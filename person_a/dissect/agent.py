# The dissection agent: an OpenAI model that proposes parts, tests them on Modal, and
# submits the best ones to the validation gate. Only parts that pass the gate reach catalog.jsonl.
#   cd ~/swarmy && .venv/bin/python dissect/agent.py [max_tool_calls]
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import client  # noqa: E402
from weight_sandbox import WeightSandbox  # noqa: E402
from config import PLANNER_MODEL, WORKER_MODEL  # noqa: E402
from experiments import log  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"), override=True)  # the key in the shell is the wrong one

NOTES_PATH = os.path.join(HERE, "notes.jsonl")
RUNS_DIR = os.path.join(HERE, "runs")
SEARCH_TASKS = ["gsm8k", "trivia", "wiki", "wiki_digits"]  # held-out GSM8K is hidden from the agent
MAX_VALIDATIONS = 3
MAX_MLP_UNITS = 2048
MAX_BLOCK_LAYERS = 4
MAX_PYTHON_RUNS = 40          # weight-sandbox runs, counted separately from the other tools
REASONING_EFFORT = "high"     # the blog used high to max
PRICE_IN = 10.0 / 1e6         # gpt-6-astra, $ per input token
PRICE_OUT = 50.0 / 1e6        # $ per output token

PART_SCHEMA = {
    "type": "object",
    "description": "mlp_cluster/attn_heads need layer+indices; block needs layers.",
    "properties": {
        "kind": {"type": "string", "enum": ["mlp_cluster", "attn_heads", "block"]},
        "layer": {"type": "integer"},
        "indices": {"type": "array", "items": {"type": "integer"}},
        "layers": {"type": "array", "items": {"type": "integer"}},
        "handle": {"type": "string", "description": "candidate handle from top_units_in_layer (e.g. 'cand1'); "
                                                      "replaces layer/indices. Use 'cand1:256' for its first 256 units. "
                                                      "Or 'py:NAME' to use a JSON file you saved in the weight sandbox "
                                                      "at /work/NAME.json with {kind, layer, indices} or {kind, layers}."},
    },
    "required": ["kind"],
}

TOOLS = [
    {"type": "function", "name": "run_python",
     "description": "Weight-space reading. Runs one Python block in a sandbox holding the donor's raw weights "
                    "(no network, no model class: you cannot run the network on inputs here). Pre-imported: torch, "
                    "CONFIG (config.json dict), names() (all tensor names), shape(name), W(name) (float32 tensor), "
                    "tok (tokenizer), encode(text) -> ids, decode(ids) -> token strings. Files in /work persist "
                    "between runs; each run is a fresh process. Print what you want to see (output is capped at "
                    "20k chars). At most " + str(MAX_PYTHON_RUNS) + " runs.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"type": "function", "name": "model_info", "description": "Shape of the donor model.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "attribution_scan",
     "description": "Attribution patching. Estimated increase in mean NLL (math loss on GSM8K solution digits, and control loss "
                    "on trivia+wiki) if each unit is mean-ablated. rank_by 'selective' = math effect minus max(control effect, 0). Returns top units "
                    "and a per-layer summary.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["mlp_cluster", "attn_heads"]},
         "top_k": {"type": "integer"},
         "layers": {"type": "array", "items": {"type": "integer"}},
         "rank_by": {"type": "string", "enum": ["selective", "target"]}}, "required": ["kind"]}},
    {"type": "function", "name": "activation_profile",
     "description": "Mean |activation| per unit on math text minus on control text. Returns top units and per-layer summary.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["mlp_cluster", "attn_heads"]},
         "top_k": {"type": "integer"},
         "layers": {"type": "array", "items": {"type": "integer"}}}, "required": ["kind"]}},
    {"type": "function", "name": "top_units_in_layer",
     "description": "The k best units in one layer (by attribution or activation), returned as a ready part's indices.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["mlp_cluster", "attn_heads"]},
         "layer": {"type": "integer"}, "k": {"type": "integer"},
         "source": {"type": "string", "enum": ["attribution", "activation"]},
         "rank_by": {"type": "string", "enum": ["selective", "target"]}},
         "required": ["kind", "layer", "k"]}},
    {"type": "function", "name": "ablate_and_score",
     "description": "Switch a part off (zero or mean ablation) and return relative score drops on gsm8k (target) and "
                    "trivia, wiki, wiki_digits (controls). Drop = (base - ablated)/base.",
     "parameters": {"type": "object", "properties": {
         "part": PART_SCHEMA, "method": {"type": "string", "enum": ["zero", "mean"]},
         "include_generation": {"type": "boolean", "description": "also measure greedy GSM8K accuracy on 60 "
                                "problems (the gated metric; adds ~30 s)"}},
         "required": ["part", "method"]}},
    {"type": "function", "name": "validate",
     "description": "Run the full validation gate (both methods, hidden held-out set (pass/fail only), random baseline of same-size parts, "
                    "generation accuracy). Expensive: at most " + str(MAX_VALIDATIONS) + " per run. Passing parts are "
                    "written to the catalog.",
     "parameters": {"type": "object", "properties": {
         "part": PART_SCHEMA, "label": {"type": "string", "description": "short label, e.g. 'arithmetic'"},
         "evidence_used": {"type": "array", "items": {"type": "string", "enum": ["weights", "activation", "attribution", "ablation"]},
                           "description": "which kinds of evidence led you to this part"},
         "weight_rationale": {"type": "string", "description": "what in the weights points to this part (one or two sentences)"}},
         "required": ["part", "label", "evidence_used"]}},
    {"type": "function", "name": "finish", "description": "End the run with a short summary of what was found.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


def read_notes(limit=30):
    notes = []
    if os.path.exists(NOTES_PATH):
        with open(NOTES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    notes.append(json.loads(line)["lesson"])
    return notes[-limit:]


def instructions(use_notes=True):
    text = (
        "You are a weight-space programming research agent. Like a programmer reading source code, you read a neural "
        "network's raw weights and work out what each part computes. Start from the weights: use run_python to "
        "inspect the weight matrices directly (e.g. project MLP down_proj columns or attention o_proj/v_proj "
        "outputs through the unembedding and read the tokens they promote; compare norms, structure and outliers "
        "across layers and heads; find units whose weights read or write number/operator tokens). Form a "
        "hypothesis from the weights, then use the activation and attribution tools as a second, independent "
        "line of evidence, and ablate_and_score to test. Never run the network inside the sandbox.\n\n"
        "You are dissecting the open-weight model " + client.DONOR + ". Goal: find parts "
        "(mlp_cluster = neurons of ONE layer's MLP; attn_heads = query heads of ONE layer; block = a few whole layers) "
        "whose removal destroys math reasoning (GSM8K) while leaving controls intact: trivia (factual recall), wiki "
        "(plain text), wiki_digits (numbers in prose, no arithmetic). Use the scans to propose, ablate_and_score to "
        "test cheaply, then validate your best candidates. The gsm8k log-prob score is a fast proxy that UNDERSTATES "
        "reasoning damage (in one test a 9% proxy drop came with a 48% accuracy drop), so check promising parts with "
        "include_generation=true. Gate: relative drop in GSM8K generation accuracy >= 0.30 under BOTH zero and mean "
        "ablation and on a hidden held-out set, every control (log-prob) drop < 0.10, and larger than the accuracy "
        "drop of random same-size parts in the same layer. Limits: mlp_cluster <= " + str(MAX_MLP_UNITS) + " neurons, block <= " + str(MAX_BLOCK_LAYERS) +
        " layers, at most " + str(MAX_VALIDATIONS) + " validate calls. The last " + str(MAX_VALIDATIONS) + " tool "
        "calls are reserved for validate, so screening ends early; validate at least your best 2 candidates even if "
        "they look short of the threshold, because the gate result is the evidence. Prefer small, selective parts. "
        "Use candidate handles (e.g. part={'kind': 'mlp_cluster', 'handle': 'cand1:256'}) instead of long index lists. "
        "Grow a part "
        "step by step and watch the controls. A failed validation is still useful evidence. Be efficient with "
        "tool calls; call finish when done."
    )
    notes = []
    if use_notes:
        notes = read_notes()
    if notes:
        text = text + "\n\nLessons from earlier runs:\n"
        for n in notes:
            text = text + "- " + n + "\n"
    return text


def sandbox(state):
    if state["box"] is None:
        print("[sandbox] starting weight sandbox ...")
        box = WeightSandbox(client.DONOR)
        state["box"] = box  # set before start() so the finally-block can always stop it
        box.start()
    return state["box"]


def run_tool(name, args, state):
    if name == "run_python":
        if state["python_runs"] >= MAX_PYTHON_RUNS:
            return {"error": "python run budget used up"}
        state["python_runs"] += 1
        return {"output": sandbox(state).run(args.get("code", ""))}
    d = client.dissector()
    if name == "model_info":
        return d.model_info.remote()
    if "layers" in args and not args["layers"]:
        args["layers"] = None  # an empty list means all layers
    if name == "attribution_scan":
        top_k = min(int(args.get("top_k", 20)), 40)
        return d.attribution_scan.remote(args["kind"], top_k, args.get("layers"), args.get("rank_by", "selective"))
    if name == "activation_profile":
        top_k = min(int(args.get("top_k", 20)), 40)
        return d.activation_profile.remote(args["kind"], top_k, args.get("layers"))
    if name == "top_units_in_layer":
        k = int(args["k"])
        if args["kind"] == "mlp_cluster":
            k = min(k, MAX_MLP_UNITS)
        res = d.top_units_in_layer.remote(args["kind"], int(args["layer"]), k,
                                          args.get("source", "attribution"), args.get("rank_by", "selective"))
        # store full indices locally; give the agent a handle so it doesn't have to echo thousands of ints
        handle = "cand" + str(len(state["candidates"]) + 1)
        state["candidates"][handle] = {"kind": res["kind"], "layer": res["layer"], "indices": res["indices"]}
        return {"candidate_handle": handle, "kind": res["kind"], "layer": res["layer"], "k": len(res["indices"]),
                "first_units": res["units"],
                "how_to_use": "pass part={'kind': '" + res["kind"] + "', 'handle': '" + handle + "'} for all "
                              + str(len(res["indices"])) + " units, or 'handle': '" + handle + ":N' for the top N"}
    part = resolve_part(args.get("part"), state)
    if isinstance(part, dict) and part.get("_py"):
        part = load_py_part(part["_py"], state)
        if isinstance(part, dict) and part.get("_py"):
            part = "a py: file cannot point to another py: file"
    if isinstance(part, str):
        return {"error": part}
    if name == "ablate_and_score":
        tasks = list(SEARCH_TASKS)
        if args.get("include_generation"):
            tasks.append("gsm8k_gen")
        r = d.ablate_and_score.remote(part, args.get("method", "mean"), tasks)
        return {"drop": r["drop"], "base": r["base"], "part_size": client.part_size(part)}
    if name == "validate":
        if state["validations"] >= MAX_VALIDATIONS:
            return {"error": "validation budget used up"}
        if not args.get("label"):
            return {"error": "validate needs a label"}  # checked before spending the budget
        state["validations"] += 1
        evidence_used = args.get("evidence_used") or []
        proposed_by = "agent:" + PLANNER_MODEL + " evidence=" + "+".join(evidence_used)
        if args.get("weight_rationale"):
            proposed_by = proposed_by + " | weights: " + args["weight_rationale"][:300]
        r = client.validate_part(part, args["label"], proposed_by, state["run_id"])
        return client.summarize_validation(r)
    return {"error": "unknown tool " + name}


def resolve_part(part, state):
    if part is None:
        return "missing part"
    part = dict(part)
    handle = part.pop("handle", None)
    if handle and handle.startswith("py:"):
        return {"_py": handle[3:]}
    if handle:
        limit = None
        if ":" in handle:
            pieces = handle.split(":")
            handle = pieces[0]
            limit = int(pieces[1])
        if handle not in state["candidates"]:
            return "unknown handle " + handle
        cand = state["candidates"][handle]
        indices = list(cand["indices"])
        if limit is not None:
            indices = indices[:limit]
        part = {"kind": cand["kind"], "layer": cand["layer"], "indices": indices}
    # drop fields that don't belong to the kind (the model sometimes sends empty extras)
    if part.get("kind") == "block":
        part = {"kind": "block", "layers": part.get("layers", [])}
    else:
        part = {"kind": part.get("kind"), "layer": part.get("layer"), "indices": part.get("indices", [])}
    err = client.check_part(part)
    if err:
        return err
    if part["kind"] == "mlp_cluster" and len(part["indices"]) > MAX_MLP_UNITS:
        return "mlp_cluster too large"
    if part["kind"] == "block" and len(part["layers"]) > MAX_BLOCK_LAYERS:
        return "block too large"
    return part


def load_py_part(name, state):
    # Reads /work/NAME.json from the weight sandbox and checks it like any other part.
    safe = ""
    for ch in name:
        if ch.isalnum() or ch in "_-":
            safe = safe + ch
    out = sandbox(state).run("print(open('/work/" + safe + ".json').read())")
    try:
        start = out.find("{")
        end = out.rfind("}")
        part = json.loads(out[start:end + 1])
    except Exception:
        return "could not read /work/" + safe + ".json: " + out[:300]
    return resolve_part(part, state)


def summarize_lessons(llm, transcript_text):
    prompt = ("Here is the log of a dissection run on " + client.DONOR + ". Write 3 to 6 short, concrete lessons "
              "for future runs (what worked, what failed, with numbers). Return JSON: {\"lessons\": [\"...\"]}.\n\n"
              + transcript_text[-30000:])
    resp = llm.responses.create(model=WORKER_MODEL, input=prompt)
    text = resp.output_text.strip()
    start = text.find("{")
    end = text.rfind("}")
    lessons = []
    try:
        lessons = json.loads(text[start:end + 1])["lessons"]
    except Exception:
        lessons = [text[:500]]
    return lessons, resp.usage


def main():
    max_calls = 30
    use_notes = True
    for a in sys.argv[1:]:
        if a == "--no-notes":
            use_notes = False
        else:
            max_calls = int(a)
    run_id = "run-" + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_path = os.path.join(RUNS_DIR, run_id + ".jsonl")
    llm = OpenAI()
    state = {"run_id": run_id, "validations": 0, "screening_calls": 0, "candidates": {},
             "python_runs": 0, "box": None}
    system_text = instructions(use_notes)
    usage = {"input_tokens": 0, "output_tokens": 0}
    transcript = []
    calls = 0
    summary = None

    response = llm.responses.create(model=PLANNER_MODEL, instructions=system_text, tools=TOOLS,
                                    reasoning={"effort": REASONING_EFFORT},
                                    input="Begin. Donor: " + client.DONOR + ".")
    try:
      while True:
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        tool_outputs = []
        for item in response.output:
            if item.type == "message":
                text = response.output_text
                transcript.append({"role": "assistant", "text": text})
                print("[agent]", text[:400])
            if item.type != "function_call":
                continue
            if item.name not in ["run_python", "validate", "finish"]:
                calls += 1  # sandbox runs and validations have their own budgets
            args = json.loads(item.arguments or "{}")
            print("[tool]", item.name, json.dumps(args)[:200])
            if item.name == "finish":
                summary = args.get("summary", "")
                result = {"ok": True}
            elif item.name == "run_python":
                try:
                    result = run_tool(item.name, args, state)
                except Exception as e:
                    result = {"error": repr(e)[:400]}
            elif item.name == "validate":
                # validate has its own budget (MAX_VALIDATIONS); refused or failed screening calls can't eat it
                try:
                    result = run_tool(item.name, args, state)
                except Exception as e:
                    result = {"error": repr(e)[:400]}
            elif calls > max_calls:
                result = {"error": "tool call budget reached; call finish now"}
            elif item.name != "validate" and state["screening_calls"] >= max_calls - MAX_VALIDATIONS:
                result = {"error": "screening budget used up; validate your best candidates (validate calls left: "
                                   + str(MAX_VALIDATIONS - state["validations"]) + ") or call finish"}
            else:
                if item.name != "validate":
                    state["screening_calls"] += 1
                try:
                    result = run_tool(item.name, args, state)
                except Exception as e:
                    result = {"error": repr(e)[:400]}
            out_text = json.dumps(result)
            print("   ->", out_text[:300])
            transcript.append({"tool": item.name, "args": args, "result": result})
            with open(run_path, "a") as f:
                f.write(json.dumps({"tool": item.name, "args": args, "result": result}) + "\n")
            tool_outputs.append({"type": "function_call_output", "call_id": item.call_id, "output": out_text})
        state["turns"] = state.get("turns", 0) + 1
        if summary is not None or len(tool_outputs) == 0 or calls > max_calls + 3:
            break
        if state["turns"] > max_calls + MAX_PYTHON_RUNS + 10:
            print("[stop] turn cap reached")
            break
        response = llm.responses.create(model=PLANNER_MODEL, instructions=system_text, tools=TOOLS,
                                        reasoning={"effort": REASONING_EFFORT},
                                        input=tool_outputs, previous_response_id=response.id)
    finally:
        if state["box"] is not None:
            state["box"].stop()

    transcript_text = json.dumps(transcript)
    lessons, worker_usage = summarize_lessons(llm, transcript_text)
    with open(NOTES_PATH, "a") as f:
        for lesson in lessons:
            f.write(json.dumps({"run_id": run_id, "model": client.DONOR, "lesson": lesson}) + "\n")
    usage["worker_input_tokens"] = worker_usage.input_tokens
    usage["worker_output_tokens"] = worker_usage.output_tokens

    print("\nSUMMARY:", summary)
    print("LESSONS:", json.dumps(lessons, indent=1))
    cost = usage["input_tokens"] * PRICE_IN + usage["output_tokens"] * PRICE_OUT
    usage["planner_cost_usd_est"] = round(cost, 2)
    usage["python_runs"] = state["python_runs"]
    print("USAGE:", usage, "tool calls:", calls, "validations:", state["validations"])
    log("dissect_agent_run", {"run_id": run_id, "planner": PLANNER_MODEL, "worker": WORKER_MODEL,
                              "max_calls": max_calls, "use_notes": use_notes, "effort": REASONING_EFFORT},
        {"summary": summary, "tool_calls": calls, "validations": state["validations"], "usage": usage,
         "lessons": lessons, "transcript": os.path.relpath(run_path, ROOT)})


if __name__ == "__main__":
    main()
