# The v2 mining agent: a weight-space programming agent (OpenAI gpt-6-astra) that reads a donor's raw
# weights, hypothesises what parts do (open-vocabulary labels), writes tests for its own labels, and
# screens parts by zero-ablation. Called by mine/run.py.
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

import common
import skills
from client import Miner, agent_view, part_brief
from weight_sandbox import WeightSandbox

PLANNER_MODEL = "gpt-6-astra"
WORKER_MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "high"
PRICE_IN = 10.0 / 1e6
PRICE_OUT = 50.0 / 1e6
WORKER_PRICE_IN = 0.75 / 1e6   # rough; only used for the short end-of-run lesson summary
WORKER_PRICE_OUT = 4.5 / 1e6
GPU_PRICE_PER_HOUR = {"L4": 0.80, "L40S": 1.95, "A100-40GB": 2.10, "H100": 3.95, "H200": 4.54}

MAX_PYTHON_RUNS = 60
UNLOCK_AFTER_PY = 5
MAX_SCAN_UNITS = 4096

PART_SCHEMA = {
    "type": "object",
    "description": "mlp_cluster / attn_heads / expert need layer + indices (or handle); block needs layers.",
    "properties": {
        "kind": {"type": "string", "enum": ["mlp_cluster", "attn_heads", "expert", "block"]},
        "layer": {"type": "integer"},
        "indices": {"type": "array", "items": {"type": "integer"}},
        "layers": {"type": "array", "items": {"type": "integer"}},
        "handle": {"type": "string", "description": "instead of indices: 'py:NAME' = /work/NAME.json in the weight "
                   "sandbox holding {kind, layer, indices} or {kind, layers}; or 'scanN:SKILL:K' = top K units of "
                   "SKILL in scan N, restricted to the given layer"},
    },
    "required": ["kind"],
}

TOOLS = [
    {"type": "function", "name": "run_python",
     "description": "Weight-space reading. Runs one Python block in a network-blocked sandbox holding the donor's "
                    "raw weights (read-only; no model class, you cannot run the network). Pre-imported: torch, "
                    "CONFIG, names(), shape(name), W(name) -> float32 tensor, tok, encode(text), decode(ids). /work "
                    "persists between runs. Print what you need (12k char cap; print summaries, not raw tensors).",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"type": "function", "name": "model_structure",
     "description": "Layers, attention heads, MLP widths, MoE experts per layer as the engine sees them, plus the "
                    "base scores of the active skills and which built-in skills are skipped.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "define_skill",
     "description": "Create a test for a label you hypothesised from the weights. " + skills.GENERATOR_CONTRACT +
                    " Or pass items/control_items directly (flagged as hand-written). Rejected if the unablated "
                    "model is near chance. Once accepted it is frozen (half hidden as held-out) and joins all scoring.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"}, "description": {"type": "string"},
         "generator_code": {"type": "string"},
         "items": {"type": "array", "items": {"type": "object"}},
         "control_items": {"type": "array", "items": {"type": "object"}}},
         "required": ["name", "description"]}},
    {"type": "function", "name": "ablate",
     "description": "Zero-ablate a part; returns relative score drops on every active skill (search halves). "
                    "~seconds. Needs a weight_rationale: what in the weights made you pick this part.",
     "parameters": {"type": "object", "properties": {
         "part": PART_SCHEMA, "weight_rationale": {"type": "string"}}, "required": ["part", "weight_rationale"]}},
    {"type": "function", "name": "screen",
     "description": "Light gate for a labelled part: target drop >= 0.15 (search AND hidden held-out), every other "
                    "skill drops by at most max(0.05, target/3) and the target's matched control moves by at most +- that much, and larger than 5 random "
                    "same-size parts in the same layer. Passing parts are written to the catalog (tier 'screened').",
     "parameters": {"type": "object", "properties": {
         "part": PART_SCHEMA, "target_skill": {"type": "string"},
         "label": {"type": "string", "description": "short open-vocabulary label for what the part does"},
         "weight_rationale": {"type": "string"},
         "evidence_used": {"type": "array", "items": {"type": "string",
                           "enum": ["weights", "activation", "attribution", "routing", "ablation"]}}},
         "required": ["part", "target_skill", "label", "weight_rationale", "evidence_used"]}},
    {"type": "function", "name": "confirm_math",
     "description": "Optional stricter check for a screened MATH part: greedy GSM8K accuracy on 40 problems must "
                    "drop >= 20% relative. Upgrades the catalog record to tier 'confirmed'.",
     "parameters": {"type": "object", "properties": {"catalog_id": {"type": "string"}}, "required": ["catalog_id"]}},
    {"type": "function", "name": "activation_scan",
     "description": "LOCKED until you have used run_python 5 times and submitted one ablate/screen with a weight "
                    "rationale. Mean |activation| per unit on each skill vs the other skills. kind=expert gives MoE "
                    "routing frequency per skill.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["mlp_cluster", "attn_heads", "expert"]},
         "skills": {"type": "array", "items": {"type": "string"}},
         "layers": {"type": "array", "items": {"type": "integer"}}}, "required": ["kind", "skills"]}},
    {"type": "function", "name": "attribution_scan",
     "description": "LOCKED like activation_scan. Attribution patching: estimated rise in each skill's loss if a "
                    "unit is zeroed, ranked by selectivity (own effect minus the largest effect on other skills).",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["mlp_cluster", "attn_heads", "expert"]},
         "skills": {"type": "array", "items": {"type": "string"}},
         "layers": {"type": "array", "items": {"type": "integer"}}}, "required": ["kind", "skills"]}},
    {"type": "function", "name": "finish", "description": "End the run with a short summary.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
]


def instructions(miner, use_notes):
    s = miner.structure()
    text = (
        "You are a weight-space programming research agent. Like a programmer reading source code, you read a "
        "neural network's raw weights and work out what its parts compute. Your MAIN GOAL is to discover labelled "
        "parts of the donor model " + miner.hf_id + " - open-vocabulary labels you infer from the weights (e.g. "
        "'calendar arithmetic', 'Python indentation', 'French morphology', 'negation', 'unit conversion'). The five "
        "built-in skills (math, code, facts, multilingual; general_text is control only) are just a baseline and "
        "control suite.\n\n"
        "Method (weight-first, like the Sparks of Weight-Space Programming harness):\n"
        "1. Read the weights with run_python: e.g. project MLP down-projection columns, attention out-projection "
        "head slices, or expert down-projections through the unembedding (lm_head / tied embeddings, after the final "
        "norm) and read the tokens they promote; read which input tokens a unit responds to via its up/gate rows "
        "dotted with the embedding; look at norms, outliers, structure across layers/heads/experts.\n"
        "2. Form a hypothesis (a label + where it lives). If no built-in skill tests it, call define_skill with a "
        "generator that computes answers, plus a matched control.\n"
        "3. ablate to check cheaply, then screen the best parts with a label. Screens are cheap: try many candidates "
        "across many labels and layers rather than polishing one.\n"
        "4. activation_scan / attribution_scan unlock after 5 run_python calls and one weight-justified proposal; use "
        "them as a second, independent line of evidence, not as the main source.\n"
        "Never run the network in the sandbox. Prefer small, selective parts. Part kinds: mlp_cluster (columns of "
        "one layer's MLP down-projection input), attn_heads (heads of one layer), expert (routed experts of one MoE "
        "layer), block (contiguous whole layers). Give every proposal a concrete weight_rationale.\n\n"
        "Model structure: " + json.dumps({"n_layers": s["n_layers"], "d_model": s["d_model"],
                                          "layers_with": s["layers_with"], "example": s["example"]}) + "\n"
    )
    notes = []
    if use_notes:
        notes = read_notes(miner.short)
    if notes:
        text = text + "\nLessons from earlier runs (this and other models):\n"
        for n in notes:
            text = text + "- " + n + "\n"
    done = []
    for key in miner.state.get("screens", {}):
        r = miner.state["screens"][key]
        done.append(json.dumps(part_brief(r["part"])) + " target=" + r["target"] + " passed=" + str(r["gate"]["passed"]))
    if done:
        text = text + "\nAlready screened in earlier sessions (don't repeat):\n" + "\n".join(done[-40:]) + "\n"
    if miner.agent_skills():
        text = text + "\nYour skills defined earlier: " + ", ".join(miner.agent_skills().keys()) + "\n"
    return text


def read_notes(short, limit=25):
    notes = []
    logs_root = os.path.join(common.ROOT, "logs")
    if not os.path.isdir(logs_root):
        return notes
    names = sorted(os.listdir(logs_root))
    for name in names:
        path = os.path.join(logs_root, name, "notes.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                prefix = "" if name == short else "[" + name + "] "
                notes.append(prefix + rec["lesson"])
    return notes[-limit:]


class Session:
    def __init__(self, short, budget_usd, max_hours, max_screens):
        self.miner = Miner(short)
        self.short = short
        self.budget_usd = budget_usd
        self.deadline = time.time() + max_hours * 3600
        self.max_screens = max_screens
        self.box = None
        self.py_runs = 0
        self.weight_proposals = 0
        self.screens = 0
        self.scans = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "worker_input_tokens": 0, "worker_output_tokens": 0}

    def sandbox(self):
        if self.box is None:
            box = WeightSandbox(self.miner.hf_id, self.miner.revision, self.miner.params_b)
            self.box = box  # set before start() so finally can always stop it
            box.start()
        return self.box

    def cost(self):
        return (self.usage["input_tokens"] * PRICE_IN + self.usage["output_tokens"] * PRICE_OUT
                + self.usage["worker_input_tokens"] * WORKER_PRICE_IN + self.usage["worker_output_tokens"] * WORKER_PRICE_OUT)

    def unlocked(self):
        return self.py_runs >= UNLOCK_AFTER_PY and self.weight_proposals >= 1

    # ---------- part resolution ----------

    def resolve(self, part):
        if not isinstance(part, dict):
            return "missing part"
        part = dict(part)
        handle = part.pop("handle", None) or ""
        if handle.startswith("py:"):
            safe = common.clean_name(handle[3:])
            raw = self.sandbox().read_file("/work/" + safe + ".json")
            if raw is None:
                return "no /work/" + safe + ".json in the sandbox"
            try:
                part = json.loads(raw)
            except Exception:
                return "could not parse /work/" + safe + ".json"
            if str(part.get("handle", "")).startswith("py:"):
                return "a py: file cannot point to another py: file"
            part.pop("handle", None)
        elif handle.startswith("scan"):
            pieces = handle.split(":")
            if len(pieces) != 3 or pieces[0] not in self.scans:
                return "bad scan handle; use scanN:SKILL:K"
            scan = self.scans[pieces[0]]
            units = scan["full"].get(pieces[1])
            if units is None:
                return "skill " + pieces[1] + " not in " + pieces[0]
            k = int(pieces[2])
            layer = part.get("layer")
            picked = []
            for u in units:
                if layer is None or u["layer"] == layer:
                    picked.append(u["unit"])
                    if layer is None:
                        layer = u["layer"]
                if len(picked) >= k:
                    break
            part = {"kind": scan["kind"], "layer": layer, "indices": picked}
        if part.get("kind") == "block":
            part = {"kind": "block", "layers": [int(x) for x in part.get("layers") or []]}
        else:
            idx = []
            for x in part.get("indices") or []:
                idx.append(int(x))
            part = {"kind": part.get("kind"), "layer": part.get("layer"), "indices": idx}
        return part

    # ---------- tools ----------

    def tool(self, name, args):
        m = self.miner
        if name == "run_python":
            if self.py_runs >= MAX_PYTHON_RUNS:
                return {"error": "python run budget used up"}
            self.py_runs += 1
            return {"output": self.sandbox().run(args.get("code", ""))}
        if name == "model_structure":
            base = m.base()
            scores = {}
            for key in base["scores"]:
                if key.endswith("|search"):
                    scores[key.split("|")[0]] = base["scores"][key]["score"]
            return {"structure": m.structure()["example"], "n_layers": m.structure()["n_layers"],
                    "layers_with": m.structure()["layers_with"], "base_scores_search": scores,
                    "skipped": m.state.get("skipped"), "agent_skills": list(m.agent_skills().keys())}
        if name == "define_skill":
            return skills.define(m, self.sandbox, args.get("name", ""), args.get("description", ""),
                                 args.get("generator_code"), args.get("items"), args.get("control_items"))
        if name in ["activation_scan", "attribution_scan"]:
            if not self.unlocked():
                return {"error": "locked: needs " + str(UNLOCK_AFTER_PY) + " run_python calls (have " + str(self.py_runs)
                                 + ") and 1 weight-justified ablate/screen (have " + str(self.weight_proposals) + ")"}
            known = m.target_skills() + common.CONTROL_ONLY
            wanted = []
            for sname in args.get("skills") or []:
                if sname in known:
                    wanted.append(sname)
            if not wanted:
                return {"error": "no known skills given; known: " + ", ".join(known)}
            if "general_text" not in wanted:
                wanted.append("general_text")  # reference
            layers = args.get("layers") or []
            if name == "activation_scan":
                res = m.call("activation_scan", m.payload(), args["kind"], wanted, layers)
            else:
                res = m.call("attribution_scan", m.payload(), args["kind"], wanted, layers, 24,
                             common.scan_token_budget(m.params_b))
            sid = "scan" + str(len(self.scans) + 1)
            self.scans[sid] = {"kind": args["kind"], "full": res["stats"]}
            brief = {}
            for sname in res["stats"]:
                brief[sname] = res["stats"][sname][:12]
            common.log_event(self.short, name, {"kind": args["kind"], "skills": wanted, "layers": layers},
                             {"seconds": res["seconds"]})
            return {"scan_id": sid, "top_units": brief, "seconds": res["seconds"],
                    "use": "part={'kind': '" + args["kind"] + "', 'layer': L, 'handle': '" + sid + ":SKILL:K'}"}
        if name in ["ablate", "screen"]:
            part = self.resolve(args.get("part"))
            if isinstance(part, str):
                return {"error": part}
            if not (args.get("weight_rationale") or "").strip():
                return {"error": "weight_rationale is required"}
            if name == "ablate":
                self.weight_proposals += 1
                res = m.ablate(part)
                drops = {}
                for key in res["drops"]:
                    if key.endswith("|search"):
                        drops[key.split("|")[0]] = res["drops"][key]
                return {"part": part_brief(part), "drops": drops, "seconds": res["seconds"]}
            if self.screens >= self.max_screens:
                return {"error": "screen budget used up"}
            target = args.get("target_skill")
            if target not in m.target_skills():
                return {"error": "target_skill must be one of " + ", ".join(m.target_skills())}
            if not args.get("label"):
                return {"error": "label is required"}
            self.weight_proposals += 1
            self.screens += 1
            res = m.screen(part, target)
            view = agent_view(res)
            if res["gate"]["passed"]:
                rec = m.write_part(part, args["label"], target, res, args.get("evidence_used") or [],
                                   args["weight_rationale"], "screened")
                view["catalog_id"] = rec["id"]
            return view
        if name == "confirm_math":
            path = common.catalog_path(self.short)
            records = []
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
            target = None
            for r in records:
                if r["id"] == args.get("catalog_id"):
                    target = r
            if target is None:
                return {"error": "unknown catalog_id"}
            if target["skill"]["name"] != "math":
                return {"error": "confirm_math only applies to math parts"}
            if target["type"] == "block":
                part = {"kind": "block", "layers": target["layers"]}
            else:
                import torch
                idx = torch.load(os.path.join(common.ROOT, target["units"]))["indices"].tolist()
                part = {"kind": target["type"], "layer": target["layer"], "indices": idx}
            res = m.confirm(part)
            if res["passed"]:
                for r in records:
                    if r["id"] == target["id"]:
                        r["tier"] = "confirmed"
                        r["signature"]["confirm"] = res
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                os.replace(tmp, path)
            return res
        return {"error": "unknown tool " + name}


def run(short, budget_usd=25.0, max_hours=3.0, max_screens=60, use_notes=True, effort=REASONING_EFFORT):
    load_dotenv(os.path.join(common.ROOT, ".env"), override=True)
    session = Session(short, budget_usd, max_hours, max_screens)
    miner = session.miner
    miner.structure()
    miner.base()
    run_id = "run-" + time.strftime("%Y%m%d-%H%M%S")
    log_dir = common.logs_dir(short)
    transcript_path = os.path.join(log_dir, run_id + ".jsonl")
    llm = OpenAI()
    system_text = instructions(miner, use_notes)
    summary = None
    stop_reason = None
    turns = 0
    print("[" + short + "] run", run_id, "GPU", miner.gpu, "revision", miner.revision[:10])
    try:
        response = llm.responses.create(model=PLANNER_MODEL, instructions=system_text, tools=TOOLS,
                                        reasoning={"effort": effort},
                                        input="Begin dissecting " + miner.hf_id + ".")
        while True:
            turns += 1
            session.usage["input_tokens"] += response.usage.input_tokens
            session.usage["output_tokens"] += response.usage.output_tokens
            outputs = []
            for item in response.output:
                if item.type == "message":
                    print("[" + short + "][agent]", response.output_text[:300])
                    common.append_jsonl(transcript_path, {"agent": response.output_text})
                if item.type != "function_call":
                    continue
                try:
                    args = json.loads(item.arguments or "{}")
                except Exception:
                    args = {}
                if item.name == "finish":
                    summary = args.get("summary", "")
                    result = {"ok": True}
                else:
                    try:
                        result = session.tool(item.name, args)
                    except Exception as e:
                        result = {"error": repr(e)[:500]}
                text = json.dumps(result)
                print("[" + short + "][tool]", item.name, text[:240])
                common.append_jsonl(transcript_path, {"tool": item.name, "args": args, "result": result})
                outputs.append({"type": "function_call_output", "call_id": item.call_id, "output": text[:30000]})
            if summary is not None:
                stop_reason = "finished"
                break
            if not outputs:
                stop_reason = "no tool call"
                break
            if session.cost() >= budget_usd:
                stop_reason = "OpenAI budget reached"
                break
            if time.time() >= session.deadline:
                stop_reason = "time limit reached"
                break
            if session.screens >= max_screens:
                outputs.append({"type": "message", "role": "user", "content": "Screen budget used up. Call finish."})
            if turns > 400:
                stop_reason = "turn cap"
                break
            response = llm.responses.create(model=PLANNER_MODEL, instructions=system_text, tools=TOOLS,
                                            reasoning={"effort": effort},
                                            input=outputs, previous_response_id=response.id)
    finally:
        if session.box is not None:
            session.box.stop()

    lessons = summarize_lessons(llm, session, transcript_path)
    for lesson in lessons:
        common.append_jsonl(os.path.join(log_dir, "notes.jsonl"), {"run_id": run_id, "model": miner.hf_id, "lesson": lesson})
    gpu_cost = miner.modal_seconds / 3600.0 * GPU_PRICE_PER_HOUR.get(miner.gpu, 4.0)
    n_parts = 0
    if os.path.exists(common.catalog_path(short)):
        with open(common.catalog_path(short)) as f:
            for line in f:
                if line.strip():
                    n_parts += 1
    cost = {"run_id": run_id, "openai_usd_est": round(session.cost(), 2),
            "modal_gpu_usd_est": round(gpu_cost, 2), "modal_busy_seconds": round(miner.modal_seconds),
            "gpu": miner.gpu, "note": "modal estimate counts busy call time only (not idle keep-warm or first load)",
            "usage": session.usage, "python_runs": session.py_runs, "screens": session.screens,
            "stop_reason": stop_reason, "summary": summary, "parts_in_catalog": n_parts}
    common.append_jsonl(os.path.join(log_dir, "costs.jsonl"), cost)
    common.log_event(short, "agent_run", {"run_id": run_id, "budget_usd": budget_usd, "max_hours": max_hours,
                                          "effort": effort}, cost)
    print("[" + short + "] DONE", json.dumps(cost)[:1500])
    return cost


def summarize_lessons(llm, session, transcript_path):
    if not os.path.exists(transcript_path):
        return []
    with open(transcript_path) as f:
        text = f.read()[-30000:]
    prompt = ("Here is the log of a weight-space dissection run on " + session.miner.hf_id + ". Write 3 to 6 short, "
              "concrete lessons for future runs on this and other models (what in the weights predicted real parts, "
              "what failed, with numbers). Return JSON {\"lessons\": [\"...\"]}.\n\n" + text)
    try:
        resp = llm.responses.create(model=WORKER_MODEL, input=prompt)
        session.usage["worker_input_tokens"] += resp.usage.input_tokens
        session.usage["worker_output_tokens"] += resp.usage.output_tokens
        out = resp.output_text
        return json.loads(out[out.find("{"):out.rfind("}") + 1])["lessons"]
    except Exception as e:
        return ["(lesson summary failed: " + repr(e)[:120] + ")"]
