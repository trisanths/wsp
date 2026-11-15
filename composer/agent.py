"""The composer loop: an LLM writes wsp programs, practices them, and finishes.

    from composer.agent import run_request
    summary = run_request("composer/requests/math_plus.json", practice_fn)

practice_fn(program_dict, payload) compiles and scores one program; on
Modal it is modal_composer.compile_and_practice.remote. The loop calls the
OpenAI Responses API with function tools only (see composer/tools.py),
and archives every request, reply, tool call, and result in
composer/runs/<request>/<timestamp>/.
"""

import json
import os

from composer.archive import Archive
from composer.llm import OpenAIClient, read_key
from composer.skills import DEFAULT_DRIFT_TOLERANCE, SKILLS
from composer.sources import Sources
from composer.tools import Toolbox, read_language_text, tool_specs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(REPO_ROOT, "composer", "runs")
DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_MAX_TURNS = 60
DEFAULT_MAX_TOKENS = 3000000

SHORT_LANGUAGE = """Short language notes (call language_reference for the full text):
- One statement per line. `host M` names the model the program changes.
- `graft ID at N rank R stitch mlp objective donor_lm steps S` puts a mined part or a donor layer block
  inside the host at layer N, with a trained stitch. Block ids: block/<org>/<model>/<first>-<last>.
  objective donor_lm needs the same tokenizer as the host; downstream works across tokenizers.
- `graft ID at N gate on gate_bias -1.0`: a token gate on the graft, so several grafts share one host.
- `experts DONOR1,DONOR2 top_k K` makes each layer's MLP a router over the host MLP and the donors' MLPs.
- `insert DONOR layers A..B at N stitch mlp` adds donor layers as a gated residual branch.
- `from M` plus `carve target 1.5b select influence` keeps a source's most useful layers to fit a size.
- `splice base a sources a=M1,b=M2 plan a:0..13, b:14..27` builds one plain model from layer segments.
- `mask` and `tv` apply catalog deletion masks and task vectors to the host.
- `lora rank R` adds a host LoRA trained by heal. `heal steps S data D lr X` trains the new parts and LoRAs.
  `heal ... distill TEACHER kd_weight 1.0` adds a distillation loss (teacher with the host's tokenizer).
- `budget total 2b active 1b heal_minutes 30` can only tighten the request's limits.
- `control random` on a graft, experts, or insert builds the same shape from random components of the same size.
- Not available: `tap` and `tap_train`. A tap runs a source model at inference, so the result is not one network.
"""


def load_request(path):
    with open(path) as f:
        request = json.load(f)
    for name in ["name", "skills", "constraints"]:
        if name not in request:
            raise ValueError(f"{path}: a request needs {name!r}")
    for skill in request["skills"]:
        if skill not in SKILLS:
            raise ValueError(f"{path}: unknown skill {skill!r}; the skills are {', '.join(sorted(SKILLS))}")
    check_request_options(path, request)
    return request


def check_request_options(path, request):
    """Check the optional request keys: limits, item caps, and the report options."""
    constraints = request["constraints"]
    for name in ["max_total_params_b", "max_run_params_b", "max_heal_minutes", "heal_steps_cap"]:
        value = constraints.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"{path}: constraints.{name} is a number above 0 or null, got {value!r}")
    base = request.get("base")
    if base is not None and (not isinstance(base, str) or base == ""):
        raise ValueError(f"{path}: base is a model id or null, got {base!r}")
    tolerance = request.get("drift_tolerance")
    if tolerance is not None and (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                                  or tolerance < 0):
        raise ValueError(f"{path}: drift_tolerance is a number of 0 or more (nats) or null, got {tolerance!r}")
    for name in ["practice_items", "heldout_items"]:
        value = request.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{path}: {name} is an integer of 1 or more or null, got {value!r}")
    report = request.get("report", {})
    if not isinstance(report, dict):
        raise ValueError(f"{path}: report is an object")
    for name in report:
        if name not in ("control_seeds", "far_factor", "noise_floor", "probe_noise_floor"):
            raise ValueError(f"{path}: unknown report option {name!r}; the options are control_seeds, "
                             f"far_factor, noise_floor, probe_noise_floor")


def system_prompt(request, max_turns, max_tokens):
    constraints = request["constraints"]
    skills = request["skills"]
    skill_lines = []
    for skill in skills:
        skill_lines.append(f"- {skill} (weight {skills[skill]}): probed on held-out text of the "
                           f"{SKILLS[skill]['pool']} pool")
    tolerance = request.get("drift_tolerance")
    if tolerance is None:
        tolerance = DEFAULT_DRIFT_TOLERANCE

    def cap(name):
        value = constraints.get(name)
        if value is None:
            return "no cap"
        return str(value)

    lines = [
        "You are the composer for Sapien (the wsp package), a language and compiler that writes one new network",
        "by composing components mined from existing open models: grafted donor blocks and mined parts with",
        "trained stitches (optionally gated), routed experts, inserted donor layers, splices, carves, deletion",
        "masks, task vectors, and a short heal (a LoRA heal or a distillation heal).",
        "",
        "Goal: " + request.get("description", request["name"]),
        ("Base: the program's host must be " + request["base"] + "." if request.get("base") else
         "Base: you choose the host."),
        "Objective: " + request.get("objective", "weighted mean of skill scores") +
        ". You are scored on the probe objective below while you work; benchmarks score only the final model.",
        "",
        "Skills:",
    ]
    lines += skill_lines
    lines += [
        "",
        "Rules:",
        "- Compose components only. You never write, set, or edit raw weight values, and no tool lets you.",
        "  Your output is a program in the wsp language.",
        "- Stitches and the heal are the only training. Each stage has at most "
        + cap("heal_steps_cap") + " steps.",
        "- The result is ONE network that runs alone. You cannot write taps: a tap runs a source model at",
        "  inference, and a program with one is rejected.",
        "- A host with only a LoRA and heal (or a distillation heal) is the fine-tuning baseline and is rejected.",
        "- Limits: total params at most " + cap("max_total_params_b") + "b; params run per token at most "
        + cap("max_run_params_b") + "b; heal at most " + cap("max_heal_minutes") + " minutes. The limits are",
        "  added to your program as its budget. compile_and_practice checks an estimate before compiling and",
        "  the measured numbers after. The model may grow past the base when the limits allow it.",
        "- The final report scores your program on held-out items against: the base; the base with a LoRA; the",
        "  base with a distillation heal from each source that has the host's tokenizer; your program with",
        "  `control random` on every component (random components of the same size); and, when the size limit is",
        "  far above the base, the best existing model of that size. All arms get the same heal data and budget.",
        "  A program that does not beat its own random control by more than seed noise has not shown that its",
        "  components matter, so prefer components whose transfer drops under `control random`.",
        "- No benchmark data trains or selects anything. Benchmarks are chosen per user request, so you never",
        "  practice on them and no tool shows a benchmark score. The heal data names are approved pool text.",
        "  Write `data NAME` on every heal: the language default, gsm8k_train, is a benchmark and is rejected.",
        "",
        "Budgets:",
        f"- Compile budget: {constraints.get('compile_budget', 10)} compiles. A parse error, a constraint",
        "  rejection, or a repeat of a program already practiced costs nothing.",
        f"- At most {max_turns} model turns and {max_tokens} tokens. If you stop without finish, the loop",
        "  submits your best practiced program.",
        "- A compile takes minutes to an hour on one H200, so think before you compile.",
        "",
        "Benchmarks are unseen. They score the final program and its rivals once, after finish.",
        "Practice feedback is the benchmark-free probe on held-out pool text, never on heal text:",
        "- transfer, per skill: the share of the base-to-donor gap in next-token KL that your model closes on the",
        "  skill's held-out text (0 is the base, 1 is the donor). The donor of a skill is the source model of the",
        "  components you used for it. A skill with no such component has transfer 0. A donor with another",
        "  tokenizer than the host is probed on text that donor generated.",
        f"- drift: KL from the frozen base on general text. The request's tolerance is {tolerance} nats.",
        "- objective: mean transfer over the skills minus 10 points per nat of drift above the tolerance.",
        "A skill with a donor that is no better than the base on the text has no gap, so its transfer is 0.",
        "Transfer on 128 held-out texts has noise, so do not chase differences under about 0.03.",
        "",
        "How to work: read atlas() for models, sizes, and structure, and catalog() for mined parts. Pick a",
        "host and components, write a program, call compile_and_practice, read the probe and errors,",
        "and revise. When you are done, call finish with the program text and a short rationale.",
        "",
        SHORT_LANGUAGE,
    ]
    return "\n".join(lines)


def response_calls(response):
    calls = []
    for item in response.get("output", []) or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            calls.append(item)
    return calls


def response_text(response):
    texts = []
    for item in response.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                texts.append(part.get("text", ""))
    return "\n".join(texts)


def run_loop(request, toolbox, client, archive, max_turns=DEFAULT_MAX_TURNS, max_tokens=DEFAULT_MAX_TOKENS):
    """Run the composer until finish, the turn cap, or the token budget. Return a summary dict."""
    instructions = system_prompt(request, max_turns, max_tokens)
    items = [{"role": "user", "content": "Start. The request is: " + json.dumps(request)}]
    tokens_used = 0
    turns = 0
    stop_reason = None
    errors = []
    archive.log(f"request {request['name']}: start, model {client.model}")

    while True:
        if toolbox.finished is not None:
            stop_reason = "finish"
            break
        if turns >= max_turns:
            stop_reason = f"turn cap ({max_turns})"
            break
        if tokens_used >= max_tokens:
            stop_reason = f"token budget ({tokens_used} of {max_tokens})"
            break
        turns += 1
        body = {
            "model": client.model,
            "instructions": instructions,
            "input": items,
            "tools": tool_specs(),
            "reasoning": {"effort": client.reasoning_effort},
        }
        archive.write("request", body)
        try:
            response = client.create(body)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            errors.append(message)
            archive.write("api_error", {"error": message})
            archive.log("API error, stopping: " + message[:300])
            stop_reason = "API error"
            break
        archive.write("response", response)
        usage = response.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        if tokens <= 0:
            # No usage in the reply: count about 4 characters per token, so the
            # token budget still stops the loop.
            tokens = (len(json.dumps(body)) + len(json.dumps(response.get("output") or []))) // 4
        tokens_used += tokens

        for item in response.get("output", []) or []:
            items.append(item)
        calls = response_calls(response)
        text = response_text(response)
        if text:
            archive.log("model says: " + text[:500])
        if len(calls) == 0:
            items.append({"role": "user", "content": "Use a tool. When you are done, call finish."})
            continue

        for call in calls:
            name = call.get("name")
            args = call.get("arguments")
            result = None
            try:
                args = json.loads(call.get("arguments") or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as error:
                result = {"ok": False, "error": f"bad arguments: {error}"}
            if result is None:
                try:
                    result = toolbox.call(name, args)
                except Exception as error:
                    result = {"ok": False, "error": f"the tool failed: {type(error).__name__}: {str(error)[:1000]}"}
            archive.write("tool_call", {"name": name, "call_id": call.get("call_id"), "arguments": args})
            archive.write("tool_result", {"name": name, "call_id": call.get("call_id"), "result": result})
            if name == "compile_and_practice":
                archive.log(f"compile_and_practice: ok={result.get('ok')} stage={result.get('stage')} "
                            f"objective={result.get('practice_objective')} budget_left={result.get('budget_left')}")
            items.append({"type": "function_call_output", "call_id": call.get("call_id"),
                          "output": json.dumps(result)})
            if toolbox.finished is not None:
                break
        turns_left = max_turns - turns
        if toolbox.finished is None and 0 < turns_left <= 3:
            items.append({"role": "user", "content": f"{turns_left} turns left. Call finish soon."})

    if toolbox.finished is None:
        toolbox.fallback_finish(f"the loop stopped: {stop_reason}")
    summary = {
        "request": request["name"],
        "stop_reason": stop_reason,
        "turns": turns,
        "tokens_used": tokens_used,
        "compile_budget": toolbox.budget,
        "compile_budget_used": toolbox.budget_used,
        "errors": errors,
        "final": toolbox.finished,
        "practiced": list(toolbox.practiced.values()),
        "archive_dir": archive.dir,
    }
    archive.write_named("final.json", summary)
    archive.log(f"stop: {stop_reason}; compiles {toolbox.budget_used} of {toolbox.budget}; tokens {tokens_used}")
    return summary


def run_request(request_path, practice_fn, root=REPO_ROOT, client=None, runs_root=None, stamp=None):
    """Load a request, build the tools and the client, and run the loop."""
    request = load_request(request_path)
    agent_options = request.get("agent", {})
    if client is None:
        key = read_key(os.path.join(root, ".env"))
        client = OpenAIClient(key, model=agent_options.get("model", DEFAULT_MODEL),
                              reasoning_effort=agent_options.get("reasoning_effort", "high"))
    sources = Sources(root)
    toolbox = Toolbox(request, sources, practice_fn, read_language_text(root))
    if runs_root is None:
        runs_root = RUNS_ROOT
    archive = Archive(runs_root, request["name"], client.secrets(), stamp=stamp)
    return run_loop(request, toolbox, client, archive,
                    max_turns=agent_options.get("max_turns", DEFAULT_MAX_TURNS),
                    max_tokens=agent_options.get("max_tokens", DEFAULT_MAX_TOKENS))
