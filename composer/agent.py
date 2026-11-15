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
from composer.skills import SET_NOTES, SKILLS, practice_sets
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
- `tap DONOR layer L at N` lets host layer N read a donor's layer L states; it needs `tap_train`.
- `splice base a sources a=M1,b=M2 plan a:0..13, b:14..27` builds one plain model from layer segments.
- `from M` plus `carve target 1.5b select influence` keeps a source's most useful layers to fit a size.
- `mask` and `tv` apply catalog deletion masks and task vectors to the host.
- `lora rank R` adds a host LoRA trained by heal. `heal steps S data D lr X` trains the glue and LoRAs.
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
    return request


def system_prompt(request, max_turns, max_tokens):
    constraints = request["constraints"]
    skills = request["skills"]
    skill_lines = []
    for skill in skills:
        sets = []
        for name in practice_sets([skill]):
            sets.append(SET_NOTES[name])
        skill_lines.append(f"- {skill} (weight {skills[skill]}): practice on " + "; ".join(sets))

    def cap(name):
        value = constraints.get(name)
        if value is None:
            return "no cap"
        return str(value)

    lines = [
        "You are the composer for wsp, a language and compiler that writes new models by composing",
        "proven components of existing open models: grafted donor blocks and mined parts with trained",
        "stitches, taps, splices, carves, deletion masks, task vectors, and a short heal.",
        "",
        "Goal: " + request.get("description", request["name"]),
        "Objective: " + request.get("objective", "weighted mean of skill scores") +
        ". The score is the weighted mean of the skill scores below.",
        "",
        "Skills:",
    ]
    lines += skill_lines
    lines += [
        "",
        "Rules:",
        "- Compose components only. You never write, set, or edit raw weight values, and no tool lets you.",
        "  Your output is a program in the wsp language.",
        "- Heal and stitch (and tap_train for taps) are the only training. Each stage has at most "
        + cap("heal_steps_cap") + " steps.",
        "- A host with only a LoRA and heal is the fine-tuning baseline and is rejected. The final report",
        "  compares your program with that LoRA, with the base alone, and with each source model.",
        "- Constraints: total params at most " + cap("max_total_params_b") + "b; params run per token at most "
        + cap("max_run_params_b") + "b. compile_and_practice checks an estimate before compiling and the",
        "  measured size after.",
        "",
        "Budgets:",
        f"- Compile budget: {constraints.get('compile_budget', 10)} compiles. A parse error, a constraint",
        "  rejection, or a repeat of a program already practiced costs nothing.",
        f"- At most {max_turns} model turns and {max_tokens} tokens. If you stop without finish, the loop",
        "  submits your best practiced program.",
        "- A compile takes minutes to an hour on one H200, so think before you compile.",
        "",
        "Held-out tests are unseen. They run once, after finish, on test splits with fixed settings.",
        "Practice scores come from small train-split sets (100 to 200 items per skill), so expect noise",
        "of a few points between programs.",
        "",
        "How to work: read atlas() for models, sizes, and scores, and catalog() for mined parts. Pick a",
        "host and components, write a program, call compile_and_practice, read the scores and errors,",
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
