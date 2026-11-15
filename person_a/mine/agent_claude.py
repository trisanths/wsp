# Claude variant of the v2 mining agent. It uses the same tools, prompt, skills and gate as agent.py,
# but the Claude Agent SDK runs the agent loop. The SDK starts the `claude` CLI with its own login, so
# the run uses the Claude plan of the person who is logged in, not the OpenAI API.
#   .venv/bin/python mine/run.py --model <short_name> --llm claude
# Model: env WSP_CLAUDE_MODEL (default claude-opus-5-5). Needs `pip install claude-agent-sdk` and a
# logged-in `claude` CLI.
import asyncio
import json
import os
import time

from dotenv import load_dotenv

import agent as base
import common
from shared_memory import SharedMemory

CLAUDE_MODEL = os.environ.get("WSP_CLAUDE_MODEL", "claude-opus-5-5")
MAX_TURNS = 400
SERVER = "wsp"


def clean_env():
    # A run started from inside a Claude Code session inherits that session's variables. Remove them,
    # so the child CLI uses its own login and does not attach to the parent session.
    for k in list(os.environ.keys()):
        if k.startswith("CLAUDE_CODE_") or k.startswith("CLAUDE_AGENT_SDK") or k in [
                "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN", "CLAUDE_PREVIEW_CLASSIFIER_FLOOR"]:
            os.environ.pop(k, None)


def run(short, budget_usd=25.0, max_hours=3.0, max_screens=60, use_notes=True, effort="medium",
        person=None, focus=None):
    return asyncio.run(_run(short, budget_usd, max_hours, max_screens, use_notes, effort, person, focus))


async def _run(short, budget_usd, max_hours, max_screens, use_notes, effort, person, focus):
    import claude_agent_sdk as sdk

    load_dotenv(os.path.join(common.ROOT, ".env"), override=True)
    clean_env()
    run_id = "run-" + time.strftime("%Y%m%d-%H%M%S")
    lane = os.environ.get("WSP_LANE", "")
    if lane != "":
        run_id = run_id + "-lane" + lane
    person = person or os.environ.get("WSP_PERSON") or "unknown"
    memory = SharedMemory(person, short, run_id)
    session = base.Session(short, budget_usd, max_hours, max_screens, memory)
    miner = session.miner
    structure = miner.structure()
    miner.base()
    log_dir = common.logs_dir(short)
    transcript_path = os.path.join(log_dir, run_id + ".jsonl")
    system_text = base.instructions(miner, use_notes)
    # agent.py words its limits in OpenAI dollars; here the limits are time and screens
    system_text = system_text + ("\nLimits for this run: " + str(round(max_hours, 2)) + " hours of wall clock and "
                                 + str(max_screens) + " screens. There is no dollar budget.\n")
    if focus:
        system_text = system_text + ("\nYour focus in this run (other agents are covering the rest in parallel "
                                     "and sharing results through memory): " + focus + "\n")
    if memory.enabled:
        arch = structure.get("model_class", "") + " " + miner.row.get("architecture", "")
        recalled = memory.recall_text("parts found in " + arch + " " + miner.row.get("kind", "") + " models; "
                                      "what worked and failed in weight-space dissection")
        if recalled:
            system_text = system_text + "\nShared memory from teammates' runs (past.dev):\n" + recalled + "\n"

    state = {"summary": None, "finish_refused": 0, "calls": 0}
    lock = asyncio.Lock()  # the session is not thread-safe; run one tool at a time

    def make_handler(name):
        async def handler(args):
            args = args or {}
            if name == "finish":
                minutes_left = (session.deadline - time.time()) / 60.0
                if state["finish_refused"] < 6 and session.screens < max_screens and minutes_left > 10:
                    state["finish_refused"] += 1
                    result = {"error": "Not yet: " + str(int(minutes_left)) + " minutes and "
                              + str(max_screens - session.screens) + " screens are left. Keep mining: try layers, "
                              "part kinds and skills you have not covered yet (check recall_shared for what "
                              "teammates already found, and avoid repeating it), define new skills from what the "
                              "weights suggest, and screen more candidates."}
                else:
                    state["summary"] = args.get("summary", "")
                    result = {"ok": True, "note": "Run finished. Reply with one short line and stop."}
            elif time.time() >= session.deadline:
                result = {"error": "Time limit reached. Call finish now with your summary."}
            else:
                async with lock:
                    try:
                        result = await asyncio.to_thread(session.tool, name, args)
                    except Exception as e:
                        result = {"error": repr(e)[:500]}
            state["calls"] += 1
            text = json.dumps(result, default=str)
            print("[" + short + "][tool]", name, text[:240], flush=True)
            common.append_jsonl(transcript_path, {"tool": name, "args": args, "result": result})
            return {"content": [{"type": "text", "text": text[:15000]}]}
        return handler

    tools = []
    for spec in base.TOOLS:
        schema = spec.get("parameters") or {"type": "object", "properties": {}}
        tools.append(sdk.tool(spec["name"], spec["description"], schema)(make_handler(spec["name"])))
    server = sdk.create_sdk_mcp_server(SERVER, tools=tools)
    allowed = ["mcp__" + SERVER + "__" + spec["name"] for spec in base.TOOLS]
    options = sdk.ClaudeAgentOptions(
        model=CLAUDE_MODEL, system_prompt=system_text, mcp_servers={SERVER: server},
        tools=[], allowed_tools=allowed, setting_sources=[], max_turns=MAX_TURNS,
        effort=effort, cwd=log_dir, permission_mode="default")

    usage = {"turns": 0, "claude_equiv_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_input_tokens": 0}
    stop_reason = None
    lessons = []
    print("[" + short + "] run", run_id, "GPU", miner.gpu, "revision", miner.revision[:10], "| llm", CLAUDE_MODEL,
          "| shared memory", "on" if memory.enabled else "off", flush=True)

    async def drain(client):
        text_out = []
        async for msg in client.receive_response():
            if isinstance(msg, sdk.AssistantMessage):
                for block in msg.content:
                    if isinstance(block, sdk.TextBlock) and block.text.strip():
                        print("[" + short + "][agent]", block.text[:300], flush=True)
                        common.append_jsonl(transcript_path, {"agent": block.text})
                        text_out.append(block.text)
            elif isinstance(msg, sdk.ResultMessage):
                usage["turns"] += int(getattr(msg, "num_turns", 0) or 0)
                usage["claude_equiv_usd"] += float(getattr(msg, "total_cost_usd", 0) or 0)
                u = getattr(msg, "usage", None) or {}
                for k in ["input_tokens", "output_tokens", "cache_read_input_tokens"]:
                    usage[k] += int(u.get(k, 0) or 0)
                if getattr(msg, "is_error", False):
                    print("[" + short + "] result error:", str(getattr(msg, "result", ""))[:300], flush=True)
        return "\n".join(text_out)

    try:
        async with sdk.ClaudeSDKClient(options=options) as client:
            await client.query("Begin dissecting " + miner.hf_id + ".")
            nudges = 0
            while True:
                calls_before = state["calls"]
                await drain(client)
                if state["summary"] is not None:
                    stop_reason = "finished"
                    break
                if time.time() >= session.deadline:
                    stop_reason = "time limit reached"
                    break
                if session.screens >= max_screens:
                    stop_reason = "screen budget used up"
                    break
                if usage["turns"] >= MAX_TURNS:
                    stop_reason = "turn cap"
                    break
                if state["calls"] == calls_before:
                    nudges += 1
                    if nudges > 3:
                        stop_reason = "no tool call"
                        break
                await client.query("Continue mining with the tools. Call finish only when you are done.")
            try:
                await client.query("The run is over. Write 3 to 6 short, concrete lessons for future runs on this "
                                   "and other models (what in the weights predicted real parts, what failed, with "
                                   "numbers). Reply with only JSON {\"lessons\": [\"...\"]}. Do not call tools.")
                out = await drain(client)
                lessons = json.loads(out[out.find("{"):out.rfind("}") + 1])["lessons"]
            except Exception as e:
                lessons = ["(lesson summary failed: " + repr(e)[:120] + ")"]
    finally:
        if session.box is not None:
            session.box.stop()

    for lesson in lessons:
        common.append_jsonl(os.path.join(log_dir, "notes.jsonl"), {"run_id": run_id, "model": miner.hf_id, "lesson": lesson})
        memory.record("lesson", "Model " + short + " (" + miner.row.get("architecture", "") + "): " + lesson)
    memory.flush()
    gpu_cost = miner.modal_seconds / 3600.0 * base.GPU_PRICE_PER_HOUR.get(miner.gpu, 5.0)
    n_parts = 0
    if os.path.exists(common.catalog_path(short)):
        with open(common.catalog_path(short)) as f:
            for line in f:
                if line.strip():
                    n_parts += 1
    cost = {"run_id": run_id, "person": person, "llm": CLAUDE_MODEL, "openai_usd_est": 0.0,
            "claude_equiv_usd": round(usage["claude_equiv_usd"], 2),
            "modal_gpu_usd_est": round(gpu_cost, 2), "modal_busy_seconds": round(miner.modal_seconds),
            "gpu": miner.gpu, "note": "claude_equiv_usd is the API-price equivalent the CLI reports; the run uses "
            "the Claude plan. Modal estimate counts busy call time only (not idle keep-warm or first load)",
            "usage": usage, "turns": usage["turns"], "python_runs": session.py_runs, "screens": session.screens,
            "stop_reason": stop_reason, "summary": state["summary"], "parts_in_catalog": n_parts,
            "shared_memory": {"enabled": memory.enabled, "records": memory.count}}
    common.append_jsonl(os.path.join(log_dir, "costs.jsonl"), cost)
    common.log_event(short, "agent_run", {"run_id": run_id, "max_hours": max_hours, "effort": effort,
                                          "llm": CLAUDE_MODEL}, cost)
    print("[" + short + "] DONE", json.dumps(cost, default=str)[:1500], flush=True)
    return cost
