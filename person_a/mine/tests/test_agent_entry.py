# Test of the entry points that mine/run.py calls: agent.run (OpenAI) and agent_claude.run (Claude
# Agent SDK). run.py passes the same keywords to both, so both must accept them, and --open-domain
# must have the same effects in both.
#   .venv/bin/python mine/tests/test_agent_entry.py
# No engine call, no LLM call, and no SDK process: the test stops the Claude run at the point where
# the SDK options are built, and reads the system prompt and the miner state there.
import inspect
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import agent  # noqa: E402
import agent_claude  # noqa: E402
import common  # noqa: E402
import run  # noqa: E402


class Stop(Exception):
    pass


class FakeMiner:
    def __init__(self):
        self.state = {}
        self.row = {}
        self.hf_id = "org/Toy"
        self.revision = "rev-toy"
        self.gpu = "EngineL4"
        self.modal_seconds = 0.0

    def structure(self):
        return {"model_class": "Toy"}

    def base(self):
        return {}


class FakeSession:
    made = []

    def __init__(self, short, budget_usd, max_hours, max_screens, memory=None):
        self.miner = FakeMiner()
        self.box = None
        self.screens = 0
        self.py_runs = 0
        FakeSession.made.append(self)


class FakeMemory:
    enabled = False
    count = 0

    def __init__(self, person, short, run_id):
        pass


def fake_sdk(seen):
    def options(**kw):
        seen["options"] = kw
        raise Stop()  # the prompt and the state are final here; nothing of the SDK runs
    return types.SimpleNamespace(
        tool=lambda name, description, schema: (lambda handler: handler),
        create_sdk_mcp_server=lambda name, tools: object(),
        ClaudeAgentOptions=options)


def start_claude_run(**kw):
    # agent_claude.run up to the SDK options. Returns (system prompt, miner state).
    seen = {}
    real = (agent.Session, agent.instructions, agent_claude.SharedMemory, agent_claude.clean_env,
            agent_claude.load_dotenv, sys.modules.get("claude_agent_sdk"))
    agent.Session = FakeSession
    agent.instructions = lambda miner, use_notes: "BASE PROMPT\n"
    agent_claude.SharedMemory = FakeMemory
    agent_claude.clean_env = lambda: None
    agent_claude.load_dotenv = lambda *a, **k: None
    sys.modules["claude_agent_sdk"] = fake_sdk(seen)
    try:
        agent_claude.run("toy", **kw)
        raise AssertionError("the run must stop at the SDK options")
    except Stop:
        pass
    finally:
        (agent.Session, agent.instructions, agent_claude.SharedMemory, agent_claude.clean_env,
         agent_claude.load_dotenv) = real[:5]
        if real[5] is None:
            del sys.modules["claude_agent_sdk"]
        else:
            sys.modules["claude_agent_sdk"] = real[5]
    return seen["options"]["system_prompt"], FakeSession.made[-1].miner.state


def test_signatures():
    # both entry points take the same arguments, in the same order
    names = list(inspect.signature(agent.run).parameters)
    assert names == list(inspect.signature(agent_claude.run).parameters), names
    assert "open_domain" in names and inspect.signature(agent_claude.run).parameters["open_domain"].default is False


def test_run_py_reaches_both():
    # mine/run.py with each --llm, with and without --open-domain: the call must reach run()
    real_argv = sys.argv
    real_runs = (agent.run, agent_claude.run)
    real_sigs = [inspect.signature(f) for f in real_runs]
    reached = []

    def stand_in(which, signature):
        def fake(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)   # raises TypeError the way the real function would
            reached.append((which, bound.arguments.get("open_domain", False)))
            return {"stop_reason": "time limit reached"}
        return fake
    agent.run = stand_in("openai", real_sigs[0])
    agent_claude.run = stand_in("claude", real_sigs[1])
    try:
        for llm in ["openai", "claude"]:
            for extra in [[], ["--open-domain"]]:
                sys.argv = ["run.py", "--model", "zz_entry_test", "--llm", llm] + extra
                run.main()
    finally:
        sys.argv = real_argv
        agent.run, agent_claude.run = real_runs
    assert reached == [("openai", False), ("openai", True), ("claude", False), ("claude", True)], reached


def test_open_domain_effects():
    prompt, state = start_claude_run(open_domain=True, focus="layer 3")
    # the three effects of agent.run: the prompt text, the larger skill cap (skills.define reads it),
    # and the flag that Session.tool reads to refuse built-in targets
    assert prompt.count(agent.OPEN_DOMAIN_TEXT) == 1 and prompt.startswith("BASE PROMPT\n"), prompt
    assert prompt.index(agent.OPEN_DOMAIN_TEXT) < prompt.index("Your focus in this run"), prompt
    assert state == {"max_agent_skills": 16, "open_domain": True}, state
    prompt, state = start_claude_run()
    assert "BENCHMARK-AGNOSTIC" not in prompt and state == {}, (prompt, state)


def main():
    real_root = common.ROOT
    tmp = tempfile.mkdtemp(prefix="agent-entry-")
    common.ROOT = tmp
    try:
        test_signatures()
        test_run_py_reaches_both()
        test_open_domain_effects()
    finally:
        common.ROOT = real_root
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
    print("ok")
