"""Interfaces for memory (past.dev), literature (Exa) and LLMs, with null versions.

The integration agents implement these classes against the docstrings here:
- Memory: wsp/swarm/memory.py, PastMemory(config)            (config memory = "past")
- Literature: wsp/swarm/literature.py, ExaLiterature(config, mode)   (literature = "replay" or "live")
- LLM: wsp/swarm/llm.py, MockLLM, ModalVLLM, OpenAICompatLLM, AnthropicLLM, each (config)

Every network-backed implementation fails soft: it logs, degrades and keeps
the run going. None of them ever touches the eval or compute budget.

make_memory_listener(memory) is the env listener that turns log rows into
memory.record(...) calls. Memory implementations do not need to write it.
"""


class Memory:
    """Run-scoped memory of lessons (spec 9.2, pastdev.md section 6)."""

    enabled = False

    def start_run(self, run_id, roles):
        """Open the memory for one run.

        roles: role names such as ["analyst", "planner-1", "planner-2", "judge"].
        The implementation creates the run's audience and one identity per
        role (for past.dev: audience "<run>-lessons", identities "<run>:<role>"),
        then asserts a cold start: a recall must come back empty.
        Raises RuntimeError if that recall is not empty (memory from another
        run would leak in). If the service is unreachable it degrades to off,
        logs memory_degraded and returns. Returns None.
        """
        raise NotImplementedError

    def record(self, kind, text, meta=None, round_index=None):
        """Queue one lesson for the next batch. Never blocks, never raises.

        kind: "step", "probe", "collapse", "verdict", "capmap" or "plan".
        text: one self-contained plain-English sentence or short paragraph,
            with the numbers in it (score, fit_r2).
        meta: a dict for bookkeeping (program_hash, token, pattern, status, ...).
        round_index: the swarm round, or None.
        Ids are made by the implementation as "<run>:<kind>:<n>".
        """
        raise NotImplementedError

    def flush(self, timeout_s=30):
        """Send the queued batch and wait (up to timeout_s) until earlier sends are searchable.

        Called once at the start of each round, before planners recall.
        Returns {"ok": bool, "lag": bool, "pending": int}: lag is True when the
        wait timed out (the caller logs memory_lag), pending is the number of
        records not yet searchable. Never raises.
        """
        raise NotImplementedError

    def recall(self, role, query, level="low", limit=20):
        """Return lessons for one role, best first, from this run only.

        role: a role name passed to start_run, for example "planner-2".
        level: "low" or "medium" retrieval depth.
        Returns a list of {"text", "source_id", "occurred_at", "kind"}.
        Superseded results are left out. Returns [] on any failure.
        """
        raise NotImplementedError

    def end_run(self, cleanup=False):
        """Close the run. With cleanup=True, delete every point this run wrote. Never raises."""
        raise NotImplementedError


class Literature:
    """A frozen literature prior plus cached lookups (spec 9.3, exa.md section 5)."""

    mode = "off"

    def priors(self):
        """Return the frozen priors pack, built before eval 1.

        Returns {"sha": str, "components": {part_id: [cite]}, "topics": [claim]},
        where cite = {"url", "title", "quote"}. "sha" is the sha256 of the pack
        file and goes into the run manifest. In replay mode this never calls
        the network.
        """
        raise NotImplementedError

    def search(self, role, query, k=3):
        """At most k citations for one hypothesis: a list of {"url", "title", "quote"}.

        Uses the cache (replay mode never calls the network). Never sends
        held-out text, never touches the budget, never raises: returns [] on
        any failure.
        """
        raise NotImplementedError

    def known_url(self, url):
        """True when url is in the priors pack or the cache (the judge logs other citations as ungrounded)."""
        raise NotImplementedError


class LLM:
    """A chat model for planners and analysts (spec 9.4).

    messages is a list of {"role": "system" | "user" | "assistant", "content": str}.
    """

    name = "null"

    def complete(self, messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        """Return the reply text (str).

        schema: an optional JSON schema the reply should follow; the caller
        still parses and validates the text. seed makes sampling repeatable
        where the backend allows it. On a network failure it returns "" and
        logs the error, so the caller treats it as a reply with no proposals.
        """
        raise NotImplementedError

    def complete_many(self, list_of_messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        """Return one reply text per message list, in input order (one batch call where possible)."""
        raise NotImplementedError


class NullMemory(Memory):
    """Memory off: remembers nothing, recalls nothing."""

    enabled = False

    def __init__(self, config=None):
        self.config = config
        self.run_id = None

    def start_run(self, run_id, roles):
        self.run_id = run_id
        return None

    def record(self, kind, text, meta=None, round_index=None):
        return None

    def flush(self, timeout_s=30):
        return {"ok": True, "lag": False, "pending": 0}

    def recall(self, role, query, level="low", limit=20):
        return []

    def end_run(self, cleanup=False):
        return None


class NullLiterature(Literature):
    """Literature off: an empty, fixed priors pack and no search."""

    mode = "off"

    def __init__(self, config=None, mode="off"):
        self.config = config

    def priors(self):
        return {"sha": "", "components": {}, "topics": []}

    def search(self, role, query, k=3):
        return []

    def known_url(self, url):
        return False


class LLMNotConfigured(RuntimeError):
    """Raised when code asks for an LLM in a run configured with llm = "none"."""


class NullLLM(LLM):
    """No LLM. Any call raises, so a scripted run can never call a model by accident."""

    name = "null"

    def __init__(self, config=None):
        self.config = config

    def complete(self, messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        raise LLMNotConfigured("this run has llm = 'none'; set config llm to use a model")

    def complete_many(self, list_of_messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        raise LLMNotConfigured("this run has llm = 'none'; set config llm to use a model")


# The env listener for memory.

def format_score(value):
    if value is None:
        return "none"
    return "%.3f" % value


def format_fits(fit_r2):
    if not fit_r2:
        return "no grafts"
    parts = []
    for key in sorted(fit_r2):
        value = fit_r2[key]
        if value is None:
            parts.append(key + " skipped")
        else:
            parts.append(key + " " + "%.2f" % value)
    return ", ".join(parts)


def step_sentence(row):
    text = ("Eval " + str(row.get("token")) + " by " + str(row.get("agent")) + " on the " + str(row.get("split")) +
            " split: " + str(row.get("describe")) + ". Status " + str(row.get("status")) +
            ", combined score " + format_score(row.get("combined")) +
            ", fit_r2 " + format_fits(row.get("fit_r2")) + ".")
    if row.get("hypothesis"):
        text += " Hypothesis: " + str(row["hypothesis"])
    return text


def collapse_sentence(row):
    scores = row.get("scores") or {}
    parts = []
    for task in sorted(scores):
        parts.append(task + " " + "%.3f" % scores[task])
    return ("Collapse: " + str(row.get("describe")) + " fell below half its base score (" + ", ".join(parts) +
            ") with fit_r2 " + format_fits(row.get("fit_r2")) + ". Lesson: avoid patterns " +
            ", ".join((row.get("extra") or {}).get("patterns", [])) + ".")


def probe_sentence(row):
    return ("Probe by " + str(row.get("agent")) + ": " + str(row.get("describe")) + " has fit_r2 " +
            format_fits(row.get("fit_r2")) + " before any heal (status " + str(row.get("status")) + ").")


def row_meta(row):
    extra = row.get("extra") or {}
    return {"program_hash": row.get("program_hash"), "token": row.get("token"), "status": row.get("status"),
            "split": row.get("split"), "combined": row.get("combined"), "patterns": list(extra.get("patterns", []))}


def make_memory_listener(memory):
    """Return fn(row) for env.listeners that records each log row in memory.

    Eval rows on the search split become "step" records, plus a "collapse"
    record when the program collapsed. Probe rows become "probe" records.
    Judge verdict and analyst map events become "verdict" and "capmap".
    Held-out rows are never recorded, so held-out results cannot reach planners.
    """

    def listener(row):
        row_type = row.get("type")
        round_index = row.get("round")
        if row_type == "eval":
            if row.get("split") != "search":
                return
            if row.get("status") in ("invalid", "budget_exhausted"):
                return
            memory.record("step", step_sentence(row), row_meta(row), round_index)
            if row.get("status") == "collapsed":
                memory.record("collapse", collapse_sentence(row), row_meta(row), round_index)
        elif row_type == "probe":
            if row.get("status") not in ("ok",):
                return
            memory.record("probe", probe_sentence(row), row_meta(row), round_index)
        elif row_type == "event":
            if row.get("kind") == "judge_verdict":
                memory.record("verdict", row.get("text", ""), {"kind": row.get("kind")}, round_index)
            elif row.get("kind") == "analyst_map":
                memory.record("capmap", row.get("text", ""), {"kind": row.get("kind")}, round_index)

    return listener
