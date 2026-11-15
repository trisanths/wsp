"""Live smoke test of wsp/swarm/memory.py against past.dev.

It starts a run with a unique id, records a step, a collapse and a verdict,
flushes, measures how long until a recall returns them, recalls as a planner
and as the judge, proves isolation (a second run and an outside identity
recall nothing), then runs end_run(cleanup=True) and confirms the run's
memory is gone. Last it reuses the run id: the state file refuses it before
any request, and without the state file (as if outputs/ were wiped) the
first batch is refused by past.dev and memory degrades. It prints one JSON
report and exits 1 if a check fails.

Cost: about 3 ingest credits and at most 23 recalls (one credit each); the
latency loop stops after 12 recalls. The refused batch of the reuse check
costs nothing. Credits used are read from GET /usage (free) before and after.

Needs PAST_API_KEY in the environment or in .env. The key is never printed.

Example:
    .venv/Scripts/python scripts/memory_smoke.py
"""

import json
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wsp.swarm.config import load_dotenv  # noqa: E402
from wsp.swarm.memory import MemoryLeakError, PastMemory  # noqa: E402

ROLES = ["analyst", "planner-1", "planner-2", "judge"]
PLANNER_QUERY = "what collapsed, and which graft had the best combined score"
JUDGE_QUERY = "collapses and frontier changes in the last rounds"
MAX_LATENCY_RECALLS = 12
LATENCY_PAUSE_S = 1.5

STEP_TEXT = ("Eval 1 by planner-1 on the search split: graft Qwen2.5-Math layer 14 into host layer 8 with "
             "300 heal steps. Status ok, combined score 0.412, fit_r2 0.61.")
COLLAPSE_TEXT = ("Collapse: grafting Qwen2.5-Math layer 27 into host layer 15 with no heal fell below half its "
                 "base score (gsm8k 0.031, if_guard 0.120) with fit_r2 0.08. Lesson: avoid late-layer grafts "
                 "without heal.")
VERDICT_TEXT = ("Round 0 frontier top-1: Qwen2.5-Math layer 14 grafted at host layer 8, combined score 0.412. "
                "Killed: late-layer grafts without heal (1 collapse). Budget used 2 of 60 evals.")


def read_usage(memory):
    """Credits spent this month (organization) and by this project, from the free GET /usage."""
    status, data = memory.call("GET", "/usage")
    if status != 200 or not isinstance(data, dict):
        return None
    balance = data.get("balance") or {}
    project = data.get("project") or {}
    return {"spent": balance.get("spent"), "left": balance.get("left"), "project_credits": project.get("credits")}


def own_ids(lessons, sent_ids):
    out = []
    for lesson in lessons:
        if lesson["source_id"] in sent_ids:
            out.append(lesson["source_id"])
    return out


def describe(lessons):
    out = []
    for lesson in lessons:
        out.append(lesson["kind"] + "/" + lesson["artifact"] + ": " + lesson["text"][:90])
    return out


def add(checks, name, ok, detail):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ": " + json.dumps(detail, default=str))


def run_smoke():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    if os.environ.get("PAST_API_KEY", "") == "":
        print("PAST_API_KEY is not set (checked the environment and .env); nothing was run")
        return 1
    print("PAST_API_KEY: set (value hidden)")

    stamp = str(int(time.time()))
    run_id = "smoke-" + stamp
    config = {"outputs_dir": os.path.join(REPO_ROOT, "outputs"), "memory_verify_cleanup": True,
              "memory_poll_s": 1.0}
    checks = []
    report = {"run_id": run_id}

    probe = PastMemory(config)
    usage_before = read_usage(probe)
    report["usage_before"] = usage_before

    memory = PastMemory(config)
    other = PastMemory(config)
    reused = None
    try:
        # 1. Cold start.
        began = time.time()
        try:
            memory.start_run(run_id, ROLES)
            leak = None
        except MemoryLeakError as error:
            leak = str(error)
        add(checks, "cold_start", leak is None and memory.enabled,
            {"seconds": round(time.time() - began, 2), "audience": memory.audience,
             "membership_ready": memory.membership_ready, "leak": leak, "degraded": memory.degraded_reason})
        if not memory.enabled:
            return finish(report, checks, memory, other, probe, usage_before)

        # 2. Record and send.
        memory.record("step", STEP_TEXT, {"token": 1, "status": "ok", "combined": 0.412}, 0)
        memory.record("collapse", COLLAPSE_TEXT, {"token": 2, "status": "collapsed"}, 0)
        memory.record("verdict", VERDICT_TEXT, {"kind": "judge_verdict"}, 0)
        sent_at = time.time()
        first = memory.flush(timeout_s=0)
        add(checks, "batch_sent", len(memory.sent_ids) == 3 and memory.enabled,
            {"flush": first, "ids": list(memory.sent_ids), "ingestions": list(memory.ingestion_ids)})
        if len(memory.sent_ids) == 0:
            return finish(report, checks, memory, other, probe, usage_before)

        # 3. Ingest-to-recall latency: recall until this run's points come back.
        text_latency = None
        latency_recalls = 0
        while latency_recalls < MAX_LATENCY_RECALLS:
            lessons = memory.recall("planner-1", PLANNER_QUERY)
            latency_recalls += 1
            if len(own_ids(lessons, memory.sent_ids)) > 0:
                text_latency = round(time.time() - sent_at, 2)
                break
            time.sleep(LATENCY_PAUSE_S)
        add(checks, "recall_latency", text_latency is not None,
            {"send_to_first_recall_s": text_latency, "recalls_used": latency_recalls})

        # 4. Wait for full processing (derived claims and observations).
        waited = memory.flush(timeout_s=60)
        complete_latency = round(time.time() - sent_at, 2)
        add(checks, "ingestion_completed", waited["ok"] and not waited["lag"],
            {"flush": waited, "send_to_completed_s": complete_latency})

        # 5. Recall as a planner and as the judge.
        planner = memory.recall("planner-1", PLANNER_QUERY)
        planner_own = own_ids(planner, memory.sent_ids)
        planner_kinds = []
        for lesson in planner:
            if lesson["kind"] not in planner_kinds:
                planner_kinds.append(lesson["kind"])
        add(checks, "planner_recall", len(planner) > 0 and len(planner_own) == len(planner),
            {"results": len(planner), "kinds": planner_kinds, "pending": memory.last_pending,
             "top": describe(planner[:6])})
        judge = memory.recall("judge", JUDGE_QUERY)
        judge_own = own_ids(judge, memory.sent_ids)
        add(checks, "judge_recall", len(judge) > 0 and len(judge_own) == len(judge),
            {"results": len(judge), "top": describe(judge[:4])})
        medium = memory.recall_raw(memory.identities["planner-2"], PLANNER_QUERY, "medium", 20)
        add(checks, "medium_level_accepted", medium is not None,
            {"results": None if medium is None else len(medium)})

        # 6. Isolation: a second run and an identity in no audience see nothing of this run.
        try:
            other.start_run(run_id + "-other", ROLES)
            other_leak = None
        except MemoryLeakError as error:
            other_leak = str(error)
        other_docs = None
        if other.enabled:
            other_docs = other.recall_raw(other.identities["planner-1"], PLANNER_QUERY, "low", 20)
        outsider_docs = memory.recall_raw("smoke-outsider-" + stamp, PLANNER_QUERY, "low", 20)
        add(checks, "isolation", other_leak is None and other_docs == [] and outsider_docs == [],
            {"other_run_cold_start_leak": other_leak, "other_run_results": other_docs,
             "outsider_results": outsider_docs, "other_audience": other.audience})
        other.end_run(cleanup=True)

        # 7. Cleanup, then confirm the run's memory is gone.
        ingestion_ids = list(memory.ingestion_ids)
        planner_identity = memory.identities["planner-1"]
        cleaned = memory.end_run(cleanup=True)
        after = memory.recall_raw(planner_identity, PLANNER_QUERY, "low", 20)
        deleted_counts = []
        for ingestion_id in ingestion_ids:
            status, data = memory.call("GET", "/ingest/" + ingestion_id)
            if status == 200 and isinstance(data, dict):
                deleted_counts.append(data.get("sources"))
        add(checks, "cleanup", cleaned["deleted"] == 3 and cleaned["verified_empty"] is True and
            cleaned["audience_deleted"] and after == [],
            {"end_run": cleaned, "recall_after_cleanup": after, "ingestion_sources": deleted_counts})
        report["latency"] = {"send_to_first_recall_s": text_latency, "send_to_completed_s": complete_latency}

        # 8. Reuse the run id. The state file refuses it before any request.
        local_error = None
        local = PastMemory(config)
        try:
            local.start_run(run_id, ROLES)
        except MemoryLeakError as error:
            local_error = str(error)
        # end_run touches nothing after a refusal; it cleans up if the refusal failed to happen.
        local.end_run(cleanup=True)
        add(checks, "reuse_refused_locally", local_error is not None and "was used before" in local_error,
            {"error": local_error})

        # Without the state file, past.dev itself refuses the first batch and memory degrades.
        no_state = dict(config)
        no_state["memory_state_path"] = ""
        reused = PastMemory(no_state)
        reuse_leak = None
        try:
            reused.start_run(run_id, ROLES)
        except MemoryLeakError as error:
            reuse_leak = str(error)
        reuse_started = reused.enabled
        reused.record("step", "Eval 1 after reusing the run id: combined score 0.100.", {"token": 9}, 0)
        reuse_flush = reused.flush(timeout_s=0)
        reason = reused.degraded_reason or ""
        add(checks, "reuse_degrades_at_first_batch", reuse_leak is None and reuse_started and
            not reused.enabled and "run id reused" in reason and len(reused.sent_ids) == 0,
            {"cold_start_leak": reuse_leak, "started": reuse_started, "audience_replaced": reused.audience_replaced,
             "membership_ready": reused.membership_ready, "flush": reuse_flush, "degraded": reason})
    finally:
        if not memory.ended:
            memory.end_run(cleanup=True)
        if other.run_id is not None and not other.ended:
            other.end_run(cleanup=True)
        if reused is not None and not reused.ended:
            report["reuse_end_run"] = reused.end_run(cleanup=True)
    return finish(report, checks, memory, other, probe, usage_before)


def finish(report, checks, memory, other, probe, usage_before):
    if not memory.ended:
        memory.end_run(cleanup=True)
    usage_after = read_usage(probe)
    report["usage_after"] = usage_after
    if usage_before and usage_after and usage_before.get("spent") is not None and usage_after.get("spent") is not None:
        report["credits_used"] = usage_after["spent"] - usage_before["spent"]
    report["stats"] = memory.stats
    report["degraded"] = memory.degraded_reason
    report["checks"] = checks
    print(json.dumps(report, indent=1, default=str))
    failed = 0
    for check in checks:
        if not check["ok"]:
            failed += 1
    if failed > 0 or len(checks) == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke())
