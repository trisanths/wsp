"""Tests for wsp/swarm/memory.py (past.dev memory). CPU only, no network.

FakePast stands in for the past.dev API: it keeps audiences and data points
in dicts and answers recall with the points an identity may read. Like the
real service it refuses a deleted id (409 data-point-deleted) and a used
batch key with another body (409 idempotency-conflict). Setting
membership_lag makes every identity reach only "project". Tests that need a
particular failure queue one answer with fake.queue(...).
"""

import json
import logging

import pytest
import requests

from wsp.swarm import memory as memory_mod
from wsp.swarm.memory import MemoryLeakError, MemoryScopeError, PastMemory, audience_slug

PLACEHOLDER_KEY = "past_sk_placeholder_for_tests"
ROLES = ["analyst", "planner-1", "planner-2", "judge"]


class FakeResponse:
    def __init__(self, status, data=None, headers=None):
        self.status_code = status
        self.data = data
        self.headers = headers or {}

    def json(self):
        if self.data is None:
            raise ValueError("no body")
        return self.data


class FakePast:
    """A tiny in-memory past.dev. Records every call as (method, path, body)."""

    def __init__(self):
        self.calls = []
        self.queued = []
        self.audiences = {}
        self.points = {}
        self.poll_states = {}
        self.extra_docs = []
        self.ingestions = 0
        self.deleted = []
        self.keys = {}
        self.membership_lag = False

    def queue(self, method, path_start, response):
        """Answer the next matching call with response (a FakeResponse or an exception)."""
        self.queued.append((method, path_start, response))

    def request(self, method, url, headers=None, json=None, timeout=None):
        assert headers["Authorization"] == "Bearer " + PLACEHOLDER_KEY
        path = url.split("/api/v1", 1)[1]
        self.calls.append((method, path, json))
        for index in range(len(self.queued)):
            queued_method, path_start, response = self.queued[index]
            if queued_method == method and path.startswith(path_start):
                del self.queued[index]
                if isinstance(response, Exception):
                    raise response
                return response
        return self.answer(method, path, json)

    def reached(self, identity):
        names = ["project"]
        if self.membership_lag:
            return names
        for slug in self.audiences:
            if identity in self.audiences[slug]:
                names.append(slug)
        return names

    def answer(self, method, path, body):
        if method == "PUT" and path.startswith("/audiences/"):
            slug = path[len("/audiences/"):]
            created = slug not in self.audiences
            self.audiences[slug] = list(body["identities"])
            if created:
                return FakeResponse(201, {"slug": slug})
            return FakeResponse(200, {"slug": slug})
        if method == "POST" and path.endswith("/recompute"):
            return FakeResponse(200, {})
        if method == "GET" and path.startswith("/identities/"):
            identity = requests.utils.unquote(path[len("/identities/"):])
            return FakeResponse(200, {"identity": identity, "audiences": self.reached(identity)})
        if method == "POST" and path == "/recall":
            return FakeResponse(200, {"results": self.recall_docs(body["identity"])})
        if method == "POST" and path == "/ingest/batch":
            return self.ingest(body)
        if method == "GET" and path.startswith("/ingest/"):
            ingestion_id = path[len("/ingest/"):]
            states = self.poll_states.get(ingestion_id, [])
            state = "completed"
            if len(states) > 0:
                state = states.pop(0)
            return FakeResponse(200, {"status": state, "failure": None})
        if method == "POST" and path.endswith("/retry"):
            return FakeResponse(202, {"status": "queued"})
        if method == "POST" and path == "/data-points/delete":
            missing = []
            for source_id in body["ids"]:
                if source_id not in self.points:
                    missing.append(source_id)
            if len(missing) > 0:
                return FakeResponse(404, {"code": "data-point-not-found", "data": {"ids": missing}})
            for source_id in body["ids"]:
                del self.points[source_id]
                self.deleted.append(source_id)
            return FakeResponse(202, {"operationId": "op", "ids": body["ids"]})
        if method == "DELETE" and path.startswith("/data-points/"):
            source_id = requests.utils.unquote(path[len("/data-points/"):])
            if source_id not in self.points:
                return FakeResponse(404, {"code": "data-point-not-found"})
            del self.points[source_id]
            self.deleted.append(source_id)
            return FakeResponse(202, {"operationId": "op", "ids": [source_id]})
        if method == "DELETE" and path.startswith("/audiences/"):
            slug = path[len("/audiences/"):]
            if slug in self.audiences:
                del self.audiences[slug]
            return FakeResponse(204)
        if method == "DELETE" and path.startswith("/identities/"):
            return FakeResponse(204)
        return FakeResponse(404, {"code": "not-found"})

    def ingest(self, body):
        """POST /ingest/batch: replay a known key, refuse reuse, else store the points."""
        key = body["idempotencyKey"]
        if key in self.keys:
            if self.keys[key]["body"] == body:
                return FakeResponse(202, self.keys[key]["receipt"])
            return FakeResponse(409, {"code": "idempotency-conflict", "debugMessage": "key used with another body"})
        for item in body["items"]:
            if item["id"] in self.deleted:
                return FakeResponse(409, {"code": "data-point-deleted",
                                          "debugMessage": "a deleted id is never reused: " + item["id"]})
        self.ingestions += 1
        for item in body["items"]:
            self.points[item["id"]] = item
        receipt = {"ingestionId": "ing-" + str(self.ingestions), "status": "queued"}
        self.keys[key] = {"body": body, "receipt": receipt}
        return FakeResponse(202, receipt)

    def recall_docs(self, identity):
        reached = self.reached(identity)
        docs = []
        for source_id in sorted(self.points):
            item = self.points[source_id]
            audience = item.get("audience") or "project"
            if audience in reached:
                docs.append({"content": item["content"], "occurredAt": item.get("timestamp"),
                             "artifact": {"kind": "source"}, "sources": [{"sourceId": source_id}]})
        for doc in self.extra_docs:
            docs.append(doc)
        return docs

    def count(self, method, path_start):
        total = 0
        for call in self.calls:
            if call[0] == method and call[1].startswith(path_start):
                total += 1
        return total


class FakeTime:
    """A clock that only moves when the code sleeps."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += max(seconds, 0.5)


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("PAST_API_KEY", PLACEHOLDER_KEY)


def make_memory(tmp_path, fake, extra=None):
    config = {"outputs_dir": str(tmp_path)}
    if extra is not None:
        for name in extra:
            config[name] = extra[name]
    memory = PastMemory(config)
    clock = FakeTime()
    memory.http = fake
    memory.sleep = clock.sleep
    memory.clock = clock.clock
    return memory, clock


def started(tmp_path, fake, run_id="s0-swarm-1", extra=None):
    memory, clock = make_memory(tmp_path, fake, extra)
    memory.start_run(run_id, ROLES)
    assert memory.enabled
    return memory, clock


# Audience slug.

def test_audience_slug_is_valid_short_and_stable():
    run_ids = ["s0-swarm-1794698437123", "S0_Swarm Run #7", "x" * 120, "___", "r", "Ünïcode-ряд-42",
               "a" * 30 + "-" * 5 + "b"]
    for run_id in run_ids:
        slug = audience_slug(run_id)
        assert 2 <= len(slug) <= 40
        for ch in slug:
            assert ch in "abcdefghijklmnopqrstuvwxyz0123456789-"
        assert slug[0] != "-" and slug[-1] != "-"
        assert "--" not in slug
        assert audience_slug(run_id) == slug
    assert audience_slug("s0-swarm-1794698437123").startswith("s0-swarm-1794698437123-")
    # Two run ids that clean up to the same text still get different slugs.
    assert audience_slug("S0_swarm") != audience_slug("s0-swarm")
    assert audience_slug("x" * 120) != audience_slug("x" * 121)


def test_identities_and_ids_use_the_run_and_role(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    assert memory.identities["planner-2"] == "s0-swarm-1:planner-2"
    assert fake.audiences[memory.audience] == ["s0-swarm-1:analyst", "s0-swarm-1:judge",
                                               "s0-swarm-1:planner-1", "s0-swarm-1:planner-2"]
    state = json.loads((tmp_path / "runs" / "s0-swarm-1" / "memory.json").read_text(encoding="utf-8"))
    assert state["audience"] == audience_slug("s0-swarm-1")
    assert state["identities"]["judge"] == "s0-swarm-1:judge"


# Start of run: audience first, then the cold-start guard.

def test_start_run_creates_audience_before_recall_and_waits_for_membership(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    methods = []
    for call in fake.calls:
        methods.append(call[0] + " " + call[1].split("/")[1])
    assert methods == ["PUT audiences", "POST audiences", "GET identities", "POST recall"]
    assert fake.calls[1][1].endswith("/recompute")
    assert fake.calls[3][2]["identity"] == "s0-swarm-1:planner-1"
    assert memory.membership_ready


def test_cold_start_guard_raises_when_anything_comes_back(tmp_path, key):
    fake = FakePast()
    fake.points["old-global-note"] = {"id": "old-global-note", "content": "a project-visible note"}
    memory, clock = make_memory(tmp_path, fake)
    with pytest.raises(MemoryLeakError) as caught:
        memory.start_run("s0-swarm-2", ROLES)
    assert isinstance(caught.value, RuntimeError)
    assert "old-global-note" in str(caught.value)
    assert not memory.enabled
    assert fake.count("POST", "/ingest") == 0


def test_cold_start_guard_catches_a_reused_run_id(tmp_path, key):
    fake = FakePast()
    first, clock = started(tmp_path, fake, "s0-swarm-3")
    first.record("step", "Eval 1 scored 0.41.")
    first.flush()
    first.end_run(cleanup=False)
    # No state file for the second run, so the guard recall is what must catch the reuse.
    second, clock = make_memory(tmp_path, fake, {"memory_state_path": ""})
    with pytest.raises(MemoryLeakError) as caught:
        second.start_run("s0-swarm-3", ROLES)
    assert "s0-swarm-3:step:1" in str(caught.value)
    assert second.audience_replaced


def test_state_file_refuses_a_reused_run_id_before_any_request(tmp_path, key):
    fake = FakePast()
    first, clock = started(tmp_path, fake, "s0-swarm-6")
    first.record("step", "Eval 1 scored 0.41.")
    first.flush()
    first.end_run(cleanup=True)
    state_path = tmp_path / "runs" / "s0-swarm-6" / "memory.json"
    state_before = state_path.read_text(encoding="utf-8")
    calls_before = len(fake.calls)
    second, clock = make_memory(tmp_path, fake)
    with pytest.raises(MemoryLeakError) as caught:
        second.start_run("s0-swarm-6", ROLES)
    assert "was used before" in str(caught.value)
    assert not second.enabled
    second.record("step", "Eval 1 scored 0.10.")
    assert second.flush() == {"ok": False, "lag": False, "pending": 0}
    result = second.end_run(cleanup=True)
    assert result == {"deleted": 0, "audience_deleted": False, "verified_empty": None}
    assert len(fake.calls) == calls_before
    # The earlier run's file still lists what it sent, for a later cleanup.
    assert state_path.read_text(encoding="utf-8") == state_before
    assert json.loads(state_before)["sent_ids"] == ["s0-swarm-6:step:1"]


def test_state_file_of_an_unused_or_other_run_does_not_refuse(tmp_path, key):
    fake = FakePast()
    unused = tmp_path / "runs" / "s0-swarm-7" / "memory.json"
    unused.parent.mkdir(parents=True)
    unused.write_text(json.dumps({"run_id": "s0-swarm-7", "sent_ids": [], "ended": False}), encoding="utf-8")
    memory, clock = started(tmp_path, fake, "s0-swarm-7")
    assert memory.enabled
    shared = tmp_path / "shared-memory.json"
    shared.write_text(json.dumps({"run_id": "s0-swarm-other", "sent_ids": ["x"], "ended": True}), encoding="utf-8")
    other, clock = started(tmp_path, fake, "s0-swarm-8", {"memory_state_path": str(shared)})
    assert other.enabled


def test_replaced_audience_with_lagging_membership_does_not_enable(tmp_path, key):
    fake = FakePast()
    slug = audience_slug("probe-2")
    fake.audiences[slug] = ["probe-2:analyst", "probe-2:judge", "probe-2:planner-1", "probe-2:planner-2"]
    fake.points["probe-2:step:7"] = {"id": "probe-2:step:7", "content": "stale old-run lesson", "audience": slug}
    fake.membership_lag = True
    memory, clock = make_memory(tmp_path, fake, {"memory_state_path": ""})
    memory.start_run("probe-2", ROLES)
    assert memory.audience_replaced
    assert not memory.membership_ready
    assert not memory.enabled
    assert "cold start not proven" in memory.degraded_reason
    assert fake.count("POST", "/recall") == 0
    # Even once membership arrives, the stale lesson never reaches this run.
    fake.membership_lag = False
    memory.record("step", "Eval 1 scored 0.41.")
    assert memory.flush() == {"ok": False, "lag": False, "pending": 0}
    assert memory.recall("planner-1", "lesson") == []
    assert fake.count("POST", "/ingest/batch") == 0
    assert fake.count("POST", "/recall") == 0


def test_new_audience_with_lagging_membership_still_starts(tmp_path, key):
    fake = FakePast()
    fake.membership_lag = True
    memory, clock = make_memory(tmp_path, fake)
    memory.start_run("s0-swarm-9", ROLES)
    assert not memory.audience_replaced
    assert not memory.membership_ready
    assert memory.enabled
    kinds = []
    for event in memory.events:
        kinds.append(event["kind"])
    assert "memory_membership_lag" in kinds
    assert fake.count("POST", "/recall") == 1


def test_unreachable_service_degrades_instead_of_raising(tmp_path, key):
    fake = FakePast()
    for attempt in range(4):
        fake.queue("PUT", "/audiences/", requests.ConnectionError("down"))
    memory, clock = make_memory(tmp_path, fake)
    memory.start_run("s0-swarm-4", ROLES)
    assert not memory.enabled
    assert "failed after 4 tries" in memory.degraded_reason
    assert fake.count("POST", "/recall") == 0


def test_missing_key_degrades(tmp_path, monkeypatch):
    monkeypatch.delenv("PAST_API_KEY", raising=False)
    fake = FakePast()
    memory, clock = make_memory(tmp_path, fake)
    memory.start_run("s0-swarm-5", ROLES)
    assert not memory.enabled
    assert "PAST_API_KEY" in memory.degraded_reason
    assert len(fake.calls) == 0


# The audience guard.

def test_post_batch_refuses_items_without_the_run_audience(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    calls_before = len(fake.calls)
    good = {"id": "s0-swarm-1:step:9", "content": "x", "audience": memory.audience}
    missing = {"id": "s0-swarm-1:step:10", "content": "x"}
    project = {"id": "s0-swarm-1:step:11", "content": "x", "audience": "project"}
    other = {"id": "s0-swarm-1:step:12", "content": "x", "audience": "someone-else"}
    for bad in [missing, project, other]:
        with pytest.raises(MemoryScopeError):
            memory.post_batch([good, bad], "key")
    assert len(fake.calls) == calls_before


def test_record_before_start_run_is_refused(tmp_path, key):
    fake = FakePast()
    memory, clock = make_memory(tmp_path, fake)
    memory.record("step", "Eval 1 scored 0.41.")
    assert memory.buffer == []
    kinds = []
    for event in memory.events:
        kinds.append(event["kind"])
    assert "memory_scope_refused" in kinds


def test_flush_with_an_unscoped_item_sends_nothing_and_degrades(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    memory.record("step", "Eval 1 scored 0.41.")
    memory.buffer[0]["audience"] = None
    result = memory.flush()
    assert result == {"ok": False, "lag": False, "pending": 0}
    assert fake.count("POST", "/ingest/batch") == 0
    assert not memory.enabled
    assert "MemoryScopeError" in memory.degraded_reason


# record and flush.

def test_flush_sends_one_scoped_batch_with_an_idempotency_key(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    memory.record("step", "Eval 1 scored 0.41.", {"token": 1}, 0)
    memory.record("collapse", "Collapse: late overlap deletion fell to 0.02.", {"token": 1}, 0)
    memory.record("step", "Eval 2 scored 0.44.", {"token": 2}, 0)
    memory.record("nonsense", "dropped")
    memory.record("verdict", "   ")
    result = memory.flush()
    assert result == {"ok": True, "lag": False, "pending": 0}
    assert fake.count("POST", "/ingest/batch") == 1
    batch = None
    for call in fake.calls:
        if call[1] == "/ingest/batch":
            batch = call[2]
    assert batch["idempotencyKey"] == "s0-swarm-1:batch:1"
    ids = []
    for item in batch["items"]:
        ids.append(item["id"])
        assert item["audience"] == memory.audience
        assert item["timestamp"].endswith("Z") and "T" in item["timestamp"]
        assert item["metadata"]["run_id"] == "s0-swarm-1"
        assert item["metadata"]["round"] == 0
    assert ids == ["s0-swarm-1:step:1", "s0-swarm-1:collapse:1", "s0-swarm-1:step:2"]
    assert batch["items"][1]["metadata"]["meta"] == {"token": 1}
    assert memory.sent_ids == ids


def test_flush_reports_lag_on_timeout(tmp_path, key):
    fake = FakePast()
    fake.poll_states["ing-1"] = ["processing"] * 100
    memory, clock = started(tmp_path, fake)
    memory.record("step", "Eval 1 scored 0.41.")
    memory.record("step", "Eval 2 scored 0.44.")
    start = clock.now
    result = memory.flush(timeout_s=10)
    assert result == {"ok": True, "lag": True, "pending": 2}
    assert clock.now - start >= 10
    assert clock.now - start < 15
    # The lagging ingestion is still watched by the next flush.
    fake.poll_states["ing-1"] = []
    result = memory.flush(timeout_s=10)
    assert result == {"ok": True, "lag": False, "pending": 0}


def test_flush_retries_a_failed_ingestion_once(tmp_path, key):
    fake = FakePast()
    fake.poll_states["ing-1"] = ["failed", "completed"]
    memory, clock = started(tmp_path, fake)
    memory.record("step", "Eval 1 scored 0.41.")
    assert memory.flush()["ok"]
    assert fake.count("POST", "/ingest/ing-1/retry") == 1

    fake.poll_states["ing-2"] = ["failed", "failed"]
    memory.record("step", "Eval 2 scored 0.44.")
    result = memory.flush()
    assert result == {"ok": False, "lag": False, "pending": 0}
    assert fake.count("POST", "/ingest/ing-2/retry") == 1
    assert memory.stats["failed_ingestions"] == 1
    assert memory.enabled


def test_5xx_retries_with_the_same_idempotency_key(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    fake.queue("POST", "/ingest/batch", FakeResponse(503, {"code": "organization-not-ready"}))
    memory.record("step", "Eval 1 scored 0.41.")
    assert memory.flush()["ok"]
    bodies = []
    for call in fake.calls:
        if call[1] == "/ingest/batch":
            bodies.append(call[2])
    assert len(bodies) == 2
    assert bodies[0] == bodies[1]
    assert bodies[0]["idempotencyKey"] == "s0-swarm-1:batch:1"


def test_429_backs_off_and_retries(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    fake.queue("POST", "/ingest/batch", FakeResponse(429))
    fake.queue("POST", "/ingest/batch", FakeResponse(429))
    memory.record("step", "Eval 1 scored 0.41.")
    clock.sleeps = []
    assert memory.flush()["ok"]
    assert clock.sleeps[:2] == [1.0, 2.0]
    assert memory.enabled


def test_429_cap_reached_degrades_without_retry(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    fake.queue("POST", "/ingest/batch", FakeResponse(429, {"code": "project-cap-reached"}))
    memory.record("step", "Eval 1 scored 0.41.")
    memory.flush()
    assert fake.count("POST", "/ingest/batch") == 1
    assert not memory.enabled


def test_402_degrades_and_turns_record_and_recall_into_no_ops(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    seen = []

    def hook(kind, text, extra):
        seen.append(kind)

    memory.on_event = hook
    fake.queue("POST", "/ingest/batch", FakeResponse(402, {"code": "out-of-credits"}))
    memory.record("step", "Eval 1 scored 0.41.")
    result = memory.flush()
    assert result == {"ok": False, "lag": False, "pending": 0}
    assert not memory.enabled
    assert "out-of-credits" in memory.degraded_reason
    assert seen == ["memory_degraded"]
    assert fake.count("POST", "/ingest/batch") == 1
    calls_before = len(fake.calls)
    memory.record("step", "Eval 2 scored 0.44.")
    assert memory.buffer == []
    assert memory.recall("planner-1", "anything") == []
    assert memory.flush() == {"ok": False, "lag": False, "pending": 0}
    assert len(fake.calls) == calls_before


def test_409_data_point_deleted_degrades_as_a_reused_run_id(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    fake.queue("POST", "/ingest/batch", FakeResponse(409, {"code": "data-point-deleted",
                                                           "debugMessage": "a deleted id is never reused"}))
    memory.record("step", "Eval 1 scored 0.41.")
    result = memory.flush()
    assert result == {"ok": False, "lag": False, "pending": 0}
    assert not memory.enabled
    assert "run id reused" in memory.degraded_reason
    assert "data-point-deleted" in memory.degraded_reason
    assert memory.sent_ids == []
    calls_before = len(fake.calls)
    assert memory.recall("planner-1", "anything") == []
    assert len(fake.calls) == calls_before


def test_reused_run_id_after_cleanup_without_a_state_file_degrades_at_the_first_batch(tmp_path, key):
    fake = FakePast()
    no_state = {"memory_state_path": ""}
    first, clock = started(tmp_path, fake, "s0-swarm-r", no_state)
    first.record("step", "Eval 1 scored 0.41.")
    first.flush()
    first.end_run(cleanup=True)
    assert fake.deleted == ["s0-swarm-r:step:1"]
    # The cleaned-up run leaves nothing to recall, so the second start is cold.
    second, clock = started(tmp_path, fake, "s0-swarm-r", no_state)
    second.record("step", "Eval 1 scored 0.10.")
    assert second.flush()["ok"] is False
    assert not second.enabled
    assert "run id reused" in second.degraded_reason
    assert fake.points == {}


def test_two_refused_batches_in_a_row_degrade(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    refusal = FakeResponse(400, {"code": "invalid-ingestion"})
    fake.queue("POST", "/ingest/batch", refusal)
    memory.record("step", "Eval 1 scored 0.41.")
    assert memory.flush()["ok"] is False
    assert memory.enabled
    memory.record("step", "Eval 2 scored 0.44.")
    assert memory.flush()["ok"] is True
    fake.queue("POST", "/ingest/batch", refusal)
    memory.record("step", "Eval 3 scored 0.45.")
    memory.flush()
    assert memory.enabled
    fake.queue("POST", "/ingest/batch", refusal)
    memory.record("step", "Eval 4 scored 0.46.")
    memory.flush()
    assert not memory.enabled
    assert "2 batches in a row" in memory.degraded_reason


def test_repeated_5xx_degrades(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    for attempt in range(4):
        fake.queue("POST", "/ingest/batch", FakeResponse(500))
    memory.record("step", "Eval 1 scored 0.41.")
    assert memory.flush()["ok"] is False
    assert fake.count("POST", "/ingest/batch") == 4
    assert not memory.enabled
    assert "failed after 4 tries" in memory.degraded_reason


# recall.

def test_recall_drops_superseded_mentions_and_other_runs(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    memory.record("collapse", "Collapse: late overlap deletion fell to 0.02.")
    memory.flush()
    own = [{"sourceId": "s0-swarm-1:collapse:1"}]
    fake.extra_docs = [
        {"content": "late overlap score: 0.37", "artifact": {"kind": "claim", "supersededAt": "2026-11-14T00:00:00Z"},
         "sources": own},
        {"content": "late overlap", "artifact": {"kind": "mention"}, "sources": own},
        {"content": "a note from another run", "artifact": {"kind": "claim"},
         "sources": [{"sourceId": "s9-swarm-9:step:1"}]},
        {"content": "Deleting overlap in late layers collapses the model.", "rank": 2, "occurredAt": "t",
         "artifact": {"kind": "observation"}, "sources": own},
    ]
    lessons = memory.recall("planner-2", "collapses")
    assert len(lessons) == 2
    assert lessons[0]["text"] == "Collapse: late overlap deletion fell to 0.02."
    assert lessons[0]["kind"] == "collapse"
    assert lessons[0]["artifact"] == "source"
    assert lessons[1]["kind"] == "collapse"
    assert lessons[1]["artifact"] == "observation"
    for lesson in lessons:
        for name in ["text", "source_id", "occurred_at", "kind"]:
            assert name in lesson
        assert lesson["source_id"] == "s0-swarm-1:collapse:1"
    assert memory.stats["foreign_dropped"] == 1
    recall_body = fake.calls[-1][2]
    assert recall_body["identity"] == "s0-swarm-1:planner-2"
    assert recall_body["level"] == "low"
    assert recall_body["limit"] == 20


def test_recall_makes_no_call_for_unknown_roles_or_before_any_send(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    calls_before = len(fake.calls)
    assert memory.recall("planner-1", "anything") == []
    memory.record("step", "Eval 1 scored 0.41.")
    memory.flush()
    calls_after_flush = len(fake.calls)
    assert memory.recall("planner-9", "anything") == []
    assert len(fake.calls) == calls_after_flush
    assert calls_after_flush > calls_before


def test_recall_failure_returns_empty(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    memory.record("step", "Eval 1 scored 0.41.")
    memory.flush()
    fake.queue("POST", "/recall", FakeResponse(400, {"code": "recall-level-invalid"}))
    assert memory.recall("planner-1", "anything", level="medium") == []
    assert memory.enabled


def test_runs_are_isolated(tmp_path, key):
    fake = FakePast()
    first, clock = started(tmp_path, fake, "s0-swarm-a")
    first.record("step", "Eval 1 scored 0.41.")
    first.flush()
    second, clock = started(tmp_path, fake, "s0-swarm-b")
    second.record("step", "Eval 1 scored 0.10.")
    second.flush()
    texts = []
    for lesson in second.recall("planner-1", "scores"):
        texts.append(lesson["text"])
    assert texts == ["Eval 1 scored 0.10."]


# end_run and cleanup.

def test_cleanup_deletes_in_chunks_of_1000_then_the_scope(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    for index in range(2500):
        memory.record("step", "Eval " + str(index) + " scored 0.41.")
    assert memory.flush()["ok"]
    assert fake.count("POST", "/ingest/batch") == 3
    result = memory.end_run(cleanup=True)
    sizes = []
    for call in fake.calls:
        if call[1] == "/data-points/delete":
            sizes.append(len(call[2]["ids"]))
    assert sizes == [1000, 1000, 500]
    assert result["deleted"] == 2500
    assert result["audience_deleted"]
    assert fake.points == {}
    assert memory.audience not in fake.audiences
    assert fake.count("DELETE", "/identities/") == len(ROLES)
    assert not memory.enabled


def test_cleanup_falls_back_to_single_deletes_on_404(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake, extra={"memory_verify_cleanup": True})
    memory.record("step", "Eval 1 scored 0.41.")
    memory.record("step", "Eval 2 scored 0.44.")
    memory.flush()
    del fake.points["s0-swarm-1:step:1"]
    result = memory.end_run(cleanup=True)
    assert result["deleted"] == 1
    assert result["verified_empty"] is True
    assert fake.count("DELETE", "/data-points/") == 2


def test_end_run_without_cleanup_sends_the_rest_and_deletes_nothing(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    memory.record("verdict", "Round 0 frontier top score 0.44.")
    result = memory.end_run(cleanup=False)
    assert result["deleted"] == 0
    assert fake.count("POST", "/ingest/batch") == 1
    assert fake.count("POST", "/data-points/delete") == 0
    assert memory.audience in fake.audiences


def test_degraded_run_still_cleans_up_what_it_sent(tmp_path, key):
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    memory.record("step", "Eval 1 scored 0.41.")
    memory.flush()
    fake.queue("POST", "/ingest/batch", FakeResponse(402, {"code": "out-of-credits"}))
    memory.record("step", "Eval 2 scored 0.44.")
    memory.flush()
    assert not memory.enabled
    result = memory.end_run(cleanup=True)
    assert result["deleted"] == 1
    assert fake.points == {}


def test_the_key_never_reaches_logs_or_the_state_file(tmp_path, key, caplog):
    caplog.set_level(logging.INFO, logger="wsp.swarm.memory")
    fake = FakePast()
    memory, clock = started(tmp_path, fake)
    fake.queue("POST", "/ingest/batch", FakeResponse(402, {"code": "out-of-credits"}))
    memory.record("step", "Eval 1 scored 0.41.")
    memory.flush()
    memory.end_run(cleanup=True)
    assert PLACEHOLDER_KEY not in caplog.text
    state_text = (tmp_path / "runs" / "s0-swarm-1" / "memory.json").read_text(encoding="utf-8")
    assert PLACEHOLDER_KEY not in state_text
    assert "memory_degraded" in caplog.text


def test_memory_module_matches_the_interface():
    from wsp.swarm.interfaces import Memory
    assert issubclass(PastMemory, Memory)
    assert issubclass(memory_mod.MemoryLeakError, RuntimeError)
