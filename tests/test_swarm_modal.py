"""Tests for modal_swarm.py and wsp/swarm/interim_eval.py.

CPU only, no network: every Modal Queue and Dict is a fake defined here.
"""

import threading
import warnings

import pytest
import torch

import modal_swarm
from wsp.swarm import interim_eval


class FakeQueue:
    """A thread-safe stand-in for modal.Queue with partitions."""

    def __init__(self):
        self.partitions = {}
        self.lock = threading.Lock()
        self.get_calls = 0
        self.fail_gets = 0
        self.ttls = []

    def put_many(self, values, partition=None, partition_ttl=None):
        with self.lock:
            if partition not in self.partitions:
                self.partitions[partition] = []
            for value in values:
                self.partitions[partition].append(value)
            self.ttls.append(partition_ttl)

    def get(self, block=True, timeout=None, partition=None):
        with self.lock:
            self.get_calls += 1
            if self.fail_gets > 0:
                self.fail_gets -= 1
                raise ConnectionError("fake network error")
            items = self.partitions.get(partition, [])
            if len(items) == 0:
                return None
            return items.pop(0)

    def len(self, partition=None, total=False):
        with self.lock:
            return len(self.partitions.get(partition, []))


class FakeDict:
    """A thread-safe stand-in for modal.Dict. put(skip_if_exists=True) is an atomic claim."""

    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
        self.fail_puts = 0
        self.fail_reads = False

    def put(self, key, value, skip_if_exists=False):
        with self.lock:
            if self.fail_puts > 0:
                self.fail_puts -= 1
                raise ConnectionError("fake network error")
            if skip_if_exists and key in self.data:
                return False
            self.data[key] = value
            return True

    def get(self, key, default=None):
        if self.fail_reads:
            raise ConnectionError("fake network error")
        with self.lock:
            return self.data.get(key, default)

    def contains(self, key):
        with self.lock:
            return key in self.data

    def items(self):
        if self.fail_reads:
            raise ConnectionError("fake network error")
        with self.lock:
            pairs = list(self.data.items())
        return iter(pairs)


@pytest.fixture
def fakes(monkeypatch):
    """Replace every named Modal object with a fake. Returns {name: fake}."""
    objects = {modal_swarm.BUDGET_QUEUE: FakeQueue()}
    for name in modal_swarm.DICT_NAMES:
        objects[name] = FakeDict()

    def fake_queue(name):
        return objects[name]

    def fake_dict(name):
        return objects[name]

    monkeypatch.setattr(modal_swarm, "get_queue", fake_queue)
    monkeypatch.setattr(modal_swarm, "get_dict", fake_dict)
    return objects


def graft_program(part_id="pilot_qmath15_l18_top512", control=None, steps=50):
    return {"host": modal_swarm.HOST, "seed": 0, "masks": None, "task_vectors": [],
            "grafts": [{"id": part_id, "at": None, "control": control, "control_seed": 1}],
            "heal": {"steps": steps, "data": "gsm8k_train", "lr": 0.0003, "batch_size": 8, "seq_len": 384}}


def call_local(function, *args, **kwargs):
    """Run a Modal function's body here. Modal warns that the volume is not mounted; ignore that."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return function.local(*args, **kwargs)


class FakeClock:
    """Stands in for the time module inside modal_swarm. Only the fakes below move it."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ContainerKilled(BaseException):
    """A container kill (timeout or preemption): not an Exception, so the function body cannot catch it."""


FAKE_REPORT = {"host": "h", "grafts": [{"id": "pilot_qmath15_l18_top512", "kind": "mlp_cluster", "control": None,
                                        "indices": [1, 2, 3], "fit": "ridge", "fit_r2": 0.64}],
               "heal": {"seconds": 10.0, "losses": [1.4, 1.2]}}


class FakeWork:
    """Fake setup, compile and eval for the container functions: setup 50 s, compile+heal 30 s, eval 5 s.

    kill_eval makes the eval phase die the way a killed container does.
    fail_in makes that phase raise an ordinary error.
    """

    def __init__(self, clock):
        self.clock = clock
        self.kill_eval = False
        self.fail_in = None
        self.calls = []

    def setup_run(self, program, catalog_name, split, eval_limit, align_samples):
        self.calls.append("setup")
        self.clock.advance(50.0)
        if self.fail_in == "setup":
            raise FileNotFoundError("catalog missing")
        return {"catalog": None, "align": None, "heal_input": None, "items": [], "warmed": ["m"]}

    def compile_into(self, program, work_dir, setup):
        self.calls.append("compile")
        self.clock.advance(30.0)
        if self.fail_in == "compile":
            raise RuntimeError("CUDA out of memory")
        report = dict(FAKE_REPORT)
        if program.heal_spec is None:
            report["heal"] = None
        return report

    def evaluate_model(self, work_dir, items, report):
        self.calls.append("eval")
        self.clock.advance(5.0)
        if self.kill_eval:
            raise ContainerKilled()
        return {"accuracy": 0.25, "n_items": 20, "samples": [" 18"]}


@pytest.fixture
def work(fakes, monkeypatch):
    """Fake clock and fake GPU work inside modal_swarm. Returns the FakeWork."""
    clock = FakeClock()
    fake = FakeWork(clock)
    monkeypatch.setattr(modal_swarm, "time", clock)
    monkeypatch.setattr(modal_swarm, "setup_run", fake.setup_run)
    monkeypatch.setattr(modal_swarm, "compile_into", fake.compile_into)
    monkeypatch.setattr(modal_swarm, "evaluate_model", fake.evaluate_model)
    return fake


def compute_rows(fakes, run_id):
    rows = []
    for key in sorted(fakes[modal_swarm.LEDGER_DICT].data):
        row = fakes[modal_swarm.LEDGER_DICT].data[key]
        if row["run_id"] == run_id and row["axis"] == "compute":
            rows.append(row)
    return rows


def phases_of(rows):
    phases = []
    for row in rows:
        phases.append(row["phase"])
    return sorted(phases)


# ---------------------------------------------------------------- spend and the pool


def test_spend_hands_out_tokens_in_order_then_exhausts(fakes):
    modal_swarm.fill_pool("r1", "search", 3)
    assert modal_swarm.eval_remaining("r1", "search") == 3
    tokens = []
    for index in range(3):
        tokens.append(modal_swarm.spend("r1", "search", "planner-1"))
    assert tokens == [0, 1, 2]
    assert modal_swarm.spend("r1", "search", "planner-1") is None
    assert modal_swarm.spend("r1", "search", "planner-1") is None
    assert modal_swarm.eval_remaining("r1", "search") == 0
    assert modal_swarm.eval_spent("r1", "search") == 3


def test_fill_sets_partition_ttl_and_refuses_a_second_fill(fakes):
    modal_swarm.fill_pool("r1", "search", 2)
    assert fakes[modal_swarm.BUDGET_QUEUE].ttls == [modal_swarm.POOL_TTL_S]
    modal_swarm.spend("r1", "search", "a")
    modal_swarm.spend("r1", "search", "a")
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.fill_pool("r1", "search", 2)
    assert modal_swarm.eval_remaining("r1", "search") == 0


def test_fill_rejects_bad_sizes_and_names(fakes):
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.fill_pool("r1", "search", 0)
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.fill_pool("r1", "search", modal_swarm.POOL_MAX + 1)
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.fill_pool("bad|run", "search", 3)


def test_pools_and_runs_are_separate_partitions(fakes):
    modal_swarm.fill_pool("r1", "search", 2)
    modal_swarm.fill_pool("r1", "heldout", 1)
    modal_swarm.fill_pool("r2", "search", 1)
    assert modal_swarm.spend("r1", "heldout", "judge") == 0
    assert modal_swarm.spend("r1", "heldout", "judge") is None
    assert modal_swarm.spend("r2", "search", "evolve") == 0
    assert modal_swarm.eval_remaining("r1", "search") == 2


def test_spend_does_not_retry_a_failed_pop(fakes):
    queue = fakes[modal_swarm.BUDGET_QUEUE]
    modal_swarm.fill_pool("r1", "search", 3)
    queue.fail_gets = 1
    calls_before = queue.get_calls
    assert modal_swarm.spend("r1", "search", "planner-1") is None
    assert queue.get_calls == calls_before + 1
    assert modal_swarm.eval_spent("r1", "search") == 0
    assert modal_swarm.eval_remaining("r1", "search") == 3


def test_spend_drops_the_token_when_its_ledger_row_fails(fakes):
    modal_swarm.fill_pool("r1", "search", 3)
    fakes[modal_swarm.LEDGER_DICT].fail_puts = 1
    assert modal_swarm.spend("r1", "search", "planner-1") is None
    assert modal_swarm.eval_remaining("r1", "search") == 2
    assert modal_swarm.spend("r1", "search", "planner-1") == 1


def test_spend_refuses_a_token_that_already_has_a_row(fakes):
    modal_swarm.fill_pool("r1", "search", 1)
    fakes[modal_swarm.BUDGET_QUEUE].put_many([0], partition=modal_swarm.partition_key("r1", "search"))
    assert modal_swarm.spend("r1", "search", "a") == 0
    assert modal_swarm.spend("r1", "search", "b") is None


def test_eval_remaining_fails_closed(fakes, monkeypatch):
    def broken_queue(name):
        raise ConnectionError("no network")

    monkeypatch.setattr(modal_swarm, "get_queue", broken_queue)
    assert modal_swarm.eval_remaining("r1", "search") == 0
    assert modal_swarm.spend("r1", "search", "a") is None


def test_spends_from_four_threads_have_no_duplicates_and_no_losses(fakes):
    modal_swarm.fill_pool("threads", "search", 10)
    got = []
    got_lock = threading.Lock()

    def worker(name):
        while True:
            token = modal_swarm.spend("threads", "search", name)
            if token is None:
                return
            with got_lock:
                got.append(token)

    threads = []
    for index in range(4):
        threads.append(threading.Thread(target=worker, args=("planner-" + str(index),)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(got) == list(range(10))
    assert modal_swarm.eval_spent("threads", "search") == 10


# ---------------------------------------------------------------- the write-once ledger


def test_ledger_rows_are_write_once(fakes):
    modal_swarm.fill_pool("r1", "search", 1)
    modal_swarm.spend("r1", "search", "planner-1", method="swarm")
    ledger = fakes[modal_swarm.LEDGER_DICT]
    key = "r1|search|eval|000000"
    assert ledger.data[key]["token"] == 0
    assert ledger.data[key]["who"] == "planner-1"
    assert ledger.data[key]["method"] == "swarm"
    assert ledger.put(key, {"forged": True}, skip_if_exists=True) is False
    assert ledger.data[key]["token"] == 0


def test_charge_compute_with_the_same_id_writes_one_row(fakes):
    first = modal_swarm.charge_compute("r1", "search", 41.0, "planner-1", "compile", charge_id="abc:0:compile")
    second = modal_swarm.charge_compute("r1", "search", 41.0, "planner-1", "compile", charge_id="abc:0:compile")
    assert first == second == "r1|search|compute|abc:0:compile"
    assert modal_swarm.compute_used("r1", "search") == 41.0


def test_compute_sums_rows_per_run_and_pool(fakes):
    modal_swarm.charge_compute("r1", "search", 41.0, "planner-1", "compile")
    modal_swarm.charge_compute("r1", "search", 150.5, "planner-1", "heal")
    modal_swarm.charge_compute("r1", "search", 120.0, "planner-2", "eval")
    modal_swarm.charge_compute("r1", "heldout", 99.0, "judge", "eval")
    modal_swarm.charge_compute("r2", "search", 7.0, "evolve", "eval")
    assert modal_swarm.charge_compute("r1", "search", 0, "planner-1", "eval") is None
    assert modal_swarm.charge_compute("r1", "search", -3, "planner-1", "eval") is None
    assert modal_swarm.compute_used("r1", "search") == pytest.approx(311.5)
    assert modal_swarm.compute_used("r1", "heldout") == pytest.approx(99.0)
    assert modal_swarm.compute_used("r2", "search") == pytest.approx(7.0)
    assert modal_swarm.compute_used("r3", "search") == 0.0


def test_concurrent_compute_charges_are_all_kept(fakes):
    def worker():
        for index in range(25):
            modal_swarm.charge_compute("r1", "search", 2.0, "w", "eval")

    threads = []
    for index in range(4):
        threads.append(threading.Thread(target=worker))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert modal_swarm.compute_used("r1", "search") == pytest.approx(200.0)


def test_compute_charge_and_reads_fail_closed(fakes):
    fakes[modal_swarm.LEDGER_DICT].fail_puts = 1
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.charge_compute("r1", "search", 5.0, "w", "eval")
    fakes[modal_swarm.LEDGER_DICT].fail_reads = True
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.compute_used("r1", "search")
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.ledger_rows("r1")


def test_ledger_rows_lists_one_run_oldest_first(fakes, monkeypatch):
    clock = [1000.0]

    def fake_time():
        clock[0] += 1.0
        return clock[0]

    monkeypatch.setattr(modal_swarm.time, "time", fake_time)
    modal_swarm.fill_pool("r1", "search", 2)
    modal_swarm.spend("r1", "search", "a")
    modal_swarm.charge_compute("r1", "search", 3.0, "a", "eval")
    modal_swarm.fill_pool("r2", "search", 1)
    rows = modal_swarm.ledger_rows("r1")
    axes = []
    for row in rows:
        axes.append(row["axis"])
        assert row["run_id"] == "r1"
    assert axes == ["fill", "eval", "compute"]
    assert rows[0]["key"] == "r1|search|fill|pool"


def test_put_exp_row_is_write_once(fakes):
    assert modal_swarm.put_exp_row("r1", 3, {"score": 0.1}) is True
    assert modal_swarm.put_exp_row("r1", 3, {"score": 0.9}) is False
    assert fakes[modal_swarm.EXP_DICT].data["r1:000003"] == {"score": 0.1}


# ---------------------------------------------------------------- keys


def test_partition_key_fits_modal_limit():
    assert modal_swarm.partition_key("s0-swarm-1731627600", "search") == "s0-swarm-1731627600:search"
    long_key = modal_swarm.partition_key("x" * 80, "search")
    assert long_key.startswith("h:")
    assert len(long_key.encode("utf-8")) <= 64
    assert long_key == modal_swarm.partition_key("x" * 80, "search")
    assert long_key != modal_swarm.partition_key("x" * 80, "heldout")
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.partition_key("", "search")


def test_program_hash_ignores_key_order_and_fills_defaults():
    program = graft_program()
    reordered = {}
    for key in reversed(list(program.keys())):
        reordered[key] = program[key]
    assert modal_swarm.program_hash(program) == modal_swarm.program_hash(reordered)
    short = {"host": modal_swarm.HOST, "grafts": [{"id": "pilot_qmath15_l18_top512", "control_seed": 1}],
             "heal": {"steps": 50, "data": "gsm8k_train", "lr": 0.0003, "batch_size": 8, "seq_len": 384}}
    assert modal_swarm.program_hash(short) == modal_swarm.program_hash(program)
    assert len(modal_swarm.program_hash(program)) == 16


def test_program_hash_matches_core_space_when_present():
    space = pytest.importorskip("wsp.swarm.space")
    program = graft_program()
    assert modal_swarm.program_hash(program) == space.program_hash(program)
    if hasattr(space, "probe_hash"):
        assert modal_swarm.program_hash(modal_swarm.without_heal(program)) == space.probe_hash(program)


def test_probe_key_ignores_heal_but_not_grafts_or_catalog():
    base = modal_swarm.probe_key(graft_program(steps=50), "pilot", "r1")
    assert modal_swarm.probe_key(graft_program(steps=300), "pilot", "r1") == base
    assert modal_swarm.probe_key(graft_program(control="random"), "pilot", "r1") != base
    assert modal_swarm.probe_key(graft_program(part_id="pilot_qmath15_l6_top256"), "pilot", "r1") != base
    assert modal_swarm.probe_key(graft_program(), "main", "r1") != base
    assert modal_swarm.probe_key(graft_program(), "pilot", "r1", align_samples=200) != base


def test_result_key_separates_split_limit_heal_control_and_backend():
    program = graft_program()
    base = modal_swarm.result_key(program, "search", "pilot", "r1", 20)
    assert base.endswith("|interim")
    assert modal_swarm.result_key(program, "heldout", "pilot", "r1", 20) != base
    assert modal_swarm.result_key(program, "search", "pilot", "r1", None) != base
    assert modal_swarm.result_key(graft_program(steps=300), "search", "pilot", "r1", 20) != base
    assert modal_swarm.result_key(graft_program(control="random"), "search", "pilot", "r1", 20) != base
    assert modal_swarm.result_key(program, "search", "pilot", "r1", 20) == base
    assert modal_swarm.host_key(modal_swarm.HOST, "search", 20) != modal_swarm.host_key(modal_swarm.HOST, "search")


def test_keys_are_scoped_by_run_except_the_host():
    program = graft_program()
    assert modal_swarm.probe_key(program, "pilot", "s0-swarm") != modal_swarm.probe_key(program, "pilot", "s0-evolve")
    assert modal_swarm.probe_key(program, "pilot", "s0-swarm") != modal_swarm.probe_key(program, "pilot", "s1-swarm")
    assert (modal_swarm.result_key(program, "search", "pilot", "s0-swarm", 20) !=
            modal_swarm.result_key(program, "search", "pilot", "s0-evolve", 20))
    assert modal_swarm.host_key(modal_swarm.HOST, "search", 20).startswith("host|")
    for scope in ["", "a|b", None]:
        with pytest.raises(modal_swarm.BudgetError):
            modal_swarm.probe_key(program, "pilot", scope)
        with pytest.raises(modal_swarm.BudgetError):
            modal_swarm.result_key(program, "search", "pilot", scope, 20)


def test_two_runs_never_share_a_probe_or_a_result(fakes, work):
    program = graft_program()
    fakes[modal_swarm.PROBE_DICT].put(modal_swarm.probe_key(program, "pilot", "run-a"),
                                      {"ok": True, "graft_report": [], "seconds": {"compile": 40.0}})
    fakes[modal_swarm.CACHE_DICT].put(modal_swarm.result_key(program, "search", "pilot", "run-a", 20),
                                      {"ok": True, "scores": {"gsm8k": 0.25}, "seconds": {"compile": 1.0}})
    assert modal_swarm.lookup_probe(program, "pilot", "run-a") is not None
    assert modal_swarm.lookup_probe(program, "pilot", "run-b") is None
    assert modal_swarm.lookup_result(program, "search", "pilot", "run-a", 20) is not None
    assert modal_swarm.lookup_result(program, "search", "pilot", "run-b", 20) is None

    probe = call_local(modal_swarm.probe_candidate, program, "pilot", run_id="run-b", pool="search")
    assert probe["cache_hit"] is False
    assert probe["seconds"]["compile"] == 30.0
    run = call_local(modal_swarm.run_candidate, program, "search", 0, "pilot", 20, run_id="run-b", pool="search")
    assert run["cache_hit"] is False
    assert run["ok"] is True
    assert modal_swarm.compute_used("run-b", "search") == pytest.approx(30.0 + 20.0 + 10.0 + 5.0)
    assert modal_swarm.compute_used("run-a", "search") == 0.0


def test_cache_lookup_fails_soft(fakes):
    fakes[modal_swarm.CACHE_DICT].fail_reads = True
    assert modal_swarm.lookup_result(graft_program(), "search", "pilot", "r1", 20) is None


def test_cached_probe_returns_a_hit_with_zero_seconds(fakes):
    program = graft_program()
    key = modal_swarm.probe_key(program, "pilot", "r1")
    fakes[modal_swarm.PROBE_DICT].put(key, {"ok": True, "graft_report": [{"fit_r2": 0.5}],
                                            "seconds": {"compile": 40.0}, "cache_hit": False})
    result = call_local(modal_swarm.probe_candidate, program, "pilot", run_id="r1", pool="search")
    assert result["cache_hit"] is True
    assert result["seconds"] == {"compile": 0.0}
    assert result["cached_seconds"] == {"compile": 40.0}
    assert result["graft_report"][0]["fit_r2"] == 0.5
    assert result["charged"] == [] and result["uncharged"] == {}
    assert compute_rows(fakes, "r1") == []


def test_cached_run_returns_a_hit_with_the_callers_token(fakes):
    program = graft_program()
    key = modal_swarm.result_key(program, "search", "pilot", "r1", 20)
    fakes[modal_swarm.CACHE_DICT].put(key, {"ok": True, "scores": {"gsm8k": 0.25}, "token": 4,
                                            "seconds": {"compile": 1.0, "heal": 2.0, "eval": 3.0}})
    result = call_local(modal_swarm.run_candidate, program, "search", 9, "pilot", 20, run_id="r1", pool="search")
    assert result["cache_hit"] is True
    assert result["token"] == 9
    assert result["seconds"] == {"compile": 0.0, "heal": 0.0, "eval": 0.0}
    assert compute_rows(fakes, "r1") == []


def test_run_candidate_refuses_without_a_token(fakes):
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", None, "pilot", 20, run_id="r1",
                        pool="search")
    assert result["ok"] is False
    assert "token" in result["error"]
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", True, "pilot", 20, run_id="r1",
                        pool="search")
    assert result["ok"] is False
    result = call_local(modal_swarm.run_candidate, graft_program(), "train", 0, "pilot", 20, run_id="r1",
                        pool="search")
    assert "split" in result["error"]


def test_calls_without_a_run_id_or_pool_are_refused(fakes, work):
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", 20)
    assert result["ok"] is False
    assert "run_id" in result["error"]
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", 20, run_id="r1")
    assert "pool" in result["error"]
    result = call_local(modal_swarm.probe_candidate, graft_program(), "pilot", run_id="bad|id", pool="search")
    assert result["ok"] is False
    assert work.calls == []


# ---------------------------------------------------------------- compute charged from the container


def test_run_candidate_charges_each_phase_and_never_setup(fakes, work):
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", 3, "pilot", 20, run_id="r1",
                        pool="search", who="planner-1", method="swarm")
    assert result["ok"] is True
    assert result["setup_seconds"] == 50.0
    assert result["seconds"] == {"compile": 20.0, "heal": 10.0, "eval": 5.0}
    rows = compute_rows(fakes, "r1")
    assert phases_of(rows) == ["compile", "eval", "heal"]
    for row in rows:
        assert row["who"] == "planner-1"
        assert row["method"] == "swarm"
        assert row["gpu"] == "none"
    assert modal_swarm.compute_used("r1", "search") == pytest.approx(35.0)
    assert len(result["charged"]) == 3
    prefix = "r1|search|compute|" + result["program_hash"] + ":3:"
    for key in result["charged"]:
        assert key.startswith(prefix)
        assert key.endswith(":" + result["attempt"])
    assert result["uncharged"] == {}


def test_a_retried_attempt_adds_rows_and_keeps_the_first_attempts(fakes, work):
    program = graft_program()
    work.kill_eval = True
    with pytest.raises(ContainerKilled):
        call_local(modal_swarm.run_candidate, program, "search", 0, "pilot", 20, run_id="r1", pool="search")
    assert phases_of(compute_rows(fakes, "r1")) == ["compile", "heal"]
    assert modal_swarm.lookup_result(program, "search", "pilot", "r1", 20) is None

    work.kill_eval = False
    retry = call_local(modal_swarm.run_candidate, program, "search", 0, "pilot", 20, run_id="r1", pool="search")
    assert retry["ok"] is True
    assert phases_of(compute_rows(fakes, "r1")) == ["compile", "compile", "eval", "heal", "heal"]
    assert modal_swarm.compute_used("r1", "search") == pytest.approx(30.0 + 35.0)


def test_a_failed_phase_is_charged_for_the_time_it_ran(fakes, work):
    work.fail_in = "compile"
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", 20, run_id="r1",
                        pool="search")
    assert result["ok"] is False
    assert result["phase"] == "compile"
    assert result["seconds"]["compile"] == 30.0
    assert phases_of(compute_rows(fakes, "r1")) == ["compile"]
    assert modal_swarm.compute_used("r1", "search") == pytest.approx(30.0)


def test_a_failed_setup_charges_nothing(fakes, work):
    work.fail_in = "setup"
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", 20, run_id="r1",
                        pool="search")
    assert result["ok"] is False
    assert result["phase"] == "setup"
    assert result["setup_seconds"] == 50.0
    assert compute_rows(fakes, "r1") == []


def test_uncharged_phases_are_settled_by_the_caller_once(fakes, work):
    fakes[modal_swarm.LEDGER_DICT].fail_puts = 3
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", 20, run_id="r1",
                        pool="search", who="p1")
    assert result["ok"] is True
    assert result["uncharged"] == {"compile": 20.0, "heal": 10.0, "eval": 5.0}
    assert compute_rows(fakes, "r1") == []
    first = modal_swarm.charge_uncharged(result)
    second = modal_swarm.charge_uncharged(result)
    assert len(first) == 3
    assert first == second
    assert modal_swarm.compute_used("r1", "search") == pytest.approx(35.0)
    fakes[modal_swarm.LEDGER_DICT].fail_puts = 1
    with pytest.raises(modal_swarm.BudgetError):
        modal_swarm.charge_uncharged({"uncharged": {"eval": 1.0}, "run_id": "r1", "pool": "search", "who": "p1",
                                      "method": "", "charge_prefix": "x", "attempt": "y", "gpu": None})


def test_probe_charges_one_probe_row_and_a_repeat_in_the_same_run_is_free(fakes, work):
    program = graft_program()
    first = call_local(modal_swarm.probe_candidate, program, "pilot", run_id="r1", pool="search")
    assert first["ok"] is True
    assert first["setup_seconds"] == 50.0
    assert first["seconds"] == {"compile": 30.0}
    assert first["graft_report"][0]["n_units"] == 3
    rows = compute_rows(fakes, "r1")
    assert phases_of(rows) == ["probe"]
    assert rows[0]["amount"] == 30.0
    again = call_local(modal_swarm.probe_candidate, graft_program(steps=300), "pilot", run_id="r1", pool="search")
    assert again["cache_hit"] is True
    assert len(compute_rows(fakes, "r1")) == 1


# ---------------------------------------------------------------- GPU type and container setup


def test_parse_gpu_pins_one_type_or_makes_a_preference_list():
    assert modal_swarm.parse_gpu("L4") == "L4"
    assert modal_swarm.parse_gpu("") == "L4"
    assert modal_swarm.parse_gpu(" L4 , A10 ") == ["L4", "A10"]
    assert modal_swarm.gpu_text(["L4", "A10"]) == "L4,A10"
    assert modal_swarm.gpu_text("L4") == "L4"


def test_compute_rows_record_the_gpu_so_mixed_runs_show(fakes):
    modal_swarm.charge_compute("r1", "search", 5.0, "a", "compile", gpu="NVIDIA L4")
    modal_swarm.charge_compute("r1", "search", 5.0, "a", "eval", gpu="NVIDIA L4")
    assert modal_swarm.compute_devices("r1", "search") == ["NVIDIA L4"]
    modal_swarm.charge_compute("r1", "search", 5.0, "a", "eval", gpu="NVIDIA A10")
    modal_swarm.charge_compute("r1", "search", 5.0, "a", "eval")
    assert modal_swarm.compute_devices("r1", "search") == ["NVIDIA A10", "NVIDIA L4", "unknown"]
    assert modal_swarm.compute_devices("r2", "search") == []


def test_host_baseline_times_setup_apart_and_is_shared_by_every_run(fakes, work, monkeypatch):
    def fake_setup_host(host, split, eval_limit):
        work.clock.advance(40.0)
        return {"items": [], "warmed": [host]}

    def fake_score_host(host, items):
        work.clock.advance(7.0)
        return {"accuracy": 0.25, "n_items": 20, "samples": []}

    monkeypatch.setattr(modal_swarm, "setup_host", fake_setup_host)
    monkeypatch.setattr(modal_swarm, "score_host", fake_score_host)
    first = call_local(modal_swarm.host_baseline, "search", 20)
    assert first["ok"] is True
    assert first["setup_seconds"] == 40.0
    assert first["seconds"] == {"eval": 7.0}
    assert modal_swarm.lookup_host("search", 20)["scores"] == {"gsm8k": 0.25}
    again = call_local(modal_swarm.host_baseline, "search", 20)
    assert again["cache_hit"] is True
    assert again["seconds"] == {"eval": 0.0}
    assert len(fakes[modal_swarm.LEDGER_DICT].data) == 0


def test_gsm8k_data_loads_once_per_container(monkeypatch):
    import wsp.data

    calls = []

    def fake_splits(seed=0):
        calls.append(seed)
        return [{"question": "q", "answer": "#### 1"}], [{"question": "h", "answer": "#### 2"}]

    monkeypatch.setattr(wsp.data, "gsm8k_splits", fake_splits)
    monkeypatch.setattr(modal_swarm, "CONTAINER_STATE", {"gsm8k": None, "gsm8k_test": None, "warmed": [],
                                                         "cuda_ready": False})
    first = modal_swarm.gsm8k_data()
    second = modal_swarm.gsm8k_data()
    assert calls == [0]
    assert first is second
    assert first[1] == ["Question: h\nAnswer: #### 2"]
    assert modal_swarm.split_items("search", 1) == [{"question": "q", "answer": "#### 1"}]
    assert calls == [0]
    with pytest.raises(ValueError):
        modal_swarm.split_items("train", 1)


def test_pilot_record_shape():
    record = modal_swarm.pilot_record(18, 512)
    assert record["id"] == "pilot_qmath15_l18_top512"
    assert record["type"] == "mlp_cluster"
    assert record["model"] == "Qwen/Qwen2.5-Math-1.5B"
    assert record["layer"] == 18
    assert record["units"] == "units/pilot/pilot_qmath15_l18_top512.pt"
    assert record["signature"] == {"effect": {}, "fit_r2": {}}
    assert record["pilot"] is True
    assert record["source"] == "wsp/extract.py stub extractor; pilot set, not the Person A catalog"


def test_pilot_units_resolve_on_the_volume(tmp_path):
    from wsp.catalog import Catalog

    record = modal_swarm.pilot_record(6, 256)
    catalog = Catalog([record], root=str(tmp_path))
    folder = tmp_path / "units" / "pilot"
    folder.mkdir(parents=True)
    indices = torch.arange(256)
    torch.save({"kind": "mlp_cluster", "model": record["model"], "layer": 6, "indices": indices},
               str(folder / (record["id"] + ".pt")))
    from wsp.assemble import read_units

    units = read_units(catalog, record["id"])
    assert units["layer"] == 6
    assert torch.equal(units["indices"], indices)


# ---------------------------------------------------------------- the interim scorer


def test_final_number_cases():
    assert interim_eval.final_number("so the total is 18.\n#### 18") == 18.0
    assert interim_eval.final_number("#### 1,234") == 1234.0
    assert interim_eval.final_number("#### -7") == -7.0
    assert interim_eval.final_number("first 3 then 4.5 dollars.") == 4.5
    assert interim_eval.final_number("She has 12.") == 12.0
    assert interim_eval.final_number("no digits here") is None
    assert interim_eval.final_number("") is None


def test_item_correct_matches_modal_graft_rules():
    gold = "Janet sells 16 - 3 - 4 = 9 eggs.\n#### 18"
    assert interim_eval.item_correct(" She makes 9 * 2 = 18 dollars.", gold)
    assert interim_eval.item_correct(" 18\nQuestion: next one has 99", gold)
    assert not interim_eval.item_correct(" The answer is 17.", gold)
    assert not interim_eval.item_correct(" I am not sure.", gold)
    assert interim_eval.item_correct(" #### 18.0", gold)


def test_score_completions_counts_and_labels():
    items = [{"question": "a", "answer": "#### 5"}, {"question": "b", "answer": "#### 7"},
             {"question": "c", "answer": "#### 1,000"}, {"question": "d", "answer": "#### 2"}]
    completions = [" 2 + 3 = 5", " 8", " #### 1000", " 2\nQuestion: then 9"]
    result = interim_eval.score_completions(completions, items)
    assert result["correct"] == 3
    assert result["n_items"] == 4
    assert result["accuracy"] == 0.75
    assert result["eval_backend"] == "interim"
    assert result["label"] == "INTERIM"
    assert result["samples"] == [" 2 + 3 = 5", " 8", " #### 1000"]
    with pytest.raises(ValueError):
        interim_eval.score_completions([], [])


def test_final_number_matches_modal_graft():
    modal_graft = pytest.importorskip("modal_graft")
    cases = ["#### 18", "#### 1,234.", "x 3 y 4", "none", "-0.5 and 2.", "#### .5", "1,2,3", "12.3.4", ""]
    for text in cases:
        assert interim_eval.final_number(text) == modal_graft.final_number(text)


class FakeEncoded(dict):
    def __init__(self, input_ids):
        dict.__init__(self, input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class FakeTokenizer:
    """Prompt i becomes token id i; decode maps id i back to a canned completion."""

    def __init__(self, completions):
        self.completions = completions
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.seen = 0

    def __call__(self, prompts, return_tensors=None, padding=None):
        ids = []
        for prompt in prompts:
            assert prompt.startswith("Question: ") and prompt.endswith("\nAnswer:")
            ids.append([self.seen])
            self.seen += 1
        return FakeEncoded(torch.tensor(ids))

    def decode(self, ids, skip_special_tokens=True):
        return self.completions[int(ids[0])]


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.options = []

    def generate(self, input_ids=None, **options):
        self.options.append(options)
        return torch.cat([input_ids, input_ids], dim=1)


def test_score_batches_generation_like_modal_graft():
    items = []
    completions = []
    for index in range(5):
        items.append({"question": "q" + str(index), "answer": "#### " + str(index)})
        completions.append(" it is " + str(index) + "\nQuestion: then 99")
    completions[4] = " it is 40"
    model = FakeModel()
    tokenizer = FakeTokenizer(completions)
    result = interim_eval.score(model, tokenizer, items, batch_size=2)
    assert result["correct"] == 4
    assert result["accuracy"] == 0.8
    assert result["label"] == "INTERIM"
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "<eos>"
    assert len(model.options) == 3
    assert model.options[0] == {"max_new_tokens": 256, "do_sample": False, "pad_token_id": 0}


def test_score_passes_use_cache_false_only_when_asked():
    items = [{"question": "q", "answer": "#### 1"}]
    model = FakeModel()
    interim_eval.score(model, FakeTokenizer([" 1"]), items, use_cache=False)
    assert model.options[0]["use_cache"] is False


# ---------------------------------------------------------------- the harness scorer


def test_result_key_separates_the_scorers_and_tasks():
    program = graft_program()
    base = modal_swarm.result_key(program, "search", "pilot", "r1", None)
    assert modal_swarm.result_key(program, "search", "pilot", "r1", None, tasks=["gsm8k"]) == base
    harness = modal_swarm.result_key(program, "search", "pilot", "r1", None, eval_backend="harness",
                                     tasks=["gsm8k", "if_guard"])
    assert harness != base and harness.endswith("|harness")
    assert "gsm8k+if_guard" in harness
    assert modal_swarm.result_key(program, "search", "pilot", "r1", None, eval_backend="harness",
                                  tasks=["gsm8k"]) != harness


def fake_harness(monkeypatch, work):
    """Replace the harness setup and grading. The grade advances the fake clock by 8 s."""
    seen = {"setup": [], "eval": []}

    def setup_harness(split, tasks):
        seen["setup"].append((split, list(tasks)))
        work.clock.advance(4.0)

    def evaluate_harness(work_dir, split, tasks, namespace):
        seen["eval"].append((split, list(tasks), namespace))
        work.clock.advance(8.0)
        scores = {}
        per_task = {}
        for name in tasks:
            scores[name] = 0.5
            per_task[name] = 100
        return {"scores": scores, "n_items": 100 * len(tasks), "n_items_per_task": per_task}

    monkeypatch.setattr(modal_swarm, "setup_harness", setup_harness)
    monkeypatch.setattr(modal_swarm, "evaluate_harness", evaluate_harness)
    return seen


def test_run_candidate_grades_every_task_with_the_harness(fakes, work, monkeypatch):
    seen = fake_harness(monkeypatch, work)
    result = call_local(modal_swarm.run_candidate, graft_program(), "search", 2, "pilot", None, run_id="r1",
                        pool="search", tasks=["gsm8k", "if_guard"], scorer="harness")
    assert result["ok"] is True
    assert result["scores"] == {"gsm8k": 0.5, "if_guard": 0.5}
    assert result["eval_backend"] == "harness" and result["label"] == "HARNESS"
    assert result["n_items"] == 200 and result["n_items_per_task"] == {"gsm8k": 100, "if_guard": 100}
    assert seen["setup"] == [("search", ["gsm8k", "if_guard"])]
    assert seen["eval"] == [("search", ["gsm8k", "if_guard"], "r1")]
    # The harness setup lands in the untimed setup; only the grade is charged as eval.
    assert result["setup_seconds"] == 54.0
    assert result["seconds"]["eval"] == 8.0
    assert work.calls == ["setup", "compile"]
    key = modal_swarm.result_key(graft_program(), "search", "pilot", "r1", None, eval_backend="harness",
                                 tasks=["gsm8k", "if_guard"])
    assert fakes[modal_swarm.CACHE_DICT].get(key) is not None
    assert modal_swarm.lookup_result(graft_program(), "search", "pilot", "r1") is None


def test_run_candidate_refuses_what_a_scorer_cannot_grade(fakes, work, monkeypatch):
    fake_harness(monkeypatch, work)
    limited = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", 20, run_id="r1",
                         pool="search", tasks=["gsm8k"], scorer="harness")
    assert limited["ok"] is False and "whole splits" in limited["error"]
    guard = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", 20, run_id="r1",
                       pool="search", tasks=["gsm8k", "if_guard"], scorer="interim")
    assert guard["ok"] is False and "if_guard" in guard["error"]
    unknown = call_local(modal_swarm.run_candidate, graft_program(), "search", 0, "pilot", None, run_id="r1",
                         pool="search", scorer="vllm")
    assert unknown["ok"] is False and "unknown scorer" in unknown["error"]
    assert work.calls == []
    assert compute_rows(fakes, "r1") == []


def test_host_baseline_grades_any_harness_task_and_keys_it_apart(fakes, work, monkeypatch):
    seen = []

    def fake_evaluate_harness(model, split, tasks, namespace):
        seen.append((model, split, list(tasks), namespace))
        work.clock.advance(9.0)
        scores = {}
        per_task = {}
        for name in tasks:
            scores[name] = 0.2
            per_task[name] = 150
        return {"scores": scores, "n_items": 150 * len(tasks), "n_items_per_task": per_task}

    monkeypatch.setattr(modal_swarm, "import_heavy_modules", lambda: None)
    monkeypatch.setattr(modal_swarm, "ready_cuda", lambda: None)
    monkeypatch.setattr(modal_swarm, "warm_models", lambda ids: list(ids))
    monkeypatch.setattr(modal_swarm, "setup_harness", lambda split, tasks: work.clock.advance(30.0))
    monkeypatch.setattr(modal_swarm, "evaluate_harness", fake_evaluate_harness)
    tasks = ["mmlupro:law", "if_guard"]
    first = call_local(modal_swarm.host_baseline, "search", None, tasks=tasks, scorer="harness")
    assert first["ok"] is True and first["scores"] == {"mmlupro:law": 0.2, "if_guard": 0.2}
    assert first["label"] == "HARNESS" and first["setup_seconds"] == 30.0 and first["seconds"] == {"eval": 9.0}
    assert seen == [(modal_swarm.HOST, "search", tasks, "host")]
    assert modal_swarm.lookup_host("search", None, eval_backend="harness", tasks=tasks) is not None
    assert modal_swarm.lookup_host("search", None) is None
    again = call_local(modal_swarm.host_baseline, "search", None, tasks=tasks, scorer="harness")
    assert again["cache_hit"] is True and len(seen) == 1
    refused = call_local(modal_swarm.host_baseline, "search", 20, tasks=tasks, scorer="harness")
    assert refused["ok"] is False and "whole splits" in refused["error"]
    assert len(fakes[modal_swarm.LEDGER_DICT].data) == 0
