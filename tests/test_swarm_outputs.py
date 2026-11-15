"""Tests for the outputs wave: compare.py, plot_curves.py, agent_log.py, sync_log.py.

CPU only, stub backend, no network. Most tests use small synthetic run logs
written to a temp folder, so every expected number is known by hand.
"""

import html.parser
import importlib
import importlib.util
import io
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script(name):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    path = os.path.join(REPO_ROOT, "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location("wsp_script_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load_script("plot_curves")
agent_log = load_script("agent_log")
sync_log = load_script("sync_log")
compare = load_script("compare")


def baselines_ready():
    try:
        module = importlib.import_module("wsp.swarm.baselines")
    except ImportError:
        return False
    return hasattr(module, "run_baseline")


# Synthetic logs.

def stamp(index):
    return "2026-11-14T10:00:%02d.000Z" % index


def eval_row(run_id, method, seconds, evals, combined, status="ok", token=None, fit=None, stub=True,
             split="search", fidelity="full", second=0, hypothesis="a hypothesis", describe="host + x@3"):
    if fit is None:
        fit = {"part_a@3": 0.5}
    return {"timestamp": stamp(second), "type": "eval", "run_id": run_id, "method": method,
            "backend": "stub" if stub else "local", "stub": stub, "agent": "planner-1", "round": 0,
            "hypothesis": hypothesis, "describe": describe, "split": split, "fidelity": fidelity,
            "fit_r2": fit, "scores": {}, "combined": combined, "status": status, "token": token,
            "eval_count": evals, "compute_used_s": seconds, "extra": {}}


def write_run(folder, run_id, seed, rows):
    """Write runs/<run_id>/experiments.jsonl and a small manifest. Returns the log path."""
    run_dir = os.path.join(str(folder), "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "experiments.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "seed": seed}, handle)
    return path


def scored(run_id, method, points, stub=True):
    """Eval rows from (seconds, evals, combined) triples, tokens in order."""
    rows = []
    for index in range(len(points)):
        seconds, evals, combined = points[index]
        rows.append(eval_row(run_id, method, seconds, evals, combined, token=index, stub=stub, second=index))
    return rows


# The evolve runs reach 0.3 (the median final best) at 900 s / 9 evals and 600 s / 6 evals;
# the swarm runs reach it at 300 s / 3 evals and 450 s / 4.5 evals. Both ratios are 2.0.
EVOLVE_POINTS = [
    [(500.0, 5.0, 0.2)],
    [(300.0, 3.0, 0.1), (900.0, 9.0, 0.3)],
    [(600.0, 6.0, 0.35), (1200.0, 12.0, 0.4)],
]
SWARM_POINTS = [
    [(300.0, 3.0, 0.31)],
    [(200.0, 2.0, 0.1), (450.0, 4.5, 0.3)],
    [(250.0, 2.5, 0.05), (700.0, 7.0, 0.25)],
]


def make_runs(points_by_seed, method, stub=True):
    runs = []
    for seed in range(len(points_by_seed)):
        run_id = method + "-s" + str(seed)
        runs.append({"condition": method, "seed": seed, "run_id": run_id,
                     "rows": scored(run_id, method, points_by_seed[seed], stub), "path": None})
    return runs


def make_compare_folder(tmp_path, stub=True):
    """A compare folder whose manifest lists synthetic swarm and evolve runs."""
    outputs = tmp_path / "outputs"
    records = []
    for method, points_by_seed in [("swarm", SWARM_POINTS), ("evolve", EVOLVE_POINTS)]:
        for seed in range(len(points_by_seed)):
            run_id = "t-" + method + "-s" + str(seed)
            path = write_run(outputs, run_id, seed, scored(run_id, method, points_by_seed[seed], stub))
            records.append({"condition": method, "seed": seed, "run_id": run_id, "log_path": path})
    folder = outputs / "compare" / "t"
    os.makedirs(str(folder))
    with open(str(folder / "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"name": "t", "runs": records}, handle)
    return str(folder)


# Curves.

def test_best_so_far_uses_only_full_fidelity_ok_search_rows():
    rows = [
        eval_row("r", "swarm", 100.0, 1.0, 0.1, token=0, second=1),
        eval_row("r", "swarm", 200.0, 2.0, 0.5, status="collapsed", token=1, second=2),
        eval_row("r", "swarm", 250.0, 2.34, 0.9, fidelity="low", token=2, second=3),
        eval_row("r", "swarm", 300.0, 3.34, 0.05, token=3, second=4),
        eval_row("r", "swarm", 400.0, 4.34, 0.3, token=4, second=5),
        eval_row("r", "swarm", 400.0, 4.34, 0.3, status="cache_hit", second=6),
        eval_row("r", "swarm", 9999.0, 1.0, 0.95, split="heldout", token=0, second=8),
        {"timestamp": stamp(7), "type": "probe", "run_id": "r", "status": "ok", "combined": None,
         "compute_used_s": 500.0, "eval_count": 4.34, "split": None},
    ]
    assert plot.curve_points(rows, "compute") == [(100.0, 0.1), (300.0, 0.05), (400.0, 0.3)]
    steps = plot.best_so_far(plot.curve_points(rows, "compute"))
    assert steps == [(100.0, 0.1), (400.0, 0.3)]
    assert plot.best_so_far(plot.curve_points(rows, "evals")) == [(1.0, 0.1), (4.34, 0.3)]
    assert plot.value_at(steps, 50.0) is None
    assert plot.value_at(steps, 399.0) == 0.1
    assert plot.value_at(steps, 1e9) == 0.3
    assert plot.first_reach(steps, 0.2) == 400.0
    assert plot.first_reach(steps, 0.31) is None
    assert plot.run_end(rows, "compute") == 500.0
    assert plot.final_best(rows) == 0.3


def test_median_and_min_max_band():
    curves = [
        ([(1.0, 0.1), (3.0, 0.4)], 5.0),
        ([(2.0, 0.2)], 4.0),
        ([(1.0, 0.3), (4.0, 0.5)], 6.0),
    ]
    result = plot.band(curves)
    assert result["x"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["median"][0] is None and result["low"][0] is None and result["high"][0] is None
    assert result["median"][1:] == pytest.approx([0.2, 0.3, 0.4, 0.4, 0.4])
    assert result["low"][1:] == pytest.approx([0.1, 0.2, 0.2, 0.2, 0.2])
    assert result["high"][1:] == pytest.approx([0.3, 0.4, 0.5, 0.5, 0.5])


def test_spearman_on_known_data():
    assert plot.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert plot.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert plot.spearman([1, 2, 3, 4, 5], [1, 8, 27, 64, 125]) == pytest.approx(1.0)
    # No ties: 1 - 6 * sum(d^2) / (n (n^2 - 1)) = 1 - 6 * 4 / 120 = 0.8.
    assert plot.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)
    # Ties share their average rank: x ranks 1, 2.5, 2.5, 4 give 4.5 / sqrt(4.5 * 5).
    assert plot.average_ranks([1, 2, 2, 3]) == [1.0, 2.5, 2.5, 4.0]
    assert plot.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(4.5 / (4.5 * 5) ** 0.5)
    assert plot.spearman([1, 1, 1], [1, 2, 3]) is None
    assert plot.spearman([1], [1]) is None


def test_fit_points_one_per_graft_in_scored_full_search_rows():
    rows = [
        eval_row("r", "swarm", 1.0, 1.0, 0.2, token=0, fit={"a@3": 0.7, "b@5": 0.4}),
        eval_row("r", "swarm", 2.0, 2.0, -0.5, status="collapsed", token=1, fit={"a@3": 0.7}),
        eval_row("r", "swarm", 3.0, 2.0, None, status="rejected_fit", fit={"c@1": 0.1}),
        eval_row("r", "swarm", 4.0, 3.0, 0.3, token=2, fit={"d@2": None}),
        eval_row("r", "swarm", 5.0, 3.34, 0.9, fidelity="low", token=3, fit={"e@2": 0.9}),
        eval_row("r", "swarm", 6.0, 1.0, 0.8, split="heldout", token=0, fit={"f@2": 0.9}),
    ]
    assert plot.fit_points(rows) == [(0.7, 0.2), (0.4, 0.2), (0.7, -0.5)]


def test_stub_watermark_decision():
    assert plot.needs_stub_watermark([]) is False
    assert plot.needs_stub_watermark([{"backend": "local", "stub": False}]) is False
    assert plot.needs_stub_watermark([{"backend": "local", "stub": False}, {"backend": "stub"}]) is True
    assert plot.needs_stub_watermark([{"backend": "modal", "stub": True}]) is True


def test_times_fewer_numbers():
    result = plot.times_fewer(make_runs(SWARM_POINTS, "swarm"), make_runs(EVOLVE_POINTS, "evolve"))
    assert result["target"] == 0.3
    assert result["compute"]["fast"] == [300.0, 450.0, None]
    assert result["compute"]["slow"] == [None, 900.0, 600.0]
    assert result["compute"]["fast_median"] == 450.0 and result["compute"]["slow_median"] == 900.0
    assert result["compute"]["ratio"] == pytest.approx(2.0)
    assert result["evals"]["ratio"] == pytest.approx(2.0)
    # An even seed count takes the lower middle final best, so at least half the baseline seeds reach it.
    two = plot.times_fewer(make_runs(SWARM_POINTS, "swarm"), make_runs(EVOLVE_POINTS[1:], "evolve"))
    assert two["target"] == 0.3 and two["evals"]["slow_median"] == 7.5
    assert plot.times_fewer(make_runs(SWARM_POINTS, "swarm"), []) is None


def test_plot_main_draws_three_charts(tmp_path, capsys):
    folder = make_compare_folder(tmp_path)
    result = plot.main([folder])
    assert len(result["paths"]) == 3
    for path in result["paths"]:
        assert os.path.dirname(path) == folder
        with open(path, "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
    assert result["stub"] is True
    assert result["conditions"] == ["swarm", "evolve"]
    assert result["n_fit_points"] == 10
    printed = capsys.readouterr().out
    assert "[SEARCH][STUB]" in printed and "2.00x fewer" in printed
    only = plot.main([folder, "--conditions", "evolve", "--out", str(tmp_path / "only")])
    assert only["conditions"] == ["evolve"] and only["n_fit_points"] == 5
    capsys.readouterr()

    clean = tmp_path / "clean"
    path = write_run(clean, "real-s0", 0, scored("real-s0", "swarm", SWARM_POINTS[0], stub=False))
    result = plot.main([path, "--out", str(clean / "plots")])
    assert result["stub"] is False
    printed = capsys.readouterr().out
    assert "STUB" not in printed and "[SEARCH]" in printed and "needs a 'evolve' condition" in printed


# Sync.

def test_sync_merges_sorts_and_dedupes(tmp_path):
    outputs = tmp_path / "outputs"
    run_a = [
        eval_row("a", "swarm", 10.0, 1.0, 0.1, token=0, second=2),
        {"timestamp": stamp(4), "type": "probe", "run_id": "a", "token": None},
        {"timestamp": stamp(4), "type": "event", "run_id": "a", "kind": "judge_verdict", "text": "v"},
        eval_row("a", "swarm", 20.0, 1.0, 0.1, status="cache_hit", second=5),
    ]
    run_b = [
        {"timestamp": stamp(1), "type": "event", "run_id": "b", "kind": "reset", "text": "start"},
        eval_row("b", "evolve", 10.0, 1.0, 0.2, token=0, second=3),
    ]
    write_run(outputs, "a", 0, run_a)
    write_run(outputs, "b", 0, run_b)
    shared = tmp_path / "experiments.jsonl"
    synced_before = dict(run_a[0])
    synced_before["seq"] = 7
    old = {"timestamp": stamp(0), "type": "event", "run_id": "old", "kind": "reset", "text": "an older run", "seq": 0}
    with open(str(shared), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(old) + "\n")
        handle.write(json.dumps(synced_before) + "\n")

    modal_items = [("a:000001", {"timestamp": stamp(4), "type": "probe", "run_id": "a", "token": None}),
                   ("c:000000", {"timestamp": stamp(6), "type": "event", "run_id": "c", "kind": "reset"}),
                   ("c:000001", "not a row")]
    modal = sync_log.rows_from_dict_items(modal_items)
    assert modal[0]["seq"] == 1 and len(modal) == 2

    result = sync_log.sync(str(outputs), str(shared), [modal])
    rows = []
    with open(str(shared), encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    keys = []
    for row in rows:
        keys.append((row["run_id"], row["type"], row.get("seq")))
    assert keys == [("old", "event", 0), ("b", "event", 0), ("a", "eval", 7), ("b", "eval", 1),
                    ("a", "probe", 1), ("a", "event", 2), ("a", "eval", 3), ("c", "event", 0)]
    assert result["duplicates"] == 2 and result["rows_written"] == 8

    again = sync_log.sync(str(outputs), str(shared), [modal])
    with open(str(shared), encoding="utf-8") as handle:
        assert len(handle.readlines()) == 8
    assert again["duplicates"] == again["rows_read"] - 8
    assert sync_log.seq_from_key("run:x:000042") == 42 and sync_log.seq_from_key("nokey") is None


def test_sync_keeps_search_and_heldout_rows_with_the_same_token_and_time(tmp_path):
    outputs = tmp_path / "outputs"
    search = eval_row("a", "full_ft", 10.0, 1.0, 0.1, token=0, second=2)
    heldout = eval_row("a", "full_ft", 10.0, 1.0, 0.1, token=0, second=2, split="heldout")
    write_run(outputs, "a", 0, [search, heldout])
    shared = str(tmp_path / "shared.jsonl")
    result = sync_log.sync(str(outputs), shared)
    assert result["rows_written"] == 2 and result["duplicates"] == 0
    again = sync_log.sync(str(outputs), shared)
    assert again["rows_written"] == 2 and again["duplicates"] == 2


def test_sync_skips_a_half_written_line(tmp_path):
    outputs = tmp_path / "outputs"
    path = write_run(outputs, "a", 0, [eval_row("a", "swarm", 1.0, 1.0, 0.1, token=0)])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"timestamp": "2026')
    result = sync_log.sync(str(outputs), str(tmp_path / "shared.jsonl"))
    assert result["rows_written"] == 1 and result["bad_lines"] == 1


# Agent log.

class TableCounter(html.parser.HTMLParser):
    def __init__(self):
        html.parser.HTMLParser.__init__(self)
        self.rows = 0
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "tr":
            self.rows += 1


def test_agent_log_html_page(tmp_path):
    rows = [
        eval_row("a", "swarm", 1.0, 1.0, 0.1234, token=0, second=1, hypothesis="older idea",
                 describe="host + qmath.l14.c0@8, heal 300"),
        eval_row("b", "evolve", 1.0, 2.0, None, status="rejected_fit", second=2, hypothesis="<script>x</script>"),
        eval_row("a", "swarm", 1.0, 3.0, 0.2, token=1, second=3, hypothesis="newest idea"),
    ]
    page = agent_log.build_html(agent_log.time_order(rows), ["a.jsonl", "b.jsonl"])
    assert "STUB: these rows come from the stub backend" in page
    assert page.index("newest idea") < page.index("older idea")
    assert "host + qmath.l14.c0@8, heal 300" in page and "part_a@3 0.50" in page
    assert "0.1234 SEARCH STUB" in page and "rejected_fit" in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert "<link" not in page and "src=" not in page and "http" not in page
    assert "prefers-color-scheme: dark" in page
    counter = TableCounter()
    counter.feed(page)
    assert counter.rows == 4 and "table" in counter.tags

    clean = []
    for row in rows:
        copy = dict(row)
        copy["stub"] = False
        copy["backend"] = "local"
        clean.append(copy)
    assert "STUB" not in agent_log.build_html(clean, ["a.jsonl"])

    path = write_run(tmp_path, "a", 0, rows)
    out = io.StringIO()
    written = agent_log.main([os.path.dirname(path), "--html", str(tmp_path / "log.html")], out)
    assert os.path.exists(written) and "wrote" in out.getvalue()


def test_agent_log_terminal_and_tail(tmp_path):
    path = write_run(tmp_path, "a", 0, [
        eval_row("a", "swarm", 1.0, 1.0, 0.1234, token=0, second=2, hypothesis="graft the math block"),
        {"timestamp": stamp(1), "type": "event", "run_id": "a", "kind": "reset", "text": "x"},
        eval_row("a", "swarm", 2.0, 2.0, 0.3, token=0, split="heldout", second=3),
    ])
    out = io.StringIO()
    shown = agent_log.main([path], out)
    text = out.getvalue()
    assert len(shown) == 2 and "[STUB]" in text.splitlines()[0]
    assert "planner-1" in text and "why: graft the math block" in text and "host + x@3" in text
    assert "0.1234 SEARCH STUB" in text and "0.3000 HELDOUT STUB" in text and "part_a@3 0.50" in text

    growing = str(tmp_path / "growing.jsonl")
    tail = agent_log.LogTail(growing)
    assert tail.poll() == []
    line = json.dumps(eval_row("g", "swarm", 1.0, 1.0, 0.5, token=0))
    with open(growing, "w", encoding="utf-8") as handle:
        handle.write(line + "\n" + line[:20])
    assert len(tail.poll()) == 1
    assert tail.poll() == []
    with open(growing, "a", encoding="utf-8") as handle:
        handle.write(line[20:] + "\n")
    assert len(tail.poll()) == 1

    out = io.StringIO()
    agent_log.follow(growing, out, poll_s=0, max_polls=1)
    assert out.getvalue().count("SEARCH STUB") == 2


# Compare.

def test_parse_conditions():
    assert compare.parse_conditions("swarm, evolve,swarm") == ["swarm", "evolve"]
    with pytest.raises(compare.CompareError):
        compare.parse_conditions("swarm,cmaes")


def test_compare_fails_before_any_run_without_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "load_dotenv", lambda: [])
    monkeypatch.setitem(sys.modules, "wsp.swarm.baselines", None)  # import now fails as if not built
    with pytest.raises(compare.CompareError) as error:
        compare.main(["--name", "t", "--outputs-dir", str(tmp_path), "--conditions", "swarm,evolve"])
    assert "not built yet" in str(error.value) and "evolve" in str(error.value)
    assert not os.path.exists(str(tmp_path / "compare")) and not os.path.exists(str(tmp_path / "runs"))


def test_compare_preflight_refuses_a_baseline_that_cannot_run(tmp_path, monkeypatch):
    if not baselines_ready():
        pytest.skip("wsp.swarm.baselines.run_baseline is not built yet")
    monkeypatch.setattr(compare, "load_dotenv", lambda: [])
    with pytest.raises(compare.CompareError) as error:
        compare.main(["--name", "t", "--outputs-dir", str(tmp_path), "--backend", "local", "--conditions",
                      "swarm,lora"])
    assert "nothing ran" in str(error.value) and "t-lora-s0" in str(error.value)
    assert not os.path.exists(str(tmp_path / "compare")) and not os.path.exists(str(tmp_path / "runs"))


def test_compare_swarm_llm_preflight_and_llm_config(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "load_dotenv", lambda: [])
    outputs = str(tmp_path / "outputs")
    with pytest.raises(compare.CompareError) as error:
        compare.main(["--name", "a", "--outputs-dir", outputs, "--conditions", "swarm,swarm_llm", "--llm", "mock"])
    assert "nothing ran" in str(error.value) and "mock_replies" in str(error.value)
    assert not os.path.exists(os.path.join(outputs, "compare"))

    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps({"eval_budget": 99}))
    with pytest.raises(compare.CompareError) as error:
        compare.main(["--name", "b", "--outputs-dir", outputs, "--conditions", "swarm_llm", "--llm", "mock",
                      "--llm-config", str(shared)])
    assert "may not set 'eval_budget'" in str(error.value)

    replies = tmp_path / "mock.json"
    replies.write_text(json.dumps({"mock_replies": ["no proposals here"]}))
    manifest = compare.main(["--name", "c", "--outputs-dir", outputs, "--seeds", "1", "--eval-budget", "4",
                             "--conditions", "swarm_llm", "--llm", "mock", "--llm-config", str(replies)])
    assert manifest["ran"] == ["swarm_llm"] and manifest["skipped"] == {}
    assert manifest["swarm_llm"]["llm_config"] == {"mock_replies": ["no proposals here"]}
    record = manifest["runs"][0]
    assert record["method"] == "swarm_llm" and os.path.exists(record["log_path"])
    assert record["run_budgets"]["eval_budget"] == 4.0
    assert record["run_settings"]["fidelities"] == {"check": 0.16, "low": 0.34, "full": 1.0}
    assert manifest["budget_check"]["equal"] is True


def test_llm_config_refuses_shared_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "load_dotenv", lambda: [])
    outputs = str(tmp_path / "outputs")
    refused = [
        ({"fidelities": {"check": 0.16, "low": 0.34, "full": 0.1}}, "may not set 'fidelities'"),
        ({"glue_ranks": [None, 64, 8]}, "may not set 'glue_ranks'"),
        ({"fit_threshold": 0.0}, "may not set 'fit_threshold'"),
        ({"memory_cleanup": False}, "may not set 'memory_cleanup'"),
        ({"openai_model": "m", "max_rounds_extra": 9}, "'max_rounds_extra' is not one"),
    ]
    index = 0
    for settings, message in refused:
        path = tmp_path / ("refused" + str(index) + ".json")
        path.write_text(json.dumps(settings))
        with pytest.raises(compare.CompareError) as error:
            compare.main(["--name", "r" + str(index), "--outputs-dir", outputs, "--conditions", "swarm,swarm_llm",
                          "--llm", "mock", "--llm-config", str(path)])
        assert message in str(error.value)
        index += 1
    assert not os.path.exists(outputs)

    for key in ["openai_model", "anthropic_effort", "llm_temperature", "modal_llm_app", "mock_replies",
                "memory_level", "past_base_url", "exa_cache_dir", "priors_path"]:
        assert compare.llm_key_allowed(key), key
    for key in ["glue_ranks", "fidelities", "stub_seed", "max_rounds", "k", "memory_cleanup", "seed", "host"]:
        assert not compare.llm_key_allowed(key), key


def test_budget_check_compares_shared_settings():
    budgets = {"eval_budget": 6.0}
    settings = {"fidelities": {"check": 0.16, "low": 0.34, "full": 1.0}, "k": 4, "glue_ranks": None}

    def record(run_id, used):
        return {"run_id": run_id, "run_budgets": {"eval_budget": 6}, "run_settings": used}

    same = record("a", {"fidelities": {"full": 1.0, "low": 0.34, "check": 0.16}, "k": 4})
    assert compare.budget_check(budgets, [same], settings) == {"equal": True, "mismatches": []}

    cheap = record("b", {"fidelities": {"check": 0.16, "low": 0.34, "full": 0.1}, "k": 4})
    ranks = record("c", {"fidelities": settings["fidelities"], "k": 4, "glue_ranks": [64]})
    no_k = record("d", {"fidelities": settings["fidelities"]})
    result = compare.budget_check(budgets, [same, cheap, ranks, no_k], settings)
    assert result["equal"] is False and len(result["mismatches"]) == 3
    assert result["mismatches"][0].startswith("b: fidelities")
    assert result["mismatches"][1] == "c: glue_ranks [64] != null"
    assert result["mismatches"][2] == "d: k missing from its manifest"


def test_compare_on_stub(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, "load_dotenv", lambda: [])
    conditions = "swarm,swarm_llm"
    ran = ["swarm"]
    if baselines_ready():
        conditions = "swarm,swarm_llm,evolve,random"
        ran = ["swarm", "evolve", "random"]
    outputs = str(tmp_path / "outputs")
    manifest = compare.main(["--name", "t", "--outputs-dir", outputs, "--seeds", "2", "--eval-budget", "6",
                             "--compute-budget-s", "20000", "--conditions", conditions])
    assert manifest["status"] == "done" and manifest["ran"] == ran
    assert "no LLM configured" in manifest["skipped"]["swarm_llm"]
    assert manifest["seeds"] == [0, 1] and manifest["backend"] == "stub" and manifest["stub"] is True
    assert manifest["budgets"]["eval_budget"] == 6.0 and manifest["budgets"]["compute_budget_s"] == 20000.0
    assert "git_commit" in manifest and manifest["budget_check"]["equal"] is True
    assert manifest["shared_settings"]["k"] == 4 and manifest["shared_settings"]["glue_ranks"] == [None, 64]
    assert len(manifest["runs"]) == 2 * len(ran)
    for record in manifest["runs"]:
        assert os.path.exists(record["log_path"])
        assert record["run_id"] == "t-" + record["condition"] + "-s" + str(record["seed"])
        assert record["ledger_search"]["eval"] <= 6.0
        if record["best_search"] is not None:
            assert record["best_search"]["label"] == "SEARCH"
    with open(os.path.join(outputs, "compare", "t", "manifest.json"), encoding="utf-8") as handle:
        assert json.load(handle)["status"] == "done"
    assert "[SEARCH][STUB]" in capsys.readouterr().out

    result = plot.main([os.path.join(outputs, "compare", "t")])
    assert result["stub"] is True and len(result["paths"]) == 3 and result["n_fit_points"] > 0
    if baselines_ready():
        assert result["conditions"] == ["swarm", "evolve", "random"]
        assert "swarm" in result["times_fewer"]

    with pytest.raises(compare.CompareError) as error:
        compare.main(["--name", "t", "--outputs-dir", outputs, "--seeds", "2", "--conditions", "swarm"])
    assert "already" in str(error.value)
