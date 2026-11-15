"""Tests for scripts/export_dashboard.py, the JSON exporter behind dashboard/.

CPU only, no network. Each test writes a tiny synthetic repo (one compare
folder with two runs, a smoke file and a pilot catalog) to a temp folder, so
every expected number is known by hand.
"""

import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script(name):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    path = os.path.join(REPO_ROOT, "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location("wsp_script_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load_script("export_dashboard")

HOST = "unsloth/Llama-3.2-1B-Instruct"


def program(part_id, at, steps=300):
    return {"host": HOST, "seed": 0, "masks": None, "task_vectors": [],
            "grafts": [{"id": part_id, "at": at, "control": None, "control_seed": 0, "glue_rank": None}],
            "heal": {"steps": steps, "data": "gsm8k_train", "lr": 0.0003, "batch_size": 8, "seq_len": 384}}


def eval_row(run_id, method, eval_count, compute, combined, status="ok", split="search", fidelity="full",
             program_hash="h1", hypothesis="try it", part_id="stub_qmath_l14_c0", at=8, fit=0.8):
    return {
        "timestamp": "2026-11-15T01:00:%02d.000Z" % int(eval_count * 2), "type": "eval", "run_id": run_id,
        "method": method, "backend": "stub", "stub": True, "agent": "planner-1", "round": 0,
        "hypothesis": hypothesis, "citations": [], "program": program(part_id, at), "program_hash": program_hash,
        "describe": "host + qmath.l14.c0@" + str(at) + ", heal 300 on gsm8k_train", "split": split,
        "fidelity": fidelity, "fit_r2": {part_id + "@" + str(at): fit}, "fit": {part_id + "@" + str(at): "ridge"},
        "graft_report": [{"id": part_id, "kind": "mlp_cluster", "control": None, "donor": "Qwen/Qwen2.5-Math-1.5B",
                          "donor_layer": 14, "host_layer": at, "d_host": 2048, "d_donor": 1536,
                          "part_params": 589824, "glue_params": 6291456, "glue_rank": None, "fit": "ridge",
                          "fit_r2": fit, "fit_pairs": 4096, "stub": True}],
        "heal_loss": [2.4, 2.0, 1.9], "scores": {"gsm8k": 0.3, "if_guard": 0.7}, "combined": combined,
        "status": status, "cache_hit": False, "token": int(eval_count), "eval_count": eval_count,
        "eval_budget": 4.0, "compute_s": {"compile": 35.0, "heal": 150.0, "eval": 120.0},
        "compute_used_s": compute, "compute_budget_s": 2000.0, "extra": {"params": 6881280, "eval_cost": 1.0},
    }


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_repo(root, hypothesis="try it"):
    """A compare folder with a swarm run and an evolve run, a smoke file and a pilot catalog."""
    swarm_rows = [
        eval_row("tiny-swarm-s0", "swarm", 1.0, 305.0, 0.10, program_hash="a1", hypothesis=hypothesis),
        eval_row("tiny-swarm-s0", "swarm", 1.34, 500.0, 0.50, fidelity="low", program_hash="a2"),
        eval_row("tiny-swarm-s0", "swarm", 2.0, 610.0, 0.05, program_hash="a3"),
        eval_row("tiny-swarm-s0", "swarm", 3.0, 915.0, 0.30, program_hash="a4", at=9, fit=0.6),
        eval_row("tiny-swarm-s0", "swarm", 1.0, 400.0, 0.20, split="heldout", program_hash="a4", at=9),
    ]
    evolve_rows = [
        eval_row("tiny-evolve-s0", "evolve", 1.0, 305.0, 0.20, program_hash="b1"),
        eval_row("tiny-evolve-s0", "evolve", 2.0, 610.0, -0.40, status="collapsed", program_hash="b2"),
        eval_row("tiny-evolve-s0", "evolve", 3.0, 915.0, 0.25, program_hash="b3"),
    ]
    swarm_log = os.path.join(root, "outputs", "runs", "tiny-swarm-s0", "experiments.jsonl")
    evolve_log = os.path.join(root, "outputs", "runs", "tiny-evolve-s0", "experiments.jsonl")
    write_jsonl(swarm_log, swarm_rows)
    write_jsonl(evolve_log, evolve_rows)
    manifest = {
        "name": "tiny", "backend": "stub", "cascade": False, "created": "2026-11-15T01:00:00Z",
        "budgets": {"eval_budget": 4.0, "compute_budget_s": 2000.0, "heldout_eval_budget": 3.0,
                    "heldout_compute_budget_s": 1000.0},
        "budget_check": {"equal": True, "mismatches": []},
        "outputs_dir": os.path.join(root, "outputs"),
        "runs": [
            {"run_id": "tiny-swarm-s0", "condition": "swarm", "seed": 0, "log_path": swarm_log,
             "run_dir": os.path.dirname(swarm_log), "compute_overshoot_s": 0.0},
            {"run_id": "tiny-evolve-s0", "condition": "evolve", "seed": 0, "log_path": evolve_log,
             "run_dir": os.path.dirname(evolve_log), "compute_overshoot_s": 0.0},
        ],
    }
    os.makedirs(os.path.join(root, "outputs", "compare", "tiny"), exist_ok=True)
    with open(os.path.join(root, "outputs", "compare", "tiny", "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    smoke = {
        "run_id": "smoke-1794705164", "labels": ["SMOKE", "SEARCH", "INTERIM"], "part": "pilot_x", "heal_steps": 50,
        "eval_limit": 20, "gpu_spec": "L4", "checks": {"graft_ok": True},
        "host": {"result": {"scores": {"gsm8k": 0.25}, "label": "INTERIM", "n_items": 20, "gpu": "NVIDIA L4",
                            "samples": ["an answer"]}},
        "graft": {"result": {"scores": {"gsm8k": 0.15}, "label": "INTERIM", "n_items": 20, "gpu": "NVIDIA L4",
                             "compile_report": {"grafts": [{"id": "pilot_x", "donor_layer": 18, "host_layer": 10,
                                                            "fit_r2": 0.64}],
                                                "heal": {"spec": {"steps": 50}, "losses": [1.6, 1.2]}}}},
        "ledger": {"compute_used_s": 84.0, "spends": 2, "tokens_left": 1, "devices": ["NVIDIA L4"]},
    }
    os.makedirs(os.path.join(root, "outputs", "smoke"), exist_ok=True)
    with open(os.path.join(root, "outputs", "smoke", "modal_smoke_1794705164.json"), "w", encoding="utf-8") as handle:
        json.dump(smoke, handle)
    write_jsonl(os.path.join(root, "data", "pilot_catalog.jsonl"), [
        {"id": "pilot_x_top256", "layer": 6, "model": "Qwen/Qwen2.5-Math-1.5B", "pilot": True,
         "signature": {"effect": {}, "fit_r2": {}}, "type": "mlp_cluster", "trained_on": "top 256 neurons",
         "units": "units/pilot/pilot_x_top256.pt"},
    ])
    return root


def build(root, max_bytes=export.MAX_BYTES):
    files = export.build_export(str(root), max_bytes)
    return files, json.loads(files["index.json"]), json.loads(files["dataset-tiny.json"])


def test_curves_match_best_so_far_on_both_axes(tmp_path):
    files, index, dataset = build(make_repo(str(tmp_path)))
    runs = {}
    for run in dataset["runs"]:
        runs[run["condition"]] = run
    swarm = runs["swarm"]["curves"]
    # The low-fidelity 0.50 and the held-out 0.20 never count; 0.05 does not beat 0.10.
    assert swarm["evals"]["steps"] == [[1, 0.1], [3, 0.3]]
    assert swarm["compute"]["steps"] == [[305, 0.1], [915, 0.3]]
    assert swarm["evals"]["end"] == 3
    evolve = runs["evolve"]["curves"]
    assert evolve["evals"]["steps"] == [[1, 0.2], [3, 0.25]]
    assert runs["swarm"]["final_best"] == 0.3
    # One seed per arm, so the band's median is that seed.
    swarm_arm = dataset["arms"][dataset["conditions"].index("swarm")]
    assert swarm_arm["band"]["evals"]["median"][-1] == 0.3
    assert index["datasets"][0]["id"] == "tiny"


def test_labels_carry_through_from_rows(tmp_path):
    files, index, dataset = build(make_repo(str(tmp_path)))
    defaults = dataset["row_defaults"]
    labels = []
    for row in dataset["log"]:
        merged = dict(defaults)
        merged.update(row)
        labels.append((merged["split"], merged["labels"]))
    assert ("search", ["SEARCH", "STUB"]) in labels
    assert ("heldout", ["HELD-OUT", "STUB"]) in labels
    assert dataset["stub"] is True
    assert "STUB" in dataset["labels"]
    assert dataset["fit"]["labels"] == ["SEARCH", "STUB"]
    assert index["all_search_stub"] is True
    smoke = json.loads(files["smoke.json"])
    for model in smoke["runs"][0]["models"]:
        assert "SMOKE" in model["labels"]
        assert "INTERIM" in model["labels"]


def test_program_and_fit_tables_rebuild_the_logged_program(tmp_path):
    files, index, dataset = build(make_repo(str(tmp_path)))
    entry = dataset["programs"]["a4"]
    rebuilt = dict(dataset["program_base"])
    rebuilt.update(entry["program"])
    assert rebuilt == program("stub_qmath_l14_c0", 9)
    assert entry["grafts"][0]["host_layer"] == 9
    assert dataset["parts"]["stub_qmath_l14_c0"]["donor_layer"] == 14
    # Fit points: every graft of each scored full-fidelity search eval (ok or collapsed), not low or held-out.
    assert dataset["fit"]["n"] == 6
    assert dataset["budget_check"]["equal_caps"] is True


def test_paths_are_relative_and_local_paths_never_appear(tmp_path):
    root = make_repo(str(tmp_path), hypothesis="read " + str(tmp_path) + "/notes.txt first")
    files, index, dataset = build(root)
    for name in files:
        text = files[name]
        assert str(tmp_path) not in text
        assert str(tmp_path).replace("\\", "/") not in text
        assert json.dumps(str(tmp_path))[1:-1] not in text
    sources = []
    for run in dataset["runs"]:
        sources.append(run["source"])
    assert "outputs/runs/tiny-swarm-s0/experiments.jsonl" in sources
    assert dataset["source"] == "outputs/compare/tiny"
    hypotheses = []
    for row in dataset["log"]:
        hypotheses.append(row["hypothesis"])
    assert "read notes.txt first" in hypotheses
    assert "units" not in json.loads(files["parts.json"])["catalogs"][0]["parts"][0]


def test_env_values_and_token_shapes_block_the_export(tmp_path):
    secret = "zq" + "9" * 30
    root = make_repo(str(tmp_path), hypothesis="the key is " + secret)
    with open(os.path.join(root, ".env"), "w", encoding="utf-8") as handle:
        handle.write("SOME_TOKEN=" + secret + "\n")
    with pytest.raises(export.ExportError) as caught:
        export.build_export(root)
    assert "SOME_TOKEN" in str(caught.value)
    assert secret not in str(caught.value)
    token_root = make_repo(str(tmp_path / "other"), hypothesis="ak-" + "A" * 20)
    with pytest.raises(export.ExportError):
        export.build_export(token_root)


def test_size_guard_trims_then_refuses(tmp_path):
    root = make_repo(str(tmp_path))
    # Sixty extra low-fidelity rows give the trim steps something to cut.
    log_path = os.path.join(root, "outputs", "runs", "tiny-swarm-s0", "experiments.jsonl")
    extra = []
    for index in range(60):
        extra.append(eval_row("tiny-swarm-s0", "swarm", 3.0 + 0.34 * (index + 1), 1000.0 + index, 0.01 * index,
                              fidelity="low", program_hash="low" + str(index)))
    with open(log_path, "a", encoding="utf-8") as handle:
        for row in extra:
            handle.write(json.dumps(row) + "\n")
    files, index, dataset = build(root)
    assert index["trimmed"] == []
    full_size = export.total_size(files)
    # A cap under the full size forces trim steps, the result fits, and the index says what was cut.
    cap = full_size - 4000
    trimmed_files = export.build_export(root, cap)
    trimmed_index = json.loads(trimmed_files["index.json"])
    assert len(trimmed_index["trimmed"]) >= 1
    assert export.total_size(trimmed_files) <= cap + 200
    trimmed_dataset = json.loads(trimmed_files["dataset-tiny.json"])
    assert trimmed_dataset["counts"]["eval_rows_kept"] < trimmed_dataset["counts"]["eval_rows"]
    # Held-out and collapsed rows survive every trim.
    statuses = []
    for row in trimmed_dataset["log"]:
        merged = dict(trimmed_dataset["row_defaults"])
        merged.update(row)
        statuses.append((merged["split"], merged["status"]))
    assert ("heldout", "ok") in statuses
    assert ("search", "collapsed") in statuses
    with pytest.raises(export.ExportError):
        export.build_export(root, 1000)


def test_write_export_removes_stale_dataset_files(tmp_path):
    root = make_repo(str(tmp_path / "repo"))
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "dataset-old.json"), "w", encoding="utf-8") as handle:
        handle.write("{}")
    files = export.build_export(root)
    export.write_export(files, out_dir)
    names = sorted(os.listdir(out_dir))
    assert "dataset-old.json" not in names
    assert "dataset-tiny.json" in names
    assert "index.json" in names
    final = json.loads(open(os.path.join(out_dir, "final.json"), encoding="utf-8").read())
    assert final["available"] is False
