"""Tests for how the studio loads and shows the whole parts library. CPU only, stub backend, no network."""

import json
import os
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from studio.server import catalog as catalog_mod
from studio.server import library
from studio.server import programs
from studio.server import settings
from studio.server import specs as specs_mod
from studio.server.app import app
from wsp.swarm import program_schema as ps

REAL_LIBRARY = os.path.join(settings.REPO_ROOT, "library")
GRAFTABLE = ("mlp_cluster", "attn_heads", "expert", "block")


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() != "":
                rows.append(json.loads(line))
    return rows


@pytest.fixture
def real_library(monkeypatch):
    """The repo library, loaded fresh."""
    monkeypatch.delenv("WSP_STUDIO_LIBRARY", raising=False)
    monkeypatch.delenv("WSP_STUDIO_ATLAS", raising=False)
    return library.get(reload=True)


def make_client(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", backend)
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    return TestClient(app)


@pytest.fixture
def stub_client(tmp_path, monkeypatch, real_library):
    return make_client(tmp_path, monkeypatch, "stub")


@pytest.fixture
def modal_client(tmp_path, monkeypatch, real_library):
    return make_client(tmp_path, monkeypatch, "modal")


# Loading.

def test_counts_match_the_index_by_kind(real_library):
    rows = read_rows(os.path.join(REAL_LIBRARY, "index.jsonl"))
    expected = {}
    for row in rows:
        expected[row["kind"]] = expected.get(row["kind"], 0) + 1
    assert real_library["failed"] == []
    assert real_library["rows"] == len(rows)
    for kind, count in expected.items():
        assert real_library["counts"][kind] == count, kind
    assert len(real_library["parts"]) == sum(expected.get(kind, 0) for kind in GRAFTABLE) + \
        real_library["counts"]["inserted_layer"]
    assert len(real_library["masks"]) == expected["deletion_mask"]
    assert len(real_library["model_list"]) == expected["model"]
    assert len(real_library["recipes"]) == expected["recipe"]
    assert len(real_library["findings"]) == expected["finding"]
    assert len(real_library["stitches"]) == expected["stitch"]


def test_loading_the_whole_library_is_fast(real_library):
    start = time.time()
    library.load()
    assert time.time() - start < 8.0  # a loose guard: over 1000 rows, and other jobs may share the machine


def test_a_broken_row_is_reported_and_the_rest_load(tmp_path, monkeypatch, real_library):
    copy = tmp_path / "library"
    shutil.copytree(REAL_LIBRARY, copy)
    good = real_library["rows"]
    with open(copy / "index.jsonl", "a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
        handle.write(json.dumps({"id": "ghost_part", "kind": "expert", "path": "components/none/ghost.json",
                                 "title": "ghost", "status": "candidate"}) + "\n")
        handle.write(json.dumps({"id": "odd_kind", "kind": "gadget", "path": "x.json"}) + "\n")
        handle.write(json.dumps({"id": "f_missing_line", "kind": "finding", "path": "findings/findings.jsonl#f_missing_line"}) + "\n")
    monkeypatch.setenv("WSP_STUDIO_LIBRARY", str(copy))
    data = library.get(reload=True)
    reasons = {}
    for item in data["failed"]:
        reasons[item["id"]] = item["reason"]
    assert len(data["failed"]) == 4
    assert "not JSON" in reasons["line " + str(good + 1)]
    assert "cannot read" in reasons["ghost_part"]
    assert "kind" in reasons["odd_kind"]
    assert "no line" in reasons["f_missing_line"]
    # The good rows still load: the count equals the expert rows of the real index, whatever its size today.
    expert_rows = 0
    with open(os.path.join(str(copy), "index.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except ValueError:
                continue  # the deliberately broken line
            if isinstance(row, dict) and row.get("kind") == "expert":
                expert_rows += 1
    assert data["counts"]["expert"] == expert_rows - 1  # every expert row but the ghost one
    assert data["counts"]["finding"] == 44
    line = library.counts_line(data)
    assert "failed 4" in line
    assert "ghost_part" in line
    with TestClient(app) as running:
        assert running.get("/api/health").status_code == 200
        assert running.get("/api/catalog").status_code == 200
    monkeypatch.delenv("WSP_STUDIO_LIBRARY")
    library.get(reload=True)


def test_incomplete_models_are_listed_and_flagged(stub_client, real_library):
    expected = []
    model_rows = 0
    for row in read_rows(os.path.join(REAL_LIBRARY, "index.jsonl")):
        if row["kind"] != "model":
            continue
        model_rows += 1
        with open(os.path.join(REAL_LIBRARY, row["path"]), encoding="utf-8") as handle:
            data = json.load(handle)
        if not data.get("hf_id") or not data.get("params_b"):
            expected.append(row["id"])
    # The library grows, so the counts come from the index, not fixed numbers.
    models = stub_client.get("/api/library/models").json()["models"]
    assert len(models) == model_rows
    flagged = sorted(item["id"] for item in models if item["incomplete"])
    assert flagged == sorted(expected)
    for item in models:
        if item["incomplete"]:
            assert len(item["missing"]) > 0
            assert item["title"]


# Routes.

def test_catalog_keeps_its_shape_and_has_counts(stub_client):
    palette = stub_client.get("/api/catalog").json()
    for key in ["product", "stub", "host", "parts", "masks", "train", "kinds", "library"]:
        assert key in palette
    counts = palette["counts"]
    assert counts["total"] == len(palette["parts"]) + len(palette["masks"])
    assert counts["kinds"]["deletion_mask"]["total"] == len(palette["masks"])
    assert sum(counts["status"].values()) == counts["total"]
    for part in palette["parts"]:
        assert isinstance(part["findings"], list)


def test_recipes_route(stub_client):
    data = stub_client.get("/api/library/recipes").json()["recipes"]
    assert len(data) == 23
    starting = 0
    for recipe in data:
        for key in ["id", "title", "status", "builds", "evidence", "starting_program", "no_start_reason", "start_problems"]:
            assert key in recipe
        assert recipe["status"] in ("candidate", "validated", "rejected")
        if recipe["starting_program"] is not None:
            starting += 1
            assert recipe["starting_program"]["host"]
            assert recipe["no_start_reason"] is None
        else:
            assert recipe["no_start_reason"]
    assert starting >= 5
    by_id = {}
    for recipe in data:
        by_id[recipe["id"]] = recipe
    assert by_id["bake_tap_mmlu_llama1b_qwen7b"]["starting_program"] is None
    assert by_id["lora_r1_gsm8k_llama1b"]["starting_program"]["lora"]["rank"] == 1


def test_findings_route_and_findings_on_parts(stub_client):
    findings = stub_client.get("/api/library/findings").json()["findings"]
    assert len(findings) == 44
    for item in findings:
        for key in ["id", "title", "claim", "numbers", "status", "about"]:
            assert key in item
        assert "parts" in item["about"] and "models" in item["about"]
    linked = [item for item in findings if len(item["about"]["parts"]) > 0]
    assert len(linked) > 0
    palette = stub_client.get("/api/catalog").json()
    by_id = {}
    for part in palette["parts"]:
        by_id[part["id"]] = part
    ids = set(item["id"] for item in findings)
    attached = 0
    for item in linked:
        for part_id in item["about"]["parts"]:
            if part_id in by_id:
                attached += 1
                assert item["id"] in [note["id"] for note in by_id[part_id]["findings"]]
    assert attached > 0
    for part in palette["parts"]:
        for note in part["findings"]:
            assert note["id"] in ids and note["title"]


def test_stitches_route(stub_client):
    stitches = stub_client.get("/api/library/stitches").json()["stitches"]
    assert len(stitches) == 58
    for item in stitches:
        for key in ["id", "host", "donor", "host_layers", "donor_layer", "method", "status", "fit"]:
            assert key in item
    assert any(item["fit"] for item in stitches)


# Placeable parts.

def graftable_in_root_catalog():
    count = 0
    for record in read_rows(os.path.join(settings.REPO_ROOT, "catalog.jsonl")):
        if record.get("type") in GRAFTABLE:
            count += 1
    return count


def test_more_parts_are_placeable_on_the_modal_catalog(modal_client):
    palette = modal_client.get("/api/catalog").json()
    placeable = [part for part in palette["parts"] if part["usable"]]
    before = graftable_in_root_catalog()
    assert before == 18
    assert len(placeable) > before + 300
    kinds = set(part["kind"] for part in placeable)
    assert {"expert", "block", "mlp_cluster", "attn_heads", "inserted_layer"} <= kinds
    assert palette["counts"]["placeable"] >= len(placeable)


def test_new_placeable_parts_have_what_the_compiler_needs(modal_client):
    palette = modal_client.get("/api/catalog").json()
    path = os.path.join(os.environ["WSP_STUDIO_OUT"], "catalog_modal.jsonl")
    records = {}
    for record in read_rows(path):
        records[record["id"]] = record
    built = 0
    for part in palette["parts"]:
        if not part["usable"] or not part["from_library"]:
            continue
        built += 1
        record = records[part["id"]]
        assert record["library_entry"]
        assert record["model"]
        assert "built from the library file" in part["evidence"]
        if record["type"] == "block":
            assert len(record["layers"]) == 2 and record["layers"][0] <= record["layers"][1]
        else:
            assert isinstance(record["layer"], int)
            assert record["type"] in ("mlp_cluster", "attn_heads", "expert")
            assert os.path.isfile(os.path.join(settings.REPO_ROOT, *record["units"].split("/")))
    assert built > 300


def test_a_library_expert_and_block_compile_into_a_valid_program(modal_client):
    palette = modal_client.get("/api/catalog").json()
    config, catalog = catalog_mod.load_catalog("modal")
    specs_mod.add_task_types(config)  # the server's validate route does the same
    chosen = {}
    for part in palette["parts"]:
        if part["usable"] and part["from_library"] and part["kind"] not in chosen:
            chosen[part["kind"]] = part
    assert "expert" in chosen and "block" in chosen
    for kind in ["expert", "block"]:
        part = chosen[kind]
        program = ps.make_program(config["host"], [ps.make_graft(part["id"], 6, glue_rank=64)],
                                  ps.make_heal(300, data="math"))
        assert programs.validate(program, catalog, config) == []


def test_locked_parts_say_what_is_missing(modal_client):
    palette = modal_client.get("/api/catalog").json()
    locked = [part for part in palette["parts"] if not part["usable"]]
    assert len(locked) > 0
    seen = set()
    for part in locked:
        reason = part["usable_reason"]
        assert reason
        assert "Not in the compiler catalog yet" not in reason
        seen.add(reason.split(".")[0])
    assert any("No unit indices in the library file" in reason for reason in seen)
    assert any("tap" in reason for reason in seen)


def test_library_parts_stay_locked_on_the_stub_backend(stub_client):
    palette = stub_client.get("/api/catalog").json()
    for part in palette["parts"]:
        if part["source"] in ("library", "atlas") and not part["usable"]:
            assert part["usable_reason"]
            assert part["from_library"] is False
