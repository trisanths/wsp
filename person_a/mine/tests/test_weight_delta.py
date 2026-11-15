# Tests for mine/weight_delta.py. Plain asserts, no pytest:
#   .venv/bin/python mine/tests/test_weight_delta.py
# The math and the name matching run on small synthetic safetensors files in a temp folder.
# Nothing here touches Modal or the network.
import json
import math
import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import weight_delta as wd  # noqa: E402


def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def t(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype)


# ---------- math ----------

def test_pair_sums_by_hand():
    # member (1, 2, 2), reference (1, 0, 0): difference (0, 2, 2).
    sums = wd.pair_sums(t([1.0, 2.0, 2.0]), t([1.0, 0.0, 0.0]))
    assert sums == [8.0, 9.0, 1.0, 1.0], sums
    rel, cosine = wd.rel_and_cosine(sums)
    assert close(rel, math.sqrt(8.0))          # ||d|| / ||r|| = sqrt(8) / 1
    assert close(cosine, 1.0 / 3.0)            # 1 / (3 * 1)

    # Equal tensors: no difference, cosine exactly 1.
    rel, cosine = wd.rel_and_cosine(wd.pair_sums(t([[3.0, 4.0]]), t([[3.0, 4.0]])))
    assert rel == 0.0 and cosine == 1.0

    # Opposite tensors: rel 2, cosine -1.
    rel, cosine = wd.rel_and_cosine(wd.pair_sums(t([3.0, 4.0]), t([-3.0, -4.0])))
    assert close(rel, 2.0) and close(cosine, -1.0)

    # A zero reference has no relative difference and no cosine.
    assert wd.rel_and_cosine(wd.pair_sums(t([1.0, 1.0]), t([0.0, 0.0]))) == (None, None)
    # A zero member against a real reference: rel 1, no cosine.
    rel, cosine = wd.rel_and_cosine(wd.pair_sums(t([0.0, 0.0]), t([3.0, 4.0])))
    assert close(rel, 1.0) and cosine is None


def test_bf16_cast_before_subtraction():
    # Two weights that bf16 stores exactly: m = 2^-8 and r = 2^-10 * (1 + 2^-7).
    # Exact difference: 2^-9 * (1 + 127/256). That needs 9 significant bits; bf16 has 8, and the
    # tie rounds to even, so a bf16 subtraction gives 2^-9 * 1.5.
    m_value = 2.0 ** -8
    r_value = 2.0 ** -10 * (1 + 2.0 ** -7)
    m = t([m_value], torch.bfloat16)
    r = t([r_value], torch.bfloat16)
    assert m.double().item() == m_value and r.double().item() == r_value   # stored exactly
    exact = 2.0 ** -9 * (1 + 127.0 / 256.0)
    assert exact == m_value - r_value == 0.00292205810546875

    in_bf16 = (m - r).double().item()
    assert in_bf16 == 2.0 ** -9 * 1.5 == 0.0029296875
    assert in_bf16 != exact

    # Cast to float32 first: the difference is kept exactly.
    assert (m.float() - r.float()).double().item() == exact

    # The job's own function (float64) keeps it too.
    sums = wd.pair_sums(m, r)
    assert sums[0] == exact * exact
    rel, cosine = wd.rel_and_cosine(sums)
    assert close(rel, exact / r_value, 1e-15)
    # A subtraction in bf16 would report a different relative difference.
    assert abs(in_bf16 / r_value - rel) > 0.007

    # A one-step change of a bf16 weight near 1 (2^-7) is exact in both, so the measure sees it.
    a = t([1.0 + 2.0 ** -7, 0.5], torch.bfloat16)
    b = t([1.0, 0.5], torch.bfloat16)
    sums = wd.pair_sums(a, b)
    assert sums[0] == 2.0 ** -14 and sums[2] == 1.25
    assert close(wd.rel_and_cosine(sums)[0], 2.0 ** -7 / math.sqrt(1.25), 1e-15)


# ---------- names ----------

def test_normalise_prefixes():
    want = (3, "layers.3.self_attn.q_proj.weight", "attn")
    for name in ["model.layers.3.self_attn.q_proj.weight",
                 "layers.3.self_attn.q_proj.weight",
                 "language_model.model.layers.3.self_attn.q_proj.weight",
                 "model.language_model.layers.3.self_attn.q_proj.weight",
                 "language_model.layers.3.self_attn.q_proj.weight"]:
        assert wd.normalise(name) == want, (name, wd.normalise(name))

    # Layer 1 and layer 12 must not be confused.
    assert wd.normalise("model.layers.12.mlp.down_proj.weight") == (12, "layers.12.mlp.down_proj.weight", "mlp")
    assert wd.normalise("model.layers.1.mlp.down_proj.weight")[0] == 1

    # A vision tower has "layers.<N>." names too. They are not decoder layers.
    layer, key, kind = wd.normalise("model.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.weight")
    assert layer is None and kind == "other"
    assert key == "vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.weight"
    assert wd.normalise("vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight")[0] is None

    # Tensors outside the layers, with and without wrappers.
    for name in ["model.embed_tokens.weight", "embed_tokens.weight",
                 "language_model.model.embed_tokens.weight", "model.language_model.embed_tokens.weight"]:
        assert wd.normalise(name) == (None, "embed_tokens.weight", "embed"), name
    for name in ["lm_head.weight", "language_model.lm_head.weight"]:
        assert wd.normalise(name) == (None, "lm_head.weight", "lm_head"), name
    assert wd.normalise("model.norm.weight") == (None, "norm.weight", "norm")
    assert wd.normalise("model.language_model.norm.weight") == (None, "norm.weight", "norm")
    assert wd.normalise("model.embedding_norm.weight") == (None, "embedding_norm.weight", "norm")
    assert wd.normalise("model.multi_modal_projector.mm_input_projection_weight")[2] == "other"


def test_module_kinds():
    cases = {
        "self_attn.q_proj.weight": "attn", "self_attn.o_proj.weight": "attn", "self_attn.out_proj.weight": "attn",
        "self_attn.q_proj.bias": "attn",
        "mlp.gate_proj.weight": "mlp", "mlp.down_proj.weight": "mlp",
        "feed_forward.w1.weight": "mlp", "feed_forward.w3.weight": "mlp",
        # LFM2 conv mixer: its own kind, not hidden in a catch-all
        "conv.conv.weight": "conv", "conv.in_proj.weight": "conv", "conv.out_proj.weight": "conv",
        # norm weights, also the ones inside the attention module
        "input_layernorm.weight": "norm", "post_attention_layernorm.weight": "norm",
        "pre_feedforward_layernorm.weight": "norm", "post_feedforward_layernorm.weight": "norm",
        "operator_norm.weight": "norm", "ffn_norm.weight": "norm",
        "self_attn.q_norm.weight": "norm", "self_attn.k_layernorm.weight": "norm",
        # an unknown module keeps its own name
        "gate.weight": "gate",
    }
    for suffix in cases:
        assert wd.layer_kind(suffix) == cases[suffix], (suffix, wd.layer_kind(suffix))
        assert wd.normalise("model.layers.5." + suffix)[2] == cases[suffix]


def test_name_collision_is_an_error():
    table = {"model.layers.0.mlp.up_proj.weight": {"shape": [2, 2], "dtype": "F32"},
             "layers.0.mlp.up_proj.weight": {"shape": [2, 2], "dtype": "F32"}}
    try:
        wd.index_names(table)
    except wd.PairError as e:
        assert "same name" in str(e)
    else:
        raise AssertionError("a collision must raise")


# ---------- the 90% rule ----------

def layer_table(prefix, count, shape=None):
    table = {}
    for i in range(count):
        table[prefix + "layers." + str(i) + ".mlp.up_proj.weight"] = {"shape": shape or [2, 2], "dtype": "F32"}
    return table


def run_rule(member_table, reference_table):
    member_index = wd.index_names(member_table)
    reference_index = wd.index_names(reference_table)
    matched, unmatched_member, unmatched_reference, shape_mismatch = wd.match_names(member_index, reference_index)
    return wd.check_layer_match(member_index, matched, unmatched_member, shape_mismatch)


def test_ninety_percent_rule():
    member = layer_table("model.", 10)
    # Tensors outside the layers do not count for the rule.
    member["model.embed_tokens.weight"] = {"shape": [5, 2], "dtype": "F32"}

    # 9 of 10 layer tensors match: exactly 90%, allowed.
    good, total, fraction = run_rule(member, layer_table("language_model.model.", 9))
    assert (good, total) == (9, 10) and close(fraction, 0.9)

    # 8 of 10: an error that says how many matched.
    try:
        run_rule(member, layer_table("language_model.model.", 8))
    except wd.PairError as e:
        text = str(e)
        assert "only 8 of the member's 10 layer tensors" in text and "80.0%" in text, text
        assert "model.layers.8.mlp.up_proj.weight" in text, text
    else:
        raise AssertionError("80% must raise")

    # A shape mismatch is not a match: 10 names agree, but 2 shapes differ.
    reference = layer_table("", 10)
    reference["layers.0.mlp.up_proj.weight"] = {"shape": [3, 2], "dtype": "F32"}
    reference["layers.1.mlp.up_proj.weight"] = {"shape": [3, 2], "dtype": "F32"}
    try:
        run_rule(member, reference)
    except wd.PairError as e:
        assert "only 8 of" in str(e) and "[2, 2] against [3, 2]" in str(e), str(e)
    else:
        raise AssertionError("shape mismatches must count against the rule")

    # A checkpoint with no decoder layer names at all.
    try:
        run_rule({"transformer.h.0.attn.weight": {"shape": [2], "dtype": "F32"}}, layer_table("", 3))
    except wd.PairError as e:
        assert "no tensor named like a decoder layer" in str(e)
    else:
        raise AssertionError("no layers must raise")


# ---------- whole pair on synthetic files ----------

def write_pair(folder, dtype=torch.float32):
    # Member: a text-only checkpoint, names "model.layers.N.", one file.
    member = {
        "model.embed_tokens.weight": t([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "model.norm.weight": t([1.0, 1.0]),
        "lm_head.weight": t([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "model.layers.0.self_attn.q_proj.weight": t([[1.0, 0.0], [0.0, 2.0]]),
        "model.layers.0.self_attn.v_proj.weight": t([[2.0, 0.0], [0.0, 0.0]]),
        "model.layers.0.self_attn.q_norm.weight": t([1.0, 1.0]),
        "model.layers.0.mlp.up_proj.weight": t([[0.0, 3.0], [4.0, 0.0]]),
        "model.layers.0.input_layernorm.weight": t([1.0, 2.0]),
        "model.layers.1.self_attn.q_proj.weight": t([[1.0, 1.0], [1.0, 1.0]]),
        "model.layers.1.self_attn.v_proj.weight": t([[1.0, 0.0], [0.0, 1.0]]),
        "model.layers.1.self_attn.q_norm.weight": t([2.0, 2.0]),
        "model.layers.1.mlp.up_proj.weight": t([[1.0, 0.0], [0.0, 1.0]]),
        "model.layers.1.input_layernorm.weight": t([1.0, 1.0]),
        "model.layers.1.conv.conv.weight": torch.ones(2, 1, 3),
        # only the member has this one
        "model.layers.1.adapter.weight": t([1.0, 1.0]),
    }
    # Reference: a multimodal checkpoint, decoder under "model.language_model.", two shards.
    conv = torch.ones(2, 1, 3)
    conv[1, 0, 2] = 0.0
    reference_a = {
        # one more row than the member (new tokens)
        "model.language_model.embed_tokens.weight": t([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0]]),
        "model.language_model.norm.weight": t([1.0, 1.0]),
        # different width: no shared rows to compare
        "lm_head.weight": t([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        "model.language_model.layers.0.self_attn.q_proj.weight": t([[1.0, 0.0], [0.0, 1.0]]),
        "model.language_model.layers.0.self_attn.v_proj.weight": t([[2.0, 0.0], [0.0, 0.0]]),
        "model.language_model.layers.0.self_attn.q_norm.weight": t([1.0, 1.0]),
        "model.language_model.layers.0.mlp.up_proj.weight": t([[0.0, 3.0], [0.0, 0.0]]),
        "model.language_model.layers.0.input_layernorm.weight": t([1.0, 1.0]),
    }
    reference_b = {
        "model.language_model.layers.1.self_attn.q_proj.weight": t([[1.0, 1.0], [1.0, 1.0]]),
        "model.language_model.layers.1.self_attn.v_proj.weight": t([[0.0, 1.0], [1.0, 0.0]]),
        "model.language_model.layers.1.self_attn.q_norm.weight": t([2.0, 2.0]),
        "model.language_model.layers.1.mlp.up_proj.weight": t([[1.0, 0.0], [0.0, 1.0]]),
        "model.language_model.layers.1.input_layernorm.weight": t([1.0, 1.0]),
        "model.language_model.layers.1.conv.conv.weight": conv,
        # only the reference has these two
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight": t([[9.0, 9.0], [9.0, 9.0]]),
        "model.multi_modal_projector.mm_input_projection_weight": t([[1.0, 2.0], [3.0, 4.0]]),
    }
    member_dir = os.path.join(folder, "member")
    reference_dir = os.path.join(folder, "reference")
    os.makedirs(member_dir)
    os.makedirs(reference_dir)
    save_file({k: v.to(dtype) for k, v in member.items()}, os.path.join(member_dir, "model.safetensors"))
    save_file({k: v.to(dtype) for k, v in reference_a.items()},
              os.path.join(reference_dir, "model-00001-of-00002.safetensors"))
    save_file({k: v.to(dtype) for k, v in reference_b.items()},
              os.path.join(reference_dir, "model-00002-of-00002.safetensors"))
    weight_map = {}
    for name in reference_a:
        weight_map[name] = "model-00001-of-00002.safetensors"
    for name in reference_b:
        weight_map[name] = "model-00002-of-00002.safetensors"
    with open(os.path.join(reference_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    # A stray safetensors file that the index does not list must be ignored.
    save_file({"model.language_model.norm.weight": t([7.0, 7.0])}, os.path.join(reference_dir, "extra.safetensors"))
    return member_dir, reference_dir


def check_synthetic_report(report):
    assert report["layers"] == 2
    assert report["kinds"] == ["attn", "conv", "mlp", "norm"], report["kinds"]
    assert report["dtype"] == "float64"
    counts = report["counts"]
    assert counts["member_tensors"] == 15 and counts["reference_tensors"] == 16
    assert counts["matched"] == 12 and counts["unmatched_member"] == 1
    assert counts["unmatched_reference"] == 2 and counts["shape_mismatch"] == 2
    assert counts["member_layer_tensors"] == 12 and counts["member_layer_tensors_matched"] == 11
    assert close(counts["layer_match_fraction"], 11.0 / 12.0)

    # Layer 0, attn: q_proj (diff^2 1, m^2 5, r^2 2, m.r 3) + v_proj (0, 4, 4, 4)
    #   -> rel sqrt(1/6), cosine 7 / sqrt(9 * 6). The q_norm weight is not in this group.
    layer0 = report["per_layer"][0]
    assert layer0["layer"] == 0
    assert close(layer0["attn"]["rel"], math.sqrt(1.0 / 6.0))
    assert close(layer0["attn"]["cosine"], 7.0 / math.sqrt(54.0))
    assert layer0["attn"]["tensors"] == 2 and layer0["attn"]["params"] == 8
    # Layer 0, mlp: up_proj (16, 25, 9, 9) -> rel 4/3, cosine 9/15.
    assert close(layer0["mlp"]["rel"], 4.0 / 3.0) and close(layer0["mlp"]["cosine"], 0.6)
    # Layer 0, norm: q_norm (0, 2, 2, 2) + input_layernorm (1, 5, 2, 3) -> rel sqrt(1/4), cosine 5/sqrt(28).
    assert close(layer0["norm"]["rel"], 0.5) and close(layer0["norm"]["cosine"], 5.0 / math.sqrt(28.0))
    assert layer0["norm"]["tensors"] == 2
    assert "conv" not in layer0

    # Layer 1, attn: q_proj equal (0, 4, 4, 4) + v_proj (4, 2, 2, 0) -> rel sqrt(4/6), cosine 4/6.
    layer1 = report["per_layer"][1]
    assert close(layer1["attn"]["rel"], math.sqrt(4.0 / 6.0)) and close(layer1["attn"]["cosine"], 4.0 / 6.0)
    assert layer1["mlp"]["rel"] == 0.0 and layer1["mlp"]["cosine"] == 1.0
    assert layer1["norm"]["rel"] == 0.0
    # Layer 1, conv: one of six ones is zero in the reference: (1, 6, 5, 5).
    assert close(layer1["conv"]["rel"], 1.0 / math.sqrt(5.0))
    assert close(layer1["conv"]["cosine"], 5.0 / math.sqrt(30.0))
    # The member-only adapter is in no group.
    assert "adapter" not in layer1

    # Outside the layers: only the final norm matched. embed and lm_head differ in shape.
    outside = report["outside_layers"]
    assert outside["embed"] is None and outside["lm_head"] is None and outside["other"] is None
    assert outside["norm"]["rel"] == 0.0 and outside["norm"]["cosine"] == 1.0

    # Summary over both layers, attn: diff^2 1 + 4, r^2 6 + 6.
    assert close(report["summary"]["layers"]["attn"]["rel"], math.sqrt(5.0 / 12.0))
    assert report["summary"]["all"]["tensors"] == 12

    # Per-tensor table: one row per matched tensor, with both names when they differ.
    table = {row["name"]: row for row in report["tensors"]}
    assert len(table) == 12
    row = table["model.layers.0.self_attn.q_proj.weight"]
    assert row["reference_name"] == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert row["shape"] == [2, 2] and row["layer"] == 0 and row["kind"] == "attn"
    assert close(row["rel"], 1.0 / math.sqrt(2.0)) and close(row["cosine"], 3.0 / math.sqrt(10.0))
    assert close(row["diff_norm"], 1.0) and close(row["reference_norm"], math.sqrt(2.0))
    assert table["model.layers.0.self_attn.q_norm.weight"]["kind"] == "norm"
    assert table["model.norm.weight"]["layer"] is None

    # Nothing is dropped without a record.
    assert [row["name"] for row in report["unmatched_member"]] == ["model.layers.1.adapter.weight"]
    names = sorted(row["name"] for row in report["unmatched_reference"])
    assert names == ["model.multi_modal_projector.mm_input_projection_weight",
                     "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"], names
    mismatch = {row["normalised"]: row for row in report["shape_mismatch"]}
    assert sorted(mismatch) == ["embed_tokens.weight", "lm_head.weight"]
    embed = mismatch["embed_tokens.weight"]
    assert embed["member_shape"] == [3, 2] and embed["reference_shape"] == [4, 2]
    # The 3 shared rows: difference only in row 2 (1 element): (1, 4, 3, 3).
    assert embed["shared_rows"]["rows"] == 3
    assert close(embed["shared_rows"]["rel"], math.sqrt(1.0 / 3.0))
    assert close(embed["shared_rows"]["cosine"], 3.0 / math.sqrt(12.0))
    # Different width: nothing to compare.
    assert "shared_rows" not in mismatch["lm_head.weight"]


def test_compare_folders():
    with tempfile.TemporaryDirectory() as folder:
        member_dir, reference_dir = write_pair(folder)
        report = wd.compare_folders(member_dir, reference_dir)
        check_synthetic_report(report)
        assert report["source_dtypes"]["member"] == {"F32": 15}
        # The report must survive a JSON round trip (it goes to a file).
        check_synthetic_report(json.loads(json.dumps(report)))

        # Reading in small slices gives the same numbers as reading whole tensors.
        old = wd.CHUNK_ELEMS
        wd.CHUNK_ELEMS = 2
        try:
            check_synthetic_report(wd.compare_folders(member_dir, reference_dir))
        finally:
            wd.CHUNK_ELEMS = old

    # The same values stored as bf16 (all of them are exact in bf16) give the same report.
    with tempfile.TemporaryDirectory() as folder:
        member_dir, reference_dir = write_pair(folder, torch.bfloat16)
        report = wd.compare_folders(member_dir, reference_dir)
        check_synthetic_report(report)
        assert report["source_dtypes"]["member"] == {"BF16": 15}


def test_bf16_files_keep_small_difference():
    # The pair of test_bf16_cast_before_subtraction, read back from bf16 files.
    m_value = 2.0 ** -8
    r_value = 2.0 ** -10 * (1 + 2.0 ** -7)
    with tempfile.TemporaryDirectory() as folder:
        for name, value in [("member", m_value), ("reference", r_value)]:
            os.makedirs(os.path.join(folder, name))
            save_file({"model.layers.0.mlp.up_proj.weight": t([[value, 1.0]], torch.bfloat16)},
                      os.path.join(folder, name, "model.safetensors"))
        report = wd.compare_folders(os.path.join(folder, "member"), os.path.join(folder, "reference"))
    exact = 0.00292205810546875
    row = report["tensors"][0]
    assert close(row["diff_norm"], exact, 1e-18)
    assert close(row["rel"], exact / math.sqrt(r_value * r_value + 1.0), 1e-15)
    assert close(report["per_layer"][0]["mlp"]["rel"], row["rel"], 0.0)


def test_low_match_pair_fails_on_files():
    with tempfile.TemporaryDirectory() as folder:
        os.makedirs(os.path.join(folder, "member"))
        os.makedirs(os.path.join(folder, "reference"))
        member = {}
        reference = {}
        for i in range(4):
            member["model.layers." + str(i) + ".mlp.up_proj.weight"] = torch.ones(2, 2)
            # a different width in every layer: no tensor matches
            reference["model.layers." + str(i) + ".mlp.up_proj.weight"] = torch.ones(2, 3)
        save_file(member, os.path.join(folder, "member", "model.safetensors"))
        save_file(reference, os.path.join(folder, "reference", "model.safetensors"))
        try:
            wd.compare_folders(os.path.join(folder, "member"), os.path.join(folder, "reference"))
        except wd.PairError as e:
            assert "only 0 of the member's 4 layer tensors" in str(e), str(e)
        else:
            raise AssertionError("a pair with no matching layer tensor must raise")


def test_duplicate_tensor_in_two_files_fails():
    with tempfile.TemporaryDirectory() as folder:
        save_file({"model.norm.weight": t([1.0])}, os.path.join(folder, "a.safetensors"))
        save_file({"model.norm.weight": t([1.0])}, os.path.join(folder, "b.safetensors"))
        try:
            wd.Checkpoint(folder)
        except wd.PairError as e:
            assert "stored in two files" in str(e)
        else:
            raise AssertionError("a tensor in two files must raise")


def test_reader_agrees_with_safetensors():
    # The job reads byte ranges by itself. Check every dtype and slice against safetensors' loader.
    from safetensors.torch import load_file

    torch.manual_seed(0)
    tensors = {
        "bf16": torch.randn(7, 5).to(torch.bfloat16),
        "f16": torch.randn(4, 3).to(torch.float16),
        "f32": torch.randn(6, 2, 3),
        "f64": torch.randn(3, 3, dtype=torch.float64),
        "i64": torch.arange(10, dtype=torch.int64).reshape(5, 2),
        "flag": t([True, False, True], torch.bool),
        "scalar": torch.tensor(2.5),
        "vector": torch.randn(9),
        "empty": torch.zeros(0, 4),
    }
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "model.safetensors")
        save_file(tensors, path, metadata={"format": "pt"})
        truth = load_file(path)
        ckpt = wd.Checkpoint(folder)
        assert sorted(ckpt.tensors) == sorted(tensors)
        for name in tensors:
            got = ckpt.read(name)
            assert got.dtype == truth[name].dtype and list(got.shape) == list(truth[name].shape), name
            assert torch.equal(got, truth[name]), name
            assert ckpt.tensors[name]["shape"] == list(truth[name].shape)
        for name, first, last in [("bf16", 0, 3), ("bf16", 3, 7), ("bf16", 6, 7), ("f32", 2, 5),
                                  ("vector", 4, 9), ("i64", 1, 2), ("f16", 0, 4)]:
            got = ckpt.read(name, first, last)
            assert torch.equal(got, truth[name][first:last]), (name, first, last)
        assert ckpt.dtype_counts() == {"BF16": 1, "F16": 1, "F32": 4, "F64": 1, "I64": 1, "BOOL": 1}
        assert ckpt.bytes == os.path.getsize(path)
        ckpt.close()

        # A file cut short must fail loudly, not give a tensor of zeros.
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            data = f.read()
        cut = os.path.join(folder, "cut")
        os.makedirs(cut)
        with open(os.path.join(cut, "model.safetensors"), "wb") as f:
            f.write(data[:size - 8])
        ckpt = wd.Checkpoint(cut)
        last_name = max(ckpt.tensors, key=lambda name: ckpt.tensors[name]["offset"])
        try:
            ckpt.read(last_name)
        except wd.PairError as e:
            assert "short read" in str(e)
        else:
            raise AssertionError("a short file must raise")
        ckpt.close()


# ---------- families, paths, table ----------

def test_families_file():
    families = wd.load_families()
    names = [fam["name"] for fam in families]
    assert names == ["qwen7b", "llama8b", "gemma27", "qwen32b", "lfm2", "qwen3_8b"], names
    by_name = {fam["name"]: fam for fam in families}
    assert by_name["llama8b"]["host"] == "unsloth/Meta-Llama-3.1-8B-Instruct"
    with_ancestor = sorted(row["id"] for row in by_name["llama8b"]["members"] if row.get("ancestor"))
    assert with_ancestor == ["NousResearch/Hermes-3-Llama-3.1-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                             "nvidia/OpenMath2-Llama3.1-8B"]
    for row in by_name["llama8b"]["members"]:
        assert row.get("ancestor") in [None, "unsloth/Meta-Llama-3.1-8B"]
    sizes = {fam["name"]: len(fam["members"]) for fam in families}
    assert sizes == {"qwen7b": 2, "llama8b": 7, "gemma27": 1, "qwen32b": 6, "lfm2": 3, "qwen3_8b": 1}

    # 20 members against their host, 3 more against the ancestor.
    pairs = wd.select_pairs(families, everything=True)
    assert len(pairs) == 23
    assert len([p for p in pairs if p["relation"] == "ancestor"]) == 3
    assert len(set((p["member"], p["reference"]) for p in pairs)) == 23

    lfm2 = wd.select_pairs(families, family="lfm2")
    assert [p["member"] for p in lfm2] == ["LiquidAI/LFM2-1.2B-Tool", "LiquidAI/LFM2-1.2B-RAG",
                                           "LiquidAI/LFM2-1.2B-Extract"]
    assert all(p["reference"] == "LiquidAI/LFM2-1.2B" and p["relation"] == "host" for p in lfm2)

    one = wd.select_pairs(families, member="nvidia/OpenMath2-Llama3.1-8B", reference="unsloth/Meta-Llama-3.1-8B")
    assert len(one) == 1 and one[0]["relation"] == "ancestor" and one[0]["family"] == "llama8b"
    assert one[0]["host"] == "unsloth/Meta-Llama-3.1-8B-Instruct"

    # Every host and member is a model of atlas/zoo.json.
    with open(os.path.join(wd.REPO, "atlas", "zoo.json")) as f:
        zoo = set(row["id"] for row in json.load(f)["models"])
    for fam in families:
        assert fam["host"] in zoo, fam["host"]
        for row in fam["members"]:
            assert row["id"] in zoo, row["id"]

    for bad in [{"family": "nope"}, {}, {"member": "a/b"}]:
        try:
            wd.select_pairs(families, **bad)
        except SystemExit:
            pass
        else:
            raise AssertionError("select_pairs must refuse " + str(bad))


def test_paths_and_atomic_write():
    path = wd.pair_path("LiquidAI/LFM2-1.2B-Tool", "LiquidAI/LFM2-1.2B", out_dir="/x")
    assert path == "/x/LiquidAI__LFM2-1.2B-Tool__vs__LiquidAI__LFM2-1.2B.json", path
    assert wd.pair_path("a/b", "c/d").startswith(os.path.join(wd.REPO, "atlas", "lineage", "weight_delta"))
    with tempfile.TemporaryDirectory() as folder:
        target = os.path.join(folder, "sub", "out.json")
        wd.write_json_atomic(target, {"a": 1})
        wd.write_json_atomic(target, {"a": 2})
        with open(target) as f:
            assert json.load(f) == {"a": 2}
        assert os.listdir(os.path.dirname(target)) == ["out.json"]


def test_format_table():
    with tempfile.TemporaryDirectory() as folder:
        member_dir, reference_dir = write_pair(folder)
        report = wd.compare_folders(member_dir, reference_dir)
    report.update({"member": "org/member", "reference": "org/host", "relation": "host"})
    text = wd.format_table(report)
    lines = text.split("\n")
    assert "2 layers, 12 tensors compared, 1 only in member, 2 only in reference, 2 shape mismatches" in lines[0]
    # attn: layer 1 (0.816) is larger than layer 0 (0.408), so it comes first.
    start = lines.index("  largest attn rel (2 layers have attn):")
    assert "layer   1" in lines[start + 1] and "0.816497" in lines[start + 1]
    assert "layer   0" in lines[start + 2] and "0.408248" in lines[start + 2]
    start = lines.index("  largest mlp rel (2 layers have mlp):")
    assert "layer   0" in lines[start + 1] and "1.333333" in lines[start + 1]
    assert "  largest conv rel (1 layers have conv):" in lines
    assert "largest norm rel" not in text
    assert lines[-1].startswith("  outside the layers: norm rel 0.000000")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("pass", name)
    print("ok")
