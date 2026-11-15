import json
import os

import pytest
import torch
from safetensors.torch import save_file

from wsp import lineage

HIDDEN = 16
LAYERS = 4
VOCAB = 64
MLP = 32


def base_weights(seed=0, hidden=HIDDEN, vocab=VOCAB, mlp=MLP, experts=0):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen)

    w = {"model.embed_tokens.weight": rnd(vocab, hidden), "model.norm.weight": torch.ones(hidden)}
    for i in range(LAYERS):
        p = "model.layers.%d." % i
        w[p + "self_attn.q_proj.weight"] = rnd(hidden, hidden)
        w[p + "self_attn.k_proj.weight"] = rnd(hidden, hidden)
        w[p + "input_layernorm.weight"] = torch.ones(hidden)
        if experts:
            for e in range(experts):
                w[p + "mlp.experts.%d.down_proj.weight" % e] = rnd(hidden, mlp)
                w[p + "mlp.experts.%d.up_proj.weight" % e] = rnd(mlp, hidden)
            w[p + "mlp.gate.weight"] = rnd(experts, hidden)
        else:
            w[p + "mlp.down_proj.weight"] = rnd(hidden, mlp)
            w[p + "mlp.up_proj.weight"] = rnd(mlp, hidden)
    return w


def noisy(weights, rel, seed=1, only=None):
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, t in weights.items():
        if t.dim() < 2 or (only is not None and only not in name):
            out[name] = t.clone()
            continue
        out[name] = t + rel * t.std() * torch.randn(t.shape, generator=gen)
    return out


def write_model(path, weights, rope=10000.0, hidden=HIDDEN, vocab=VOCAB, mlp=MLP, tok="a", experts=0,
                model_type="qwen2", card=None, template="T1", dtype=None, tok_vocab=None):
    os.makedirs(path, exist_ok=True)
    if dtype is not None:
        weights = {k: v.to(dtype) for k, v in weights.items()}
    save_file({k: v.contiguous() for k, v in weights.items()}, os.path.join(path, "model.safetensors"))
    cfg = {
        "model_type": model_type, "architectures": ["X"], "num_hidden_layers": LAYERS,
        "hidden_size": hidden, "num_attention_heads": 2, "num_key_value_heads": 2,
        "intermediate_size": mlp, "vocab_size": vocab, "rope_theta": rope,
        "max_position_embeddings": 128, "tie_word_embeddings": True,
    }
    if experts:
        cfg["num_experts"] = experts
        cfg["num_experts_per_tok"] = 2
        cfg["moe_intermediate_size"] = mlp
        cfg["model_type"] = model_type + "_moe"
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    tokenizer = {"model": {"vocab": {"tok%s%d" % (tok, i): i for i in range(tok_vocab or vocab)}, "merges": []},
                 "added_tokens": [{"id": vocab, "content": "<eos>", "special": True}]}
    with open(os.path.join(path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"chat_template": template, "eos_token": "<eos>"}, f)
    if card:
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write("---\nlicense: apache-2.0\nbase_model: %s\n---\ntext\n" % card)
    return str(path)


@pytest.fixture
def world(tmp_path):
    base_w = base_weights()
    paths = {}
    paths["base"] = write_model(tmp_path / "base", base_w)
    paths["copy"] = write_model(tmp_path / "copy", base_w)
    paths["bf16copy"] = write_model(tmp_path / "bf16copy", base_w, dtype=torch.bfloat16)
    paths["ft"] = write_model(tmp_path / "ft", noisy(base_w, 0.03))
    paths["tiny"] = write_model(tmp_path / "tiny", noisy(base_w, 0.02, only="q_proj"))
    paths["far"] = write_model(tmp_path / "far", noisy({k: 0.5 * v for k, v in base_w.items()}, 1.0))
    paths["theta"] = write_model(tmp_path / "theta", noisy(base_w, 0.03), rope=1000000.0)
    paths["wide"] = write_model(tmp_path / "wide", base_weights(seed=3, hidden=24, mlp=48), hidden=24, mlp=48)
    paths["moe"] = write_model(tmp_path / "moe", base_weights(seed=4, experts=4), experts=4)
    paths["foreign"] = write_model(tmp_path / "foreign", base_w, tok="b")
    padded = dict(base_w)
    padded["model.embed_tokens.weight"] = torch.cat([base_w["model.embed_tokens.weight"], torch.zeros(8, HIDDEN)])
    paths["bigvocab"] = write_model(tmp_path / "bigvocab", padded, vocab=VOCAB + 8, tok_vocab=VOCAB)
    wrapped = {}
    for k, v in noisy(base_w, 0.03).items():
        wrapped[k.replace("model.", "model.language_model.", 1)] = v
    paths["wrapped"] = write_model(tmp_path / "wrapped", wrapped)
    # ancestor test: base = A + delta, source = A + other delta
    anc = base_weights(seed=7)
    paths["anc"] = write_model(tmp_path / "anc", anc)
    paths["anc_base"] = write_model(tmp_path / "anc_base", noisy(anc, 0.05, seed=11), card="org/anc")
    paths["anc_src"] = write_model(tmp_path / "anc_src", noisy(anc, 0.05, seed=12), card="org/anc")
    paths["cache"] = str(tmp_path / "cache")
    return paths


def opener_for(world):
    mapping = {"org/anc": world["anc"]}

    def opener(model_id):
        return lineage.LocalSource(mapping.get(model_id, model_id), model_id=model_id)

    return opener


def rel(world, source, base="base", **kw):
    return lineage.relation(world[base], world[source], cache_dir=world["cache"], opener=opener_for(world), **kw)


def test_facts_fields(world):
    f = lineage.facts(world["moe"], cache_dir=None)
    cfg = f["config"]
    assert cfg["experts"] == 4 and cfg["top_k"] == 2 and cfg["expert_size"] == MLP
    assert cfg["hidden"] == HIDDEN and cfg["rope_theta"] == 10000.0
    assert f["tokenizer"]["core_hash"] and f["tokenizer"]["chat_template_hash"]
    assert f["tokenizer"]["special_tokens"]["eos_token"] == "<eos>"
    name = "model.layers.0.self_attn.q_proj.weight"
    shard, dtype, shape, begin, end = f["tensors"][name]
    assert shard == "model.safetensors" and dtype == "F32" and shape == [HIDDEN, HIDDEN] and end > begin
    assert f["params"] > 0


def test_card_and_license(world):
    f = lineage.facts(world["anc_base"], cache_dir=None)
    assert f["base_model"] == ["org/anc"]
    assert f["license"] == "apache-2.0"


def test_cache_round_trip(world):
    first = lineage.facts(world["base"], cache_dir=world["cache"])
    assert first["from_cache"] is False
    path = lineage.cache_path(world["base"], world["cache"])
    assert os.path.exists(path)
    second = lineage.facts(world["base"], cache_dir=world["cache"])
    assert second["from_cache"] is True
    assert second["tensors"] == first["tensors"] and second["revision"] == first["revision"]
    # A changed file changes the revision and rebuilds.
    with open(os.path.join(world["base"], "config.json")) as f:
        cfg = json.load(f)
    cfg["rope_theta"] = 5.0
    with open(os.path.join(world["base"], "config.json"), "w") as f:
        json.dump(cfg, f)
    third = lineage.facts(world["base"], cache_dir=world["cache"])
    assert third["from_cache"] is False and third["config"]["rope_theta"] == 5.0


def test_cache_path_name():
    assert lineage.cache_path("Qwen/Qwen2.5-7B", "/x").endswith("Qwen__Qwen2.5-7B.json")


def test_hub_source_through_fake_network(world):
    """HubSource code path with the three network methods replaced by local reads."""

    class FakeHub(lineage.HubSource):
        def __init__(self, model_id, root):
            super().__init__(model_id)
            self.root = root
            self.local = lineage.LocalSource(root)

        def revision(self):
            return "abc123"

        def files(self):
            return self.local.files()

        def _download(self, name):
            return self.local.read_file(name)

        def _make_reader(self, name):
            return self.local.range_reader(name)

        def card(self):
            return {"base_model": ["org/x"], "base_model_relation": "finetune", "license": "mit"}

    src = FakeHub("org/model", world["base"])
    f = lineage.facts("org/model", cache_dir=world["cache"], opener=lambda mid: src)
    assert f["revision"] == "abc123" and f["source_kind"] == "hub"
    assert f["base_model"] == ["org/x"] and f["license"] == "mit"
    again = lineage.facts("org/model", cache_dir=world["cache"], opener=lambda mid: src)
    assert again["from_cache"] is True
    assert os.path.exists(os.path.join(world["cache"], "org__model.json"))


def test_card_tags():
    card = lineage._card_from_meta({}, ["license:mit", "base_model:finetune:org/a", "base_model:quantized:org/b"])
    assert card["base_model"] == ["org/a", "org/b"]
    assert card["base_model_relation"] == "finetune" and card["license"] == "mit"


def test_canonical_names():
    a = lineage.canonical("model.layers.3.mlp.down_proj.weight")
    b = lineage.canonical("model.language_model.layers.3.mlp.down_proj.weight")
    c = lineage.canonical("language_model.model.layers.3.mlp.down_proj.weight")
    assert a == b == c == ("layers.3.mlp.down_proj.weight", "mlp_down", 3)
    assert lineage.canonical("model.layers.0.feed_forward.w2.weight")[1] == "mlp_down"
    assert lineage.canonical("model.layers.0.self_attn.qkv_proj.weight")[1] == "attn_q"
    assert lineage.canonical("model.layers.0.mlp.experts.1.down_proj.weight")[1] is None
    assert lineage.canonical("model.embedding_norm.weight")[0] == "final_norm"
    assert lineage.canonical("model.vision_tower.layers.0.mlp.down_proj.weight") is None


def test_relation_same(world):
    r = rel(world, "copy")
    assert r["kind"] == "same"
    assert r["flags"]["tokenizer_same"] and r["flags"]["embeddings_mixable"]
    assert methods_names(r) == []


def test_relation_fine_tune(world):
    r = rel(world, "ft")
    assert r["kind"] == "fine_tune"
    assert 0.99 < r["evidence"]["mean_cosine"] < 0.9999
    assert r["evidence"]["relative_delta_norm"] > 0.01
    assert r["flags"]["embeddings_mixable"] and r["flags"]["teacher_usable_for_logit_kd"]
    assert not r["flags"]["positional_base_differs"] and not r["flags"]["vocab_differs"]
    assert methods_names(r)[0] == "weave"


def test_tiny_delta_warning(world):
    r = rel(world, "tiny")
    assert r["kind"] == "fine_tune"
    assert any("under 0.01" in w for w in r["warnings"])


def test_relation_diverged(world):
    r = rel(world, "far")
    assert r["kind"] == "diverged_same_shape"
    assert r["evidence"]["mean_cosine"] < 0.9
    names = methods_names(r)
    assert "weave" not in names and "delta" not in names and "transplant" not in names
    assert names[0] == "synth"


def test_relation_other_rope_theta(world):
    r = rel(world, "theta")
    assert r["kind"] == "diverged_same_shape"
    assert r["flags"]["positional_base_differs"] is True
    assert "weave" not in methods_names(r)


def test_relation_other_size(world):
    r = rel(world, "wide")
    assert r["kind"] == "same_family_other_size"
    assert r["flags"]["tokenizer_same"] and not r["flags"]["embeddings_mixable"]
    assert methods_names(r)[0] == "synth"


def test_relation_moe_same_width(world):
    r = rel(world, "moe")
    assert r["kind"] == "moe_same_width"
    assert r["flags"]["same_family"] is True
    assert methods_names(r)[0] == "transplant"


def test_relation_foreign_tokenizer(world):
    r = rel(world, "foreign")
    assert r["kind"] == "foreign"
    assert r["flags"]["tokenizer_same"] is False
    assert r["flags"]["teacher_usable_for_logit_kd"] is False
    assert "weave" not in methods_names(r) and "transplant" not in methods_names(r)


def test_template_flag(world, tmp_path):
    path = write_model(tmp_path / "t2", base_weights(), template="T2")
    world["t2"] = path
    r = rel(world, "t2")
    assert r["flags"]["template_same"] is False and r["flags"]["tokenizer_same"] is True


def test_vocab_difference_uses_shared_rows(world):
    r = rel(world, "bigvocab")
    assert r["flags"]["vocab_differs"] is True
    assert r["kind"] == "same"
    rows = [t for t in lineage.probe(world["base"], world["bigvocab"], cache_dir=None)["tensors"] if t["key"] == "embed"]
    assert rows and rows[0]["shared_rows"] == VOCAB


def test_prefix_rename(world):
    r = rel(world, "wrapped")
    assert r["kind"] == "fine_tune"
    keys = [t["key"] for t in r["evidence"]["probe"]["tensors"]]
    assert "embed" in keys and any("down_proj" in k for k in keys)


def test_fine_tune_of_ancestor(world):
    r = lineage.relation(world["anc_base"], world["anc_src"], cache_dir=world["cache"], opener=opener_for(world))
    assert 0.9 < r["evidence"]["mean_cosine"] < 0.999
    assert r["kind"] == "fine_tune_of_ancestor"
    assert r["subtract_reference"] == "org/anc"
    add = lineage.methods(r, "add")
    assert add[0]["method"] == "weave" and "org/anc" in add[0]["reason"]


def test_explicit_tensor_list_and_sample_limit(world):
    names = ["model.layers.1.self_attn.q_proj.weight", "model.embed_tokens.weight"]
    out = lineage.probe(world["base"], world["ft"], tensors=names, sample_bytes=256, cache_dir=None)
    assert [t["key"] for t in out["tensors"]] == ["layers.1.self_attn.q_proj.weight", "embed"]
    assert all(t["elements"] <= 64 for t in out["tensors"])


def test_fallback_when_names_unknown(tmp_path):
    gen = torch.Generator().manual_seed(0)
    weights = {"blk.0.ffn.weight": torch.randn(8, 8, generator=gen), "blk.1.ffn.weight": torch.randn(8, 8, generator=gen)}
    a = write_model(tmp_path / "a", weights)
    b = write_model(tmp_path / "b", weights)
    out = lineage.probe(a, b, cache_dir=None)
    assert out["n_matrices"] == 2 and out["mean_cosine"] == pytest.approx(1.0)


def methods_names(r):
    return [m["method"] for m in lineage.methods(r, "add")]


def test_methods_table_encodes_evidence():
    for kind in ["fine_tune", "fine_tune_of_ancestor"]:
        names = [m["method"] for m in lineage.methods(kind)]
        assert names[0] == "weave" and "delta" in names
    for kind in ["diverged_same_shape", "same_family_other_size", "foreign"]:
        names = [m["method"] for m in lineage.methods(kind)]
        assert "weave" not in names and "delta" not in names and "transplant" not in names
        assert "synth" in names and "whole_model" in names
    assert "feedback" in [m["method"] for m in lineage.methods("diverged_same_shape")]
    assert lineage.methods("same") == []
    assert lineage.methods("moe_same_width")[0]["method"] == "synth"
    moe = {"kind": "moe_same_width", "flags": {"same_family": True}}
    assert lineage.methods(moe)[0]["method"] == "transplant"
    for kind in ["fine_tune", "foreign"]:
        out = lineage.methods(kind, want="remove")
        assert [m["method"] for m in out] == ["slim"]
    for m in lineage.methods("foreign"):
        assert m["reason"] and m["cost"]
    with pytest.raises(ValueError):
        lineage.methods("nonsense")


def test_cli(world, capsys, monkeypatch):
    monkeypatch.setattr(lineage, "DEFAULT_CACHE_DIR", world["cache"])
    monkeypatch.setattr(lineage.facts, "__defaults__", (world["cache"], False, None))
    monkeypatch.setattr(lineage.relation, "__defaults__", (world["cache"], None, lineage.SAMPLE_BYTES, None))
    assert lineage.main(["relation", world["base"], world["ft"]]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["kind"] == "fine_tune" and out["methods"]["add"][0]["method"] == "weave"
    assert lineage.main(["facts", world["base"]]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["config"]["hidden"] == HIDDEN and "tensors" not in out


# ---- review fixes 2026-11-14


def test_feedback_needs_the_hosts_tokenizer():
    foreign = {"kind": "foreign", "flags": {"tokenizer_same": False}}
    assert "feedback" not in [m["method"] for m in lineage.methods(foreign)]
    other_size = {"kind": "same_family_other_size", "flags": {"tokenizer_same": False}}
    assert "feedback" not in [m["method"] for m in lineage.methods(other_size)]
    same_tok = {"kind": "foreign", "flags": {"tokenizer_same": True}}
    assert "feedback" in [m["method"] for m in lineage.methods(same_tok)]


def test_frozen_embeddings_do_not_lift_the_gate(world, tmp_path):
    base_w = base_weights()
    # Layers near cosine 0.89 with the embedding held at 1.0: the old mean over all matrices passed 0.9.
    path = write_model(tmp_path / "frozen", noisy(base_w, 0.5, only="layers."))
    world["frozen"] = path
    r = rel(world, "frozen")
    assert r["kind"] == "diverged_same_shape"
    assert r["evidence"]["probe"]["embed_cosine"] == pytest.approx(1.0)
    assert [m["method"] for m in lineage.methods(r)][0] == "synth"


def test_one_far_layer_matrix_blocks_weave(world, tmp_path):
    base_w = base_weights()
    path = write_model(tmp_path / "onefar", noisy(base_w, 2.0, only="layers.2.self_attn.q_proj"))
    world["onefar"] = path
    r = rel(world, "onefar")
    assert r["kind"] == "diverged_same_shape"
    assert "weave" not in methods_names(r)


def test_quantized_layers_are_never_same_or_fine_tune(world, tmp_path):
    base_w = base_weights()
    quant = {}
    for name, t in base_w.items():
        quant[name] = (t * 10).to(torch.int8) if "layers." in name and t.dim() == 2 else t
    world["fp8"] = write_model(tmp_path / "fp8", quant)
    r = rel(world, "fp8")
    assert r["kind"] == "diverged_same_shape" and r["flags"]["quantized"]
    assert methods_names(r) == ["whole_model"]


def test_renamed_quantized_tensors_keep_the_architecture_label(world, tmp_path):
    base_w = base_weights()
    renamed = {}
    for name, t in base_w.items():
        if name.endswith("down_proj.weight"):
            renamed[name.replace("weight", "qweight")] = t.to(torch.int8)
            renamed[name.replace("weight", "scales")] = torch.ones(1)
        else:
            renamed[name] = t
    world["awq"] = write_model(tmp_path / "awq", renamed)
    r = rel(world, "awq")
    assert r["kind"] == "diverged_same_shape" and r["flags"]["quantized"]
    assert methods_names(r) == ["whole_model"]


def test_added_tokens_block_logit_distillation(world, tmp_path):
    path = write_model(tmp_path / "think", noisy(base_weights(), 0.03))
    with open(os.path.join(path, "tokenizer.json")) as f:
        data = json.load(f)
    data["added_tokens"].append({"id": VOCAB + 1, "content": "<think>", "special": True})
    with open(os.path.join(path, "tokenizer.json"), "w") as f:
        json.dump(data, f)
    world["think"] = path
    r = rel(world, "think")
    assert r["flags"]["tokenizer_same"] is True
    assert r["flags"]["teacher_usable_for_logit_kd"] is False
    assert r["evidence"]["added_tokens_same"] is False
    assert rel(world, "ft")["flags"]["teacher_usable_for_logit_kd"] is True


def test_local_cache_keys_differ_for_equal_basenames(tmp_path):
    a = tmp_path / "x" / "m"
    b = tmp_path / "y" / "m"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    cache = str(tmp_path / "c")
    assert lineage.cache_path(str(a), cache) != lineage.cache_path(str(b), cache)


def set_merges(path, merges):
    with open(os.path.join(path, "tokenizer.json")) as f:
        data = json.load(f)
    data["model"]["merges"] = merges
    with open(os.path.join(path, "tokenizer.json"), "w") as f:
        json.dump(data, f)


def set_tokenizer_config(path, **fields):
    with open(os.path.join(path, "tokenizer_config.json")) as f:
        conf = json.load(f)
    conf.update(fields)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump(conf, f)


def test_same_vocab_other_template_pad_and_format_is_the_same_tokenizer(world, tmp_path):
    set_merges(world["base"], ["taka1 taka2", "taka3 taka4"])
    path = write_model(tmp_path / "fmt", noisy(base_weights(), 0.03))
    set_merges(path, [["taka1", "taka2"], ["taka3", "taka4"]])
    set_tokenizer_config(path, pad_token="<pad>", chat_template="OTHER")
    world["fmt"] = path
    r = rel(world, "fmt")
    assert r["flags"]["tokenizer_same"] is True
    assert r["flags"]["template_same"] is False
    assert r["flags"]["tokenizer_strict_same"] is False
    assert r["evidence"]["special_tokens_same"] is False
    assert r["kind"] == "fine_tune"
    twin = write_model(tmp_path / "twin", noisy(base_weights(), 0.03))
    set_merges(twin, ["taka1 taka2", "taka3 taka4"])
    world["twin"] = twin
    assert rel(world, "twin")["flags"]["tokenizer_strict_same"] is True


def test_other_vocab_is_still_a_different_tokenizer(world):
    r = rel(world, "foreign")
    assert r["flags"]["tokenizer_same"] is False and r["flags"]["tokenizer_strict_same"] is False


def test_reserved_ids_reused_is_a_flag_not_a_new_tokenizer(world, tmp_path):
    path = write_model(tmp_path / "hermes", noisy(base_weights(), 0.03))
    with open(os.path.join(path, "tokenizer.json")) as f:
        data = json.load(f)
    data["added_tokens"][0]["content"] = "<tool_call>"
    with open(os.path.join(path, "tokenizer.json"), "w") as f:
        json.dump(data, f)
    world["hermes"] = path
    r = rel(world, "hermes")
    assert r["flags"]["tokenizer_same"] is True
    assert r["flags"]["reserved_ids_reused"] is True
    assert r["flags"]["teacher_usable_for_logit_kd"] is False
    assert r["kind"] == "fine_tune"


def test_small_noise_fine_tune_is_not_same(world, tmp_path):
    path = write_model(tmp_path / "light", noisy(base_weights(), 0.007))
    world["light"] = path
    r = rel(world, "light")
    assert 0.003 < r["evidence"]["relative_delta_norm"] < 0.02
    assert r["kind"] == "fine_tune"
    assert any("small" in w for w in r["warnings"])
    assert methods_names(r)[0] == "weave"


def test_exact_copy_is_same_and_rounded_copy_is_not(world):
    assert rel(world, "copy")["kind"] == "same"
    rounded = rel(world, "bf16copy")
    assert rounded["kind"] == "fine_tune"
    assert any("small" in w for w in rounded["warnings"])


def test_gated_ancestor_falls_back_to_a_mirror(world, tmp_path):
    mapping = {"unsloth/Meta-Llama-X": world["anc"]}
    gated = {"meta-llama/Llama-X"}

    def opener(model_id):
        if model_id in gated:
            raise PermissionError("gated repo %s" % model_id)
        return lineage.LocalSource(mapping.get(model_id, model_id), model_id=model_id)

    base = write_model(tmp_path / "gb", noisy(base_weights(seed=7), 0.05, seed=11), card="meta-llama/Llama-X")
    src = write_model(tmp_path / "gs", noisy(base_weights(seed=7), 0.05, seed=12), card="meta-llama/Llama-X")
    r = lineage.relation(base, src, cache_dir=world["cache"], opener=opener)
    assert r["kind"] == "fine_tune_of_ancestor"
    assert r["subtract_reference"] == "unsloth/Meta-Llama-X"
    assert r["evidence"]["ancestor_search"][0]["mirror"] == "unsloth/Meta-Llama-X"


def test_unreadable_ancestor_reports_fine_tune_with_a_note(world, tmp_path):
    def opener(model_id):
        if model_id.startswith("meta-llama/") or model_id.startswith("unsloth/") or model_id.startswith("NousResearch/"):
            raise PermissionError("gated")
        return lineage.LocalSource(model_id, model_id=model_id)

    base = write_model(tmp_path / "gb", noisy(base_weights(seed=7), 0.05, seed=11), card="meta-llama/Llama-X")
    src = write_model(tmp_path / "gs", noisy(base_weights(seed=7), 0.05, seed=12), card="meta-llama/Llama-X")
    r = lineage.relation(base, src, cache_dir=world["cache"], opener=opener)
    assert r["kind"] == "fine_tune" and r["subtract_reference"] is None
    assert any("ancestor" in w for w in r["warnings"])


def test_added_token_renamed_inside_the_vocab_keeps_the_tokenizer(world, tmp_path):
    for name, content in (("base", "<|reserved_1|>"), ("renamed", "<tool_call>")):
        path = world["base"] if name == "base" else write_model(tmp_path / name, noisy(base_weights(), 0.03))
        with open(os.path.join(path, "tokenizer.json")) as f:
            data = json.load(f)
        data["model"]["vocab"].pop("toka3", None)
        data["model"]["vocab"][content] = 3
        data["added_tokens"] = [{"id": 3, "content": content, "special": True}]
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f)
        world[name] = path
    r = rel(world, "renamed")
    assert r["flags"]["tokenizer_same"] is True
    assert r["flags"]["reserved_ids_reused"] is True
    assert r["kind"] == "fine_tune"


def test_range_reader_retries_timeouts_and_rate_limits(monkeypatch):
    monkeypatch.setattr(lineage.time, "sleep", lambda seconds: None)
    calls = []

    def flaky(offset, length):
        calls.append(offset)
        if len(calls) == 1:
            raise RuntimeError("range request failed with status 429: url")
        if len(calls) == 2:
            raise RuntimeError("Read timed out.")
        return b"ok"

    assert lineage._retrying(flaky)(0, 2) == b"ok" and len(calls) == 3

    def broken(offset, length):
        raise RuntimeError("range request failed with status 404: url")

    with pytest.raises(RuntimeError):
        lineage._retrying(broken)(0, 2)


def test_tops_is_valid_for_same_tokenizer_other_size_and_diverged_sources():
    for kind in ("same_family_other_size", "diverged_same_shape"):
        names = [m["method"] for m in lineage.methods(kind)]
        assert names.index("tops") > names.index("synth")
        assert [m["module"] for m in lineage.methods(kind) if m["method"] == "tops"] == ["wsp/tops.py"]
    other = {"kind": "same_family_other_size", "flags": {"tokenizer_same": False}, "base_is_moe": False,
             "subtract_reference": None}
    assert "tops" not in [m["method"] for m in lineage.methods(other)]
    assert "tops" not in [m["method"] for m in lineage.methods("fine_tune")]
