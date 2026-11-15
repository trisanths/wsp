"""The graft sink, RMS matching, the ridge default, and the stitch sink.

The audit found that grafts fed the donor part a mapped vector at the first
position, where the donor's attention sink lives. These tests check that a
graft with a sink feeds the donor its own sink state there, on the first
real token under left padding, and never again on a cached decode step.
"""

import transformers as _transformers
import pytest as _pytest

pytestmark = _pytest.mark.skipif(
    int(_transformers.__version__.split(".")[0]) >= 5,
    reason="the Transformers 4 version of this test file; the 5.x version is the unsuffixed file",
)

import copy
import json
import os

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file

from wsp import Catalog, Program, load
from wsp.align import ridge_fit
from wsp.assemble import rebuild_graft
from wsp.graft import Graft, attach_grafts, glue_parameters, part_lora_parameters
from wsp.heal import heal
from wsp.lora import LoRALinear, apply_part_lora, graft_budget
from wsp.parts import build_attn_heads, build_block_part, build_mlp_cluster
from wsp.program import GraftSpec
from wsp.stitch import build_stitch, donor_sink, ridge_init, stitch_penalty, train_stitch
from tests.test_block import capture_block, donor_tensors
from tests.test_expert import part_from_disk, save_donor, tiny_mixtral
from tests.test_graft import tiny_qwen2 as graft_tiny_qwen2
from tests.test_mixed_compile import local_fetch, save_tiny
from tests.test_part_cache import SETTINGS, build_part, grafted_host, padded_prompts
from tests.test_review_align import tiny_tokenizer
from tests.test_review_parts import tiny_phi3
from tests.test_stitch import cycle_texts, perturbed_copy

BOS_TEXTS = []
for _index in range(12):
    BOS_TEXTS.append(" ".join(["the", "cat", "sat", "on", "a", "mat", "dog", "ran"][_index % 4:_index % 4 + 5]))


class Recorder(nn.Module):
    """A donor part with no weights: records its input and returns it."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return x


def tiny_llama(hidden=32):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def tiny_qwen2():
    torch.manual_seed(2)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.Qwen2ForCausalLM(config).eval()


def save_same_model(tmp_path):
    """Save one tiny Qwen2 as both host and donor, with a BOS tokenizer, and a block catalog record."""
    model = tiny_qwen2()
    host_dir = str(tmp_path / "host")
    donor_dir = str(tmp_path / "donor")
    for path in [host_dir, donor_dir]:
        model.save_pretrained(path)
        tiny_tokenizer(with_bos=True).save_pretrained(path)
    record = {"id": "qblock", "type": "block", "model": donor_dir, "layers": [1, 2]}
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(json.dumps(record) + "\n")
    return host_dir, donor_dir, Catalog.load(str(catalog_path)), model


def part_input_at_first_layer(model, ids):
    """Run a grafted model and return the input of the first donor layer of its block part."""
    store = {}

    def grab(module, args, kwargs):
        if len(args) > 0:
            store["x"] = args[0].detach().clone()
        else:
            store["x"] = kwargs["hidden_states"].detach().clone()

    handle = model.wsp_grafts[0].part.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return store["x"]


def test_block_graft_sink_is_the_donor_own_state_at_position_0(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog, donor = save_same_model(tmp_path)
    tokenizer = tiny_tokenizer(with_bos=True)
    out = str(tmp_path / "out")
    # Host layer 0 output is donor layer 1 input, since host and donor are one model.
    Program(host=host_dir).graft("qblock", at=0).compile(out, catalog, align_texts=BOS_TEXTS, device="cpu")

    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert meta["sink"] is True
    assert meta["sink_norm"] > 0
    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["grafts"][0]["sink"] is True
    saved = load_file(os.path.join(out, "graft_weights.safetensors"))
    assert saved["0.sink"].shape == (48,)

    model, _ = load(out, dtype=torch.float32)
    assert torch.equal(model.wsp_grafts[0].sink, saved["0.sink"])
    ids = tokenizer("a dog ran to the big red hat", return_tensors="pt").input_ids
    assert ids[0, 0].item() == 3
    seen = part_input_at_first_layer(model, ids)
    # The compiler runs the donor in bf16, so the sink is the bf16 donor's
    # own state. The BOS state does not depend on the text, so the mean
    # over the align texts is that state for any text.
    own_bf16, _ = capture_block(copy.deepcopy(donor).to(torch.bfloat16), 1, 2, ids)
    assert torch.equal(seen[0, 0], own_bf16[0, 0].float())
    own, _ = capture_block(donor, 1, 2, ids)
    assert torch.allclose(seen[0, 0], own[0, 0], atol=1e-3)

    # With the sink off, position 0 gets the ridge map, which is fit
    # without position-0 rows and misses the sink.
    out_off = str(tmp_path / "off")
    Program(host=host_dir).graft("qblock", at=0, sink=False).compile(out_off, catalog, align_texts=BOS_TEXTS,
                                                                      device="cpu")
    meta_off = json.load(open(os.path.join(out_off, "grafts.json")))[0]
    assert meta_off["sink"] is False
    assert "0.sink" not in load_file(os.path.join(out_off, "graft_weights.safetensors"))
    model_off, _ = load(out_off, dtype=torch.float32)
    assert model_off.wsp_grafts[0].sink is None
    seen_off = part_input_at_first_layer(model_off, ids)
    assert not torch.allclose(seen_off[0, 0], own[0, 0], atol=1e-2)


def test_no_align_texts_leaves_sink_off(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog, _ = save_same_model(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qblock", at=0).compile(out, catalog, device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert meta["sink"] is False
    model, _ = load(out, dtype=torch.float32)
    assert model.wsp_grafts[0].sink is None


def test_left_padding_puts_sink_on_first_real_token():
    host = tiny_llama()
    recorder = Recorder()
    graft = Graft(32, 16, recorder)
    sink = torch.full((16,), 7.0)
    graft.set_sink(sink)
    attach_grafts(host, [(1, graft)])
    ids, mask = padded_prompts()
    with torch.no_grad():
        host(input_ids=ids, attention_mask=mask)
    x = recorder.seen[0]
    for row in range(3):
        first = int(mask[row].nonzero()[0])
        for position in range(x.shape[1]):
            is_sink = torch.equal(x[row, position], sink)
            assert is_sink == (position == first), f"row {row} position {position}"

    # Without a mask, the sink goes on position 0.
    recorder.seen.clear()
    with torch.no_grad():
        host(input_ids=ids[:1])
    assert torch.equal(recorder.seen[0][0, 0], sink)
    assert not torch.equal(recorder.seen[0][0, 1], sink)


def test_decode_steps_do_not_put_the_sink_again():
    host = tiny_llama()
    recorder = Recorder()
    graft = Graft(32, 16, recorder)
    sink = torch.full((16,), 7.0)
    graft.set_sink(sink)
    attach_grafts(host, [(1, graft)])
    ids, mask = padded_prompts()
    with torch.no_grad():
        host.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
    prefill = recorder.seen[0]
    assert prefill.shape[1] == ids.shape[1]
    assert torch.equal(prefill[0, 0], sink)
    steps = recorder.seen[1:]
    assert len(steps) == SETTINGS["max_new_tokens"] - 1
    for step in steps:
        assert step.shape[1] == 1
        for row in range(3):
            assert not torch.equal(step[row, 0], sink)


def test_sink_graft_never_puts_the_sink_on_a_pad_under_a_static_cache():
    # A static cache gives the host a 4D mask, which hides the padding. The
    # sink would then go on position 0 of every row, a pad under left
    # padding. attach_grafts clears a static cache in the generation
    # config, and a static cache asked for at call time raises.
    host = tiny_llama()
    host.generation_config.cache_implementation = "static"
    recorder = Recorder()
    graft = Graft(32, 16, recorder)
    sink = torch.full((16,), 7.0)
    graft.set_sink(sink)
    attach_grafts(host, [(1, graft)])
    assert host.generation_config.cache_implementation is None
    ids, mask = padded_prompts()
    with torch.no_grad():
        host.generate(input_ids=ids, attention_mask=mask, **SETTINGS)
    prefill = recorder.seen[0]
    for row in range(3):
        first = int(mask[row].nonzero()[0])
        for position in range(prefill.shape[1]):
            assert torch.equal(prefill[row, position], sink) == (position == first), f"row {row} {position}"

    with pytest.raises(ValueError, match="sink"):
        with torch.no_grad():
            host.generate(input_ids=ids, attention_mask=mask, cache_implementation="static", **SETTINGS)


@pytest.mark.parametrize("part_kind", ["attn_heads", "block"])
def test_sink_cached_and_chunked_generate_equal_uncached(part_kind):
    # Uncached generate runs the whole sequence each step, so the sink sits
    # on the first real token every time. The cached and chunked paths must
    # put it there once and give the same tokens.
    host, graft = grafted_host("gemma3", part_kind)
    torch.manual_seed(9)
    graft.set_sink(torch.randn(48) * 20)
    ids, mask = padded_prompts()
    with torch.no_grad():
        uncached = host.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
        cached = host.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
        chunked = host.generate(input_ids=ids, attention_mask=mask, prefill_chunk_size=2, **SETTINGS)
    assert torch.equal(cached, uncached)
    assert torch.equal(chunked, uncached)
    # The sink must change the output, or the test checks nothing.
    plain, _ = grafted_host("gemma3", part_kind)
    with torch.no_grad():
        without = plain.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
    assert not torch.equal(without, uncached)


def test_old_saved_graft_loads_with_sink_off(monkeypatch):
    import wsp.assemble

    graft = Graft(32, 16, Recorder())
    assert graft.sink is None
    assert graft.rms_scale is None
    assert "sink" not in graft.state_dict()

    # rebuild_graft reads only the keys an old meta has. The part rebuild is
    # stubbed out, so no donor config is needed.
    def fake_part_graft(meta, state, index):
        return Graft(32, 16, Recorder())

    monkeypatch.setattr(wsp.assemble, "rebuild_part_graft", fake_part_graft)
    rebuilt = rebuild_graft({"kind": "mlp_cluster", "d_host": 32, "d_donor": 16}, {}, 0)
    assert rebuilt.sink is None
    assert rebuilt.rms_scale is None


# RMS matching.

def test_rms_match_rescales_mapped_rows():
    torch.manual_seed(0)
    recorder = Recorder()
    graft = Graft(32, 16, recorder)
    graft.set_rms_match(5.0)
    assert float(graft.rms_scale.detach()) == 1.0
    h = torch.randn(2, 6, 32) * 3
    with torch.no_grad():
        graft(h)
    rms = recorder.seen[0].pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 5.0), atol=1e-4)

    # The scale is glue and trains: doubling it doubles the RMS.
    host = tiny_llama()
    attach_grafts(host, [(1, graft)])
    found = False
    for param in glue_parameters(host):
        if param is graft.rms_scale:
            found = True
    assert found
    with torch.no_grad():
        graft.rms_scale.fill_(2.0)
        recorder.seen.clear()
        graft(h)
    rms = recorder.seen[0].pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 10.0), atol=1e-4)

    # The sink row keeps the sink, not the rescaled vector.
    graft.set_sink(torch.full((16,), 100.0))
    recorder.seen.clear()
    with torch.no_grad():
        graft(h)
    assert torch.equal(recorder.seen[0][:, 0], torch.full((2, 16), 100.0))
    rest = recorder.seen[0][:, 1:].pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rest, torch.full_like(rest, 10.0), atol=1e-4)


def test_rms_match_compiles_heals_and_reloads(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog, _ = save_same_model(tmp_path)
    out = str(tmp_path / "out")
    program = Program(host=host_dir).graft("qblock", at=0, rms_match=True).heal(steps=3, lr=1e-2, batch_size=2,
                                                                               seq_len=16)
    program.compile(out, catalog, align_texts=BOS_TEXTS, heal_texts=BOS_TEXTS, device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert meta["rms_match"] is True
    assert meta["donor_rms"] > 0
    assert meta["glue_params"] == 48 * 48 * 2 + 1
    saved = load_file(os.path.join(out, "graft_weights.safetensors"))
    assert float(saved["0.rms_scale"]) != 1.0
    model, _ = load(out, dtype=torch.float32)
    graft = model.wsp_grafts[0]
    assert abs(float(graft.donor_rms) - meta["donor_rms"]) < 1e-5
    for key, value in graft.state_dict().items():
        assert torch.equal(value, saved["0." + key].to(value.dtype))
    saved_program = json.load(open(os.path.join(out, "program.json")))
    assert saved_program["grafts"][0]["rms_match"] is True


def test_rms_match_needs_align_texts(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog, _ = save_same_model(tmp_path)
    with pytest.raises(ValueError, match="rms_match"):
        Program(host=host_dir).graft("qblock", rms_match=True).compile(str(tmp_path / "out"), catalog, device="cpu")


# The ridge default.

def test_ridge_default_recovers_exact_identity():
    torch.manual_seed(0)
    x = torch.randn(600, 32)
    # One dim with huge values, like a massive activation.
    x[:, 3] *= 100.0
    w, r2, r2_dim = ridge_fit(x, x)
    assert (w - torch.eye(32)).abs().max() < 1e-3
    assert r2_dim > 0.99999
    # The old default shrank the map, the cost the audit measured.
    w_old, _, _ = ridge_fit(x, x, lam=1e-2)
    assert (w_old - torch.eye(32)).abs().max() > 0.05
    # The fit is scale-invariant: scaled inputs give a scaled map.
    w_scaled, _, _ = ridge_fit(10.0 * x, x)
    assert torch.allclose(w_scaled * 10.0, w, atol=1e-4)


# The stitch sink.

@pytest.fixture(scope="module")
def bos_donor():
    """A tiny Qwen2 with a BOS and a huge BOS embedding, trained on cycle texts.

    The BOS is always at position 0, so its state does not depend on the
    text: an attention sink, as in real models.
    """
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    donor = transformers.Qwen2ForCausalLM(config)
    with torch.no_grad():
        donor.model.embed_tokens.weight[3] *= 30.0
    params = []
    for name, param in donor.named_parameters():
        if "embed" not in name:
            params.append(param)
    heal(donor, tiny_tokenizer(with_bos=True), cycle_texts(64, seed=0), params, steps=60, lr=1e-2, batch_size=16,
         seq_len=32, log_every=0)
    return donor.eval()


def test_stitch_penalty_is_lower_with_the_sink(bos_donor):
    tokenizer = tiny_tokenizer(with_bos=True)
    host = perturbed_copy(bos_donor, noise=0.5, seed=1)
    texts = cycle_texts(64, seed=0)
    train = texts[:48]
    held_out = texts[48:]
    ridge = build_stitch({"kind": "linear"}, 48, 48)
    ridge_init(ridge, host, tokenizer, 1, bos_donor, tokenizer, 2, train)
    sink = donor_sink(bos_donor, tokenizer, 2, train)
    on = stitch_penalty(host, tokenizer, 1, bos_donor, 2, ridge, held_out, sink=sink)
    off = stitch_penalty(host, tokenizer, 1, bos_donor, 2, ridge, held_out, sink=False)
    assert on["sink"] is True
    assert off["sink"] is False
    assert on["native_loss"] == off["native_loss"]
    assert on["penalty"] < 0.5 * off["penalty"]
    # sink=True measures the sink on the texts given. The BOS state is the
    # same for every text, so it gives the same penalty.
    measured = stitch_penalty(host, tokenizer, 1, bos_donor, 2, ridge, held_out)
    assert abs(measured["penalty"] - on["penalty"]) < 1e-4


def test_train_stitch_uses_the_sink_and_records_both_penalties(bos_donor):
    tokenizer = tiny_tokenizer(with_bos=True)
    host = perturbed_copy(bos_donor, noise=0.5, seed=1)
    texts = cycle_texts(48, seed=0)
    stitch = build_stitch({"kind": "linear"}, 48, 48)
    ridge_init(stitch, host, tokenizer, 1, bos_donor, tokenizer, 2, texts[:40])
    before = copy.deepcopy(stitch)
    result = train_stitch(host, tokenizer, 1, bos_donor, tokenizer, 2, stitch, texts[:40], objective="donor_lm",
                          steps=10, lr=1e-3, batch_size=8, seq_len=32, eval_texts=texts[40:])
    assert result["sink"] is True
    assert result["sink_norm"] > 0
    for key in ["before", "after"]:
        assert result[key]["sink"] is True
        assert result[key]["penalty"] < result[key]["penalty_no_sink"]
    sink = donor_sink(bos_donor, tokenizer, 2, texts[:40])
    expected = stitch_penalty(host, tokenizer, 1, bos_donor, 2, before, texts[40:], max_len=32, sink=sink)
    assert abs(result["before"]["penalty"] - expected["penalty"]) < 1e-5

    # The downstream objective takes the sink too.
    stitch = build_stitch({"kind": "linear"}, 48, 48)
    ridge_init(stitch, host, tokenizer, 1, bos_donor, tokenizer, 2, texts[:40])
    on = train_stitch(host, tokenizer, 1, bos_donor, tokenizer, 2, copy.deepcopy(stitch), texts[:40],
                      objective="downstream", steps=0, batch_size=8, seq_len=32, end_layer=3)
    off = train_stitch(host, tokenizer, 1, bos_donor, tokenizer, 2, copy.deepcopy(stitch), texts[:40],
                       objective="downstream", steps=0, batch_size=8, seq_len=32, end_layer=3, sink=False)
    assert on["sink"] is True
    assert off["sink"] is False
    assert on["before"]["loss"] < off["before"]["loss"]


def test_stitch_graft_trains_with_the_graft_sink(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog, _ = save_same_model(tmp_path)
    out = str(tmp_path / "out")
    stitch = {"kind": "linear", "objective": "donor_lm", "steps": 2, "lr": 1e-3, "batch_size": 2, "seq_len": 16}
    Program(host=host_dir).graft("qblock", at=0, stitch=stitch).compile(out, catalog, align_texts=BOS_TEXTS,
                                                                        device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert meta["sink"] is True
    assert meta["stitch"]["sink"] is True
    assert "penalty_no_sink" in meta["stitch"]["after"]

    out_off = str(tmp_path / "off")
    Program(host=host_dir).graft("qblock", at=0, stitch=stitch, sink=False).compile(
        out_off, catalog, align_texts=BOS_TEXTS, device="cpu")
    meta_off = json.load(open(os.path.join(out_off, "grafts.json")))[0]
    assert meta_off["stitch"]["sink"] is False
    assert "penalty_no_sink" not in meta_off["stitch"]["after"]


# Program JSON.

def test_graft_options_round_trip_and_keep_old_json():
    default = GraftSpec("x", at=3).to_dict()
    assert sorted(default) == ["at", "control", "control_seed", "glue_rank", "id"]
    program = (Program(host="some/model")
               .graft("x", at=3, sink=False, rms_match=True, part_lora_rank=4)
               .graft("y"))
    data = json.loads(json.dumps(program.to_dict()))
    assert data["grafts"][0]["sink"] is False
    assert data["grafts"][0]["rms_match"] is True
    assert data["grafts"][0]["part_lora_rank"] == 4
    for key in ["sink", "rms_match", "part_lora_rank"]:
        assert key not in data["grafts"][1]
    again = Program.from_dict(data)
    assert again.grafts[0].sink is False
    assert again.grafts[0].rms_match is True
    assert again.grafts[0].part_lora_rank == 4
    assert again.grafts[1].sink is True
    assert again.to_dict() == program.to_dict()

    old = Program.from_dict({"host": "m", "grafts": [{"id": "x", "at": 1, "control": None, "control_seed": 0,
                                                      "glue_rank": None}]})
    assert old.grafts[0].sink is True
    assert old.grafts[0].rms_match is False
    assert old.grafts[0].part_lora_rank is None
    assert old.to_dict()["grafts"][0] == {"id": "x", "at": 1, "control": None, "control_seed": 0, "glue_rank": None}

    for bad in [{"sink": "yes"}, {"rms_match": 1.5}, {"part_lora_rank": 0}, {"part_lora_rank": True}]:
        with pytest.raises(ValueError):
            GraftSpec("x", **bad)


# Part LoRA (the receiver).

def phi2_block():
    config = transformers.PhiConfig(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4, vocab_size=300,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    donor = transformers.PhiForCausalLM(config).eval()
    return build_block_part(config, 1, 1, donor_tensors(donor)), 48


def phi3_parts(part_kind):
    config, donor = tiny_phi3()
    if part_kind == "attn_heads":
        return build_attn_heads(config, 1, [0, 3], donor_tensors(donor)), 32
    return build_mlp_cluster(config, 1, torch.tensor([7, 1, 30]), donor_tensors(donor)), 32


def qwen2_cluster():
    config, donor = graft_tiny_qwen2()
    return build_mlp_cluster(config, 1, torch.arange(16), donor_tensors(donor)), 48


def experts_part(tmp_path):
    _, donor_dir = save_donor(tmp_path, tiny_mixtral)
    config, part = part_from_disk(donor_dir, 1, [1, 4])
    return part, config.hidden_size


# (name, builder, wrapped linear names). Each name is the last piece of the
# module path, counted once per linear.
PART_CASES = [
    ("mlp_cluster", qwen2_cluster, ["down_proj", "gate_proj", "up_proj"]),
    ("phi3_mlp_cluster", lambda: phi3_parts("mlp_cluster"), ["down_proj", "gate_up_proj"]),
    ("attn_heads", lambda: build_part("llama", "attn_heads"), ["k_proj", "o_proj", "q_proj", "v_proj"]),
    ("phi3_attn_heads", lambda: phi3_parts("attn_heads"), ["o_proj", "qkv_proj"]),
    ("gemma3_block", lambda: build_part("gemma3", "block"),
     ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"] * 2),
    ("phi2_block", phi2_block, ["dense", "fc1", "fc2", "k_proj", "q_proj", "v_proj"]),
]


def wrapped_names(part):
    names = []
    for name, module in part.named_modules():
        if isinstance(module, LoRALinear):
            names.append(name.split(".")[-1])
    return sorted(names)


def check_part_lora(part, d_donor, expected):
    part = part.float()
    torch.manual_seed(4)
    x = torch.randn(2, 6, d_donor)
    with torch.no_grad():
        before = part(x)
    params = apply_part_lora(part, 2)
    assert wrapped_names(part) == sorted(expected)
    assert len(params) == 2 * len(expected)
    # B starts at zero, so the wrapped part gives the same output.
    with torch.no_grad():
        assert torch.equal(part(x), before)
    # Only the LoRA factors train. The donor weights stay frozen.
    lora_ids = set()
    for param in params:
        lora_ids.add(id(param))
    for name, param in part.named_parameters():
        assert param.requires_grad == (id(param) in lora_ids), name


@pytest.mark.parametrize("name,builder,expected", PART_CASES)
def test_part_lora_wraps_part_linears_and_starts_as_noop(name, builder, expected):
    part, d_donor = builder()
    check_part_lora(part, d_donor, expected)


def test_part_lora_wraps_every_expert_linear(tmp_path):
    # Mixtral experts name their linears w1, w2 and w3.
    part, d_donor = experts_part(tmp_path)
    check_part_lora(part, d_donor, ["w1", "w2", "w3"] * 2)


def capture_compiled_model(monkeypatch):
    """Keep a reference to the host the compiler attaches grafts to."""
    import wsp.graft

    store = {}
    real = wsp.graft.attach_grafts

    def keep(model, placements):
        store["model"] = model
        return real(model, placements)

    monkeypatch.setattr(wsp.graft, "attach_grafts", keep)
    return store


def test_part_lora_heals_only_glue_and_lora_and_reloads(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    store = capture_compiled_model(monkeypatch)
    host_dir, donor_dir, catalog, donor = save_same_model(tmp_path)
    out = str(tmp_path / "out")
    program = Program(host=host_dir).graft("qblock", at=0, part_lora_rank=2).heal(steps=3, lr=1e-2, batch_size=2,
                                                                                 seq_len=16)
    program.compile(out, catalog, align_texts=BOS_TEXTS, heal_texts=BOS_TEXTS, device="cpu")

    # Two Qwen2 layers, rank 2: q and o are 48 to 48, k and v 48 to 24,
    # gate and up 48 to 96, down 96 to 48. Each LoRA has 2 * (in + out).
    per_layer = 2 * ((48 + 48) * 2 + (48 + 24) * 2 + (48 + 96) * 3)
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert meta["part_lora_rank"] == 2
    assert meta["part_lora_params"] == 2 * per_layer
    assert meta["glue_params"] == 48 * 48 * 2
    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["heal"]["part_lora_params"] == 2 * per_layer
    assert report["heal"]["trainable_params"] == meta["glue_params"] + meta["part_lora_params"]
    assert graft_budget(report["grafts"]) == meta["glue_params"] + meta["part_params"] + meta["part_lora_params"]

    # Only the glue and the part LoRA were set to train.
    model = store["model"]
    graft = model.wsp_grafts[0]
    train_ids = set()
    for param in glue_parameters(model) + part_lora_parameters(model):
        train_ids.add(id(param))
    assert len(part_lora_parameters(model)) == 28
    for name, param in model.named_parameters():
        assert param.requires_grad == (id(param) in train_ids), name

    # The donor weights did not move, and the LoRA did.
    saved = load_file(os.path.join(out, "graft_weights.safetensors"))
    native = donor.state_dict()
    for index in range(2):
        for proj in ["self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj"]:
            key = f"0.part.layers.{index}.{proj}.base.weight"
            expected = native[f"model.layers.{index + 1}.{proj}.weight"].to(saved[key].dtype)
            assert torch.equal(saved[key], expected), key
    moved = 0.0
    for key in saved:
        if key.endswith("lora_B"):
            moved += float(saved[key].abs().sum())
    assert moved > 0

    # wsp.load rebuilds the part LoRA, with the saved weights and the same
    # logits. The compiler runs the host in bf16, so the load does too.
    loaded, _ = load(out, dtype=torch.bfloat16)
    assert wrapped_names(loaded.wsp_grafts[0].part) == wrapped_names(graft.part)
    for key, value in loaded.wsp_grafts.state_dict().items():
        assert torch.equal(value, saved[key].to(value.dtype)), key
    tokenizer = tiny_tokenizer(with_bos=True)
    ids = tokenizer("a dog ran to the big red hat", return_tensors="pt").input_ids
    model.eval()
    with torch.no_grad():
        compiled_logits = model(ids).logits.float()
        loaded_logits = loaded(ids).logits.float()
    assert torch.equal(loaded_logits, compiled_logits)

    # The LoRA matters: with B zeroed the logits change.
    with torch.no_grad():
        for module in loaded.wsp_grafts[0].part.modules():
            if isinstance(module, LoRALinear):
                module.lora_B.zero_()
        plain_logits = loaded(ids).logits.float()
    assert not torch.allclose(plain_logits, loaded_logits)


def test_random_control_gets_the_same_part_lora(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    metas = []
    for control in [None, "random"]:
        out = str(tmp_path / f"out_{control}")
        Program(host=host_dir).graft("qcluster", control=control, control_seed=1, part_lora_rank=2).compile(
            out, catalog, device="cpu")
        metas.append(json.load(open(os.path.join(out, "grafts.json")))[0])
        model, _ = load(out, dtype=torch.float32)
        assert wrapped_names(model.wsp_grafts[0].part) == ["down_proj", "gate_proj", "up_proj"]
    # 8 neurons of a 48-wide donor: gate and up are 48 to 8, down 8 to 48.
    assert metas[0]["part_lora_params"] == 3 * 2 * (48 + 8)
    assert metas[1]["part_lora_params"] == metas[0]["part_lora_params"]
    assert metas[1]["control"] == "random"
    assert metas[1]["indices"] != metas[0]["indices"]
