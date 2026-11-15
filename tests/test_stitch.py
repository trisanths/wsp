import copy
import json
import os
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from wsp import Program, load
from wsp.assemble import stitch_settings
from wsp.graft import Graft, attach_grafts, glue_parameters
from wsp.heal import heal
from wsp.parts import build_block_part
from wsp.program import GraftSpec
from wsp.stitch import (
    MLPStitch,
    MultiStitch,
    build_stitch,
    feed_hooks,
    init_from_ridge,
    remove_hooks,
    ridge_init,
    stitch_penalty,
    train_stitch,
)
from tests.test_align import WORDS, tiny_tokenizer
from tests.test_block import donor_tensors, save_block_tiny
from tests.test_mixed_compile import TEXTS, local_fetch, logits


def other_tokenizer():
    """The same words as tiny_tokenizer, with the ids in another order."""
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in reversed(WORDS):
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def cycle_texts(n, seed):
    """Texts that walk the word list with step 1 or 5, from a random start."""
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        start = rng.randrange(len(WORDS))
        step = rng.choice([1, 5])
        words = []
        for i in range(12):
            words.append(WORDS[(start + step * i) % len(WORDS)])
        texts.append(" ".join(words))
    return texts


@pytest.fixture(scope="module")
def trained_donor():
    """A tiny Qwen2 trained on cycle texts, so its native loss is low."""
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    donor = transformers.Qwen2ForCausalLM(config)
    heal(donor, tiny_tokenizer(), cycle_texts(64, seed=0), list(donor.parameters()), steps=60, lr=1e-2,
         batch_size=16, seq_len=32, log_every=0)
    donor.eval()
    return donor


def perturbed_copy(model, noise, seed):
    """A copy of model with Gaussian noise on every matrix, scaled to its std."""
    host = copy.deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in host.parameters():
            if param.dim() == 2:
                param.add_(noise * param.std() * torch.randn_like(param))
    return host.eval()


def tiny_llama_host():
    torch.manual_seed(5)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def test_mlp_stitch_starts_equal_to_linear_map():
    torch.manual_seed(0)
    w = torch.randn(12, 16)
    stitch = build_stitch({"kind": "mlp"}, 16, 12)
    assert isinstance(stitch, MLPStitch)
    assert stitch.hidden == 24
    init_from_ridge(stitch, w)
    x = torch.randn(3, 5, 16) * 10
    with torch.no_grad():
        assert torch.allclose(stitch(x), x @ w.T, atol=1e-5)


def test_build_stitch_kinds():
    linear = build_stitch({"kind": "linear"}, 16, 12)
    assert isinstance(linear, torch.nn.Linear)
    assert linear.bias is None
    lowrank = build_stitch({"kind": "lowrank", "rank": 4}, 16, 12)
    assert lowrank.down.weight.shape == (4, 16)
    multi = build_stitch({"kind": "multi", "host_layers": [0, 2]}, 16, 12)
    assert multi.linear.weight.shape == (12, 32)
    with pytest.raises(ValueError):
        build_stitch({"kind": "lowrank"}, 16, 12)
    with pytest.raises(ValueError):
        build_stitch({"kind": "nope"}, 16, 12)


def test_feed_hooks_with_native_state_give_native_logits(trained_donor):
    # Feeding the donor its own state at the donor point must change nothing.
    donor = trained_donor
    ids = tiny_tokenizer()(TEXTS[0], return_tensors="pt").input_ids
    with torch.no_grad():
        native = donor(ids).logits
    store = {}

    def layer_pre(module, args, kwargs):
        store["layer"] = args[0] if len(args) > 0 else kwargs["hidden_states"]

    def norm_pre(module, args):
        store["norm"] = args[0]

    layer = donor.model.layers[2]
    handles = [layer.register_forward_pre_hook(layer_pre, with_kwargs=True),
               layer.post_attention_layernorm.register_forward_pre_hook(norm_pre)]
    with torch.no_grad():
        donor(ids)
    remove_hooks(handles)

    for prefix, key in [(None, "layer"), ("model.layers.2.post_attention_layernorm", "norm")]:
        handles = feed_hooks(donor, 2, prefix, {"x": store[key].clone()})
        with torch.no_grad():
            fed = donor(ids).logits
        remove_hooks(handles)
        assert torch.allclose(fed, native, atol=1e-5)


def test_identity_stitch_has_no_penalty(trained_donor):
    # Host layer 1 output is donor layer 2 input when the host is the donor.
    host = copy.deepcopy(trained_donor)
    stitch = build_stitch({"kind": "linear"}, 48, 48)
    init_from_ridge(stitch, torch.eye(48))
    found = stitch_penalty(host, tiny_tokenizer(), 1, trained_donor, 2, stitch, cycle_texts(8, seed=3))
    assert abs(found["penalty"]) < 1e-5
    assert found["r2"] > 0.9999
    assert found["pairs"] > 0


def test_donor_lm_lowers_donor_loss_and_beats_ridge(trained_donor):
    tokenizer = tiny_tokenizer()
    host = perturbed_copy(trained_donor, noise=1.2, seed=1)
    texts = cycle_texts(64, seed=0)
    train = texts[:48]
    held_out = texts[48:]

    ridge = build_stitch({"kind": "linear"}, 48, 48)
    ridge_init(ridge, host, tokenizer, 1, trained_donor, tokenizer, 2, train)
    ridge_found = stitch_penalty(host, tokenizer, 1, trained_donor, 2, ridge, held_out)
    assert ridge_found["penalty"] > 0.05

    stitch = build_stitch({"kind": "mlp"}, 48, 48)
    init_from_ridge(stitch, ridge.weight.detach())
    before = []
    for param in trained_donor.parameters():
        before.append(param.detach().clone())
    result = train_stitch(host, tokenizer, 1, trained_donor, tokenizer, 2, stitch, train, objective="donor_lm",
                          steps=30, lr=3e-3, batch_size=8, seq_len=32, eval_texts=held_out)
    assert len(result["losses"]) == 30
    assert sum(result["losses"][-5:]) < sum(result["losses"][:5])
    # The stitch starts from the ridge map, so "before" is the ridge penalty.
    assert abs(result["before"]["penalty"] - ridge_found["penalty"]) < 1e-5
    assert result["after"]["stitched_loss"] < result["before"]["stitched_loss"]
    assert result["after"]["penalty"] < ridge_found["penalty"]
    # Host and donor stay frozen and unchanged.
    index = 0
    for param in trained_donor.parameters():
        assert torch.equal(param, before[index])
        assert param.requires_grad is True
        index += 1


def test_donor_lm_raises_on_mismatched_vocab(trained_donor):
    host = perturbed_copy(trained_donor, noise=0.5, seed=1)
    stitch = build_stitch({"kind": "linear"}, 48, 48)
    with pytest.raises(ValueError, match="share a tokenizer"):
        train_stitch(host, tiny_tokenizer(), 1, trained_donor, other_tokenizer(), 2, stitch, TEXTS,
                     objective="donor_lm", steps=1)


def test_same_model_as_host_and_donor_keeps_requires_grad(trained_donor):
    # Freezing the same model twice must not leave it frozen afterwards.
    model = copy.deepcopy(trained_donor)
    stitch = build_stitch({"kind": "linear"}, 48, 48)
    init_from_ridge(stitch, torch.eye(48))
    train_stitch(model, tiny_tokenizer(), 1, model, tiny_tokenizer(), 2, stitch, cycle_texts(8, seed=1),
                 objective="donor_lm", steps=1, batch_size=2, seq_len=16)
    for param in model.parameters():
        assert param.requires_grad is True


def test_one_token_texts_give_no_nan_loss(trained_donor):
    # A batch of one-token texts has no labels; donor_lm must skip such texts.
    tokenizer = tiny_tokenizer()
    host = perturbed_copy(trained_donor, noise=0.5, seed=7)
    stitch = build_stitch({"kind": "linear"}, 48, 48)
    init_from_ridge(stitch, torch.eye(48))
    texts = ["cat", "dog", "the cat sat on a mat"]
    result = train_stitch(host, tokenizer, 1, trained_donor, tokenizer, 2, stitch, texts, objective="donor_lm",
                          steps=6, lr=1e-3, batch_size=2, seq_len=16)
    assert result["train_texts"] == 1
    for value in result["losses"]:
        assert value == value
    with pytest.raises(ValueError, match="at least one training text"):
        train_stitch(host, tokenizer, 1, trained_donor, tokenizer, 2, stitch, ["cat"], objective="donor_lm",
                     steps=1)


def test_multi_stitch_spec_rejects_host_layers():
    # Offsets set the host layers; a host_layers key would be dropped silently.
    spec = GraftSpec("x", stitch={"kind": "multi", "host_layers": [0, 1]})
    with pytest.raises(ValueError, match="offsets"):
        stitch_settings(spec, None, 2, 48)
    spec = GraftSpec("x", stitch={"kind": "multi", "offsets": [-1, 0]})
    assert stitch_settings(spec, None, 2, 48)["host_layers"] == [1, 2]


def test_mse_objective_lowers_standardized_error(trained_donor):
    tokenizer = tiny_tokenizer()
    host = tiny_llama_host()
    stitch = build_stitch({"kind": "lowrank", "rank": 8}, 64, 48)
    texts = cycle_texts(32, seed=2)
    result = train_stitch(host, tokenizer, 1, trained_donor, tokenizer, 2, stitch, texts, objective="mse",
                          steps=40, lr=1e-2, batch_size=4, seq_len=32)
    assert result["after"]["loss"] < result["before"]["loss"]
    assert result["after"]["r2_dim"] > result["before"]["r2_dim"]


def test_downstream_objective_runs_across_families_and_tokenizers(trained_donor):
    host = tiny_llama_host()
    stitch = build_stitch({"kind": "mlp"}, 64, 48)
    texts = cycle_texts(32, seed=4)
    result = train_stitch(host, tiny_tokenizer(), 1, trained_donor, other_tokenizer(), 1, stitch, texts,
                          objective="downstream", steps=30, lr=3e-3, batch_size=8, seq_len=32)
    assert result["end_layer"] == 3
    assert result["before"]["rows"] > 0
    assert result["after"]["loss"] < result["before"]["loss"]
    assert sum(result["losses"][-5:]) < sum(result["losses"][:5])


def test_downstream_at_pre_mlp_norm(trained_donor):
    host = tiny_llama_host()
    stitch = build_stitch({"kind": "linear"}, 64, 48)
    texts = cycle_texts(24, seed=5)
    ridge_init(stitch, host, tiny_tokenizer(), 1, trained_donor, tiny_tokenizer(), 2, texts,
               donor_norm_prefix="model.layers.2.post_attention_layernorm")
    result = train_stitch(host, tiny_tokenizer(), 1, trained_donor, tiny_tokenizer(), 2, stitch, texts,
                          objective="downstream", steps=20, lr=3e-3, batch_size=8, seq_len=32, end_layer=2,
                          donor_norm_prefix="model.layers.2.post_attention_layernorm")
    assert result["after"]["loss"] < result["before"]["loss"]


def test_multi_stitch_trains_with_donor_lm(trained_donor):
    tokenizer = tiny_tokenizer()
    host = perturbed_copy(trained_donor, noise=1.2, seed=1)
    stitch = build_stitch({"kind": "multi", "host_layers": [0, 1]}, 48, 48)
    texts = cycle_texts(48, seed=0)
    fit = ridge_init(stitch, host, tokenizer, 1, trained_donor, tokenizer, 2, texts)
    assert fit["pairs"] > 0
    result = train_stitch(host, tokenizer, 1, trained_donor, tokenizer, 2, stitch, texts, objective="donor_lm",
                          steps=20, lr=3e-3, batch_size=8, seq_len=32)
    assert result["host_layers"] == [0, 1]
    assert result["after"]["stitched_loss"] < result["before"]["stitched_loss"]


def test_multi_stitch_graft_reads_earlier_host_layer():
    host = tiny_llama_host()
    torch.manual_seed(0)
    part = torch.nn.Sequential(torch.nn.Linear(48, 48, bias=False))
    stitch = MultiStitch(64, 48, [0, 2])
    graft = Graft(64, 48, part, w_in=stitch)
    with torch.no_grad():
        graft.w_out.weight.normal_()
    ids = tiny_tokenizer()("the cat sat on a mat", return_tensors="pt").input_ids

    # Reference: read layers 0 and 2 by hand and add the graft output at layer 2.
    reference = copy.deepcopy(host)
    store = {}

    def keep(module, args, output):
        store["h0"] = output[0] if isinstance(output, tuple) else output

    def add(module, args, output):
        h2 = output[0] if isinstance(output, tuple) else output
        x = stitch([store["h0"], h2])
        return h2 + graft.w_out(part(x))

    reference.model.layers[0].register_forward_hook(keep)
    reference.model.layers[2].register_forward_hook(add)

    attach_grafts(host, [(2, graft)])
    with torch.no_grad():
        assert torch.allclose(host(ids).logits, reference(ids).logits, atol=1e-5)
    assert graft.stitch_inputs == {}


def test_multi_stitch_cannot_read_later_layer():
    host = tiny_llama_host()
    part = torch.nn.Sequential(torch.nn.Linear(48, 48, bias=False))
    graft = Graft(64, 48, part, w_in=MultiStitch(64, 48, [1, 3]))
    with pytest.raises(ValueError):
        attach_grafts(host, [(2, graft)])


def test_stitch_params_are_glue():
    host = tiny_llama_host()
    part = torch.nn.Sequential(torch.nn.Linear(48, 48, bias=False))
    stitch = MLPStitch(64, 48)
    graft = Graft(64, 48, part, w_in=stitch)
    attach_grafts(host, [(1, graft)])
    params = glue_parameters(host)
    assert len(params) == 4
    found = False
    for param in params:
        if param is stitch.w1.weight:
            found = True
    assert found


def test_graft_spec_stitch_round_trip():
    stitch = {"kind": "mlp", "objective": "donor_lm", "steps": 500, "lr": 1e-3}
    program = Program(host="some/model").graft("x", at=3, stitch=stitch).graft("y")
    data = json.loads(json.dumps(program.to_dict()))
    assert data["grafts"][0]["stitch"] == stitch
    # A graft without a stitch keeps the old JSON.
    assert "stitch" not in data["grafts"][1]
    again = Program.from_dict(data)
    assert again.grafts[0].stitch == stitch
    assert again.grafts[1].stitch is None
    assert again.to_dict() == program.to_dict()
    old = Program.from_dict({"host": "m", "grafts": [{"id": "x", "at": 1, "control": None, "control_seed": 0,
                                                      "glue_rank": None}]})
    assert old.grafts[0].stitch is None
    with pytest.raises(ValueError):
        GraftSpec("x", stitch={"kind": "bad"})
    with pytest.raises(ValueError):
        GraftSpec("x", stitch={"kind": "mlp", "objective": "bad"})


def expected_logits(out, host_dir, donor_dir, host_layer, w_in, text):
    """Build the saved block graft by hand from the donor and the saved weights."""
    state = load_file(os.path.join(out, "graft_weights.safetensors"))
    donor = transformers.Qwen2ForCausalLM.from_pretrained(donor_dir, dtype=torch.float32)
    part = build_block_part(donor.config, 1, 2, donor_tensors(donor))
    graft = Graft(64, 48, part, w_in=w_in)
    own = {}
    for key in state:
        if key.startswith("0."):
            own[key[2:]] = state[key].float()
    graft.load_state_dict(own, strict=True)
    host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32).eval()
    attach_grafts(host, [(host_layer, graft)])
    return logits(host, text)


def test_program_with_stitch_compiles_saves_and_loads(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_block_tiny(tmp_path)
    out = str(tmp_path / "out")
    stitch = {"kind": "mlp", "objective": "donor_lm", "steps": 3, "lr": 1e-2, "batch_size": 2, "seq_len": 16}
    program = Program(host=host_dir).graft("qblock", at=1, stitch=stitch).heal(steps=2, lr=1e-2, batch_size=2,
                                                                             seq_len=16)
    program.compile(out, catalog, align_texts=TEXTS, heal_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    meta = report["grafts"][0]
    assert meta["fit"] == "ridge"
    info = meta["stitch"]
    assert info["kind"] == "mlp"
    assert info["hidden"] == 96
    assert len(info["losses"]) == 3
    assert info["train_texts"] == 11
    assert info["eval_texts"] == 1
    for key in ["native_loss", "stitched_loss", "penalty"]:
        assert key in info["before"]
        assert key in info["after"]
    assert meta["glue_params"] == (64 * 48 + 64 * 96 + 96 * 48) + 48 * 64
    saved_program = json.load(open(os.path.join(out, "program.json")))
    assert saved_program["grafts"][0]["stitch"] == stitch

    model, _ = load(out, dtype=torch.float32)
    assert isinstance(model.wsp_grafts[0].w_in, MLPStitch)
    loaded = logits(model, "the cat sat")
    host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert not torch.equal(loaded, logits(host, "the cat sat"))
    expected = expected_logits(out, host_dir, donor_dir, 1, MLPStitch(64, 48, hidden=96), "the cat sat")
    assert torch.equal(loaded, expected)


def test_program_with_multi_stitch_compiles_and_loads(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_block_tiny(tmp_path)
    out = str(tmp_path / "out")
    stitch = {"kind": "multi", "objective": "mse", "steps": 3, "lr": 1e-2, "batch_size": 2, "seq_len": 16,
              "offsets": [-1, 0]}
    program = Program(host=host_dir).graft("qblock", at=2, stitch=stitch).heal(steps=2, lr=1e-2, batch_size=2,
                                                                             seq_len=16)
    program.compile(out, catalog, align_texts=TEXTS, heal_texts=TEXTS, device="cpu")

    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert meta["stitch"]["host_layers"] == [1, 2]
    assert "r2_dim" in meta["stitch"]["after"]
    model, _ = load(out, dtype=torch.float32)
    assert model.wsp_grafts[0].w_in.host_layers == [1, 2]
    loaded = logits(model, "the cat sat")
    expected = expected_logits(out, host_dir, donor_dir, 2, MultiStitch(64, 48, [1, 2]), "the cat sat")
    assert torch.equal(loaded, expected)


def test_stitch_needs_align_texts(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_block_tiny(tmp_path)
    program = Program(host=host_dir).graft("qblock", stitch={"kind": "linear", "steps": 1})
    with pytest.raises(ValueError, match="align_texts"):
        program.compile(str(tmp_path / "out"), catalog, device="cpu")
