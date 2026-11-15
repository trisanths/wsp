
import transformers as _transformers
import pytest as _pytest

pytestmark = _pytest.mark.skipif(
    int(_transformers.__version__.split(".")[0]) >= 5,
    reason="the Transformers 4 version of this test file; the 5.x version is the unsuffixed file",
)
import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file

from wsp import Catalog, Program, load
from wsp.assemble import random_block_range
from wsp.graft import Graft, attach_grafts
from wsp.parts import BlockPart, block_tensor_names, build_block_part
from tests.test_mixed_compile import local_fetch, logits, TEXTS
from tests.test_align import tiny_tokenizer


def tiny_donor(kind):
    torch.manual_seed(3)
    if kind == "llama":
        config = transformers.LlamaConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        return config, transformers.LlamaForCausalLM(config).eval()
    if kind == "qwen2_sliding":
        # Layers 1 and up use a sliding window of 3 tokens.
        config = transformers.Qwen2Config(
            hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
            use_sliding_window=True, sliding_window=3, max_window_layers=1,
        )
        config._attn_implementation = "eager"
        return config, transformers.Qwen2ForCausalLM(config).eval()
    if kind == "gemma3":
        config = transformers.Gemma3TextConfig(
            hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=12, vocab_size=300, max_position_embeddings=64,
            sliding_window=3, rope_local_base_freq=100.0,
            layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
        )
        config._attn_implementation = "eager"
        return config, transformers.Gemma3ForCausalLM(config).eval()
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return config, transformers.Qwen2ForCausalLM(config).eval()


def donor_tensors(model):
    out = {}
    for name, value in model.state_dict().items():
        out[name] = value.detach().clone()
    return out


def capture_block(model, first, last, input_ids, attention_mask=None, position_ids=None):
    """Run the full donor and return (input of layer first, output of layer last)."""
    layers = model.model.layers
    store = {}

    def pre_hook(module, args, kwargs):
        if len(args) > 0:
            store["in"] = args[0].detach().clone()
        else:
            store["in"] = kwargs["hidden_states"].detach().clone()

    def post_hook(module, args, output):
        if isinstance(output, tuple):
            output = output[0]
        store["out"] = output.detach().clone()

    handles = [
        layers[first].register_forward_pre_hook(pre_hook, with_kwargs=True),
        layers[last].register_forward_hook(post_hook),
    ]
    try:
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return store["in"], store["out"]


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen2_sliding", "gemma3"])
def test_block_equals_donor_layers(kind):
    config, donor = tiny_donor(kind)
    part = build_block_part(config, 1, 2, donor_tensors(donor))
    assert isinstance(part, BlockPart)
    ids = torch.randint(3, 300, (2, 7))
    x, y = capture_block(donor, 1, 2, ids)
    with torch.no_grad():
        out = part(x)
    assert torch.allclose(out, y - x, atol=1e-5)


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen2_sliding", "gemma3"])
def test_block_with_left_padding(kind):
    config, donor = tiny_donor(kind)
    part = build_block_part(config, 1, 2, donor_tensors(donor))
    ids = torch.randint(3, 300, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, :3] = 0
    ids[0, :3] = 2
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    x, y = capture_block(donor, 1, 2, ids, attention_mask=mask, position_ids=position_ids)
    with torch.no_grad():
        out = part(x, attention_mask=mask)
    real = mask.bool()
    assert torch.allclose(out[real], (y - x)[real], atol=1e-5)


def test_block_tensor_names_cover_whole_layers():
    config, donor = tiny_donor("qwen2")
    names = block_tensor_names(config, 1, 2)
    expected = []
    for name in donor.state_dict().keys():
        if name.startswith("model.layers.1.") or name.startswith("model.layers.2."):
            expected.append(name)
    assert sorted(names) == sorted(expected)


def test_identity_block_adds_nothing():
    config, donor = tiny_donor("llama")
    tensors = donor_tensors(donor)
    # Zero output projections make both layers pass the residual through.
    for name in tensors:
        if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
            tensors[name] = torch.zeros_like(tensors[name])
    part = build_block_part(config, 1, 2, tensors)
    x = torch.randn(2, 5, config.hidden_size)
    with torch.no_grad():
        assert torch.equal(part(x), torch.zeros_like(x))


def test_block_ignores_train_mode():
    # The heal pass calls model.train(). Donor dropout must stay off.
    config = transformers.PhiConfig(
        hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        vocab_size=300, resid_pdrop=0.5, attention_dropout=0.5,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    donor = transformers.PhiForCausalLM(config).eval()
    part = build_block_part(config, 1, 2, donor_tensors(donor))
    host = nn_wrapper(part)
    host.train()
    assert part.training is False
    x = torch.randn(1, 6, 48)
    with torch.no_grad():
        assert torch.equal(part(x), part(x))


def nn_wrapper(part):
    wrapper = torch.nn.Module()
    wrapper.part = part
    return wrapper


def test_random_block_range():
    first, last = random_block_range(6, 1, 2, seed=0)
    assert last - first == 1
    assert last < 1 or first > 2
    # With no room to stay apart, the range still differs.
    first, last = random_block_range(3, 1, 2, seed=0)
    assert (first, last) == (0, 1)
    with pytest.raises(ValueError):
        random_block_range(3, 0, 2, seed=0)


def save_block_tiny(tmp_path, donor_layers=3):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    host = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    donor = Qwen2ForCausalLM(Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=donor_layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    host_dir = str(tmp_path / "host")
    donor_dir = str(tmp_path / "donor")
    for model, path in [(host, host_dir), (donor, donor_dir)]:
        model.save_pretrained(path)
        tiny_tokenizer().save_pretrained(path)
    record = {"id": "qblock", "type": "block", "model": donor_dir, "layers": [1, 2]}
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(json.dumps(record) + "\n")
    return host_dir, donor_dir, Catalog.load(str(catalog_path))


def test_fresh_block_graft_gives_host_logits(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_block_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qblock").compile(out, catalog, align_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    graft = report["grafts"][0]
    assert graft["kind"] == "block"
    assert graft["fit"] == "ridge"
    assert graft["fit_pairs"] > 0
    assert graft["layers"] == [1, 2]
    assert graft["host_layer"] == round(1 / 3 * 4)

    model, _ = load(out, dtype=torch.float32)
    # A block part keeps its own KV cache, so the cache stays on.
    assert model.config.use_cache is not False
    host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_healed_block_graft_reloads_with_saved_weights(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_block_tiny(tmp_path)
    out = str(tmp_path / "out")
    program = Program(host=host_dir).graft("qblock", at=2).heal(steps=3, lr=1e-2, batch_size=2, seq_len=16)
    program.compile(out, catalog, heal_texts=TEXTS, device="cpu")

    model, _ = load(out, dtype=torch.float32)
    host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    loaded = logits(model, "the cat sat")
    assert not torch.equal(loaded, logits(host, "the cat sat"))

    # Build the same graft from the original donor and the saved glue.
    state = load_file(os.path.join(out, "graft_weights.safetensors"))
    donor = transformers.Qwen2ForCausalLM.from_pretrained(donor_dir, dtype=torch.float32)
    part = build_block_part(donor.config, 1, 2, donor_tensors(donor))
    graft = Graft(64, 48, part)
    with torch.no_grad():
        graft.w_in.weight.copy_(state["0.w_in.weight"])
        graft.w_out.weight.copy_(state["0.w_out.weight"])
    expected_host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32).eval()
    attach_grafts(expected_host, [(2, graft)])
    assert torch.equal(loaded, logits(expected_host, "the cat sat"))


def test_random_control_picks_other_range(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_block_tiny(tmp_path, donor_layers=6)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qblock", control="random", control_seed=1).compile(out, catalog, device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    first, last = meta["layers"]
    assert meta["kind"] == "block"
    assert meta["control"] == "random"
    assert last - first == 1
    assert last < 1 or first > 2
    # The control sits where the real block would.
    assert meta["host_layer"] == round(1 / 6 * 4)

    model, _ = load(out, dtype=torch.float32)
    host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))
