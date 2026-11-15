import json
import os

import pytest
import torch
import transformers
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from wsp import atlas, giant

SKILL_WORDS = ["add", "sum", "plus", "minus", "times", "equals", "number", "digit"]
GENERAL_WORDS = ["the", "river", "city", "king", "song", "film", "tree", "road"]
EXTRA = ["A", "B", "C", "D", "Question:", "Answer:", "yes", "no", "."]


def tokenizer():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in SKILL_WORDS + GENERAL_WORDS + EXTRA:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def texts(words, count, length, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    out = []
    for _ in range(count):
        picks = torch.randint(len(words), (length,), generator=generator).tolist()
        out.append(" ".join(words[pick] for pick in picks))
    return out


SKILL = texts(SKILL_WORDS + GENERAL_WORDS, 5, 12, seed=1)
OTHER = texts(SKILL_WORDS, 4, 10, seed=2)
GENERAL = texts(GENERAL_WORDS + SKILL_WORDS[:2], 5, 12, seed=3)

FAMILIES = ["qwen3_moe", "olmoe", "mixtral", "deepseek_v2", "deepseek_v3"]


def tiny_model(family, layers=3):
    torch.manual_seed(0)
    T = transformers
    if family == "qwen3_moe":
        model = T.Qwen3MoeForCausalLM(T.Qwen3MoeConfig(
            hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=40, max_position_embeddings=64, num_experts=6,
            num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[1], head_dim=12))
    elif family == "olmoe":
        model = T.OlmoeForCausalLM(T.OlmoeConfig(
            hidden_size=48, intermediate_size=40, num_hidden_layers=layers, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=40, max_position_embeddings=64, num_experts=6, num_experts_per_tok=2))
    elif family == "mixtral":
        model = T.MixtralForCausalLM(T.MixtralConfig(
            hidden_size=48, intermediate_size=40, num_hidden_layers=layers, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=40, max_position_embeddings=64, num_local_experts=6,
            num_experts_per_tok=2))
    elif family == "deepseek_v2":
        model = T.DeepseekV2ForCausalLM(T.DeepseekV2Config(
            hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=4, vocab_size=40, max_position_embeddings=64,
            n_routed_experts=6, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1, kv_lora_rank=16,
            q_lora_rank=None, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, n_group=1, topk_group=1,
            topk_method="greedy"))
    else:
        model = T.DeepseekV3ForCausalLM(T.DeepseekV3Config(
            hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=4, vocab_size=40, max_position_embeddings=64,
            n_routed_experts=6, num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1, kv_lora_rank=16,
            q_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, n_group=1, topk_group=1))
        # A non-zero correction bias, so the router bias is a real tensor to compare.
        for layer in model.model.layers:
            gate = getattr(layer.mlp, "gate", None)
            bias = getattr(gate, "e_score_correction_bias", None)
            if bias is not None:
                bias.data = torch.randn_like(bias) * 0.1
    return model.eval()


def saved(tmp_path, family, layers=3):
    model = tiny_model(family, layers)
    folder = str(tmp_path / family)
    model.save_pretrained(folder)
    return model, folder


def same_map(left, right):
    if left["top_k"] != right["top_k"] or left["num_experts"] != right["num_experts"]:
        return False
    if set(left["layers"]) != set(right["layers"]):
        return False
    for layer in left["layers"]:
        if left["layers"][layer] != right["layers"][layer]:
            return False
    return True


def reference_maps(model, tok):
    return {"a": atlas.expert_skill_map(model, tok, SKILL, GENERAL, top=3),
            "b": atlas.expert_skill_map(model, tok, OTHER, GENERAL, top=3)}


def stream(folder, model, tok, **kwargs):
    source = giant.LocalSource(folder)
    kwargs.setdefault("prefetch", False)
    kwargs.setdefault("log", lambda text: None)
    return giant.profile_streaming(source, model.config, tok, {"a": SKILL, "b": OTHER}, GENERAL, top=3, **kwargs)


# ---------------------------------------------------------------------------
# Exactness against the fully loaded model.


@pytest.mark.parametrize("family", FAMILIES)
def test_streamed_profile_equals_atlas_expert_map(tmp_path, family):
    tok = tokenizer()
    model, folder = saved(tmp_path, family)
    expected = reference_maps(model, tok)
    result = stream(folder, model, tok)
    assert result["info"]["layout"] == "hf"
    for skill in expected:
        assert same_map(expected[skill], result["maps"][skill]), family
    moe_layers = sorted(expected["a"]["layers"])
    assert result["info"]["layers_with_routers"] == moe_layers
    assert result["info"]["layers_done"] == 3


def test_dense_layers_have_no_entry(tmp_path):
    tok = tokenizer()
    model, folder = saved(tmp_path, "qwen3_moe")
    result = stream(folder, model, tok)
    assert sorted(result["maps"]["a"]["layers"]) == [0, 2]


@pytest.mark.parametrize("family", ["qwen3_moe", "mixtral", "deepseek_v3"])
def test_streamed_hidden_states_equal_full_forward(tmp_path, family):
    tok = tokenizer()
    model, folder = saved(tmp_path, family)
    result = stream(folder, model, tok, max_layers=2, keep_state=True)
    states = result["info"]["state"]
    items = []
    for name, group in [("general", GENERAL), ("a", SKILL), ("b", OTHER)]:
        for ids in atlas.encode_texts(tok, group, 256):
            items.append(ids)
    assert len(states) == len(items)
    for ids, state in zip(items, states):
        with torch.no_grad():
            full = model(input_ids=ids.unsqueeze(0), use_cache=False, output_hidden_states=True).hidden_states[2]
        assert torch.equal(full, state), family


def test_max_layers_limits_the_map(tmp_path):
    tok = tokenizer()
    model, folder = saved(tmp_path, "olmoe")
    result = stream(folder, model, tok, max_layers=1)
    assert sorted(result["maps"]["a"]["layers"]) == [0]
    assert result["info"]["layers_done"] == 1


def test_prefetch_gives_the_same_maps(tmp_path):
    tok = tokenizer()
    model, folder = saved(tmp_path, "mixtral")
    plain = stream(folder, model, tok, prefetch=False)
    fetched = stream(folder, model, tok, prefetch=True)
    for skill in plain["maps"]:
        assert same_map(plain["maps"][skill], fetched["maps"][skill])


def test_resume_continues_from_saved_state(tmp_path):
    tok = tokenizer()
    model, folder = saved(tmp_path, "qwen3_moe", layers=4)
    expected = stream(folder, model, tok)
    state_dir = str(tmp_path / "state")
    first = stream(folder, model, tok, max_layers=2, state_dir=state_dir, checkpoint_every=1)
    assert first["info"]["layers_done"] == 2
    second = stream(folder, model, tok, state_dir=state_dir, checkpoint_every=1)
    assert len(second["info"]["layer_seconds"]) == 2
    for skill in expected["maps"]:
        assert same_map(expected["maps"][skill], second["maps"][skill])


def test_resume_ignores_state_of_other_texts(tmp_path):
    tok = tokenizer()
    model, folder = saved(tmp_path, "qwen3_moe")
    state_dir = str(tmp_path / "state")
    stream(folder, model, tok, max_layers=2, state_dir=state_dir, checkpoint_every=1)
    source = giant.LocalSource(folder)
    result = giant.profile_streaming(source, model.config, tok, {"a": OTHER}, GENERAL, top=3, prefetch=False,
                                     state_dir=state_dir, checkpoint_every=1, log=lambda text: None)
    assert len(result["info"]["layer_seconds"]) == 3


def test_dense_model_is_refused(tmp_path):
    tok = tokenizer()
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=40, max_position_embeddings=64)).eval()
    folder = str(tmp_path / "dense")
    model.save_pretrained(folder)
    with pytest.raises(ValueError, match="no MoE layers"):
        giant.profile_streaming(giant.LocalSource(folder), model.config, tok, {"a": SKILL}, GENERAL,
                                prefetch=False, log=lambda text: None)


def test_missing_tensor_is_reported(tmp_path):
    from safetensors.torch import load_file

    tok = tokenizer()
    model, folder = saved(tmp_path, "mixtral")
    path = os.path.join(folder, "model.safetensors")
    tensors = load_file(path)
    del tensors["model.layers.1.block_sparse_moe.gate.weight"]
    save_file(tensors, path)
    with pytest.raises(ValueError, match="layer 1"):
        stream(folder, model, tok)


# ---------------------------------------------------------------------------
# Single experts and byte counts.


def expert_state(model, family, layer, expert):
    state = model.state_dict()
    base = f"model.layers.{layer}."
    if family == "mixtral":
        return (state[base + f"block_sparse_moe.experts.{expert}.w1.weight"],
                state[base + f"block_sparse_moe.experts.{expert}.w3.weight"],
                state[base + f"block_sparse_moe.experts.{expert}.w2.weight"])
    return (state[base + f"mlp.experts.{expert}.gate_proj.weight"], state[base + f"mlp.experts.{expert}.up_proj.weight"],
            state[base + f"mlp.experts.{expert}.down_proj.weight"])


@pytest.mark.parametrize("family", ["qwen3_moe", "olmoe", "mixtral", "deepseek_v3"])
def test_fetch_expert_equals_state_dict_reads(tmp_path, family):
    model, folder = saved(tmp_path, family)
    source = giant.LocalSource(folder)
    layout = giant.layout_for(model.config, source)
    picks = [4, 1]
    result = giant.fetch_expert(source, 2, picks, layout=layout, dtype=torch.float32)
    assert result["experts"] == picks
    state = model.state_dict()
    for position, expert in enumerate(picks):
        gate, up, down = expert_state(model, family, 2, expert)
        assert torch.equal(result["gate"][position], gate)
        assert torch.equal(result["up"][position], up)
        assert torch.equal(result["down"][position], down)
    prefix = "model.layers.2."
    router = state[prefix + ("block_sparse_moe.gate.weight" if family == "mixtral" else "mlp.gate.weight")]
    assert torch.equal(result["router_rows"], router[picks])
    if family == "deepseek_v3":
        bias = state[prefix + "mlp.gate.e_score_correction_bias"]
        assert torch.equal(result["router_bias"], bias[picks].float())
    else:
        assert result["router_bias"] is None


def test_fetch_expert_reads_only_the_chosen_bytes(tmp_path):
    model, folder = saved(tmp_path, "qwen3_moe")
    source = giant.LocalSource(folder)
    source.gap = 0
    layout = giant.layout_for(model.config, source)
    total_file = os.path.getsize(os.path.join(folder, "model.safetensors"))
    result = giant.fetch_expert(source, 2, [3], layout=layout, dtype=torch.float32)
    assert 0 < result["bytes"] < total_file / 4
    predicted = giant.expert_download_bytes(source, layout, {2: [3]})
    assert predicted["total"] == result["bytes"]
    assert predicted["per_layer"] == {2: result["bytes"]}


def test_expert_download_bytes_adds_up_over_layers(tmp_path):
    model, folder = saved(tmp_path, "olmoe")
    source = giant.LocalSource(folder)
    layout = giant.layout_for(model.config, source)
    one = giant.expert_download_bytes(source, layout, {0: [1]})["total"]
    two = giant.expert_download_bytes(source, layout, {0: [1], 1: [1, 2]})["total"]
    assert two == one * 3


def test_fetch_expert_from_fused_checkpoint_slices_rows(tmp_path):
    torch.manual_seed(0)
    experts, hidden, inter = 5, 8, 6
    gate_up = torch.randn(experts, 2 * inter, hidden)
    down = torch.randn(experts, hidden, inter)
    router = torch.randn(experts, hidden)
    folder = tmp_path / "fused"
    folder.mkdir()
    save_file({"model.layers.0.mlp.experts.gate_up_proj": gate_up, "model.layers.0.mlp.experts.down_proj": down,
               "model.layers.0.mlp.gate.weight": router}, str(folder / "model.safetensors"))
    source = giant.LocalSource(str(folder))
    layout = giant.Layout()
    result = giant.fetch_expert(source, 0, [3, 0], layout=layout, dtype=torch.float32)
    assert torch.equal(result["gate"], gate_up[[3, 0], :inter])
    assert torch.equal(result["up"], gate_up[[3, 0], inter:])
    assert torch.equal(result["down"], down[[3, 0]])
    assert torch.equal(result["router_rows"], router[[3, 0]])
    row_bytes = (2 * inter * hidden + hidden * inter + hidden) * 4 * 2
    assert result["bytes"] == row_bytes
    assert giant.expert_download_bytes(source, layout, {0: [3, 0]})["total"] == row_bytes


# ---------------------------------------------------------------------------
# Sources.


def test_read_merges_close_tensors_into_one_request(tmp_path):
    tensors = {f"t{i}": torch.randn(4, 4) for i in range(6)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    source = giant.LocalSource(str(tmp_path))
    before = source.requests
    got = source.read(list(tensors))
    for name in tensors:
        assert torch.equal(got[name], tensors[name])
    assert source.requests - before == 1


def test_read_rows_matches_indexing(tmp_path):
    table = torch.randn(10, 3, 5).to(torch.bfloat16)
    save_file({"w": table}, str(tmp_path / "model.safetensors"))
    source = giant.LocalSource(str(tmp_path))
    rows = source.read_rows("w", [7, 2, 3])
    assert torch.equal(rows, table[[7, 2, 3]])
    with pytest.raises(IndexError):
        source.read_rows("w", [10])


def test_plan_ranges_splits_far_spans_and_caps_size():
    spans = [(0, 10, "a"), (12, 20, "b"), (5000, 5010, "c")]
    groups = giant.plan_ranges(spans, gap=100)
    assert [(g[0], g[1]) for g in groups] == [(0, 20), (5000, 5010)]
    capped = giant.plan_ranges([(0, 10, "a"), (10, 20, "b")], gap=100, cap=15)
    assert len(capped) == 2


def test_decode_bytes_reads_float8_types():
    values = torch.tensor([0.5, -2.0, 3.0]).to(torch.float8_e4m3fn)
    got = giant.decode_bytes(values.view(torch.uint8).numpy().tobytes(), "F8_E4M3", [3])
    assert got.dtype == torch.float8_e4m3fn
    assert torch.equal(got.float(), values.float())
    exponent = giant.decode_bytes(bytes([127, 130]), "F8_E8M0", [2])
    assert exponent.float().tolist() == [1.0, 8.0]
    with pytest.raises(ValueError):
        giant.decode_bytes(b"", "F4", [1])


def test_missing_tensor_name_raises_key_error(tmp_path):
    save_file({"a": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    source = giant.LocalSource(str(tmp_path))
    with pytest.raises(KeyError, match="nope"):
        source.read(["nope"])


# ---------------------------------------------------------------------------
# Stored quantization.


def quantize_fp8(weight, block=128):
    """FP8 e4m3 with power-of-two block scales. Returns (stored, scale as e8m0, decoded float32)."""
    rows, cols = weight.shape
    br = -(-rows // block)
    bc = -(-cols // block)
    scale = torch.zeros(br, bc)
    for i in range(br):
        for j in range(bc):
            tile = weight[i * block:(i + 1) * block, j * block:(j + 1) * block]
            top = tile.abs().max().clamp(min=1e-8) / 448.0
            scale[i, j] = 2.0 ** torch.ceil(torch.log2(top))
    full = scale.repeat_interleave(block, 0)[:rows].repeat_interleave(block, 1)[:, :cols]
    stored = (weight / full).to(torch.float8_e4m3fn)
    return stored, scale.to(torch.float8_e8m0fnu), stored.float() * full


def quantize_fp4(weight, block=32):
    """FP4 e2m1 packed two per byte, one power-of-two scale per 32 inputs. Returns (packed int8, scale, decoded)."""
    rows, cols = weight.shape
    table = torch.tensor(giant.FP4_TABLE)
    tiles = weight.reshape(rows, cols // block, block)
    top = tiles.abs().amax(dim=-1).clamp(min=1e-8) / 6.0
    scale = 2.0 ** torch.ceil(torch.log2(top))
    scaled = tiles / scale.unsqueeze(-1)
    codes = (scaled.reshape(-1, 1) - table.reshape(1, -1)).abs().argmin(dim=1).reshape(rows, cols)
    decoded = (table[codes].reshape(rows, cols // block, block) * scale.unsqueeze(-1)).reshape(rows, cols)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed.view(torch.int8), scale.to(torch.float8_e8m0fnu), decoded


def test_block_fp8_dequant_with_partial_tiles():
    torch.manual_seed(0)
    weight = torch.randn(200, 300) * 0.3
    stored, scale, decoded = quantize_fp8(weight)
    assert scale.shape == (2, 3)
    got = giant.dequant_block_fp8(stored, scale, 128, torch.float32)
    assert torch.equal(got, decoded)
    assert (got - weight).abs().max() < 0.1
    with pytest.raises(ValueError):
        giant.dequant_block_fp8(stored, scale[:1], 128, torch.float32)


def test_fp4_pack_unpack_and_dequant_roundtrip():
    torch.manual_seed(0)
    weight = torch.randn(6, 64)
    packed, scale, decoded = quantize_fp4(weight)
    assert packed.shape == (6, 32)
    assert scale.shape == (6, 2)
    assert torch.equal(giant.dequant_fp4(packed, scale, 32, torch.float32), decoded)
    values = giant.unpack_fp4(packed)
    assert torch.equal(giant.pack_fp4(values).view(torch.int8), packed)


def test_fp4_low_nibble_is_the_even_column():
    byte = torch.tensor([[0x21]], dtype=torch.uint8)
    assert giant.unpack_fp4(byte).tolist() == [[0.5, 1.0]]
    negative = torch.tensor([[0x9F]], dtype=torch.uint8)
    assert giant.unpack_fp4(negative).tolist() == [[-6.0, -0.5]]


def test_uint8_scale_is_an_exponent_with_bias_127():
    assert giant.scale_to_float(torch.tensor([127, 128, 126], dtype=torch.uint8)).tolist() == [1.0, 2.0, 0.5]


# ---------------------------------------------------------------------------
# Layout names, on real key names of the giants.


def test_deepseek_v4_names_translate_to_module_keys():
    layout = giant.DeepSeekV4Layout()
    cases = {
        "layers.5.attn.wq_a.weight": "self_attn.q_a_proj.weight",
        "layers.5.attn.wq_b.scale": "self_attn.q_b_proj.scale",
        "layers.5.attn.wkv.weight": "self_attn.kv_proj.weight",
        "layers.5.attn.wo_a.weight": "self_attn.o_a_proj.weight",
        "layers.5.attn.wo_b.weight": "self_attn.o_b_proj.weight",
        "layers.5.attn.attn_sink": "self_attn.sinks",
        "layers.5.attn.q_norm.weight": "self_attn.q_a_norm.weight",
        "layers.5.attn.kv_norm.weight": "self_attn.kv_norm.weight",
        "layers.5.attn.compressor.ape": "self_attn.compressor.position_bias",
        "layers.5.attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
        "layers.5.attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
        "layers.5.attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
        "layers.5.attn_norm.weight": "input_layernorm.weight",
        "layers.5.ffn_norm.weight": "post_attention_layernorm.weight",
        "layers.5.ffn.gate.weight": "mlp.gate.weight",
        "layers.5.ffn.gate.bias": "mlp.gate.e_score_correction_bias",
        "layers.0.ffn.gate.tid2eid": "mlp.gate.tid2eid",
        "layers.5.ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
        "layers.5.ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
        "layers.5.ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
        "layers.5.hc_attn_fn": "attn_hc.fn",
        "layers.5.hc_ffn_scale": "ffn_hc.scale",
    }
    for name in cases:
        layer = int(name.split(".")[1])
        assert layout.translate(name, layer) == cases[name], name


def test_glm_and_kimi_names_translate_to_module_keys():
    glm = giant.GlmLayout()
    base = "model.language_model.layers.5."
    assert glm.translate(base + "self_attn.f_a_proj.weight", 5) == "self_attn.forget_gate.f_a_proj.weight"
    assert glm.translate(base + "self_attn.A_log", 5) == "self_attn.forget_gate.A_log"
    assert glm.translate(base + "hc_attn_base", 5) == "attn_hc.base"
    assert glm.translate(base + "mlp.shared_experts.gate_proj.weight", 5) == "mlp.shared_experts.gate_proj.weight"
    kimi = giant.KimiLayout()
    base = "language_model.model.layers.5."
    assert kimi.translate(base + "block_sparse_moe.experts.3.w1.weight_packed", 5) == \
        "block_sparse_moe.experts.3.w1.weight"
    assert kimi.translate(base + "mlp_res_proj.weight", 5) == "mlp_res_proj.weight"


def test_layout_for_picks_the_family_and_the_plain_prefix():
    class Cfg:
        def __init__(self, model_type):
            self.model_type = model_type

    class Plain:
        def weight_map(self):
            return {"model.language_model.layers.0.mlp.gate.weight": "a", "model.visual.layers.0.x": "a",
                    "model.language_model.embed_tokens.weight": "a"}

    assert isinstance(giant.layout_for(Cfg("deepseek_v4"), Plain()), giant.DeepSeekV4Layout)
    assert isinstance(giant.layout_for(Cfg("kimi_k31"), Plain()), giant.KimiLayout)
    plain = giant.layout_for(Cfg("qwen3_5_moe_text"), Plain())
    assert plain.prefix(7) == "model.language_model.layers.7."
    assert plain.embed_name() == "model.language_model.embed_tokens.weight"


# ---------------------------------------------------------------------------
# Toy giants: fused experts with FP8 and FP4, per-expert mxfp4, two-state layers, hash routing.


class ToyConfig(transformers.PretrainedConfig):
    model_type = "toy_giant"

    def __init__(self, hidden_size=32, inter=32, vocab_size=40, num_hidden_layers=3, num_experts=4, moe_top_k=2,
                 style="fused", hash_layers=(), **kwargs):
        self.hidden_size = hidden_size
        self.inter = inter
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.moe_top_k = moe_top_k
        self.style = style
        self.hash_layers = list(hash_layers)
        super().__init__(**kwargs)


class ToyNestedConfig(ToyConfig):
    model_type = "toy_giant_nested"


class ToyGate(torch.nn.Module):
    def __init__(self, config, layer_index, kimi_output):
        super().__init__()
        self.top_k = config.moe_top_k
        self.kimi_output = kimi_output
        self.weight = torch.nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        self.e_score_correction_bias = torch.nn.Parameter(torch.zeros(config.num_experts))
        self.is_hash = layer_index in config.hash_layers
        if self.is_hash:
            self.register_buffer("tid2eid", torch.zeros(config.vocab_size, config.moe_top_k, dtype=torch.long))

    def forward(self, hidden, input_ids=None):
        flat = hidden.reshape(-1, hidden.shape[-1])
        logits = flat @ self.weight.t()
        scores = logits.sigmoid()
        if self.is_hash:
            indices = self.tid2eid[input_ids.reshape(-1)]
        else:
            indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / weights.sum(-1, keepdim=True)
        if self.kimi_output:
            return indices, weights
        return logits, weights, indices


class ToyFusedExperts(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.gate_up_proj = torch.nn.Parameter(torch.empty(config.num_experts, 2 * config.inter, config.hidden_size))
        self.down_proj = torch.nn.Parameter(torch.empty(config.num_experts, config.hidden_size, config.inter))

    def forward(self, flat, indices, weights):
        out = torch.zeros_like(flat)
        for expert in range(self.num_experts):
            rows, slots = torch.where(indices == expert)
            if rows.numel() == 0:
                continue
            gate, up = (flat[rows] @ self.gate_up_proj[expert].t()).chunk(2, dim=-1)
            value = (torch.nn.functional.silu(gate) * up) @ self.down_proj[expert].t()
            out.index_add_(0, rows, value * weights[rows, slots, None])
        return out


class ToyExpert(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = torch.nn.Linear(config.hidden_size, config.inter, bias=False)
        self.w2 = torch.nn.Linear(config.inter, config.hidden_size, bias=False)
        self.w3 = torch.nn.Linear(config.hidden_size, config.inter, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class ToyShared(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.inter, bias=False)
        self.up_proj = torch.nn.Linear(config.hidden_size, config.inter, bias=False)
        self.down_proj = torch.nn.Linear(config.inter, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyMoe(torch.nn.Module):
    def __init__(self, config, layer_index, kimi):
        super().__init__()
        self.kimi = kimi
        self.top_k = config.moe_top_k
        self.gate = ToyGate(config, layer_index, kimi)
        if kimi:
            self.experts = torch.nn.ModuleList([ToyExpert(config) for _ in range(config.num_experts)])
        else:
            self.experts = ToyFusedExperts(config)
        self.shared_experts = ToyShared(config)

    def forward(self, hidden, input_ids=None):
        flat = hidden.reshape(-1, hidden.shape[-1])
        routed = self.gate(hidden, input_ids)
        if self.kimi:
            indices, weights = routed
            out = torch.zeros_like(flat)
            for expert in range(len(self.experts)):
                rows, slots = torch.where(indices == expert)
                if rows.numel() == 0:
                    continue
                out.index_add_(0, rows, self.experts[expert](flat[rows]) * weights[rows, slots, None])
        else:
            _, weights, indices = routed
            out = self.experts(flat, indices, weights)
        return out.reshape(hidden.shape) + self.shared_experts(hidden)


class ToyConv(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(size, 1))


class ToyAttn(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_a_proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.conv1d = ToyConv(3 * config.hidden_size)
        self.sinks = torch.nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, x):
        scale = self.conv1d.weight.reshape(3, -1).mean(0)
        return self.q_a_proj(x) * (1.0 + 0.1 * torch.tanh(scale)) + self.sinks


class ToyLayer(torch.nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        self.layer_idx = layer_index
        self.flag = layer_index % 2 == 0
        self.two_state = config.style == "kimi"
        self.input_layernorm = torch.nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = torch.nn.LayerNorm(config.hidden_size)
        self.self_attn = ToyAttn(config)
        if self.two_state:
            self.block_sparse_moe = ToyMoe(config, layer_index, True)
        else:
            self.mlp = ToyMoe(config, layer_index, False)

    def forward(self, hidden_states, input_ids=None, block_residual=None, big=True, **kwargs):
        hidden = hidden_states * (1.0 if big else 0.9)
        if self.two_state:
            if block_residual is not None and block_residual.shape[2] > 0:
                hidden = hidden + 0.1 * block_residual.mean(dim=2)
            if self.layer_idx % 2 == 0:
                new = hidden.unsqueeze(2)
                block_residual = new if block_residual is None or block_residual.shape[2] == 0 else \
                    torch.cat([block_residual, new], dim=2)
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
        normed = self.post_attention_layernorm(hidden)
        if self.two_state:
            hidden = hidden + self.block_sparse_moe(normed)
            return hidden, block_residual
        return hidden + self.mlp(normed, input_ids=input_ids)


class ToyBase(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = torch.nn.ModuleList([ToyLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = torch.nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids=None, use_cache=None, **kwargs):
        hidden = self.embed_tokens(input_ids)
        block = hidden.new_zeros(hidden.shape[0], hidden.shape[1], 0, hidden.shape[2])
        for layer in self.layers:
            if layer.two_state:
                hidden, block = layer(hidden, block_residual=block, big=layer.flag)
            else:
                hidden = layer(hidden, input_ids=input_ids, big=layer.flag)
        return self.norm(hidden)


class ToyCausal(PreTrainedModel):
    config_class = ToyConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = ToyBase(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids=None, use_cache=None, **kwargs):
        return self.lm_head(self.model(input_ids=input_ids))


class ToyOuter(PreTrainedModel):
    config_class = ToyNestedConfig

    def __init__(self, config):
        super().__init__(config)
        self.language_model = ToyCausal(config)

    def forward(self, input_ids=None, use_cache=None, **kwargs):
        return self.language_model(input_ids=input_ids)


for _config, _model in [(ToyConfig, ToyCausal), (ToyNestedConfig, ToyOuter)]:
    try:
        transformers.AutoConfig.register(_config.model_type, _config)
        transformers.AutoModelForCausalLM.register(_config, _model)
    except ValueError:
        pass


def toy_weights(model, seed):
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        with torch.no_grad():
            if param.dim() >= 2:
                param.copy_(torch.randn_like(param) * 0.3)
            elif "bias" in name:
                param.copy_(torch.randn_like(param) * 0.05)
            else:
                param.copy_(torch.randn_like(param) * 0.1)
    for name, buffer in model.named_buffers():
        if name.endswith("tid2eid"):
            buffer.copy_(torch.randint(0, 4, buffer.shape))


def collect_checkpoint(model, style):
    """Quantize the toy weights in place and return {checkpoint name: tensor} in the family naming."""
    config = model.config
    base = model.language_model.model if style == "kimi" else model.model
    out = {}
    for index, layer in enumerate(base.layers):
        if style == "v4":
            prefix = f"layers.{index}."
            names = {"attn": prefix + "attn.", "ffn": prefix + "ffn.", "an": prefix + "attn_norm.",
                     "fn": prefix + "ffn_norm."}
        elif style == "glm":
            prefix = f"model.language_model.layers.{index}."
            names = {"attn": prefix + "self_attn.", "ffn": prefix + "mlp.", "an": prefix + "input_layernorm.",
                     "fn": prefix + "post_attention_layernorm."}
        else:
            prefix = f"language_model.model.layers.{index}."
            names = {"attn": prefix + "self_attn.", "ffn": prefix + "block_sparse_moe.",
                     "an": prefix + "input_layernorm.", "fn": prefix + "post_attention_layernorm."}
        for key, value in layer.input_layernorm.state_dict().items():
            out[names["an"] + key] = value.clone()
        for key, value in layer.post_attention_layernorm.state_dict().items():
            out[names["fn"] + key] = value.clone()
        attn = layer.self_attn
        with torch.no_grad():
            if style == "kimi":
                out[names["attn"] + "q_a_proj.weight"] = attn.q_a_proj.weight.clone()
                out[names["attn"] + "conv1d.weight"] = attn.conv1d.weight.detach().clone()
                out[names["attn"] + "sinks"] = attn.sinks.clone()
            else:
                stored, scale, decoded = quantize_fp8(attn.q_a_proj.weight)
                attn.q_a_proj.weight.copy_(decoded)
                if style == "v4":
                    out[names["attn"] + "wq_a.weight"] = stored
                    out[names["attn"] + "wq_a.scale"] = scale
                    out[names["attn"] + "attn_sink"] = attn.sinks.clone()
                    out[names["attn"] + "conv1d.weight"] = attn.conv1d.weight.detach().clone()
                else:
                    out[names["attn"] + "q_a_proj.weight"] = stored
                    out[names["attn"] + "q_a_proj.weight_scale_inv"] = scale.float()
                    out[names["attn"] + "sinks"] = attn.sinks.clone()
                    third = attn.conv1d.weight.shape[0] // 3
                    for position, letter in enumerate("qkv"):
                        out[names["attn"] + f"{letter}_conv1d.weight"] = attn.conv1d.weight[position * third:(position + 1) * third].detach().clone()
        moe = layer.block_sparse_moe if style == "kimi" else layer.mlp
        gate_names = {"v4": ("gate.weight", "gate.bias", "gate.tid2eid"),
                      "glm": ("gate.weight", "gate.e_score_correction_bias", "gate.tid2eid"),
                      "kimi": ("gate.weight", "gate.e_score_correction_bias", "gate.tid2eid")}[style]
        out[names["ffn"] + gate_names[0]] = moe.gate.weight.detach().clone()
        out[names["ffn"] + gate_names[1]] = moe.gate.e_score_correction_bias.detach().clone()
        if moe.gate.is_hash:
            out[names["ffn"] + gate_names[2]] = moe.gate.tid2eid.clone()
        shared = moe.shared_experts
        for part, v4_name in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
            weight = getattr(shared, part).weight
            if style == "kimi":
                out[names["ffn"] + f"shared_experts.{part}.weight"] = weight.detach().clone()
                continue
            stored, scale, decoded = quantize_fp8(weight)
            with torch.no_grad():
                weight.copy_(decoded)
            if style == "v4":
                out[names["ffn"] + f"shared_experts.{v4_name}.weight"] = stored
                out[names["ffn"] + f"shared_experts.{v4_name}.scale"] = scale
            else:
                out[names["ffn"] + f"shared_experts.{part}.weight"] = stored
                out[names["ffn"] + f"shared_experts.{part}.weight_scale_inv"] = scale.float()
        for expert in range(config.num_experts):
            if style == "kimi":
                for part in ["w1", "w2", "w3"]:
                    weight = getattr(moe.experts[expert], part).weight
                    packed, scale, decoded = quantize_fp4(weight)
                    with torch.no_grad():
                        weight.copy_(decoded)
                    stem = names["ffn"] + f"experts.{expert}.{part}."
                    out[stem + "weight_packed"] = packed.view(torch.uint8)
                    out[stem + "weight_scale"] = scale.view(torch.uint8)
                continue
            inter = config.inter
            for part, rows, v4_name, glm_name in [("gate", slice(0, inter), "w1", "gate_proj"),
                                                  ("up", slice(inter, 2 * inter), "w3", "up_proj")]:
                weight = moe.experts.gate_up_proj[expert, rows]
                if style == "v4":
                    packed, scale, decoded = quantize_fp4(weight.detach())
                    stem = names["ffn"] + f"experts.{expert}.{v4_name}."
                    out[stem + "weight"] = packed
                    out[stem + "scale"] = scale
                else:
                    packed, scale, decoded = quantize_fp8(weight.detach())
                    stem = names["ffn"] + f"experts.{expert}.{glm_name}."
                    out[stem + "weight"] = packed
                    out[stem + "weight_scale_inv"] = scale.float()
                with torch.no_grad():
                    moe.experts.gate_up_proj[expert, rows].copy_(decoded)
            weight = moe.experts.down_proj[expert]
            if style == "v4":
                packed, scale, decoded = quantize_fp4(weight.detach())
                stem = names["ffn"] + f"experts.{expert}.w2."
                out[stem + "weight"] = packed
                out[stem + "scale"] = scale
            else:
                packed, scale, decoded = quantize_fp8(weight.detach())
                stem = names["ffn"] + f"experts.{expert}.down_proj."
                out[stem + "weight"] = packed
                out[stem + "weight_scale_inv"] = scale.float()
            with torch.no_grad():
                moe.experts.down_proj[expert].copy_(decoded)
    embed = base.embed_tokens.weight.detach().clone()
    out[{"v4": "embed.weight", "glm": "model.language_model.embed_tokens.weight",
         "kimi": "language_model.model.embed_tokens.weight"}[style]] = embed
    return out


LAYOUT_OF = {"v4": giant.DeepSeekV4Layout, "glm": giant.GlmLayout, "kimi": giant.KimiLayout}


def toy_giant(tmp_path, style, layers=3, hash_layers=()):
    config_class = ToyNestedConfig if style == "kimi" else ToyConfig
    config = config_class(style="kimi" if style == "kimi" else "fused", num_hidden_layers=layers,
                          hash_layers=list(hash_layers))
    model = (ToyOuter if style == "kimi" else ToyCausal)(config)
    toy_weights(model, seed=3)
    tensors = collect_checkpoint(model, style)
    folder = tmp_path / style
    folder.mkdir()
    half = len(tensors) // 2
    names = sorted(tensors)
    shards = {"model-1.safetensors": names[:half], "model-2.safetensors": names[half:]}
    weight_map = {}
    for shard in shards:
        save_file({name: tensors[name].contiguous() for name in shards[shard]}, str(folder / shard))
        for name in shards[shard]:
            weight_map[name] = shard
    with open(folder / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return model.eval(), str(folder), config


@pytest.mark.parametrize("style", ["v4", "glm", "kimi"])
def test_quantized_toy_giant_matches_the_full_model(tmp_path, style):
    tok = tokenizer()
    hash_layers = [0] if style == "v4" else []
    model, folder, config = toy_giant(tmp_path, style, hash_layers=hash_layers)
    source = giant.LocalSource(folder)
    layout = LAYOUT_OF[style]()
    expected = atlas.expert_skill_map(model, tok, SKILL, GENERAL, top=2)
    result = giant.profile_streaming(source, config, tok, {"a": SKILL}, GENERAL, layout=layout, top=2,
                                     prefetch=False, log=lambda text: None, keep_state=True, max_layers=3)
    assert same_map(expected, result["maps"]["a"]), style
    assert result["info"]["layout"] == layout.family
    assert sorted(expected["layers"]) == [0, 1, 2]


@pytest.mark.parametrize("style", ["v4", "kimi"])
def test_quantized_toy_giant_hidden_states_match(tmp_path, style):
    tok = tokenizer()
    model, folder, config = toy_giant(tmp_path, style, hash_layers=[0] if style == "v4" else [])
    source = giant.LocalSource(folder)
    result = giant.profile_streaming(source, config, tok, {"a": SKILL}, GENERAL, layout=LAYOUT_OF[style](), top=2,
                                     prefetch=True, log=lambda text: None, keep_state=True, max_layers=2)
    items = atlas.encode_texts(tok, GENERAL, 256) + atlas.encode_texts(tok, SKILL, 256)
    base = model.language_model.model if style == "kimi" else model.model
    for ids, state in zip(items, result["info"]["state"]):
        hidden = base.embed_tokens(ids.unsqueeze(0))
        block = hidden.new_zeros(1, hidden.shape[1], 0, hidden.shape[2])
        with torch.no_grad():
            for layer in list(base.layers)[:2]:
                if layer.two_state:
                    hidden, block = layer(hidden, block_residual=block, big=layer.flag)
                else:
                    hidden = layer(hidden, input_ids=ids.unsqueeze(0), big=layer.flag)
        if style == "kimi":
            assert torch.allclose(state[0], hidden, atol=1e-6)
            assert torch.allclose(state[1], block, atol=1e-6)
        else:
            assert torch.allclose(state, hidden, atol=1e-6)


@pytest.mark.parametrize("style", ["v4", "glm", "kimi"])
def test_fetch_expert_decodes_stored_quantization(tmp_path, style):
    model, folder, config = toy_giant(tmp_path, style)
    source = giant.LocalSource(folder)
    layout = LAYOUT_OF[style]()
    result = giant.fetch_expert(source, 1, [3, 0], layout=layout, dtype=torch.float32)
    base = model.language_model.model if style == "kimi" else model.model
    moe = base.layers[1].block_sparse_moe if style == "kimi" else base.layers[1].mlp
    for position, expert in enumerate([3, 0]):
        if style == "kimi":
            gate = moe.experts[expert].w1.weight
            up = moe.experts[expert].w3.weight
            down = moe.experts[expert].w2.weight
        else:
            gate = moe.experts.gate_up_proj[expert, :config.inter]
            up = moe.experts.gate_up_proj[expert, config.inter:]
            down = moe.experts.down_proj[expert]
        assert torch.equal(result["gate"][position], gate.detach())
        assert torch.equal(result["up"][position], up.detach())
        assert torch.equal(result["down"][position], down.detach())
    assert torch.equal(result["router_rows"], moe.gate.weight.detach()[[3, 0]])
    raw = giant.fetch_expert(source, 1, [3], layout=layout, decode=False)
    assert raw["gate"][0].dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn)
    assert raw["scales"]["gate"][0] is not None


@pytest.mark.parametrize("style", ["v4", "kimi"])
def test_stored_experts_cost_less_than_decoded_ones(tmp_path, style):
    model, folder, config = toy_giant(tmp_path, style)
    source = giant.LocalSource(folder)
    layout = LAYOUT_OF[style]()
    source.gap = 0
    predicted = giant.expert_download_bytes(source, layout, {1: [0, 1]})
    result = giant.fetch_expert(source, 1, [0, 1], layout=layout, dtype=torch.float32)
    assert predicted["total"] == result["bytes"]
    decoded_bytes = 2 * 3 * config.inter * config.hidden_size * 4
    assert predicted["total"] < decoded_bytes


def test_stream_bytes_counts_every_tensor_of_the_chosen_layers(tmp_path):
    model, folder, config = toy_giant(tmp_path, "glm")
    source = giant.LocalSource(folder)
    layout = giant.GlmLayout()
    both = giant.stream_bytes(source, layout, 3)
    one = giant.stream_bytes(source, layout, 3, layers=[1])
    assert one["total"] < both["total"]
    assert both["total"] == both["embedding"] + sum(both["per_layer"].values())
    source.gap = 0
    tokens_before = source.bytes_read
    source.read(giant.layer_tensor_names(layout, source, 1))
    assert source.bytes_read - tokens_before == one["per_layer"][1]


def test_estimate_hours_scales_with_bytes():
    download, compute, total = giant.estimate_hours(400e9 * 3600, 10, mb_per_second=400.0, seconds_per_layer=360.0)
    assert download == pytest.approx(1000.0)
    assert compute == pytest.approx(1.0)
    assert total == pytest.approx(1001.0)


# ---------------------------------------------------------------------------
# modal_giant.py selection, without Modal.


def test_modal_giant_selects_only_supported_streaming_models():
    import modal_giant

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zoo = atlas.load_zoo(os.path.join(root, "atlas", "zoo.json"))
    chosen = modal_giant.giant_models(zoo, "")
    ids = [model["id"] for model in chosen]
    assert "deepseek-ai/DeepSeek-V4-Flash" in ids
    assert "zai-org/GLM-5.3-Flash" in ids
    assert "Qwen/Qwen3.8-2.4T-A95B" in ids
    assert "moonshotai/Kimi-K3.1-Pro" not in ids
    only = modal_giant.giant_models(zoo, "deepseek-ai__DeepSeek-V4-Pro,zai-org/GLM-5.3-Flash")
    assert sorted(model["id"] for model in only) == ["deepseek-ai/DeepSeek-V4-Pro", "zai-org/GLM-5.3-Flash"]
    seen = []
    for shard in range(3):
        seen.extend(model["id"] for model in modal_giant.shard_cells(chosen, shard, 3))
    assert sorted(seen) == sorted(ids)
    with pytest.raises(ValueError):
        modal_giant.shard_cells(chosen, 3, 3)


def test_zoo_stream_layouts_are_known():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zoo = atlas.load_zoo(os.path.join(root, "atlas", "zoo.json"))
    for model in zoo["models"]:
        stream = model.get("stream")
        if stream is None:
            continue
        assert stream["layout"] in ("deepseek_v4", "glm5_next", "kimi", "hf")
        assert model["moe"]
        assert stream["api_total_gb"] > 100


# ---------------------------------------------------------------------------
# Regressions found in review.


class MemorySource(giant.Source):
    """One shard held in memory, with a log of the range requests."""

    def __init__(self, tensors, cut=0):
        giant.Source.__init__(self)
        import io

        blob = io.BytesIO()
        from safetensors.torch import save as save_bytes

        self.blob = save_bytes(tensors)
        self.names = list(tensors)
        self.log = []
        self.cut = cut

    def weight_map(self):
        return {name: "m.safetensors" for name in self.names}

    def range_reader(self, shard):
        def read(offset, length):
            self.log.append(length)
            return self.blob[offset:offset + length - self.cut]

        return read


def test_long_read_is_split_into_requests_and_stays_exact():
    table = torch.randn(64, 32)
    source = MemorySource({"w": table, "v": torch.arange(5, dtype=torch.int64)})
    source.header("m.safetensors")
    source.log.clear()
    source.max_request = 1000
    got = source.read(["w"])["w"]
    assert torch.equal(got, table)
    assert len(source.log) == 9
    assert max(source.log) <= 1000
    assert source.bytes_read == table.numel() * 4


def test_short_range_answer_is_an_error():
    source = MemorySource({"w": torch.randn(8, 8)})
    source.header("m.safetensors")
    source.cut = 4
    with pytest.raises(RuntimeError):
        source.read(["w"])


def test_unaligned_neighbours_decode_exactly():
    tensors = {"a": torch.arange(3, dtype=torch.uint8), "b": torch.randn(5, 3), "c": torch.arange(7, dtype=torch.int8),
               "d": torch.randn(4).to(torch.bfloat16)}
    source = MemorySource(tensors)
    got = source.read(list(tensors))
    for name in tensors:
        assert torch.equal(got[name], tensors[name])


def test_resume_ignores_state_of_other_weights(tmp_path):
    tok = tokenizer()
    model, folder = saved(tmp_path / "one", "qwen3_moe")
    state_dir = str(tmp_path / "state")
    stream(folder, model, tok, max_layers=2, state_dir=state_dir, checkpoint_every=1)
    other = tiny_model("qwen3_moe")
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    other_folder = str(tmp_path / "two")
    other.save_pretrained(other_folder)
    expected = reference_maps(other, tok)
    result = stream(other_folder, other, tok, state_dir=state_dir, checkpoint_every=1)
    assert len(result["info"]["layer_seconds"]) == 3
    for skill in expected:
        assert same_map(expected[skill], result["maps"][skill])


def test_resumed_states_are_moved_to_the_layer_device(tmp_path, monkeypatch):
    tok = tokenizer()
    model, folder = saved(tmp_path, "qwen3_moe", layers=4)
    state_dir = str(tmp_path / "state")
    stream(folder, model, tok, max_layers=2, state_dir=state_dir, checkpoint_every=1)
    seen = []
    real = giant.to_device_state

    def spy(state, device):
        seen.append(device)
        return real(state, device)

    monkeypatch.setattr(giant, "to_device_state", spy)
    stream(folder, model, tok, state_dir=state_dir, checkpoint_every=1)
    assert len(seen) > 0 and set(seen) == {"cpu"}
    pair = (torch.zeros(2), torch.ones(3))
    moved = giant.to_device_state(pair, "meta")
    assert isinstance(moved, tuple) and all(part.device.type == "meta" for part in moved)
    assert giant.to_device_state(None, "meta") is None


def test_layer_name_cache_does_not_confuse_sources(tmp_path):
    save_file({"model.layers.0.w": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    first = giant.LocalSource(str(tmp_path))
    other = tmp_path / "b"
    other.mkdir()
    save_file({"model.layers.0.x": torch.zeros(2)}, str(other / "model.safetensors"))
    second = giant.LocalSource(str(other))
    layout = giant.Layout()
    assert layout.layer_names(second, 0) == ["model.layers.0.x"]
    # Another source that gets the id of a freed one must not see its names.
    layout._by_layer[id(first)] = (second, {0: ["model.layers.0.x"]})
    assert layout.layer_names(first, 0) != ["model.layers.0.x"]


def test_glm_layout_lists_the_keys_it_combines():
    state = {"self_attn.q_conv1d.weight": torch.zeros(2, 1, 4), "self_attn.k_conv1d.weight": torch.zeros(2, 1, 4),
             "self_attn.v_conv1d.weight": torch.zeros(2, 1, 4)}
    layout = giant.GlmLayout()
    assert set(layout.combine_keys) == set(state)
    assert list(layout.combine(state)) == ["self_attn.conv1d.weight"]


def test_layout_from_name_covers_every_zoo_layout(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zoo = atlas.load_zoo(os.path.join(root, "atlas", "zoo.json"))
    model, folder = saved(tmp_path, "mixtral")
    source = giant.LocalSource(folder)
    for entry in zoo["models"]:
        stream_info = entry.get("stream")
        if stream_info is not None:
            assert giant.layout_from_name(stream_info["layout"], source).family
    assert giant.layout_from_name("kimi", source).family == "kimi_k3"
    with pytest.raises(ValueError):
        giant.layout_from_name("nope", source)


def test_estimate_runs_without_a_transformers_config(tmp_path, monkeypatch):
    import modal_giant

    def refuse(*args, **kwargs):
        raise AssertionError("estimate must not load a transformers config")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", refuse)
    model, folder = saved(tmp_path, "mixtral")
    source = giant.LocalSource(folder)
    row = {"id": "tiny/mixtral", "stream": {"layout": "hf"}}
    got = modal_giant.estimate(row, 300.0, 30.0, source=source, layer_count=3)
    layout = giant.layout_from_name("hf", source)
    sizes = giant.stream_bytes(source, layout, 3)
    assert got["download_gb"] == round(sizes["total"] / 1e9, 1)
    assert got["layers"] == 3
    limited = modal_giant.estimate(row, 300.0, 30.0, max_layers=1, source=source, layer_count=3)
    assert limited["layers"] == 1
    assert limited["compute_hours"] < got["compute_hours"]


def zoo_row():
    return {"id": "org/giant", "key": "org__giant", "kind": "reasoning", "maker": "x", "skills": ["general"],
            "params_b": 1.0, "layers": 4, "hidden": 8, "family": "f"}


def test_smoke_run_keeps_the_layers_of_a_finished_entry():
    import modal_giant

    done = {"layers_done": 4, "layers_total": 4, "texts": 10, "bytes_read": 100}
    full = modal_giant.merge_expert_maps({}, {"math": {"0": [1], "1": [2], "2": [3], "3": [4]}}, done, zoo_row())
    assert "partial" not in full
    smoke = {"layers_done": 1, "layers_total": 4, "texts": 2, "bytes_read": 5}
    again = modal_giant.merge_expert_maps(full, {"math": {"0": [9]}}, smoke, zoo_row())
    assert again["expert_maps"]["math"] == {"0": [9], "1": [2], "2": [3], "3": [4]}
    assert "partial" not in again
    assert again["streamed"] == done


def test_partial_run_marks_the_entry_partial_and_a_full_run_clears_it():
    import modal_giant

    smoke = {"layers_done": 1, "layers_total": 4, "texts": 2, "bytes_read": 5}
    first = modal_giant.merge_expert_maps({}, {"math": {"0": [9]}}, smoke, zoo_row())
    assert first["partial"] is True
    second = modal_giant.merge_expert_maps(first, {"code": {"0": [1]}}, smoke, zoo_row())
    assert second["partial"] is True
    assert sorted(second["expert_maps"]) == ["code", "math"]
    done = {"layers_done": 4, "layers_total": 4, "texts": 10, "bytes_read": 100}
    last = modal_giant.merge_expert_maps(second, {"math": {"0": [1], "1": [2], "2": [3], "3": [4]}}, done, zoo_row())
    assert "partial" not in last
    assert sorted(last["expert_maps"]["math"]) == ["0", "1", "2", "3"]
