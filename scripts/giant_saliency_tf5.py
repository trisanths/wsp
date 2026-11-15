"""Check the saliency pass of wsp.giant on the real transformers 5.x MoE blocks. CPU, tiny random weights.

tests/test_giant.py runs on transformers 4.57.1, where DeepSeek-V4, GLM-5.3 and Qwen3.5-MoE do not
exist. This script builds one MoE block of each of those classes, fills it with random weights, and
compares giant.SaliencyMeter with a reference that takes one token and one top-k slot at a time from
the block's own parameters and its router output. It also checks that the block's output does not
change while the meter is on. Every experts implementation that transformers registers is tried.

The meter checks itself on the first call of a block: weight times the expert outputs it computed
again, summed over the top-k slots, must be the output of the experts module (check_fused). The
script prints how far the two are apart, in float32 and again with the block in bfloat16, the dtype
of the H200 run. In bfloat16 only that self-check is judged, since the reference is float32 math.

Run it from the repo root with a Python that has transformers 5.23.0, for example in the image of
modal_giant.py. It needs no network and no GPU.

    python scripts/giant_saliency_tf5.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import transformers

from wsp import giant

TOKENS = 11
HIDDEN = 32
EXPERTS = 8
TOP_K = 3


def fill(block, seed):
    """Random weights, a random hash-routing table, and a non-zero router bias."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)
        for name, buffer in block.named_buffers():
            if name.endswith("tid2eid"):
                buffer.copy_(torch.randint(0, EXPERTS, buffer.shape))
            elif buffer.is_floating_point():
                buffer.copy_(torch.randn_like(buffer) * 0.1)


def reference(block, router_output, flat):
    """Saliency per expert from the definition. The router returns (logits, weights, indices)."""
    _, weights, indices = router_output
    experts = block.experts
    out = torch.zeros(EXPERTS, dtype=torch.float64)
    for token in range(flat.shape[0]):
        for slot in range(indices.shape[1]):
            expert = int(indices[token, slot])
            inner = F.linear(flat[token:token + 1], experts.gate_up_proj[expert])
            if hasattr(experts, "_apply_gate"):
                inner = experts._apply_gate(inner)
            else:
                gate, up = inner.chunk(2, dim=-1)
                inner = experts.act_fn(gate) * up
            value = F.linear(inner, experts.down_proj[expert])
            out[expert] += float(weights[token, slot]) * float(value.float().norm())
    return out


def check(name, block, layout, call, hidden, config):
    """Compare the meter with the reference on one block. Returns the largest relative difference."""
    seen = {}
    handle = block.gate.register_forward_hook(lambda module, args, output: seen.update(router=output))
    with torch.no_grad():
        plain = call(block)
        handle.remove()
        want = reference(block, seen["router"], hidden.reshape(-1, HIDDEN))
        found = giant.find_layer_router(torch.nn.ModuleDict({"mlp": block}), config)
        if found is None or found[2] != TOP_K or found[3] != EXPERTS:
            raise AssertionError(f"{name}: find_layer_router gave {found}")
        meter = giant.SaliencyMeter(0, None, block, block.gate, TOP_K, EXPERTS, ["general"], layout)
        meter.group = "general"
        measured = call(block)
        meter.close()
    if meter.method != "fused" or not torch.equal(plain, measured) or meter.tokens["general"] != TOKENS:
        raise AssertionError(f"{name}: method {meter.method}, tokens {meter.tokens}")
    if len(block.experts._forward_hooks) != 0:
        raise AssertionError(f"{name}: the meter left a hook")
    got = meter.sums["general"]
    picked = want > 0
    worst = ((got - want).abs()[picked] / want[picked]).max().item()
    if int(picked.sum()) < 2 or worst > 1e-5 or float(got[~picked].abs().sum()) != 0.0:
        raise AssertionError(f"{name}: meter {got.tolist()}, reference {want.tolist()}")
    if len(meter.notes) != 1:
        raise AssertionError(f"{name}: the meter ran {len(meter.notes)} self-checks, expected 1")
    return worst, check_share(meter), narrow_share(block, layout, call)


def check_share(meter):
    """The number in the note of check_fused: the difference as a share of the largest output value."""
    return float(meter.notes[0].split(" by ")[1].split()[0])


def narrow_share(block, layout, call):
    """The self-check of the meter with the block in bfloat16. Returns its share, or the error as text."""
    block.to(torch.bfloat16)
    meter = giant.SaliencyMeter(0, None, block, block.gate, TOP_K, EXPERTS, ["general"], layout)
    meter.group = "general"
    try:
        with torch.no_grad():
            call(block, torch.bfloat16)
        return check_share(meter)
    except Exception as error:
        return repr(error)[:200]
    finally:
        meter.close()
        block.to(torch.float32)


def implementations():
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    return ["eager"] + [name for name in ["grouped_mm", "batched_mm"] if name in ALL_EXPERTS_FUNCTIONS]


def deepseek_v4():
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

    config = DeepseekV4Config(vocab_size=40, hidden_size=HIDDEN, moe_intermediate_size=16, num_hidden_layers=5,
                              num_attention_heads=4, head_dim=8, q_lora_rank=8, o_lora_rank=8, o_groups=2,
                              num_experts_per_tok=TOP_K, n_routed_experts=EXPERTS, index_n_heads=2,
                              index_head_dim=8, index_topk=4)
    rows = []
    # Layer 0 routes by the token-id table, layer 4 by the learned router.
    for layer_index in [0, 4]:
        for implementation in implementations():
            config._experts_implementation = implementation
            block = DeepseekV4SparseMoeBlock(config, layer_index).eval()
            fill(block, layer_index)
            hidden = torch.randn(1, TOKENS, HIDDEN)
            ids = torch.randint(0, 40, (1, TOKENS))

            def call(module, dtype=torch.float32, hidden=hidden, ids=ids):
                if module.is_hash:
                    return module(hidden.to(dtype), ids)
                return module(hidden.to(dtype))

            kind = "hash" if block.is_hash else "learned"
            name = f"DeepseekV4SparseMoeBlock ({kind} router, {implementation})"
            rows.append((name, check(name, block, giant.DeepSeekV4Layout(), call, hidden, config)))
    return rows


def glm5_next():
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMoE

    config = Glm5NextTextConfig(vocab_size=40, hidden_size=HIDDEN, intermediate_size=64, moe_intermediate_size=16,
                                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                                n_routed_experts=EXPERTS, num_experts_per_tok=TOP_K, kv_lora_rank=8, q_lora_rank=8,
                                v_head_dim=8, qk_nope_head_dim=8, pad_token_id=0)
    rows = []
    for implementation in implementations():
        config._experts_implementation = implementation
        block = Glm5NextTextMoE(config).eval()
        fill(block, 7)
        hidden = torch.randn(1, TOKENS, HIDDEN)
        name = f"Glm5NextTextMoE ({implementation})"
        rows.append((name, check(name, block, giant.GlmLayout(),
                                 lambda module, dtype=torch.float32, hidden=hidden: module(hidden.to(dtype)), hidden,
                                 config)))
    return rows


def qwen3_5_moe():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    config = Qwen3_5MoeTextConfig(vocab_size=40, hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=8, moe_intermediate_size=16,
                                  shared_expert_intermediate_size=16, num_experts_per_tok=TOP_K,
                                  num_experts=EXPERTS, linear_key_head_dim=8, linear_value_head_dim=8,
                                  linear_num_key_heads=2, linear_num_value_heads=4)
    block = Qwen3_5MoeSparseMoeBlock(config).eval()
    fill(block, 9)
    hidden = torch.randn(1, TOKENS, HIDDEN)
    name = "Qwen3_5MoeSparseMoeBlock"
    return [(name, check(name, block, giant.Layout(),
                         lambda module, dtype=torch.float32: module(hidden.to(dtype)), hidden, config))]


def main():
    print(f"transformers {transformers.__version__}, torch {torch.__version__}")
    if int(transformers.__version__.split(".")[0]) < 5:
        print("this check needs transformers 5.x")
        return 1
    status = 0
    for family in [deepseek_v4, glm5_next, qwen3_5_moe]:
        for name, (worst, share, narrow) in family():
            print(f"ok {name}: largest relative difference {worst:.1e}, self-check {share:.1e} of the largest output")
            if isinstance(narrow, float):
                print(f"   in bfloat16 the self-check passes at {narrow:.1e} (limit "
                      f"{giant.check_tolerance(torch.bfloat16):.0e})")
            else:
                print(f"   in bfloat16 the self-check did NOT pass: {narrow}")
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
