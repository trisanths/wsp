import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import clinic, pgraft
from wsp.lora import apply_lora
from tests.test_align import tiny_tokenizer
from tests.test_pgraft import TEXTS, logits, tiny_donor, tiny_host


def test_prompts_match_the_shared_format():
    row = {"question": "Which?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer_idx": "C"}
    prompt, gold = clinic.medqa_item(row)
    assert prompt == "Question: Which?\nA. a\nB. b\nC. c\nD. d\nAnswer:" and gold == 2
    assert clinic.medqa_item({"question": "x", "options": {"A": "a"}, "answer_idx": "A"}) is None
    pub = {"question": "Is it?", "context": {"contexts": ["one", "two"]}, "final_decision": "maybe"}
    prompt, gold = clinic.pubmedqa_item(pub)
    assert prompt.endswith("C. maybe\nAnswer:") and "one two" in prompt and gold == 2
    assert clinic.pubmedqa_item({"question": "q", "context": {"contexts": []}, "final_decision": "yes"}) is None
    med = {"question": "Q", "opa": "1", "opb": "2", "opc": "3", "opd": "4", "cop": 1}
    assert clinic.medmcqa_item(med)[1] == 1
    assert clinic.medmcqa_item(dict(med, cop=-1)) is None
    assert clinic.medmcqa_item(dict(med, opb=None)) is None


def test_filter_items_drops_banned_repeats_and_long_prompts():
    items = [("a", 0), ("b", 1), ("a", 2), ("c" * 50, 3), ("d", 0)]
    out = clinic.filter_items(items, banned={"b"}, max_chars=10)
    assert out == [["a", 0], ["d", 0]]


def test_mix_soft_mean_is_normalised_and_single_donor_passes_through():
    softs = {"q": [[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], "m": [[0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0]]}
    mixed = clinic.mix_soft(softs, "mean")
    assert mixed[0] == pytest.approx([0.25, 0.5, 0.25, 0.0]) and mixed[1] == pytest.approx([0.5, 0.5, 0, 0])
    assert clinic.mix_soft(softs, "m") == softs["m"]


def second_donor():
    torch.manual_seed(3)
    config = transformers.LlamaConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=6,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
                                      max_position_embeddings=128)
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def two_donor_host(dtype):
    """Host with a Qwen-type donor graft at layer 1 and a Llama-type donor graft (other width) at layer 2."""
    base = tiny_host().to(dtype)
    for p in base.parameters():
        p.requires_grad_(False)
    lora = apply_lora(base, 4)
    for p in lora:
        p.requires_grad_(True)
    tok = tiny_tokenizer()
    grafts = []
    for donor, layer in ((tiny_donor().to(dtype), 1), (second_donor().to(dtype), 2)):
        g = pgraft.build(base, donor, host_layer=layer, first=2, last=4, enc_depth=2, dec_rank=8)
        grafts.append(g)
    pgraft.attach(base, grafts)
    return base, tok, grafts, lora


def perturb(grafts, lora):
    torch.manual_seed(5)
    with torch.no_grad():
        for g in grafts:
            for p in list(g.w_out.parameters()) + list(g.token_gate.parameters()):
                p.add_(0.05 * torch.randn_like(p))
        for p in lora:
            p.add_(0.05 * torch.randn_like(p))


def test_two_donor_state_roundtrips_in_float32():
    base, tok, grafts, lora = two_donor_host(torch.float32)
    perturb(grafts, lora)
    state = clinic.graft_state(grafts, lora)
    before = logits(base, tok, TEXTS[:3])
    base2, _, grafts2, lora2 = two_donor_host(torch.float32)
    assert not torch.allclose(logits(base2, tok, TEXTS[:3]), before, atol=1e-5)
    clinic.load_graft_state(grafts2, state, lora2)
    assert torch.allclose(logits(base2, tok, TEXTS[:3]), before, atol=1e-5)
    with pytest.raises(ValueError):
        clinic.load_graft_state(grafts2[:1], state, lora2)


def test_two_donor_forward_and_state_in_bf16():
    base, tok, grafts, lora = two_donor_host(torch.bfloat16)
    perturb(grafts, lora)
    out = logits(base, tok, TEXTS[:3])
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    state = clinic.graft_state(grafts, lora)
    base2, _, grafts2, lora2 = two_donor_host(torch.bfloat16)
    clinic.load_graft_state(grafts2, state, lora2)
    assert torch.allclose(logits(base2, tok, TEXTS[:3]).float(), out.float(), atol=1e-2)


def test_only_context_isolates_one_graft_and_restores():
    base, tok, grafts, lora = two_donor_host(torch.float32)
    perturb(grafts, lora)
    full = logits(base, tok, TEXTS[:2])
    with clinic.only(grafts, 0):
        assert [g.forced_off for g in grafts] == [False, True]
        one = logits(base, tok, TEXTS[:2])
    assert [g.forced_off for g in grafts] == [False, False]
    assert not torch.allclose(one, full, atol=1e-5)
    assert torch.allclose(logits(base, tok, TEXTS[:2]), full, atol=1e-6)
