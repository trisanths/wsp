import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tests.test_align import WORDS, tiny_tokenizer
from wsp.tops import SkillTops
from wsp.tops_identity import (IdentityTops, fit_identity_router, identity_sizes, require_true_fine_tune,
                               route_accuracy, shape_report)

VOCAB = 300
A_WORDS = ["the", "cat", "sat", "on"]
B_WORDS = ["dog", "ran", "to", "big"]
G_WORDS = ["a", "mat", "red", "hat"]


def tiny_qwen2(seed, hidden=48, layers=4, theta=1000000.0, heads=4, kv=2, intermediate=96):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv, vocab_size=VOCAB, max_position_embeddings=64, tie_word_embeddings=False,
        rope_theta=theta,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def perturbed(model, scale, seed=1):
    """A member: the host with every layer, the norm, and the head moved a little. Embeddings stay equal."""
    member = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        params = list(member.model.layers.parameters()) + list(member.model.norm.parameters()) + [
            member.lm_head.weight]
        for param in params:
            param.add_(scale * torch.randn(param.shape, generator=gen).to(param.dtype) * param.abs().mean())
    return member.eval()


def planted(words, n, seed, length=10):
    import random

    rng = random.Random(seed)
    return [" ".join(rng.choice(words) for _ in range(length)) for _ in range(n)]


def encode(texts):
    tok = tiny_tokenizer()
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def real(logits, mask):
    """Logits at real positions only: a padded query row has no defined output."""
    return logits * mask.unsqueeze(-1).to(logits.dtype)


def logits_of(net, ids, mask, top):
    with torch.no_grad():
        return real(net(ids, attention_mask=mask, top=top).logits, mask)


def own_logits(model, ids, mask):
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        return real(model(input_ids=ids, attention_mask=mask, position_ids=pos).logits, mask)


def padded_batch():
    texts = [planted(WORDS, 1, 3)[0], planted(WORDS, 1, 4, length=6)[0], planted(WORDS, 1, 5, length=8)[0]]
    ids, mask = encode(texts)
    assert int(mask.sum()) < mask.numel()
    return ids, mask


DTYPES = [(torch.float32, 1e-5), (torch.bfloat16, 3e-2)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cut", [0, 1, 2, 3])
def test_an_identity_top_of_the_host_gives_the_host(dtype, cut):
    host = tiny_qwen2(0).to(dtype)
    net = IdentityTops(host, tiny_tokenizer())
    net.add_top("m", copy.deepcopy(host), cut, donor_tokenizer=tiny_tokenizer())
    ids, mask = padded_batch()
    got = logits_of(net, ids, mask, "m").float()
    want = logits_of(net, ids, mask, "base").float()
    assert torch.equal(got, want)


def test_the_float32_host_top_is_exact_to_rounding():
    host = tiny_qwen2(0)
    net = IdentityTops(host, tiny_tokenizer())
    net.add_top("m", copy.deepcopy(host), 2, donor_tokenizer=tiny_tokenizer())
    ids, mask = padded_batch()
    assert (logits_of(net, ids, mask, "m") - logits_of(net, ids, mask, "base")).abs().max().item() < 1e-5


@pytest.mark.parametrize("dtype,tol", DTYPES)
def test_a_perturbed_member_at_cut_zero_gives_the_members_own_logits(dtype, tol):
    host = tiny_qwen2(0)
    member = perturbed(host, 0.2).to(dtype)
    host = host.to(dtype)
    net = IdentityTops(host, tiny_tokenizer())
    net.add_top("m", member, 0, donor_tokenizer=tiny_tokenizer())
    ids, mask = padded_batch()
    got = logits_of(net, ids, mask, "m").float()
    want = own_logits(member, ids, mask).float()
    assert (got - want).abs().max().item() <= tol
    # The member really differs from the host, so the check is not vacuous.
    assert (want - own_logits(host, ids, mask).float()).abs().max().item() > 0.1


def test_the_error_against_the_member_grows_with_the_trunk():
    host = tiny_qwen2(0)
    member = perturbed(host, 0.3)
    net = IdentityTops(host, tiny_tokenizer())
    ids, mask = padded_batch()
    want = own_logits(member, ids, mask)
    errors = []
    for cut in (0, 1, 2, 3):
        name = "m" + str(cut)
        net.add_top(name, member, cut, donor_tokenizer=tiny_tokenizer())
        errors.append((logits_of(net, ids, mask, name) - want).abs().max().item())
    assert errors[0] < 1e-5
    assert errors == sorted(errors)
    assert errors[-1] > 100 * errors[0] + 1e-3


def test_the_error_against_the_member_shrinks_with_the_perturbation():
    host = tiny_qwen2(0)
    ids, mask = padded_batch()
    errors = []
    for scale in (0.4, 0.04, 0.004):
        member = perturbed(host, scale)
        net = IdentityTops(host, tiny_tokenizer())
        net.add_top("m", member, 2, donor_tokenizer=tiny_tokenizer())
        errors.append((logits_of(net, ids, mask, "m") - own_logits(member, ids, mask)).abs().max().item())
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.05 * errors[0]


def reference_generate(net, ids, mask, new, top):
    ids = ids.clone()
    mask = mask.clone()
    out = []
    for _ in range(new):
        logits = net(ids, attention_mask=mask, top=top).logits
        token = logits[:, -1].argmax(dim=-1)
        out.append(token)
        ids = torch.cat([ids, token[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones(ids.shape[0], 1, dtype=mask.dtype)], dim=1)
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("top", ["m", "base"])
def test_batched_left_padded_cached_generation_equals_uncached_two_calls(dtype, top):
    host = tiny_qwen2(0)
    member = perturbed(host, 0.2)
    if dtype == torch.float32:
        host = host.double()
        member = member.double()
    else:
        host = host.to(dtype)
        member = member.to(dtype)
    net = IdentityTops(host, tiny_tokenizer())
    net.add_top("m", member, 2, donor_tokenizer=tiny_tokenizer())
    matches = 0
    total = 0
    for call in range(2):
        texts = [planted(WORDS, 1, 30 + call)[0], planted(WORDS, 1, 40 + call, length=5)[0],
                 planted(WORDS, 1, 50 + call, length=7)[0]]
        ids, mask = encode(texts)
        assert int(mask.sum()) < mask.numel()
        with torch.no_grad():
            made = net.generate(ids, attention_mask=mask, max_new_tokens=9, top=top, pad_token_id=0)
            want = reference_generate(net, ids, mask, 9, top)
        if dtype == torch.float32:
            assert torch.equal(made[:, ids.shape[1]:], want)
        else:
            # bf16 rounds differently with and without a cache; most greedy tokens still agree.
            matches += int((made[:, ids.shape[1]:] == want).sum())
            total += want.numel()
    if dtype != torch.float32:
        assert matches / total > 0.8


def routed_net():
    host = tiny_qwen2(0)
    net = IdentityTops(host, tiny_tokenizer())
    net.add_top("a", perturbed(host, 0.2, 1), 1, donor_tokenizer=tiny_tokenizer())
    net.add_top("b", perturbed(host, 0.2, 2), 1, donor_tokenizer=tiny_tokenizer())
    return net


def class_texts(seed=0, n=60):
    return {"base": planted(G_WORDS, n, seed), "a": planted(A_WORDS, n, seed + 1), "b": planted(B_WORDS, n, seed + 2)}


def test_the_router_routes_planted_domains():
    net = routed_net()
    report = fit_identity_router(net, class_texts(), lam=0.01, probe_by_class=class_texts(20, 20))
    assert report["classes"] == ["base", "a", "b"]
    assert report["lam"] == 0.01
    assert report["cut"] == 1
    accuracy = route_accuracy(net, class_texts(50, 12))
    assert accuracy["overall"] == 1.0
    assert net.route_texts(planted(A_WORDS, 6, 77)) == ["a"] * 6


def test_a_routed_batch_with_mixed_tops_generates_like_forced_rows():
    net = routed_net()
    fit_identity_router(net, class_texts())
    texts = [planted(A_WORDS, 1, 3, length=7)[0], planted(B_WORDS, 1, 4, length=5)[0],
             planted(G_WORDS, 1, 5, length=9)[0]]
    ids, mask = encode(texts)
    with torch.no_grad():
        made = net.generate(ids, attention_mask=mask, max_new_tokens=6, pad_token_id=0)
        choice = list(net.last_choice)
        assert choice == ["a", "b", "base"]
        for row in range(3):
            n = int(mask[row].sum())
            alone = net.generate(ids[row:row + 1, -n:], max_new_tokens=6, top=choice[row], pad_token_id=0)
            assert torch.equal(made[row, ids.shape[1]:], alone[0, n:])


def test_stitch_and_sink_count_as_zero_and_nothing_trains():
    net = routed_net()
    counts = net.count_params()
    for entry in counts["tops"].values():
        assert entry["stitch"] == 0
        assert entry["sink"] == 0
        assert entry["stored"] == entry["layers"] + entry["norm"] + entry["head"]
    assert all(not top.has_sink for top in net.tops.values())
    assert net.tops["a"].spec["seam"] == "identity"
    with pytest.raises(NotImplementedError):
        net.fit_stitch("a", ["the cat"])
    from wsp.tops_identity import trainable_count

    assert trainable_count(net) == 0


def test_identity_sizes_match_count_params():
    host = tiny_qwen2(0)
    member = perturbed(host, 0.1)
    net = IdentityTops(host, tiny_tokenizer())
    net.add_top("m", member, 2, donor_tokenizer=tiny_tokenizer())
    net.fit_router({"base": planted(G_WORDS, 20, 0), "m": planted(A_WORDS, 20, 1)})
    counts = net.count_params()
    sizes = identity_sizes(host.config, [{"name": "m", "donor_config": member.config, "cut_host": 2}],
                           router_classes=2)
    assert sizes["stored"] == counts["stored"]
    assert sizes["run"]["m"] == counts["run"]["m"]
    assert sizes["tops"]["m"]["stored"] == counts["tops"]["m"]["stored"]


def test_the_gate_refuses_other_shapes_other_rope_and_a_foreign_lineage():
    host = tiny_qwen2(0)
    net = IdentityTops(host, tiny_tokenizer())
    with pytest.raises(ValueError, match="not shaped like the host"):
        net.add_top("w", tiny_qwen2(1, hidden=32, intermediate=64), 1)
    with pytest.raises(ValueError, match="rope_theta"):
        net.add_top("r", tiny_qwen2(1, theta=10000.0), 1)
    with pytest.raises(ValueError, match="lineage says"):
        net.add_top("f", copy.deepcopy(host), 1, lineage={"kind": "foreign"})
    with pytest.raises(ValueError, match="cut_donor"):
        net.add_top("c", copy.deepcopy(host), 1, cut_donor=2)
    net.add_top("ok", copy.deepcopy(host), 1, lineage={"kind": "fine_tune_of_ancestor"})
    assert net.tops["ok"].spec["lineage"] == "fine_tune_of_ancestor"
    ok, differences = shape_report(host.config, tiny_qwen2(1, theta=10000.0).config)
    assert not ok and "rope_theta" in differences
    require_true_fine_tune(host.config, host.config, "fine_tune")


def test_a_scrambled_identity_top_differs_from_the_member():
    host = tiny_qwen2(0)
    member = perturbed(host, 0.05)
    net = IdentityTops(host, tiny_tokenizer())
    net.add_top("real", member, 2, donor_tokenizer=tiny_tokenizer())
    net.add_top("scr", member, 2, control="scrambled", control_seed=3, donor_tokenizer=tiny_tokenizer())
    ids, mask = padded_batch()
    a = logits_of(net, ids, mask, "real")
    b = logits_of(net, ids, mask, "scr")
    assert (a - b).abs().max().item() > 1e-3


def test_save_and_load_round_trip_an_identity_network(tmp_path):
    host = tiny_qwen2(0)
    member = perturbed(host, 0.1)
    path = str(tmp_path / "member")
    member.save_pretrained(path)
    tiny_tokenizer().save_pretrained(path)
    net = IdentityTops(host, tiny_tokenizer(), base_id="tiny-host")
    net.add_top("m", path, 2)
    net.fit_router({"base": planted(G_WORDS, 20, 0), "m": planted(A_WORDS, 20, 1)})
    ids, mask = padded_batch()
    want = logits_of(net, ids, mask, "m")
    net.save(str(tmp_path / "net"))
    loaded = IdentityTops.load(str(tmp_path / "net"), base_model=host, tokenizer=tiny_tokenizer())
    assert isinstance(loaded.tops["m"], type(net.tops["m"]))
    assert (logits_of(loaded, ids, mask, "m") - want).abs().max().item() < 1e-6
    assert loaded.route_texts(planted(A_WORDS, 3, 9)) == net.route_texts(planted(A_WORDS, 3, 9))
