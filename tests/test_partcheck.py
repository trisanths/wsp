"""wsp.partcheck on tiny random models and tiny synthetic skill files (CPU, no network)."""

import json
import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from wsp import partcheck as pc
from wsp import synth
from tests.test_align import WORDS, tiny_tokenizer
from tests.test_synth import sentences, tiny_donor, tiny_host, trained_permutation_donor


def build_tokenizer(words):
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in words:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def number_tokenizer():
    """The tiny tokenizer plus three tokens that hold a digit."""
    return build_tokenizer(WORDS + ["n1", "n2", "n3"])


# ------------------------------------------------------------ catalog files


def item(prompt, answer, digits=False):
    return {"context": prompt, "continuation": answer, "digits_only": digits}


def write_skill(path, search, heldout, control_search=None, control_heldout=None, hash_value="h1"):
    """Write a skill file in the miners' frozen format: items and control as {search, heldout}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    control = None
    if control_search is not None:
        control = {"search": control_search, "heldout": control_heldout or []}
    data = {"name": path.stem, "description": "d", "hash": hash_value, "agent_items": True, "generator_code": "",
            "items": {"search": search, "heldout": heldout}, "control": control, "base_margin": 1.0}
    path.write_text(json.dumps(data))


def plain_items(count, offset=0):
    return [item(" ".join(WORDS[(offset + i + j) % 12] for j in range(3)) + " x%d" % i, " " + WORDS[(i + offset) % 12])
            for i in range(count)]


def make_catalog(tmp_path, rows):
    folder = tmp_path / "library"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "catalog.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def mlp_row(tmp_path, name, num_units=3, folder="library", **extra):
    """A row whose skill file is under person_a/mine/skills and whose units file is under <folder>/tiny."""
    skill = tmp_path / "person_a" / "mine" / "skills" / "tiny" / (name + ".json")
    write_skill(skill, plain_items(20), plain_items(20, 5), plain_items(10, 3), plain_items(10, 4))
    units = tmp_path / folder / "tiny" / (name + ".pt")
    units.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "mlp_cluster", "model": "donorA", "layer": 2, "indices": torch.arange(num_units) + 3}, units)
    row = {"id": name, "type": "mlp_cluster", "model": "donorA", "tier": "screened", "layer": 2,
           "num_units": num_units, "d_model": 48, "units": folder + "\\tiny\\" + name + ".pt",
           "skill": {"name": name, "path": "mine\\skills\\tiny\\" + name + ".json", "hash": "h1", "source": "agent",
                     "description": "x"}}
    row.update(extra)
    return row


def write_donor_catalog(tmp_path, name, rows):
    folder = tmp_path / "person_a" / "catalog"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# --------------------------------------------------------------- tests: files


def test_paths_with_backslashes_resolve_under_person_a_and_the_repo_root(tmp_path):
    row = mlp_row(tmp_path, "one")
    catalog = make_catalog(tmp_path, [row])
    groups, missing = pc.load_parts(catalog)
    assert missing == []
    part = groups["donorA"][0]
    assert part["skill_file"].endswith("person_a/mine/skills/tiny/one.json")
    assert part["units_file"].endswith("library/tiny/one.pt")
    assert part["indices"] == [3, 4, 5]
    assert part["skill"] == "one"
    assert part["hash_ok"] is True
    row2 = mlp_row(tmp_path, "two")
    row2["skill"]["path"] = "person_a\\mine\\skills\\tiny\\two.json"
    groups, missing = pc.load_parts(make_catalog(tmp_path, [row2]))
    assert missing == []
    assert groups["donorA"][0]["skill_file"].endswith("mine/skills/tiny/two.json")
    assert pc.fix_path("a\\b\\c.json") == "a/b/c.json"


def test_per_donor_files_are_read_deduped_and_hash_mismatch_is_recorded(tmp_path):
    merged = mlp_row(tmp_path, "shared")
    only_donor = mlp_row(tmp_path, "solo", folder="person_a/parts")
    again = mlp_row(tmp_path, "shared", folder="person_a/parts")
    again["num_units"] = 99
    bad_hash = mlp_row(tmp_path, "badhash", folder="person_a/parts")
    bad_hash["skill"]["hash"] = "something else"
    write_donor_catalog(tmp_path, "donorA.jsonl", [only_donor, again, bad_hash])
    write_donor_catalog(tmp_path, "donorA.rejected.jsonl", [mlp_row(tmp_path, "rejected", folder="person_a/parts")])
    groups, missing = pc.load_parts(make_catalog(tmp_path, [merged]))
    ids = [p["id"] for p in groups["donorA"]]
    assert ids == ["shared", "solo", "badhash"]
    assert groups["donorA"][0]["num_units"] == 3
    assert [(m["id"], m["reason"]) for m in missing] == [("shared", "duplicate id with different content")]
    assert [m["id"] for m in pc.hash_mismatches(groups)] == ["badhash"]
    groups, missing = pc.load_parts(make_catalog(tmp_path, [merged]), donor_dir=None)
    assert [p["id"] for p in groups["donorA"]] == ["shared"]


def test_identical_duplicates_are_silent_and_one_id_from_two_donors_keeps_both(tmp_path):
    first = mlp_row(tmp_path, "same")
    second = mlp_row(tmp_path, "same", folder="person_a/parts")
    other = mlp_row(tmp_path, "same", folder="person_a/parts")
    other["model"] = "donorB"
    other["skill"]["hash"] = "h2"
    write_donor_catalog(tmp_path, "x.jsonl", [second, other])
    groups, missing = pc.load_parts(make_catalog(tmp_path, [first]), tiers=("screened",))
    assert [p["id"] for p in groups["donorA"]] == ["same"]
    assert [p["id"] for p in groups["donorB"]] == ["same"]
    assert [m["reason"].split(":")[0] for m in missing] == ["id also used by another donor"]


def test_builtin_skill_rows_read_the_builtin_file(tmp_path):
    data = tmp_path / "person_a" / "mine" / "data"
    data.mkdir(parents=True)
    (data / "builtin_skills.json").write_text(json.dumps(
        {"math": {"description": "m", "items": {"search": [item("1 + 1 =", " 2", True)] * 3,
                                                "heldout": [item("2 + 2 =", " 4", True)] * 2}}}))
    row = mlp_row(tmp_path, "b")
    row["skill"] = {"source": "builtin", "name": "math"}
    groups, missing = pc.load_parts(make_catalog(tmp_path, [row]))
    assert missing == []
    part = groups["donorA"][0]
    got = pc.skill_items(part, "heldout")
    assert [i["answer"] for i in got["items"]] == [" 4", " 4"]
    assert got["items"][0]["digits_only"] is True
    assert got["control"] == []
    assert part["hash_ok"] is None


def test_prefers_parts_with_64_units_when_the_list_is_cut():
    parts = [{"id": "s1", "kind": "mlp_cluster", "num_units": 1, "skill": "a"},
             {"id": "s2", "kind": "mlp_cluster", "num_units": 4, "skill": "b"},
             {"id": "big", "kind": "mlp_cluster", "num_units": 64, "skill": "a"},
             {"id": "blk", "kind": "block", "num_units": None, "skill": "c"}]
    assert [p["id"] for p in pc.pick_parts(parts, 2)] == ["big", "blk"]
    assert [p["id"] for p in pc.pick_parts(parts, 3)] == ["big", "blk", "s2"]
    assert len(pc.pick_parts(parts, None)) == 4


def test_missing_files_are_listed_not_fatal(tmp_path):
    good = mlp_row(tmp_path, "good")
    no_skill_file = mlp_row(tmp_path, "gone_skill")
    (tmp_path / "person_a" / "mine" / "skills" / "tiny" / "gone_skill.json").unlink()
    no_units = mlp_row(tmp_path, "gone_units")
    (tmp_path / "library" / "tiny" / "gone_units.pt").unlink()
    bare = {"id": "bare", "type": "mlp_cluster", "model": "donorA", "tier": "screened", "layer": 1}
    other_tier = mlp_row(tmp_path, "prov")
    other_tier["tier"] = "provisional"
    groups, missing = pc.load_parts(make_catalog(tmp_path, [good, no_skill_file, no_units, bare, other_tier]))
    assert [p["id"] for p in groups["donorA"]] == ["good"]
    reasons = {m["id"]: m["reason"] for m in missing}
    assert set(reasons) == {"gone_skill", "gone_units", "bare"}
    assert "skill file not found" in reasons["gone_skill"]
    assert "units file not found" in reasons["gone_units"]
    assert "no skill" in reasons["bare"]
    groups, missing = pc.load_parts(make_catalog(tmp_path, [good, other_tier]), tiers=None)
    assert len(groups["donorA"]) == 2
    groups, missing = pc.load_parts(make_catalog(tmp_path, [good]), models=["someone else"])
    assert groups == {}


def test_rows_with_unknown_extra_fields_load(tmp_path):
    row = mlp_row(tmp_path, "extra", routing={"router": "x"}, selection_data={"pools": ["a"]},
                  future_field=[1, 2, 3])
    row["signature"] = {"effect": {}, "z": 4.2, "guard": {"passed": True, "gate_version": 3}}
    groups, missing = pc.load_parts(make_catalog(tmp_path, [row]))
    assert missing == []
    part = groups["donorA"][0]
    assert part["extra"]["routing"] == {"router": "x"}
    assert part["extra"]["selection_data"] == {"pools": ["a"]}
    assert part["signature"]["z"] == 4.2
    assert part["row"]["future_field"] == [1, 2, 3]
    carried = pc.carried_fields(part)
    assert carried["signature_z"] == 4.2 and carried["routing"] == {"router": "x"}


# ---------------------------------------------------------------- tests: split


def test_split_is_the_files_own_halves_and_never_mixes(tmp_path):
    path = tmp_path / "person_a" / "mine" / "skills" / "tiny" / "halves.json"
    search = plain_items(7)
    heldout = plain_items(5, 6)
    write_skill(path, search, heldout, plain_items(3), plain_items(2, 2))
    pc._SKILL_CACHE.clear()
    part = {"skill_file": str(path)}
    got = pc.skill_items(part, "search")
    assert [i["prompt"] for i in got["items"]] == [i["context"] for i in search]
    assert [i["answer"] for i in got["items"]] == [i["continuation"] for i in search]
    assert len(got["control"]) == 3
    held = pc.skill_items(part, "heldout")
    assert [i["prompt"] for i in held["items"]] == [i["context"] for i in heldout]
    assert len(held["control"]) == 2
    assert len(pc.skill_items(part, "all")["items"]) == 12
    with pytest.raises(ValueError):
        pc.skill_items(part, "fit")
    write_skill(path, search, heldout, None)
    pc._SKILL_CACHE.clear()
    assert pc.skill_items(part, "heldout")["control"] == []


def test_drop_overlap_removes_heldout_items_that_repeat_the_search_half():
    search = [item("Q: a?", " 1"), item("Q: b?", " 2"), item("Q: c? A: 3\nQ: d? A:", " 4")]
    search = pc.normalize_halves({"search": search})["search"]
    held = pc.normalize_halves({"search": [], "heldout": [
        item("Q: a?", " 1"),          # same pair
        item("Q: b?", " 9"),          # same context, other answer
        item("Q: c? A:", " 3"),       # the question and answer sit inside a search demo
        item("Q: e?", " 5")]})["heldout"]
    kept, dropped = pc.drop_overlap(held, search)
    assert [i["prompt"] for i in kept] == ["Q: e?"]
    assert dropped == 3
    kept, dropped = pc.drop_overlap([], search)
    assert kept == [] and dropped == 0


def test_check_parts_scores_only_independent_heldout_items_and_reports_skips(tmp_path):
    """A skill whose heldout half is a copy of its search half has nothing left to score: skipped, with a row."""
    skills = tmp_path / "person_a" / "mine" / "skills" / "tiny"
    same = plain_items(12)
    write_skill(skills / "copy.json", same, same, plain_items(4), plain_items(4))
    mixed = plain_items(12) + plain_items(8, 20)
    write_skill(skills / "mixed.json", plain_items(12), mixed[:12] + mixed[12:], None)
    rows = []
    for name in ("copy", "mixed"):
        rows.append({"id": name, "type": "mlp_cluster", "model": "donorA", "tier": "screened", "layer": 2,
                     "num_units": 3, "units": "library\\tiny\\" + name + ".pt",
                     "skill": {"name": name, "path": "mine\\skills\\tiny\\" + name + ".json", "hash": "h1"}})
        units = tmp_path / "library" / "tiny" / (name + ".pt")
        units.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"indices": torch.arange(3)}, units)
    groups, missing = pc.load_parts(make_catalog(tmp_path, rows))
    assert missing == []
    pc._SKILL_CACHE.clear()
    tok = tiny_tokenizer()
    lines = []
    report = pc.check_parts((tiny_host(seed=1), tok), (tiny_donor(), tok), groups["donorA"][:1], [], base_id="b",
                            log=lines.append, thresholds={"min_items": 6})
    assert report["parts"] == []
    assert report["skipped"][0]["id"] == "copy"
    assert report["skipped"][0]["n_heldout_dropped_overlap"] == 12
    out = pc.rows_from_report(dict(report, base="b", missing=[
        {"id": "gone", "model": "donorA", "reason": "skill file not found"},
        {"id": "dup", "model": "donorA", "reason": "duplicate id with different content"}]))
    assert [r["id"] for r in out] == ["copy", "gone"]
    for r in out:
        assert tuple(r.keys()) == pc.RESULT_FIELDS
        assert r["verdict"] == "skipped" and r["reason"]
    assert out[0]["n_heldout_dropped_overlap"] == 12


# --------------------------------------------------------------- tests: margin


def hand_margin(model, tok, items):
    """The margin with plain loops: one sequence per forward pass, no padding, no batching."""
    n = len(items)

    def lp_of(rows):
        total = 0.0
        count = 0
        for prompt, answer, digits in rows:
            ctx = tok(prompt, add_special_tokens=True)["input_ids"]
            cont = tok(answer, add_special_tokens=False)["input_ids"]
            ids = ctx + cont
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits[0].float()
            for t in range(len(ctx), len(ids)):
                text = tok.convert_ids_to_tokens(ids[t])
                if digits and not any(ch.isdigit() for ch in text):
                    continue
                total += float(torch.log_softmax(logits[t - 1], dim=-1)[ids[t]])
                count += 1
        return total / count

    real = lp_of([(i["prompt"], i["answer"], i.get("digits_only", False)) for i in items])
    shuffled = lp_of([(items[k]["prompt"], items[(k + 1) % n]["answer"], items[k].get("digits_only", False))
                      for k in range(n)])
    return real - shuffled


def test_item_margin_equals_a_hand_computation_with_padding_and_batches():
    model = tiny_host(seed=2)
    tok = tiny_tokenizer()
    items = [{"prompt": "the cat", "answer": " sat on"},
             {"prompt": "dog ran to a big red", "answer": " hat"},
             {"prompt": "a mat", "answer": " the cat sat"},
             {"prompt": "red hat on", "answer": " mat"},
             {"prompt": "big", "answer": " dog"}]
    expected = hand_margin(model, tok, items)
    assert abs(expected) > 1e-4
    for batch_size in (1, 2, 16):
        got = pc.item_margin(model, tok, items, batch_size=batch_size)
        assert got == pytest.approx(expected, abs=1e-4)
    assert pc.item_margin(model, tok, []) is None
    assert pc.item_margin(model, tok, items[:1]) == pytest.approx(0.0, abs=1e-6)


def test_item_margin_scores_only_digit_tokens_when_digits_only():
    model = tiny_host(seed=2)
    tok = number_tokenizer()
    items = [{"prompt": "the cat", "answer": " sat n1 on", "digits_only": True},
             {"prompt": "dog ran", "answer": " n2 big n3", "digits_only": True},
             {"prompt": "a mat", "answer": " the n1", "digits_only": False},
             {"prompt": "red hat", "answer": " n3 mat", "digits_only": True}]
    expected = hand_margin(model, tok, items)
    for batch_size in (1, 3, 16):
        assert pc.item_margin(model, tok, items, batch_size=batch_size) == pytest.approx(expected, abs=1e-4)
    plain = [dict(i, digits_only=False) for i in items]
    assert abs(pc.item_margin(model, tok, plain) - expected) > 1e-4
    no_digits = [{"prompt": "the cat", "answer": " sat", "digits_only": True}] * 2
    assert pc.item_margin(model, tok, no_digits) is None


def test_item_margin_accepts_context_continuation_items():
    model = tiny_host(seed=2)
    tok = tiny_tokenizer()
    a = [{"prompt": "the cat", "answer": " sat"}, {"prompt": "a mat", "answer": " dog"}]
    b = [{"context": "the cat", "continuation": " sat"}, {"context": "a mat", "continuation": " dog"}]
    assert pc.item_margin(model, tok, a) == pytest.approx(pc.item_margin(model, tok, b), abs=1e-6)


# ------------------------------------------------------------ tests: gate closing


def two_branch_model():
    host = tiny_host()
    tok = tiny_tokenizer()
    comps = [{"id": "A", "kind": "block", "donor": "donorA", "layers": [1, 2], "skill": "s", "host_layer": 1},
             {"id": "B", "kind": "mlp_cluster", "donor": "donorA", "layer": 2, "indices": list(range(8)),
              "host_layer": 2}]
    s = dict(synth.DEFAULTS)
    model, branches, _, _ = synth.place(host, tok, comps, {"donorA": (tiny_donor(), tok)}, s)
    torch.manual_seed(0)
    for b in branches:
        b.ramp = 1.0
        torch.nn.init.normal_(b.w_out.weight, std=0.5)
    return model


def logits_on(model):
    ids = torch.tensor([[3, 5, 7, 9, 4, 6], [4, 8, 6, 5, 9, 3]])
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


def test_gate_closed_measurement_isolates_one_part():
    model = two_branch_model()
    full = logits_on(model)
    with pc.closed_part(model, "A"):
        without_a = logits_on(model)
    assert not torch.allclose(full, without_a, atol=1e-5)
    assert torch.equal(full, logits_on(model))
    with pc.only_part(model, "B"):
        only_b = logits_on(model)
    # Only B open equals A closed, since the network has just these two branches.
    assert torch.allclose(without_a, only_b, atol=1e-6)
    with pc.closed_part(model, "A"):
        with pc.closed_part(model, "B"):
            none_open = logits_on(model)
    with synth.branches_off(model):
        base_logits = logits_on(model)
    assert torch.equal(none_open, base_logits)
    assert not torch.allclose(without_a, base_logits, atol=1e-5)
    flags = [b.forced_off for b in model.wsp_grafts]
    assert flags == [False, False]
    with pytest.raises(KeyError):
        with pc.closed_part(model, "Z"):
            pass


# ------------------------------------------------------------ tests: planted part

KEYS = 100         # words k1..k100; the donor memorizes a fixed random map between them
NSEARCH = 8        # keys the seam trains on: too few to learn a 100 key hash-like map
NHELD = 40         # keys in the heldout half
DONOR_STEPS = 1500


def key_words():
    return ["k%d" % i for i in range(0, KEYS + 1)]


def key_perm():
    """A fixed random map of 1..KEYS onto 1..KEYS. It has no rule a few examples could reveal."""
    gen = torch.Generator().manual_seed(4)
    order = torch.randperm(KEYS, generator=gen).tolist()
    return {i + 1: order[i] + 1 for i in range(KEYS)}


def memorizing_donor(tok):
    """A 6-layer donor, 4 times wider in the MLP, that maps the prompt "k0 kN" to "k<perm(N)>" for every key.

    Layers 0, 1, 4, 5 are identity (zero o_proj and down_proj, frozen), so the
    map sits only in layers 2 and 3. The map is memorized, so it cannot be
    inferred from a few examples: a compile that sees NSEARCH keys cannot know
    the other keys unless it carries the donor's weights.
    """
    torch.manual_seed(11)
    config = transformers.LlamaConfig(
        hidden_size=48, intermediate_size=384, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    donor = transformers.LlamaForCausalLM(config)
    for i in (0, 1, 4, 5):
        layer = donor.model.layers[i]
        layer.self_attn.o_proj.weight.data.zero_()
        layer.mlp.down_proj.weight.data.zero_()
        for p in layer.parameters():
            p.requires_grad_(False)
    params = [p for p in donor.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-3)
    perm = key_perm()
    first = tok("k0")["input_ids"][0]
    ids = torch.tensor([[first, tok("k%d" % k)["input_ids"][0]] for k in range(1, KEYS + 1)])
    target = torch.tensor([tok("k%d" % perm[k])["input_ids"][0] for k in range(1, KEYS + 1)])
    for step in range(DONOR_STEPS):
        logits = donor(input_ids=ids).logits[:, 1]
        loss = torch.nn.functional.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    donor.eval()
    return donor


def key_sentences(count, seed):
    """General text: random sentences of six keys that never start with k0 and follow no map."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        out.append(" ".join("k%d" % (1 + int(torch.randint(0, KEYS, (1,), generator=gen))) for _ in range(6)))
    return out


def build_planted(tmp, controls, seeds, per_part, swap_real_for_random=False):
    """Return (report, host, tokenizer, parts, log lines) for one donor with a memorized map and two block parts.

    Part "mem" is block [2, 3] of the donor (it holds the map). Part "decoy" is block [0, 1] (identity: its output
    is a zero delta). Search half: keys 1 to NSEARCH. Heldout half: NHELD other keys, never seen by the seam.
    With swap_real_for_random the "real" compile is a random-control compile with its own seeds, so no part
    carries donor content: the verdict must not be "survives compile".
    """
    tok = build_tokenizer(key_words())
    perm = key_perm()

    def make(keys):
        return [item("k0 k%d" % k, " k%d" % perm[k]) for k in keys]

    search = make(range(1, NSEARCH + 1))
    heldout = make(range(NSEARCH + 1, NSEARCH + 1 + NHELD))
    control_search = [item("k0 k%d" % k, " k%d" % k) for k in range(1, 7)]
    control_heldout = [item("k0 k%d" % k, " k%d" % k) for k in range(7, 19)]
    skills = tmp / "person_a" / "mine" / "skills" / "tiny"
    rows = []
    for name, layers in (("mem", [2, 3]), ("decoy", [0, 1])):
        write_skill(skills / (name + ".json"), search, heldout, control_search, control_heldout)
        rows.append({"id": name, "type": "block", "model": "mem_donor", "tier": "screened", "layers": layers,
                     "layer": layers[0], "skill": {"name": name, "path": "mine\\skills\\tiny\\" + name + ".json",
                                                   "hash": "h1", "source": "agent"}})
    groups, missing = pc.load_parts(make_catalog(tmp, rows))
    assert missing == []
    donor = memorizing_donor(tok)
    host = tiny_host(seed=5)
    host.lm_head.weight.data *= 40
    budget = {"stage1_steps": 20, "stage2_steps": 0, "lr_translation": 3e-2, "lr_gate": 1e-2, "batch_size": 8,
              "seq_len": 16, "steps_in": 30, "steps_lm": 10, "batch_in": 4, "lora_rank": 2}
    lines = []
    original = pc.compile_group
    if swap_real_for_random:
        def swapped(base, donor_pair, parts, fit, texts, budget_, stage2, control=None, seed=0):
            if control is None:
                return original(base, donor_pair, parts, fit, texts, budget_, stage2, "random", seed + 100)
            return original(base, donor_pair, parts, fit, texts, budget_, stage2, control, seed)

        pc.compile_group = swapped
    try:
        report = pc.check_parts((host, tok), (donor, tok), groups["mem_donor"], key_sentences(80, 7), base_id="tiny-host",
                                relation={"kind": "diverged_same_shape"}, seeds=seeds, controls=controls, budget=budget,
                                per_part=per_part, thresholds={"min_items": 6, "min_seeds_survive": 3},
                                log=lines.append, cell="planted|tiny-host")
    finally:
        pc.compile_group = original
    return report, host, tok, groups["mem_donor"], lines


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """The planted compile with random and scrambled controls over 3 seeds, and the one-at-a-time check."""
    return build_planted(tmp_path_factory.mktemp("planted"), controls=("random", "scrambled"), seeds=(0, 1, 2),
                         per_part=True)


@pytest.fixture(scope="module")
def planted_swapped(tmp_path_factory):
    """The same world with the real compile replaced by a random control compile."""
    return build_planted(tmp_path_factory.mktemp("planted_swapped"), controls=("random",), seeds=(0, 1, 2),
                         per_part=False, swap_real_for_random=True)


def by_id(report):
    return {r["id"]: r for r in report["parts"]}


def test_planted_part_beats_the_control_contributions_on_heldout_items(planted):
    report = planted[0]
    rows = by_id(report)
    mem = rows["mem"]
    decoy = rows["decoy"]
    assert mem["n_seeds"] == 3
    assert mem["n_heldout"] == NHELD and mem["n_heldout_dropped_overlap"] == 0
    assert mem["margin_donor"] - mem["margin_base"] > 1.0
    # The donor's block carries the map: its own gate matters, on keys the seam never trained on.
    random_best = max(pc.mean([v]) for v in mem["contribution_random"])
    scrambled_mean = pc.mean(mem["contribution_scrambled"])
    assert mem["contribution"] > pc.mean(mem["contribution_random"]) + pc.THRESHOLDS["min_gain"]
    assert mem["contribution"] > scrambled_mean + pc.THRESHOLDS["min_gain"]
    assert mem["control_best_contribution"] is not None
    assert mem["contribution"] - mem["control_best_contribution"] >= pc.THRESHOLDS["min_gain"]
    assert mem["verdict"] == "survives compile"
    # The decoy block is identity in the donor: its gate changes nothing, and it never survives. Its gain is the
    # group's gain (one compile), which is why gain alone cannot be used per part.
    assert abs(decoy["contribution"]) < 1e-6
    assert decoy["verdict"] != "survives compile"
    assert decoy["gain"] == pytest.approx(mem["gain"], abs=1e-9)
    assert report["group"]["n_survive"] == 1
    assert random_best is not None


def test_a_random_control_swapped_in_for_the_real_part_does_not_survive(planted_swapped):
    report = planted_swapped[0]
    for row in report["parts"]:
        assert row["verdict"] != "survives compile"
        assert row["verdict"] != pc.PROVISIONAL
    assert report["group"]["n_survive"] == 0


def test_planted_controls_are_measured_per_part_and_logged_with_the_cell(planted):
    report, host, tok, parts, lines = planted
    mem = by_id(report)["mem"]
    assert len(mem["margin_random"]) == 3 and len(mem["margin_scrambled"]) == 3
    assert len(mem["contribution_random"]) == 3 and len(mem["contribution_scrambled"]) == 3
    assert "random" in mem["content_z_detail"] and "scrambled" in mem["content_z_detail"]
    assert report["settings"]["controls"] == ["random", "scrambled"]
    assert mem["drift_ref"] is not None
    assert len(lines) > 10
    assert all(line.startswith("partcheck cell=planted|tiny-host ") for line in lines)
    assert any("control=random" in line and "part=mem" in line for line in lines)


def test_planted_per_part_run_and_group_verdict(planted):
    report = planted[0]
    mem = by_id(report)["mem"]
    assert mem["margin_alone"] is not None
    assert mem["alone_share"] is not None
    assert report["group"]["alone_checked"] == 1
    assert report["group"]["verdict"] in ("parts survive together and alone", "some parts need the others")


def test_base_model_is_not_changed_by_check_parts(planted):
    report, host, tok, parts, lines = planted
    assert not hasattr(host, "wsp_grafts")
    mem = by_id(report)["mem"]
    heldout = pc.skill_items(parts[0], "heldout")["items"]
    assert mem["margin_base"] == pytest.approx(pc.item_margin(host, tok, heldout), abs=1e-5)


# -------------------------------------------------------------- tests: output rows


def test_output_rows_have_every_field_and_merge_by_id_and_base(planted, tmp_path):
    report = planted[0]
    rows = pc.rows_from_report(report)
    assert [r["id"] for r in rows] == ["mem", "decoy"]
    for r in rows:
        assert tuple(r.keys()) == pc.RESULT_FIELDS
        assert all(key in r for key in pc.RESULT_FIELDS)
        assert r["base"] == "tiny-host"
        assert r["relation"] == "diverged_same_shape"
        assert r["verdict"] is not None
    path = tmp_path / "out" / "rows.json"
    pc.write_rows(rows, str(path))
    changed = [dict(rows[0], verdict="harmful")]
    pc.write_rows(changed, str(path))
    stored = json.loads(path.read_text())
    assert len(stored) == 2
    assert {r["id"]: r["verdict"] for r in stored}["mem"] == "harmful"
    other_base = [dict(rows[0], base="another-base")]
    pc.write_rows(other_base, str(path))
    assert len(json.loads(path.read_text())) == 3
    pc.write_donor_report(report, str(tmp_path / "out" / "donor.json"))
    again = json.loads((tmp_path / "out" / "donor.json").read_text())
    assert again["donor"] == "mem_donor"
    assert math.isfinite(again["parts"][0]["margin_base"])


# --------------------------------------------------------------- tests: helpers


def test_rows_keep_signature_and_routing_fields_when_present():
    report = {"base": "b", "relation": {"kind": "fine_tune"}, "parts": [
        {"id": "p", "margin_base": 0.0, "margin_donor": 1.0, "margin_compiled": 0.5, "recovered": 0.5,
         "contribution": 0.4, "content_z": 3.0, "control_effect": 0.0, "drift": 0.0, "verdict": "survives compile",
         "carried": {"signature_z": 4.2, "routing": {"router": "x"}, "base": "must not replace"}},
        {"id": "q", "margin_base": 0.0, "margin_donor": 0.0, "margin_compiled": 0.0, "recovered": None,
         "contribution": 0.0, "content_z": None, "control_effect": None, "drift": None, "verdict": "no gap"}]}
    rows = pc.rows_from_report(report)
    assert rows[0]["signature_z"] == 4.2 and rows[0]["routing"] == {"router": "x"}
    assert rows[0]["base"] == "b" and rows[0]["relation"] == "fine_tune"
    assert tuple(rows[1].keys()) == pc.RESULT_FIELDS


def test_part_verdict_rules():
    th = dict(pc.THRESHOLDS)
    ok = {"donor_gap": 1.0, "contribution": 0.5, "gain": 0.6, "control_best": 0.1, "content_z": 5.0, "drift": 0.0,
          "drift_ref": 0.0, "n_seeds": 5}
    assert pc.part_verdict(ok, th) == "survives compile"
    assert pc.part_verdict(dict(ok, donor_gap=0.01), th) == "no gap"
    assert pc.part_verdict(dict(ok, contribution=-0.2), th) == "harmful"
    assert pc.part_verdict(dict(ok, drift=0.5), th) == "harmful"
    assert pc.part_verdict(dict(ok, contribution=0.01), th) == "no better than controls"
    # control_best is the best control CONTRIBUTION, not a group gain.
    assert pc.part_verdict(dict(ok, control_best=0.45), th) == "no better than controls"
    assert pc.part_verdict(dict(ok, content_z=1.0), th) == "no better than controls"
    assert pc.part_verdict(dict(ok, content_z=None), th) == "survives compile"
    assert pc.part_verdict(dict(ok, n_seeds=3), th) == pc.PROVISIONAL
    assert pc.part_verdict(dict(ok, n_seeds=None), th) == "survives compile"


def test_drift_is_judged_against_the_controls_and_after_no_gap():
    th = dict(pc.THRESHOLDS)
    base = {"donor_gap": 1.0, "contribution": 0.5, "control_best": 0.0, "content_z": 5.0, "n_seeds": 5}
    # Chat-format text gives every compile a few nats of drift. A part at the control level is not harmful.
    assert pc.part_verdict(dict(base, drift=4.0, drift_ref=4.0), th) == "survives compile"
    assert pc.part_verdict(dict(base, drift=4.0, drift_ref=0.0), th) == "harmful"
    assert pc.part_verdict(dict(base, drift=4.0, drift_ref=3.0), th) == "harmful"
    # No gap comes before a drift verdict.
    assert pc.part_verdict(dict(base, drift=9.0, drift_ref=0.0, donor_gap=0.0), th) == "no gap"


def test_stage1_steps_grow_with_the_part_count():
    assert pc.steps_for_parts(3, 3) == 100
    assert pc.steps_for_parts(63, 3) == 2100
    assert pc.steps_for_parts(18, 3) == 600
    assert pc.steps_for_parts(2, 5) == 100


def test_unsupported_model_types_are_refused():
    assert "post_attention_layernorm" in pc.unsupported_reason("olmo2")
    assert pc.unsupported_reason("qwen2") is None
    assert pc.unsupported_reason(None) is None


def test_pick_parts_takes_one_per_skill_first():
    parts = [{"id": i, "skill": i[0], "kind": "block"} for i in ("a1", "a2", "b1", "c1")]
    assert [p["id"] for p in pc.pick_parts(parts, 3)] == ["a1", "b1", "c1"]
    assert [p["id"] for p in pc.pick_parts(parts, 4)] == ["a1", "a2", "b1", "c1"]
    assert len(pc.pick_parts(parts, None)) == 4


def test_choose_base_prefers_the_closest_relation_and_survives_errors():
    kinds = {"small": "same_family_other_size", "sibling": "fine_tune", "dead": None}

    def relate(base, donor):
        if kinds[base] is None:
            raise RuntimeError("hub says no")
        return {"kind": kinds[base], "evidence": {}, "flags": {}, "warnings": []}

    base, rel, tried = pc.choose_base("donor", ["dead", "small", "sibling"], default="small", relate=relate)
    assert base == "sibling"
    assert [t["kind"] for t in tried] == ["unknown", "same_family_other_size", "fine_tune"]
    base, rel, tried = pc.choose_base("donor", ["dead"], default="small", relate=relate)
    assert base == "small" and rel["kind"] == "same_family_other_size"
    with pytest.raises(ValueError):
        pc.choose_base("donor", ["dead"], relate=relate)


def test_raw_tokenizer_hides_the_chat_template_and_keeps_encoding():
    tok = tiny_tokenizer()
    tok.chat_template = "{{ messages }}"
    raw = pc.RawTokenizer(tok)
    assert raw.chat_template is None
    assert raw("the cat")["input_ids"] == tok("the cat")["input_ids"]
    assert raw.eos_token_id == tok.eos_token_id
    enc = synth.encode_item(raw, {"prompt": "the cat", "answer": " sat"}, 16)
    assert enc is not None and enc[1][-1] == 1


def test_compile_group_gives_every_part_enough_stage1_steps(monkeypatch):
    seen = {}

    def fake_synthesize(model_tok, comps, texts, settings, donors=None):
        seen.update(settings)
        return model_tok[0], None, {}

    monkeypatch.setattr(pc, "synthesize", fake_synthesize)
    parts = [{"id": "p%d" % i, "kind": "block", "layers": [0, 1], "model": "d", "skill": "s"} for i in range(63)]
    fit = {p["id"]: {"items": []} for p in parts}
    pc.compile_group((torch.nn.Linear(2, 2), tiny_tokenizer()), (None, None), parts, fit, {}, {"stage1_steps": 300}, False)
    assert seen["components_per_step"] == 3
    assert seen["stage1_steps"] == 2100
    assert seen["stage2_steps"] == 0
    seen.clear()
    pc.compile_group((torch.nn.Linear(2, 2), tiny_tokenizer()), (None, None), parts[:3], fit, {}, {"stage1_steps": 300}, False)
    assert seen["stage1_steps"] == 300


# --------------------------------------------------------------- tests: modal_round23


@pytest.fixture(scope="module")
def r23():
    pytest.importorskip("modal")
    import modal_round23

    return modal_round23


def cell(key, donor, base):
    return (key, donor, base, None, "", False)


def test_round23_refuses_a_big_fan_out_before_any_row_exists(r23, tmp_path):
    cells = [cell("a%d" % i, "d%d" % i, "b") for i in range(4)]
    rows = tmp_path / "rows.json"
    assert "run --smoke first" in r23.gate(cells, False, False, False, rows_path=str(rows))
    assert r23.gate(cells, False, False, True, rows_path=str(rows)) is None
    assert r23.gate(cells, True, False, False, rows_path=str(rows)) is None
    assert r23.gate(cells[:3], False, False, False, rows_path=str(rows)) is None
    rows.write_text("[]")
    assert r23.gate(cells, False, False, False, rows_path=str(rows)) is not None
    rows.write_text(json.dumps([{"id": "x"}]))
    assert r23.gate(cells, False, False, False, rows_path=str(rows)) is None
    assert "tf5 smoke" in r23.gate(cells[:1], False, True, False, rows_path=str(rows), marker_exists=lambda p: False)
    assert r23.gate(cells[:1], False, True, False, rows_path=str(rows), marker_exists=lambda p: True) is None


def test_round23_warms_shared_models_and_limits_containers(r23):
    cells = [cell("math", "Qwen/Math", "Qwen/Inst"), cell("coder", "Qwen/Coder", "Qwen/Inst"),
             cell("alone", "Other/Donor", "Other/Base")]
    assert r23.shared_models(cells) == ["Qwen/Inst"]
    assert r23.MAX_CONTAINERS == 4
    assert r23.MAX_PARTS_DEFAULT == 24
    assert "olmo2_7b" in r23.REFUSED
    assert "modal_round23.py" in open("tests/test_no_benchmark_training.py").read()
