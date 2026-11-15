"""wsp.xfam: spec, glue count, training on tiny models, artifact round trip. CPU only, no network.

The host is Llama-like with one tokenizer, the donor Qwen-like with another (different words and vocab size), so the
cross-tokenizer alignment path runs.
"""

import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from wsp import xfam  # noqa: E402

HOST_WORDS = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat", "Question", "Answer", "A", "B", "C", "D"]
DONOR_WORDS = ["hat", "red", "big", "to", "ran", "dog", "mat", "a", "on", "sat", "cat", "the", "extra1", "extra2",
               "Answer", "Question", "D", "C", "B", "A"]


def make_tok(words):
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for w in words:
        vocab[w] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def tiny_host(dtype=torch.float32):
    torch.manual_seed(1)
    config = transformers.LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                                      num_key_value_heads=2, vocab_size=40, max_position_embeddings=128)
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).to(dtype).eval()


def tiny_donor(dtype=torch.float32):
    torch.manual_seed(2)
    config = transformers.Qwen2Config(hidden_size=48, intermediate_size=96, num_hidden_layers=7, num_attention_heads=4,
                                      num_key_value_heads=2, vocab_size=60, max_position_embeddings=128)
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def items(n=48):
    words = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat"]
    out = []
    for i in range(n):
        q = " ".join(words[(i * 5 + j * 3) % 12] for j in range(6))
        choices = "\n".join("%s %s" % ("ABCD"[k], words[(i + k * 7) % 12]) for k in range(4))
        out.append(["Question %s\n%s\nAnswer" % (q, choices), i % 4])
    return out


def texts():
    words = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat"]
    return [" ".join(words[(i * 7 + j * 3) % 12] for j in range(10 + i % 5)) for i in range(40)]


SPEC_SMALL = {"steps": 6, "align_steps": 4, "geo_steps": 2, "batch": 4, "enc_depth": 2, "dec_rank": 8, "lora_rank": 2,
              "host_layer": 1, "first": 2, "last": 4}


def small_spec(host, donor):
    base_cfg = {"layers": 4, "hidden": 32}
    donor_cfg = {"layers": 7, "hidden": 48}
    spec = xfam.spec_for(base_cfg, donor_cfg, SPEC_SMALL)
    spec.update(seq_len=24, donor_seq_len=32)
    return spec


def pool_for(n=48):
    its = items(n)
    soft = [[0.7 if k == g else 0.1 for k in range(4)] for _, g in its]
    return {"items": its, "soft": soft}


def run_arm(arm="graft", dtype=torch.float32, seed=3):
    host_tok, donor_tok = make_tok(HOST_WORDS), make_tok(DONOR_WORDS)
    base, donor = tiny_host(dtype), tiny_donor(dtype)
    spec = small_spec(base, donor)
    t = texts()
    data = {"align": t[:24], "align_eval": t[24:32], "general": t[32:] * 3}
    built = xfam.train_network(base, host_tok, donor, donor_tok, pool_for(), data, spec, seed, arm=arm, log=lambda *a: None)
    return base, donor, host_tok, donor_tok, spec, built


def test_spec_follows_the_measured_recipe():
    spec = xfam.spec_for({"layers": 16, "hidden": 2048}, {"layers": 28, "hidden": 3584})
    assert (spec["host_layer"], spec["first"], spec["last"]) == (7, 4, 20)
    assert spec["steps"] == 4000 and spec["align_steps"] == 600 and spec["lora_rank"] == 16
    # the trainable Enc, Dec and gate count of the stored wi_xfam rows
    assert xfam.glue_params(spec) == 15214466


def test_spec_refuses_a_span_outside_the_donor():
    with pytest.raises(ValueError):
        xfam.spec_for({"layers": 4, "hidden": 8}, {"layers": 28, "hidden": 8}, {"first": 20, "last": 40})


def test_letter_scoring_matches_the_round_6_function():
    mod = pytest.importorskip("modal_round6")
    tok = make_tok(HOST_WORDS)
    base = tiny_host()
    its = items(20)
    assert xfam.score_letters(base, tok, its, bs=7) == mod.score_mmlu(base, tok, [tuple(x) for x in its], batch_size=7)


def test_graft_trains_and_counts_match_the_spec():
    base, donor, tok, dtok, spec, built = run_arm()
    row = built.row
    assert row["trainable_glue"] == xfam.glue_params(spec)
    assert row["slice_params"] == sum(p.numel() for p in built.grafts[0].part.parameters()) > 0
    assert row["params_per_token"] == row["params_base"] + row["slice_params"] + row["trainable_glue"]
    assert row["train_steps"] == spec["steps"]
    assert row["alignment"] is not None and row["tokens_trained"] > 0


def test_controls_spend_the_alignment_steps_on_the_task():
    _, _, _, _, spec, built = run_arm("identity")
    assert built.row["train_steps"] == spec["steps"] + spec["align_steps"]
    assert built.row["slice_params"] == 0
    _, _, _, _, spec, built = run_arm("scrambled")
    assert built.row["train_steps"] == spec["steps"] + spec["align_steps"] and built.row["slice_params"] > 0


def test_artifact_round_trip_gives_the_same_logits(tmp_path):
    base, donor, tok, dtok, spec, built = run_arm()
    path = xfam.save_artifact(str(tmp_path), base, built, spec, {"host": "tiny-host", "donor": "tiny-donor"})
    assert os.path.basename(path) == "pgraft_v1"
    meta = xfam.read_meta(path)
    assert meta["format_version"] == 1 and meta["donor_span"] == [2, 4]
    # the artifact holds glue and LoRA only, not the donor slice
    size = sum(os.path.getsize(os.path.join(path, f)) for f in ("glue.safetensors", "lora.safetensors"))
    donor_bytes = sum(p.numel() * 4 for p in donor.parameters())
    assert size < donor_bytes
    enc = tok([x[0] for x in items(6)], return_tensors="pt", padding=True)
    with torch.no_grad():
        want = base(**enc).logits
    fresh_base, fresh_donor = tiny_host(), tiny_donor()
    model, _ = xfam.load_artifact(path, base=fresh_base, donor=fresh_donor, device="cpu")
    with torch.no_grad():
        got = model(**enc).logits
    assert torch.allclose(want, got, atol=1e-5)
    assert len(model.wsp_grafts) == 1
    its = items(16)
    assert xfam.score_letters(base, tok, its) == xfam.score_letters(model, tok, its)


def test_the_loader_refuses_a_wrong_host_and_a_wrong_version(tmp_path):
    base, donor, tok, dtok, spec, built = run_arm()
    path = xfam.save_artifact(str(tmp_path), base, built, spec, {"host": "h", "donor": "d"})
    other = tiny_host()
    torch.manual_seed(99)
    other.get_input_embeddings().weight.data.normal_()
    with pytest.raises(ValueError, match="does not match"):
        xfam.load_artifact(path, base=other, donor=tiny_donor(), device="cpu")
    meta_path = os.path.join(path, "meta.json")
    meta = json.load(open(meta_path))
    meta["format_version"] = 2
    json.dump(meta, open(meta_path, "w"))
    with pytest.raises(ValueError, match="version"):
        xfam.read_meta(path)


def test_save_needs_host_and_donor_ids(tmp_path):
    base, donor, tok, dtok, spec, built = run_arm()
    with pytest.raises(ValueError, match="needs"):
        xfam.save_artifact(str(tmp_path), base, built, spec, {})


def test_loader_reads_models_from_disk(tmp_path):
    base, donor, tok, dtok, spec, built = run_arm()
    # the host on disk is the frozen host, so save a fresh copy of the same weights
    host_dir, donor_dir = tmp_path / "host", tmp_path / "donor"
    fresh_host = tiny_host()
    fresh_host.save_pretrained(host_dir)
    donor.save_pretrained(donor_dir)
    path = xfam.save_artifact(str(tmp_path / "out"), base, built, spec, {"host": str(host_dir), "donor": str(donor_dir)})
    model, meta = xfam.load_artifact(path, device="cpu", dtype=torch.float32)
    enc = tok([x[0] for x in items(4)], return_tensors="pt", padding=True)
    with torch.no_grad():
        assert torch.allclose(base(**enc).logits, model(**enc).logits, atol=1e-5)


def test_bf16_train_save_load_and_score(tmp_path):
    base, donor, tok, dtok, spec, built = run_arm(dtype=torch.bfloat16)
    path = xfam.save_artifact(str(tmp_path), base, built, spec, {"host": "h", "donor": "d"})
    model, _ = xfam.load_artifact(path, base=tiny_host(torch.bfloat16), donor=tiny_donor(torch.bfloat16), device="cpu")
    its = items(12)
    a, b = xfam.score_letters(base, tok, its), xfam.score_letters(model, tok, its)
    assert 0.0 <= a <= 1.0 and a == b
    probs = xfam.letter_probs(model, tok, its)
    assert all(abs(sum(p) - 1) < 1e-3 for p in probs)


def test_load_or_build_data_reads_the_cache(tmp_path):
    pool = pool_for(8)
    json.dump({"items": pool["items"], "soft": pool["soft"], "n_aux": 5, "donor_train_acc": 0.5}, open(tmp_path / "pool.json", "w"))
    json.dump({"mmlu_dev": items(4)}, open(tmp_path / "evals.json", "w"))
    data = xfam.load_or_build_data(str(tmp_path), "host")
    assert len(data["items"]) == 8 and data["source"].startswith("cache")
    with pytest.raises(FileNotFoundError):
        xfam.load_or_build_data(str(tmp_path / "none"), "host")
