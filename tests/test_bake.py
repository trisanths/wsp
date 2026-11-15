"""Tests for wsp.bake: distill a tapped host into a plain host LoRA."""

import copy
import json

import pytest
import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors

transformers = pytest.importorskip("transformers")

from wsp.bake import (bake, bare_host_copy, distill_kl, is_plain, mean_kl, save_baked, teacher_answer_logits,
                      teacher_sequences)
from wsp.lora import LoRALinear
from wsp.tap import TapBranch, attach_taps, prepare, clear_tap_context

WORDS = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat", "is", "what"]
PROMPTS = ["the cat sat on a mat", "a dog ran", "what is the big red hat", "the dog sat on the hat",
           "a big cat ran to the mat", "what is a red dog"]


def host_tokenizer():
    """Word-level, with a BOS token, like Llama."""
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<s>": 3}
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 3)])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>", bos_token="<s>",
    )


def donor_tokenizer():
    """Character-level, with no BOS token, like Qwen. Spaces are dropped."""
    vocab = {"<unk>": 0, "<|end|>": 1}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        vocab[letter] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Split(Regex("."), behavior="isolated"),
    ])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="<|end|>", pad_token="<|end|>",
    )


def tiny_host(seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=24, max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(config).eval()


def tiny_donor(seed=1):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=24, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def make_donors():
    return {"d": (tiny_donor(), donor_tokenizer())}


def tapped_teacher(seed=5):
    """A tiny host with one tap that changes its outputs a lot, like a trained tap."""
    host = tiny_host()
    torch.manual_seed(seed)
    branch = TapBranch(32, 24, attn_dim=16, heads=4, donor_id="d", donor_layer=2)
    with torch.no_grad():
        branch.gate.fill_(1.0)
        branch.o.weight.mul_(20.0)
    attach_taps(host, [(1, branch, 0)])
    return host


def plain_logits(model, tokenizer, prompts):
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        return model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits


def test_bare_copy_has_no_taps_and_keeps_the_teacher():
    tok = host_tokenizer()
    host = tiny_host()
    bare_expected = plain_logits(host, tok, PROMPTS)
    donors = make_donors()
    teacher = tapped_teacher()
    teacher_before = teacher_answer_logits(teacher, tok, donors, PROMPTS)[0]

    student = bare_host_copy(teacher)
    assert is_plain(student)
    assert not hasattr(student, "wsp_taps")
    for module in student.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0
    # The copy is the bare host and runs with no donors.
    assert torch.equal(plain_logits(student, tok, PROMPTS), bare_expected)

    # The teacher keeps its taps, and its outputs do not change.
    assert hasattr(teacher, "wsp_taps")
    teacher_after = teacher_answer_logits(teacher, tok, donors, PROMPTS)[0]
    assert torch.equal(teacher_after, teacher_before)
    assert not torch.allclose(teacher_after, bare_expected[:, -1, :], atol=1e-3)


def test_answer_bake_lowers_kl_and_needs_no_donors():
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    teacher_state = copy.deepcopy(teacher.state_dict())
    before = mean_kl(teacher, bare_host_copy(teacher), tok, donors, PROMPTS)

    result = bake(teacher, tok, donors, PROMPTS, rank=4, steps=40, lr=1e-2, batch_size=4, mode="answer",
                  seed=0, log_every=0)
    student = result["student"]
    after = mean_kl(teacher, student, tok, donors, PROMPTS)
    assert len(result["losses"]) == 40
    assert result["seconds"] > 0
    assert after < 0.3 * before

    # The student has no taps and runs with plain input ids.
    assert not hasattr(student, "wsp_taps")
    assert not hasattr(student, "wsp_tap_donors")
    logits = plain_logits(student, tok, PROMPTS)
    assert torch.isfinite(logits).all()

    # Only LoRA trained: the teacher did not change.
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, teacher_state[key]), key


def test_merge_gives_plain_model_with_same_logits():
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    result = bake(teacher, tok, donors, PROMPTS, rank=4, steps=10, lr=1e-2, batch_size=4, log_every=0)
    student = result["student"]
    lora_count = 0
    for module in student.modules():
        if isinstance(module, LoRALinear):
            lora_count += 1
    assert lora_count > 0
    unmerged = plain_logits(student, tok, PROMPTS)

    merged = copy.deepcopy(student)
    from wsp.lora import merge_lora

    merge_lora(merged)
    assert is_plain(merged)
    assert type(merged) is transformers.LlamaForCausalLM
    assert torch.allclose(plain_logits(merged, tok, PROMPTS), unmerged, atol=1e-5)

    # merge=True in bake gives the same kind of plain model.
    again = bake(teacher, tok, donors, PROMPTS, rank=4, steps=10, lr=1e-2, batch_size=4, merge=True,
                 log_every=0)["student"]
    assert is_plain(again)
    assert torch.allclose(plain_logits(again, tok, PROMPTS), unmerged, atol=1e-5)


def test_save_baked_writes_a_plain_model(tmp_path):
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    result = bake(teacher, tok, donors, PROMPTS, rank=4, steps=5, lr=1e-2, batch_size=4, log_every=0)
    student = result["student"]
    expected = plain_logits(student, tok, PROMPTS)
    save_baked(student, tok, str(tmp_path), info=result["info"])
    loaded = transformers.LlamaForCausalLM.from_pretrained(str(tmp_path)).eval()
    assert torch.allclose(plain_logits(loaded, tok, PROMPTS), expected, atol=1e-5)
    assert (tmp_path / "bake.json").exists()


def test_option_ids_restrict_the_distribution():
    torch.manual_seed(0)
    options = [4, 7, 9]
    student = torch.randn(3, 24)
    teacher = torch.randn(3, 24)
    loss = distill_kl(student, teacher, option_ids=options)

    # Logits outside the options do not change the loss.
    other_student = student.clone()
    other_teacher = teacher.clone()
    for index in range(24):
        if index not in options:
            other_student[:, index] += torch.randn(3) * 5
            other_teacher[:, index] += torch.randn(3) * 5
    assert torch.allclose(distill_kl(other_student, other_teacher, option_ids=options), loss)
    assert not torch.allclose(distill_kl(other_student, other_teacher), distill_kl(student, teacher))

    # Same option logits give zero loss, even when the rest of the vocab differs.
    same = teacher.clone()
    same[:, 0] += 10.0
    assert float(distill_kl(same, teacher, option_ids=options)) < 1e-6
    assert float(distill_kl(same, teacher)) > 1e-3

    # Baking with option ids matches the teacher over the options.
    tok = host_tokenizer()
    donors = make_donors()
    tapped = tapped_teacher()
    before = mean_kl(tapped, bare_host_copy(tapped), tok, donors, PROMPTS, option_ids=options)
    result = bake(tapped, tok, donors, PROMPTS, rank=4, steps=40, lr=1e-2, batch_size=4, option_ids=options,
                  log_every=0)
    after = mean_kl(tapped, result["student"], tok, donors, PROMPTS, option_ids=options)
    assert after < 0.3 * before
    assert result["info"]["option_ids"] == options


def test_teacher_sequences_match_generate_and_mask_after_eos():
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    prompts = PROMPTS[:3]
    found = teacher_sequences(teacher, tok, donors, prompts, max_new_tokens=5)

    tok.padding_side = "left"
    inputs = prepare(teacher, tok, donors, prompts)
    with torch.no_grad():
        generated = teacher.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id)
    clear_tap_context(teacher)
    prompt_len = inputs["input_ids"].shape[1]
    kept = found["attention_mask"][:, prompt_len:]
    for row in range(len(prompts)):
        for index in range(kept.shape[1]):
            if int(kept[row, index]) == 1:
                assert int(found["input_ids"][row, prompt_len + index]) == int(generated[row, prompt_len + index])

    # The teacher logits along the sequence predict the greedy tokens.
    weights = found["weights"]
    for row in range(len(prompts)):
        for position in range(weights.shape[1]):
            if float(weights[row, position]) == 1.0:
                predicted = int(found["logits"][row, position].argmax())
                assert predicted == int(found["input_ids"][row, position + 1])
    assert float(weights[:, :prompt_len - 1].sum()) == 0.0


def test_sequence_bake_lowers_kl_over_continuations():
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    found = teacher_sequences(teacher, tok, donors, PROMPTS, max_new_tokens=4)

    def continuation_kl(student):
        with torch.no_grad():
            positions = found["attention_mask"].long().cumsum(-1) - 1
            positions = positions.masked_fill(found["attention_mask"] == 0, 1)
            logits = student(input_ids=found["input_ids"], attention_mask=found["attention_mask"],
                             position_ids=positions).logits
        return float(distill_kl(logits, found["logits"], weights=found["weights"]))

    before = continuation_kl(bare_host_copy(teacher))
    result = bake(teacher, tok, donors, PROMPTS, rank=4, steps=40, lr=1e-2, batch_size=6, mode="sequence",
                  max_new_tokens=4, log_every=0)
    after = continuation_kl(result["student"])
    assert after < 0.5 * before
    assert result["info"]["max_new_tokens"] == 4


def test_bad_inputs_raise():
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    with pytest.raises(ValueError):
        bake(teacher, tok, donors, PROMPTS, mode="other", steps=1)
    with pytest.raises(ValueError):
        bake(teacher, tok, donors, PROMPTS, mode="sequence", option_ids=[4], steps=1)
    with pytest.raises(ValueError):
        bake(tiny_host(), tok, donors, PROMPTS, steps=1)
    with pytest.raises(ValueError):
        bake(teacher, tok, donors, PROMPTS, student_host=tapped_teacher(), steps=1)
    with pytest.raises(ValueError):
        bake(teacher, tok, donors, PROMPTS, seq_len=2, steps=1)


def test_pairs_use_only_the_prompt():
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    pairs = []
    for prompt in PROMPTS:
        pairs.append((prompt, " the"))
    one = bake(teacher, tok, donors, PROMPTS, rank=4, steps=3, lr=1e-2, batch_size=4, log_every=0)
    two = bake(teacher, tok, donors, pairs, rank=4, steps=3, lr=1e-2, batch_size=4, log_every=0)
    assert one["losses"] == two["losses"]


def test_generator_prompts_work():
    # A generator has no len, so bake must read it into a list first.
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    prompts = (prompt for prompt in PROMPTS)
    result = bake(teacher, tok, donors, prompts, rank=2, steps=2, batch_size=2, log_every=0)
    assert result["info"]["prompts_given"] == len(PROMPTS)


def test_bake_keeps_teacher_and_donor_requires_grad():
    # Tap training must still work on the teacher after a bake.
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    teacher_flags = []
    for param in teacher.parameters():
        teacher_flags.append(param.requires_grad)
    donor_flags = []
    for param in donors["d"][0].parameters():
        donor_flags.append(param.requires_grad)
    bake(teacher, tok, donors, PROMPTS, rank=2, steps=1, batch_size=2, log_every=0)
    after = []
    for param in teacher.parameters():
        after.append(param.requires_grad)
    assert after == teacher_flags
    after = []
    for param in donors["d"][0].parameters():
        after.append(param.requires_grad)
    assert after == donor_flags


def test_sequences_stop_on_generation_config_eos():
    # Instruct models stop on more than one id. Tokens after any of them are masked.
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    with torch.no_grad():
        teacher.lm_head.weight.mul_(60.0)
    plain = teacher_sequences(teacher, tok, donors, PROMPTS, max_new_tokens=6)
    prompt_len = len(tok(PROMPTS, padding=True)["input_ids"][0])

    # Pick a token the teacher writes mid-continuation and make it a stop id.
    stop = int(plain["input_ids"][0, prompt_len])
    teacher.generation_config.eos_token_id = [tok.eos_token_id, stop]
    found = teacher_sequences(teacher, tok, donors, PROMPTS, max_new_tokens=6)
    ids = found["input_ids"]
    mask = found["attention_mask"]
    checked = 0
    for row in range(ids.shape[0]):
        stopped = False
        for index in range(prompt_len, ids.shape[1]):
            if stopped:
                assert int(mask[row, index]) == 0
                assert float(found["weights"][row, index - 1]) == 0.0
                checked += 1
            if int(mask[row, index]) == 1 and int(ids[row, index]) in (tok.eos_token_id, stop):
                stopped = True
    assert checked > 0


def test_tensor_option_ids_save_as_json(tmp_path):
    tok = host_tokenizer()
    donors = make_donors()
    teacher = tapped_teacher()
    options = torch.tensor([4, 7, 9])
    result = bake(teacher, tok, donors, PROMPTS, rank=2, steps=2, batch_size=2, option_ids=options, log_every=0)
    assert result["info"]["option_ids"] == [4, 7, 9]
    save_baked(result["student"], tok, str(tmp_path), info=result["info"])
    with open(tmp_path / "bake.json") as f:
        saved = json.load(f)
    assert saved["info"]["option_ids"] == [4, 7, 9]
