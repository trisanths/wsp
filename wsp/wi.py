"""Helpers for the write-in tracks: same-tokenizer distillation, letter scoring, TriviaQA exact match.

Pure torch and string code, so the CPU tests run it. The drivers (modal_wi_*.py)
do the model loading and the training loop.
"""

import re
import string

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

LETTERS = ["A", "B", "C", "D", "E"]


def _kl_rows(teacher, student):
    t = torch.log_softmax(teacher.float(), dim=-1)
    s = torch.log_softmax(student.float(), dim=-1)
    return (t.exp() * (t - s)).sum(-1)


def kd_logits_loss(student_logits, teacher_logits, mask, chunk=512):
    """Mean KL(teacher || student) over masked positions of (batch, seq, vocab) logits.

    Both models share a tokenizer, so position i of one is position i of the
    other. The vocab axis is cut to the smaller of the two. Positions go
    through the softmax in checkpointed chunks, so the float copies of a
    152k-wide vocabulary never exist for all positions at once.
    """
    v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    keep = mask.bool()
    s = student_logits[..., :v][keep]
    t = teacher_logits[..., :v][keep].detach()
    n = s.shape[0]
    if n == 0:
        return student_logits.sum() * 0.0
    total = s.new_zeros((), dtype=torch.float32)
    for start in range(0, n, chunk):
        sc = s[start:start + chunk]
        tc = t[start:start + chunk]
        if sc.requires_grad:
            part = checkpoint(lambda a, b: _kl_rows(b, a).sum(), sc, tc, use_reentrant=False)
        else:
            part = _kl_rows(tc, sc).sum()
        total = total + part
    return total / n


def soft_letter_loss(student_letters, teacher_letters, temperature=1.0):
    """Mean KL(teacher || student) over the answer letters. Inputs are (batch, n_letters) logits."""
    t = torch.log_softmax(teacher_letters.float() / temperature, dim=-1)
    s = torch.log_softmax(student_letters.float() / temperature, dim=-1)
    return (t.exp() * (t - s)).sum(-1).mean()


def normalize_answer(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def first_line(text):
    """The part of a generated answer before the first newline, stripped."""
    return text.strip().split("\n")[0].strip()


def trivia_em(prediction, aliases):
    """Exact match of the first line of the prediction against any normalized alias."""
    p = normalize_answer(first_line(prediction))
    if not p:
        return False
    for alias in aliases:
        if p == normalize_answer(alias):
            return True
    return False


def mc_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def arc_item(row):
    """(prompt, gold index, n options) from an ai2_arc row, or None when the label is not found."""
    labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    if row["answerKey"] not in labels or len(texts) < 2 or len(texts) > len(LETTERS):
        return None
    return mc_prompt(row["question"], texts), labels.index(row["answerKey"]), len(texts)


def sciq_item(row, rng):
    """(prompt, gold index, 4) from an allenai/sciq row with shuffled options."""
    choices = [row["distractor1"], row["distractor2"], row["distractor3"], row["correct_answer"]]
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    return mc_prompt(row["question"], shuffled), order.index(3), 4


def letter_token_ids(tokenizer):
    out = []
    for letter in LETTERS:
        out.append(tokenizer(" " + letter, add_special_tokens=False).input_ids[-1])
    return out


@torch.no_grad()
def letter_logits(model, tokenizer, prompts, letter_ids, batch_size=16, device=None):
    """Float logits at the last prompt token over letter_ids, shape (len(prompts), len(letter_ids)).

    Left padding. Prompts are processed in length order and returned in the input order.
    """
    from wsp.stitch import autocast

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if device is None:
        device = next(model.parameters()).device
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = torch.zeros(len(prompts), len(letter_ids))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in chunk], return_tensors="pt", padding=True).to(device)
        with autocast(device):
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           use_cache=False).logits[:, -1, :].float()
        out[torch.tensor(chunk)] = logits[:, letter_ids].cpu()
    tokenizer.padding_side = old
    return out


def score_letters(model, tokenizer, items, letter_ids, batch_size=16):
    """Accuracy over (prompt, gold, n_options) items; the argmax is taken over the first n_options letters."""
    if not items:
        return 0.0
    logits = letter_logits(model, tokenizer, [i[0] for i in items], letter_ids, batch_size)
    right = 0
    for row, (_, gold, n) in zip(logits, items):
        if int(row[:n].argmax()) == gold:
            right += 1
    return right / len(items)
