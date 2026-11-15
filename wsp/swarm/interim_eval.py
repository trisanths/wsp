"""The interim GSM8K exact-match scorer.

This is the same logic as final_number and score in modal_graft.py
(Trisanth's first graft run), so part 3 numbers line up with his until
Person B's eval harness lands. Every score this module returns carries
"eval_backend": "interim" and "label": "INTERIM". Never report one as a
final result.

Why the logic is copied and not imported: modal_graft.py builds its own
Modal app ("wsp-graft") and volume ("wsp-models") when it is imported, and
it is not mounted in the wsp-swarm container. Importing it there would be
neither clean nor possible. tests/test_swarm_modal.py checks that
final_number here gives the same answers as modal_graft.final_number.

Sources (modal_graft.py as last changed in commit 2ce7ddd):
- final_number: lines 40-54, copied unchanged.
- score: lines 57-83. The prompt, the left padding, greedy generation
  with max_new_tokens 256, the split on "Question:", and the 1e-6 match
  are unchanged. Two additions: the per-item check is its own function
  (item_correct) so it can be tested on plain strings, and score() takes
  use_cache, which defaults to the original behavior (attn_heads grafts
  need use_cache=False, per the brief).
"""

import re

EVAL_BACKEND = "interim"
LABEL = "INTERIM"
TASK = "gsm8k"


def final_number(text):
    """Return the number after '####', else the last number in the text."""
    # modal_graft.py lines 40-54, unchanged.
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if match is None:
        numbers = re.findall(r"-?[\d,]*\.?\d+", text)
        if len(numbers) == 0:
            return None
        value = numbers[-1]
    else:
        value = match.group(1)
    value = value.replace(",", "").rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


def prompt_for(item):
    """The prompt modal_graft.py line 69 builds for one GSM8K item."""
    return "Question: " + item["question"] + "\nAnswer:"


def trim_completion(text):
    """Cut a completion where the model starts a new question (modal_graft.py line 76)."""
    return text.split("Question:")[0]


def item_correct(completion, gold_answer):
    """True when the completion's final number matches the gold answer's.

    completion is the decoded text after the prompt. This is modal_graft.py
    lines 76-80 as one function.
    """
    text = trim_completion(completion)
    predicted = final_number(text)
    gold = final_number(gold_answer)
    if predicted is not None and gold is not None and abs(predicted - gold) < 1e-6:
        return True
    return False


def labeled_result(correct, n_items, samples):
    """The result dict every scorer here returns, labeled interim."""
    if n_items == 0:
        raise ValueError("cannot score zero items")
    return {
        "accuracy": correct / n_items,
        "correct": correct,
        "n_items": n_items,
        "samples": samples,
        "task": TASK,
        "eval_backend": EVAL_BACKEND,
        "label": LABEL,
    }


def score_completions(completions, items):
    """Score decoded completions against GSM8K items. No model needed.

    completions[i] is the text the model wrote after prompt_for(items[i]).
    Keeps the first 3 trimmed completions (300 characters each) as samples,
    like modal_graft.py lines 81-82.
    """
    if len(completions) != len(items):
        raise ValueError("got " + str(len(completions)) + " completions for " + str(len(items)) + " items")
    correct = 0
    samples = []
    for index in range(len(items)):
        if item_correct(completions[index], items[index]["answer"]):
            correct += 1
        if len(samples) < 3:
            samples.append(trim_completion(completions[index])[:300])
    return labeled_result(correct, len(items), samples)


def score(model, tokenizer, items, batch_size=32, max_new_tokens=256, use_cache=True):
    """Generate greedily for each item and return the labeled interim result.

    Same generation as modal_graft.py lines 57-83. use_cache=False is only
    passed to generate when asked for, so the default call is unchanged.
    """
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    completions = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            prompts.append(prompt_for(item))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        options = {"max_new_tokens": max_new_tokens, "do_sample": False, "pad_token_id": tokenizer.pad_token_id}
        if not use_cache:
            options["use_cache"] = False
        with torch.no_grad():
            out = model.generate(**encoded, **options)
        prompt_length = encoded.input_ids.shape[1]
        for i in range(len(batch)):
            completions.append(tokenizer.decode(out[i, prompt_length:], skip_special_tokens=True))
    return score_completions(completions, items)
