"""Helpers for the clinic write-in track (modal_wi_clinic.py, docs/write_in_clinic.md).

Prompt builders for the medical multiple-choice sets, the soft-label mix over several donors, and the save and load of
a multi-donor parallel graft (so a demo can rebuild the trained model without retraining).
"""

import contextlib

import torch

LETTERS = ["A", "B", "C", "D"]
MAX_PROMPT_CHARS = 1400


def mcq_prompt(question, choices):
    """The shared multiple-choice prompt: question, lettered choices, then 'Answer:'."""
    lines = ["Question: " + str(question).strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def medqa_item(row):
    """(prompt, gold) from a GBaker/MedQA-USMLE-4-options row, or None when it cannot be read."""
    options = row.get("options")
    if not isinstance(options, dict) or any(k not in options for k in LETTERS):
        return None
    answer = row.get("answer_idx")
    if answer not in LETTERS or not row.get("question"):
        return None
    return mcq_prompt(row["question"], [options[k] for k in LETTERS]), LETTERS.index(answer)


def pubmedqa_item(row):
    """(prompt, gold) from a PubMedQA pqa_labeled row: three options (yes, no, maybe) in the letters A to C."""
    answer = row.get("final_decision")
    contexts = (row.get("context") or {}).get("contexts") or []
    if answer not in ("yes", "no", "maybe") or not row.get("question") or not contexts:
        return None
    body = " ".join(str(c).strip() for c in contexts)
    question = "Context: " + body + "\n" + str(row["question"]).strip()
    return mcq_prompt(question, ["yes", "no", "maybe"]), ("yes", "no", "maybe").index(answer)


def medmcqa_item(row):
    """(prompt, gold) from a MedMCQA row, or None for a row without a usable answer."""
    try:
        gold = int(row["cop"])
    except (KeyError, TypeError, ValueError):
        return None
    choices = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
    if gold not in range(4) or any(c is None for c in choices) or not row.get("question"):
        return None
    return mcq_prompt(row["question"], choices), gold


def filter_items(items, banned, seen=None, max_chars=MAX_PROMPT_CHARS):
    """Drop items with a banned (evaluation) prompt, a repeated prompt, or an overlong prompt. Order is kept."""
    seen = set() if seen is None else seen
    out = []
    for prompt, gold in items:
        if prompt in banned or prompt in seen or len(prompt) > max_chars:
            continue
        seen.add(prompt)
        out.append([prompt, int(gold)])
    return out


def mix_soft(softs, mode):
    """One soft letter distribution per item from several donors.

    softs maps a donor name to a list of probability rows. mode is a donor name (use that donor's rows) or "mean"
    (the renormalised average of every donor). Rows have equal length.
    """
    if mode != "mean":
        return [list(row) for row in softs[mode]]
    names = sorted(softs)
    out = []
    for rows in zip(*[softs[n] for n in names]):
        avg = [sum(r[j] for r in rows) / len(rows) for j in range(len(rows[0]))]
        total = sum(avg) or 1.0
        out.append([x / total for x in avg])
    return out


@contextlib.contextmanager
def only(grafts, keep):
    """Switch every graft off except grafts[keep] inside the context."""
    old = [g.forced_off for g in grafts]
    for i, g in enumerate(grafts):
        g.forced_off = i != keep
    try:
        yield
    finally:
        for g, o in zip(grafts, old):
            g.forced_off = o


def graft_state(grafts, lora_params=()):
    """CPU copy of everything a trained graft set holds that is not the frozen donor or host weights."""
    state = {"grafts": [], "lora": [p.detach().cpu().clone() for p in lora_params]}
    for g in grafts:
        state["grafts"].append({
            "enc": {k: v.detach().cpu().clone() for k, v in g.w_in.state_dict().items()},
            "dec": {k: v.detach().cpu().clone() for k, v in g.w_out.state_dict().items()},
            "gate": {k: v.detach().cpu().clone() for k, v in g.token_gate.state_dict().items()},
            "in_scale": g.in_scale.detach().cpu().clone(),
        })
    return state


def load_graft_state(grafts, state, lora_params=()):
    """Inverse of graft_state, into grafts built the same way and LoRA parameters in the same order."""
    if len(state["grafts"]) != len(grafts) or len(state["lora"]) != len(lora_params):
        raise ValueError("graft count or LoRA parameter count differs from the saved state")
    with torch.no_grad():
        for g, s in zip(grafts, state["grafts"]):
            g.w_in.load_state_dict(s["enc"])
            g.w_out.load_state_dict(s["dec"])
            g.token_gate.load_state_dict(s["gate"])
            g.in_scale.copy_(s["in_scale"])
        for p, v in zip(lora_params, state["lora"]):
            p.copy_(v.to(p.dtype))


def train_accum(base, groups, steps, step_fn, seed=0, warmup=30, log_every=100):
    """Like pgraft.train, but step_fn calls backward itself on each loss term and returns an info dict.

    The gradient is the same sum of terms. Backward per term frees each term's activations before the next term runs,
    which keeps two donor slices in one training step inside one GPU. zero_grad runs before step_fn, so the gradients
    from every term add up until the optimiser step. info must hold "loss" (the summed value of the terms).
    """
    import math

    torch.manual_seed(seed)
    groups = [g for g in groups if len(g["params"]) > 0]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    log = []
    base.train()
    try:
        for step in range(steps):
            opt.zero_grad()
            info = dict(step_fn(step))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            info["step"] = step
            log.append(info)
            if log_every and step % log_every == 0:
                print("train " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                          for k, v in info.items()), flush=True)
    finally:
        base.eval()
    return log
