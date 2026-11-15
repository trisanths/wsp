"""Benchmark-free measures of a compile: did the donor's behaviour carry over, and did the base survive.

Benchmarks are chosen per request, so the compiler cannot tune on one.
These probes need only text from wsp.healdata pools (held out from any
training):

  transfer      on a skill's text, how much of the gap between the base
                and the donor the written model closes. 0 is the base,
                1 is the donor.
  drift         on general text, the KL from the frozen base to the
                written model. 0 means general behaviour is unchanged.
  content gain  transfer of the real compile minus transfer of the same
                compile with random components. With several random
                controls, content_z gives the gap in standard deviations.

With a shared tokenizer the measure is the KL between next-token
distributions. Without one it is the loss on text the donor generated.
"""

import torch


def model_device(model):
    return next(model.parameters()).device


def encode(tokenizer, texts, max_length, device):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tokenizer.padding_side = side
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def token_mask(attention_mask):
    """Real tokens except the first one of each sequence (the attention sink carries odd states)."""
    mask = attention_mask.clone().float()
    mask[:, 0] = 0.0
    return mask


@torch.no_grad()
def logit_kl(teacher, student, tokenizer, texts, max_length=384, batch_size=8):
    """Mean KL(teacher || student) per token in nats, and the top-1 agreement rate.

    Both models read the same token ids, so they must share a tokenizer.
    When the vocabulary sizes differ only by unused tail rows, the shared
    prefix is compared. Returns {"kl": float, "agree": float, "tokens": int}.
    """
    teacher_device = model_device(teacher)
    student_device = model_device(student)
    total_kl = 0.0
    total_agree = 0.0
    total_tokens = 0.0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        ids, attention = encode(tokenizer, batch, max_length, teacher_device)
        t_logits = teacher(input_ids=ids, attention_mask=attention).logits.float()
        s_logits = student(input_ids=ids.to(student_device), attention_mask=attention.to(student_device)).logits
        s_logits = s_logits.float().to(teacher_device)
        vocab = min(t_logits.shape[-1], s_logits.shape[-1])
        t = torch.log_softmax(t_logits[..., :vocab], dim=-1)
        s = torch.log_softmax(s_logits[..., :vocab], dim=-1)
        mask = token_mask(attention)
        kl = (t.exp() * (t - s)).sum(-1)
        total_kl += float((kl * mask).sum())
        same = (t.argmax(-1) == s.argmax(-1)).float()
        total_agree += float((same * mask).sum())
        total_tokens += float(mask.sum())
    if total_tokens == 0:
        return {"kl": 0.0, "agree": 0.0, "tokens": 0}
    return {"kl": total_kl / total_tokens, "agree": total_agree / total_tokens, "tokens": int(total_tokens)}


@torch.no_grad()
def text_nll(model, tokenizer, texts, max_length=384, batch_size=8):
    """Mean next-token loss of a model on texts, in nats per token. Returns {"nll": float, "tokens": int}."""
    device = model_device(model)
    total = 0.0
    count = 0.0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        ids, attention = encode(tokenizer, batch, max_length, device)
        logits = model(input_ids=ids, attention_mask=attention).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        target = ids[:, 1:]
        picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mask = attention[:, 1:].float()
        total += float((-picked * mask).sum())
        count += float(mask.sum())
    if count == 0:
        return {"nll": 0.0, "tokens": 0}
    return {"nll": total / count, "tokens": int(count)}


@torch.no_grad()
def donor_samples(donor, donor_tokenizer, prompts, max_new_tokens=128, batch_size=8):
    """Return the donor's own greedy answers to chat prompts, as plain answer strings."""
    device = model_device(donor)
    if donor_tokenizer.pad_token is None:
        donor_tokenizer.pad_token = donor_tokenizer.eos_token
    side = donor_tokenizer.padding_side
    donor_tokenizer.padding_side = "left"
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = []
        for prompt in prompts[start:start + batch_size]:
            if getattr(donor_tokenizer, "chat_template", None) is not None:
                batch.append(donor_tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                                 tokenize=False, add_generation_prompt=True))
            else:
                batch.append(prompt)
        enc = donor_tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        ids = donor.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=donor_tokenizer.pad_token_id)
        for row in range(ids.shape[0]):
            out.append(donor_tokenizer.decode(ids[row, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    donor_tokenizer.padding_side = side
    return out


def gap_closed(base_value, written_value, donor_value):
    """Share of the base-to-donor gap that the written model closes, for a lower-is-better measure.

    0 means no better than the base and 1 means equal to the donor. The
    value can be negative (worse than the base). When the donor is not
    better than the base on this text, there is no gap and the result is
    None.
    """
    gap = base_value - donor_value
    if gap <= 1e-9:
        return None
    return (base_value - written_value) / gap


def transfer(written, base, donor, tokenizer, skill_texts, max_length=384, batch_size=8):
    """Transfer on one skill with a shared tokenizer. Returns a dict with the KLs and the gap closed.

    KL(donor || base) is the gap. KL(donor || written) is what is left.
    """
    base_kl = logit_kl(donor, base, tokenizer, skill_texts, max_length, batch_size)
    written_kl = logit_kl(donor, written, tokenizer, skill_texts, max_length, batch_size)
    return {"donor_kl_base": base_kl["kl"], "donor_kl_written": written_kl["kl"],
            "agree_base": base_kl["agree"], "agree_written": written_kl["agree"],
            "transfer": gap_closed(base_kl["kl"], written_kl["kl"], 0.0), "tokens": written_kl["tokens"]}


def transfer_by_text(written, base, tokenizer, donor_texts, donor_self_nll=None, max_length=384, batch_size=8):
    """Transfer on one skill without a shared tokenizer, from text the donor generated.

    donor_texts are full chat texts whose answers the donor wrote, rendered
    in the base's chat format. donor_self_nll is the donor's own loss on
    its text when known, else 0 is used as the floor.
    """
    base_nll = text_nll(base, tokenizer, donor_texts, max_length, batch_size)
    written_nll = text_nll(written, tokenizer, donor_texts, max_length, batch_size)
    floor = 0.0
    if donor_self_nll is not None:
        floor = donor_self_nll
    return {"nll_base": base_nll["nll"], "nll_written": written_nll["nll"], "nll_donor": donor_self_nll,
            "transfer": gap_closed(base_nll["nll"], written_nll["nll"], floor), "tokens": written_nll["tokens"]}


def drift(written, base, tokenizer, general_texts, max_length=384, batch_size=8):
    """KL(base || written) on general text, and top-1 agreement. 0 KL means the base's behaviour is kept."""
    out = logit_kl(base, written, tokenizer, general_texts, max_length, batch_size)
    return {"base_kl": out["kl"], "agree": out["agree"], "tokens": out["tokens"]}


def report(written, base, tokenizer, skills, general_texts, random_written=None, max_length=384, batch_size=8):
    """Probe one compile. skills maps a skill name to {"donor": model, "texts": [...]}.

    Returns {"skills": {name: transfer dict (+ "random_transfer", "content_gain" when random_written is given)},
    "drift": drift dict, "verdict": {name: "contributes" | "no better than random" | "no transfer" | "harmful"}}.
    """
    out = {"skills": {}, "drift": drift(written, base, tokenizer, general_texts, max_length, batch_size),
           "verdict": {}}
    for name in skills:
        entry = skills[name]
        row = transfer(written, base, entry["donor"], tokenizer, entry["texts"], max_length, batch_size)
        if random_written is not None:
            control = transfer(random_written, base, entry["donor"], tokenizer, entry["texts"], max_length,
                               batch_size)
            row["random_transfer"] = control["transfer"]
            if row["transfer"] is not None and control["transfer"] is not None:
                row["content_gain"] = row["transfer"] - control["transfer"]
        out["skills"][name] = row
        out["verdict"][name] = verdict(row)
    return out


def content_z(real_transfer, random_transfers):
    """How far the real compile stands above several random-component compiles, in standard deviations.

    One random control cannot separate content from seed noise. With two
    or more controls this returns {"mean", "std", "z", "controls"}; z is
    None when the controls do not vary. With fewer than two it returns
    None.
    """
    values = []
    for value in random_transfers:
        if value is not None:
            values.append(float(value))
    if real_transfer is None or len(values) < 2:
        return None
    mean = sum(values) / len(values)
    spread = 0.0
    for value in values:
        spread += (value - mean) ** 2
    std = (spread / (len(values) - 1)) ** 0.5
    z = None
    if std > 1e-9:
        z = (real_transfer - mean) / std
    return {"mean": mean, "std": std, "z": z, "controls": len(values)}


def verdict(row, min_transfer=0.05, min_gain=0.03, min_z=2.0):
    """One word for a skill's row: does the written model carry the donor's behaviour, beyond random controls.

    When the row has "content_z" (several random controls), the real
    compile must stand min_z standard deviations above them. With one
    control ("content_gain") it must beat it by min_gain.
    """
    value = row.get("transfer")
    if value is None:
        return "no gap"
    if value < -min_transfer:
        return "harmful"
    if value < min_transfer:
        return "no transfer"
    stats = row.get("content_z")
    if stats is not None and stats.get("z") is not None:
        if stats["z"] < min_z:
            return "no better than random"
        return "contributes"
    gain = row.get("content_gain")
    if gain is not None and gain < min_gain:
        return "no better than random"
    return "contributes"
