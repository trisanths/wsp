"""Short healing runs: train a few parameters with a causal LM loss.

heal trains one parameter list with the plain LM loss. heal_groups is the
general loop: AdamW groups with their own learning rates, an optional
teacher for a distillation loss, an optional extra loss (for example the
router balance loss of a skill MoE), and an optional time limit.
distill_heal is heal_groups with a teacher and one learning rate.
"""

import time

import torch


def heal(model, tokenizer, texts, params, steps=300, lr=3e-4, batch_size=8, seq_len=384,
         seed=0, log_every=10, max_seconds=None):
    """Train only params on texts. Return the list of per-step losses.

    max_seconds: stop after this many seconds (None runs every step). The
    returned list is then shorter than steps.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_autocast = device.type == "cuda"

    model.train()
    losses = []
    order = []
    start = time.perf_counter()
    for step in range(steps):
        if max_seconds is not None and time.perf_counter() - start >= max_seconds:
            break
        batch = []
        while len(batch) < batch_size:
            if len(order) == 0:
                order = torch.randperm(len(texts), generator=generator).tolist()
            batch.append(texts[order.pop()])

        enc = tokenizer(batch, truncation=True, max_length=seq_len, padding=True,
                        return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        loss = float(out.loss.detach())
        losses.append(loss)
        if log_every and (step + 1) % log_every == 0:
            print(f"heal step {step + 1}/{steps} loss {loss:.4f}")

    model.eval()
    return losses


def check_same_tokenizer(student_tokenizer, teacher_tokenizer, teacher_id="the teacher"):
    """Raise ValueError unless both tokenizers map the same tokens to the same ids.

    A distillation heal compares the two models' next-token distributions
    position by position, so the ids must mean the same tokens.
    """
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if student_vocab == teacher_vocab:
        return
    differences = 0
    for token in teacher_vocab:
        if student_vocab.get(token) != teacher_vocab[token]:
            differences += 1
    raise ValueError(
        f"distillation needs the same tokenizer, but {teacher_id} differs from the student: "
        f"{len(teacher_vocab)} teacher tokens vs {len(student_vocab)} student tokens, "
        f"{differences} teacher tokens have another id or are missing. "
        "Use a teacher of the student's tokenizer family, or drop distill from the heal."
    )


def kd_loss(student_logits, teacher_logits, attention_mask, input_ids, kd_weight, temperature):
    """Return (total, kd, lm): kd_weight * KL(teacher || student) plus the LM loss.

    Both logit tensors are cut to the smaller vocabulary, since padded
    embedding tables of two sizes can share one tokenizer. The KL averages
    over real tokens, and the LM loss over real next-token targets.
    """
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    vocab = min(teacher_logits.shape[-1], student_logits.shape[-1])
    t = torch.log_softmax(teacher_logits[..., :vocab] / temperature, dim=-1)
    s = torch.log_softmax(student_logits[..., :vocab] / temperature, dim=-1)
    mask = attention_mask.float()
    kl = (t.exp() * (t - s)).sum(-1)
    kd = (kl * mask).sum() / mask.sum()
    labels = input_ids[:, 1:]
    lm = torch.nn.functional.cross_entropy(student_logits[:, :-1].reshape(-1, student_logits.shape[-1]),
                                           labels.reshape(-1), reduction="none")
    lm_mask = mask[:, 1:].reshape(-1)
    lm = (lm * lm_mask).sum() / lm_mask.sum()
    return kd_weight * kd + lm, kd, lm


def heal_groups(model, tokenizer, texts, groups, steps=300, batch_size=8, seq_len=384, seed=0, log_every=10,
                teacher=None, kd_weight=1.0, temperature=1.0, extra_loss=None, extra_weight=0.0,
                max_seconds=None, add_special_tokens=True, use_cache=False):
    """Train AdamW parameter groups on texts. Return a dict with the per-step losses.

    groups: a list of dicts with "params" and "lr" (and any other AdamW
        option, such as "eps").
    teacher: None for the LM loss, or a frozen model of the same tokenizer
        for kd_weight * KL(teacher || student) + LM loss.
    extra_loss: None, or a function model -> tensor (or None) that is added
        with extra_weight, for example wsp.skill_moe.load_balance_loss.
    max_seconds: stop after this many seconds (None runs every step).

    The result has "losses" (the LM loss per step, or the total with a
    teacher), "kd" (the KL per step, empty without a teacher), "extra"
    (the extra loss per step), "steps" (steps run), and "stopped_early".
    The batches are drawn like heal: seeded shuffled passes.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    every = []
    for group in groups:
        for param in group["params"]:
            every.append(param)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_autocast = device.type == "cuda"

    model.train()
    if teacher is not None:
        teacher.eval()
    losses = []
    kd_losses = []
    extra_losses = []
    order = []
    start = time.perf_counter()
    stopped = False
    for step in range(steps):
        if max_seconds is not None and time.perf_counter() - start >= max_seconds:
            stopped = True
            break
        batch = []
        while len(batch) < batch_size:
            if len(order) == 0:
                order = torch.randperm(len(texts), generator=generator).tolist()
            batch.append(texts[order.pop()])

        enc = tokenizer(batch, truncation=True, max_length=seq_len, padding=True, return_tensors="pt",
                        add_special_tokens=add_special_tokens)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        optimizer.zero_grad()
        if teacher is not None:
            teacher_device = next(teacher.parameters()).device
            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids.to(teacher_device),
                                         attention_mask=attention_mask.to(teacher_device)).logits.to(device)
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                           use_cache=use_cache).logits
            else:
                student_logits = model(input_ids=input_ids, attention_mask=attention_mask,
                                       use_cache=use_cache).logits
            loss, kd, lm = kd_loss(student_logits, teacher_logits, attention_mask, input_ids, kd_weight,
                                   temperature)
            shown = float(loss.detach())
            kd_losses.append(float(kd.detach()))
        else:
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                use_cache=use_cache)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            use_cache=use_cache)
            loss = out.loss
            shown = float(loss.detach())
        if extra_loss is not None and extra_weight > 0:
            extra = extra_loss(model)
            if extra is not None:
                loss = loss + extra_weight * extra.to(loss.device)
                extra_losses.append(float(extra.detach()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(every, 1.0)
        optimizer.step()
        losses.append(shown)
        if log_every and (step + 1) % log_every == 0:
            print(f"heal step {step + 1}/{steps} loss {shown:.4f}")

    model.eval()
    return {"losses": losses, "kd": kd_losses, "extra": extra_losses, "steps": len(losses),
            "stopped_early": stopped}


def distill_heal(student, teacher, tokenizer, texts, params, steps=300, lr=3e-4, batch_size=8, seq_len=384,
                 seed=0, kd_weight=1.0, temperature=1.0, teacher_tokenizer=None, max_seconds=None,
                 log_every=10):
    """Train params so the student matches the teacher's next-token distribution and the LM loss.

    loss = kd_weight * KL(teacher || student at temperature) + LM loss. The
    teacher is frozen and has the student's tokenizer. Pass
    teacher_tokenizer to check that: a different tokenizer raises
    ValueError before any step runs. Returns the heal_groups result.
    """
    if teacher_tokenizer is not None:
        check_same_tokenizer(tokenizer, teacher_tokenizer)
    for param in teacher.parameters():
        param.requires_grad_(False)
    return heal_groups(student, tokenizer, texts, [{"params": list(params), "lr": lr}], steps=steps,
                       batch_size=batch_size, seq_len=seq_len, seed=seed, log_every=log_every, teacher=teacher,
                       kd_weight=kd_weight, temperature=temperature, max_seconds=max_seconds)
