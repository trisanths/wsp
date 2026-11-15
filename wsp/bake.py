"""Baking: distill a tapped host into a plain host LoRA, so no donor runs at inference.

A tapped host needs its donors at runtime: each prompt runs every donor up
to its tap layer. Baking trains a copy of the bare host (no taps) to match
the tapped host's predictions on domain prompts. The student trains only a
host LoRA from wsp.lora, so its runtime cost is the host alone. After
merge_lora it is a plain Hugging Face model.

Two modes:
- "answer": for multiple-choice prompts. The student matches the teacher's
  next-token distribution at the last prompt position. With option_ids
  (for example the ids of " A" to " D"), both distributions are taken over
  those ids only.
- "sequence": the teacher writes a greedy continuation of each prompt. The
  student matches the teacher's distribution at every position that
  predicts a continuation token.

The loss is KL(teacher || student) on logits divided by the temperature,
times temperature squared (the usual distillation scale).

Usage:

    result = bake(tapped_host, host_tokenizer, donors, prompts, rank=16, steps=500,
                  lr=1e-4, batch_size=8, mode="answer", option_ids=letter_ids)
    student = result["student"]
    save_baked(student, host_tokenizer, "outputs/baked_mmlu", info=result["info"])

prompts are strings, or (prompt, completion) pairs. Baking reads only the
prompt of a pair: the teacher gives the targets, not the labels.
"""

import copy
import json
import os
import time

import torch

from wsp.lora import DEFAULT_TARGETS, LoRALinear, apply_lora, lora_modules, merge_lora
from wsp.tap import clear_tap_context, donor_model_of, host_device, prepare

MODES = ("answer", "sequence")

# Attributes that attach_taps and set_tap_sources put on a host.
TAP_ATTRIBUTES = ["wsp_tap_state", "wsp_tap_handles", "wsp_tap_stats", "wsp_tap_tokenizer", "wsp_tap_donors"]

# The hook tables of an nn.Module. A tap handle id is in one of them.
HOOK_TABLES = ["_forward_hooks", "_forward_pre_hooks", "_forward_hooks_with_kwargs",
               "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called"]


def has_taps(model):
    return getattr(model, "wsp_taps", None) is not None


def bare_host_copy(host_model, model_id=None, **from_pretrained_kwargs):
    """Return a copy of the host with no taps, no tap hooks, and no donors.

    With model_id, the host is loaded again with from_pretrained (pass dtype
    and other options as keyword arguments) and moved to the host device.
    Without it, the host is deep copied. The tap branches, their hooks, the
    tap state, and the donor references are left out of the copy, and the
    host itself stays as it was. A host LoRA on the source is folded into
    the copy's weights, so the copy is a plain model that starts from the
    same host function.
    """
    device = host_device(host_model)
    if model_id is not None:
        import transformers

        student = transformers.AutoModelForCausalLM.from_pretrained(model_id, **from_pretrained_kwargs)
        student = student.to(device)
    else:
        if getattr(host_model, "wsp_grafts", None) is not None:
            raise ValueError("bare_host_copy cannot deep copy a host with grafts; pass model_id to reload it")
        student = deep_copy_without_taps(host_model)
    if len(lora_modules(student)) > 0:
        merge_lora(student)
    student.eval()
    return student


def deep_copy_without_taps(host_model):
    """Deep copy host_model, but not its tap branches, tap state, donors, or tap hooks."""
    # Take the tap parts off the host for the copy, and put them back after.
    stash = {}
    if "wsp_taps" in host_model._modules:
        stash["wsp_taps"] = host_model._modules.pop("wsp_taps")
    for name in TAP_ATTRIBUTES:
        if name in host_model.__dict__:
            stash[name] = host_model.__dict__.pop(name)
    handle_ids = set()
    if "wsp_tap_handles" in stash:
        for handle in stash["wsp_tap_handles"]:
            handle_ids.add(handle.id)
    try:
        copied = copy.deepcopy(host_model)
    finally:
        for name in stash:
            if name == "wsp_taps":
                host_model._modules["wsp_taps"] = stash[name]
            else:
                host_model.__dict__[name] = stash[name]

    # The copy has the same hook tables as the host, so it still holds the
    # tap hooks. They point at the host's branches, so we remove them.
    for module in copied.modules():
        for table_name in HOOK_TABLES:
            table = getattr(module, table_name, None)
            if table is None:
                continue
            for hook_id in list(table.keys()):
                if hook_id in handle_ids:
                    del table[hook_id]
    return copied


def prompt_text(item):
    """Return the prompt of a string or of a (prompt, completion) pair."""
    if isinstance(item, str):
        return item
    return item[0]


def fit_prompts(tokenizer, prompts, seq_len):
    """Return the prompt texts whose host encoding has at most seq_len tokens."""
    kept = []
    for item in prompts:
        text = prompt_text(item)
        if len(tokenizer(text).input_ids) <= seq_len:
            kept.append(text)
    return kept


def position_ids_from_mask(attention_mask):
    """Position ids that count only real tokens, as generate computes them under left padding."""
    positions = attention_mask.long().cumsum(-1) - 1
    positions = positions.masked_fill(attention_mask == 0, 1)
    return positions


def restrict(logits, option_ids):
    """Return logits over option_ids only, or all logits when option_ids is None."""
    if option_ids is None:
        return logits
    index = torch.tensor(list(option_ids), dtype=torch.long, device=logits.device)
    return logits.index_select(-1, index)


def distill_kl(student_logits, teacher_logits, temperature=1.0, option_ids=None, weights=None):
    """Return the mean KL(teacher || student) per position, times temperature squared.

    The logits have shape (..., vocab). option_ids restricts both to those
    ids before the softmax. weights, with the shape of the logits minus the
    last dim, gives each position's weight (1 for a position to train on,
    0 to skip it). The mean is over positions with weight.
    """
    s = restrict(student_logits.float(), option_ids) / temperature
    t = restrict(teacher_logits.float(), option_ids) / temperature
    log_s = torch.log_softmax(s, dim=-1)
    log_t = torch.log_softmax(t, dim=-1)
    per_position = (log_t.exp() * (log_t - log_s)).sum(dim=-1)
    if weights is None:
        loss = per_position.mean()
    else:
        weights = weights.to(per_position.dtype)
        loss = (per_position * weights).sum() / weights.sum().clamp(min=1.0)
    return loss * temperature * temperature


def encode_left(tokenizer, prompts, device):
    """Tokenize prompts with left padding. Return (input_ids, attention_mask) on device."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        encoded = tokenizer(list(prompts), padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = old_side
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def teacher_prepare(teacher, tokenizer, donors, prompts):
    """Run the donors, set the teacher's tap context, and return left padded host inputs."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = prepare(teacher, tokenizer, donors, prompts)
    finally:
        tokenizer.padding_side = old_side
    return inputs


@torch.no_grad()
def teacher_answer_logits(teacher, tokenizer, donors, prompts):
    """Return the teacher's logits at the last prompt position, and the host inputs.

    Returns (logits, input_ids, attention_mask). logits is (batch, vocab)
    float32. Prompts are left padded, so the last position is the last
    prompt token of every row.
    """
    inputs = teacher_prepare(teacher, tokenizer, donors, prompts)
    try:
        positions = position_ids_from_mask(inputs["attention_mask"])
        out = teacher(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                      position_ids=positions, use_cache=False)
    finally:
        clear_tap_context(teacher)
    return out.logits[:, -1, :].float(), inputs["input_ids"], inputs["attention_mask"]


def stop_token_ids(model, tokenizer):
    """Return the set of ids that end a generate call: the tokenizer EOS and the generation config EOS.

    generate stops on generation_config.eos_token_id, which can be a list
    (Llama 3 and Qwen instruct models stop on more than one id).
    """
    found = []
    found.append(tokenizer.eos_token_id)
    config = getattr(model, "generation_config", None)
    if config is not None:
        found.append(config.eos_token_id)
    stops = set()
    for value in found:
        if value is None:
            continue
        if isinstance(value, int):
            stops.add(value)
            continue
        if torch.is_tensor(value):
            value = value.flatten().tolist()
        for item in value:
            stops.add(int(item))
    return stops


@torch.no_grad()
def teacher_sequences(teacher, tokenizer, donors, prompts, max_new_tokens):
    """Generate greedy teacher continuations and return the teacher's logits along them.

    Returns a dict with:
    - input_ids, attention_mask: (batch, prompt_len + n), the left padded
      prompt and then the continuation. Tokens after a row's first stop
      token (see stop_token_ids) are pads with mask 0.
    - logits: (batch, prompt_len + n, vocab) float32 teacher logits.
    - weights: (batch, prompt_len + n) float, 1 at each position that
      predicts a continuation token (the last prompt position and the
      continuation positions before the last kept token).
    """
    inputs = teacher_prepare(teacher, tokenizer, donors, prompts)
    try:
        generated = teacher.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, use_cache=True)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated[:, prompt_len:]
        batch = new_tokens.shape[0]
        count = new_tokens.shape[1]

        # Keep each row's tokens up to and including its first stop token.
        keep = torch.ones(batch, count, dtype=torch.long, device=new_tokens.device)
        stops = stop_token_ids(teacher, tokenizer)
        for row in range(batch):
            for index in range(count):
                if int(new_tokens[row, index]) in stops:
                    keep[row, index + 1:] = 0
                    break
        new_tokens = new_tokens.masked_fill(keep == 0, tokenizer.pad_token_id)
        input_ids = torch.cat([inputs["input_ids"], new_tokens], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"], keep], dim=1)

        # The tap context from prepare is still set. Positions past the
        # prompt read all valid memory, the same as decode steps.
        positions = position_ids_from_mask(attention_mask)
        out = teacher(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions,
                      use_cache=False)
    finally:
        clear_tap_context(teacher)

    # Logits at position p predict token p + 1, so continuation token j
    # (at position prompt_len + j) is predicted at prompt_len + j - 1.
    weights = torch.zeros(input_ids.shape, dtype=torch.float32, device=input_ids.device)
    weights[:, prompt_len - 1:prompt_len - 1 + count] = keep.float()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "logits": out.logits.float(),
            "weights": weights}


def student_logits(student, input_ids, attention_mask):
    positions = position_ids_from_mask(attention_mask)
    out = student(input_ids=input_ids, attention_mask=attention_mask, position_ids=positions, use_cache=False)
    return out.logits


def bake(teacher_host, tokenizer, donors, prompts, student_host=None, rank=16, steps=100, lr=1e-4,
         batch_size=8, seq_len=256, temperature=1.0, mode="answer", option_ids=None, max_new_tokens=64,
         seed=0, merge=False, lora_targets=None, lora_layers=None, log_every=10):
    """Distill the tapped teacher_host into a host LoRA on a bare copy of the host.

    teacher_host has its taps attached. donors is the dict of donor_id ->
    (donor_model, donor_tokenizer) the taps read. tokenizer is the host
    tokenizer, shared by teacher and student.

    student_host: a bare host with no taps, or None for bare_host_copy of
    the teacher. Every student parameter is frozen, then a rank-r LoRA
    goes on and trains with AdamW at lr. The teacher and the donors run in
    no_grad and do not change.

    prompts: strings or (prompt, completion) pairs. Prompts longer than
    seq_len host tokens are skipped. mode is "answer" or "sequence" (see
    the module doc). option_ids works in answer mode only. max_new_tokens
    is the teacher continuation length in sequence mode. seed sets the
    torch seed (for the LoRA init) and the batch order.

    merge=True folds the LoRA into the student weights at the end, so the
    student is a plain Hugging Face model.

    Returns a dict: student, losses (one per step), seconds (training
    wall time), teacher_seconds (teacher and donor time inside it), and
    info (the settings, for save_baked).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "sequence" and option_ids is not None:
        raise ValueError("option_ids works in answer mode only")
    if not has_taps(teacher_host):
        raise ValueError("the teacher has no taps; attach or load them first")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # A generator of prompts is read once here, so len works at the end.
    prompts = list(prompts)
    if option_ids is not None:
        ids = []
        for value in option_ids:
            ids.append(int(value))
        option_ids = ids

    texts = fit_prompts(tokenizer, prompts, seq_len)
    if len(texts) == 0:
        raise ValueError(f"no prompt fits in seq_len {seq_len} host tokens")

    if student_host is None:
        student = bare_host_copy(teacher_host)
    else:
        if has_taps(student_host):
            raise ValueError("the student must be a bare host; it has taps")
        student = student_host

    # The teacher and donors run under no_grad, so their requires_grad flags
    # stay as the caller set them (tap training can go on after a bake).
    teacher_host.eval()
    for donor_id in donors:
        donor_model = donor_model_of(donors[donor_id])
        if donor_model is None:
            continue
        donor_model.eval()

    for param in student.parameters():
        param.requires_grad_(False)
    torch.manual_seed(seed)
    targets = lora_targets
    if targets is None:
        targets = DEFAULT_TARGETS
    lora_params = apply_lora(student, rank, targets=targets, layers=lora_layers)
    for param in lora_params:
        param.requires_grad_(True)
    lora_count = 0
    for param in lora_params:
        lora_count += param.numel()

    device = host_device(student)
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_autocast = device.type == "cuda"

    losses = []
    order = []
    teacher_seconds = 0.0
    start = time.time()
    student.train()
    try:
        for step in range(steps):
            batch = []
            while len(batch) < batch_size:
                if len(order) == 0:
                    order = torch.randperm(len(texts), generator=generator).tolist()
                batch.append(texts[order.pop()])

            teacher_start = time.time()
            if mode == "answer":
                target, input_ids, attention_mask = teacher_answer_logits(teacher_host, tokenizer, donors, batch)
                weights = None
            else:
                found = teacher_sequences(teacher_host, tokenizer, donors, batch, max_new_tokens)
                target = found["logits"]
                input_ids = found["input_ids"]
                attention_mask = found["attention_mask"]
                weights = found["weights"].to(device)
            teacher_seconds += time.time() - teacher_start
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = student_logits(student, input_ids, attention_mask)
            else:
                logits = student_logits(student, input_ids, attention_mask)
            if mode == "answer":
                logits = logits[:, -1, :]
            loss = distill_kl(logits, target, temperature, option_ids, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            losses.append(float(loss.detach()))
            if log_every and (step + 1) % log_every == 0:
                print(f"bake step {step + 1}/{steps} kl {losses[-1]:.4f}")
    finally:
        student.eval()
    seconds = time.time() - start

    for param in student.parameters():
        param.requires_grad_(False)
    if merge:
        merge_lora(student)

    info = {"mode": mode, "rank": rank, "steps": steps, "lr": lr, "batch_size": batch_size,
            "seq_len": seq_len, "temperature": temperature, "seed": seed, "prompts_used": len(texts),
            "prompts_given": len(prompts), "lora_params": lora_count, "merged": merge,
            "seconds": round(seconds, 2), "teacher_seconds": round(teacher_seconds, 2)}
    if option_ids is not None:
        info["option_ids"] = list(option_ids)
    if mode == "sequence":
        info["max_new_tokens"] = max_new_tokens
    return {"student": student, "losses": losses, "seconds": seconds, "teacher_seconds": teacher_seconds,
            "info": info}


@torch.no_grad()
def mean_kl(teacher_host, student, tokenizer, donors, prompts, temperature=1.0, option_ids=None,
            batch_size=8):
    """Return the mean last-position KL(teacher || student) over prompts, in answer mode terms."""
    texts = []
    for item in prompts:
        texts.append(prompt_text(item))
    total = 0.0
    count = 0
    device = host_device(student)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        target, input_ids, attention_mask = teacher_answer_logits(teacher_host, tokenizer, donors, batch)
        logits = student_logits(student, input_ids.to(device), attention_mask.to(device))[:, -1, :]
        kl = distill_kl(logits, target.to(device), temperature, option_ids)
        total += float(kl) * len(batch)
        count += len(batch)
    return total / count


def save_baked(student, tokenizer, path, info=None):
    """Merge any LoRA into the student, then save it as a plain Hugging Face model.

    Writes the model, the tokenizer, and bake.json (info) to path. The
    merge changes the student in place.
    """
    if has_taps(student):
        raise ValueError("a baked student has no taps; this model has some")
    if len(lora_modules(student)) > 0:
        merge_lora(student)
    os.makedirs(path, exist_ok=True)
    student.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    with open(os.path.join(path, "bake.json"), "w") as f:
        json.dump({"info": info}, f, indent=2)


def is_plain(model):
    """True when the model has no taps, no tap hooks state, and no LoRA modules."""
    if has_taps(model):
        return False
    for name in TAP_ATTRIBUTES:
        if hasattr(model, name):
            return False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            return False
    return True
