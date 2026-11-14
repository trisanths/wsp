"""Short healing run: train a few parameters with a causal LM loss."""

import torch


def heal(model, tokenizer, texts, params, steps=300, lr=3e-4, batch_size=8, seq_len=384,
         seed=0, log_every=10):
    """Train only params on texts. Return the list of per-step losses."""
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
    for step in range(steps):
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
