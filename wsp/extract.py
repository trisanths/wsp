"""Pick which MLP neurons to cut out of a donor model."""

import torch


def top_activation_neurons(model, tokenizer, texts, mlp_module, n, max_tokens=50000):
    """Return the n neurons with the largest mean absolute activation, sorted.

    Activations are the inputs of the MLP's down_proj. Texts run one at a time
    with no padding. Stops after max_tokens tokens.
    """
    down = None
    for name, child in mlp_module.named_children():
        if name == "down_proj":
            down = child
    if down is None:
        raise ValueError("mlp_module has no child named down_proj")

    device = next(model.parameters()).device
    sums = torch.zeros(down.in_features, dtype=torch.float64)
    counts = [0]

    def hook(module, args):
        acts = args[0].detach().float().reshape(-1, args[0].shape[-1])
        sums.add_(acts.abs().sum(dim=0).double().cpu())
        counts[0] += acts.shape[0]

    handle = down.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for text in texts:
                if counts[0] >= max_tokens:
                    break
                ids = tokenizer(text, return_tensors="pt")["input_ids"]
                room = max_tokens - counts[0]
                ids = ids[:, :room]
                if ids.shape[1] == 0:
                    continue
                model(ids.to(device))
    finally:
        handle.remove()

    if counts[0] == 0:
        raise ValueError("no tokens were processed")
    mean_abs = sums / counts[0]
    top = torch.topk(mean_abs, n).indices
    return torch.sort(top).values


def random_neurons(intermediate_size, n, seed):
    """Return n distinct random neuron indices, sorted, from a seeded generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    picks = torch.randperm(intermediate_size, generator=generator)[:n]
    return torch.sort(picks).values


def top_routed_experts(model, tokenizer, texts, moe_module, n, max_tokens=50000):
    """Return the n experts with the highest mean routing probability, sorted.

    Hooks the router (the gate) of the MoE block, takes a softmax over
    experts for every token, and averages. Texts run one at a time with no
    padding. Stops after max_tokens tokens. In Transformers 5 the gate is a
    router module that returns (router_logits, weights, indices); older
    gates are a Linear that returns the logits.
    """
    gate = getattr(moe_module, "gate", None)
    if gate is None:
        raise ValueError("moe_module has no child named gate")

    device = next(model.parameters()).device
    counts = [0]
    state = {"sums": None}

    def hook(module, args, output):
        if isinstance(output, tuple):
            output = output[0]
        logits = output.detach().float().reshape(-1, output.shape[-1])
        probs = torch.softmax(logits, dim=-1)
        if state["sums"] is None:
            state["sums"] = torch.zeros(probs.shape[-1], dtype=torch.float64)
        state["sums"].add_(probs.sum(dim=0).double().cpu())
        counts[0] += probs.shape[0]

    handle = gate.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for text in texts:
                if counts[0] >= max_tokens:
                    break
                ids = tokenizer(text, return_tensors="pt")["input_ids"]
                room = max_tokens - counts[0]
                ids = ids[:, :room]
                if ids.shape[1] == 0:
                    continue
                model(ids.to(device))
    finally:
        handle.remove()

    if counts[0] == 0:
        raise ValueError("no tokens were processed")
    mean_prob = state["sums"] / counts[0]
    top = torch.topk(mean_prob, n).indices
    return torch.sort(top).values


def _find_down_proj(mlp_module):
    # LFM2 calls its down projection w2.
    for name, child in mlp_module.named_children():
        if name == "down_proj" or name == "w2":
            return child
    raise ValueError("mlp module has no down_proj child")


def mean_abs_activation(model, tokenizer, texts, mlp_module, max_tokens=50000):
    """Return a float tensor (intermediate,) of mean |activation| per neuron."""
    down_proj = _find_down_proj(mlp_module)
    totals = {"sum": None, "tokens": 0}

    def hook(module, args):
        acts = args[0].detach().float().abs()
        acts = acts.reshape(-1, acts.shape[-1])
        if totals["sum"] is None:
            totals["sum"] = acts.sum(dim=0)
        else:
            totals["sum"] += acts.sum(dim=0)
        totals["tokens"] += acts.shape[0]

    handle = down_proj.register_forward_pre_hook(hook)
    device = next(model.parameters()).device
    try:
        for text in texts:
            if totals["tokens"] >= max_tokens:
                break
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                model(ids)
    finally:
        handle.remove()
    return totals["sum"] / max(1, totals["tokens"])


def contrast_neurons(model, tokenizer, task_texts, generic_texts, mlp_module, n, max_tokens=50000):
    """Top n neurons by mean |activation| on task text minus on generic text.

    This picks neurons that fire more on the task than in general, not
    neurons that are always loud.
    """
    task = mean_abs_activation(model, tokenizer, task_texts, mlp_module, max_tokens)
    generic = mean_abs_activation(model, tokenizer, generic_texts, mlp_module, max_tokens)
    score = task - generic
    top = torch.topk(score, n).indices
    return torch.sort(top).values


def attribution_neurons(model, tokenizer, texts, mlp_module, n, max_tokens=20000):
    """Top n neurons by estimated loss increase when the neuron is zeroed.

    First-order estimate (attribution patching): zeroing activation a
    changes the loss by about -a * dL/da. We sum that over tokens of the
    task text, using the model's own next-token loss.
    """
    down_proj = _find_down_proj(mlp_module)
    store = {}

    def hook(module, args):
        acts = args[0]
        acts.retain_grad()
        store["acts"] = acts

    handle = down_proj.register_forward_pre_hook(hook)
    device = next(model.parameters()).device
    score = None
    tokens = 0
    try:
        for text in texts:
            if tokens >= max_tokens:
                break
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            model.zero_grad(set_to_none=True)
            loss = model(ids, labels=ids).loss
            loss.backward()
            acts = store["acts"]
            effect = -(acts.detach().float() * acts.grad.detach().float())
            effect = effect.reshape(-1, effect.shape[-1]).sum(dim=0)
            if score is None:
                score = effect
            else:
                score += effect
            tokens += ids.shape[1]
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    top = torch.topk(score, n).indices
    return torch.sort(top).values
