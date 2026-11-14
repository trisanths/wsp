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
