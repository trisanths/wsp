"""Align two models that use different tokenizers, and fit a linear map.

We match tokens from the host and donor tokenizers by their character end
offsets. Each match gives one (host hidden, donor hidden) row pair. A ridge
fit on those rows gives W_in, which maps host hidden size to donor hidden size.
"""

import torch


def token_end_offsets(tokenizer, text):
    """Return the character end offset of every token in text (no special tokens)."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = []
    for start, end in encoded["offset_mapping"]:
        ends.append(int(end))
    return ends


def match_offsets(host_ends, donor_ends):
    """Return (i, j) pairs where host_ends[i] == donor_ends[j]. Both lists are sorted."""
    pairs = []
    i = 0
    j = 0
    while i < len(host_ends) and j < len(donor_ends):
        if host_ends[i] == donor_ends[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif host_ends[i] < donor_ends[j]:
            i += 1
        else:
            j += 1
    return pairs


def ridge_fit(x, y, lam=1e-2, holdout=0.1, seed=0):
    """Fit y ~ x @ w.T with ridge. Return (w, r2) with r2 scored on a holdout.

    w has shape (d_donor, d_host). lam is scaled by the mean diagonal of X^T X,
    so it has no units.
    """
    n = x.shape[0]
    if n < 20:
        raise ValueError(f"need at least 20 rows to fit, got {n}")
    x = x.float()
    y = y.float()

    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    n_hold = max(2, int(n * holdout))
    hold_idx = order[:n_hold]
    train_idx = order[n_hold:]

    x_train = x[train_idx].double()
    y_train = y[train_idx].double()
    x_hold = x[hold_idx].double()
    y_hold = y[hold_idx].double()

    d_host = x_train.shape[1]
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train
    scale = lam * torch.trace(xtx) / d_host
    ridge = xtx + scale * torch.eye(d_host, dtype=xtx.dtype)
    w_t = torch.linalg.solve(ridge, xty)

    pred = x_hold @ w_t
    sse = ((y_hold - pred) ** 2).sum()
    sst = ((y_hold - y_hold.mean(dim=0, keepdim=True)) ** 2).sum()
    r2 = 1.0 - float(sse / sst)

    w = w_t.T.contiguous().float()
    return w, r2


def collect_pairs(host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer,
                  donor_capture, texts, max_pairs=20000):
    """Run both models on each text and stack matched (host, donor) hidden rows.

    The host row is the output of decoder layer host_layer. The donor row comes
    from donor_capture(donor_model, input_ids), which returns (seq, d_donor).
    Returns (x, y) as float32 tensors.
    """
    host_device = next(host_model.parameters()).device
    donor_device = next(donor_model.parameters()).device
    xs = []
    ys = []
    total = 0
    for text in texts:
        if total >= max_pairs:
            break
        host_ends = token_end_offsets(host_tokenizer, text)
        donor_ends = token_end_offsets(donor_tokenizer, text)
        pairs = match_offsets(host_ends, donor_ends)
        if len(pairs) == 0:
            continue

        host_ids = host_tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        donor_ids = donor_tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = host_model(host_ids.to(host_device), output_hidden_states=True)
            host_hidden = out.hidden_states[host_layer + 1][0]
            donor_hidden = donor_capture(donor_model, donor_ids.to(donor_device))

        host_rows = torch.tensor([p[0] for p in pairs])
        donor_rows = torch.tensor([p[1] for p in pairs])
        xs.append(host_hidden.float().cpu()[host_rows])
        ys.append(donor_hidden.float().cpu()[donor_rows])
        total += len(pairs)

    if len(xs) == 0:
        raise ValueError("no matching tokens found in any text")
    x = torch.cat(xs, dim=0)[:max_pairs]
    y = torch.cat(ys, dim=0)[:max_pairs]
    return x, y


def mlp_input_capture(layer_module_norm):
    """Return a donor_capture function that records the input of the given norm.

    The norm sits right before the donor MLP, so its input is the residual
    stream after attention.
    """

    def capture(donor_model, input_ids):
        store = []

        def hook(module, args):
            store.append(args[0].detach())

        handle = layer_module_norm.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                donor_model(input_ids)
        finally:
            handle.remove()
        return store[0][0]

    return capture


def layer_input_capture(layer_module):
    """Return a donor_capture function that records the input of a decoder layer.

    The input is the residual stream before the layer's input norm. The hook
    reads the hidden state from the positional args or from the
    hidden_states keyword.
    """

    def capture(donor_model, input_ids):
        store = []

        def hook(module, args, kwargs):
            if len(args) > 0:
                store.append(args[0].detach())
            else:
                store.append(kwargs["hidden_states"].detach())

        handle = layer_module.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                donor_model(input_ids)
        finally:
            handle.remove()
        return store[0][0]

    return capture
