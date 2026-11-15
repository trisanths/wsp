"""Align two models that use different tokenizers, and fit a linear map.

We match tokens from the host and donor tokenizers by their character end
offsets. Each match gives one (host hidden, donor hidden) row pair. A ridge
fit on those rows gives W_in, which maps host hidden size to donor hidden size.
"""

import torch


def alignment_tokens(tokenizer, text):
    """Tokenize text the way heal and inference do, and pick the rows to fit.

    The tokenizer adds its default special tokens (for example BOS), so the
    model sees the same kind of sequence as at run time. Returns
    (input_ids, positions, ends). input_ids has shape (1, seq). positions
    holds the index of each token we may fit on, and ends holds its
    character end offset. Special tokens are left out. When the sequence has
    no special token at index 0, the first text token is the attention sink,
    so it is left out too.
    """
    encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                        return_tensors="pt")
    special = encoded["special_tokens_mask"][0].tolist()
    offsets = encoded["offset_mapping"][0].tolist()
    has_bos = len(special) > 0 and special[0] == 1

    positions = []
    ends = []
    for index in range(len(offsets)):
        start, end = offsets[index]
        if special[index] == 1:
            continue
        if end <= start:
            # A token with no characters has nothing to match on.
            continue
        positions.append(index)
        ends.append(int(end))

    if not has_bos and len(positions) > 0:
        positions = positions[1:]
        ends = ends[1:]
    return encoded["input_ids"], positions, ends


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
    """Fit y ~ x @ w.T with ridge. Return (w, r2, r2_dim), scored on a holdout.

    w has shape (d_donor, d_host). lam is scaled by the mean diagonal of X^T X,
    so it has no units. r2 pools the error over all output dims, so a few
    dims with huge variance can decide it. r2_dim is the mean of each output
    dim's own r2, so every dim counts the same. Dims that are constant on the
    holdout are left out of r2_dim.
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

    sse_dim = ((y_hold - pred) ** 2).sum(dim=0)
    sst_dim = ((y_hold - y_hold.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    varying = sst_dim > 0
    if varying.any():
        r2_dim = float((1.0 - sse_dim[varying] / sst_dim[varying]).mean())
    else:
        r2_dim = float("nan")

    w = w_t.T.contiguous().float()
    return w, r2, r2_dim


def collect_pairs(host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer,
                  donor_capture, texts, max_pairs=20000):
    """Run both models on each text and stack matched (host, donor) hidden rows.

    The host row is the output of decoder layer host_layer, read with a
    forward hook on that layer. That is the same tensor the graft hook gets.
    The donor row comes from donor_capture(donor_model, input_ids), which
    returns (seq, d_donor). Both models see their default special tokens.
    Only rows that alignment_tokens picks enter the fit. Returns (x, y) as
    float32 tensors.
    """
    from wsp.parts import find_decoder_layers

    host_device = next(host_model.parameters()).device
    donor_device = next(donor_model.parameters()).device
    _, host_layers = find_decoder_layers(host_model, host_layer)

    host_store = []

    def host_hook(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        host_store.append(output.detach())

    xs = []
    ys = []
    total = 0
    for text in texts:
        if total >= max_pairs:
            break
        host_ids, host_positions, host_ends = alignment_tokens(host_tokenizer, text)
        donor_ids, donor_positions, donor_ends = alignment_tokens(donor_tokenizer, text)
        pairs = match_offsets(host_ends, donor_ends)
        if len(pairs) == 0:
            continue

        host_store.clear()
        handle = host_layers[host_layer].register_forward_hook(host_hook)
        try:
            with torch.no_grad():
                host_model(host_ids.to(host_device))
        finally:
            handle.remove()
        host_hidden = host_store[0][0]
        donor_hidden = donor_capture(donor_model, donor_ids.to(donor_device))

        host_rows = []
        donor_rows = []
        for i, j in pairs:
            host_rows.append(host_positions[i])
            donor_rows.append(donor_positions[j])
        host_rows = torch.tensor(host_rows)
        donor_rows = torch.tensor(donor_rows)
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
