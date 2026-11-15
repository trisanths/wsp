"""Train the stitch: the map from a host hidden state into a donor layer.

Run 4 found that a ridge-fit linear stitch from host layer k into donor
layer l raises the donor's own next-token loss from 0.80 to between 2.2
and 3.0 nats, even when host and donor share a tokenizer. The donor's
computation does not survive the stitch. This module trains the stitch to
drive that stitch penalty down, before any task heal.

Stitch kinds (build_stitch), all bias-free and float32:
  linear   a full linear map d_host -> d_donor (the default graft w_in).
  lowrank  LowRankLinear from wsp.graft, rank r.
  mlp      w2(gelu(w1(rms(x)))) + w0(x). w2 starts at zero, so a new MLP
           stitch equals the linear map w0.
  multi    concatenates host states from several host layers and maps the
           result to d_donor with one linear map.
Each stitch takes the host state at layer k (a list of states, one per
host layer, for multi) and returns a donor state.

Objectives (train_stitch):
  mse         regress the donor state at offset-matched positions, with
              each target dim divided by its std.
  donor_lm    same tokenizer only. Feed the stitch output into the donor
              and minimize the donor's own next-token loss.
  downstream  any tokenizers. Feed the stitch output into the donor, run
              donor layers l..end_layer, and match the donor's own
              layer-end_layer states at offset-matched positions.

The donor point is where the stitch output enters the donor. By default
it is the input of decoder layer donor_layer. With donor_norm_prefix (the
module name of that layer's pre-MLP norm), it is the input of that norm,
which is the residual stream after attention. That is where mlp_cluster
and expert parts read.
"""

import contextlib
import time

import torch
from torch import nn
from torch.nn import functional as F

from wsp.align import alignment_tokens, layer_input_capture, match_offsets, mlp_input_capture, ridge_fit
from wsp.graft import LowRankLinear, replace_hidden
from wsp.parts import find_decoder_layers

KINDS = ("linear", "lowrank", "mlp", "multi")
OBJECTIVES = ("mse", "donor_lm", "downstream")


class MLPStitch(nn.Module):
    """x -> w2(gelu(w1(rms(x)))) + w0(x), bias-free.

    rms(x) divides each row by its root mean square and has no weights.
    Host states have a few dims with huge values (attention sinks), and
    without it those dims would swamp the hidden layer. w0 holds the linear
    map. w2 starts at zero, so a new MLPStitch equals w0.
    """

    def __init__(self, d_in, d_out, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = 2 * d_out
        self.hidden = hidden
        self.w0 = nn.Linear(d_in, d_out, bias=False)
        self.w1 = nn.Linear(d_in, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_out, bias=False)
        nn.init.zeros_(self.w2.weight)

    def forward(self, h):
        scale = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return self.w2(F.gelu(self.w1(h * scale))) + self.w0(h)

    def set_from_full(self, w):
        """Set the linear map w0 to w, shape (d_out, d_in)."""
        self.w0.weight.copy_(w.to(self.w0.weight.dtype))


class MultiStitch(nn.Module):
    """Concatenate host states from host_layers, then map them to d_donor.

    host_layers are absolute host layer indices. forward takes a list of
    states in the same order.
    """

    def __init__(self, d_host, d_donor, host_layers):
        super().__init__()
        self.host_layers = list(host_layers)
        self.linear = nn.Linear(d_host * len(self.host_layers), d_donor, bias=False)

    def forward(self, states):
        if len(states) != len(self.host_layers):
            raise ValueError(f"multi stitch needs {len(self.host_layers)} host states, got {len(states)}")
        return self.linear(torch.cat(list(states), dim=-1))

    def set_from_full(self, w):
        """Set the map from a (d_donor, n_layers * d_host) matrix."""
        self.linear.weight.copy_(w.to(self.linear.weight.dtype))


def build_stitch(config, d_host, d_donor):
    """Build a float32 stitch from a config dict.

    config["kind"] is one of KINDS. lowrank needs "rank". mlp takes an
    optional "hidden" (default 2 * d_donor). multi needs "host_layers".
    """
    kind = config.get("kind", "linear")
    if kind == "linear":
        stitch = nn.Linear(d_host, d_donor, bias=False)
    elif kind == "lowrank":
        if config.get("rank") is None:
            raise ValueError("a lowrank stitch needs a rank")
        stitch = LowRankLinear(d_host, d_donor, int(config["rank"]))
    elif kind == "mlp":
        stitch = MLPStitch(d_host, d_donor, config.get("hidden"))
    elif kind == "multi":
        if not config.get("host_layers"):
            raise ValueError("a multi stitch needs host_layers")
        stitch = MultiStitch(d_host, d_donor, config["host_layers"])
    else:
        raise ValueError(f"unknown stitch kind: {kind!r}")
    return stitch.float()


def init_from_ridge(stitch, w):
    """Start the stitch from a full (d_donor, d_in) matrix, for example a ridge fit."""
    with torch.no_grad():
        if isinstance(stitch, nn.Linear):
            stitch.weight.copy_(w.to(stitch.weight.dtype))
        else:
            stitch.set_from_full(w)


def stitch_host_layers(stitch, host_layer):
    """Return the host layers the stitch reads: its own list for multi, else [host_layer]."""
    layers = getattr(stitch, "host_layers", None)
    if layers is not None:
        return list(layers)
    return [host_layer]


def run_stitch(stitch, states):
    """Apply the stitch to a list of host states, one per host layer it reads."""
    if isinstance(stitch, MultiStitch):
        return stitch(states)
    return stitch(states[0])


def device_of(module):
    return next(module.parameters()).device


def autocast(device):
    """bf16 autocast on CUDA, nothing elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def check_same_vocab(host_tokenizer, donor_tokenizer):
    if host_tokenizer.get_vocab() != donor_tokenizer.get_vocab():
        raise ValueError("the donor_lm objective needs host and donor to share a tokenizer, and their "
                         "vocabularies differ; use the downstream or mse objective instead")


def last_layer(model):
    _, layers = find_decoder_layers(model, 0)
    return len(layers) - 1


class StopForward(Exception):
    """Raised by a hook to end a forward pass once the needed layers have run."""


def layer_outputs(model, layer_indices, input_ids, attention_mask=None):
    """Run model until the last of layer_indices has run. Return {index: output hidden state}.

    The rest of the model does not run. Hooks registered on these layers
    before this call run first, so their changes show in the result.
    """
    last = max(layer_indices)
    _, layers = find_decoder_layers(model, last)
    store = {}

    def make_hook(index):
        def hook(module, args, output):
            if isinstance(output, tuple):
                output = output[0]
            store[index] = output
            if index == last:
                raise StopForward()

        return hook

    handles = []
    seen = []
    for index in layer_indices:
        if index in seen:
            continue
        seen.append(index)
        handles.append(layers[index].register_forward_hook(make_hook(index)))
    kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        model(**kwargs)
    except StopForward:
        pass
    finally:
        for handle in handles:
            handle.remove()
    return store


def encode(tokenizer, texts, seq_len=None):
    """Tokenize each text like alignment_tokens, cut it to seq_len, and right-pad a batch.

    Returns (input_ids, attention_mask, positions, ends). positions[b] and
    ends[b] are the rows alignment_tokens picks for text b, kept only when
    they are inside the cut.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    rows = []
    positions = []
    ends = []
    for text in texts:
        ids, text_positions, text_ends = alignment_tokens(tokenizer, text)
        ids = ids[0]
        if seq_len is not None:
            ids = ids[:seq_len]
        kept_positions = []
        kept_ends = []
        for index in range(len(text_positions)):
            if text_positions[index] < len(ids):
                kept_positions.append(text_positions[index])
                kept_ends.append(text_ends[index])
        rows.append(ids)
        positions.append(kept_positions)
        ends.append(kept_ends)

    longest = 1
    for ids in rows:
        if len(ids) > longest:
            longest = len(ids)
    input_ids = torch.full((len(rows), longest), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for b in range(len(rows)):
        n = len(rows[b])
        input_ids[b, :n] = rows[b]
        attention_mask[b, :n] = 1
    return input_ids, attention_mask, positions, ends


def matched_index(host_positions, host_ends, donor_positions, donor_ends):
    """Return index tensors (batch, host_pos, donor_pos) for offset-matched tokens in a batch."""
    batch_rows = []
    host_rows = []
    donor_rows = []
    for b in range(len(host_positions)):
        pairs = match_offsets(host_ends[b], donor_ends[b])
        for i, j in pairs:
            batch_rows.append(b)
            host_rows.append(host_positions[b][i])
            donor_rows.append(donor_positions[b][j])
    return torch.tensor(batch_rows, dtype=torch.long), torch.tensor(host_rows, dtype=torch.long), \
        torch.tensor(donor_rows, dtype=torch.long)


def map_host(host_model, host_layers, stitch, input_ids, attention_mask):
    """Run the frozen host to its last stitch layer and apply the stitch. Gradients reach only the stitch."""
    host_device = device_of(host_model)
    stitch_device = device_of(stitch)
    with torch.no_grad():
        with autocast(host_device):
            found = layer_outputs(host_model, host_layers, input_ids.to(host_device),
                                  attention_mask.to(host_device))
    states = []
    for layer in host_layers:
        states.append(found[layer].to(device=stitch_device, dtype=torch.float32))
    with autocast(stitch_device):
        return run_stitch(stitch, states)


def feed_hooks(donor_model, donor_layer, donor_norm_prefix, holder):
    """Hook the donor so holder["x"] replaces its state at the donor point. Return the handles.

    At a layer input, a pre-hook replaces the hidden state. At a pre-MLP
    norm, a pre-hook gives the norm holder["x"] in place of its input s,
    and a hook on the layer adds holder["x"] - s to the layer output. The
    layer output is s + f(norm(s)), so it becomes x + f(norm(x)), as if
    the residual stream after attention were x.
    """
    _, layers = find_decoder_layers(donor_model, donor_layer)
    layer = layers[donor_layer]
    handles = []
    if donor_norm_prefix is None:
        def layer_pre(module, args, kwargs):
            if len(args) > 0:
                x = holder["x"].to(args[0].dtype)
                return (x,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = holder["x"].to(kwargs["hidden_states"].dtype)
            return args, kwargs

        handles.append(layer.register_forward_pre_hook(layer_pre, with_kwargs=True))
        return handles

    norm = donor_model.get_submodule(donor_norm_prefix)
    seen = {}

    def norm_pre(module, args):
        seen["s"] = args[0]
        return (holder["x"].to(args[0].dtype),) + tuple(args[1:])

    def layer_post(module, args, output):
        s = seen.pop("s")

        def shift(out):
            return out + (holder["x"].to(out.dtype) - s.to(out.dtype))

        return replace_hidden(output, shift)

    handles.append(norm.register_forward_pre_hook(norm_pre))
    handles.append(layer.register_forward_hook(layer_post))
    return handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def donor_capture_for(donor_model, donor_layer, donor_norm_prefix):
    """Return an align-style capture function for the donor point."""
    if donor_norm_prefix is None:
        _, layers = find_decoder_layers(donor_model, donor_layer)
        return layer_input_capture(layers[donor_layer])
    return mlp_input_capture(donor_model.get_submodule(donor_norm_prefix))


def collect_rows(host_model, host_tokenizer, host_layers, donor_model, donor_tokenizer, donor_capture, texts,
                 max_rows=20000):
    """Stack offset-matched (host, donor) rows, like align.collect_pairs.

    The host row concatenates the outputs of host_layers, in that order.
    Returns (x, y) as float32 CPU tensors.
    """
    host_device = device_of(host_model)
    donor_device = device_of(donor_model)
    xs = []
    ys = []
    total = 0
    for text in texts:
        if total >= max_rows:
            break
        host_ids, host_positions, host_ends = alignment_tokens(host_tokenizer, text)
        donor_ids, donor_positions, donor_ends = alignment_tokens(donor_tokenizer, text)
        pairs = match_offsets(host_ends, donor_ends)
        if len(pairs) == 0:
            continue
        with torch.no_grad():
            found = layer_outputs(host_model, host_layers, host_ids.to(host_device))
        parts = []
        for layer in host_layers:
            parts.append(found[layer][0].float().cpu())
        host_hidden = torch.cat(parts, dim=-1)
        donor_hidden = donor_capture(donor_model, donor_ids.to(donor_device)).float().cpu()
        host_rows = []
        donor_rows = []
        for i, j in pairs:
            host_rows.append(host_positions[i])
            donor_rows.append(donor_positions[j])
        xs.append(host_hidden[torch.tensor(host_rows)])
        ys.append(donor_hidden[torch.tensor(donor_rows)])
        total += len(pairs)
    if len(xs) == 0:
        raise ValueError("no matching tokens found in any text")
    x = torch.cat(xs, dim=0)[:max_rows]
    y = torch.cat(ys, dim=0)[:max_rows]
    return x, y


def split_rows(x, n_layers):
    """Split concatenated host rows back into one tensor per host layer."""
    width = x.shape[-1] // n_layers
    return list(torch.split(x, width, dim=-1))


def r2_scores(pred, y):
    """Return (pooled r2, mean per-dim r2), as in align.ridge_fit."""
    pred = pred.double()
    y = y.double()
    centered = y - y.mean(dim=0, keepdim=True)
    sse = ((y - pred) ** 2).sum()
    sst = (centered ** 2).sum()
    r2 = 1.0 - float(sse / sst)
    sse_dim = ((y - pred) ** 2).sum(dim=0)
    sst_dim = (centered ** 2).sum(dim=0)
    varying = sst_dim > 0
    if varying.any():
        r2_dim = float((1.0 - sse_dim[varying] / sst_dim[varying]).mean())
    else:
        r2_dim = float("nan")
    return r2, r2_dim


def dim_std(y):
    """Per-dim std of target rows, floored so near-constant dims do not blow up the loss."""
    std = y.float().std(dim=0)
    floor = 1e-3 * float(std.mean()) + 1e-8
    return std.clamp(min=floor)


def ridge_init(stitch, host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer, donor_layer, texts,
               donor_norm_prefix=None, lam=1e-2):
    """Fit a ridge map on matched rows and start the stitch from it. Return {r2, r2_dim, pairs}.

    A multi stitch is fit on the concatenation of its host layers.
    """
    host_layers = stitch_host_layers(stitch, host_layer)
    capture = donor_capture_for(donor_model, donor_layer, donor_norm_prefix)
    x, y = collect_rows(host_model, host_tokenizer, host_layers, donor_model, donor_tokenizer, capture, texts)
    w, r2, r2_dim = ridge_fit(x, y, lam=lam)
    init_from_ridge(stitch, w)
    return {"r2": r2, "r2_dim": r2_dim, "pairs": int(x.shape[0])}


def stitched_lm_loss(host_model, tokenizer, host_layers, donor_model, donor_layer, donor_norm_prefix, stitch,
                     texts, seq_len=None):
    """The donor's next-token loss on texts with the stitch output at the donor point."""
    donor_device = device_of(donor_model)
    input_ids, attention_mask, _, _ = encode(tokenizer, texts, seq_len)
    mapped = map_host(host_model, host_layers, stitch, input_ids, attention_mask)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    holder = {"x": mapped}
    handles = feed_hooks(donor_model, donor_layer, donor_norm_prefix, holder)
    try:
        with autocast(donor_device):
            out = donor_model(input_ids=input_ids.to(donor_device), attention_mask=attention_mask.to(donor_device),
                              labels=labels.to(donor_device), use_cache=False)
    finally:
        remove_hooks(handles)
    return out.loss


def native_lm_loss(donor_model, tokenizer, texts, seq_len=None):
    donor_device = device_of(donor_model)
    input_ids, attention_mask, _, _ = encode(tokenizer, texts, seq_len)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    with autocast(donor_device):
        out = donor_model(input_ids=input_ids.to(donor_device), attention_mask=attention_mask.to(donor_device),
                          labels=labels.to(donor_device), use_cache=False)
    return out.loss


def stitch_penalty(host_model, tokenizer, host_layer, donor_model, donor_layer, stitch, texts,
                   donor_norm_prefix=None, max_len=None):
    """Donor native loss, stitched loss, and their difference, for a same-tokenizer pair.

    Each text runs on its own. Its loss counts once per predicted token
    (length - 1), so the result is a token-weighted mean, as in
    modal_round4.stitch_penalty. The stitch replaces the donor state at
    every position. Also returns the stitch's pooled and per-dim r2 on the
    offset-matched rows of texts (r2, r2_dim, pairs).
    """
    host_layers = stitch_host_layers(stitch, host_layer)
    native_total = 0.0
    stitched_total = 0.0
    count = 0
    with torch.no_grad():
        for text in texts:
            input_ids, _, _, _ = encode(tokenizer, [text], max_len)
            n = input_ids.shape[1] - 1
            if n < 1:
                continue
            native = native_lm_loss(donor_model, tokenizer, [text], max_len)
            stitched = stitched_lm_loss(host_model, tokenizer, host_layers, donor_model, donor_layer,
                                        donor_norm_prefix, stitch, [text], max_len)
            native_total += float(native) * n
            stitched_total += float(stitched) * n
            count += n
    if count == 0:
        raise ValueError("no text has two or more tokens")
    native_loss = native_total / count
    stitched_loss = stitched_total / count

    capture = donor_capture_for(donor_model, donor_layer, donor_norm_prefix)
    x, y = collect_rows(host_model, tokenizer, host_layers, donor_model, tokenizer, capture, texts)
    with torch.no_grad():
        pred = run_stitch(stitch, split_rows(x.to(device_of(stitch)), len(host_layers))).cpu()
    r2, r2_dim = r2_scores(pred, y)
    return {"native_loss": native_loss, "stitched_loss": stitched_loss, "penalty": stitched_loss - native_loss,
            "r2": r2, "r2_dim": r2_dim, "pairs": int(x.shape[0])}


def downstream_batch(host_model, host_tokenizer, host_layers, donor_model, donor_tokenizer, donor_layer,
                     donor_norm_prefix, stitch, texts, seq_len, end_layer, std):
    """Standardized MSE between stitched and native donor states at end_layer. Return (loss, rows).

    The stitched run feeds the donor pad tokens on the host's sequence
    length and replaces the donor state at the donor point, so only the
    stitch output reaches layers donor_layer..end_layer. loss is None when
    no tokens match.
    """
    donor_device = device_of(donor_model)
    host_ids, host_mask, host_positions, host_ends = encode(host_tokenizer, texts, seq_len)
    donor_ids, donor_mask, donor_positions, donor_ends = encode(donor_tokenizer, texts, seq_len)
    batch_rows, host_rows, donor_rows = matched_index(host_positions, host_ends, donor_positions, donor_ends)
    if len(batch_rows) == 0:
        return None, 0

    with torch.no_grad():
        with autocast(donor_device):
            native = layer_outputs(donor_model, [end_layer], donor_ids.to(donor_device),
                                   donor_mask.to(donor_device))[end_layer]
    target = native[batch_rows.to(donor_device), donor_rows.to(donor_device)].float()

    mapped = map_host(host_model, host_layers, stitch, host_ids, host_mask)
    holder = {"x": mapped.to(donor_device)}
    handles = feed_hooks(donor_model, donor_layer, donor_norm_prefix, holder)
    filler = torch.zeros_like(host_ids).to(donor_device)
    try:
        with autocast(donor_device):
            out = layer_outputs(donor_model, [end_layer], filler, host_mask.to(donor_device))[end_layer]
    finally:
        remove_hooks(handles)
    pred = out[batch_rows.to(donor_device), host_rows.to(donor_device)].float()
    diff = (pred - target) / std.to(donor_device)
    return (diff ** 2).mean(), int(len(batch_rows))


def native_rows_std(host_tokenizer, donor_model, donor_tokenizer, texts, seq_len, end_layer, batch_size,
                    max_texts=64):
    """Per-dim std of the donor's native end_layer states at offset-matched rows."""
    donor_device = device_of(donor_model)
    rows = []
    texts = texts[:max_texts]
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        _, _, host_positions, host_ends = encode(host_tokenizer, batch, seq_len)
        donor_ids, donor_mask, donor_positions, donor_ends = encode(donor_tokenizer, batch, seq_len)
        batch_rows, _, donor_rows = matched_index(host_positions, host_ends, donor_positions, donor_ends)
        if len(batch_rows) == 0:
            continue
        with torch.no_grad():
            with autocast(donor_device):
                native = layer_outputs(donor_model, [end_layer], donor_ids.to(donor_device),
                                       donor_mask.to(donor_device))[end_layer]
        rows.append(native[batch_rows.to(donor_device), donor_rows.to(donor_device)].float().cpu())
    if len(rows) == 0:
        raise ValueError("no matching tokens found in any text")
    return dim_std(torch.cat(rows, dim=0))


def text_sampler(texts, batch_size, seed):
    """Return next_batch(), which draws batch_size texts per call, in seeded shuffled passes."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = []

    def next_batch():
        batch = []
        while len(batch) < batch_size:
            if len(order) == 0:
                order.extend(torch.randperm(len(texts), generator=generator).tolist())
            batch.append(texts[order.pop()])
        return batch

    return next_batch


def freeze_models(models):
    """Set requires_grad False on every parameter. Return the old flags to restore."""
    flags = []
    for model in models:
        for param in model.parameters():
            flags.append((param, param.requires_grad))
            param.requires_grad_(False)
    return flags


def restore_flags(flags):
    # Go in reverse, so a parameter seen twice (host and donor are the same
    # model) gets back the flag it had before the first freeze.
    for index in range(len(flags) - 1, -1, -1):
        param, flag = flags[index]
        param.requires_grad_(flag)


def lm_texts(tokenizer, texts, seq_len):
    """Keep the texts that give at least one next-token label after the cut to seq_len."""
    kept = []
    for text in texts:
        input_ids, _, _, _ = encode(tokenizer, [text], seq_len)
        if input_ids.shape[1] >= 2:
            kept.append(text)
    return kept


def optimize(stitch, steps, lr, next_loss, log_every=0, name="stitch"):
    """Run AdamW on the stitch with grad clip 1.0. Return the per-step losses."""
    params = []
    for param in stitch.parameters():
        param.requires_grad_(True)
        params.append(param)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = next_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        if log_every and (step + 1) % log_every == 0:
            print(f"{name} step {step + 1}/{steps} loss {value:.4f}")
    return losses


def train_stitch(host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer, donor_layer, stitch, texts,
                 objective="donor_lm", steps=500, lr=1e-3, batch_size=8, seq_len=256, seed=0, end_layer=None,
                 eval_texts=None, donor_norm_prefix=None, log_every=0):
    """Train the stitch on texts. Return a dict with per-step losses and before and after metrics.

    Host and donor stay frozen; only the stitch trains. The stitch moves to
    the donor's device. eval_texts (default: texts) give the before and
    after metrics:
      mse         loss (standardized MSE), r2, r2_dim on matched rows.
      donor_lm    stitch_penalty: native_loss, stitched_loss, penalty, r2,
                  r2_dim, and loss (the stitched loss).
      downstream  loss (standardized MSE at end_layer).
    For mse, each step uses batch_size * seq_len random matched rows.
    end_layer (downstream only) defaults to min(donor_layer + 4, last layer).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown stitch objective: {objective!r}")
    if objective == "donor_lm":
        check_same_vocab(host_tokenizer, donor_tokenizer)
        # A batch of one-token texts has no labels and gives a NaN loss.
        texts = lm_texts(host_tokenizer, texts, seq_len)
    if len(texts) == 0:
        raise ValueError("train_stitch needs at least one training text")
    if eval_texts is None:
        eval_texts = texts
    host_layers = stitch_host_layers(stitch, host_layer)
    donor_device = device_of(donor_model)
    stitch.to(donor_device)
    start = time.time()
    result = {"objective": objective, "host_layers": host_layers, "donor_layer": donor_layer,
              "donor_norm_prefix": donor_norm_prefix, "steps": steps, "lr": lr, "batch_size": batch_size,
              "seq_len": seq_len, "seed": seed, "train_texts": len(texts), "eval_texts": len(eval_texts)}

    flags = freeze_models([host_model, donor_model])
    try:
        if objective == "mse":
            capture = donor_capture_for(donor_model, donor_layer, donor_norm_prefix)
            x, y = collect_rows(host_model, host_tokenizer, host_layers, donor_model, donor_tokenizer, capture, texts)
            if eval_texts is texts:
                x_eval = x
                y_eval = y
            else:
                x_eval, y_eval = collect_rows(host_model, host_tokenizer, host_layers, donor_model, donor_tokenizer,
                                              capture, eval_texts)
            std = dim_std(y).to(donor_device)
            x = x.to(donor_device)
            y = y.to(donor_device)
            generator = torch.Generator()
            generator.manual_seed(seed)
            rows_per_step = min(x.shape[0], batch_size * seq_len)

            def mse_metrics():
                with torch.no_grad():
                    pred = run_stitch(stitch, split_rows(x_eval.to(donor_device), len(host_layers))).float()
                diff = (pred - y_eval.to(donor_device)) / std
                r2, r2_dim = r2_scores(pred.cpu(), y_eval)
                return {"loss": float((diff ** 2).mean()), "r2": r2, "r2_dim": r2_dim, "pairs": int(x_eval.shape[0])}

            def next_loss():
                pick = torch.randperm(x.shape[0], generator=generator)[:rows_per_step].to(donor_device)
                with autocast(donor_device):
                    pred = run_stitch(stitch, split_rows(x[pick], len(host_layers)))
                diff = (pred.float() - y[pick]) / std
                return (diff ** 2).mean()

            result["before"] = mse_metrics()
            result["losses"] = optimize(stitch, steps, lr, next_loss, log_every)
            result["after"] = mse_metrics()

        elif objective == "donor_lm":
            next_batch = text_sampler(texts, batch_size, seed)

            def lm_metrics():
                found = stitch_penalty(host_model, host_tokenizer, host_layer, donor_model, donor_layer, stitch,
                                       eval_texts, donor_norm_prefix=donor_norm_prefix, max_len=seq_len)
                found["loss"] = found["stitched_loss"]
                return found

            def next_loss():
                return stitched_lm_loss(host_model, host_tokenizer, host_layers, donor_model, donor_layer,
                                        donor_norm_prefix, stitch, next_batch(), seq_len)

            result["before"] = lm_metrics()
            result["losses"] = optimize(stitch, steps, lr, next_loss, log_every)
            result["after"] = lm_metrics()

        else:
            if end_layer is None:
                end_layer = min(donor_layer + 4, last_layer(donor_model))
            if end_layer < donor_layer or end_layer > last_layer(donor_model):
                raise ValueError(f"end_layer {end_layer} must be in [{donor_layer}, {last_layer(donor_model)}]")
            result["end_layer"] = end_layer
            std = native_rows_std(host_tokenizer, donor_model, donor_tokenizer, texts, seq_len, end_layer,
                                  batch_size)
            next_batch = text_sampler(texts, batch_size, seed)

            def downstream_metrics():
                total = 0.0
                count = 0
                with torch.no_grad():
                    for first in range(0, len(eval_texts), batch_size):
                        batch = eval_texts[first:first + batch_size]
                        loss, rows = downstream_batch(host_model, host_tokenizer, host_layers, donor_model,
                                                      donor_tokenizer, donor_layer, donor_norm_prefix, stitch,
                                                      batch, seq_len, end_layer, std)
                        if loss is None:
                            continue
                        total += float(loss) * rows
                        count += rows
                if count == 0:
                    raise ValueError("no matching tokens found in any eval text")
                return {"loss": total / count, "rows": count}

            def next_loss():
                # A batch with no matched tokens gives no signal, so draw again.
                for _ in range(100):
                    loss, _ = downstream_batch(host_model, host_tokenizer, host_layers, donor_model,
                                               donor_tokenizer, donor_layer, donor_norm_prefix, stitch,
                                               next_batch(), seq_len, end_layer, std)
                    if loss is not None:
                        return loss
                raise ValueError("no matching tokens found in 100 training batches")

            result["before"] = downstream_metrics()
            result["losses"] = optimize(stitch, steps, lr, next_loss, log_every)
            result["after"] = downstream_metrics()
    finally:
        restore_flags(flags)
    result["seconds"] = round(time.time() - start, 1)
    return result
