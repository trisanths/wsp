"""Skill tops: one shared trunk of base layers and a skill-specific top on top of it.

Evidence (results/README.md). A trained stitch lets a donor's layers run on
the base's hidden states almost losslessly when the path then continues
through the donor's own remaining layers and head (stitch study: 0.03 to
0.08 nats). Every attempt to send a donor block's output back into the
base's upper layers failed or gave no gain (runs 14, 14b, 20, 22). Swapping
a specialist's layers into the base with no stitch gives chance (run 16c).
So a skill keeps its own readout.

The network. The base keeps layers 0..cut-1 as a frozen, shared trunk.
Each skill gets a top:

    trunk state at the cut -> stitch (+ donor sink at the first real token)
    -> donor layers cut_donor..end (own rotary base, private KV cache)
    -> donor final norm -> donor head

The base's own upper layers and head are the top named "base". A router
picks one top per sequence from the trunk's mean state at the smallest cut.
Only one path runs per token, so compute per token stays near the base
while stored params grow by each top.

Pieces.
  SkillTops(base_model, tokenizer)        the network
  SkillTops.add_top(...)                  build a top from a donor, by byte range
  SkillTops.fit_stitch(top, texts, ...)   ridge init, then the donor's own LM loss
  SkillTops.fit_router(texts_by_class)    ridge router over the trunk state
  SkillTops.forward / generate            forced or routed, batched, cached
  SkillTops.count_params()                stored and per-path params
  SkillTops.save / SkillTops.load         one directory
  fit_kd_lora(...)                        reference arm: donor distilled into a LoRA on the base
  sizes_from_configs(...)                 stored and per-token params from configs alone

Tokenizer rule. A top with its own head needs the base tokenizer's ids to
mean the same tokens to the donor. check_vocab compares the token to id
maps (not file hashes). A shared token with two ids, or an id with two
tokens, raises. The donor head must cover every shared id. Rows the head
has beyond the base's logit width are dropped, and a narrower head is padded
with masked rows, so every path returns logits of the base's width. A base
token the donor tokenizer lacks gets its column masked to the dtype minimum
(never chosen). Donor-only ids and ids in neither tokenizer pass through
as the donor head gives them (they are the padding rows of the table).

Stitch penalty. The donor's own next-token loss through the top minus its
native loss, on the same texts, token weighted. The prediction made from
the first real token is left out of both when the top uses a sink, because
the sink is a constant and that one token cannot be predicted from it. The
study's number (all tokens) is reported as penalty_all.

Not built. own_head=False (the donor state into the base head) is the
write-back design that failed in runs 14 to 22, so it raises.
"""

import copy
import json
import os
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from wsp import stitch as stitch_lib
from wsp.parts import (_build_meta_model, _layer_window, _text_config, block_tensor_names, build_block_part,
                       causal_mask, find_decoder_layers)
from wsp.scramble import scramble_module, scramble_tensors

CONTROLS = (None, "scrambled")
RESERVED = "base"
ROUTER_SCALE = 8.0
DEFAULT_MAX_PENALTY = 0.25
LR_BY_KIND = {"linear": 1e-3, "mlp": 3e-4}


# ---------------------------------------------------------------- small helpers


def positions_of(mask):
    """Position ids that count only real tokens, so left padding does not shift a prompt."""
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


def first_index(mask):
    """Index of the first real token of every row (0 for a row with none)."""
    return mask.long().argmax(dim=-1)


def layer_dtype(module):
    return next(module.parameters()).dtype


def model_device(module):
    return next(module.parameters()).device


def count(params):
    seen = set()
    total = 0
    for param in params:
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    return total


def run_layers(layers, windows, rotary, h, attention_mask, position_ids, cache, past):
    """Run decoder layers over h with an explicit mask. Return the new hidden state.

    attention_mask is the 2D padding mask (batch, past + seq) or None. cache
    is a DynamicCache, or None for a plain pass. past is the number of tokens
    the cache held before this call. The mask is additive, so it works for
    eager and sdpa layers alike.
    """
    if len(layers) == 0:
        return h
    batch, seq, _ = h.shape
    cos_sin = rotary(h, position_ids)
    cache_position = torch.arange(past, past + seq, device=h.device)
    masks = {}
    for index in range(len(layers)):
        window = windows[index]
        if window not in masks:
            mask = causal_mask(batch, seq, h.dtype, h.device, window, past)
            if attention_mask is not None:
                pad_keys = attention_mask[:, None, None, :] == 0
                mask = mask.clone().masked_fill(pad_keys, torch.finfo(mask.dtype).min)
            masks[window] = mask
        out = layers[index](h, attention_mask=masks[window], position_ids=position_ids, past_key_values=cache,
                            use_cache=cache is not None, cache_position=cache_position,
                            position_embeddings=cos_sin)
        if isinstance(out, tuple):
            out = out[0]
        h = out
    return h


def lm_token_losses(logits, ids, mask, drop_first):
    """Per-token cross entropy of predicting ids[:, 1:] from logits[:, :-1], and the validity mask.

    A prediction counts when both tokens are real. With drop_first the one
    made from the first real token does not count.
    """
    vocab = logits.shape[-1]
    ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, vocab), ids[:, 1:].reshape(-1), reduction="none")
    ce = ce.reshape(ids.shape[0], ids.shape[1] - 1)
    valid = (mask[:, 1:] == 1) & (mask[:, :-1] == 1)
    if drop_first:
        first = first_index(mask)
        columns = torch.arange(ids.shape[1] - 1, device=ids.device).unsqueeze(0)
        valid = valid & (columns != first.unsqueeze(1))
    return ce, valid


def kd_loss_value(student_logits, teacher_logits, mask, drop_first):
    """Mean KL(teacher || student) per token over the shared vocabulary, on real tokens.

    The last position predicts past the text and is kept; the first real
    token's prediction is dropped when drop_first is set (the sink row).
    """
    rows = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = torch.log_softmax(student_logits[..., :rows].float(), dim=-1)
    t = torch.log_softmax(teacher_logits[..., :rows].float(), dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    weight = mask.to(kl.device).float()
    if drop_first:
        first = first_index(mask).to(kl.device)
        weight = weight.clone()
        weight[torch.arange(weight.shape[0], device=kl.device), first] = 0.0
    return (kl * weight).sum() / weight.sum().clamp(min=1.0)


def lm_loss_value(logits, ids, mask, drop_first):
    ce, valid = lm_token_losses(logits, ids, mask, drop_first)
    weight = valid.to(ce.dtype)
    return (ce * weight).sum() / weight.sum().clamp(min=1.0)


# ---------------------------------------------------------------- vocabulary


def check_vocab(base_tokenizer, donor_tokenizer, donor_rows, out_rows):
    """Compare the token to id maps of the base and donor tokenizers. Return a report with the ids to mask.

    Raises ValueError when a shared token has two ids, when an id names two
    tokens, or when the donor head has no row for a shared id. base_only
    tokens (the donor tokenizer lacks them) are returned in masked_ids.
    """
    base_vocab = base_tokenizer.get_vocab()
    donor_vocab = donor_tokenizer.get_vocab()
    moved = []
    for token in base_vocab:
        if token in donor_vocab and base_vocab[token] != donor_vocab[token]:
            moved.append(token)
    if len(moved) > 0:
        raise ValueError(f"{len(moved)} shared tokens have different ids in the base and donor tokenizers "
                         f"(first: {moved[:3]}); a top with its own head needs the same ids")
    donor_by_id = {}
    for token, index in donor_vocab.items():
        donor_by_id[index] = token
    clash = []
    for token, index in base_vocab.items():
        if index in donor_by_id and donor_by_id[index] != token:
            clash.append(index)
    if len(clash) > 0:
        raise ValueError(f"{len(clash)} ids name different tokens in the base and donor tokenizers "
                         f"(first ids: {clash[:3]})")
    shared_ids = []
    masked = []
    for token, index in base_vocab.items():
        if token in donor_vocab:
            shared_ids.append(index)
        else:
            masked.append(index)
    top_shared = max(shared_ids) if len(shared_ids) > 0 else -1
    if top_shared >= donor_rows:
        raise ValueError(f"the donor head has {donor_rows} rows but the shared vocabulary reaches id {top_shared}")
    extra_donor = 0
    for token in donor_vocab:
        if token not in base_vocab:
            extra_donor += 1
    kept_masked = []
    for index in masked:
        if index < out_rows:
            kept_masked.append(index)
    return {"checked": True, "shared": len(shared_ids), "base_only": len(masked), "donor_only": extra_donor,
            "masked_ids": sorted(kept_masked), "donor_rows": int(donor_rows), "out_rows": int(out_rows)}


def fit_head_rows(weight, out_rows):
    """Cut a head to out_rows rows, or pad it with zero rows. Return (weight, pad_start).

    pad_start is the first padded row, or None when nothing was padded. A
    head with more rows loses its tail (rows no base token id reaches).
    """
    rows = weight.shape[0]
    if rows == out_rows:
        return weight, None
    if rows > out_rows:
        return weight[:out_rows].contiguous(), None
    pad = torch.zeros(out_rows - rows, weight.shape[1], dtype=weight.dtype, device=weight.device)
    return torch.cat([weight, pad], dim=0), rows


# ---------------------------------------------------------------- donor parts


def donor_config_of(donor):
    import transformers

    return transformers.AutoConfig.from_pretrained(donor)


def head_tensor_name(text_config):
    if getattr(text_config, "tie_word_embeddings", False):
        return "model.embed_tokens.weight"
    return "lm_head.weight"


def donor_parts(donor, cut_donor, control, control_seed, device, cache_dir=None):
    """Build the donor's layers cut_donor..end, final norm, and head weight.

    donor is a repo id or local directory (fetched by byte range) or a
    loaded model (its layers are copied). Returns a dict with layers,
    rotary, windows, norm, head_weight, d_model, n_layers, rows, tied, config_id.
    """
    if isinstance(donor, nn.Module):
        return model_donor_parts(donor, cut_donor, control, control_seed)
    from wsp.skill_moe import fetch_expert_tensors

    config = donor_config_of(donor)
    text = _text_config(config)
    n_layers = int(text.num_hidden_layers)
    if cut_donor < 0 or cut_donor >= n_layers:
        raise ValueError(f"cut_donor {cut_donor} must be in [0, {n_layers - 1}] for {donor}")
    names = block_tensor_names(config, cut_donor, n_layers - 1)
    head_name = head_tensor_name(text)
    tensors = fetch_expert_tensors(donor, names + ["model.norm.weight", head_name], cache_dir)
    if control == "scrambled":
        layer_tensors = {}
        for name in names:
            layer_tensors[name] = tensors[name].to(device)
        scramble_tensors(layer_tensors, control_seed, "spectrum")
        for name in names:
            tensors[name] = layer_tensors[name]
    part = build_block_part(config, cut_donor, n_layers - 1, tensors)
    meta = _build_meta_model(config)
    meta_norm = meta.model.norm
    eps = getattr(meta_norm, "variance_epsilon", getattr(text, "rms_norm_eps", 1e-6))
    dtype = tensors["model.norm.weight"].dtype
    norm = type(meta_norm)(text.hidden_size, eps=eps)
    norm.load_state_dict({"weight": tensors["model.norm.weight"].to(dtype)})
    for param in norm.parameters():
        param.requires_grad_(False)
    head_weight = tensors[head_name].to(dtype).contiguous()
    return {"layers": part.layers, "rotary": part._rotary[0], "windows": list(part.windows), "norm": norm,
            "head_weight": head_weight, "d_model": int(text.hidden_size), "n_layers": n_layers,
            "rows": int(head_weight.shape[0]), "tied": bool(getattr(text, "tie_word_embeddings", False))}


def model_donor_parts(model, cut_donor, control, control_seed):
    inner = model.model
    n_layers = len(inner.layers)
    if cut_donor < 0 or cut_donor >= n_layers:
        raise ValueError(f"cut_donor {cut_donor} must be in [0, {n_layers - 1}]")
    layers = nn.ModuleList()
    windows = []
    text = _text_config(model.config)
    for index in range(cut_donor, n_layers):
        layer = copy.deepcopy(inner.layers[index])
        layers.append(layer)
        windows.append(_layer_window(layer, text))
    if control == "scrambled":
        scramble_module(layers, control_seed, "spectrum")
    norm = copy.deepcopy(inner.norm)
    head = model.get_output_embeddings().weight.detach().clone()
    for param in list(layers.parameters()) + list(norm.parameters()):
        param.requires_grad_(False)
    return {"layers": layers, "rotary": copy.deepcopy(inner.rotary_emb), "windows": windows, "norm": norm,
            "head_weight": head, "d_model": int(text.hidden_size), "n_layers": n_layers,
            "rows": int(head.shape[0]), "tied": model.get_output_embeddings().weight is inner.embed_tokens.weight}


# ---------------------------------------------------------------- one top


class TopModule(nn.Module):
    """One skill top: stitch, donor layers, donor norm, donor head. The layers are frozen."""

    def __init__(self, spec, stitch, layers, rotary, windows, norm, head, head_mask, owner):
        super().__init__()
        self.spec = spec
        self.stitch = stitch
        self.layers = layers
        self._rotary = [rotary]
        self.windows = list(windows)
        self.norm = norm
        self.head = head
        self.register_buffer("head_mask", head_mask, persistent=False)
        self.has_head_mask = bool(head_mask.any())
        self.register_buffer("sink", torch.zeros(spec["d_donor"], dtype=torch.float32), persistent=False)
        self.has_sink = False
        self.accepted = True
        self.flag = None
        self.fit_report = None
        self.lora = None
        self._owner = [owner]

    @property
    def name(self):
        return self.spec["name"]

    @property
    def cut_host(self):
        return self.spec["cut_host"]

    @property
    def cut_donor(self):
        return self.spec["cut_donor"]

    def owner(self):
        return self._owner[0]

    def train(self, mode=True):
        # Donor layers stay in eval mode. The stitch has no dropout.
        return super().train(False)

    def rotary(self, reference):
        rotary = self._rotary[0]
        if rotary.inv_freq.device != reference.device:
            rotary.to(reference.device)
        return rotary

    def map_state(self, h, first):
        """The stitch output for trunk states h. first: the first real index per row, or None (no sink)."""
        param = next(self.stitch.parameters())
        x = h.to(param.dtype)
        with stitch_lib.autocast(x.device):
            y = self.stitch(x)
        y = y.to(layer_dtype(self.layers))
        if first is not None and self.has_sink:
            hit = torch.zeros(y.shape[0], y.shape[1], 1, dtype=torch.bool, device=y.device)
            hit[torch.arange(y.shape[0], device=y.device), first.to(y.device)] = True
            sink = self.sink.to(device=y.device, dtype=y.dtype).view(1, 1, -1)
            y = torch.where(hit, sink, y)
        return y

    def run(self, h, mask, position_ids, cache, past, first):
        """Trunk states in, normed donor states out. first is used only on a prefill (past 0)."""
        if past != 0:
            first = None
        x = self.map_state(h, first)
        out = run_layers(self.layers, self.windows, self.rotary(x), x, mask, position_ids, cache, past)
        return self.norm(out)

    def logits(self, normed):
        # The donor's final norm can return float32 (Qwen2 RMSNorm upcasts) while the head is bf16 on a GPU.
        out = self.head(normed.to(self.head.weight.dtype))
        if self.has_head_mask:
            out = out.masked_fill(self.head_mask, torch.finfo(out.dtype).min)
        return out

    def trainable(self):
        params = []
        for param in self.stitch.parameters():
            params.append(param)
        for name, param in self.named_parameters():
            if name.endswith("lora_A") or name.endswith("lora_B"):
                params.append(param)
        return params


# ---------------------------------------------------------------- the network


class SkillTops(nn.Module):
    """A frozen base with a shared trunk, any number of skill tops, and a router."""

    def __init__(self, base_model, tokenizer, base_id=None):
        super().__init__()
        self.base = base_model
        self.base.eval()
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.tokenizer = tokenizer
        inner = base_model.model
        self.n_layers = len(inner.layers)
        self.text_config = _text_config(base_model.config)
        self.d_model = int(self.text_config.hidden_size)
        self.rows = int(base_model.get_output_embeddings().weight.shape[0])
        self.base_id = base_id or getattr(base_model.config, "_name_or_path", None)
        self.trunk_windows = []
        for layer in inner.layers:
            self.trunk_windows.append(_layer_window(layer, self.text_config))
        self.tops = nn.ModuleDict()
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.register_buffer("router_w", None, persistent=False)
        self.register_buffer("router_b", None, persistent=False)
        self.router_names = None
        self.router_cut = None
        self.router_scale = ROUTER_SCALE
        self.base_margin = 0.0
        self.allow_flagged = False
        self.last_choice = None
        self.tally = None

    # ---- access

    def train(self, mode=True):
        return super().train(False)

    def get_input_embeddings(self):
        return self.base.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base.get_output_embeddings()

    def top_names(self):
        return list(self.tops.keys())

    def get_top(self, top):
        if isinstance(top, TopModule):
            return top
        if top not in self.tops:
            raise ValueError(f"no top named {top!r}; the tops are {self.top_names()}")
        return self.tops[top]

    # ---- building

    def add_top(self, name, donor, cut_host, cut_donor=None, stitch_kind="mlp", own_head=True, control=None,
                control_seed=0, donor_tokenizer=None, check=True, device=None, cache_dir=None, stitch_hidden=None):
        """Add a top. donor is a repo id, a local directory, or a loaded model.

        The trunk is base layers 0..cut_host-1. cut_donor defaults to the
        same relative depth in the donor. control="scrambled" rewrites the
        donor layers with wsp.scramble (seed control_seed) before any
        training. The stitch starts as the identity when the widths match.
        """
        if name == RESERVED or name in self.tops:
            raise ValueError(f"top name {name!r} is taken")
        if "." in name:
            raise ValueError("a top name may not hold a dot")
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}, got {control!r}")
        if not own_head:
            raise NotImplementedError("own_head=False sends the donor state into the base head, the write-back "
                                      "design that failed in runs 14 to 22; a top keeps its own head")
        if cut_host < 0 or cut_host > self.n_layers:
            raise ValueError(f"cut_host must be in [0, {self.n_layers}], got {cut_host}")
        if device is None:
            device = model_device(self.base)
        if cut_donor is None:
            if isinstance(donor, nn.Module):
                n_donor = len(donor.model.layers)
            else:
                n_donor = int(_text_config(donor_config_of(donor)).num_hidden_layers)
            cut_donor = min(n_donor - 1, int(round(cut_host / self.n_layers * n_donor)))
        parts = donor_parts(donor, cut_donor, control, control_seed, device, cache_dir)

        report = {"checked": False}
        if check:
            if donor_tokenizer is None and not isinstance(donor, nn.Module):
                import transformers

                donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor)
            if donor_tokenizer is not None:
                report = check_vocab(self.tokenizer, donor_tokenizer, parts["rows"], self.rows)
        head_weight, pad_start = fit_head_rows(parts["head_weight"], self.rows)
        head_mask = torch.zeros(self.rows, dtype=torch.bool)
        if pad_start is not None:
            head_mask[pad_start:] = True
        for index in report.get("masked_ids", []):
            head_mask[index] = True
        head = nn.Linear(parts["d_model"], self.rows, bias=False)
        head.weight = nn.Parameter(head_weight.clone(), requires_grad=False)
        head_mask = head_mask.to(device)

        stitch = stitch_lib.build_stitch({"kind": stitch_kind, "hidden": stitch_hidden}, self.d_model,
                                         parts["d_model"])
        if self.d_model == parts["d_model"]:
            with torch.no_grad():
                eye = torch.eye(self.d_model)
                stitch_lib.init_from_ridge(stitch, eye)
        spec = {"name": name, "donor": donor if isinstance(donor, str) else None, "cut_host": int(cut_host),
                "cut_donor": int(cut_donor), "stitch_kind": stitch_kind, "stitch_hidden": stitch_hidden,
                "own_head": True, "control": control, "control_seed": int(control_seed),
                "d_donor": parts["d_model"], "donor_layers": parts["n_layers"], "donor_rows": parts["rows"],
                "head_rows_dropped": max(0, parts["rows"] - self.rows), "head_rows_padded": 0 if pad_start is None
                else self.rows - parts["rows"], "head_masked": int(head_mask.sum()), "vocab": report,
                "donor_tied_head": parts["tied"]}
        top = TopModule(spec, stitch, parts["layers"], parts["rotary"], parts["windows"], parts["norm"], head,
                        head_mask, self)
        top.to(device)
        self.tops[name] = top
        return top

    # ---- running pieces

    def encode(self, texts, seq_len=None):
        """Tokenize a batch with left padding. Return (input_ids, attention_mask) on the model device."""
        tok = self.tokenizer
        old = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            kwargs = {"return_tensors": "pt", "padding": True}
            if seq_len is not None:
                kwargs["truncation"] = True
                kwargs["max_length"] = seq_len
            enc = tok(list(texts), **kwargs)
        finally:
            tok.padding_side = old
        device = model_device(self.base)
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def run_trunk(self, ids, mask, position_ids, upto, cache, past):
        """Base layers 0..upto-1 over token ids. Return the hidden state after layer upto-1 (the embeddings for 0)."""
        inner = self.base.model
        h = inner.embed_tokens(ids)
        if upto == 0:
            return h
        return run_layers(inner.layers[:upto], self.trunk_windows[:upto], inner.rotary_emb, h, mask, position_ids,
                          cache, past)

    def check_usable(self, top):
        if not top.accepted and not self.allow_flagged:
            raise ValueError(f"top {top.name!r} is flagged ({top.flag}); set allow_flagged to run it anyway")

    def forward_group(self, name, ids, mask, keep):
        pos = positions_of(mask)
        if name == RESERVED:
            return self.base(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=keep).logits
        top = self.tops[name]
        h = self.run_trunk(ids, mask, pos, top.cut_host, None, 0)
        x = top.run(h, mask, pos, None, 0, first_index(mask))
        if keep:
            x = x[:, -keep:]
        return top.logits(x)

    # ---- router

    def trunk_features(self, ids, mask, cut):
        """Mean over the prompt tokens of the normed trunk state at the cut, leaving out the first real token."""
        pos = positions_of(mask)
        h = self.run_trunk(ids, mask, pos, cut, None, 0)
        inner = self.base.model
        if cut < self.n_layers:
            norm = inner.layers[cut].input_layernorm
        else:
            norm = inner.norm
        x = norm(h).float()
        keep = mask.clone().long()
        first = first_index(mask)
        rows = torch.arange(ids.shape[0], device=ids.device)
        several = keep.sum(dim=-1) > 1
        keep[rows[several], first[several]] = 0
        weight = keep.to(x.dtype).unsqueeze(-1)
        return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)

    def router_classes(self):
        names = [RESERVED]
        for name in self.tops:
            if self.tops[name].accepted or self.allow_flagged:
                names.append(name)
        return names

    def batch_features(self, texts, cut, max_length=384, batch_size=16):
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                ids, mask = self.encode(texts[start:start + batch_size], max_length)
                out.append(self.trunk_features(ids, mask, cut).double().cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def fit_router(self, texts_by_class, lam=0.01, scale=ROUTER_SCALE, base_margin=0.0, max_length=384,
                   batch_size=16):
        """Ridge router: one-hot class labels regressed on the trunk's mean state at the smallest cut.

        texts_by_class maps "base" (general text) and each accepted top name
        to a list of prompts. lam is relative to the mean diagonal of the
        covariance, since Qwen states have outlier dims. The logits are
        scale times the regression output. Returns a report.
        """
        classes = self.router_classes()
        if len(classes) < 2:
            raise ValueError("a router needs at least one accepted top")
        for name in classes:
            if name not in texts_by_class or len(texts_by_class[name]) == 0:
                raise ValueError(f"no router texts for class {name!r}")
        cuts = []
        for name in classes[1:]:
            cuts.append(self.tops[name].cut_host)
        cut = min(cuts)
        feats = []
        labels = []
        for index, name in enumerate(classes):
            f = self.batch_features(texts_by_class[name], cut, max_length, batch_size)
            feats.append(f)
            labels += [index] * f.shape[0]
        x = torch.cat(feats, dim=0)
        y = torch.zeros(x.shape[0], len(classes), dtype=torch.float64)
        y[torch.arange(x.shape[0]), torch.tensor(labels)] = 1.0
        mx = x.mean(dim=0)
        my = y.mean(dim=0)
        xc = x - mx
        yc = y - my
        cxx = xc.T @ xc / x.shape[0]
        cxy = xc.T @ yc / x.shape[0]
        lam_eff = lam * float(torch.diagonal(cxx).mean())
        weight = torch.linalg.solve(cxx + lam_eff * torch.eye(cxx.shape[0], dtype=cxx.dtype), cxy)
        bias = my - mx @ weight
        device = model_device(self.base)
        self.router_w = (scale * weight).float().to(device)
        self.router_b = (scale * bias).float().to(device)
        self.router_names = classes
        self.router_cut = cut
        self.router_scale = scale
        self.base_margin = float(base_margin)
        scores = x.float().to(device) @ self.router_w + self.router_b
        scores[:, 0] += self.base_margin
        pick = scores.argmax(dim=-1).cpu()
        accuracy = {}
        start = 0
        for index, name in enumerate(classes):
            n = feats[index].shape[0]
            accuracy[name] = float((pick[start:start + n] == index).float().mean())
            start += n
        return {"classes": classes, "cut": cut, "train_accuracy": accuracy, "lam": lam, "scale": scale,
                "base_margin": self.base_margin, "texts": {k: len(texts_by_class[k]) for k in classes}}

    def route_scores(self, ids, mask):
        feats = self.trunk_features(ids, mask, self.router_cut)
        scores = feats.to(self.router_w.dtype) @ self.router_w + self.router_b
        scores = scores.clone()
        scores[:, 0] += self.base_margin
        return scores

    @torch.no_grad()
    def route(self, ids, mask):
        """The chosen top name for every row of a batch."""
        if self.router_w is None:
            raise ValueError("no router is fitted; call fit_router or pass top=")
        picks = self.route_scores(ids, mask).argmax(dim=-1).tolist()
        names = []
        for pick in picks:
            names.append(self.router_names[pick])
        return names

    @torch.no_grad()
    def route_texts(self, texts, max_length=384, batch_size=16):
        """The router's choice for each text, as names."""
        out = []
        for start in range(0, len(texts), batch_size):
            ids, mask = self.encode(texts[start:start + batch_size], max_length)
            out += self.route(ids, mask)
        return out

    def route_shares(self, texts, max_length=384, batch_size=16):
        picks = self.route_texts(texts, max_length, batch_size)
        shares = {}
        for name in self.router_names:
            shares[name] = 0.0
        for pick in picks:
            shares[pick] += 1.0 / max(1, len(picks))
        return shares

    def tune_base_margin(self, probe_by_class, margins=(0.0, 1.0, 2.0, 3.0, 4.0, 6.0), min_base_share=0.98,
                         min_recall=0.9, max_length=384):
        """Pick the smallest margin that keeps general probe text on the base and domain probe text on its top.

        Scans margins in order. The pick is the first margin where the base
        class sends at least min_base_share of its probe texts to the base
        and every top keeps at least min_recall. When none qualifies, the
        margin with the best (base share + worst recall) wins. Sets
        base_margin and returns the table.
        """
        feats = {}
        for name in self.router_names:
            feats[name] = []
        with torch.no_grad():
            for name in self.router_names:
                texts = probe_by_class[name]
                f = self.batch_features(texts, self.router_cut, max_length)
                feats[name] = f.float().to(self.router_w.device)
        table = []
        best = None
        chosen = None
        for margin in margins:
            row = {"margin": margin, "recall": {}}
            for index, name in enumerate(self.router_names):
                scores = feats[name] @ self.router_w + self.router_b
                scores[:, 0] += margin
                row["recall"][name] = float((scores.argmax(dim=-1) == index).float().mean())
            base_share = row["recall"][RESERVED]
            worst = min([row["recall"][n] for n in self.router_names[1:]])
            row["base_share"] = base_share
            row["worst_top_recall"] = worst
            table.append(row)
            if chosen is None and base_share >= min_base_share and worst >= min_recall:
                chosen = margin
            quality = base_share + worst
            if best is None or quality > best[0]:
                best = (quality, margin)
        if chosen is None:
            chosen = best[1]
        self.base_margin = float(chosen)
        return {"chosen": chosen, "table": table}

    # ---- forward and generation

    def choose(self, ids, mask, top):
        if top is not None:
            if top != RESERVED:
                self.check_usable(self.get_top(top))
            return [top] * ids.shape[0]
        names = self.route(ids, mask)
        for name in names:
            if name != RESERVED:
                self.check_usable(self.tops[name])
            if self.tally is not None:
                self.tally[name] = self.tally.get(name, 0) + 1
        return names

    @staticmethod
    def group_rows(names):
        groups = {}
        order = []
        for row, name in enumerate(names):
            if name not in groups:
                groups[name] = []
                order.append(name)
            groups[name].append(row)
        return order, groups

    def forward(self, input_ids, attention_mask=None, top=None, logits_to_keep=0, **ignored):
        """Logits for a batch. top forces one top ("base" for the base); None routes each row.

        Returns an object with .logits and .choice (the top name of every
        row). Rows with different choices run in groups.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        names = self.choose(input_ids, attention_mask, top)
        order, groups = self.group_rows(names)
        if len(order) == 1:
            self.last_choice = names
            logits = self.forward_group(order[0], input_ids, attention_mask, logits_to_keep)
            return SimpleNamespace(logits=logits, choice=names)
        out = None
        for name in order:
            rows = torch.tensor(groups[name], device=input_ids.device)
            part = self.forward_group(name, input_ids[rows], attention_mask[rows], logits_to_keep)
            if out is None:
                out = part.new_zeros((input_ids.shape[0],) + tuple(part.shape[1:]))
            out[rows] = part.to(out.dtype)
        self.last_choice = names
        return SimpleNamespace(logits=out, choice=names)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=20, eos_token_id=None, pad_token_id=None,
                 top=None, do_sample=False, stopping_criteria=None, **extra):
        """Greedy generation. Routes once from the prompt, then decodes with KV caches.

        Rows with different choices decode in separate groups. A group keeps
        a trunk cache (base layers up to the chosen cut) and the top's own
        private cache. Rows that end are padded. The base group runs the
        base model's own generate. Returns prompt plus new tokens.
        """
        if do_sample:
            raise ValueError("SkillTops.generate is greedy only")
        for key in extra:
            if key in ("temperature", "top_p", "top_k", "use_cache") and extra[key] in (None, True):
                continue
            if key == "repetition_penalty" and extra[key] in (None, 1.0):
                continue
            if extra[key] is None:
                continue
            raise TypeError(f"generate does not support {key}={extra[key]!r}")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        eos = []
        if isinstance(eos_token_id, int):
            eos = [eos_token_id]
        elif eos_token_id is not None:
            eos = [int(x) for x in eos_token_id]
        if pad_token_id is None:
            pad_token_id = eos[0] if len(eos) > 0 else 0
        names = self.choose(input_ids, attention_mask, top)
        order, groups = self.group_rows(names)
        pieces = {}
        width = 0
        for name in order:
            rows = torch.tensor(groups[name], device=input_ids.device)
            ids = input_ids[rows]
            mask = attention_mask[rows]
            if name == RESERVED:
                made = self.base.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                          do_sample=False, temperature=None, top_p=None, top_k=None,
                                          repetition_penalty=1.0, pad_token_id=pad_token_id,
                                          eos_token_id=eos if len(eos) > 0 else None,
                                          stopping_criteria=stopping_criteria)
                new = made[:, ids.shape[1]:]
            else:
                new = self.generate_top(self.tops[name], ids, mask, max_new_tokens, eos, pad_token_id,
                                        stopping_criteria)
            pieces[name] = (rows, new)
            width = max(width, new.shape[1])
        out = torch.full((input_ids.shape[0], width), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        for name in order:
            rows, new = pieces[name]
            out[rows, :new.shape[1]] = new.to(out.dtype)
        self.last_choice = names
        return torch.cat([input_ids, out], dim=1)

    def generate_top(self, top, ids, mask, max_new, eos, pad, criteria):
        from transformers import DynamicCache

        batch, length = ids.shape
        trunk_cache = DynamicCache()
        top_cache = DynamicCache()
        pos = positions_of(mask)
        h = self.run_trunk(ids, mask, pos, top.cut_host, trunk_cache, 0)
        x = top.run(h, mask, pos, top_cache, 0, first_index(mask))
        logits = top.logits(x[:, -1:])[:, -1]
        eos_ids = torch.tensor(eos, dtype=torch.long, device=ids.device) if len(eos) > 0 else None
        finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
        new = []
        current = mask
        total = length
        for step in range(max_new):
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, pad), token)
            new.append(token)
            if eos_ids is not None:
                finished = finished | torch.isin(token, eos_ids)
            if criteria is not None:
                done = criteria(torch.cat([ids, torch.stack(new, dim=1)], dim=1), None)
                finished = finished | done.to(finished.device)
            if bool(finished.all()) or step == max_new - 1:
                break
            current = torch.cat([current, torch.ones(batch, 1, dtype=current.dtype, device=current.device)], dim=1)
            position = current.long().sum(dim=-1, keepdim=True) - 1
            h = self.run_trunk(token[:, None], current, position, top.cut_host, trunk_cache, total)
            x = top.run(h, current, position, top_cache, total, None)
            logits = top.logits(x)[:, -1]
            total += 1
        return torch.stack(new, dim=1)

    # ---- fitting

    def per_token_losses(self, top, texts, donor_model, seq_len, native=None):
        """Per text: (native CE, top CE, keep), one text per batch. The donor side may be passed in.

        CE vectors run over every predicted token of the text. keep marks the
        ones that count when the top has a sink (the prediction made from the
        first real token is dropped). A passed-in native list lines up with
        the texts that have three or more tokens, in order, as an earlier
        call returned it.
        """
        out = []
        kept = 0
        with torch.no_grad():
            for text in texts:
                ids, mask = self.encode([text], seq_len)
                if ids.shape[1] < 3:
                    continue
                if native is not None:
                    native_ce = native[kept]
                else:
                    donor_logits = donor_model(input_ids=ids, attention_mask=mask).logits
                    ce, valid = lm_token_losses(donor_logits, ids, mask, False)
                    native_ce = ce[0][valid[0]].cpu()
                kept += 1
                logits = self.forward_group(top.name, ids, mask, 0)
                ce, valid = lm_token_losses(logits, ids, mask, False)
                top_ce = ce[0][valid[0]].cpu()
                keep = torch.ones(top_ce.shape[0], dtype=torch.bool)
                keep[0] = False
                out.append((native_ce, top_ce, keep))
        return out

    def penalty(self, top, texts, donor_model, seq_len=256, native=None):
        """Donor loss through the top minus the donor's native loss, token weighted, on texts.

        penalty leaves out the prediction made from the first real token
        when the top has a sink. penalty_all counts every token, as the
        stitch study did.
        """
        top = self.get_top(top)
        rows = self.per_token_losses(top, texts, donor_model, seq_len, native)
        sums = {"native": 0.0, "top": 0.0, "native_all": 0.0, "top_all": 0.0}
        n = 0
        n_all = 0
        for native_ce, top_ce, keep in rows:
            sums["native_all"] += float(native_ce.sum())
            sums["top_all"] += float(top_ce.sum())
            n_all += int(native_ce.numel())
            if top.has_sink:
                sums["native"] += float(native_ce[keep].sum())
                sums["top"] += float(top_ce[keep].sum())
                n += int(keep.sum())
            else:
                sums["native"] += float(native_ce.sum())
                sums["top"] += float(top_ce.sum())
                n += int(native_ce.numel())
        if n == 0 or n_all == 0:
            raise ValueError("no text has three or more tokens")
        return {"native_loss": sums["native"] / n, "top_loss": sums["top"] / n,
                "penalty": (sums["top"] - sums["native"]) / n,
                "penalty_all": (sums["top_all"] - sums["native_all"]) / n_all,
                "tokens": n, "texts": len(rows), "drop_first": top.has_sink,
                "native_ce": [row[0] for row in rows]}

    def fit_stitch(self, top, texts, steps=800, lr=None, batch_size=8, seq_len=256, seed=0, eval_texts=None,
                   donor_model=None, ridge_texts=None, max_penalty=DEFAULT_MAX_PENALTY, lora_rank=0,
                   lora_alpha=None, log_every=0, log_name="stitch", objective="kd", lm_weight=0.0):
        """Ridge init, then train the stitch (and an optional top LoRA) on the donor's own LM loss.

        Reports the penalty against the donor run natively on the same eval
        texts, before and after. A penalty above max_penalty flags the top:
        accepted becomes False, and the top is refused (forward raises, the
        router leaves it out) unless allow_flagged is set. A loss that more
        than doubles after the first 25 steps stops training, restores the
        start, and flags the top.
        """
        top = self.get_top(top)
        start = time.time()
        if donor_model is None:
            donor_model = self.load_donor_model(top)
        donor_model.eval()
        for param in donor_model.parameters():
            param.requires_grad_(False)
        texts = stitch_lib.lm_texts(self.tokenizer, texts, seq_len)
        if len(texts) < 4:
            raise ValueError("fit_stitch needs at least four training texts")
        if eval_texts is None:
            hold = max(2, min(100, len(texts) // 10))
            eval_texts = texts[-hold:]
            texts = texts[:-hold]
        report = {"steps": steps, "batch_size": batch_size, "seq_len": seq_len, "seed": seed,
                  "train_texts": len(texts), "eval_texts": len(eval_texts), "max_penalty": max_penalty,
                  "lora_rank": lora_rank, "stitch_kind": top.spec["stitch_kind"]}
        device = model_device(self.base)
        ridge_texts = ridge_texts if ridge_texts is not None else texts[:300]
        with torch.no_grad():
            if top.cut_donor > 0:
                sink = stitch_lib.donor_sink(donor_model, self.tokenizer, top.cut_donor, ridge_texts)
                top.sink.copy_(sink.to(top.sink.device))
                top.has_sink = True
            if top.cut_host > 0:
                report["ridge"] = stitch_lib.ridge_init(top.stitch, self.base, self.tokenizer, top.cut_host - 1,
                                                        donor_model, self.tokenizer, top.cut_donor, ridge_texts)
        before = self.penalty(top, eval_texts, donor_model, seq_len)
        native = before.pop("native_ce")
        report["before"] = before
        if lora_rank > 0:
            from wsp.lora import apply_lora, default_alpha

            alpha = lora_alpha if lora_alpha is not None else default_alpha(lora_rank)
            apply_lora(top, lora_rank, alpha)
            top.lora = {"rank": int(lora_rank), "alpha": alpha}
        if lr is None:
            lr = LR_BY_KIND.get(top.spec["stitch_kind"], 3e-4)
        params = top.trainable()
        for param in params:
            param.requires_grad_(True)
        report["lr"] = lr
        report["objective"] = objective
        report["lm_weight"] = lm_weight
        report["trainable_params"] = count(params)
        start_state = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        next_batch = stitch_lib.text_sampler(texts, batch_size, seed)
        losses = []
        diverged = False
        fit_start = time.time()
        for step in range(steps):
            ids, mask = self.encode(next_batch(), seq_len)
            pos = positions_of(mask)
            with torch.no_grad():
                h = self.run_trunk(ids, mask, pos, top.cut_host, None, 0)
            x = top.run(h, mask, pos, None, 0, first_index(mask))
            logits = top.logits(x)
            if objective == "kd":
                # Match the donor's own next-token distribution, so the top
                # reproduces what the donor would say. Training on text alone
                # (objective "lm") rewards a better language model, which can
                # move the top away from the donor's skill: round 24's smoke
                # beat the donor's loss by 0.7 nats and lost 26 GSM8K points.
                with torch.no_grad():
                    teacher = donor_model(input_ids=ids.to(model_device(donor_model)),
                                          attention_mask=mask.to(model_device(donor_model))).logits
                loss = kd_loss_value(logits, teacher.to(logits.device), mask, top.has_sink)
                if lm_weight > 0:
                    loss = loss + lm_weight * lm_loss_value(logits, ids, mask, top.has_sink)
            elif objective == "lm":
                loss = lm_loss_value(logits, ids, mask, top.has_sink)
            else:
                raise ValueError("objective must be 'kd' or 'lm'")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            if log_every and (step + 1) % log_every == 0:
                print(f"[{log_name}] step {step + 1}/{steps} loss {value:.4f}", flush=True)
            if not (value == value) or value == float("inf"):
                diverged = True
            if step == 24:
                early = sum(losses) / len(losses)
            if step >= 49 and step % 25 == 24:
                recent = sum(losses[-25:]) / 25.0
                if recent > 2.0 * early:
                    diverged = True
            if diverged:
                break
        if diverged:
            with torch.no_grad():
                for param, saved in zip(params, start_state):
                    param.copy_(saved)
        for param in params:
            param.requires_grad_(False)
        report["fit_seconds"] = round(time.time() - fit_start, 1)
        report["loss_first"] = losses[0] if losses else None
        report["loss_last25"] = sum(losses[-25:]) / max(1, len(losses[-25:]))
        report["steps_run"] = len(losses)
        report["diverged"] = diverged
        after = self.penalty(top, eval_texts, donor_model, seq_len, native=native)
        after.pop("native_ce")
        report["after"] = after
        report["penalty"] = after["penalty"]
        top.accepted = (after["penalty"] <= max_penalty) and not diverged
        top.flag = None
        if diverged:
            top.flag = "training diverged"
        elif after["penalty"] > max_penalty:
            top.flag = f"penalty {after['penalty']:.3f} nats is above {max_penalty}"
        report["accepted"] = top.accepted
        report["flag"] = top.flag
        report["seconds"] = round(time.time() - start, 1)
        top.fit_report = report
        return report

    def load_donor_model(self, top):
        import transformers

        donor = top.spec["donor"]
        if donor is None:
            raise ValueError("this top came from a loaded model; pass donor_model to fit_stitch")
        model = transformers.AutoModelForCausalLM.from_pretrained(donor, dtype=layer_dtype(top.layers))
        return model.to(model_device(self.base)).eval()

    # ---- counts

    def count_params(self):
        """Stored params and the params run per token on each path.

        Run per token follows wsp.skills_eval.model_params: the input
        embedding is a lookup (not counted) unless the head is tied to it.
        A top path runs the trunk layers, the stitch, the top's layers, norm,
        and head. The router runs once per sequence and is listed apart.
        """
        inner = self.base.model
        embed = self.base.get_input_embeddings().weight
        head = self.base.get_output_embeddings().weight
        tied = head is embed
        base_total = count(self.base.parameters())
        base_run = base_total if tied else base_total - embed.numel()
        layer_params = []
        for layer in inner.layers:
            layer_params.append(count(layer.parameters()))
        tops = {}
        stored = base_total
        run = {RESERVED: base_run}
        for name in self.tops:
            top = self.tops[name]
            lora = 0
            layers_only = 0
            for key, param in top.layers.named_parameters():
                if key.endswith("lora_A") or key.endswith("lora_B"):
                    lora += param.numel()
                else:
                    layers_only += param.numel()
            entry = {"stitch": count(top.stitch.parameters()), "layers": layers_only, "lora": lora,
                     "norm": count(top.norm.parameters()), "head": count(top.head.parameters()),
                     "sink": top.spec["d_donor"] if top.has_sink else 0}
            entry["stored"] = entry["stitch"] + entry["layers"] + entry["lora"] + entry["norm"] + entry["head"] \
                + entry["sink"]
            entry["trunk"] = sum(layer_params[:top.cut_host])
            entry["run"] = entry["trunk"] + entry["stored"]
            tops[name] = entry
            stored += entry["stored"]
            run[name] = entry["run"]
        router = 0
        if self.router_w is not None:
            router = self.router_w.numel() + self.router_b.numel()
        stored += router
        return {"base_stored": base_total, "base_tied_head": tied, "tops": tops, "router": router,
                "stored": stored, "run": run, "added_over_base": (stored - base_total) / base_total,
                "stored_over_base": stored / base_total}

    # ---- save and load

    def save(self, directory, names=None):
        """Write stitches, sinks, LoRAs, and the router to one directory. Donor layers are not saved: they refetch.

        names limits the save to some tops (and leaves the router out).
        """
        from safetensors.torch import save_file

        os.makedirs(directory, exist_ok=True)
        tensors = {}
        metas = []
        for name in self.tops:
            if names is not None and name not in names:
                continue
            top = self.tops[name]
            if top.spec["donor"] is None:
                raise ValueError(f"top {name!r} came from a loaded model and cannot be saved by reference")
            for key, value in top.stitch.state_dict().items():
                tensors[f"{name}.stitch.{key}"] = value.detach().cpu().contiguous()
            if top.has_sink:
                tensors[f"{name}.sink"] = top.sink.detach().cpu().contiguous()
            for key, param in top.layers.named_parameters():
                if key.endswith("lora_A") or key.endswith("lora_B"):
                    tensors[f"{name}.lora.{key}"] = param.detach().cpu().float().contiguous()
            meta = dict(top.spec)
            meta["has_sink"] = top.has_sink
            meta["accepted"] = top.accepted
            meta["flag"] = top.flag
            meta["lora"] = top.lora
            meta["fit_report"] = json_safe(top.fit_report)
            metas.append(meta)
        router = None
        if self.router_w is not None and names is None:
            tensors["router.w"] = self.router_w.detach().cpu().contiguous()
            tensors["router.b"] = self.router_b.detach().cpu().contiguous()
            router = {"names": self.router_names, "cut": self.router_cut, "scale": self.router_scale,
                      "base_margin": self.base_margin}
        if len(tensors) == 0:
            tensors["empty"] = torch.zeros(1)
        save_file(tensors, os.path.join(directory, "tops.safetensors"))
        with open(os.path.join(directory, "tops.json"), "w") as f:
            json.dump({"version": 1, "base": self.base_id, "tops": metas, "router": router}, f, indent=2)
        return directory

    def add_saved_tops(self, directory, names=None, device=None, cache_dir=None):
        """Add the tops saved in a directory (all, or the named ones). Donor layers refetch by byte range."""
        from safetensors.torch import load_file

        with open(os.path.join(directory, "tops.json")) as f:
            meta = json.load(f)
        tensors = load_file(os.path.join(directory, "tops.safetensors"))
        added = []
        for entry in meta["tops"]:
            name = entry["name"]
            if names is not None and name not in names:
                continue
            top = self.add_top(name, entry["donor"], entry["cut_host"], entry["cut_donor"],
                               stitch_kind=entry["stitch_kind"], control=entry["control"],
                               control_seed=entry["control_seed"], device=device, cache_dir=cache_dir,
                               stitch_hidden=entry.get("stitch_hidden"))
            if entry.get("lora"):
                from wsp.lora import apply_lora

                apply_lora(top, entry["lora"]["rank"], entry["lora"]["alpha"])
                top.lora = entry["lora"]
            state = {}
            prefix = f"{name}.stitch."
            for key in tensors:
                if key.startswith(prefix):
                    state[key[len(prefix):]] = tensors[key]
            top.stitch.load_state_dict(state)
            lora_prefix = f"{name}.lora."
            for key, param in top.layers.named_parameters():
                full = lora_prefix + key
                if full in tensors:
                    with torch.no_grad():
                        param.copy_(tensors[full].to(param.device))
            if entry.get("has_sink"):
                with torch.no_grad():
                    top.sink.copy_(tensors[f"{name}.sink"].to(top.sink.device))
                top.has_sink = True
            top.accepted = entry.get("accepted", True)
            top.flag = entry.get("flag")
            top.fit_report = entry.get("fit_report")
            added.append(name)
        if meta["router"] is not None and names is None:
            target = model_device(self.base)
            self.router_w = tensors["router.w"].to(target)
            self.router_b = tensors["router.b"].to(target)
            self.router_names = meta["router"]["names"]
            self.router_cut = meta["router"]["cut"]
            self.router_scale = meta["router"]["scale"]
            self.base_margin = meta["router"]["base_margin"]
        return added

    @classmethod
    def load(cls, directory, base_model=None, tokenizer=None, device=None, dtype=None, cache_dir=None):
        """Rebuild a saved network. The base loads by its id unless base_model is given."""
        with open(os.path.join(directory, "tops.json")) as f:
            meta = json.load(f)
        if base_model is None:
            import transformers

            base_model = transformers.AutoModelForCausalLM.from_pretrained(meta["base"], dtype=dtype)
            if device is not None:
                base_model = base_model.to(device)
        if tokenizer is None:
            import transformers

            tokenizer = transformers.AutoTokenizer.from_pretrained(meta["base"])
        network = cls(base_model, tokenizer, base_id=meta["base"])
        network.add_saved_tops(directory, None, device, cache_dir)
        return network


class ForcedView:
    """A model-shaped handle that forces one top, for code that calls model(...) and model.generate(...)."""

    def __init__(self, network, top):
        self.network = network
        self.top = top
        self.generation_config = network.generation_config
        self.config = network.config

    def parameters(self):
        return self.network.parameters()

    def get_input_embeddings(self):
        return self.network.get_input_embeddings()

    def get_output_embeddings(self):
        return self.network.get_output_embeddings()

    def __call__(self, *args, **kwargs):
        kwargs["top"] = self.top
        return self.network(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs["top"] = self.top
        return self.network.generate(*args, **kwargs)

    def eval(self):
        return self


def fit_stitch(top, texts, **kwargs):
    """Module level form of SkillTops.fit_stitch: top is a TopModule."""
    return top.owner().fit_stitch(top, texts, **kwargs)


def json_safe(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


# ---------------------------------------------------------------- reference arm: distillation into a LoRA


def fit_kd_lora(base_model, donor_model, tokenizer, texts, rank, steps=800, lr=2e-4, batch_size=8, seq_len=256,
                seed=0):
    """Distill the donor into a LoRA on the base: KL(donor || base) on the same texts and steps as the stitch fit.

    The LoRA goes on every decoder layer (attention and MLP linears). Both
    models share a tokenizer. The KL runs over the rows both heads have.
    Returns {"params", "seconds", "loss_first", "loss_last25"}; the base
    model is changed in place.
    """
    from wsp.lora import apply_lora, default_alpha

    start = time.time()
    texts = stitch_lib.lm_texts(tokenizer, texts, seq_len)
    params = apply_lora(base_model, rank, default_alpha(rank))
    for param in params:
        param.requires_grad_(True)
    base_model.eval()
    donor_model.eval()
    for param in donor_model.parameters():
        param.requires_grad_(False)
    device = model_device(base_model)
    rows = min(base_model.get_output_embeddings().weight.shape[0], donor_model.get_output_embeddings().weight.shape[0])
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    next_batch = stitch_lib.text_sampler(texts, batch_size, seed)
    old = tokenizer.padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    losses = []
    for step in range(steps):
        tokenizer.padding_side = "left"
        try:
            enc = tokenizer(next_batch(), return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        finally:
            tokenizer.padding_side = old
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        pos = positions_of(mask)
        with torch.no_grad():
            teacher = donor_model(input_ids=ids.to(model_device(donor_model)),
                                  attention_mask=mask.to(model_device(donor_model)),
                                  position_ids=pos.to(model_device(donor_model))).logits[..., :rows].float()
        student = base_model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[..., :rows].float()
        log_t = F.log_softmax(teacher.to(device), dim=-1)
        log_s = F.log_softmax(student, dim=-1)
        kl = (log_t.exp() * (log_t - log_s)).sum(dim=-1)
        weight = mask.to(kl.dtype)
        loss = (kl * weight).sum() / weight.sum().clamp(min=1.0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    for param in params:
        param.requires_grad_(False)
    return {"params": count(params), "rank": rank, "seconds": round(time.time() - start, 1),
            "loss_first": losses[0] if losses else None,
            "loss_last25": sum(losses[-25:]) / max(1, len(losses[-25:])), "steps": steps}


# ---------------------------------------------------------------- sizes from configs


def config_counts(config):
    """Param counts of one model from its config alone (built on the meta device)."""
    meta = _build_meta_model(config)
    text = _text_config(config)
    _, layers = find_decoder_layers(meta, 0)
    per_layer = []
    for layer in layers:
        per_layer.append(sum(p.numel() for p in layer.parameters()))
    inner = meta.model
    embed = inner.embed_tokens.weight.numel()
    norm = inner.norm.weight.numel()
    rows = int(inner.embed_tokens.weight.shape[0])
    tied = bool(getattr(text, "tie_word_embeddings", False))
    head = 0 if tied else meta.get_output_embeddings().weight.numel()
    total = embed + sum(per_layer) + norm + head
    run = total if tied else total - embed
    return {"layers": per_layer, "embed": embed, "norm": norm, "head": head, "total": total, "run": run,
            "tied": tied, "rows": rows, "d": int(text.hidden_size), "n_layers": len(per_layer)}


def stitch_param_count(kind, d_in, d_out, hidden=None):
    with torch.device("meta"):
        stitch = stitch_lib.build_stitch({"kind": kind, "hidden": hidden}, d_in, d_out)
    return sum(p.numel() for p in stitch.parameters())


def sizes_from_configs(base_config, tops, router_classes=None):
    """Stored and per-token params of a SkillTops network, from configs alone.

    tops is a list of dicts: name, donor_config, cut_host, and optional
    cut_donor, stitch_kind (mlp), lora_rank (0). Matches count_params of
    the built network (with a sink on every top, the router when
    router_classes is set). Returns stored, per-path run, the multiple of the
    base, and the per-token cost of the base and of each donor.
    """
    from wsp.lora import lora_param_count

    base = config_counts(base_config)
    stored = base["total"]
    run = {RESERVED: base["run"]}
    out_tops = {}
    for spec in tops:
        donor = config_counts(spec["donor_config"])
        cut_host = spec["cut_host"]
        cut_donor = spec.get("cut_donor")
        if cut_donor is None:
            cut_donor = min(donor["n_layers"] - 1, int(round(cut_host / base["n_layers"] * donor["n_layers"])))
        kind = spec.get("stitch_kind", "mlp")
        stitch = stitch_param_count(kind, base["d"], donor["d"], spec.get("stitch_hidden"))
        layers = sum(donor["layers"][cut_donor:])
        head = base["rows"] * donor["d"]
        lora = 0
        rank = spec.get("lora_rank", 0)
        if rank > 0:
            lora = lora_param_count(spec["donor_config"], rank, layers=list(range(cut_donor, donor["n_layers"])))
        sink = donor["d"] if cut_donor > 0 else 0
        entry = {"cut_host": cut_host, "cut_donor": cut_donor, "stitch": stitch, "layers": layers,
                 "norm": donor["norm"], "head": head, "lora": lora, "sink": sink,
                 "trunk": sum(base["layers"][:cut_host]), "donor_total": donor["total"], "donor_run": donor["run"]}
        entry["stored"] = stitch + layers + donor["norm"] + head + lora + sink
        entry["run"] = entry["trunk"] + entry["stored"]
        out_tops[spec["name"]] = entry
        stored += entry["stored"]
        run[spec["name"]] = entry["run"]
    router = 0
    if router_classes:
        router = router_classes * base["d"] + router_classes
    stored += router
    return {"base_total": base["total"], "base_run": base["run"], "tops": out_tops, "router": router,
            "stored": stored, "stored_over_base": stored / base["total"], "run": run}
