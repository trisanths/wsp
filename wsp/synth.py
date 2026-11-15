"""Synthesis compiler: mined components in, one network out.

Other agents mine components (blocks, attention heads, MLP clusters,
experts) from donor models. synthesize() compiles them into one host
network. Every component is a residual branch at its host depth:

    h + gate * T_out(component(T_in(h)))

A fresh network equals the base exactly, because every branch starts closed
(ramp 0). Nothing but the one network runs at inference. The seven stages
are described in docs/compiler.md:

    1 place        build the parts, attach the branches, count params
    2 translate in ridge fit, then match the component's native output
    3 translate out logit-preserving init or ridge to the MISSING residual,
                   then stage-1 heal (translations, gates only)
    4 synthesis    stage-2 heal: adds LoRA on host layers after the first
                   seam, with a preservation term against the frozen base
    5 gates        token gates train in both heals; reported per text set
    6 verify       practice sets, rollback, component test, random and scrambled controls
    7 budget       params total and per token, refusal over the limits

Heal data rule (owner, 2026-11-14): no benchmark data trains anything here.
The caller passes texts from wsp.healdata pools, and practice sets that
never receive gradient.
"""

import contextlib
import copy
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

from wsp.align import collect_pairs, mean_rms, ridge_fit
from wsp.assemble import default_host_layer, host_hidden_size, host_layer_count, load_donor, random_block_range, random_heads
from wsp.extract import random_neurons
from wsp.gate import TokenGate, real_tokens
from wsp.graft import Graft, attach_grafts, part_dtype
from wsp.lora import LoRALinear, apply_lora, load_lora_state, lora_modules, lora_param_count, lora_state_dict, merge_lora
from wsp.scramble import scramble_tensors
from wsp.parts import (
    AttnHeadsPart,
    BlockPart,
    _text_config,
    attn_tensor_names,
    block_layer_prefixes,
    block_tensor_names,
    build_attn_heads,
    build_block_part,
    build_expert_part,
    build_mlp_cluster,
    expert_tensor_names,
    find_decoder_layers,
    find_layer_modules,
    find_moe_block,
    mlp_tensor_names,
)
from wsp.skills_eval import evaluate_skill, model_params
from wsp.stitch import (
    MLPStitch,
    StopForward,
    check_same_vocab,
    donor_capture_for,
    donor_sink,
    stitch_penalty,
    text_sampler,
    train_stitch,
)

CONTROLS = (None, "random", "scrambled")
KINDS = ("block", "attn_heads", "mlp_cluster", "expert")
FORMAT = 1

DEFAULTS = {
    "steps_in": 60,            # T_in output-matching steps
    "steps_lm": 40,            # T_in donor-LM steps (blocks, same tokenizer)
    "batch_in": 4,
    "lr_in": 1e-3,
    "stage1_steps": 200,
    "stage2_steps": 200,
    "batch_size": 4,
    "seq_len": 256,
    "lr_translation": 1e-3,
    "lr_gate": 1e-2,
    "lr_translation2": 1e-4,
    "lr_gate2": 1e-3,
    "lr_lora": 5e-5,
    "lora_rank": 16,
    "kd_weight": 1.0,
    "lm_weight_kd": 0.1,
    "pres_weight": 1.0,
    "feat_weight": 0.1,
    "ramp_frac": 0.2,
    "warmup_frac": 0.05,
    "tolerance": 0.03,
    "margin": 0.01,
    "near_top": 2,
    "out_alpha": 0.3,
    "ridge_out_lam": 1e-2,
    "unembed_lam": 1e-4,
    "output_init": "auto",     # "auto" or "old" (zero T_out, LM only)
    "gate_bias": 2.0,
    "t_in_kind": "linear",
    "max_rows": 20000,
    "control_run": False,      # True: build the random control (same as controls=["random"])
    "controls": None,          # list of control kinds to build in verify, e.g. ["random", "scrambled"]
    "control_mode": "spectrum",  # scrambled controls: "spectrum" or "permute" (a component may set its own)
    "scramble_max_numel": None,  # spectrum mode: tensors with more entries use "permute" instead
    "kd": True,                # donor KL when the tokenizers match (False: LM loss only)
    "components_per_step": None,  # None: every component each step; k: k of them, round robin
    "component_tests": True,
    "test_limit": None,        # items per practice skill in the component test
    "seed": 0,
    "max_params": None,
    "max_params_per_token": None,
}


class BudgetError(ValueError):
    """The compiled network is over the requested parameter limits."""


# ---------------------------------------------------------------- components


def normalize_component(raw, index=0):
    """Return a plain component dict from a spec or a library catalog row.

    Accepted keys: kind or type; donor or model (an id string, or a loaded
    model); layers [first, last] for a block; layer plus indices (or heads,
    units, experts) for the other kinds; units may be the path of a .pt file
    with an "indices" list; skill (a name or a dict with "name"); texts;
    host_layer, or at ("auto" or None: same relative depth); control
    ("random", "scrambled", or None) with control_seed. "scrambled" keeps the
    real donor layers or units and rewrites their weights (wsp.scramble) with
    control_mode "spectrum" (default) or "permute".
    """
    c = dict(raw)
    kind = c.get("kind") or c.get("type")
    if kind not in KINDS:
        raise ValueError(f"component {index}: kind {kind!r} is not one of {KINDS}")
    donor = c.get("donor")
    if donor is None:
        donor = c.get("model")
    out = {"kind": kind, "donor": None, "donor_model": None, "donor_tokenizer": None}
    if isinstance(donor, str):
        out["donor"] = donor
    elif donor is not None:
        out["donor_model"] = donor
        out["donor"] = c.get("donor_id") or getattr(getattr(donor, "config", None), "_name_or_path", None) or f"donor{index}"
        out["donor_tokenizer"] = c.get("donor_tokenizer")
    else:
        raise ValueError(f"component {index}: no donor model id")
    out["layers"] = None
    out["layer"] = None
    out["indices"] = None
    if kind == "block":
        if c.get("layers") is None:
            raise ValueError(f"component {index}: a block needs layers [first, last]")
        out["layers"] = [int(c["layers"][0]), int(c["layers"][1])]
        out["layer"] = out["layers"][0]
    else:
        if c.get("layer") is None:
            raise ValueError(f"component {index}: kind {kind} needs a donor layer")
        out["layer"] = int(c["layer"])
        indices = None
        for key in ("indices", "heads", "experts", "units"):
            value = c.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                path = value
                if not os.path.isabs(path) and not os.path.exists(path):
                    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    path = os.path.join(root, path)
                value = torch.load(path)["indices"]
            indices = [int(v) for v in torch.as_tensor(value).reshape(-1).tolist()]
            break
        if indices is None:
            raise ValueError(f"component {index}: kind {kind} needs indices (heads, units, or experts)")
        out["indices"] = indices
    skill = c.get("skill")
    if isinstance(skill, dict):
        skill = skill.get("name")
    out["skill"] = skill
    out["texts"] = c.get("texts")
    at = c.get("host_layer", c.get("at"))
    if at == "auto":
        at = None
    out["host_layer"] = at
    control = c.get("control")
    if control not in CONTROLS:
        raise ValueError(f"component {index}: control must be None, 'random', or 'scrambled', got {control!r}")
    out["control"] = control
    control_mode = c.get("control_mode")
    if control_mode not in (None, "spectrum", "permute"):
        raise ValueError(f"component {index}: control_mode must be 'spectrum' or 'permute', got {control_mode!r}")
    out["control_mode"] = control_mode
    out["control_seed"] = int(c.get("control_seed", c.get("seed", 0)))
    out["id"] = c.get("id") or f"{kind}{index}"
    out["row"] = {k: c[k] for k in ("id", "label", "tier") if k in c}
    return out


def control_block_range(num_layers, first, last, seed):
    """Return a seeded [start, end] range of the length of [first, last] with the least overlap.

    random_block_range falls back to any shifted window when no disjoint
    range exists, which can share most layers with the real block. Here the
    choice is limited to the starts with the smallest overlap. Return
    (start, end, overlap_in_layers).
    """
    length = last - first + 1
    overlaps = {}
    for start in range(num_layers - length + 1):
        if start == first:
            continue
        end = start + length - 1
        overlaps[start] = max(0, min(end, last) - max(start, first) + 1)
    if len(overlaps) == 0:
        raise ValueError(f"block [{first}, {last}] covers all {num_layers} donor layers; no control range exists")
    least = min(overlaps.values())
    choices = [start for start in sorted(overlaps) if overlaps[start] == least]
    generator = torch.Generator()
    generator.manual_seed(seed)
    start = choices[int(torch.randint(len(choices), (1,), generator=generator))]
    return start, start + length - 1, least


def locate(comp, config):
    """Return comp with its concrete donor location, after the random control (if any).

    real_last is the last donor layer of the REAL block. translate_out reads
    it, so a control picks the same output init as the part it replaces.
    """
    comp = dict(comp)
    text_config = _text_config(config)
    layers = text_config.num_hidden_layers
    seed = comp["control_seed"]
    comp["donor_layers"] = layers
    comp["real_last"] = comp["layers"][1] if comp["kind"] == "block" else None
    comp["control_overlap"] = None
    if comp["control"] != "random":
        return comp
    if comp["kind"] == "block":
        first, last, overlap = control_block_range(layers, comp["layers"][0], comp["layers"][1], seed)
        comp["layers"] = [first, last]
        comp["control_overlap"] = overlap
    elif comp["kind"] == "attn_heads":
        comp["indices"] = random_heads(text_config.num_attention_heads, len(comp["indices"]), seed).tolist()
    elif comp["kind"] == "mlp_cluster":
        comp["indices"] = random_neurons(text_config.intermediate_size, len(comp["indices"]), seed).tolist()
    else:
        found = find_moe_block(config, comp["layer"])
        comp["indices"] = random_neurons(found["num_experts"], len(comp["indices"]), seed).tolist()
    return comp


def pick_tensors(model, names):
    """Return {name: tensor} for the donor checkpoint names, from the loaded donor."""
    state = model.state_dict()
    out = {}
    for name in names:
        if name not in state:
            raise KeyError(f"donor has no tensor {name}; the donor must be a causal LM with checkpoint names")
        out[name] = state[name].detach().clone()
    return out


def build_part(comp, donor_model, scramble_report=None):
    """Build the frozen part of a located component from a loaded donor. Return (part, tensors).

    A scrambled control takes the same tensors as the real component and
    rewrites them with wsp.scramble (comp["control_seed"], comp["control_mode"])
    before the part is built. The donor model itself is not changed. If
    scramble_report is a dict, the scramble report is copied into it.
    """
    config = donor_model.config
    kind = comp["kind"]
    if kind == "block":
        first, last = comp["layers"]
        tensors = pick_tensors(donor_model, block_tensor_names(config, first, last))
    else:
        layer = comp["layer"]
        indices = torch.as_tensor(comp["indices"], dtype=torch.long)
        if kind == "attn_heads":
            tensors = pick_tensors(donor_model, attn_tensor_names(config, layer))
        elif kind == "mlp_cluster":
            tensors = pick_tensors(donor_model, mlp_tensor_names(config, layer))
        else:
            tensors = pick_tensors(donor_model, expert_tensor_names(config, layer, indices.tolist()))
    if comp.get("control") == "scrambled":
        report = scramble_tensors(tensors, comp["control_seed"], comp.get("control_mode") or "spectrum",
                                  max_numel=comp.get("scramble_max_numel"))
        if scramble_report is not None:
            scramble_report.update(report)
    if kind == "block":
        return build_block_part(config, first, last, tensors), tensors
    if kind == "attn_heads":
        return build_attn_heads(config, layer, indices, tensors), tensors
    if kind == "mlp_cluster":
        return build_mlp_cluster(config, layer, indices, tensors), tensors
    return build_expert_part(config, layer, indices, tensors), tensors


def rebuild_part(meta, config, tensors):
    """Rebuild a part from saved tensors, the same way build_part made it."""
    kind = meta["kind"]
    if kind == "block":
        return build_block_part(config, meta["layers"][0], meta["layers"][1], tensors)
    indices = torch.as_tensor(meta["indices"], dtype=torch.long)
    if kind == "attn_heads":
        return build_attn_heads(config, meta["layer"], indices, tensors)
    if kind == "mlp_cluster":
        return build_mlp_cluster(config, meta["layer"], indices, tensors)
    return build_expert_part(config, meta["layer"], indices, tensors)


# -------------------------------------------------------------------- branch


class Branch(Graft):
    """One component as a residual branch: h + ramp * gate(h) * T_out(part(T_in(h))).

    Subclass of Graft, so attach_grafts keeps the KV caches of attention
    parts in step with the host cache. ramp is a plain float: 0 means the
    branch returns h untouched, so a fresh network equals the base exactly.
    The delta is zero at each sequence's first real position (the host's own
    sink state stays as it is).
    """

    def __init__(self, d_host, d_donor, part, t_in="linear", hidden=None, gate_bias=2.0):
        super().__init__(d_host, d_donor, part)
        self.t_in_kind = t_in
        if t_in == "mlp":
            self.w_in = MLPStitch(d_host, d_donor, hidden).float()
        elif t_in != "linear":
            raise ValueError(f"unknown t_in kind {t_in!r}")
        self.token_gate = TokenGate(d_host, gate_bias)
        self.ramp = 0.0
        self.forced_off = False
        self.closed = False
        self.record = False
        self.last_part_out = None

    def donor_input(self, h):
        """Translate host states to the donor's input space, with the sink at the first real token."""
        x = self.map_in(h)
        if self.sink is not None:
            x = self.put_sink(x)
        return x

    def run_part(self, x):
        dtype = part_dtype(self.part)
        if isinstance(self.part, (AttnHeadsPart, BlockPart)):
            if self.host_uses_cache:
                y = self.part(x.to(dtype), attention_mask=self.pad_mask, cache_len=self.host_cache_len)
            else:
                y = self.part(x.to(dtype), attention_mask=self.pad_mask)
        else:
            y = self.part(x.to(dtype))
        return y.to(torch.float32)

    def first_position(self, h):
        """Return a (batch, seq, 1) bool mask of each sequence's first real position."""
        batch, seq, _ = h.shape
        start = 0
        if self.host_uses_cache:
            start = self.host_cache_len
        if self.pad_mask is None:
            first = torch.zeros(batch, dtype=torch.long, device=h.device)
            has_real = torch.ones(batch, dtype=torch.bool, device=h.device)
        else:
            real = self.pad_mask.to(h.device) != 0
            first = real.long().argmax(dim=-1)
            has_real = real.any(dim=-1)
        positions = torch.arange(start, start + seq, device=h.device)
        hit = (positions.view(1, seq) == first.view(batch, 1)) & has_real.view(batch, 1)
        return hit.unsqueeze(-1)

    def forward(self, h):
        if self.forced_off or self.ramp == 0.0:
            return h
        x = self.donor_input(h)
        y = self.run_part(x)
        if self.record:
            self.last_part_out = y
        delta = self.w_out(y)
        gate = self.token_gate(h, real_tokens(self, h))
        delta = delta * gate * self.ramp
        delta = delta.masked_fill(self.first_position(h), 0.0)
        return h + delta.to(h.dtype)


@contextlib.contextmanager
def branches_off(model):
    """Run with every branch returning h untouched."""
    branches = list(getattr(model, "wsp_grafts", []))
    old = []
    for b in branches:
        old.append(b.forced_off)
        b.forced_off = True
    try:
        yield
    finally:
        for b, value in zip(branches, old):
            b.forced_off = value


@contextlib.contextmanager
def lora_off(model):
    """Run with every host LoRA scaled to zero."""
    mods = [m for _, m in lora_modules(model)]
    old = [m.scale for m in mods]
    for m in mods:
        m.scale = 0.0
    try:
        yield
    finally:
        for m, value in zip(mods, old):
            m.scale = value


@contextlib.contextmanager
def frozen_base(model):
    """The base network: no branches, no LoRA."""
    with branches_off(model):
        with lora_off(model):
            yield


def device_of(model):
    return next(model.parameters()).device


def same_vocab(a, b):
    try:
        check_same_vocab(a, b)
    except ValueError:
        return False
    return True


# ------------------------------------------------------------------- data


def encode_item(tokenizer, item, seq_len, truncate=False):
    """Return (ids, loss_mask) for one text or one prompt/answer pair, or None when it does not fit.

    A string trains on every token after the first. A pair (dict with
    prompt and answer, or a 2-tuple) trains on the answer tokens only,
    including the end token. An item longer than seq_len is dropped, never
    cut, unless truncate is True (used for fitting maps, not for targets).
    """
    if isinstance(item, str):
        ids = list(tokenizer(item)["input_ids"])
        mask = [0] + [1] * (len(ids) - 1)
    else:
        if isinstance(item, dict):
            prompt, answer = item["prompt"], item["answer"]
        else:
            prompt, answer = item
        if getattr(tokenizer, "chat_template", None):
            msgs = [{"role": "user", "content": prompt}]
            p_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            f_text = tokenizer.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)
            p_ids = list(tokenizer(p_text, add_special_tokens=False)["input_ids"])
            ids = list(tokenizer(f_text, add_special_tokens=False)["input_ids"])
            cut = 0
            while cut < len(p_ids) and cut < len(ids) and p_ids[cut] == ids[cut]:
                cut += 1
        else:
            p_ids = list(tokenizer(prompt)["input_ids"])
            a_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
            if tokenizer.eos_token_id is not None:
                a_ids = a_ids + [tokenizer.eos_token_id]
            ids = p_ids + a_ids
            cut = len(p_ids)
        mask = [0] * cut + [1] * (len(ids) - cut)
    if len(ids) > seq_len:
        if not truncate:
            return None
        ids = ids[:seq_len]
        mask = mask[:seq_len]
    if len(ids) < 2 or sum(mask[1:]) == 0:
        return None
    return ids, mask


def encode_items(tokenizer, items, seq_len, truncate=False):
    """Return (encoded, dropped): the encodable items and the count that did not fit."""
    out = []
    dropped = 0
    for item in items:
        e = encode_item(tokenizer, item, seq_len, truncate)
        if e is None:
            dropped += 1
        else:
            out.append(e)
    return out, dropped


def collate(encoded, pad_id, device):
    """Right-pad a list of (ids, loss_mask) into (ids, attention_mask, loss_mask) tensors."""
    width = max(len(e[0]) for e in encoded)
    ids = torch.full((len(encoded), width), int(pad_id), dtype=torch.long)
    attn = torch.zeros((len(encoded), width), dtype=torch.long)
    lmask = torch.zeros((len(encoded), width), dtype=torch.float32)
    for row, (i, m) in enumerate(encoded):
        ids[row, :len(i)] = torch.tensor(i)
        attn[row, :len(i)] = 1
        lmask[row, :len(m)] = torch.tensor(m, dtype=torch.float32)
    return ids.to(device), attn.to(device), lmask.to(device)


def pad_id_of(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def example_mean(values, mask):
    """Mean of values over mask per example, then over the examples that have tokens."""
    count = mask.sum(dim=1)
    per = (values * mask).sum(dim=1) / count.clamp(min=1.0)
    keep = (count > 0).float()
    return (per * keep).sum() / keep.sum().clamp(min=1.0)


def lm_loss(logits, ids, lmask):
    """Next-token cross entropy on the loss-mask tokens, averaged per example first."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return example_mean(nll, lmask[:, 1:])


def kl_tokens(teacher_logits, student_logits, temperature=1.0):
    """KL(teacher || student) per next-token position, on the shared vocabulary."""
    v = min(teacher_logits.shape[-1], student_logits.shape[-1])
    t = torch.log_softmax(teacher_logits[..., :v].float() / temperature, dim=-1)
    s = torch.log_softmax(student_logits[..., :v].float() / temperature, dim=-1)
    return (t.exp() * (t - s)).sum(-1) * temperature * temperature


def kl_loss(teacher_logits, student_logits, lmask, temperature=1.0):
    kl = kl_tokens(teacher_logits[:, :-1], student_logits[:, :-1], temperature)
    return example_mean(kl, lmask[:, 1:])


def forward_logits(model, ids, attn):
    return model(input_ids=ids, attention_mask=attn, use_cache=False).logits


def host_capture(model, ids, attn, layer_indices):
    """Run the host once. Return ({layer: output state}, logits). Hooks run after any branch."""
    _, layers = find_decoder_layers(model, 0)
    store = {}
    handles = []
    for idx in layer_indices:
        def hook(module, args, output, idx=idx):
            store[idx] = output[0] if isinstance(output, tuple) else output

        handles.append(layers[idx].register_forward_hook(hook))
    try:
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return store, out.logits


@contextlib.contextmanager
def scrambled_donor_layers(donor_model, branch, active):
    """Put the branch's scrambled tensors into the donor model, then restore the real ones.

    The donor-LM stitch fit runs the donor's own layers after the seam. For a
    scrambled control those layers must be the scrambled ones, or T_in would
    be fit to the real block.
    """
    if not active:
        yield
        return
    state = donor_model.state_dict()
    saved = {}
    try:
        with torch.no_grad():
            for name, tensor in branch.part_tensors.items():
                saved[name] = state[name].detach().clone()
                state[name].copy_(tensor.to(device=state[name].device, dtype=state[name].dtype))
        yield
    finally:
        with torch.no_grad():
            for name, tensor in saved.items():
                state[name].copy_(tensor)


def donor_point(donor_model, comp, ids, attn):
    """The donor's state at the component's input point on ids: (batch, seq, d_donor)."""
    holder = {}
    handles = []
    if comp["kind"] in ("block", "attn_heads"):
        _, layers = find_decoder_layers(donor_model, comp["layer"])
        module = layers[comp["layer"]]

        def pre(m, args, kwargs):
            holder["x"] = args[0] if len(args) > 0 else kwargs["hidden_states"]
            raise StopForward()

        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
    else:
        norm = donor_model.get_submodule(find_layer_modules(donor_model.config, comp["layer"])["pre_norm_prefix"])

        def pre_norm(m, args):
            holder["x"] = args[0]
            raise StopForward()

        handles.append(norm.register_forward_pre_hook(pre_norm))
    try:
        donor_model(input_ids=ids, attention_mask=attn, use_cache=False)
    except StopForward:
        pass
    finally:
        for handle in handles:
            handle.remove()
    return holder["x"]


def donor_exit_state(donor_model, comp, ids, attn):
    """The donor's residual state after the component's last layer (for blocks)."""
    layer = comp["layers"][1] if comp["kind"] == "block" else comp["layer"]
    _, layers = find_decoder_layers(donor_model, layer)
    holder = {}

    def post(m, args, output):
        holder["x"] = output[0] if isinstance(output, tuple) else output
        raise StopForward()

    handle = layers[layer].register_forward_hook(post)
    try:
        donor_model(input_ids=ids, attention_mask=attn, use_cache=False)
    except StopForward:
        pass
    finally:
        handle.remove()
    return holder["x"]


# ------------------------------------------------------------ unembed map


def final_norm_gain(model):
    """The final RMS norm gain of a causal LM, as a float32 vector."""
    inner = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    norm = getattr(inner, "norm", None)
    if norm is None:
        norm = getattr(inner, "final_layernorm", None)
    if norm is None or not hasattr(norm, "weight"):
        d = model.get_output_embeddings().weight.shape[1]
        return torch.ones(d)
    w = norm.weight.detach().float()
    if "Gemma" in type(norm).__name__:
        w = 1.0 + w
    return w


def token_weights(tokenizer, texts, vocab):
    """Unigram weights over the first vocab ids, smoothed, summing to 1. Uniform without texts."""
    if not texts:
        return torch.full((vocab,), 1.0 / vocab)
    counts = torch.ones(vocab)
    for text in texts:
        for i in tokenizer(text)["input_ids"]:
            if 0 <= i < vocab:
                counts[i] += 1.0
    return counts / counts.sum()


def effective_unembedding(model, device, weights):
    """E * diag(final gain), with the weighted mean row removed (logits are shift invariant)."""
    e = model.get_output_embeddings().weight.detach().float().to(device)
    e = e * final_norm_gain(model).to(device).view(1, -1)
    w = weights.to(device).view(-1, 1)
    mu = (e[: w.shape[0]] * w).sum(dim=0, keepdim=True) / w.sum()
    return e - mu


def unembed_map(host, donor, weights=None, lam=1e-4):
    """Least-squares map M (d_host, d_donor) that keeps the donor's logits through the host unembedding.

    M minimizes the weighted error of E_h M - E_d, where E is the unembedding
    times the final norm gain, centered over the vocabulary. A unit-RMS donor
    final state u goes to M u, and the host logits of M u match the donor
    logits of u as closely as rank d_host allows. Returns a dict with M and
    "capture", the share of the donor's centered logit variance that M keeps.
    """
    device = device_of(host)
    v = min(host.get_output_embeddings().weight.shape[0], donor.get_output_embeddings().weight.shape[0])
    if weights is None:
        weights = torch.full((v,), 1.0 / v)
    weights = weights[:v].float()
    e_h = effective_unembedding(host, device, weights)[:v]
    e_d = effective_unembedding(donor, device, weights)[:v]
    w = weights.to(device).view(-1, 1)
    a = (e_h * w).T @ e_h
    b = (e_h * w).T @ e_d
    d_h = a.shape[0]
    a64 = a.double() + lam * (torch.trace(a).double() / d_h) * torch.eye(d_h, dtype=torch.float64, device=device)
    m = torch.linalg.solve(a64, b.double()).float()
    err = (((e_h @ m - e_d) ** 2) * w).sum()
    tot = ((e_d ** 2) * w).sum()
    capture = 1.0 - float(err / tot)
    return {"M": m.cpu(), "capture": capture, "rank": min(d_h, e_d.shape[1]), "vocab_rows": int(v)}


# --------------------------------------------------------------- stages 1-3


def settings_of(budget, options):
    s = dict(DEFAULTS)
    if budget:
        s.update(budget)
    s.update(options)
    return s


def place(base, tokenizer, components, donors=None, settings=None, device=None):
    """Stage 1. Build every part, attach the branches closed, and return (model, branches, donor_map).

    components: raw component dicts. donors: {id: (model, tokenizer)} for
    donors that are already loaded; others load from the hub in bf16.
    """
    s = settings or dict(DEFAULTS)
    donors = dict(donors or {})
    comps = []
    for i, raw in enumerate(components):
        comps.append(normalize_component(raw, i))
    if device is None:
        device = device_of(base)
    n_layers = host_layer_count(base)
    d_host = host_hidden_size(base)
    handles_of = {}
    branches = []
    placements = []
    located = []
    for i, comp in enumerate(comps):
        key = comp["donor"]
        if comp["donor_model"] is not None:
            tok = comp["donor_tokenizer"]
            if tok is None and key in donors:
                tok = donors[key][1]
            if tok is None:
                raise ValueError(f"component {comp['id']}: a loaded donor needs donor_tokenizer, or donors[{key!r}]")
            handles_of[key] = (comp["donor_model"], tok)
        elif key in donors:
            handles_of[key] = donors[key]
        elif key not in handles_of:
            handles_of[key] = load_donor(key, device)
        donor_model, donor_tokenizer = handles_of[key]
        comp["donor_tokenizer"] = donor_tokenizer
        config = donor_model.config
        # The host layer comes from the real range, so a random control sits at the same depth.
        text_config = _text_config(config)
        at = comp["host_layer"]
        if at is None:
            at = default_host_layer(comp["layer"], text_config.num_hidden_layers, n_layers)
        elif at < 0:
            at += n_layers
        if at < 0 or at >= n_layers:
            raise ValueError(f"component {comp['id']}: host layer {comp['host_layer']} is outside {n_layers} layers")
        loc = locate(comp, config)
        if loc["control"] == "scrambled":
            loc["control_mode"] = loc.get("control_mode") or s["control_mode"]
            loc["scramble_max_numel"] = s["scramble_max_numel"]
        scramble_report = {}
        part, tensors = build_part(loc, donor_model, scramble_report)
        d_donor = text_config.hidden_size
        branch = Branch(d_host, d_donor, part, s["t_in_kind"], None, s["gate_bias"])
        branch.to(device)
        branch.synth = {"id": comp["id"], "kind": loc["kind"], "donor": loc["donor"], "layer": loc["layer"],
                        "layers": loc["layers"], "indices": loc["indices"], "host_layer": at,
                        "d_host": d_host, "d_donor": d_donor, "skill": comp["skill"],
                        "control": comp["control"], "control_seed": comp["control_seed"],
                        "control_mode": loc.get("control_mode") if comp["control"] == "scrambled" else None,
                        "scramble": scramble_report or None,
                        "control_overlap": loc.get("control_overlap"),
                        "t_in_kind": s["t_in_kind"], "donor_config": json.loads(config.to_json_string()),
                        "donor_layers": loc["donor_layers"]}
        branch.part_tensors = tensors
        branches.append(branch)
        placements.append((at, branch))
        located.append(loc)
    handles = attach_grafts(base, placements)
    base.__dict__["wsp_synth_handles"] = handles
    for p in base.parameters():
        p.requires_grad_(False)
    return base, branches, located, handles_of


def stage_trainables(model, stage):
    """Return the names of the parameters a stage trains.

    Stage 1: T_in, T_out, and the token gates of every branch. Stage 2 adds
    the LoRA on the host layers after the first seam. Parts and the base are
    never trained.
    """
    names = []
    for name, p in model.named_parameters():
        if name.startswith("wsp_grafts."):
            rest = name.split(".", 2)[2]
            if rest.startswith("part."):
                continue
            names.append(name)
        elif stage == 2 and (name.endswith("lora_A") or name.endswith("lora_B")):
            names.append(name)
    return names


def parameter_groups(model, stage, s):
    """AdamW groups for a stage, and the params, set to require grad. All others are frozen."""
    for p in model.parameters():
        p.requires_grad_(False)
    translation, gate, lora = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("wsp_grafts."):
            rest = name.split(".", 2)[2]
            if rest.startswith("part."):
                continue
            if rest.startswith("token_gate."):
                gate.append(p)
            else:
                translation.append(p)
        elif stage == 2 and (name.endswith("lora_A") or name.endswith("lora_B")):
            lora.append(p)
    for p in translation + gate + lora:
        p.requires_grad_(True)
    suffix = "2" if stage == 2 else ""
    groups = [{"params": translation, "lr": s["lr_translation" + suffix]},
              {"params": gate, "lr": s["lr_gate" + suffix]}]
    if len(lora) > 0:
        groups.append({"params": lora, "lr": s["lr_lora"]})
    return [g for g in groups if len(g["params"]) > 0]


def norm_error(pred, target, valid):
    """Relative squared error over valid positions, plain (for reports)."""
    v = valid.unsqueeze(-1).float()
    return float((((pred - target) ** 2) * v).sum() / ((target ** 2) * v).sum().clamp(min=1e-12))


def match_batch(model, branch, donor_model, comp, ids, attn):
    """Return (pred, target, valid) for the T_in objective on one batch.

    pred: the part's output on the translated host state (with the sink).
    target: the part's output on the donor's native input. Same tokenizer only.
    """
    with torch.no_grad():
        with branches_off(model):
            states, _ = host_capture(model, ids, attn, [branch.host_layer])
        h = states[branch.host_layer]
        native = donor_point(donor_model, comp, ids.to(device_of(donor_model)), attn.to(device_of(donor_model)))
        branch.pad_mask = attn
        branch.host_uses_cache = False
        target = branch.run_part(native.to(device=h.device))
    branch.pad_mask = attn
    branch.host_uses_cache = False
    x = branch.donor_input(h)
    pred = branch.run_part(x)
    valid = attn.clone()
    valid[:, 0] = 0
    return pred, target, valid


def std_mse(pred, target, valid):
    """Mean squared error in units of each dim's target std, over valid positions."""
    v = valid.bool()
    t = target[v]
    std = t.std(dim=0).clamp(min=1e-3)
    return (((pred[v] - t) / std) ** 2).mean()


def optimize_groups(groups, steps, next_loss, warmup_frac=0.05, clip=1.0):
    """AdamW with warmup and cosine decay to zero. next_loss(step) returns the loss tensor (it may run backward itself)."""
    params = []
    for g in groups:
        for p in g["params"]:
            params.append(p)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    base_lrs = [g["lr"] for g in groups]
    warm = max(1, int(warmup_frac * steps))
    losses = []
    for step in range(steps):
        if step < warm:
            factor = (step + 1) / warm
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * factor
        opt.zero_grad()
        value = next_loss(step)
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        losses.append(value)
    return losses


def translate_in(model, tokenizer, branch, comp, donor_model, donor_tokenizer, align, s):
    """Stage 2. Ridge init of T_in, then match the component's native output. Return a report dict."""
    t0 = time.time()
    k = branch.host_layer
    if comp["kind"] in ("block", "attn_heads"):
        _, dl = find_decoder_layers(donor_model, comp["layer"])
        capture = donor_capture_for(donor_model, comp["layer"], None)
        norm_prefix = None
    else:
        norm_prefix = find_layer_modules(donor_model.config, comp["layer"])["pre_norm_prefix"]
        capture = donor_capture_for(donor_model, comp["layer"], norm_prefix)
    stats = {}
    with branches_off(model):
        x, y = collect_pairs(model, tokenizer, k, donor_model, donor_tokenizer, capture, align,
                             max_pairs=s["max_rows"], stats=stats)
    w, r2, r2_dim = ridge_fit(x, y)
    with torch.no_grad():
        branch.set_w_in(w.to(branch.w_out.weight.device))
    branch.set_sink(stats["sink"].to(branch.w_out.weight.device))
    out = {"ridge_r2": r2, "ridge_r2_dim": r2_dim, "ridge_rows": int(x.shape[0]), "sink_norm": float(stats["sink"].norm()),
           "trained": False, "rel_err_ridge": None, "rel_err": None, "donor_lm_penalty": None,
           "donor_lm_penalty_ridge": None}
    if not same_vocab(tokenizer, donor_tokenizer):
        out["note"] = "tokenizers differ: ridge fit only, no output matching"
        out["seconds"] = round(time.time() - t0, 2)
        return out
    pad = pad_id_of(tokenizer)
    enc, _ = encode_items(tokenizer, align, s["seq_len"], truncate=True)
    n_eval = max(1, len(enc) // 10)
    train_enc, eval_enc = enc[:-n_eval] or enc, enc[-n_eval:]
    dev = device_of(model)
    sampler = text_sampler(train_enc, s["batch_in"], s["seed"])

    def error_on(batch):
        ids, attn, _ = collate(batch, pad, dev)
        with torch.no_grad():
            pred, target, valid = match_batch(model, branch, donor_model, comp, ids, attn)
        return norm_error(pred, target, valid)

    out["rel_err_ridge"] = error_on(eval_enc)
    if s["steps_in"] > 0:
        def next_loss(step):
            ids, attn, _ = collate(sampler(), pad, dev)
            pred, target, valid = match_batch(model, branch, donor_model, comp, ids, attn)
            loss = std_mse(pred, target, valid)
            loss.backward()
            return float(loss.detach())

        for p in branch.w_in.parameters():
            p.requires_grad_(True)
        groups = [{"params": list(branch.w_in.parameters()), "lr": s["lr_in"]}]
        losses = optimize_groups(groups, s["steps_in"], next_loss, s["warmup_frac"])
        for p in branch.w_in.parameters():
            p.requires_grad_(False)
        out["match_loss_start"] = losses[0]
        out["match_loss_end"] = losses[-1]
        out["trained"] = True
    out["rel_err"] = error_on(eval_enc)
    if comp["kind"] == "block":
        plain = [t for t in align if isinstance(t, str)]
        if len(plain) > 0:
            with branches_off(model), scrambled_donor_layers(donor_model, branch, comp.get("control") == "scrambled"):
                if s["steps_lm"] > 0:
                    res = train_stitch(model, tokenizer, k, donor_model, donor_tokenizer, comp["layer"], branch.w_in,
                                       plain, objective="donor_lm", steps=s["steps_lm"], lr=s["lr_in"],
                                       batch_size=s["batch_in"], seq_len=s["seq_len"], seed=s["seed"],
                                       sink=branch.sink.detach().cpu())
                    out["donor_lm_penalty_ridge"] = res["before"]["penalty"]
                    out["donor_lm_penalty"] = res["after"]["penalty"]
                else:
                    res = stitch_penalty(model, tokenizer, k, donor_model, comp["layer"], branch.w_in, plain,
                                         max_len=s["seq_len"], sink=branch.sink.detach().cpu())
                    out["donor_lm_penalty"] = res["penalty"]
            branch.w_in.to(branch.w_out.weight.device)
            for p in branch.w_in.parameters():
                p.requires_grad_(False)
        out["rel_err_final"] = error_on(eval_enc)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def part_rows(model, branch, ids, attn, states=None):
    """Run the branch's translation and part on the host states at its layer. Return (y, valid).

    y has shape (batch, seq, d_donor), float32; valid marks the real tokens
    after the first position of each sequence.
    """
    if states is None:
        with torch.no_grad():
            with branches_off(model):
                states, _ = host_capture(model, ids, attn, [branch.host_layer])
    h = states[branch.host_layer]
    branch.pad_mask = attn
    branch.host_uses_cache = False
    with torch.no_grad():
        y = branch.run_part(branch.donor_input(h))
    valid = attn.clone()
    valid[:, 0] = 0
    return y, valid


def translate_out(model, tokenizer, branch, comp, donor_model, donor_tokenizer, align, s):
    """Stage 3a. Initialize T_out. Return a report dict.

    auto, same tokenizer, component exits near the donor top: the logit
    preserving least-squares map. Otherwise ridge to the residual the host is
    MISSING at its top (never to the host's own state). output_init "old"
    keeps the zero init of wsp.graft.
    """
    t0 = time.time()
    mode = s["output_init"]
    if mode == "old":
        nn.init.zeros_(branch.w_out.weight)
        return {"init": "zero", "seconds": 0.0}
    dev = device_of(model)
    pad = pad_id_of(tokenizer)
    enc, _ = encode_items(tokenizer, align, s["seq_len"], truncate=True)
    same = same_vocab(tokenizer, donor_tokenizer)
    n_layers = host_layer_count(model)
    real_last = comp.get("real_last")
    if real_last is None and comp["kind"] == "block":
        real_last = comp["layers"][1]
    near_top = (comp["kind"] == "block" and real_last >= comp["donor_layers"] - s["near_top"])
    out = {}
    if same and near_top:
        v = min(model.get_output_embeddings().weight.shape[0], donor_model.get_output_embeddings().weight.shape[0])
        weights = token_weights(tokenizer, [t for t in align if isinstance(t, str)], v)
        res = unembed_map(model, donor_model, weights, s["unembed_lam"])
        r_h, r_d = [], []
        for first in range(0, min(len(enc), 32), 8):
            ids, attn, _ = collate(enc[first:first + 8], pad, dev)
            with torch.no_grad():
                with branches_off(model):
                    states, _ = host_capture(model, ids, attn, [n_layers - 1])
                dstate = donor_exit_state(donor_model, comp, ids.to(device_of(donor_model)), attn.to(device_of(donor_model)))
            valid = attn.clone()
            valid[:, 0] = 0
            r_h.append(states[n_layers - 1].float()[valid.bool()].pow(2).mean(-1).sqrt())
            r_d.append(dstate.float().to(dev)[valid.bool()].pow(2).mean(-1).sqrt())
        ratio = float(torch.cat(r_h).mean() / torch.cat(r_d).mean())
        with torch.no_grad():
            branch.w_out.weight.copy_((ratio * res["M"]).to(branch.w_out.weight))
        out = {"init": "unembed_map", "logit_capture": res["capture"], "rms_ratio": ratio, "rank": res["rank"]}
    elif not same:
        nn.init.zeros_(branch.w_out.weight)
        return {"init": "zero_no_teacher", "seconds": round(time.time() - t0, 2),
                "note": "tokenizers differ: no donor logits to fit the missing residual, T_out stays zero"}
    else:
        out = missing_residual_fit(model, tokenizer, branch, comp, donor_model, donor_tokenizer, enc, same, s)
        out["init"] = "missing_residual_donor_logits"
    out.update(calibrate_out(model, branch, enc, s, pad) if out["init"] != "unembed_map" else {})
    out["seconds"] = round(time.time() - t0, 2)
    return out


def missing_residual_fit(model, tokenizer, branch, comp, donor_model, donor_tokenizer, enc, same, s):
    """Ridge from the part output to the residual the host lacks at its top. See translate_out."""
    dev = device_of(model)
    pad = pad_id_of(tokenizer)
    n_layers = host_layer_count(model)
    k = branch.host_layer
    v = model.get_output_embeddings().weight.shape[0]
    weights = token_weights(tokenizer, [], v)
    e = effective_unembedding(model, dev, weights)
    w = weights.to(dev).view(-1, 1)
    a = (e * w).T @ e
    d_h = a.shape[0]
    a64 = a.double() + s["unembed_lam"] * (torch.trace(a).double() / d_h) * torch.eye(d_h, dtype=torch.float64, device=dev)
    proj = torch.linalg.solve(a64, (e * w).T.double()).float()  # (d_h, V)
    ys, ts = [], []
    total = 0
    for first in range(0, len(enc), 8):
        if total >= s["max_rows"]:
            break
        ids, attn, _ = collate(enc[first:first + 8], pad, dev)
        with torch.no_grad():
            with branches_off(model):
                states, logits = host_capture(model, ids, attn, sorted({k, n_layers - 1}))
            y, valid = part_rows(model, branch, ids, attn, states)
            logits = logits.float()
            if same:
                dl = forward_logits(donor_model, ids.to(device_of(donor_model)), attn.to(device_of(donor_model))).float().to(dev)
                vv = min(dl.shape[-1], logits.shape[-1])
                gap = (dl[..., :vv] - dl[..., :vv].mean(-1, keepdim=True)) - (logits[..., :vv] - logits[..., :vv].mean(-1, keepdim=True))
                if vv < v:
                    gap = F.pad(gap, (0, v - vv))
            else:
                p = torch.softmax(logits, dim=-1)
                gold = F.one_hot(ids.roll(-1, dims=1), v).float()
                gap = gold - p
            final = states[n_layers - 1].float()
            r = final.pow(2).mean(-1, keepdim=True).sqrt()
            target = r * (gap @ proj.T)
            keep = valid.bool()
            keep[:, -1] = False
            keep &= attn.roll(-1, dims=1).bool()
            ys.append(y[keep].cpu())
            ts.append(target[keep].cpu())
            total += int(keep.sum())
    y = torch.cat(ys)[: s["max_rows"]]
    t = torch.cat(ts)[: s["max_rows"]]
    if float(y.pow(2).sum()) == 0.0 or y.shape[0] < 20:
        return {"fit_r2": None, "fit_rows": int(y.shape[0]), "fit": "degenerate: the part output is zero or too few rows"}
    wmat, r2, r2_dim = ridge_fit(y, t, lam=s["ridge_out_lam"])
    with torch.no_grad():
        branch.w_out.weight.copy_(wmat.to(branch.w_out.weight))
    return {"fit_r2": r2, "fit_rows": int(y.shape[0])}


def calibrate_out(model, branch, enc, s, pad):
    """Scale T_out so the branch's delta has RMS out_alpha times the host's own update at that depth."""
    dev = device_of(model)
    k = branch.host_layer
    ids, attn, _ = collate(enc[:8], pad, dev)
    with torch.no_grad():
        with branches_off(model):
            layers = sorted({max(k - 1, 0), k})
            states, _ = host_capture(model, ids, attn, layers)
        y, valid = part_rows(model, branch, ids, attn, states)
        delta = branch.w_out(y)
        keep = valid.bool()
        d_rms = float(delta[keep].pow(2).mean(-1).sqrt().mean())
        if k > 0:
            own = (states[k] - states[k - 1]).float()
        else:
            own = states[k].float()
        rho = float(own[keep].pow(2).mean(-1).sqrt().mean())
        if d_rms > 0:
            factor = s["out_alpha"] * rho / d_rms
            branch.w_out.weight.mul_(factor)
        else:
            factor = 1.0
    return {"out_rms_before": d_rms, "host_update_rms": rho, "out_scale": factor}


# ------------------------------------------------------------------- heal


def preservation_kl(model, ids, attn, lmask):
    """KL(frozen base || model) on a general batch. Zero when the model equals the base."""
    with torch.no_grad():
        with frozen_base(model):
            ref = forward_logits(model, ids, attn).float()
    cur = forward_logits(model, ids, attn).float()
    return kl_loss(ref, cur, lmask)


def snapshot(model):
    """Copy the trainable state of the branches and the LoRA, plus ramps and closed flags."""
    state = {"params": {}, "lora": lora_state_dict(model), "branches": []}
    for name, p in model.named_parameters():
        if name.startswith("wsp_grafts.") and not name.split(".", 2)[2].startswith("part."):
            state["params"][name] = p.detach().clone()
    for b in model.wsp_grafts:
        state["branches"].append((b.ramp, b.closed))
    return state


def restore(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state["params"].items():
            params[name].copy_(value)
    for b, (ramp, closed) in zip(model.wsp_grafts, state["branches"]):
        b.ramp = ramp
        b.closed = closed


def strip_lora(model):
    """Remove the host LoRA (its delta is zeroed first, so the network is the one before the LoRA)."""
    mods = lora_modules(model)
    if len(mods) == 0:
        return
    with torch.no_grad():
        for _, m in mods:
            m.lora_B.zero_()
    merge_lora(model)


def lora_delta_ratio(model):
    """Frobenius norm of all LoRA deltas over the Frobenius norm of the weights they sit on."""
    num = 0.0
    den = 0.0
    with torch.no_grad():
        for _, m in lora_modules(model):
            delta = m.scale * (m.lora_B.float() @ m.lora_A.float())
            num += float((delta ** 2).sum())
            den += float((m.base.weight.float() ** 2).sum())
    if den == 0:
        return 0.0
    return math.sqrt(num / den)


def run_heal(model, tokenizer, comps, branches, donor_map, data, stage, s, steps):
    """One heal stage. Returns a report dict with the per-step losses.

    Per step and per component: donor KL on its domain batch when the
    tokenizer matches (function matching), plus lm_weight_kd times the LM
    loss; plain LM loss otherwise. Stage 1 adds feature matching at the seam.
    Both stages add pres_weight times KL(frozen base || model) on a general
    batch. Loss is on answer tokens only where the item has a prompt.
    """
    t0 = time.time()
    dev = device_of(model)
    pad = pad_id_of(tokenizer)
    groups = parameter_groups(model, stage, s)
    old_mode = "old" if s["output_init"] == "old" else "new"
    samplers = []
    for i in range(len(comps)):
        samplers.append(text_sampler(data["components"][i], s["batch_size"], s["seed"] + 17 * i + stage))
    general = text_sampler(data["general"], s["batch_size"], s["seed"] + 1000 + stage)
    ramp_steps = max(1, int(s["ramp_frac"] * steps)) if stage == 1 else 1
    n = len(comps) if s["components_per_step"] is None else min(len(comps), s["components_per_step"])
    n = max(n, 1)
    log = {"task": [], "pres": [], "feat": []}

    def next_loss(step):
        ramp = min(1.0, (step + 1) / ramp_steps)
        for b in branches:
            if not b.closed:
                b.ramp = ramp
        task_total, feat_total = 0.0, 0.0
        if s["components_per_step"] is None or s["components_per_step"] >= len(comps):
            active = list(range(len(comps)))
        else:
            k = s["components_per_step"]
            active = [(step * k + j) % len(comps) for j in range(k)]
        for i in active:
            comp = comps[i]
            ids, attn, lmask = collate(samplers[i](), pad, dev)
            donor_model, donor_tokenizer = donor_map[comp["donor"]]
            use_kd = (s["kd"] and old_mode == "new" and same_vocab(tokenizer, donor_tokenizer))
            feat = None
            for b in branches:
                b.record = use_kd and stage == 1 and s["feat_weight"] > 0 and b.synth["id"] == comp["id"]
            logits = forward_logits(model, ids, attn).float()
            if use_kd:
                with torch.no_grad():
                    d_ids = ids.to(device_of(donor_model))
                    d_attn = attn.to(device_of(donor_model))
                    teacher = forward_logits(donor_model, d_ids, d_attn).float().to(dev)
                task = s["kd_weight"] * kl_loss(teacher, logits, lmask) + s["lm_weight_kd"] * lm_loss(logits, ids, lmask)
                if stage == 1 and s["feat_weight"] > 0:
                    branch = [b for b in branches if b.synth["id"] == comp["id"]][0]
                    if branch.last_part_out is not None and not branch.closed:
                        with torch.no_grad():
                            native = donor_point(donor_model, comp, d_ids, d_attn)
                            tgt = branch.run_part(native.to(part_dtype(branch.part)).to(dev))
                        valid = attn.clone()
                        valid[:, 0] = 0
                        feat = std_mse(branch.last_part_out, tgt, valid)
            else:
                task = lm_loss(logits, ids, lmask)
            term = task
            if feat is not None:
                term = term + s["feat_weight"] * feat
                feat_total += float(feat.detach()) / n
            (term / n).backward()
            task_total += float(task.detach()) / n
            for b in branches:
                b.record = False
                b.last_part_out = None
        g_ids, g_attn, g_mask = collate(general(), pad, dev)
        pres = preservation_kl(model, g_ids, g_attn, g_mask)
        if s["pres_weight"] > 0:
            (s["pres_weight"] * pres).backward()
        log["task"].append(task_total)
        log["pres"].append(float(pres.detach()))
        log["feat"].append(feat_total)
        return task_total

    model.eval()
    optimize_groups(groups, steps, next_loss, s["warmup_frac"])
    for b in branches:
        if not b.closed:
            b.ramp = 1.0
    for p in model.parameters():
        p.requires_grad_(False)
    out = {"steps": steps, "seconds": round(time.time() - t0, 2),
           "task_start": log["task"][0] if steps else None, "task_end": log["task"][-1] if steps else None,
           "pres_start": log["pres"][0] if steps else None, "pres_end": log["pres"][-1] if steps else None,
           "feat_start": log["feat"][0] if steps else None, "feat_end": log["feat"][-1] if steps else None,
           "losses": [round(v, 5) for v in log["task"]]}
    return out


# ------------------------------------------------------------ verification


def score_practice(model, tokenizer, practice, skills=None, limit=None):
    """Score each practice skill. A value is a callable (model, tokenizer) -> float, or {"items", "limit"}."""
    out = {}
    names = skills if skills is not None else list(practice)
    model.eval()
    for skill in names:
        spec = practice[skill]
        with torch.no_grad():
            if callable(spec):
                out[skill] = float(spec(model, tokenizer))
            else:
                cap = spec.get("limit")
                if limit is not None:
                    cap = limit if cap is None else min(cap, limit)
                res = evaluate_skill(model, tokenizer, skill, items=spec["items"], limit=cap,
                                     batch_size=spec.get("batch_size", 32))
                out[skill] = float(res["score"])
    return out


def guard_failures(scores, base_scores, guard, tol):
    bad = {}
    for skill in guard:
        if scores[skill] < base_scores[skill] - tol:
            bad[skill] = round(scores[skill] - base_scores[skill], 5)
    return bad


def text_loss(model, tokenizer, enc, s):
    """Mean per-example LM loss over answer tokens, on encoded items."""
    if len(enc) == 0:
        return None
    dev = device_of(model)
    pad = pad_id_of(tokenizer)
    total, count = 0.0, 0
    with torch.no_grad():
        for first in range(0, len(enc), s["batch_size"]):
            batch = enc[first:first + s["batch_size"]]
            ids, attn, lmask = collate(batch, pad, dev)
            total += float(lm_loss(forward_logits(model, ids, attn), ids, lmask)) * len(batch)
            count += len(batch)
    return total / count


def gate_report(model, tokenizer, comps, data, s, batches=4):
    """Mean token-gate value of every branch on each text set, after the heal."""
    dev = device_of(model)
    pad = pad_id_of(tokenizer)
    sets = {"general": data["general"]}
    for i, comp in enumerate(comps):
        sets["domain:" + comp["id"]] = data["components"][i]
    out = {}
    for name, enc in sets.items():
        for b in model.wsp_grafts:
            b.token_gate.reset_stats()
        with torch.no_grad():
            for first in range(0, min(len(enc), batches * s["batch_size"]), s["batch_size"]):
                ids, attn, _ = collate(enc[first:first + s["batch_size"]], pad, dev)
                forward_logits(model, ids, attn)
        out[name] = {b.synth["id"]: (round(b.token_gate.mean(), 5) if b.token_gate.mean() is not None else None)
                     for b in model.wsp_grafts}
    for b in model.wsp_grafts:
        b.token_gate.reset_stats()
    return out


def param_counts(model):
    total, run = model_params(model)
    parts, glue, gates = 0, 0, 0
    for b in getattr(model, "wsp_grafts", []):
        for name, p in b.named_parameters():
            if name.startswith("part."):
                parts += p.numel()
            elif name.startswith("token_gate."):
                gates += p.numel()
            else:
                glue += p.numel()
    lora = 0
    for _, m in lora_modules(model):
        lora += m.lora_A.numel() + m.lora_B.numel()
    return {"total": total, "per_token": run, "parts": parts, "glue": glue, "gates": gates, "lora": lora}


def check_budget(model, s, planned_lora=0):
    counts = param_counts(model)
    total = counts["total"] + planned_lora
    run = counts["per_token"] + planned_lora
    problems = []
    if s["max_params"] is not None and total > s["max_params"]:
        problems.append(f"params total {total} is over the limit {s['max_params']}")
    if s["max_params_per_token"] is not None and run > s["max_params_per_token"]:
        problems.append(f"params per token {run} is over the limit {s['max_params_per_token']}")
    if problems:
        raise BudgetError("; ".join(problems) + f" (parts {counts['parts']}, glue {counts['glue']}, "
                          f"gates {counts['gates']}, planned LoRA {planned_lora})")
    counts["planned_lora"] = planned_lora
    return counts


# ---------------------------------------------------------------- compile


def prepare_data(tokenizer, comps, texts, s):
    """Encode the heal texts. Returns (data, report). Domain texts hold out the last 10% for the component test."""
    if isinstance(texts, (list, tuple)):
        texts = {"align": list(texts), "general": list(texts)}
    align = texts.get("align") or texts.get("general")
    general_raw = texts.get("general") or align
    if not align:
        raise ValueError("texts needs an 'align' or 'general' list")
    skill_texts = texts.get("skills", {})
    data = {"components": [], "heldout": [], "align": align}
    dropped = {}
    for comp in comps:
        raw = comp["texts"]
        if raw is None:
            raw = skill_texts.get(comp["skill"])
        if not raw:
            raw = general_raw
        enc, d = encode_items(tokenizer, raw, s["seq_len"])
        dropped[comp["id"]] = d
        if len(enc) == 0:
            raise ValueError(f"component {comp['id']}: no domain text fits seq_len {s['seq_len']} ({d} dropped)")
        n_hold = max(1, len(enc) // 10) if len(enc) > 1 else 0
        data["components"].append(enc[:len(enc) - n_hold] if n_hold else enc)
        data["heldout"].append(enc[len(enc) - n_hold:] if n_hold else enc)
    gen, d = encode_items(tokenizer, general_raw, s["seq_len"])
    dropped["general"] = d
    if len(gen) == 0:
        raise ValueError(f"no general text fits seq_len {s['seq_len']} ({d} dropped)")
    data["general"] = gen
    return data, {"dropped": dropped}


def synthesize(base, components, texts, budget=None, practice=None, guard=None, donors=None, out_dir=None,
               tokenizer=None, device=None, **options):
    """Compile components into one network. Returns (model, tokenizer, report).

    base: a model id, or a loaded (model, tokenizer) pair (modified in place).
    components: raw component dicts (see normalize_component), or library
        catalog rows.
    texts: {"align": [...], "general": [...], "skills": {skill: [...]}},
        or one list. Items are strings or {"prompt", "answer"} pairs.
    budget: heal and limit settings (see DEFAULTS): max_params,
        max_params_per_token, stage1_steps, stage2_steps, and so on.
    practice: {skill: callable(model, tokenizer) -> score, or
        {"items": [...]}}. Never receives gradient. Guards and verdicts use it.
    guard: skills that may not fall more than tolerance below the base
        (default: every practice skill).
    donors: {id: (model, tokenizer)} for donors already loaded.
    out_dir: save the network here (read it with load_synth).
    options: any DEFAULTS key, for example control_run=True.
    """
    s = settings_of(budget, options)
    t_start = time.time()
    if isinstance(base, str):
        import transformers

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = transformers.AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(base)
        base_id = base
    else:
        model, tokenizer = base
        base_id = getattr(model.config, "_name_or_path", None) or None
    model.eval()
    practice = practice or {}
    guard = list(practice) if guard is None else list(guard)
    report = {"base": base_id, "settings": {k: v for k, v in s.items()}, "events": []}

    control_names = control_list(s)
    base_copies = {name: copy.deepcopy(model) for name in control_names}
    comps_in = [normalize_component(c, i) for i, c in enumerate(components)]
    model, branches, comps, donor_map = place(model, tokenizer, components, donors, s)
    for c_in, c in zip(comps_in, comps):
        c["texts"] = c_in["texts"]
        c["skill"] = c_in["skill"]
    # stage 7 (budget) runs before any heal, with the LoRA that stage 2 will add.
    n_layers = host_layer_count(model)
    first_seam = min(b.host_layer for b in branches)
    lora_layers = list(range(first_seam + 1, n_layers))
    planned = lora_param_count(model.config, s["lora_rank"], layers=lora_layers) if lora_layers and s["stage2_steps"] > 0 else 0
    report["params"] = check_budget(model, s, planned)
    report["planned_lora"] = planned
    report["place"] = [{"id": c["id"], "kind": c["kind"], "donor": c["donor"], "host_layer": b.host_layer,
                        "layers": c["layers"], "indices": c["indices"], "control": c["control"],
                        "scramble": b.synth.get("scramble"),
                        "part_params": sum(p.numel() for n, p in b.named_parameters() if n.startswith("part."))}
                       for c, b in zip(comps, branches)]
    data, drop_report = prepare_data(tokenizer, comps, texts, s)
    report["data"] = drop_report

    # stage 2: translate in
    report["translate_in"] = {}
    report["translate_out"] = {}
    for c, b in zip(comps, branches):
        dm, dt = donor_map[c["donor"]]
        report["translate_in"][c["id"]] = translate_in(model, tokenizer, b, c, dm, dt, data["align"], s)
    # stage 3: translate out init
    for c, b in zip(comps, branches):
        dm, dt = donor_map[c["donor"]]
        report["translate_out"][c["id"]] = translate_out(model, tokenizer, b, c, dm, dt, data["align"], s)

    # stage 6 base scores (a fresh network is the base)
    scores = {}
    if practice:
        scores["base"] = score_practice(model, tokenizer, practice)
    state0 = snapshot(model)
    guard_events = []
    final_stage = "stage0"

    # stage 3b: stage-1 heal
    report["stage1"] = run_heal(model, tokenizer, comps, branches, donor_map, data, 1, s, s["stage1_steps"])
    state1 = snapshot(model)
    stage1_ok = True
    if practice:
        scores["stage1"] = score_practice(model, tokenizer, practice)
        bad = guard_failures(scores["stage1"], scores["base"], guard, s["tolerance"])
        if bad:
            stage1_ok = False
            guard_events.append({"stage": 1, "failed": bad, "action": "stage 1 fails the guard"})
    final_stage = "stage1"

    # stage 4: stage-2 heal with LoRA after the first seam
    ran_stage2 = False
    if s["stage2_steps"] > 0 and stage1_ok:
        if lora_layers:
            apply_lora(model, s["lora_rank"], layers=lora_layers)
        state1 = snapshot(model)
        report["stage2"] = run_heal(model, tokenizer, comps, branches, donor_map, data, 2, s, s["stage2_steps"])
        report["stage2"]["lora_delta_ratio"] = lora_delta_ratio(model)
        report["stage2"]["lora_layers"] = lora_layers
        ran_stage2 = True
        final_stage = "stage2"
        if practice:
            scores["stage2"] = score_practice(model, tokenizer, practice)
            bad = guard_failures(scores["stage2"], scores["base"], guard, s["tolerance"])
            if bad:
                restore(model, state1)
                strip_lora(model)
                final_stage = "stage1"
                guard_events.append({"stage": 2, "failed": bad, "action": "rolled back to stage 1"})
                scores["after_rollback"] = score_practice(model, tokenizer, practice)
    elif s["stage2_steps"] > 0:
        report["stage2"] = {"skipped": "stage 1 failed the guard"}

    # a failing stage 1 (or a failing rollback target): close the offending gates, else go back to the base
    if practice:
        current = scores.get("after_rollback") or scores.get(final_stage)
        bad = guard_failures(current, scores["base"], guard, s["tolerance"])
        if bad:
            if ran_stage2 and final_stage == "stage2":
                pass
            strip_lora(model)
            closed = []
            while bad and len(closed) < len(branches):
                best = None
                for b in branches:
                    if b.closed:
                        continue
                    b.forced_off = True
                    trial = score_practice(model, tokenizer, practice, guard)
                    b.forced_off = False
                    margin = min(trial[g] - scores["base"][g] for g in guard)
                    if best is None or margin > best[0]:
                        best = (margin, b)
                best[1].closed = True
                best[1].ramp = 0.0
                closed.append(best[1].synth["id"])
                current = score_practice(model, tokenizer, practice)
                bad = guard_failures(current, scores["base"], guard, s["tolerance"])
            guard_events.append({"stage": final_stage, "failed": bad, "action": "closed gates",
                                 "closed": closed})
            scores["after_closing"] = current
            final_stage = "stage1 with closed gates" if bad == {} else "base"
    report["guard"] = {"events": guard_events, "final_stage": final_stage, "tolerance": s["tolerance"],
                       "guarded": guard}
    for e in guard_events:
        report["events"].append(e)

    # stage 5 report: gates
    report["gates"] = gate_report(model, tokenizer, comps, data, s)
    report["gates"]["ramp"] = {b.synth["id"]: b.ramp for b in model.wsp_grafts}
    report["gates"]["closed"] = {b.synth["id"]: b.closed for b in model.wsp_grafts}

    # stage 6: component test and random control
    if practice:
        final = score_practice(model, tokenizer, practice)
        scores["final"] = final
    report["scores"] = scores
    if s["component_tests"]:
        report["components"] = component_tests(model, tokenizer, comps, branches, data, practice, scores, s)
    else:
        report["components"] = [{"id": c["id"], "kind": c["kind"], "skill": c["skill"], "host_layer": b.host_layer,
                                 "closed": b.closed, "verdict": "not tested"} for c, b in zip(comps, branches)]

    if control_names:
        report["controls"] = {}
        for name in control_names:
            report["controls"][name] = build_control(name, base_copies.pop(name), model, tokenizer, components, texts,
                                                     budget, practice, guard, donors, donor_map, s, options, report)
        if "random" in report["controls"]:
            report["random_control"] = report["controls"]["random"]
        for c in report["components"]:
            by_control = {}
            for name in control_names:
                by_control[name] = report["controls"][name]["gain"].get(c["id"])
            c["content_gain_by_control"] = by_control
            first = "random" if "random" in by_control else control_names[0]
            c["content_gain"] = by_control[first]
        for c in report["components"]:
            if c["verdict"] != "not tested":
                c["verdict"] = verdict(c, s)

    report["params"] = param_counts(model)
    report["heal_seconds"] = round(report["stage1"]["seconds"] + report.get("stage2", {}).get("seconds", 0.0), 2)
    report["seconds"] = round(time.time() - t_start, 2)
    for k in ("stage1", "stage2"):
        if k in report and "losses" in report[k]:
            report[k]["losses"] = report[k]["losses"][::max(1, len(report[k]["losses"]) // 50)]
    model.wsp_synth_report = report
    model.wsp_synth_donors = None
    if out_dir:
        save_synth(model, tokenizer, out_dir, base_id, report)
        report["saved_to"] = out_dir
    return model, tokenizer, report


def lora_baseline(model, tokenizer, comps, donor_map, data, s, steps):
    """Base plus LoRA on every layer, trained like stage 2 without branches: the competitor arm.

    comps need "id", "donor", and the encoded data in data["components"]. s["kd"]
    picks the donor KL (distillation) or the LM loss only. The preservation
    term and the schedule are the same as in the synthesis heal.
    """
    apply_lora(model, s["lora_rank"], layers=list(range(host_layer_count(model))))
    out = run_heal(model, tokenizer, comps, [], donor_map, data, 2, s, steps)
    out["lora_delta_ratio"] = lora_delta_ratio(model)
    return out


def component_tests(model, tokenizer, comps, branches, data, practice, scores, s):
    """Per component: score with its gate forced to 0, and LM loss on its held-out domain text with and without it."""
    out = []
    final = scores.get("final")
    if final is not None and s["test_limit"] is not None:
        # Score the full network on the same items, so the difference is paired.
        final = score_practice(model, tokenizer, practice, limit=s["test_limit"])
    for i, (c, b) in enumerate(zip(comps, branches)):
        row = {"id": c["id"], "kind": c["kind"], "skill": c["skill"], "host_layer": b.host_layer,
               "closed": b.closed, "contribution": None, "text_gain": None, "content_gain": None,
               "scores_without": None}
        loss_on = text_loss(model, tokenizer, data["heldout"][i], s)
        b.forced_off = True
        loss_off = text_loss(model, tokenizer, data["heldout"][i], s)
        without = None
        if practice:
            without = score_practice(model, tokenizer, practice, limit=s["test_limit"])
        b.forced_off = False
        row["heldout_loss_on"] = loss_on
        row["heldout_loss_off"] = loss_off
        if loss_on is not None and loss_off is not None:
            row["text_gain"] = loss_off - loss_on
        if without is not None:
            row["scores_without"] = without
            if c["skill"] in without:
                row["contribution"] = final[c["skill"]] - without[c["skill"]]
        row["verdict"] = verdict(row, s)
        out.append(row)
    return out


def control_list(s):
    """The control kinds verify builds: s["controls"], or ["random"] when control_run is set."""
    names = s.get("controls")
    if names is None:
        names = ["random"] if s["control_run"] else []
    out = []
    for name in names:
        if name not in ("random", "scrambled"):
            raise ValueError(f"controls must name 'random' or 'scrambled', got {name!r}")
        if name not in out:
            out.append(name)
    return out


def verdict(row, s):
    """contributes, harmful, or one of the "no better" verdicts.

    Random stands in for "nothing specific to this part" (another layer range
    of the same donor). Scrambled has the same shapes and weight statistics
    but no learned structure, so it tests whether the donor's knowledge is
    used at all. With a scrambled control built, "contributes" needs the real
    component to beat it by more than margin:
      beats scrambled, not random   donor helps, choice does not matter
      beats random, not scrambled   no better than scrambled
      beats neither                 no better than random
    Without a scrambled control the verdict is the old one.
    """
    contribution = row.get("contribution")
    if contribution is None:
        contribution = row.get("text_gain")
    if row.get("closed"):
        return "harmful"
    if contribution is None:
        return "no better than random"
    if contribution < -s["margin"]:
        return "harmful"
    if contribution <= s["margin"]:
        return "no better than random"
    by_control = row.get("content_gain_by_control")
    if by_control is None:
        gain = row.get("content_gain")
        by_control = {"random": gain} if gain is not None else {}
    random_gain = by_control.get("random")
    scrambled_gain = by_control.get("scrambled")
    beats_random = random_gain is None or random_gain > s["margin"]
    if scrambled_gain is None:
        return "contributes" if beats_random else "no better than random"
    beats_scrambled = scrambled_gain > s["margin"]
    if not beats_scrambled:
        if random_gain is not None and beats_random:
            return "no better than scrambled"
        if random_gain is None:
            return "no better than scrambled"
        return "no better than random"
    if not beats_random:
        return "donor helps, choice does not matter"
    return "contributes"


def random_control(base_copy, model, tokenizer, components, texts, budget, practice, guard, donors, donor_map, s, options, report):
    """Compile the same recipe with random same-size parts and report real minus random per component."""
    return build_control("random", base_copy, model, tokenizer, components, texts, budget, practice, guard, donors,
                         donor_map, s, options, report)


def build_control(kind, base_copy, model, tokenizer, components, texts, budget, practice, guard, donors, donor_map, s,
                  options, report):
    """Compile the same recipe with every component replaced by a control. Report real minus control per component.

    kind "random": another range of the same donor (wsp.synth.locate). kind
    "scrambled": the same donor layers or units with weights rewritten by
    wsp.scramble. A component keeps its own control_mode. Others get s["control_mode"].
    """
    rand = []
    for c in components:
        c = dict(c)
        c["control"] = kind
        rand.append(c)
    opts = dict(options)
    opts["control_run"] = False
    opts["controls"] = None
    fresh = base_copy
    known = dict(donors or {})
    for key, pair in donor_map.items():
        known.setdefault(key, pair)
    rmodel, rtok, rreport = synthesize((fresh, tokenizer), rand, texts, budget, practice, guard, known, **opts)
    gain = {}
    rscores = rreport["scores"].get("final")
    real_scores = report["scores"].get("final")
    for i, c in enumerate(report["components"]):
        row = rreport["components"][i]
        skill = c["skill"]
        if rscores is not None and real_scores is not None and skill in rscores:
            gain[c["id"]] = real_scores[skill] - rscores[skill]
        elif c.get("heldout_loss_on") is not None and row.get("heldout_loss_on") is not None:
            gain[c["id"]] = row["heldout_loss_on"] - c["heldout_loss_on"]
    out = {"kind": kind, "scores": rscores, "gain": gain, "gates": rreport["gates"], "guard": rreport["guard"],
           "scramble": {r["id"]: r.get("scramble") for r in rreport["place"]} if kind == "scrambled" else None,
           "heldout_loss_on": [r.get("heldout_loss_on") for r in rreport["components"]]}
    del rmodel
    return out


# --------------------------------------------------------------- save/load


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def save_synth(model, tokenizer, out_dir, base_id, report=None):
    """Write the network to out_dir. The base is saved there too unless base_id is a hub id or local path."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    branches = list(model.wsp_grafts)
    lora = lora_modules(model)
    lora_info = None
    if lora:
        first = lora[0][1]
        layers = sorted({int(n.split(".layers.")[1].split(".")[0]) for n, _ in lora if ".layers." in n})
        lora_info = {"rank": first.rank, "alpha": first.alpha, "layers": layers}
    saved_base = None
    if base_id is None or not (os.path.exists(str(base_id)) or "/" in str(base_id)):
        saved_base = os.path.join(out_dir, "base")
        plain = fresh_copy_for_save(model)
        plain.save_pretrained(saved_base)
    tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))
    tensors = {}
    glue = {"branches": [], "lora": lora_state_dict(model)}
    metas = []
    for i, b in enumerate(branches):
        for name, t in b.part_tensors.items():
            tensors[f"{i}/{name}"] = t.detach().contiguous().cpu().clone()
        state = {k: v.detach().cpu() for k, v in b.state_dict().items() if not k.startswith("part.")}
        glue["branches"].append(state)
        meta = dict(b.synth)
        meta["ramp"] = b.ramp
        meta["closed"] = b.closed
        meta["has_sink"] = b.sink is not None
        metas.append(meta)
    save_file(tensors, os.path.join(out_dir, "parts.safetensors"))
    torch.save(glue, os.path.join(out_dir, "glue.pt"))
    manifest = {"format": FORMAT, "base_id": base_id if saved_base is None else None,
                "base_dir": "base" if saved_base else None, "branches": metas, "lora": lora_info,
                "report": json_safe(report) if report is not None else None}
    with open(os.path.join(out_dir, "synth.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def fresh_copy_for_save(model):
    """The host weights without LoRA wrappers or branches, for save_pretrained (the LoRA is saved apart)."""
    copy_model = copy.deepcopy(model)
    strip_lora_inplace(copy_model)
    if hasattr(copy_model, "wsp_grafts"):
        del copy_model.wsp_grafts
    copy_model.__dict__.pop("wsp_synth_handles", None)
    return copy_model


def strip_lora_inplace(model):
    """Replace each LoRALinear with its base linear, dropping the delta (the base weight is untouched)."""
    for name, module in lora_modules(model):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.base)


def load_synth(path, device="cpu", dtype=torch.float32, base=None):
    """Read a network written by synthesize. Returns (model, tokenizer, report).

    base: optional (model, tokenizer) to use instead of loading the saved or named base.
    """
    import transformers
    from safetensors.torch import load_file

    with open(os.path.join(path, "synth.json")) as f:
        manifest = json.load(f)
    if base is not None:
        model, tokenizer = base
    else:
        source = os.path.join(path, manifest["base_dir"]) if manifest["base_dir"] else manifest["base_id"]
        model = transformers.AutoModelForCausalLM.from_pretrained(source, dtype=dtype).to(device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    model.eval()
    tensors = load_file(os.path.join(path, "parts.safetensors"))
    glue = torch.load(os.path.join(path, "glue.pt"))
    placements = []
    branches = []
    for i, meta in enumerate(manifest["branches"]):
        cfg = dict(meta["donor_config"])
        config = transformers.AutoConfig.for_model(cfg.pop("model_type"), **cfg)
        prefix = f"{i}/"
        part_tensors = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
        part = rebuild_part(meta, config, part_tensors)
        branch = Branch(meta["d_host"], meta["d_donor"], part, meta["t_in_kind"], None, 0.0)
        state = dict(glue["branches"][i])
        if meta["has_sink"]:
            branch.set_sink(state["sink"])
        missing, unexpected = branch.load_state_dict(state, strict=False)
        bad = [k for k in missing if not k.startswith("part.")]
        if bad or unexpected:
            raise ValueError(f"branch {i}: state mismatch, missing {bad[:3]}, unexpected {list(unexpected)[:3]}")
        branch.to(device)
        branch.synth = {k: v for k, v in meta.items() if k not in ("ramp", "closed", "has_sink")}
        branch.part_tensors = part_tensors
        branch.ramp = meta["ramp"]
        branch.closed = meta["closed"]
        branches.append(branch)
        placements.append((meta["host_layer"], branch))
    handles = attach_grafts(model, placements)
    model.__dict__["wsp_synth_handles"] = handles
    if manifest["lora"] is not None:
        apply_lora(model, manifest["lora"]["rank"], alpha=manifest["lora"]["alpha"], layers=manifest["lora"]["layers"])
        load_lora_state(model, glue["lora"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, tokenizer, manifest["report"]


def read_catalog_rows(path, model=None, kind=None, skill=None):
    """Read rows of a library catalog.jsonl, filtered by donor model id, type, or skill name."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if model is not None and row.get("model") != model:
                continue
            if kind is not None and row.get("type") != kind:
                continue
            if skill is not None and (row.get("skill") or {}).get("name") != skill:
                continue
            rows.append(row)
    return rows
