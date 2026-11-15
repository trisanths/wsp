"""Insert donor decoder layers into a base model: one network, one forward pass.

The new network runs base layers 0..at-1, then the inserted section, then
base layers at..end. The inserted section is a residual branch:

    x   = stitch_in(h)                    base width -> donor width
    out = h + gate * stitch_out(block(x) - x)

block runs the donor's own decoder layers first..last (wsp.parts.BlockPart:
the donor's layer classes and rotary, any family). stitch_out maps donor
width back to base width. gate is one trained scalar that starts at 0, so
a fresh network equals the base exactly. The heal opens the gate.

When the base and donor widths match, stitch_in and stitch_out start as
identity maps. Then a gate of 1 gives exactly "base layers 0..at-1, donor
layers first..last, base layers at..end" as one plain stack. Otherwise
stitch_in is a linear, lowrank, or mlp stitch from wsp.stitch (ridge_stitch
fits it on matched states), and stitch_out starts as the pseudo-inverse of
the stitch's linear map (or a ridge fit back to the base).

While the gate is exactly 0, only the gate gets a gradient: the stitches
start to learn once the gate moves. So pre-train stitch_in first with
wsp.stitch.train_stitch on the plain base, then pass it to insert_layers.

An insertion is a wsp.graft.Graft whose w_out is the gated stitch_out, so
it reuses the graft machinery: the donor sink at each sequence's first
real token, part LoRA, and the part KV cache that follows the base cache
through generate (cached output equals uncached output). Donor layers
with a conv state (LFM2 as a donor) cannot use that cache, so a network
with such an insertion runs with use_cache=False (an LFM2 base is fine).

Trainable: stitches, gates, an optional part LoRA on the donor layers,
and an optional LoRA on the base layers ("the base learns to read").

save_inserted writes a directory with a manifest (insert.json), the
trained tensors, and the tokenizer. load_inserted rebuilds the network
and fetches only the donor tensors each insertion needs.
"""

import json
import os

import torch
from torch import nn

from wsp.graft import Graft, LowRankLinear, attach_grafts, disable_cache
from wsp.parts import _text_config, block_layer_prefixes, block_tensor_names, build_block_part, find_decoder_layers

FORMAT = "wsp-insert-1"
STITCH_KINDS = ("identity", "linear", "lowrank", "mlp")
# LoRA targets for the base: the wsp.lora defaults plus the LFM2 names
# (in_proj and out_proj in conv and attention, w1 to w3 in the MLP).
BASE_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                     "qkv_proj", "gate_up_proj", "in_proj", "out_proj", "w1", "w2", "w3")


class GatedOut(nn.Module):
    """x -> gate * linear(x), with a scalar gate that starts at gate_init."""

    def __init__(self, d_donor, d_base, gate_init=0.0):
        super().__init__()
        self.linear = nn.Linear(d_donor, d_base, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def forward(self, x):
        return self.gate * self.linear(x)


class Insertion(Graft):
    """h + gate * stitch_out(block(stitch_in(h)) - stitch_in(h)).

    w_in is stitch_in and w_out is the GatedOut. meta holds what
    load_inserted needs to build the insertion again.
    """

    def __init__(self, d_base, d_donor, part, stitch_in, meta, gate_init=0.0):
        super().__init__(d_base, d_donor, part, w_in=stitch_in)
        self.w_out = GatedOut(d_donor, d_base, gate_init)
        self.meta = meta
        self.cacheable = part_can_cache(part)

    @property
    def gate(self):
        return self.w_out.gate

    def forward(self, h):
        if self.host_uses_cache and not self.cacheable:
            raise ValueError(
                "this insertion has donor layers with a conv state (LFM2 style), which the part KV "
                "cache cannot hold; run the network with use_cache=False"
            )
        return super().forward(h)


def part_can_cache(part):
    """Return False when the donor block has a conv mixer (an nn.Conv1d), as in LFM2.

    The part cache is a plain KV cache, so such layers run only without a
    cache. attach_insertions then turns the cache off on the base.
    """
    for module in part.modules():
        if isinstance(module, nn.Conv1d):
            return False
    return True


# ---------------------------------------------------------------- stitches


def build_stitch_in(kind, d_base, d_donor, hidden=None, rank=None):
    """Build a float32 stitch_in. A linear or mlp stitch starts as identity when the widths match."""
    from wsp.stitch import MLPStitch

    if kind not in STITCH_KINDS:
        raise ValueError(f"unknown stitch kind {kind!r}; the kinds are {', '.join(STITCH_KINDS)}")
    if kind == "identity":
        if d_base != d_donor:
            raise ValueError(f"an identity stitch needs equal widths, got base {d_base} and donor {d_donor}")
        return nn.Identity()
    if kind == "linear":
        stitch = nn.Linear(d_base, d_donor, bias=False)
    elif kind == "lowrank":
        if rank is None:
            raise ValueError("a lowrank stitch needs a rank")
        stitch = LowRankLinear(d_base, d_donor, int(rank))
    else:
        stitch = MLPStitch(d_base, d_donor, hidden)
    stitch = stitch.float()
    if d_base == d_donor and kind in ("linear", "mlp"):
        set_linear_map(stitch, torch.eye(d_donor))
    return stitch


def stitch_config(stitch):
    """Return the config dict that rebuilds this stitch: kind, plus hidden or rank."""
    from wsp.stitch import MLPStitch

    if isinstance(stitch, nn.Identity):
        return {"kind": "identity"}
    if isinstance(stitch, nn.Linear):
        return {"kind": "linear"}
    if isinstance(stitch, LowRankLinear):
        return {"kind": "lowrank", "rank": int(stitch.down.weight.shape[0])}
    if isinstance(stitch, MLPStitch):
        return {"kind": "mlp", "hidden": int(stitch.hidden)}
    raise ValueError(f"insertions do not support a {type(stitch).__name__} stitch")


def linear_map(stitch):
    """Return the (d_donor, d_base) linear part of a stitch, or None for identity."""
    from wsp.stitch import MLPStitch

    if isinstance(stitch, nn.Identity):
        return None
    if isinstance(stitch, nn.Linear):
        return stitch.weight.detach().float()
    if isinstance(stitch, LowRankLinear):
        return (stitch.up.weight.float() @ stitch.down.weight.float()).detach()
    if isinstance(stitch, MLPStitch):
        return stitch.w0.weight.detach().float()
    raise ValueError(f"insertions do not support a {type(stitch).__name__} stitch")


@torch.no_grad()
def set_linear_map(stitch, w):
    """Set the linear part of a stitch to w, shape (d_donor, d_base)."""
    if isinstance(stitch, nn.Linear):
        stitch.weight.copy_(w.to(stitch.weight.dtype))
    else:
        stitch.set_from_full(w)


@torch.no_grad()
def init_stitch_out(insertion, w=None):
    """Start stitch_out from w (d_base, d_donor), else identity or the pseudo-inverse of stitch_in."""
    weight = insertion.w_out.linear.weight
    if w is None:
        d_base, d_donor = weight.shape
        forward = linear_map(insertion.w_in)
        if forward is None:
            w = torch.eye(d_base, d_donor)
        elif d_base == d_donor and torch.allclose(forward, torch.eye(d_donor, device=forward.device)):
            w = torch.eye(d_base, d_donor)
        else:
            w = torch.linalg.pinv(forward.double()).float()
    weight.copy_(w.to(device=weight.device, dtype=weight.dtype))


def ridge_stitch(base_model, base_tokenizer, at, donor_model, donor_tokenizer, first, texts, kind="linear",
                 hidden=None, rank=None):
    """Fit stitch_in and the stitch_out matrix by ridge on offset-matched states.

    The base side is the output of base layer at - 1, the donor side the
    input of donor layer first. Returns (stitch_in, w_out, report). w_out is
    the (d_base, d_donor) ridge map back, for insert_layers(stitch_out=...).
    """
    from wsp.align import ridge_fit
    from wsp.stitch import collect_rows, donor_capture_for

    check_at(base_model, at)
    d_base = _text_config(base_model.config).hidden_size
    d_donor = _text_config(donor_model.config).hidden_size
    stitch = build_stitch_in(kind, d_base, d_donor, hidden, rank)
    capture = donor_capture_for(donor_model, first, None)
    x, y = collect_rows(base_model, base_tokenizer, [at - 1], donor_model, donor_tokenizer, capture, texts)
    w_in, r2_in, r2_dim_in = ridge_fit(x, y)
    if not isinstance(stitch, nn.Identity):
        set_linear_map(stitch, w_in)
    w_out, r2_out, r2_dim_out = ridge_fit(y, x)
    report = {"pairs": int(x.shape[0]), "r2_in": r2_in, "r2_dim_in": r2_dim_in, "r2_out": r2_out,
              "r2_dim_out": r2_dim_out}
    return stitch, w_out, report


# ---------------------------------------------------------------- donor tensors


def read_local_tensors(folder, names):
    """Read the named tensors from a local safetensors model folder (one file or sharded)."""
    from wsp.fetch import local_range_reader, read_tensors

    index_path = os.path.join(folder, "model.safetensors.index.json")
    shards = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for name in names:
            if name not in weight_map:
                raise KeyError(f"tensor not found in index: {name}")
            shard = weight_map[name]
            if shard not in shards:
                shards[shard] = []
            shards[shard].append(name)
    else:
        shards["model.safetensors"] = list(names)
    out = {}
    for shard in shards:
        found = read_tensors(local_range_reader(os.path.join(folder, shard)), shards[shard])
        for name in found:
            out[name] = found[name]
    return out


def donor_block_tensors(donor, config, first, last, cache_dir=None):
    """Return the tensors of donor layers first..last.

    donor is a loaded model (its state_dict is used), a dict of tensors
    (returned as is), a local model folder, or a Hugging Face repo id
    (only these tensors are fetched, by HTTP range reads).
    """
    if isinstance(donor, nn.Module):
        prefixes = block_layer_prefixes(config, first, last)
        out = {}
        for name, value in donor.state_dict().items():
            for prefix in prefixes:
                if name.startswith(prefix + "."):
                    out[name] = value.detach()
                    break
        return out
    if isinstance(donor, dict):
        return donor
    names = block_tensor_names(config, first, last)
    if os.path.isdir(donor):
        return read_local_tensors(donor, names)
    from wsp.fetch import fetch_tensors

    return fetch_tensors(donor, names, cache_dir=cache_dir)


def donor_config_of(donor, donor_config=None):
    """Return the donor config: the one given, the loaded model's, or one read by id or folder."""
    if donor_config is not None:
        return donor_config
    if isinstance(donor, nn.Module):
        return donor.config
    if isinstance(donor, str):
        import transformers

        return transformers.AutoConfig.from_pretrained(donor)
    raise ValueError("donor tensors need a donor_config")


# ---------------------------------------------------------------- building


def layer_count(model):
    _, layers = find_decoder_layers(model, 0)
    return len(layers)


def check_at(base_model, at):
    count = layer_count(base_model)
    if at < 1 or at > count:
        raise ValueError(f"at must be in [1, {count}] (the insertion goes after base layer at - 1), got {at}")


def insertions_of(model):
    """Return the insertions attached to model, in order (empty when there are none)."""
    grafts = getattr(model, "wsp_grafts", None)
    if grafts is None:
        return []
    out = []
    for graft in grafts:
        if not isinstance(graft, Insertion):
            raise ValueError("this model holds grafts that are not insertions; insert_layers cannot mix them")
        out.append(graft)
    return out


def attach_insertions(model, insertions):
    """Hook every insertion onto the base, replacing hooks made by an earlier call."""
    old = getattr(model, "_wsp_insert_handles", None)
    if old is not None:
        for handle in old:
            handle.remove()
    placements = []
    cacheable = True
    for insertion in insertions:
        placements.append((insertion.meta["at"] - 1, insertion))
        if not insertion.cacheable:
            cacheable = False
    model._wsp_insert_handles = attach_grafts(model, placements)
    if not cacheable:
        # Donor conv layers cannot use the part KV cache, so generate and
        # plain calls run without a cache by default.
        disable_cache(model)
    return model._wsp_insert_handles


def build_insertion(base_model, donor, donor_layers, at, stitch_kind="linear", donor_config=None, donor_id=None,
                    stitch_in=None, stitch_out=None, stitch_hidden=None, stitch_rank=None, part_lora_rank=None,
                    part_lora_alpha=None, sink=None, gate_init=0.0, cache_dir=None):
    """Build one Insertion without attaching it. See insert_layers for the arguments."""
    from wsp.lora import apply_part_lora

    check_at(base_model, at)
    first = int(donor_layers[0])
    last = int(donor_layers[1])
    config = donor_config_of(donor, donor_config)
    if donor_id is None:
        if isinstance(donor, str):
            donor_id = donor
        else:
            donor_id = getattr(config, "_name_or_path", None) or None
    tensors = donor_block_tensors(donor, config, first, last, cache_dir)
    part = build_block_part(config, first, last, tensors)
    if part_lora_rank is not None:
        apply_part_lora(part, part_lora_rank, part_lora_alpha)

    d_base = _text_config(base_model.config).hidden_size
    d_donor = _text_config(config).hidden_size
    if stitch_in is None:
        stitch_in = build_stitch_in(stitch_kind, d_base, d_donor, stitch_hidden, stitch_rank)
    meta = {"donor_id": donor_id, "donor_layers": [first, last], "at": int(at), "d_base": int(d_base),
            "d_donor": int(d_donor), "stitch": stitch_config(stitch_in), "part_lora_rank": part_lora_rank,
            "part_lora_alpha": part_lora_alpha, "sink": sink is not None and sink is not False}
    insertion = Insertion(d_base, d_donor, part, stitch_in, meta, gate_init)
    # The config is a plain attribute, not a module; save_inserted writes it.
    insertion.donor_config = config
    init_stitch_out(insertion, stitch_out)
    if meta["sink"]:
        insertion.set_sink(sink)
    device = base_model.get_input_embeddings().weight.device
    insertion.to(device)
    return insertion


def insert_layers(base_model, donor, donor_layers, at, stitch_kind="linear", **options):
    """Insert donor layers donor_layers = [first, last] (inclusive) after base layer at - 1.

    donor: a loaded donor model, a dict of its tensors (with
    donor_config=...), a local model folder, or a Hugging Face repo id.
    stitch_kind: identity, linear, lowrank (stitch_rank=r) or mlp
    (stitch_hidden=n). Options:
      donor_config    the donor config (needed for a dict of tensors)
      donor_id        the repo id or folder that load_inserted fetches from
                      (default: the donor string, or config._name_or_path)
      stitch_in       a ready stitch module, for example a trained one
      stitch_out      a (d_base, d_donor) start matrix for stitch_out
      part_lora_rank  a LoRA of this rank on the donor layers
      sink            a (d_donor,) donor state fed at each first real token
      gate_init       the gate's start value (default 0: equals the base)
      cache_dir       the tensor cache for fetched donor tensors
    Call it again to add insertions at other depths. Returns the new
    Insertion. The base gets the insertions in base_model.wsp_grafts.
    """
    insertion = build_insertion(base_model, donor, donor_layers, at, stitch_kind, **options)
    insertions = insertions_of(base_model)
    insertions.append(insertion)
    attach_insertions(base_model, insertions)
    return insertion


def add_base_lora(model, rank, alpha=None, layers=None, targets=BASE_LORA_TARGETS):
    """Put a LoRA on the base decoder layers. Return its parameters.

    The names in targets that a base lacks are skipped. The config goes in
    model._wsp_base_lora for save_inserted.
    """
    from wsp.lora import DEFAULT_TARGETS, apply_lora, default_alpha, target_linears

    found = []
    for _, name, _ in target_linears(model, targets, layers):
        if name not in found:
            found.append(name)
    if len(found) == 0:
        found = list(DEFAULT_TARGETS)
    if alpha is None:
        alpha = default_alpha(rank)
    params = apply_lora(model, rank, alpha, tuple(found), layers)
    model._wsp_base_lora = {"rank": int(rank), "alpha": alpha, "targets": found, "layers": layers}
    return params


# ---------------------------------------------------------------- training


def insertion_parameters(model):
    """Return a dict of trainable parameter lists: stitch, gate, part_lora, base_lora."""
    from wsp.lora import LoRALinear, lora_modules

    out = {"stitch": [], "gate": [], "part_lora": [], "base_lora": []}
    for insertion in insertions_of(model):
        for param in insertion.w_in.parameters():
            out["stitch"].append(param)
        for param in insertion.w_out.linear.parameters():
            out["stitch"].append(param)
        out["gate"].append(insertion.w_out.gate)
        for module in insertion.part.modules():
            if isinstance(module, LoRALinear):
                out["part_lora"].append(module.lora_A)
                out["part_lora"].append(module.lora_B)
    for _, module in lora_modules(model):
        out["base_lora"].append(module.lora_A)
        out["base_lora"].append(module.lora_B)
    return out


def heal_groups(model, lrs):
    """Freeze everything, then return AdamW groups for the parameters named in lrs.

    lrs maps stitch, gate, part_lora, and base_lora to a learning rate. A
    name left out, or with lr None, stays frozen.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    found = insertion_parameters(model)
    groups = []
    for name in ["stitch", "gate", "part_lora", "base_lora"]:
        lr = lrs.get(name)
        if lr is None or len(found[name]) == 0:
            continue
        for param in found[name]:
            param.requires_grad_(True)
        groups.append({"params": found[name], "lr": lr, "name": name})
    if len(groups) == 0:
        raise ValueError("nothing to train: give a learning rate for stitch, gate, part_lora, or base_lora")
    return groups


def heal(model, tokenizer, texts, lrs, steps=1500, batch_size=8, seq_len=512, seed=0, log_every=50,
         add_special_tokens=True):
    """Train the parameters named in lrs on texts with the causal LM loss. Return per-step losses.

    The batches are drawn like wsp.heal.heal (seeded shuffled passes). Each
    group has its own learning rate, so the gate can move faster than the
    stitches. Pass add_special_tokens=False for chat-template texts, which
    already hold the BOS.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    groups = heal_groups(model, lrs)
    params = []
    for group in groups:
        for param in group["params"]:
            params.append(param)
    device = model.get_input_embeddings().weight.device
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
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
        enc = tokenizer(batch, truncation=True, max_length=seq_len, padding=True, return_tensors="pt",
                        add_special_tokens=add_special_tokens)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        loss = float(out.loss.detach())
        losses.append(loss)
        if log_every and (step + 1) % log_every == 0:
            print(f"insert heal step {step + 1}/{steps} loss {loss:.4f} gates {gate_values(model)}")
    model.eval()
    return losses


def gate_values(model):
    values = []
    for insertion in insertions_of(model):
        values.append(round(float(insertion.w_out.gate.detach()), 4))
    return values


# ---------------------------------------------------------------- counts


def param_counts(model):
    """Count parameters: total, per_token, and by kind (base, donor, stitch, gate, part_lora, base_lora).

    Every inserted layer runs on every token, so per_token is the total
    minus the input embedding table (a lookup), unless the output head
    shares it, as in wsp.skills_eval.model_params. trainable counts the
    parameters with requires_grad set.
    """
    counts = {"base": 0, "donor": 0, "stitch": 0, "gate": 0, "part_lora": 0, "base_lora": 0}
    seen = set()
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        if name.startswith("wsp_grafts."):
            if ".part." in name:
                if is_lora:
                    counts["part_lora"] += n
                else:
                    counts["donor"] += n
            elif name.endswith("w_out.gate"):
                counts["gate"] += n
            else:
                counts["stitch"] += n
        elif is_lora:
            counts["base_lora"] += n
        else:
            counts["base"] += n
    per_token = total
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed is not None:
        tied = head is not None and head.weight is embed.weight
        if not tied:
            per_token -= embed.weight.numel()
    counts["total"] = total
    counts["per_token"] = per_token
    counts["trainable"] = trainable
    return counts


# ---------------------------------------------------------------- save and load


def trained_state(model):
    """Return the insertion tensors to save: everything except the frozen donor weights."""
    state = {}
    grafts = getattr(model, "wsp_grafts", None)
    if grafts is None:
        # A LoRA-only model (no insertions) has no insertion tensors.
        return state
    for name, value in grafts.state_dict().items():
        if ".part." in name and not (name.endswith("lora_A") or name.endswith("lora_B")):
            continue
        state[name] = value.detach().cpu().contiguous()
    return state


def save_inserted(model, tokenizer, out_dir, base_id, extra=None):
    """Save an inserted network to out_dir. Return the manifest.

    base_id is the base's repo id or local folder. Donor weights are not
    saved: each donor config goes in donors/<i>/, and load_inserted fetches
    the donor tensors again by donor_id.
    """
    from safetensors.torch import save_file

    from wsp.lora import lora_state_dict

    os.makedirs(out_dir, exist_ok=True)
    insertions = insertions_of(model)
    metas = []
    for index, insertion in enumerate(insertions):
        meta = dict(insertion.meta)
        if meta["donor_id"] is None:
            raise ValueError(f"insertion {index} has no donor_id, so load_inserted could not fetch its donor")
        folder = os.path.join(out_dir, "donors", str(index))
        os.makedirs(folder, exist_ok=True)
        insertion.donor_config.save_pretrained(folder)
        metas.append(meta)
    save_file(trained_state(model), os.path.join(out_dir, "insert_weights.safetensors"))
    base_lora = getattr(model, "_wsp_base_lora", None)
    if base_lora is not None:
        save_file(lora_state_dict(model), os.path.join(out_dir, "lora_weights.safetensors"))
    manifest = {"format": FORMAT, "base": base_id, "insertions": metas, "base_lora": base_lora,
                "params": param_counts(model), "gates": gate_values(model)}
    if extra is not None:
        manifest["extra"] = extra
    with open(os.path.join(out_dir, "insert.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return manifest


def load_inserted(path, device="cpu", dtype=torch.bfloat16, cache_dir=None, base_model=None, donors=None):
    """Load a directory written by save_inserted. Return (model, tokenizer).

    base_model: a loaded base to use in place of the manifest's base id.
    donors: an optional dict of donor_id -> loaded model or tensors, used in
    place of a fetch. The network generates with a KV cache; the donor
    layers keep their own cache in step with the base cache.
    """
    import transformers
    from safetensors.torch import load_file

    from wsp.lora import apply_lora, load_lora_state

    with open(os.path.join(path, "insert.json")) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} directory")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    if base_model is None:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(manifest["base"], dtype=dtype)
    model = base_model
    lora = manifest.get("base_lora")
    if lora is not None:
        apply_lora(model, lora["rank"], lora["alpha"], tuple(lora["targets"]), lora["layers"])
        load_lora_state(model, load_file(os.path.join(path, "lora_weights.safetensors")))
        model._wsp_base_lora = lora

    insertions = []
    for index, meta in enumerate(manifest["insertions"]):
        config = transformers.AutoConfig.from_pretrained(os.path.join(path, "donors", str(index)))
        source = meta["donor_id"]
        if donors is not None and source in donors:
            source = donors[source]
        stitch = meta["stitch"]
        stitch_in = build_stitch_in(stitch["kind"], meta["d_base"], meta["d_donor"], stitch.get("hidden"),
                                    stitch.get("rank"))
        sink = None
        if meta.get("sink"):
            # A placeholder; the saved state fills in the real sink.
            sink = torch.zeros(meta["d_donor"])
        insertion = build_insertion(model, source, meta["donor_layers"], meta["at"], donor_config=config,
                                    donor_id=meta["donor_id"], stitch_in=stitch_in,
                                    part_lora_rank=meta.get("part_lora_rank"),
                                    part_lora_alpha=meta.get("part_lora_alpha"), sink=sink, cache_dir=cache_dir)
        insertions.append(insertion)
    attach_insertions(model, insertions)

    state = load_file(os.path.join(path, "insert_weights.safetensors"))
    result = model.wsp_grafts.load_state_dict(state, strict=False)
    if len(result.unexpected_keys) > 0:
        raise KeyError(f"saved insertion tensors the model does not have: {result.unexpected_keys[:5]}")
    for name in result.missing_keys:
        frozen_donor = ".part." in name and not (name.endswith("lora_A") or name.endswith("lora_B"))
        if not frozen_donor:
            raise KeyError(f"insertion tensor {name} is missing from the saved weights")
    model = model.to(device).eval()
    return model, tokenizer
