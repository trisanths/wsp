"""Same-family composition: a searched per-layer mix of weight differences.

A specialist that shares a lineage with the host has the same shapes and the
same residual basis, so its per-tensor difference from a reference is a
component that needs no translation layer. This module builds those
components, mixes them with a small genome, searches the genome with an
evolution strategy, and writes the result as a plain Hugging Face model of
exactly the host's size.

Terms:
  host       the model that gets written (also the model that runs).
  member     a specialist of the same lineage: tensors of its own weights and
             a reference. The reference is the member's true parent (the
             common ancestor, or the host when the member was trained from the
             host). One reference can be shared by many members.
  component  the difference D = W_member - W_ref of one tensor, in float32.
             Tensors fall in groups: (layer, attention), (layer, MLP),
             embeddings, norms.
  genome     one weight per (member, layer bucket, attention or MLP), two
             global weights per member (embeddings, norms), and one density per
             member (keep the top fraction of each difference by magnitude and
             rescale to keep the norm).
  apply      W = W_host + sum over members of weight * trimmed D, written in
             place layer by layer in float32, cast once to the model dtype.
  restore    copies the host values back, bit for bit.

Members with a different positional base (rope_theta or rope scaling) mix
only in MLP groups. Differences are computed on the fly from the members'
stored tensors (bf16 on the CPU) and the shared references, so the stored
size is 2 bytes per parameter per member plus one copy per distinct reference.
bf16 values subtract exactly in float32.

Reference recipes for comparison: uniform average, task arithmetic at a few
scales, and a corrected TIES (fp32, per-member references, mixable groups
only, disjoint mean).

The search is a diagonal CMA-ES written with numpy. Controls: the same genome
on differences shuffled inside each tensor (same norms, no knowledge).
"""

import hashlib
import json
import math
import os
import re
import time
import zlib

import numpy as np
import torch

from wsp import probe

KINDS = ("attn", "mlp")
GLOBALS = ("embed", "norm")
SKIP_WORDS = ("vision", "multi_modal", "projector", "audio")


# ---------------------------------------------------------------- names


def canonical(name):
    """Return a name that is the same for the model and for checkpoint files, or None to ignore the tensor.

    Layer tensors keep the text from "layers.N." on. Other tensors lose the
    leading "model." and "language_model." parts. Vision and projector
    tensors return None.
    """
    for word in SKIP_WORDS:
        if word in name:
            return None
    match = re.search(r"(?:^|\.)(layers\.\d+\..*)$", name)
    if match is not None:
        return match.group(1)
    changed = True
    while changed:
        changed = False
        for prefix in ("model.", "language_model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def classify(name):
    """Return (kind, layer) for a parameter name, or None when it is in no group.

    kind is "attn", "mlp", "embed", or "norm". layer is the layer index for
    attn and mlp, else None. Norm tensors inside layers belong to the global
    norm group.
    """
    text = canonical(name)
    if text is None:
        return None
    if "norm" in text:
        return ("norm", None)
    match = re.match(r"layers\.(\d+)\.(.*)$", text)
    if match is not None:
        layer = int(match.group(1))
        rest = match.group(2)
        if "mlp" in rest or "feed_forward" in rest or "experts" in rest:
            return ("mlp", layer)
        return ("attn", layer)
    if "embed" in text or text.startswith("lm_head"):
        return ("embed", None)
    return None


def bucket_of(layer, n_layers, n_buckets):
    """Return the layer bucket of a layer index (equal slices over depth)."""
    return min(layer * n_buckets // max(1, n_layers), n_buckets - 1)


def positional_signature(config):
    """Return the settings that fix the positional base of a config, as a JSON string."""
    if hasattr(config, "get_text_config"):
        try:
            config = config.get_text_config()
        except Exception:
            pass
    sig = {}
    for key in ("rope_theta", "rope_scaling", "rope_parameters", "rope_local_base_freq"):
        value = getattr(config, key, None)
        if value is not None:
            sig[key] = value
    return json.dumps(sig, sort_keys=True, default=str)


def positional_differs(config_a, config_b):
    """True when the two configs use a different positional base."""
    return positional_signature(config_a) != positional_signature(config_b)


# ---------------------------------------------------------------- members


class Member:
    """One specialist: tensors by host parameter name, and where its reference lives.

    tensors: {param name: CPU tensor in storage dtype}.
    ref: {param name: CPU tensor} shared with other members, or None when the
    reference is the host itself.
    """

    def __init__(self, name, skill, tensors, ref=None, ref_name="host", mlp_only=False, meta=None):
        self.name = name
        self.skill = skill
        self.tensors = tensors
        self.ref = ref
        self.ref_name = ref_name
        self.mlp_only = mlp_only
        self.meta = dict(meta or {})

    def covers(self, name, kind):
        """True when this member may write the tensor in a group of this kind."""
        if name not in self.tensors:
            return False
        if self.mlp_only and kind != "mlp":
            return False
        return True


def host_index(model):
    """Return {canonical name: (param name, shape)} for the model's classified parameters."""
    out = {}
    for name, param in model.named_parameters():
        if classify(name) is None:
            continue
        out[canonical(name)] = (name, tuple(param.shape))
    return out


def map_state(state, index, groups=("attn", "mlp"), embed_policy="skip"):
    """Return {param name: tensor} for the entries of state that match the host index.

    state: any mapping name -> tensor (or an iterator of pairs). Only groups in
    groups are kept. A shape that differs from the host is dropped, except
    that with embed_policy "slice" an embedding table with fewer or more rows
    is cut to the common rows, padded with the host rows by the caller (the
    mixed difference is zero outside the common rows).
    """
    out = {}
    items = state.items() if hasattr(state, "items") else state
    for name, tensor in items:
        key = canonical(name)
        if key is None or key not in index:
            continue
        host_name, shape = index[key]
        info = classify(host_name)
        if info[0] not in groups:
            continue
        if tuple(tensor.shape) != shape:
            if info[0] == "embed" and embed_policy == "slice" and tensor.dim() == 2 and tensor.shape[1] == shape[1]:
                rows = min(tensor.shape[0], shape[0])
                padded = torch.zeros(shape, dtype=tensor.dtype)
                padded[:rows] = tensor[:rows]
                out[host_name] = ("slice", rows, padded)
            continue
        out[host_name] = tensor
    return out


def read_checkpoint(directory, wanted=None):
    """Yield (name, tensor) from every .safetensors file in a directory.

    wanted: an optional function name -> bool; only tensors it accepts load.
    """
    from safetensors import safe_open

    files = sorted(f for f in os.listdir(directory) if f.endswith(".safetensors"))
    for file in files:
        with safe_open(os.path.join(directory, file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if wanted is not None and not wanted(name):
                    continue
                yield name, handle.get_tensor(name)


def member_from_state(name, skill, state, model, ref_state=None, ref=None, ref_name="host", mlp_only=False,
                      groups=("attn", "mlp"), meta=None):
    """Build a Member from a mapping of tensors (a state dict or a checkpoint iterator).

    ref_state: a mapping of the reference's tensors, or None to use the host as
    the reference. ref: an already mapped reference (shared between members).
    Tensors the reference lacks are dropped. Returns the Member.
    """
    index = host_index(model)
    tensors = map_state(state, index, groups)
    clean = {}
    for key, value in tensors.items():
        if isinstance(value, tuple):
            continue
        clean[key] = value.detach().cpu()
    mapped_ref = ref
    if mapped_ref is None and ref_state is not None:
        mapped_ref = map_ref(ref_state, index, groups)
    if mapped_ref is not None:
        for key in list(clean.keys()):
            if key not in mapped_ref:
                del clean[key]
    return Member(name, skill, clean, mapped_ref, ref_name if mapped_ref is not None else "host", mlp_only, meta)


def map_ref(state, index, groups=("attn", "mlp")):
    """Return a reference mapping {param name: CPU tensor} to share between members."""
    out = {}
    for key, value in map_state(state, index, groups).items():
        if isinstance(value, tuple):
            continue
        out[key] = value.detach().cpu()
    return out


def member_from_directory(name, skill, directory, model, ref=None, ref_name="host", mlp_only=False,
                          groups=("attn", "mlp"), meta=None):
    """Build a Member from a checkpoint directory, reading only the tensors the host groups use."""
    index = host_index(model)

    def wanted(tensor_name):
        key = canonical(tensor_name)
        if key is None or key not in index:
            return False
        return classify(index[key][0])[0] in groups

    tensors = map_state(read_checkpoint(directory, wanted), index, groups)
    clean = {}
    for key, value in tensors.items():
        if isinstance(value, tuple):
            continue
        clean[key] = value
    if ref is not None:
        for key in list(clean.keys()):
            if key not in ref:
                del clean[key]
    return Member(name, skill, clean, ref, ref_name if ref is not None else "host", mlp_only, meta)


def reference_from_directory(directory, model, groups=("attn", "mlp")):
    """Read a reference checkpoint (the common ancestor) into a mapping to share between members."""
    index = host_index(model)

    def wanted(tensor_name):
        key = canonical(tensor_name)
        if key is None or key not in index:
            return False
        return classify(index[key][0])[0] in groups

    return map_ref(read_checkpoint(directory, wanted), index, groups)


def gate_member(member, model, limit=24, max_rel=0.3, min_cos=0.97):
    """Measure how far a member is from its reference on a sample of matrices (biases and norms are skipped).

    Returns {"rel_l2_mean", "rel_l2_max", "cos_mean", "cos_min", "n", "passed"}.
    rel_l2 is the norm of the difference over the norm of the reference. A
    member passes when the mean rel_l2 is at most max_rel and the mean cosine
    is at least min_cos. A heavy continued-pretraining shift fails.
    """
    params = dict(model.named_parameters())
    names = []
    for name in sorted(member.tensors.keys()):
        if member.tensors[name].dim() >= 2:
            names.append(name)
    if len(names) == 0:
        return {"rel_l2_mean": None, "rel_l2_max": None, "cos_mean": None, "cos_min": None, "n": 0, "passed": False}
    step = max(1, len(names) // limit)
    picked = names[::step][:limit]
    rels = []
    coss = []
    for name in picked:
        spec = member.tensors[name].float()
        if member.ref is None:
            ref = params[name].detach().cpu().float()
        else:
            ref = member.ref[name].float()
        diff = spec - ref
        ref_norm = float(ref.norm())
        rels.append(float(diff.norm()) / max(ref_norm, 1e-12))
        denom = float(spec.norm()) * ref_norm
        coss.append(float((spec * ref).sum()) / max(denom, 1e-12))
    rel_mean = sum(rels) / len(rels)
    cos_mean = sum(coss) / len(coss)
    return {"rel_l2_mean": rel_mean, "rel_l2_max": max(rels), "cos_mean": cos_mean, "cos_min": min(coss),
            "n": len(picked), "passed": bool(rel_mean <= max_rel and cos_mean >= min_cos)}


# ---------------------------------------------------------------- genome


class Genome:
    """The mixing weights. weights[s, b, k]: member s, bucket b, kind k (attn, mlp)."""

    def __init__(self, n_members, n_buckets, weights=None, globals_=None, density=None):
        self.n_members = n_members
        self.n_buckets = n_buckets
        self.weights = np.zeros((n_members, n_buckets, 2)) if weights is None else np.array(weights, dtype=float)
        self.globals_ = np.zeros((n_members, 2)) if globals_ is None else np.array(globals_, dtype=float)
        self.density = np.ones(n_members) if density is None else np.array(density, dtype=float)

    @property
    def dim_per_member(self):
        return self.n_buckets * 2 + 3

    def copy(self):
        return Genome(self.n_members, self.n_buckets, self.weights, self.globals_, self.density)

    def to_vector(self):
        parts = []
        for s in range(self.n_members):
            parts.append(self.weights[s].reshape(-1))
            parts.append(self.globals_[s])
            parts.append(self.density[s:s + 1])
        return np.concatenate(parts)

    @classmethod
    def from_vector(cls, vector, n_members, n_buckets):
        vector = np.asarray(vector, dtype=float)
        per = n_buckets * 2 + 3
        genome = cls(n_members, n_buckets)
        for s in range(n_members):
            chunk = vector[s * per:(s + 1) * per]
            genome.weights[s] = chunk[:n_buckets * 2].reshape(n_buckets, 2)
            genome.globals_[s] = chunk[n_buckets * 2:n_buckets * 2 + 2]
            genome.density[s] = chunk[n_buckets * 2 + 2]
        return genome

    def to_dict(self):
        return {"n_members": self.n_members, "n_buckets": self.n_buckets, "weights": self.weights.tolist(),
                "globals": self.globals_.tolist(), "density": self.density.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["n_members"], data["n_buckets"], data["weights"], data["globals"], data["density"])

    def is_zero(self):
        return not (np.any(self.weights != 0) or np.any(self.globals_ != 0))


def zero_genome(n_members, n_buckets):
    return Genome(n_members, n_buckets)


def one_hot_genome(n_members, n_buckets, index, value=1.0, with_globals=False):
    """Weight value on every group of one member, zero elsewhere."""
    genome = Genome(n_members, n_buckets)
    genome.weights[index] = value
    if with_globals:
        genome.globals_[index] = value
    return genome


def uniform_genome(n_members, n_buckets, include_host=True, with_globals=False):
    """Uniform average: each member at 1/(members + 1) when the host counts as one, else 1/members."""
    value = 1.0 / (n_members + (1 if include_host else 0))
    genome = Genome(n_members, n_buckets)
    genome.weights[:] = value
    if with_globals:
        genome.globals_[:] = value
    return genome


def task_arithmetic_genome(n_members, n_buckets, scale, with_globals=False):
    """Task arithmetic: host plus scale times the sum of all differences."""
    genome = Genome(n_members, n_buckets)
    genome.weights[:] = scale
    if with_globals:
        genome.globals_[:] = scale
    return genome


def layer_prior_genome(layer_scores, member_skills, n_buckets, n_layers=None, gain=1.0, include_host=True):
    """A starting genome from per-skill layer scores (for example wsp.atlas.layer_skill_map "scores").

    layer_scores: {skill: list of one score per layer}. member_skills: one
    skill name per member (a skill without scores gets the uniform weight).
    Each member's weight in a bucket is u * (1 + gain * z), where u is the
    uniform weight and z in [-1, 1] says how much more the skill lives in
    that bucket than in the average bucket. Weights stay at or above zero.
    """
    count = len(member_skills)
    genome = uniform_genome(count, n_buckets, include_host)
    u = genome.weights[0, 0, 0]
    for s, skill in enumerate(member_skills):
        if skill not in layer_scores:
            continue
        scores = np.asarray(layer_scores[skill], dtype=float)
        layers = len(scores) if n_layers is None else n_layers
        sums = np.zeros(n_buckets)
        counts = np.zeros(n_buckets)
        for layer in range(layers):
            b = bucket_of(layer, layers, n_buckets)
            sums[b] += scores[layer]
            counts[b] += 1
        means = sums / np.maximum(counts, 1)
        spread = means.max() - means.min()
        if spread <= 0:
            continue
        z = (means - means.mean()) / spread
        z = np.clip(2.0 * z, -1.0, 1.0)
        genome.weights[s, :, :] = np.maximum(u * (1.0 + gain * z), 0.0)[:, None]
    return genome


def genome_bounds(n_members, n_buckets, weight_range=(0.0, 1.0), density_range=(0.5, 1.0), global_range=(0.0, 1.0)):
    """Return (lower, upper) vectors in the genome's vector layout."""
    lower = []
    upper = []
    for s in range(n_members):
        for _ in range(n_buckets * 2):
            lower.append(weight_range[0])
            upper.append(weight_range[1])
        for _ in range(2):
            lower.append(global_range[0])
            upper.append(global_range[1])
        lower.append(density_range[0])
        upper.append(density_range[1])
    return np.array(lower), np.array(upper)


def active_mask(n_members, n_buckets, free_globals=False, search_density=True):
    """Boolean vector: which genome dimensions the search may move."""
    mask = []
    for s in range(n_members):
        for _ in range(n_buckets * 2):
            mask.append(True)
        for _ in range(2):
            mask.append(bool(free_globals))
        mask.append(bool(search_density))
    return np.array(mask)


# ---------------------------------------------------------------- trimming


def rank_cutoff(delta, keep_count):
    """Return (threshold, need) for an exact top-keep_count selection by magnitude.

    threshold is the magnitude of the keep_count-th largest entry. need is how
    many of the entries that equal the threshold must be kept, so that
    (count above threshold) + need == keep_count. Both can be cached.
    """
    flat = delta.abs().reshape(-1)
    count = flat.numel()
    threshold = torch.kthvalue(flat, count - keep_count + 1).values
    above = int((flat > threshold).sum())
    return threshold, keep_count - above


def top_mask(delta, keep_count, cutoff=None):
    """Boolean mask with exactly keep_count True entries: the largest magnitudes.

    Entries tied at the threshold are taken in flat index order until the
    count is reached, so the choice is deterministic.
    """
    if cutoff is None:
        cutoff = rank_cutoff(delta, keep_count)
    threshold, need = cutoff
    magnitude = delta.abs()
    threshold = threshold.to(magnitude.device)
    mask = magnitude > threshold
    if need > 0:
        tied = magnitude == threshold
        tie_count = int(tied.sum())
        if tie_count <= need:
            mask |= tied
        else:
            order = torch.cumsum(tied.reshape(-1).to(torch.int32), dim=0).reshape(tied.shape)
            mask |= tied & (order <= need)
    return mask


def trim_top(delta, density, rescale=True, cutoff=None):
    """Keep exactly the top density fraction of a difference by magnitude.

    density 1.0 returns the difference unchanged. The kept count is
    round(density * numel); ties at the threshold are cut to that count.
    With rescale, the kept entries are multiplied by |delta|_2 / |kept|_2, so
    the trimmed difference has the norm of the original. A 1 over density
    factor is unbiased only for random dropping; for a top-by-magnitude cut it
    overshoots (a Gaussian at density 0.5 keeps 90 percent of the energy), and
    when most entries are zero it would amplify a difference that lost nothing.
    """
    count = delta.numel()
    keep = int(round(count * density))
    if keep >= count:
        return delta
    if keep <= 0:
        return torch.zeros_like(delta)
    mask = top_mask(delta, keep, cutoff)
    out = delta * mask.to(delta.dtype)
    if rescale:
        kept_norm = float(out.double().norm())
        if kept_norm > 0.0:
            out = out * (float(delta.double().norm()) / kept_norm)
    return out


# ---------------------------------------------------------------- weaver


class Weaver:
    """Holds the host base and the members; applies a genome to a model in place.

    The model must be the host the members were built against. Build the
    Weaver before any change to the model: it copies the host values of every
    tensor a member can write.
    """

    def __init__(self, model, members, n_buckets=8, density_step=0.05):
        self.members = list(members)
        self.n_buckets = n_buckets
        self.density_step = density_step
        self.shuffle_seed = None
        self.model_id = id(model)
        params = dict(model.named_parameters())
        self.info = {}
        layers = 0
        for name in params:
            info = classify(name)
            if info is None:
                continue
            self.info[name] = info
            if info[1] is not None:
                layers = max(layers, info[1] + 1)
        self.n_layers = layers
        names = set()
        for member in self.members:
            for name, tensor in member.tensors.items():
                if name not in params:
                    raise ValueError(f"{member.name}: {name} is not a host parameter")
                if tuple(tensor.shape) != tuple(params[name].shape):
                    raise ValueError(f"{member.name}: {name} has shape {tuple(tensor.shape)}, host {tuple(params[name].shape)}")
                names.add(name)
        self.names = sorted(names)
        self.host = {}
        for name in self.names:
            self.host[name] = params[name].detach().to("cpu").clone()
        self.dirty = set()
        self.thresholds = {}

    # --- clones

    def shuffled(self, seed):
        """A Weaver on the same data whose differences are shuffled inside each tensor (the random control)."""
        clone = Weaver.__new__(Weaver)
        clone.__dict__.update(self.__dict__)
        clone.shuffle_seed = int(seed)
        clone.thresholds = {}
        return clone

    # --- differences

    def reference_tensor(self, index, name):
        member = self.members[index]
        if member.ref is None:
            return self.host[name]
        return member.ref[name]

    def delta(self, index, name, device=None):
        """The float32 difference W_member - W_ref of one tensor (shuffled for a control Weaver)."""
        member = self.members[index]
        spec = member.tensors[name]
        ref = self.reference_tensor(index, name)
        if device is not None:
            spec = spec.to(device)
            ref = ref.to(device)
        d = spec.float() - ref.float()
        if self.shuffle_seed is not None:
            seed = (self.shuffle_seed * 1000003 + zlib.crc32(name.encode()) + index * 7919) % (2 ** 63)
            generator = torch.Generator(device=d.device)
            generator.manual_seed(seed)
            perm = torch.randperm(d.numel(), generator=generator, device=d.device)
            d = d.reshape(-1)[perm].reshape(d.shape)
        return d

    def snap(self, density):
        """Round a density to the grid so thresholds can be cached."""
        step = self.density_step
        value = round(float(density) / step) * step
        return float(min(1.0, max(step, value)))

    def trimmed(self, index, name, d, density):
        density = self.snap(density)
        if density >= 1.0:
            return d
        keep = int(round(d.numel() * density))
        key = (index, name, keep, self.shuffle_seed)
        cutoff = self.thresholds.get(key)
        if cutoff is None:
            threshold, need = rank_cutoff(d, keep)
            cutoff = (threshold.detach().clone(), need)
            self.thresholds[key] = cutoff
        return trim_top(d, density, rescale=True, cutoff=cutoff)

    # --- terms

    def terms(self, genome, name):
        """Return [(member index, weight, density)] with nonzero weight for one tensor."""
        kind, layer = self.info[name]
        out = []
        for index, member in enumerate(self.members):
            if not member.covers(name, kind):
                continue
            if kind in KINDS:
                bucket = bucket_of(layer, self.n_layers, self.n_buckets)
                weight = genome.weights[index, bucket, KINDS.index(kind)]
            else:
                weight = genome.globals_[index, GLOBALS.index(kind)]
            if weight == 0:
                continue
            out.append((index, float(weight), float(genome.density[index])))
        return out

    # --- writes

    def check(self, model):
        if id(model) != self.model_id:
            raise ValueError("this Weaver was built for another model")

    def put(self, params, name, value):
        param = params[name]
        with torch.no_grad():
            param.data.copy_(value)
        self.dirty.add(name)

    def apply(self, model, genome):
        """Write host + sum of weight * trimmed difference into the model, in float32, cast once."""
        self.check(model)
        if genome.n_members != len(self.members) or genome.n_buckets != self.n_buckets:
            raise ValueError("genome does not match this Weaver")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name in self.names:
                terms = self.terms(genome, name)
                if len(terms) == 0:
                    if name in self.dirty:
                        params[name].data.copy_(self.host[name])
                        self.dirty.discard(name)
                    continue
                device = params[name].device
                acc = self.host[name].to(device=device, dtype=torch.float32, copy=True)
                for index, weight, density in terms:
                    d = self.delta(index, name, device)
                    d = self.trimmed(index, name, d, density)
                    acc.add_(d, alpha=weight)
                self.put(params, name, acc.to(params[name].dtype))

    def restore(self, model):
        """Copy the host values back into every tensor that changed."""
        self.check(model)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name in list(self.dirty):
                params[name].data.copy_(self.host[name])
        self.dirty.clear()

    def load_member(self, model, index):
        """Write one member's own weights into the model (to score the member in the host's slot)."""
        self.check(model)
        params = dict(model.named_parameters())
        member = self.members[index]
        with torch.no_grad():
            for name, tensor in member.tensors.items():
                params[name].data.copy_(tensor.to(params[name].device))
                self.dirty.add(name)

    def apply_ties(self, model, keep=0.5, scale=1.0):
        """Corrected TIES: trim each difference to its top fraction, elect signs, take the disjoint mean.

        Differences are taken against each member's own reference in float32,
        only mixable groups (attention and MLP, and only MLP for members with a
        different positional base) take part, embeddings and norms stay with
        the host, and the result is cast once. The write is host + scale * mean.
        """
        self.check(model)
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name in self.names:
                kind, layer = self.info[name]
                if kind not in KINDS:
                    if name in self.dirty:
                        params[name].data.copy_(self.host[name])
                        self.dirty.discard(name)
                    continue
                device = params[name].device
                trimmed = []
                for index, member in enumerate(self.members):
                    if not member.covers(name, kind):
                        continue
                    d = self.delta(index, name, device)
                    trimmed.append(trim_top(d, keep, rescale=False))
                if len(trimmed) == 0:
                    if name in self.dirty:
                        params[name].data.copy_(self.host[name])
                        self.dirty.discard(name)
                    continue
                stacked = torch.stack(trimmed)
                sign = torch.sign(stacked.sum(dim=0))
                agree = (torch.sign(stacked) == sign) & (stacked != 0)
                total = (stacked * agree.to(stacked.dtype)).sum(dim=0)
                count = agree.sum(dim=0).clamp(min=1).to(stacked.dtype)
                value = self.host[name].to(device).float() + scale * total / count
                self.put(params, name, value.to(params[name].dtype))

    # --- diagnostics

    def cast_survival(self, genome, limit=8):
        """Fraction of nonzero mixed entries that change the model-dtype value after the cast, on a sample of tensors."""
        total = 0
        survive = 0
        step = max(1, len(self.names) // limit)
        for name in self.names[::step][:limit]:
            terms = self.terms(genome, name)
            if len(terms) == 0:
                continue
            host = self.host[name]
            acc = host.float()
            moved = torch.zeros(host.shape, dtype=torch.bool)
            for index, weight, density in terms:
                d = self.trimmed(index, name, self.delta(index, name), density) * weight
                moved |= d != 0
                acc = acc + d
            changed = acc.to(host.dtype) != host
            total += int(moved.sum())
            survive += int((changed & moved).sum())
        if total == 0:
            return None
        return survive / total

    def summary(self):
        """A small description of the members for weave.json."""
        out = []
        for member in self.members:
            out.append({"name": member.name, "skill": member.skill, "reference": member.ref_name,
                        "mlp_only": member.mlp_only, "tensors": len(member.tensors), "meta": member.meta})
        return out


# ---------------------------------------------------------------- objective


class Objective:
    """The search objective on one practice split.

    value = mean over target skills of (score - host) / max(best member - host, floor)
            - guard_weight * sum over guard skills of max(0, host - tol - score).
    All scores are fractions in [0, 1].
    """

    def __init__(self, targets, guards, host_scores, best_member_scores, tol=0.01, guard_weight=20.0, floor=0.05):
        self.targets = list(targets)
        self.guards = list(guards)
        self.host = dict(host_scores)
        self.best = dict(best_member_scores)
        self.tol = tol
        self.guard_weight = guard_weight
        self.floor = floor

    @classmethod
    def from_reference(cls, targets, guards, host_scores, member_scores, **kwargs):
        """Build the objective from the host's scores and a list of the members' score dicts."""
        best = {}
        for skill in targets:
            values = [m[skill] for m in member_scores if skill in m]
            best[skill] = max(values) if len(values) > 0 else host_scores[skill]
        return cls(targets, guards, host_scores, best, **kwargs)

    def __call__(self, scores):
        """Return (value, detail)."""
        gains = {}
        total = 0.0
        for skill in self.targets:
            denom = max(self.best[skill] - self.host[skill], self.floor)
            gains[skill] = (scores[skill] - self.host[skill]) / denom
            total += gains[skill]
        value = total / max(1, len(self.targets))
        penalty = 0.0
        drops = {}
        for skill in self.guards:
            excess = max(0.0, self.host[skill] - self.tol - scores[skill])
            drops[skill] = excess
            penalty += excess
        value -= self.guard_weight * penalty
        return value, {"gains": gains, "guard_excess": drops, "denom_clipped": self.clipped()}

    def clipped(self):
        """Per target skill: True when the best member is within the floor of the host, so the floor sets the scale."""
        return {skill: bool(self.best[skill] - self.host[skill] <= self.floor) for skill in self.targets}


# ---------------------------------------------------------------- probe objective
#
# A benchmark-free, forward-only objective. Each member is the teacher on text
# of its own skill pool. A teacher cache keeps, per position, the top-k token
# ids, their log-probabilities, and the leftover tail mass, so a candidate is
# scored with forward passes only. No generation, no benchmark item.

PROBE_TOP_K = 64


def text_hash(texts, max_length=384, top_k=PROBE_TOP_K):
    """MD5 of the texts and the cache settings; part of a cache key."""
    digest = hashlib.md5()
    digest.update(json.dumps([int(max_length), int(top_k)]).encode())
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def cache_path(directory, label, revision, texts_hash):
    """Path of one teacher cache: keyed by the model label, the repo revision, and the text hash."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    rev = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(revision))[:16]
    return os.path.join(directory, f"{safe}.{rev}.{texts_hash[:16]}.npz")


@torch.no_grad()
def build_teacher_cache(model, tokenizer, texts, top_k=PROBE_TOP_K, max_length=384, batch_size=8, vocab=None):
    """Run texts through a model, teacher-forced, and keep the top-k distribution at every position.

    Returns a dict of numpy arrays. The tokens of all texts are stored flat:
      lengths  int32 [N]      tokens per text
      ids      int32 [T]      the input token ids
      top_ids  int32 [T, k]   the k most likely next tokens at each position
      top_logp float16 [T, k] their log-probabilities (full softmax)
      tail     float16 [T]    the probability mass outside the top k
    vocab cuts the logits to the first vocab entries (a shared prefix when
    vocabularies differ only by unused tail rows).
    """
    device = probe.model_device(model)
    lengths = []
    flat_ids = []
    flat_top = []
    flat_logp = []
    flat_tail = []
    used_vocab = None
    used_k = None
    for start in range(0, len(texts), batch_size):
        ids, attention = probe.encode(tokenizer, texts[start:start + batch_size], max_length, device)
        logits = model(input_ids=ids, attention_mask=attention).logits.float()
        if vocab is not None:
            logits = logits[..., :vocab]
        used_vocab = int(logits.shape[-1])
        used_k = min(int(top_k), used_vocab)
        logz = torch.logsumexp(logits, dim=-1, keepdim=True)
        values, index = torch.topk(logits, used_k, dim=-1)
        logp = values - logz
        rest = logits.scatter(-1, index, float("-inf"))
        tail = torch.exp(torch.logsumexp(rest, dim=-1) - logz.squeeze(-1))
        for row in range(ids.shape[0]):
            count = int(attention[row].sum())
            lengths.append(count)
            flat_ids.append(ids[row, :count].cpu().numpy().astype(np.int32))
            flat_top.append(index[row, :count].cpu().numpy().astype(np.int32))
            flat_logp.append(logp[row, :count].cpu().numpy().astype(np.float16))
            flat_tail.append(tail[row, :count].cpu().numpy().astype(np.float16))
        del logits, rest
    return {"lengths": np.array(lengths, dtype=np.int32), "ids": np.concatenate(flat_ids),
            "top_ids": np.concatenate(flat_top), "top_logp": np.concatenate(flat_logp),
            "tail": np.concatenate(flat_tail), "top_k": int(used_k), "vocab": int(used_vocab),
            "max_length": int(max_length), "texts_hash": text_hash(texts, max_length, top_k)}


def save_cache(cache, path):
    """Write a teacher cache to an .npz file (atomic: temp file, then rename)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp = path + ".tmp.npz"
    np.savez(temp, lengths=cache["lengths"], ids=cache["ids"], top_ids=cache["top_ids"],
             top_logp=cache["top_logp"], tail=cache["tail"],
             meta=np.array(json.dumps({"top_k": cache["top_k"], "vocab": cache["vocab"],
                                       "max_length": cache["max_length"], "texts_hash": cache["texts_hash"]})))
    os.replace(temp, path)


def load_cache(path):
    """Read a teacher cache written by save_cache."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return {"lengths": data["lengths"], "ids": data["ids"], "top_ids": data["top_ids"],
                "top_logp": data["top_logp"], "tail": data["tail"], "top_k": meta["top_k"],
                "vocab": meta["vocab"], "max_length": meta["max_length"], "texts_hash": meta["texts_hash"]}


@torch.no_grad()
def kl_to_cache(model, cache, batch_size=8):
    """Mean KL(teacher || model) per token in nats, on the cached top-k plus one tail bucket.

    The teacher is the cached distribution. The model reads the cached token
    ids. Per position: the sum over the top-k of p (log p - log q), plus the
    tail bucket p_tail (log p_tail - log q_tail). This is a lower bound of
    the full KL and is close to it when the top k holds most of the mass. The
    first position of each text is left out, as in wsp.probe. Returns
    {"kl": float, "tokens": int}.
    """
    device = probe.model_device(model)
    lengths = cache["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    vocab = int(cache["vocab"])
    total = 0.0
    tokens = 0.0
    for start in range(0, len(lengths), batch_size):
        stop = min(start + batch_size, len(lengths))
        width = int(lengths[start:stop].max())
        count = stop - start
        k = int(cache["top_k"])
        ids = np.zeros((count, width), dtype=np.int64)
        attention = np.zeros((count, width), dtype=np.int64)
        top_ids = np.zeros((count, width, k), dtype=np.int64)
        top_logp = np.zeros((count, width, k), dtype=np.float32)
        tail = np.zeros((count, width), dtype=np.float32)
        for row in range(count):
            lo = int(offsets[start + row])
            hi = int(offsets[start + row + 1])
            n = hi - lo
            ids[row, :n] = cache["ids"][lo:hi]
            attention[row, :n] = 1
            top_ids[row, :n] = cache["top_ids"][lo:hi]
            top_logp[row, :n] = cache["top_logp"][lo:hi].astype(np.float32)
            tail[row, :n] = cache["tail"][lo:hi].astype(np.float32)
        ids_t = torch.from_numpy(ids).to(device)
        att_t = torch.from_numpy(attention).to(device)
        logits = model(input_ids=ids_t, attention_mask=att_t).logits.float()[..., :vocab]
        logz = torch.logsumexp(logits, dim=-1, keepdim=True)
        index = torch.from_numpy(top_ids).to(logits.device)
        s_top = logits.gather(-1, index) - logz
        rest = logits.scatter(-1, index, float("-inf"))
        s_tail = (torch.logsumexp(rest, dim=-1) - logz.squeeze(-1)).clamp(min=-30.0)
        t_logp = torch.from_numpy(top_logp).to(logits.device)
        t_tail = torch.from_numpy(tail).to(logits.device)
        kl = (t_logp.exp() * (t_logp - s_top)).sum(-1)
        kl = kl + t_tail * (torch.log(t_tail.clamp(min=1e-8)) - s_tail)
        mask = probe.token_mask(att_t).to(logits.device)
        total += float((kl * mask).sum())
        tokens += float(mask.sum())
        del logits, rest
    if tokens == 0:
        return {"kl": 0.0, "tokens": 0}
    return {"kl": total / tokens, "tokens": int(tokens)}


class ProbeObjective:
    """Forward-only search objective from teacher caches. No generation and no benchmark.

    For each member i with its own pool texts: the teacher is the member's
    cache. transfer_i = gap closed between KL(member_i || host) and 0, so the
    host scores 0 and the member itself scores 1. Drift is KL(host || model)
    on general texts, from the host's cache. The objective is

        mean over members of transfer_i - drift_weight * max(0, drift - drift_tolerance)

    Members whose gap to the host is zero (no signal on their pool) are
    left out of the mean and reported as None. Call calibrate(host model)
    once with the host weights in the model, then scores(model) and
    objective(scores) work like the score and Objective pair that search()
    takes: search(weaver, model, po_a.scores, po_b.scores, po_a, po_b).
    """

    def __init__(self, member_names, member_caches, drift_cache, drift_tolerance=0.05, drift_weight=10.0,
                 batch_size=8, host_pool_caches=None):
        if len(set(member_names)) != len(member_names):
            raise ValueError("member names must be unique")
        self.names = list(member_names)
        self.caches = dict(member_caches)
        self.drift_cache = drift_cache
        self.drift_tolerance = float(drift_tolerance)
        self.drift_weight = float(drift_weight)
        self.batch_size = batch_size
        self.host_pool_caches = dict(host_pool_caches or {})
        self.baseline = None
        self.drift_floor = None

    def calibrate(self, model):
        """Measure KL(member || host) on every member pool with the host weights in the model."""
        base = {}
        for name in self.names:
            base[name] = kl_to_cache(model, self.caches[name], self.batch_size)["kl"]
        self.baseline = base
        self.drift_floor = kl_to_cache(model, self.drift_cache, self.batch_size)["kl"]
        return {"baseline_kl": dict(base), "drift_floor": self.drift_floor}

    def scores(self, model):
        """Forward passes only. Returns {"transfer": {name: value or None}, "member_kl": {...}, "drift": float}."""
        if self.baseline is None:
            raise ValueError("calibrate the objective on the host first")
        transfers = {}
        kls = {}
        for name in self.names:
            kl = kl_to_cache(model, self.caches[name], self.batch_size)["kl"]
            kls[name] = kl
            transfers[name] = probe.gap_closed(self.baseline[name], kl, 0.0)
        drift = kl_to_cache(model, self.drift_cache, self.batch_size)["kl"]
        out = {"transfer": transfers, "member_kl": kls, "drift": drift}
        if len(self.host_pool_caches) > 0:
            pool_drift = {}
            for label in sorted(self.host_pool_caches):
                pool_drift[label] = kl_to_cache(model, self.host_pool_caches[label], self.batch_size)["kl"]
            out["pool_drift"] = pool_drift
        return out

    def mean_transfer(self, scores):
        """Mean transfer over the members that have a gap (None when none has)."""
        values = []
        for name in self.names:
            value = scores["transfer"][name]
            if value is not None:
                values.append(value)
        if len(values) == 0:
            return None
        return sum(values) / len(values)

    def __call__(self, scores):
        """Return (value, detail)."""
        mean = self.mean_transfer(scores)
        if mean is None:
            mean = 0.0
        excess = max(0.0, scores["drift"] - self.drift_tolerance)
        penalty = self.drift_weight * excess
        detail = {"transfer": dict(scores["transfer"]), "mean_transfer": mean, "drift": scores["drift"],
                  "drift_excess": excess, "penalty": penalty}
        return mean - penalty, detail


# ---------------------------------------------------------------- CMA-ES


class SepCMAES:
    """Separable (diagonal) CMA-ES in the unit cube, maximizing a fitness. Seeded, numpy only."""

    def __init__(self, mean, sigma, lower, upper, popsize, seed):
        self.lower = np.asarray(lower, dtype=float)
        self.span = np.maximum(np.asarray(upper, dtype=float) - self.lower, 1e-12)
        n = len(mean)
        self.n = n
        self.popsize = popsize
        self.rng = np.random.default_rng(seed)
        self.mean = np.clip((np.asarray(mean, dtype=float) - self.lower) / self.span, 0.0, 1.0)
        self.sigma = float(sigma)
        mu = max(1, popsize // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.w = w / w.sum()
        self.mu = mu
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        cmu = min(1 - c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        scale = (n + 2) / 3.0
        c1 *= scale
        cmu *= scale
        if c1 + cmu > 0.9:
            shrink = 0.9 / (c1 + cmu)
            c1 *= shrink
            cmu *= shrink
        self.c1 = c1
        self.cmu = cmu
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.C = np.ones(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.generation = 0

    def to_genes(self, unit):
        return self.lower + self.span * unit

    def ask(self):
        """Return popsize candidate vectors (clipped to the bounds)."""
        z = self.rng.standard_normal((self.popsize, self.n))
        unit = self.mean + self.sigma * z * np.sqrt(self.C)
        unit = np.clip(unit, 0.0, 1.0)
        return self.to_genes(unit)

    def move_mean(self, genes):
        """Put the mean at a point that was not drawn from the distribution (a seed). C, sigma and the paths stay."""
        unit = (np.asarray(genes, dtype=float) - self.lower) / self.span
        self.mean = np.clip(unit, 0.0, 1.0)

    def tell(self, genes, fitness):
        """Update from evaluated candidates (any number at least 1); higher fitness is better."""
        genes = np.asarray(genes, dtype=float)
        unit = (genes - self.lower) / self.span
        order = np.argsort(-np.asarray(fitness, dtype=float))
        mu = min(self.mu, len(order))
        w = self.w[:mu] / self.w[:mu].sum()
        top = unit[order[:mu]]
        y = (top - self.mean) / self.sigma
        ymean = (w[:, None] * y).sum(axis=0)
        self.mean = np.clip(self.mean + self.sigma * ymean, 0.0, 1.0)
        mueff = 1.0 / np.sum(w ** 2)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * mueff) * ymean / np.sqrt(self.C)
        self.generation += 1
        norm_ps = np.linalg.norm(self.ps)
        hsig = norm_ps / math.sqrt(1 - (1 - self.cs) ** (2 * self.generation)) / self.chi < 1.4 + 2 / (self.n + 1)
        hsig = 1.0 if hsig else 0.0
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * mueff) * ymean
        rank_mu = (w[:, None] * y ** 2).sum(axis=0)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.C = np.maximum(self.C, 1e-8)
        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chi - 1))
        self.sigma = float(min(max(self.sigma, 1e-3), 0.5))


# ---------------------------------------------------------------- search


def search(weaver, model, score_a, score_b, objective_a, objective_b, popsize=12, generations=8, seed=0,
           init=None, sigma=0.2, free_globals=False, search_density=True, weight_range=(0.0, 1.0),
           density_range=(0.5, 1.0), log=print, progress=None):
    """Search the genome with a diagonal CMA-ES.

    score_a(model) and score_b(model) return {skill: value} on two disjoint
    practice splits. Generation 0 holds the zero genome, one one-hot genome per
    member, and the uniform average, plus the first popsize samples. The
    CMA-ES update sees only the samples, which came from its distribution;
    when a seed beats every sample, the mean moves to that seed. Each
    generation scores its best candidate again on split B; a candidate's
    validated value is min(objective A, objective B), and the returned best
    is the highest validated value over generations.

    Returns {"best": {...}, "best_genome": Genome, "log": [entries], "summary": {...}}.
    Every entry keeps its genome vector, scores, and objective. The model is
    restored to the host at the end.
    """
    count = len(weaver.members)
    buckets = weaver.n_buckets
    lower, upper = genome_bounds(count, buckets, weight_range, density_range)
    mask = active_mask(count, buckets, free_globals, search_density)
    fixed = np.zeros(len(lower))
    for index in range(len(lower)):
        if not mask[index]:
            fixed[index] = 1.0 if (index % (buckets * 2 + 3)) == buckets * 2 + 2 else 0.0
    if init is None:
        init = uniform_genome(count, buckets)
    start = init.to_vector()
    start = np.clip(start, lower, upper)
    start = np.where(mask, start, fixed)
    active = np.where(mask)[0]
    cma = SepCMAES(start[active], sigma, lower[active], upper[active], popsize, seed)

    def expand(vector_active):
        full = fixed.copy()
        full[active] = vector_active
        return full

    log_entries = []
    best = None
    started = time.time()
    names = [m.name for m in weaver.members]

    def evaluate(vector, label, generation):
        genome = Genome.from_vector(vector, count, buckets)
        weaver.apply(model, genome)
        scores = score_a(model)
        value, detail = objective_a(scores)
        entry = {"generation": generation, "label": label, "genome": vector.tolist(), "scores_a": scores,
                 "objective_a": value, "detail_a": detail, "seconds": round(time.time() - started, 1)}
        log_entries.append(entry)
        if progress is not None:
            progress(entry)
        return entry

    try:
        for generation in range(generations):
            batch = []
            if generation == 0:
                seeds = [("zero", zero_genome(count, buckets).to_vector())]
                for index in range(count):
                    seeds.append(("one_hot:" + names[index],
                                  one_hot_genome(count, buckets, index, 1.0, with_globals=free_globals).to_vector()))
                seeds.append(("uniform", uniform_genome(count, buckets, with_globals=free_globals).to_vector()))
                for label, vector in seeds:
                    vector = np.where(mask, np.clip(vector, lower, upper), fixed)
                    batch.append((label, vector))
            for vector in cma.ask():
                batch.append(("sample", expand(vector)))
            entries = []
            for label, vector in batch:
                entries.append(evaluate(vector, label, generation))
                log(f"gen {generation} {label}: objective {entries[-1]['objective_a']:.4f}")
            sampled = [e for e in entries if e["label"] == "sample"]
            injected = [e for e in entries if e["label"] != "sample"]
            cma.tell(np.array([e["genome"] for e in sampled])[:, active], [e["objective_a"] for e in sampled])
            if len(injected) > 0:
                lead = max(injected, key=lambda e: e["objective_a"])
                if lead["objective_a"] > max(e["objective_a"] for e in sampled):
                    cma.move_mean(np.array(lead["genome"])[active])
            top = max(entries, key=lambda e: e["objective_a"])
            genome = Genome.from_vector(np.array(top["genome"]), count, buckets)
            weaver.apply(model, genome)
            scores_b = score_b(model)
            value_b, detail_b = objective_b(scores_b)
            top["scores_b"] = scores_b
            top["objective_b"] = value_b
            top["detail_b"] = detail_b
            top["validated"] = min(top["objective_a"], value_b)
            log(f"gen {generation} best A {top['objective_a']:.4f} B {value_b:.4f}")
            if best is None or top["validated"] > best["validated"]:
                best = top
    finally:
        weaver.restore(model)
    best_genome = Genome.from_vector(np.array(best["genome"]), count, buckets)
    summary = {"evaluations": len(log_entries), "generations": generations, "popsize": popsize, "seed": seed,
               "best_generation": best["generation"], "best_label": best["label"],
               "best_objective_a": best["objective_a"], "best_objective_b": best["objective_b"],
               "best_validated": best["validated"],
               "per_generation": [
                   {"generation": e["generation"], "label": e["label"], "objective_a": e["objective_a"],
                    "objective_b": e.get("objective_b"), "validated": e.get("validated")}
                   for e in log_entries if "validated" in e],
               "members": names, "seconds": round(time.time() - started, 1)}
    return {"best": best, "best_genome": best_genome, "log": log_entries, "summary": summary}


# ---------------------------------------------------------------- output


def save(weaver, model, genome, path, tokenizer=None, host_id=None, search_summary=None, extra=None):
    """Write the model with the genome applied as a plain Hugging Face directory, plus weave.json.

    The directory has exactly the host's tensors and loads with
    AutoModelForCausalLM; it needs no wsp code. The model is restored to the
    host afterwards.
    """
    weaver.apply(model, genome)
    try:
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)
    finally:
        weaver.restore(model)
    record = {"host": host_id, "genome": genome.to_dict(), "members": weaver.summary(),
              "reference": [m.ref_name for m in weaver.members], "n_buckets": weaver.n_buckets,
              "n_layers": weaver.n_layers, "density_step": weaver.density_step}
    if search_summary is not None:
        record["search"] = search_summary
    if extra is not None:
        record["extra"] = extra
    with open(os.path.join(path, "weave.json"), "w") as handle:
        json.dump(record, handle, indent=2, default=str)
    return record
