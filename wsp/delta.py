"""Skill deltas: mine a specialist as a compressed weight delta, switch it per sequence.

A specialist fine-tune keeps its skill only when its weights run as it was
trained (run 16b). A skill delta stores the specialist as D = W_spec - W_host
per weight matrix, compressed to a rank-r factor pair, plus the small vectors
(norms, biases) exactly. One network holds the host and a bank of deltas.
One delta is active per sequence. Params run per token stay at the host
plus the active delta's factors.

Pieces:
  mine_delta         mine one specialist against the host (SVD, per matrix stats)
  Delta              one mined delta: factors, vectors, attributes, stats
  SkillDeltaNetwork  host model plus a bank of deltas, applied on the fly by
                     forward hooks (out = W x + scale * B (A x))
  SequenceRouter     ridge classifier on the host's mean-pooled early hidden state
  merge              fold one delta into a plain model copy (a check)
  save / load        one directory

Module names are parameter names without ".weight", for example
"model.layers.3.self_attn.q_proj". Embeddings and the output head are
optional; they are applied by writing the dense weight when the expert
changes, not by a hook.

The rope base is part of how a specialist was trained (Qwen2.5-Math has
rope_theta 10000, the instruct and coder models 1e6). The network switches
the rotary frequencies with the expert when the thetas differ.
"""

import copy
import glob
import json
import os

import torch
import torch.nn.functional as F

BASE = "base"

KIND_OF = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
    "embed_tokens": "embed", "lm_head": "head",
}
DENSE_KINDS = ["embed", "head"]
TENSOR_EXTRAS = ["*.json", "*.txt", "*.jinja", "*.safetensors"]


# ---------------------------------------------------------------- sources


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class WeightSource:
    """Lazy access to the weights of a model object, a local directory, or a hub id."""

    def __init__(self, obj):
        self.id = None
        self.model = None
        self.params = {}
        self.handles = {}
        self.weight_map = {}
        self.config = {}
        self.generation_config = None
        self.chat_template = None
        if isinstance(obj, str):
            self.id = obj
            path = obj
            if not os.path.isdir(obj):
                from huggingface_hub import snapshot_download

                path = snapshot_download(obj, allow_patterns=TENSOR_EXTRAS)
            self.open_dir(path)
        else:
            self.model = obj
            self.params = dict(obj.named_parameters())
            self.config = obj.config.to_dict()
            if getattr(obj, "generation_config", None) is not None:
                self.generation_config = obj.generation_config.to_dict()
            self.chat_template = getattr(obj, "chat_template", None)

    def open_dir(self, path):
        from safetensors import safe_open

        self.config = read_json(os.path.join(path, "config.json")) or {}
        self.generation_config = read_json(os.path.join(path, "generation_config.json"))
        tokenizer_config = read_json(os.path.join(path, "tokenizer_config.json")) or {}
        self.chat_template = tokenizer_config.get("chat_template")
        template_file = os.path.join(path, "chat_template.jinja")
        if self.chat_template is None and os.path.exists(template_file):
            with open(template_file) as f:
                self.chat_template = f.read()
        index = read_json(os.path.join(path, "model.safetensors.index.json"))
        files = {}
        if index is not None:
            self.weight_map = dict(index["weight_map"])
        else:
            for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                with safe_open(file, framework="pt") as handle:
                    for key in handle.keys():
                        self.weight_map[key] = os.path.basename(file)
        for key in self.weight_map:
            files[self.weight_map[key]] = True
        for file in files:
            self.handles[file] = safe_open(os.path.join(path, file), framework="pt")

    def names(self):
        if self.model is not None:
            return list(self.params.keys())
        return list(self.weight_map.keys())

    def shape(self, name):
        if self.model is not None:
            return tuple(self.params[name].shape)
        return tuple(self.handles[self.weight_map[name]].get_slice(name).get_shape())

    def get(self, name, device="cpu"):
        """Return the tensor as float32 on the device."""
        if self.model is not None:
            return self.params[name].detach().to(device=device, dtype=torch.float32)
        tensor = self.handles[self.weight_map[name]].get_tensor(name)
        return tensor.to(device=device, dtype=torch.float32)


def rope_theta_of(config):
    """Return rope_theta from a config dict or object, or None."""
    if config is None:
        return None
    if not isinstance(config, dict):
        config = config.to_dict()
    if config.get("rope_theta") is not None:
        return float(config["rope_theta"])
    parameters = config.get("rope_parameters")
    if isinstance(parameters, dict) and parameters.get("rope_theta") is not None:
        return float(parameters["rope_theta"])
    return None


ATTR_KEYS = ["rope_theta", "max_position_embeddings", "rope_scaling", "sliding_window", "rms_norm_eps",
             "vocab_size", "bos_token_id", "eos_token_id", "torch_dtype", "dtype"]


def non_weight_attrs(source):
    """The differences that matter besides weights: rope, positions, template, generation config."""
    attrs = {}
    for key in ATTR_KEYS:
        if key in source.config:
            attrs[key] = source.config[key]
    attrs["rope_theta"] = rope_theta_of(source.config)
    attrs["chat_template"] = source.chat_template
    attrs["generation_config"] = source.generation_config
    return attrs


def attr_differences(host_attrs, spec_attrs):
    """Return {key: [host value, specialist value]} for every attribute that differs."""
    out = {}
    for key in sorted(set(host_attrs) | set(spec_attrs)):
        a = host_attrs.get(key)
        b = spec_attrs.get(key)
        if a != b:
            if key == "chat_template":
                out[key] = ["same text" if a == b else "differs", len(a or ""), len(b or "")]
            else:
                out[key] = [a, b]
    return out


# ---------------------------------------------------------------- ranks


def kind_of(module_name):
    return KIND_OF.get(module_name.split(".")[-1], "other")


def rank_for_kind(spec, kind):
    """Pick the spec of one kind from a dict spec (with an optional "default")."""
    if isinstance(spec, dict):
        if kind in spec:
            return spec[kind]
        return spec.get("default", 0)
    return spec


def energy_rank(sv, target):
    """Smallest rank whose cumulative energy (sum of squared singular values) reaches target."""
    energy = sv.double() ** 2
    total = float(energy.sum())
    if total <= 0:
        return 1
    cum = torch.cumsum(energy, 0) / total
    count = int((cum < target - 1e-12).sum()) + 1
    return min(count, len(sv))


def resolve_rank(spec, kind, shape, sv=None):
    """Turn a rank spec into 0 (skip), "full", or an int.

    spec: an int rank, "full", a float in (0, 1] as an energy target, None or 0
    to skip, or a dict of those by kind (q, k, v, o, gate, up, down, embed,
    head, with "default" for the rest).
    """
    spec = rank_for_kind(spec, kind)
    if spec is None or spec == 0:
        return 0
    if spec == "full":
        return "full"
    limit = min(shape)
    if isinstance(spec, float):
        if sv is None:
            raise ValueError("an energy target needs the singular values")
        return energy_rank(sv, spec)
    return max(1, min(int(spec), limit))


def frob_of_product(B, A):
    """||B A||_F from the small r x r Gram matrices."""
    gram_b = B.double().T @ B.double()
    gram_a = A.double() @ A.double().T
    return float((gram_b * gram_a).sum().clamp(min=0).sqrt())


# ---------------------------------------------------------------- delta


class Delta:
    """One mined delta.

    factors: {module: {"A", "B"}} (low rank) or {module: {"D"}} (dense, rank "full").
    vectors: {param name: specialist value} for 1-D params that differ.
    attrs: non-weight attributes of the specialist and its differences from the host.
    stats: {module: {kind, shape, rank, rel_err, energy_kept, rank90, rank99}}.
    spectrum: {module: singular values}, kept in memory only.
    """

    def __init__(self, name, source_id, rank, factors, vectors, attrs, stats, spectrum=None, scale=1.0):
        self.name = name
        self.source_id = source_id
        self.rank = rank
        self.factors = factors
        self.vectors = vectors
        self.attrs = attrs
        self.stats = stats
        self.spectrum = spectrum or {}
        self.scale = scale

    def stored_params(self):
        total = 0
        for module in self.factors:
            for tensor in self.factors[module].values():
                total += tensor.numel()
        for name in self.vectors:
            total += self.vectors[name].numel()
        return total

    def factor_params(self):
        """Params of the matrix factors only (what runs per token on top of the host)."""
        total = 0
        for module in self.factors:
            for tensor in self.factors[module].values():
                total += tensor.numel()
        return total

    def summary(self):
        return summarize_stats(self.stats)

    def truncate(self, rank, name=None):
        """Return a copy at a lower rank, cut from the SVD factors (best rank-r approximation)."""
        if self.attrs.get("control"):
            raise ValueError("a random control cannot be truncated")
        factors = {}
        stats = {}
        for module in self.factors:
            old = self.stats[module]
            kind = old["kind"]
            sv = self.spectrum.get(module)
            parts = self.factors[module]
            if "D" in parts:
                u, s, vh = torch.linalg.svd(parts["D"].float(), full_matrices=False)
                full_B = u * s
                full_A = vh
                sv = s if sv is None else sv
            else:
                full_B = parts["B"]
                full_A = parts["A"]
            shape = tuple(old["shape"])
            target = resolve_rank(rank, kind, shape, sv)
            entry = dict(old)
            if target == 0:
                continue
            if target == "full":
                if "D" in parts:
                    factors[module] = {"D": parts["D"].clone()}
                    stats[module] = entry
                    continue
                raise ValueError("cannot grow low-rank factors to full")
            keep = min(target, full_A.shape[0])
            B = full_B[:, :keep].contiguous().clone()
            A = full_A[:keep].contiguous().clone()
            factors[module] = {"A": A, "B": B}
            entry["rank"] = keep
            if sv is not None:
                cum = torch.cumsum(sv.double() ** 2, 0) / sv.double().pow(2).sum().clamp(min=1e-30)
                entry["energy_kept"] = float(cum[keep - 1])
                entry["rel_err"] = float((1.0 - cum[keep - 1]).clamp(min=0).sqrt())
            else:
                entry["energy_kept"] = None
                entry["rel_err"] = None
            stats[module] = entry
        vectors = {}
        for key in self.vectors:
            vectors[key] = self.vectors[key].clone()
        return Delta(name or self.name, self.source_id, rank, factors, vectors, copy.deepcopy(self.attrs), stats,
                     spectrum=self.spectrum, scale=self.scale)


def summarize_stats(stats):
    """Per kind and overall: count, mean and median rank for 90% and 99% energy, error, energy kept."""
    groups = {"all": []}
    for module in stats:
        kind = stats[module]["kind"]
        groups.setdefault(kind, []).append(stats[module])
        groups["all"].append(stats[module])
    out = {}
    for kind in groups:
        rows = groups[kind]
        if len(rows) == 0:
            continue
        entry = {"n": len(rows)}
        for key in ["rank90", "rank99", "rel_err", "energy_kept", "rank"]:
            values = []
            for row in rows:
                if row.get(key) is not None:
                    values.append(float(row[key]))
            if len(values) > 0:
                values.sort()
                entry[key + "_mean"] = sum(values) / len(values)
                entry[key + "_median"] = values[len(values) // 2]
                entry[key + "_max"] = values[-1]
        fractions = []
        for row in rows:
            if row.get("rank90") is not None:
                fractions.append(row["rank90"] / min(row["shape"]))
        if len(fractions) > 0:
            entry["rank90_over_min_dim_mean"] = sum(fractions) / len(fractions)
        out[kind] = entry
    return out


# ---------------------------------------------------------------- mining


def random_factors(shape_out, shape_in, rank, target_norm, generator):
    """Random A (r x in) and B (out x r) with ||B A||_F equal to target_norm."""
    A = torch.randn(rank, shape_in, generator=generator)
    B = torch.randn(shape_out, rank, generator=generator)
    current = frob_of_product(B, A)
    if current > 0:
        B = B * (target_norm / current)
    return A, B


def mine_delta(host, specialist, rank=64, name=None, include_embeddings=False, device="cpu", control=None,
               seed=0, factor_dtype=torch.float32, spectrum=True, keep_spectrum=True):
    """Mine D = W_specialist - W_host for every shared weight matrix and keep a low-rank form.

    host, specialist: a model, a local directory, or a hub id. Both are read
    one tensor at a time (lazily for directories and ids), so memory stays
    near one matrix plus its SVD.
    rank: int, "full" (store D), a float energy target, or a dict by kind.
    control: None, or "random": same shapes and the same Frobenius norm of
    the stored product per matrix, random directions (seeded). A control keeps
    the host's norms, biases, and rope (it carries no vectors).
    spectrum: also compute singular values for "full" matrices (for the rank
    90 and 99 stats); low-rank matrices always get them.
    Returns a Delta.
    """
    if control not in [None, "random"]:
        raise ValueError("control is None or 'random'")
    host_src = WeightSource(host)
    spec_src = WeightSource(specialist)
    host_attrs = non_weight_attrs(host_src)
    spec_attrs = non_weight_attrs(spec_src)
    if name is None:
        name = spec_src.id or "specialist"
    factors = {}
    vectors = {}
    stats = {}
    spectra = {}
    skipped = []
    generator = torch.Generator().manual_seed(seed)
    host_names = set(host_src.names())
    embed_rel_diff = None
    for param_name in spec_src.names():
        if param_name not in host_names:
            skipped.append([param_name, "not in host"])
            continue
        shape = spec_src.shape(param_name)
        if tuple(host_src.shape(param_name)) != tuple(shape):
            skipped.append([param_name, "shape differs"])
            continue
        if len(shape) == 1:
            if control is not None:
                continue
            w_host = host_src.get(param_name)
            w_spec = spec_src.get(param_name)
            if not torch.equal(w_host, w_spec):
                vectors[param_name] = w_spec.to(factor_dtype).cpu()
            continue
        if len(shape) != 2 or not param_name.endswith(".weight"):
            skipped.append([param_name, "not a matrix"])
            continue
        module = param_name[:-len(".weight")]
        kind = kind_of(module)
        if kind == "other":
            skipped.append([param_name, "unknown matrix kind"])
            continue
        if kind in DENSE_KINDS and not include_embeddings:
            if kind == "embed":
                w_host = host_src.get(param_name, device)
                diff = spec_src.get(param_name, device) - w_host
                embed_rel_diff = float(diff.norm() / w_host.norm().clamp(min=1e-30))
            continue
        w_host = host_src.get(param_name, device)
        D = spec_src.get(param_name, device) - w_host
        del w_host
        fro = float(D.norm())
        entry = {"kind": kind, "shape": list(shape), "fro": fro}
        if fro == 0.0:
            entry.update({"rank": 0, "rel_err": 0.0, "energy_kept": 1.0, "rank90": 0, "rank99": 0})
            stats[module] = entry
            continue
        spec = rank_for_kind(rank, kind)
        dense = spec == "full"
        sv = None
        u = vh = None
        if dense:
            if spectrum:
                sv = torch.linalg.svdvals(D)
        elif spec is None or spec == 0:
            stats[module] = dict(entry, rank=0)
            continue
        else:
            u, sv, vh = torch.linalg.svd(D, full_matrices=False)
        if sv is not None:
            cum = torch.cumsum(sv.double() ** 2, 0) / sv.double().pow(2).sum()
            entry["rank90"] = int((cum < 0.9 - 1e-12).sum()) + 1
            entry["rank99"] = int((cum < 0.99 - 1e-12).sum()) + 1
            spectra[module] = sv.float().cpu() if keep_spectrum else None
        else:
            cum = None
            entry["rank90"] = None
            entry["rank99"] = None
        if dense:
            entry["rank"] = min(shape)
            entry["energy_kept"] = 1.0
            entry["rel_err"] = 0.0
            if control == "random":
                R = torch.randn(shape[0], shape[1], generator=generator)
                R = R * (fro / float(R.norm()))
                factors[module] = {"D": R.to(factor_dtype)}
                entry["rel_err"] = float((D.cpu() - R).norm() / fro)
            else:
                factors[module] = {"D": D.to(factor_dtype).cpu()}
            stats[module] = entry
            continue
        r = resolve_rank(spec, kind, shape, sv)
        B = (u[:, :r] * sv[:r]).contiguous()
        A = vh[:r].contiguous()
        entry["rank"] = r
        entry["energy_kept"] = float(cum[r - 1])
        if control == "random":
            A, B = random_factors(shape[0], shape[1], r, frob_of_product(B.cpu(), A.cpu()), generator)
            A = A.to(device)
            B = B.to(device)
        entry["rel_err"] = float((D - B @ A).norm() / fro)
        factors[module] = {"A": A.to(factor_dtype).cpu(), "B": B.to(factor_dtype).cpu()}
        stats[module] = entry
        del u, vh, D
    attrs = dict(spec_attrs)
    attrs["host_attrs"] = host_attrs
    attrs["differences"] = attr_differences(host_attrs, spec_attrs)
    attrs["skipped"] = skipped
    attrs["include_embeddings"] = include_embeddings
    attrs["embedding_rel_diff_if_excluded"] = embed_rel_diff
    attrs["control"] = control
    if control is not None:
        attrs["rope_theta"] = None
    return Delta(name, spec_src.id, rank, factors, vectors, attrs, stats, spectrum=spectra)


def random_control(delta, rank=None, seed=0, name=None):
    """A random delta shaped like `delta` (cut to `rank` first), with the same Frobenius norm per matrix.

    Directions are random and seeded. The control carries no vectors and no
    rope change, so it tests the matrix directions only. Needs SVD-ordered
    factors or dense factors, so the stored norms are the true ones.
    """
    source = delta if rank is None else delta.truncate(rank)
    generator = torch.Generator().manual_seed(seed)
    factors = {}
    for module in source.factors:
        parts = source.factors[module]
        if "D" in parts:
            dense = parts["D"]
            fresh = torch.randn(dense.shape[0], dense.shape[1], generator=generator)
            fresh = fresh * (float(dense.float().norm()) / float(fresh.norm()))
            factors[module] = {"D": fresh.to(dense.dtype)}
        else:
            A, B = parts["A"], parts["B"]
            target = frob_of_product(B.float().cpu(), A.float().cpu())
            fresh_a, fresh_b = random_factors(B.shape[0], A.shape[1], A.shape[0], target, generator)
            factors[module] = {"A": fresh_a.to(A.dtype), "B": fresh_b.to(B.dtype)}
    attrs = copy.deepcopy(source.attrs)
    attrs["control"] = "random"
    attrs["rope_theta"] = None
    stats = copy.deepcopy(source.stats)
    return Delta(name or source.name + "_random", source.source_id, source.rank, factors, {}, attrs, stats,
                 spectrum=source.spectrum, scale=source.scale)


# ---------------------------------------------------------------- rope


def rotary_modules(model):
    found = []
    for module in model.modules():
        if hasattr(module, "inv_freq") and hasattr(module, "rope_init_fn"):
            found.append(module)
    return found


def set_rope_theta(model, theta):
    """Rewrite the rotary frequencies of a model for a new rope base."""
    with torch.no_grad():
        for rot in rotary_modules(model):
            config = copy.deepcopy(rot.config)
            config.rope_theta = theta
            inv_freq, scaling = rot.rope_init_fn(config, rot.inv_freq.device)
            rot.inv_freq.copy_(inv_freq.to(rot.inv_freq.dtype))
            if hasattr(rot, "original_inv_freq"):
                rot.original_inv_freq = rot.inv_freq.clone()
            rot.attention_scaling = scaling
            rot.config.rope_theta = theta
    if hasattr(model.config, "rope_theta"):
        model.config.rope_theta = theta


def host_param_counts(model):
    """(total, run per token) with the skills_eval convention: an untied input table is a lookup."""
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    run = total
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed is not None:
        tied = head is not None and head.weight is embed.weight
        if not tied:
            run -= embed.weight.numel()
    return total, run


# ---------------------------------------------------------------- router


class SequenceRouter:
    """Ridge classifier over standardized mean-pooled hidden states.

    names: expert names in column order, including "base".
    The route is the top-scoring expert, except that a non-base winner must
    lead the runner-up by base_margin, else the prompt stays on the base.
    """

    def __init__(self, names, layer, W, mean, std, base_margin=0.0):
        self.names = list(names)
        self.layer = layer
        self.W = W
        self.mean = mean
        self.std = std
        self.base_margin = base_margin

    def numel(self):
        return self.W.numel() + self.mean.numel() + self.std.numel()

    def scores(self, features):
        x = (features.double() - self.mean) / self.std
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
        return torch.cat([x, ones], dim=1) @ self.W

    def pick(self, features):
        score = self.scores(features)
        out = []
        for row in range(score.shape[0]):
            top2 = torch.topk(score[row], min(2, score.shape[1]))
            best = int(top2.indices[0])
            name = self.names[best]
            if name != BASE and len(top2.values) > 1:
                if float(top2.values[0] - top2.values[1]) < self.base_margin:
                    name = BASE
            out.append(name)
        return out

    @classmethod
    def fit(cls, names, layer, features, labels, ridge=1.0, base_margin=0.0):
        """features: (n, d); labels: list of names. One-hot targets, standardized inputs."""
        x = features.double().cpu()
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp(min=1e-6)
        x = (x - mean) / std
        x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
        y = torch.zeros(x.shape[0], len(names), dtype=x.dtype)
        for row, label in enumerate(labels):
            y[row, names.index(label)] = 1.0
        reg = ridge * x.shape[0] * torch.eye(x.shape[1], dtype=x.dtype)
        reg[-1, -1] = 0.0
        W = torch.linalg.solve(x.T @ x + reg, x.T @ y)
        return cls(names, layer, W, mean, std, base_margin)


class StopForward(Exception):
    pass


# ---------------------------------------------------------------- network


class SkillDeltaNetwork:
    """A host model plus a bank of named deltas. One expert is active per sequence.

    The host model is changed in place: forward hooks add scale * B (A x) to
    the output of every adapted linear layer, and the active expert's norms,
    biases, and rope base are copied in when it is selected. Expert "base"
    means no delta. Call close() to remove the hooks and restore the host.
    """

    def __init__(self, model, deltas=None, router=None, tokenizer=None, host_id=None):
        self.model = model
        self.tokenizer = tokenizer
        self.host_id = host_id
        self.deltas = {}
        self.runtime = {}
        self.scales = {}
        self.router = router
        self.active = BASE
        self.hooks = {}
        self.active_factors = {}
        self.host_vectors = {}
        self.host_dense = {}
        self.host_theta = rope_theta_of(model.config)
        self.max_route_tokens = 256
        if deltas is not None:
            for delta in deltas:
                self.add_delta(delta)

    # ---- bank

    def expert_names(self):
        return [BASE] + list(self.deltas.keys())

    def param_of(self, module):
        return self.model.get_parameter(module + ".weight")

    def add_delta(self, delta):
        if self.active != BASE:
            raise RuntimeError("select the base expert before adding a delta")
        if delta.name in self.deltas or delta.name == BASE:
            raise ValueError(f"expert name {delta.name!r} is taken")
        runtime = {}
        for module in delta.factors:
            param = self.param_of(module)
            entry = {}
            for key in delta.factors[module]:
                entry[key] = delta.factors[module][key].to(device=param.device, dtype=param.dtype)
            runtime[module] = entry
            kind = kind_of(module)
            if kind in DENSE_KINDS:
                if module not in self.host_dense:
                    self.host_dense[module] = param.detach().clone()
            elif module not in self.hooks:
                self.hooks[module] = self.model.get_submodule(module).register_forward_hook(self.make_hook(module))
        for key in delta.vectors:
            if key not in self.host_vectors:
                self.host_vectors[key] = self.model.get_parameter(key).detach().clone()
        self.deltas[delta.name] = delta
        self.runtime[delta.name] = runtime
        self.scales[delta.name] = delta.scale

    def make_hook(self, module):
        def hook(layer, args, output):
            entry = self.active_factors.get(module)
            if entry is None:
                return output
            x = args[0]
            scale = self.active_scale
            if "D" in entry:
                return output + scale * F.linear(x, entry["D"])
            return output + scale * F.linear(F.linear(x, entry["A"]), entry["B"])

        return hook

    active_scale = 1.0

    def set_expert(self, name):
        """Make one expert active for the next sequences."""
        if name == self.active:
            return
        if name != BASE and name not in self.deltas:
            raise KeyError(f"unknown expert {name!r}; the bank has {self.expert_names()}")
        self.apply_state(name)
        self.active = name

    def apply_state(self, name):
        delta = None
        if name != BASE:
            delta = self.deltas[name]
        self.active_scale = 1.0
        self.active_factors = {}
        if delta is not None:
            self.active_scale = self.scales[name]
            for module in self.runtime[name]:
                if kind_of(module) not in DENSE_KINDS:
                    self.active_factors[module] = self.runtime[name][module]
        with torch.no_grad():
            for key in self.host_vectors:
                param = self.model.get_parameter(key)
                if delta is not None and key in delta.vectors:
                    param.copy_(delta.vectors[key].to(device=param.device, dtype=param.dtype))
                else:
                    param.copy_(self.host_vectors[key])
            for module in self.host_dense:
                param = self.param_of(module)
                if delta is not None and module in self.runtime[name]:
                    entry = self.runtime[name][module]
                    if "D" in entry:
                        change = entry["D"].float()
                    else:
                        change = entry["B"].float() @ entry["A"].float()
                    param.copy_((self.host_dense[module].float() + self.active_scale * change).to(param.dtype))
                else:
                    param.copy_(self.host_dense[module])
        theta = self.host_theta
        if delta is not None and delta.attrs.get("rope_theta") is not None:
            theta = float(delta.attrs["rope_theta"])
        if theta is not None and rope_theta_of(self.model.config) != theta:
            set_rope_theta(self.model, theta)

    def using(self, name):
        return ExpertContext(self, name)

    def close(self):
        """Restore the host and remove the hooks."""
        self.set_expert(BASE)
        for module in self.hooks:
            self.hooks[module].remove()
        self.hooks = {}

    # ---- encoding and features

    def encode(self, prompt):
        """A prompt as a list of token ids: text, a list of ids, or a 1-D tensor."""
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("a text prompt needs a tokenizer")
            ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        elif torch.is_tensor(prompt):
            ids = prompt.reshape(-1).tolist()
        else:
            ids = list(prompt)
        return ids[:self.max_route_tokens]

    def features(self, prompts, layer, batch_size=8):
        """Mean-pooled hidden state after `layer` layers, host weights, one row per prompt."""
        if layer < 1:
            raise ValueError("layer must be at least 1")
        encoded = []
        for prompt in prompts:
            encoded.append(self.encode(prompt))
        device = next(self.model.parameters()).device
        layers = self.model.model.layers
        out = [None] * len(encoded)
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            width = max(max(len(encoded[i]) for i in chunk), 1)
            ids = torch.zeros(len(chunk), width, dtype=torch.long)
            mask = torch.zeros(len(chunk), width, dtype=torch.long)
            for row, i in enumerate(chunk):
                ids[row, :len(encoded[i])] = torch.tensor(encoded[i], dtype=torch.long)
                mask[row, :len(encoded[i])] = 1
            captured = {}

            def capture(layer_module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured["hidden"] = hidden
                raise StopForward()

            handle = layers[layer - 1].register_forward_hook(capture)
            with torch.no_grad(), self.using(BASE):
                try:
                    self.model.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
                except StopForward:
                    pass
                finally:
                    handle.remove()
            hidden = captured["hidden"].float().cpu()
            weights = mask.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            for row, i in enumerate(chunk):
                out[i] = pooled[row]
        return torch.stack(out)

    # ---- router

    def fit_router(self, texts_by_expert, layer=4, ridge=1.0, base_margin=0.0, holdout=0.0, seed=0, batch_size=8):
        """Fit the router. texts_by_expert needs "base" and one list per expert in the bank.

        holdout: fraction of each list kept out; the report has accuracy on it.
        Returns a report dict.
        """
        names = self.expert_names()
        for name in names:
            if name not in texts_by_expert or len(texts_by_expert[name]) == 0:
                raise ValueError(f"texts_by_expert has no texts for {name!r}")
        import random

        rng = random.Random(seed)
        train = {}
        held = {}
        for name in names:
            texts = list(texts_by_expert[name])
            rng.shuffle(texts)
            cut = int(len(texts) * holdout)
            held[name] = texts[:cut]
            train[name] = texts[cut:]
        features = []
        labels = []
        for name in names:
            if len(train[name]) == 0:
                continue
            features.append(self.features(train[name], layer, batch_size))
            labels += [name] * len(train[name])
        x = torch.cat(features)
        self.router = SequenceRouter.fit(names, layer, x, labels, ridge=ridge, base_margin=base_margin)
        report = {"layer": layer, "ridge": ridge, "base_margin": base_margin, "n_train": len(labels)}
        picked = self.router.pick(x)
        right = 0
        for guess, label in zip(picked, labels):
            if guess == label:
                right += 1
        report["train_accuracy"] = right / len(labels)
        total = 0
        right = 0
        per = {}
        for name in names:
            if len(held[name]) == 0:
                continue
            guesses = self.router.pick(self.features(held[name], layer, batch_size))
            hits = 0
            for guess in guesses:
                if guess == name:
                    hits += 1
            per[name] = hits / len(guesses)
            total += len(guesses)
            right += hits
        if total > 0:
            report["holdout_accuracy"] = right / total
            report["holdout_by_expert"] = per
        return report

    def route_many(self, prompts, batch_size=8):
        if self.router is None:
            raise RuntimeError("no router; call fit_router first")
        return self.router.pick(self.features(prompts, self.router.layer, batch_size))

    def route(self, prompt):
        """Return the expert name for one prompt."""
        return self.route_many([prompt])[0]

    # ---- running

    def pad_id(self):
        config = getattr(self.model, "generation_config", None)
        if config is not None and config.pad_token_id is not None:
            return config.pad_token_id
        return 0

    def rows_by_expert(self, input_ids, attention_mask, experts):
        batch = input_ids.shape[0]
        if experts is None:
            prompts = []
            for row in range(batch):
                if attention_mask is None:
                    prompts.append(input_ids[row])
                else:
                    prompts.append(input_ids[row][attention_mask[row].bool()])
            experts = self.route_many(prompts)
        elif isinstance(experts, str):
            experts = [experts] * batch
        groups = {}
        for row in range(batch):
            groups.setdefault(experts[row], []).append(row)
        return experts, groups

    def forward(self, input_ids, attention_mask=None, experts=None):
        """Logits (B, T, V). Rows are grouped by expert; experts is a name, a list, or None (route)."""
        experts, groups = self.rows_by_expert(input_ids, attention_mask, experts)
        out = None
        previous = self.active
        try:
            for name in groups:
                rows = groups[name]
                self.set_expert(name)
                sub_mask = None
                if attention_mask is not None:
                    sub_mask = attention_mask[rows]
                with torch.no_grad():
                    logits = self.model(input_ids=input_ids[rows], attention_mask=sub_mask).logits
                if out is None:
                    out = torch.zeros(input_ids.shape[0], *logits.shape[1:], dtype=logits.dtype, device=logits.device)
                out[rows] = logits
        finally:
            self.set_expert(previous)
        return out

    def generate(self, input_ids, attention_mask=None, experts=None, **kwargs):
        """Greedy or sampled generation. Route once per row from its prompt, then run with that delta.

        The expert is fixed for the whole sequence, so the KV cache stays valid.
        Rows with the same expert run together. Rows that end early are padded
        on the right with the pad id.
        """
        experts, groups = self.rows_by_expert(input_ids, attention_mask, experts)
        pieces = {}
        previous = self.active
        try:
            for name in groups:
                rows = groups[name]
                self.set_expert(name)
                sub_mask = None
                if attention_mask is not None:
                    sub_mask = attention_mask[rows]
                with torch.no_grad():
                    out = self.model.generate(input_ids=input_ids[rows], attention_mask=sub_mask, **kwargs)
                for index, row in enumerate(rows):
                    pieces[row] = out[index]
        finally:
            self.set_expert(previous)
        width = max(piece.shape[0] for piece in pieces.values())
        result = torch.full((input_ids.shape[0], width), self.pad_id(), dtype=input_ids.dtype,
                            device=input_ids.device)
        for row in pieces:
            result[row, :pieces[row].shape[0]] = pieces[row]
        return result

    # ---- counts

    def count_params(self, expert=None):
        """Stored params (host + every delta + router) and params run per token (host + active factors)."""
        host_total, host_run = host_param_counts(self.model)
        per_delta = {}
        for name in self.deltas:
            per_delta[name] = self.deltas[name].stored_params()
        router = 0
        if self.router is not None:
            router = self.router.numel()
        active = self.active if expert is None else expert
        extra = 0
        if active != BASE:
            extra = self.deltas[active].factor_params()
        return {"host": host_total, "host_run": host_run, "deltas": per_delta, "router": router,
                "stored": host_total + sum(per_delta.values()) + router,
                "per_token": host_run + extra, "active": active}

    # ---- save and load

    def save(self, path):
        """Write meta.json, one safetensors file per expert, and router.safetensors."""
        from safetensors.torch import save_file

        os.makedirs(path, exist_ok=True)
        meta = {"format": 1, "host_id": self.host_id, "experts": [], "router": None,
                "host_theta": self.host_theta}
        for name in self.deltas:
            delta = self.deltas[name]
            tensors = {}
            for module in delta.factors:
                for key in delta.factors[module]:
                    tensors[key + "|" + module] = delta.factors[module][key].contiguous().cpu()
            for key in delta.vectors:
                tensors["v|" + key] = delta.vectors[key].contiguous().cpu()
            file = "delta_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in name) + ".safetensors"
            save_file(tensors, os.path.join(path, file))
            meta["experts"].append({"name": name, "file": file, "source_id": delta.source_id, "rank": delta.rank,
                                    "scale": self.scales[name], "attrs": delta.attrs, "stats": delta.stats})
        if self.router is not None:
            save_file({"W": self.router.W.contiguous(), "mean": self.router.mean.contiguous(),
                       "std": self.router.std.contiguous()}, os.path.join(path, "router.safetensors"))
            meta["router"] = {"names": self.router.names, "layer": self.router.layer,
                              "base_margin": self.router.base_margin}
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)

    @classmethod
    def load(cls, path, host=None, tokenizer=None, dtype=None, device=None):
        """Load a saved network. host: a model, or None to load meta host_id from the hub."""
        from safetensors.torch import load_file

        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        if host is None or isinstance(host, str):
            import transformers

            host_id = host or meta["host_id"]
            host = transformers.AutoModelForCausalLM.from_pretrained(host_id, torch_dtype=dtype)
            if device is not None:
                host = host.to(device)
            if tokenizer is None:
                tokenizer = transformers.AutoTokenizer.from_pretrained(host_id)
        network = cls(host, tokenizer=tokenizer, host_id=meta.get("host_id"))
        for entry in meta["experts"]:
            tensors = load_file(os.path.join(path, entry["file"]))
            factors = {}
            vectors = {}
            for key in tensors:
                head, rest = key.split("|", 1)
                if head == "v":
                    vectors[rest] = tensors[key]
                else:
                    factors.setdefault(rest, {})[head] = tensors[key]
            delta = Delta(entry["name"], entry["source_id"], entry["rank"], factors, vectors, entry["attrs"],
                          entry["stats"], scale=entry["scale"])
            network.add_delta(delta)
        if meta["router"] is not None:
            router_tensors = load_file(os.path.join(path, "router.safetensors"))
            network.router = SequenceRouter(meta["router"]["names"], meta["router"]["layer"], router_tensors["W"],
                                            router_tensors["mean"], router_tensors["std"],
                                            meta["router"]["base_margin"])
        return network


class ExpertContext:
    """with network.using("math"): run with one expert, then restore the previous one."""

    def __init__(self, network, name):
        self.network = network
        self.name = name
        self.previous = None

    def __enter__(self):
        self.previous = self.network.active
        self.network.set_expert(self.name)
        return self.network

    def __exit__(self, *exc):
        self.network.set_expert(self.previous)
        return False


# ---------------------------------------------------------------- merge


def merge(network, expert):
    """Return a plain model copy with one delta folded in (the host for "base").

    A copy of the host is made with the hooks removed. Low-rank deltas add
    scale * B A; dense deltas add scale * D. Vectors and the rope base are set
    to the specialist's. With rank "full" this equals the specialist up to
    float rounding of W_host + (W_spec - W_host).
    """
    previous = network.active
    network.set_expert(BASE)
    model = copy.deepcopy(network.model)
    network.set_expert(previous)
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_hooks_with_kwargs.clear()
        module._forward_hooks_always_called.clear()
    if expert == BASE:
        return model
    delta = network.deltas[expert]
    scale = network.scales[expert]
    with torch.no_grad():
        for module in delta.factors:
            param = model.get_parameter(module + ".weight")
            parts = delta.factors[module]
            if "D" in parts:
                change = parts["D"].float()
            else:
                change = parts["B"].float() @ parts["A"].float()
            param.copy_((param.float() + scale * change.to(param.device)).to(param.dtype))
        for key in delta.vectors:
            param = model.get_parameter(key)
            param.copy_(delta.vectors[key].to(device=param.device, dtype=param.dtype))
    if delta.attrs.get("rope_theta") is not None and rope_theta_of(model.config) != delta.attrs["rope_theta"]:
        set_rope_theta(model, float(delta.attrs["rope_theta"]))
    return model
