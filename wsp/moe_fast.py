"""Fast frozen Qwen3-MoE donor: a batched expert forward and a direct safetensors loader.

transformers 4.57 runs a Python loop over the experts that a batch hits. That loop
is CPU bound (about 4 kernel launches and one host sync per expert). FastMoeBlock
stacks the experts of a layer into three tensors and runs all of them with batched
matmuls: tokens are sorted by expert, written into an (experts, capacity, d) buffer,
and pushed through bmm. The result matches the native block (tests/test_moe_fast.py).

load_qwen3_moe builds the model on the meta device with FastMoeBlock in place of the
native block (so no per-expert module is ever created), allocates it on the target
device, and reads each tensor from the safetensors shards with safe_open. Only the
shards that hold the wanted tensors are touched.
"""

import contextlib
import json
import os
import time

import torch
import torch.nn.functional as F
from torch import nn


def _module():
    import torch._dynamo  # noqa: F401  (import before any meta context, see LESSONS.md)
    from transformers.models.qwen3_moe import modeling_qwen3_moe

    return modeling_qwen3_moe


class FastMoeBlock(_module().Qwen3MoeSparseMoeBlock):
    """Qwen3MoeSparseMoeBlock with the experts stacked: w_gate, w_up (E, I, D) and w_down (E, D, I)."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        d, i, e = config.hidden_size, config.moe_intermediate_size, config.num_experts
        self.gate = nn.Linear(d, e, bias=False)
        self.w_gate = nn.Parameter(torch.empty(e, i, d), requires_grad=False)
        self.w_up = nn.Parameter(torch.empty(e, i, d), requires_grad=False)
        self.w_down = nn.Parameter(torch.empty(e, d, i), requires_grad=False)
        self.act_fn = F.silu if config.hidden_act == "silu" else _act(config.hidden_act)
        self.cap_factor = 2.0  # padded capacity per expert, in multiples of the mean load
        self.ckpt = False  # recompute the expert pass in backward to save memory

    def route(self, flat):
        logits = self.gate(flat)
        weights = F.softmax(logits, dim=1, dtype=torch.float)
        weights, chosen = torch.topk(weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.to(flat.dtype), chosen

    def forward(self, hidden_states):
        batch, seq, dim = hidden_states.shape
        flat = hidden_states.reshape(-1, dim)
        logits, weights, chosen = self.route(flat)
        if self.ckpt and torch.is_grad_enabled() and flat.requires_grad:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(self.experts_forward, flat, weights, chosen, use_reentrant=False)
        else:
            out = self.experts_forward(flat, weights, chosen)
        return out.reshape(batch, seq, dim), logits

    def experts_forward(self, flat, weights, chosen):
        """Sorted dispatch. Each expert's first `capacity` tokens go through padded bmm. A crowded expert's overflow
        (padding tokens and sink tokens pile onto a few experts) runs in a per-expert loop, so the padded buffer
        stays near the mean load instead of the maximum."""
        tokens, dim = flat.shape
        k = chosen.shape[1]
        experts = self.num_experts
        expert_of = chosen.reshape(-1)
        order = torch.argsort(expert_of, stable=True)
        sorted_expert = expert_of[order]
        token_of = order // k
        counts = torch.bincount(expert_of, minlength=experts)
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(order.numel(), device=flat.device) - starts[sorted_expert]
        weight_of = weights.reshape(-1)[order].unsqueeze(-1)
        counts_list = counts.tolist()
        capacity = max(1, min(max(counts_list), int(self.cap_factor * order.numel() / experts) + 1))
        main = slot < capacity
        e_main, s_main, t_main = sorted_expert[main], slot[main], token_of[main]
        buf = flat.new_zeros(experts, capacity, dim)
        buf[e_main, s_main] = flat[t_main]
        act = self.act_fn(torch.bmm(buf, self.w_gate.transpose(1, 2).to(buf.dtype)))
        act = act * torch.bmm(buf, self.w_up.transpose(1, 2).to(buf.dtype))
        out = torch.bmm(act, self.w_down.transpose(1, 2).to(buf.dtype))
        picked = out[e_main, s_main] * weight_of[main]
        result = flat.new_zeros(tokens, dim)
        result.index_add_(0, t_main, picked.to(flat.dtype))
        starts_list = starts.tolist()
        for e in range(experts):
            if counts_list[e] <= capacity:
                continue
            lo, hi = starts_list[e] + capacity, starts_list[e] + counts_list[e]
            x = flat[token_of[lo:hi]]
            h = self.act_fn(x @ self.w_gate[e].t().to(x.dtype)) * (x @ self.w_up[e].t().to(x.dtype))
            y = (h @ self.w_down[e].t().to(x.dtype)) * weight_of[lo:hi]
            result.index_add_(0, token_of[lo:hi], y.to(flat.dtype))
        return result


def _act(name):
    from transformers.activations import ACT2FN

    return ACT2FN[name]


def stack_native(block):
    """Copy the weights of a native Qwen3MoeSparseMoeBlock into a new FastMoeBlock on the same device and dtype."""
    cfg = block.experts[0].config
    cfg = _block_config(block, cfg)
    fast = FastMoeBlock(cfg)
    dtype = block.gate.weight.dtype
    device = block.gate.weight.device
    fast.to(device=device, dtype=dtype)
    with torch.no_grad():
        fast.gate.weight.copy_(block.gate.weight)
        for e, expert in enumerate(block.experts):
            fast.w_gate[e].copy_(expert.gate_proj.weight)
            fast.w_up[e].copy_(expert.up_proj.weight)
            fast.w_down[e].copy_(expert.down_proj.weight)
    for p in fast.parameters():
        p.requires_grad_(False)
    return fast


def _block_config(block, expert_config):
    import copy

    cfg = copy.copy(expert_config)
    cfg.num_experts = block.num_experts
    cfg.num_experts_per_tok = block.top_k
    cfg.norm_topk_prob = block.norm_topk_prob
    cfg.moe_intermediate_size = block.experts[0].intermediate_size
    return cfg


def swap_native_blocks(model):
    """Replace every native sparse block of a loaded model with a FastMoeBlock (same numbers, batched forward)."""
    native = _module().Qwen3MoeSparseMoeBlock
    count = 0
    for layer in model.model.layers:
        if type(layer.mlp) is native:
            layer.mlp = stack_native(layer.mlp)
            count += 1
    return count


@contextlib.contextmanager
def fast_blocks_in_constructor():
    """While active, Qwen3MoeDecoderLayer builds FastMoeBlock instead of the native block."""
    mod = _module()
    saved = mod.Qwen3MoeSparseMoeBlock
    mod.Qwen3MoeSparseMoeBlock = FastMoeBlock
    try:
        yield
    finally:
        mod.Qwen3MoeSparseMoeBlock = saved


def weight_map(snapshot):
    """{tensor name: shard file name} from model.safetensors.index.json, or from the single shard."""
    index = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    from safetensors import safe_open

    name = "model.safetensors"
    with safe_open(os.path.join(snapshot, name), "pt") as f:
        return {k: name for k in f.keys()}


def wanted(name, layers, extras):
    """Is checkpoint tensor `name` needed for decoder layers `layers` (None means all) plus the extras?"""
    if name.startswith("model.layers."):
        if layers is None:
            return True
        return int(name.split(".")[2]) in layers
    return extras


def shards_needed(snapshot, layers=None, extras=True):
    wmap = weight_map(snapshot)
    return sorted({shard for name, shard in wmap.items() if wanted(name, layers, extras)})


def copy_shards(snapshot, dest, shards, workers=8):
    """Copy shard files (plus small config files) to a local directory with large sequential reads."""
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(snapshot):
        if not name.endswith(".safetensors") and os.path.isfile(os.path.join(snapshot, name)):
            shutil.copyfile(os.path.join(snapshot, name), os.path.join(dest, name))

    def one(shard):
        src, dst = os.path.join(snapshot, shard), os.path.join(dest, shard)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return os.path.getsize(dst)
        with open(src, "rb") as a, open(dst + ".tmp", "wb") as b:
            shutil.copyfileobj(a, b, 64 << 20)
        os.replace(dst + ".tmp", dst)
        return os.path.getsize(dst)

    with ThreadPoolExecutor(workers) as pool:
        return sum(pool.map(one, shards))


class Skipped(nn.Module):
    """Stands where an unloaded decoder layer would be. Not a valid forward; for slice-only loads."""

    def forward(self, hidden_states, *args, **kwargs):
        raise RuntimeError("this decoder layer was not loaded")


def load_qwen3_moe(snapshot, device, dtype=torch.bfloat16, layers=None, head=True, log=print):
    """Qwen3MoeForCausalLM with FastMoeBlock, loaded straight from the shards in `snapshot` onto `device`.

    layers: iterable of decoder layer indices to load (others become Skipped), or None for all.
    head: also load the embeddings, final norm and LM head. Returns (model, stats).
    """
    import transformers

    mod = _module()
    layers = None if layers is None else set(int(i) for i in layers)
    config = transformers.AutoConfig.from_pretrained(snapshot)
    config._attn_implementation = "eager"
    t0 = time.time()
    with torch.device("meta"):
        with fast_blocks_in_constructor():
            model = mod.Qwen3MoeForCausalLM(config)
    model = model.to(dtype)
    if layers is not None:
        for i in range(config.num_hidden_layers):
            if i not in layers:
                model.model.layers[i] = Skipped()
    model.to_empty(device=device)
    model.model.rotary_emb = mod.Qwen3MoeRotaryEmbedding(config=config, device=device)
    names = dict(model.named_parameters())
    names.update(dict(model.named_buffers()))
    filled = set()
    wmap = weight_map(snapshot)
    by_shard = {}
    for name, shard in wmap.items():
        if wanted(name, layers, head):
            by_shard.setdefault(shard, []).append(name)
    from safetensors import safe_open

    nbytes = 0
    stacked = ("gate_proj", "up_proj", "down_proj")
    target = {"gate_proj": "w_gate", "up_proj": "w_up", "down_proj": "w_down"}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(os.path.join(snapshot, shard), "pt", device=str(device)) as f:
            for key in keys:
                tensor = f.get_tensor(key)
                nbytes += tensor.numel() * tensor.element_size()
                parts = key.split(".")
                if "experts" in parts:
                    j = parts.index("experts")
                    expert, proj = int(parts[j + 1]), parts[j + 2]
                    assert proj in stacked, key
                    param = names[".".join(parts[:j] + [target[proj]])]
                    param.data[expert].copy_(tensor)
                    filled.add((".".join(parts[:j] + [target[proj]]), expert))
                    continue
                if key == "lm_head.weight" and key not in names:
                    continue
                names[key].data.copy_(tensor.to(dtype))
                filled.add(key)
        log(f"loaded shard {shard} {round(time.time() - t0, 1)}s")
    expect = {n for n in names if _expected(n, layers, head, config)}
    got = {k if isinstance(k, str) else k[0] for k in filled}
    missing = sorted(expect - got)
    if missing:
        raise RuntimeError(f"{len(missing)} tensors not loaded, first {missing[:3]}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    if config.tie_word_embeddings:
        model.tie_weights()
    return model, {"seconds": round(time.time() - t0, 1), "bytes": nbytes, "shards": len(by_shard)}


def _expected(name, layers, head, config):
    if name.startswith("model.layers."):
        return layers is None or int(name.split(".")[2]) in layers
    if name.endswith("inv_freq"):
        return False
    if name == "lm_head.weight" and config.tie_word_embeddings:
        return False
    return head


def block_part(donor, first, last, windows=None):
    """A BlockPart that runs donor layers first..last in place (no copy: the part shares the donor's frozen modules)."""
    from wsp import parts

    layers = [donor.model.layers[i] for i in range(first, last + 1)]
    cfg = donor.config
    wins = [parts._layer_window(layer, cfg) for layer in layers]
    if hasattr(parts, "_layer_type"):
        types = [parts._layer_type(cfg, first + o) for o in range(len(layers))]
        part = parts.BlockPart(layers, donor.model.rotary_emb, wins, types, first)
    else:
        # Transformers 4.x copy of wsp/parts.py: (layers, rotary, local_rotary, windows)
        part = parts.BlockPart(layers, donor.model.rotary_emb, None, wins)
    part.eval()
    for p in part.parameters():
        p.requires_grad_(False)
    return part


def active_params(part, top_k, num_experts):
    """Params run per token in a donor slice: dense tensors in full, stacked routed experts at top_k / num_experts."""
    total, stored = 0.0, 0
    seen = set()
    for name, p in part.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        stored += p.numel()
        routed = name.split(".")[-1] in ("w_gate", "w_up", "w_down")
        total += p.numel() * (top_k / num_experts if routed else 1.0)
    return int(total), stored


def set_checkpoint(model_or_part, on=True):
    for m in model_or_part.modules():
        if isinstance(m, FastMoeBlock):
            m.ckpt = on


def fast_grafted_final(donor, layer, x, mask):
    """Same result as pgraft._grafted_final, but runs only donor layers layer..end on x (no work on zero embeddings)."""
    last = len(donor.model.layers) - 1
    cache = getattr(donor, "_tail_parts", None)
    if cache is None:
        cache = donor._tail_parts = {}
    if layer not in cache:
        cache[layer] = block_part(donor, layer, last)
    part = cache[layer]
    from wsp.stitch import autocast

    x = x.to(donor.model.norm.weight.dtype)
    with autocast(x.device):
        out = x + part(x, attention_mask=mask)
        return donor.model.norm(out).to(x.dtype)


def install_fast_alignment():
    """Make wsp.pgraft use fast_grafted_final in stage 1 and in the functional regularizer."""
    from wsp import pgraft

    pgraft._grafted_final = fast_grafted_final


def scramble_part(part, seed, mode="spectrum"):
    """Scramble every weight of a donor slice in place (wsp.scramble semantics), expert by expert for the stacked tensors.

    Each routed expert matrix keeps its shape, norm and spectrum and loses its learned structure. Returns a report.
    """
    from wsp.scramble import scramble_tensor, tensor_seed

    t0 = time.time()
    changed, worst = 0, 0.0
    seen = set()
    for name, param in part.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        with torch.no_grad():
            if param.dim() == 3:
                slices = [param[e] for e in range(param.shape[0])]
                keys = [f"{name}.{e}" for e in range(param.shape[0])]
            else:
                slices, keys = [param], [name]
            for sl, key in zip(slices, keys):
                old = sl.detach().clone()
                new, _ = scramble_tensor(old, tensor_seed(seed, key), mode, None, "permute")
                gap = (new.float().norm() - old.float().norm()).abs() / (old.float().norm() + 1e-12)
                worst = max(worst, float(gap))
                sl.copy_(new)
                changed += 1
    return {"mode": mode, "tensors_changed": changed, "rel_fro_error": worst, "seconds": round(time.time() - t0, 1)}
