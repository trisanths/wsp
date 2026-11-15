"""Layer-streaming expert profiling for giant open MoE models.

No machine holds a whole giant, but profiling needs only one decoder layer at
a time. The loop is:

    for each layer L:
        fetch the tensors of layer L by byte range
        build the layer from the config with the Hugging Face class
        run the cached hidden states of every calibration text through it
        record which experts the router picks, per text set
        keep the output hidden states for layer L + 1, free the layer

The result has the format of wsp.atlas.expert_skill_map, so atlas entries
and catalog rows work unchanged.

How a layer sees the right inputs. The model's own forward builds the masks
and position embeddings. We build the model on the meta device, replace every
decoder layer with a stub, and call the model's forward with each text. Stub k
before the active layer passes its input through. The active stub injects the
cached hidden state, runs the real layer, stores the output and stops the
forward. This works for any model that loops over `model.layers`, and it keeps
the masks and rotary code of the real class.

Stored quantization. The giants ship quantized tensors. Layouts decode them
to the compute dtype while loading: FP8 with 128x128 block scales, FP4 (two
values per byte) with one scale per 32 values, and compressed-tensors mxfp4
with the same block shape. Decoding runs on the device that runs the layer.
"""

import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
import time

import torch

from wsp import atlas
from wsp import fetch
from wsp.parts import find_decoder_layers

# ---------------------------------------------------------------------------
# Raw tensor reading. wsp.fetch has no 8-bit float types, so decode here.

RAW_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}

ITEM_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}


def decode_bytes(data, dtype_name, shape):
    """Build a tensor from the raw bytes of one safetensors entry."""
    if dtype_name not in RAW_TYPES:
        raise ValueError(f"unsupported safetensors dtype {dtype_name}")
    dtype = RAW_TYPES[dtype_name]
    count = 1
    for size in shape:
        count *= size
    if count == 0:
        return torch.zeros(shape, dtype=dtype)
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return raw.view(dtype).reshape(shape).clone()


def entry_bytes(entry):
    begin, end = entry["data_offsets"]
    return end - begin


def plan_ranges(spans, gap=1 << 20, cap=1 << 29):
    """Merge sorted (start, end, key) spans into reads.

    Two spans join when the hole between them is at most `gap` bytes and the
    merged read stays under `cap` bytes. Returns [(start, end, [span, ...])].
    """
    spans = sorted(spans)
    groups = []
    for span in spans:
        if groups:
            start, end, members = groups[-1]
            if span[0] - end <= gap and span[1] - start <= cap:
                members.append(span)
                groups[-1] = (start, max(end, span[1]), members)
                continue
        groups.append((span[0], span[1], [span]))
    return groups


class Source:
    """Safetensors tensors in a folder or repo, read by byte range.

    Subclasses give weight_map() (tensor name to shard file name) and
    range_reader(shard). bytes_read and requests count the traffic.
    """

    def __init__(self):
        self._headers = {}
        self._readers = {}
        self._stash = {}
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.requests = 0
        self.gap = 1 << 20
        self._prefetch = None

    def weight_map(self):
        raise NotImplementedError

    def range_reader(self, shard):
        raise NotImplementedError

    def reader(self, shard):
        with self._lock:
            if shard not in self._readers:
                self._readers[shard] = self.range_reader(shard)
            return self._readers[shard]

    def header(self, shard):
        """(header dict, data start) of one shard, read once."""
        with self._lock:
            known = self._headers.get(shard)
        if known is not None:
            return known
        value = fetch.read_header(self.reader(shard))
        with self._lock:
            self._headers[shard] = value
        return value

    def has(self, name):
        return name in self.weight_map()

    def preload_headers(self, workers=6):
        """Read the header of every shard, in parallel. One small range request per shard."""
        shards = sorted(set(self.weight_map().values()))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(self.header, shards))

    def entry(self, name):
        """(shard, header entry, data start) of one tensor."""
        weight_map = self.weight_map()
        if name not in weight_map:
            raise KeyError(f"tensor not found in the weight map: {name}")
        shard = weight_map[name]
        header, start = self.header(shard)
        if name not in header:
            raise KeyError(f"tensor {name} is not in the header of {shard}")
        return shard, header[name], start

    def nbytes(self, name):
        return entry_bytes(self.entry(name)[1])

    def shape(self, name):
        return list(self.entry(name)[1]["shape"])

    def dtype_name(self, name):
        return self.entry(name)[1]["dtype"]

    def _read_span(self, shard, begin, end):
        data = self.reader(shard)(begin, end - begin)
        with self._lock:
            self.bytes_read += end - begin
            self.requests += 1
        return data

    def read(self, names, workers=8, gap=None):
        """Read whole tensors by name. Returns {name: tensor} in the stored dtype.

        Reads whose gap is at most `gap` bytes (default self.gap) are merged into one request.
        """
        if gap is None:
            gap = self.gap
        out = {}
        wanted = []
        with self._lock:
            for name in names:
                if name in self._stash:
                    out[name] = self._stash.pop(name)
                else:
                    wanted.append(name)
        by_shard = {}
        for name in wanted:
            shard, entry, start = self.entry(name)
            begin, end = entry["data_offsets"]
            by_shard.setdefault(shard, []).append((start + begin, start + end, name))
        jobs = []
        for shard, spans in by_shard.items():
            for group in plan_ranges(spans, gap=gap):
                jobs.append((shard, group))

        def run(job):
            shard, group = job
            data = self._read_span(shard, group[0], group[1])
            pieces = {}
            for begin, end, name in group[2]:
                entry = self.entry(name)[1]
                chunk = data[begin - group[0]:end - group[0]]
                pieces[name] = decode_bytes(chunk, entry["dtype"], entry["shape"])
            return pieces

        if workers <= 1 or len(jobs) <= 1:
            results = [run(job) for job in jobs]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run, jobs))
        for pieces in results:
            out.update(pieces)
        return out

    def read_rows(self, name, rows, workers=8, gap=0):
        """Read rows of the first dimension of one tensor: shape (len(rows), ...)."""
        shard, entry, start = self.entry(name)
        shape = list(entry["shape"])
        row_count = 1
        for size in shape[1:]:
            row_count *= size
        row_bytes = row_count * ITEM_BYTES[entry["dtype"]]
        base = start + entry["data_offsets"][0]
        spans = []
        for row in rows:
            if row < 0 or row >= shape[0]:
                raise IndexError(f"row {row} is outside {name} with {shape[0]} rows")
            spans.append((base + row * row_bytes, base + (row + 1) * row_bytes, row))
        pieces = {}
        for group in plan_ranges(spans, gap=gap):
            data = self._read_span(shard, group[0], group[1])
            for begin, end, row in group[2]:
                pieces[row] = decode_bytes(data[begin - group[0]:end - group[0]], entry["dtype"], shape[1:])
        return torch.stack([pieces[row] for row in rows])

    def prefetch(self, names, workers=8):
        """Start reading names in a background thread. read() takes them from the stash."""
        names = list(names)

        def work():
            got = self.read(names, workers=workers)
            with self._lock:
                self._stash.update(got)

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        self._prefetch = thread
        return thread

    def wait_prefetch(self):
        if self._prefetch is not None:
            self._prefetch.join()
            self._prefetch = None


class LocalSource(Source):
    """Shards in a folder, with or without model.safetensors.index.json."""

    def __init__(self, folder):
        Source.__init__(self)
        self.folder = folder
        index = os.path.join(folder, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self._map = json.load(f)["weight_map"]
        else:
            path = os.path.join(folder, "model.safetensors")
            header, _ = fetch.read_header(fetch.local_range_reader(path))
            self._map = {}
            for name in header:
                self._map[name] = "model.safetensors"

    def weight_map(self):
        return self._map

    def range_reader(self, shard):
        return fetch.local_range_reader(os.path.join(self.folder, shard))


def with_retries(read_range, tries=6, wait=2.0, longest=60.0):
    """Retry a range read with doubling waits. The hub answers 429 when requests come too fast."""

    def read(offset, length):
        last = None
        for attempt in range(tries):
            try:
                return read_range(offset, length)
            except Exception as error:
                last = error
                time.sleep(min(longest, wait * (2 ** attempt)))
        raise last

    return read


class HubSource(Source):
    """A Hugging Face repo, read by HTTP range requests. Nothing is stored on disk."""

    def __init__(self, repo_id, revision="main", token=None):
        Source.__init__(self)
        self.repo_id = repo_id
        self.revision = revision
        self.token = token
        self._map = None

    def weight_map(self):
        if self._map is None:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError

            try:
                path = hf_hub_download(self.repo_id, "model.safetensors.index.json", revision=self.revision)
            except EntryNotFoundError:
                path = None
            if path is None:
                url = self._url("model.safetensors")
                header, _ = fetch.read_header(with_retries(fetch.http_range_reader(url, token=self._token())))
                self._map = {name: "model.safetensors" for name in header}
            else:
                with open(path) as f:
                    self._map = json.load(f)["weight_map"]
        return self._map

    def _token(self):
        if self.token is not None:
            return self.token
        from huggingface_hub import get_token

        return get_token()

    def _url(self, shard):
        from huggingface_hub import hf_hub_url

        return hf_hub_url(self.repo_id, shard, revision=self.revision)

    def range_reader(self, shard):
        return with_retries(fetch.http_range_reader(self._url(shard), token=self._token()))


# ---------------------------------------------------------------------------
# Decoding of stored quantization.

FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def scale_to_float(scale):
    """Float32 scale from an e8m0 tensor, a uint8 exponent (bias 127) or a float tensor."""
    if scale.dtype == torch.uint8:
        return torch.exp2(scale.float() - 127.0)
    return scale.float()


def dequant_block_fp8(weight, scale, block=128, dtype=torch.bfloat16):
    """FP8 weight (m, n) with one scale per block x block tile: scale shape (ceil(m/b), ceil(n/b))."""
    rows, cols = weight.shape
    factor = scale_to_float(scale)
    if factor.shape != (math.ceil(rows / block), math.ceil(cols / block)):
        raise ValueError(f"scale {tuple(factor.shape)} does not fit weight {tuple(weight.shape)} at block {block}")
    factor = factor.repeat_interleave(block, dim=0)[:rows].repeat_interleave(block, dim=1)[:, :cols]
    return (weight.float() * factor).to(dtype)


def unpack_fp4(packed):
    """(m, n/2) bytes to (m, n) float32 e2m1 values. The low nibble is the even column."""
    raw = packed.view(torch.uint8) if packed.dtype == torch.int8 else packed
    table = torch.tensor(FP4_TABLE, dtype=torch.float32, device=raw.device)
    low = table[(raw & 0x0F).long()]
    high = table[(raw >> 4).long()]
    return torch.stack([low, high], dim=-1).reshape(raw.shape[0], raw.shape[1] * 2)


def dequant_fp4(packed, scale, block=32, dtype=torch.bfloat16):
    """FP4 weight with one scale per `block` values along the input dimension."""
    values = unpack_fp4(packed)
    factor = scale_to_float(scale)
    if factor.shape != (values.shape[0], values.shape[1] // block):
        raise ValueError(f"scale {tuple(factor.shape)} does not fit fp4 weight {tuple(values.shape)} at block {block}")
    factor = factor.repeat_interleave(block, dim=1)
    return (values * factor).to(dtype)


def pack_fp4(values):
    """Inverse of unpack_fp4 for values that are exactly e2m1. For tests."""
    table = torch.tensor(FP4_TABLE, dtype=torch.float32)
    flat = values.reshape(-1, 1)
    codes = (flat - table.reshape(1, -1)).abs().argmin(dim=1)
    codes = codes.reshape(values.shape)
    low = codes[:, 0::2]
    high = codes[:, 1::2]
    return (low | (high << 4)).to(torch.uint8)


# ---------------------------------------------------------------------------
# Layouts: how a family names and stores the tensors of one layer.


class Load:
    """One tensor to write into the layer: target key, source name, and where it goes."""

    def __init__(self, target, source, index=None, rows=None):
        self.target = target
        self.source = source
        self.index = index
        self.rows = rows


class Layout:
    """Checkpoint naming for one family. The base class is the plain Hugging Face layout.

    A plain layout has the same key names as the module state dict, apart
    from a prefix, and its experts are ordinary keys. Other layouts rename
    keys, decode stored quantization, and write experts one by one.
    """

    family = "hf"
    # Matches "<prefix>experts.<e>.<part>.<kind>" names that need special loading. None for plain layouts.
    expert_re = None
    expert_parts = {}
    fused_experts = False
    fused_targets = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
    list_target = "mlp.experts.{e}.{part}.weight"
    router_weight = "mlp.gate.weight"
    router_bias = "mlp.gate.e_score_correction_bias"
    fp32_names = ("e_score_correction_bias",)
    # Layer-relative keys that the loader may leave unset. Regexes.
    optional = ()

    def __init__(self, prefix="model.layers.{layer}.", embed="model.embed_tokens.weight"):
        self.prefix_format = prefix
        self.embed = embed
        self._by_layer = {}

    # Names.

    def prefix(self, layer):
        return self.prefix_format.format(layer=layer)

    def embed_name(self):
        return self.embed

    def layer_names(self, source, layer):
        """Every tensor name of one layer, from the weight map."""
        key = id(source)
        if key not in self._by_layer:
            head, tail = self.prefix_format.split("{layer}")
            pattern = re.compile("^" + re.escape(head) + r"(\d+)" + re.escape(tail))
            table = {}
            for name in source.weight_map():
                match = pattern.match(name)
                if match is not None:
                    table.setdefault(int(match.group(1)), []).append(name)
            self._by_layer[key] = table
        return sorted(self._by_layer[key].get(layer, []))

    def translate(self, name, layer):
        """Layer-relative key in the module state dict for a checkpoint name, or None to drop."""
        return name[len(self.prefix(layer)):]

    def scale_for(self, source, name):
        """Name of the scale tensor that belongs to a weight, or None."""
        return None

    def is_expert(self, name):
        return self.expert_re is not None and self.expert_re.match(name) is not None

    # Decoding.

    def decode(self, source, name, tensor, scale, dtype, device):
        if tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    def combine(self, state):
        """Merge tensors that the module keeps in one parameter. Default: nothing."""
        return state

    # Expert loading for layouts with expert_re.

    def expert_loads(self, source, layer):
        """([Load] for every expert tensor of one layer, {name: scale name or None})."""
        return [], {}

    # Hidden state conventions. The default is one tensor.

    def state_of(self, output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def inject(self, args, kwargs, state):
        if state is None:
            return args, kwargs
        if args:
            return (state,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = state
        return args, kwargs

    def passthrough(self, args, kwargs, width):
        hidden = args[0] if args else kwargs["hidden_states"]
        if width is None:
            return hidden
        return (hidden,) + (None,) * (width - 1)


class RenamedLayout(Layout):
    """A layout with ordered regex renames from checkpoint names to module keys."""

    renames = ()

    def translate(self, name, layer):
        key = name
        for pattern, replacement in self.renames:
            key = re.sub(pattern, replacement, key)
        return key[len(self.module_prefix(layer)):]

    def module_prefix(self, layer):
        return self.prefix(layer)


def sibling_scale(source, name, suffixes):
    """First existing name among name with ".weight" replaced by each suffix."""
    if not name.endswith(".weight"):
        return None
    stem = name[:-len(".weight")]
    for suffix in suffixes:
        candidate = stem + suffix
        if source.has(candidate):
            return candidate
    return None


class FusedExpertLayout(RenamedLayout):
    """Per-expert checkpoint tensors that the transformers module keeps fused: gate_up_proj (E, 2I, H), down_proj (E, H, I)."""

    fused_experts = True

    def expert_loads(self, source, layer):
        loads = []
        scales = {}
        for name in self.layer_names(source, layer):
            match = self.expert_re.match(name)
            if match is None or not name.endswith(".weight"):
                continue
            part = self.expert_parts[match.group(3)]
            scales[name] = self.scale_for(source, name)
            target = self.fused_targets[1] if part == "down" else self.fused_targets[0]
            loads.append(Load(target, name, index=int(match.group(2)), rows=part))
        return loads, scales

    def decode(self, source, name, tensor, scale, dtype, device):
        tensor = tensor.to(device)
        if scale is None:
            if tensor.is_floating_point() and tensor.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                return tensor.to(dtype=dtype)
            return tensor
        scale = scale.to(device)
        if tensor.dtype == torch.float8_e4m3fn:
            return dequant_block_fp8(tensor, scale, 128, dtype)
        if tensor.dtype == torch.int8:
            return dequant_fp4(tensor, scale, 32, dtype)
        raise ValueError(f"{name}: stored {tensor.dtype} with a scale has no decoder")


class DeepSeekV4Layout(FusedExpertLayout):
    """DeepSeek-V4 Flash and Pro: native names, FP8 dense weights, FP4 experts, hash routing in layers 0 to 2.

    Dense weights are F8_E4M3 with a `.scale` tensor of 128x128 blocks. Experts are
    I8 (two FP4 values per byte) with a `.scale` of one e8m0 value per 32 inputs.
    """

    family = "deepseek_v4"
    expert_re = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.(weight|scale)$")
    expert_parts = {"w1": "gate", "w3": "up", "w2": "down"}
    # The rename order is the order of transformers 5.23.0 conversion_mapping["deepseek_v4"].
    renames = (
        (r"\.attn\.", ".self_attn."),
        (r"\.ffn\.", ".mlp."),
        (r"\.indexer\.compressor\.", ".compressor.indexer."),
        (r"\.attn_norm\.", ".input_layernorm."),
        (r"\.ffn_norm\.", ".post_attention_layernorm."),
        (r"\.hc_attn_fn", ".attn_hc.fn"),
        (r"\.hc_attn_base", ".attn_hc.base"),
        (r"\.hc_attn_scale", ".attn_hc.scale"),
        (r"\.hc_ffn_fn", ".ffn_hc.fn"),
        (r"\.hc_ffn_base", ".ffn_hc.base"),
        (r"\.hc_ffn_scale", ".ffn_hc.scale"),
        (r"\.attn_sink", ".sinks"),
        (r"\.indexer\.weights_proj\.", ".compressor.indexer.scorer.weights_proj."),
        (r"\.indexer\.wq_b\.", ".compressor.indexer.q_b_proj."),
        (r"\.norm\.", ".kv_norm."),
        (r"\.ape", ".position_bias"),
        (r"\.wq_a\.", ".q_a_proj."),
        (r"\.self_attn\.wq_b\.", ".self_attn.q_b_proj."),
        (r"\.wkv\.", ".kv_proj."),
        (r"\.wgate\.", ".gate_proj."),
        (r"\.wo_a\.", ".o_a_proj."),
        (r"\.wo_b\.", ".o_b_proj."),
        (r"\.q_norm\.", ".q_a_norm."),
        (r"\.gate\.bias", ".gate.e_score_correction_bias"),
        (r"\.shared_experts\.w1\.", ".shared_experts.gate_proj."),
        (r"\.shared_experts\.w2\.", ".shared_experts.down_proj."),
        (r"\.shared_experts\.w3\.", ".shared_experts.up_proj."),
    )

    def __init__(self):
        RenamedLayout.__init__(self, prefix="layers.{layer}.", embed="embed.weight")

    def scale_for(self, source, name):
        return sibling_scale(source, name, [".scale"])

    def is_scale(self, name):
        return name.endswith(".scale")


class GlmLayout(FusedExpertLayout):
    """GLM-5.3-Flash: names under model.language_model, FP8 weights with `weight_scale_inv`, fused experts in transformers."""

    family = "glm5_next"
    expert_re = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$")
    expert_parts = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
    renames = (
        (r"self_attn\.f_a_proj\.", "self_attn.forget_gate.f_a_proj."),
        (r"self_attn\.f_b_proj\.", "self_attn.forget_gate.f_b_proj."),
        (r"self_attn\.dt_bias", "self_attn.forget_gate.dt_bias"),
        (r"self_attn\.A_log", "self_attn.forget_gate.A_log"),
        ("hc_attn_fn", "attn_hc.fn"),
        ("hc_attn_base", "attn_hc.base"),
        ("hc_attn_scale", "attn_hc.scale"),
        ("hc_ffn_fn", "ffn_hc.fn"),
        ("hc_ffn_base", "ffn_hc.base"),
        ("hc_ffn_scale", "ffn_hc.scale"),
    )

    def __init__(self):
        RenamedLayout.__init__(self, prefix="model.language_model.layers.{layer}.",
                               embed="model.language_model.embed_tokens.weight")

    def scale_for(self, source, name):
        return sibling_scale(source, name, [".weight_scale_inv"])

    def is_scale(self, name):
        return name.endswith(".weight_scale_inv")

    def combine(self, state):
        """The module has one conv1d for q, k and v. The checkpoint has three."""
        names = ["self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight"]
        if all(name in state for name in names):
            state["self_attn.conv1d.weight"] = torch.cat([state.pop(name) for name in names], dim=0)
        return state



class KimiLayout(Layout):
    """Kimi K3 and K3.1: remote code, compressed-tensors mxfp4 experts, latent MoE, block attention residuals.

    Names match the module state dict. Experts are `w1` (gate), `w2` (down), `w3` (up)
    stored as `weight_packed` (uint8, two FP4 values per byte) plus `weight_scale`
    (uint8 e8m0, one per 32 inputs). The decoder layer takes and returns a pair:
    the running hidden state and the block residual list.
    """

    family = "kimi_k3"
    expert_re = re.compile(
        r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.(weight_packed|weight_scale)$")
    expert_parts = {"w1": "gate", "w3": "up", "w2": "down"}
    fused_experts = False
    list_target = "block_sparse_moe.experts.{e}.{part}.weight"
    router_weight = "block_sparse_moe.gate.weight"
    router_bias = "block_sparse_moe.gate.e_score_correction_bias"

    def __init__(self):
        Layout.__init__(self, prefix="language_model.model.layers.{layer}.",
                        embed="language_model.model.embed_tokens.weight")

    def scale_for(self, source, name):
        if name.endswith(".weight_packed"):
            candidate = name[:-len(".weight_packed")] + ".weight_scale"
            if source.has(candidate):
                return candidate
        return None

    def is_scale(self, name):
        return name.endswith(".weight_scale")

    def translate(self, name, layer):
        key = name[len(self.prefix(layer)):]
        if key.endswith(".weight_packed"):
            key = key[:-len(".weight_packed")] + ".weight"
        return key

    def decode(self, source, name, tensor, scale, dtype, device):
        tensor = tensor.to(device)
        if name.endswith(".weight_packed"):
            return dequant_fp4(tensor, scale.to(device), 32, dtype)
        return Layout.decode(self, source, name, tensor, scale, dtype, device)

    def expert_loads(self, source, layer):
        loads = []
        scales = {}
        for name in self.layer_names(source, layer):
            match = self.expert_re.match(name)
            if match is None or match.group(4) != "weight_packed":
                continue
            expert = int(match.group(2))
            part = match.group(3)
            scales[name] = self.scale_for(source, name)
            loads.append(Load(self.list_target.format(e=expert, part=part), name))
        return loads, scales

    def state_of(self, output):
        return (output[0], output[1])

    def inject(self, args, kwargs, state):
        if state is None:
            return args, kwargs
        kwargs = dict(kwargs)
        kwargs["block_residual"] = state[1]
        if args:
            return (state[0],) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = state[0]
        return args, kwargs

    def passthrough(self, args, kwargs, width):
        hidden = args[0] if args else kwargs["hidden_states"]
        return hidden, kwargs.get("block_residual")


LAYOUTS = {
    "deepseek_v4": DeepSeekV4Layout,
    "glm5_next": GlmLayout,
    "glm5_next_text": GlmLayout,
    "kimi_k3": KimiLayout,
    "kimi_k31": KimiLayout,
}


def auto_prefix(weight_map):
    """Checkpoint prefix of the language model layers, as "<head>layers.{layer}."."""
    counts = {}
    pattern = re.compile(r"^(.*?)layers\.(\d+)\.")
    for name in weight_map:
        match = pattern.match(name)
        if match is None:
            continue
        head = match.group(1)
        lowered = head.lower()
        if "vision" in lowered or "visual" in lowered or "mtp" in lowered or "encoder" in lowered:
            continue
        counts[head] = counts.get(head, 0) + 1
    if not counts:
        raise ValueError("no decoder layer names in the weight map")
    best = None
    for head in counts:
        score = counts[head]
        if "language_model" in head:
            score += 1 << 40
        if best is None or score > best[0]:
            best = (score, head)
    return best[1] + "layers.{layer}."


def layout_for(config, source):
    """The Layout for a config and source. Unknown families get the plain Hugging Face layout."""
    names = [getattr(config, "model_type", None)]
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    names.append(getattr(text, "model_type", None))
    for name in names:
        if name in LAYOUTS:
            return LAYOUTS[name]()
    prefix = auto_prefix(source.weight_map())
    head = prefix[:-len("layers.{layer}.")]
    return Layout(prefix=prefix, embed=head + "embed_tokens.weight")


# ---------------------------------------------------------------------------
# Building one layer.


class skip_init:
    """Context that turns the torch.nn.init functions into no-ops, so a layer is built without random fills."""

    names = ["uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
             "xavier_normal_", "orthogonal_"]

    def __enter__(self):
        self.saved = {}
        for name in self.names:
            self.saved[name] = getattr(torch.nn.init, name)
            setattr(torch.nn.init, name, lambda tensor, *args, **kwargs: tensor)
        return self

    def __exit__(self, *exc):
        for name in self.saved:
            setattr(torch.nn.init, name, self.saved[name])
        return False


def text_config_of(config):
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return config


def find_layer_router(layer, config):
    """(moe module, router module, top_k, num_experts) of one decoder layer, or None for a dense layer.

    Same rules as wsp.atlas.find_routers, for one layer.
    """
    holders = []
    for _, child in layer.named_children():
        holders.append(child)
    holders.append(layer)
    for child in holders:
        if child is layer and "experts" not in layer._modules:
            continue
        if not hasattr(child, "experts"):
            continue
        router = getattr(child, "gate", None)
        if router is None:
            router = getattr(child, "router", None)
        if router is None:
            continue
        top_k = atlas.first_attr(["top_k"], child, router, child.experts)
        if top_k is None:
            top_k = atlas.first_attr(["num_experts_per_tok", "top_k_experts", "moe_top_k", "num_experts_per_token"],
                                     config)
        num_experts = atlas.first_attr(["num_experts"], child, router, child.experts)
        if num_experts is None and hasattr(child.experts, "__len__"):
            num_experts = len(child.experts)
        if num_experts is None:
            num_experts = atlas.first_attr(["num_experts", "num_local_experts", "n_routed_experts"], config)
        if top_k is None or num_experts is None:
            continue
        return child, router, int(top_k), int(num_experts)
    return None


def build_layer(layer_class, config, index, device, dtype):
    """A layer from its class, with uninitialized parameters on the device."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with skip_init():
            with torch.device(device):
                layer = layer_class(config, index)
    finally:
        torch.set_default_dtype(old)
    layer.eval()
    return layer


def load_layer(layer, source, layout, index, device, dtype, expert_chunk=32, workers=8):
    """Fill a built layer from the source. Returns the number of tensors written.

    Raises ValueError when a state dict key was not written and the layout
    does not list it as optional, or when a checkpoint tensor has no target.
    """
    target = layer.state_dict()
    written = set()
    names = layout.layer_names(source, index)
    scale_names = set()
    plain = []
    for name in names:
        if layout.is_expert(name):
            continue
        scale = layout.scale_for(source, name)
        if scale is not None:
            scale_names.add(scale)
        plain.append((name, scale))
    state = {}
    wanted = []
    for name, scale in plain:
        wanted.append(name)
        if scale is not None:
            wanted.append(scale)
    raw = source.read(wanted, workers=workers)
    for name, scale in plain:
        if name in scale_names:
            continue
        key = layout.translate(name, index)
        if key is None:
            continue
        state[key] = layout.decode(source, name, raw[name], raw[scale] if scale is not None else None, dtype, device)
    raw = None
    state = layout.combine(state)
    unexpected = []
    with torch.no_grad():
        for key, value in state.items():
            if key not in target:
                unexpected.append(key)
                continue
            if tuple(target[key].shape) != tuple(value.shape):
                raise ValueError(f"layer {index}: {key} has shape {tuple(value.shape)}, the module wants "
                                 f"{tuple(target[key].shape)}")
            target[key].copy_(value)
            written.add(key)
    state = None
    if layout.expert_re is not None:
        loads, scales = layout.expert_loads(source, index)
        written.update(write_experts(layer, target, source, layout, loads, scales, device, dtype, expert_chunk,
                                     workers))
    missing = []
    for key in target:
        if key in written:
            continue
        if any(re.search(pattern, key) for pattern in layout.optional):
            continue
        missing.append(key)
    if unexpected or missing:
        raise ValueError(f"layer {index}: checkpoint tensors without a module key {unexpected[:6]}; "
                         f"module keys without a tensor {missing[:6]}")
    for name, tensor in layer.named_buffers():
        for fp32 in layout.fp32_names:
            if name.endswith(fp32) and tensor.is_floating_point():
                tensor.data = tensor.data.float()
    return len(written)


def write_experts(layer, target, source, layout, loads, scales, device, dtype, chunk, workers):
    """Decode experts in chunks and write them into the fused parameters or per-expert linears."""
    written = set()
    by_expert = {}
    for load in loads:
        by_expert.setdefault(load.index if load.index is not None else load.target, []).append(load)
    groups = list(by_expert.values())
    for start in range(0, len(groups), chunk):
        part = groups[start:start + chunk]
        wanted = []
        for group in part:
            for load in group:
                wanted.append(load.source)
                if scales.get(load.source) is not None:
                    wanted.append(scales[load.source])
        raw = source.read(wanted, workers=workers)
        with torch.no_grad():
            for group in part:
                for load in group:
                    scale = scales.get(load.source)
                    value = layout.decode(source, load.source, raw[load.source],
                                          raw[scale] if scale is not None else None, dtype, device)
                    slot = target[load.target]
                    if load.index is None:
                        slot.copy_(value)
                    elif load.rows == "gate":
                        half = slot.shape[1] // 2
                        slot[load.index, :half].copy_(value)
                    elif load.rows == "up":
                        half = slot.shape[1] // 2
                        slot[load.index, half:].copy_(value)
                    else:
                        slot[load.index].copy_(value)
                    written.add(load.target)
        raw = None
    return written


# ---------------------------------------------------------------------------
# The skeleton and the stub controller.


class Stop(Exception):
    """Raised by the active stub once the real layer has run."""


class Stub(torch.nn.Module):
    """Stands in for decoder layer `index` inside the model's own forward."""

    def __init__(self, controller, index, template):
        super().__init__()
        self.controller = controller
        self.index = index
        # Models read plain attributes of their layers, for example attention_type or is_linear_attn.
        for name, value in vars(template).items():
            if isinstance(value, (bool, int, float, str)) and not name.startswith("_"):
                setattr(self, name, value)

    def forward(self, *args, **kwargs):
        controller = self.controller
        if self.index < controller.active:
            return controller.layout.passthrough(args, kwargs, controller.width)
        if self.index > controller.active:
            raise RuntimeError("the forward went past the active layer")
        args, kwargs = controller.layout.inject(args, kwargs, controller.state)
        output = controller.layer(*args, **kwargs)
        if controller.width is None and isinstance(output, (tuple, list)):
            controller.width = len(output)
        controller.output = output
        raise Stop()


class Controller:
    """Holds the active layer and the state to inject."""

    def __init__(self, layout):
        self.layout = layout
        self.active = 0
        self.layer = None
        self.state = None
        self.output = None
        self.width = None


def fresh_rotary(module, config):
    cls = type(module)
    try:
        return cls(config=config)
    except TypeError:
        return cls(config)


class Skeleton:
    """The model on the meta device with stub layers, a real embedding and real rotary modules."""

    def __init__(self, config, source, layout, device, dtype, trust_remote_code=False):
        import transformers

        self.config = config
        self.text_config = text_config_of(config)
        self.layout = layout
        self.device = device
        self.dtype = dtype
        model = None
        errors = []
        for auto_name in ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"]:
            auto = getattr(transformers, auto_name, None)
            if auto is None:
                continue
            try:
                with torch.device("meta"):
                    model = auto.from_config(config, trust_remote_code=trust_remote_code)
                break
            except (ValueError, KeyError, OSError, AttributeError, TypeError) as error:
                errors.append(f"{auto_name}: {error!r}")
        if model is None:
            raise ValueError("cannot build the model skeleton: " + "; ".join(errors))
        name, layers = find_decoder_layers(model, 0)
        if "." in name:
            parent = model.get_submodule(name.rsplit(".", 1)[0])
        else:
            parent = model
        self.parent = parent
        self.layer_count = len(layers)
        self.controller = Controller(layout)
        self.classes = []
        stubs = []
        for index in range(len(layers)):
            self.classes.append(type(layers[index]))
            stubs.append(Stub(self.controller, index, layers[index]))
        setattr(parent, name.rsplit(".", 1)[-1], torch.nn.ModuleList(stubs))
        embedding = parent.embed_tokens
        weight = source.read([layout.embed_name()])[layout.embed_name()]
        real = torch.nn.Embedding(embedding.num_embeddings, embedding.embedding_dim,
                                  padding_idx=getattr(embedding, "padding_idx", None))
        with torch.no_grad():
            real.weight = torch.nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        parent.embed_tokens = real
        for child_name, child in list(parent.named_children()):
            if "rotary" in type(child).__name__.lower():
                setattr(parent, child_name, fresh_rotary(child, self.text_config).to(device))
        parent.eval()

    def run(self, ids, layer_index, layer, state):
        """Run one text through layer `layer_index`. Returns the layer output."""
        controller = self.controller
        controller.active = layer_index
        controller.layer = layer
        controller.state = state
        controller.output = None
        try:
            with torch.no_grad():
                self.parent(input_ids=ids.unsqueeze(0).to(self.device), use_cache=False)
        except Stop:
            pass
        output = controller.output
        controller.layer = None
        controller.state = None
        controller.output = None
        if output is None:
            raise RuntimeError(f"layer {layer_index} did not run")
        return output


# ---------------------------------------------------------------------------
# Profiling.


def map_from_counts(skill, general, top_k, num_experts, top=8, eps=1e-4, min_freq=0.0):
    """Same math as wsp.atlas.expert_skill_map, from {layer: (counts, tokens)} of two text sets."""
    result = {"layers": {}, "top_k": top_k, "num_experts": num_experts}
    for layer in skill:
        skill_counts, skill_tokens = skill[layer]
        general_counts, general_tokens = general[layer]
        skill_freq = (skill_counts / max(1, skill_tokens)).tolist()
        general_freq = (general_counts / max(1, general_tokens)).tolist()
        ratios = []
        for s_value, g_value in zip(skill_freq, general_freq):
            ratios.append(math.log((s_value + eps) / (g_value + eps)))
        order = list(range(len(ratios)))
        order.sort(key=lambda expert: -ratios[expert])
        chosen = []
        for expert in order:
            if len(chosen) >= top:
                break
            if ratios[expert] <= 0 or skill_freq[expert] < min_freq:
                continue
            chosen.append(expert)
        result["layers"][layer] = {
            "skill_freq": atlas.rounded(skill_freq),
            "general_freq": atlas.rounded(general_freq),
            "log_ratio": atlas.rounded(ratios),
            "top": chosen,
        }
    return result


def texts_key(groups, max_length, extra):
    digest = hashlib.sha256()
    digest.update(json.dumps([groups, max_length, extra]).encode("utf-8"))
    return digest.hexdigest()[:16]


def profile_streaming(source, config, tokenizer, skill_texts, general_texts, layout=None, top=8, max_length=256,
                      eps=1e-4, min_freq=0.0, device="cpu", dtype=torch.float32, max_layers=None, first_layer=0,
                      state_dir=None, checkpoint_every=4, prefetch=True, workers=8, expert_chunk=32,
                      trust_remote_code=False, keep_state=False, attn_implementation=None, log=print):
    """Expert maps of a model that is read one layer at a time.

    skill_texts is {skill: [text, ...]}. Returns {"maps": {skill: map}, "info": {...}}.
    Each map has the format of wsp.atlas.expert_skill_map. With keep_state the
    info also holds "state", the hidden states after the last processed layer
    (one per text, general texts first), for tests. attn_implementation, when
    given, replaces the attention backend of the layers that are built.

    The model's forward must hand the output of one layer straight to the
    next. A model that changes the hidden state between layers loses that
    change at the injection point.

    state_dir enables resume: counts of every finished layer are saved, and the
    hidden states every `checkpoint_every` layers. A rerun with the same texts
    continues from the last saved hidden states.
    """
    started = time.time()
    if layout is None:
        layout = layout_for(config, source)
    names = ["general"] + list(skill_texts.keys())
    groups = {"general": general_texts}
    for name in skill_texts:
        groups[name] = skill_texts[name]
    items = []
    for name in names:
        for ids in atlas.encode_texts(tokenizer, groups[name], max_length):
            items.append((name, ids))
    if not items:
        raise ValueError("no texts to profile")
    skeleton = Skeleton(config, source, layout, device, dtype, trust_remote_code=trust_remote_code)
    text_config = skeleton.text_config
    if attn_implementation is not None:
        # Kimi K3 code forces flash_attention_2 in the model constructor. Layers read the setting from the config.
        text_config._attn_implementation = attn_implementation
    total = skeleton.layer_count
    last = total
    if max_layers is not None:
        last = min(total, first_layer + max_layers)

    key = texts_key({name: groups[name] for name in names}, max_length, [getattr(config, "model_type", ""), total])
    counts = {}
    states = [None] * len(items)
    start = first_layer
    if state_dir is not None:
        os.makedirs(state_dir, exist_ok=True)
        start, counts, states = resume_state(state_dir, key, items, first_layer, last)
        if start > first_layer:
            log(f"resumed at layer {start}")

    router_info = {"top_k": None, "num_experts": None}
    layer_seconds = []
    for index in range(start, last):
        tick = time.time()
        source.wait_prefetch()
        layer_class = skeleton.classes[index]
        layer = build_layer(layer_class, text_config, index, device, dtype)
        load_layer(layer, source, layout, index, device, dtype, expert_chunk=expert_chunk, workers=workers)
        if prefetch and index + 1 < last:
            source.prefetch(layer_tensor_names(layout, source, index + 1), workers=workers)
        found = find_layer_router(layer, text_config)
        layer_counts = {}
        handle = None
        current = {"group": None}
        if found is not None:
            _, router, top_k, num_experts = found
            if router_info["top_k"] is None:
                router_info["top_k"] = top_k
                router_info["num_experts"] = num_experts
            for name in names:
                layer_counts[name] = [torch.zeros(num_experts, dtype=torch.float64), 0]

            def hook(module, args, output, top_k=top_k, num_experts=num_experts):
                picks = atlas.selected_experts(output, top_k, num_experts).reshape(-1).cpu()
                row = layer_counts[current["group"]]
                row[0] += torch.bincount(picks, minlength=num_experts).double()
                row[1] += picks.numel() // top_k

            handle = router.register_forward_hook(hook)
        try:
            for position, (name, ids) in enumerate(items):
                current["group"] = name
                output = skeleton.run(ids, index, layer, states[position])
                states[position] = detach_state(layout.state_of(output))
        finally:
            if handle is not None:
                handle.remove()
        if found is not None:
            counts[index] = {name: (layer_counts[name][0], layer_counts[name][1]) for name in names}
        del layer
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        layer_seconds.append(round(time.time() - tick, 2))
        log(f"layer {index + 1}/{total} done in {layer_seconds[-1]} s, read {source.bytes_read / 1e9:.2f} GB")
        if state_dir is not None:
            save_progress(state_dir, key, index, counts, states, checkpoint_every, last)
    source.wait_prefetch()

    maps = {}
    for skill in skill_texts:
        skill_side = {}
        general_side = {}
        for layer_index in counts:
            skill_side[layer_index] = counts[layer_index][skill]
            general_side[layer_index] = counts[layer_index]["general"]
        if not skill_side:
            raise ValueError("model has no MoE layers among the processed layers")
        maps[skill] = map_from_counts(skill_side, general_side, router_info["top_k"], router_info["num_experts"],
                                      top=top, eps=eps, min_freq=min_freq)
    info = {"layers_total": total, "layers_done": last - first_layer, "bytes_read": source.bytes_read,
            "requests": source.requests, "seconds": round(time.time() - started, 1), "layer_seconds": layer_seconds,
            "texts": len(items), "layout": layout.family, "layers_with_routers": sorted(counts.keys())}
    if keep_state:
        info["state"] = states
    return {"maps": maps, "info": info}


def detach_state(state):
    if isinstance(state, (tuple, list)):
        return tuple(detach_state(part) for part in state)
    if state is None:
        return None
    return state.detach()


def layer_tensor_names(layout, source, index):
    """Every tensor name of layer `index`, scale tensors included."""
    return layout.layer_names(source, index)


# Resume files.


def progress_paths(state_dir, index):
    return (os.path.join(state_dir, f"counts_{index:04d}.pt"), os.path.join(state_dir, f"state_{index:04d}.pt"))


def save_progress(state_dir, key, index, counts, states, checkpoint_every, last):
    counts_path, state_path = progress_paths(state_dir, index)
    if index in counts:
        torch.save({"key": key, "counts": counts[index]}, counts_path)
    else:
        torch.save({"key": key, "counts": None}, counts_path)
    if (index + 1) % checkpoint_every == 0 or index + 1 == last:
        torch.save({"key": key, "states": [to_cpu_state(state) for state in states]}, state_path)


def to_cpu_state(state):
    if isinstance(state, (tuple, list)):
        return tuple(to_cpu_state(part) for part in state)
    if state is None:
        return None
    return state.cpu()


def resume_state(state_dir, key, items, first_layer, last):
    """(start layer, counts, states) from the newest hidden state file of the same texts."""
    best = None
    for index in range(last - 1, first_layer - 1, -1):
        counts_path, state_path = progress_paths(state_dir, index)
        if not os.path.exists(state_path):
            continue
        saved = torch.load(state_path, weights_only=False)
        if saved.get("key") != key or len(saved["states"]) != len(items):
            continue
        best = (index, saved)
        break
    if best is None:
        return first_layer, {}, [None] * len(items)
    index, saved = best
    counts = {}
    for layer_index in range(first_layer, index + 1):
        counts_path, _ = progress_paths(state_dir, layer_index)
        if not os.path.exists(counts_path):
            return first_layer, {}, [None] * len(items)
        entry = torch.load(counts_path, weights_only=False)
        if entry.get("key") != key:
            return first_layer, {}, [None] * len(items)
        if entry["counts"] is not None:
            counts[layer_index] = entry["counts"]
    return index + 1, counts, list(saved["states"])


# ---------------------------------------------------------------------------
# Single experts by byte range.


def expert_tensor_names(layout, source, layer, expert):
    """{"gate", "up", "down"} to the checkpoint names of one expert, and the scale names.

    Returns (names, scales). For checkpoints that store the experts of a layer
    fused in one tensor (the plain layout with `experts.gate_up_proj`), names holds
    the fused tensor names and "fused" is True in the third value.
    """
    prefix = layout.prefix(layer)
    if layout.expert_re is not None:
        names = {}
        for name in layout.layer_names(source, layer):
            match = layout.expert_re.match(name)
            if match is None or int(match.group(2)) != expert:
                continue
            if name.endswith(".scale") or name.endswith("_scale_inv") or name.endswith(".weight_scale"):
                continue
            names[layout.expert_parts[match.group(3)]] = name
        scales = {}
        for part in names:
            scales[part] = layout.scale_for(source, names[part])
        return names, scales, False
    fused = prefix + "mlp.experts.gate_up_proj"
    if source.has(fused):
        return {"gate_up": fused, "down": prefix + "mlp.experts.down_proj"}, {}, True
    names = {}
    for part, patterns in [("gate", ["mlp.experts.{e}.gate_proj.weight", "block_sparse_moe.experts.{e}.w1.weight"]),
                           ("up", ["mlp.experts.{e}.up_proj.weight", "block_sparse_moe.experts.{e}.w3.weight"]),
                           ("down", ["mlp.experts.{e}.down_proj.weight", "block_sparse_moe.experts.{e}.w2.weight"])]:
        for pattern in patterns:
            name = prefix + pattern.format(e=expert)
            if source.has(name):
                names[part] = name
                break
    return names, {}, False


def router_names_of(layout, source, layer):
    """(weight name, bias name or None) of the router of one layer."""
    prefix = layout.prefix(layer)
    weight = None
    bias = None
    for candidate in [layout.router_weight, "mlp.gate.weight", "block_sparse_moe.gate.weight", "mlp.router.weight",
                      "ffn.gate.weight"]:
        if source.has(prefix + candidate):
            weight = prefix + candidate
            break
    for candidate in [layout.router_bias, "mlp.gate.e_score_correction_bias", "ffn.gate.bias",
                      "block_sparse_moe.gate.e_score_correction_bias"]:
        if source.has(prefix + candidate):
            bias = prefix + candidate
            break
    if weight is None:
        raise KeyError(f"layer {layer} has no router weight under {prefix}")
    return weight, bias


def fetch_expert(source, layer, expert_ids, layout=None, config=None, dtype=torch.bfloat16, device="cpu",
                 decode=True):
    """FFN tensors of chosen experts plus their router rows, by byte range only.

    Returns {"layer", "experts", "gate" (n, I, H), "up" (n, I, H), "down" (n, H, I), "router_rows" (n, H),
    "router_bias" (n,) or None, "bytes"}. With decode False, gate, up and down stay in the stored dtype and
    the scales are returned under "scales" (same keys).
    """
    if layout is None:
        if config is None:
            raise ValueError("pass a layout or a config")
        layout = layout_for(config, source)
    expert_ids = [int(e) for e in expert_ids]
    if len(expert_ids) == 0:
        raise ValueError("expert_ids is empty")
    before = source.bytes_read
    parts = {"gate": [], "up": [], "down": []}
    scale_parts = {"gate": [], "up": [], "down": []}
    first_names, _, fused = expert_tensor_names(layout, source, layer, expert_ids[0])
    if fused:
        gate_up = source.read_rows(first_names["gate_up"], expert_ids)
        down = source.read_rows(first_names["down"], expert_ids)
        half = gate_up.shape[1] // 2
        for position in range(len(expert_ids)):
            parts["gate"].append(gate_up[position, :half])
            parts["up"].append(gate_up[position, half:])
            parts["down"].append(down[position])
        if decode:
            for part in parts:
                parts[part] = [row.to(device=device, dtype=dtype) for row in parts[part]]
    else:
        for expert in expert_ids:
            names, scales, _ = expert_tensor_names(layout, source, layer, expert)
            for part in parts:
                if part not in names:
                    raise KeyError(f"layer {layer} expert {expert} has no {part} tensor")
            wanted = [names[part] for part in parts]
            for part in parts:
                if scales.get(part) is not None:
                    wanted.append(scales[part])
            raw = source.read(wanted)
            for part in parts:
                scale = raw[scales[part]] if scales.get(part) is not None else None
                if decode:
                    parts[part].append(layout.decode(source, names[part], raw[names[part]], scale, dtype, device))
                else:
                    parts[part].append(raw[names[part]])
                    scale_parts[part].append(scale)
    weight_name, bias_name = router_names_of(layout, source, layer)
    router = source.read_rows(weight_name, expert_ids)
    bias = None
    if bias_name is not None:
        bias = source.read_rows(bias_name, expert_ids)
    result = {"layer": layer, "experts": expert_ids}
    for part in parts:
        if decode:
            result[part] = torch.stack(parts[part])
        else:
            result[part] = parts[part]
    if not decode:
        result["scales"] = scale_parts
    result["router_rows"] = router.to(device=device, dtype=dtype) if decode else router
    result["router_bias"] = None if bias is None else bias.float()
    result["bytes"] = source.bytes_read - before
    return result


def expert_download_bytes(source, layout, picks, include_router=True):
    """Bytes to download for chosen experts, from the shard headers.

    picks is {layer: [expert ids]}. Returns {"total", "per_layer", "tensors"}.
    """
    total = 0
    per_layer = {}
    tensors = 0
    for layer in picks:
        layer_total = 0
        for expert in picks[layer]:
            names, scales, fused = expert_tensor_names(layout, source, layer, expert)
            if fused:
                for key in names:
                    shape = source.shape(names[key])
                    row = 1
                    for size in shape[1:]:
                        row *= size
                    layer_total += row * ITEM_BYTES[source.dtype_name(names[key])]
                    tensors += 1
                continue
            for part in names:
                layer_total += source.nbytes(names[part])
                tensors += 1
                if scales.get(part) is not None:
                    layer_total += source.nbytes(scales[part])
                    tensors += 1
        if include_router and len(picks[layer]) > 0:
            weight_name, bias_name = router_names_of(layout, source, layer)
            for name in [weight_name, bias_name]:
                if name is None:
                    continue
                shape = source.shape(name)
                row = 1
                for size in shape[1:]:
                    row *= size
                layer_total += row * ITEM_BYTES[source.dtype_name(name)] * len(picks[layer])
                tensors += 1
        per_layer[layer] = layer_total
        total += layer_total
    return {"total": total, "per_layer": per_layer, "tensors": tensors}


# ---------------------------------------------------------------------------
# Estimates for a whole run.


def stream_bytes(source, layout, layer_count, layers=None):
    """Bytes a streaming run reads: embedding plus the tensors of the chosen layers, from the shard headers."""
    if layers is None:
        layers = range(layer_count)
    source.preload_headers()
    total = source.nbytes(layout.embed_name())
    per_layer = {}
    for index in layers:
        size = 0
        for name in layer_tensor_names(layout, source, index):
            size += source.nbytes(name)
        per_layer[index] = size
        total += size
    return {"total": total, "per_layer": per_layer, "embedding": source.nbytes(layout.embed_name())}


def estimate_hours(total_bytes, layer_count, mb_per_second=400.0, seconds_per_layer=20.0):
    """(download hours, compute hours, total hours) from assumed rates. Both rates are guesses."""
    download = total_bytes / (mb_per_second * 1e6) / 3600.0
    compute = layer_count * seconds_per_layer / 3600.0
    return download, compute, download + compute
