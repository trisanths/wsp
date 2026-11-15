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
    if not isinstance(data, (bytearray, memoryview)) or (isinstance(data, memoryview) and data.readonly):
        data = bytearray(data)
    # The tensor shares the buffer. Callers that read one big range keep it alive through the views.
    raw = torch.frombuffer(data, dtype=torch.uint8)
    return raw.view(dtype).reshape(shape)


def slice_buffer(data, start, end):
    """data[start:end] as a writable buffer. A view when it is aligned to 8 bytes, a copy otherwise."""
    if isinstance(data, bytearray) and start % 8 == 0:
        return memoryview(data)[start:end]
    return bytearray(data[start:end])


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
        self.max_request = 1 << 28
        self._prefetch = None

    def weight_map(self):
        raise NotImplementedError

    def range_reader(self, shard):
        raise NotImplementedError

    def identity(self):
        """A JSON-able value that changes when the weights change. Part of the resume key."""
        return ["source", type(self).__name__]

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
        """Bytes of [begin, end) of one shard. Long spans go out as several requests of at most max_request bytes."""
        reader = self.reader(shard)
        size = end - begin
        if size <= self.max_request:
            data = reader(begin, size)
            if len(data) != size:
                raise RuntimeError(f"{shard}: asked for {size} bytes at {begin}, got {len(data)}")
            requests = 1
        else:
            data = bytearray(size)
            offsets = list(range(0, size, self.max_request))

            def piece(offset):
                length = min(self.max_request, size - offset)
                part = reader(begin + offset, length)
                if len(part) != length:
                    raise RuntimeError(f"{shard}: asked for {length} bytes at {begin + offset}, got {len(part)}")
                data[offset:offset + length] = part

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(piece, offsets))
            requests = len(offsets)
        with self._lock:
            self.bytes_read += size
            self.requests += requests
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
                chunk = slice_buffer(data, begin - group[0], end - group[0])
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
                chunk = slice_buffer(data, begin - group[0], end - group[0])
                pieces[row] = decode_bytes(chunk, entry["dtype"], shape[1:])
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

    def identity(self):
        files = []
        for shard in sorted(set(self._map.values())):
            stat = os.stat(os.path.join(self.folder, shard))
            files.append([shard, stat.st_size, stat.st_mtime_ns])
        return ["local", os.path.abspath(self.folder), files]


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

    def identity(self):
        return ["hub", self.repo_id, self.revision]

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
    # Module keys that combine() builds from several checkpoint tensors. Regexes need not apply: exact keys.
    combine_keys = ()
    # Name of the child of the MoE block whose output the experts read, for blocks that project the
    # hidden state before the experts. None: the experts read the input of the block.
    expert_input = None
    # Name of the child of the MoE block that normalizes the weighted sum of the expert outputs before
    # the block adds it to the residual stream. None: the block uses the weighted sum as it is.
    expert_norm = None

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
        # Keep the source in the entry: a freed source can hand its id to a new one.
        if key not in self._by_layer or self._by_layer[key][0] is not source:
            head, tail = self.prefix_format.split("{layer}")
            pattern = re.compile("^" + re.escape(head) + r"(\d+)" + re.escape(tail))
            table = {}
            for name in source.weight_map():
                match = pattern.match(name)
                if match is not None:
                    table.setdefault(int(match.group(1)), []).append(name)
            self._by_layer[key] = (source, table)
        return sorted(self._by_layer[key][1].get(layer, []))

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

    combine_keys = ("self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight")

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
    # Latent MoE: the block maps the hidden state to the expert width with this linear, then calls the experts.
    expert_input = "routed_expert_down_proj"
    # config.json of Kimi-K3 sets latent_moe_use_norm, and the index has routed_expert_norm.weight in all
    # 92 MoE layers. The block applies this RMS norm to the weighted sum of the expert outputs, per token,
    # and then the shared up-projection.
    expert_norm = "routed_expert_norm"

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


def layout_from_name(name, source):
    """The Layout of a zoo `stream.layout` name: deepseek_v4, glm5_next, kimi, or hf (plain, prefix found from the weights)."""
    if name == "kimi":
        return KimiLayout()
    if name in LAYOUTS:
        return LAYOUTS[name]()
    if name != "hf":
        raise ValueError(f"unknown layout {name}")
    prefix = auto_prefix(source.weight_map())
    head = prefix[:-len("layers.{layer}.")]
    return Layout(prefix=prefix, embed=head + "embed_tokens.weight")


def hub_layer_count(repo_id, revision="main"):
    """num_hidden_layers of the text model, from config.json alone. Needs no transformers class, so it works for new families."""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo_id, "config.json", revision=revision)) as f:
        config = json.load(f)
    text = config.get("text_config") or config
    return int(text["num_hidden_layers"])


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
    unexpected = []

    def put(key, value):
        if key not in target:
            unexpected.append(key)
            return
        if tuple(target[key].shape) != tuple(value.shape):
            raise ValueError(f"layer {index}: {key} has shape {tuple(value.shape)}, the module wants "
                             f"{tuple(target[key].shape)}")
        with torch.no_grad():
            target[key].copy_(value)
        written.add(key)

    # Decode one tensor at a time and copy it into the layer at once, so a layer with big dense
    # tensors never holds the stored, the decoded and the module copy together. Tensors that
    # the layout merges (combine_keys) wait in `state`.
    state = {}
    for name, scale in plain:
        if name in scale_names:
            continue
        key = layout.translate(name, index)
        if key is None:
            continue
        value = layout.decode(source, name, raw.pop(name), raw.pop(scale) if scale is not None else None, dtype,
                              device)
        if key in layout.combine_keys:
            state[key] = value
        else:
            put(key, value)
        value = None
    raw = None
    state = layout.combine(state)
    for key, value in state.items():
        put(key, value)
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
    """The model on the meta device with stub layers, a real embedding and real rotary modules.

    routed[k] says whether decoder layer k has a router and experts. It comes from the meta model,
    so it is known before any weight is read. With embedding false the embedding table stays on the
    meta device until load_embedding() is called, so a caller can stop first without the download.
    """

    def __init__(self, config, source, layout, device, dtype, trust_remote_code=False, embedding=True):
        import transformers

        # Load torch._dynamo before any device context is active. transformers 5.x model modules import
        # it, and its first import inside `with torch.device("meta")` fails with a circular import
        # ("partially initialized module 'torch._dynamo' has no attribute 'disable'"), which
        # transformers reports as "Could not import module <class>". Seen on GLM-5.3-Flash and
        # DeepSeek-V4 with torch 2.7.1 and transformers 5.23.0.
        import torch._dynamo  # noqa: F401

        self.config = config
        self.text_config = text_config_of(config)
        self.source = source
        self.layout = layout
        self.device = device
        self.dtype = dtype
        self.embedded = False
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
            except (ValueError, KeyError, OSError, AttributeError, TypeError, ImportError) as error:
                # ImportError: when an auto class does not list the config, transformers builds its
                # "should be one of" message by importing every config class, and one of them can need
                # a package the image does not have (GLM-5.3-Flash hit this on AriaTextConfig). The next
                # auto class may still build the model.
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
        self.routed = []
        stubs = []
        for index in range(len(layers)):
            self.classes.append(type(layers[index]))
            self.routed.append(find_layer_router(layers[index], self.text_config) is not None)
            stubs.append(Stub(self.controller, index, layers[index]))
        setattr(parent, name.rsplit(".", 1)[-1], torch.nn.ModuleList(stubs))
        for child_name, child in list(parent.named_children()):
            if "rotary" in type(child).__name__.lower():
                setattr(parent, child_name, fresh_rotary(child, self.text_config).to(device))
        parent.eval()
        if embedding:
            self.load_embedding()

    def load_embedding(self):
        """Read the embedding table and put it in place of the one on the meta device."""
        if self.embedded:
            return
        embedding = self.parent.embed_tokens
        name = self.layout.embed_name()
        weight = self.source.read([name])[name]
        real = torch.nn.Embedding(embedding.num_embeddings, embedding.embedding_dim,
                                  padding_idx=getattr(embedding, "padding_idx", None))
        with torch.no_grad():
            real.weight = torch.nn.Parameter(weight.to(device=self.device, dtype=self.dtype), requires_grad=False)
        self.parent.embed_tokens = real.eval()
        self.embedded = True

    def run(self, ids, layer_index, layer, state):
        """Run one text through layer `layer_index`. Returns the layer output."""
        self.load_embedding()
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
# Saliency.
#
# The saliency of expert e in one MoE layer, on a set of texts, is the sum over the tokens whose
# router picks e of (the routing weight the model gives e on that token) times (the L2 norm of the
# output of e on that token). It says how much the expert writes into the residual stream on those
# texts. wsp.slim.collect_moe_stats is the reference, and a streamed layer gives the same numbers.
#
# The forward of a model never shows one expert's output next to its routing weight, so the meter
# computes the outputs once more, one per (token, picked expert) pair. There are two methods. The
# way the family keeps its experts decides which one runs.
#
# "list": the experts are a ModuleList (the transformers 4.x families and the Kimi K3 remote code).
#   A hook on an expert sees its rows in the order the block chose: by top-k slot in Qwen3-MoE and
#   Mixtral, by an unstable argsort in DeepSeek-V2 and Kimi. A hooked output therefore cannot be
#   paired with the routing weight of its token. The meter calls each picked expert on the rows of
#   its tokens, as wsp.slim does. The weights come from the router output (routing_of). The expert
#   input is the input of the MoE block, or the output of the child named by layout.expert_input
#   when the block has it (Kimi's latent MoE). On the first text of a layer the meter compares that
#   input with the rows one expert received in the block's own forward, and stops if they differ.
#
#   A block can normalize the weighted sum of the expert outputs before it adds it to the residual
#   stream. Kimi K3 does: y = sum of weight * output per token, then routed_expert_norm(y), an RMS
#   norm, then the shared up-projection. What a token's experts write then has a fixed size, however
#   large their outputs are, so weight * norm of the output would rank the experts by a size that
#   the model removes. When the block has the child named by layout.expert_norm, the meter reads
#   the input of that child and divides the gain of every (token, expert) pair by the RMS of the
#   token's weighted sum, the number the norm divides by. The result is the L2 norm of the
#   expert's share of the normalized sum, before the learned weight of the norm and before the
#   up-projection, which are the same two maps for every expert. On the first text of a layer the
#   meter checks that the child is an RMS norm, and stops if it is not.
#
# "fused": the experts are one module, called as experts(hidden, top_k_index, top_k_weights), that
#   returns only the weighted sum per token (transformers 5.x: DeepSeek-V4, GLM-5.3, Qwen3.5-MoE).
#   A hook on the module gives the indices and weights the model uses. The meter then calls
#   module.forward once on tokens * top_k rows. Row j holds the hidden state of token j // top_k,
#   the one index of slot j % top_k, and weight 1, so output row j is the output of that expert on
#   that token. The call goes to forward, not to the module, so the hook does not run again. On the
#   first text of a layer the meter adds the rows up again, weight times output over the slots of
#   each token, and stops unless the sum is the tensor the module returned (check_fused).
#
# Cost. Each method evaluates tokens * top_k (token, expert) pairs per text, the same pairs as the
# layer's own forward, grouped by expert in the same way. The routed experts run twice. The rest
# of the layer (attention, router, shared experts, norms) runs once, and nothing is read or
# decoded again. Memory on top of the layer: the list method holds the output of one expert; the
# fused method holds the repeated hidden states and the outputs of one text (2 * tokens * top_k *
# hidden values), plus what the experts module allocates inside its own forward on those rows.
# The norms are taken in float32, 256 rows at a time, so the outputs get no float32 copy.

# Upper bound of the forward work of a layer with saliency, relative to the same layer without it.
SALIENCY_COMPUTE_FACTOR = 2.0

# Version of the saliency sums in the resume files. Raise it when the numbers the saliency pass
# writes change their meaning, so a rerun does not continue from sums of other code. The first
# saliency code wrote no version. Version 2 divides by the RMS of the routed sum for Kimi K3.
SALIENCY_STATE_VERSION = 2

SALIENCY_OFF = ("Run again with saliency off (saliency=False, or --no-saliency in modal_giant.py) to get the pick "
                "counts alone.")


def check_tolerance(dtype):
    """Largest difference, as a share of the largest value, that a self-check of the saliency pass accepts.

    A wrong pairing of tokens and experts gives differences of the size of the values themselves.
    Rounding gives about 2 ** -8 per value in the 16-bit float types and 2 ** -24 in float32.
    """
    if dtype in (torch.bfloat16, torch.float16):
        return 5e-2
    return 1e-4


def row_norms(out, block=256):
    """L2 norm of every row, in float32. The rows are converted `block` at a time, so no float32 copy of all of them exists."""
    norms = torch.empty(out.shape[0], dtype=torch.float32, device=out.device)
    for start in range(0, out.shape[0], block):
        norms[start:start + block] = out[start:start + block].float().norm(dim=-1)
    return norms


def routing_of(moe, output, top_k, num_experts):
    """(weights, indices), both (tokens, top_k): the routing weights and the experts of a router output.

    A router that returns an integer index tensor also returns the weights the block uses:
    (indices, weights) in DeepSeek-V2, DeepSeek-V3 and Kimi, (logits, weights, indices) in
    transformers 5.x. Those weights are taken as they are, scaling factor included. A router that
    returns logits leaves the rest to the block: softmax in float32, top-k, and division by the
    sum unless the block has norm_topk_prob false. That branch is the math of wsp.slim.route.
    """
    logits = output
    if isinstance(output, (tuple, list)):
        indices = None
        for part in output:
            if torch.is_tensor(part) and not torch.is_floating_point(part) and part.shape[-1] == top_k:
                indices = part.reshape(-1, top_k)
                break
        if indices is not None:
            weights = None
            for part in output:
                # Keep the last match. With top_k equal to num_experts the logits have this size too. They come first.
                if torch.is_tensor(part) and torch.is_floating_point(part) and part.numel() == indices.numel():
                    weights = part
            if weights is None:
                raise NotImplementedError("router output has expert indices but no weights of the same size")
            return weights.detach().reshape(-1, top_k).float(), indices
        logits = None
        for part in output:
            if torch.is_tensor(part) and torch.is_floating_point(part) and part.shape[-1] == num_experts:
                logits = part
                break
        if logits is None:
            raise NotImplementedError(f"router output has no index tensor or logits of width {num_experts}")
    if logits.shape[-1] != num_experts:
        raise NotImplementedError(f"router output has width {logits.shape[-1]}, expected {num_experts} experts")
    probs = torch.softmax(logits.detach().float().reshape(-1, num_experts), dim=-1)
    weights, indices = torch.topk(probs, top_k, dim=-1)
    if getattr(moe, "norm_topk_prob", True):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


def first_tensor(args, kwargs):
    for value in list(args) + list(kwargs.values()):
        if torch.is_tensor(value):
            return value
    return None


def call_value(args, kwargs, place):
    if isinstance(place, int):
        return args[place]
    return kwargs[place]


def call_with(args, kwargs, values):
    """A copy of (args, kwargs) with {place: value} written in."""
    args = list(args)
    kwargs = dict(kwargs)
    for place in values:
        if isinstance(place, int):
            args[place] = values[place]
        else:
            kwargs[place] = values[place]
    return tuple(args), kwargs


def experts_call_parts(args, kwargs):
    """Where the hidden states, expert indices and routing weights sit in one call of a fused experts module.

    Returns {"hidden", "index", "weights"}: each value is a position in args or a keyword. The
    indices are the first integer tensor, the hidden states the first float tensor, and the
    weights the next float tensor with the shape of the indices.
    """
    places = list(enumerate(args)) + list(kwargs.items())
    found = {"hidden": None, "index": None, "weights": None}
    for place, value in places:
        if not torch.is_tensor(value) or value.dtype == torch.bool:
            continue
        if torch.is_floating_point(value):
            if found["hidden"] is None:
                found["hidden"] = place
        elif found["index"] is None:
            found["index"] = place
    if found["index"] is not None:
        shape = call_value(args, kwargs, found["index"]).shape
        for place, value in places:
            if place == found["hidden"] or not torch.is_tensor(value) or not torch.is_floating_point(value):
                continue
            if value.shape == shape:
                found["weights"] = place
                break
    missing = [name for name in found if found[name] is None]
    if missing:
        raise NotImplementedError(f"the fused experts call has no {', '.join(missing)} tensor. " + SALIENCY_OFF)
    return found


class SaliencyMeter:
    """Adds up the saliency of every expert of one MoE layer while the texts run through it.

    names are the text sets. Set `group` to the set of the text that runs next. Afterwards
    sums[name] is the saliency per expert (float64, on the CPU) and tokens[name] the number of
    tokens. close() removes the hooks and lets go of the layer.
    """

    def __init__(self, index, layer, moe, router, top_k, num_experts, names, layout):
        self.index = index
        self.moe = moe
        self.top_k = top_k
        self.num_experts = num_experts
        self.sums = {}
        self.tokens = {}
        for name in names:
            self.sums[name] = torch.zeros(num_experts, dtype=torch.float64)
            self.tokens[name] = 0
        self.group = None
        self.handles = []
        self.probe_handles = []
        self.probe = None
        self.seen = {}
        # Name of the norm child whose RMS divides the gains, or None. Set below for the list method.
        self.norm = None
        self.norm_checked = False
        self.fused_checked = False
        # One line per self-check that passed, for the log of the layer.
        self.notes = []
        if not isinstance(moe.experts, torch.nn.ModuleList):
            self.method = "fused"
            self.handles.append(moe.experts.register_forward_hook(self.measure_fused, with_kwargs=True))
            return
        self.method = "list"
        if moe is layer:
            raise NotImplementedError(f"layer {index}: the decoder layer itself holds the experts, so the input of "
                                      "the experts is not known. " + SALIENCY_OFF)
        self.handles.append(router.register_forward_hook(self.keep_router))
        self.handles.append(moe.register_forward_pre_hook(self.keep_input, with_kwargs=True))
        reader = None
        if layout.expert_input is not None:
            reader = getattr(moe, layout.expert_input, None)
        if reader is not None:
            self.handles.append(reader.register_forward_hook(self.keep_expert_input))
        norm = None
        if layout.expert_norm is not None:
            norm = getattr(moe, layout.expert_norm, None)
        if norm is not None:
            self.norm = layout.expert_norm
            self.handles.append(norm.register_forward_hook(self.keep_norm))
        self.handles.append(moe.register_forward_hook(self.measure_list))
        for expert_id in range(len(moe.experts)):
            def keep_probe(module, args, kwargs, expert_id=expert_id):
                if self.probe is None:
                    self.probe = (expert_id, first_tensor(args, kwargs))

            self.probe_handles.append(moe.experts[expert_id].register_forward_pre_hook(keep_probe, with_kwargs=True))

    def keep_router(self, module, args, output):
        self.seen["router"] = output

    def keep_input(self, module, args, kwargs):
        self.seen["input"] = first_tensor(args, kwargs)

    def keep_expert_input(self, module, args, output):
        self.seen["expert_input"] = output

    def keep_norm(self, module, args, output):
        self.seen["norm"] = (module, args[0], output)

    def measure_list(self, module, args, output):
        with torch.no_grad():
            weights, indices = routing_of(self.moe, self.seen["router"], self.top_k, self.num_experts)
            source = self.seen.get("expert_input")
            if source is None:
                source = self.seen["input"]
            x = source.reshape(-1, source.shape[-1])
            if x.shape[0] != indices.shape[0]:
                raise RuntimeError(f"layer {self.index}: the router picked experts for {indices.shape[0]} tokens, the "
                                   f"MoE block got {x.shape[0]}. " + SALIENCY_OFF)
            if self.probe_handles:
                self.check_probe(x, indices)
            flat = indices.reshape(-1)
            # One stable sort puts the (token, slot) pairs of each expert side by side, tokens in rising order.
            order = torch.argsort(flat, stable=True)
            sizes = torch.bincount(flat, minlength=self.num_experts).tolist()
            norms = torch.zeros(flat.shape[0], dtype=torch.float32, device=x.device)
            start = 0
            for expert in range(len(sizes)):
                if sizes[expert] == 0:
                    continue
                pairs = order[start:start + sizes[expert]]
                start += sizes[expert]
                if expert >= self.num_experts:
                    continue
                out = self.moe.experts[expert](x[pairs // self.top_k])
                norms[pairs] = row_norms(out)
            gains = weights.reshape(-1).float() * norms
            if self.norm is not None:
                gains = gains / self.routed_rms(x.shape[0]).repeat_interleave(self.top_k)
            self.add(flat, gains, x.shape[0])
        self.seen.clear()

    def routed_rms(self, tokens):
        """RMS per token of the weighted sum of the expert outputs: the number the norm after the experts divides by."""
        if "norm" not in self.seen:
            raise RuntimeError(f"layer {self.index}: the MoE block has {self.norm} but did not call it. "
                               + SALIENCY_OFF)
        module, total, normed = self.seen["norm"]
        total = total.detach().reshape(-1, total.shape[-1]).float()
        if total.shape[0] != tokens:
            raise RuntimeError(f"layer {self.index}: {self.norm} got {total.shape[0]} rows, the MoE block has "
                               f"{tokens} tokens. " + SALIENCY_OFF)
        eps = 0.0
        for name in ["variance_epsilon", "eps"]:
            if isinstance(getattr(module, name, None), float):
                eps = getattr(module, name)
                break
        rms = torch.sqrt(total.pow(2).mean(dim=-1) + eps)
        if not self.norm_checked:
            self.norm_checked = True
            want = total / rms.unsqueeze(-1)
            weight = getattr(module, "weight", None)
            if torch.is_tensor(weight) and weight.shape == want.shape[-1:]:
                want = want * weight.detach().float()
            got = normed.detach().reshape(want.shape).float()
            scale = float(got.abs().max())
            error = float((got - want).abs().max())
            if not error <= check_tolerance(normed.dtype) * scale:
                raise RuntimeError(f"layer {self.index}: {self.norm} is not an RMS norm of the weighted sum of the "
                                   f"expert outputs (largest difference {error:.3g}, largest output {scale:.3g}), so "
                                   "the saliency pass does not know what it divides by. " + SALIENCY_OFF)
            self.notes.append(f"{self.norm} is an RMS norm: its output differs from weight * x / rms(x) by "
                              f"{error / max(scale, 1e-30):.2g} of its largest value")
        return rms

    def check_probe(self, x, indices):
        """Stop unless the first expert the block called received the rows that the meter would give it."""
        for handle in self.probe_handles:
            handle.remove()
        self.probe_handles = []
        if self.probe is None or self.probe[1] is None:
            raise RuntimeError(f"layer {self.index}: the MoE block did not call an expert module, so the expert "
                               "input cannot be confirmed. " + SALIENCY_OFF)
        expert, got = self.probe
        self.probe = None
        got = got.reshape(-1, got.shape[-1]).float()
        want = x[torch.where(indices == expert)[0]].float()
        # The block may hand over the rows in another order. Sorting each column removes the order.
        same = got.shape == want.shape and torch.equal(torch.sort(got, dim=0).values, torch.sort(want, dim=0).values)
        if not same:
            raise RuntimeError(f"layer {self.index}: expert {expert} did not receive the rows of the tensor that the "
                               "saliency pass takes as the expert input. Set layout.expert_input to the child of "
                               "the MoE block that feeds the experts. " + SALIENCY_OFF)
        self.notes.append(f"expert {expert} received the {got.shape[0]} rows that the saliency pass gives it")

    def measure_fused(self, module, args, kwargs, output):
        with torch.no_grad():
            places = experts_call_parts(args, kwargs)
            hidden = call_value(args, kwargs, places["hidden"])
            index = call_value(args, kwargs, places["index"])
            weights = call_value(args, kwargs, places["weights"])
            top_k = index.shape[-1]
            flat = index.reshape(-1)
            if hidden.dim() != 2 or hidden.shape[0] * top_k != flat.shape[0]:
                raise NotImplementedError(
                    f"layer {self.index}: the fused experts got hidden states of shape {tuple(hidden.shape)} and "
                    f"indices of shape {tuple(index.shape)}; the saliency pass needs (tokens, hidden) and "
                    "(tokens, top_k). " + SALIENCY_OFF)
            wide = hidden.repeat_interleave(top_k, dim=0)
            ones = torch.ones_like(weights).reshape(-1, 1)
            values = {places["hidden"]: wide, places["index"]: flat.reshape(-1, 1), places["weights"]: ones}
            new_args, new_kwargs = call_with(args, kwargs, values)
            out = module.forward(*new_args, **new_kwargs)
            weights = weights.detach().reshape(-1, top_k).float()
            if not self.fused_checked:
                self.fused_checked = True
                self.check_fused(out, weights, output, hidden.shape[0], top_k)
            self.add(flat, weights.reshape(-1) * row_norms(out), hidden.shape[0])

    def check_fused(self, out, weights, output, tokens, top_k):
        """Stop unless the expert outputs that were computed again add up to what the experts module returned.

        The module returns, per token, the sum over the top-k slots of weight times expert output.
        out holds one row per (token, slot) pair, so the same sum over its rows must give the same
        tensor, up to the rounding of the dtype. The sum is taken slot by slot in float32, which
        needs tokens * hidden values and no copy of out.
        """
        result = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(result) or result.numel() != tokens * out.shape[-1]:
            raise NotImplementedError(f"layer {self.index}: the fused experts did not return one row per token, so "
                                      "the saliency pass cannot be checked against it. " + SALIENCY_OFF)
        result = result.detach().reshape(tokens, -1).float()
        rows = out.reshape(tokens, top_k, -1)
        again = torch.zeros_like(result)
        for slot in range(top_k):
            again += weights[:, slot, None] * rows[:, slot].float()
        scale = float(result.abs().max())
        error = float((again - result).abs().max())
        if not error <= check_tolerance(out.dtype) * scale:
            raise RuntimeError(f"layer {self.index}: the expert outputs of the saliency pass do not add up to the "
                               f"output of the experts module (largest difference {error:.3g}, largest output "
                               f"{scale:.3g}), so they are not the outputs the model used. " + SALIENCY_OFF)
        self.notes.append("weight * recomputed expert output, summed over the slots, differs from the output of "
                          f"the experts module by {error / max(scale, 1e-30):.2g} of its largest value")

    def add(self, flat, gains, tokens):
        """Add the gain of every (token, slot) pair to its expert. An index past the last expert means "no expert"."""
        keep = flat < self.num_experts
        if not bool(torch.isfinite(gains[keep]).all()):
            raise RuntimeError(f"layer {self.index}: the saliency of text set '{self.group}' has a value that is "
                               "not finite (an expert output or a routing weight is inf or nan). One such value "
                               "would spoil the ranking of the whole layer. " + SALIENCY_OFF)
        total = torch.zeros(self.num_experts, dtype=torch.float32, device=gains.device)
        total.index_add_(0, flat[keep], gains[keep])
        self.sums[self.group] += total.double().cpu()
        self.tokens[self.group] += tokens

    def check(self, layer_counts):
        """Stop unless the saliency pass saw every token that the router hook counted."""
        for name in self.tokens:
            if self.tokens[name] != layer_counts[name][1]:
                raise RuntimeError(f"layer {self.index}: the router routed {layer_counts[name][1]} tokens of "
                                   f"'{name}', the saliency pass saw {self.tokens[name]}. " + SALIENCY_OFF)

    def close(self):
        for handle in self.handles + self.probe_handles:
            handle.remove()
        self.handles = []
        self.probe_handles = []
        self.moe = None
        self.probe = None
        self.seen = {}


def significant(values, digits=7):
    """Floats rounded to `digits` significant digits. Saliency has no fixed scale, so decimal places do not fit."""
    out = []
    for value in values:
        out.append(float(f"{float(value):.{digits}g}"))
    return out


def top_by_saliency(saliency, counts=None, top=8):
    """Up to `top` expert ids with the highest saliency, best first. Experts with saliency 0 are left out.

    Ties go to the expert that is picked more often (when counts are given), then to the lower
    id, as in wsp.slim.rank_experts.
    """
    values = [float(value) for value in saliency]
    picks = [0.0] * len(values)
    if counts is not None:
        picks = [float(value) for value in counts]
    order = list(range(len(values)))
    order.sort(key=lambda expert: (-values[expert], -picks[expert], expert))
    chosen = []
    for expert in order:
        if len(chosen) >= top or values[expert] <= 0:
            break
        chosen.append(expert)
    return chosen


def add_saliency(result, skill, general, top=8):
    """Write the saliency fields into a map made by map_from_counts.

    skill and general are {layer: (saliency sums, pick counts, tokens)} of the two text sets.
    Each layer gets "skill_saliency" and "general_saliency" (per expert, divided by the tokens of
    the text set, so sets of different size compare) and "top_saliency" (up to `top` experts with
    the highest skill saliency, best first).
    """
    for layer in result["layers"]:
        skill_sums, skill_counts, skill_tokens = skill[layer]
        general_sums, _, general_tokens = general[layer]
        per_token = (skill_sums / max(1, skill_tokens)).tolist()
        entry = result["layers"][layer]
        entry["skill_saliency"] = significant(per_token)
        entry["general_saliency"] = significant((general_sums / max(1, general_tokens)).tolist())
        entry["top_saliency"] = top_by_saliency(per_token, skill_counts.tolist(), top)
    return result


def saliency_report(name, sums, counts, tokens, top=8):
    """One log line on the saliency of one text set in one layer, for a person who checks a run.

    It gives the sum, the three experts with the highest saliency next to the three that are
    picked most often, how many of the `top` best by saliency are also among the `top` most
    picked, and how many picked experts have saliency 0 (which a working pass does not produce).
    """
    values = [float(value) for value in sums]
    picks = [float(value) for value in counts]
    total = sum(values)
    by_saliency = top_by_saliency(values, picks, top)
    by_picks = top_by_saliency(picks, None, top)
    shared = len(set(by_saliency) & set(by_picks))
    picked = sum(1 for value in picks if value > 0)
    silent = sum(1 for expert in range(len(picks)) if picks[expert] > 0 and values[expert] <= 0)
    return (f"saliency '{name}': {tokens} tokens, sum {total:.6g}, {total / max(1, tokens):.6g} per token; "
            f"top 3 by saliency {by_saliency[:3]}, by picks {by_picks[:3]}; {shared} of the top {len(by_saliency)} "
            f"by saliency are in the top {len(by_picks)} by picks; {picked} of {len(picks)} experts picked, "
            f"{silent} of them with saliency 0")


def top_saliency_experts(expert_map):
    """The compact saliency form of one map: {str(layer): [expert ids]}, or None for a map without saliency."""
    out = {}
    for layer in expert_map["layers"]:
        if "top_saliency" not in expert_map["layers"][layer]:
            return None
        out[str(layer)] = list(expert_map["layers"][layer]["top_saliency"])
    return out


def compact_maps(result):
    """The two compact maps per skill of a profile_streaming result, for an atlas entry.

    Returns {"expert_maps": {skill: {str(layer): [expert ids]}}, "expert_saliency_maps": the same
    form}. expert_maps holds the experts with the highest ratio of skill pick rate to general pick
    rate ("top", as wsp.atlas.top_experts). expert_saliency_maps holds the experts with the highest
    skill saliency ("top_saliency"), and is None when the run had saliency off.
    """
    maps = result["maps"] if "maps" in result else result
    picked = {}
    salient = {}
    for skill in maps:
        picked[skill] = atlas.top_experts(maps[skill])
        salient[skill] = top_saliency_experts(maps[skill])
    if not salient or any(value is None for value in salient.values()):
        salient = None
    return {"expert_maps": picked, "expert_saliency_maps": salient}


def saliency_table(result):
    """Saliency per token of every expert, for a results file.

    Returns {"general": {str(layer): [...]}, "skills": {skill: {str(layer): [...]}}}, or None when
    the run had saliency off. With these lists top_by_saliency gives the best experts of a layer
    for any count, not only for the `top` of the run.
    """
    maps = result["maps"] if "maps" in result else result
    table = {"general": {}, "skills": {}}
    for skill in maps:
        table["skills"][skill] = {}
        for layer in maps[skill]["layers"]:
            entry = maps[skill]["layers"][layer]
            if "skill_saliency" not in entry:
                return None
            table["skills"][skill][str(layer)] = entry["skill_saliency"]
            table["general"][str(layer)] = entry["general_saliency"]
    return table


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
                      trust_remote_code=False, keep_state=False, attn_implementation=None, log=print,
                      saliency=True):
    """Expert maps of a model that is read one layer at a time.

    skill_texts is {skill: [text, ...]}. Returns {"maps": {skill: map}, "info": {...}}.
    Each map has the format of wsp.atlas.expert_skill_map. With keep_state the
    info also holds "state", the hidden states after the last processed layer
    (one per text, general texts first), for tests. attn_implementation, when
    given, replaces the attention backend of the layers that are built.

    With saliency (the default) every layer of a map also has "skill_saliency",
    "general_saliency" and "top_saliency" (see add_saliency and the Saliency
    section above), and the info has "saliency_method" and "saliency_norm" (the
    norm child whose RMS divides the gains, or None). The routed experts of each
    layer then run twice, and the log gets one line per MoE layer and text set
    (saliency_report). With saliency false none of this runs, and the result has
    only the pick-rate fields.

    The model's forward must hand the output of one layer straight to the
    next. A model that changes the hidden state between layers loses that
    change at the injection point.

    state_dir enables resume: counts (and saliency sums) of every finished layer
    are saved, and the hidden states every `checkpoint_every` layers. A rerun
    with the same texts continues from the last saved hidden states. A rerun
    with saliency stops with a ValueError when the saved layers have none, or
    have saliency of another SALIENCY_STATE_VERSION. That check runs before the
    model skeleton is built, so nothing is downloaded first. A run whose layer
    range holds no MoE layer stops before the embedding is read.
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

    def saved_progress(total):
        """What state_dir holds for a model of `total` layers. Raises ValueError for saliency that cannot be continued."""
        last = total
        if max_layers is not None:
            last = min(total, first_layer + max_layers)
        key = texts_key({name: groups[name] for name in names}, max_length,
                        [getattr(config, "model_type", ""), total, source.identity(), layout.family, str(dtype)])
        progress = {"total": total, "last": last, "key": key, "start": first_layer, "counts": {},
                    "states": [None] * len(items), "sums": {} if saliency else None, "tag": None}
        if state_dir is None:
            return progress
        os.makedirs(state_dir, exist_ok=True)
        start, records, states = resume_state(state_dir, key, items, first_layer, last, log)
        progress["start"] = start
        progress["states"] = states
        for layer_index in sorted(records):
            if records[layer_index]["counts"] is not None:
                progress["counts"][layer_index] = records[layer_index]["counts"]
        if saliency:
            progress["sums"], progress["tag"] = saved_saliency(state_dir, records, first_layer)
        return progress

    # Read the saved state before the skeleton is built when the config gives the layer count. A
    # state that cannot be continued then stops the run before the embedding is downloaded.
    progress = None
    config_total = getattr(text_config_of(config), "num_hidden_layers", None)
    if state_dir is not None and config_total is not None:
        progress = saved_progress(int(config_total))
    skeleton = Skeleton(config, source, layout, device, dtype, trust_remote_code=trust_remote_code, embedding=False)
    text_config = skeleton.text_config
    if attn_implementation is not None:
        # Kimi K3 code forces flash_attention_2 in the model constructor. Layers read the setting from the config.
        text_config._attn_implementation = attn_implementation
    total = skeleton.layer_count
    if progress is None or progress["total"] != total:
        progress = saved_progress(total)
    last = progress["last"]
    key = progress["key"]
    start = progress["start"]
    counts = progress["counts"]
    # {layer: {text set: saliency sum per expert}}. Stays None with saliency off.
    sums = progress["sums"]
    # (method, norm) of the saliency pass: from the saved layers, or from the first MoE layer that runs.
    tag = progress["tag"]

    # Stop before the embedding and the first layer are read when no layer of the range has a router.
    routed = [index for index in range(total) if skeleton.routed[index]]
    if not counts and not any(first_layer <= index < last for index in routed):
        if not routed:
            raise ValueError("model has no MoE layers")
        later = [index for index in routed if index >= first_layer]
        advice = f"no MoE layer comes at or after layer {first_layer}"
        if later:
            advice = (f"the first MoE layer is layer {later[0]}, so max_layers must be at least "
                      f"{later[0] + 1 - first_layer} (--max-layers in modal_giant.py)")
        raise ValueError(f"model has no MoE layers among the processed layers {first_layer} to {last - 1}: {advice}")
    skeleton.load_embedding()
    states = [to_device_state(state, device) for state in progress["states"]]
    progress = None
    if start > first_layer:
        line = f"resumed at layer {start}: {len(counts)} MoE layers have saved pick counts"
        if saliency:
            line += (f", {len(sums)} have saved saliency (state version {SALIENCY_STATE_VERSION}, "
                     f"method {tag[0] if tag else None})")
        log(line)

    router_info = {"top_k": None, "num_experts": None}
    if counts:
        # A run that resumes every layer builds none, so the router shape comes from the saved counts.
        router_info["top_k"], router_info["num_experts"] = router_shape(counts[min(counts)])
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
        meter = None
        moe = None
        router = None
        current = {"group": None}
        if found is not None:
            moe, router, top_k, num_experts = found
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
            if found is not None and saliency:
                meter = SaliencyMeter(index, layer, moe, router, top_k, num_experts, names, layout)
                if tag is None:
                    tag = (meter.method, meter.norm)
                elif tag != (meter.method, meter.norm):
                    raise ValueError(
                        f"layer {index}: the saliency pass runs as {(meter.method, meter.norm)}, the layers before "
                        f"it ran as {tag}. Sums of two methods cannot go into one result. " + SALIENCY_OFF
                        + stale_state_advice(state_dir, first_layer))
            for position, (name, ids) in enumerate(items):
                current["group"] = name
                if meter is not None:
                    meter.group = name
                output = skeleton.run(ids, index, layer, states[position])
                states[position] = detach_state(layout.state_of(output))
        finally:
            if handle is not None:
                handle.remove()
            if meter is not None:
                meter.close()
        if found is not None:
            counts[index] = {name: (layer_counts[name][0], layer_counts[name][1]) for name in names}
        if meter is not None:
            meter.check(layer_counts)
            sums[index] = meter.sums
            norm_note = "" if meter.norm is None else f", divided by the RMS of {meter.norm}"
            log(f"layer {index + 1}/{total} saliency method {meter.method}{norm_note}")
            for note in meter.notes:
                log(f"layer {index + 1}/{total} saliency check: {note}")
            for name in names:
                log(f"layer {index + 1}/{total} " + saliency_report(name, meter.sums[name], layer_counts[name][0],
                                                                    layer_counts[name][1], top))
        # These names hold the experts of the layer. Drop them with the layer, before the next one is built.
        found = moe = router = meter = handle = None
        del layer
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        layer_seconds.append(round(time.time() - tick, 2))
        log(f"layer {index + 1}/{total} done in {layer_seconds[-1]} s, read {source.bytes_read / 1e9:.2f} GB")
        if state_dir is not None:
            save_progress(state_dir, key, index, counts, states, checkpoint_every, last, sums, tag)
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
        if saliency:
            skill_sums = {}
            general_sums = {}
            for layer_index in counts:
                skill_sums[layer_index] = (sums[layer_index][skill],) + tuple(counts[layer_index][skill])
                general_sums[layer_index] = (sums[layer_index]["general"],) + tuple(counts[layer_index]["general"])
            add_saliency(maps[skill], skill_sums, general_sums, top=top)
    info = {"layers_total": total, "layers_done": last - first_layer, "bytes_read": source.bytes_read,
            "requests": source.requests, "seconds": round(time.time() - started, 1), "layer_seconds": layer_seconds,
            "texts": len(items), "layout": layout.family, "layers_with_routers": sorted(counts.keys())}
    if saliency:
        info["saliency_method"] = tag[0] if tag else None
        info["saliency_norm"] = tag[1] if tag else None
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


def save_atomic(value, path):
    """torch.save to a temporary name, then a rename. A container that stops during the write leaves no torn file at `path`."""
    torch.save(value, path + ".tmp")
    os.replace(path + ".tmp", path)


def save_progress(state_dir, key, index, counts, states, checkpoint_every, last, sums=None, tag=None):
    """Save the counts of layer `index`, and the hidden states when a checkpoint is due.

    sums is the {layer: saliency sums} of a run with saliency, or None. A run with saliency
    writes a "saliency" field for every layer (None for a dense layer), the
    SALIENCY_STATE_VERSION, and the method and norm of the pass (tag). A run without it writes
    none of these fields, so the two kinds of file can be told apart.
    """
    counts_path, state_path = progress_paths(state_dir, index)
    record = {"key": key, "counts": counts.get(index)}
    if sums is not None:
        record["saliency"] = sums.get(index)
        record["saliency_version"] = SALIENCY_STATE_VERSION
        record["saliency_method"] = tag[0] if tag else None
        record["saliency_norm"] = tag[1] if tag else None
    save_atomic(record, counts_path)
    if (index + 1) % checkpoint_every == 0 or index + 1 == last:
        save_atomic({"key": key, "states": [to_cpu_state(state) for state in states]}, state_path)


def clear_state(state_dir):
    """Delete the resume files of state_dir, so the next run starts at the first layer. Returns the number of files."""
    removed = 0
    if not os.path.isdir(state_dir):
        return removed
    for name in sorted(os.listdir(state_dir)):
        if re.match(r"^(counts|state)_\d{4}\.pt(\.tmp)?$", name):
            os.remove(os.path.join(state_dir, name))
            removed += 1
    return removed


def to_device_state(state, device):
    """Move a saved hidden state (a tensor, a tuple of them, or None) to the device of the layers."""
    if isinstance(state, (tuple, list)):
        return tuple(to_device_state(part, device) for part in state)
    if state is None:
        return None
    return state.to(device)


def to_cpu_state(state):
    if isinstance(state, (tuple, list)):
        return tuple(to_cpu_state(part) for part in state)
    if state is None:
        return None
    return state.cpu()


def load_saved(path, log=None):
    """The object in one resume file, or None when the file is missing or does not load.

    A file that does not load is a torn write: older code wrote the files in place, and a
    container that stops during a write leaves half a file.
    """
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception as error:
        if log is not None:
            log(f"resume skips {path}: it does not load ({error!r})"[:300])
        return None


def resume_state(state_dir, key, items, first_layer, last, log=None):
    """(start layer, records, states) from the newest usable checkpoint of the same texts.

    records is {layer: the saved counts record} of every layer before the start layer. A
    checkpoint is usable when its hidden state file and the counts file of every layer up to it
    load and carry the key. Otherwise the next older checkpoint is tried, and with none left the
    run starts at first_layer.
    """
    for index in range(last - 1, first_layer - 1, -1):
        _, state_path = progress_paths(state_dir, index)
        saved = load_saved(state_path, log)
        if saved is None or saved.get("key") != key or len(saved["states"]) != len(items):
            continue
        records = {}
        for layer_index in range(first_layer, index + 1):
            entry = load_saved(progress_paths(state_dir, layer_index)[0], log)
            if entry is None or entry.get("key") != key:
                records = None
                break
            records[layer_index] = entry
        if records is not None:
            return index + 1, records, list(saved["states"])
    return first_layer, {}, [None] * len(items)


def router_shape(layer_counts):
    """(top_k, num_experts) of one MoE layer, from its saved pick counts: every token picks top_k experts."""
    picks, tokens = next(iter(layer_counts.values()))
    return int(round(float(picks.sum()) / max(1, tokens))), int(picks.numel())


def stale_state_advice(state_dir, first_layer):
    if state_dir is None:
        return ""
    return (f" Or delete {state_dir} (--fresh in modal_giant.py, or pass another state_dir) and run again from "
            f"layer {first_layer} to measure saliency in every layer.")


def saved_saliency(state_dir, records, first_layer):
    """({layer: saliency sums}, (method, norm) or None) of the resumed MoE layers, from their counts records.

    Raises ValueError when a record has no saliency (older code, or a run with saliency off, wrote
    it) or has saliency of another SALIENCY_STATE_VERSION (other saliency code wrote it). Such
    layers cannot go into one result with new ones, and measuring them again needs their layers.
    """
    sums = {}
    tag = None
    for layer_index in sorted(records):
        entry = records[layer_index]
        if entry["counts"] is None:
            continue
        counts_path, _ = progress_paths(state_dir, layer_index)
        if entry.get("saliency") is None:
            raise ValueError(
                f"{counts_path} holds pick counts without saliency: older code or a run with saliency off wrote "
                f"it. Its layers cannot be mixed with layers that have saliency. Either run again with saliency "
                f"off (saliency=False, or --no-saliency in modal_giant.py) to finish with pick counts alone, or "
                f"delete {state_dir} (--fresh in modal_giant.py, or pass another state_dir) and run again from "
                f"layer {first_layer} to measure saliency in every layer.")
        if entry.get("saliency_version") != SALIENCY_STATE_VERSION:
            raise ValueError(
                f"{counts_path} holds saliency of state version {entry.get('saliency_version')}, and this code "
                f"writes version {SALIENCY_STATE_VERSION}: other saliency code wrote it, so its sums cannot be "
                f"mixed with new layers or reported as they are. " + SALIENCY_OFF
                + stale_state_advice(state_dir, first_layer))
        sums[layer_index] = entry["saliency"]
        if tag is None:
            tag = (entry.get("saliency_method"), entry.get("saliency_norm"))
    return sums, tag


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


def estimate_hours(total_bytes, layer_count, mb_per_second=400.0, seconds_per_layer=20.0, compute_factor=1.0):
    """(download hours, compute hours, total hours) from assumed rates. Both rates are guesses.

    compute_factor multiplies the compute hours. Pass SALIENCY_COMPUTE_FACTOR for a run with
    saliency. It is an upper bound: seconds_per_layer also covers the decode, which runs once.
    """
    download = total_bytes / (mb_per_second * 1e6) / 3600.0
    compute = layer_count * seconds_per_layer * compute_factor / 3600.0
    return download, compute, download + compute
