"""Lineage probe: how can a source model be compiled into a base model.

The probe never downloads a model. It reads the config, the tokenizer files,
the card, and the safetensors headers, then compares a few small slices of
matching tensors by byte-range reads.

    facts(model_id)                 what the model is (cached to library/lineage/facts)
    probe(base_id, source_id)       cosine and relative L2 on sampled tensors
    relation(base_id, source_id)    kind, evidence, flags
    methods(relation, want)         ordered compile methods valid for the relation

All file access goes through the small Source interface (files, read_file,
range_reader, revision, card). LocalSource reads a directory. HubSource reads
the Hugging Face Hub. Tests use LocalSource through the same code path.

Command line:
    python -m wsp.lineage relation BASE SOURCE
    python -m wsp.lineage facts MODEL
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

from wsp import fetch

SCHEMA = 3
SAMPLE_BYTES = 4 * 1024 * 1024
DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "library", "lineage", "facts"
)

# Thresholds from the evidence in results/README.md.
SAME_REL_L2 = 1e-6
FINE_TUNE_COSINE = 0.9
ANCESTOR_BAND_TOP = 0.999
ANCESTOR_COSINE = 0.99
SMALL_DELTA = 0.01
LAYER_MIN_COSINE = 0.8
QUANT_NAMES = (".qweight", ".scales", ".weight_scale_inv", ".weight_scale", ".qzeros", ".g_idx")

ELEMENT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}

MODEL_TYPE_ALIASES = {
    "qwen2_moe": "qwen2",
    "qwen3_moe": "qwen3",
    "mixtral": "mistral",
    "granitemoe": "granite",
    "lfm2_moe": "lfm2",
    "olmoe": "olmo",
    "phimoe": "phi3",
}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class Source:
    """The small interface the lineage code reads files through."""

    model_id = ""
    kind = ""

    def revision(self):
        """Return a string that changes when the model files change."""
        raise NotImplementedError

    def files(self):
        """Return the file names in the model root."""
        raise NotImplementedError

    def read_file(self, name):
        """Return the bytes of a small file, or None when it is absent."""
        raise NotImplementedError

    def range_reader(self, name):
        """Return read_range(offset, length) for a file."""
        raise NotImplementedError

    def card(self):
        """Return {"base_model": [...], "base_model_relation": str, "license": str}."""
        raise NotImplementedError


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _parse_front_matter(text):
    """Return the YAML front matter of a model card as a dict."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    import yaml

    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _card_from_meta(meta, tags):
    """Build the card dict from card metadata and Hub tags."""
    base_models = _as_list(meta.get("base_model"))
    relation = ""
    license_name = meta.get("license")
    for tag in tags:
        if tag.startswith("base_model:"):
            parts = tag.split(":", 2)
            if len(parts) == 3:
                if parts[2] not in base_models:
                    base_models.append(parts[2])
                if relation == "":
                    relation = parts[1]
            elif parts[1] not in base_models:
                base_models.append(parts[1])
        if tag.startswith("license:") and not license_name:
            license_name = tag[len("license:"):]
    return {
        "base_model": base_models,
        "base_model_relation": relation,
        "license": license_name if license_name else None,
    }


class LocalSource(Source):
    """A model in a local directory."""

    kind = "local"

    def __init__(self, path, model_id=None):
        self.path = os.path.abspath(path)
        self.model_id = model_id if model_id else path

    def revision(self):
        entries = []
        for name in sorted(os.listdir(self.path)):
            full = os.path.join(self.path, name)
            if os.path.isfile(full):
                stat = os.stat(full)
                entries.append([name, stat.st_size, stat.st_mtime_ns])
        blob = json.dumps(entries).encode("utf-8")
        return "local-" + hashlib.sha256(blob).hexdigest()[:16]

    def files(self):
        names = []
        for name in sorted(os.listdir(self.path)):
            if os.path.isfile(os.path.join(self.path, name)):
                names.append(name)
        return names

    def read_file(self, name):
        full = os.path.join(self.path, name)
        if not os.path.isfile(full):
            return None
        with open(full, "rb") as f:
            return f.read()

    def range_reader(self, name):
        return fetch.local_range_reader(os.path.join(self.path, name))

    def card(self):
        data = self.read_file("README.md")
        meta = {}
        if data is not None:
            meta = _parse_front_matter(data.decode("utf-8", errors="replace"))
        return _card_from_meta(meta, [])


def _retrying(read_range, tries=5, pause=5.0):
    """Wrap a range reader: back off and retry on HTTP 429, 5xx, and timeouts."""

    def read(offset, length):
        for attempt in range(tries):
            try:
                return read_range(offset, length)
            except Exception as error:
                text = str(error)
                transient = "status 429" in text or "status 5" in text or "imed out" in text or "Connection" in text
                if not transient or attempt == tries - 1:
                    raise
                time.sleep(pause * (attempt + 1) * (4 if "status 429" in text else 1))

    return read


class HubSource(Source):
    """A model on the Hugging Face Hub.

    The three methods _info, _download, and _make_reader are the only places
    that touch the network, so a test can override them.
    """

    kind = "hub"

    def __init__(self, model_id):
        self.model_id = model_id
        self._info_cache = None

    def _info(self):
        if self._info_cache is None:
            from huggingface_hub import HfApi

            self._info_cache = HfApi().model_info(self.model_id)
        return self._info_cache

    def _download(self, name):
        """Return the bytes of one small file, or None when it is absent."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        try:
            path = hf_hub_download(self.model_id, name, revision=self.revision())
        except EntryNotFoundError:
            return None
        with open(path, "rb") as f:
            return f.read()

    def _make_reader(self, name):
        from huggingface_hub import get_token, hf_hub_url

        url = hf_hub_url(self.model_id, name, revision=self.revision())
        return _retrying(fetch.http_range_reader(url, token=get_token()))

    def revision(self):
        return self._info().sha

    def files(self):
        names = []
        for sibling in self._info().siblings or []:
            if "/" not in sibling.rfilename:
                names.append(sibling.rfilename)
        return sorted(names)

    def read_file(self, name):
        if name not in self.files():
            return None
        return self._download(name)

    def range_reader(self, name):
        return self._make_reader(name)

    def card(self):
        info = self._info()
        meta = {}
        if info.card_data is not None:
            meta = info.card_data.to_dict()
        return _card_from_meta(meta, list(info.tags or []))


def open_source(model_id):
    """Open a local directory or a Hub id."""
    if os.path.isdir(model_id):
        return LocalSource(model_id)
    return HubSource(model_id)


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()[:16]


def _hash_json(obj):
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return _hash_bytes(blob)


def _first(cfg, names):
    for name in names:
        if cfg.get(name) is not None:
            return cfg[name]
    return None


def _rope(cfg):
    """Return (rope_theta, rope_local_base_freq), None when unknown."""
    theta = cfg.get("rope_theta")
    local = cfg.get("rope_local_base_freq")
    params = cfg.get("rope_parameters")
    if isinstance(params, dict):
        if "rope_theta" in params:
            theta = params["rope_theta"]
        if isinstance(params.get("full_attention"), dict):
            theta = params["full_attention"].get("rope_theta", theta)
        if isinstance(params.get("sliding_attention"), dict):
            local = params["sliding_attention"].get("rope_theta", local)
    scaling = cfg.get("rope_scaling")
    if theta is None and isinstance(scaling, dict):
        theta = scaling.get("rope_theta")
    return theta, local


def normalize_config(raw):
    """Pull the fields lineage needs out of a transformers config dict."""
    core = dict(raw)
    nested = raw.get("text_config")
    if isinstance(nested, dict):
        core.update(nested)
    theta, local = _rope(core)
    hidden = core.get("hidden_size")
    heads = core.get("num_attention_heads")
    head_dim = core.get("head_dim")
    if head_dim is None and hidden and heads:
        head_dim = hidden // heads
    experts = _first(core, ["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"])
    expert_size = core.get("moe_intermediate_size")
    mlp_size = _first(core, ["intermediate_size", "block_ff_dim"])
    if expert_size is None and experts:
        expert_size = core.get("intermediate_size")
    scaling = core.get("rope_scaling")
    return {
        "model_type": core.get("model_type"),
        "architectures": raw.get("architectures") or [],
        "layers": _first(core, ["num_hidden_layers", "n_layer", "num_layers"]),
        "hidden": hidden,
        "heads": heads,
        "kv_heads": core.get("num_key_value_heads"),
        "head_dim": head_dim,
        "mlp_size": mlp_size,
        "vocab": core.get("vocab_size"),
        "rope_theta": theta,
        "rope_local_base_freq": local,
        "rope_scaling": scaling if isinstance(scaling, dict) else None,
        "max_positions": core.get("max_position_embeddings"),
        "tied_embeddings": core.get("tie_word_embeddings"),
        "experts": experts,
        "top_k": _first(core, ["num_experts_per_tok", "moe_top_k", "top_k"]),
        "expert_size": expert_size,
        "shared_experts": core.get("n_shared_experts") or core.get("num_shared_experts"),
    }


def _token_text(value):
    if isinstance(value, dict):
        return value.get("content")
    return value


def _merge_text(merge):
    """A BPE merge is "a b" in older files and ["a", "b"] in newer ones."""
    if isinstance(merge, (list, tuple)):
        return " ".join(merge)
    return merge


def _tokenizer_facts(src, files):
    """Return the tokenizer identity: file hashes, template hash, special tokens."""
    result = {
        "kind": "none",
        "core_hash": None,
        "pre_hash": None,
        "file_hash": None,
        "added_hash": None,
        "vocab_size": None,
        "chat_template_hash": None,
        "special_tokens": {},
    }
    raw = src.read_file("tokenizer.json") if "tokenizer.json" in files else None
    if raw is not None:
        data = json.loads(raw.decode("utf-8"))
        model = data.get("model") or {}
        added_ids = {token.get("id") for token in data.get("added_tokens") or []}
        vocab_map = model.get("vocab")
        if isinstance(vocab_map, dict):
            # Ordinary tokens only: an added token may sit in the vocab under a reserved name.
            vocab_map = {tok: tid for tok, tid in vocab_map.items() if tid not in added_ids}
        core = {
            "type": model.get("type"),
            "vocab": vocab_map,
            "merges": [_merge_text(m) for m in model.get("merges") or []],
        }
        result["pre_hash"] = _hash_json({"normalizer": data.get("normalizer"), "pre_tokenizer": data.get("pre_tokenizer")})
        added = []
        for token in data.get("added_tokens") or []:
            added.append([token.get("content"), token.get("id"), bool(token.get("special"))])
        added.sort(key=lambda row: (row[1] is None, row[1], row[0]))
        result["kind"] = "tokenizer.json"
        result["core_hash"] = _hash_json(core)
        result["file_hash"] = _hash_bytes(raw)
        result["added_hash"] = _hash_json(added)
        vocab = model.get("vocab")
        if isinstance(vocab, dict):
            result["vocab_size"] = len(vocab) + len(added)
    else:
        parts = []
        for name in ("vocab.json", "merges.txt", "tokenizer.model", "vocab.txt", "spiece.model"):
            if name in files:
                parts.append(name.encode("utf-8") + src.read_file(name))
        if parts:
            digest = hashlib.sha256()
            for part in parts:
                digest.update(part)
            result["kind"] = "vocab_files"
            result["core_hash"] = digest.hexdigest()[:16]
            result["file_hash"] = result["core_hash"]

    config_raw = src.read_file("tokenizer_config.json") if "tokenizer_config.json" in files else None
    tok_cfg = json.loads(config_raw.decode("utf-8")) if config_raw is not None else {}
    template = tok_cfg.get("chat_template")
    if template is None and "chat_template.jinja" in files:
        template = src.read_file("chat_template.jinja").decode("utf-8")
    if template is None and "chat_template.json" in files:
        template = json.loads(src.read_file("chat_template.json").decode("utf-8")).get("chat_template")
    if template is not None:
        result["chat_template_hash"] = _hash_json(template)
    special = {}
    for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = _token_text(tok_cfg.get(name))
        if value is not None:
            special[name] = value
    if len(special) == 0 and "special_tokens_map.json" in files:
        smap = json.loads(src.read_file("special_tokens_map.json").decode("utf-8"))
        for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
            value = _token_text(smap.get(name))
            if value is not None:
                special[name] = value
    result["special_tokens"] = special
    return result


def _safetensors_facts(src, files):
    """Read the safetensors headers. Return (tensors, shards, has_index)."""
    shard_names = []
    has_index = False
    if "model.safetensors.index.json" in files:
        index = json.loads(src.read_file("model.safetensors.index.json").decode("utf-8"))
        for shard in index.get("weight_map", {}).values():
            if shard not in shard_names:
                shard_names.append(shard)
        has_index = True
    elif "model.safetensors" in files:
        shard_names = ["model.safetensors"]
    else:
        for name in files:
            if name.endswith(".safetensors"):
                shard_names.append(name)
    tensors = {}
    shards = {}
    for shard in sorted(shard_names):
        header, data_start = fetch.read_header(src.range_reader(shard))
        size = 0
        for name, entry in header.items():
            begin, end = entry["data_offsets"]
            tensors[name] = [shard, entry["dtype"], list(entry["shape"]), begin, end]
            if end > size:
                size = end
        shards[shard] = {"data_start": data_start, "size": data_start + size}
    return tensors, shards, has_index


def _build_facts(src):
    files = src.files()
    raw_config = {}
    config_bytes = src.read_file("config.json") if "config.json" in files else None
    if config_bytes is not None:
        raw_config = json.loads(config_bytes.decode("utf-8"))
    tensors, shards, has_index = _safetensors_facts(src, files)
    params = 0
    nbytes = 0
    for entry in tensors.values():
        count = 1
        for dim in entry[2]:
            count *= dim
        params += count
        nbytes += entry[4] - entry[3]
    card = src.card()
    return {
        "schema": SCHEMA,
        "model_id": src.model_id,
        "source_kind": src.kind,
        "revision": src.revision(),
        "config": normalize_config(raw_config),
        "tokenizer": _tokenizer_facts(src, files),
        "license": card.get("license"),
        "base_model": card.get("base_model", []),
        "base_model_relation": card.get("base_model_relation", ""),
        "files": files,
        "safetensors_index": has_index,
        "params": params,
        "bytes": nbytes,
        "shards": shards,
        "tensors": tensors,
    }


def cache_path(model_id, cache_dir):
    """Return the cache file path for a model id."""
    if os.path.isdir(model_id):
        absolute = os.path.abspath(model_id)
        digest = hashlib.sha256(absolute.encode("utf-8")).hexdigest()[:12]
        name = "local__%s__%s" % (os.path.basename(absolute), digest)
    else:
        name = model_id.replace("/", "__")
    return os.path.join(cache_dir, name + ".json")


def facts(model_id, cache_dir=DEFAULT_CACHE_DIR, refresh=False, opener=None):
    """Return the facts of a model. Cache them when cache_dir is not None.

    A cached file is used when its revision matches the model's current
    revision, or when the revision cannot be read (offline).
    """
    opener = opener if opener else open_source
    src = opener(model_id)
    path = cache_path(model_id, cache_dir) if cache_dir else None
    cached = None
    if path and os.path.exists(path) and not refresh:
        with open(path) as f:
            cached = json.load(f)
        if cached.get("schema") != SCHEMA:
            cached = None
    if cached is not None:
        try:
            current = src.revision()
        except Exception:
            current = None
        if current is None or current == cached["revision"]:
            cached["from_cache"] = True
            return cached
    result = _build_facts(src)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, separators=(",", ":"))
    result["from_cache"] = False
    return result


# ---------------------------------------------------------------------------
# Tensor names
# ---------------------------------------------------------------------------

AUX_WORDS = ("vision", "visual", "audio", "multi_modal", "mm_projector", "projector", "image_newline")
PREFIXES = ("model.", "language_model.", "transformer.", "backbone.")
LAYER_RE = re.compile(r"^(?:layers|h)\.(\d+)\.(.+)$")
MLP_DOWN_RE = re.compile(r"^(mlp\.down_proj|mlp\.c_proj|mlp\.fc2|feed_forward\.w2|feed_forward\.down_proj)\.weight$")
ATTN_Q_RE = re.compile(r"^(self_attn\.q_proj|self_attn\.qkv_proj|attention\.q_proj|attn\.q_proj|self_attn\.query_key_value)\.weight$")
EMBED_NAMES = ("embed_tokens.weight", "wte.weight", "tok_embeddings.weight", "embeddings.word_embeddings.weight")
HEAD_NAMES = ("lm_head.weight", "output.weight")
NORM_NAMES = ("norm.weight", "final_layernorm.weight", "ln_f.weight", "embedding_norm.weight", "final_norm.weight", "norm_f.weight")


def strip_prefix(name):
    """Remove wrapper prefixes: model., language_model., transformer., backbone."""
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name


def canonical(name):
    """Return (key, role, layer) for a tensor name, or None for auxiliary tensors.

    The key is the same for model.layers.3.mlp.down_proj.weight and
    model.language_model.layers.3.mlp.down_proj.weight.
    """
    lowered = name.lower()
    for word in AUX_WORDS:
        if word in lowered:
            return None
    stripped = strip_prefix(name)
    if stripped in EMBED_NAMES:
        return "embed", "embed", None
    if stripped in HEAD_NAMES:
        return "lm_head", "lm_head", None
    if stripped in NORM_NAMES:
        return "final_norm", "final_norm", None
    match = LAYER_RE.match(stripped)
    if match:
        layer = int(match.group(1))
        rest = match.group(2)
        role = None
        if "experts" not in rest:
            if MLP_DOWN_RE.match(rest):
                role = "mlp_down"
            elif ATTN_Q_RE.match(rest):
                role = "attn_q"
        return "layers.%d.%s" % (layer, rest), role, layer
    return stripped, None, None


def _key_table(f):
    """Return key -> (tensor name, role, layer) for a facts dict."""
    table = {}
    for name in f["tensors"]:
        info = canonical(name)
        if info is None:
            continue
        key, role, layer = info
        if key not in table:
            table[key] = (name, role, layer)
    return table


def _pick_three(layers):
    """Pick an early, a middle, and a late layer from a sorted list."""
    if len(layers) == 0:
        return []
    count = len(layers)
    picks = []
    for index in (count // 8, count // 2, (7 * count) // 8):
        layer = layers[min(index, count - 1)]
        if layer not in picks:
            picks.append(layer)
    return picks


def _float_dtype(f, name):
    return f["tensors"][name][1] in ELEMENT_BYTES


def _auto_keys(base_table, source_table, base_facts):
    """Choose the tensor keys to compare. Return a list of keys."""
    keys = []
    for fixed in ("embed", "final_norm", "lm_head"):
        if fixed in base_table and fixed in source_table:
            keys.append(fixed)
    for role in ("mlp_down", "attn_q"):
        layers = []
        for key, (name, key_role, layer) in base_table.items():
            if key_role == role and key in source_table:
                layers.append(layer)
        layers.sort()
        for layer in _pick_three(layers):
            for key, (name, key_role, key_layer) in base_table.items():
                if key_role == role and key_layer == layer and key in source_table:
                    keys.append(key)
                    break
    matrices = 0
    for key in keys:
        if key not in ("final_norm",) and len(base_facts["tensors"][base_table[key][0]][2]) == 2:
            matrices += 1
    if matrices <= 1:
        # Fallback: the first tensors that exist in both with equal shapes.
        extra = 0
        for key, (name, role, layer) in base_table.items():
            if key in keys or key not in source_table:
                continue
            shape = base_facts["tensors"][name][2]
            if len(shape) != 2:
                continue
            keys.append(key)
            extra += 1
            if extra >= 4:
                break
    return keys


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def _read_window(src, f, name, start, count):
    """Read count elements from a flat offset of a tensor. Return a float64 vector."""
    shard, dtype, shape, begin, end = f["tensors"][name]
    size = ELEMENT_BYTES[dtype]
    data_start = f["shards"][shard]["data_start"]
    reader = _reader_for(src, shard)
    data = reader(data_start + begin + start * size, count * size)
    tensor = fetch._tensor_from_bytes(data, dtype, [count])
    return tensor.double()


def _reader_for(src, shard):
    """Return a range reader for a shard, kept on the source object."""
    readers = src.__dict__.setdefault("_readers", {})
    if shard not in readers:
        readers[shard] = src.range_reader(shard)
    return readers[shard]


def _compare_pair(base_src, base_facts, source_src, source_facts, key, base_name, source_name, role, sample_bytes):
    """Compare one tensor pair. Return a result row or a skip row."""
    base_shape = base_facts["tensors"][base_name][2]
    source_shape = source_facts["tensors"][source_name][2]
    base_dtype = base_facts["tensors"][base_name][1]
    source_dtype = source_facts["tensors"][source_name][1]
    if base_dtype not in ELEMENT_BYTES or source_dtype not in ELEMENT_BYTES:
        return {"key": key, "skipped": "dtype %s/%s not float" % (base_dtype, source_dtype)}
    shape = list(base_shape)
    rows_shared = None
    if base_shape != source_shape:
        row_roles = ("embed", "lm_head")
        if role in row_roles and len(base_shape) == 2 and len(source_shape) == 2 and base_shape[1] == source_shape[1]:
            shape = [min(base_shape[0], source_shape[0]), base_shape[1]]
            rows_shared = shape[0]
        else:
            return {"key": key, "skipped": "shape %s against %s" % (base_shape, source_shape)}
    total = 1
    for dim in shape:
        total *= dim
    widest = max(ELEMENT_BYTES[base_dtype], ELEMENT_BYTES[source_dtype])
    budget = max(1, sample_bytes // widest)
    take = min(total, budget)
    start = 0
    if take < total:
        if len(shape) == 2 and shape[1] <= take:
            rows = take // shape[1]
            start_row = (shape[0] - rows) // 2
            start = start_row * shape[1]
            take = rows * shape[1]
        else:
            start = (total - take) // 2
    a = _read_window(base_src, base_facts, base_name, start, take)
    b = _read_window(source_src, source_facts, source_name, start, take)
    norm_a = float(a.norm())
    norm_b = float(b.norm())
    if norm_a == 0.0 or norm_b == 0.0:
        cosine = 1.0 if norm_a == norm_b else 0.0
    else:
        cosine = float(a.dot(b)) / (norm_a * norm_b)
    rel = float((a - b).norm()) / norm_a if norm_a > 0 else (0.0 if norm_b == 0 else float("inf"))
    row = {
        "key": key,
        "base_name": base_name,
        "source_name": source_name,
        "role": role,
        "shape": shape,
        "elements": take,
        "kind": "matrix" if len(shape) >= 2 else "vector",
        "cosine": cosine,
        "rel_l2": rel,
    }
    if rows_shared is not None:
        row["shared_rows"] = rows_shared
        row["rows"] = [base_shape[0], source_shape[0]]
    return row


def probe(base_id, source_id, tensors=None, sample_bytes=SAMPLE_BYTES, cache_dir=DEFAULT_CACHE_DIR, opener=None):
    """Compare sampled slices of matching tensors of two models.

    tensors: optional list of tensor keys or raw base tensor names. Without
    it the probe takes the embeddings, the final norm, and an early, a
    middle, and a late layer's MLP down projection and attention q
    projection.
    """
    opener = opener if opener else open_source
    base_facts = facts(base_id, cache_dir=cache_dir, opener=opener)
    source_facts = facts(source_id, cache_dir=cache_dir, opener=opener)
    return probe_facts(base_facts, source_facts, opener, tensors=tensors, sample_bytes=sample_bytes)


def probe_facts(base_facts, source_facts, opener, tensors=None, sample_bytes=SAMPLE_BYTES):
    """Same as probe, with the facts already in hand."""
    base_table = _key_table(base_facts)
    source_table = _key_table(source_facts)
    if tensors is None:
        keys = _auto_keys(base_table, source_table, base_facts)
    else:
        keys = []
        for item in tensors:
            if item in base_table:
                keys.append(item)
                continue
            info = canonical(item)
            if info is not None:
                keys.append(info[0])
    base_src = opener(base_facts["model_id"])
    source_src = opener(source_facts["model_id"])
    rows = []
    skipped = []
    for key in keys:
        if key not in base_table or key not in source_table:
            skipped.append({"key": key, "skipped": "missing in one model"})
            continue
        base_name, role, layer = base_table[key]
        source_name = source_table[key][0]
        row = _compare_pair(base_src, base_facts, source_src, source_facts, key, base_name, source_name, role, sample_bytes)
        if "skipped" in row:
            skipped.append(row)
        else:
            rows.append(row)
    return summarize(rows, skipped)


def summarize(rows, skipped):
    """Add the mean cosine and relative L2 over the weight matrices."""
    matrices = []
    for row in rows:
        if row["kind"] == "matrix":
            matrices.append(row)
    summary = {
        "tensors": rows,
        "skipped": skipped,
        "n_compared": len(rows),
        "n_matrices": len(matrices),
        "mean_cosine": None,
        "mean_rel_l2": None,
        "min_cosine": None,
        "max_rel_l2": None,
        "embed_cosine": None,
        "n_layer_matrices": 0,
        "layer_mean_cosine": None,
        "layer_min_cosine": None,
    }
    layer_rows = [r for r in matrices if str(r["key"]).startswith("layers.")]
    summary["n_layer_matrices"] = len(layer_rows)
    if len(layer_rows) > 0:
        summary["layer_mean_cosine"] = sum(r["cosine"] for r in layer_rows) / len(layer_rows)
        summary["layer_min_cosine"] = min(r["cosine"] for r in layer_rows)
    if len(matrices) > 0:
        summary["mean_cosine"] = sum(r["cosine"] for r in matrices) / len(matrices)
        summary["mean_rel_l2"] = sum(r["rel_l2"] for r in matrices) / len(matrices)
    if len(rows) > 0:
        summary["min_cosine"] = min(r["cosine"] for r in rows)
        summary["max_rel_l2"] = max(r["rel_l2"] for r in rows)
    for row in rows:
        if row["key"] == "embed":
            summary["embed_cosine"] = row["cosine"]
    return summary


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------


def family(cfg):
    """Return the architecture family: model type with _text and MoE suffixes folded."""
    model_type = cfg.get("model_type") or ""
    if model_type.endswith("_text"):
        model_type = model_type[: -len("_text")]
    return MODEL_TYPE_ALIASES.get(model_type, model_type)


def _dims(cfg):
    keys = ("layers", "hidden", "heads", "kv_heads", "head_dim", "mlp_size", "experts", "expert_size")
    return {key: cfg.get(key) for key in keys}


def _layer_shapes(f):
    """Return key -> shape for the layer tensors (no embeddings or head)."""
    shapes = {}
    for name, entry in f["tensors"].items():
        info = canonical(name)
        if info is None or info[2] is None:
            continue
        if info[0] not in shapes:
            shapes[info[0]] = entry[2]
    return shapes


def _same_shapes(base_facts, source_facts):
    """Return (bool, reason) for equal layer geometry."""
    base_cfg = base_facts["config"]
    source_cfg = source_facts["config"]
    base_dims = _dims(base_cfg)
    source_dims = _dims(source_cfg)
    for key in base_dims:
        if base_dims[key] != source_dims[key]:
            return False, "%s differs: %s against %s" % (key, base_dims[key], source_dims[key])
    if bool(base_cfg.get("experts")) != bool(source_cfg.get("experts")):
        return False, "one model is a MoE and the other is dense"
    if len(base_facts["tensors"]) > 0 and len(source_facts["tensors"]) > 0:
        base_shapes = _layer_shapes(base_facts)
        source_shapes = _layer_shapes(source_facts)
        if base_shapes != source_shapes:
            return False, "layer tensor names or shapes differ"
    return True, ""


def is_quantized(f):
    """True when a model stores quantized weights: a non-float tensor dtype or a quantization tensor name."""
    for name, entry in f["tensors"].items():
        if entry[1] not in ELEMENT_BYTES:
            return True
        for word in QUANT_NAMES:
            if name.endswith(word):
                return True
    return False


def _positional_differs(base_cfg, source_cfg):
    for key in ("rope_theta", "rope_local_base_freq"):
        a = base_cfg.get(key)
        b = source_cfg.get(key)
        if a is not None and b is not None and float(a) != float(b):
            return True
    return False


def _vocab_differs(base_facts, source_facts, probe_result):
    a = base_facts["config"].get("vocab")
    b = source_facts["config"].get("vocab")
    if a is not None and b is not None and a != b:
        return True
    if probe_result is not None:
        for row in probe_result["tensors"]:
            if "rows" in row and row["rows"][0] != row["rows"][1]:
                return True
    return False


def _probe_summary(result):
    if result is None:
        return None
    short = {}
    for name in ("n_compared", "n_matrices", "mean_cosine", "mean_rel_l2", "min_cosine", "max_rel_l2", "embed_cosine",
                 "n_layer_matrices", "layer_mean_cosine", "layer_min_cosine"):
        short[name] = result[name]
    short["tensors"] = [
        {"key": r["key"], "cosine": round(r["cosine"], 6), "rel_l2": round(r["rel_l2"], 6)} for r in result["tensors"]
    ]
    short["skipped"] = result["skipped"]
    return short


def _mirrors(model_id):
    """Ungated mirrors of a gated repo, most trusted first."""
    org, _, name = model_id.partition("/")
    if org != "meta-llama" or not name:
        return []
    plain = "Meta-" + name if name.startswith("Llama-") else name
    return ["unsloth/" + plain, "NousResearch/" + plain]


def _ancestor_facts(ancestor, base_id, source_id, cache_dir, opener):
    """Return (facts, id_used, error). Fall back to a mirror when the repo is gated."""
    try:
        return facts(ancestor, cache_dir=cache_dir, opener=opener), ancestor, None
    except Exception as error:
        message = "%s: %s" % (type(error).__name__, str(error).splitlines()[0] if str(error) else "")
        first = message
    for mirror in _mirrors(ancestor):
        if mirror in (base_id, source_id):
            continue
        try:
            return facts(mirror, cache_dir=cache_dir, opener=opener), mirror, None
        except Exception:
            continue
    return None, ancestor, first


def _try_ancestor(base_facts, source_facts, direct_mean, opener, cache_dir, sample_bytes):
    """Look for a common ancestor named in the cards. Return a dict or None."""
    candidates = []
    for ancestor in base_facts["base_model"] + source_facts["base_model"]:
        if ancestor in (base_facts["model_id"], source_facts["model_id"]):
            continue
        if ancestor not in candidates:
            candidates.append(ancestor)
    tried = []
    for ancestor in candidates[:3]:
        entry = {"ancestor": ancestor}
        ancestor_facts, used, failure = _ancestor_facts(
            ancestor, base_facts["model_id"], source_facts["model_id"], cache_dir, opener
        )
        if ancestor_facts is None:
            entry["error"] = failure
            tried.append(entry)
            continue
        if used != ancestor:
            entry["mirror"] = used
        try:
            ok, why = _same_shapes(base_facts, ancestor_facts)
            if not ok:
                entry["error"] = "ancestor shapes: " + why
                tried.append(entry)
                continue
            source_result = probe_facts(ancestor_facts, source_facts, opener, sample_bytes=sample_bytes)
            base_result = probe_facts(ancestor_facts, base_facts, opener, sample_bytes=sample_bytes)
        except Exception as error:
            entry["error"] = "%s: %s" % (type(error).__name__, error)
            tried.append(entry)
            continue
        entry["source_vs_ancestor"] = source_result["layer_mean_cosine"]
        entry["base_vs_ancestor"] = base_result["layer_mean_cosine"]
        tried.append(entry)
        if source_result["layer_mean_cosine"] is None or base_result["layer_mean_cosine"] is None:
            continue
        if (
            source_result["layer_mean_cosine"] >= ANCESTOR_COSINE
            and base_result["layer_mean_cosine"] >= ANCESTOR_COSINE
            and source_result["layer_mean_cosine"] > direct_mean
        ):
            return {"ancestor": used, "tried": tried}
    return {"ancestor": None, "tried": tried}


def relation(base_id, source_id, cache_dir=DEFAULT_CACHE_DIR, opener=None, sample_bytes=SAMPLE_BYTES, probe_result=None):
    """Classify how a source model relates to a base model.

    Returns {"kind", "evidence", "flags", "warnings", ...}. The kinds are
    same, fine_tune, fine_tune_of_ancestor, diverged_same_shape,
    same_family_other_size, moe_same_width, foreign.
    """
    opener = opener if opener else open_source
    base_facts = facts(base_id, cache_dir=cache_dir, opener=opener)
    source_facts = facts(source_id, cache_dir=cache_dir, opener=opener)
    base_cfg = base_facts["config"]
    source_cfg = source_facts["config"]
    base_tok = base_facts["tokenizer"]
    source_tok = source_facts["tokenizer"]

    tokenizer_same = base_tok["core_hash"] is not None and base_tok["core_hash"] == source_tok["core_hash"]
    template_same = base_tok["chat_template_hash"] == source_tok["chat_template_hash"]
    same_family = family(base_cfg) == family(source_cfg) and family(base_cfg) != ""
    positional = _positional_differs(base_cfg, source_cfg)
    added_same = base_tok["added_hash"] == source_tok["added_hash"]
    special_same = base_tok["special_tokens"] == source_tok["special_tokens"]
    pre_same = base_tok.get("pre_hash") == source_tok.get("pre_hash")
    quantized = is_quantized(source_facts) or is_quantized(base_facts)
    flags = {
        "positional_base_differs": positional,
        "vocab_differs": False,
        "tokenizer_same": tokenizer_same,
        "template_same": template_same,
        "added_tokens_same": added_same,
        "tokenizer_strict_same": tokenizer_same and added_same and template_same and special_same,
        "reserved_ids_reused": tokenizer_same and not added_same,
        "pretokenizer_same": pre_same,
        "teacher_usable_for_logit_kd": tokenizer_same and added_same,
        "quantized": quantized,
        "embeddings_mixable": False,
        "same_family": same_family,
    }
    evidence = {
        "base": _dims(base_cfg),
        "source": _dims(source_cfg),
        "base_family": family(base_cfg),
        "source_family": family(source_cfg),
        "rope_theta": [base_cfg.get("rope_theta"), source_cfg.get("rope_theta")],
        "added_tokens_same": added_same,
        "special_tokens_same": base_tok["special_tokens"] == source_tok["special_tokens"],
        "source_license": source_facts["license"],
        "card_links": {
            "source_declares_base": base_id in source_facts["base_model"],
            "base_declares_source": source_id in base_facts["base_model"],
            "shared_ancestors": sorted(set(base_facts["base_model"]) & set(source_facts["base_model"])),
        },
    }
    warnings = []
    if tokenizer_same and not added_same:
        warnings.append("added tokens differ from the base (reserved ids reused or new tokens), so logit distillation from this source is not used")
    if tokenizer_same and not pre_same:
        warnings.append("the pre-tokenizer or normalizer differs from the base although the vocabulary is the same")
    if tokenizer_same and base_tok["special_tokens"] != source_tok["special_tokens"]:
        warnings.append("special tokens differ from the base: %s against %s" % (base_tok["special_tokens"],
                                                                                source_tok["special_tokens"]))
    result = {
        "base": base_id,
        "source": source_id,
        "kind": None,
        "evidence": evidence,
        "flags": flags,
        "warnings": warnings,
        "subtract_reference": None,
        "base_is_moe": bool(base_cfg.get("experts")),
        "source_is_moe": bool(source_cfg.get("experts")),
    }

    # A MoE source with the base's width and tokenizer comes first, since a
    # MoE model type differs from the dense one by name.
    if (
        source_cfg.get("experts")
        and not base_cfg.get("experts")
        and base_cfg.get("hidden") is not None
        and base_cfg.get("hidden") == source_cfg.get("hidden")
        and tokenizer_same
    ):
        result["kind"] = "moe_same_width"
        try:
            embed = probe_facts(base_facts, source_facts, opener, tensors=["embed"], sample_bytes=sample_bytes)
            evidence["probe"] = _probe_summary(embed)
            flags["embeddings_mixable"] = embed["embed_cosine"] is not None and embed["embed_cosine"] >= FINE_TUNE_COSINE
        except Exception as error:
            warnings.append("embedding probe failed: %s" % error)
        if not same_family:
            warnings.append("the MoE is a different architecture family from the base")
        flags["vocab_differs"] = _vocab_differs(base_facts, source_facts, None)
        return result

    if not same_family or not tokenizer_same:
        result["kind"] = "foreign"
        if not same_family:
            evidence["reason"] = "different architecture family"
        else:
            evidence["reason"] = "different tokenizer"
        flags["vocab_differs"] = _vocab_differs(base_facts, source_facts, None)
        return result

    if quantized:
        dims_equal = _dims(base_cfg) == _dims(source_cfg)
        result["kind"] = "diverged_same_shape" if dims_equal else "same_family_other_size"
        evidence["reason"] = "quantized weights: the tensors cannot be compared or mixed"
        warnings.append("a quantized model is only usable whole; weights were not compared")
        flags["vocab_differs"] = _vocab_differs(base_facts, source_facts, None)
        return result

    same_shape, why = _same_shapes(base_facts, source_facts)
    if not same_shape:
        result["kind"] = "same_family_other_size"
        evidence["reason"] = why
        flags["vocab_differs"] = _vocab_differs(base_facts, source_facts, None)
        if bool(base_cfg.get("experts")) and not source_cfg.get("experts"):
            warnings.append("the base is a MoE and the source is dense")
        return result

    if probe_result is None:
        probe_result = probe_facts(base_facts, source_facts, opener, sample_bytes=sample_bytes)
    evidence["probe"] = _probe_summary(probe_result)
    flags["vocab_differs"] = _vocab_differs(base_facts, source_facts, probe_result)
    mean = probe_result["layer_mean_cosine"]
    if mean is None:
        result["kind"] = "diverged_same_shape"
        warnings.append("no layer weight matrix was compared, so the source is treated as not mixable")
        return result
    layer_min = probe_result["layer_min_cosine"]
    embed_cos = probe_result["embed_cosine"]
    flags["embeddings_mixable"] = embed_cos is not None and embed_cos >= FINE_TUNE_COSINE
    evidence["mean_cosine"] = mean
    evidence["min_layer_cosine"] = layer_min
    evidence["relative_delta_norm"] = probe_result["mean_rel_l2"]

    if positional:
        result["kind"] = "diverged_same_shape"
        evidence["reason"] = "different positional base (rope_theta)"
        return result
    if probe_result["max_rel_l2"] < SAME_REL_L2:
        result["kind"] = "same"
        return result
    if mean <= FINE_TUNE_COSINE:
        result["kind"] = "diverged_same_shape"
        evidence["reason"] = "mean layer cosine %.3f is at or below %.1f" % (mean, FINE_TUNE_COSINE)
        return result
    if layer_min < LAYER_MIN_COSINE:
        result["kind"] = "diverged_same_shape"
        evidence["reason"] = "a layer matrix has cosine %.3f, under %.1f, so the source is not a uniform fine-tune" % (
            layer_min, LAYER_MIN_COSINE)
        return result

    result["kind"] = "fine_tune"
    if mean < ANCESTOR_BAND_TOP:
        found = _try_ancestor(base_facts, source_facts, mean, opener, cache_dir, sample_bytes)
        evidence["ancestor_search"] = found["tried"]
        if found["ancestor"] is not None:
            result["kind"] = "fine_tune_of_ancestor"
            result["subtract_reference"] = found["ancestor"]
            evidence["ancestor"] = found["ancestor"]
        elif any("error" in t and "shapes" not in t["error"] for t in found["tried"]):
            warnings.append("a common ancestor named in the cards could not be read (gated or missing), so the relation "
                            "is reported as fine_tune")
    if probe_result["mean_rel_l2"] < SMALL_DELTA:
        warnings.append(
            "the difference from the base is small: relative delta norm %.5f is under %.2f, so it carries little"
            % (probe_result["mean_rel_l2"], SMALL_DELTA)
        )
    return result


# ---------------------------------------------------------------------------
# Methods table
# ---------------------------------------------------------------------------

METHOD_INFO = {
    "weave": ("wsp/weave.py", "base_size"),
    "delta": ("wsp/delta.py", "base_size_plus_low_rank"),
    "transplant": ("wsp/transplant.py", "extra_routed_params"),
    "synth": ("wsp/synth.py", "seam_params_plus_donor_blocks"),
    "feedback": ("wsp/feedback.py", "seam_params_plus_donor_blocks"),
    "whole_model": ("route to the whole source model", "source_params_added"),
    "slim": ("wsp/slim.py", "smaller_than_base"),
}


def _entry(name, reason):
    module, cost = METHOD_INFO[name]
    return {"method": name, "module": module, "cost": cost, "reason": reason}


def methods(rel, want="add"):
    """Return the ordered list of compile methods valid for a relation.

    rel is the dict from relation(), or just a kind string. want is "add" to
    bring a skill in from the source, or "remove" to take a skill out.
    """
    if isinstance(rel, str):
        rel = {"kind": rel, "flags": {}, "base_is_moe": False, "subtract_reference": None}
    kind = rel["kind"]
    flags = rel.get("flags", {})
    if want == "remove":
        if rel.get("base_is_moe"):
            reason = "keep the top experts per layer for the skills that stay; drops experts the removed skill uses"
        else:
            reason = "remove layers, neurons, and heads that carry the skill; no donor is needed"
        return [_entry("slim", reason)]
    if want != "add":
        raise ValueError("want must be add or remove")

    out = []
    if kind == "same":
        return out
    if kind == "fine_tune":
        out.append(_entry("weave", "same shapes and positional base, cosine near 1: mix per layer at exactly the base size"))
        out.append(_entry("delta", "same-shape specialist as a switchable low-rank difference"))
        out.append(_entry("whole_model", "fallback: run the source whole as a routed alternative"))
    elif kind == "fine_tune_of_ancestor":
        ref = rel.get("subtract_reference")
        out.append(
            _entry(
                "weave",
                "close enough to mix per layer; to isolate the skill, subtract %s from the source first" % ref,
            )
        )
        out.append(_entry("delta", "take source minus %s as the difference and add it to the base as a switchable delta" % ref))
        out.append(_entry("whole_model", "fallback: run the source whole as a routed alternative"))
    elif kind == "diverged_same_shape":
        why = "weights are far from the base (mixing any part below the whole model gives chance)"
        if flags.get("positional_base_differs"):
            why = "another positional base (rope_theta) and far weights"
        out.append(_entry("synth", why + ": staged seam with translate in, translate out, and a synthesis heal"))
        if flags.get("tokenizer_same", True):
            out.append(_entry("feedback", why + ": block output written into the next token's input"))
        out.append(_entry("whole_model", "route to the whole source model, which keeps its own embeddings and head"))
    elif kind == "same_family_other_size":
        out.append(_entry("synth", "other width or depth: donor layers go through a seam with translation layers"))
        if flags.get("tokenizer_same", True):
            out.append(_entry("feedback", "other width or depth: block output written into the next token's input"))
        out.append(_entry("whole_model", "route to the whole source model"))
    elif kind == "moe_same_width":
        if flags.get("same_family", False):
            out.append(_entry("transplant", "same family and width: routed experts next to each MLP"))
            out.append(_entry("synth", "seam for experts that do not fit the base's MLP slots"))
        else:
            out.append(_entry("synth", "MoE of another family: experts go through a seam"))
        out.append(_entry("whole_model", "route to the whole source model"))
    elif kind == "foreign":
        out.append(_entry("synth", "other tokenizer or family: components need translation in and out"))
        if flags.get("tokenizer_same", False):
            out.append(_entry("feedback", "same tokenizer, other family: block output written into the next token's input"))
        out.append(_entry("whole_model", "route to the whole source model"))
    else:
        raise ValueError("unknown relation kind: %s" % kind)
    if flags.get("quantized"):
        out = [e for e in out if e["method"] == "whole_model"]
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m wsp.lineage")
    sub = parser.add_subparsers(dest="command", required=True)
    p_facts = sub.add_parser("facts", help="print the facts of a model")
    p_facts.add_argument("model")
    p_facts.add_argument("--full", action="store_true", help="include the tensor index")
    p_facts.add_argument("--refresh", action="store_true")
    p_rel = sub.add_parser("relation", help="classify a source against a base")
    p_rel.add_argument("base")
    p_rel.add_argument("source")
    p_rel.add_argument("--sample-bytes", type=int, default=SAMPLE_BYTES)
    p_probe = sub.add_parser("probe", help="print the per-tensor comparison")
    p_probe.add_argument("base")
    p_probe.add_argument("source")
    p_probe.add_argument("--sample-bytes", type=int, default=SAMPLE_BYTES)
    args = parser.parse_args(argv)

    if args.command == "facts":
        out = facts(args.model, refresh=args.refresh)
        if not args.full:
            out = {k: v for k, v in out.items() if k not in ("tensors", "shards")}
            out["n_tensors"] = len(facts(args.model)["tensors"])
    elif args.command == "relation":
        out = relation(args.base, args.source, sample_bytes=args.sample_bytes)
        out["methods"] = {"add": methods(out, "add"), "remove": methods(out, "remove")}
    else:
        out = probe(args.base, args.source, sample_bytes=args.sample_bytes)
    print(json.dumps(_json_safe(out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
