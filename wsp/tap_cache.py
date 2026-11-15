"""Donor memory cache: a donor's hidden states at a few layers, for a fixed set of prompts.

A tap reads only the donor's states at one layer. For a fixed prompt set
(for example MMLU train and test prompts) we can run the donor once, store
those states, and then train and score taps with no donor model loaded.
wsp.tap.CachedDonor reads a cache back.

Layout of a cache directory:

    meta.json                   donor id, layers, hidden size, tokenizer id, dtype, ...
    index.json                  {prompt hash: [shard number, row]}
    shard_00000.safetensors     the tensors of up to shard_size prompts
    shard_00000.json            the prompts of that shard, in row order
    ...

One shard holds these tensors. Prompts have different lengths, so each
prompt's tokens are stored one after another, and offsets says where each
row starts:

    layer_<L>   (tokens, hidden) states at donor layer L, in the cache dtype
    offsets     (rows + 1,) int64; row r is tokens offsets[r]:offsets[r + 1]
    ends        (tokens,) int64 character end of each donor token, -1 for specials
    valid       (tokens,) bool, True for real text tokens

The prompt hash is the sha256 of the prompt text in UTF-8. A shard's JSON
file is written after its tensors, so a shard with a JSON file is complete.
A writer that finds complete shards in its directory keeps them and skips
their prompts (resume after a crash or a timeout).
"""

import hashlib
import json
import os
import time

import torch

FORMAT_VERSION = 1
DEFAULT_SHARD_SIZE = 512
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def prompt_hash(prompt):
    """Return the sha256 hex digest of the prompt text."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def shard_name(number):
    return f"shard_{number:05d}"


def dtype_name(dtype):
    for name in DTYPES:
        if DTYPES[name] == dtype:
            return name
    raise ValueError(f"unsupported cache dtype {dtype}; use one of {sorted(DTYPES)}")


def write_json(path, data):
    """Write JSON to path through a temporary file, so a reader never sees half a file."""
    temp = path + ".tmp"
    with open(temp, "w") as f:
        json.dump(data, f)
    os.replace(temp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def complete_shards(out_dir):
    """Return the sorted shard numbers in out_dir that have both their tensors and their JSON file."""
    numbers = []
    if not os.path.isdir(out_dir):
        return numbers
    for name in os.listdir(out_dir):
        if not name.startswith("shard_") or not name.endswith(".json"):
            continue
        stem = name[:-len(".json")]
        if not os.path.exists(os.path.join(out_dir, stem + ".safetensors")):
            continue
        numbers.append(int(stem[len("shard_"):]))
    numbers.sort()
    return numbers


class CacheWriter:
    """Write a donor memory cache one prompt at a time.

    layers: the donor layers stored for every prompt. hidden_size: the
    donor hidden size. meta: extra JSON data for meta.json (donor id,
    tokenizer id, and so on). on_shard: a function called after each shard
    is written (for example a Modal volume commit).

    add() takes one prompt's states. close() writes the last shard,
    index.json, and meta.json, and returns the meta dict.
    """

    def __init__(self, out_dir, layers, hidden_size, meta=None, shard_size=DEFAULT_SHARD_SIZE,
                 dtype=torch.float16, on_shard=None):
        self.out_dir = out_dir
        self.layers = sorted(set(int(layer) for layer in layers))
        self.hidden_size = int(hidden_size)
        self.extra = dict(meta or {})
        self.shard_size = int(shard_size)
        self.dtype = dtype
        self.dtype_name = dtype_name(dtype)
        self.on_shard = on_shard
        self.pending = []
        self.pending_hashes = set()
        self.done_hashes = set()
        self.started = time.time()
        os.makedirs(out_dir, exist_ok=True)

        meta_path = os.path.join(out_dir, "meta.json")
        if os.path.exists(meta_path):
            old = read_json(meta_path)
            if old["layers"] != self.layers or old["hidden_size"] != self.hidden_size:
                raise ValueError(f"{out_dir} already holds a cache with layers {old['layers']} and hidden size "
                                 f"{old['hidden_size']}; asked for {self.layers} and {self.hidden_size}")
        self.shards = complete_shards(out_dir)
        for number in self.shards:
            info = read_json(os.path.join(out_dir, shard_name(number) + ".json"))
            for prompt in info["prompts"]:
                self.done_hashes.add(prompt_hash(prompt))
        self.next_shard = 0
        if len(self.shards) > 0:
            self.next_shard = self.shards[-1] + 1

    def has(self, prompt):
        """Return True when the prompt is already written or waiting to be written."""
        key = prompt_hash(prompt)
        return key in self.done_hashes or key in self.pending_hashes

    def add(self, prompt, states_by_layer, ends, valid):
        """Add one prompt: states_by_layer is a dict of layer -> (tokens, hidden); ends and valid are (tokens,).

        A prompt that is already in the cache is skipped. Returns True when
        the prompt was added.
        """
        if self.has(prompt):
            return False
        ends = torch.as_tensor(ends).reshape(-1).to(torch.long).cpu()
        valid = torch.as_tensor(valid).reshape(-1).to(torch.bool).cpu()
        count = ends.shape[0]
        if valid.shape[0] != count:
            raise ValueError(f"ends has {count} tokens but valid has {valid.shape[0]}")
        states = {}
        for layer in self.layers:
            if layer not in states_by_layer:
                raise ValueError(f"no states for layer {layer}; the cache stores layers {self.layers}")
            state = states_by_layer[layer].detach()
            if state.dim() != 2 or state.shape[0] != count or state.shape[1] != self.hidden_size:
                raise ValueError(f"layer {layer} states have shape {tuple(state.shape)}, want ({count}, "
                                 f"{self.hidden_size})")
            states[layer] = state.to(device="cpu", dtype=self.dtype).contiguous()
        self.pending.append((prompt, states, ends, valid))
        self.pending_hashes.add(prompt_hash(prompt))
        if len(self.pending) >= self.shard_size:
            self.flush()
        return True

    def flush(self):
        """Write the waiting prompts as one shard."""
        if len(self.pending) == 0:
            return
        from safetensors.torch import save_file

        offsets = [0]
        for prompt, states, ends, valid in self.pending:
            offsets.append(offsets[-1] + ends.shape[0])
        tensors = {"offsets": torch.tensor(offsets, dtype=torch.long)}
        end_parts = []
        valid_parts = []
        for prompt, states, ends, valid in self.pending:
            end_parts.append(ends)
            valid_parts.append(valid)
        tensors["ends"] = torch.cat(end_parts)
        tensors["valid"] = torch.cat(valid_parts)
        for layer in self.layers:
            parts = []
            for prompt, states, ends, valid in self.pending:
                parts.append(states[layer])
            tensors[f"layer_{layer}"] = torch.cat(parts).contiguous()

        name = shard_name(self.next_shard)
        temp = os.path.join(self.out_dir, name + ".safetensors.tmp")
        save_file(tensors, temp)
        os.replace(temp, os.path.join(self.out_dir, name + ".safetensors"))
        prompts = []
        for prompt, states, ends, valid in self.pending:
            prompts.append(prompt)
        # The JSON file goes last: it marks the shard as complete.
        write_json(os.path.join(self.out_dir, name + ".json"), {"prompts": prompts, "tokens": offsets[-1]})

        self.shards.append(self.next_shard)
        self.next_shard += 1
        for prompt in prompts:
            self.done_hashes.add(prompt_hash(prompt))
        self.pending = []
        self.pending_hashes = set()
        if self.on_shard is not None:
            self.on_shard()

    def close(self):
        """Write the last shard, index.json, and meta.json. Return the meta dict."""
        self.flush()
        index = {}
        tokens = 0
        names = []
        for number in self.shards:
            name = shard_name(number)
            names.append(name)
            info = read_json(os.path.join(self.out_dir, name + ".json"))
            tokens += info["tokens"]
            for row in range(len(info["prompts"])):
                index[prompt_hash(info["prompts"][row])] = [number, row]
        write_json(os.path.join(self.out_dir, "index.json"), index)

        meta = dict(self.extra)
        meta["format_version"] = FORMAT_VERSION
        meta["layers"] = self.layers
        meta["hidden_size"] = self.hidden_size
        meta["dtype"] = self.dtype_name
        meta["shard_size"] = self.shard_size
        meta["shards"] = names
        meta["num_prompts"] = len(index)
        meta["num_tokens"] = tokens
        meta["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["write_seconds"] = round(time.time() - self.started, 1)
        write_json(os.path.join(self.out_dir, "meta.json"), meta)
        if self.on_shard is not None:
            self.on_shard()
        return meta


def row_items(value, count, name):
    """Turn a per-prompt list, or a padded (count, m, ...) tensor, into a list of count tensors."""
    if torch.is_tensor(value):
        if value.shape[0] != count:
            raise ValueError(f"{name} has batch {value.shape[0]}, want {count}")
        rows = []
        for row in range(count):
            rows.append(value[row])
        return rows
    rows = list(value)
    if len(rows) != count:
        raise ValueError(f"{name} has {len(rows)} rows, want {count}")
    return rows


def write_cache(out_dir, prompts, layer_states_by_layer, ends, valid, meta, shard_size=DEFAULT_SHARD_SIZE,
                dtype=torch.float16, on_shard=None):
    """Write a donor memory cache for prompts and return its meta dict.

    layer_states_by_layer: dict of layer -> either a list of (tokens_i, d)
    tensors, one per prompt, or one padded (n, m, d) tensor, as
    run_donor_layers returns. ends and valid: a list of (tokens_i,)
    tensors, or padded (n, m) tensors. For padded inputs, each row keeps
    its tokens up to its last valid token, so right pads are dropped. The
    tokens a cache keeps past the last valid one never matter, because no
    host token can read an invalid memory token.

    meta: a dict for meta.json, for example {"donor_id": ..., "tokenizer_id": ...}.
    dtype: the stored dtype (float16 by default; float32 keeps the states exactly).
    """
    prompts = list(prompts)
    count = len(prompts)
    layers = sorted(layer_states_by_layer)
    if len(layers) == 0:
        raise ValueError("write_cache needs states for at least one layer")
    padded = torch.is_tensor(valid)
    end_rows = row_items(ends, count, "ends")
    valid_rows = row_items(valid, count, "valid")
    state_rows = {}
    hidden = None
    for layer in layers:
        state_rows[layer] = row_items(layer_states_by_layer[layer], count, f"layer {layer} states")
        for state in state_rows[layer]:
            hidden = state.shape[-1]

    writer = CacheWriter(out_dir, layers, hidden, meta=meta, shard_size=shard_size, dtype=dtype, on_shard=on_shard)
    for row in range(count):
        row_valid = valid_rows[row].reshape(-1).bool()
        length = row_valid.shape[0]
        if padded:
            # Keep the tokens up to the last valid one; the rest are pads.
            length = 0
            for index in range(row_valid.shape[0]):
                if bool(row_valid[index]):
                    length = index + 1
        states = {}
        for layer in layers:
            states[layer] = state_rows[layer][row][:length]
        writer.add(prompts[row], states, end_rows[row].reshape(-1)[:length], row_valid[:length])
    return writer.close()


def read_meta(path):
    return read_json(os.path.join(path, "meta.json"))


def read_index(path):
    return read_json(os.path.join(path, "index.json"))


def read_shard_rows(path, number):
    """Return (offsets, ends, valid, prompts) for one shard: the small tensors and the prompt list."""
    from safetensors import safe_open

    name = shard_name(number)
    with safe_open(os.path.join(path, name + ".safetensors"), framework="pt") as f:
        offsets = f.get_tensor("offsets")
        ends = f.get_tensor("ends")
        valid = f.get_tensor("valid")
    prompts = read_json(os.path.join(path, name + ".json"))["prompts"]
    return offsets, ends, valid, prompts


def read_shard_layer(path, number, layer):
    """Return the (tokens, hidden) states of one layer in one shard, in the cache dtype."""
    from safetensors import safe_open

    with safe_open(os.path.join(path, shard_name(number) + ".safetensors"), framework="pt") as f:
        return f.get_tensor(f"layer_{layer}")
