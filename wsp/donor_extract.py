"""Run a big donor once over a fixed prompt set and write its tap memories to a cache.

modal_extract_frontier.py uses these helpers on an 8-GPU node. They are
plain torch and transformers code, so the CPU tests run them on tiny
models.

The steps:
1. choose_shards picks the checkpoint files that hold layers
   0..max_layer and the other weights the run needs (embeddings, the
   stream readout of hyper-connection models). filter_weight_map writes a
   safetensors index that names only those files.
2. truncate_text_config cuts the (text) config to max_layer + 1 layers,
   so the model is built and loaded with no later layers at all.
3. extract_prompts runs the language model on prompts of one exact token
   length per batch (so no batch has pads), stops after the deepest read
   layer, and gives each prompt's states at every read layer to a
   wsp.tap_cache.CacheWriter.

States come from layer_state. For a plain decoder layer that is its
output hidden state. Some donors keep more than one residual stream:
- DeepSeek-V4 and GLM-5.3-Flash (hyper-connections) keep hc_mult streams,
  (batch, seq, hc_mult, hidden). We collapse them with the model's own
  hc_head, the readout it uses before its final norm (a learned mix for
  DeepSeek-V4, a mean for GLM-5.3-Flash).
- Kimi-K3 (attention residuals) returns (prefix_sum, block_residual). We
  mix them with the model's own output attention residual, the readout it
  uses before its final norm.
"""

import re
import sys
import time

import torch

from wsp.parts import find_decoder_layers

LAYER_KEY = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Weights a tap extraction never needs: the output head, multi-token
# prediction layers, and the final norm.
SKIP_KEYS = [re.compile(r"(^|\.)lm_head\."), re.compile(r"(^|\.)mtp\."), re.compile(r"(^|\.)norm\.weight$")]


class _StopRun(Exception):
    """Raised by the hook on the deepest read layer, so later layers never run."""


def layer_number(key):
    """Return the decoder layer number in a checkpoint key, or None for other weights.

    Vision tower weights count as other weights, even when their module
    list is called layers.
    """
    if "visual" in key or "vision" in key:
        return None
    found = LAYER_KEY.search(key)
    if found is None:
        return None
    return int(found.group(1))


def skipped_key(key):
    for pattern in SKIP_KEYS:
        if pattern.search(key):
            return True
    return False


def choose_shards(weight_map, max_layer):
    """Return the sorted checkpoint files that hold layers 0..max_layer or a needed non-layer weight.

    weight_map is the "weight_map" of model.safetensors.index.json (key ->
    file). Files that hold only later layers, the output head, the final
    norm, or multi-token prediction layers are left out.
    """
    files = set()
    for key in weight_map:
        number = layer_number(key)
        if number is None:
            if skipped_key(key):
                continue
            files.add(weight_map[key])
        elif number <= max_layer:
            files.add(weight_map[key])
    return sorted(files)


def filter_weight_map(weight_map, files, max_layer):
    """Return the weight map with only keys that sit in files and belong to layers 0..max_layer or no layer."""
    kept = set(files)
    out = {}
    for key in weight_map:
        if weight_map[key] not in kept:
            continue
        number = layer_number(key)
        if number is not None and number > max_layer:
            continue
        out[key] = weight_map[key]
    return out


def text_config_of(config):
    """Return the config that holds num_hidden_layers for the language model.

    A multimodal config keeps it in text_config. That can be a config
    object or, in some remote code, a plain dict.
    """
    text = getattr(config, "text_config", None)
    if text is not None:
        return text
    return config


def config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def set_config_value(config, name, value):
    if isinstance(config, dict):
        config[name] = value
    else:
        setattr(config, name, value)


def truncate_text_config(config, layers):
    """Cut the language model config to its first layers layers. Return the old layer count.

    Every list attribute with one entry per layer (layer_types,
    mlp_layer_types, indexer_types, and so on) is cut too, so the config
    stays consistent. The model then builds only those layers.
    """
    text = text_config_of(config)
    old = int(config_value(text, "num_hidden_layers"))
    if layers > old:
        raise ValueError(f"the model has {old} layers; cannot keep {layers}")
    names = []
    if isinstance(text, dict):
        names = list(text.keys())
    else:
        names = list(vars(text).keys())
    per_layer = []
    for name in names:
        value = config_value(text, name)
        if isinstance(value, (list, tuple)) and len(value) == old and name != "architectures":
            per_layer.append(name)
    set_config_value(text, "num_hidden_layers", layers)
    for name in per_layer:
        value = config_value(text, name)
        set_config_value(text, name, list(value[:layers]))
    return old


def find_decoder(model, layer):
    """Return (decoder, layers): the module that owns the decoder layer list, and that list."""
    name, layers = find_decoder_layers(model, layer)
    if "." not in name:
        return model, layers
    return model.get_submodule(name.rsplit(".", 1)[0]), layers


def module_device(module, fallback):
    for param in module.parameters():
        return param.device
    return fallback


def layer_state(decoder, output):
    """Return the (batch, seq, hidden) state a decoder layer output stands for. See the module notes."""
    if isinstance(output, tuple):
        if getattr(decoder, "use_attn_residuals", False) and len(output) > 1 and torch.is_tensor(output[1]) \
                and output[1].dim() == 3:
            prefix_sum = output[0]
            block_residual = output[1]
            device = module_device(decoder.output_attn_res_proj, prefix_sum.device)
            return decoder._apply_output_attn_res(prefix_sum.to(device), block_residual.to(device))
        output = output[0]
    if output.dim() == 4:
        head = getattr(decoder, "hc_head", None)
        if head is None:
            return output.mean(dim=2)
        return head(output.to(module_device(head, output.device)))
    return output


def token_ends(tokenizer, text):
    """Tokenize one text with default special tokens. Return (ids, ends, valid) as lists.

    ends holds each token's character end in text, -1 for special tokens.
    With a fast tokenizer the ends are its offsets, the same ones
    run_donor uses. A slow tokenizer (for example Kimi's tiktoken one) has
    no offsets: there each token's end is the number of characters that
    the decode of the text tokens up to it shares with text, so a token
    never claims text past its own end.
    """
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        ids = list(encoded["input_ids"])
        special = list(encoded["special_tokens_mask"])
        ends = []
        valid = []
        for index in range(len(ids)):
            if special[index] == 1:
                ends.append(-1)
                valid.append(False)
            else:
                ends.append(int(encoded["offset_mapping"][index][1]))
                valid.append(True)
        return ids, ends, valid

    ids = list(tokenizer(text)["input_ids"])
    special_ids = set(tokenizer.all_special_ids)
    ends = []
    valid = []
    kept = []
    for token in ids:
        if token in special_ids:
            ends.append(-1)
            valid.append(False)
            continue
        kept.append(token)
        prefix = tokenizer.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        shared = 0
        limit = min(len(prefix), len(text))
        while shared < limit and prefix[shared] == text[shared]:
            shared += 1
        ends.append(shared)
        valid.append(True)
    return ids, ends, valid


def length_batches(lengths, max_batch=64, max_tokens=16384):
    """Group prompt indices into batches of one exact token length.

    Each batch holds at most max_batch prompts and about max_tokens
    tokens (at least one prompt). Batches come shortest first.
    """
    by_length = {}
    for index in range(len(lengths)):
        length = lengths[index]
        if length not in by_length:
            by_length[length] = []
        by_length[length].append(index)
    batches = []
    for length in sorted(by_length):
        size = max(1, min(max_batch, max_tokens // max(length, 1)))
        indices = by_length[length]
        for start in range(0, len(indices), size):
            batches.append(indices[start:start + size])
    return batches


def run_read_layers(decoder, layers, input_ids, read_layers):
    """Run decoder on input_ids (no pads) and return dict layer -> (batch, seq, hidden) state.

    The hook on the deepest read layer stops the run, so later layers
    (and the final norm) never run.
    """
    wanted = sorted(set(read_layers))
    last = wanted[-1]
    store = {}
    handles = []
    for layer in wanted:

        def hook(module, inputs, output, layer=layer):
            store[layer] = layer_state(decoder, output).detach()
            if layer == last:
                raise _StopRun()

        handles.append(layers[layer].register_forward_hook(hook))
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.no_grad():
            decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except _StopRun:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if last not in store:
        raise RuntimeError(f"donor layer {last} did not run")
    return store


def input_device(decoder):
    embeddings = None
    try:
        embeddings = decoder.get_input_embeddings()
    except (AttributeError, NotImplementedError):
        embeddings = getattr(decoder, "embed_tokens", None)
    if embeddings is not None:
        return embeddings.weight.device
    return module_device(decoder, torch.device("cpu"))


def extract_prompts(model, tokenizer, prompts, read_layers, writer, max_batch=64, max_tokens=16384, log_every=20,
                    log=print):
    """Run model's language model on every prompt the writer lacks, and add its states to the writer.

    Prompts go in batches of one exact donor token length, so the donor
    sees no pads: linear attention, compressed attention, and sparse
    indexers then never meet a padding mask. Returns a stats dict.
    """
    decoder, layers = find_decoder(model, max(read_layers))
    device = input_device(decoder)
    todo = []
    seen = set()
    encoded = []
    for prompt in prompts:
        if writer.has(prompt) or prompt in seen:
            continue
        seen.add(prompt)
        todo.append(prompt)
        encoded.append(token_ends(tokenizer, prompt))
    lengths = []
    for ids, ends, valid in encoded:
        lengths.append(len(ids))
    batches = length_batches(lengths, max_batch=max_batch, max_tokens=max_tokens)
    log(f"extract: {len(todo)} prompts to run ({len(prompts) - len(todo)} cached), {len(batches)} batches, "
        f"{sum(lengths)} tokens")

    start = time.time()
    tokens = 0
    for number in range(len(batches)):
        batch = batches[number]
        rows = []
        for index in batch:
            rows.append(encoded[index][0])
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        states = run_read_layers(decoder, layers, input_ids, read_layers)
        host_states = {}
        for layer in states:
            host_states[layer] = states[layer].to(device="cpu", dtype=torch.float32)
        for row in range(len(batch)):
            ids, ends, valid = encoded[batch[row]]
            row_states = {}
            for layer in host_states:
                row_states[layer] = host_states[layer][row]
            writer.add(todo[batch[row]], row_states, torch.tensor(ends), torch.tensor(valid))
        tokens += input_ids.numel()
        if log_every and (number + 1) % log_every == 0:
            seconds = time.time() - start
            log(f"extract: batch {number + 1}/{len(batches)}, {tokens} tokens, {tokens / max(seconds, 1e-9):.0f} tok/s")
    seconds = time.time() - start
    return {"prompts_run": len(todo), "tokens_run": tokens, "batches": len(batches),
            "extract_seconds": round(seconds, 1), "tokens_per_second": round(tokens / max(seconds, 1e-9), 1)}


def patch_remote_mask_calls(model):
    """Make remote code written for transformers 4.x call create_causal_mask with the 5.x arguments.

    Kimi-K3's modeling_kimi_linear.py calls create_causal_mask(input_embeds=...,
    cache_position=...). Transformers 5.23 names the first inputs_embeds and
    has no cache_position. We wrap the function in each loaded remote
    module that imports it. Returns the module names we patched.
    """
    from transformers import masking_utils

    patched = []
    names = set()
    for module in model.modules():
        names.add(type(module).__module__)
    for name in sorted(names):
        if not name.startswith("transformers_modules"):
            continue
        remote = sys.modules.get(name)
        if remote is None or not hasattr(remote, "create_causal_mask"):
            continue
        if getattr(remote.create_causal_mask, "wsp_patched", False):
            continue

        def create_causal_mask(*args, **kwargs):
            if "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            kwargs.pop("cache_position", None)
            return masking_utils.create_causal_mask(*args, **kwargs)

        create_causal_mask.wsp_patched = True
        remote.create_causal_mask = create_causal_mask
        patched.append(name)
    return patched


def count_layer_params(layers, read_layers):
    """Return {str(L): stored parameter count of layers 0..L} for each read layer (packed sizes for quantized weights)."""
    out = {}
    total = 0
    deepest = max(read_layers)
    for index in range(deepest + 1):
        for param in layers[index].parameters():
            total += param.numel()
        if index in read_layers:
            out[str(index)] = total
    return out
