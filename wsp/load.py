"""Load a compiled model: wsp.load(path) -> (model, tokenizer).

A compiled directory holds program.json and a tokenizer. A program with
grafts also holds grafts.json and graft_weights.safetensors, plus host/
when masks or task vectors changed the host, and lora.json with
lora_weights.safetensors when it has a LoRA. A program without grafts is
a plain Hugging Face model directory (its LoRA, if any, is merged in).
"""

import json
import os

import torch


# tokenizer_config.json names these for a plain fast tokenizer.
GENERIC_TOKENIZERS = ("TokenizersBackend", "PreTrainedTokenizerFast")


def load_tokenizer(source):
    """Load the tokenizer of a folder or repo the way it was saved.

    For some model types (qwen2 and phi3, for example) Transformers 5 uses
    the family's own tokenizer class even when tokenizer_config.json names
    the generic fast tokenizer. That class builds a new pipeline from the
    vocabulary and ignores the saved tokenizer.json. When the saved class
    is the generic one, we load it with that class, as Transformers 4 did.
    Otherwise AutoTokenizer picks the class.
    """
    import transformers
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    saved = get_tokenizer_config(source).get("tokenizer_class")
    if saved in GENERIC_TOKENIZERS:
        return transformers.PreTrainedTokenizerFast.from_pretrained(source)
    return transformers.AutoTokenizer.from_pretrained(source)


def load(path, device="cpu", dtype=torch.bfloat16):
    import transformers
    from safetensors.torch import load_file

    from wsp.assemble import rebuild_graft
    from wsp.graft import attach_grafts

    tokenizer = load_tokenizer(path)
    grafts_path = os.path.join(path, "grafts.json")
    if not os.path.exists(grafts_path):
        model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        return model.to(device).eval(), tokenizer

    with open(os.path.join(path, "program.json")) as f:
        program = json.load(f)
    host_source = program.get("host") or program.get("base")
    if os.path.isdir(os.path.join(path, "host")):
        host_source = os.path.join(path, "host")
    model = transformers.AutoModelForCausalLM.from_pretrained(host_source, dtype=dtype)

    lora_path = os.path.join(path, "lora.json")
    if os.path.exists(lora_path):
        from wsp.lora import apply_lora, load_lora_state

        with open(lora_path) as f:
            lora = json.load(f)
        apply_lora(model, lora["rank"], lora["alpha"], tuple(lora["targets"]), lora["layers"])
        load_lora_state(model, load_file(os.path.join(path, "lora_weights.safetensors")))

    with open(grafts_path) as f:
        metas = json.load(f)
    state = load_file(os.path.join(path, "graft_weights.safetensors"))
    placements = []
    for index, meta in enumerate(metas):
        placements.append((meta["host_layer"], rebuild_graft(meta, state, index)))
    # attn_heads and block parts keep their own KV caches, so the model
    # generates with use_cache=True. wsp.graft.disable_cache turns it off.
    attach_grafts(model, placements)
    model.wsp_grafts.load_state_dict(state, strict=True)
    return model.to(device).eval(), tokenizer
