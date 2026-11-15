"""Load a compiled model: wsp.load(path) -> (model, tokenizer).

A compiled directory holds program.json and a tokenizer. A program with
grafts also holds grafts.json and graft_weights.safetensors, plus host/
when masks or task vectors changed the host. A program without grafts is
a plain Hugging Face model directory.
"""

import json
import os

import torch


def load(path, device="cpu", dtype=torch.bfloat16):
    import transformers
    from safetensors.torch import load_file

    from wsp.assemble import rebuild_graft
    from wsp.graft import attach_grafts

    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
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

    with open(grafts_path) as f:
        metas = json.load(f)
    state = load_file(os.path.join(path, "graft_weights.safetensors"))
    placements = []
    for index, meta in enumerate(metas):
        placements.append((meta["host_layer"], rebuild_graft(meta, state, index)))
    # attach_grafts also turns use_cache off when a graft holds an
    # attn_heads part, because that part has no KV cache.
    attach_grafts(model, placements)
    model.wsp_grafts.load_state_dict(state, strict=True)
    return model.to(device).eval(), tokenizer
