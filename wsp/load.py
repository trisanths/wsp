"""Load a compiled model: wsp.load(path) -> (model, tokenizer).

wsp.load reads every kind of compiled directory. It looks at the files:

  insert.json          inserted donor layers (wsp.insert.load_inserted)
  manifest.json        routed experts, a skill MoE (wsp.skill_moe.load_skill_moe)
  grafts.json, taps.json, or neither
                       the layout below. Grafts with gate on in program.json
                       also get their token gates (gates.json), unless
                       gates=False.

Directories from wsp.fuse.compile_fusion have gates.json too, but their
program has no gate flag, so wsp.load leaves the gates off as before and
wsp.fuse.load adds them.

A compiled directory holds program.json and a tokenizer. A program with
grafts or taps also holds grafts.json (an empty list for a tap program
without grafts) and graft_weights.safetensors when it has grafts, plus
host/ when masks or task vectors changed the host, lora.json with
lora_weights.safetensors when it has a LoRA, and taps.json with
tap_weights.safetensors when it has taps. A program without grafts or
taps is a plain Hugging Face model directory (its LoRA, if any, is merged
in).

A loaded tap model loads each donor by its id, in donor_dtype on the same
device, and auto-prepares: plain model(input_ids) and
model.generate(input_ids, attention_mask=...) run the donors on the
prompt text by themselves (see wsp.tap.auto_context). wsp.tap.prepare
still works and takes priority until wsp.tap.clear_tap_context.
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

    if int(transformers.__version__.split(".")[0]) < 5:
        return transformers.AutoTokenizer.from_pretrained(source)
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    saved = get_tokenizer_config(source).get("tokenizer_class")
    if saved in GENERIC_TOKENIZERS:
        return transformers.PreTrainedTokenizerFast.from_pretrained(source)
    return transformers.AutoTokenizer.from_pretrained(source)


def load(path, device="cpu", dtype=torch.bfloat16, donor_dtype=torch.bfloat16, gates=True):
    """Return (model, tokenizer) for any compiled directory.

    The experts and inserts layouts fetch their donor tensors by donor id
    (a Hugging Face repo or a local folder) with byte-range reads.
    gates=False skips the token gates of gated grafts.
    """
    import transformers
    from safetensors.torch import load_file

    from wsp.assemble import rebuild_graft
    from wsp.graft import attach_grafts

    if os.path.exists(os.path.join(path, "insert.json")):
        from wsp.insert import load_inserted

        return load_inserted(path, device=device, dtype=dtype)
    manifest_path = os.path.join(path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("format") == "skill_moe/1":
            from wsp.skill_moe import load_skill_moe

            model, tokenizer = load_skill_moe(path, device=device, dtype=dtype)
            if tokenizer is None:
                tokenizer = load_tokenizer(path)
            return model, tokenizer

    tokenizer = load_tokenizer(path)
    grafts_path = os.path.join(path, "grafts.json")
    taps_path = os.path.join(path, "taps.json")
    if not os.path.exists(grafts_path) and not os.path.exists(taps_path):
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

    metas = []
    if os.path.exists(grafts_path):
        with open(grafts_path) as f:
            metas = json.load(f)
    if len(metas) > 0:
        state = load_file(os.path.join(path, "graft_weights.safetensors"))
        placements = []
        for index, meta in enumerate(metas):
            placements.append((meta["host_layer"], rebuild_graft(meta, state, index)))
        # attn_heads and block parts keep their own KV caches, so the model
        # generates with use_cache=True. wsp.graft.disable_cache turns it off.
        attach_grafts(model, placements)
        model.wsp_grafts.load_state_dict(state, strict=True)

    model = model.to(device).eval()
    if gates and has_gated_grafts(program):
        from wsp.gate import load_gates

        load_gates(model, path)
        model.eval()
    if os.path.exists(taps_path):
        load_tapped(model, tokenizer, path, device, donor_dtype)
    return model, tokenizer


def has_gated_grafts(program):
    """True when the program dict (program.json) has a graft with gate on."""
    for graft in program.get("grafts", []):
        if graft.get("gate", False):
            return True
    return False


def load_tapped(model, tokenizer, path, device, donor_dtype):
    """Rebuild the taps saved in path, load their donors, and turn on auto-prepare."""
    import transformers

    from wsp.tap import load_taps, set_tap_sources

    info = load_taps(model, path)
    donors = {}
    for row in info["taps"]:
        donor_id = row["donor_id"]
        if donor_id in donors:
            continue
        donor = transformers.AutoModelForCausalLM.from_pretrained(donor_id, dtype=donor_dtype)
        donor = donor.to(device).eval()
        for param in donor.parameters():
            param.requires_grad_(False)
        donors[donor_id] = (donor, load_tokenizer(donor_id))
    model.wsp_taps.to(device=device, dtype=torch.float32)
    set_tap_sources(model, tokenizer, donors)
    return info
