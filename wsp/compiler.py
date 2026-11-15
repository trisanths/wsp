"""compile(program) -> model_path.

Same-base programs only for now: every component must name the program's
base. Cross-family stitching and the heal pass come later.

Steps:
1. Load the base model in bf16.
2. Add each task vector: weight * (fine-tuned - base), on its layers and modules.
3. Evaluate the mask expression and zero every deleted weight.
4. Add the LoRA, if any. Build and attach the grafts, if any.
5. Taps, if any: load each donor, attach the branches, and train only the
   taps on (prompt, completion) pairs.
6. Heal: train the glue and the LoRA together. With taps, the taps are
   frozen first and heal trains on pairs with the taps reading the
   donors (staged training, since joint training collapsed the tap).
7. Save the model, its tokenizer, program.json, and compile_report.json.
   A LoRA without grafts or taps is merged into the weights, so the
   output is a plain Hugging Face model. With grafts or taps, the LoRA
   goes to lora_weights.safetensors and lora.json, and the taps go to
   taps.json and tap_weights.safetensors. A tap program also writes
   grafts.json (an empty list when it has no grafts), so tools that look
   for grafts.json load it with wsp.load.

A written program (base_source, carve, or splice) takes another path,
compile_written: load the sources, splice, carve layers, carve width, heal
a LoRA and merge it, and save one plain Hugging Face model.
"""

import json
import os
import shutil

import torch

from wsp import masks as mask_ops


class CompileError(Exception):
    pass


def check_same_base(program, catalog):
    """Masks and task vectors edit the host's own weights, so they must be built on the host.

    Grafts can come from any model and are not checked here.
    """
    for component_id in program.host_component_ids():
        record = catalog.get(component_id)
        if record["base"] != program.host:
            raise CompileError(
                f"{component_id} is built on {record['base']}, but the program host is {program.host}. "
                "Use a graft to bring in a part from another model."
            )


def make_mask_loader(catalog):
    """Return load_mask(component_id) that reads each mask once."""
    cache = {}

    def load_mask(component_id):
        if component_id not in cache:
            record = catalog.get(component_id)
            if record["type"] != "deletion_mask":
                raise CompileError(f"{component_id} is a {record['type']}, not a deletion_mask")
            cache[component_id] = mask_ops.load_mask_dir(catalog.local_path(component_id))
        return cache[component_id]

    return load_mask


def selected(name, layers, modules):
    if layers is not None and mask_ops.layer_of(name) not in layers:
        return False
    if modules is not None and mask_ops.module_of(name) not in modules:
        return False
    return True


@torch.no_grad()
def add_task_vector(params, base_params, finetuned_params, weight, layers=None, modules=None):
    """params += weight * (finetuned - base) on selected names. Return the names changed."""
    changed = []
    for name in sorted(params):
        if not selected(name, layers, modules):
            continue
        if name not in finetuned_params:
            continue
        if finetuned_params[name].shape != base_params[name].shape:
            raise CompileError(f"{name}: fine-tuned shape {tuple(finetuned_params[name].shape)} "
                               f"differs from base {tuple(base_params[name].shape)}")
        delta = finetuned_params[name].float() - base_params[name].float()
        new_value = params[name].float() + weight * delta
        params[name].copy_(new_value.to(params[name].dtype))
        changed.append(name)
    return changed


@torch.no_grad()
def apply_mask(params, mask):
    """Zero every deleted weight in params. Return the number deleted."""
    deleted = 0
    for name in sorted(mask):
        if name not in params:
            raise CompileError(f"mask entry {name} is not a parameter of the base model")
        delete = mask[name]
        if delete.shape != params[name].shape:
            raise CompileError(f"{name}: mask shape {tuple(delete.shape)} vs weight {tuple(params[name].shape)}")
        keep = (~delete).to(device=params[name].device, dtype=params[name].dtype)
        params[name].mul_(keep)
        deleted += int(delete.sum())
    return deleted


def remove_stale_outputs(output_dir):
    """Remove the files a past compile into output_dir can leave behind.

    load() picks its path from the files that exist. A host/ or grafts.json
    from an older program would make it load the wrong model.
    """
    host_dir = os.path.join(output_dir, "host")
    if os.path.isdir(host_dir):
        shutil.rmtree(host_dir)
    for name in ["grafts.json", "graft_weights.safetensors", "lora.json", "lora_weights.safetensors",
                 "taps.json", "tap_weights.safetensors"]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def compile_program(program, catalog, output_dir, align_texts=None, heal_texts=None, device=None, cache_dir=None,
                    tap_pairs=None, heal_pairs=None, calib_texts=None, domain_texts=None):
    """Build the program's model and save it to output_dir. Return output_dir.

    A written program (base_source, carve, or splice) goes to
    compile_written, which uses heal_texts, heal_pairs, calib_texts, and
    domain_texts, and needs no catalog.

    align_texts: texts for the W_in ridge fit. None skips the fit, so W_in
        starts random and the heal pass trains it.
    heal_texts: training texts for the heal pass. None loads them from the
        program's heal data name. Not used when the program has taps.
    tap_pairs: (prompt, completion) pairs for the tap training. None loads
        them from wsp.pairs by the tap_train data name.
    heal_pairs: pairs for the heal pass of a program with taps. None uses
        tap_pairs when the heal data name is the tap data name, and loads
        them from wsp.pairs otherwise.
    """
    import transformers

    if program.is_written():
        return compile_written(program, output_dir, heal_texts=heal_texts, heal_pairs=heal_pairs,
                               calib_texts=calib_texts, domain_texts=domain_texts, device=device)
    check_same_base(program, catalog)
    if program.lora_spec is not None and program.heal_spec is None:
        raise CompileError("a program with a LoRA needs a heal spec: call .heal(...) to train it")
    if len(program.taps) > 0 and program.tap_train_spec is None:
        raise CompileError("a program with taps needs a tap_train spec: call .tap_train(...) to train them")
    if len(program.taps) == 0 and program.tap_train_spec is not None:
        raise CompileError("a program with a tap_train spec needs at least one tap")
    if len(program.taps) > 0:
        check_pair_names(program, tap_pairs, heal_pairs)
    torch.manual_seed(program.seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(program.host, dtype=torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(program.host)
    params = dict(model.named_parameters())
    report = {"host": program.host, "task_vectors": [], "mask": None, "grafts": [], "heal": None, "lora": None}

    if len(program.task_vectors) > 0:
        base_params = {}
        for name in params:
            base_params[name] = params[name].detach().clone()
        for tv in program.task_vectors:
            record = catalog.get(tv.component_id)
            if record["type"] != "task_vector":
                raise CompileError(f"{tv.component_id} is a {record['type']}, not a task_vector")
            finetuned = transformers.AutoModelForCausalLM.from_pretrained(record["model"], torch_dtype=torch.bfloat16)
            finetuned_params = dict(finetuned.named_parameters())
            changed = add_task_vector(params, base_params, finetuned_params, tv.weight, tv.layers, tv.modules)
            report["task_vectors"].append({"id": tv.component_id, "weight": tv.weight, "params_changed": len(changed)})
            del finetuned
            del finetuned_params
        del base_params

    if program.masks is not None:
        mask = program.masks.evaluate(make_mask_loader(catalog))
        deleted = apply_mask(params, mask)
        total_params = 0
        for name in params:
            total_params += params[name].numel()
        report["mask"] = {
            "expression": repr(program.masks),
            "deleted": deleted,
            "fraction_of_all_params": deleted / total_params,
            "deleted_per_layer": mask_ops.deleted_per_layer(mask),
        }

    os.makedirs(output_dir, exist_ok=True)
    remove_stale_outputs(output_dir)
    tokenizer.save_pretrained(output_dir)
    host_changed = program.masks is not None or len(program.task_vectors) > 0
    if (len(program.grafts) > 0 or len(program.taps) > 0) and host_changed:
        # host/ holds the host without the LoRA. The LoRA is saved on its own.
        model.save_pretrained(os.path.join(output_dir, "host"))

    lora_params = []
    if program.lora_spec is not None:
        lora_params = add_lora(program.lora_spec, model, report)

    if len(program.taps) > 0:
        compile_taps(program, catalog, model, tokenizer, output_dir, report, align_texts, device, cache_dir,
                     lora_params, tap_pairs, heal_pairs)
        if program.lora_spec is not None:
            save_lora(model, output_dir, report)
    elif len(program.grafts) > 0:
        compile_grafts(program, catalog, model, tokenizer, output_dir, report,
                       align_texts, heal_texts, device, cache_dir, lora_params)
        if program.lora_spec is not None:
            save_lora(model, output_dir, report)
    else:
        if program.lora_spec is not None:
            from wsp.lora import merge_lora

            model.to(device)
            for param in model.parameters():
                param.requires_grad_(False)
            for param in lora_params:
                param.requires_grad_(True)
            run_heal(program, model, tokenizer, heal_texts, lora_params, report)
            # Fold the LoRA into the weights, so the output is a plain HF model.
            merge_lora(model)
            report["lora"]["merged"] = True
        model.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "program.json"), "w") as f:
        json.dump(program.to_dict(), f, indent=2)
    with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return output_dir


def add_lora(spec, model, report):
    """Apply the program's LoRA to the host. Record it in report["lora"]. Return its parameters."""
    from wsp.lora import DEFAULT_TARGETS, default_alpha, apply_lora, rank_for_budget

    targets = DEFAULT_TARGETS
    if spec.targets is not None:
        targets = tuple(spec.targets)
    rank = spec.rank
    if rank is None:
        rank = rank_for_budget(model, spec.budget_params, targets, spec.layers)
    alpha = spec.alpha
    if alpha is None:
        alpha = default_alpha(rank)
    params = apply_lora(model, rank, alpha, targets, spec.layers)
    count = 0
    for param in params:
        count += param.numel()
    report["lora"] = {"rank": rank, "alpha": alpha, "targets": list(targets), "layers": spec.layers,
                      "params": count, "budget_params": spec.budget_params, "merged": False}
    return params


def save_lora(model, output_dir, report):
    """Save the LoRA tensors and lora.json, so wsp.load can rebuild the LoRA."""
    from safetensors.torch import save_file

    from wsp.lora import lora_state_dict

    save_file(lora_state_dict(model), os.path.join(output_dir, "lora_weights.safetensors"))
    info = {}
    for key in ["rank", "alpha", "targets", "layers", "params"]:
        info[key] = report["lora"][key]
    with open(os.path.join(output_dir, "lora.json"), "w") as f:
        json.dump(info, f, indent=2)


def run_heal(program, model, tokenizer, heal_texts, params, report):
    """Run the program's heal pass on params. Record it in report["heal"].

    The caller sets requires_grad. Grafts and the LoRA use the same loop,
    data, and seed, so their results compare fairly.
    """
    import time

    from wsp.heal import heal

    spec = program.heal_spec
    if heal_texts is None:
        from wsp.data import task_texts

        heal_texts = task_texts(spec.data, seed=0)
    trainable = 0
    for param in params:
        trainable += param.numel()
    start = time.time()
    losses = heal(model, tokenizer, heal_texts, params, steps=spec.steps, lr=spec.lr,
                  batch_size=spec.batch_size, seq_len=spec.seq_len, seed=program.seed)
    report["heal"] = {"spec": spec.to_dict(), "losses": losses, "seconds": round(time.time() - start, 1),
                      "trainable_params": trainable}


def attach_program_grafts(program, catalog, model, tokenizer, report, align_texts, device, cache_dir):
    """Build, fit, and attach the program's grafts. Record them in report["grafts"]. Return their metas."""
    from wsp.assemble import build_graft
    from wsp.graft import attach_grafts

    placements = []
    metas = []
    for spec in program.grafts:
        host_layer, graft, meta = build_graft(spec, catalog, model, tokenizer, align_texts, device, cache_dir)
        placements.append((host_layer, graft))
        metas.append(meta)
    attach_grafts(model, placements)
    model.wsp_grafts.to(device)
    report["grafts"] = metas
    return metas


def heal_parameters(model, lora_params):
    """Freeze the host and return the glue, part LoRA, and host LoRA parameters, and the part LoRA count.

    With no grafts on the model, only the host LoRA parameters come back.
    """
    from wsp.graft import freeze_all_but_glue, glue_parameters, part_lora_parameters

    params = []
    part_lora = []
    if getattr(model, "wsp_grafts", None) is not None:
        freeze_all_but_glue(model)
        params = glue_parameters(model)
        # A part LoRA (the receiver) trains with the glue.
        part_lora = part_lora_parameters(model)
        for param in part_lora:
            params.append(param)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
    if lora_params is not None:
        for param in lora_params:
            param.requires_grad_(True)
            params.append(param)
    count = 0
    for param in part_lora:
        count += param.numel()
    return params, count


def save_grafts(model, output_dir, metas):
    """Write grafts.json and, when there are grafts, graft_weights.safetensors."""
    from safetensors.torch import save_file

    if len(metas) > 0:
        state = {}
        for key, value in model.wsp_grafts.state_dict().items():
            state[key] = value.detach().cpu().contiguous()
        save_file(state, os.path.join(output_dir, "graft_weights.safetensors"))
    with open(os.path.join(output_dir, "grafts.json"), "w") as f:
        json.dump(metas, f, indent=2)


def compile_grafts(program, catalog, model, tokenizer, output_dir, report, align_texts, heal_texts, device, cache_dir,
                   lora_params=None):
    """Build, fit, attach, and heal the grafts. Save their weights.

    lora_params: LoRA parameters on the host, trained by the same heal pass.
    """
    model.to(device)
    model.eval()
    metas = attach_program_grafts(program, catalog, model, tokenizer, report, align_texts, device, cache_dir)

    if program.heal_spec is not None:
        params, part_lora_count = heal_parameters(model, lora_params)
        run_heal(program, model, tokenizer, heal_texts, params, report)
        report["heal"]["part_lora_params"] = part_lora_count

    save_grafts(model, output_dir, metas)


def model_width(model):
    """Return the hidden size of a model's decoder (the text part of a multimodal config)."""
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return config.hidden_size


def load_tap_donors(program, device):
    """Load each distinct tap donor once, in bf16 on device. Return donor_id -> (model, tokenizer)."""
    import transformers

    donors = {}
    for spec in program.taps:
        if spec.donor in donors:
            continue
        donor = transformers.AutoModelForCausalLM.from_pretrained(spec.donor, dtype=torch.bfloat16)
        donor.to(device)
        donor.eval()
        for param in donor.parameters():
            param.requires_grad_(False)
        donor_tokenizer = transformers.AutoTokenizer.from_pretrained(spec.donor)
        donors[spec.donor] = (donor, donor_tokenizer)
    return donors


def build_taps(program, model, donors, device):
    """Make one TapBranch per tap and attach them to the host. Return the placements.

    Taps with the same donor and donor_layer share one tap index, so they
    read one memory.
    """
    from wsp.parts import find_decoder_layers
    from wsp.tap import TapBranch, attach_taps

    _, host_layers = find_decoder_layers(model, 0)
    indices = {}
    placements = []
    for number, spec in enumerate(program.taps):
        if spec.at >= len(host_layers):
            raise CompileError(f"tap {number} is at host layer {spec.at}, but the host has {len(host_layers)} layers")
        donor = donors[spec.donor][0]
        _, donor_layers = find_decoder_layers(donor, 0)
        if spec.donor_layer >= len(donor_layers):
            raise CompileError(f"tap {number} reads donor layer {spec.donor_layer}, but {spec.donor} has "
                               f"{len(donor_layers)} layers")
        key = (spec.donor, spec.donor_layer)
        if key not in indices:
            indices[key] = len(indices)
        tap_id = spec.tap_id
        if tap_id is None:
            tap_id = f"tap{number}"
        branch = TapBranch(model_width(model), model_width(donor), attn_dim=spec.attn_dim, heads=spec.heads,
                           donor_id=spec.donor, donor_layer=spec.donor_layer, tap_id=tap_id)
        placements.append((spec.at, branch, indices[key]))
    attach_taps(model, placements)
    model.wsp_taps.to(device=device, dtype=torch.float32)
    return placements


def tap_param_groups(model):
    """Return optimizer groups for the taps: the gates in one, the rest in the other (no lr set)."""
    gates = []
    others = []
    for branch in model.wsp_taps:
        for name, param in branch.named_parameters():
            if name == "gate":
                gates.append(param)
            else:
                others.append(param)
    return gates, others


def check_pair_names(program, tap_pairs, heal_pairs):
    """Raise CompileError for a pairs name that wsp.pairs does not know.

    This runs before any model loads. Without it, a bad heal data name
    fails only after the donors load and the taps train.
    """
    from wsp.pairs import pair_names

    names = pair_names()
    spec = program.tap_train_spec
    if tap_pairs is None and spec.data not in names:
        raise CompileError(f"unknown tap_train data {spec.data!r}; the pairs names are {', '.join(names)}")
    heal = program.heal_spec
    if heal is None or heal_pairs is not None or heal.data == spec.data:
        return
    if len(program.grafts) == 0 and program.lora_spec is None:
        # This heal does not run (see compile_taps).
        return
    if heal.data not in names:
        raise CompileError(f"unknown heal data {heal.data!r}: in a program with taps, heal data is a pairs "
                           f"name, one of {', '.join(names)}")


def pairs_by_name(name, tokenizer, seed=0):
    from wsp.pairs import pairs_for

    return pairs_for(name, tokenizer, seed=seed)


def train_program_taps(program, model, tokenizer, donors, pairs, report):
    """Train only the taps (stage 1). Record it in report["taps"]."""
    import time

    from wsp.tap import tap_meta, tap_param_count, tap_train

    spec = program.tap_train_spec
    gates, others = tap_param_groups(model)
    groups = [{"params": others, "lr": spec.branch_lr}, {"params": gates, "lr": spec.gate_lr}]
    start = time.time()
    losses = tap_train(model, tokenizer, donors, pairs, groups, steps=spec.steps, lr=spec.branch_lr,
                       batch_size=spec.batch_size, seq_len=spec.seq_len, seed=program.seed)
    seconds = round(time.time() - start, 1)
    final_gates = []
    for branch in model.wsp_taps:
        final_gates.append(float(branch.gate.detach()))
    donor_ids = []
    for donor_id in donors:
        donor_ids.append(donor_id)
    report["taps"] = {"spec": spec.to_dict(), "taps": tap_meta(model), "donors": donor_ids, "losses": losses,
                      "gates": final_gates, "trainable_params": tap_param_count(model), "pairs": len(pairs),
                      "seconds": seconds}


def heal_after_taps(program, model, tokenizer, donors, pairs, params, part_lora_count, report):
    """Train params with the taps frozen (stage 2). Record it in report["heal"].

    The loop is wsp.tap.tap_train on (prompt, completion) pairs, so the
    taps read the donors the same way as in stage 1. The seed is the
    program seed plus 50, the same offset as the staged arm in round 7.
    """
    import time

    from wsp.tap import tap_parameters, tap_train

    spec = program.heal_spec
    for param in tap_parameters(model):
        param.requires_grad_(False)
    trainable = 0
    for param in params:
        trainable += param.numel()
    start = time.time()
    losses = tap_train(model, tokenizer, donors, pairs, params, steps=spec.steps, lr=spec.lr,
                       batch_size=spec.batch_size, seq_len=spec.seq_len, seed=program.seed + 50)
    report["heal"] = {"spec": spec.to_dict(), "losses": losses, "seconds": round(time.time() - start, 1),
                      "trainable_params": trainable, "part_lora_params": part_lora_count,
                      "staged_after_taps": True, "pairs": len(pairs)}


def save_program_taps(model, output_dir, report):
    """Save the tap branches and their donor ids, so wsp.load can rebuild them."""
    from wsp.tap import save_taps

    save_taps(model, output_dir, meta={"donors": report["taps"]["donors"]})


def compile_taps(program, catalog, model, tokenizer, output_dir, report, align_texts, device, cache_dir,
                 lora_params, tap_pairs, heal_pairs):
    """Attach grafts and taps, train the taps, then heal with the taps frozen. Save grafts and taps.

    The LoRA, if any, is saved by the caller.
    """
    from wsp.tap import set_tap_sources, tap_cost

    model.to(device)
    model.eval()
    metas = []
    if len(program.grafts) > 0:
        metas = attach_program_grafts(program, catalog, model, tokenizer, report, align_texts, device, cache_dir)

    donors = load_tap_donors(program, device)
    build_taps(program, model, donors, device)
    set_tap_sources(model, tokenizer, donors)

    spec = program.tap_train_spec
    if tap_pairs is None:
        tap_pairs = pairs_by_name(spec.data, tokenizer)
    # tap_train freezes everything but its params, so a LoRA or the glue
    # stays at its start during stage 1.
    train_program_taps(program, model, tokenizer, donors, tap_pairs, report)

    if program.heal_spec is not None and (len(metas) > 0 or program.lora_spec is not None):
        if heal_pairs is None:
            if program.heal_spec.data == spec.data:
                heal_pairs = tap_pairs
            else:
                heal_pairs = pairs_by_name(program.heal_spec.data, tokenizer)
        params, part_lora_count = heal_parameters(model, lora_params)
        heal_after_taps(program, model, tokenizer, donors, heal_pairs, params, part_lora_count, report)

    for param in model.parameters():
        param.requires_grad_(False)
    save_grafts(model, output_dir, metas)
    save_program_taps(model, output_dir, report)
    report["taps"]["cost"] = tap_cost(model)


# Written programs: one plain model from source models.


def pair_texts(pairs):
    """Join (prompt, completion) pairs into training texts."""
    texts = []
    for prompt, completion in pairs:
        texts.append(prompt + completion)
    return texts


def written_heal_texts(spec, tokenizer, heal_texts, heal_pairs, seed):
    """Texts for the heal of a written program: given texts, given pairs, a wsp.pairs name, or a wsp.data name."""
    if heal_texts is not None:
        return list(heal_texts)
    if heal_pairs is not None:
        return pair_texts(heal_pairs)
    from wsp.pairs import pair_names

    if spec.data in pair_names():
        return pair_texts(pairs_by_name(spec.data, tokenizer, seed=seed))
    from wsp.data import task_texts

    return task_texts(spec.data, seed=seed)


def check_written(program):
    """Raise CompileError for parts a written program cannot hold."""
    if len(program.grafts) > 0 or len(program.taps) > 0 or program.tap_train_spec is not None:
        raise CompileError("a written program is one plain model, so it cannot have grafts or taps")
    if program.masks is not None or len(program.task_vectors) > 0:
        raise CompileError("a written program does not take masks or task vectors")
    if program.lora_spec is not None and program.heal_spec is None:
        raise CompileError("a program with a LoRA needs a heal spec: call .heal(...) to train it")
    if program.heal_spec is not None and program.lora_spec is None:
        raise CompileError("in a written program, heal trains a LoRA that is merged at the end: add .lora(...)")


def load_written_sources(program):
    """Load the source models in bf16. Return ({key: model}, base key)."""
    import transformers

    if program.splice_spec is None:
        model = transformers.AutoModelForCausalLM.from_pretrained(program.host, dtype=torch.bfloat16)
        return {"base": model}, "base"
    spec = program.splice_spec
    models = {}
    loaded = {}
    for key in spec.sources:
        name = spec.sources[key]
        # Keys that name one model share it; splice copies a layer used twice.
        if name not in loaded:
            loaded[name] = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        models[key] = loaded[name]
    return models, spec.base


def carve_written(spec, model, tokenizer, calib_texts, domain_texts, seed):
    """Apply a CarveSpec: choose and keep layers, then cut the width. Return the carve report."""
    from wsp import carve as carve_ops

    _, layers = carve_ops.find_decoder_layers(model, 0)
    layers_before = len(layers)
    keep_mlp = 1.0
    keep_heads = 1.0
    if spec.width is not None:
        keep_mlp = spec.width["keep_mlp"]
        keep_heads = spec.width["keep_heads"]

    keep = layers_before
    if spec.keep is not None:
        keep = min(spec.keep, layers_before)
    elif spec.target_params_b is not None:
        try:
            keep = carve_ops.layers_for_budget(model, spec.target_params_b * 1e9, keep_mlp, keep_heads)
        except ValueError as error:
            raise CompileError(str(error))

    report = {"spec": spec.to_dict(), "layers_before": layers_before, "layers_after": keep, "keep_layers": None,
              "influence": None, "width": None}
    if keep < layers_before:
        if spec.select == "uniform":
            scores = layers_before
        elif spec.select == "domain":
            if domain_texts is None:
                if spec.domain_data is None:
                    raise CompileError("carve select domain needs domain_data or domain_texts")
                domain_texts = pair_texts(pairs_by_name(spec.domain_data, tokenizer, seed=seed))
            if calib_texts is None:
                raise CompileError("carve select domain needs general texts: pass calib_texts or heal texts")
            scores = carve_ops.domain_influence(model, tokenizer, domain_texts[:256], calib_texts)
            report["influence"] = scores
        else:
            if calib_texts is None:
                raise CompileError("carve select influence needs calibration texts: pass calib_texts or heal texts")
            scores = carve_ops.layer_influence(model, tokenizer, calib_texts)
            report["influence"] = scores
        keep_layers = carve_ops.choose_layers(scores, keep, spec.select, spec.domain_weight)
        carve_ops.carve_layers(model, keep_layers)
        report["keep_layers"] = keep_layers
    else:
        report["keep_layers"] = list(range(layers_before))

    if spec.width is not None:
        scores = None
        texts = calib_texts
        if domain_texts is not None:
            texts = list(domain_texts[:256]) + list(calib_texts or [])
        if texts is not None and (keep_mlp < 1 or keep_heads < 1):
            scores = carve_ops.unit_scores(model, tokenizer, texts)
        _, width_report = carve_ops.carve_width(model, keep_mlp, keep_heads, scores)
        # Leave out the kept unit lists, which are long for a big model.
        summary = {}
        for name in width_report:
            if name != "layers":
                summary[name] = width_report[name]
        first = width_report["layers"][0]
        for name in ["mlp_before", "mlp_after", "q_heads_before", "q_heads_after", "kv_heads_before",
                     "kv_heads_after"]:
            summary[name] = first[name]
        summary["scores"] = "weights" if scores is None else "activations"
        report["width"] = summary
    return report


def compile_written(program, output_dir, heal_texts=None, heal_pairs=None, calib_texts=None, domain_texts=None,
                    device=None):
    """Write one plain model: splice, then carve (layers, then width), then a merged LoRA heal. Return output_dir.

    heal_texts or heal_pairs: the heal data. None loads the heal data name
        (a wsp.pairs name, or a wsp.data name such as gsm8k_train).
    calib_texts: general texts for layer influence and unit scores. None
        uses the first 256 heal texts, when there is a heal.
    domain_texts: domain texts for select domain. None loads the carve's
        domain_data pairs.
    The output directory holds a plain Hugging Face model and tokenizer,
    program.json, and compile_report.json. No wsp code runs at load time.
    """
    import time

    import transformers

    from wsp import carve as carve_ops
    from wsp.lora import merge_lora

    check_written(program)
    start = time.time()
    seconds = {}
    torch.manual_seed(program.seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sources, base_key = load_written_sources(program)
    tokenizer = transformers.AutoTokenizer.from_pretrained(program.host)
    model = sources[base_key]
    report = {"host": program.host, "written": True, "params_before": carve_ops.count_params(model),
              "layers_before": len(carve_ops.find_decoder_layers(model, 0)[1]), "splice": None, "carve": None,
              "lora": None, "heal": None}
    seconds["load"] = round(time.time() - start, 1)

    if program.splice_spec is not None:
        step = time.time()
        spec = program.splice_spec
        configs = {}
        for key in sources:
            configs[key] = sources[key].config
        plan = []
        for key, layer in spec.plan:
            plan.append((key, layer))
        try:
            carve_ops.splice(sources, plan, base_key)
        except ValueError as error:
            raise CompileError(str(error))
        report["splice"] = {"spec": spec.to_dict(), "layers": len(plan),
                            "config_differences": carve_ops.config_differences(configs, base_key)}
        # Only the spliced model stays.
        sources = {base_key: model}
        seconds["splice"] = round(time.time() - step, 1)

    model.to(device)
    model.eval()
    texts = None
    if program.heal_spec is not None:
        texts = written_heal_texts(program.heal_spec, tokenizer, heal_texts, heal_pairs, program.seed)
    if calib_texts is None and texts is not None:
        calib_texts = texts[:256]

    if program.carve_spec is not None:
        step = time.time()
        report["carve"] = carve_written(program.carve_spec, model, tokenizer, calib_texts, domain_texts,
                                        program.seed)
        seconds["carve"] = round(time.time() - step, 1)
        # Check the size before the heal, so an over-budget carve stops early.
        # Merging the LoRA does not change the parameter count.
        spec = program.carve_spec
        size = carve_ops.count_params(model)
        if spec.target_params_b is not None and size > spec.target_params_b * 1e9:
            raise CompileError(f"the written model has {size} parameters, over the target {spec.target_params_b}b")

    if program.lora_spec is not None:
        step = time.time()
        lora_params = add_lora(program.lora_spec, model, report)
        for param in model.parameters():
            param.requires_grad_(False)
        for param in lora_params:
            param.requires_grad_(True)
        run_heal(program, model, tokenizer, texts, lora_params, report)
        # Fold the LoRA into the weights, so the output is a plain HF model.
        merge_lora(model)
        report["lora"]["merged"] = True
        seconds["heal"] = round(time.time() - step, 1)

    for param in model.parameters():
        param.requires_grad_(False)
    report["params_after"] = carve_ops.count_params(model)
    report["layers_after"] = len(carve_ops.find_decoder_layers(model, 0)[1])

    step = time.time()
    os.makedirs(output_dir, exist_ok=True)
    remove_stale_outputs(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "program.json"), "w") as f:
        json.dump(program.to_dict(), f, indent=2)
    seconds["save"] = round(time.time() - step, 1)
    seconds["total"] = round(time.time() - start, 1)
    report["seconds"] = seconds
    with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return output_dir
