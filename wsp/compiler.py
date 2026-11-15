"""compile(program) -> model_path.

Same-base programs only for now: every component must name the program's
base. Cross-family stitching and the heal pass come later.

Steps:
1. Load the base model in bf16.
2. Add each task vector: weight * (fine-tuned - base), on its layers and modules.
3. Evaluate the mask expression and zero every deleted weight.
4. Add the LoRA, if any. Build and attach the grafts, if any.
5. Heal: train the glue and the LoRA together.
6. Save the model, its tokenizer, program.json, and compile_report.json.
   A LoRA without grafts is merged into the weights, so the output is a
   plain Hugging Face model. With grafts, the LoRA goes to
   lora_weights.safetensors and lora.json.
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
    for name in ["grafts.json", "graft_weights.safetensors", "lora.json", "lora_weights.safetensors"]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def compile_program(program, catalog, output_dir, align_texts=None, heal_texts=None, device=None, cache_dir=None):
    """Build the program's model and save it to output_dir. Return output_dir.

    align_texts: texts for the W_in ridge fit. None skips the fit, so W_in
        starts random and the heal pass trains it.
    heal_texts: training texts for the heal pass. None loads them from the
        program's heal data name.
    """
    import transformers

    check_same_base(program, catalog)
    if program.lora_spec is not None and program.heal_spec is None:
        raise CompileError("a program with a LoRA needs a heal spec: call .heal(...) to train it")
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
    if len(program.grafts) > 0 and host_changed:
        # host/ holds the host without the LoRA. The LoRA is saved on its own.
        model.save_pretrained(os.path.join(output_dir, "host"))

    lora_params = []
    if program.lora_spec is not None:
        lora_params = add_lora(program.lora_spec, model, report)

    if len(program.grafts) > 0:
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


def compile_grafts(program, catalog, model, tokenizer, output_dir, report, align_texts, heal_texts, device, cache_dir,
                   lora_params=None):
    """Build, fit, attach, and heal the grafts. Save their weights.

    lora_params: LoRA parameters on the host, trained by the same heal pass.
    """
    from safetensors.torch import save_file

    from wsp.assemble import build_graft
    from wsp.graft import attach_grafts, freeze_all_but_glue, glue_parameters, part_lora_parameters

    model.to(device)
    model.eval()
    placements = []
    metas = []
    for spec in program.grafts:
        host_layer, graft, meta = build_graft(spec, catalog, model, tokenizer, align_texts, device, cache_dir)
        placements.append((host_layer, graft))
        metas.append(meta)
    attach_grafts(model, placements)
    model.wsp_grafts.to(device)
    report["grafts"] = metas

    if program.heal_spec is not None:
        freeze_all_but_glue(model)
        params = glue_parameters(model)
        # A part LoRA (the receiver) trains with the glue.
        part_lora = part_lora_parameters(model)
        for param in part_lora:
            params.append(param)
        if lora_params is not None:
            for param in lora_params:
                param.requires_grad_(True)
                params.append(param)
        run_heal(program, model, tokenizer, heal_texts, params, report)
        count = 0
        for param in part_lora:
            count += param.numel()
        report["heal"]["part_lora_params"] = count

    state = {}
    for key, value in model.wsp_grafts.state_dict().items():
        state[key] = value.detach().cpu().contiguous()
    save_file(state, os.path.join(output_dir, "graft_weights.safetensors"))
    with open(os.path.join(output_dir, "grafts.json"), "w") as f:
        json.dump(metas, f, indent=2)
