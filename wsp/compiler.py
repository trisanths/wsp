"""compile(program) -> model_path.

Same-base programs only for now: every component must name the program's
base. Cross-family stitching and the heal pass come later.

Steps:
1. Load the base model in bf16.
2. Add each task vector: weight * (fine-tuned - base), on its layers and modules.
3. Evaluate the mask expression and zero every deleted weight.
4. Save the model, its tokenizer, program.json, and compile_report.json.
"""

import json
import os

import torch

from wsp import masks as mask_ops


class CompileError(Exception):
    pass


def check_same_base(program, catalog):
    for component_id in program.component_ids():
        record = catalog.get(component_id)
        if record["base"] != program.base:
            raise CompileError(
                f"{component_id} is built on {record['base']}, but the program base is {program.base}. "
                "Cross-family components need a stitch layer, which is not built yet."
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


def compile_program(program, catalog, output_dir):
    import transformers

    check_same_base(program, catalog)
    torch.manual_seed(program.seed)

    model = transformers.AutoModelForCausalLM.from_pretrained(program.base, torch_dtype=torch.bfloat16)
    params = dict(model.named_parameters())
    report = {"base": program.base, "task_vectors": [], "mask": None}

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
    model.save_pretrained(output_dir)
    transformers.AutoTokenizer.from_pretrained(program.base).save_pretrained(output_dir)
    with open(os.path.join(output_dir, "program.json"), "w") as f:
        json.dump(program.to_dict(), f, indent=2)
    with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return output_dir
