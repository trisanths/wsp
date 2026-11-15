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

Three more kinds of program have their own builders, and wsp.load reads
all of them from one directory format:
  experts   compile_experts: routed expert MLPs from donors (skill MoE).
            Saves manifest.json and skill_moe_weights.safetensors.
  inserts   compile_inserts: donor layers inserted into the host.
            Saves insert.json, insert_weights.safetensors, and donors/.
  gates     grafts with gate on: compile_grafts attaches a token gate to
            each and trains the gates in the heal. Saves gates.json and
            gate_weights.safetensors next to the grafts.
Every directory has program.json, compile_report.json, and the tokenizer.
The report always has params_total, params_active (run per token), and
heal_seconds, and a budget entry when the program has a budget.
"""

import json
import os
import shutil

import torch

from wsp import masks as mask_ops


class CompileError(Exception):
    pass


def text_config_of(config):
    """Return the text part of a (possibly multimodal) config."""
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return config


def check_program(program):
    """Raise CompileError for a program that mixes parts the compiler cannot build together.

    A program holds one kind of added component: grafts (taps may join
    them), routed experts, or inserted layers. Masks and task vectors edit
    the host, so they go with grafts or taps only. The checks that need
    models (sizes, tokenizers) run during the compile.
    """
    kinds = []
    if len(program.grafts) > 0:
        kinds.append("grafts")
    if program.experts_spec is not None:
        kinds.append("experts")
    if len(program.inserts) > 0:
        kinds.append("inserts")
    if program.is_written():
        # compile_written has its own checks and messages (check_written).
        return
    if "experts" in kinds and len(kinds) > 1:
        raise CompileError("a program takes experts or inserts or grafts, not a mix: this one has " + ", ".join(kinds))
    if "inserts" in kinds and len(kinds) > 1:
        raise CompileError("a program takes experts or inserts or grafts, not a mix: this one has " + ", ".join(kinds))
    if (program.experts_spec is not None or len(program.inserts) > 0) and len(program.taps) > 0:
        raise CompileError("taps cannot join experts or inserts in one program")
    if (program.experts_spec is not None or len(program.inserts) > 0) and (
            program.masks is not None or len(program.task_vectors) > 0):
        raise CompileError("masks and task vectors cannot join experts or inserts in one program")
    if program.experts_spec is not None and program.lora_spec is not None and program.lora_spec.layers is not None:
        raise CompileError("an experts program puts its LoRA on the attention of every layer; drop layers")
    gated = []
    for graft in program.grafts:
        if graft.gate:
            gated.append(graft)
    if len(gated) > 0:
        if len(program.taps) > 0:
            raise CompileError("gates are not available in a program with taps")
        first = gated[0]
        for graft in gated:
            if graft.gate_bias != first.gate_bias or graft.gate_lr != first.gate_lr:
                raise CompileError("grafts with gate on must share gate_bias and gate_lr")
    spec = program.heal_spec
    if spec is not None and spec.distill is not None and len(program.taps) > 0:
        raise CompileError("a distillation heal is not available in a program with taps")
    if program.lora_spec is not None and program.heal_spec is None:
        raise CompileError("a program with a LoRA needs a heal spec: call .heal(...) to train it")
    if program.heal_spec is not None and program.lora_spec is None and len(kinds) == 0 and len(program.taps) == 0:
        raise CompileError("this program has a heal but nothing to train: add a lora statement, a graft, "
                           "experts, or an insert")


def count_model_params(model, exclude_lora=False):
    """Return (params_total, params_active) of a built model.

    params_total counts every distinct parameter once. params_active is
    what one token runs: the total minus the input embedding table (a
    lookup, unless the output head shares it) and minus, in every routed
    expert layer, the experts that the top_k routing leaves out (the top_k
    largest count). Taps count their branches but not their donors, which
    are separate models. exclude_lora leaves out LoRA factors, for a LoRA
    that the compile merges at the end.
    """
    from wsp.skill_moe import moe_layers

    seen = set()
    total = 0
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if exclude_lora and (name.endswith("lora_A") or name.endswith("lora_B")):
            continue
        total += param.numel()
    active = total
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed is not None:
        tied = head is not None and head.weight is embed.weight
        if not tied:
            active -= embed.weight.numel()
    # Host layers that map to one donor layer share its MLP tensors, and the
    # total counts them once. So take every distinct expert parameter out
    # of the active count once, then add back what each layer runs.
    expert_ids = set()
    run = 0
    for moe in moe_layers(model):
        sizes = []
        for index in range(moe.n_experts()):
            size = 0
            for param in moe.experts[index].parameters():
                size += param.numel()
                if id(param) not in expert_ids:
                    expert_ids.add(id(param))
                    active -= param.numel()
            sizes.append(size)
        run += sum(sorted(sizes, reverse=True)[:moe.top_k])
    active += run
    return total, active


def heal_limit_seconds(program):
    """The budget's heal limit in seconds, or None."""
    spec = program.budget_spec
    if spec is None or spec.max_heal_minutes is None:
        return None
    return spec.max_heal_minutes * 60.0


def budget_problem(spec, total, active, heal_seconds, stopped_early=False):
    """Return a sentence that names every limit the numbers break, or None when all hold."""
    problems = []
    if spec.max_total_params_b is not None and total > spec.max_total_params_b * 1e9:
        problems.append(f"params_total {total} ({total / 1e9:.4f}b) is over max_total_params_b "
                        f"{spec.max_total_params_b}")
    if spec.max_active_params_b is not None and active > spec.max_active_params_b * 1e9:
        problems.append(f"params_active {active} ({active / 1e9:.4f}b) is over max_active_params_b "
                        f"{spec.max_active_params_b}")
    if spec.max_heal_minutes is not None and not stopped_early and heal_seconds > spec.max_heal_minutes * 60.0:
        problems.append(f"the heal took {heal_seconds / 60.0:.2f} minutes, over max_heal_minutes "
                        f"{spec.max_heal_minutes}")
    if len(problems) == 0:
        return None
    return "the compiled model is over its budget: " + "; ".join(problems)


def check_size_budget(program, model, report, exclude_lora=False):
    """Raise CompileError when the built model is over a size limit. Run before the heal.

    The message holds the measured params_total and params_active.
    """
    total, active = count_model_params(model, exclude_lora=exclude_lora)
    report["params_total"] = total
    report["params_active"] = active
    spec = program.budget_spec
    if spec is None:
        return
    problem = budget_problem(spec, total, active, 0.0)
    if problem is not None:
        raise CompileError(f"{problem}. Measured before the heal: params_total {total}, params_active {active}.")


def finish_outputs(program, model, report, output_dir):
    """Write program.json and compile_report.json, with the final params and heal seconds in the report.

    The report gets params_total, params_active, heal_seconds, and (with a
    budget) a budget entry with the measured numbers and ok. A heal that
    ran over max_heal_minutes raises CompileError here, after the files are
    written, so the model is kept.
    """
    total, active = count_model_params(model)
    report["params_total"] = total
    report["params_active"] = active
    heal_seconds = 0.0
    stopped_early = False
    if report.get("heal") is not None:
        heal_seconds = float(report["heal"].get("seconds", 0.0))
        stopped_early = bool(report["heal"].get("stopped_early", False))
    report["heal_seconds"] = heal_seconds
    problem = None
    spec = program.budget_spec
    if spec is not None:
        problem = budget_problem(spec, total, active, heal_seconds, stopped_early)
        report["budget"] = {"spec": spec.to_dict(), "params_total": total, "params_active": active,
                            "params_total_b": round(total / 1e9, 6), "params_active_b": round(active / 1e9, 6),
                            "heal_seconds": heal_seconds, "heal_stopped_at_limit": stopped_early,
                            "ok": problem is None}
    with open(os.path.join(output_dir, "program.json"), "w") as f:
        json.dump(program.to_dict(), f, indent=2)
    with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if problem is not None:
        raise CompileError(f"{problem}. Measured: params_total {total}, params_active {active}, "
                           f"heal_seconds {heal_seconds}. The model and report were saved to {output_dir}.")


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
    donors_dir = os.path.join(output_dir, "donors")
    if os.path.isdir(donors_dir):
        shutil.rmtree(donors_dir)
    for name in ["grafts.json", "graft_weights.safetensors", "lora.json", "lora_weights.safetensors",
                 "taps.json", "tap_weights.safetensors", "gates.json", "gate_weights.safetensors",
                 "manifest.json", "skill_moe_weights.safetensors", "insert.json", "insert_weights.safetensors"]:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)


def compile_program(program, catalog, output_dir, align_texts=None, heal_texts=None, device=None, cache_dir=None,
                    tap_pairs=None, heal_pairs=None, calib_texts=None, domain_texts=None):
    """Build the program's model and save it to output_dir. Return output_dir.

    A written program (base_source, carve, or splice) goes to
    compile_written, which uses heal_texts, heal_pairs, calib_texts, and
    domain_texts, and needs no catalog. A program with experts goes to
    compile_experts, one with inserts to compile_inserts, and neither needs
    a catalog. Grafts with gate on get token gates (see compile_grafts).

    align_texts: texts for the W_in ridge fit. None skips the fit, so W_in
        starts random and the heal pass trains it.
    heal_texts: training texts for the heal pass. None loads them from the
        program's heal data name. Not used when the program has taps.
    tap_pairs: (prompt, completion) pairs for the tap training. None loads
        them from wsp.pairs by the tap_train data name.
    heal_pairs: pairs for the heal pass of a program with taps. None uses
        tap_pairs when the heal data name is the tap data name, and loads
        them from wsp.pairs otherwise.

    compile_report.json always has params_total, params_active, and
    heal_seconds. With a budget, a model over a size limit stops the
    compile before the heal with a CompileError that gives the measured
    numbers.
    """
    import transformers

    check_program(program)
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
    check_teacher_tokenizer(program, tokenizer)
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

    if program.experts_spec is not None:
        compile_experts(program, model, tokenizer, output_dir, report, heal_texts, heal_pairs, device, cache_dir)
        return output_dir
    if len(program.inserts) > 0:
        compile_inserts(program, model, tokenizer, output_dir, report, heal_texts, heal_pairs, device, cache_dir)
        return output_dir

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
                       align_texts, heal_texts, device, cache_dir, lora_params, heal_pairs)
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
            check_size_budget(program, model, report, exclude_lora=True)
            run_heal(program, model, tokenizer, heal_texts, lora_params, report)
            # Fold the LoRA into the weights, so the output is a plain HF model.
            merge_lora(model)
            report["lora"]["merged"] = True
        else:
            check_size_budget(program, model, report)
        model.save_pretrained(output_dir)

    finish_outputs(program, model, report, output_dir)
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


def check_teacher_tokenizer(program, tokenizer):
    """Raise CompileError when the heal distills from a teacher with another tokenizer.

    Only the teacher's tokenizer files are read, so this runs before the
    teacher model loads.
    """
    spec = program.heal_spec
    if spec is None or spec.distill is None:
        return None
    import transformers

    from wsp.heal import check_same_tokenizer

    try:
        teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(spec.distill)
    except OSError as error:
        raise CompileError(f"cannot read the tokenizer of the distill teacher {spec.distill}: {error}")
    try:
        check_same_tokenizer(tokenizer, teacher_tokenizer, spec.distill)
    except ValueError as error:
        raise CompileError(str(error))
    return teacher_tokenizer


def load_teacher(program, device):
    """Load the distill teacher in bf16 on device, frozen. Return it."""
    import transformers

    teacher = transformers.AutoModelForCausalLM.from_pretrained(program.heal_spec.distill, dtype=torch.bfloat16)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def run_heal(program, model, tokenizer, heal_texts, params, report, extra_groups=None, extra_loss=None,
             extra_weight=0.0):
    """Run the program's heal pass on params. Record it in report["heal"].

    The caller sets requires_grad. Grafts and the LoRA use the same loop,
    data, and seed, so their results compare fairly. With no extra groups
    and no distillation this is wsp.heal.heal. extra_groups are more AdamW
    groups (each a dict with "params" and "lr"), trained in the same loop.
    extra_loss is a function model -> tensor added with extra_weight.
    A distill heal (the heal spec's distill) loads the teacher on the
    model's device for the heal only. A budget with max_heal_minutes stops
    the loop at the limit and records stopped_early.
    """
    import time

    from wsp.heal import distill_heal, heal, heal_groups

    spec = program.heal_spec
    if heal_texts is None:
        if spec.data in ("gsm8k_train",):
            from wsp.data import task_texts

            heal_texts = task_texts(spec.data, seed=program.seed)
        else:
            heal_texts = written_heal_texts(spec, tokenizer, None, None, program.seed)
    trainable = 0
    for param in params:
        trainable += param.numel()
    if extra_groups is not None:
        for group in extra_groups:
            for param in group["params"]:
                trainable += param.numel()
    limit = heal_limit_seconds(program)
    teacher = None
    if spec.distill is not None:
        check_teacher_tokenizer(program, tokenizer)
        teacher = load_teacher(program, next(model.parameters()).device)
    start = time.time()
    result = None
    if teacher is None and extra_groups is None and extra_loss is None and len(params) > 0:
        losses = heal(model, tokenizer, heal_texts, params, steps=spec.steps, lr=spec.lr,
                      batch_size=spec.batch_size, seq_len=spec.seq_len, seed=program.seed, max_seconds=limit)
    else:
        groups = []
        if len(params) > 0:
            groups.append({"params": list(params), "lr": spec.lr})
        if extra_groups is not None:
            for group in extra_groups:
                groups.append(group)
        result = heal_groups(model, tokenizer, heal_texts, groups, steps=spec.steps, batch_size=spec.batch_size,
                             seq_len=spec.seq_len, seed=program.seed, teacher=teacher,
                             kd_weight=spec.kd_weight, temperature=spec.temperature, extra_loss=extra_loss,
                             extra_weight=extra_weight, max_seconds=limit)
        losses = result["losses"]
    seconds = round(time.time() - start, 3)
    report["heal"] = {"spec": spec.to_dict(), "losses": losses, "seconds": seconds, "trainable_params": trainable,
                      "steps_run": len(losses), "stopped_early": len(losses) < spec.steps}
    if teacher is not None:
        report["heal"]["teacher"] = spec.distill
        report["heal"]["kd_losses"] = result["kd"]
        del teacher
    if result is not None and len(result["extra"]) > 0:
        report["heal"]["extra_losses"] = result["extra"]


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
                   lora_params=None, heal_pairs=None):
    """Build, fit, attach, and heal the grafts. Save their weights.

    lora_params: LoRA parameters on the host, trained by the same heal pass.
    Grafts with gate on get a token gate (wsp.gate), trained in the heal at
    the graft's gate_lr. The gates go to gates.json and
    gate_weights.safetensors, and wsp.load reads them back.
    """
    model.to(device)
    model.eval()
    if heal_texts is None and heal_pairs is not None:
        heal_texts = pair_texts(heal_pairs)
    metas = attach_program_grafts(program, catalog, model, tokenizer, report, align_texts, device, cache_dir)
    gated = []
    for number, spec in enumerate(program.grafts):
        if spec.gate:
            gated.append(number)
    if len(gated) > 0:
        from wsp.gate import attach_gates

        attach_gates(model, init_bias=program.grafts[gated[0]].gate_bias, which=gated)
    check_size_budget(program, model, report)

    if program.heal_spec is not None:
        params, part_lora_count = heal_parameters(model, lora_params)
        extra_groups = None
        gate_count = 0
        if len(gated) > 0:
            from wsp.gate import gate_parameters

            gate_params = gate_parameters(model)
            for param in gate_params:
                param.requires_grad_(True)
                gate_count += param.numel()
            extra_groups = [{"params": gate_params, "lr": program.grafts[gated[0]].gate_lr}]
        run_heal(program, model, tokenizer, heal_texts, params, report, extra_groups=extra_groups)
        report["heal"]["part_lora_params"] = part_lora_count
        if len(gated) > 0:
            report["heal"]["gate_params"] = gate_count
            report["heal"]["gate_lr"] = program.grafts[gated[0]].gate_lr

    save_grafts(model, output_dir, metas)
    if len(gated) > 0:
        from wsp.gate import gate_meta, save_gates

        report["gates"] = gate_meta(model)
        save_gates(model, output_dir)


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
    check_size_budget(program, model, report)

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
    if program.experts_spec is not None or len(program.inserts) > 0:
        raise CompileError("a written program is one plain model, so it cannot have experts or inserts")
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(program.host)
    check_teacher_tokenizer(program, tokenizer)
    sources, base_key = load_written_sources(program)
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

    # Merging the LoRA does not change the parameter count, so the size is
    # checked on the model before the LoRA goes on.
    check_size_budget(program, model, report)

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
    seconds["save"] = round(time.time() - step, 1)
    seconds["total"] = round(time.time() - start, 1)
    report["seconds"] = seconds
    finish_outputs(program, model, report, output_dir)
    return output_dir


# Routed experts: a skill MoE over donor MLPs.


def random_layer_map(n_donor, layer_map, seed):
    """Return a seeded random layer map with the same sharing as layer_map, avoiding the mapped layers.

    layer_map gives, for each host layer, the donor layer its MLP comes
    from. The result uses as many distinct donor layers as layer_map does
    (so two host layers that share a donor layer still share one, and the
    parameter count matches), and sends each used donor layer to a
    different, random donor layer. This is the random control of an experts
    statement: the same donor, the same shapes, other layers. When the
    donor has too few layers to avoid the mapped ones, the fewest possible
    stay.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    distinct = sorted(set(layer_map))
    if len(distinct) > n_donor:
        raise ValueError(f"the layer map uses {len(distinct)} donor layers but the donor has {n_donor}")
    best = None
    for _ in range(200):
        order = torch.randperm(n_donor, generator=generator).tolist()
        mapping = {}
        for position, value in enumerate(distinct):
            mapping[value] = order[position]
        fixed = 0
        for value in distinct:
            if mapping[value] == value:
                fixed += 1
        if best is None or fixed < best[0]:
            best = (fixed, mapping)
        if fixed == 0:
            break
    out = []
    for value in layer_map:
        out.append(best[1][value])
    return out


def expert_kind(host_config, donor_config, spec):
    """Return "plain" for a donor of the host's model type, width, and depth, else "stitched".

    An experts statement with a stitch kind makes every donor stitched.
    """
    if spec.stitch is not None:
        return "stitched"
    host = text_config_of(host_config)
    donor = text_config_of(donor_config)
    if getattr(host, "model_type", None) != getattr(donor, "model_type", None):
        return "stitched"
    if host.hidden_size != donor.hidden_size or host.num_hidden_layers != donor.num_hidden_layers:
        return "stitched"
    return "stitched" if getattr(donor, "num_experts", None) or getattr(donor, "n_routed_experts", None) else "plain"


def build_expert_layers(program, model, cache_dir):
    """Fetch each donor's MLP tensors and build the per-layer experts. Return (layer_experts, rows).

    Only the MLP and norm tensors of the layers in use are fetched, with
    wsp.skill_moe.expert_layer_modules. rows is the manifest's expert list.
    """
    import transformers

    from wsp import skill_moe

    spec = program.experts_spec
    n_base = len(skill_moe.decoder_layers(model))
    host_config = model.config
    d_host = skill_moe.text_config(model).hidden_size
    dtype = next(model.parameters()).dtype
    layer_experts = []
    for _ in range(n_base):
        layer_experts.append([])
    rows = []
    for number, donor in enumerate(spec.donors):
        donor_config = transformers.AutoConfig.from_pretrained(donor)
        kind = expert_kind(host_config, donor_config, spec)
        inner = text_config_of(donor_config)
        n_donor = inner.num_hidden_layers
        if kind == "plain":
            layer_map = list(range(n_base))
        else:
            layer_map = skill_moe.depth_map(n_base, n_donor)
        if spec.control == "random":
            layer_map = random_layer_map(n_donor, layer_map, spec.control_seed + 1000 * number)
        try:
            _, modules = skill_moe.expert_layer_modules(donor, layer_map, dtype, cache_dir)
        except (KeyError, ValueError, OSError) as error:
            raise CompileError(f"cannot build the experts of {donor}: {error}")
        stitch = None
        if kind == "stitched":
            stitch = {"kind": spec.stitch or "linear"}
            if spec.stitch_rank is not None:
                stitch["rank"] = spec.stitch_rank
            if spec.stitch_hidden is not None:
                stitch["hidden"] = spec.stitch_hidden
        row = {"id": donor, "kind": kind, "stitch": stitch, "layer_map": list(layer_map)}
        if kind == "stitched":
            row["d_donor"] = inner.hidden_size
        for index in range(n_base):
            pre_norm, mlp, post_norm = modules[layer_map[index]]
            if kind == "plain":
                layer_experts[index].append(skill_moe.PlainExpert(pre_norm, mlp, post_norm))
            else:
                w_in = skill_moe.make_stitch(stitch, d_host, inner.hidden_size)
                w_out = skill_moe.make_w_out(inner.hidden_size, d_host)
                layer_experts[index].append(skill_moe.StitchedExpert(w_in, pre_norm, mlp, w_out, post_norm))
        rows.append(row)
    return layer_experts, rows


def compile_experts(program, model, tokenizer, output_dir, report, heal_texts, heal_pairs, device, cache_dir):
    """Build the skill MoE, train its routers (and stitches and LoRA) in the heal, and save it.

    The directory gets manifest.json and skill_moe_weights.safetensors
    (wsp.skill_moe.save_skill_moe), the tokenizer, program.json, and
    compile_report.json. wsp.load rebuilds the network and fetches only the
    donor MLP tensors it needs. With a lora statement, a LoRA of that rank
    goes on the attention of the host layers, and the heal trains it.
    """
    from wsp import skill_moe
    from wsp.lora import rank_for_budget

    spec = program.experts_spec
    lora_rank = None
    lora = program.lora_spec
    if lora is not None:
        lora_rank = lora.rank
        if lora_rank is None:
            targets = tuple(lora.targets) if lora.targets is not None else skill_moe.ATTN_TARGETS
            lora_rank = rank_for_budget(model, lora.budget_params, targets, None)
    layer_experts, rows = build_expert_layers(program, model, cache_dir)
    info = {"base_id": program.host, "experts": rows, "top_k": spec.top_k, "router_init": spec.router_init,
            "base_logit": float(spec.base_logit), "load_balance": spec.load_balance,
            "router_noise": float(skill_moe.ROUTER_NOISE), "lora": None}
    skill_moe.assemble(model, layer_experts, info)
    lora_params = []
    if lora_rank is not None:
        targets = skill_moe.ATTN_TARGETS
        if lora.targets is not None:
            targets = tuple(lora.targets)
        lora_params = skill_moe.add_lora(model, lora_rank, lora.alpha, targets)
        report["lora"] = {"rank": lora_rank, "alpha": info["lora"]["alpha"], "targets": list(targets),
                          "layers": None, "params": sum(param.numel() for param in lora_params), "merged": False}
    model.to(device)
    model.eval()
    report["experts"] = {"donors": rows, "top_k": spec.top_k, "router_init": spec.router_init,
                         "control": spec.control, "control_seed": spec.control_seed}
    check_size_budget(program, model, report)

    if program.heal_spec is not None:
        for param in model.parameters():
            param.requires_grad_(False)
        routers = skill_moe.router_parameters(model)
        for param in routers:
            param.requires_grad_(True)
        others = []
        for param in skill_moe.stitch_parameters(model):
            param.requires_grad_(True)
            others.append(param)
        for param in lora_params:
            param.requires_grad_(True)
            others.append(param)
        groups = [{"params": routers, "lr": spec.router_lr, "eps": skill_moe.ROUTER_ADAM_EPS}]
        if len(others) > 0:
            groups.append({"params": others, "lr": program.heal_spec.lr})
        texts = written_heal_texts(program.heal_spec, tokenizer, heal_texts, heal_pairs, program.seed)
        run_heal(program, model, tokenizer, texts, [], report, extra_groups=groups,
                 extra_loss=skill_moe.load_balance_loss if spec.load_balance else None,
                 extra_weight=spec.balance_weight if spec.load_balance else 0.0)
        report["heal"]["router_lr"] = spec.router_lr
    for param in model.parameters():
        param.requires_grad_(False)
    counts = skill_moe.count_params(model)
    report["experts"]["params"] = counts
    skill_moe.save_skill_moe(model, output_dir, tokenizer)
    finish_outputs(program, model, report, output_dir)


# Inserted layers.


def compile_inserts(program, model, tokenizer, output_dir, report, heal_texts, heal_pairs, device, cache_dir):
    """Insert the donor layers, train the stitches, gates, and LoRAs in the heal, and save the network.

    The directory gets insert.json, insert_weights.safetensors, donors/
    (the donor configs), the tokenizer, program.json, and
    compile_report.json (wsp.insert.save_inserted). wsp.load rebuilds the
    network and fetches only the donor layers it needs. Learning rates in
    the heal: stitches, part LoRAs, and the host LoRA at the heal lr, the
    gates at INSERT_GATE_LR (the round 17 value).
    """
    import transformers

    from wsp import insert as insert_ops
    from wsp.assemble import random_block_range

    model.to(device)
    model.eval()
    base_layers = insert_ops.layer_count(model)
    d_base = text_config_of(model.config).hidden_size
    rows = []
    for number, spec in enumerate(program.inserts):
        try:
            config = transformers.AutoConfig.from_pretrained(spec.donor)
        except OSError as error:
            raise CompileError(f"cannot read the config of the insert donor {spec.donor}: {error}")
        inner = text_config_of(config)
        n_donor = inner.num_hidden_layers
        first, last = spec.layers
        if last >= n_donor:
            raise CompileError(f"insert {number} takes donor layers {first}..{last}, but {spec.donor} has "
                               f"{n_donor} layers")
        if spec.at > base_layers:
            raise CompileError(f"insert {number} is at host layer {spec.at}, but the host has {base_layers} layers")
        if spec.control == "random":
            try:
                first, last = random_block_range(n_donor, first, last, spec.control_seed)
            except ValueError as error:
                raise CompileError(str(error))
        kind = spec.stitch
        if kind is None:
            kind = "identity" if inner.hidden_size == d_base else "linear"
        if kind == "identity" and inner.hidden_size != d_base:
            raise CompileError(f"insert {number} has an identity stitch, but the donor width {inner.hidden_size} "
                               f"differs from the host width {d_base}")
        insertion = insert_ops.insert_layers(
            model, spec.donor, [first, last], spec.at, kind, donor_config=config, donor_id=spec.donor,
            stitch_rank=spec.stitch_rank, stitch_hidden=spec.stitch_hidden, part_lora_rank=spec.part_lora_rank,
            gate_init=spec.gate_init, cache_dir=cache_dir)
        rows.append({"donor": spec.donor, "layers": [first, last], "at": spec.at, "stitch": kind,
                     "control": spec.control, "control_seed": spec.control_seed,
                     "d_donor": inner.hidden_size, "part_params": count_module_params(insertion.part)})
    report["inserts"] = rows

    lora = program.lora_spec
    base_lora = False
    if lora is not None:
        rank = lora.rank
        targets = insert_ops.BASE_LORA_TARGETS
        if lora.targets is not None:
            targets = tuple(lora.targets)
        if rank is None:
            from wsp.lora import rank_for_budget

            rank = rank_for_budget(model, lora.budget_params, targets, lora.layers)
        insert_ops.add_base_lora(model, rank, lora.alpha, lora.layers, targets)
        report["lora"] = {"rank": rank, "alpha": model._wsp_base_lora["alpha"],
                          "targets": list(model._wsp_base_lora["targets"]), "layers": lora.layers,
                          "merged": False}
        base_lora = True
    check_size_budget(program, model, report)

    if program.heal_spec is not None:
        spec = program.heal_spec
        lrs = {"stitch": spec.lr, "gate": INSERT_GATE_LR, "part_lora": spec.lr}
        if base_lora:
            lrs["base_lora"] = spec.lr
        groups = []
        for group in insert_ops.heal_groups(model, lrs):
            groups.append({"params": group["params"], "lr": group["lr"]})
        texts = written_heal_texts(spec, tokenizer, heal_texts, heal_pairs, program.seed)
        run_heal(program, model, tokenizer, texts, [], report, extra_groups=groups)
        report["heal"]["gates"] = insert_ops.gate_values(model)
    for param in model.parameters():
        param.requires_grad_(False)
    report["gates"] = insert_ops.gate_values(model)
    insert_ops.save_inserted(model, tokenizer, output_dir, program.host)
    finish_outputs(program, model, report, output_dir)


INSERT_GATE_LR = 1e-2


def count_module_params(module):
    total = 0
    for param in module.parameters():
        total += param.numel()
    return total
