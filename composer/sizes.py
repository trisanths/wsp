"""Size estimates and constraint checks for a program, before it compiles.

The estimate uses model sizes from the atlas and profiles.json. It is
rough on purpose and leans high: a decoder layer counts as
(params - embedding) / layers. The compile on Modal measures the real
total, and the composer checks the constraints again on that number.

Params run per token: a graft, splice, or carved layer runs on every
token. An MoE model runs active_params_b of params_b. A tap donor runs
on the prompt, up to the layer the tap reads; we count those layers as
run per token, which over-counts decode cost but keeps the check safe.
"""

# Vocabulary sizes by tokenizer family, for the embedding share.
VOCAB = {
    "qwen2": 151936,
    "llama3": 128256,
    "gemma": 256000,
    "phi3": 32064,
    "phi4": 100352,
    "mistral_v3": 32768,
    "mistral_v1": 32000,
    "lfm2": 65536,
}

# An MLP is about two thirds of a decoder layer (the rest is attention and norms).
MLP_SHARE = 2.0 / 3.0
BLOCK_PREFIX = "block/"
# GraftSpec trains a stitch for 500 steps when the stitch names no steps.
DEFAULT_STITCH_STEPS = 500


def block_record(component_id):
    """Read a graft id of the form block/<org>/<model>/<first>-<last>.

    Return a catalog block record, or None when the id is not of that form.
    """
    if not component_id.startswith(BLOCK_PREFIX):
        return None
    rest = component_id[len(BLOCK_PREFIX):]
    if "/" not in rest:
        return None
    model, span = rest.rsplit("/", 1)
    if "-" not in span:
        return None
    first, last = span.split("-", 1)
    if not first.isdigit() or not last.isdigit():
        return None
    if model == "":
        return None
    return {"id": component_id, "type": "block", "model": model, "layers": [int(first), int(last)]}


def embedding_params(info):
    vocab = info.get("vocab_size")
    if vocab is None:
        vocab = VOCAB.get(info.get("tokenizer_family"))
    hidden = info.get("hidden")
    if vocab is None or hidden is None:
        return 0.0
    return vocab * hidden / 1e9


def layer_params(info):
    """Billions of parameters in one decoder layer (rough)."""
    layers = info.get("layers")
    if not layers:
        return None
    body = info["params_b"] - embedding_params(info)
    if body <= 0:
        body = info["params_b"]
    return body / layers


def active_ratio(info):
    total = info.get("params_b")
    active = info.get("active_params_b")
    if not total or not active:
        return 1.0
    return min(1.0, active / total)


class Estimate:
    def __init__(self):
        self.total = 0.0
        self.run = 0.0
        self.unknown = []
        self.notes = []

    def add(self, total, run, note):
        self.total += total
        self.run += run
        self.notes.append(f"{note}: +{total:.3f}b total, +{run:.3f}b run")

    def to_dict(self):
        return {"total_params_b": round(self.total, 3), "run_params_b": round(self.run, 3),
                "unknown_models": list(self.unknown), "notes": list(self.notes)}


def find_info(models, name):
    if name in models:
        return models[name]
    # Match by the last part of the id too (unsloth/x vs org/x copies are
    # different models, so only exact ids count here).
    return None


def part_size(record, models, estimate):
    """Add one catalog part (not a block) to the estimate."""
    kind = record.get("type")
    model = record.get("model") or record.get("base")
    info = find_info(models, model) if model else None
    d = record.get("d_model")
    if d is None and info is not None:
        d = info.get("hidden")
    units = record.get("num_units")
    signature = record.get("signature")
    if units is None and isinstance(signature, dict):
        units = signature.get("n_units")
    if kind == "mlp_cluster":
        if d is None or units is None:
            estimate.add(0.05, 0.05, f"{record['id']} (mlp_cluster, size guessed)")
            return
        size = 3.0 * units * d / 1e9
        estimate.add(size, size, f"{record['id']} ({units} neurons)")
        return
    if kind == "attn_heads":
        if d is None or units is None:
            estimate.add(0.05, 0.05, f"{record['id']} (attn_heads, size guessed)")
            return
        size = 4.0 * units * 128 * d / 1e9
        estimate.add(size, size, f"{record['id']} ({units} heads)")
        return
    if kind == "expert":
        size = record.get("params_b")
        if size is None:
            size = 0.1
        estimate.add(size, size, f"{record['id']} (expert)")
        return
    # Masks and task vectors do not change the size.


def host_info_of(program, models):
    info = find_info(models, program.host)
    if info is None or not info.get("params_b"):
        return None
    return info


def experts_size(program, host_info, models, estimate, info_for):
    """Add the donors' MLPs, the routers, and the stitches of an experts statement."""
    spec = program.experts_spec
    if host_info is None or not host_info.get("layers") or not host_info.get("hidden"):
        if host_info is not None:
            estimate.unknown.append(f"{program.host} (layers or hidden size missing, needed for experts)")
        return
    layers = host_info["layers"]
    d_host = host_info["hidden"]
    host_mlp = layer_params(host_info) * MLP_SHARE
    pool = [host_mlp]
    count = len(spec.donors) + 1
    for donor in spec.donors:
        info = info_for(donor)
        if info is None:
            continue
        per_layer = layer_params(info)
        if per_layer is None:
            continue
        mlp = per_layer * MLP_SHARE
        stitch = 0.0
        d_donor = info.get("hidden")
        if d_donor is not None and (d_donor != d_host or info.get("layers") != layers or spec.stitch is not None):
            stitch = 2.0 * d_host * d_donor / 1e9
        used = min(layers, info.get("layers") or layers)
        total = mlp * used + stitch * layers
        estimate.add(total, 0.0, f"experts donor {donor} ({used} MLPs)")
        pool.append(mlp + stitch)
    router = layers * (d_host * count + count) / 1e9
    ranked = sorted(pool, reverse=True)
    run = layers * (sum(ranked[:spec.top_k]) - host_mlp) + router
    estimate.add(router, max(0.0, run), f"experts routers and top_{spec.top_k}")


def insert_stitch_params(spec, d_host, d_donor):
    """Parameters of the stitches around inserted layers, in billions (rough, high)."""
    kind = spec.stitch
    if kind is None:
        kind = "identity" if d_host == d_donor else "linear"
    if kind == "identity":
        return 0.0
    if kind == "linear":
        return 2.0 * d_host * d_donor / 1e9
    if kind == "lowrank":
        return 2.0 * (spec.stitch_rank or 1) * (d_host + d_donor) / 1e9
    hidden = spec.stitch_hidden or 2 * d_donor
    return (2.0 * hidden * (d_host + d_donor)) / 1e9


def insert_size(spec, host_info, estimate, info_for):
    """Add the donor layers and the stitches of one insert statement."""
    info = info_for(spec.donor)
    if info is None or host_info is None:
        return
    per_layer = layer_params(info)
    if per_layer is None:
        return
    count = spec.layers[1] - spec.layers[0] + 1
    stitch = 0.0
    if info.get("hidden") and host_info.get("hidden"):
        stitch = insert_stitch_params(spec, host_info["hidden"], info["hidden"])
    size = per_layer * count * 1.0 + stitch
    estimate.add(size, size * active_ratio(info), f"insert {spec.donor} layers {spec.layers[0]}..{spec.layers[1]}")


def graft_glue_params(graft, host_info, donor_info):
    """Billions of parameters in the w_in and w_out glue of a block graft (full rank unless glue_rank is set)."""
    if host_info is None or not host_info.get("hidden") or not donor_info.get("hidden"):
        return 0.0
    d_host = host_info["hidden"]
    d_donor = donor_info["hidden"]
    if graft.glue_rank:
        return 2.0 * graft.glue_rank * (d_host + d_donor) / 1e9
    return 2.0 * d_host * d_donor / 1e9


def estimate_program(program, models, records):
    """Return an Estimate of the compiled program's size.

    models: {hf_id: {"params_b", "active_params_b", "layers", "hidden", ...}}.
    records: {catalog id: record}.
    """
    estimate = Estimate()

    def info_for(name):
        info = find_info(models, name)
        if info is None or not info.get("params_b"):
            if name not in estimate.unknown:
                estimate.unknown.append(name)
            return None
        return info

    if program.splice_spec is not None:
        spec = program.splice_spec
        base_info = info_for(spec.sources[spec.base])
        if base_info is not None:
            embed = embedding_params(base_info)
            estimate.add(embed, embed, "splice base embeddings")
        for key, layer in spec.plan:
            info = info_for(spec.sources[key])
            if info is None:
                continue
            size = layer_params(info)
            if size is None:
                continue
            estimate.add(size, size * active_ratio(info), f"splice {key}:{layer}")
    else:
        host_info = info_for(program.host)
        if host_info is not None:
            total = host_info["params_b"]
            estimate.add(total, total * active_ratio(host_info), f"host {program.host}")

    if program.carve_spec is not None and len(estimate.unknown) == 0:
        spec = program.carve_spec
        if spec.target_params_b is not None:
            ratio = estimate.run / estimate.total if estimate.total > 0 else 1.0
            estimate.total = spec.target_params_b
            estimate.run = spec.target_params_b * ratio
            estimate.notes.append(f"carve to target {spec.target_params_b}b")
        elif spec.keep is not None:
            layers = None
            host_info = find_info(models, program.host)
            if program.splice_spec is not None:
                layers = len(program.splice_spec.plan)
            elif host_info is not None:
                layers = host_info.get("layers")
            if layers:
                keep = min(spec.keep, layers)
                embed = 0.0
                if host_info is not None:
                    embed = embedding_params(host_info)
                body_total = estimate.total - embed
                body_run = estimate.run - embed
                estimate.total = embed + body_total * keep / layers
                estimate.run = embed + body_run * keep / layers
                estimate.notes.append(f"carve keep {keep} of {layers} layers")
        if spec.width is not None:
            keep_mlp = spec.width.get("keep_mlp", 1.0)
            keep_heads = spec.width.get("keep_heads", 1.0)
            # MLP is about two thirds of a layer, attention one third.
            scale = (2.0 * keep_mlp + keep_heads) / 3.0
            estimate.total = estimate.total * scale
            estimate.run = estimate.run * scale
            estimate.notes.append(f"carve width mlp {keep_mlp} heads {keep_heads}")

    for graft in program.grafts:
        record = records.get(graft.component_id)
        if record is None:
            record = block_record(graft.component_id)
        if record is None:
            estimate.unknown.append(graft.component_id)
            continue
        if record.get("type") == "block":
            info = info_for(record["model"])
            if info is None:
                continue
            size = layer_params(info)
            if size is None:
                continue
            first, last = record["layers"]
            # A reversed range is a violation; it must never lower the estimate.
            count = max(1, last - first + 1)
            glue = graft_glue_params(graft, find_info(models, program.host), info)
            estimate.add(size * count + glue, size * count * active_ratio(info) + glue,
                         f"graft {graft.component_id} ({count} layers)")
        else:
            part_size(record, models, estimate)

    if program.experts_spec is not None:
        experts_size(program, host_info_of(program, models), models, estimate, info_for)
    for spec in program.inserts:
        insert_size(spec, host_info_of(program, models), estimate, info_for)

    tap_donors = {}
    for tap in program.taps:
        deepest = tap_donors.get(tap.donor, -1)
        if tap.donor_layer > deepest:
            tap_donors[tap.donor] = tap.donor_layer
    for donor in tap_donors:
        info = info_for(donor)
        if info is None:
            continue
        layers = info.get("layers") or 1
        share = min(1.0, (tap_donors[donor] + 1) / layers)
        estimate.add(info["params_b"], info["params_b"] * share * active_ratio(info), f"tap donor {donor}")
    return estimate


def training_steps(program):
    """Return a list of (what, steps) for every training stage the program runs."""
    stages = []
    if program.heal_spec is not None:
        stages.append(("heal", program.heal_spec.steps))
    if program.tap_train_spec is not None:
        stages.append(("tap_train", program.tap_train_spec.steps))
    for graft in program.grafts:
        if isinstance(graft.stitch, dict):
            steps = graft.stitch.get("steps", DEFAULT_STITCH_STEPS)
            stages.append((f"stitch on {graft.component_id}", steps))
    return stages


def layer_range_violations(program, models, records):
    """Layer numbers outside a model: a reversed or too deep block, a too deep splice layer or tap."""
    violations = []
    for graft in program.grafts:
        record = records.get(graft.component_id)
        if record is None:
            record = block_record(graft.component_id)
        if record is None or record.get("type") != "block":
            continue
        layers = record.get("layers")
        if not isinstance(layers, (list, tuple)) or len(layers) != 2:
            violations.append(f"block {graft.component_id} needs layers [first, last]")
            continue
        first, last = layers
        if first > last:
            violations.append(f"block {graft.component_id}: the first layer {first} is after the last {last}")
        info = find_info(models, record.get("model"))
        if info is not None and info.get("layers") and last >= info["layers"]:
            violations.append(f"block {graft.component_id}: {record.get('model')} has layers 0 to "
                              f"{info['layers'] - 1}")
    if program.splice_spec is not None:
        spec = program.splice_spec
        for key, layer in spec.plan:
            info = find_info(models, spec.sources[key])
            if info is not None and info.get("layers") and layer >= info["layers"]:
                violations.append(f"splice {key}:{layer}: {spec.sources[key]} has layers 0 to {info['layers'] - 1}")
    host_info = find_info(models, program.host)
    for spec in program.inserts:
        first, last = spec.layers
        info = find_info(models, spec.donor)
        if info is not None and info.get("layers") and last >= info["layers"]:
            violations.append(f"insert {spec.donor} layers {first}..{last}: the donor has layers 0 to "
                              f"{info['layers'] - 1}")
        if host_info is not None and host_info.get("layers") and spec.at > host_info["layers"]:
            violations.append(f"insert at {spec.at}: the host {program.host} has {host_info['layers']} layers, "
                              f"so at is 1 to {host_info['layers']}")
    for tap in program.taps:
        info = find_info(models, tap.donor)
        if info is not None and info.get("layers") and tap.donor_layer >= info["layers"]:
            violations.append(f"tap {tap.donor} layer {tap.donor_layer}: the donor has layers 0 to "
                              f"{info['layers'] - 1}")
    return violations


def composes_something(program):
    """True when the program uses at least one mined or source component beyond its host."""
    if len(program.grafts) > 0 or len(program.taps) > 0:
        return True
    if program.experts_spec is not None or len(program.inserts) > 0:
        return True
    if program.splice_spec is not None or program.carve_spec is not None:
        return True
    if program.masks is not None or len(program.task_vectors) > 0:
        return True
    return False


TAP_MESSAGE = ("the composer cannot write taps: a tap runs a source model beside the host at inference, so the "
               "result is not one network that runs alone, which Sapien requires. Bring the donor's computation "
               "into the host with a graft, experts, or an insert instead.")


def added_kinds(program):
    """The kinds of added components the program holds: graft, experts, insert, tap."""
    kinds = []
    if len(program.grafts) > 0:
        kinds.append("graft")
    if program.experts_spec is not None:
        kinds.append("experts")
    if len(program.inserts) > 0:
        kinds.append("insert")
    if len(program.taps) > 0:
        kinds.append("tap")
    return kinds


def tokenizer_family(models, name):
    info = find_info(models, name)
    if info is None:
        return None
    return info.get("tokenizer_family")


def same_tokenizer(models, first, second):
    """True or False when both tokenizer families are known, else None."""
    a = tokenizer_family(models, first)
    b = tokenizer_family(models, second)
    if a is None or b is None:
        return None
    return a == b


def compile_conflicts(program, kinds):
    """Combinations that wsp.compiler.check_program and check_written refuse. Each would cost a compile."""
    problems = []
    has_mask = program.masks is not None or len(program.task_vectors) > 0
    if program.is_written():
        # compile_written has its own rules and ignores the others.
        for kind in kinds:
            if kind in ("graft", "experts", "insert"):
                problems.append("a program that writes one plain model (from, carve, splice) cannot have "
                                "grafts, experts, or inserts")
                break
        if has_mask:
            problems.append("a program that writes one plain model (from, carve, splice) cannot have masks or "
                            "task vectors")
        if program.lora_spec is None and program.heal_spec is not None:
            problems.append("in a program that writes one plain model, heal trains a LoRA that is merged at the "
                            "end: add a lora statement")
    else:
        if ("experts" in kinds or "insert" in kinds) and has_mask:
            problems.append("masks and task vectors cannot join experts or inserts in one program")
        spec = program.lora_spec
        if program.experts_spec is not None and spec is not None and spec.layers is not None:
            problems.append("an experts program puts its LoRA on the attention of every layer: drop the lora "
                            "layers")
        if program.heal_spec is not None and spec is None and len(kinds) == 0:
            problems.append("this program has a heal but nothing to train: add a lora statement, a graft, "
                            "experts, or an insert")
    if program.lora_spec is not None and program.heal_spec is None:
        problems.append("a program with a lora needs a heal statement to train it")
    return problems


def structure_violations(program, models):
    """Violations of the language rules that the compile would raise after it has started."""
    violations = []
    if len(program.taps) > 0 or program.tap_train_spec is not None:
        violations.append(TAP_MESSAGE)
    kinds = added_kinds(program)
    if len(kinds) > 1:
        violations.append("a program holds one kind of added component, and this one mixes " + ", ".join(kinds) +
                          ". Use grafts (gated grafts fuse several donors), or experts, or inserts.")
    gated = []
    for graft in program.grafts:
        if graft.gate:
            gated.append(graft)
    for graft in gated[1:]:
        if graft.gate_bias != gated[0].gate_bias or graft.gate_lr != gated[0].gate_lr:
            violations.append("grafts with gate on must share gate_bias and gate_lr")
            break
    violations += compile_conflicts(program, kinds)
    if program.heal_spec is not None and program.heal_spec.distill is not None:
        teacher = program.heal_spec.distill
        match = same_tokenizer(models, program.host, teacher)
        if match is None:
            violations.append(f"cannot confirm that the distillation teacher {teacher} has the tokenizer of "
                              f"{program.host}: one of them is not in the atlas. Use models listed by atlas().")
        elif not match:
            violations.append(f"the distillation teacher {teacher} has another tokenizer than {program.host}, "
                              f"and a distillation heal needs the same tokenizer")
    return violations


def budget_dict(constraints, program_budget=None):
    """The Sapien budget for a compile: the request limits, tightened by the program's own budget.

    Return a dict for BudgetSpec.from_dict, or None when nothing is limited.
    """
    limits = {"max_total_params_b": constraints.get("max_total_params_b"),
              "max_active_params_b": constraints.get("max_run_params_b"),
              "max_heal_minutes": constraints.get("max_heal_minutes")}
    if program_budget is not None:
        own = program_budget.to_dict()
        for name in limits:
            value = own.get(name)
            if value is None:
                continue
            if limits[name] is None or value < limits[name]:
                limits[name] = value
    if limits["max_total_params_b"] is None and limits["max_active_params_b"] is None and \
            limits["max_heal_minutes"] is None:
        return None
    return limits


def with_request_budget(program, constraints):
    """Set the request limits as the program's budget, so the compile enforces them too."""
    from wsp.program import BudgetSpec

    limits = budget_dict(constraints, program.budget_spec)
    if limits is None:
        return program
    program.budget_spec = BudgetSpec.from_dict(limits)
    return program


def check_program(program, constraints, models, records):
    """Check a parsed program against a request's constraints.

    Return {"ok": bool, "violations": [text], "estimate": {...}}.
    """
    violations = structure_violations(program, models)
    violations += layer_range_violations(program, models, records)
    estimate = estimate_program(program, models, records)
    if len(estimate.unknown) > 0:
        violations.append("unknown size for: " + ", ".join(estimate.unknown) +
                          ". Use models and parts listed by atlas() and catalog(), or a block id "
                          "block/<org>/<model>/<first>-<last> for a model in the atlas.")
    if len(program.taps) == 0:
        # A tap program is rejected above; its donor sizes would only repeat the message.
        max_total = constraints.get("max_total_params_b")
        if max_total is not None and estimate.total > max_total:
            violations.append(f"estimated total params {estimate.total:.3f}b is over the cap {max_total}b")
        max_run = constraints.get("max_run_params_b")
        if max_run is not None and estimate.run > max_run:
            violations.append(f"estimated params run per token {estimate.run:.3f}b is over the cap {max_run}b")
    cap = constraints.get("heal_steps_cap")
    if cap is not None:
        for what, steps in training_steps(program):
            if steps is not None and steps > cap:
                violations.append(f"{what} has {steps} steps, over the cap of {cap}")
    if not composes_something(program):
        violations.append("the program composes nothing: a host with only a LoRA and heal (or a distillation "
                          "heal) is a fine-tuning baseline. Add grafts, experts, inserts, a splice, a carve, "
                          "masks, or task vectors.")
    return {"ok": len(violations) == 0, "violations": violations, "estimate": estimate.to_dict()}


def check_measured(measured, constraints):
    """Check measured numbers from a compile. Return a list of violations."""
    violations = []
    max_total = constraints.get("max_total_params_b")
    total = measured.get("params_total_b")
    if max_total is not None and total is not None and total > max_total:
        violations.append(f"measured total params {total:.3f}b is over the cap {max_total}b")
    max_run = constraints.get("max_run_params_b")
    run = measured.get("params_run_b")
    if max_run is not None and run is not None and run > max_run:
        violations.append(f"measured params run per token {run:.3f}b is over the cap {max_run}b")
    max_minutes = constraints.get("max_heal_minutes")
    seconds = measured.get("heal_seconds")
    if max_minutes is not None and seconds is not None and seconds > max_minutes * 60.0 and \
            not measured.get("heal_stopped_at_limit"):
        violations.append(f"the heal took {seconds / 60.0:.2f} minutes, over the cap {max_minutes}")
    return violations
