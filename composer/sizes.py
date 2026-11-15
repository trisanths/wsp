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
            estimate.add(size * count, size * count * active_ratio(info),
                         f"graft {graft.component_id} ({count} layers)")
        else:
            part_size(record, models, estimate)

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
    if program.splice_spec is not None or program.carve_spec is not None:
        return True
    if program.masks is not None or len(program.task_vectors) > 0:
        return True
    return False


def check_program(program, constraints, models, records):
    """Check a parsed program against a request's constraints.

    Return {"ok": bool, "violations": [text], "estimate": {...}}.
    """
    violations = layer_range_violations(program, models, records)
    estimate = estimate_program(program, models, records)
    if len(estimate.unknown) > 0:
        violations.append("unknown size for: " + ", ".join(estimate.unknown) +
                          ". Use models and parts listed by atlas() and catalog(), or a block id "
                          "block/<org>/<model>/<first>-<last> for a model in the atlas.")
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
        violations.append("the program composes nothing: a host with only a LoRA and heal is the fine-tuning "
                          "baseline. Add grafts, taps, a splice, a carve, masks, or task vectors.")
    return {"ok": len(violations) == 0, "violations": violations, "estimate": estimate.to_dict()}


def check_measured(measured, constraints):
    """Check measured sizes from a compile. Return a list of violations."""
    violations = []
    max_total = constraints.get("max_total_params_b")
    total = measured.get("params_total_b")
    if max_total is not None and total is not None and total > max_total:
        violations.append(f"measured total params {total:.3f}b is over the cap {max_total}b")
    max_run = constraints.get("max_run_params_b")
    run = measured.get("params_run_b")
    if max_run is not None and run is not None and run > max_run:
        violations.append(f"measured params run per token {run:.3f}b is over the cap {max_run}b")
    return violations
