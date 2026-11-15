"""Size and time estimates for a program, and the limit checks that refuse a build.

A spec has three limits: max total parameters, max parameters run per token,
and max heal minutes. The server estimates a program's numbers before any
compute is spent. A build over a limit is refused with a plain message.

The estimates are rough and say so. Part sizes come from the library model
profiles (layer count, hidden size, parameters), as in composer/sizes.py. On
the stub backend they come from the stub world. After a compile, the
measured parameter count replaces the estimate in the result.
"""

from composer import sizes
from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm import stub

from studio.server import library

# A rough heal speed. One heal step on a 1B model at batch 8 and length 384
# takes about 0.3 s; the cost grows with the parameters run per token. This
# is not a measurement of the compile path. Replace it with measured heal
# times when the runs record them.
HEAL_S_PER_STEP_PER_B = 0.3
MIN_HEAL_SCALE_B = 0.5
DEFAULT_STITCH_STEPS = 500
DEFAULT_HOST_HIDDEN = 2048
EPSILON = 1e-6

TAP_MESSAGE = ("A tap runs a second model at inference. Sapien builds one network that runs alone, so a program "
               "cannot have taps.")


class SizeContext:
    """What an estimate needs: the host, the compile catalog, model profiles and, for the stub, its world."""

    def __init__(self, config, catalog, host_b):
        self.config = config
        self.catalog = catalog
        self.host = config["host"]
        self.host_b = float(host_b)
        self.world = None
        if config.get("backend") == "stub":
            self.world = stub.StubWorld(config["stub_seed"])
        self.models = dict(library.get()["models"])
        self.library_parts = library.by_catalog_id()
        info = self.models.get(self.host)
        if info is not None and info.get("hidden"):
            self.host_hidden = info["hidden"]
        else:
            self.host_hidden = DEFAULT_HOST_HIDDEN

    def n_host_layers(self):
        return space.host_layers(self.config, self.catalog)


def record_for(part_id, ctx):
    """A record in the composer's shape (type, model, layers, signature) for a part id, or None."""
    record = ctx.catalog.records.get(part_id)
    if record is None:
        record = sizes.block_record(part_id)
    if record is None:
        entry = ctx.library_parts.get(part_id)
        if entry is not None:
            record = {"id": part_id, "type": entry["kind"], "model": entry["model"], "layers": entry["layers"],
                      "signature": {"n_units": entry["units"]}, "params_b": entry.get("params_b")}
    if record is not None and record.get("type") == "inserted_layer":
        record = dict(record)
        record["type"] = "block"
    return record


def part_size_b(part_id, ctx):
    """(total_b, run_b, donor hidden size) of one part in billions of parameters, or None when unknown."""
    if ctx.world is not None and part_id in ctx.world.parts:
        part = ctx.world.parts[part_id]
        size = ctx.world.part_params(part) / 1e9
        return size, size, stub.STUB_DONORS[part["model"]]["hidden"]
    record = record_for(part_id, ctx)
    if record is None:
        return None
    info = ctx.models.get(record.get("model"))
    hidden = None
    if info is not None:
        hidden = info.get("hidden")
    estimate = sizes.Estimate()
    if record.get("type") == "block":
        layers = record.get("layers")
        if info is None or not isinstance(layers, (list, tuple)) or len(layers) == 0:
            return None
        per_layer = sizes.layer_params(info)
        if per_layer is None:
            return None
        count = max(1, int(layers[-1]) - int(layers[0]) + 1)
        return per_layer * count, per_layer * count * sizes.active_ratio(info), hidden
    sizes.part_size(record, ctx.models, estimate)
    if estimate.total <= 0:
        return None
    return estimate.total, estimate.run, hidden


def glue_b(graft, donor_hidden, ctx):
    """Billions of parameters in the glue of one graft."""
    if donor_hidden is None:
        donor_hidden = ctx.host_hidden
    return stub.glue_params(ctx.host_hidden, donor_hidden, ps.graft_glue_rank(graft)) / 1e9


def stitch_steps(program):
    """Training steps spent on stitches: each graft with a stitch spec counts its steps or 500."""
    total = 0
    for graft in program.get("grafts") or []:
        stitch = graft.get("stitch")
        if isinstance(stitch, dict):
            total += int(stitch.get("steps") or DEFAULT_STITCH_STEPS)
    return total


def heal_minutes(program, run_b):
    """Rough minutes of heal and stitch training for a program."""
    steps = (ps.heal_steps(program) or 0) + stitch_steps(program)
    scale = max(float(run_b), MIN_HEAL_SCALE_B)
    return steps * HEAL_S_PER_STEP_PER_B * scale / 60.0


def estimate(program, ctx):
    """Estimated size and heal time of a program.

    Returns {"total_params_b", "run_params_b", "heal_min", "host_params_b", "added_params_b", "unknown",
    "parts"}. A part whose size is unknown adds nothing and is listed in "unknown".
    """
    total = ctx.host_b
    run = ctx.host_b
    unknown = []
    parts = []
    for graft in ps.grafts(program):
        if ps.graft_control(graft) is not None:
            continue
        part_id = ps.graft_id(graft)
        found = part_size_b(part_id, ctx)
        if found is None:
            unknown.append(part_id)
            continue
        part_total, part_run, hidden = found
        glue = glue_b(graft, hidden, ctx)
        total += part_total + glue
        run += part_run + glue
        parts.append({"id": part_id, "params_b": round(part_total + glue, 4)})
    unsized = []
    for key in ("carve", "splice", "base_source"):
        if program.get(key) is not None:
            unsized.append("a " + key + " step")
            unknown.append("a " + key + " step")
    add_total, add_run = inserts_size_b(program, ctx, unsized)
    total += add_total
    run += add_run
    add_total, add_run = experts_size_b(program, ctx, unsized)
    total += add_total
    run += add_run
    return {"total_params_b": round(total, 4), "run_params_b": round(run, 4),
            "heal_min": round(heal_minutes(program, run), 2), "host_params_b": round(ctx.host_b, 4),
            "added_params_b": round(total - ctx.host_b, 4), "unknown": unknown, "parts": parts,
            "unsized_steps": unsized, "estimated": True}


def stitch_b(kind, rank, hidden_width, host_hidden, donor_hidden):
    """Billions of parameters in the two stitches of an inserted layer or an expert (rough)."""
    if kind is None or kind == "identity":
        if host_hidden == donor_hidden:
            return 0.0
        kind = "linear"
    if kind == "linear":
        return 2.0 * host_hidden * donor_hidden / 1e9
    if kind == "lowrank":
        return 2.0 * (rank or 1) * (host_hidden + donor_hidden) / 1e9
    width = hidden_width or host_hidden
    return 2.0 * (host_hidden + donor_hidden) * width / 1e9


def inserts_size_b(program, ctx, unsized):
    """(total, run) billions added by inserted donor layers. A donor with no profile goes to unsized."""
    total = 0.0
    run = 0.0
    for item in program.get("inserts") or []:
        info = ctx.models.get(item.get("donor"))
        per_layer = None
        if info is not None:
            per_layer = sizes.layer_params(info)
        if per_layer is None:
            unsized.append("inserted layers from " + str(item.get("donor")))
            continue
        first, last = item["layers"]
        count = max(1, int(last) - int(first) + 1)
        donor_hidden = info.get("hidden") or ctx.host_hidden
        glue = stitch_b(item.get("stitch"), item.get("stitch_rank"), item.get("stitch_hidden"), ctx.host_hidden,
                        donor_hidden)
        total += per_layer * count + glue
        run += per_layer * count * sizes.active_ratio(info) + glue
    return total, run


MLP_SHARE = 2.0 / 3.0


def experts_size_b(program, ctx, unsized):
    """(total, run) billions added by routed donor experts: each donor adds its MLPs for the host's layers.

    An MLP is taken as two thirds of a layer, as in composer/sizes.py. The run size adds the top_k - 1 largest
    donor MLP sets, so it is an upper bound.
    """
    spec = program.get("experts")
    if not spec:
        return 0.0, 0.0
    n_host = ctx.n_host_layers()
    donor_sets = []
    for donor in spec.get("donors") or []:
        info = ctx.models.get(donor)
        per_layer = None
        if info is not None:
            per_layer = sizes.layer_params(info)
        if per_layer is None:
            unsized.append("experts from " + str(donor))
            continue
        count = min(n_host, int(info.get("layers") or n_host))
        glue = stitch_b(spec.get("stitch"), spec.get("stitch_rank"), spec.get("stitch_hidden"), ctx.host_hidden,
                        info.get("hidden") or ctx.host_hidden)
        donor_sets.append(per_layer * MLP_SHARE * count + glue * count)
    total = sum(donor_sets)
    donor_sets.sort(reverse=True)
    extra_run = sum(donor_sets[:max(0, int(spec.get("top_k") or 1) - 1)])
    return total, extra_run


def fmt_b(value):
    """A size in billions with up to 3 decimals, so a build just over a limit does not read as equal to it."""
    return ("%.3f" % value).rstrip("0").rstrip(".")


def fmt_min(value):
    if value >= 10:
        return str(int(round(value)))
    return ("%.1f" % value).rstrip("0").rstrip(".")


def limit_problems(limits, est):
    """Plain messages for every limit the estimate is over. An empty list means the build fits."""
    problems = []
    if limits is None:
        return problems
    if len(est.get("unsized_steps") or []) > 0:
        problems.append("The studio cannot size " + ", ".join(est["unsized_steps"]) + ", so it cannot show that " +
                        "this build fits the limits. Remove it or build it outside the studio.")
    max_total = limits.get("max_total_params_b")
    if max_total is not None and est["total_params_b"] > max_total + EPSILON:
        problems.append("Over the total size limit. This build is estimated at " + fmt_b(est["total_params_b"]) +
                        "B parameters (the host is " + fmt_b(est["host_params_b"]) + "B and the added parts are " +
                        fmt_b(est["added_params_b"]) + "B). The limit is " + fmt_b(max_total) + "B.")
    max_run = limits.get("max_run_params_b")
    if max_run is not None and est["run_params_b"] > max_run + EPSILON:
        problems.append("Over the per-token limit. This build runs an estimated " + fmt_b(est["run_params_b"]) +
                        "B parameters per token. The limit is " + fmt_b(max_run) + "B.")
    max_heal = limits.get("max_heal_min")
    if max_heal is not None and est["heal_min"] > max_heal + EPSILON:
        problems.append("Over the heal time limit. The heal and stitch training is estimated at " +
                        fmt_min(est["heal_min"]) + " minutes. The limit is " + fmt_min(max_heal) + " minutes.")
    return problems


def forbidden_problems(program):
    """Problems that no limit can fix. Taps run a second model at inference, which Sapien does not do."""
    problems = []
    if len(program.get("taps") or []) > 0 or program.get("tap_train") is not None:
        problems.append(TAP_MESSAGE)
    return problems


def check(program, limits, ctx):
    """Estimate a program and check it. Returns (estimate, problems)."""
    est = estimate(program, ctx)
    problems = forbidden_problems(program)
    problems.extend(limit_problems(limits, est))
    return est, problems


def measured_problems(limits, params_b):
    """Limit messages for a measured parameter count after a compile (total size only)."""
    if limits is None or params_b is None:
        return []
    max_total = limits.get("max_total_params_b")
    if max_total is not None and params_b > max_total + EPSILON:
        return ["Over the total size limit. The compiled model has " + fmt_b(params_b) + "B parameters. The limit is " +
                fmt_b(max_total) + "B."]
    return []
