"""The sapien command line: check, compile, score, library, atlas.

    sapien check prog.sap [--catalog catalog.jsonl] [--json]
    sapien compile prog.sap -o out_dir [--catalog catalog.jsonl] [--control random]
    sapien score out_dir --skills gsm8k,mmlu
    sapien library query --kind block --skill math
    sapien atlas best --skill math --max-params-b 2

python -m wsp.cli does the same. A program is a text file (.sap, or .wsp)
or a .json file from Program.to_dict(). The syntax is in docs/language.md.

check parses and validates a program and estimates its size from the
model configs (config.json only, no weights), then compares the estimate
with the budget. compile builds the model. Both refuse a program with
taps unless --allow-parallel is given, because a tap runs a source model
at inference, so the result is not one network that runs alone. That is
not Sapien.

Exit codes: 0 on success, 1 on an error or an estimate over the budget.
"""

import argparse
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_ENTRIES = os.path.join(REPO, "atlas", "entries")

TAPS_REFUSAL = ("this program has taps. A tap reads a source model at inference, so the compiled model is not one "
                "network that runs alone. That is not Sapien. Pass --allow-parallel to run it as a research "
                "program anyway.")


class CliError(Exception):
    """A problem to print as "error: ..." with exit code 1."""


# Reading a program


def read_program(path):
    """Read a program file (.sap, .wsp, or .json). Raise CliError with the file name on a problem."""
    from wsp.lang import load_program

    try:
        return load_program(path)
    except OSError as error:
        raise CliError(f"{path}: {error.strerror}")
    except ValueError as error:
        raise CliError(f"{path}: {error}")


def refuse_taps(program, allow_parallel):
    if program.has_taps() and not allow_parallel:
        raise CliError(TAPS_REFUSAL)


# Size estimate from configs


def format_size(value):
    if value is None:
        return "unknown"
    return f"{value / 1e9:.3f}b"


def config_counts(config):
    """Build the model from a config on the meta device and measure it. Return a dict.

    Keys: total, embed (the input embedding table), tied, layers (a list of
    {"total", "mlp", "attn"} per decoder layer), other (everything outside
    the decoder layers), hidden, n_layers, per_rank (LoRA parameters per
    unit of rank on the default targets).
    """
    import torch
    import transformers

    from wsp.lora import DEFAULT_TARGETS, params_per_rank
    from wsp.parts import find_decoder_layers

    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    embed = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    tied = head is not None and head.weight is embed
    _, layers = find_decoder_layers(model, 0)
    rows = []
    in_layers = 0
    for layer in layers:
        row = {"total": 0, "mlp": 0, "attn": 0}
        for param in layer.parameters():
            row["total"] += param.numel()
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for param in mlp.parameters():
                row["mlp"] += param.numel()
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for param in attn.parameters():
                row["attn"] += param.numel()
        in_layers += row["total"]
        rows.append(row)
    inner = config.get_text_config() if hasattr(config, "get_text_config") else config
    def per_rank(targets=DEFAULT_TARGETS, layers=None):
        """LoRA parameters per unit of rank on these targets (0 when the model has none)."""
        try:
            return params_per_rank(model, tuple(targets), layers)
        except ValueError:
            return 0

    return {"total": total, "embed": embed.numel(), "tied": bool(tied), "layers": rows, "other": total - in_layers,
            "hidden": inner.hidden_size, "n_layers": len(rows), "per_rank": per_rank,
            "sparse": bool(getattr(inner, "num_experts", None) or getattr(inner, "n_routed_experts", None))}


def default_config_loader(model_id):
    import transformers

    return transformers.AutoConfig.from_pretrained(model_id)


def lora_param_estimate(spec, host_counts, default_targets=None):
    """LoRA parameters of the host for a lora statement, or None when the rank is unknown.

    default_targets: the linears the LoRA wraps when the statement names
    none (the compiler uses another list for experts and inserts).
    """
    if spec is None:
        return 0
    targets = spec.targets
    if targets is None:
        targets = default_targets
    if targets is None:
        per_rank = host_counts["per_rank"]()
    else:
        per_rank = host_counts["per_rank"](targets, spec.layers)
    if spec.rank is not None:
        return spec.rank * per_rank
    if spec.budget_params is not None and per_rank > 0:
        return (spec.budget_params // per_rank) * per_rank
    return None


def estimate_program(program, catalog=None, load_config=default_config_loader):
    """Estimate params_total and params_active from model configs. Return a dict.

    The dict has params_total, params_active (None when unknown), notes
    (what is approximate), and unknown (parts the estimate cannot size).
    The estimate builds each config on the meta device, so no weights
    load. A donor that cannot be read goes in unknown.
    """
    notes = []
    unknown = []
    cache = {}

    def counts_of(model_id):
        if model_id not in cache:
            cache[model_id] = config_counts(load_config(model_id))
        return cache[model_id]

    host = counts_of(program.host)
    total = host["total"]
    active = host["total"] - (0 if host["tied"] else host["embed"])
    if host["sparse"]:
        notes.append("the host is a sparse MoE, so params_active here counts every expert")

    if program.is_written():
        return estimate_written(program, host, notes)

    merged_lora = len(program.grafts) == 0 and program.experts_spec is None and len(program.inserts) == 0 and \
        len(program.taps) == 0
    if program.lora_spec is not None and not merged_lora and program.experts_spec is None:
        from wsp.insert import BASE_LORA_TARGETS

        count = lora_param_estimate(program.lora_spec, host, BASE_LORA_TARGETS if len(program.inserts) > 0 else None)
        if count is None:
            unknown.append("lora")
        else:
            total += count
            active += count
            notes.append(f"the LoRA stays unmerged and adds {count} parameters")
    elif program.lora_spec is not None and merged_lora:
        notes.append("the LoRA is merged into the weights at the end, so it adds no parameters")
    if program.masks is not None or len(program.task_vectors) > 0:
        notes.append("masks zero weights and task vectors edit them, so neither changes the size")

    for number, graft in enumerate(program.grafts):
        size = graft_size(graft, host, catalog, counts_of, unknown)
        if size is not None:
            total += size
            active += size

    if program.experts_spec is not None:
        added_total, added_active = experts_size(program, host, counts_of, unknown, notes)
        total += added_total
        active += added_active
    for spec in program.inserts:
        if spec.part_lora_rank is not None:
            notes.append("the LoRA inside the inserted donor layers is not counted")
        size = insert_size(spec, host, counts_of, unknown)
        if size is not None:
            total += size
            active += size
    if program.has_taps():
        notes.append("taps add their branches only; the donors they read are separate models and are not counted")
    if len(unknown) > 0:
        return {"params_total": None, "params_active": None, "partial_total": total, "partial_active": active,
                "notes": notes, "unknown": unknown}
    return {"params_total": total, "params_active": active, "notes": notes, "unknown": unknown}


def graft_size(graft, host, catalog, counts_of, unknown):
    """Parameters a graft adds (part plus glue), or None when the estimate cannot size it."""
    if catalog is None:
        unknown.append(f"graft {graft.component_id} (pass --catalog)")
        return None
    try:
        record = catalog.get(graft.component_id)
    except KeyError:
        unknown.append(f"graft {graft.component_id} (not in the catalog)")
        return None
    donor_id = record.get("model")
    if donor_id is None:
        unknown.append(f"graft {graft.component_id} (no donor model in its record)")
        return None
    try:
        donor = counts_of(donor_id)
    except Exception as error:
        unknown.append(f"graft {graft.component_id} (cannot read the config of {donor_id}: {error})")
        return None
    d_host = host["hidden"]
    d_donor = donor["hidden"]
    kind = record.get("type")
    if kind == "block" and record.get("layers") is not None:
        first, last = record["layers"]
        part = 0
        for index in range(first, last + 1):
            if index >= donor["n_layers"]:
                unknown.append(f"graft {graft.component_id} (layers {first}..{last} are past the donor's depth)")
                return None
            part += donor["layers"][index]["total"]
    elif kind == "mlp_cluster":
        units = read_units_count(catalog, graft.component_id)
        if units is None:
            unknown.append(f"graft {graft.component_id} (cannot read its units file)")
            return None
        part = 3 * d_donor * units + d_donor
    else:
        unknown.append(f"graft {graft.component_id} (size of a {kind} part is not estimated)")
        return None
    if graft.glue_rank is None:
        glue = 2 * d_host * d_donor
    else:
        glue = graft.glue_rank * (d_host + d_donor) * 2
    return part + glue


def read_units_count(catalog, component_id):
    import torch

    record = catalog.get(component_id)
    path = record.get("units")
    if path is None:
        return None
    full = os.path.join(catalog.root, *path.split("/")) if not os.path.isabs(path) else path
    try:
        data = torch.load(full, map_location="cpu", weights_only=False)
        return int(len(data["indices"]))
    except Exception:
        return None


def experts_size(program, host, counts_of, unknown, notes):
    """Return (added_total, added_active) of an experts statement."""
    spec = program.experts_spec
    n_experts = len(spec.donors) + 1
    layers = host["n_layers"]
    d_host = host["hidden"]
    added_total = 0
    donor_sizes = []
    for donor_id in spec.donors:
        try:
            donor = counts_of(donor_id)
        except Exception as error:
            unknown.append(f"expert donor {donor_id} (cannot read its config: {error})")
            return 0, 0
        stitched = spec.stitch is not None or donor["hidden"] != d_host or donor["n_layers"] != layers
        per_layer = []
        used = {}
        stitch_size = 0
        if stitched:
            stitch_size = 2 * d_host * donor["hidden"]
        for index in range(layers):
            donor_index = index if donor["n_layers"] == layers else round(index * (donor["n_layers"] - 1) /
                                                                         max(1, layers - 1))
            row = donor["layers"][donor_index]
            # The MLP, plus the norms around it (half of what is not attention or MLP).
            size = row["mlp"] + (row["total"] - row["mlp"] - row["attn"]) // 2
            # Host layers that map to one donor layer share its MLP, so it counts once in the total.
            used[donor_index] = size
            per_layer.append(size + stitch_size)
        donor_sizes.append(per_layer)
        added_total += sum(used.values()) + stitch_size * layers
    router = layers * (d_host * n_experts + n_experts)
    added_total += router
    # Active: in each layer the top_k largest of the host MLP and the donor experts run.
    added_active = router
    for index in range(layers):
        row = host["layers"][index]
        sizes = [row["mlp"] + (row["total"] - row["mlp"] - row["attn"]) // 2]
        for per_layer in donor_sizes:
            sizes.append(per_layer[index])
        largest = sorted(sizes, reverse=True)
        added_active += sum(largest[:spec.top_k]) - sizes[0]
    notes.append("expert sizes are the donor MLPs plus norms and, for a donor of another width, the stitches")
    from wsp.skill_moe import ATTN_TARGETS

    lora = lora_param_estimate(program.lora_spec, host, ATTN_TARGETS)
    if lora is None:
        unknown.append("lora")
    else:
        added_total += lora
        added_active += lora
    return added_total, added_active


def insert_size(spec, host, counts_of, unknown):
    """Parameters an insert adds: the donor layers, the stitches, and the gate."""
    try:
        donor = counts_of(spec.donor)
    except Exception as error:
        unknown.append(f"insert donor {spec.donor} (cannot read its config: {error})")
        return None
    first, last = spec.layers
    if last >= donor["n_layers"]:
        unknown.append(f"insert {spec.donor} layers {first}..{last} are past the donor's depth {donor['n_layers']}")
        return None
    part = 0
    for index in range(first, last + 1):
        part += donor["layers"][index]["total"]
    d_host = host["hidden"]
    d_donor = donor["hidden"]
    stitch_in = 0
    kind = spec.stitch
    if kind is None:
        kind = "identity" if d_host == d_donor else "linear"
    if kind == "linear":
        stitch_in = d_host * d_donor
    elif kind == "lowrank":
        stitch_in = spec.stitch_rank * (d_host + d_donor)
    elif kind == "mlp":
        hidden = spec.stitch_hidden or 2 * d_donor
        stitch_in = d_host * hidden + hidden * d_donor
    return part + stitch_in + d_donor * d_host + 1


def estimate_written(program, host, notes):
    """Estimate a written program (carve and splice) from the source config."""
    layers = host["layers"]
    n = host["n_layers"]
    average_mlp = sum(row["mlp"] for row in layers) / n
    average_attn = sum(row["attn"] for row in layers) / n
    average_other = sum(row["total"] - row["mlp"] - row["attn"] for row in layers) / n
    keep_mlp = 1.0
    keep_heads = 1.0
    spec = program.carve_spec
    keep = n
    if spec is not None:
        if spec.width is not None:
            keep_mlp = spec.width["keep_mlp"]
            keep_heads = spec.width["keep_heads"]
        if spec.keep is not None:
            keep = min(spec.keep, n)
    layer_size = average_mlp * keep_mlp + average_attn * keep_heads + average_other
    if spec is not None and spec.target_params_b is not None:
        budget = spec.target_params_b * 1e9 - host["other"]
        keep = max(2, min(n, int(budget // layer_size)))
        notes.append("the compiler picks the layer count that fits the target, from exact counts")
    if program.splice_spec is not None:
        keep = min(keep, len(program.splice_spec.plan)) if spec is not None else len(program.splice_spec.plan)
        notes.append("a splice takes layers of sources with the base's shapes, so the size follows the base")
    total = int(host["other"] + keep * layer_size)
    active = total - (0 if host["tied"] else host["embed"])
    notes.append(f"the written model keeps {keep} of {n} layers; width cuts are estimated from average layer sizes")
    return {"params_total": total, "params_active": active, "notes": notes, "unknown": [], "layers_kept": keep}


def budget_report(program, estimate):
    """Compare an estimate with the program's budget. Return a list of {"limit", "value", "ok"} rows."""
    spec = program.budget_spec
    rows = []
    if spec is None:
        return rows
    total = estimate["params_total"]
    active = estimate["params_active"]
    if spec.max_total_params_b is not None:
        row = {"limit": "max_total_params_b", "value": spec.max_total_params_b, "estimate": format_size(total)}
        row["ok"] = None if total is None else total <= spec.max_total_params_b * 1e9
        rows.append(row)
    if spec.max_active_params_b is not None:
        row = {"limit": "max_active_params_b", "value": spec.max_active_params_b, "estimate": format_size(active)}
        row["ok"] = None if active is None else active <= spec.max_active_params_b * 1e9
        rows.append(row)
    if spec.max_heal_minutes is not None:
        rows.append({"limit": "max_heal_minutes", "value": spec.max_heal_minutes, "estimate": "not estimated",
                     "ok": None})
    return rows


def run_check(path, catalog_path, allow_parallel, as_json, load_config=default_config_loader, out=None):
    """Check a program file. Return (exit code, result dict). Write the report to out (default stdout)."""
    from wsp.compiler import CompileError, check_program

    if out is None:
        out = sys.stdout
    program = read_program(path)
    problems = []
    warnings = []
    if program.has_taps() and not allow_parallel:
        problems.append(TAPS_REFUSAL)
    elif program.has_taps():
        warnings.append("this program has taps: it runs source models at inference, so it is not Sapien")
    try:
        check_program(program)
    except CompileError as error:
        problems.append(str(error))
    catalog = None
    if catalog_path is not None:
        from wsp.catalog import Catalog

        try:
            catalog = Catalog.load(catalog_path)
        except (OSError, ValueError) as error:
            problems.append(f"cannot read the catalog {catalog_path}: {error}")
    if catalog is not None:
        for graft in program.grafts:
            try:
                catalog.get(graft.component_id)
            except KeyError:
                problems.append(f"graft {graft.component_id} is not in the catalog {catalog_path}")
    elif program.needs_catalog() and (len(program.grafts) > 0 or program.masks is not None or
                                      len(program.task_vectors) > 0):
        warnings.append("this program reads catalog records; pass --catalog to resolve and size them")
    if program.heal_spec is not None and program.heal_spec.distill is not None:
        warnings.append("distill needs the teacher to share the host's tokenizer; the compile checks that")

    estimate = {"params_total": None, "params_active": None, "notes": [], "unknown": []}
    try:
        estimate = estimate_program(program, catalog, load_config)
    except Exception as error:
        warnings.append(f"cannot estimate the size: {error}")
    rows = budget_report(program, estimate)
    for row in rows:
        if row["ok"] is False:
            problems.append(f"estimated {row['limit']} is over the limit: estimate {row['estimate']}, "
                            f"limit {row['value']}")
    if len(estimate["unknown"]) > 0:
        warnings.append("the size estimate leaves out: " + "; ".join(estimate["unknown"]))
    result = {"program": path, "host": program.host, "ok": len(problems) == 0, "problems": problems,
              "warnings": warnings, "params_total": estimate["params_total"],
              "params_active": estimate["params_active"], "budget": rows, "notes": estimate["notes"]}
    if as_json:
        out.write(json.dumps(result, indent=2) + "\n")
    else:
        out.write(render_check(result, program))
    return (0 if result["ok"] else 1), result


def render_check(result, program):
    lines = [f"{result['program']}: host {result['host']}"]
    parts = []
    if len(program.grafts) > 0:
        parts.append(f"{len(program.grafts)} grafts")
    if program.experts_spec is not None:
        parts.append(f"experts from {len(program.experts_spec.donors)} donors")
    if len(program.inserts) > 0:
        parts.append(f"{len(program.inserts)} inserts")
    if program.is_written():
        parts.append("a written model")
    if len(parts) > 0:
        lines.append("  parts: " + ", ".join(parts))
    lines.append(f"  params_total  {format_size(result['params_total'])} (estimate)")
    lines.append(f"  params_active {format_size(result['params_active'])} (estimate)")
    for row in result["budget"]:
        state = {True: "ok", False: "OVER", None: "not checked"}[row["ok"]]
        lines.append(f"  budget {row['limit']} {row['value']}: {state}")
    for note in result["notes"]:
        lines.append(f"  note: {note}")
    for warning in result["warnings"]:
        lines.append(f"  warning: {warning}")
    for problem in result["problems"]:
        lines.append(f"  error: {problem}")
    lines.append("  " + ("ok" if result["ok"] else "failed"))
    return "\n".join(lines) + "\n"


# compile


def run_compile(args):
    from wsp.compiler import CompileError
    from wsp.lang import compile_file

    program = read_program(args.path)
    refuse_taps(program, args.allow_parallel)
    if args.control is not None:
        try:
            program = program.with_control(args.control, args.control_seed)
        except ValueError as error:
            raise CliError(str(error))
    if args.catalog is None and program.needs_catalog():
        raise CliError(f"{args.path}: this program needs --catalog")
    try:
        report = compile_file(program, args.out, args.catalog, args.align, args.device)
    except (CompileError, ValueError, KeyError) as error:
        raise CliError(str(error))
    summary = {"out": args.out, "params_total": report.get("params_total"),
               "params_active": report.get("params_active"), "heal_seconds": report.get("heal_seconds")}
    if report.get("budget") is not None:
        summary["budget_ok"] = report["budget"]["ok"]
    print(json.dumps(summary, indent=2))
    return 0


# score


def run_score(args):
    import torch

    from wsp import skills_eval
    from wsp.load import load

    skills = []
    for name in args.skills.split(","):
        name = name.strip()
        if name == "":
            continue
        if name not in skills_eval.SKILLS:
            raise CliError(f"unknown skill {name!r}; the skills are {', '.join(skills_eval.SKILLS)}")
        skills.append(name)
    if len(skills) == 0:
        raise CliError("--skills needs at least one skill name")
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(args.model_dir):
        raise CliError(f"{args.model_dir} is not a directory")
    model, tokenizer = load(args.model_dir, device=device)
    result = skills_eval.evaluate_suite(model, tokenizer, skills=skills, batch_size=args.batch_size, limit=args.limit)
    from wsp.compiler import count_model_params

    # The compiler's count, so the score row agrees with compile_report.json: an experts model
    # runs only its top_k experts per token.
    total, run = count_model_params(model)
    row = {"model": args.model_dir, "scores": result["scores"], "suite": result["suite"], "params_total": total,
           "params_run_per_token": run}
    print(json.dumps(row, indent=2))
    if args.json is not None:
        with open(args.json, "w") as f:
            json.dump({"summary": row, "full": skills_eval.strip_outputs(result, keep_output=False)}, f)
    return 0


# library and atlas


def load_library_module():
    path = os.path.join(REPO, "scripts", "library.py")
    if not os.path.exists(path):
        raise CliError(f"{path} is missing; sapien library needs the repo checkout")
    spec = importlib.util.spec_from_file_location("sapien_library_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_library(args):
    rest = list(args.rest)
    if len(rest) == 0:
        rest = ["--help"]
    return load_library_module().main(rest)


def run_atlas(args):
    from wsp.atlas import best_models_for, load_entries

    entries_dir = args.entries
    entries = load_entries(entries_dir)
    if len(entries) == 0:
        raise CliError(f"no atlas entries in {entries_dir}. modal_atlas.py writes them (see docs/atlas.md).")
    rows = best_models_for(args.skill, entries, args.max_params_b)
    if args.json:
        for row in rows[:args.top]:
            print(json.dumps(row))
        return 0
    if len(rows) == 0:
        raise CliError(f"no atlas entry has the skill {args.skill!r} within the size limit")
    print(f"{'model':48} {'score':>8} {'params_b':>9}")
    for row in rows[:args.top]:
        params = "" if row["params_b"] is None else f"{row['params_b']:.2f}"
        print(f"{row['model']:48} {row['score']:8.4f} {params:>9}")
    return 0


# Entry point


def build_parser():
    parser = argparse.ArgumentParser(prog="sapien", description="Write, size, compile, and score Sapien programs.")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="parse a program, validate it, and estimate its size from configs")
    check.add_argument("path")
    check.add_argument("--catalog", default=None, help="catalog.jsonl, to resolve graft ids and sizes")
    check.add_argument("--allow-parallel", action="store_true", help="accept a program with taps (not Sapien)")
    check.add_argument("--json", action="store_true", help="print the result as JSON")

    build = commands.add_parser("compile", help="compile a program into one model directory")
    build.add_argument("path")
    build.add_argument("-o", "--out", required=True, help="output directory")
    build.add_argument("--catalog", default=None, help="catalog.jsonl (grafts, masks, task vectors)")
    build.add_argument("--align", type=int, default=200,
                       help="heal texts for the W_in fit when the program has grafts (0 skips the fit)")
    build.add_argument("--device", default=None, help="cpu or cuda (default: cuda if present)")
    build.add_argument("--control", default=None, choices=["random"],
                       help="compile the program's own control: random same-size components of the same donors")
    build.add_argument("--control-seed", type=int, default=0, help="base seed of the random control")
    build.add_argument("--allow-parallel", action="store_true", help="accept a program with taps (not Sapien)")

    score = commands.add_parser("score", help="score a compiled directory on skills")
    score.add_argument("model_dir")
    score.add_argument("--skills", required=True, help="comma list: gsm8k, math500, humaneval, mbpp, medmcqa, mmlu")
    score.add_argument("--limit", type=int, default=None, help="score only the first N items of each skill")
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--device", default=None)
    score.add_argument("--json", default=None, help="also write the full result to this file")

    library = commands.add_parser("library", help="the shared component library (scripts/library.py)")
    library.add_argument("rest", nargs=argparse.REMAINDER, help="build, index, validate, add, or query, with options")

    atlas = commands.add_parser("atlas", help="the skill atlas")
    atlas_commands = atlas.add_subparsers(dest="atlas_command", required=True)
    best = atlas_commands.add_parser("best", help="rank atlas models for a skill")
    best.add_argument("--skill", required=True)
    best.add_argument("--max-params-b", type=float, default=None, help="largest model size in billions")
    best.add_argument("--entries", default=ATLAS_ENTRIES, help="atlas entries folder (default: atlas/entries)")
    best.add_argument("--top", type=int, default=10)
    best.add_argument("--json", action="store_true", help="print JSON lines")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            code, _ = run_check(args.path, args.catalog, args.allow_parallel, args.json)
            return code
        if args.command == "compile":
            return run_compile(args)
        if args.command == "score":
            return run_score(args)
        if args.command == "library":
            return run_library(args)
        if args.command == "atlas":
            return run_atlas(args)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
