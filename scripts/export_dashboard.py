"""Export the part 3 runs as compact JSON for the static dashboard in dashboard/.

Examples:
    python scripts/export_dashboard.py
    python scripts/export_dashboard.py --out dashboard/public/data --max-bytes 3000000

It reads:
- outputs/compare/<name>/manifest.json and the run logs it lists (one dataset per compare folder);
- outputs/runs/<run_id>/experiments.jsonl for runs no compare folder lists (one dataset each;
  exact repeats of the same run are dropped and named in index.json);
- outputs/smoke/modal_smoke_*.json (the real Modal smokes);
- data/pilot_catalog*.jsonl (the pilot parts);
- outputs/final/<run_id>/final.json and model_card.md, when they exist.

It writes index.json, one dataset-<id>.json per dataset, smoke.json, parts.json and final.json.

Curves, bands, fit points, Spearman and the N-times-fewer numbers come from
scripts/plot_curves.py, and log helpers from scripts/agent_log.py, so the
dashboard shows the same numbers as the PNG charts and the HTML agent log.

Every score keeps the labels of the row it came from (STUB, SEARCH, HELD-OUT,
INTERIM, SMOKE), in a "labels" list next to it.

The site is public, so two guards run before anything is written:
- scrub() rewrites every string: the repo path becomes relative, other local
  paths become "<local path>", emails become "<email>" and the local user
  names become "<user>";
- check_public() scans every output file for .env values, token shapes, local
  paths and emails, and refuses to write when it finds one. It never prints a
  secret, only the name of the check that failed.

The total output stays under --max-bytes (default 3,000,000). When it would
not, trim() applies steps in a fixed order and lists each step it took in
index.json "trimmed".
"""

import argparse
import datetime
import getpass
import glob
import importlib.util
import json
import math
import os
import re
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_OUT = os.path.join(REPO_ROOT, "dashboard", "public", "data")
MAX_BYTES = 3000000

HOST_LAYERS_DEFAULT = 16
HYPOTHESIS_CHARS = 240
EVENT_CHARS = 300
HEAL_POINTS = 12
SAMPLE_CHARS = 280
SKIPPED_EVENT_KINDS = ["reset"]


def load_script(name):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    path = os.path.join(SCRIPTS_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location("wsp_dashboard_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load_script("plot_curves")
agent_log = load_script("agent_log")


class ExportError(Exception):
    """The export would leak something or would not fit the size cap."""


# Small helpers.

def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    """Rows of a JSONL file, skipping blank and half-written lines."""
    return plot.read_rows(path)


def round_number(value, digits=4):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        rounded = round(float(value), digits)
        if rounded == int(rounded) and abs(rounded) < 1e15:
            return int(rounded)
        return rounded
    return value


def round_dict(values, digits=4):
    out = {}
    if not isinstance(values, dict):
        return out
    for key in values:
        out[key] = round_number(values[key], digits)
    return out


def truncate(text, limit):
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


def downsample(values, limit):
    """At most limit evenly spaced values, always keeping the first and last."""
    if not isinstance(values, list):
        return []
    numbers = []
    for value in values:
        numbers.append(round_number(value, 4))
    if len(numbers) <= limit or limit < 2:
        return numbers
    out = []
    step = (len(numbers) - 1) / float(limit - 1)
    for index in range(limit):
        out.append(numbers[int(round(index * step))])
    return out


def relative_path(path, root):
    """A path relative to the repo root with forward slashes ("<local path>" when outside it)."""
    if path is None:
        return None
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, root)
    except ValueError:
        return "<local path>"
    if relative.startswith(".."):
        return "<local path>"
    return relative.replace("\\", "/")


def slug(text):
    out = ""
    for char in str(text):
        if char.isalnum() or char in "-_.":
            out += char
        else:
            out += "-"
    return out.strip("-") or "dataset"


# Labels.

def is_interim(row):
    """True when the row says its scores came from the interim scorer."""
    candidates = [row.get("label"), row.get("eval_backend"), row.get("labels")]
    extra = row.get("extra") or {}
    if isinstance(extra, dict):
        candidates.append(extra.get("label"))
        candidates.append(extra.get("eval_backend"))
        candidates.append(extra.get("labels"))
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if "interim" in str(item).lower():
                    return True
        elif "interim" in str(value).lower():
            return True
    return False


def row_labels(row):
    """The labels a score from this row carries, for example ["SEARCH", "STUB"]."""
    labels = []
    if row.get("split") == "heldout":
        labels.append("HELD-OUT")
    elif row.get("split") == "search":
        labels.append("SEARCH")
    if agent_log.is_stub_row(row):
        labels.append("STUB")
    if is_interim(row):
        labels.append("INTERIM")
    if row.get("method") == "smoke":
        labels.append("SMOKE")
    return labels


def union_labels(label_lists):
    order = ["SMOKE", "STUB", "INTERIM", "SEARCH", "HELD-OUT"]
    seen = set()
    for labels in label_lists:
        for label in labels:
            seen.add(label)
    out = []
    for label in order:
        if label in seen:
            out.append(label)
    for label in sorted(seen):
        if label not in out:
            out.append(label)
    return out


# Privacy.

class Scrubber:
    """Rewrites strings so no local path, email or user name reaches the public site."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.home = os.path.expanduser("~")
        self.user_names = local_user_names(root)
        self.root_forms = path_forms(self.root)
        self.home_forms = path_forms(self.home)

    def text(self, value):
        out = value
        for form in self.root_forms:
            out = replace_insensitive(out, form + "\\", "")
            out = replace_insensitive(out, form + "/", "")
            out = replace_insensitive(out, form, ".")
        for form in self.home_forms:
            out = replace_insensitive(out, form, "<local path>")
        out = DRIVE_PATH.sub("<local path>", out)
        out = UNIX_HOME_PATH.sub("<local path>", out)
        out = EMAIL.sub("<email>", out)
        for name in self.user_names:
            out = re.sub(r"(?i)(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", "<user>", out)
        return out

    def value(self, value):
        """A deep copy of value with every string scrubbed and every NaN or infinity set to None."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, dict):
            out = {}
            for key in value:
                out[self.text(str(key))] = self.value(value[key])
            return out
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.append(self.value(item))
            return out
        return self.text(str(value))


DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\"'<>|\r\n]*")
UNIX_HOME_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[^\s\"'<>|]+")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
TOKEN_SHAPES = [
    ("modal token", re.compile(r"(?<![A-Za-z0-9])a[ks]-[A-Za-z0-9]{16,}")),
    ("api key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")),
    ("github token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("slack token", re.compile(r"xox[abpr]-[A-Za-z0-9-]{10,}")),
    ("bearer header", re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]


def path_forms(path):
    """The spellings of a local path that can appear in logs (backslashes, forward slashes)."""
    forms = []
    for form in [path, path.replace("\\", "/"), path.replace("/", "\\")]:
        if form not in forms and len(form) > 3:
            forms.append(form)
    forms.sort(key=len, reverse=True)
    return forms


def replace_insensitive(text, old, new):
    if old == "":
        return text
    return re.sub(re.escape(old), lambda match: new, text, flags=re.IGNORECASE)


def git_config(root, key):
    try:
        result = subprocess.run(["git", "-C", root, "config", "--get", key], capture_output=True,
                                text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if value == "":
        return None
    return value


def local_user_names(root):
    """The OS user name and git user name (3 or more characters), which the site never shows."""
    names = []
    candidates = []
    try:
        candidates.append(getpass.getuser())
    except (OSError, KeyError, ImportError):
        pass
    candidates.append(os.path.basename(os.path.expanduser("~")))
    candidates.append(git_config(root, "user.name"))
    for name in candidates:
        if name is None:
            continue
        name = name.strip()
        if len(name) >= 3 and name.lower() not in lowered(names):
            names.append(name)
    return names


def lowered(names):
    out = []
    for name in names:
        out.append(name.lower())
    return out


def env_values(root):
    """(key, value) pairs from .env with values of 6 or more characters. Never printed."""
    path = os.path.join(root, ".env")
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line == "" or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip().strip('"').strip("'")
            if len(value) >= 6:
                pairs.append((key, value))
    return pairs


def check_public(files, root, scrubber):
    """Raise ExportError if any output text holds a secret, a local path, an email or a user name."""
    secrets = env_values(root)
    git_email = git_config(root, "user.email")
    problems = []
    for name in sorted(files):
        text = files[name]
        lower = text.lower()
        for key, value in secrets:
            if value in text:
                problems.append(name + ": holds the value of .env key " + key)
        for label, pattern in TOKEN_SHAPES:
            if pattern.search(text):
                problems.append(name + ": holds something shaped like a " + label)
        for form in scrubber.root_forms + scrubber.home_forms:
            if form.lower() in lower or json.dumps(form)[1:-1].lower() in lower:
                problems.append(name + ": holds a local path")
                break
        if re.search(r"(?<![A-Za-z])[A-Za-z]:\\\\", text) or re.search(r"(?<![A-Za-z])[A-Za-z]:/[A-Za-z]", text):
            problems.append(name + ": holds a drive path")
        if EMAIL.search(text):
            problems.append(name + ": holds an email address")
        if git_email is not None and git_email.lower() in lower:
            problems.append(name + ": holds the git email")
        for user in scrubber.user_names:
            if re.search(r"(?i)(?<![A-Za-z0-9])" + re.escape(user) + r"(?![A-Za-z0-9])", text):
                problems.append(name + ": holds a local user name")
    if len(problems) > 0:
        raise ExportError("refusing to write the dashboard data:\n  " + "\n  ".join(problems))


# Datasets.

def compare_folders(root):
    out = []
    for path in sorted(glob.glob(os.path.join(root, "outputs", "compare", "*"))):
        if plot.is_compare_folder(path):
            out.append(path)
    return out


def compare_runs(folder):
    """The runs of a compare folder, as plot_curves.load_inputs reads them, plus their manifest records."""
    manifest = read_json(os.path.join(folder, "manifest.json"))
    runs = []
    for record in manifest.get("runs", []):
        log_path = plot.compare_log_path(folder, record)
        if not os.path.exists(log_path):
            continue
        run = plot.run_from_log(log_path, record.get("condition"), record.get("seed"))
        run["record"] = record
        runs.append(run)
    return manifest, runs


def eval_signature(rows):
    """What a run did, ignoring its run id and timestamps: (program_hash, combined, eval_count) per eval."""
    out = []
    for row in rows:
        if row.get("type") == "eval":
            out.append((row.get("program_hash"), row.get("combined"), row.get("eval_count"), row.get("split")))
    return tuple(out)


def loose_runs(root, listed_run_ids):
    """Runs in outputs/runs that no compare folder lists. Returns (runs, notes about what was skipped)."""
    runs = []
    notes = []
    no_log = []
    for folder in sorted(glob.glob(os.path.join(root, "outputs", "runs", "*"))):
        if not os.path.isdir(folder):
            continue
        name = os.path.basename(folder)
        if name in listed_run_ids:
            continue
        log_path = os.path.join(folder, "experiments.jsonl")
        if not os.path.exists(log_path):
            no_log.append(name)
            continue
        run = plot.run_from_log(log_path)
        if len(agent_log.eval_rows(run["rows"])) == 0:
            no_log.append(name)
            continue
        run["record"] = None
        run["folder_name"] = name
        run["started"] = first_timestamp(run["rows"])
        runs.append(run)
    if len(no_log) > 0:
        notes.append("skipped run folders with no eval rows (memory or setup smokes): " + ", ".join(no_log))
    groups = {}
    order = []
    for run in runs:
        key = eval_signature(run["rows"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(run)
    kept = []
    for key in order:
        group = sorted(groups[key], key=lambda item: str(item["started"]))
        keeper = group[-1]
        kept.append(keeper)
        if len(group) > 1:
            names = []
            for run in group[:-1]:
                names.append(run["folder_name"])
            notes.append("dropped exact repeats of run " + keeper["folder_name"] + ": " + ", ".join(names))
    kept.sort(key=lambda item: str(item["started"]))
    return kept, notes


def first_timestamp(rows):
    for row in rows:
        if row.get("timestamp"):
            return row["timestamp"]
    return ""


def first_number(rows, key, skip_heldout=True):
    for row in rows:
        if skip_heldout and row.get("split") == "heldout":
            continue
        value = row.get(key)
        if value is not None:
            return value
    return None


def phase_totals(rows):
    """GPU-seconds per phase over the search budget: eval rows by phase, probes as "probe"."""
    totals = {}
    for row in rows:
        if row.get("split") == "heldout":
            continue
        if row.get("type") not in ("eval", "probe"):
            continue
        phases = row.get("compute_s") or {}
        for phase in phases:
            value = phases[phase]
            if value is None:
                continue
            name = phase
            if row.get("type") == "probe":
                name = "probe"
            totals[name] = totals.get(name, 0.0) + float(value)
    return round_dict(totals, 1)


def heldout_usage(rows):
    eval_used = None
    compute_used = None
    eval_budget = None
    compute_budget = None
    for row in rows:
        if row.get("split") != "heldout":
            continue
        if row.get("eval_count") is not None:
            eval_used = max(eval_used or 0.0, float(row["eval_count"]))
        if row.get("compute_used_s") is not None:
            compute_used = max(compute_used or 0.0, float(row["compute_used_s"]))
        if eval_budget is None:
            eval_budget = row.get("eval_budget")
        if compute_budget is None:
            compute_budget = row.get("compute_budget_s")
    return {"eval_used": round_number(eval_used, 2), "eval_budget": round_number(eval_budget, 2),
            "compute_used_s": round_number(compute_used, 1), "compute_budget_s": round_number(compute_budget, 1)}


def run_budget(run):
    """Budget use of one run, read from its rows (and its compare record or summary when present)."""
    rows = run["rows"]
    eval_used = plot.run_end(rows, "evals")
    compute_used = plot.run_end(rows, "compute")
    eval_budget = first_number(rows, "eval_budget")
    compute_budget = first_number(rows, "compute_budget_s")
    overshoot = None
    record = run.get("record")
    if isinstance(record, dict):
        overshoot = record.get("compute_overshoot_s")
    if overshoot is None:
        summary_path = os.path.join(os.path.dirname(run["path"]), "summary.json")
        if os.path.exists(summary_path):
            try:
                overshoot = read_json(summary_path).get("compute_overshoot_s")
            except ValueError:
                overshoot = None
    if overshoot is None and compute_used is not None and compute_budget:
        overshoot = max(0.0, compute_used - float(compute_budget))
    statuses = {}
    probes = 0
    for row in rows:
        if row.get("type") == "probe":
            probes += 1
        if row.get("type") == "eval" and row.get("split") != "heldout":
            status = str(row.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
    return {
        "eval_used": round_number(eval_used, 2),
        "eval_budget": round_number(eval_budget, 2),
        "compute_used_s": round_number(compute_used, 1),
        "compute_budget_s": round_number(compute_budget, 1),
        "overshoot_s": round_number(overshoot, 1),
        "phases_s": phase_totals(rows),
        "probes": probes,
        "statuses": statuses,
        "heldout": heldout_usage(rows),
    }


def run_curves(rows):
    """Best-so-far steps and the run's end on both axes, the way plot_curves.draw_curves draws them."""
    out = {}
    for axis in ["evals", "compute"]:
        steps = plot.best_so_far(plot.curve_points(rows, axis))
        if len(steps) == 0:
            out[axis] = None
            continue
        end = plot.run_end(rows, axis)
        if end is None or end < steps[-1][0]:
            end = steps[-1][0]
        points = []
        for step in steps:
            points.append([round_number(step[0], 2), round_number(step[1], 4)])
        out[axis] = {"steps": points, "end": round_number(end, 2)}
    return out


def band_json(curves):
    """plot_curves.band over the seeds' curves, rounded for JSON."""
    pairs = []
    for curve in curves:
        if curve is None:
            continue
        steps = []
        for point in curve["steps"]:
            steps.append((point[0], point[1]))
        pairs.append((steps, curve["end"]))
    if len(pairs) == 0:
        return None
    summary = plot.band(pairs)
    out = {"x": [], "median": [], "low": [], "high": []}
    for index in range(len(summary["x"])):
        out["x"].append(round_number(summary["x"][index], 2))
        out["median"].append(round_number(summary["median"][index], 4))
        out["low"].append(round_number(summary["low"][index], 4))
        out["high"].append(round_number(summary["high"][index], 4))
    return out


def heldout_rows(rows):
    out = []
    for row in rows:
        if row.get("type") != "eval" or row.get("split") != "heldout":
            continue
        extra = row.get("extra") or {}
        out.append({
            "describe": row.get("describe"),
            "program_hash": row.get("program_hash"),
            "combined": round_number(row.get("combined"), 4),
            "scores": round_dict(row.get("scores")),
            "status": row.get("status"),
            "control": extra.get("control"),
            "control_of": extra.get("control_of"),
            "labels": row_labels(row),
        })
    return out


PART_FIELDS = ["kind", "donor", "donor_layer", "part_params", "n_units", "d_donor", "d_host"]
PLACEMENT_FIELDS = ["host_layer", "fit", "fit_r2", "glue_params"]


def graft_meta(row):
    """One entry per graft of the row's program, from its compile report entry.

    Keys: id; kind, donor, donor_layer, part_params, n_units, d_donor, d_host
    (fixed per part); host_layer, fit, fit_r2, glue_params (per placement). The
    program itself holds at, glue_rank, control and control_seed.
    """
    program = row.get("program") or {}
    grafts = program.get("grafts") or []
    report = row.get("graft_report") or []
    fits = row.get("fit_r2") or {}
    out = []
    for index in range(len(grafts)):
        graft = grafts[index]
        entry = {}
        if index < len(report) and isinstance(report[index], dict):
            entry = report[index]
        host_layer = entry.get("host_layer")
        if host_layer is None:
            host_layer = graft.get("at")
        fit = entry.get("fit_r2")
        if fit is None and host_layer is not None:
            key = str(graft.get("id")) + "@" + str(host_layer)
            if graft.get("control"):
                key += "~random"
            fit = fits.get(key)
        meta = {"id": graft.get("id"), "host_layer": host_layer, "fit": entry.get("fit"),
                "fit_r2": round_number(fit, 4), "glue_params": entry.get("glue_params")}
        for field in PART_FIELDS:
            meta[field] = entry.get(field)
        out.append(meta)
    return out


def note_parts(parts_info, grafts):
    """Record the fixed properties of each part (kind, donor, donor layer, sizes) once per dataset."""
    for graft in grafts:
        part_id = graft.get("id")
        if part_id is None or part_id in parts_info:
            continue
        if graft.get("kind") is None and graft.get("donor") is None:
            continue
        info = {}
        for field in PART_FIELDS:
            info[field] = graft.get(field)
        parts_info[part_id] = info


def placements(grafts):
    out = []
    for graft in grafts:
        entry = {}
        for field in PLACEMENT_FIELDS:
            entry[field] = graft.get(field)
        out.append(entry)
    return out


def program_base_of(runs):
    """The most common value of each top-level program key other than grafts and heal.

    Program entries store only the keys that differ from this base; the dashboard
    merges the base back in, so the program JSON it shows is the logged one.
    """
    counts = {}
    for run in runs:
        for row in run["rows"]:
            if row.get("type") != "eval":
                continue
            program = row.get("program") or {}
            for key in program:
                if key in ("grafts", "heal"):
                    continue
                pair = (key, json.dumps(program[key], sort_keys=True))
                counts[pair] = counts.get(pair, 0) + 1
    best = {}
    for pair in counts:
        key = pair[0]
        if key not in best or counts[pair] > counts[best[key]]:
            best[key] = pair
    base = {}
    for key in best:
        base[key] = json.loads(best[key][1])
    return base


def program_diff(program, base):
    """The program without the keys whose value equals the dataset's program base."""
    out = {}
    for key in program:
        same = key in base and json.dumps(program[key], sort_keys=True) == json.dumps(base[key], sort_keys=True)
        if same and key not in ("grafts", "heal"):
            continue
        out[key] = program[key]
    return out


def program_entry(row, grafts, base):
    extra = row.get("extra") or {}
    program = row.get("program") or {}
    entry = {"program": program_diff(program, base), "describe": row.get("describe"),
             "grafts": placements(grafts)}
    if len(program) == 0:
        entry["empty_program"] = True
    training = extra.get("training")
    if isinstance(training, dict):
        entry["training"] = training
    return entry


def fit_list(row, grafts):
    out = []
    for graft in grafts:
        out.append(graft.get("fit_r2"))
    return out


def log_row(row, index, run_index, grafts, heal_index, condition):
    """One trimmed eval row. The run index gives its arm and seed; "heal" indexes the dataset's heal curves."""
    extra = row.get("extra") or {}
    out = {
        "i": index,
        "run": run_index,
        "t": row.get("timestamp"),
        "eval_count": round_number(row.get("eval_count"), 2),
        "agent": row.get("agent"),
        "round": row.get("round"),
        "hypothesis": truncate(row.get("hypothesis") or "", HYPOTHESIS_CHARS),
        "program_hash": row.get("program_hash"),
        "fit_r2": fit_list(row, grafts),
        "combined": round_number(row.get("combined"), 4),
        "scores": round_dict(row.get("scores")),
        "status": row.get("status"),
        "fidelity": row.get("fidelity"),
        "split": row.get("split"),
        "stub": agent_log.is_stub_row(row),
        "compute_s": round_dict(row.get("compute_s"), 1),
        "compute_used_s": round_number(row.get("compute_used_s"), 1),
        "labels": row_labels(row),
        "heal": heal_index,
    }
    if row.get("method") is not None and row.get("method") != condition:
        out["method"] = row.get("method")
    if extra.get("control"):
        out["control"] = extra.get("control")
        out["control_of"] = extra.get("control_of")
    if extra.get("parent"):
        out["parent"] = extra.get("parent")
    if extra.get("params") is not None:
        out["params"] = extra.get("params")
    if extra.get("step") is not None:
        out["step"] = extra.get("step")
    if row.get("cache_hit"):
        out["cache_hit"] = True
    citations = row.get("citations") or []
    if len(citations) > 0:
        kept = []
        for citation in citations[:3]:
            if isinstance(citation, dict):
                kept.append({"title": truncate(citation.get("title"), 140), "url": truncate(citation.get("url"), 200)})
            else:
                kept.append({"title": truncate(citation, 140), "url": None})
        out["citations"] = kept
    return out


def event_row(row, run_index):
    return {
        "run": run_index,
        "t": row.get("timestamp"),
        "eval_count": round_number(row.get("eval_count"), 2),
        "agent": row.get("agent"),
        "round": row.get("round"),
        "kind": row.get("kind"),
        "text": truncate(row.get("text") or "", EVENT_CHARS),
        "labels": row_labels(row),
    }


def build_dataset(dataset_id, title, kind, source, runs, manifest, root):
    """One dataset: arms with per-seed curves, bands, budgets, the log, events and fit points."""
    groups = plot.group_by_condition(runs)
    conditions = plot.ordered_conditions(list(groups.keys()))
    run_list = []
    arms = []
    run_index_of = {}
    for condition in conditions:
        arm = {"condition": condition, "slot": slot_of(condition), "runs": []}
        evals_curves = []
        compute_curves = []
        for run in groups[condition]:
            curves = run_curves(run["rows"])
            record = run.get("record") or {}
            entry = {
                "run_id": run.get("run_id"),
                "condition": condition,
                "seed": run.get("seed"),
                "source": relative_path(run["path"], root),
                "curves": curves,
                "final_best": round_number(plot.final_best(run["rows"]), 4),
                "budget": run_budget(run),
                "heldout": heldout_rows(run["rows"]),
                "labels": union_labels(row_label_lists(run["rows"])),
            }
            if isinstance(record.get("best_search"), dict):
                best = dict(record["best_search"])
                best["combined"] = round_number(best.get("combined"), 4)
                best["scores"] = round_dict(best.get("scores"))
                entry["judge_pick"] = best
            run_index_of[id(run)] = len(run_list)
            arm["runs"].append(len(run_list))
            run_list.append(entry)
            evals_curves.append(curves["evals"])
            compute_curves.append(curves["compute"])
        arm["band"] = {"evals": band_json(evals_curves), "compute": band_json(compute_curves)}
        arms.append(arm)

    log = []
    events = []
    programs = {}
    parts_info = {}
    heal_curves = []
    heal_index_of = {}
    program_base = program_base_of(runs)
    fit_points = []
    all_rows = []
    arm_index_of = {}
    for index in range(len(arms)):
        arm_index_of[arms[index]["condition"]] = index
    ordered = []
    for run in runs:
        for row in run["rows"]:
            ordered.append((run, row))
    ordered.sort(key=lambda pair: agent_log.timestamp_of(pair[1]))
    for run, row in ordered:
        all_rows.append(row)
        run_index = run_index_of[id(run)]
        if row.get("type") == "event":
            if row.get("kind") in SKIPPED_EVENT_KINDS:
                continue
            events.append(event_row(row, run_index))
            continue
        if row.get("type") != "eval":
            continue
        grafts = graft_meta(row)
        note_parts(parts_info, grafts)
        program_hash = row.get("program_hash")
        if program_hash is not None and program_hash not in programs:
            programs[program_hash] = program_entry(row, grafts, program_base)
        heal_index = None
        curve = downsample(row.get("heal_loss") or [], HEAL_POINTS)
        if len(curve) > 0:
            key = json.dumps(curve)
            if key not in heal_index_of:
                heal_index_of[key] = len(heal_curves)
                heal_curves.append(curve)
            heal_index = heal_index_of[key]
        index = len(log)
        log.append(log_row(row, index, run_index, grafts, heal_index, run["condition"]))
        arm_index = arm_index_of.get(run["condition"], -1)
        for point in plot.fit_points([row]):
            fit_points.append([round_number(point[0], 4), round_number(point[1], 4), arm_index, index])

    xs = []
    ys = []
    for point in fit_points:
        xs.append(point[0])
        ys.append(point[1])
    for arm_index in range(len(arms)):
        arm_x = []
        arm_y = []
        for point in fit_points:
            if point[2] == arm_index:
                arm_x.append(point[0])
                arm_y.append(point[1])
        arms[arm_index]["fit"] = {"n": len(arm_x), "spearman": round_number(plot.spearman(arm_x, arm_y), 3)}

    fewer = {}
    if "evolve" in groups:
        for condition in plot.SWARM_CONDITIONS:
            if condition in groups:
                result = plot.times_fewer(groups[condition], groups["evolve"])
                if result is not None:
                    fewer[condition] = times_fewer_json(result)

    stub = plot.needs_stub_watermark(all_rows)
    dataset = {
        "id": dataset_id,
        "title": title,
        "kind": kind,
        "source": source,
        "stub": stub,
        "labels": union_labels(row_label_lists(all_rows)),
        "host": host_of(all_rows),
        "host_layers": host_layers_of(runs),
        "conditions": conditions,
        "seeds": seeds_of(runs),
        "cascade": cascade_of(manifest, all_rows),
        "budgets": budgets_of(manifest, run_list),
        "budget_check": budget_check_of(manifest, run_list),
        "created": (manifest or {}).get("created") or first_timestamp(all_rows),
        "runs": run_list,
        "arms": arms,
        "times_fewer": fewer,
        "fit": {"points": fit_points, "n": len(fit_points), "spearman": round_number(plot.spearman(xs, ys), 3),
                "labels": search_labels(stub)},
        "program_base": program_base,
        "programs": programs,
        "parts": parts_info,
        "heal_curves": heal_curves,
        "row_defaults": compact_rows(log),
        "log": log,
        "events": events,
        "counts": {"eval_rows": len(log), "events": len(events), "probes": count_type(all_rows, "probe")},
    }
    return dataset


DEFAULTED_FIELDS = ["split", "fidelity", "stub", "labels", "status"]


def compact_rows(log):
    """Drop fields that equal the dataset's most common value; returns those defaults.

    The dashboard reads a row as {...row_defaults, ...row}, so nothing is lost.
    """
    defaults = {}
    for field in DEFAULTED_FIELDS:
        counts = {}
        for row in log:
            if field in row:
                text = json.dumps(row[field], sort_keys=True)
                counts[text] = counts.get(text, 0) + 1
        if len(counts) == 0:
            continue
        best = None
        for text in counts:
            if best is None or counts[text] > counts[best]:
                best = text
        defaults[field] = json.loads(best)
    for row in log:
        for field in defaults:
            if field in row and json.dumps(row[field], sort_keys=True) == json.dumps(defaults[field], sort_keys=True):
                del row[field]
    return defaults


def row_value(dataset, row, key):
    """A log row's field, falling back to the dataset's row defaults."""
    if key in row:
        return row[key]
    return dataset.get("row_defaults", {}).get(key)


def row_label_lists(rows):
    out = []
    for row in rows:
        if row.get("type") == "eval":
            out.append(row_labels(row))
    return out


def count_type(rows, kind):
    total = 0
    for row in rows:
        if row.get("type") == kind:
            total += 1
    return total


def search_labels(stub):
    if stub:
        return ["SEARCH", "STUB"]
    return ["SEARCH"]


def slot_of(condition):
    if condition in plot.CONDITION_ORDER:
        return plot.CONDITION_ORDER.index(condition)
    return -1


def times_fewer_json(result):
    out = {"target": round_number(result["target"], 4), "baseline": "evolve"}
    for axis in ["compute", "evals"]:
        item = result[axis]
        fast = []
        for value in item["fast"]:
            fast.append(round_number(value, 2))
        slow = []
        for value in item["slow"]:
            slow.append(round_number(value, 2))
        out[axis] = {"fast": fast, "slow": slow, "fast_median": round_number(item["fast_median"], 2),
                     "slow_median": round_number(item["slow_median"], 2), "ratio": round_number(item["ratio"], 2)}
    return out


def host_of(rows):
    for row in rows:
        program = row.get("program") or {}
        host = program.get("host") or program.get("base")
        if host:
            return host
    return None


def host_layers_of(runs):
    for run in runs:
        manifest_path = os.path.join(os.path.dirname(run["path"]), "manifest.json")
        if os.path.exists(manifest_path):
            try:
                config = read_json(manifest_path).get("config") or {}
            except ValueError:
                config = {}
            if config.get("host_layers"):
                return config["host_layers"]
    return HOST_LAYERS_DEFAULT


def seeds_of(runs):
    seeds = set()
    for run in runs:
        if run.get("seed") is not None:
            seeds.add(run["seed"])
    return sorted(seeds)


def cascade_of(manifest, rows):
    if isinstance(manifest, dict) and "cascade" in manifest:
        return bool(manifest["cascade"])
    for row in rows:
        if row.get("type") == "eval" and row.get("fidelity") not in (None, "full"):
            return True
    return False


def budgets_of(manifest, run_list):
    if isinstance(manifest, dict) and isinstance(manifest.get("budgets"), dict):
        return round_dict(manifest["budgets"], 1)
    if len(run_list) == 0:
        return {}
    budget = run_list[0]["budget"]
    return {"eval_budget": budget["eval_budget"], "compute_budget_s": budget["compute_budget_s"],
            "heldout_eval_budget": budget["heldout"]["eval_budget"],
            "heldout_compute_budget_s": budget["heldout"]["compute_budget_s"]}


def budget_check_of(manifest, run_list):
    """Whether every run had the same caps on both axes (from the rows), plus compare.py's own check."""
    search_caps = set()
    heldout_caps = set()
    for run in run_list:
        budget = run["budget"]
        search_caps.add((budget["eval_budget"], budget["compute_budget_s"]))
        heldout = (budget["heldout"]["eval_budget"], budget["heldout"]["compute_budget_s"])
        if heldout != (None, None):
            heldout_caps.add(heldout)
    out = {"equal_caps": len(search_caps) <= 1 and len(heldout_caps) <= 1, "n_runs": len(run_list)}
    if isinstance(manifest, dict) and isinstance(manifest.get("budget_check"), dict):
        out["compare_check"] = manifest["budget_check"]
    return out


# Smoke, parts and final.

def smoke_model(role, title, block, run_labels):
    result = block.get("result") if isinstance(block, dict) else None
    if not isinstance(result, dict):
        return None
    report = result.get("compile_report") or {}
    grafts = report.get("grafts") or []
    graft = None
    if len(grafts) > 0 and isinstance(grafts[0], dict):
        first = grafts[0]
        graft = {}
        for key in ["id", "kind", "donor", "donor_layer", "host_layer", "control", "fit", "fit_r2", "fit_pairs",
                    "n_units", "part_params", "glue_params", "d_donor", "d_host"]:
            graft[key] = round_number(first.get(key), 4)
    heal = report.get("heal") or {}
    labels = list(run_labels)
    if result.get("label") and str(result["label"]).upper() not in labels:
        labels.append(str(result["label"]).upper())
    samples = result.get("samples") or []
    sample = None
    if len(samples) > 0:
        sample = truncate(str(samples[0]).strip(), SAMPLE_CHARS)
    return {
        "role": role,
        "title": title,
        "scores": round_dict(result.get("scores")),
        "labels": union_labels([labels]),
        "n_items": result.get("n_items") or result.get("eval_limit"),
        "eval_backend": result.get("eval_backend"),
        "gpu": result.get("gpu"),
        "ok": result.get("ok"),
        "cache_hit": result.get("cache_hit"),
        "seconds": round_dict(result.get("seconds"), 2),
        "cached_seconds": round_dict(result.get("cached_seconds"), 2),
        "program_hash": result.get("program_hash"),
        "graft": graft,
        "heal": heal.get("spec"),
        "heal_seconds": round_number(heal.get("seconds"), 2),
        "heal_loss": downsample(result.get("heal_loss") or heal.get("losses") or [], 25),
        "sample": sample,
        "error": truncate(result.get("error"), 300) if result.get("error") else None,
    }


def smoke_entry(path, root):
    data = read_json(path)
    labels = []
    for label in data.get("labels") or []:
        labels.append(str(label).upper())
    for label in ["SMOKE", "INTERIM"]:
        if label not in labels:
            labels.append(label)
    models = []
    for role, title in [("host", "Host only"), ("graft", "Host + graft"), ("control_random", "Random-part control")]:
        model = smoke_model(role, title, data.get(role), labels)
        if model is not None:
            models.append(model)
    probe = None
    probe_result = (data.get("probe") or {}).get("result")
    if isinstance(probe_result, dict):
        fits = []
        for graft in probe_result.get("graft_report") or []:
            fits.append(round_number(graft.get("fit_r2"), 4))
        probe = {"fit_r2": fits, "seconds": round_dict(probe_result.get("seconds"), 2),
                 "gpu": probe_result.get("gpu")}
    ledger = data.get("ledger") or {}
    checks = {}
    for key in sorted((data.get("checks") or {}).keys()):
        checks[key] = bool(data["checks"][key])
    run_id = data.get("run_id")
    started = None
    digits = re.search(r"(\d{10})", str(run_id or ""))
    if digits:
        started = datetime.datetime.fromtimestamp(int(digits.group(1)), datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    gpu = data.get("gpu_spec") or data.get("gpu")
    return {
        "run_id": run_id,
        "source": relative_path(path, root),
        "started": started,
        "labels": union_labels([labels]),
        "part": data.get("part"),
        "heal_steps": data.get("heal_steps"),
        "eval_limit": data.get("eval_limit"),
        "gpu": gpu,
        "cost_estimate_usd": round_number(data.get("cost_estimate_usd"), 3),
        "gpu_call_wall_s": round_number(data.get("gpu_call_wall_s"), 1),
        "checks": checks,
        "ledger": {"compute_used_s": round_number(ledger.get("compute_used_s"), 2), "spends": ledger.get("spends"),
                   "tokens_left": ledger.get("tokens_left"), "devices": ledger.get("devices")},
        "probe": probe,
        "models": models,
    }


def build_smoke(root):
    entries = []
    for path in sorted(glob.glob(os.path.join(root, "outputs", "smoke", "modal_smoke_*.json"))):
        try:
            entries.append(smoke_entry(path, root))
        except (ValueError, OSError, AttributeError, TypeError):
            continue
    entries.sort(key=lambda item: str(item.get("started") or item.get("run_id")), reverse=True)
    return {"runs": entries,
            "note": "SMOKE INTERIM numbers: a plumbing check on 20 GSM8K search items, scored with the interim "
                    "scorer. They are not a result."}


def part_entry(record):
    signature = record.get("signature") or {}
    n_units = signature.get("n_units")
    if n_units is None:
        match = re.search(r"top(\d+)", str(record.get("id")))
        if match:
            n_units = int(match.group(1))
    return {
        "id": record.get("id"),
        "model": record.get("model"),
        "layer": record.get("layer"),
        "type": record.get("type"),
        "n_units": n_units,
        "function": signature.get("function"),
        "method": signature.get("method"),
        "mean_effect": round_number(signature.get("mean_effect"), 4),
        "random_mean_effect": round_number(signature.get("random_mean_effect"), 4),
        "layer_rank": signature.get("layer_rank"),
        "trained_on": truncate(record.get("trained_on"), 320),
        "n_controls": len(record.get("controls") or []),
        "source": truncate(record.get("source"), 160),
        "pilot": bool(record.get("pilot")),
    }


def build_parts(root):
    catalogs = []
    for path in sorted(glob.glob(os.path.join(root, "data", "pilot_catalog*.jsonl"))):
        parts = []
        for record in read_jsonl(path):
            if isinstance(record, dict) and record.get("id"):
                parts.append(part_entry(record))
        name = os.path.basename(path)
        title = "Pilot parts"
        if "mined" in name:
            title = "Mined pilot parts"
        catalogs.append({"file": relative_path(path, root), "title": title, "parts": parts})
    return {"catalogs": catalogs}


def shrink_lists(value, limit):
    """A copy of value where every list longer than limit is cut down to limit evenly spaced items."""
    if isinstance(value, dict):
        out = {}
        for key in value:
            out[key] = shrink_lists(value[key], limit)
        return out
    if isinstance(value, list):
        items = value
        if len(items) > limit:
            kept = []
            step = (len(items) - 1) / float(limit - 1)
            for index in range(limit):
                kept.append(items[int(round(index * step))])
            items = kept
        out = []
        for item in items:
            out.append(shrink_lists(item, limit))
        return out
    return value


def build_final(root):
    runs = []
    for folder in sorted(glob.glob(os.path.join(root, "outputs", "final", "*"))):
        final_path = os.path.join(folder, "final.json")
        if not os.path.isdir(folder) or not os.path.exists(final_path):
            continue
        try:
            final = read_json(final_path)
        except ValueError:
            continue
        entry = {"run_id": os.path.basename(folder), "source": relative_path(final_path, root),
                 "final": shrink_lists(final, 40), "card_md": None}
        card_path = os.path.join(folder, "model_card.md")
        if os.path.exists(card_path):
            with open(card_path, encoding="utf-8", errors="replace") as handle:
                entry["card_md"] = truncate(handle.read(), 40000)
        runs.append(entry)
    return {"available": len(runs) > 0, "runs": runs}


# Trimming.

def dumps(data):
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def total_size(files):
    total = 0
    for name in files:
        total += len(files[name].encode("utf-8"))
    return total


def trim_heal(datasets, points):
    """Cut every heal loss curve to points values; 0 drops the curves."""
    for dataset in datasets:
        if points == 0:
            dataset["heal_curves"] = []
            for row in dataset["log"]:
                row["heal"] = None
            continue
        curves = []
        for curve in dataset["heal_curves"]:
            curves.append(downsample(curve, points))
        dataset["heal_curves"] = curves


def trim_events(datasets, kinds, chars):
    for dataset in datasets:
        kept = []
        for event in dataset["events"]:
            if event.get("kind") in kinds:
                continue
            event["text"] = truncate(event.get("text"), chars)
            kept.append(event)
        dataset["events"] = kept


def keep_row_always(dataset, row):
    """Rows a trim never drops: held-out rows, controls and anything that did not end "ok"."""
    return row_value(dataset, row, "split") == "heldout" or row_value(dataset, row, "status") != "ok"


def trim_low_fidelity(datasets, keep_every):
    for dataset in datasets:
        kept = []
        seen = {}
        for row in dataset["log"]:
            if row_value(dataset, row, "fidelity") in (None, "full") or keep_row_always(dataset, row):
                kept.append(row)
                continue
            count = seen.get(row["run"], 0)
            seen[row["run"]] = count + 1
            if count % keep_every == 0:
                kept.append(row)
        dataset["log"] = kept


def trim_per_run(datasets, cap):
    """At most cap log rows per run: best-so-far steps and always-kept rows first, then an even sample."""
    for dataset in datasets:
        by_run = {}
        for row in dataset["log"]:
            if row["run"] not in by_run:
                by_run[row["run"]] = []
            by_run[row["run"]].append(row)
        kept_ids = set()
        for run_index in by_run:
            rows = by_run[run_index]
            if len(rows) <= cap:
                for row in rows:
                    kept_ids.add(row["i"])
                continue
            must = []
            best = None
            for row in rows:
                improves = (row_value(dataset, row, "split") == "search" and
                            row_value(dataset, row, "fidelity") in (None, "full") and
                            row_value(dataset, row, "status") == "ok" and row.get("combined") is not None and
                            (best is None or row["combined"] > best))
                if improves:
                    best = row["combined"]
                if improves or keep_row_always(dataset, row):
                    must.append(row["i"])
            for row_id in must:
                kept_ids.add(row_id)
            room = cap - len(must)
            others = []
            for row in rows:
                if row["i"] not in kept_ids:
                    others.append(row["i"])
            if room > 0 and len(others) > 0:
                step = len(others) / float(room)
                position = 0.0
                while int(position) < len(others) and room > 0:
                    kept_ids.add(others[int(position)])
                    position += step
                    room -= 1
        kept = []
        for row in dataset["log"]:
            if row["i"] in kept_ids:
                kept.append(row)
        dataset["log"] = kept


def prune_programs(datasets):
    for dataset in datasets:
        used = set()
        for row in dataset["log"]:
            used.add(row.get("program_hash"))
        kept = {}
        for program_hash in dataset["programs"]:
            if program_hash in used:
                kept[program_hash] = dataset["programs"][program_hash]
        dataset["programs"] = kept


TRIM_STEPS = [
    ("heal loss curves in the log cut to 6 points per row (from up to 12)", lambda sets: trim_heal(sets, 6)),
    ("evolve 'population' events dropped and event text cut to 160 characters",
     lambda sets: trim_events(sets, ["population"], 160)),
    ("low-fidelity log rows sampled 1 in 3 per run (every collapse, control and held-out row kept)",
     lambda sets: trim_low_fidelity(sets, 3)),
    ("log capped at 60 rows per run (best-so-far steps, collapses, controls and held-out rows kept)",
     lambda sets: trim_per_run(sets, 60)),
    ("heal loss curves dropped from the log", lambda sets: trim_heal(sets, 0)),
    ("log capped at 25 rows per run", lambda sets: trim_per_run(sets, 25)),
]


def finish_counts(datasets):
    for dataset in datasets:
        dataset["counts"]["eval_rows_kept"] = len(dataset["log"])
        dataset["counts"]["events_kept"] = len(dataset["events"])


# Assembly.

def git_snapshot(root):
    out = {"git_commit": None, "git_dirty": None}
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            out["git_commit"] = head.stdout.strip()
        status = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True,
                                timeout=20)
        if status.returncode == 0:
            out["git_dirty"] = status.stdout.strip() != ""
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def collect_datasets(root):
    """Every dataset as (id, title, kind, source, runs, manifest), compare folders first. Plus notes."""
    items = []
    listed = set()
    for folder in compare_folders(root):
        manifest, runs = compare_runs(folder)
        for record in manifest.get("runs", []):
            if record.get("run_id"):
                listed.add(record["run_id"])
        if len(runs) == 0:
            continue
        name = manifest.get("name") or os.path.basename(folder)
        items.append((slug(name), name, "compare", relative_path(folder, root), runs, manifest))
    loose, notes = loose_runs(root, listed)
    for run in loose:
        name = run["folder_name"]
        items.append(("run-" + slug(name), name, "run", relative_path(os.path.dirname(run["path"]), root),
                      [run], None))
    return items, notes


def build_export(root=REPO_ROOT, max_bytes=MAX_BYTES, now=None):
    """Build every output file as {name: JSON text}. Raises ExportError on a leak or when over max_bytes."""
    root = os.path.abspath(root)
    scrubber = Scrubber(root)
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    items, notes = collect_datasets(root)
    datasets = []
    for dataset_id, title, kind, source, runs, manifest in items:
        datasets.append(scrubber.value(build_dataset(dataset_id, title, kind, source, runs, manifest, root)))
    smoke = scrubber.value(build_smoke(root))
    parts = scrubber.value(build_parts(root))
    final = scrubber.value(build_final(root))

    trimmed = []
    snapshot = git_snapshot(root)
    snapshot["generated_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    step_index = 0
    while True:
        finish_counts(datasets)
        prune_programs(datasets)
        index = build_index(datasets, smoke, parts, final, snapshot, notes, trimmed)
        files = render_files(datasets, smoke, parts, final, index)
        size = total_size(files)
        if size <= max_bytes:
            break
        if step_index >= len(TRIM_STEPS):
            raise ExportError("the dashboard data is " + str(size) + " bytes after every trim step, over the cap of "
                              + str(max_bytes) + " bytes")
        label, step = TRIM_STEPS[step_index]
        step(datasets)
        trimmed.append(label)
        step_index += 1
    # Write the final sizes into the index (a few bytes more than measured; the cap has slack for it).
    index["total_bytes"] = size
    files["index.json"] = dumps(index)
    check_public(files, root, scrubber)
    return files


def build_index(datasets, smoke, parts, final, snapshot, notes, trimmed):
    entries = []
    all_stub = len(datasets) > 0
    for dataset in datasets:
        if not dataset["stub"]:
            all_stub = False
        entries.append({
            "id": dataset["id"],
            "title": dataset["title"],
            "kind": dataset["kind"],
            "file": "dataset-" + dataset["id"] + ".json",
            "source": dataset["source"],
            "stub": dataset["stub"],
            "labels": dataset["labels"],
            "conditions": dataset["conditions"],
            "seeds": dataset["seeds"],
            "n_runs": len(dataset["runs"]),
            "cascade": dataset["cascade"],
            "created": dataset["created"],
            "counts": dataset["counts"],
        })
    default = None
    for entry in entries:
        if entry["kind"] == "compare" and not entry["stub"] and default is None:
            default = entry["id"]
    for entry in entries:
        if entry["kind"] == "compare" and default is None:
            default = entry["id"]
    if default is None and len(entries) > 0:
        default = entries[0]["id"]
    return {
        "schema": 1,
        "snapshot": snapshot,
        "all_search_stub": all_stub,
        "default_dataset": default,
        "datasets": entries,
        "files": {"smoke": "smoke.json", "parts": "parts.json", "final": "final.json"},
        "smoke_runs": len(smoke["runs"]),
        "parts_catalogs": len(parts["catalogs"]),
        "final_available": final["available"],
        "notes": notes,
        "trimmed": list(trimmed),
        "sizes": {},
        "total_bytes": None,
    }


def render_files(datasets, smoke, parts, final, index):
    files = {}
    for dataset in datasets:
        files["dataset-" + dataset["id"] + ".json"] = dumps(dataset)
    files["smoke.json"] = dumps(smoke)
    files["parts.json"] = dumps(parts)
    files["final.json"] = dumps(final)
    sizes = {}
    for name in sorted(files):
        sizes[name] = len(files[name].encode("utf-8"))
    index["sizes"] = sizes
    files["index.json"] = dumps(index)
    return files


def write_export(files, out_dir):
    """Write the files and remove stale dataset-*.json files from earlier exports."""
    os.makedirs(out_dir, exist_ok=True)
    for path in glob.glob(os.path.join(out_dir, "dataset-*.json")):
        if os.path.basename(path) not in files:
            os.remove(path)
    for name in sorted(files):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(files[name])
    return out_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export run data as JSON for the static dashboard.")
    parser.add_argument("--root", default=REPO_ROOT, help="the repo root to read outputs/ and data/ from")
    parser.add_argument("--out", default=DEFAULT_OUT, help="the folder for the JSON files")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES, help="the cap on the total output size")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        files = build_export(args.root, args.max_bytes)
    except ExportError as error:
        print("export_dashboard: " + str(error))
        raise SystemExit(1)
    write_export(files, args.out)
    index = json.loads(files["index.json"])
    print("wrote " + str(len(files)) + " files, " + str(index["total_bytes"]) + " bytes, to " +
          relative_path(args.out, os.path.abspath(args.root)))
    for entry in index["datasets"]:
        counts = entry["counts"]
        print("  " + entry["file"] + ": " + str(entry["n_runs"]) + " runs, " + str(counts["eval_rows_kept"]) +
              " of " + str(counts["eval_rows"]) + " eval rows kept, labels " + " ".join(entry["labels"]))
    for note in index["notes"]:
        print("  note: " + note)
    for label in index["trimmed"]:
        print("  trimmed: " + label)
    if index["all_search_stub"]:
        print("  every search dataset is STUB")
    return files


if __name__ == "__main__":
    main()
