"""Draw Chart 1a, Chart 1b and Chart 2 from a compare folder or from run logs.

Examples:
    python scripts/plot_curves.py outputs/compare/stub-e60
    python scripts/plot_curves.py outputs/runs/run-a outputs/runs/run-b/experiments.jsonl --out outputs/plots

Chart 1a: best combined score so far on the search split against cumulative
    GPU-seconds (the primary axis, papers.md section 8.5).
Chart 1b: the same against evals spent, in full-eval equivalents (FEE).
    Both show one thin line per seed, the median per condition and the
    min-to-max band across seeds (papers.md sections 5 and 7.5). Only
    full-fidelity search rows with status "ok" count toward best-so-far, so a
    low-fidelity score is never plotted as a best. The median and band start
    where every seed of a condition has a score, and a seed that stopped early
    keeps its final best.
Chart 2: fit_r2 (per graft, before heal) against the final combined score of
    its eval row, over every graft in every scored full-fidelity search eval,
    with the Spearman correlation in the title.

Every chart says "search split" and SEARCH; a STUB watermark goes across the
plot when any row is a stub. The script also prints the "N times fewer"
numbers (papers.md 7.5 and 8.5): the GPU-seconds and evals the swarm needs to
reach the evolve median final best, labeled SEARCH, and STUB on the stub.
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402

CONDITION_ORDER = ["swarm", "evolve", "swarm_llm", "random", "random_part", "greedy", "lora", "full_ft"]
# Fixed categorical slots (light mode), one per condition, so a color always means the same arm.
SLOT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER_COLOR = "#8a8984"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
AXES = {"compute": "compute_used_s", "evals": "eval_count"}
SWARM_CONDITIONS = ["swarm", "swarm_llm"]


# Reading logs.

def read_rows(path):
    """Read a JSONL log. A last line still being written (not valid JSON yet) is skipped."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def first_value(rows, key):
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def run_from_log(path, condition=None, seed=None):
    """One run: its rows plus condition (default: the rows' method) and seed (default: its manifest)."""
    rows = read_rows(path)
    if condition is None:
        condition = first_value(rows, "method")
    if condition is None:
        condition = "unknown"
    if seed is None:
        manifest_path = os.path.join(os.path.dirname(path), "manifest.json")
        if os.path.exists(manifest_path):
            seed = read_json(manifest_path).get("seed")
    return {"condition": condition, "seed": seed, "run_id": first_value(rows, "run_id"), "rows": rows, "path": path}


def compare_log_path(folder, record):
    """A compare record's log path, falling back to <outputs>/runs/<run_id> if the folder moved."""
    path = record.get("log_path")
    if path is not None and os.path.exists(path):
        return path
    outputs = os.path.dirname(os.path.dirname(os.path.abspath(folder)))
    return os.path.join(outputs, "runs", record["run_id"], "experiments.jsonl")


def is_compare_folder(path):
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isdir(path) or not os.path.exists(manifest_path):
        return False
    return "runs" in read_json(manifest_path) and not os.path.exists(os.path.join(path, "experiments.jsonl"))


def load_inputs(paths):
    """Return (runs, compare_folder) from compare folders, run folders or experiments.jsonl files."""
    runs = []
    compare_folder = None
    for path in paths:
        if is_compare_folder(path):
            if compare_folder is None:
                compare_folder = path
            manifest = read_json(os.path.join(path, "manifest.json"))
            for record in manifest["runs"]:
                log_path = compare_log_path(path, record)
                if not os.path.exists(log_path):
                    print("missing log for " + str(record.get("run_id")) + ": " + log_path)
                    continue
                runs.append(run_from_log(log_path, record.get("condition"), record.get("seed")))
        elif os.path.isdir(path):
            runs.append(run_from_log(os.path.join(path, "experiments.jsonl")))
        else:
            runs.append(run_from_log(path))
    return runs, compare_folder


def group_by_condition(runs):
    """{condition: [runs sorted by seed]}, in the fixed condition order, unknown names last."""
    groups = {}
    for run in runs:
        if run["condition"] not in groups:
            groups[run["condition"]] = []
        groups[run["condition"]].append(run)
    for condition in groups:
        groups[condition].sort(key=seed_sort_key)
    return groups


def seed_sort_key(run):
    if run["seed"] is None:
        return (1, 0, str(run["run_id"]))
    return (0, run["seed"], str(run["run_id"]))


def ordered_conditions(names):
    out = []
    for name in CONDITION_ORDER:
        if name in names:
            out.append(name)
    for name in sorted(names):
        if name not in out:
            out.append(name)
    return out


def color_for(condition):
    if condition in CONDITION_ORDER:
        return SLOT_COLORS[CONDITION_ORDER.index(condition)]
    return OTHER_COLOR


# Curves.

def is_curve_row(row):
    """A full-fidelity search eval with status ok and a combined score."""
    if row.get("type") != "eval" or row.get("split") != "search":
        return False
    fidelity = row.get("fidelity")
    if fidelity is not None and fidelity != "full":
        return False
    return row.get("status") == "ok" and row.get("combined") is not None


def curve_points(rows, axis):
    """(x, combined) for each curve row; x is cumulative GPU-seconds or evals spent."""
    key = AXES[axis]
    points = []
    for row in rows:
        if is_curve_row(row) and row.get(key) is not None:
            points.append((float(row[key]), float(row["combined"])))
    return points


def best_so_far(points):
    """The steps (x, best) where the best score so far goes up, in x order."""
    steps = []
    best = None
    for point in sorted(points):
        if best is None or point[1] > best:
            best = point[1]
            steps.append((point[0], best))
    return steps


def run_end(rows, axis):
    """The last search-budget value of the axis in a run (held-out rows use another budget)."""
    key = AXES[axis]
    end = None
    for row in rows:
        if row.get("split") == "heldout":
            continue
        value = row.get(key)
        if value is None:
            continue
        if end is None or float(value) > end:
            end = float(value)
    return end


def value_at(steps, x):
    """The best so far at x, or None before the first scored row."""
    value = None
    for step in steps:
        if step[0] <= x:
            value = step[1]
        else:
            break
    return value


def band(curves):
    """Median and min-to-max band over seeds.

    curves is a list of (steps, end), one per seed. Returns {"x", "median",
    "low", "high"}; the values are None where some seed has no score yet.
    """
    xs = set()
    for steps, end in curves:
        for step in steps:
            xs.add(step[0])
        if end is not None:
            xs.add(end)
    grid = sorted(xs)
    out = {"x": grid, "median": [], "low": [], "high": []}
    for x in grid:
        values = []
        for steps, end in curves:
            value = value_at(steps, x)
            if value is None:
                values = None
                break
            values.append(value)
        if values is None or len(values) == 0:
            out["median"].append(None)
            out["low"].append(None)
            out["high"].append(None)
        else:
            out["median"].append(float(numpy.median(values)))
            out["low"].append(min(values))
            out["high"].append(max(values))
    return out


def final_best(rows):
    """The best full-fidelity ok search score of a run, or None."""
    steps = best_so_far(curve_points(rows, "evals"))
    if len(steps) == 0:
        return None
    return steps[-1][1]


def first_reach(steps, target):
    """The first x where the best so far reaches target, or None if it never does."""
    for step in steps:
        if step[1] >= target - 1e-12:
            return step[0]
    return None


def lower_median(values):
    """The median for an odd count; the lower middle value for an even count."""
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def median_reach(values):
    """Median of per-seed x values, with never-reached (None) counted as infinite. None if the median is."""
    numbers = []
    for value in values:
        if value is None:
            numbers.append(float("inf"))
        else:
            numbers.append(float(value))
    if len(numbers) == 0:
        return None
    middle = float(numpy.median(numbers))
    if math.isinf(middle) or math.isnan(middle):
        return None
    return middle


def times_fewer(fast_runs, slow_runs):
    """The "N times fewer" numbers for a fast condition against a slow (baseline) one.

    target = the slow condition's median final best (the lower middle value
    for an even seed count, so at least half its seeds reach it). For each
    axis, each seed's first x reaching the target, the medians, and the ratio
    slow median / fast median. Returns None when the slow runs never scored.
    """
    finals = []
    for run in slow_runs:
        value = final_best(run["rows"])
        if value is None:
            value = float("-inf")
        finals.append(value)
    if len(finals) == 0:
        return None
    target = lower_median(finals)
    if math.isinf(target):
        return None
    out = {"target": target, "slow_finals": finals}
    for axis in ["compute", "evals"]:
        fast = []
        for run in fast_runs:
            fast.append(first_reach(best_so_far(curve_points(run["rows"], axis)), target))
        slow = []
        for run in slow_runs:
            slow.append(first_reach(best_so_far(curve_points(run["rows"], axis)), target))
        fast_median = median_reach(fast)
        slow_median = median_reach(slow)
        ratio = None
        if fast_median is not None and slow_median is not None and fast_median > 0:
            ratio = slow_median / fast_median
        out[axis] = {"fast": fast, "slow": slow, "fast_median": fast_median, "slow_median": slow_median,
                     "ratio": ratio}
    return out


# Chart 2 helpers.

def fit_points(rows):
    """(fit_r2, combined) for every graft in every scored full-fidelity search eval (ok or collapsed)."""
    points = []
    for row in rows:
        if row.get("type") != "eval" or row.get("split") != "search":
            continue
        fidelity = row.get("fidelity")
        if fidelity is not None and fidelity != "full":
            continue
        if row.get("status") not in ("ok", "collapsed") or row.get("combined") is None:
            continue
        fits = row.get("fit_r2") or {}
        for key in sorted(fits):
            value = fits[key]
            if value is None:
                continue
            points.append((float(value), float(row["combined"])))
    return points


def average_ranks(values):
    """Ranks starting at 1, with tied values sharing their average rank."""
    order = numpy.argsort(numpy.asarray(values, dtype=float), kind="mergesort")
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def spearman(xs, ys):
    """Spearman's rho: the Pearson correlation of the average ranks. None when undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rank_x = numpy.asarray(average_ranks(xs))
    rank_y = numpy.asarray(average_ranks(ys))
    dx = rank_x - rank_x.mean()
    dy = rank_y - rank_y.mean()
    denominator = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if denominator == 0:
        return None
    return float((dx * dy).sum()) / denominator


def needs_stub_watermark(rows):
    """True when any row came from the stub backend."""
    for row in rows:
        if row.get("stub") is True or row.get("backend") == "stub":
            return True
    return False


def all_rows(runs):
    rows = []
    for run in runs:
        for row in run["rows"]:
            rows.append(row)
    return rows


def labels(stub):
    if stub:
        return "[SEARCH][STUB]"
    return "[SEARCH]"


# Drawing.

def new_figure():
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(TEXT_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def set_titles(ax, title, subtitle, xlabel, ylabel):
    ax.set_title(title + "\n" + subtitle, loc="left", fontsize=11, color=TEXT_PRIMARY)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)


def add_watermark(fig):
    fig.text(0.5, 0.5, "STUB", fontsize=110, color=TEXT_SECONDARY, alpha=0.13, rotation=28,
             ha="center", va="center", fontweight="bold", zorder=10)


def nan_list(values):
    out = []
    for value in values:
        if value is None:
            out.append(float("nan"))
        else:
            out.append(value)
    return numpy.asarray(out, dtype=float)


def spread_labels(ys, gap):
    """Nudge label heights apart so no two sit closer than gap. Returns new heights in input order."""
    order = sorted(range(len(ys)), key=lambda index: ys[index])
    placed = list(ys)
    previous = None
    for index in order:
        if previous is not None and placed[index] < previous + gap:
            placed[index] = previous + gap
        previous = placed[index]
    return placed


def save(fig, path):
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_curves(groups, axis, title, xlabel, path, stub):
    """Chart 1a or 1b: per-seed lines, median and min-to-max band per condition."""
    fig, ax = new_figure()
    ends = []
    for condition in ordered_conditions(list(groups.keys())):
        color = color_for(condition)
        curves = []
        for run in groups[condition]:
            steps = best_so_far(curve_points(run["rows"], axis))
            if len(steps) == 0:
                continue
            end = run_end(run["rows"], axis)
            if end is None or end < steps[-1][0]:
                end = steps[-1][0]
            curves.append((steps, end))
            xs = []
            ys = []
            for step in steps:
                xs.append(step[0])
                ys.append(step[1])
            xs.append(end)
            ys.append(steps[-1][1])
            ax.step(xs, ys, where="post", color=color, linewidth=0.9, alpha=0.5)
        if len(curves) == 0:
            print("chart " + axis + ": " + condition + " has no scored full-fidelity search rows")
            continue
        summary = band(curves)
        grid = numpy.asarray(summary["x"], dtype=float)
        ax.fill_between(grid, nan_list(summary["low"]), nan_list(summary["high"]), step="post",
                        color=color, alpha=0.14, linewidth=0)
        ax.step(grid, nan_list(summary["median"]), where="post", color=color, linewidth=2.2,
                label=condition + " (median of " + str(len(curves)) + " seeds)")
        last = None
        for index in range(len(summary["x"])):
            if summary["median"][index] is not None:
                last = (summary["x"][index], summary["median"][index])
        if last is not None:
            ends.append((condition, last[0], last[1]))
    if len(ends) > 0:
        ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=TEXT_PRIMARY)
    if 0 < len(ends) <= 4:
        low, high = ax.get_ylim()
        heights = []
        for item in ends:
            heights.append(item[2])
        placed = spread_labels(heights, (high - low) * 0.05)
        for index in range(len(ends)):
            ax.annotate(ends[index][0], xy=(ends[index][1], placed[index]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY,
                        annotation_clip=False)
    set_titles(ax, title + " " + labels(stub),
               "search split, full-fidelity ok rows only; thin lines are seeds, thick lines the median, "
               "bands min to max",
               xlabel, "best combined score so far (search split)")
    if stub:
        add_watermark(fig)
    return save(fig, path)


def draw_fit_chart(runs, path, stub):
    """Chart 2: fit_r2 against final combined score, with Spearman's rho in the title."""
    points = fit_points(all_rows(runs))
    xs = []
    ys = []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
    rho = spearman(xs, ys)
    fig, ax = new_figure()
    if len(points) > 0:
        ax.scatter(xs, ys, s=26, color=SLOT_COLORS[0], alpha=0.75, edgecolors=SURFACE, linewidths=0.8)
    rho_text = "n/a"
    if rho is not None:
        rho_text = "%.3f" % rho
    set_titles(ax, "Chart 2: fit_r2 vs final combined score, Spearman rho = " + rho_text + " (n = " +
               str(len(points)) + " grafts) " + labels(stub),
               "search split; one point per graft in each scored full-fidelity search eval; "
               "rows the fit gate rejected have no score",
               "fit_r2 before heal (per graft)", "final combined score (search split)")
    if stub:
        add_watermark(fig)
    return save(fig, path), rho, len(points)


# Printing.

def format_number(value, digits):
    if value is None:
        return "never"
    return ("%." + str(digits) + "f") % value


def seed_list(runs, values, digits):
    parts = []
    for index in range(len(runs)):
        parts.append("s" + str(runs[index]["seed"]) + " " + format_number(values[index], digits))
    return "[" + ", ".join(parts) + "]"


def print_final_table(groups, stub):
    print("final best-so-far search score per seed " + labels(stub) + ":")
    print("  (the highest full-fidelity ok row, where each curve ends; compare.py prints the judge's pick, "
          "which can sit up to tie_margin below it)")
    for condition in ordered_conditions(list(groups.keys())):
        finals = []
        for run in groups[condition]:
            finals.append(final_best(run["rows"]))
        scored = []
        for value in finals:
            if value is not None:
                scored.append(value)
        line = "  " + condition.ljust(12) + " seeds " + str(len(finals))
        if len(scored) > 0:
            line += "  median " + "%.4f" % float(numpy.median(scored)) + "  min " + "%.4f" % min(scored)
            line += "  max " + "%.4f" % max(scored)
        line += "  " + seed_list(groups[condition], finals, 4)
        print(line)


def report_times_fewer(groups, baseline, stub):
    """Print and return the N-times-fewer numbers for each swarm condition against the baseline."""
    results = {}
    tag = labels(stub)
    if baseline not in groups:
        print("N times fewer " + tag + ": needs a '" + baseline + "' condition; skipped")
        return results
    for condition in SWARM_CONDITIONS:
        if condition not in groups:
            continue
        result = times_fewer(groups[condition], groups[baseline])
        if result is None:
            print("N times fewer " + tag + ": " + baseline + " has no scored rows; skipped")
            return results
        results[condition] = result
        print("N times fewer " + tag + ": target = " + baseline + " median final best " +
              "%.4f" % result["target"] + " (" + str(len(groups[baseline])) + " seeds)")
        names = {"compute": "GPU-seconds", "evals": "evals (FEE)"}
        digits = {"compute": 0, "evals": 2}
        for axis in ["compute", "evals"]:
            item = result[axis]
            line = "  " + condition + " " + names[axis] + " to target: median " + \
                format_number(item["fast_median"], digits[axis]) + " vs " + baseline + " " + \
                format_number(item["slow_median"], digits[axis])
            if item["ratio"] is None:
                line += " -> ratio not defined (a median seed never reached the target)"
            else:
                line += " -> " + "%.2f" % item["ratio"] + "x fewer"
            print(line + " " + tag)
            print("    per seed " + condition + " " + seed_list(groups[condition], item["fast"], digits[axis]) +
                  ", " + baseline + " " + seed_list(groups[baseline], item["slow"], digits[axis]))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Draw the search curves and the fit chart.")
    parser.add_argument("paths", nargs="+", help="a compare folder, run folders or experiments.jsonl files")
    parser.add_argument("--out", default=None, help="folder for the PNGs (default: the compare folder)")
    parser.add_argument("--baseline", default="evolve", help="the condition the N-times-fewer target comes from")
    parser.add_argument("--conditions", default=None,
                        help="comma list of conditions to draw, for example swarm,evolve (default: all)")
    return parser.parse_args(argv)


def keep_conditions(runs, text):
    """Only the runs whose condition is in the comma list (all runs when text is None)."""
    if text is None:
        return runs
    wanted = []
    for piece in text.split(","):
        if piece.strip() != "":
            wanted.append(piece.strip())
    kept = []
    for run in runs:
        if run["condition"] in wanted:
            kept.append(run)
    return kept


def main(argv=None):
    """Draw the three charts, print their paths and the numbers. Returns a dict for tests."""
    args = parse_args(argv)
    runs, compare_folder = load_inputs(args.paths)
    runs = keep_conditions(runs, args.conditions)
    if len(runs) == 0:
        raise SystemExit("plot_curves: no run logs found in " + ", ".join(args.paths))
    out_dir = args.out
    if out_dir is None:
        out_dir = compare_folder
    if out_dir is None:
        out_dir = os.path.join("outputs", "plots")
    os.makedirs(out_dir, exist_ok=True)
    stub = needs_stub_watermark(all_rows(runs))
    groups = group_by_condition(runs)

    paths = []
    paths.append(draw_curves(groups, "compute", "Chart 1a: best combined score vs GPU-seconds",
                             "cumulative GPU-seconds (search budget)",
                             os.path.join(out_dir, "chart1a_gpu_seconds.png"), stub))
    paths.append(draw_curves(groups, "evals", "Chart 1b: best combined score vs evals",
                             "evals spent, FEE (search budget)",
                             os.path.join(out_dir, "chart1b_evals.png"), stub))
    fit_path, rho, n_points = draw_fit_chart(runs, os.path.join(out_dir, "chart2_fit_vs_score.png"), stub)
    paths.append(fit_path)
    for path in paths:
        print("wrote " + path)
    rho_text = "n/a"
    if rho is not None:
        rho_text = "%.3f" % rho
    print("fit_r2 vs final score: Spearman rho " + rho_text + " over " + str(n_points) + " grafts " + labels(stub))
    print_final_table(groups, stub)
    fewer = report_times_fewer(groups, args.baseline, stub)
    return {"paths": paths, "stub": stub, "spearman": rho, "n_fit_points": n_points, "times_fewer": fewer,
            "conditions": ordered_conditions(list(groups.keys()))}


if __name__ == "__main__":
    main()
