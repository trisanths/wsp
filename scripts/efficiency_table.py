"""Markdown tables from results/efficiency_<name>.json rows.

    python scripts/efficiency_table.py results/efficiency_full.json results/efficiency_slim64.json

Prints one table per file: the start row (config "base" on H200) first, then every other non-smoke row sorted by
HumanEval. Columns are the axes measured. A "-" means the config did not measure it or failed (see the error list
below the table).
"""

import json
import sys


def pct(block):
    return "-" if not block else f"{100 * block['score']:.1f}"


def num(value, digits=0):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def accept(row):
    block = (row.get("accept") or {}).get("he_b1") or (row.get("accept") or {}).get("humaneval")
    if not block:
        return "-"
    return f"{100 * block['rate']:.0f}% / {block['mean_accept_len']:.2f}"


def table(rows):
    rows = [r for r in rows if not r.get("smoke")]
    good = [r for r in rows if not r.get("error")]
    start = [r for r in good if r["config"] == "base" and (r.get("gpu_label") or r["gpu"]) == "H200"]
    rest = [r for r in good if r not in start]
    rest.sort(key=lambda r: -((r.get("humaneval") or {}).get("score") or 0))
    head = ("| config | GPU | HumanEval | MBPP | tok/s b1 | tok/s b8 | tok/s b32 | he_b1 tok/s | prefill tok/s | TTFT ms | "
            "disk GiB | weights GiB | KV tokens | accept (rate / len) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in start + rest:
        d = r.get("decode_tps") or {}
        lines.append("| " + " | ".join([
            r["config"], r.get("gpu_label") or r["gpu"], pct(r.get("humaneval")), pct(r.get("mbpp")),
            num(d.get("1")), num(d.get("8")), num(d.get("32")), num(r.get("he_b1_tps")), num(r.get("prefill_tps")),
            num(r.get("ttft_ms_median"), 1), num(r.get("weights_disk_gib"), 1), num(r.get("model_loading_gib"), 1),
            num(r.get("kv_cache_tokens")), accept(r)]) + " |")
    errors = [r for r in rows if r.get("error")]
    out = "\n".join(lines)
    if errors:
        out += "\n\nFailed configs:\n"
        for r in errors:
            out += f"- {r['config']} on {r.get('gpu_label') or r['gpu']}: {r['error'][:260]}\n"
    return out


def main(paths):
    for path in paths:
        with open(path) as f:
            rows = json.load(f)
        model = next((r["model"] for r in rows if r.get("model")), path)
        print(f"### {model} ({path})\n")
        print(table(rows))
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
