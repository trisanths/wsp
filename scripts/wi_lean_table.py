"""Summarise results/wi_lean/rows into markdown tables: screens, lr pilots, finals, and the Pareto table.

Rows come from `modal volume get wsp-models wi_lean/rows results/wi_lean/rows` (profile trisanth). Rows of the
parent track (results/wi_xfam/rows) supply the Llama-1B base, identity, LoRA and full fine-tune arms, since those do
not depend on the donor span.

    python3 scripts/wi_lean_table.py [rows dir] [xfam rows dir] [bench json]
"""
import glob
import json
import os
import statistics as st
import sys

SETS = ["mmlu_held", "arc_c", "obqa", "mmlu_dev"]
root = sys.argv[1] if len(sys.argv) > 1 else "results/wi_lean/rows"
xfam = sys.argv[2] if len(sys.argv) > 2 else "results/wi_xfam/rows"
bench_path = sys.argv[3] if len(sys.argv) > 3 else "results/wi_lean/bench.json"


def load(path):
    return json.load(open(path))


def fmt(r, k):
    return f"{100 * r['scores'][k]:.1f}"


def mean_sd(rs, k):
    v = [100 * r["scores"][k] for r in rs]
    return f"{st.mean(v):.1f} +-{st.stdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f} (1 seed)"


rows = [load(p) | {"file": os.path.basename(p)} for p in sorted(glob.glob(f"{root}/*.json"))]


def host_of(r):
    c = r["config"]
    if c.get("host"):
        return c["host"]
    tag = c.get("tag", "")
    for key, name in (("llama3b", "unsloth/Llama-3.2-3B-Instruct"), ("qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
                      ("qwen3b", "Qwen/Qwen2.5-3B-Instruct")):
        if key in tag:
            return name
    return "unsloth/Llama-3.2-1B-Instruct"


def span_of(r):
    c = r["config"]
    keep = c.get("keep") or (list(range(4, 21, 2)) if "sp4_20" in c.get("tag", "") else None)
    if keep:
        return f"sparse {len(keep)} layers of {c['first']}..{c['last']}"
    return f"{c['first']}..{c['last']} ({c['last'] - c['first'] + 1} layers)"


print("## Donor span screens (seed 0, 4000 steps, graft + LoRA 16)\n")
print("| span | MMLU dev 1000 | MMLU held-out 2000 | ARC-C | OBQA | slice params | params/token | GPU h | file |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["name"] == "graft_lora" and r["seed"] == 0:
        print(f"| {span_of(r)} | {fmt(r, 'mmlu_dev')} | {fmt(r, 'mmlu_held')} | {fmt(r, 'arc_c')} | {fmt(r, 'obqa')} "
              f"| {r.get('slice_params')} | {r.get('params_per_token')} | {r['gpu_hours']} | {r['file']} |")
screen_ref = {}
for p in ("graft_lora_0_s4k.json",):
    if os.path.exists(f"{xfam}/{p}"):
        r = load(f"{xfam}/{p}")
        print(f"| 4..20 (17 layers), from wi_xfam | {fmt(r, 'mmlu_dev')} | {fmt(r, 'mmlu_held')} | {fmt(r, 'arc_c')} | {fmt(r, 'obqa')} "
              f"| {r.get('slice_params')} | {r.get('params_per_token')} | {r['gpu_hours']} | wi_xfam/{p} |")

print("\n## Native full fine-tune lr pilots (seed 0, 1200 steps, MMLU dev 1000)\n")
print("| host | ft_lr | MMLU dev | MMLU held-out | file |")
print("|---|---|---|---|---|")
for r in rows:
    if r["name"] == "fullft" and r["config"]["steps"] == 1200:
        print(f"| {host_of(r)} | {r['config']['ft_lr']} | {fmt(r, 'mmlu_dev')} | {fmt(r, 'mmlu_held')} | {r['file']} |")

print("\n## Per-seed rows (everything else)\n")
print("| arm | host | span or lr | seed | MMLU held-out 2000 | ARC-C | OBQA | MMLU dev | params/token | trainable | GPU h | file |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if (r["name"] == "graft_lora" and r["seed"] == 0) or (r["name"] == "fullft" and r["config"]["steps"] == 1200):
        continue
    what = span_of(r) if r["name"] in ("graft_lora", "scrambled_l") else (
        f"lr {r['config']['ft_lr']}" if r["name"] == "fullft" else "")
    print(f"| {r['name']} | {host_of(r)} | {what} | {r['seed']} | {fmt(r, 'mmlu_held')} | {fmt(r, 'arc_c')} | {fmt(r, 'obqa')} "
          f"| {fmt(r, 'mmlu_dev')} | {r.get('params_per_token', r.get('params_base'))} | {r.get('trainable', '')} "
          f"| {r['gpu_hours']} | {r['file']} |")

if os.path.exists(bench_path):
    print("\n## Decode speed (one H200, 128 prompt tokens, 128 new tokens, bf16, greedy, median of 3)\n")
    print("| model | params/token | bs1 tok/s | bs8 tok/s per seq | bs8 tok/s total |")
    print("|---|---|---|---|---|")
    for b in load(bench_path):
        p = b.get("params_per_token", b.get("params"))
        b8 = b.get("bs8", {})
        print(f"| {b['label']} | {p} | {b['bs1']['tok_s_per_seq']} | {b8.get('tok_s_per_seq', '')} | {b8.get('tok_s_total', '')} |")
