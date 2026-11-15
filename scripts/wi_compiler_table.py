"""Build results/wi_compiler/table.md: the compiler's write-in rows next to the wi_xfam rows of the same seed.

    python scripts/wi_compiler_table.py

Reads results/wi_compiler/xfam_knowledge_report.json (sapien_report.json of the compile) and the seed 1 rows in
results/wi_xfam/rows. Scores are percent on held-out items nothing trained on.
"""

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = os.path.join(HERE, "results", "wi_xfam", "rows")
REPORT = os.path.join(HERE, "results", "wi_compiler", "xfam_knowledge_report.json")
OUT = os.path.join(HERE, "results", "wi_compiler", "table.md")
KEYS = ("mmlu_held", "arc_c", "obqa", "mmlu_dev")


def pct(scores, key):
    return "%.1f" % (100 * scores[key]) if scores and key in scores else ""


def fmt(value):
    return "" if value in (None, "") else str(value)


def line(arm, seed, source, scores, params, trainable, hours, tps):
    cells = [arm, str(seed)] + [pct(scores, k) for k in KEYS] + [fmt(params), fmt(trainable), fmt(hours), fmt(tps), source]
    return "| " + " | ".join(cells) + " |"


def xfam_row(name):
    path = os.path.join(ROWS, name)
    with open(path) as f:
        row = json.load(f)
    return row, "results/wi_xfam/rows/" + name


def main():
    with open(REPORT) as f:
        report = json.load(f)
    pg = report["pgraft"]
    graft = pg["graft"]
    out = ["| arm | seed | MMLU held-out 2000 | ARC-C test | OBQA test | MMLU dev 1000 | params/token | trainable | GPU h | tok/s | source |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    ref, src = xfam_row("reference_0_final.json")
    out.append(line("base (wi_xfam scoring)", 0, src, ref["scores"], ref["params_base"], 0, ref["gpu_hours"], ref.get("decode_tps")))
    ref_c = pg["reference"]
    out.append(line("base (compiler scoring)", 1, "results/wi_compiler/xfam_knowledge_report.json", ref_c["scores"],
                    ref_c["params"], 0, "", ref_c.get("decode_tps")))
    out.append(line("graft, compiler path", 1, "results/wi_compiler/xfam_knowledge_report.json", graft["scores"],
                    graft["params_per_token"], graft["trainable"], graft["gpu_hours"], graft.get("decode_tps")))
    row, src = xfam_row("graft_lora_1_final.json")
    out.append(line("graft, wi_xfam driver", 1, src, row["scores"], row["params_per_token"], row["trainable"], row["gpu_hours"],
                    row.get("decode_tps")))
    for name, label in (("identity", "identity control"), ("scrambled", "scrambled-donor control")):
        c = pg["controls"].get(name)
        if c is not None:
            out.append(line(label + ", compiler path", 1, "results/wi_compiler/xfam_knowledge_report.json", c["scores"],
                            c["params_per_token"], c["trainable"], c["gpu_hours"], ""))
        else:
            out.append("| %s, compiler path | 1 | not run | | | | | | | | the compile lost its GPU workspace during this control |" % label)
    for preset, label in (("identity_l", "identity control, wi_xfam driver"), ("scrambled_l", "scrambled-donor control, wi_xfam driver"),
                          ("lora_match_l", "LoRA matched to graft, wi_xfam driver"), ("lora_big", "LoRA rank 128, wi_xfam driver"),
                          ("fullft", "full fine-tune, wi_xfam driver")):
        row, src = xfam_row("%s_1_final.json" % preset)
        out.append(line(label, 1, src, row["scores"], row["params_per_token"], row["trainable"], row["gpu_hours"], row.get("decode_tps")))
    gap = 100 * (graft["scores"]["mmlu_held"] - xfam_row("graft_lora_1_final.json")[0]["scores"]["mmlu_held"])
    out.append("")
    out.append("Compiler graft against the wi_xfam graft of seed 1 on MMLU held-out 2000: %+.1f points (limit 1.5). One seed each." % gap)
    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
