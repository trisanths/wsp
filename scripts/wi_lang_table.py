"""Summarise results/wi_lang/rows/*_final.json into markdown tables (per-language scores, controls, costs).

    python scripts/wi_lang_table.py [rows_dir] > results/wi_lang/table.md
"""
import glob
import json
import statistics as st
import sys

BELE_TARGET = ["arb_Arab", "jpn_Jpan", "zho_Hans", "kor_Hang", "hin_Deva"]
BELE_TRANSFER = ["spa_Latn", "fra_Latn", "ben_Beng", "swh_Latn"]
MMMLU = ["ar", "ja", "zh", "ko", "hi"]
ORDER = ["reference", "donor", "graft_lora", "identity_l", "scrambled_l", "lora_match_l", "lora_big", "fullft"]
NATIVE_ORDER = ["qwen1.5b", "qwen3b", "llama3b"]


def pct(x):
    return 100 * x


def derived(scores):
    s = scores
    out = {}
    out["bele_eng"] = pct(s["bele_eng_Latn"])
    for k in BELE_TARGET:
        out["bele_" + k[:3]] = pct(s["bele_" + k])
    out["bele_target5"] = st.mean(pct(s["bele_" + k]) for k in BELE_TARGET)
    out["bele_transfer4"] = st.mean(pct(s["bele_" + k]) for k in BELE_TRANSFER)
    for k in BELE_TRANSFER:
        out["bele_" + k[:3]] = pct(s["bele_" + k])
    for k in MMMLU:
        out["mmmlu_" + k] = pct(s["mmmlu_" + k])
    out["mmmlu_mean5"] = st.mean(pct(s["mmmlu_" + k]) for k in MMMLU)
    out["mmlu_en"] = pct(s["mmlu_held"])
    if "dev_mmmlu" in s:
        out["dev_mmmlu"] = pct(s["dev_mmmlu"])
    return out


def load(root):
    rows = {}
    for path in sorted(glob.glob(f"{root}/*_final.json")):
        r = json.load(open(path))
        key = r["name"] if r["name"] != "native" else "native_" + r["config"]["tag"].replace("_final", "").strip("_")
        rows.setdefault(key, []).append(r)
    return rows


def ms(values):
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{st.mean(values):.1f} +-{st.stdev(values):.1f}"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/wi_lang/rows"
    rows = load(root)
    names = [n for n in ORDER if n in rows] + ["native_" + n for n in NATIVE_ORDER if "native_" + n in rows]
    cols = ["bele_eng", "bele_arb", "bele_jpn", "bele_zho", "bele_kor", "bele_hin", "bele_target5", "bele_transfer4",
            "mmmlu_mean5", "mmlu_en"]
    head = ["Belebele eng", "arb", "jpn", "zho", "kor", "hin", "target 5 mean", "transfer 4 mean (no text)",
            "MMMLU 5 mean", "MMLU en"]
    print("## Scores, mean +-sd over seeds (percent)\n")
    print("| arm | seeds | " + " | ".join(head) + " |")
    print("|---|---|" + "---|" * len(head))
    for n in names:
        d = [derived(r["scores"]) for r in rows[n]]
        print(f"| {n} | {len(d)} | " + " | ".join(ms([x[c] for x in d]) for c in cols) + " |")
    print("\n## Transfer languages (no training text), Belebele\n")
    print("| arm | spa | fra | ben | swh |")
    print("|---|---|---|---|---|")
    for n in names:
        d = [derived(r["scores"]) for r in rows[n]]
        print(f"| {n} | " + " | ".join(ms([x["bele_" + k[:3]] for x in d]) for k in BELE_TRANSFER) + " |")
    print("\n## MMMLU held-out 1000 per language (percent)\n")
    print("| arm | ar | ja | zh | ko | hi |")
    print("|---|---|---|---|---|---|")
    for n in names:
        d = [derived(r["scores"]) for r in rows[n]]
        print(f"| {n} | " + " | ".join(ms([x["mmmlu_" + k] for x in d]) for k in MMMLU) + " |")
    print("\n## Cost per arm\n")
    print("| arm | seed | params/token | trainable | train tokens | GPU | GPU h | list $ | decode tok/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    total = 0.0
    for n in names:
        for r in sorted(rows[n], key=lambda r: r["seed"]):
            total += r.get("dollars_list", 0)
            print(f"| {n} | {r['seed']} | {r.get('params_per_token', '')} | {r.get('trainable', '')} | "
                  f"{r.get('tokens_trained', '')} | {r.get('gpu', '')} | {r['gpu_hours']} | {r.get('dollars_list', '')} | "
                  f"{r.get('decode_tps', '')} |")
    print(f"\nFinal rows total: {total:.2f} dollars at list price (pilots and smokes not included).")


if __name__ == "__main__":
    main()
