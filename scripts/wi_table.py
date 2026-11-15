"""Markdown table of the big-donor same-tokenizer main rows (run scripts/wi_rows.py first to pull them)."""
import glob
import json

rows = {}
for f in glob.glob("/private/tmp/claude-501/wi_rows/*/**/*.json", recursive=True):
    r = json.load(open(f))
    if "smoke" not in r["name"]:
        rows[r["name"]] = r
order = ["ref_base", "ref_donor", "ref_sib3b", "ref_sib7b"]
for arm in ["graft", "identity", "scrambled", "lora", "lora128", "fullft"]:
    for s in (0, 1):
        order.append(f"main_{arm}_s{s}")
head = "| row | MMLU held-out | MMLU dev | ARC-C test | TriviaQA EM | GSM8K | params total (B) | params per token (B) | trainable (M) | train tokens (M) | GPU h (train/cell) | tok/s b=1 |"
print(head)
print("|" + "---|" * 12)


def f(r, k, scale=100, nd=1):
    v = r.get(k)
    return "" if v is None else f"{v * scale:.{nd}f}"


for name in order:
    r = rows.get(name)
    if r is None:
        continue
    if r.get("arm") in ("graft", "scrambled"):
        # model_params(base) already counts the graft's block through the base module tree; slice_params counted it again.
        dup = r["slice_params"] - 778572800
        r = dict(r, params_total=r["params_total"] - dup, params_run=r["params_run"] - dup)
    print("| {} | {} | {} | {} | {} | {} | {:.2f} | {:.2f} | {:.1f} | {} | {}/{} | {} |".format(
        name, f(r, "mmlu_heldout"), f(r, "mmlu_dev"), f(r, "arc_c_test"), f(r, "trivia_em"), f(r, "gsm8k"),
        r["params_total"] / 1e9, r["params_run"] / 1e9, r.get("trainable", 0) / 1e6,
        f(r, "train_tokens", 1e-6, 1) if r.get("train_tokens") else "0", f(r, "train_gpu_hours", 1, 2),
        f(r, "gpu_hours", 1, 2), f(r, "decode_tok_s", 1, 1)))
total = sum(r.get("gpu_hours", 0) for r in rows.values())
print("cell GPU hours in table:", round(total, 2))
