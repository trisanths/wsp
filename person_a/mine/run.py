# Run the mining pipeline on one donor model.
#   .venv/bin/python mine/run.py --model <short_name> --smoke                  # setup check, no agent, no OpenAI
#   .venv/bin/python mine/run.py --model <short_name> --budget-usd 25 --max-hours 3
# Re-running the same command resumes: finished screens and pinned revision are kept in logs/<model>/state.json.
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from client import Miner  # noqa: E402


def smoke(short):
    start = time.time()
    m = Miner(short)
    print("model:", m.hf_id, "| revision:", m.revision, "| GPU:", m.gpu)
    s = m.structure()
    brief = dict(s)
    brief.pop("layers_detail", None)
    print("structure:", json.dumps(brief, indent=1))
    base = m.base()
    print("base scores (geometric-mean prob of answer tokens):")
    for key in sorted(base["scores"]):
        print("  ", key, base["scores"][key]["score"])
    print("margin vs shuffled answers (nats; < 0.25 => built-in skill skipped):", base["margin_vs_shuffled"])
    print("skipped skills:", m.state.get("skipped"))
    # one ablation: the first two attention heads (or MLP columns) of the middle layer
    mid = s["n_layers"] // 2
    entry = s["layers_detail"][mid]
    if "attn" in entry:
        part = {"kind": "attn_heads", "layer": mid, "indices": [0, 1]}
    elif "mlp" in entry:
        part = {"kind": "mlp_cluster", "layer": mid, "indices": list(range(64))}
    else:
        part = {"kind": "block", "layers": [mid]}
    res = m.ablate(part)
    print("one ablation", json.dumps(part), "->", json.dumps(res["drops"]), "in", res["seconds"], "s")
    # MoE models: also check the expert hook on the first layer that has routed experts
    for e in s["layers_detail"]:
        if "experts" in e:
            epart = {"kind": "expert", "layer": e["layer"], "indices": [0, 1]}
            eres = m.ablate(epart)
            print("expert ablation", json.dumps(epart), "->", json.dumps(eres["drops"]), "in", eres["seconds"], "s")
            break
    total = round(time.time() - start, 1)
    common.log_event(short, "smoke", {"gpu": m.gpu}, {"ok": True, "wall_seconds": total, "ablation_seconds": res["seconds"]})
    print("smoke OK in", total, "s (wall clock, includes model load)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=25.0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--max-screens", type=int, default=60)
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--person", default=None, help="your name for the shared memory (else env WSP_PERSON)")
    ap.add_argument("--focus", default=None, help="what this agent should concentrate on (set by run_many --agents)")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh"],
                    help="Astra reasoning effort (high = the blog's setting; medium is cheaper)")
    args = ap.parse_args()
    if args.smoke:
        smoke(args.model)
        return
    import shutil

    import agent

    # a new parallel lane starts from the model's main state (pinned revision, base scores, skipped skills)
    lane_state = common.state_path(args.model)
    main_state = os.path.join(common.logs_dir(args.model), "state.json")
    if lane_state != main_state and not os.path.exists(lane_state) and os.path.exists(main_state):
        shutil.copyfile(main_state, lane_state)
    # Keep going until the budget, time or screen limit is really used: if the agent stops early,
    # start a fresh round with what is left (state, skills and shared memory carry over).
    start = time.time()
    spent = 0.0
    screens = 0
    for round_no in range(1, 6):
        budget_left = args.budget_usd - spent
        hours_left = args.max_hours - (time.time() - start) / 3600.0
        if budget_left < 0.15 * args.budget_usd or hours_left < 0.1 or screens >= args.max_screens:
            break
        print("[" + args.model + "] round", round_no, "| budget left $" + str(round(budget_left, 2)),
              "| hours left", round(hours_left, 2))
        cost = agent.run(args.model, budget_usd=budget_left, max_hours=hours_left,
                         max_screens=args.max_screens - screens, use_notes=not args.no_notes, effort=args.effort,
                         person=args.person, focus=args.focus)
        spent = spent + float(cost.get("openai_usd_est") or 0)
        screens = screens + int(cost.get("screens") or 0)
        if cost.get("stop_reason") not in ["finished", "no tool call"]:
            break  # budget, time or turn cap reached


if __name__ == "__main__":
    main()
