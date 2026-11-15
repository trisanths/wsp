# Manual steps of the pipeline. Run from ~/swarmy after deploying dissect/dissect_app.py.
#   .venv/bin/python dissect/cli.py base          base scores on every task (incl. generation accuracy)
#   .venv/bin/python dissect/cli.py sanity        empty part must reproduce base scores exactly
#   .venv/bin/python dissect/cli.py propose       attribution + activation scans (mlp and heads)
#   .venv/bin/python dissect/cli.py ablate '<part json>' [zero|mean]
#   .venv/bin/python dissect/cli.py validate '<part json>' <label>
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import client  # noqa: E402
from client import log  # noqa: E402

ALL_TASKS = ["gsm8k", "gsm8k_heldout", "trivia", "wiki", "wiki_digits", "gsm8k_gen"]
LP_TASKS = ["gsm8k", "gsm8k_heldout", "trivia", "wiki", "wiki_digits"]
OUT = os.path.join(HERE, "results")


def save(name, data):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(data, f, indent=1)


def main():
    step = sys.argv[1]
    d = client.dissector()
    start = time.time()

    if step == "base":
        info = d.model_info.remote()
        base = d.base_scores.remote(ALL_TASKS)
        print(json.dumps(info, indent=1))
        print(json.dumps(base, indent=1))
        save("base_scores.json", {"info": info, "base": base})
        log("dissect_base_scores", {"model": client.DONOR, "tasks": ALL_TASKS}, base)

    elif step == "sanity":
        results = {}
        for method in ["zero", "mean"]:
            empty_mlp = {"kind": "mlp_cluster", "layer": 0, "indices": []}
            r = d.ablate_and_score.remote(empty_mlp, method, LP_TASKS)
            results[method] = r
        exact = True
        for method in results:
            for t in LP_TASKS:
                b = results[method]["base"][t]["nll"]
                a = results[method]["ablated"][t]["nll"]
                if b != a:
                    exact = False
                print(method, t, "base nll", b, "ablated nll", a)
        print("EXACT MATCH:", exact)
        save("sanity.json", {"exact": exact, "results": results})
        log("dissect_sanity_empty_part", {"model": client.DONOR}, {"exact_match": exact})

    elif step == "propose":
        out = {}
        for kind in ["attn_heads", "mlp_cluster"]:
            out["attr_" + kind] = d.attribution_scan.remote(kind, 20)
            out["act_" + kind] = d.activation_profile.remote(kind, 20)
            print(kind, "attribution top 5:", json.dumps(out["attr_" + kind]["top_units"][:5]))
            print(kind, "activation top 5:", json.dumps(out["act_" + kind]["top_units"][:5]))
        save("propose_scan.json", out)
        log("dissect_propose_scan", {"model": client.DONOR}, {"saved": "dissect/results/propose_scan.json"})

    elif step == "ablate":
        part = json.loads(sys.argv[2])
        method = sys.argv[3] if len(sys.argv) > 3 else "mean"
        r = d.ablate_and_score.remote(part, method, LP_TASKS)
        print(json.dumps(r["drop"], indent=1))
        log("dissect_ablate", {"part": part, "method": method}, r["drop"])

    elif step == "validate":
        part = json.loads(sys.argv[2])
        label = sys.argv[3]
        r = client.validate_part(part, label, "manual", "manual-" + time.strftime("%H%M%S"))
        print(json.dumps(client.summarize_validation(r), indent=1))

    else:
        print("unknown step", step)
    print("seconds:", round(time.time() - start, 1))


if __name__ == "__main__":
    main()
