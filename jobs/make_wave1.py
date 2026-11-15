"""Write jobs/wave1.json: the six donor-span screens, three zero-shot references, nine native lr pilots."""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
jobs = json.load(open(os.path.join(here, "screen.json")))
hosts = {"llama3b": "unsloth/Llama-3.2-3B-Instruct", "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
         "qwen3b": "Qwen/Qwen2.5-3B-Instruct"}
for k, h in hosts.items():
    jobs.append({"name": "reference", "seed": 0, "host": h, "overrides": {"tag": "_ref_" + k}})
for k, h in hosts.items():
    for lr in ("1e-6", "3e-6", "1e-5"):
        jobs.append({"name": "fullft", "seed": 0, "host": h,
                     "overrides": {"ft_lr": float(lr), "steps": 1200, "tag": f"_pilot_{k}_{lr}"}})
json.dump(jobs, open(os.path.join(here, "wave1.json"), "w"), indent=1)
print(len(jobs))
