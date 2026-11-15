"""Write jobs/wave2.json: two extra donor-span screens, then the native fine-tune finals at the lr picked on MMLU dev.

Pilot picks (seed 0, 1200 steps, MMLU dev 1000): llama3b 1e-6 (64.8), qwen1.5b 1e-6 (60.8), qwen3b 3e-6 (65.0).
"""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
hosts = {"llama3b": ("unsloth/Llama-3.2-3B-Instruct", 1e-6), "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", 3e-6),
         "qwen1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", 1e-6)}
jobs = [
    {"name": "graft_lora", "seed": 0, "overrides": {"first": 4, "last": 17, "steps": 4000, "tag": "_s4_17"}},
    {"name": "graft_lora", "seed": 0, "overrides": {"first": 8, "last": 19, "steps": 4000, "tag": "_s8_19"}},
]
for key, (host, lr) in hosts.items():
    for seed in (1, 2, 3):
        jobs.append({"name": "fullft", "seed": seed, "host": host,
                     "overrides": {"ft_lr": lr, "steps": 4000, "tag": f"_final_{key}"}})
json.dump(jobs, open(os.path.join(here, "wave2.json"), "w"), indent=1)
print(len(jobs))
