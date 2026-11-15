"""Fetch the math-rl rows from the volume and print the table (one line per cell, base first)."""
import glob
import json
import os
import subprocess
import sys
import tempfile

profile = sys.argv[1] if len(sys.argv) > 1 else "ztrisanth"
out = tempfile.mkdtemp()
env = dict(os.environ, MODAL_PROFILE=profile)
modal = os.path.join(os.path.dirname(sys.executable), "modal")
subprocess.run([modal, "volume", "get", "--force", "wsp-models", "wi_mathrl/v1/rows", out], env=env,
               check=True, capture_output=True)
order = ["base", "donor", "graft", "identity", "scrambled", "lora_match", "lora_big", "fullft"]
rows = []
for path in glob.glob(out + "/**/*.json", recursive=True):
    if path.endswith("_smoke.json"):
        continue
    rows.append(json.load(open(path)))
rows.sort(key=lambda r: (order.index(r["arm"]), r["seed"]))
print("cell | status | GSM8K | MATH500 | MMLU | devGSM | trainable | params_run | tokens | GPUh | tok/s")
for r in rows:
    s = r.get("scores", {})
    print(r["cell"], "|", r.get("status", "done"), "|", s.get("gsm8k"), "|", s.get("math500"), "|", s.get("mmlu"), "|",
          s.get("dev_gsm8k"), "|", r.get("trainable"), "|", r.get("params_run"), "|", r.get("train_tokens"), "|",
          r.get("gpu_hours"), "|", r.get("tok_s"))
