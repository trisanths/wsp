# Pre-download donor weights into the Modal volume so mining runs don't wait on downloads.
# Uses the same cache (HF_HOME=/models/hf_cache) and the same pinned revision as the engine.
# CPU containers only, one per model, all in parallel.
#
#   .venv/bin/modal run mine/predownload.py                       # every donor in registry/donors.csv
#   .venv/bin/modal run mine/predownload.py --models nanbeige33b,gemma4_31b
import os
import sys
import time

import modal

app = modal.App("swarmy-predownload")
volume = modal.Volume.from_name("swarmy-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]")
    .env({"HF_HOME": "/models/hf_cache"})
)

PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja", "*.py"]


@app.function(image=image, volumes={"/models": volume}, secrets=[modal.Secret.from_name("hf-token")],
              cpu=4, memory=8192, timeout=3 * 3600)
def fetch(short, hf_id, revision):
    from huggingface_hub import snapshot_download

    start = time.time()
    path = snapshot_download(hf_id, revision=revision, allow_patterns=PATTERNS,
                             token=os.environ.get("HF_TOKEN"), max_workers=16)
    volume.commit()
    total = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            total = total + os.path.getsize(os.path.join(root, name))
    return {"model": short, "gb": round(total / 1e9, 1), "minutes": round((time.time() - start) / 60, 1)}


@app.local_entrypoint()
def main(models: str = ""):
    # local-only helper (the Modal containers don't have it, so import it here, not at the top)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common

    donors = common.load_donors()
    names = []
    if models:
        for m in models.split(","):
            if m.strip():
                names.append(m.strip())
    else:
        names = list(donors.keys())

    args = []
    for short in names:
        row = donors[short]
        # same revision the engine will load: the one pinned in the model's state file, else the current HEAD
        state = common.read_json_safe(os.path.join(common.logs_dir(short), "state.json"), {})
        revision = state.get("revision") or common.pin_revision(row["hf_id"])
        args.append((short, row["hf_id"], revision))
        print("queued", short, row["hf_id"], revision[:10])

    for result in fetch.starmap(args, return_exceptions=True):
        if isinstance(result, Exception):
            print("FAILED:", repr(result)[:300])
        else:
            print("done:", result)
