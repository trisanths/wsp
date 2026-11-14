"""Write catalog.jsonl entries for the public Deletion Tuning SFT masks.

Source: Hugging Face repo Jacklu0831/deletion-tuning-masks.
Base: unsloth/Llama-3.2-1B-Instruct, an ungated copy of
meta-llama/Llama-3.2-1B-Instruct (the meta repo needs an access token).

This is a seed catalog for compiler tests. The library owner (Person A)
owns the real catalog.
"""

import json
import urllib.request

REPO = "Jacklu0831/deletion-tuning-masks"
BASE = "unsloth/Llama-3.2-1B-Instruct"
SETS = [
    ("sft/llama-3.2-1b-instruct-mmlu-pro", "mmlupro", "MMLU-Pro {task} SFT"),
    ("sft/llama-3.2-1b-instruct-bbeh", "bbeh", "BBEH {task} SFT"),
]


def list_dirs(subdir):
    url = f"https://huggingface.co/api/models/{REPO}/tree/main/{subdir}"
    with urllib.request.urlopen(url) as response:
        entries = json.load(response)
    names = []
    for entry in entries:
        if entry["type"] == "directory":
            names.append(entry["path"].split("/")[-1])
    return names


def main():
    lines = []
    for subdir, prefix, trained_on in SETS:
        for task in list_dirs(subdir):
            short = task.removeprefix("bbeh_")
            record = {
                "id": f"mask_{prefix}_{short}_llama1b",
                "type": "deletion_mask",
                "base": BASE,
                "family": "llama3.2",
                "trained_on": trained_on.format(task=short),
                "path": f"hf://{REPO}/{subdir}/{task}",
                "source": "Lu et al. 2026, arXiv:2611.10866",
            }
            lines.append(json.dumps(record))
    with open("catalog.jsonl", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} records")


if __name__ == "__main__":
    main()
