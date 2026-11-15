"""The component catalog: one JSON record per line in catalog.jsonl.

A record has at least: id, type, base, path.
type is one of: deletion_mask, task_vector, lora, stitch.

path can be:
- a local directory (relative paths are relative to the catalog file), or
- "hf://<repo_id>/<subdir>" for a directory inside a Hugging Face repo.

A task_vector record also has "model": the fine-tuned model id. The
vector is (model - base), computed when the program compiles.
"""

import json
import os


class Catalog:
    def __init__(self, records, root="."):
        self.records = {}
        for record in records:
            if record["id"] in self.records:
                raise ValueError(f"duplicate catalog id: {record['id']}")
            self.records[record["id"]] = record
        self.root = root

    @classmethod
    def load(cls, path):
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                records.append(json.loads(line))
        return cls(records, root=os.path.dirname(os.path.abspath(path)))

    def get(self, component_id):
        if component_id not in self.records:
            raise KeyError(f"unknown component id: {component_id}")
        return self.records[component_id]

    def local_path(self, component_id):
        """Return a local directory for the component, downloading it if needed."""
        path = self.get(component_id)["path"]
        if path.startswith("hf://"):
            return download_hf_dir(path)
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, *path.split("/"))


def download_hf_dir(path):
    """Download hf://<owner>/<repo>/<subdir> and return the local subdir."""
    from huggingface_hub import snapshot_download

    parts = path[len("hf://"):].split("/")
    repo_id = parts[0] + "/" + parts[1]
    subdir = "/".join(parts[2:])
    local_root = snapshot_download(repo_id=repo_id, allow_patterns=[subdir + "/*"])
    return os.path.join(local_root, subdir)
