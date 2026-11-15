"""A hash of the code that the Modal app bakes into its image.

modal_studio.py computes it when it is deployed. The server computes it from the local files and compares
the two, so a deploy that is older than the local code gets a warning.
"""

import hashlib
import os

SKIP_DIRS = ["__pycache__", "parts_extracted"]


def code_files(repo_root):
    """Paths of modal_studio.py and every .py file under wsp/ and wsp_eval/, in a fixed order."""
    paths = [os.path.join(repo_root, "studio", "modal_studio.py")]
    for folder in ("wsp", "wsp_eval"):
        for current, dirs, files in os.walk(os.path.join(repo_root, folder)):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS)
            for name in sorted(files):
                if name.endswith(".py"):
                    paths.append(os.path.join(current, name))
    return paths


def local_hash(repo_root):
    """A short hash of the names and contents of the code files."""
    digest = hashlib.sha256()
    for path in code_files(repo_root):
        digest.update(os.path.relpath(path, repo_root).encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:16]
