# Runs dissect/prepare_data.py on Modal (no local datasets install needed) and writes dissect/data/tasks.json.
#   cd ~/swarmy && .venv/bin/modal run dissect/prepare_modal.py
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("swarmy-dissect-prep")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("datasets")
    .add_local_file(os.path.join(HERE, "prepare_data.py"), "/root/prepare_data.py")
)


@app.function(image=image, timeout=1200)
def build():
    import sys

    sys.path.insert(0, "/root")
    import prepare_data

    return prepare_data.build_tasks()


@app.local_entrypoint()
def main():
    import sys

    sys.path.insert(0, HERE)
    import prepare_data

    tasks = build.remote()
    prepare_data.write_tasks(tasks)
