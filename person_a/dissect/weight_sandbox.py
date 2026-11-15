# A locked-down Modal Sandbox that holds one donor's weights for the agent to read.
#
# Follows the "Sparks of Weight-Space Programming" harness: the agent writes one Python
# block per turn, it runs in a sandbox with no network, a working folder is kept between
# runs, and the printed output comes back. The sandbox has torch + safetensors + tokenizers
# but no transformers and no model class, so the network cannot be run on inputs.
#
# Quick check:  .venv/bin/python dissect/weight_sandbox.py
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))

MAX_OUTPUT_CHARS = 20000
RUN_TIMEOUT_SECONDS = 600

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("safetensors", "tokenizers", "numpy")
    .add_local_file(os.path.join(HERE, "wsp_helpers.py"), "/wsp/wsp_helpers.py")
)

PRELUDE = "import sys\nsys.path.insert(0, '/wsp')\nfrom wsp_helpers import *\n"


class WeightSandbox:
    def __init__(self, hf_id):
        self.hf_id = hf_id
        self.app = modal.App.lookup("swarmy-wsp", create_if_missing=True)
        self.volume = modal.Volume.from_name("swarmy-models")
        self.sb = None
        self.model_dir = None
        self.runs = 0

    def start(self):
        # Sandbox lives up to 6 hours; CPU only (weight analysis does not need a GPU at 1.5B).
        self.sb = modal.Sandbox.create(
            app=self.app,
            image=image,
            volumes={"/models": self.volume.read_only()},  # agent code must not be able to change the weights
            block_network=True,
            cpu=8,
            memory=32768,
            timeout=6 * 3600,
            workdir="/work",
        )
        try:
            self.model_dir = self._find_model_dir()
            if self.model_dir is None:
                raise RuntimeError("weights for " + self.hf_id + " not found in the swarmy-models volume hf_cache")
        except Exception:
            self.stop()  # never leave a paid sandbox running after a failed start
            raise

    def _find_model_dir(self):
        folder = "models--" + self.hf_id.replace("/", "--")
        # pick a snapshot that has config, tokenizer and at least one weight file (not just the first one)
        script = ("for d in /models/hf_cache/hub/" + folder + "/snapshots/*/; do "
                  "if [ -f \"$d/config.json\" ] && [ -f \"$d/tokenizer.json\" ] && ls \"$d\"*.safetensors >/dev/null 2>&1; "
                  "then echo \"$d\"; break; fi; done")
        p = self.sb.exec("bash", "-c", script)
        p.wait()
        path = p.stdout.read().strip()
        if path == "":
            return None
        return path.rstrip("/")

    def run(self, code):
        # Runs one block of agent code in a fresh Python process; files in /work persist.
        self.runs = self.runs + 1
        p = self.sb.exec(
            "env", "WSP_MODEL_DIR=" + self.model_dir,
            "python", "-c", PRELUDE + code,
            timeout=RUN_TIMEOUT_SECONDS,
            workdir="/work",
        )
        p.wait()
        out = p.stdout.read()
        err = p.stderr.read()
        text = out
        if err.strip() != "":
            text = text + "\n[stderr]\n" + err
        if p.returncode != 0:
            text = text + "\n[exit code " + str(p.returncode) + "]"
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + "\n[output truncated at " + str(MAX_OUTPUT_CHARS) + " characters]"
        if text.strip() == "":
            text = "[no output]"
        return text

    def stop(self):
        if self.sb is not None:
            self.sb.terminate()
            self.sb = None


if __name__ == "__main__":
    hf_id = "Qwen/Qwen2.5-Math-1.5B"
    if len(sys.argv) > 1:
        hf_id = sys.argv[1]
    box = WeightSandbox(hf_id)
    box.start()
    try:
        print("model dir:", box.model_dir)
        print(box.run("print(len(names()), 'tensors'); print(names()[:5]); print(shape('model.layers.0.mlp.down_proj.weight'))"))
        print(box.run("print(decode(encode(' 12 + 7 = 19')))"))
        print(box.run("import urllib.request\nurllib.request.urlopen('https://example.com', timeout=5)"))
        print(box.run("import transformers"))
    finally:
        box.stop()
