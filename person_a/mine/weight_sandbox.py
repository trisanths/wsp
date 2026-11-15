# Network-blocked Modal Sandbox holding one donor's raw weights, read-only.
# Same harness as "Sparks of Weight-Space Programming": the agent writes one Python block per turn,
# it runs in a fresh process, /work persists between runs, printed output comes back.
# No transformers / no model class is installed, so the network cannot be run here.
# Also used to run the agent's skill-test generators (deterministic, no network).
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_OUTPUT_CHARS = 12000  # every turn re-sends the whole context, so big outputs cost money
RUN_TIMEOUT_SECONDS = 600

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("safetensors", "tokenizers", "numpy")
    .add_local_file(os.path.join(HERE, "wsp_helpers.py"), "/wsp/wsp_helpers.py")
)

PRELUDE = "import sys\nsys.path.insert(0, '/wsp')\nfrom wsp_helpers import *\n"


class WeightSandbox:
    def __init__(self, hf_id, revision, params_b):
        self.hf_id = hf_id
        self.revision = revision
        self.params_b = params_b
        self.sb = None
        self.model_dir = None

    def start(self):
        app = modal.App.lookup("swarmy-mine-wsp", create_if_missing=True)
        volume = modal.Volume.from_name("swarmy-models", create_if_missing=True)
        memory = 32768
        if self.params_b > 16:
            memory = 65536  # big embedding / fused-expert tensors as float32
        self.sb = modal.Sandbox.create(app=app, image=image, volumes={"/models": volume.read_only()},
                                       block_network=True, cpu=8, memory=memory, timeout=6 * 3600, workdir="/work")
        try:
            self.model_dir = self._find_model_dir()
            if self.model_dir is None:
                raise RuntimeError("weights for " + self.hf_id + "@" + self.revision[:8] + " are not in the volume "
                                   "yet (the engine downloads them on first load; run --smoke first)")
        except Exception:
            self.stop()
            raise

    def _find_model_dir(self):
        folder = "/models/hf_cache/hub/models--" + self.hf_id.replace("/", "--") + "/snapshots/" + self.revision
        script = ("d=" + folder + "; if [ -f \"$d/config.json\" ] && ls \"$d\"/*.safetensors >/dev/null 2>&1; "
                  "then echo \"$d\"; fi")
        p = self.sb.exec("bash", "-c", script)
        p.wait()
        path = p.stdout.read().strip()
        return path or None

    def run(self, code, max_chars=MAX_OUTPUT_CHARS):
        p = self.sb.exec("env", "WSP_MODEL_DIR=" + self.model_dir, "python", "-c", PRELUDE + code,
                         timeout=RUN_TIMEOUT_SECONDS, workdir="/work")
        p.wait()
        out = p.stdout.read()
        err = p.stderr.read()
        text = out
        if err.strip():
            text = text + "\n[stderr]\n" + err[-4000:]
        if p.returncode != 0:
            text = text + "\n[exit code " + str(p.returncode) + "]"
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n[output truncated at " + str(max_chars) + " characters]"
        return text if text.strip() else "[no output]"

    def read_file(self, path):
        # raw file read (no truncation), for generated skill tests
        p = self.sb.exec("cat", path)
        p.wait()
        if p.returncode != 0:
            return None
        return p.stdout.read()

    def stop(self):
        if self.sb is not None:
            try:
                self.sb.terminate()
            finally:
                self.sb = None
