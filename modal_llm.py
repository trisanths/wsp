"""Serve an open-weight planner model on Modal with vLLM (spec 9.1; docs/research/modal.md sections 5 and 8g).

App "wsp-llm" holds one class, PlannerLLM, with one batched method:

    planner = modal.Cls.from_name("wsp-llm", "PlannerLLM")()
    texts = planner.generate.remote(list_of_messages, schema, temperature, seed, max_tokens)

wsp/swarm/llm.py ModalVLLM wraps that call (config llm = "modal_vllm"). It is
reached only through Modal auth. There is no web endpoint of any kind.

Deploy from the repo root. Auth, and so the workspace, come only from
MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env:

    set -a && . ./.env && set +a
    .venv/Scripts/modal deploy modal_llm.py

Settings, read from the environment when you deploy and baked into the image
so the container sees the same values:
    WSP_LLM_MODEL      Hugging Face repo id. Default Qwen/Qwen2.5-7B-Instruct (Apache-2.0).
    WSP_LLM_REVISION   repo commit. Default: a pinned commit for the default model, "main" for others.
    WSP_LLM_GPU        Modal GPU string. Default L40S (48 GB).
    WSP_LLM_MAX_LEN    context length in tokens. Default 16384.
    WSP_LLM_FAST_BOOT  "1" (default) runs vLLM in eager mode: no CUDA graph capture,
                       so a faster cold start and a slower decode. "0" turns it off.

Engine: vLLM 0.21.0 offline API (vllm.LLM.chat), with the image from Modal's
vLLM example (CUDA 12.9 devel base, vllm 0.21.0, transformers 5.8.1). JSON
output uses SamplingParams(structured_outputs=StructuredOutputsParams(json=schema)),
the vLLM 0.21 names (the guided_* fields were removed in vLLM 0.12).

Cost controls: max_containers=1, no min_containers, scaledown_window=300 s,
timeout=600 s per call. L40S lists at $1.95 per hour, and an idle container
bills for the 300 s scaledown window after its last call. Weights and vLLM
compile artifacts live on two Volumes, so only the first boot downloads.

Calls to this model never touch the eval or compute budget.
"""

import os
import time

import modal

APP_NAME = "wsp-llm"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"   # main on 2026-11-14 (last change 2025-01-12)

MODEL_NAME = os.environ.get("WSP_LLM_MODEL", DEFAULT_MODEL)
if MODEL_NAME == DEFAULT_MODEL:
    REVISION = os.environ.get("WSP_LLM_REVISION", DEFAULT_REVISION)
else:
    REVISION = os.environ.get("WSP_LLM_REVISION", "main")
GPU = os.environ.get("WSP_LLM_GPU", "L40S")
MAX_MODEL_LEN = int(os.environ.get("WSP_LLM_MAX_LEN", "16384"))
FAST_BOOT = os.environ.get("WSP_LLM_FAST_BOOT", "1") == "1"

MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "WSP_LLM_MODEL": MODEL_NAME,
        "WSP_LLM_REVISION": REVISION,
        "WSP_LLM_GPU": GPU,
        "WSP_LLM_MAX_LEN": str(MAX_MODEL_LEN),
        "WSP_LLM_FAST_BOOT": "1" if FAST_BOOT else "0",
    })
)

hf_cache = modal.Volume.from_name("wsp-llm-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("wsp-llm-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


def make_sampling_params(count, schema, temperature, seed, max_tokens):
    """One SamplingParams per conversation. Item i samples with seed + i (or no seed)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    params = []
    for index in range(count):
        item_seed = None
        if seed is not None:
            item_seed = int(seed) + index
        structured = None
        if schema is not None:
            structured = StructuredOutputsParams(json=schema)
        params.append(SamplingParams(temperature=float(temperature), seed=item_seed,
                                     max_tokens=int(max_tokens), structured_outputs=structured))
    return params


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    scaledown_window=300,
    max_containers=1,
    timeout=10 * MINUTES,
    startup_timeout=20 * MINUTES,
)
class PlannerLLM:
    """A vLLM offline engine for one open-weight instruct model, loaded once per container."""

    @modal.enter()
    def load(self):
        from vllm import LLM

        start = time.time()
        self.llm = LLM(
            model=MODEL_NAME,
            revision=REVISION,
            tokenizer_revision=REVISION,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.90,
            enforce_eager=FAST_BOOT,
            seed=0,
        )
        print("wsp-llm: loaded " + MODEL_NAME + "@" + REVISION[:12] + " in " +
              str(round(time.time() - start, 1)) + " s")

    @modal.method()
    def generate(self, list_of_messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        """Reply to every conversation in one vLLM batch. Returns a list of str, in input order.

        list_of_messages: a list of chats, each a list of {"role", "content"}.
        schema: a JSON schema dict; when given, every reply is constrained to it.
        seed: item i samples with seed + i, so a batch is repeatable and its items differ.
        """
        if len(list_of_messages) == 0:
            return []
        start = time.time()
        params = make_sampling_params(len(list_of_messages), schema, temperature, seed, max_tokens)
        outputs = self.llm.chat(list_of_messages, sampling_params=params, use_tqdm=False)
        texts = []
        prompt_tokens = 0
        reply_tokens = 0
        for output in outputs:
            texts.append(output.outputs[0].text)
            prompt_tokens += len(output.prompt_token_ids)
            reply_tokens += len(output.outputs[0].token_ids)
        print("wsp-llm: " + str(len(texts)) + " replies, " + str(prompt_tokens) + " prompt tokens, " +
              str(reply_tokens) + " reply tokens, " + str(round(time.time() - start, 2)) + " s")
        return texts
