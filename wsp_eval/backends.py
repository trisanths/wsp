"""Generation backends. All use greedy decoding.

  vllm  the GPU backend for plain models (needs CUDA and the vllm package)
  hf    transformers backend: grafted models (vLLM cannot load them), ablation
        probes, and laptops (CUDA, MPS or CPU)
  fake  deterministic pseudo-outputs; used by the hour-1 stub and the tests

Choose with WSP_BACKEND, or let `get_backend` pick vllm when it can and
hf otherwise. A model path starting with "fake:" always uses the fake backend.
Grafted models (program.json lists grafts) always use hf and load with
wsp.load(path).

Both real backends render prompts with the same code (`render_prompt`) and pass
token ids, not text, so the prompts are identical across backends.
"""
import gc
import hashlib
import json
import os

from wsp_eval import config

# Fixed date for chat templates (Llama 3.x inserts today's date otherwise,
# which would change prompts from day to day).
TEMPLATE_DATE = "26 Jul 2024"

_backends = {}


def render_item(backend, model_path, item):
    """The exact prompt text for an item: raw `prompt`, or chat `messages` through the template."""
    if "prompt" in item:
        return backend.raw(model_path, item["prompt"])
    return backend.render(model_path, item["messages"])


def raw_prompt(tokenizer, text):
    """A raw text prompt, with the BOS token a plain tokenizer() call would add."""
    bos = tokenizer.bos_token
    if bos and tokenizer("x")["input_ids"][:1] == [tokenizer.bos_token_id] and not text.startswith(bos):
        return bos + text
    return text


def render_prompt(tokenizer, messages):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             date_string=TEMPLATE_DATE)
    parts = []
    for message in messages:
        parts.append(message["content"])
    return "\n\n".join(parts) + "\n"


def read_program(model_path):
    """The program.json saved next to a compiled model, or None."""
    path = os.path.join(str(model_path), "program.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def is_grafted(model_path):
    """A compiled model with grafts has grafts.json (see wsp/load.py)."""
    return os.path.isfile(os.path.join(str(model_path), "grafts.json"))


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        # A grafted output may not save a tokenizer; use the host's.
        program = read_program(model_path)
        if program is None:
            raise
        host = program.get("host") or program.get("base")
        return AutoTokenizer.from_pretrained(host)


class FakeBackend:
    name = "fake"

    def render(self, model_path, messages):
        parts = []
        for message in messages:
            parts.append(message["role"] + ": " + message["content"])
        return "\n".join(parts)

    def raw(self, model_path, text):
        return text

    def generate(self, model_path, prompts, max_new_tokens, model_hash, ablation=None):
        from wsp_eval import units
        outputs = []
        for prompt in prompts:
            digest = hashlib.sha256((model_hash + units.ablation_key(ablation) + prompt).encode("utf-8")).hexdigest()
            value = int(digest[:8], 16)
            letter = "ABCDEFGHIJ"[value % 10]
            number = value % 100
            if prompt.endswith("\nAnswer:"):
                outputs.append(" #### " + str(number) + "\n\nQuestion: next")
            elif "The final answer is" in prompt:
                outputs.append("The final answer is: " + str(number))
            elif "The answer is: <number>" in prompt:
                outputs.append("The answer is: " + str(number))
            else:
                outputs.append(letter)
        return outputs


class HFBackend:
    name = "hf"

    def __init__(self, batch_size=None):
        self.loaded_path = None
        self.model = None
        self.tokenizer = None
        if batch_size is None:
            batch_size = int(os.environ.get("WSP_HF_BATCH_SIZE", "16"))
        self.batch_size = batch_size

    def _device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load(self, model_path):
        if self.loaded_path == model_path:
            return
        self.unload()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(config.SEED)
        device = self._device()
        tokenizer = load_tokenizer(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        if is_grafted(model_path):
            # wsp.load returns (model, tokenizer); it rebuilds the host and attaches the grafts.
            from wsp.load import load as wsp_load
            model, tokenizer = wsp_load(model_path, device=device, dtype=dtype)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if os.environ.get("WSP_GRAFT_NO_CACHE") == "1":
                from wsp.graft import disable_cache
                disable_cache(model)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
        model.to(device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.loaded_path = model_path

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.loaded_path = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def render(self, model_path, messages):
        self._load(model_path)
        return render_prompt(self.tokenizer, messages)

    def raw(self, model_path, text):
        self._load(model_path)
        return raw_prompt(self.tokenizer, text)

    def generate(self, model_path, prompts, max_new_tokens, model_hash, ablation=None):
        from wsp_eval import units
        self._load(model_path)
        handles = []
        if ablation is not None:
            handles = units.add_ablation_hooks(self.model, ablation)
        try:
            return self._generate(prompts, max_new_tokens)
        finally:
            for handle in handles:
                handle.remove()

    def _generate(self, prompts, max_new_tokens):
        import torch
        tokenizer = self.tokenizer
        device = self._device()

        # Sort by length so each batch has little padding; restore order after.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs = [None] * len(prompts)
        for start in range(0, len(order), self.batch_size):
            batch_ids = order[start:start + self.batch_size]
            batch_prompts = []
            for i in batch_ids:
                batch_prompts.append(prompts[i])
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            encoded = encoded.to(device)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for row in range(len(batch_ids)):
                new_tokens = generated[row, prompt_length:]
                outputs[batch_ids[row]] = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return outputs


class VLLMBackend:
    name = "vllm"

    def __init__(self):
        self.loaded_path = None
        self.llm = None
        self.tokenizer = None

    def _load(self, model_path):
        if self.loaded_path == model_path:
            return
        self.unload()
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            seed=config.SEED,
            dtype="bfloat16",
            gpu_memory_utilization=float(os.environ.get("WSP_VLLM_GPU_UTIL", "0.85")),
            max_model_len=int(os.environ.get("WSP_VLLM_MAX_LEN", "8192")),
        )
        self.loaded_path = model_path

    def unload(self):
        if self.llm is None:
            return
        # vLLM does not always release GPU memory cleanly. This is the usual
        # best-effort cleanup; if it leaks, run one model per process.
        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        self.llm = None
        self.tokenizer = None
        self.loaded_path = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    def render(self, model_path, messages):
        self._load(model_path)
        return render_prompt(self.tokenizer, messages)

    def raw(self, model_path, text):
        self._load(model_path)
        return raw_prompt(self.tokenizer, text)

    def generate(self, model_path, prompts, max_new_tokens, model_hash, ablation=None):
        if ablation is not None:
            raise ValueError("ablation probes need the hf backend (hooks); vLLM cannot run them")
        from vllm import SamplingParams
        self._load(model_path)
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=config.SEED)
        requests = []
        for prompt in prompts:
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            requests.append({"prompt_token_ids": ids})
        results = self.llm.generate(requests, params, use_tqdm=True)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs


def _vllm_available():
    try:
        import torch
        import vllm  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


def get_backend(model_path, needs_hooks=False, name=None):
    if name is not None and os.environ.get("WSP_BACKEND") != "fake" and not str(model_path).startswith("fake:"):
        pass  # explicit choice by the caller
    elif str(model_path).startswith("fake:"):
        name = "fake"
    elif os.environ.get("WSP_BACKEND") == "fake":
        name = "fake"
    elif is_grafted(model_path) or needs_hooks:
        name = "hf"   # vLLM cannot load grafted models or run ablation hooks
    else:
        name = os.environ.get("WSP_BACKEND")
        if name is None:
            if _vllm_available():
                name = "vllm"
            else:
                name = "hf"
    if name not in _backends:
        if name == "fake":
            _backends[name] = FakeBackend()
        elif name == "hf":
            _backends[name] = HFBackend()
        elif name == "vllm":
            _backends[name] = VLLMBackend()
        else:
            raise ValueError("unknown backend: " + name)
    return _backends[name]
