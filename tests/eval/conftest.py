import json

import pytest

from wsp_eval import tasks as tasks_module


class ToyTask(tasks_module.Task):
    """Offline multiple-choice task with 400 items (the fake backend answers it)."""
    max_new_tokens = 8

    def __init__(self, name="toy"):
        self.name = name

    def load_items(self):
        items = []
        for i in range(400):
            items.append({"id": "toy-%04d" % i,
                          "messages": [{"role": "user", "content": self.name + " question " + str(i)}],
                          "answer": "ABCD"[i % 4], "num_options": 4})
        return items

    def score(self, output, item):
        if tasks_module.extract_letter(output, item["num_options"]) == item["answer"]:
            return 1.0
        return 0.0


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_ROOT", str(tmp_path))
    monkeypatch.setenv("WSP_BACKEND", "fake")
    monkeypatch.delenv("WSP_CATALOG", raising=False)
    tasks_module.register_task(ToyTask("toy"))
    tasks_module.register_task(ToyTask("toy2"))
    return tmp_path


def _save_tiny_tokenizer(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<eos>": 1}
    for i in range(62):
        vocab["w" + str(i)] = i + 2
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>")
    fast.save_pretrained(directory)


@pytest.fixture
def tiny_model(workspace):
    """A 4-layer random Llama with a tokenizer, two packbits masks and a task vector, in catalog.jsonl."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from wsp import masks as wsp_masks

    torch.manual_seed(0)
    model_config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                               num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(model_config).to(torch.bfloat16)
    base_dir = workspace / "base"
    model.save_pretrained(base_dir)
    _save_tiny_tokenizer(base_dir)

    # Fine-tuned copy for the task vector: down_proj of layer 2 shifted by 0.25.
    with torch.no_grad():
        model.model.layers[2].mlp.down_proj.weight.add_(0.25)
    tuned_dir = workspace / "tuned"
    model.save_pretrained(tuned_dir)
    _save_tiny_tokenizer(tuned_dir)

    # Masks over gate_proj (shape 32 x 16 = 512 entries), in the packbits_zstd format.
    mask_a = {}
    mask_b = {}
    for layer in range(4):
        name = "model.layers.%d.mlp.gate_proj.weight" % layer
        a = torch.zeros(512, dtype=torch.bool)
        a[0:10 + layer] = True
        b = torch.zeros(512, dtype=torch.bool)
        b[5:20] = True
        mask_a[name] = a.reshape(32, 16)
        mask_b[name] = b.reshape(32, 16)
    wsp_masks.save_mask_dir(mask_a, str(workspace / "components" / "a"))
    wsp_masks.save_mask_dir(mask_b, str(workspace / "components" / "b"))

    records = [
        {"id": "mask_a", "type": "deletion_mask", "base": str(base_dir), "family": "toy",
         "trained_on": "toy", "path": "components/a"},
        {"id": "mask_b", "type": "deletion_mask", "base": str(base_dir), "family": "toy",
         "trained_on": "toy2", "path": "components/b", "signature": {}},
        {"id": "tv_c", "type": "task_vector", "base": str(base_dir), "family": "toy",
         "trained_on": "toy", "path": str(tuned_dir), "model": str(tuned_dir)},
    ]
    with open(workspace / "catalog.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return base_dir
