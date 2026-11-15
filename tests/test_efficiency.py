import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    if "modal_efficiency" in sys.modules:
        return sys.modules["modal_efficiency"]
    spec = importlib.util.spec_from_file_location("modal_efficiency", os.path.join(ROOT, "modal_efficiency.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_efficiency"] = module
    spec.loader.exec_module(module)
    return module


m = load()


def test_parse_config_merges_parts():
    kwargs = m.parse_config("fp8+kvfp8+ngram")
    assert kwargs["quantization"] == "fp8"
    assert kwargs["kv_cache_dtype"] == "fp8"
    assert kwargs["speculative_config"]["method"] == "ngram"
    assert kwargs["max_num_seqs"] == 64
    assert m.parse_config("seqs256")["max_num_seqs"] == 256
    assert m.parse_config("base") == m.DEFAULT_LLM_KWARGS


def test_unknown_part_and_empty_fail():
    with pytest.raises(ValueError):
        m.parse_config("fp8+nope")
    with pytest.raises(ValueError):
        m.select_configs(" , ")


def test_select_configs_dedupes():
    assert m.select_configs("base,fp8,base") == ["base", "fp8"]


def test_split_shards():
    assert m.split_shards(["a", "b", "c", "d"], 3) == [["a", "d"], ["b"], ["c"]]
    assert m.split_shards(["a"], 3) == [["a"]]


def test_model_ref():
    assert m.model_ref("hub:Qwen/x") == ("hub", "Qwen/x")
    with pytest.raises(ValueError):
        m.model_ref("Qwen/x")


def test_13gram_drop():
    test = ["def add the two numbers and return the sum of both values given in the arguments list please now"]
    texts = ["write add the two numbers and return the sum of both values given in the arguments list please now ok",
             "a completely different text about sorting arrays with a stable merge sort in python code today"]
    kept, dropped = m.drop_13gram(texts, test)
    assert dropped == 1 and kept == [texts[1]]
    kept, dropped = m.drop_13gram(["short text"], test)
    assert dropped == 0


def test_parse_log():
    lines = ["x Model loading took 29.92 GiB memory and 65.1 seconds", "Available KV cache memory: 66.79 GiB",
             "GPU KV cache size: 729,472 tokens, Maximum concurrency for 4,096 tokens per request: 178.09x",
             "Model loading took 1.2 GiB memory"]
    out = m.parse_log(lines)
    assert out["model_loading_gib"] == 29.92
    assert out["model_loading_all_gib"] == [29.92, 1.2]
    assert out["kv_cache_tokens"] == 729472
    assert out["kv_concurrency"] == 178.09


def test_spec_delta():
    a = {"drafts": 10, "draft_tokens": 40, "accepted": 10}
    b = {"drafts": 20, "draft_tokens": 80, "accepted": 40}
    d = m.spec_delta(a, b)
    assert d["rate"] == 0.75 and d["mean_accept_len"] == 1 + 30 / 10
    assert m.spec_delta(a, a) is None


def test_measure_decode_subtracts_prefill():
    t = {"now": 0.0}

    def clock():
        return t["now"]

    def generate(chunk, n):
        t["now"] += 1.0 + (0.01 * n if n > 1 else 0.0)

    tps, first = m.measure_decode(generate, [[1]] * 4, 2, 101, clock=clock)
    assert tps > 0 and len(first) == 2


def test_slim_tensor_plan():
    kept = {3: [1, 5, 9]}
    assert m.slim_tensor_plan("model.layers.3.mlp.experts.5.up_proj.weight", kept) == (
        "keep", "model.layers.3.mlp.experts.1.up_proj.weight")
    assert m.slim_tensor_plan("model.layers.3.mlp.experts.2.up_proj.weight", kept) == ("drop", None)
    assert m.slim_tensor_plan("model.layers.3.mlp.gate.weight", kept)[0] == "gate"
    assert m.slim_tensor_plan("model.embed_tokens.weight", kept) == ("keep", "model.embed_tokens.weight")


def test_build_slim_files_tiny(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    src = tmp_path / "src"
    src.mkdir()
    tensors = {"model.embed_tokens.weight": torch.zeros(2, 2),
               "model.layers.0.mlp.gate.weight": torch.arange(8.).reshape(4, 2)}
    for e in range(4):
        for p in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"model.layers.0.mlp.experts.{e}.{p}.weight"] = torch.full((2, 2), float(e))
    save_file(tensors, str(src / "a.safetensors"))
    json.dump({"weight_map": {k: "a.safetensors" for k in tensors}}, open(src / "model.safetensors.index.json", "w"))
    json.dump({"num_experts": 4}, open(src / "config.json", "w"))
    out = tmp_path / "out"
    m.build_slim_files(str(src), str(out), {"kept": {"0": [1, 3]}})
    index = json.load(open(out / "model.safetensors.index.json"))
    got = {}
    for f in set(index["weight_map"].values()):
        got.update(load_file(str(out / f)))
    assert got["model.layers.0.mlp.gate.weight"].tolist() == [[2., 3.], [6., 7.]]
    assert got["model.layers.0.mlp.experts.1.up_proj.weight"][0, 0] == 3.0
    assert "model.layers.0.mlp.experts.2.up_proj.weight" not in got
    assert json.load(open(out / "config.json"))["num_experts"] == 2


def test_merge_rows(tmp_path):
    path = str(tmp_path / "r.json")
    m.merge_rows(path, [{"cell": "a:b", "gpu_label": "H200", "smoke": False, "x": 1}])
    out = m.merge_rows(path, [{"cell": "a:b", "gpu_label": "H200", "smoke": False, "x": 2}])
    assert len(out) == 1 and out[0]["x"] == 2


def test_manifest_has_48_layers_of_64():
    with open(os.path.join(ROOT, m.MANIFEST)) as f:
        manifest = json.load(f)
    assert len(manifest["kept"]) == 48
    assert all(len(v) == 64 and v == sorted(v) for v in manifest["kept"].values())
