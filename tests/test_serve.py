import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_serve():
    if "modal_serve" in sys.modules:
        return sys.modules["modal_serve"]
    spec = importlib.util.spec_from_file_location("modal_serve", os.path.join(ROOT, "modal_serve.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_serve"] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    chat_template = "yes"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


def test_vllm_is_not_imported():
    load_serve()
    assert "vllm" not in sys.modules


def test_prompts_use_chat_template_and_prefix_order():
    s = load_serve()
    pairs = [(f"write function {i}", "answer") for i in range(10)]
    full = s.build_prompts(FakeTokenizer(), 10, pairs=pairs)
    small = s.build_prompts(FakeTokenizer(), 4, pairs=pairs)
    assert small == full[:4]
    assert full[0] == "<user>write function 0<assistant>"
    long_pairs = [("x" * 5000, "a")]
    assert len(s.build_prompts(FakeTokenizer(), 1, pairs=long_pairs)[0]) < s.PROMPT_CHARS + 40


def test_prompts_short_pool_raises():
    s = load_serve()
    with pytest.raises(ValueError):
        s.build_prompts(FakeTokenizer(), 5, pairs=[("a", "b")])


def test_encode_adds_no_special_tokens():
    s = load_serve()
    assert s.encode_prompts(FakeTokenizer(), ["a b c"]) == [[0, 1, 2]]


def test_humaneval_prompt_matches_skills_eval():
    from wsp import skills_eval

    s = load_serve()
    item = {"id": "HumanEval/0", "prompt": "def f(x):\n    pass\n", "test": "", "entry_point": "f"}
    items, prompts = s.humaneval_prompts(FakeTokenizer(), [item])
    expected = skills_eval.render_prompt(FakeTokenizer(), skills_eval.user_text("humaneval", item))
    assert prompts == [expected]
    assert "def f(x)" in prompts[0]


def test_grade_humaneval_uses_extraction():
    s = load_serve()
    item = {"id": "HumanEval/0", "prompt": "def add(a, b):\n    pass\n",
            "test": "def check(f):\n    assert f(1, 2) == 3\n", "entry_point": "add"}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return 0\n```"
    graded = s.grade_humaneval([item, item], [good, bad])
    assert graded["n"] == 2 and graded["correct"] == 1 and graded["score"] == 0.5


def test_row_shape():
    s = load_serve()
    row = s.new_row("coder_7b", s.CELLS["coder_7b"], False, "trisanth")
    assert list(row) == s.ROW_KEYS
    assert row["prompts"] == 64 and row["gen_len"] == 256 and row["batches"] == [1, 32]
    assert row["profile_volume"] == "trisanth"
    err = s.error_row("slim_64", s.CELLS["slim_64"], False, "boom", skipped=True)
    assert list(err) == s.ROW_KEYS and err["error"] == "boom" and err["skipped"] is True
    json.dumps(err)


def test_write_row(tmp_path):
    s = load_serve()
    row = s.new_row("coder_3b", s.CELLS["coder_3b"], True)
    path = s.write_row(row, str(tmp_path))
    assert os.path.basename(path) == "smoke_coder_3b.json"
    assert json.load(open(path))["cell"] == "smoke_coder_3b"


def test_local_row_name(tmp_path, monkeypatch):
    s = load_serve()
    monkeypatch.chdir(tmp_path)
    row = s.new_row("full_30b", s.CELLS["full_30b"], False)
    assert s.write_row(row, s.LOCAL_DIR).endswith("serve_full_30b.json")


def test_check_path_missing_and_present(tmp_path):
    s = load_serve()
    root = str(tmp_path)
    ok, message = s.check_path("slim_64", s.CELLS["slim_64"], root, "ztrisanth")
    assert not ok
    assert "does not exist" in message and "ztrisanth" in message and "slim_64" in message
    path = os.path.join(root, s.CELLS["slim_64"]["dir"])
    os.makedirs(path)
    assert not s.check_path("slim_64", s.CELLS["slim_64"], root)[0]
    open(os.path.join(path, "config.json"), "w").write("{}")
    ok, message = s.check_path("slim_64", s.CELLS["slim_64"], root)
    assert not ok and "safetensors" in message
    open(os.path.join(path, "model-00001.safetensors"), "w").write("x")
    assert s.check_path("slim_64", s.CELLS["slim_64"], root) == (True, "")
    assert s.check_path("coder_3b", s.CELLS["coder_3b"], root) == (True, "")


def test_smoke_selection():
    s = load_serve()
    assert s.select_cells("full_30b", smoke=True) == ["coder_3b"]
    assert s.run_settings(True) == (8, (1, 8), False)
    assert s.run_settings(False) == (64, (1, 32), True)
    assert s.row_name("coder_3b", True) == "smoke_coder_3b"
    assert s.CELLS["coder_3b"]["model"] == "Qwen/Qwen2.5-Coder-3B-Instruct"


def test_select_cells():
    s = load_serve()
    assert s.select_cells("all") == ["full_30b", "slim_64", "slim_32_heal", "coder_7b", "coder_3b"]
    assert s.select_cells("coder_7b, coder_3b") == ["coder_7b", "coder_3b"]
    with pytest.raises(ValueError):
        s.select_cells("nope")


def test_measure_decode_removes_prefill():
    s = load_serve()
    now = [0.0]
    calls = []

    def clock():
        return now[0]

    def generate(chunk, max_tokens):
        calls.append((len(chunk), max_tokens))
        now[0] += 1.0 + 0.01 * (max_tokens - 1)

    ids = [[1, 2, 3]] * 4
    tps, first = s.measure_decode(generate, ids, 2, 256, clock=clock)
    assert first == [1.0, 1.0]
    assert tps == pytest.approx(4 * 255 / (2 * 0.01 * 255))
    assert calls == [(2, 1), (2, 256)] * 2
    with pytest.raises(ValueError):
        s.measure_decode(generate, ids[:1], 2, 256, clock=clock)


def test_prefill_and_ttft():
    s = load_serve()
    ticks = iter([0.0, 2.0])
    assert s.measure_prefill(lambda ids, n: None, [[1] * 10, [1] * 30], clock=lambda: next(ticks)) == 20.0
    median, p90 = s.ttft_stats([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    assert median == pytest.approx(55.0) and p90 == pytest.approx(90.0)
    assert s.ttft_stats([]) == (None, None)


def test_gpu_sampler_keeps_peak():
    s = load_serve()
    values = iter([1.0, 5.0, 3.0] + [2.0] * 100)
    sampler = s.GpuSampler(read=lambda: next(values), interval=0.001)
    with sampler:
        pass
    assert sampler.peak is not None and sampler.peak >= 1.0
    sampler.read = lambda: None
    sampler.sample()
    assert sampler.peak is not None


def test_weights_gib(tmp_path):
    s = load_serve()
    (tmp_path / "a.safetensors").write_bytes(b"x" * 1024)
    (tmp_path / "config.json").write_text("{}")
    assert s.weights_gib(str(tmp_path)) == pytest.approx(1024 / 1024 ** 3)


def test_pinned_vllm():
    s = load_serve()
    assert s.VLLM_VERSION == "0.32.0"
