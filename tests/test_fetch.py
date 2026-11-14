import pytest
import torch
from safetensors.torch import save_file

from wsp import fetch


def make_file(tmp_path):
    tensors = {
        "a": torch.randn(3, 4),
        "b": torch.randn(5, 2).to(torch.bfloat16),
        "c": torch.arange(6, dtype=torch.int64),
    }
    path = str(tmp_path / "model.safetensors")
    save_file(tensors, path)
    return path, tensors


def test_read_tensors_from_local_file(tmp_path):
    path, tensors = make_file(tmp_path)
    read_range = fetch.local_range_reader(path)
    result = fetch.read_tensors(read_range, ["a", "b"])
    assert torch.equal(result["a"], tensors["a"])
    assert torch.equal(result["b"], tensors["b"])
    assert result["b"].dtype == torch.bfloat16


def test_read_only_requested_names(tmp_path):
    path, tensors = make_file(tmp_path)
    read_range = fetch.local_range_reader(path)
    result = fetch.read_tensors(read_range, ["c"])
    assert list(result.keys()) == ["c"]
    assert torch.equal(result["c"], tensors["c"])


def test_missing_name_raises(tmp_path):
    path, tensors = make_file(tmp_path)
    read_range = fetch.local_range_reader(path)
    with pytest.raises(KeyError, match="nope"):
        fetch.read_tensors(read_range, ["a", "nope"])


def test_http_range_reader_sends_range_header(monkeypatch):
    import requests

    calls = []

    class FakeResponse:
        status_code = 206
        content = b"0123456789"

    def fake_get(url, headers=None, **kwargs):
        calls.append((url, headers))
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    read_range = fetch.http_range_reader("https://example.com/file", token="tok")
    data = read_range(10, 10)
    assert data == b"0123456789"
    assert calls[0][1]["Range"] == "bytes=10-19"
    assert calls[0][1]["Authorization"] == "Bearer tok"


def test_fetch_tensors_uses_cache(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network function was called")

    monkeypatch.setattr(fetch, "_shard_map", fail)
    monkeypatch.setattr(fetch, "http_range_reader", fail)

    cache_dir = str(tmp_path / "cache")
    expected = torch.randn(2, 3)
    fetch._save_cached(cache_dir, "org/model", "layer.weight", expected)
    result = fetch.fetch_tensors("org/model", ["layer.weight"], cache_dir=cache_dir)
    assert torch.equal(result["layer.weight"], expected)
