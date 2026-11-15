import pytest
import torch

pytest.importorskip("modal")


def test_trim_top_keeps_largest_fraction():
    import modal_round16

    t = torch.tensor([0.1, -5.0, 0.3, 2.0, -0.2, 0.05, 1.0, -0.4, 0.0, 3.0])
    kept = modal_round16.trim_top(t, 0.2)
    assert torch.equal(kept, torch.tensor([0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]))
    assert torch.equal(modal_round16.trim_top(t, 1.0), t)
    assert torch.equal(modal_round16.trim_top(t, 0.0), torch.zeros_like(t))


def test_ties_sign_consensus_and_disjoint_mean():
    import modal_round16

    pre = {"w": torch.zeros(4), "odd": torch.zeros(2)}
    a = {"w": torch.tensor([1.0, 2.0, -3.0, 0.5]), "odd": torch.zeros(3)}
    b = {"w": torch.tensor([3.0, -1.0, -1.0, 0.5]), "odd": torch.zeros(2)}
    c = {"w": torch.tensor([-1.0, -4.0, 2.0, 0.5]), "odd": torch.zeros(2)}
    start = {"w": torch.full((4,), 9.0), "odd": torch.ones(2)}
    merged, skipped = modal_round16.ties_merge(pre, [a, b, c], keep=1.0, scale=1.0, start=start)
    # Entry 0: sum +3 -> mean of +1 and +3. Entry 1: sum -3 -> mean of -1 and -4.
    # Entry 2: sum -2 -> mean of -3 and -1. Entry 3: all agree -> 0.5.
    assert torch.allclose(merged["w"], torch.tensor([2.0, -2.5, -2.0, 0.5]))
    assert skipped == ["odd"]
    assert torch.equal(merged["odd"], torch.ones(2))


def test_ties_trim_drops_small_entries_before_the_vote():
    import modal_round16

    pre = {"w": torch.ones(4)}
    a = {"w": torch.tensor([1.0, 1.0, 1.0, 11.0])}
    b = {"w": torch.tensor([1.0, -9.0, 1.0, 1.0])}
    merged, skipped = modal_round16.ties_merge(pre, [a, b], keep=0.25, scale=0.5)
    assert skipped == []
    assert torch.allclose(merged["w"], torch.tensor([1.0, -4.0, 1.0, 6.0]))


def test_cells_and_gsm8k_solution():
    import modal_round16

    cells = modal_round16.cells()
    assert ("fuse3", 1) in cells and ("base", 0) in cells and ("base", 1) not in cells
    assert len(cells) == 6 + 6 * 2
    text = modal_round16.gsm8k_solution("She has 2*3=<<2*3=6>>6 eggs.\n#### 1,006")
    assert "<<" not in text and text.endswith("The answer is 1006.")


def test_medical_block_depth(monkeypatch):
    import modal_round16

    monkeypatch.setattr(modal_round16, "layer_count", lambda model_id: 32)
    assert modal_round16.medical_block("x", 28) == (10, 23, 8)


class FakeSibling:
    def __init__(self, name):
        self.rfilename = name


class FakeInfo:
    def __init__(self, files, gated=False):
        self.siblings = []
        for name in files:
            self.siblings.append(FakeSibling(name))
        self.gated = gated
        self.private = False
        self.disabled = False


def fake_api(infos):
    class FakeApi:
        def model_info(self, repo):
            value = infos[repo]
            if isinstance(value, Exception):
                raise value
            return value

    return FakeApi


def test_pick_medical_skips_bin_only_and_gated_repos(monkeypatch):
    import huggingface_hub

    import modal_round16

    monkeypatch.setattr(modal_round16, "MEDICAL_CANDIDATES", ["bin_only", "gated", "good"])
    infos = {
        "bin_only": FakeInfo(["pytorch_model.bin.index.json", "tokenizer.json"]),
        "gated": FakeInfo(["model.safetensors.index.json", "tokenizer.json"], gated="auto"),
        "good": FakeInfo(["model.safetensors.index.json", "tokenizer.json"]),
    }
    monkeypatch.setattr(huggingface_hub, "HfApi", fake_api(infos))
    repo, notes = modal_round16.pick_medical()
    assert repo == "good"
    assert len(notes) == 3


def test_pick_medical_stops_on_a_network_error(monkeypatch):
    import huggingface_hub

    import modal_round16

    monkeypatch.setattr(modal_round16, "MEDICAL_CANDIDATES", ["flaky", "good"])
    monkeypatch.setattr(modal_round16.time, "sleep", lambda seconds: None)
    infos = {"flaky": ConnectionError("down"), "good": FakeInfo(["model.safetensors", "tokenizer.json"])}
    monkeypatch.setattr(huggingface_hub, "HfApi", fake_api(infos))
    with pytest.raises(ConnectionError):
        modal_round16.pick_medical()


def test_medical_candidates_start_with_an_open_medical_model():
    import modal_round16

    assert modal_round16.MEDICAL_CANDIDATES[0] == "m42-health/Llama3-Med42-8B"
