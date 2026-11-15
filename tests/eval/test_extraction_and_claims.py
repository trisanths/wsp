import json
import os

import pytest
import torch

from wsp_eval import backends, claims, extraction, harness, units


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2)
    return LlamaForCausalLM(config).eval()


def _tiny_moe():
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    torch.manual_seed(0)
    config = Qwen2MoeConfig(vocab_size=64, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
                            shared_expert_intermediate_size=8, num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, num_experts=8, num_experts_per_tok=2)
    return Qwen2MoeForCausalLM(config).eval()


def _tiny_tokenizer(tmp_path):
    from tests.eval.conftest import _save_tiny_tokenizer
    from transformers import AutoTokenizer
    _save_tiny_tokenizer(tmp_path / "tok")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "tok")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_index_file_and_random_control(tmp_path):
    path = units.save_index_file(str(tmp_path / "c.pt"), "mlp_cluster", "donor/x", 5, [7, 3, 9])
    data = units.load_index_file(path)
    assert data["indices"].tolist() == [3, 7, 9]
    assert data["indices"].dtype == torch.long
    control = units.random_control(path, str(tmp_path / "ctl.pt"), {"intermediate_size": 32}, seed=1)
    control_data = units.load_index_file(control)
    assert control_data["layer"] == 5 and control_data["kind"] == "mlp_cluster"
    assert len(control_data["indices"]) == 3
    assert set(control_data["indices"].tolist()).isdisjoint({3, 7, 9})
    again = units.load_index_file(units.random_control(path, str(tmp_path / "ctl2.pt"), {"intermediate_size": 32}, 1))
    assert again["indices"].tolist() == control_data["indices"].tolist()   # seeded


@pytest.mark.parametrize("kind,indices", [("mlp_cluster", list(range(32))), ("attn_heads", [0, 1, 2, 3])])
def test_ablation_hooks_dense(kind, indices):
    model = _tiny_llama()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        before = model(ids).logits
    handles = units.add_ablation_hooks(model, {"kind": kind, "layer": 1, "indices": indices})
    with torch.no_grad():
        during = model(ids).logits
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        after = model(ids).logits
    assert not torch.allclose(before, during)
    assert torch.allclose(before, after)

    # Knocking out every neuron (or head) equals removing that MLP (or attention) output.
    layer = units.decoder_layers(model)[1]
    if kind == "mlp_cluster":
        captured = {}
        handles = units.add_ablation_hooks(model, {"kind": kind, "layer": 1, "indices": indices})
        hook = layer.mlp.register_forward_hook(lambda m, a, o: captured.update(out=o))
        with torch.no_grad():
            model(ids)
        hook.remove()
        for handle in handles:
            handle.remove()
        assert torch.allclose(captured["out"], torch.zeros_like(captured["out"]))


def test_ablation_hook_moe_router():
    model = _tiny_moe()
    router = units.moe_router(units.decoder_layers(model)[0])
    chosen = {}
    handles = units.add_ablation_hooks(model, {"kind": "expert", "layer": 0, "indices": [0, 1, 2, 3, 4, 5]})
    hook = router.register_forward_hook(lambda m, a, o: chosen.update(top=o.topk(2, dim=-1).indices))
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    hook.remove()
    for handle in handles:
        handle.remove()
    assert set(chosen["top"].reshape(-1).tolist()) <= {6, 7}


def test_extract_writes_index_files_controls_and_records(workspace, tmp_path):
    model = _tiny_llama()
    tokenizer = _tiny_tokenizer(tmp_path)
    skill = ["w1 w2 w3 w4 w5 w6"] * 6
    control = ["w40 w41 w42 w43 w44 w45"] * 6
    records = extraction.extract("toy/llama", task_name="toy", kinds=["mlp_cluster", "attn_heads", "expert"],
                                 n_units={"mlp_cluster": 4, "attn_heads": 1}, top_layers=2,
                                 out_dir=str(tmp_path / "out"), model=model, tokenizer=tokenizer, device="cpu",
                                 skill=skill, control=control)
    kinds = sorted(set(r["type"] for r in records))
    assert kinds == ["attn_heads", "mlp_cluster"]           # dense model: no experts
    assert len(records) == 4                                 # 2 layers x 2 kinds
    for record in records:
        data = units.load_index_file(record["units"])
        assert data["model"] == "toy/llama"
        assert data["kind"] == record["type"]
        assert len(record["controls"]) == 3
        control_data = units.load_index_file(record["controls"][0])
        assert len(control_data["indices"]) == len(data["indices"])
        assert set(control_data["indices"].tolist()).isdisjoint(data["indices"].tolist())
        assert data["signature"]["mean_effect"] >= data["signature"]["random_mean_effect"]
    lines = open(tmp_path / "out" / "records.jsonl").read().strip().split("\n")
    assert len(lines) == 4
    log = [json.loads(line) for line in open(workspace / "interp_log.jsonl")]
    assert any(entry["action"]["kind"] == "write_index" for entry in log)


def test_extract_finds_moe_experts(workspace, tmp_path):
    model = _tiny_moe()
    tokenizer = _tiny_tokenizer(tmp_path)
    records = extraction.extract("toy/moe", task_name="toy", kinds=["expert"], n_units={"expert": 2},
                                 top_layers=1, out_dir=str(tmp_path / "moe"), model=model, tokenizer=tokenizer,
                                 device="cpu", skill=["w1 w2 w3 w4"] * 4, control=["w50 w51 w52 w53"] * 4)
    assert len(records) == 1
    assert records[0]["type"] == "expert"
    assert len(units.load_index_file(records[0]["units"])["indices"]) == 2


def test_validate_with_fake_backend(workspace, tmp_path):
    path = units.save_index_file(str(tmp_path / "c.pt"), "mlp_cluster", "fake:donor", 1, [1, 2])
    control = units.save_index_file(str(tmp_path / "r.pt"), "mlp_cluster", "fake:donor", 1, [5, 6])
    result = extraction.validate(path, control, tasks=["toy"])
    entry = result["tasks"]["toy"]
    assert entry["drop"] == pytest.approx(entry["intact"] - entry["ablated"])
    assert units.load_index_file(path)["signature"]["ablation"]["tasks"]["toy"]["n"] == 150
    # The ablated runs are cached separately from the intact model.
    assert entry["ablated"] != entry["intact"] or entry["random_ablated"] != entry["intact"]


def test_fit_r2_seed_and_grafted_detection(workspace, tmp_path):
    model_dir = tmp_path / "grafted"
    model_dir.mkdir()
    (model_dir / "program.json").write_text(json.dumps({"host": "fake-host", "seed": 2,
                                                        "grafts": [{"id": "mlp_cluster_x"}]}))
    (model_dir / "grafts.json").write_text(json.dumps([{"id": "mlp_cluster_x"}]))
    (model_dir / "compile_report.json").write_text(json.dumps({"grafts": [
        {"id": "mlp_cluster_x", "kind": "mlp_cluster", "host_layer": 9, "fit": "ridge", "fit_r2": 0.71,
         "fit_r2_dim": 0.5, "fit_pairs": 900, "part_params": 100, "glue_params": 20, "control": None},
        {"id": "attn_heads_y", "fit": "skipped", "fit_r2": None}]}))
    assert backends.is_grafted(str(model_dir))
    fit = harness.read_fit_scores(str(model_dir))
    assert fit[0]["graft"] == "mlp_cluster_x" and fit[0]["r2"] == 0.71
    assert fit[0]["host_layer"] == 9 and fit[0]["fit"] == "ridge" and fit[0]["glue_params"] == 20
    assert fit[1]["r2"] is None and fit[1]["fit"] == "skipped"
    assert harness.program_seed(str(model_dir)) == 2

    result = harness.evaluate(str(model_dir), "search", ["toy"])
    assert result["fit_r2"][0]["r2"] == 0.71
    assert result["program_seed"] == 2
    last = json.loads(open(workspace / "experiments.jsonl").read().strip().split("\n")[-1])
    assert last["fit_r2"][0]["r2"] == 0.71

    # Without the fake override, grafted models go to the hf backend (vLLM cannot load them).
    os.environ.pop("WSP_BACKEND")
    try:
        assert backends.get_backend(str(model_dir)).name == "hf"
    finally:
        os.environ["WSP_BACKEND"] = "fake"


def test_heal_split_is_refused(workspace):
    with pytest.raises(ValueError):
        harness.evaluate("fake:m", "heal", ["toy"])


def test_claims_compare():
    graft = {0: 0.40, 1: 0.42, 2: 0.45}
    control = {0: 0.35, 1: 0.40, 2: 0.41}
    verdict = claims.compare(graft, control, "gsm8k", lora_results={0: 0.38, 1: 0.39, 2: 0.40})
    assert verdict["beats_main_control"]
    assert verdict["mean_gain"] == pytest.approx(0.0366667, abs=1e-6)
    assert verdict["gain_over_lora"] == pytest.approx(0.4233333 - 0.39, abs=1e-6)

    one_loss = claims.compare({0: 0.40, 1: 0.39, 2: 0.45}, control, "gsm8k")
    assert not one_loss["beats_main_control"]

    two_seeds = claims.compare({0: 0.5, 1: 0.5}, {0: 0.1, 1: 0.1}, "gsm8k")
    assert not two_seeds["beats_main_control"]
    assert "claims need 3" in two_seeds["why_not"][0]
    assert "DOES NOT COUNT" in claims.format_verdict(two_seeds)
