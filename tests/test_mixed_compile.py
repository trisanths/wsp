import json
import os

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

import wsp.assemble
from wsp import Catalog, Program, load
from wsp.fetch import local_range_reader, read_tensors
from tests.test_align import WORDS, tiny_tokenizer

TEXTS = []
for i in range(12):
    TEXTS.append(" ".join(WORDS[i % 5: i % 5 + 6]))


def save_tiny(tmp_path):
    torch.manual_seed(0)
    host = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    donor = Qwen2ForCausalLM(Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    host_dir = str(tmp_path / "host")
    donor_dir = str(tmp_path / "donor")
    for model, path in [(host, host_dir), (donor, donor_dir)]:
        model.save_pretrained(path)
        tiny_tokenizer().save_pretrained(path)

    units_dir = tmp_path / "units"
    units_dir.mkdir()
    indices = torch.tensor([1, 4, 9, 20, 33, 50, 71, 90])
    torch.save({"kind": "mlp_cluster", "model": donor_dir, "layer": 1, "indices": indices}, units_dir / "c.pt")
    catalog_path = tmp_path / "catalog.jsonl"
    record = {"id": "qcluster", "type": "mlp_cluster", "model": donor_dir, "layer": 1, "units": "units/c.pt"}
    catalog_path.write_text(json.dumps(record) + "\n")
    return host_dir, donor_dir, Catalog.load(str(catalog_path))


def local_fetch(monkeypatch):
    def fetch(repo_id, names, revision="main", cache_dir=None):
        return read_tensors(local_range_reader(os.path.join(repo_id, "model.safetensors")), names)

    monkeypatch.setattr(wsp.assemble, "fetch_tensors", fetch)


def logits(model, text):
    ids = tiny_tokenizer()(text, return_tensors="pt").input_ids
    with torch.no_grad():
        return model(ids).logits.float()


def test_fresh_graft_compiles_and_loads_as_host(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qcluster").compile(out, catalog, align_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    graft = report["grafts"][0]
    assert graft["fit"] == "ridge"
    assert graft["fit_pairs"] > 0
    assert graft["host_layer"] == round(1 / 3 * 4)
    assert graft["part_params"] == 3 * 48 * 8 + 48

    model, _ = load(out, dtype=torch.float32)
    host = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_heal_changes_glue_and_load_restores_it(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    program = Program(host=host_dir).graft("qcluster", at=2).heal(steps=3, lr=1e-2, batch_size=2, seq_len=16)
    program.compile(out, catalog, heal_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["grafts"][0]["fit"] == "skipped"
    assert len(report["heal"]["losses"]) == 3
    saved = load_file(os.path.join(out, "graft_weights.safetensors"))
    assert saved["0.w_out.weight"].abs().sum() > 0

    model, _ = load(out, dtype=torch.float32)
    for key, value in model.wsp_grafts.state_dict().items():
        assert torch.equal(value.cpu(), saved[key].to(value.dtype))
    host = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert not torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_random_control_uses_same_count_other_units(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qcluster", control="random", control_seed=1).compile(out, catalog, device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert len(meta["indices"]) == 8
    assert meta["indices"] != [1, 4, 9, 20, 33, 50, 71, 90]
    assert meta["control"] == "random"


def test_old_program_json_still_loads():
    program = Program.from_dict({"base": "some/model", "seed": 0, "masks": None, "task_vectors": []})
    assert program.host == "some/model"
    assert program.base == "some/model"
    again = Program.from_dict(json.loads(json.dumps(program.graft("x", at=3).heal(steps=5).to_dict())))
    assert again.to_dict() == program.to_dict()
