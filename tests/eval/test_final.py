import json
import os

from wsp_eval import final


def test_merge_ledgers_and_already_run(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "MAIN_ROOT", str(tmp_path / "main"))
    monkeypatch.setattr(final, "FINAL_ROOT", str(tmp_path / "final"))
    arm_work = tmp_path / "final" / "a__b" / "graft_s0" / "work"
    arm_work.mkdir(parents=True)
    with open(arm_work / "heldout_ledger.json", "w") as f:
        json.dump({"hash1|gsm8k": {"model": "/m/graft_s0", "time": "t"}}, f)

    assert not final.already_run("/m/graft_s0")
    assert final.merge_ledgers("a/b") == 1
    assert final.already_run("/m/graft_s0")
    assert final.merge_ledgers("a/b") == 0
    assert os.path.exists(tmp_path / "main" / "work" / "heldout_ledger.json")


def test_summarize_pairs_seeds():
    outputs = []
    for arm, scores in (("graft", [0.30, 0.31, 0.32]), ("random", [0.25, 0.26, 0.27]), ("lora", [0.4, 0.4, 0.4])):
        for seed in range(3):
            result = {"model": arm + str(seed), "split": "heldout", "program_seed": seed,
                      "tasks": {"gsm8k": {"score": scores[seed], "n": 1319}}}
            outputs.append({"row": {"arm": arm, "seed": seed, "gsm8k": scores[seed]}, "result": result})
    summary = final.summarize("a/b", "p", outputs, {"gsm8k": 0.2})
    assert summary["split"] == "heldout"
    assert len(summary["rows"]) == 9
    assert "COUNTS" in summary["verdict_text"]
