import pytest

import modal_clinic
import modal_efficiency_clinic as eff


def test_clinic_starts_and_stages():
    assert set(modal_clinic.STARTS) == {"k64", "k48", "full"}
    assert modal_clinic.estimate("k48", 2, 3, 8, 4) > 0
    with pytest.raises(SystemExit):
        modal_clinic.estimate("nope", 2, 3, 8, 4)


def test_clinic_score_sets_are_the_five_reported_sets():
    assert modal_clinic.SCORE_SETS == ["medmcqa", "medqa", "pubmedqa", "mmlu_med6", "mmlu_general_ex_med6"]
    assert eff.SCORE_SETS == modal_clinic.SCORE_SETS


def test_efficiency_clinic_parses_configs_like_the_original():
    assert eff.parse_config("base") == {"max_num_seqs": 64}
    assert eff.parse_config("marlin+eagle3b5")["moe_backend"] == "marlin"
    with pytest.raises(ValueError):
        eff.select_configs("")
    with pytest.raises(ValueError):
        eff.parse_config("nonsense")


def test_efficiency_clinic_rows_report_clinic_fields():
    assert "clinic_b1_tps" in eff.ROW_KEYS and "clinic" in eff.ROW_KEYS
    assert "humaneval" not in eff.ROW_KEYS
    row = {"cell": "c", "gpu": "L4", "clinic": {"medqa": {"cot": 0.8}}, "decode_tps": {"1": 100}}
    assert "medqa" in eff.summary_line(row)
