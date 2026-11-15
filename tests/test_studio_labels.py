"""Part names in the palette: a math or code donor names the part, else the part's own library skill."""

from studio.server import catalog


def test_skill_topic_prefers_the_specific_skill():
    assert catalog.skill_topic(["math", "roman_numeral"]) == "Roman numeral"
    assert catalog.skill_topic(["law"]) == "Law"
    assert catalog.skill_topic(["math", "gsm8k"]) == "Math"
    assert catalog.skill_topic(["multilingual", "en_to_german_words"]) == "English to German"
    assert catalog.skill_topic(["prose_ru"]) == "Russian prose"
    assert catalog.skill_topic(["code_cpp"]) == "C++ code"
    assert catalog.skill_topic([]) is None


def test_part_topic_uses_the_donor_first():
    assert catalog.part_topic("Qwen/Qwen2.5-Math-1.5B", ["math", "product_sign"]) == "Math"
    assert catalog.part_topic("Qwen/Qwen2.5-Coder-1.5B", []) == "Code"
    assert catalog.part_topic("Equall/Saul-7B-Instruct-v1", ["decision", "law"]) == "Law"
    assert catalog.part_topic("google/gemma-2-9b-it", []) == "General"


def test_part_label_takes_skills():
    record = {"id": "x", "model": "LiquidAI/LFM2-24B-A2B", "type": "block", "layers": [0, 13]}
    assert catalog.part_label(record, ["law"]).startswith("Law block")
