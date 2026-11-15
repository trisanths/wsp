from wsp import clinic, rft, skills_eval


def fake_pubmed_rows():
    rows = []
    for k in range(10):
        rows.append({"pubid": 1000 + k, "question": f"Does drug {k} help?",
                     "context": {"contexts": [f"Background {k} text.", f"Results {k} text."]},
                     "final_decision": ["yes", "no", "maybe"][k % 3]})
    return rows


def test_pubmedqa_items_keep_only_the_test_ids_and_map_labels():
    rows = fake_pubmed_rows()
    items = skills_eval.pubmedqa_items(rows=rows, test_ids=["1000", "1001", "1002", "9999"])
    assert [i["id"] for i in items] == ["pubmedqa:1000", "pubmedqa:1001", "pubmedqa:1002"]
    assert [i["gold"] for i in items] == [0, 1, 2]
    assert all(i["choices"] == ["yes", "no", "maybe"] for i in items)
    assert items[0]["prompt"].startswith("Question: Abstract: Background 0 text.")
    assert items[0]["prompt"].endswith("\nAnswer:")


def test_official_pubmedqa_test_file_has_500_ids():
    ids = skills_eval.pubmedqa_test_ids()
    assert len(ids) == 500
    assert set(ids.values()) == {"yes", "no", "maybe"}


def test_medqa_items_read_options_dict_and_answer_index():
    rows = [{"question": "q1", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer_idx": "C"},
            {"question": "q2", "options": {"A": "w", "B": "x", "C": "y", "D": "z"}, "answer_idx": "A"}]
    items = skills_eval.medqa_items("test", rows=rows)
    assert [i["gold"] for i in items] == [2, 0]
    assert items[0]["id"] == "medqa_test:0"
    assert items[0]["choices"] == ["a", "b", "c", "d"]


def mmlu_rows():
    rows = []
    for subject in skills_eval.MED6_SUBJECTS + ["astronomy", "world_religions", "virology"]:
        for k in range(6):
            rows.append({"subject": subject, "question": f"{subject} question {k}", "choices": list("abcd"),
                         "answer": k % 4})
    return rows


def test_med6_group_and_general_set_are_disjoint_and_complete():
    rows = mmlu_rows()
    med = skills_eval.mmlu_subject_items(skills_eval.MED6_SUBJECTS, rows=rows)
    gen = skills_eval.mmlu_general_items(skills_eval.MED6_SUBJECTS, count=1000, rows=rows)
    assert len(med) == 36 and len(gen) == 18
    assert not ({i["id"].split(":")[1] for i in med} & {i["id"].split(":")[1] for i in gen})
    assert not ({i["subject"] for i in gen} & set(skills_eval.MED6_SUBJECTS))


def test_load_items_routes_clinic_skills(monkeypatch):
    rows = mmlu_rows()
    monkeypatch.setattr(skills_eval, "_mmlu_test_rows", lambda: rows)
    assert len(skills_eval.load_items("mmlu_med6")) == 36
    assert len(skills_eval.load_items("mmlu_general_ex_med6")) == 18
    assert set(skills_eval.CLINIC_SKILLS) >= {"medqa", "pubmedqa", "mmlu_med6"}


def test_reason_user_letters_follow_choice_count():
    four = clinic.reason_user("q", ["a", "b", "c", "d"])
    three = clinic.reason_user("q", ["yes", "no", "maybe"])
    assert "A, B, C, or D" in four
    assert "A, B, or C" in three and "D." not in three.split("\n\n")[0]


def test_train_items_have_gold_letter_and_reward():
    row = {"question": "Which?", "opa": "x", "opb": "y", "opc": "z", "opd": "w", "cop": 2}
    item = clinic.medmcqa_train_item(row, 5)
    assert item["gold"] == "C" and item["id"] == "medmcqa_train:5" and item["kind"] == "mcq"
    assert clinic.medmcqa_train_item({**row, "cop": -1}, 6) is None
    assert rft.rewards_for([item], ["It is z. Answer: C"]) == [1.0]
    assert rft.rewards_for([item], ["Answer: A"]) == [0.0]
    pub = clinic.pubmedqa_train_item(fake_pubmed_rows()[1], "pubmedqa_labeled")
    assert pub["gold"] == "B"
    assert clinic.pubmedqa_train_item({**fake_pubmed_rows()[0], "final_decision": None}, "p") is None


def test_build_train_drops_any_item_sharing_a_13_gram_with_a_test_item_and_counts_them():
    leaked_q = "A 45 year old woman presents with fatigue weight gain cold intolerance and a elevated TSH level on labs"
    test_items = [{"prompt": "Question: " + leaked_q + "\nA. x\nB. y\nC. z\nD. w\nAnswer:"}]
    index = clinic.test_index({"medqa": test_items})
    leaked = clinic.mcq_item("a", "src", leaked_q + " today", ["x", "y", "z", "w"], 0)
    fresh = clinic.mcq_item("b", "src", "Which vitamin deficiency causes scurvy in sailors on long voyages",
                            list("pqrs"), 1)
    items, stats = clinic.build_train({"src": [leaked, fresh]}, index, {"src": 0})
    assert [i["id"] for i in items] == ["b"]
    assert stats["src"] == {"candidates": 2, "dropped_overlap": 1, "kept": 1}


def test_sample_items_is_seeded_and_ordered():
    items = [{"id": str(i)} for i in range(100)]
    a = clinic.sample_items(items, 10, 1)
    assert a == clinic.sample_items(items, 10, 1)
    assert [int(i["id"]) for i in a] == sorted(int(i["id"]) for i in a)
    assert len(clinic.sample_items(items, 0)) == 100
