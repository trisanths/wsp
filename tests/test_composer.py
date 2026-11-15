"""CPU tests for the composer. The OpenAI API and Modal are mocked; nothing goes over the network."""

import json
import os

import pytest

from composer import sizes
from composer.agent import load_request, run_loop, run_request, system_prompt
from composer.archive import Archive
from composer.llm import LLMError, OpenAIClient, read_key
from composer.report import build_results, heldout_jobs, lora_heal_spec, source_models
from composer.skills import gsm8k_correct, last_boxed, math_correct, objective, skill_scores
from composer.sources import Sources, sanitize
from composer.tools import Toolbox, read_language_text, tool_specs
from wsp.lang import parse
from wsp.program import Program

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "sk-test-composer-SECRET-0123456789abcdef"
HOST = "Qwen/Qwen2.5-0.5B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-1.5B-Instruct"
GOOD_PROGRAM = (f"host {HOST}\n"
                f"graft block/{DONOR}/8-20 at 7 rank 64 stitch mlp objective donor_lm steps 400\n"
                "heal steps 300 data gsm8k_train\n")
BAD_PROGRAM = f"host {HOST}\ngraft block/{DONOR}/8-20 at seven\nheal steps 300\n"


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


@pytest.fixture
def root(tmp_path):
    """A small repo: profiles, catalogs, an atlas entry, the language doc, and a .env with a fake key."""
    profiles = [
        {"key": "qwen05", "hf_id": HOST, "family": "qwen2", "params_b": 0.494, "active_params_b": 0.494,
         "layers": 24, "hidden": 896, "tokenizer_family": "qwen2", "scores": {"gsm8k": 0.335, "mmlu_dev": 0.48},
         "score_sources": {"gsm8k": "search split"}, "subject_scores": {}},
        {"key": "qwen_math15b", "hf_id": DONOR, "family": "qwen2", "params_b": 1.544, "active_params_b": 1.544,
         "layers": 28, "hidden": 1536, "tokenizer_family": "qwen2", "scores": {"gsm8k": 0.32},
         "subject_scores": {}},
        {"key": "qwen7b", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "family": "qwen2", "params_b": 7.616,
         "active_params_b": 7.616, "layers": 28, "hidden": 3584, "tokenizer_family": "qwen2",
         "scores": {"mmlu_dev": 0.715}, "subject_scores": {}},
    ]
    write_json(str(tmp_path / "profiles.json"), profiles)
    long_list = []
    for index in range(600):
        long_list.append(index * 0.001)
    records = [
        {"id": "mlp_math_l16", "type": "mlp_cluster", "model": DONOR, "family": "qwen2", "layer": 16,
         "units": "parts_extracted/x.pt", "d_model": 1536, "num_units": 512, "controls": ["a.pt", "b.pt"],
         "signature": {"mean_effect": 1.2, "unit_effects": long_list, "indices": list(range(512))}},
        {"id": "mask_law", "type": "deletion_mask", "base": HOST, "path": "hf://x/y/z",
         "signature": {"num_deleted": 5, "deleted_per_layer": long_list[:24],
                       "overlap_with": {"mask_bio": {"jaccard": 0.2, "jaccard_per_layer": {"0": 0.1}}}}},
    ]
    with open(tmp_path / "catalog.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    os.makedirs(tmp_path / "person_a")
    with open(tmp_path / "person_a" / "catalog.jsonl", "w") as f:
        f.write(json.dumps({"id": "heads_l22", "type": "attn_heads", "model": DONOR, "layer": 22,
                            "units": "parts/q/heads.pt", "num_units": 4, "d_model": 1536,
                            "signature": {"indices": [6, 9, 10, 11]}}) + "\n")
    # The wsp.atlas entry format (see wsp/atlas.py).
    write_json(str(tmp_path / "atlas" / "entries" / "qwen_math.json"),
               {"model": DONOR, "params_b": 1.544, "layers": 28, "hidden": 1536, "family": "qwen2",
                "skills": {"math": 0.61, "science": 0.55},
                "skill_details": {"math": {"score": 0.61, "count": 500}, "science": {"score": 0.55, "count": 500}},
                "layer_maps": {"math": {"blocks": [[8, 20]], "scores": long_list[:28]},
                               "science": {"blocks": [[8, 20]], "influence": long_list[:28] + long_list[:28]}},
                "expert_maps": {"math": {"3": [1, 2]}}})
    os.makedirs(tmp_path / "docs")
    with open(tmp_path / "docs" / "language.md", "w") as f:
        f.write("# The wsp text syntax\n\nhost, graft, heal.\n")
    with open(tmp_path / ".env", "w") as f:
        f.write(f"OTHER=1\nOPENAI_API_KEY=\"{KEY}\"\n")
    write_json(str(tmp_path / "request.json"), {
        "name": "math_test",
        "description": "best math under 2B",
        "skills": {"math": 1.0},
        "constraints": {"max_total_params_b": 2.0, "max_run_params_b": 2.0, "heal_steps_cap": 500,
                        "compile_budget": 3},
        "objective": "max math",
        "agent": {"max_turns": 10, "max_tokens": 100000},
    })
    return str(tmp_path)


def fake_practice(calls, result=None):
    def practice(program_dict, payload):
        calls.append((program_dict, payload))
        if result is not None:
            return result
        return {"ok": True, "model_dir": "/vol/composer/math_test/" + payload["program_hash"],
                "practice_sets": {"gsm8k_practice": 0.41, "math_practice": 0.19},
                "params_total_b": 1.12, "params_run_b": 1.12, "decode_tok_s": 80.0, "compile_seconds": 300.0,
                "practice_seconds": 120.0, "errors": [],
                "compile_summary": {"heal": {"loss_first20": 1.2, "loss_last20": 0.8}}}
    return practice


def toolbox_for(root, practice, constraints=None):
    request = load_request(os.path.join(root, "request.json"))
    if constraints is not None:
        request["constraints"].update(constraints)
    return Toolbox(request, Sources(root), practice, read_language_text(root))


def function_call(call_id, name, arguments):
    return {"type": "function_call", "id": "fc_" + call_id, "call_id": call_id, "name": name,
            "arguments": json.dumps(arguments), "status": "completed"}


def response(index, items, tokens=1000):
    return {"id": f"resp_{index}", "object": "response", "status": "completed",
            "output": [{"type": "reasoning", "id": f"rs_{index}", "summary": []}] + items,
            "usage": {"input_tokens": tokens - 100, "output_tokens": 100, "total_tokens": tokens}}


class FakePost:
    """Plays scripted Responses API replies, with one 429 first to exercise the retry."""

    def __init__(self, replies, fail_first=True):
        self.replies = list(replies)
        self.bodies = []
        self.headers = []
        self.urls = []
        self.fail_first = fail_first

    def __call__(self, url, headers, body, timeout):
        self.urls.append(url)
        self.headers.append(dict(headers))
        self.bodies.append(json.loads(json.dumps(body)))
        if self.fail_first:
            self.fail_first = False
            return 429, '{"error": "rate limited"}'
        if len(self.replies) == 0:
            return 200, json.dumps(response(99, [{"type": "message", "role": "assistant",
                                                  "content": [{"type": "output_text", "text": "done"}]}]))
        return 200, json.dumps(self.replies.pop(0))


def scripted_replies():
    leak = {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "I will not print " + KEY + " anywhere."}]}
    return [
        response(1, [leak, function_call("c1", "atlas", {"skill": "math"})]),
        response(2, [function_call("c2", "compile_and_practice", {"program_text": BAD_PROGRAM})]),
        response(3, [function_call("c3", "compile_and_practice", {"program_text": GOOD_PROGRAM})]),
        response(4, [function_call("c4", "finish", {"program_text": GOOD_PROGRAM,
                                                    "rationale": "math block from the 1.5B math model"})]),
    ]


def all_files(folder):
    paths = []
    for base, dirs, files in os.walk(folder):
        for name in files:
            paths.append(os.path.join(base, name))
    return paths


def long_numeric_lists(value, found):
    if isinstance(value, dict):
        for key in value:
            long_numeric_lists(value[key], found)
    elif isinstance(value, list):
        numbers = 0
        for item in value:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                numbers += 1
            else:
                long_numeric_lists(item, found)
        if numbers > 32:
            found.append(len(value))
    return found


# The agent loop.

def test_agent_loop_atlas_compile_twice_then_finish(root, capsys):
    calls = []
    post = FakePost(scripted_replies())
    sleeps = []
    client = OpenAIClient(read_key(os.path.join(root, ".env")), post=post, sleep=sleeps.append)
    runs = os.path.join(root, "composer", "runs")
    summary = run_request(os.path.join(root, "request.json"), fake_practice(calls), root=root, client=client,
                          runs_root=runs, stamp="t0")

    assert summary["stop_reason"] == "finish"
    assert summary["final"]["fallback"] is False
    assert summary["final"]["program_text"] == GOOD_PROGRAM
    assert summary["final"]["model_dir"].startswith("/vol/composer/math_test/")
    # The parse error cost nothing; the good program cost one compile.
    assert summary["compile_budget_used"] == 1
    assert len(calls) == 1
    program_dict, payload = calls[0]
    assert payload["practice_sets"] == ["gsm8k_practice", "math_practice"]
    assert payload["catalog"][0]["type"] == "block"
    assert payload["catalog"][0]["layers"] == [8, 20]
    # One 429, then a backoff, then the scripted turns.
    assert sleeps == [1.0]
    assert len(post.bodies) == 5
    for url in post.urls:
        assert url == "https://api.openai.com/v1/responses"
    body = post.bodies[-1]
    assert body["model"] == "gpt-6-astra"
    assert body["reasoning"] == {"effort": "high"}
    names = []
    for tool in body["tools"]:
        names.append(tool["name"])
    assert names == ["atlas", "catalog", "language_reference", "compile_and_practice", "finish"]
    # Tool results go back as function_call_output items with the same call ids.
    outputs = {}
    for item in body["input"]:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            outputs[item["call_id"]] = json.loads(item["output"])
    assert outputs["c2"]["stage"] == "parse"
    assert "line 2" in outputs["c2"]["error"]
    assert outputs["c3"]["ok"] is True
    assert outputs["c3"]["practice_scores"]["math"] == pytest.approx(0.3)
    assert outputs["c3"]["budget_left"] == 2

    # Archives: every request, reply, tool call, and result, plus final.json and log.txt.
    folder = os.path.join(runs, "math_test", "t0")
    kinds = {}
    for path in all_files(folder):
        name = os.path.basename(path)
        if "_" in name and name[0].isdigit():
            kind = name.split("_", 1)[1].rsplit(".", 1)[0]
            kinds[kind] = kinds.get(kind, 0) + 1
    assert kinds["request"] == 4
    assert kinds["response"] == 4
    assert kinds["tool_call"] == 4
    assert kinds["tool_result"] == 4
    assert os.path.exists(os.path.join(folder, "final.json"))
    assert os.path.exists(os.path.join(folder, "log.txt"))

    # The key is only in the Authorization header, never in a body, a file, or the output.
    for headers in post.headers:
        assert headers["Authorization"] == "Bearer " + KEY
    for body in post.bodies:
        assert KEY not in json.dumps(body).replace("I will not print " + KEY, "")
    for path in all_files(folder):
        with open(path) as f:
            text = f.read()
        assert KEY not in text, path
    assert "[redacted]" in open(os.path.join(folder, "log.txt")).read()
    captured = capsys.readouterr()
    assert KEY not in captured.out
    assert KEY not in captured.err


def test_loop_without_finish_submits_best_practiced(root):
    calls = []
    replies = [
        response(1, [function_call("c1", "compile_and_practice", {"program_text": GOOD_PROGRAM})]),
        response(2, [{"type": "message", "role": "assistant",
                      "content": [{"type": "output_text", "text": "thinking"}]}]),
    ]
    post = FakePost(replies, fail_first=False)
    client = OpenAIClient(KEY, post=post, sleep=lambda seconds: None)
    toolbox = toolbox_for(root, fake_practice(calls))
    archive = Archive(os.path.join(root, "runs"), "math_test", [KEY], stamp="t1")
    summary = run_loop(toolbox.request, toolbox, client, archive, max_turns=3, max_tokens=10 ** 6)
    assert summary["stop_reason"] == "turn cap (3)"
    assert summary["final"]["fallback"] is True
    assert summary["final"]["program_text"] == GOOD_PROGRAM
    # The reply without a tool call got a nudge.
    nudged = False
    for item in post.bodies[-1]["input"]:
        if isinstance(item, dict) and item.get("role") == "user" and "Use a tool" in str(item.get("content")):
            nudged = True
    assert nudged
    assert summary["compile_budget_used"] == 1


def test_token_budget_stops_the_loop(root):
    replies = [response(1, [function_call("c1", "atlas", {})], tokens=5000)]
    client = OpenAIClient(KEY, post=FakePost(replies, fail_first=False), sleep=lambda seconds: None)
    toolbox = toolbox_for(root, fake_practice([]))
    archive = Archive(os.path.join(root, "runs"), "math_test", [KEY], stamp="t2")
    summary = run_loop(toolbox.request, toolbox, client, archive, max_turns=10, max_tokens=4000)
    assert summary["stop_reason"].startswith("token budget")
    assert summary["turns"] == 1
    assert summary["final"]["no_model"] is True


# Budgets and constraints.

def test_budget_refusal(root):
    calls = []
    toolbox = toolbox_for(root, fake_practice(calls), {"compile_budget": 1})
    first = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert first["ok"] is True
    assert first["budget_left"] == 0
    # The same program again is cached and free.
    again = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert again["cached"] is True
    other = GOOD_PROGRAM.replace("at 7", "at 6")
    refused = toolbox.call("compile_and_practice", {"program_text": other})
    assert refused["ok"] is False
    assert refused["stage"] == "budget"
    assert "spent" in refused["error"]
    assert len(calls) == 1


def test_constraints_reject_oversized_programs(root):
    calls = []
    toolbox = toolbox_for(root, fake_practice(calls))
    big = "host Qwen/Qwen2.5-7B-Instruct\ngraft mlp_math_l16 at 10\nheal steps 300\n"
    result = toolbox.call("compile_and_practice", {"program_text": big})
    assert result["ok"] is False
    assert result["stage"] == "constraints"
    joined = " ".join(result["violations"])
    assert "total params" in joined
    assert "run per token" in joined
    assert result["estimate"]["total_params_b"] > 7.0

    too_long = GOOD_PROGRAM.replace("heal steps 300", "heal steps 5000")
    result = toolbox.call("compile_and_practice", {"program_text": too_long})
    assert result["stage"] == "constraints"
    assert "over the cap of 500" in " ".join(result["violations"])

    long_stitch = GOOD_PROGRAM.replace("steps 400", "steps 900")
    result = toolbox.call("compile_and_practice", {"program_text": long_stitch})
    assert "stitch on" in " ".join(result["violations"])

    lora_only = f"host {HOST}\nlora rank 16\nheal steps 300\n"
    result = toolbox.call("compile_and_practice", {"program_text": lora_only})
    assert "composes nothing" in " ".join(result["violations"])

    unknown = f"host {HOST}\ngraft block/acme/unknown-model/0-3\nheal steps 100\n"
    result = toolbox.call("compile_and_practice", {"program_text": unknown})
    assert "unknown size" in " ".join(result["violations"])

    # A tap donor counts in total params.
    tap = (f"host {HOST}\ntap Qwen/Qwen2.5-7B-Instruct layer 20 at 10\n"
           "tap_train data mmlu_aux steps 300\n")
    result = toolbox.call("compile_and_practice", {"program_text": tap})
    assert result["stage"] == "constraints"
    assert len(calls) == 0
    assert toolbox.budget_used == 0


def test_measured_size_over_the_cap_is_not_finishable(root):
    calls = []
    oversized = {"ok": True, "model_dir": "/vol/x", "practice_sets": {"gsm8k_practice": 0.9},
                 "params_total_b": 2.6, "params_run_b": 2.6}
    toolbox = toolbox_for(root, fake_practice(calls, oversized))
    result = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert result["within_constraints"] is False
    assert "measured total params" in result["violations"][0]
    finished = toolbox.call("finish", {"program_text": GOOD_PROGRAM, "rationale": "x"})
    assert finished["ok"] is False
    assert toolbox.finished is None


def test_size_estimates():
    models = {HOST: {"params_b": 0.494, "active_params_b": 0.494, "layers": 24, "hidden": 896,
                     "tokenizer_family": "qwen2"},
              DONOR: {"params_b": 1.544, "active_params_b": 1.544, "layers": 28, "hidden": 1536,
                      "tokenizer_family": "qwen2"}}
    program = parse(f"from {DONOR}\ncarve keep 14\nlora rank 8\nheal steps 100\n")
    estimate = sizes.estimate_program(program, models, {})
    embed = 151936 * 1536 / 1e9
    assert estimate.total == pytest.approx(embed + (1.544 - embed) / 2, rel=1e-6)
    splice = parse(f"splice base a sources a={HOST},b={DONOR} plan a:0..11, b:14..27\n")
    estimate = sizes.estimate_program(splice, models, {})
    assert estimate.total > 0.494
    assert sizes.block_record("block/Qwen/Qwen2.5-7B-Instruct/8-20")["layers"] == [8, 20]
    assert sizes.block_record("mlp_math_l16") is None
    assert sizes.block_record("block/Qwen/x/8") is None


# No tool returns raw weight values.

def test_no_tool_returns_weights_or_long_numeric_arrays(root):
    class FakeTensor:
        shape = (4096, 4096)
        dtype = "bfloat16"

    leaky = {"ok": True, "model_dir": "/vol/x", "practice_sets": {"gsm8k_practice": 0.5, "math_practice": 0.1},
             "params_total_b": 1.0, "params_run_b": 1.0,
             "compile_summary": {"heal": {"losses": [0.5] * 300}, "grafts": [{"W_in": FakeTensor(),
                                                                              "r2": [0.1] * 896}]}}
    toolbox = toolbox_for(root, fake_practice([], leaky))
    outputs = [
        toolbox.call("atlas", {}),
        toolbox.call("atlas", {"skill": "math"}),
        toolbox.call("atlas", {"skill": "law", "model": "qwen"}),
        toolbox.call("catalog", {}),
        toolbox.call("catalog", {"type": "mlp_cluster"}),
        toolbox.call("language_reference", {}),
        toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM}),
        toolbox.call("compile_and_practice", {"program_text": BAD_PROGRAM}),
        toolbox.call("finish", {"program_text": GOOD_PROGRAM, "rationale": "ok"}),
    ]
    for output in outputs:
        text = json.dumps(output)
        assert "FakeTensor" not in text
        assert long_numeric_lists(output, []) == [], text[:500]
        assert "indices" not in text
        assert "unit_effects\": [" not in text
    # The atlas entry and its long influence list came through as a summary.
    atlas = outputs[1]
    found = False
    for row in atlas["rows"]:
        if row.get("from", "").startswith("atlas"):
            found = True
            assert row["layer_maps"]["science"]["blocks"] == [[8, 20]]
            assert row["layer_maps"]["science"]["influence"]["summary"].startswith("list of 56")
    assert found
    # Earlier scores on this run's held-out items stay hidden.
    for output in outputs[:3]:
        for row in output["rows"]:
            scores = row.get("scores") or {}
            assert "mmlu_dev" not in scores
            assert "medmcqa" not in scores
    # The person_a parts paths are rebased on the repo root.
    records = Sources(root).records_by_id()
    assert records["heads_l22"]["units"] == "person_a/parts/q/heads.pt"
    # Mask overlaps are cut to the top few.
    for row in outputs[3]["rows"]:
        if row["id"] == "mask_law":
            assert row["signature"]["top_overlaps"] == [{"id": "mask_bio", "jaccard": 0.2}]


def test_sanitize_and_tool_specs():
    value = sanitize({"a": list(range(100)), "b": [1, 2, 3], "indices": [1], "c": "x" * 5000})
    assert value["a"]["summary"] == "list of 100 numbers"
    assert value["b"] == [1, 2, 3]
    assert "indices" not in value
    assert len(value["c"]) < 2000
    names = []
    for spec in tool_specs():
        names.append(spec["name"])
        assert spec["type"] == "function"
    assert names == ["atlas", "catalog", "language_reference", "compile_and_practice", "finish"]


# The client.

def test_client_retries_then_raises_without_the_key():
    statuses = [(500, "oops"), (502, "oops"), (200, json.dumps({"output": []}))]
    sleeps = []

    def post(url, headers, body, timeout):
        return statuses.pop(0)

    client = OpenAIClient(KEY, post=post, sleep=sleeps.append)
    assert client.create({"model": "m"}) == {"output": []}
    assert sleeps == [1.0, 2.0]

    def bad_request(url, headers, body, timeout):
        return 400, "bad key " + KEY

    client = OpenAIClient(KEY, post=bad_request, sleep=sleeps.append)
    with pytest.raises(LLMError) as error:
        client.create({"model": "m"})
    assert KEY not in str(error.value)
    assert "[redacted]" in str(error.value)

    def broken(url, headers, body, timeout):
        raise ConnectionError("down " + KEY)

    client = OpenAIClient(KEY, post=broken, sleep=lambda seconds: None, max_retries=2)
    with pytest.raises(LLMError) as error:
        client.create({"model": "m"})
    assert KEY not in str(error.value)


def test_read_key(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# c\nexport OPENAI_API_KEY='abc123'\n")
    assert read_key(str(path)) == "abc123"
    with pytest.raises(LLMError):
        OpenAIClient(None)


# Skills, the prompt, requests, and the report.

def test_answer_checks_and_objective():
    assert gsm8k_correct(" 3 + 4 = 7\n#### 7\nQuestion: next", "so #### 7")
    assert not gsm8k_correct("#### 8", "#### 7")
    assert last_boxed("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"
    assert math_correct("answer \\boxed{\\dfrac12}", "\\frac{1}{2}")
    assert math_correct("\\boxed{5.0}", "5")
    assert not math_correct("no box", "5")
    scores = skill_scores({"gsm8k_test": 0.6, "math500": 0.2, "humaneval": 0.5}, ["math", "code"], "heldout")
    assert scores["math"] == pytest.approx(0.4)
    assert scores["code"] == pytest.approx(0.5)
    assert objective({"math": 0.4, "code": None}, {"math": 1, "code": 1}) == pytest.approx(0.2)


def test_system_prompt_states_rules_and_budgets(root):
    request = load_request(os.path.join(root, "request.json"))
    text = system_prompt(request, 10, 100000)
    assert "never write, set, or edit raw weight values" in text
    assert "Held-out tests are unseen" in text
    assert "Compile budget: 3" in text
    assert "500 steps" in text
    assert chr(0x2014) not in text
    assert chr(0x2013) not in text


def test_shipped_requests_load():
    folder = os.path.join(REPO, "composer", "requests")
    names = sorted(os.listdir(folder))
    assert names == ["generalist_7b.json", "law_small.json", "math_code_med.json", "math_plus.json",
                     "smoke_qwen05.json"]
    caps = {}
    for name in names:
        request = load_request(os.path.join(folder, name))
        caps[request["name"]] = request["constraints"]["max_total_params_b"]
        for key in ["max_total_params_b", "max_run_params_b", "heal_steps_cap", "compile_budget"]:
            assert key in request["constraints"]
    assert caps == {"generalist_7b": 8.0, "law_small": 1.5, "math_code_med": 3.5, "math_plus": 2.0,
                    "smoke_qwen05": 2.0}


def test_report_sources_and_results():
    program = parse(GOOD_PROGRAM.replace("heal steps 300 data gsm8k_train",
                                         "tap Qwen/Qwen2.5-7B-Instruct layer 20 at 10\n"
                                         "tap_train steps 300\nheal steps 300 data gsm8k_train")).to_dict()
    models = source_models(program, {})
    assert models == [HOST, DONOR, "Qwen/Qwen2.5-7B-Instruct"]
    final = {"model_dir": "/vol/composer/r/abc", "program": program, "program_text": "t", "rationale": "r",
             "fallback": False, "no_model": False, "practice": {"params_total_b": 1.1}}
    jobs = heldout_jobs(final, program, {}, "/vol/composer/r/lora_abc")
    roles = []
    for role, ref in jobs:
        roles.append(role)
    assert roles == ["final", "base", "source", "source", "lora_on_base"]
    # The LoRA gets all the program's training steps: stitch 400, tap_train 300, heal 300.
    assert lora_heal_spec(program, {})["steps"] == 1000
    assert lora_heal_spec(program, {})["data"] == "gsm8k_train"
    no_training = parse(f"host {HOST}\nmask mask_law\n").to_dict()
    assert lora_heal_spec(no_training, {"heal_steps_cap": 100})["steps"] == 100
    request = {"name": "r", "skills": {"math": 1.0}, "constraints": {}}
    summary = {"final": final, "compile_budget_used": 2, "compile_budget": 5, "turns": 7, "tokens_used": 10,
               "stop_reason": "finish", "archive_dir": "x"}
    scored = [("final", "/vol/composer/r/abc", {"skills": {"math": 0.5}, "sets": {}}),
              ("base", HOST, {"skills": {"math": 0.3}, "sets": {}}),
              ("source", DONOR, "RuntimeError: out of memory"),
              ("lora_on_base", "/vol/composer/r/lora_abc", {"skills": {"math": 0.35}, "sets": {}})]
    out = build_results(request, summary, scored)
    assert out["final"]["objective"] == pytest.approx(0.5)
    assert out["comparisons"]["base"]["objective"] == pytest.approx(0.3)
    assert out["comparisons"]["sources"][0]["error"].startswith("RuntimeError")
    assert out["comparisons"]["lora_on_base"]["objective"] == pytest.approx(0.35)
    assert out["budget"]["compiles_used"] == 2
    assert "gsm8k_test" in out["heldout_sets"]


def test_repo_sources_load_without_an_atlas():
    sources = Sources(REPO)
    assert len(sources.catalog_records) > 0
    assert len(sources.models) > 0


# Regression tests: held-out data, budgets, checks, and fair comparisons.

def test_atlas_hides_scores_that_overlap_the_heldout_sets(root):
    profiles = json.load(open(os.path.join(root, "profiles.json")))
    profiles[0]["scores"]["mmlu_heldout"] = 0.45
    profiles[0]["scores"]["mmlu"] = 0.48
    profiles[0]["scores"]["medmcqa"] = 0.37
    profiles[0]["scores"]["newbench"] = 0.5
    profiles[0]["score_sources"]["newbench"] = "results/x.json: 500 seeded MATH test items, boxed"
    profiles[0]["subject_scores"] = {
        "mmlu": {"professional_law": 0.3}, "mmlu_heldout": {"jurisprudence": 0.4},
        "mmlu_dev": {"international_law": 0.5}, "law_train": {"jurisprudence": 0.6}}
    write_json(os.path.join(root, "profiles.json"), profiles)
    toolbox = toolbox_for(root, fake_practice([]))

    rows = toolbox.call("atlas", {})["rows"]
    for row in rows:
        scores = row.get("scores") or {}
        for name in ["mmlu_dev", "mmlu", "mmlu_heldout", "medmcqa", "newbench"]:
            assert name not in scores, (row.get("hf_id"), name)
    assert rows[0]["scores"]["gsm8k"] == 0.335

    law = toolbox.call("atlas", {"skill": "law"})["rows"]
    assert law[0]["law_subject_scores"] == {"law_train:jurisprudence": 0.6}

    # The wsp.atlas screens use this run's held-out sets, so those skill scores and the math maps hide.
    entry = None
    for row in toolbox.call("atlas", {"model": "Math"})["rows"]:
        if row["from"].startswith("atlas"):
            entry = row
    assert entry["skills"] == {"science": 0.55}
    assert list(entry["skill_details"]) == ["science"]
    assert list(entry["layer_maps"]) == ["science"]
    assert entry["expert_maps"] == {}
    text = json.dumps(toolbox.call("atlas", {"skill": "math"}))
    assert "0.61" not in text


def test_catalog_hides_person_a_signatures(root):
    with open(os.path.join(root, "person_a", "catalog.jsonl"), "w") as f:
        f.write(json.dumps({"id": "heads_l22", "type": "attn_heads", "model": DONOR, "layer": 22,
                            "units": "parts/q/heads.pt", "num_units": 4, "d_model": 1536,
                            "signature": {"effect": {"zero": {"gsm8k_heldout": 0.06, "gsm8k_gen": 0.4}},
                                          "base": {"gsm8k_gen": {"score": 0.7}}}}) + "\n")
    toolbox = toolbox_for(root, fake_practice([]))
    rows = toolbox.call("catalog", {"type": "attn_heads"})["rows"]
    assert len(rows) == 1
    assert "signature" not in rows[0]
    assert "GSM8K test" in rows[0]["heldout_overlap"]
    assert "gsm8k_gen" not in json.dumps(rows)
    # Other parts keep their signatures.
    other = toolbox.call("catalog", {"type": "mlp_cluster"})["rows"][0]
    assert other["signature"]["mean_effect"] == 1.2


def test_default_stitch_steps_count_against_the_cap(root):
    toolbox = toolbox_for(root, fake_practice([]), {"heal_steps_cap": 400})
    program = (f"host {HOST}\ngraft block/{DONOR}/8-20 at 7 rank 64 stitch mlp objective donor_lm\n"
               "heal steps 300 data gsm8k_train\n")
    result = toolbox.call("compile_and_practice", {"program_text": program})
    assert result["stage"] == "constraints"
    assert "500 steps, over the cap of 400" in " ".join(result["violations"])
    assert toolbox.budget_used == 0


def test_layer_ranges_and_component_kinds_are_checked_for_free(root):
    calls = []
    toolbox = toolbox_for(root, fake_practice(calls), {"max_total_params_b": 100.0, "max_run_params_b": 100.0})
    cases = [
        # A reversed block must not lower the size estimate.
        (f"host {HOST}\ngraft block/{DONOR}/20-8 at 7\nheal steps 100\n", "is after the last"),
        (f"host {HOST}\ngraft block/{DONOR}/8-40 at 7\nheal steps 100\n", "has layers 0 to 27"),
        (f"splice base a sources a={HOST},b={HOST} plan a:0..11, b:20..30\nheal steps 100\n",
         "has layers 0 to 23"),
        (f"host {HOST}\ntap {DONOR} layer 30 at 10\ntap_train data mmlu_aux steps 300\n", "has layers 0 to 27"),
        (f"host {DONOR}\nmask mask_law\nheal steps 100\n", "not on the host"),
        (f"host {HOST}\ngraft mask_law at 3\nheal steps 100\n", "a graft takes one of"),
        (GOOD_PROGRAM.replace("data gsm8k_train", "data gsm8k_test"), "heal data 'gsm8k_test' is unknown"),
    ]
    for text, expected in cases:
        result = toolbox.call("compile_and_practice", {"program_text": text})
        assert result["ok"] is False, text
        assert result["stage"] == "constraints", result
        assert expected in " ".join(result["violations"]), (text, result["violations"])
    assert len(calls) == 0
    assert toolbox.budget_used == 0
    estimate = sizes.estimate_program(parse(cases[0][0]), toolbox.sources.models, {})
    assert estimate.total > 0.494


def test_fallback_ranks_by_the_request_objective(root):
    request = load_request(os.path.join(root, "request.json"))
    request["skills"] = {"math": 1.0, "code": 1.0}

    def practice(program_dict, payload):
        if payload["skills"] == ["math"]:
            sets = {"gsm8k_practice": 0.9, "math_practice": 0.9}
        else:
            sets = {"gsm8k_practice": 0.6, "math_practice": 0.6, "mbpp_practice": 0.6}
        return {"ok": True, "model_dir": "/vol/x/" + payload["program_hash"], "practice_sets": sets,
                "params_total_b": 1.1, "params_run_b": 1.1}

    toolbox = Toolbox(request, Sources(root), practice, "")
    math_only = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM, "practice_tasks": ["math"]})
    other = GOOD_PROGRAM.replace("at 7", "at 6")
    both = toolbox.call("compile_and_practice", {"program_text": other})
    assert math_only["practice_objective"] == pytest.approx(0.9)
    assert math_only["request_objective"] == pytest.approx(0.45)
    assert both["request_objective"] == pytest.approx(0.6)
    final = toolbox.fallback_finish("stopped")
    assert final["program_text"] == other


def test_a_failed_practice_call_is_not_cached(root):
    attempts = []

    def flaky(program_dict, payload):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("the container was lost")
        return fake_practice([])(program_dict, payload)

    toolbox = toolbox_for(root, flaky)
    first = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert first["ok"] is False
    assert "TimeoutError" in first["errors"][0]
    second = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert second["ok"] is True
    assert second.get("cached") is None
    assert toolbox.budget_used == 2


def test_a_compile_without_a_measured_size_is_not_finishable(root):
    no_size = {"ok": True, "model_dir": "/vol/x", "practice_sets": {"gsm8k_practice": 0.9}}
    toolbox = toolbox_for(root, fake_practice([], no_size))
    result = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert result["within_constraints"] is False
    assert "did not report params_total_b" in " ".join(result["violations"])
    assert toolbox.call("finish", {"program_text": GOOD_PROGRAM, "rationale": "x"})["ok"] is False


def test_heal_data_always_uses_seed_zero(monkeypatch):
    import wsp.data
    import wsp.pairs
    from composer.evalsets import heal_texts_for

    seeds = []

    def fake_pairs_for(name, tokenizer=None, seed=0, count="default", max_prompt_tokens="default"):
        seeds.append(seed)
        return [("Question: q\nAnswer:", " a")]

    def fake_task_texts(name, seed=0):
        seeds.append(seed)
        return ["t"]

    monkeypatch.setattr(wsp.pairs, "pairs_for", fake_pairs_for)
    monkeypatch.setattr(wsp.data, "task_texts", fake_task_texts)
    # gsm8k_practice is the seed 0 search split; another seed would put those items in the heal data.
    assert heal_texts_for("gsm8k_train", None, seed=3) == ["Question: q\nAnswer: a"]
    assert heal_texts_for("not_a_pairs_name", None, seed=5) == ["t"]
    assert seeds == [0, 0]


def test_lora_baseline_gets_the_same_training(root):
    from composer.report import lora_program

    program = parse(f"host {HOST}\ngraft block/{DONOR}/8-20 at 7 stitch mlp objective donor_lm steps 400\n"
                    "lora rank 64\nheal steps 300 data math_train lr 0.0001\n").to_dict()
    baseline = lora_program(program, {"heal_steps_cap": 1000}, ["math"])
    assert baseline["host"] == HOST
    assert baseline["grafts"] == []
    assert baseline["lora"]["rank"] == 64
    assert baseline["heal"]["data"] == "math_train"
    assert baseline["heal"]["lr"] == 0.0001
    assert baseline["heal"]["steps"] == 700
    # No heal: the request's own skill data, not GSM8K for every request.
    masked = parse(f"host {HOST}\nmask mask_law\n").to_dict()
    assert lora_program(masked, {}, ["law"])["heal"]["data"] == "mmlu_aux"
    assert lora_program(masked, {}, ["law"])["lora"]["rank"] == 16
    assert lora_program(masked, {}, ["math", "code"])["heal"]["data"] == "composer_mix"


def test_results_mark_oversized_comparisons_and_heldout_parts():
    program = parse(f"host {HOST}\ngraft heads_l22 at 10\nheal steps 100\n").to_dict()
    records = {"heads_l22": {"id": "heads_l22", "type": "attn_heads", "model": DONOR,
                             "heldout_overlap": "mined on GSM8K test items"}}
    final = {"model_dir": "/vol/a", "program": program, "program_text": "t", "practice": {}}
    request = {"name": "r", "skills": {"math": 1.0}, "constraints": {"max_total_params_b": 1.0}}
    summary = {"final": final}
    scored = [("final", "/vol/a", {"skills": {"math": 0.5}, "params_total_b": 0.6}),
              ("source", DONOR, {"skills": {"math": 0.6}, "params_total_b": 1.5})]
    out = build_results(request, summary, scored, records=records)
    assert out["final"]["fits_constraints"] is True
    assert out["comparisons"]["sources"][0]["fits_constraints"] is False
    assert out["heldout_overlap"] == [{"id": "heads_l22", "note": "mined on GSM8K test items"}]


def test_token_budget_counts_replies_without_usage(root):
    reply = response(1, [function_call("c1", "atlas", {})])
    del reply["usage"]
    client = OpenAIClient(KEY, post=FakePost([reply] * 5, fail_first=False), sleep=lambda seconds: None)
    toolbox = toolbox_for(root, fake_practice([]))
    archive = Archive(os.path.join(root, "runs"), "math_test", [KEY], stamp="t3")
    summary = run_loop(toolbox.request, toolbox, client, archive, max_turns=5, max_tokens=500)
    assert summary["stop_reason"].startswith("token budget")
    assert summary["turns"] == 1


def test_modal_payloads_carry_no_key(root):
    calls = []
    post = FakePost(scripted_replies(), fail_first=False)
    client = OpenAIClient(KEY, post=post, sleep=lambda seconds: None)
    run_request(os.path.join(root, "request.json"), fake_practice(calls), root=root, client=client,
                runs_root=os.path.join(root, "runs"), stamp="t4")
    assert len(calls) == 1
    assert KEY not in json.dumps(calls)


# The unified language: taps, experts, inserts, gates, distillation, budget.

EXPERTS_PROGRAM = (f"host {HOST}\nexperts {DONOR} top_k 1\nlora rank 8\n"
                   "heal steps 200 data gsm8k_train lr 1e-4\n")
INSERT_PROGRAM = (f"host {HOST}\ninsert {DONOR} layers 4..9 at 7 stitch mlp\nlora rank 8\n"
                  "heal steps 200 data gsm8k_train\n")
GATED_PROGRAM = (f"host {HOST}\ngraft block/{DONOR}/8-14 at 3 gate on gate_bias -1.0\n"
                 f"graft block/{DONOR}/15-20 at 5 gate on gate_bias -1.0\nheal steps 200\n")
DISTILL_PROGRAM = (f"host {HOST}\ngraft block/{DONOR}/8-14 at 7 rank 32\nlora rank 8\n"
                   f"heal steps 200 data gsm8k_train distill {DONOR} kd_weight 0.5\n")
TAP_PROGRAM = (f"host {HOST}\ntap Qwen/Qwen2.5-7B-Instruct layer 20 at 10\ntap_train steps 100\n")


def check_text(root, text, constraints=None):
    toolbox = toolbox_for(root, fake_practice([]), constraints)
    program, error = toolbox.parse(text)
    assert error is None, error
    return toolbox.check(program)


def test_every_statement_but_tap_is_accepted(root):
    for text in [EXPERTS_PROGRAM, INSERT_PROGRAM, GATED_PROGRAM, DISTILL_PROGRAM]:
        result = check_text(root, text)
        assert result["ok"], (text, result["violations"])


def test_a_program_with_taps_is_rejected_with_the_reason(root):
    result = check_text(root, TAP_PROGRAM)
    assert result["ok"] is False
    text = " ".join(result["violations"])
    assert "cannot write taps" in text
    assert "runs a source model" in text
    toolbox = toolbox_for(root, fake_practice([]))
    out = toolbox.call("compile_and_practice", {"program_text": TAP_PROGRAM})
    assert out["stage"] == "constraints"
    assert out["budget_used"] == 0


def test_experts_and_insert_sizes_count_against_the_limits(root):
    experts = check_text(root, EXPERTS_PROGRAM)["estimate"]
    # The host plus the donor's MLPs (about two thirds of 24 layers of a 1.5B model).
    assert 1.0 < experts["total_params_b"] < 1.6
    # top_k 1 runs one MLP per layer: the donor's is the larger, so the run count grows a little.
    assert experts["run_params_b"] < experts["total_params_b"]
    insert = check_text(root, INSERT_PROGRAM)["estimate"]
    assert insert["total_params_b"] > 0.494
    tight = check_text(root, EXPERTS_PROGRAM, {"max_total_params_b": 0.9})
    assert tight["ok"] is False
    assert "over the cap 0.9b" in " ".join(tight["violations"])
    far = check_text(root, "host " + HOST + f"\nexperts {DONOR}\nlora rank 8\nheal steps 100\n",
                     {"max_run_params_b": 0.5})
    assert "params run per token" in " ".join(far["violations"])


def test_growing_the_base_is_allowed_when_the_limits_allow_it(root):
    big = (f"host {HOST}\ngraft block/Qwen/Qwen2.5-7B-Instruct/4-9 at 7 stitch mlp objective downstream steps 100\n"
           "heal steps 100\n")
    assert check_text(root, big, {"max_total_params_b": 8.0, "max_run_params_b": 8.0})["ok"]
    assert check_text(root, big)["ok"] is False


def test_added_kinds_cannot_mix_and_gates_must_agree(root):
    mixed = (f"host {HOST}\ngraft block/{DONOR}/8-14 at 3\nexperts {DONOR}\nlora rank 8\nheal steps 100\n")
    assert "mixes graft, experts" in " ".join(check_text(root, mixed)["violations"])
    unequal = (f"host {HOST}\ngraft block/{DONOR}/8-14 at 3 gate on gate_bias -1.0\n"
               f"graft block/{DONOR}/15-20 at 5 gate on gate_bias 0.5\nheal steps 100\n")
    assert "share gate_bias" in " ".join(check_text(root, unequal)["violations"])


def test_insert_layers_are_checked_against_both_models(root):
    deep = f"host {HOST}\ninsert {DONOR} layers 20..30 at 7\nlora rank 8\nheal steps 100\n"
    assert "the donor has layers 0 to 27" in " ".join(check_text(root, deep)["violations"])
    late = f"host {HOST}\ninsert {DONOR} layers 2..4 at 30\nlora rank 8\nheal steps 100\n"
    assert "has 24 layers" in " ".join(check_text(root, late)["violations"])


def test_distillation_needs_the_hosts_tokenizer_family(root):
    write_json(os.path.join(root, "profiles.json"), json.load(open(os.path.join(root, "profiles.json"))) + [
        {"key": "llama1b", "hf_id": "unsloth/Llama-3.2-1B-Instruct", "family": "llama", "params_b": 1.236,
         "layers": 16, "hidden": 2048, "tokenizer_family": "llama3", "scores": {}, "subject_scores": {}}])
    text = (f"host {HOST}\ngraft block/{DONOR}/8-14 at 7\nlora rank 8\n"
            "heal steps 100 distill unsloth/Llama-3.2-1B-Instruct\n")
    assert "another tokenizer" in " ".join(check_text(root, text)["violations"])
    unknown = text.replace("unsloth/Llama-3.2-1B-Instruct", "org/not-in-the-atlas")
    assert "cannot confirm" in " ".join(check_text(root, unknown)["violations"])
    # A distillation heal alone is a baseline.
    alone = f"host {HOST}\nlora rank 8\nheal steps 100 distill {DONOR}\n"
    assert "composes nothing" in " ".join(check_text(root, alone)["violations"])


def test_the_request_limits_become_the_programs_budget(root):
    calls = []
    toolbox = toolbox_for(root, fake_practice(calls), {"max_heal_minutes": 30})
    out = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert out["ok"] is True
    budget = calls[0][0]["budget"]
    assert budget == {"max_total_params_b": 2.0, "max_active_params_b": 2.0, "max_heal_minutes": 30}
    # A budget line in the program can only tighten the limits.
    tighter = GOOD_PROGRAM + "budget total 1.5b heal_minutes 60\n"
    toolbox.call("compile_and_practice", {"program_text": tighter})
    assert calls[1][0]["budget"] == {"max_total_params_b": 1.5, "max_active_params_b": 2.0, "max_heal_minutes": 30}


def test_measured_heal_time_is_checked_after_the_compile(root):
    result = {"ok": True, "model_dir": "/vol/x", "practice_sets": {"gsm8k_practice": 0.4}, "params_total_b": 1.0,
              "params_run_b": 1.0, "heal_seconds": 3600.0, "errors": []}
    toolbox = toolbox_for(root, fake_practice([], result), {"max_heal_minutes": 30})
    out = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert out["within_constraints"] is False
    assert "over the cap 30" in " ".join(out["violations"])
    assert toolbox.call("finish", {"program_text": GOOD_PROGRAM, "rationale": "x"})["ok"] is False
    stopped = dict(result)
    stopped["heal_stopped_at_limit"] = True
    ok_box = toolbox_for(root, fake_practice([], stopped), {"max_heal_minutes": 30})
    assert ok_box.call("compile_and_practice", {"program_text": GOOD_PROGRAM})["within_constraints"] is True
    missing = dict(result)
    del missing["heal_seconds"]
    box = toolbox_for(root, fake_practice([], missing), {"max_heal_minutes": 30})
    assert "did not report heal_seconds" in " ".join(
        box.call("compile_and_practice", {"program_text": GOOD_PROGRAM})["violations"])


def test_the_requests_base_is_the_only_host(root):
    toolbox = toolbox_for(root, fake_practice([]))
    toolbox.request["base"] = DONOR
    out = toolbox.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert out["stage"] == "constraints"
    assert "the request names the base" in " ".join(out["violations"])


def test_finish_refuses_a_control_program(root):
    toolbox = toolbox_for(root, fake_practice([]))
    control = GOOD_PROGRAM.replace("at 7 rank 64", "at 7 rank 64 control random")
    assert toolbox.call("compile_and_practice", {"program_text": control})["ok"] is True
    out = toolbox.call("finish", {"program_text": control, "rationale": "x"})
    assert out["ok"] is False
    assert "not a control" in out["error"]
    # The loop does not fall back to a control either.
    assert toolbox.good_runs() == []


def test_the_prompt_and_notes_describe_the_unified_language(root):
    request = load_request(os.path.join(root, "request.json"))
    text = system_prompt(request, 10, 100000)
    for word in ["experts", "insert", "distill", "control random", "ONE network", "taps", "far above the base"]:
        assert word in text
    notes = toolbox_for(root, fake_practice([])).call("language_reference", {})["text"]
    assert "tap_train" in notes
    assert chr(0x2014) not in text + notes


def test_request_options_are_validated(tmp_path):
    base = {"name": "r", "skills": {"math": 1.0}, "constraints": {"max_total_params_b": 1.0}}
    for change in [{"practice_items": 0}, {"heldout_items": "x"}, {"report": {"bogus": 1}},
                   {"constraints": {"max_heal_minutes": -1}}, {"base": 3}]:
        request = dict(base)
        request.update(change)
        write_json(str(tmp_path / "r.json"), request)
        with pytest.raises(ValueError):
            load_request(str(tmp_path / "r.json"))


def test_the_smoke_request():
    request = load_request(os.path.join(REPO, "composer", "requests", "smoke_qwen05.json"))
    assert request["base"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert list(request["skills"]) == ["math"]
    assert request["constraints"]["compile_budget"] == 3
    assert request["constraints"]["heal_steps_cap"] == 200
    assert request["practice_items"] <= 50


# The comparison set.

COMPARE_REQUEST = {"name": "math_test", "skills": {"math": 1.0},
                   "constraints": {"max_total_params_b": 2.0, "max_run_params_b": 2.0, "heal_steps_cap": 500,
                                   "max_heal_minutes": 30},
                   "report": {"control_seeds": 2}}


def compare_program():
    from composer import sizes as size_module

    program = parse(f"host {HOST}\ngraft block/{DONOR}/8-14 at 7 rank 32 stitch mlp objective donor_lm steps 100\n"
                    "lora rank 8\nheal steps 150 data gsm8k_train lr 1e-4\n")
    size_module.with_request_budget(program, COMPARE_REQUEST["constraints"])
    return program.to_dict()


def test_arms_share_the_heal_data_and_budget(root):
    from composer import arms

    models = Sources(root).models
    program = compare_program()
    planned, skipped = arms.plan_arms(program, COMPARE_REQUEST, models, {})
    roles = []
    for arm in planned:
        roles.append(arm["role"])
    assert roles == ["lora_on_base", "distill_on_base", "random_control", "random_control"]
    assert skipped == []
    lora, distill, control_a, control_b = planned
    # LoRA: the host alone, the program's heal data and settings, all its training steps (stitch 100 + heal 150).
    assert lora["program"]["grafts"] == []
    assert lora["program"]["heal"] == {"steps": 250, "data": "gsm8k_train", "lr": 1e-4, "batch_size": 8,
                                       "seq_len": 384}
    assert lora["program"]["lora"]["rank"] == 8
    assert lora["program"]["budget"]["max_heal_minutes"] == 30
    assert lora["program"]["budget"]["max_total_params_b"] is None
    # Distillation: the same heal plus a teacher from the source with the host's tokenizer.
    assert distill["teacher"] == DONOR
    assert distill["program"]["heal"]["distill"] == DONOR
    assert distill["program"]["heal"]["steps"] == 250
    assert distill["program"]["heal"]["data"] == "gsm8k_train"
    # Control: the same program with control random on every component, other seeds, the same heal and budget.
    assert control_a["program"]["grafts"][0]["control"] == "random"
    assert control_a["program"]["grafts"][0]["control_seed"] != control_b["program"]["grafts"][0]["control_seed"]
    assert control_a["program"]["heal"] == program["heal"]
    assert control_a["program"]["budget"] == program["budget"]
    assert control_a["program"]["seed"] == program["seed"]


def test_the_lora_arm_does_not_distill_and_a_programs_own_teacher_carries_over(root):
    from composer import arms

    program = Program.from_dict(parse(DISTILL_PROGRAM).to_dict()).to_dict()
    lora = arms.lora_program(program, {}, ["math"])
    assert "distill" not in lora["heal"]
    assert "kd_weight" not in lora["heal"]
    distill = arms.distill_program(program, {}, DONOR, ["math"])
    assert distill["heal"]["kd_weight"] == 0.5
    other = arms.distill_program(program, {}, "Qwen/Qwen2.5-7B-Instruct", ["math"])
    assert other["heal"]["kd_weight"] == 1.0


def test_distill_arms_need_matching_tokenizers_and_a_source(root):
    from composer import arms

    models = dict(Sources(root).models)
    models["unsloth/Llama-3.2-1B-Instruct"] = {"params_b": 1.2, "layers": 16, "hidden": 2048,
                                                "tokenizer_family": "llama3"}
    program = parse(f"host {HOST}\ninsert unsloth/Llama-3.2-1B-Instruct layers 2..5 at 7 stitch mlp\n"
                    f"lora rank 8\nheal steps 100\n").to_dict()
    planned, skipped = arms.plan_arms(program, COMPARE_REQUEST, models, {})
    assert [arm["role"] for arm in planned].count("distill_on_base") == 0
    assert skipped[0]["reason"] == "the tokenizers differ"
    unknown = parse(f"host {HOST}\ninsert org/mystery layers 2..5 at 7 stitch mlp\nlora rank 8\nheal steps 100\n")
    planned, skipped = arms.plan_arms(unknown.to_dict(), COMPARE_REQUEST, models, {})
    assert "unknown" in skipped[0]["reason"]
    # A program with only masks has no source and no random control.
    masked = parse(f"host {HOST}\nmask mask_law\n").to_dict()
    planned, skipped = arms.plan_arms(masked, COMPARE_REQUEST, models, {})
    reasons = " ".join(item["reason"] for item in skipped)
    assert "no source model" in reasons
    assert "no random control" in reasons or "no graft" in reasons or "masks" in reasons
    assert [arm["role"] for arm in planned] == ["lora_on_base"]


def test_experts_and_inserts_get_a_random_control(root):
    from composer import arms

    for text in [EXPERTS_PROGRAM, INSERT_PROGRAM]:
        program = parse(text).to_dict()
        planned, skipped = arms.plan_arms(program, COMPARE_REQUEST, Sources(root).models, {})
        controls = [arm for arm in planned if arm["role"] == "random_control"]
        assert len(controls) == 2
        key = "experts" if "experts" in program else "inserts"
        value = controls[0]["program"][key]
        if isinstance(value, list):
            value = value[0]
        assert value["control"] == "random"
    assert arms.source_models(parse(EXPERTS_PROGRAM).to_dict(), {}) == [HOST, DONOR]
    assert arms.source_models(parse(INSERT_PROGRAM).to_dict(), {}) == [HOST, DONOR]


def test_best_existing_model_only_when_the_limit_is_far_above_the_base(root):
    from composer import arms

    sources = Sources(root)
    best, reason = arms.best_existing_model(COMPARE_REQUEST, sources, HOST)
    # Atlas entry math 0.61 beats the profile's gsm8k 0.32; the 7B model is over the cap.
    assert best["model"] == DONOR
    assert best["score"] == pytest.approx(0.61)
    assert best["basis"] == {"math": "atlas"}
    near = dict(COMPARE_REQUEST)
    near["constraints"] = {"max_total_params_b": 0.8}
    best, reason = arms.best_existing_model(near, sources, HOST)
    assert best is None
    assert "not a fair rival" in reason
    wide = dict(COMPARE_REQUEST)
    wide["constraints"] = {"max_total_params_b": 9.0, "max_run_params_b": 2.0}
    best, reason = arms.best_existing_model(wide, sources, HOST)
    assert best["model"] == DONOR


def row(score, items=1000, name="gsm8k_test"):
    return {"objective": score, "skills": {"math": score}, "sets": {name: {"score": score, "items": items}}}


def test_verdicts_compare_against_noise():
    from composer import arms

    weights = {"math": 1.0}
    big = arms.judge("base", row(0.60), [row(0.40)], weights)
    assert big["verdict"] == "beat"
    assert big["diff"] == pytest.approx(0.2)
    assert "beat it by more than noise" in big["text"]
    # 1000 items: the item error of the difference is about 0.022, so the threshold is about 0.044.
    close = arms.judge("base", row(0.60), [row(0.58)], weights)
    assert close["verdict"] == "tied"
    assert "did not beat it by more than noise" in close["text"]
    assert close["threshold"] > 0.04
    lost = arms.judge("base", row(0.40), [row(0.60)], weights)
    assert lost["verdict"] == "lost"
    # Few items make the threshold wide: 20 items cannot show a 0.1 gap.
    few = arms.judge("base", row(0.60, 20), [row(0.50, 20)], weights)
    assert few["verdict"] == "tied"
    # The spread of the control seeds sets the floor: two controls 0.15 apart.
    spread = arms.judge("random", row(0.70, 100000), [row(0.50, 100000), row(0.65, 100000)], weights)
    assert spread["seed_spread"] == pytest.approx(0.15)
    assert spread["verdict"] == "tied"
    none = arms.judge("base", row(0.6), [{"error": "x"}], weights)
    assert none["verdict"] == "not scored"


def test_run_comparison_end_to_end_with_fakes(root):
    from composer.comparison import run_comparison

    sources = Sources(root)
    program = compare_program()
    final = {"program_text": "t", "program": program, "program_hash": "abc123", "rationale": "r",
             "model_dir": "/vol/composer/math_test/abc123", "fallback": False, "no_model": False,
             "practice": {"params_total_b": 1.1}}
    summary = {"final": final, "compile_budget_used": 3, "compile_budget": 5, "turns": 9, "tokens_used": 5,
               "stop_reason": "finish", "archive_dir": "x"}
    compile_calls = []
    score_calls = []
    level = {"final": 0.62, "base": 0.30, "lora": 0.33, "distill": 0.35, "control": 0.31, "best": 0.50,
             "source": 0.45}

    def compile_many(calls):
        compile_calls.extend(calls)
        out = []
        for index in range(len(calls)):
            out.append({"ok": True, "model_dir": f"/vol/arm{index}", "seconds": 1.0, "heal_steps_run": 250,
                        "heal_seconds": 5.0})
        out[1] = RuntimeError("lost container")
        return out

    def score_many(calls):
        score_calls.extend(calls)
        out = []
        for ref, payload in calls:
            if ref == "/vol/composer/math_test/abc123":
                value = level["final"]
            elif ref == HOST:
                value = level["base"]
            elif ref == DONOR:
                value = level["best"]
            elif ref == "/vol/arm0":
                value = level["lora"]
            else:
                value = level["control"]
            out.append({"skills": {"math": value}, "params_total_b": 1.1,
                        "sets": {"gsm8k_test": {"score": value, "items": 1319},
                                 "math500": {"score": value, "items": 500}}})
        return out

    results = run_comparison(COMPARE_REQUEST, summary, sources, compile_many, score_many)

    # One compile per arm; every arm program has the same heal data.
    assert len(compile_calls) == 4
    for arm_program, payload in compile_calls:
        assert arm_program["heal"]["data"] == "gsm8k_train"
        assert payload["program_hash"] == "abc123"
    # The donor is a source, a distillation teacher, and the best existing model: scored once.
    refs = [ref for ref, payload in score_calls]
    assert refs.count(DONOR) == 1
    assert refs[:2] == ["/vol/composer/math_test/abc123", HOST]
    assert "/vol/arm1" not in refs
    comparisons = results["comparisons"]
    assert comparisons["base"]["objective"] == pytest.approx(0.30)
    assert comparisons["lora_on_base"]["objective"] == pytest.approx(0.33)
    assert comparisons["best_existing"]["objective"] == pytest.approx(0.50)
    assert comparisons["best_existing"]["model"] == DONOR
    assert len(comparisons["random_control"]) == 2
    assert comparisons["distill_on_base"][0]["error"].startswith("compile failed")
    by_arm = {}
    for item in results["verdicts"]:
        by_arm[item["arm"]] = item
    assert by_arm["base"]["verdict"] == "beat"
    assert by_arm["base + LoRA"]["verdict"] == "beat"
    assert by_arm["random components of the same size"]["verdict"] == "beat"
    assert by_arm["best existing model"]["verdict"] == "beat"
    assert by_arm["base + distill from " + DONOR]["verdict"] == "not scored"
    assert results["same_heal"]["data"] == "gsm8k_train"
    assert results["same_heal"]["steps"] == 250
    assert json.dumps(results)


def test_run_comparison_says_when_the_program_does_not_beat_an_arm(root):
    from composer.comparison import run_comparison

    program = compare_program()
    final = {"program_text": "t", "program": program, "program_hash": "abc", "model_dir": "/vol/f",
             "no_model": False, "practice": {}}
    summary = {"final": final}

    def compile_many(calls):
        return [{"ok": True, "model_dir": f"/vol/a{index}"} for index in range(len(calls))]

    def score_many(calls):
        out = []
        for ref, payload in calls:
            value = 0.40
            out.append({"skills": {"math": value}, "sets": {"gsm8k_test": {"score": value, "items": 1319},
                                                             "math500": {"score": value, "items": 500}}})
        return out

    results = run_comparison(COMPARE_REQUEST, summary, Sources(root), compile_many, score_many)
    verdicts = {item["arm"]: item["verdict"] for item in results["verdicts"]}
    assert set(verdicts.values()) == {"tied"}
    assert any("did not beat it by more than noise" in item["text"] for item in results["verdicts"])


def test_no_model_means_no_comparison(root):
    from composer.comparison import run_comparison

    summary = {"final": {"no_model": True, "program": None}}
    results = run_comparison(COMPARE_REQUEST, summary, Sources(root), lambda calls: 1 / 0, lambda calls: 1 / 0)
    assert results["final"] is None
    assert results["verdicts"] == []


# Practice, heal, and held-out items stay apart, and held-out scores stay hidden.

def test_practice_heal_and_heldout_sets_are_disjoint():
    from composer.skills import BUILTIN_HEAL_DATA, COMPOSER_HEAL_DATA, SKILLS, heldout_sets, practice_sets

    for skill in SKILLS:
        assert set(SKILLS[skill]["practice"]).isdisjoint(SKILLS[skill]["heldout"])
        for heal_name in SKILLS[skill]["heal_data"]:
            assert heal_name in BUILTIN_HEAL_DATA + COMPOSER_HEAL_DATA
            assert heal_name not in practice_sets(list(SKILLS)) + heldout_sets(list(SKILLS))
    assert set(practice_sets(list(SKILLS))).isdisjoint(heldout_sets(list(SKILLS)))
    # The train pairs skip the practice items, and the item cap keeps a subset of one set.
    from composer.evalsets import SIZES, limit_items

    assert SIZES["math_practice"] == 100
    items = list(range(100))
    picked = limit_items(items, 30)
    assert len(picked) == 30
    assert set(picked) <= set(items)
    assert picked == sorted(picked)
    assert picked == limit_items(items, 30)
    assert limit_items(items, None) is items
    assert limit_items(items, 500) is items


def test_arm_heal_data_comes_from_the_programs_heal_not_the_heldout_sets(root):
    from composer import arms
    from composer.skills import BUILTIN_HEAL_DATA, COMPOSER_HEAL_DATA

    for skills in [["math"], ["math", "code"], ["law"]]:
        masked = parse(f"host {HOST}\nmask mask_law\n").to_dict()
        data = arms.lora_program(masked, {}, skills)["heal"]["data"]
        assert data in BUILTIN_HEAL_DATA + COMPOSER_HEAL_DATA


def test_comparison_scores_never_reach_the_model(root):
    from composer import arms

    # Only the report uses the earlier scores that pick the best existing model. The tools do not call it.
    box = toolbox_for(root, fake_practice([]))
    atlas = json.dumps(box.call("atlas", {"skill": "math"}))
    assert "0.61" not in atlas
    assert "best_existing" not in atlas
    names = [tool["name"] for tool in tool_specs()]
    assert names == ["atlas", "catalog", "language_reference", "compile_and_practice", "finish"]
    assert callable(arms.best_existing_model)
    # Practice results carry no held-out numbers or comparison verdicts.
    out = box.call("compile_and_practice", {"program_text": GOOD_PROGRAM})
    assert "verdict" not in json.dumps(out)
    assert "gsm8k_test" not in json.dumps(out)


def test_modal_composer_is_valid_python():
    import ast

    with open(os.path.join(REPO, "modal_composer.py")) as f:
        tree = ast.parse(f.read())
    names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    for name in ["compile_and_practice", "compile_arm", "score_heldout", "main"]:
        assert name in names
    assert "compile_lora_baseline" not in names


# Verifier regressions.

def test_arm_payload_carries_block_records_for_the_control_arm(root):
    from composer import arms
    from composer.comparison import arm_payload

    sources = Sources(root)
    program = compare_program()
    planned, skipped = arms.plan_arms(program, COMPARE_REQUEST, sources.models, sources.records_by_id())
    final = {"program_hash": "abc"}
    control = [arm for arm in planned if arm["role"] == "random_control"][0]
    payload = arm_payload(COMPARE_REQUEST, final, control, sources.records_by_id())
    # The graft is a block id with no catalog row. The arm compile needs its record built from the id.
    assert [row["id"] for row in payload["catalog"]] == [f"block/{DONOR}/8-14"]
    assert payload["catalog"][0]["layers"] == [8, 14]
    assert payload["catalog"][0]["model"] == DONOR
    lora = [arm for arm in planned if arm["role"] == "lora_on_base"][0]
    assert arm_payload(COMPARE_REQUEST, final, lora, sources.records_by_id())["catalog"] == []


def test_the_pre_check_refuses_what_the_compiler_refuses(root):
    cases = [
        (f"host {HOST}\nmask mask_law\nexperts {DONOR}\nlora rank 8\nheal steps 100\n",
         "masks and task vectors cannot join experts or inserts"),
        (f"host {HOST}\nexperts {DONOR}\nlora rank 8 layers 0,1\nheal steps 100\n", "drop the lora layers"),
        (f"host {HOST}\nmask mask_law\nheal steps 100\n", "nothing to train"),
        (f"host {HOST}\ngraft block/{DONOR}/8-14 at 7\nlora rank 8\n", "needs a heal statement"),
        (f"from {DONOR}\ncarve target 1.2b select influence\ngraft block/{DONOR}/8-14 at 7\nlora rank 8\n"
         "heal steps 100\n", "cannot have grafts"),
    ]
    for text, message in cases:
        program, error = toolbox_for(root, fake_practice([])).parse(text)
        assert error is None, (text, error)
        result = sizes.check_program(program, {}, Sources(root).models, Sources(root).records_by_id())
        assert message in " ".join(result["violations"]), (text, result["violations"])
    # The legal forms stay legal.
    for text in [EXPERTS_PROGRAM, INSERT_PROGRAM, GATED_PROGRAM, DISTILL_PROGRAM]:
        assert check_text(root, text)["ok"], text


def test_a_graft_estimate_counts_its_glue(root):
    # w_in and w_out of a 0.5B host (d 896) and a 1.5B donor (d 1536), full rank: 2 * 896 * 1536 weights.
    plain = parse(f"host {HOST}\ngraft block/{DONOR}/8-14 at 7\nheal steps 100\n")
    ranked = parse(f"host {HOST}\ngraft block/{DONOR}/8-14 at 7 rank 8\nheal steps 100\n")
    models = Sources(root).models
    full = sizes.estimate_program(plain, models, {}).total
    low = sizes.estimate_program(ranked, models, {}).total
    layers = 7 * (1.544 - 151936 * 1536 / 1e9) / 28
    assert full == pytest.approx(0.494 + layers + 2 * 896 * 1536 / 1e9, abs=1e-6)
    assert low < full


def test_a_row_with_a_missing_held_out_set_has_no_objective():
    request = {"name": "r", "skills": {"math": 1.0}, "constraints": {}}
    summary = {"final": {"program": None}}
    whole = {"skills": {"math": 0.5}, "sets": {"gsm8k_test": {"score": 0.5, "items": 100},
                                                 "math500": {"score": 0.5, "items": 100}}}
    half = {"skills": {"math": 0.9}, "sets": {"gsm8k_test": {"score": 0.9, "items": 100}}, "errors": ["x"]}
    none = {"skills": {"math": None}, "sets": {}, "errors": ["x", "y"]}
    out = build_results(request, summary, [("final", "/vol/f", whole), ("base", HOST, half),
                                           ("lora_on_base", "/vol/l", none)])
    assert out["comparisons"]["base"]["objective"] is None
    assert out["comparisons"]["base"]["missing_sets"] == ["math500"]
    assert out["comparisons"]["lora_on_base"]["objective"] is None
    verdicts = {item["arm"]: item["verdict"] for item in out["verdicts"]}
    # A half-scored arm and an unscored arm are not beaten.
    assert verdicts["base"] == "not scored"
    assert verdicts["base + LoRA"] == "not scored"


def test_humaneval_uses_the_canonical_dataset_id():
    with open(os.path.join(REPO, "composer", "evalsets.py")) as f:
        text = f.read()
    assert 'load("openai/openai_humaneval"' in text
    assert 'load("openai_humaneval"' not in text
