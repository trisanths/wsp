"""Tests for wsp/swarm/llm.py and modal_llm.py. CPU only, no network: the openai,
anthropic and Modal clients are replaced with fakes."""

import json
import types

import httpx2
import modal
import openai
import pytest

from wsp.swarm import factory
from wsp.swarm import llm as llm_mod
from wsp.swarm.llm import (AnthropicLLM, LLMError, LLMJSONError, MockLLM, ModalVLLM, OpenAICompatLLM,
                           check_json, split_system, split_type_lists)

SCHEMA = {"type": "object", "properties": {"part": {"type": "string"}}, "required": ["part"]}
CHAT = [{"role": "system", "content": "You plan grafts."}, {"role": "user", "content": "Propose one."}]


def fast(config=None):
    """Config with no backoff wait, so retry tests run instantly."""
    values = {"llm_backoff_s": 0.0}
    if config is not None:
        for key in config:
            values[key] = config[key]
    return values


def user(text):
    return [{"role": "user", "content": text}]


# Fakes for the openai client.

def fake_request():
    return httpx2.Request("POST", "http://test.invalid/v1/chat/completions")


def openai_status_error(status_code, error_class):
    response = httpx2.Response(status_code, request=fake_request())
    return error_class("status " + str(status_code), response=response, body=None)


def openai_reply(text, prompt_tokens=11, completion_tokens=7):
    message = types.SimpleNamespace(content=text)
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)


class FakeCompletions:
    """Plays a script: each create() pops the next item, raising it if it is an exception."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return openai_reply(item)


def install_fake_openai(monkeypatch, script):
    completions = FakeCompletions(script)
    made = []

    def fake_client(**kwargs):
        made.append(kwargs)
        return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    return completions, made


def openai_llm(monkeypatch, script, config=None):
    monkeypatch.setenv("WSP_TEST_LLM_KEY", "placeholder-not-a-key")
    values = {"openai_model": "test-model", "openai_base_url": "http://test.invalid/v1",
              "openai_api_key_env": "WSP_TEST_LLM_KEY"}
    if config is not None:
        for key in config:
            values[key] = config[key]
    completions, made = install_fake_openai(monkeypatch, script)
    return OpenAICompatLLM(fast(values)), completions, made


# Fakes for Modal.

class FakeGenerate:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if item == "echo":
            replies = []
            for messages in args[0]:
                replies.append(json.dumps({"part": messages[-1]["content"]}))
            return replies
        return item


def install_fake_modal(monkeypatch, script):
    generate = FakeGenerate(script)
    lookups = []

    def fake_from_name(app_name, class_name, environment_name=None):
        lookups.append((app_name, class_name, environment_name))

        def make_instance():
            return types.SimpleNamespace(generate=generate)

        return make_instance

    monkeypatch.setattr(modal.Cls, "from_name", fake_from_name)
    return generate, lookups


# MockLLM

def test_mock_cycles_replies_deterministically():
    config = {"mock_replies": ["a", "b", "c"]}
    first = MockLLM(config)
    second = MockLLM(config)
    seen_first = []
    seen_second = []
    for index in range(5):
        seen_first.append(first.complete(user("q" + str(index))))
        seen_second.append(second.complete(user("other prompt")))
    assert seen_first == ["a", "b", "c", "a", "b"]
    assert seen_second == seen_first
    assert first.complete_many([user("x"), user("y")], seed=7) == ["c", "a"]
    assert first.calls[-2]["seed"] == 7
    assert first.calls[-1]["seed"] == 8
    assert len(first.calls) == 7


def test_mock_fn_and_missing_config():
    llm = MockLLM({"mock_fn": lambda messages: "echo " + messages[-1]["content"]})
    assert llm.complete(user("hi")) == "echo hi"
    assert llm.complete_many([user("a"), user("b")]) == ["echo a", "echo b"]
    with pytest.raises(LLMError):
        MockLLM({})


def test_factory_builds_mock():
    llm_class = factory.load_class("llm", "mock")
    assert llm_class is MockLLM
    assert llm_class({"mock_replies": ["ok"]}).complete(user("hi")) == "ok"


# JSON check

def test_schema_json_check():
    good = json.dumps({"part": "qmath_l14_c3"})
    assert json.loads(MockLLM({"mock_replies": [good]}).complete(user("go"), schema=SCHEMA))["part"] == "qmath_l14_c3"
    with pytest.raises(LLMJSONError) as caught:
        MockLLM({"mock_replies": ["not json"]}).complete(user("go"), schema=SCHEMA)
    assert caught.value.texts == ["not json"]
    assert caught.value.bad == [0]
    # Without a schema any text is fine.
    assert MockLLM({"mock_replies": ["not json"]}).complete(user("go")) == "not json"
    # complete_many reports every bad index and keeps the good replies.
    with pytest.raises(LLMJSONError) as caught:
        MockLLM({"mock_replies": [good, "{oops", good]}).complete_many([user("1"), user("2"), user("3")],
                                                                      schema=SCHEMA)
    assert caught.value.bad == [1]
    assert caught.value.texts[0] == good
    assert isinstance(caught.value, LLMError)
    check_json(["anything"], None, "test")


def test_openai_schema_request_and_usage(monkeypatch):
    llm, completions, made = openai_llm(monkeypatch, [json.dumps({"part": "p1"})])
    text = llm.complete(CHAT, schema=SCHEMA, temperature=0.2, seed=3, max_tokens=64)
    assert json.loads(text) == {"part": "p1"}
    sent = completions.calls[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
    assert sent["seed"] == 3
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 64
    assert made[0]["max_retries"] == 0
    assert made[0]["base_url"] == "http://test.invalid/v1"
    assert llm.usage[0]["input_tokens"] == 11
    assert llm.usage[0]["output_tokens"] == 7
    assert llm.usage[0]["backend"] == "openai_compat"


def test_openai_falls_back_to_json_object(monkeypatch):
    refused = openai_status_error(400, openai.BadRequestError)
    llm, completions, made = openai_llm(monkeypatch, [refused, '{"part": "p2"}', '{"part": "p3"}'])
    assert json.loads(llm.complete(CHAT, schema=SCHEMA)) == {"part": "p2"}
    assert completions.calls[0]["response_format"]["type"] == "json_schema"
    assert completions.calls[1]["response_format"] == {"type": "json_object"}
    system_prompt = completions.calls[1]["messages"][0]
    assert system_prompt["role"] == "system"
    assert "You plan grafts." in system_prompt["content"]
    assert "JSON schema" in system_prompt["content"]
    # The fallback sticks: the next call goes straight to json_object.
    llm.complete(CHAT, schema=SCHEMA)
    assert len(completions.calls) == 3
    assert completions.calls[2]["response_format"] == {"type": "json_object"}


def test_openai_invalid_json_raises(monkeypatch):
    llm, completions, made = openai_llm(monkeypatch, ["Sure! Here is a plan."])
    with pytest.raises(LLMJSONError):
        llm.complete(CHAT, schema=SCHEMA)


def test_openai_needs_model_and_key(monkeypatch):
    install_fake_openai(monkeypatch, [])
    with pytest.raises(LLMError):
        OpenAICompatLLM({})
    monkeypatch.delenv("WSP_TEST_MISSING_KEY", raising=False)
    with pytest.raises(LLMError):
        OpenAICompatLLM({"openai_model": "m", "openai_api_key_env": "WSP_TEST_MISSING_KEY"})


# Retries

def test_retry_then_llm_error(monkeypatch):
    errors = []
    for index in range(3):
        errors.append(openai.APIConnectionError(request=fake_request()))
    llm, completions, made = openai_llm(monkeypatch, errors)
    with pytest.raises(LLMError) as caught:
        llm.complete(CHAT)
    assert "after 3 tries" in str(caught.value)
    assert len(completions.calls) == 3


def test_retry_recovers_from_transient_errors(monkeypatch):
    script = [openai_status_error(429, openai.RateLimitError), openai_status_error(503, openai.InternalServerError),
              "fine"]
    llm, completions, made = openai_llm(monkeypatch, script)
    assert llm.complete(CHAT) == "fine"
    assert len(completions.calls) == 3


def test_non_transient_error_is_not_retried(monkeypatch):
    llm, completions, made = openai_llm(monkeypatch, [openai_status_error(401, openai.AuthenticationError)])
    with pytest.raises(LLMError):
        llm.complete(CHAT)
    assert len(completions.calls) == 1


def test_retry_count_and_backoff_follow_config(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_mod.time, "sleep", waits.append)
    llm = MockLLM({"mock_replies": [TimeoutError("slow"), TimeoutError("slow"), TimeoutError("slow"), "late"],
                   "llm_tries": 2, "llm_backoff_s": 0.5})
    with pytest.raises(LLMError):
        llm.complete(user("hi"))
    assert waits == [0.5]
    assert len(llm.calls) == 2


def test_openai_complete_many_keeps_order(monkeypatch):
    llm, completions, made = openai_llm(monkeypatch, ["r0", "r1", "r2"], {"llm_workers": 1})
    assert llm.complete_many([user("a"), user("b"), user("c")], seed=10) == ["r0", "r1", "r2"]
    seeds = []
    for call in completions.calls:
        seeds.append(call["seed"])
    assert seeds == [10, 11, 12]


# ModalVLLM

def test_modal_complete_many_is_one_batched_call(monkeypatch):
    generate, lookups = install_fake_modal(monkeypatch, ["echo"])
    llm = ModalVLLM(fast())
    batch = [user("p0"), user("p1"), user("p2")]
    texts = llm.complete_many(batch, schema=SCHEMA, temperature=0.5, seed=4, max_tokens=99)
    assert len(generate.calls) == 1
    assert generate.calls[0] == (batch, SCHEMA, 0.5, 4, 99)
    assert json.loads(texts[2]) == {"part": "p2"}
    assert lookups == [("wsp-llm", "PlannerLLM", None)]
    assert len(llm.usage) == 1
    assert llm.usage[0]["input_tokens"] is None


def test_modal_complete_is_a_batch_of_one(monkeypatch):
    generate, lookups = install_fake_modal(monkeypatch, [["hello"]])
    llm = ModalVLLM(fast({"modal_llm_environment": "dev"}))
    assert llm.complete(user("hi"), seed=1) == "hello"
    assert generate.calls[0][0] == [user("hi")]
    assert lookups[0][2] == "dev"


def test_modal_retries_then_raises(monkeypatch):
    errors = []
    for index in range(3):
        errors.append(modal.exception.ConnectionError("down"))
    generate, lookups = install_fake_modal(monkeypatch, errors)
    llm = ModalVLLM(fast())
    with pytest.raises(LLMError):
        llm.complete_many([user("a"), user("b")])
    assert len(generate.calls) == 3


def test_modal_not_deployed_says_how_to_fix(monkeypatch):
    generate, lookups = install_fake_modal(monkeypatch, [modal.exception.NotFoundError("no app")])
    llm = ModalVLLM(fast())
    with pytest.raises(LLMError) as caught:
        llm.complete(user("hi"))
    assert "modal deploy modal_llm.py" in str(caught.value)
    assert len(generate.calls) == 1


def test_modal_reply_count_mismatch(monkeypatch):
    generate, lookups = install_fake_modal(monkeypatch, [["only one"]])
    llm = ModalVLLM(fast())
    with pytest.raises(LLMError):
        llm.complete_many([user("a"), user("b")])


# AnthropicLLM

def test_anthropic_without_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError) as caught:
        AnthropicLLM({})
    assert "ANTHROPIC_API_KEY" in str(caught.value)


class FakeAnthropicMessages:
    def __init__(self, stop_reason="end_turn", text='{"part": "p9"}'):
        self.calls = []
        self.stop_reason = stop_reason
        self.text = text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text=self.text)
        usage = types.SimpleNamespace(input_tokens=20, output_tokens=5)
        return types.SimpleNamespace(content=[block], usage=usage, stop_reason=self.stop_reason,
                                     model=kwargs["model"])


def anthropic_llm(monkeypatch, messages_api, config=None):
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    made = []

    def fake_client(**kwargs):
        made.append(kwargs)
        return types.SimpleNamespace(messages=messages_api, beta=types.SimpleNamespace(messages=messages_api))

    monkeypatch.setattr(anthropic, "Anthropic", fake_client)
    return AnthropicLLM(fast(config)), made


def test_anthropic_request_shape(monkeypatch):
    messages_api = FakeAnthropicMessages()
    llm, made = anthropic_llm(monkeypatch, messages_api)
    text = llm.complete(CHAT, schema=SCHEMA, temperature=0.9, seed=5, max_tokens=256)
    assert json.loads(text) == {"part": "p9"}
    sent = messages_api.calls[0]
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["system"] == "You plan grafts."
    assert sent["messages"] == [{"role": "user", "content": "Propose one."}]
    strict = {"type": "object", "properties": {"part": {"type": "string"}}, "required": ["part"],
              "additionalProperties": False}
    assert sent["output_config"]["format"] == {"type": "json_schema", "schema": strict}
    assert "additionalProperties" not in SCHEMA
    assert sent["max_tokens"] == 4096
    assert sent["fallbacks"] == "default"
    assert "temperature" not in sent
    assert "seed" not in sent
    assert made[0]["max_retries"] == 0
    assert llm.usage[0]["input_tokens"] == 20


def test_anthropic_schema_made_strict(monkeypatch):
    """Unsupported array limits move into the description and every object gets additionalProperties false."""
    schema = {
        "type": "object",
        "required": ["proposals"],
        "properties": {
            "proposals": {"type": "array", "minItems": 2, "maxItems": 5,
                          "items": {"type": "object", "properties": {"hypothesis": {"type": "string"}}}},
        },
    }
    original = json.dumps(schema, sort_keys=True)
    messages_api = FakeAnthropicMessages()
    llm, made = anthropic_llm(monkeypatch, messages_api)
    llm.complete(CHAT, schema=schema)
    sent = messages_api.calls[0]["output_config"]["format"]["schema"]
    assert sent["additionalProperties"] is False
    proposals = sent["properties"]["proposals"]
    assert proposals["items"]["additionalProperties"] is False
    assert "maxItems" not in proposals
    assert "minItems" not in proposals
    assert "maxItems: 5" in proposals["description"]
    assert "minItems: 2" in proposals["description"]
    assert json.dumps(schema, sort_keys=True) == original


def test_anthropic_schema_keeps_min_items_one(monkeypatch):
    messages_api = FakeAnthropicMessages()
    llm, made = anthropic_llm(monkeypatch, messages_api)
    llm.complete(CHAT, schema={"type": "object", "properties": {
        "grafts": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string"}}}})
    grafts = messages_api.calls[0]["output_config"]["format"]["schema"]["properties"]["grafts"]
    assert grafts["minItems"] == 1
    assert "maxItems: 4" in grafts["description"]


def test_anthropic_schema_type_list_becomes_any_of(monkeypatch):
    """A "type": [..., "null"] list (used by prompts.program_schema) is sent as an anyOf."""
    messages_api = FakeAnthropicMessages()
    llm, made = anthropic_llm(monkeypatch, messages_api)
    schema = {"type": "object", "properties": {
        "at": {"type": ["integer", "null"], "minimum": 0, "maximum": 15},
        "heal": {"type": ["object", "null"], "properties": {"steps": {"enum": [0, 100]}}},
    }}
    llm.complete(CHAT, schema=schema)
    sent = messages_api.calls[0]["output_config"]["format"]["schema"]
    at = sent["properties"]["at"]["anyOf"]
    assert at[0]["type"] == "integer"
    assert "maximum: 15" in at[0]["description"]
    assert at[1] == {"type": "null"}
    heal = sent["properties"]["heal"]["anyOf"]
    assert heal[0]["type"] == "object"
    assert heal[0]["additionalProperties"] is False
    assert heal[0]["properties"]["steps"] == {"enum": [0, 100]}
    assert heal[1] == {"type": "null"}
    assert schema["properties"]["at"]["type"] == ["integer", "null"]


def test_anthropic_schema_open_map_becomes_empty_object(monkeypatch):
    """Strict mode cannot express an open-keyed map; the SDK transform leaves an object with no keys."""
    messages_api = FakeAnthropicMessages()
    llm, made = anthropic_llm(monkeypatch, messages_api)
    llm.complete(CHAT, schema={"type": "object", "properties": {
        "placements": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "integer"}}}}})
    placements = messages_api.calls[0]["output_config"]["format"]["schema"]["properties"]["placements"]
    assert placements == {"type": "object", "properties": {}, "additionalProperties": False}


def test_anthropic_unconvertible_schema_raises_before_any_call(monkeypatch):
    messages_api = FakeAnthropicMessages()
    llm, made = anthropic_llm(monkeypatch, messages_api)
    with pytest.raises(LLMError) as caught:
        llm.complete(CHAT, schema={"properties": {"part": {"type": "string"}}})
    assert "output_config.format" in str(caught.value)
    assert messages_api.calls == []


def test_split_type_lists():
    schema = {"description": "layer", "type": ["integer", "null"], "minimum": 0,
              "items": {"type": ["string"]}}
    result = split_type_lists(schema)
    assert result == {"description": "layer", "anyOf": [
        {"type": "integer", "minimum": 0, "items": {"type": "string"}}, {"type": "null"}]}
    assert split_type_lists({"enum": [{"type": ["a", "b"]}]}) == {"enum": [{"type": ["a", "b"]}]}
    assert split_type_lists({"type": "string"}) == {"type": "string"}


def test_anthropic_refusal_raises(monkeypatch):
    llm, made = anthropic_llm(monkeypatch, FakeAnthropicMessages(stop_reason="refusal", text=""))
    with pytest.raises(LLMError):
        llm.complete(CHAT)


def test_split_system():
    system_text, chat = split_system([{"role": "system", "content": "a"}, {"role": "user", "content": "b"},
                                      {"role": "system", "content": "c"}])
    assert system_text == "a\n\nc"
    assert chat == [{"role": "user", "content": "b"}]


# modal_llm.py (imports without a GPU or network; vLLM is only imported inside the container)

def test_modal_llm_module_shape():
    import modal_llm

    assert modal_llm.APP_NAME == "wsp-llm"
    assert modal_llm.app.name == "wsp-llm"
    assert hasattr(modal_llm, "PlannerLLM")
    assert modal_llm.DEFAULT_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    with open(modal_llm.__file__, encoding="utf-8") as handle:
        source = handle.read()
    for word in ["web_endpoint", "fastapi_endpoint", "asgi_app", "wsgi_app", "web_server", "app.server",
                 "min_containers="]:
        assert word not in source
    assert "max_containers=1" in source
    assert "scaledown_window=300" in source
