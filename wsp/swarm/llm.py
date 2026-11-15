"""LLM clients for the swarm's planners and analyst (spec 9.4).

Four clients share the interfaces.LLM contract:
- MockLLM: scripted, deterministic replies for tests.
- OpenAICompatLLM: any OpenAI-compatible chat endpoint, through the openai client.
- ModalVLLM: the open-weight model that modal_llm.py serves (app "wsp-llm",
  class "PlannerLLM"), reached through Modal auth. complete_many is one batched call.
- AnthropicLLM: Claude, usable only when ANTHROPIC_API_KEY is set.

What every client does:
- Retries transient errors (network errors, timeouts, HTTP 408, 409, 429 and 5xx,
  Modal connection and service errors) with exponential backoff, llm_tries
  tries in all, then raises LLMError. Any other error raises LLMError at once.
- When schema is given, asks the backend for JSON output and checks that every
  reply parses as JSON. A reply that does not parse raises LLMJSONError (a
  subclass of LLMError) that carries the raw replies, so the caller can run its
  own repair retry.
- Appends one dict per API call to self.usage: backend, model, input_tokens,
  output_tokens, seconds. Token counts are None when the API reports none.
- complete_many gives item i the seed seed + i, so a batch is repeatable and
  its items differ even when their prompts are the same.
- Never touches the eval or compute budget.

Config keys (all read with config.get(key, default)):
    llm_tries                 3            tries per call, counting the first
    llm_backoff_s             2.0          wait before the second try; doubles after each failure
    llm_timeout_s             120.0        per-request timeout for the HTTP clients, in seconds
    llm_workers               4            threads for complete_many on the HTTP clients
    mock_replies              None         MockLLM: list of replies, cycled in order; an item that
                                           is an exception instance is raised instead of returned
    mock_fn                   None         MockLLM: fn(messages) -> str, used instead of mock_replies
    openai_base_url           None         OpenAICompatLLM: server URL ending in /v1 (None: OpenAI)
    openai_model              None         OpenAICompatLLM: model name (required)
    openai_api_key_env        "OPENAI_API_KEY"  OpenAICompatLLM: env var that holds the key; when it
                                           is unset and openai_base_url is set, a placeholder key is sent
    modal_llm_app             "wsp-llm"    ModalVLLM: deployed Modal app name
    modal_llm_class           "PlannerLLM" ModalVLLM: modal.Cls name in that app
    modal_llm_environment     None         ModalVLLM: Modal environment (None: the default one of the
                                           workspace that MODAL_TOKEN_ID / MODAL_TOKEN_SECRET belong to)
    anthropic_model           "claude-sonnet-5-5"  AnthropicLLM: model id
    anthropic_effort          None         AnthropicLLM: low | medium | high | xhigh | max (None: model default)
    anthropic_fallbacks       "default"    AnthropicLLM: server-side refusal fallback; None turns it off
    anthropic_min_max_tokens  4096         AnthropicLLM: floor on max_tokens, because adaptive
                                           thinking spends from the same allowance as the reply
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from wsp.swarm.interfaces import LLM

log = logging.getLogger(__name__)

TRANSIENT_STATUS = (408, 409, 429)
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# JSON schema keywords whose values hold sub-schemas, for split_type_lists.
SCHEMA_MAP_KEYS = ("properties", "$defs", "definitions")
SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
SCHEMA_KEYS = ("items", "additionalProperties", "not")


class LLMError(RuntimeError):
    """An LLM call failed: a non-transient error, or transient errors that outlasted the retries."""


class LLMJSONError(LLMError):
    """A reply that should be JSON does not parse.

    texts: every raw reply of the call (one for complete, all of them for complete_many).
    bad: the indexes in texts that do not parse.
    """

    def __init__(self, message, texts, bad):
        LLMError.__init__(self, message)
        self.texts = texts
        self.bad = bad


def error_text(error):
    """A short description of an exception for logs and LLMError messages."""
    text = str(error)
    if len(text) > 300:
        text = text[:300] + "..."
    return type(error).__name__ + ": " + text


def item_seed(seed, index):
    """The seed for item index of a batch: seed + index, or None when seed is None."""
    if seed is None:
        return None
    return int(seed) + index


def check_json(texts, schema, who):
    """Raise LLMJSONError when schema is given and any text is not valid JSON."""
    if schema is None:
        return
    bad = []
    for index in range(len(texts)):
        try:
            json.loads(texts[index])
        except ValueError:
            bad.append(index)
    if len(bad) > 0:
        numbers = []
        for index in bad:
            numbers.append(str(index))
        raise LLMJSONError(who + ": reply " + ", ".join(numbers) + " of " + str(len(texts)) +
                           " is not valid JSON", texts, bad)


def is_transient_status(status_code):
    """True for HTTP statuses worth retrying: 408, 409, 429 and every 5xx."""
    if status_code is None:
        return False
    return status_code in TRANSIENT_STATUS or status_code >= 500


def split_system(messages):
    """Split OpenAI-shaped messages into (system text, the other messages).

    Every system message is joined into one system text, in order, because the
    Anthropic API takes the system prompt as a separate field.
    """
    system_parts = []
    chat = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            chat.append({"role": message["role"], "content": message["content"]})
    return "\n\n".join(system_parts), chat


def split_type_lists(schema):
    """Return a copy of a JSON schema in which every list-valued "type" becomes an anyOf.

    {"type": ["integer", "null"], "minimum": 0} becomes
    {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]}, which
    means the same thing. The Anthropic SDK's transform_schema accepts only a
    single type name per schema. Keywords other than description and title go
    to the non-null variants only, because type-specific keywords never apply
    to null. The input schema is not changed.
    """
    if not isinstance(schema, dict):
        return schema
    copy = {}
    for key in schema:
        value = schema[key]
        if key in SCHEMA_MAP_KEYS and isinstance(value, dict):
            inner = {}
            for name in value:
                inner[name] = split_type_lists(value[name])
            copy[key] = inner
        elif key in SCHEMA_LIST_KEYS and isinstance(value, list):
            items = []
            for item in value:
                items.append(split_type_lists(item))
            copy[key] = items
        elif key in SCHEMA_KEYS:
            copy[key] = split_type_lists(value)
        else:
            copy[key] = value
    types = copy.get("type")
    if not isinstance(types, list) or len(types) == 0:
        return copy
    if len(types) == 1:
        copy["type"] = types[0]
        return copy
    variants = []
    for type_name in types:
        variant = {"type": type_name}
        if type_name != "null":
            for key in copy:
                if key not in ("type", "description", "title"):
                    variant[key] = copy[key]
        variants.append(variant)
    result = {"anyOf": variants}
    for key in ("description", "title"):
        if key in copy:
            result[key] = copy[key]
    return result


def anthropic_schema(anthropic_module, schema):
    """The schema in the strict form that Anthropic's output_config.format accepts.

    Runs split_type_lists, then the SDK's own transform_schema, which sets
    additionalProperties false on every object and moves constraints the API
    does not support (maxItems, minimum, maximum, minItems other than 0 or 1,
    and others) into the description text. The model then sees them but the
    API does not enforce them, so callers must still validate replies. An open
    map (an object whose additionalProperties is a schema) becomes an object
    with no allowed keys. Raises LLMError when the schema cannot be converted.
    """
    try:
        return anthropic_module.transform_schema(split_type_lists(schema))
    except Exception as error:
        raise LLMError("anthropic: this JSON schema cannot be sent as output_config.format: " +
                       error_text(error)) from error


def with_schema_note(messages, schema):
    """Return a copy of messages whose system prompt asks for JSON that follows schema.

    Used with the json_object fallback, where the server does not see the schema.
    """
    note = "Reply with one JSON object that follows this JSON schema:\n" + json.dumps(schema, sort_keys=True)
    result = []
    if len(messages) > 0 and messages[0]["role"] == "system":
        result.append({"role": "system", "content": messages[0]["content"] + "\n\n" + note})
        start = 1
    else:
        result.append({"role": "system", "content": note})
        start = 0
    for index in range(start, len(messages)):
        result.append(messages[index])
    return result


class RetryingLLM(LLM):
    """Shared retry loop, JSON check, usage log and complete_many for the clients below.

    A subclass sets name, calls setup(config) in __init__, and implements
    send_one(messages, schema, temperature, seed, max_tokens) -> str, which
    makes one API call. It may also override is_transient(error).
    """

    name = "base"

    def setup(self, config):
        if config is None:
            config = {}
        self.config = config
        self.tries = max(1, int(config.get("llm_tries", 3)))
        self.backoff_s = float(config.get("llm_backoff_s", 2.0))
        self.timeout_s = float(config.get("llm_timeout_s", 120.0))
        self.workers = max(1, int(config.get("llm_workers", 4)))
        self.usage = []
        self.model = None

    def is_transient(self, error):
        """True when a retry could help. The base version knows only Python's own network errors."""
        return isinstance(error, (ConnectionError, TimeoutError))

    def record_usage(self, input_tokens, output_tokens, seconds, model=None):
        if model is None:
            model = self.model
        self.usage.append({"backend": self.name, "model": model, "input_tokens": input_tokens,
                           "output_tokens": output_tokens, "seconds": round(seconds, 3)})

    def call_with_retries(self, what, fn):
        """Run fn() up to self.tries times while it fails with a transient error.

        Returns fn's result. Raises LLMError when the error is not transient
        or when the last try fails. An LLMError raised by fn passes through.
        """
        delay = self.backoff_s
        last_error = None
        for attempt in range(1, self.tries + 1):
            try:
                return fn()
            except LLMError:
                raise
            except Exception as error:
                if not self.is_transient(error):
                    raise LLMError(self.name + ": " + what + " failed: " + error_text(error)) from error
                last_error = error
                log.warning("%s: %s try %d of %d failed: %s", self.name, what, attempt, self.tries,
                            error_text(error))
                if attempt < self.tries:
                    time.sleep(delay)
                    delay = delay * 2
        raise LLMError(self.name + ": " + what + " failed after " + str(self.tries) + " tries: " +
                       error_text(last_error)) from last_error

    def complete_raw(self, messages, schema, temperature, seed, max_tokens):
        """One reply with retries and no JSON check."""
        def attempt():
            return self.send_one(messages, schema, temperature, seed, max_tokens)
        return self.call_with_retries("complete", attempt)

    def send_one(self, messages, schema, temperature, seed, max_tokens):
        raise NotImplementedError

    def complete(self, messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        """Return the reply text. Raises LLMError on failure, LLMJSONError when schema is set and the reply is not JSON."""
        text = self.complete_raw(messages, schema, temperature, seed, max_tokens)
        check_json([text], schema, self.name)
        return text

    def complete_many(self, list_of_messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        """Return one reply per message list, in input order.

        The HTTP clients have no batch call, so this sends one request per
        message list on a small thread pool (llm_workers threads).
        """
        if len(list_of_messages) == 0:
            return []
        workers = min(self.workers, len(list_of_messages))
        futures = []
        texts = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index in range(len(list_of_messages)):
                futures.append(pool.submit(self.complete_raw, list_of_messages[index], schema, temperature,
                                           item_seed(seed, index), max_tokens))
            for future in futures:
                texts.append(future.result())
        check_json(texts, schema, self.name)
        return texts


class MockLLM(RetryingLLM):
    """Scripted replies for tests, with no network.

    config["mock_fn"], when set, is called as mock_fn(messages) and its return
    value is the reply. Otherwise config["mock_replies"] is cycled in order:
    call n gets reply n modulo its length, whatever the prompt. A reply that is
    an exception instance is raised, so tests can script failures; transient
    ones (ConnectionError, TimeoutError) go through the same retry loop as the
    real clients. The JSON check applies as it does for the real clients.

    self.calls records every request as {"messages", "schema", "temperature",
    "seed", "max_tokens"}. complete_many runs its items one after another, so
    the cycling order is fixed. With several threads calling complete at once,
    use mock_fn, since the order of the threads decides who gets which reply.
    """

    name = "mock"

    def __init__(self, config=None):
        self.setup(config)
        self.fn = self.config.get("mock_fn")
        self.replies = self.config.get("mock_replies")
        if self.fn is None and not self.replies:
            raise LLMError("MockLLM needs config mock_replies (a non-empty list) or mock_fn (a callable)")
        self.model = "mock"
        self.next_index = 0
        self.calls = []
        self.lock = threading.Lock()

    def send_one(self, messages, schema, temperature, seed, max_tokens):
        with self.lock:
            self.calls.append({"messages": messages, "schema": schema, "temperature": temperature,
                               "seed": seed, "max_tokens": max_tokens})
            if self.fn is not None:
                reply = self.fn(messages)
            else:
                reply = self.replies[self.next_index % len(self.replies)]
                self.next_index += 1
        if isinstance(reply, BaseException):
            raise reply
        return str(reply)

    def complete_many(self, list_of_messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        texts = []
        for index in range(len(list_of_messages)):
            texts.append(self.complete_raw(list_of_messages[index], schema, temperature,
                                           item_seed(seed, index), max_tokens))
        check_json(texts, schema, self.name)
        return texts


class OpenAICompatLLM(RetryingLLM):
    """Any OpenAI-compatible chat completions endpoint (OpenAI, a vLLM server, a hosted model API).

    With a schema it asks for response_format json_schema. If the server
    rejects that with HTTP 400, it switches to response_format json_object for
    the rest of the run and puts the schema into the system prompt instead.
    The openai client's own retries are off (max_retries=0), so the retry
    loop here is the only one.
    """

    name = "openai_compat"

    def __init__(self, config):
        self.setup(config)
        try:
            import openai
        except ImportError as error:
            raise LLMError("OpenAICompatLLM needs the openai package (optional group swarm)") from error
        self.openai = openai
        self.base_url = self.config.get("openai_base_url")
        self.model = self.config.get("openai_model")
        if not self.model:
            raise LLMError("OpenAICompatLLM needs config openai_model")
        key_env = self.config.get("openai_api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            if self.base_url is None:
                raise LLMError("OpenAICompatLLM: env var " + key_env + " is not set (and no openai_base_url)")
            log.warning("openai_compat: %s is not set; sending a placeholder key to %s", key_env, self.base_url)
            api_key = "EMPTY"
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key, max_retries=0,
                                    timeout=self.timeout_s)
        self.json_mode = "json_schema"

    def is_transient(self, error):
        if RetryingLLM.is_transient(self, error):
            return True
        if isinstance(error, self.openai.APIConnectionError):
            return True
        if isinstance(error, self.openai.APIStatusError):
            return is_transient_status(error.status_code)
        return False

    def request(self, messages, response_format, temperature, seed, max_tokens):
        """One chat completions call. Returns the reply text."""
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        if response_format is not None:
            kwargs["response_format"] = response_format
        start = time.time()
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage
        if usage is None:
            self.record_usage(None, None, time.time() - start)
        else:
            self.record_usage(usage.prompt_tokens, usage.completion_tokens, time.time() - start)
        text = response.choices[0].message.content
        if text is None:
            return ""
        return text

    def send_one(self, messages, schema, temperature, seed, max_tokens):
        if schema is None:
            return self.request(messages, None, temperature, seed, max_tokens)
        if self.json_mode == "json_schema":
            response_format = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
            try:
                return self.request(messages, response_format, temperature, seed, max_tokens)
            except self.openai.BadRequestError as error:
                log.warning("openai_compat: json_schema output refused (%s); using json_object from now on",
                            error_text(error))
                self.json_mode = "json_object"
        return self.request(with_schema_note(messages, schema), {"type": "json_object"}, temperature, seed,
                            max_tokens)


class ModalVLLM(RetryingLLM):
    """The planner model served by modal_llm.py, called through Modal auth (no web endpoint).

    complete_many sends the whole list in ONE generate call, which vLLM runs
    as one batch. complete is a batch of one. generate returns only text, so
    usage rows carry the wall seconds of each call and no token counts.
    Deploy the server first: modal deploy modal_llm.py.
    """

    name = "modal_vllm"

    def __init__(self, config):
        self.setup(config)
        try:
            import modal
        except ImportError as error:
            raise LLMError("ModalVLLM needs the modal package (optional group swarm)") from error
        self.modal = modal
        self.app_name = self.config.get("modal_llm_app", "wsp-llm")
        self.class_name = self.config.get("modal_llm_class", "PlannerLLM")
        environment = self.config.get("modal_llm_environment")
        planner_class = modal.Cls.from_name(self.app_name, self.class_name, environment_name=environment)
        self.remote = planner_class()

    def is_transient(self, error):
        if RetryingLLM.is_transient(self, error):
            return True
        exc = self.modal.exception
        transient = (exc.ConnectionError, exc.InternalFailure, exc.InternalError, exc.ServiceError,
                     exc.ResourceExhaustedError)
        return isinstance(error, transient)

    def send_many(self, list_of_messages, schema, temperature, seed, max_tokens):
        """One remote generate call for the whole list. Returns the list of replies."""
        start = time.time()
        try:
            texts = self.remote.generate.remote(list_of_messages, schema, temperature, seed, max_tokens)
        except self.modal.exception.NotFoundError as error:
            raise LLMError("modal_vllm: " + self.app_name + "." + self.class_name + " is not deployed in this "
                           "workspace; run: modal deploy modal_llm.py (" + error_text(error) + ")") from error
        self.record_usage(None, None, time.time() - start)
        if len(texts) != len(list_of_messages):
            raise LLMError("modal_vllm: generate returned " + str(len(texts)) + " replies for " +
                           str(len(list_of_messages)) + " prompts")
        return list(texts)

    def send_one(self, messages, schema, temperature, seed, max_tokens):
        return self.send_many([messages], schema, temperature, seed, max_tokens)[0]

    def complete_many(self, list_of_messages, schema=None, temperature=0.7, seed=None, max_tokens=1024):
        """Return one reply per message list, from a single batched generate call (item i uses seed + i)."""
        if len(list_of_messages) == 0:
            return []

        def attempt():
            return self.send_many(list_of_messages, schema, temperature, seed, max_tokens)

        texts = self.call_with_retries("generate", attempt)
        check_json(texts, schema, self.name)
        return texts


class AnthropicLLM(RetryingLLM):
    """Claude through the anthropic SDK. Usable only when ANTHROPIC_API_KEY is set.

    - The model defaults to claude-sonnet-5-5 (config anthropic_model).
    - System messages become the top-level system prompt.
    - A schema is sent as output_config.format json_schema, after
      anthropic_schema converts it to the strict form the API accepts
      (additionalProperties false everywhere; maxItems and other unsupported
      constraints move into the description, so the API does not enforce them).
    - temperature and seed are not sent: Claude Sonnet 5.5 rejects non-default
      sampling settings and the API has no seed, so replies are not repeatable.
    - The server-side refusal fallback (anthropic_fallbacks, default "default")
      re-runs a declined request on a fallback model inside the same call; it
      uses the beta messages endpoint. A reply that still ends in refusal
      raises LLMError.
    - The SDK's own retries are off (max_retries=0), so the retry loop here is the only one.
    """

    name = "anthropic"

    def __init__(self, config):
        self.setup(config)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("AnthropicLLM needs ANTHROPIC_API_KEY in the environment (or .env), and it is not "
                           "set. Use llm = modal_vllm or openai_compat instead.")
        try:
            import anthropic
        except ImportError as error:
            raise LLMError("AnthropicLLM needs the anthropic package") from error
        self.anthropic = anthropic
        self.model = self.config.get("anthropic_model", "claude-sonnet-5-5")
        self.effort = self.config.get("anthropic_effort")
        self.fallbacks = self.config.get("anthropic_fallbacks", "default")
        self.min_max_tokens = int(self.config.get("anthropic_min_max_tokens", 4096))
        self.client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=self.timeout_s)

    def is_transient(self, error):
        if RetryingLLM.is_transient(self, error):
            return True
        if isinstance(error, self.anthropic.APIConnectionError):
            return True
        if isinstance(error, self.anthropic.APIStatusError):
            return is_transient_status(error.status_code)
        return False

    def send_one(self, messages, schema, temperature, seed, max_tokens):
        system_text, chat = split_system(messages)
        kwargs = {"model": self.model, "max_tokens": max(int(max_tokens), self.min_max_tokens),
                  "messages": chat}
        if system_text != "":
            kwargs["system"] = system_text
        output_config = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema",
                                       "schema": anthropic_schema(self.anthropic, schema)}
        if self.effort:
            output_config["effort"] = self.effort
        if len(output_config) > 0:
            kwargs["output_config"] = output_config
        start = time.time()
        if self.fallbacks is None:
            response = self.client.messages.create(**kwargs)
        else:
            response = self.client.beta.messages.create(betas=[ANTHROPIC_FALLBACK_BETA],
                                                        fallbacks=self.fallbacks, **kwargs)
        usage = response.usage
        if usage is None:
            self.record_usage(None, None, time.time() - start, response.model)
        else:
            self.record_usage(usage.input_tokens, usage.output_tokens, time.time() - start, response.model)
        if response.stop_reason == "refusal":
            raise LLMError("anthropic: the model declined the request (stop_reason refusal)")
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        text = "".join(parts)
        if text == "" and response.stop_reason == "max_tokens":
            raise LLMError("anthropic: hit max_tokens before any reply text; raise max_tokens or "
                           "anthropic_min_max_tokens, or lower anthropic_effort")
        return text
