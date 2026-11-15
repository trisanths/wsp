"""A small HTTPS client for the OpenAI Responses API, with retries.

The key comes from .env at run time. It goes only into the Authorization
header of requests to api.openai.com. It is never printed, logged, or
archived: errors pass through redact() before anyone sees them.
"""

import json
import os
import time

API_BASE = "https://api.openai.com/v1"
RETRY_STATUS = [408, 409, 429, 500, 502, 503, 504]


class LLMError(Exception):
    pass


def read_key(env_path):
    """Return OPENAI_API_KEY from the .env file, else from the environment, else None."""
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if not line.startswith("OPENAI_API_KEY"):
                    continue
                if "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() != "OPENAI_API_KEY":
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if value != "":
                    return value
    value = os.environ.get("OPENAI_API_KEY")
    if value:
        return value
    return None


def redact_text(text, secrets):
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


class OpenAIClient:
    """POST /v1/responses with retries and backoff.

    post(url, headers, body, timeout) can be swapped in tests; it returns
    (status_code, text). sleep can be swapped too.
    """

    def __init__(self, key, model="gpt-6-astra", reasoning_effort="high", timeout=900, max_retries=6,
                 post=None, sleep=None):
        if not key:
            raise LLMError("no OpenAI key: put OPENAI_API_KEY in .env")
        self.key = key
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.post = post if post is not None else http_post
        self.sleep = sleep if sleep is not None else time.sleep

    def secrets(self):
        return [self.key]

    def create(self, body):
        """Send one Responses API request. Return the parsed JSON response."""
        url = API_BASE + "/responses"
        headers = {"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                status, text = self.post(url, headers, body, self.timeout)
            except Exception as error:
                last_error = redact_text(f"{type(error).__name__}: {error}", self.secrets())
                status = None
                text = None
            if status == 200:
                try:
                    return json.loads(text)
                except ValueError:
                    last_error = "the API returned text that is not JSON"
            elif status is not None:
                last_error = redact_text(f"HTTP {status}: {str(text)[:1000]}", self.secrets())
                if status not in RETRY_STATUS:
                    raise LLMError(last_error)
            if attempt < self.max_retries:
                self.sleep(min(60.0, 2.0 ** attempt))
        raise LLMError(f"gave up after {self.max_retries + 1} tries: {last_error}")


def http_post(url, headers, body, timeout):
    import requests

    if not url.startswith(API_BASE + "/"):
        raise LLMError("the OpenAI client only talks to api.openai.com")
    response = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
    return response.status_code, response.text
