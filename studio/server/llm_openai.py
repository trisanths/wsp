"""An OpenAI client for the studio that works with the GPT-5 and o-series models.

The swarm's OpenAICompatLLM sends max_tokens and a temperature. The GPT-5
and o-series models reject both: they take max_completion_tokens and only
the default temperature. This subclass changes only the request.
settings.set_openai_default() points the factory's "openai_compat" choice
at this class for the studio process; wsp/swarm is not edited.
"""

import time

from wsp.swarm.llm import OpenAICompatLLM

NEW_STYLE_PREFIXES = ["gpt-5", "o1", "o3", "o4"]


def is_new_style(model):
    """True for models that need max_completion_tokens and the default temperature."""
    name = str(model or "").lower()
    for prefix in NEW_STYLE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


class StudioOpenAILLM(OpenAICompatLLM):
    name = "openai_compat"

    def request(self, messages, response_format, temperature, seed, max_tokens):
        """One chat completions call. Returns the reply text."""
        if not is_new_style(self.model):
            return OpenAICompatLLM.request(self, messages, response_format, temperature, seed, max_tokens)
        # Reasoning tokens count against max_completion_tokens, so leave room for them.
        kwargs = {"model": self.model, "messages": messages, "max_completion_tokens": max(int(max_tokens), 4096)}
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
