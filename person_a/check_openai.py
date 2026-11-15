# Checks the agent brain: can the planner model be called, and does it use tools?
#   .venv/bin/python check_openai.py
import json

from dotenv import load_dotenv
from openai import OpenAI

from config import PLANNER_MODEL, WORKER_MODEL
from experiments import log

load_dotenv(override=True)  # use the key in .env, not the old one in the shell
client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "model_info",
        "description": "Return the layer count, attention heads and MLP width of an open-weight model.",
        "parameters": {
            "type": "object",
            "properties": {"model": {"type": "string", "description": "Short model name, e.g. llama1b"}},
            "required": ["model"],
        },
    }
]

results = {}
for model_name in [PLANNER_MODEL, WORKER_MODEL]:
    try:
        response = client.responses.create(
            model=model_name,
            input="Before proposing which layers of llama1b handle arithmetic, look up its structure.",
            tools=tools,
        )
        calls = []
        for item in response.output:
            if item.type == "function_call":
                calls.append({"name": item.name, "arguments": item.arguments})
        results[model_name] = {"ok": True, "tool_calls": calls}
    except Exception as e:
        results[model_name] = {"ok": False, "error": str(e)[:300]}
    print(model_name, "->", json.dumps(results[model_name]))

log("check_openai", {"models": [PLANNER_MODEL, WORKER_MODEL]}, results)
