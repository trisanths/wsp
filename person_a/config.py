# Shared settings for the dissection agent.

# The 10 models the agent dissects. Short name -> Hugging Face id.
MODELS = {
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwenmath1.5b": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "qwencoder1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "r1qwen1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "gemma2b": "google/gemma-2-2b-it",
    "phi4mini": "microsoft/Phi-4-mini-instruct",
    "smollm1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "olmoe": "allenai/OLMoE-1B-7B-0125-Instruct",
    "eurollm1.7b": "utter-project/EuroLLM-1.7B-Instruct",
}

# Modal names
APP_NAME = "swarmy"
VOLUME_NAME = "swarmy-models"
MODELS_DIR = "/models"
GPU = "L4"  # 24 GB, enough for every model above in bf16

# OpenAI models for the agent
PLANNER_MODEL = "gpt-6-astra"
WORKER_MODEL = "gpt-5.4-mini"

SEED = 0


def local_dir(short_name):
    # Where a model's files live inside the Modal volume
    return MODELS_DIR + "/" + short_name
