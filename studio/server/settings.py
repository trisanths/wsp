"""Settings read from environment variables, and shared paths.

Variables are read each time they are asked for, so tests can change them.
"""

import os

from wsp.swarm import config as config_mod

REPO_ROOT = config_mod.REPO_ROOT
WEB_DIR = os.path.join(REPO_ROOT, "studio", "web")

_dotenv_loaded = False


def load_env():
    """Read .env once, the same way wsp/swarm/config.py does. Returns nothing."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    config_mod.load_dotenv()


def backend():
    """The swarm backend: stub (default), local or modal."""
    value = os.environ.get("WSP_STUDIO_BACKEND", "stub").strip()
    if value == "":
        return "stub"
    return value


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


def llm():
    """The LLM for spec drafting and the agents.

    WSP_STUDIO_LLM picks it: none, openai_compat, anthropic, modal_vllm, or
    auto. auto (the default) uses openai_compat when OPENAI_API_KEY is set,
    anthropic when ANTHROPIC_API_KEY is set, and none otherwise.
    """
    load_env()
    value = os.environ.get("WSP_STUDIO_LLM", "auto").strip()
    if value == "" or value == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            value = "openai_compat"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            value = "anthropic"
        else:
            value = "none"
    if value == "openai_compat":
        set_openai_default()
    return value


def openai_model():
    """The OpenAI model for openai_compat. WSP_STUDIO_OPENAI_MODEL, else gpt-5.4-mini."""
    value = os.environ.get("WSP_STUDIO_OPENAI_MODEL", "").strip()
    if value == "":
        return DEFAULT_OPENAI_MODEL
    return value


def set_openai_default():
    """Give every swarm config an openai_model, since the swarm DEFAULTS have none.

    This changes the in-memory defaults only; wsp/swarm/config.py is not edited.
    """
    config_mod.DEFAULTS["openai_model"] = openai_model()
    from wsp.swarm import factory
    factory.CHOICES["llm"]["openai_compat"] = ("studio.server.llm_openai", "StudioOpenAILLM")


def llm_model():
    """The model name shown in the UI, or None when no LLM is in use."""
    name = llm()
    if name == "openai_compat":
        return openai_model()
    if name == "anthropic":
        return "claude-sonnet-5-5"
    if name == "none":
        return None
    return name


def pace_s():
    """Seconds to wait after each stub eval row, so a run is watchable."""
    text = os.environ.get("WSP_STUDIO_PACE_S", "0.8").strip()
    try:
        value = float(text)
    except ValueError:
        return 0.8
    if value < 0:
        return 0.0
    return value


def is_stub():
    return backend() == "stub"


def pipeline():
    """The compile pipeline for programs with parts: "synth" (wsp/synth.py, the default) or "classic".

    WSP_STUDIO_PIPELINE sets it. A program with no grafts and no inserted layers always uses the classic compile.
    """
    value = os.environ.get("WSP_STUDIO_PIPELINE", "synth").strip().lower()
    if value == "classic":
        return "classic"
    return "synth"


# The note belongs to the classic pipeline only. Results from the synthesis pipeline (wsp/synth.py) carry
# no heal warning: the heal there distills from the donor, keeps the host with a KL term, and is checked
# against practice scores with rollback.
#
# The compiler session found on 2026-11-15 that healing on dataset gold text damages
# the base (Qwen2.5-1.5B with only a LoRA heal: GSM8K 72.0 to about 61, MATH-500 55.4
# to about 29). The fixed heal (distillation from the donor, a KL-to-base term and a
# practice-set guard) is being built in wsp/synth.py. Until it lands, results that
# used a heal on real GPUs are labeled as not reliable.
HEAL_NOTE = ("The heal recipe is being fixed. Healing on dataset text currently damages the base model, "
             "so scores after a heal are not reliable yet.")


def heal_note():
    """The warning to show on healed classic results, or None on the stub backend (its numbers are estimates anyway)."""
    if is_stub():
        return None
    return HEAL_NOTE


def heal_note_for(pipeline_name, healed):
    """The warning for one result: the heal note for a healed classic result on a real backend, else None."""
    if pipeline_name != "classic" or not healed:
        return None
    return heal_note()


def out_dir():
    """Where specs, uploads and run files go. outputs/studio unless WSP_STUDIO_OUT is set."""
    path = os.environ.get("WSP_STUDIO_OUT", "").strip()
    if path == "":
        path = os.path.join(REPO_ROOT, "outputs", "studio")
    os.makedirs(path, exist_ok=True)
    return path
