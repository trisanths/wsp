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


def llm():
    """The LLM choice for spec drafting and the agents: none (default) or a provider name."""
    value = os.environ.get("WSP_STUDIO_LLM", "none").strip()
    if value == "":
        return "none"
    return value


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


def out_dir():
    """Where specs, uploads and run files go. outputs/studio unless WSP_STUDIO_OUT is set."""
    path = os.environ.get("WSP_STUDIO_OUT", "").strip()
    if path == "":
        path = os.path.join(REPO_ROOT, "outputs", "studio")
    os.makedirs(path, exist_ok=True)
    return path
