"""Model profiles for donors that the library has no profile for.

The sizer needs a donor's parameter count, layer count and hidden size to size
a part. The library has profiles for most donors. model_profiles.json has the
rest, read from each model's public Hugging Face page and config.json. A
library profile always wins over one from this file.
"""

import json
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_profiles.json")
_cache = None


def extra():
    """Donor id to profile, from model_profiles.json. Empty when the file is missing."""
    global _cache
    if _cache is None:
        try:
            with open(PATH) as handle:
                _cache = dict(json.load(handle)["models"])
        except (OSError, ValueError, KeyError):
            _cache = {}
    return _cache


def merged(library_models):
    """The library profiles plus the extra ones for donors the library lacks."""
    out = dict(extra())
    out.update(library_models)
    return out
