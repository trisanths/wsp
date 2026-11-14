"""Shared settings and file locations for the eval and interp tools.

Everything is relative to WSP_ROOT (default: the current directory), so the
whole team can point at one shared checkout.
"""
import os

SEED = 0

# Number of items per task in the search split. The rest of the test set is
# the held-out split.
SEARCH_SIZE = int(os.environ.get("WSP_SEARCH_SIZE", "150"))


def root_dir():
    return os.environ.get("WSP_ROOT", os.getcwd())


def root_path(*parts):
    return os.path.join(root_dir(), *parts)


def work_dir():
    d = root_path("work")
    os.makedirs(d, exist_ok=True)
    return d


def work_path(*parts):
    return os.path.join(work_dir(), *parts)


def catalog_path():
    return os.environ.get("WSP_CATALOG", root_path("catalog.jsonl"))


def experiments_path():
    return root_path("experiments.jsonl")


def interp_log_path():
    return root_path("interp_log.jsonl")
