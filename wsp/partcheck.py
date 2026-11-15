"""Library compile test: does a mined part keep its skill after it is compiled into another model.

Miners write parts into library/catalog.jsonl. Each row has a skill file
with the part's own test (items and matched controls). This module compiles
the parts of one donor into a base model with wsp.synth.synthesize and
measures, per part, on items the seam never trained on:

    margin_base       the base model
    margin_donor      the donor (the ceiling)
    margin_compiled   the compiled model
    margin_closed     the compiled model with only this part's gate closed
    control margins   compiles of random same-size parts of the same donor
                      layer, and of scrambled copies of the real parts

The margin is the miners' own measure (see item_margin). Nothing here reads
a reported benchmark. Training uses the miners' synthetic skill items
(the "search" half of the skill file) and general text the caller passes
(wsp.healdata pools). Scoring uses the "heldout" half only, which the mining
agent never saw. Formulas and verdict rules are in
docs/partcheck.md.
"""

import copy
import gc
import hashlib
import json
import math
import os
import time

import torch

from wsp import probe
from wsp.synth import synthesize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLITS = ("search", "heldout")

LARGE_UNITS = 64  # parts with fewer units are taken last when --max-parts cuts the list

# Extra row fields that miners add over time. They are carried along, never required.
EXTRA_FIELDS = ("selection_data", "routing", "trained_on", "evidence_used", "created")

THRESHOLDS = {
    "min_gap": 0.1,           # nats: donor margin above base margin, else "no gap"
    "min_contribution": 0.05,  # nats: compiled minus gate-closed on the part's own test items
    "min_gain": 0.1,          # nats: real gain over the best control mean
    "min_z": 2.0,             # content_z floor when two or more control seeds exist
    "harm": 0.05,             # nats: contribution below minus this is harmful
    "max_drift": 0.1,         # nats of floor added to the control allowance (the whole limit with no controls): KL(base || compiled, only this part open)
    "drift_ratio": 5.0,       # a part may drift this many times the control median: the planted part that works reads 2 to 3.5 times it
    "min_seeds_survive": 5,   # a part is "survives compile" only with this many seeds, else "survives provisional"
    "min_items": 6,           # items per half; fewer and the part is skipped
    "alone_share": 0.5,       # a part "needs others" when alone gain is under this share of its group gain
}

STEPS_PER_PART = 100  # stage 1 steps each part should get; the step count grows with the part count

BUDGET = {"stage1_steps": 300, "stage2_steps": 300, "batch_size": 4, "seq_len": 256, "lora_rank": 16,
          "steps_in": 100, "steps_lm": 60, "batch_in": 4, "component_tests": False}

RELATION_ORDER = ("same", "fine_tune", "fine_tune_of_ancestor", "same_family_other_size", "diverged_same_shape",
                  "moe_same_width", "unknown", "foreign")

PROVISIONAL = "survives provisional"

# Model types whose decoder layer wsp.parts.find_layer_modules reads wrongly. Parts built from them would be
# wrong with no error, so partcheck refuses them until find_layer_modules handles the layer.
UNSUPPORTED_MODEL_TYPES = {
    "olmo2": "OLMo-2 applies post_attention_layernorm to the attention output and feeds the raw residual to the "
             "MLP (then post_feedforward_layernorm); find_layer_modules treats post_attention_layernorm as the pre-MLP norm",
}


def unsupported_reason(model_type):
    """Return why parts of this model_type cannot be compiled yet, or None when the type is allowed."""
    return UNSUPPORTED_MODEL_TYPES.get(str(model_type).lower()) if model_type is not None else None

RESULT_FIELDS = ("id", "base", "relation", "model", "margin_base", "margin_donor", "margin_compiled", "recovered",
                 "contribution", "content_z", "control_effect", "drift", "verdict", "reason", "n_search", "n_heldout",
                 "n_heldout_dropped_overlap", "n_seeds", "skill_hash_ok")


# ------------------------------------------------------------------ files


def fix_path(path):
    """Return path with backslashes turned into slashes (catalog rows were written on Windows)."""
    return str(path).replace("\\", "/")


def search_roots(catalog_path=None, root=None):
    """Directories a catalog path may be relative to, in order: root, the catalog's repo root, the repo root.

    Each one is tried as given and with person_a/ added, since skill paths are
    relative to person_a/ and unit paths are relative to person_a/ (per-donor
    files) or to the repo root (library/).
    """
    roots = []
    if root:
        roots.append(root)
    if catalog_path:
        here = os.path.dirname(os.path.abspath(catalog_path))
        roots.append(os.path.dirname(here))
        roots.append(here)
        roots.append(os.path.dirname(os.path.dirname(here)))
    roots.append(REPO)
    expanded = []
    for r in roots:
        expanded.append(r)
        expanded.append(os.path.join(r, "person_a"))
    out = []
    for r in expanded:
        if r not in out:
            out.append(r)
    return out


def resolve_file(path, roots):
    """Return the first existing file for a catalog path, trying it as given and under every root. None if none exists.

    A path may use backslashes, may start with person_a/, or may be relative
    to person_a/ (mine/skills/...) or to the repo root (library/...).
    """
    if path is None:
        return None
    fixed = fix_path(path)
    candidates = []
    if os.path.isabs(fixed):
        candidates.append(fixed)
    else:
        for r in roots:
            candidates.append(os.path.join(r, fixed))
        candidates.append(fixed)
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def normalize_item(raw):
    """Return {"prompt", "answer", "digits_only"} from a miner item {"context", "continuation", "digits_only"}.

    The older names prompt and answer are accepted too. None when the item has no text.
    """
    if not isinstance(raw, dict):
        return None
    prompt = str(raw.get("context", raw.get("prompt", "")))
    answer = str(raw.get("continuation", raw.get("answer", "")))
    if prompt.strip() == "" or answer.strip() == "":
        return None
    return {"prompt": prompt, "answer": answer, "digits_only": bool(raw.get("digits_only", False))}


def normalize_halves(value):
    """Return {"search": [...], "heldout": [...]} of normalized items, or None for a missing or empty value.

    A plain list is taken as the search half with no heldout half.
    """
    if not value:
        return None
    if isinstance(value, list):
        value = {"search": value, "heldout": []}
    out = {}
    for half in ("search", "heldout"):
        rows = []
        for raw in value.get(half) or []:
            item = normalize_item(raw)
            if item is not None:
                rows.append(item)
        out[half] = rows
    return out


_SKILL_CACHE = {}


def read_skill_file(path, name=None):
    """Read one skill source. Return {"name", "description", "hash", "items", "control", "base_margin"}.

    items and control are {"search": [...], "heldout": [...]} (control may be
    None). path is a frozen skill file ({name, description, hash, agent_items,
    generator_code, items, control, base_margin}), or the builtin_skills.json /
    pool_skills.json file ({name: {description, items}}), in which case name
    picks the entry. Those two have no control and no frozen hash.
    """
    key = (path, name)
    if key in _SKILL_CACHE:
        return _SKILL_CACHE[key]
    with open(path) as f:
        data = json.load(f)
    if name is not None:
        data = data[name]
        data = dict(data, name=name)
    out = {"name": data.get("name"), "description": data.get("description"), "hash": data.get("hash"),
           "items": normalize_halves(data.get("items")) or {"search": [], "heldout": []},
           "control": normalize_halves(data.get("control")), "base_margin": data.get("base_margin")}
    _SKILL_CACHE[key] = out
    return out


def source_file(skill, roots):
    """Return (file, entry name) for a row's skill dict. Builtin and pool skills live in person_a/mine/data."""
    source = skill.get("source")
    if source in ("builtin", "pool") and not skill.get("path"):
        fname = "builtin_skills.json" if source == "builtin" else "pool_skills.json"
        return resolve_file("mine/data/" + fname, roots), skill.get("name")
    return resolve_file(skill.get("path"), roots), None


def catalog_files(catalog_path, donor_dir="auto"):
    """Return the catalog files to read: the merged catalog, then the per-donor files (person_a/catalog/*.jsonl).

    donor_dir "auto" uses person_a/catalog next to the catalog's repo root. None reads the merged file only.
    Files named *.rejected.jsonl are never read.
    """
    files = [catalog_path]
    if donor_dir == "auto":
        top = os.path.dirname(os.path.dirname(os.path.abspath(catalog_path)))
        donor_dir = os.path.join(top, "person_a", "catalog")
    if donor_dir and os.path.isdir(donor_dir):
        for name in sorted(os.listdir(donor_dir)):
            if name.endswith(".jsonl") and ".rejected." not in name:
                path = os.path.join(donor_dir, name)
                if os.path.abspath(path) != os.path.abspath(catalog_path):
                    files.append(path)
    return files


def load_parts(catalog_path="library/catalog.jsonl", tiers=("screened",), models=None, root=None, donor_dir="auto"):
    """Read the library. Return (groups, missing).

    Sources, in this order: the merged catalog_path, then every per-donor
    file in person_a/catalog/ (donor_dir; None skips them). A part id that
    appears twice for one donor is kept once, from the first file that has
    it. When the copies differ (skill hash, layer, layers, or unit count) a
    "duplicate id with different content" entry goes to missing. An id used
    by two donors keeps both parts and adds an "id also used by another
    donor" entry, since rows merge by (id, base).

    groups: {donor model id: [part, ...]}. tiers=None takes every tier;
    models=None takes every donor. A part is a dict with id, type, kind,
    model, revision, layer, layers, indices, num_units, d_model, tier,
    pipeline, skill (name), skill_source, skill_description, skill_file
    (resolved absolute path), skill_entry (name inside a builtin or pool
    file, else None), skill_hash (from the row), file_hash (from the skill
    file), hash_ok (False on a mismatch, None when either hash is missing),
    units_file, signature (kept whole when present), extra (selection_data,
    routing, and other optional fields), and row (the whole raw row).

    missing: rows that are not usable, each {"id", "model", "reason", "path"}.
    A missing skill file, a missing or unreadable units file, or a row without
    a skill is listed and never raises. Unknown extra fields on a row are
    ignored. Rows of other tiers or models are skipped without a listing.
    """
    if not os.path.isfile(catalog_path) and not os.path.isabs(catalog_path):
        catalog_path = os.path.join(REPO, catalog_path)
    groups = {}
    missing = []
    seen = {}
    id_models = {}
    for path in catalog_files(catalog_path, donor_dir):
        roots = search_roots(path, root)
        if path != catalog_path:
            roots = search_roots(catalog_path, root) + roots
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    missing.append({"id": None, "model": None, "reason": "catalog line is not JSON", "path": path})
                    continue
                if not isinstance(row, dict):
                    continue
                model = row.get("model") or row.get("donor")
                if tiers is not None and row.get("tier") not in tiers:
                    continue
                if models is not None and model not in models:
                    continue
                key = (model, row.get("id"))
                content = row_content(row)
                if key in seen:
                    if seen[key] != content:
                        missing.append({"id": row.get("id"), "model": model, "path": path,
                                        "reason": "duplicate id with different content"})
                    continue
                seen[key] = content
                other = id_models.setdefault(row.get("id"), model)
                if other != model:
                    missing.append({"id": row.get("id"), "model": model, "path": path,
                                    "reason": "id also used by another donor: " + str(other)})
                part, reason, bad_path = part_from_row(row, roots)
                if part is None:
                    missing.append({"id": row.get("id"), "model": model, "reason": reason, "path": bad_path})
                    continue
                groups.setdefault(model, []).append(part)
    return groups, missing


def row_content(row):
    """The fields that must agree between two copies of one part row."""
    skill = row.get("skill")
    skill_hash = skill.get("hash") if isinstance(skill, dict) else None
    return (skill_hash, row.get("layer"), json.dumps(row.get("layers")), row.get("num_units"))


def part_from_row(row, roots):
    """Return (part, None, None), or (None, reason, path) when a file or field is missing."""
    skill = row.get("skill")
    if isinstance(skill, str):
        skill = {"name": skill}
    if not isinstance(skill, dict) or not (skill.get("path") or skill.get("source") in ("builtin", "pool")):
        return None, "row has no skill file path", None
    skill_file, entry = source_file(skill, roots)
    if skill_file is None:
        return None, "skill file not found", skill.get("path") or skill.get("source")
    try:
        loaded = read_skill_file(skill_file, entry)
    except Exception as error:
        return None, "skill file unreadable: " + repr(error)[:120], skill_file
    kind = row.get("type") or row.get("kind")
    indices = None
    units_file = None
    units = row.get("units")
    if isinstance(units, str):
        units_file = resolve_file(units, roots)
        if units_file is None:
            return None, "units file not found", units
        try:
            blob = torch.load(units_file, map_location="cpu")
        except Exception as error:
            return None, "units file unreadable: " + repr(error)[:120], units
        if isinstance(blob, dict) and "indices" in blob:
            indices = blob["indices"]
        else:
            indices = blob
    elif units is not None:
        indices = units
    elif row.get("indices") is not None:
        indices = row["indices"]
    layers = row.get("layers")
    if kind == "block":
        if not (isinstance(layers, (list, tuple)) and len(layers) == 2):
            return None, "block row has no layers [first, last]", None
    else:
        if indices is None or row.get("layer") is None:
            return None, "row has no layer or no unit indices", units if isinstance(units, str) else None
        indices = [int(v) for v in torch.as_tensor(indices).reshape(-1).tolist()]
    row_hash = skill.get("hash")
    file_hash = loaded.get("hash")
    hash_ok = None
    if row_hash and file_hash:
        hash_ok = row_hash == file_hash
    num_units = row.get("num_units")
    if num_units is None and indices is not None:
        num_units = len(indices)
    part = {"id": row.get("id"), "type": row.get("type"), "kind": kind, "model": row.get("model") or row.get("donor"),
            "revision": row.get("revision"), "layer": row.get("layer"), "layers": layers, "indices": indices,
            "num_units": num_units, "d_model": row.get("d_model"), "tier": row.get("tier"),
            "pipeline": row.get("pipeline"), "skill": skill.get("name") or loaded.get("name"),
            "skill_source": skill.get("source"), "skill_description": skill.get("description"),
            "skill_file": skill_file, "skill_entry": entry, "skill_hash": row_hash, "file_hash": file_hash,
            "hash_ok": hash_ok, "units_file": units_file, "signature": row.get("signature"),
            "extra": {k: row[k] for k in EXTRA_FIELDS if k in row}, "row": row}
    if part["id"] is None or part["kind"] is None:
        return None, "row has no id or no type", None
    return part, None, None


def hash_mismatches(groups):
    """Return [{"id", "row_hash", "file_hash"}] for every part whose skill file hash differs from the row hash."""
    out = []
    for parts in groups.values():
        for p in parts:
            if p["hash_ok"] is False:
                out.append({"id": p["id"], "row_hash": p["skill_hash"], "file_hash": p["file_hash"]})
    return out


def is_large(part):
    """True for a part with 64 units or more, or a block. Single neurons have never beaten random in a compile."""
    if part["kind"] == "block":
        return True
    return (part.get("num_units") or 0) >= LARGE_UNITS


def pick_parts(parts, count):
    """Return count parts. Parts with 64 units or more come first, then one per skill (round robin, catalog order).

    None or 0 means all parts, in the same order.
    """
    ordered = [p for p in parts if is_large(p)] + [p for p in parts if not is_large(p)]
    if not count or count >= len(ordered):
        return ordered
    by_skill = {}
    order = []
    for p in ordered:
        key = p["skill"]
        if key not in by_skill:
            by_skill[key] = []
            order.append(key)
        by_skill[key].append(p)
    out = []
    depth = 0
    while len(out) < count:
        took = False
        for key in order:
            if depth < len(by_skill[key]) and len(out) < count:
                out.append(by_skill[key][depth])
                took = True
        if not took:
            break
        depth += 1
    return out


# ------------------------------------------------------------------ split


def skill_items(part, split):
    """Return {"items": [...], "control": [...]}: the part's own items and matched controls on one half of its file.

    split "search" is the half the mining agent saw: the seam trains on it.
    split "heldout" was never shown to the mining agent and the seam never
    trains on it: every score is measured there. "all" joins both. The
    halves are the file's own, never a hash split. Items are {"prompt",
    "answer", "digits_only"} dicts in file order; control is [] when the file
    has none.
    """
    if split not in ("search", "heldout", "all"):
        raise ValueError("split must be 'search', 'heldout', or 'all', got " + repr(split))
    skill = read_skill_file(part["skill_file"], part.get("skill_entry"))
    out = {}
    for key in ("items", "control"):
        halves = skill[key] or {"search": [], "heldout": []}
        if split == "all":
            out[key] = list(halves["search"]) + list(halves["heldout"])
        else:
            out[key] = list(halves[split])
    return out


def drop_overlap(items, search_items):
    """Return (kept, n_dropped): the items that do not repeat anything the seam trains on.

    An item is dropped when its context equals a search context, when its
    (context, continuation) pair is in the search half, or when its context
    and continuation text appears inside a search context (a few-shot demo).
    Miners wrote the halves from one generator, and many heldout items are
    copies of search items. Scoring those would count memorized answers as
    retained skill.
    """
    contexts = set()
    pairs = set()
    for s in search_items:
        contexts.add(s["prompt"].strip())
        pairs.add((s["prompt"].strip(), s["answer"].strip()))
    kept = []
    dropped = 0
    for h in items:
        context = h["prompt"].strip()
        bad = context in contexts or (context, h["answer"].strip()) in pairs
        if not bad:
            joined = h["prompt"] + h["answer"]
            for s in search_items:
                if joined in s["prompt"]:
                    bad = True
                    break
        if bad:
            dropped += 1
        else:
            kept.append(h)
    return kept, dropped


def split_texts(texts):
    """Split general texts: one in four (by hash) is held out for the drift measure. Return (train, held_out)."""
    train = []
    held = []
    for t in texts:
        digest = hashlib.sha1(("partcheck-text:" + t).encode("utf-8")).hexdigest()
        if int(digest, 16) % 4 == 0:
            held.append(t)
        else:
            train.append(t)
    return train, held


# ----------------------------------------------------------------- margin


_DIGIT_CACHE = {}


def digit_ids(tokenizer, vocab):
    """Boolean tensor over the vocabulary: True for a token whose text holds a digit (as engine_app.py builds it)."""
    key = (id(tokenizer), vocab)
    if key in _DIGIT_CACHE:
        return _DIGIT_CACHE[key]
    flags = torch.zeros(vocab, dtype=torch.bool)
    tokens = tokenizer.convert_ids_to_tokens(list(range(min(len(tokenizer), vocab))))
    for i in range(len(tokens)):
        if tokens[i] is None:
            continue
        for ch in tokens[i]:
            if ch.isdigit():
                flags[i] = True
                break
    _DIGIT_CACHE[key] = flags
    return flags


def _rows_of(tokenizer, triples):
    """Token rows (ids, start, digits_only) for (prompt, answer, digits_only) triples, built as the miners build them."""
    rows = []
    for prompt, answer, digits in triples:
        ctx = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        cont = tokenizer(answer, add_special_tokens=False)["input_ids"]
        ids = (list(ctx) + list(cont))[:1024]
        start = min(len(ctx), len(ids) - 1)
        rows.append((ids, start, bool(digits)))
    return rows


def _pad_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


@torch.no_grad()
def _logprob_sum(model, rows, pad_id, batch_size, digits):
    """Return (sum of log p over scored answer tokens, token count). One forward pass per batch.

    A row with digits_only scores only the answer tokens whose text holds a digit.
    """
    device = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    total = 0.0
    count = 0
    for begin in range(0, len(order), batch_size):
        batch = order[begin:begin + batch_size]
        width = max(len(rows[i][0]) for i in batch)
        ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, i in enumerate(batch):
            r = rows[i][0]
            ids[j, :len(r)] = torch.tensor(r)
            mask[j, :len(r)] = 1
        ids = ids.to(device)
        mask = mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        for j, i in enumerate(batch):
            seq, start, only_digits = rows[i]
            if len(seq) - start <= 0:
                continue
            pos = torch.arange(start - 1, len(seq) - 1, device=device)
            target = ids[j, start:len(seq)]
            lp = torch.log_softmax(logits[j, pos].float(), dim=-1).gather(1, target[:, None])[:, 0]
            if only_digits:
                keep = digits[target.cpu()].to(device)
                lp = lp[keep]
            total += float(lp.sum())
            count += int(lp.numel())
    return total, count


def item_margin(model, tokenizer, items, batch_size=16):
    """Margin over shuffled answers, in nats per answer token. None when there are no scored tokens.

    Matches person_a/mine/engine_app.py (methods base() and _score_rows(), the
    "margin_vs_shuffled" number the miners accept a skill with):

        real      mean log p(answer_i | prompt_i), pooled over all scored tokens of all items
        shuffled  the same with answer_{(i+1) mod n} given prompt_i (the next item's answer)
        margin    real - shuffled

    Tokens are built the miners' way: the prompt with special tokens, the
    answer without, ids cut to 1024, right padding, only answer tokens scored.
    An item with digits_only scores only answer tokens whose text holds a
    digit; a shuffled row takes digits_only from its own prompt's item.
    Items are {"prompt", "answer", "digits_only"} dicts (context and
    continuation are accepted). With one item the shuffled answer is its own,
    so the margin is 0. Batched: one forward pass per batch. The miners' own
    run uses the whole search half of a file; here the caller picks the items.
    """
    items = [normalize_item(i) for i in items]
    items = [i for i in items if i is not None]
    n = len(items)
    if n == 0:
        return None
    pad = _pad_id(tokenizer)
    vocab = model.get_output_embeddings().weight.shape[0]
    digits = digit_ids(tokenizer, vocab)
    real_rows = _rows_of(tokenizer, [(i["prompt"], i["answer"], i["digits_only"]) for i in items])
    shuf_rows = _rows_of(tokenizer, [(items[k]["prompt"], items[(k + 1) % n]["answer"], items[k]["digits_only"])
                                     for k in range(n)])
    was_training = model.training
    model.eval()
    real_sum, real_count = _logprob_sum(model, real_rows, pad, batch_size, digits)
    shuf_sum, shuf_count = _logprob_sum(model, shuf_rows, pad, batch_size, digits)
    if was_training:
        model.train()
    if real_count == 0 or shuf_count == 0:
        return None
    return real_sum / real_count - shuf_sum / shuf_count


# --------------------------------------------------------------- compiling


class RawTokenizer:
    """A tokenizer that hides its chat template.

    wsp.synth renders a prompt and answer pair through the chat template
    when there is one. The miners' items are plain "prompt" then " answer"
    (no template), and item_margin scores them that way. This wrapper makes
    synth encode the fit items the same way, so training and scoring see the
    same format. Everything else passes through.
    """

    chat_template = None

    def __init__(self, tokenizer):
        self.wrapped = tokenizer

    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


def components_for(parts, fit, control=None, seed=0):
    """Component dicts for wsp.synth, one per part. control is None, "random", or "scrambled"."""
    comps = []
    for p in parts:
        c = {"id": p["id"], "kind": p["kind"], "donor": p["model"], "skill": p["skill"],
             "texts": fit[p["id"]]["items"], "control_seed": seed}
        if p["kind"] == "block":
            c["layers"] = [int(p["layers"][0]), int(p["layers"][1])]
        else:
            c["layer"] = int(p["layer"])
            c["indices"] = list(p["indices"])
        if control is not None:
            c["control"] = control
        comps.append(c)
    return comps


def steps_for_parts(n_parts, per_step, steps_per_part=STEPS_PER_PART):
    """Stage 1 steps so that each of n_parts parts is trained in at least steps_per_part steps.

    Each step trains per_step parts, so the count is steps_per_part * n_parts / per_step, rounded up.
    """
    per_step = max(1, min(int(per_step), n_parts))
    return int(math.ceil(steps_per_part * n_parts / per_step))


def compile_group(base, donor, parts, fit, texts, budget, stage2, control=None, seed=0):
    """Compile the parts into a copy of the base. Return (model, synth report). The base is not changed.

    base and donor are (model, tokenizer) pairs. fit maps part id to
    {"items": [...]}, the training items. texts is {"align": [...],
    "general": [...]}. Items train through RawTokenizer. There is no
    practice set, so there is no guard and no rollback: the heal runs to
    its end and the caller measures the result.
    """
    model, tokenizer = base
    work = copy.deepcopy(model)
    s = dict(BUDGET)
    s.update(budget or {})
    s["seed"] = seed
    if not stage2:
        s["stage2_steps"] = 0
    if len(parts) > 6 and "components_per_step" not in s:
        s["components_per_step"] = 3
    per_step = s.get("components_per_step") or len(parts)
    s["stage1_steps"] = max(int(s["stage1_steps"]), steps_for_parts(len(parts), per_step))
    comps = components_for(parts, fit, control, seed)
    donors = {parts[0]["model"]: donor}
    torch.manual_seed(seed)
    work, _, report = synthesize((work, RawTokenizer(tokenizer)), comps, texts, s, donors=donors)
    return work, report


class closed_part:
    """Context manager: close the gate of one branch (by part id) and nothing else."""

    def __init__(self, model, part_id):
        self.model = model
        self.part_id = part_id
        self.found = False

    def __enter__(self):
        self.saved = []
        for b in self.model.wsp_grafts:
            if b.synth["id"] == self.part_id:
                self.saved.append((b, b.forced_off))
                b.forced_off = True
                self.found = True
        if not self.found:
            raise KeyError("no branch for part " + str(self.part_id))
        return self

    def __exit__(self, *args):
        for b, value in self.saved:
            b.forced_off = value
        return False


class only_part:
    """Context manager: close every branch except the one for this part id."""

    def __init__(self, model, part_id):
        self.model = model
        self.part_id = part_id

    def __enter__(self):
        self.saved = []
        for b in self.model.wsp_grafts:
            self.saved.append((b, b.forced_off))
            if b.synth["id"] != self.part_id:
                b.forced_off = True
        return self

    def __exit__(self, *args):
        for b, value in self.saved:
            b.forced_off = value
        return False


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean(values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def sub(a, b):
    if a is None or b is None:
        return None
    return a - b


def rnd(value, digits=5):
    if value is None:
        return None
    return round(float(value), digits)


# ------------------------------------------------------------------ lineage


def relation_of(base_id, donor_id, relate=None):
    """Return the lineage relation of a donor to a base: {"kind", "methods", "evidence", ...}.

    relate is a function (base_id, donor_id) -> relation dict; the default is
    wsp.lineage.relation (it reads the Hub: configs, tokenizer files, and
    safetensors headers, no weights beyond a few MB). On any error the kind
    is "unknown" and the error text is kept.
    """
    try:
        if relate is None:
            from wsp import lineage

            relate = lineage.relation
        rel = relate(base_id, donor_id)
    except Exception as error:
        return {"kind": "unknown", "error": repr(error)[:300], "base": base_id, "donor": donor_id}
    out = {"kind": rel.get("kind", "unknown"), "base": base_id, "donor": donor_id,
           "evidence": rel.get("evidence"), "flags": rel.get("flags"), "warnings": rel.get("warnings")}
    return json_safe(out)


def choose_base(donor_id, candidates, default=None, relate=None):
    """Pick a base for a donor from candidate ids (smaller siblings, Instruct siblings) by lineage.

    Each candidate gets relation_of(). Candidates whose relation could not be
    read are dropped. Among the rest, the best kind in RELATION_ORDER wins;
    ties go to the earlier candidate. If none is left, the default is used
    with kind "unknown". Return (base id, relation dict, tried list of
    {"base", "kind"}).
    """
    tried = []
    best = None
    for cand in candidates:
        rel = relation_of(cand, donor_id, relate)
        tried.append({"base": cand, "kind": rel["kind"]})
        if rel["kind"] == "unknown":
            continue
        rank = RELATION_ORDER.index(rel["kind"]) if rel["kind"] in RELATION_ORDER else len(RELATION_ORDER)
        if best is None or rank < best[0]:
            best = (rank, cand, rel)
    if best is not None:
        return best[1], best[2], tried
    if default is None:
        raise ValueError("no candidate base could be read for " + donor_id + " and no default was given")
    return default, relation_of(default, donor_id, relate), tried


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return json_safe(value.tolist())
    return str(value)


# ------------------------------------------------------------------ verdict


def combine_z(real_value, random_values, scrambled_values):
    """content_z of the real value against each control kind. Return (z, detail).

    The value is the part's contribution (compiled margin minus the margin
    with its gate closed), and each control list holds the same number from
    that control's compiles. z is the smaller of the two kinds' z (the weaker
    comparison), or None when neither kind has two control seeds that differ.
    """
    detail = {}
    zs = []
    for name, values in (("random", random_values), ("scrambled", scrambled_values)):
        stats = probe.content_z(real_value, values)
        detail[name] = stats
        if stats is not None and stats.get("z") is not None:
            zs.append(stats["z"])
    if len(zs) == 0:
        return None, detail
    return min(zs), detail


def median(values):
    values = sorted(v for v in values if v is not None)
    if len(values) == 0:
        return None
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2


def part_verdict(m, th):
    """One word for one part. m has donor_gap, contribution, gain, control_best, content_z, drift, drift_ref, n_seeds.

    Applied in this order:
    harmful               contribution below minus harm
    no gap                the donor is not better than the base by min_gap
    harmful               drift above max_drift + drift_ratio * drift_ref (drift_ref is the median drift of the control
                          compiles, 0 when none ran, so the floor alone applies then)
    no better than controls  contribution at most min_contribution, or contribution not above the best control
                          mean contribution by min_gain, or content_z below min_z
    survives provisional  everything passed with fewer than min_seeds_survive seeds
    survives compile      everything passed
    """
    contribution = m.get("contribution")
    if contribution is not None and contribution < -th["harm"]:
        return "harmful"
    if m.get("donor_gap") is None or m["donor_gap"] < th["min_gap"]:
        return "no gap"
    if m.get("drift") is not None and m["drift"] > th["max_drift"] + th["drift_ratio"] * (m.get("drift_ref") or 0.0):
        return "harmful"
    if contribution is None or contribution <= th["min_contribution"]:
        return "no better than controls"
    best = m.get("control_best")
    if best is not None and sub(contribution, best) < th["min_gain"]:
        return "no better than controls"
    if m.get("content_z") is not None and m["content_z"] < th["min_z"]:
        return "no better than controls"
    if m.get("n_seeds") is not None and m["n_seeds"] < th["min_seeds_survive"]:
        return PROVISIONAL
    return "survives compile"


# -------------------------------------------------------------------- check


def measure_compiled(model, tokenizer, parts, test, general_held, base_model, batch_size):
    """Measure one compiled model. Return {part id: {margin, closed, control, drift}} and the group drift."""
    out = {}
    group = None
    if general_held:
        group = probe.drift(model, base_model, tokenizer, general_held)["base_kl"]
    for p in parts:
        pid = p["id"]
        row = {"margin": item_margin(model, tokenizer, test[pid]["items"], batch_size),
               "control": item_margin(model, tokenizer, test[pid]["control"], batch_size) if test[pid]["control"] else None}
        with closed_part(model, pid):
            row["closed"] = item_margin(model, tokenizer, test[pid]["items"], batch_size)
        row["drift"] = None
        if general_held:
            with only_part(model, pid):
                row["drift"] = probe.drift(model, base_model, tokenizer, general_held)["base_kl"]
        out[pid] = row
    return out, group


def check_parts(base, donor, parts, general_texts, base_id=None, relation=None, stage2=False, seeds=(0,),
                controls=("random", "scrambled"), per_part=False, budget=None, thresholds=None, max_parts=None,
                batch_size=16, log=print, cell=None, confirm_seeds=(), on_progress=None, deadline=None):
    """Compile one donor's parts into a base and report, per part, whether the skill survives.

    base, donor: (model, tokenizer) pairs. They may differ in size, family,
    and tokenizer: each margin uses the model's own tokenizer, and training
    uses text, not donor logits, when the tokenizers differ (wsp.synth).
    parts: parts of ONE donor from load_parts. general_texts: strings for the
    preservation term and the drift measure (wsp.healdata pools); one in four
    is held out for drift. stage2: also run the LoRA heal. seeds: the real
    compile and each control kind run once per seed; with two or more seeds
    content_z exists. confirm_seeds: extra seeds run only when a part would
    survive with fewer than min_seeds_survive seeds. per_part: after the group
    compile, compile each surviving part alone (seed 0 only, no controls).
    cell: a name printed on every log line (the round's cell key and base id).
    on_progress: a function called with the partial report after every
    compile, so a caller can save results before a time limit. deadline: a
    time.time() value; no compile starts after it, and the report says
    "stopped" in its settings.

    Heldout items that repeat a search item are dropped before anything is
    scored (drop_overlap), and the count is kept in n_heldout_dropped_overlap.
    A control compile measures its own per-part contribution (margin minus
    margin with that part's gate closed). The real part is compared with
    those contributions, not with group-level gains, because every part of a
    group shares one compile.

    Returns {"donor", "base", "relation", "parts": [per-part dict], "group",
    "skipped", "settings", "seconds"}. The base model is not changed. No
    reported benchmark is read; training items are the "search" half of the
    miners' own skill items.
    """
    start = time.time()
    th = dict(THRESHOLDS)
    th.update(thresholds or {})
    base_model, base_tok = base
    donor_model, donor_tok = donor
    donor_id = parts[0]["model"]
    tag = "partcheck cell=%s donor=%s" % (cell or ("%s->%s" % (donor_id, base_id)), donor_id)
    log("%s start parts=%d base=%s" % (tag, len(parts), base_id))
    chosen = pick_parts(parts, max_parts)
    fit = {}
    test = {}
    usable = []
    skipped = []
    dropped = {}
    for p in chosen:
        a = skill_items(p, "search")
        b = skill_items(p, "heldout")
        held, n_drop = drop_overlap(b["items"], a["items"])
        held_control, n_drop_control = drop_overlap(b["control"], a["items"] + a["control"])
        dropped[p["id"]] = n_drop
        if len(a["items"]) < th["min_items"] or len(held) < th["min_items"]:
            skipped.append({"id": p["id"], "reason": "too few items: search %d, heldout %d after dropping %d that repeat the search half" % (
                len(a["items"]), len(held), n_drop), "n_search": len(a["items"]), "n_heldout": len(held),
                "n_heldout_dropped_overlap": n_drop, "skill_hash_ok": p.get("hash_ok")})
            continue
        fit[p["id"]] = a
        test[p["id"]] = {"items": held, "control": held_control}
        usable.append(p)
    if len(usable) == 0:
        return {"donor": donor_id, "base": base_id, "relation": relation, "parts": [], "group": None,
                "skipped": skipped, "settings": {}, "seconds": 0.0}
    train_texts, held_texts = split_texts(general_texts)
    texts = {"align": train_texts, "general": train_texts}
    seeds = list(seeds)

    margin_base = {}
    ctrl_base = {}
    margin_donor = {}
    for p in usable:
        pid = p["id"]
        margin_base[pid] = item_margin(base_model, base_tok, test[pid]["items"], batch_size)
        ctrl_base[pid] = item_margin(base_model, base_tok, test[pid]["control"], batch_size) if test[pid]["control"] else None
        margin_donor[pid] = item_margin(donor_model, donor_tok, test[pid]["items"], batch_size)
        log("%s part=%s base=%s donor_margin=%s base_margin=%s" % (
            tag, pid, base_id, rnd(margin_donor[pid], 4), rnd(margin_base[pid], 4)))

    real = {p["id"]: [] for p in usable}
    ctrl = {name: {p["id"]: [] for p in usable} for name in controls}
    state = {"group_drift": [], "heal_seconds": [], "seeds": [], "stopped": False}

    def out_of_time():
        if deadline is not None and time.time() > deadline:
            state["stopped"] = True
            log("%s wall limit reached, no more compiles" % tag)
        return state["stopped"]

    def run_seed(seed):
        """The real compile and every control compile for one seed."""
        if out_of_time():
            return
        model, rep = compile_group(base, donor, usable, fit, texts, budget, stage2, None, seed)
        state["heal_seconds"].append(rep.get("heal_seconds"))
        res, gd = measure_compiled(model, base_tok, usable, test, held_texts, base_model, batch_size)
        state["group_drift"].append(gd)
        for p in usable:
            r = res[p["id"]]
            real[p["id"]].append(r)
            log("%s part=%s seed=%d compiled=%s closed=%s control_items=%s" % (
                tag, p["id"], seed, rnd(r["margin"], 4), rnd(r["closed"], 4), rnd(r["control"], 4)))
        del model
        free_memory()
        state["seeds"].append(seed)
        if on_progress is not None:
            on_progress(build_report(False))
        for name in controls:
            if out_of_time():
                return
            model, rep = compile_group(base, donor, usable, fit, texts, budget, stage2, name, seed)
            res, gd = measure_compiled(model, base_tok, usable, test, held_texts, base_model, batch_size)
            for p in usable:
                r = res[p["id"]]
                ctrl[name][p["id"]].append(r)
                log("%s part=%s control=%s seed=%d margin=%s contribution=%s drift=%s" % (
                    tag, p["id"], name, seed, rnd(r["margin"], 4), rnd(sub(r["margin"], r["closed"]), 4), rnd(r["drift"], 4)))
            del model
            free_memory()
            if on_progress is not None:
                on_progress(build_report(False))

    def build_rows():
        rows = []
        for p in usable:
            pid = p["id"]
            runs = real[pid]
            compiled = mean([r["margin"] for r in runs])
            closed = mean([r["closed"] for r in runs])
            ctrl_items = mean([r["control"] for r in runs])
            drift = mean([r["drift"] for r in runs])
            gap = sub(margin_donor[pid], margin_base[pid])
            gain = sub(compiled, margin_base[pid])
            contribution = sub(compiled, closed)
            recovered = None
            if gap is not None and gain is not None and gap >= th["min_gap"]:
                recovered = gain / gap
            gains = {}
            contribs = {}
            drifts = []
            for name in controls:
                gains[name] = [sub(r["margin"], margin_base[pid]) for r in ctrl[name][pid]]
                contribs[name] = [sub(r["margin"], r["closed"]) for r in ctrl[name][pid]]
                drifts += [r["drift"] for r in ctrl[name][pid]]
            z, z_detail = None, {}
            if contribution is not None and len(controls) > 0:
                z, z_detail = combine_z(contribution, [v for v in contribs.get("random", []) if v is not None],
                                        [v for v in contribs.get("scrambled", []) if v is not None])
            best_means = [mean(v) for v in contribs.values() if len([x for x in v if x is not None]) > 0]
            control_best = max(best_means) if len(best_means) > 0 else None
            gain_means = [mean(v) for v in gains.values() if len([x for x in v if x is not None]) > 0]
            control_best_gain = max(gain_means) if len(gain_means) > 0 else None
            n_ctrl_seeds = min([len(v) for v in contribs.values()]) if len(contribs) > 0 else 0
            drift_ref = median(drifts)
            m = {"donor_gap": gap, "contribution": contribution, "gain": gain, "control_best": control_best,
                 "content_z": z, "drift": drift, "drift_ref": drift_ref,
                 "n_seeds": min(len(runs), n_ctrl_seeds) if len(controls) > 0 else len(runs)}
            verdict = part_verdict(m, th)
            row = {"id": pid, "skill": p["skill"], "kind": p["kind"], "layer": p["layer"], "model": donor_id,
                   "n_search": len(fit[pid]["items"]), "n_heldout": len(test[pid]["items"]),
                   "n_heldout_dropped_overlap": dropped[pid], "n_seeds": m["n_seeds"],
                   "margin_base": margin_base[pid], "margin_donor": margin_donor[pid],
                   "margin_compiled": compiled, "margin_closed": closed, "recovered": recovered,
                   "contribution": contribution, "gain": gain, "content_z": z, "content_z_detail": z_detail,
                   "margin_random": [r["margin"] for r in ctrl.get("random", {}).get(pid, [])] if "random" in ctrl else None,
                   "margin_scrambled": [r["margin"] for r in ctrl.get("scrambled", {}).get(pid, [])] if "scrambled" in ctrl else None,
                   "contribution_random": contribs.get("random"), "contribution_scrambled": contribs.get("scrambled"),
                   "control_best_contribution": control_best, "control_best_gain": control_best_gain,
                   "drift_ref": drift_ref,
                   "control_effect": sub(ctrl_items, ctrl_base[pid]), "control_items_base": ctrl_base[pid],
                   "control_items_compiled": ctrl_items, "drift": drift,
                   "compiled_by_seed": [r["margin"] for r in runs], "verdict": verdict,
                   "num_units": p.get("num_units"), "skill_hash_ok": p.get("hash_ok"), "carried": carried_fields(p)}
            rows.append(row)
        return rows

    def build_report(final):
        rows = build_rows() if len(real[usable[0]["id"]]) > 0 else []
        group = group_report(rows, state["group_drift"], th) if rows else None
        return {"donor": donor_id, "base": base_id, "cell": cell, "relation": relation, "parts": json_safe(rows),
                "group": json_safe(group), "skipped": skipped, "partial": not final,
                "settings": {"seeds": seeds, "seeds_done": list(state["seeds"]), "stage2": bool(stage2),
                             "controls": list(controls), "per_part": bool(per_part),
                             "budget": dict(BUDGET, **(budget or {})), "thresholds": th,
                             "heal_seconds": state["heal_seconds"], "stopped": state["stopped"], "general_train": len(train_texts),
                             "general_held": len(held_texts)},
                "seconds": round(time.time() - start, 1)}

    for seed in seeds:
        run_seed(seed)
    rows = build_rows()
    extra = [s for s in confirm_seeds if s not in seeds]
    if len(extra) > 0 and any(r["verdict"] == PROVISIONAL for r in rows):
        for seed in extra:
            run_seed(seed)
        rows = build_rows()
    for row in rows:
        log("%s part=%s verdict=%s recovered=%s contribution=%s content_z=%s n_seeds=%s dropped_overlap=%s" % (
            tag, row["id"], row["verdict"], rnd(row["recovered"], 3), rnd(row["contribution"], 4),
            rnd(row["content_z"], 2), row["n_seeds"], row["n_heldout_dropped_overlap"]))

    group = group_report(rows, state["group_drift"], th)
    if per_part:
        survivors = [p for p in usable if any(r["id"] == p["id"] and r["verdict"] in ("survives compile", PROVISIONAL) for r in rows)]
        for p in survivors:
            model, rep = compile_group(base, donor, [p], fit, texts, budget, stage2, None, seeds[0])
            alone = item_margin(model, base_tok, test[p["id"]]["items"], batch_size)
            del model
            free_memory()
            for r in rows:
                if r["id"] == p["id"]:
                    r["margin_alone"] = alone
                    r["gain_alone"] = sub(alone, r["margin_base"])
                    share = None
                    if r["gain"] is not None and r["gain"] > 1e-9 and r["gain_alone"] is not None:
                        share = r["gain_alone"] / r["gain"]
                    r["alone_share"] = share
                    log("%s part=%s alone_margin=%s alone_share=%s" % (tag, p["id"], rnd(alone, 4), rnd(share, 3)))
        group = add_alone(group, rows, th)
    report = build_report(True)
    report["parts"] = json_safe(rows)
    report["group"] = json_safe(group)
    report["partial"] = False
    report["seconds"] = round(time.time() - start, 1)
    return report


def carried_fields(part):
    """Fields the miners add to rows (signature.z, signature.guard, selection_data, routing), copied when present."""
    out = {}
    sig = part.get("signature")
    if isinstance(sig, dict):
        for key in ("z", "guard"):
            if key in sig:
                out["signature_" + key] = sig[key]
    for key, value in (part.get("extra") or {}).items():
        out[key] = value
    return out


def group_report(rows, group_drift, th):
    """Group fields: counts, mean recovered of the survivors, and drift of the whole compile."""
    survivors = [r for r in rows if r["verdict"] == "survives compile"]
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    out = {"n_parts": len(rows), "counts": counts, "n_survive": len(survivors),
           "n_provisional": counts.get(PROVISIONAL, 0),
           "mean_recovered_survivors": mean([r["recovered"] for r in survivors]),
           "mean_gain_all": mean([r["gain"] for r in rows]), "drift_all_parts_open": mean(group_drift)}
    if len(survivors) == 0 and counts.get(PROVISIONAL, 0) > 0:
        out["verdict"] = "no part survives with enough seeds"
    elif len(survivors) == 0:
        out["verdict"] = "no part survives"
    else:
        out["verdict"] = "parts survive together"
    return out


def add_alone(group, rows, th):
    """Add the one-at-a-time comparison to the group report."""
    alone = [r for r in rows if r.get("alone_share") is not None]
    needs = [r["id"] for r in alone if r["alone_share"] < th["alone_share"]]
    group = dict(group)
    group["alone_checked"] = len(alone)
    group["needs_others"] = needs
    if len(alone) > 0:
        if len(needs) == 0:
            group["verdict"] = "parts survive together and alone"
        else:
            group["verdict"] = "some parts need the others"
    return group


# ------------------------------------------------------------------- output


def rows_from_report(report):
    """Return the miner rows of one donor report: one dict per part with RESULT_FIELDS first.

    Parts that were not run get a row too, with verdict "skipped" and a reason: parts skipped for too few items
    (report["skipped"]) and rows that could not be loaded (report["missing"]). A duplicate-id entry of missing is
    not a part and gets no row. Fields the miners added to the row (signature.z as signature_z, signature.guard
    as signature_guard, selection_data, routing, trained_on, evidence_used, created) follow when the part had them.
    """
    rel = report.get("relation")
    kind = rel.get("kind") if isinstance(rel, dict) else rel
    out = []
    for r in report.get("parts", []):
        row = {"id": r["id"], "base": report.get("base"), "relation": kind, "model": r.get("model", report.get("donor")),
               "margin_base": r.get("margin_base"), "margin_donor": r.get("margin_donor"),
               "margin_compiled": r.get("margin_compiled"), "recovered": r.get("recovered"),
               "contribution": r.get("contribution"), "content_z": r.get("content_z"),
               "control_effect": r.get("control_effect"), "drift": r.get("drift"), "verdict": r.get("verdict"),
               "reason": r.get("reason"), "n_search": r.get("n_search"), "n_heldout": r.get("n_heldout"),
               "n_heldout_dropped_overlap": r.get("n_heldout_dropped_overlap"), "n_seeds": r.get("n_seeds"),
               "skill_hash_ok": r.get("skill_hash_ok")}
        out.append(row)
        for key, value in (r.get("carried") or {}).items():
            if key not in row:
                row[key] = value
    for r in report.get("skipped", []) or []:
        out.append(skipped_row(report, kind, r, r.get("reason"), r.get("n_search"), r.get("n_heldout")))
    for r in report.get("missing", []) or []:
        if r.get("id") is None or str(r.get("reason", "")).startswith(("duplicate id", "id also used")):
            continue
        out.append(skipped_row(report, kind, r, r.get("reason"), None, None))
    return out


def skipped_row(report, kind, entry, reason, n_search, n_heldout):
    """A row for a part that was not compiled. Every RESULT_FIELDS key is present."""
    row = {}
    for key in RESULT_FIELDS:
        row[key] = None
    row.update({"id": entry["id"], "base": report.get("base"), "relation": kind,
                "model": entry.get("model") or report.get("donor"), "verdict": "skipped", "reason": reason,
                "n_search": n_search, "n_heldout": n_heldout,
                "n_heldout_dropped_overlap": entry.get("n_heldout_dropped_overlap"),
                "skill_hash_ok": entry.get("skill_hash_ok")})
    return row


def merge_rows(*row_lists):
    """Merge row lists by (id, base). A later list replaces an earlier row with the same id and base; the same id with another base is kept apart."""
    merged = {}
    order = []
    for rows in row_lists:
        for r in rows:
            key = (r["id"], r.get("base"))
            if key not in merged:
                order.append(key)
            merged[key] = r
    return [merged[k] for k in order]


def write_donor_report(report, path):
    """Write one donor's full report as JSON. Create the folder when it is missing."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(report), f, indent=2)
    return path


def write_rows(rows, path):
    """Write the rows file, merged with the rows already in it (by id and base)."""
    old = []
    if os.path.isfile(path):
        with open(path) as f:
            old = json.load(f)
    merged = merge_rows(old, rows)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(merged), f, indent=2)
    return path
