"""The wsp text syntax: a line-based format for programs.

A program file has one statement per line, and # starts a comment:

    host unsloth/Llama-3.2-1B-Instruct
    seed 0
    mask (bio | chem) - law restrict layers 4..12 modules mlp
    tv code_tv weight 0.5 layers 10..20 modules mlp
    lora rank 16 targets q_proj,v_proj
    graft qmath_l18_contrast512 at 6 rank 64 control random seed 101
    graft qmath_block_8_19 at 5 rank 64 stitch mlp objective downstream steps 500
    heal steps 300 data gsm8k_train lr 3e-4

parse(text) returns a wsp.program.Program, the same object agents send as
JSON. format_program(program) returns the text, and
parse(format_program(p)).to_dict() == p.to_dict().

Command line:

    python -m wsp.lang check file.wsp
    python -m wsp.lang format program.json
    python -m wsp.lang compile file.wsp out_dir --catalog catalog.jsonl

docs/language.md is the reference for people.
"""

import argparse
import json
import math
import os
import re
import sys

from wsp.program import (STITCH_KINDS, Difference, GraftSpec, HealSpec, Intersect, LoraSpec, Program, Restrict,
                         TaskVector, Union, Use)

# An id is a catalog id, a model name, a local path, a module name, or a
# data name. It may start with . or / so a local path such as
# /models/tiny works as a host. It may not start with -, the mask operator.
ID_PATTERN = re.compile(r"^[A-Za-z0-9_./][A-Za-z0-9_./\-]*$")
INT_PATTERN = re.compile(r"^-?[0-9]+$")
FLOAT_PATTERN = re.compile(r"^-?([0-9]+\.[0-9]*|\.[0-9]+|[0-9]+)([eE][-+]?[0-9]+)?$")

STATEMENTS = ["host", "seed", "mask", "tv", "graft", "lora", "heal"]

# Options for each statement: the written name, then the field it sets.
# Some fields have a second, longer name that matches the JSON key.
TV_OPTIONS = {"weight": "weight", "layers": "layers", "modules": "modules"}
GRAFT_OPTIONS = {
    "at": "at",
    "control": "control",
    "seed": "control_seed",
    "control_seed": "control_seed",
    "rank": "glue_rank",
    "glue_rank": "glue_rank",
    "sink": "sink",
    "rms_match": "rms_match",
    "part_lora_rank": "part_lora_rank",
}
LORA_OPTIONS = {
    "rank": "rank",
    "budget_params": "budget_params",
    "budget": "budget_params",
    "targets": "targets",
    "alpha": "alpha",
    "layers": "layers",
}
HEAL_OPTIONS = {"steps": "steps", "data": "data", "lr": "lr", "batch_size": "batch_size", "seq_len": "seq_len"}

# Mask operators, from loosest to tightest binding. This is Python's order.
MASK_LEVELS = {"union": 1, "intersect": 2, "difference": 3}
MASK_SYMBOLS = {"union": "|", "intersect": "&", "difference": "-"}


class LangError(ValueError):
    """A problem in program text. line is the 1-based line number, or None."""

    def __init__(self, line, message):
        self.line = line
        self.message = message
        if line is None:
            super().__init__(message)
        else:
            super().__init__(f"line {line}: {message}")


# Reading text


def tokenize(code, line):
    """Split one line (without its comment) into tokens.

    Words split on spaces. ( ) | & are tokens on their own. A [ ... ] list
    is one token with its spaces removed. A comma joins the words on each
    side, so "q_proj, v_proj" is one token.
    """
    tokens = []
    word = ""
    i = 0
    while i < len(code):
        ch = code[i]
        if ch == "[":
            end = code.find("]", i)
            if end == -1:
                raise LangError(line, "a [ has no closing ]")
            if word != "":
                tokens.append(word)
                word = ""
            tokens.append("".join(code[i:end + 1].split()))
            i = end + 1
            continue
        if ch in "()|&":
            if word != "":
                tokens.append(word)
                word = ""
            tokens.append(ch)
        elif ch.isspace():
            if word != "":
                tokens.append(word)
                word = ""
        else:
            word += ch
        i += 1
    if word != "":
        tokens.append(word)

    merged = []
    for token in tokens:
        if len(merged) > 0 and (merged[-1].endswith(",") or token.startswith(",")):
            merged[-1] = merged[-1] + token
        else:
            merged.append(token)
    return merged


def read_id(token, line, what):
    if ID_PATTERN.match(token) is None:
        raise LangError(line, f"{what} must use only letters, digits, and _ - . /, got {token!r}")
    return token


def read_int(token, line, what):
    if INT_PATTERN.match(token) is None:
        raise LangError(line, f"{what} needs an integer, got {token!r}")
    return int(token)


def read_number(token, line, what):
    """Read an int or a float. The type stays as written, so 1 is an int and 1.0 is a float."""
    if INT_PATTERN.match(token) is not None:
        return int(token)
    if FLOAT_PATTERN.match(token) is not None:
        value = float(token)
        # A number such as 1e400 turns into inf, which JSON cannot hold.
        if not math.isfinite(value):
            raise LangError(line, f"{what} {token!r} is too large")
        return value
    raise LangError(line, f"{what} needs a number, got {token!r}")


def read_switch(token, line, what):
    if token in ("on", "true"):
        return True
    if token in ("off", "false"):
        return False
    raise LangError(line, f"{what} is on or off, got {token!r}")


def read_layers(token, line, what):
    """Read a layer list such as 4..12 or 1,3,5..7. Ranges include both ends."""
    layers = []
    for part in token.split(","):
        if part == "":
            raise LangError(line, f"{what} has an empty item in {token!r}")
        if ".." in part:
            ends = part.split("..")
            if len(ends) != 2:
                raise LangError(line, f"{what} has a bad range {part!r}")
            start = read_int(ends[0], line, what)
            end = read_int(ends[1], line, what)
            if end < start:
                raise LangError(line, f"{what} range {part!r} ends before it starts")
            for value in range(start, end + 1):
                layers.append(value)
        else:
            layers.append(read_int(part, line, what))
    return layers


def read_names(token, line, what):
    """Read a comma list of names such as q_proj,v_proj."""
    names = []
    for part in token.split(","):
        if part == "":
            raise LangError(line, f"{what} has an empty item in {token!r}")
        names.append(read_id(part, line, what))
    return names


def read_value(token, line, what):
    """Read a stitch value: a number, true, false, none, a [list], or a name."""
    if token.startswith("["):
        if not token.endswith("]"):
            raise LangError(line, f"{what} has a bad list {token!r}")
        inner = token[1:-1]
        values = []
        if inner == "":
            return values
        for part in inner.split(","):
            if part == "" or part.startswith("["):
                raise LangError(line, f"{what} has a bad list {token!r}")
            values.append(read_value(part, line, what))
        return values
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "none":
        return None
    if INT_PATTERN.match(token) is not None or FLOAT_PATTERN.match(token) is not None:
        return read_number(token, line, what)
    return read_id(token, line, what)


def read_options(tokens, line, statement, names, stop=None):
    """Read "name value" pairs. Return (fields, rest).

    names maps each written option name to the field it sets. Reading ends
    at the stop word, and rest holds the tokens after it (or None).
    """
    fields = {}
    written = {}
    i = 0
    while i < len(tokens):
        name = tokens[i]
        if stop is not None and name == stop:
            return fields, tokens[i + 1:]
        if name not in names:
            choices = list(names)
            if stop is not None:
                choices.append(stop)
            raise LangError(line, f"unknown {statement} option {name!r}; the options are {', '.join(choices)}")
        field = names[name]
        if field in fields:
            raise LangError(line, f"{statement} sets {written[field]!r} twice")
        if i + 1 >= len(tokens):
            raise LangError(line, f"{name} needs a value")
        fields[field] = tokens[i + 1]
        written[field] = name
        i += 2
    return fields, None


class MaskParser:
    """Read a mask expression from tokens.

    Grammar, loosest first:
        expr   = union ("restrict" options)*
        union  = inter ("|" inter)*
        inter  = diff ("&" diff)*
        diff   = atom ("-" atom)*
        atom   = id | "(" expr ")"
    """

    def __init__(self, tokens, line):
        self.tokens = tokens
        self.line = line
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def take(self):
        token = self.peek()
        if token is None:
            raise LangError(self.line, "the mask expression ends too early")
        self.pos += 1
        return token

    def parse(self):
        if len(self.tokens) == 0:
            raise LangError(self.line, "mask needs an expression")
        expr = self.expr()
        if self.peek() is not None:
            token = self.peek()
            hint = ""
            if token.startswith("-"):
                hint = " (put spaces around -, since ids may hold a -)"
            raise LangError(self.line, f"unexpected {token!r} in the mask expression{hint}")
        return expr

    def expr(self):
        expr = self.union()
        while self.peek() == "restrict":
            self.pos += 1
            expr = self.restrict(expr)
        return expr

    def restrict(self, inner):
        layers = None
        modules = None
        while self.peek() in ("layers", "modules"):
            name = self.take()
            if self.peek() is None:
                raise LangError(self.line, f"{name} needs a value")
            value = self.take()
            if name == "layers":
                if layers is not None:
                    raise LangError(self.line, "restrict sets 'layers' twice")
                layers = read_layers(value, self.line, "layers")
            else:
                if modules is not None:
                    raise LangError(self.line, "restrict sets 'modules' twice")
                modules = read_names(value, self.line, "modules")
        return Restrict(inner, layers, modules)

    def union(self):
        left = self.intersect()
        while self.peek() == "|":
            self.pos += 1
            left = Union(left, self.intersect())
        return left

    def intersect(self):
        left = self.difference()
        while self.peek() == "&":
            self.pos += 1
            left = Intersect(left, self.difference())
        return left

    def difference(self):
        left = self.atom()
        while self.peek() == "-":
            self.pos += 1
            left = Difference(left, self.atom())
        return left

    def atom(self):
        token = self.take()
        if token == "(":
            expr = self.expr()
            if self.peek() != ")":
                raise LangError(self.line, "a ( has no closing )")
            self.pos += 1
            return expr
        if token in (")", "|", "&", "-", "restrict"):
            raise LangError(self.line, f"expected a mask id or (, got {token!r}")
        return Use(read_id(token, self.line, "a mask id"))


def parse_tv(tokens, line):
    if len(tokens) == 0:
        raise LangError(line, "tv needs a catalog id")
    component_id = read_id(tokens[0], line, "a tv id")
    fields, rest = read_options(tokens[1:], line, "tv", TV_OPTIONS)
    weight = 1.0
    if "weight" in fields:
        weight = read_number(fields["weight"], line, "weight")
    layers = None
    if "layers" in fields:
        layers = read_layers(fields["layers"], line, "layers")
    modules = None
    if "modules" in fields:
        modules = read_names(fields["modules"], line, "modules")
    return TaskVector(component_id, weight, layers, modules)


def parse_stitch(tokens, line):
    """Read the words after "stitch". The first word may be the kind."""
    stitch = {}
    i = 0
    if len(tokens) > 0 and tokens[0] in STITCH_KINDS:
        stitch["kind"] = tokens[0]
        i = 1
    while i < len(tokens):
        name = read_id(tokens[i], line, "a stitch option")
        if name in stitch:
            raise LangError(line, f"stitch sets {name!r} twice")
        if i + 1 >= len(tokens):
            raise LangError(line, f"stitch option {name} needs a value")
        stitch[name] = read_value(tokens[i + 1], line, f"stitch option {name}")
        i += 2
    return stitch


def parse_graft(tokens, line):
    if len(tokens) == 0:
        raise LangError(line, "graft needs a catalog id")
    component_id = read_id(tokens[0], line, "a graft id")
    fields, stitch_tokens = read_options(tokens[1:], line, "graft", GRAFT_OPTIONS, stop="stitch")
    options = {}
    if "at" in fields:
        options["at"] = read_int(fields["at"], line, "at")
    if "control" in fields:
        options["control"] = read_id(fields["control"], line, "control")
    if "control_seed" in fields:
        options["control_seed"] = read_int(fields["control_seed"], line, "seed")
    if "glue_rank" in fields:
        options["glue_rank"] = read_int(fields["glue_rank"], line, "rank")
    if "sink" in fields:
        options["sink"] = read_switch(fields["sink"], line, "sink")
    if "rms_match" in fields:
        options["rms_match"] = read_switch(fields["rms_match"], line, "rms_match")
    if "part_lora_rank" in fields:
        options["part_lora_rank"] = read_int(fields["part_lora_rank"], line, "part_lora_rank")
    if stitch_tokens is not None:
        options["stitch"] = parse_stitch(stitch_tokens, line)
    return GraftSpec(component_id, **options)


def parse_lora(tokens, line):
    fields, rest = read_options(tokens, line, "lora", LORA_OPTIONS)
    options = {}
    if "rank" in fields:
        options["rank"] = read_int(fields["rank"], line, "rank")
    if "budget_params" in fields:
        options["budget_params"] = read_int(fields["budget_params"], line, "budget_params")
    if "targets" in fields:
        options["targets"] = read_names(fields["targets"], line, "targets")
    if "alpha" in fields:
        options["alpha"] = read_number(fields["alpha"], line, "alpha")
    if "layers" in fields:
        options["layers"] = read_layers(fields["layers"], line, "layers")
    return LoraSpec(**options)


def parse_heal(tokens, line):
    fields, rest = read_options(tokens, line, "heal", HEAL_OPTIONS)
    options = {}
    if "steps" in fields:
        options["steps"] = read_int(fields["steps"], line, "steps")
    if "data" in fields:
        options["data"] = read_id(fields["data"], line, "data")
    if "lr" in fields:
        options["lr"] = read_number(fields["lr"], line, "lr")
    if "batch_size" in fields:
        options["batch_size"] = read_int(fields["batch_size"], line, "batch_size")
    if "seq_len" in fields:
        options["seq_len"] = read_int(fields["seq_len"], line, "seq_len")
    return HealSpec(**options)


def parse(text):
    """Parse program text into a wsp.program.Program. Raise LangError on a problem."""
    host = None
    seed = 0
    masks = None
    task_vectors = []
    grafts = []
    heal = None
    lora = None
    # The line where each single statement first appeared.
    first_line = {}

    lines = text.splitlines()
    for index in range(len(lines)):
        line = index + 1
        code = lines[index].split("#", 1)[0]
        tokens = tokenize(code, line)
        if len(tokens) == 0:
            continue
        head = tokens[0]
        rest = tokens[1:]
        if head == "base":
            head = "host"
        if head in ("host", "seed", "mask", "lora", "heal"):
            if head in first_line:
                raise LangError(line, f"{head} appears twice (first on line {first_line[head]})")
            first_line[head] = line
        try:
            if head == "host":
                if len(rest) != 1:
                    raise LangError(line, "host takes one model name, as in: host unsloth/Llama-3.2-1B-Instruct")
                host = read_id(rest[0], line, "host")
            elif head == "seed":
                if len(rest) != 1:
                    raise LangError(line, "seed takes one integer, as in: seed 0")
                seed = read_int(rest[0], line, "seed")
            elif head == "mask":
                try:
                    masks = MaskParser(rest, line).parse()
                except RecursionError:
                    raise LangError(line, "the mask expression nests too deep")
            elif head == "tv":
                task_vectors.append(parse_tv(rest, line))
            elif head == "graft":
                grafts.append(parse_graft(rest, line))
            elif head == "lora":
                lora = parse_lora(rest, line)
            elif head == "heal":
                heal = parse_heal(rest, line)
            else:
                raise LangError(line, f"unknown statement {head!r}; the statements are {', '.join(STATEMENTS)}")
        except LangError:
            raise
        except ValueError as error:
            # Checks inside wsp.program (for example an unknown control).
            raise LangError(line, str(error))

    if host is None:
        raise LangError(None, "a program needs a host line, as in: host unsloth/Llama-3.2-1B-Instruct")
    return Program(host=host, masks=masks, task_vectors=task_vectors, seed=seed, grafts=grafts, heal=heal,
                   lora=lora)


def load_program(path):
    """Read a program file. A .json file holds Program.to_dict(); any other file is text."""
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        return program_from_dict(json.loads(text))
    return parse(text)


# The keys Program.from_dict reads. A key outside these lists would be
# dropped without a word, so program_from_dict refuses it.
PROGRAM_KEYS = ["host", "base", "seed", "masks", "task_vectors", "grafts", "heal", "lora"]
TV_KEYS = ["id", "weight", "layers", "modules"]
GRAFT_KEYS = ["id", "at", "control", "control_seed", "glue_rank", "stitch", "sink", "rms_match", "part_lora_rank"]
LORA_KEYS = ["rank", "budget_params", "targets", "alpha", "layers"]
HEAL_KEYS = ["steps", "data", "lr", "batch_size", "seq_len"]
MASK_KEYS = {
    "use": ["op", "id"],
    "union": ["op", "a", "b"],
    "intersect": ["op", "a", "b"],
    "difference": ["op", "a", "b"],
    "restrict": ["op", "inner", "layers", "modules"],
}


def check_mask_keys(node):
    if not isinstance(node, dict) or node.get("op") not in MASK_KEYS:
        raise ValueError(f"a mask node needs an op of {', '.join(MASK_KEYS)}, got {node!r}")
    check_keys(node, MASK_KEYS[node["op"]], "mask " + node["op"])
    if node["op"] == "restrict":
        check_mask_keys(node["inner"])
    elif node["op"] != "use":
        check_mask_keys(node["a"])
        check_mask_keys(node["b"])


def program_from_dict(data):
    """Build a Program from JSON. Raise ValueError on a key it does not know or a missing key."""
    if not isinstance(data, dict):
        raise ValueError(f"a program is a JSON object, got {type(data).__name__}")
    check_keys(data, PROGRAM_KEYS, "program")
    if data.get("masks") is not None:
        check_mask_keys(data["masks"])
    for tv in data.get("task_vectors", []):
        check_keys(tv, TV_KEYS, "task vector")
    for graft in data.get("grafts", []):
        check_keys(graft, GRAFT_KEYS, "graft")
    if data.get("lora") is not None:
        check_keys(data["lora"], LORA_KEYS, "lora")
    if data.get("heal") is not None:
        check_keys(data["heal"], HEAL_KEYS, "heal")
    try:
        return Program.from_dict(data)
    except KeyError as error:
        raise ValueError(f"the program JSON is missing the key {error.args[0]!r}")
    except (TypeError, AttributeError) as error:
        raise ValueError(f"the program JSON has a bad value: {error}")


# Writing text


def id_text(value, what):
    if not isinstance(value, str) or ID_PATTERN.match(value) is None:
        raise ValueError(f"{what} {value!r} cannot be written as text: use only letters, digits, and _ - . /")
    return value


def number_text(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    if isinstance(value, float):
        text = repr(value)
        if FLOAT_PATTERN.match(text) is None:
            raise ValueError(f"{what} {value!r} cannot be written as text")
        return text
    return str(value)


def int_text(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return str(value)


def run_text(start, end):
    if start == end:
        return str(start)
    if end == start + 1:
        return f"{start},{end}"
    return f"{start}..{end}"


def layers_text(layers, what="layers"):
    """Write a layer list with ranges, so [4, 5, 6, 9] becomes 4..6,9."""
    if len(layers) == 0:
        raise ValueError(f"an empty {what} list cannot be written as text")
    for value in layers:
        int_text(value, what)
    parts = []
    start = layers[0]
    end = layers[0]
    for value in layers[1:]:
        if value == end + 1:
            end = value
        else:
            parts.append(run_text(start, end))
            start = value
            end = value
    parts.append(run_text(start, end))
    return ",".join(parts)


def names_text(names, what):
    if len(names) == 0:
        raise ValueError(f"an empty {what} list cannot be written as text")
    parts = []
    for name in names:
        parts.append(id_text(name, what))
    return ",".join(parts)


def value_text(value, what):
    """Write a stitch value so read_value reads it back the same."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "none"
    if isinstance(value, (int, float)):
        return number_text(value, what)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, list):
                raise ValueError(f"{what} has a nested list, which text cannot hold")
            parts.append(value_text(item, what))
        return "[" + ",".join(parts) + "]"
    if isinstance(value, str):
        if value in ("true", "false", "none") or FLOAT_PATTERN.match(value) is not None:
            raise ValueError(f"{what} {value!r} would read back as a different type")
        return id_text(value, what)
    raise ValueError(f"{what} {value!r} cannot be written as text")


def check_keys(data, known, what):
    """Fail on a JSON key this module does not know, so new options are not lost."""
    if not isinstance(data, dict):
        raise ValueError(f"a {what} is a JSON object, got {data!r}")
    for name in data:
        if name not in known:
            raise ValueError(f"wsp.lang does not know the {what} key {name!r}; update wsp/lang.py")


def mask_text(node, parent_level=0, right_side=False):
    """Write a mask dict. Parentheses go in only where the grammar needs them."""
    op = node["op"]
    if op == "use":
        # In the mask grammar, restrict is a keyword and not an id.
        if node["id"] == "restrict":
            raise ValueError("the mask id 'restrict' cannot be written as text, since restrict is a keyword")
        return id_text(node["id"], "mask id")
    if op == "restrict":
        text = mask_text(node["inner"])
        text += " restrict"
        if node["layers"] is not None:
            text += " layers " + layers_text(node["layers"])
        if node["modules"] is not None:
            text += " modules " + names_text(node["modules"], "modules")
        if parent_level > 0:
            text = "(" + text + ")"
        return text
    if op not in MASK_LEVELS:
        raise ValueError(f"unknown mask op: {op}")
    level = MASK_LEVELS[op]
    text = mask_text(node["a"], level, False) + " " + MASK_SYMBOLS[op] + " " + mask_text(node["b"], level, True)
    # Operators group to the left, so a right child at the same level needs parentheses.
    if level < parent_level or (right_side and level == parent_level):
        text = "(" + text + ")"
    return text


def tv_text(data):
    check_keys(data, TV_KEYS, "task vector")
    words = ["tv", id_text(data["id"], "tv id")]
    if data["weight"] != 1.0 or not isinstance(data["weight"], float):
        words.append("weight " + number_text(data["weight"], "weight"))
    if data["layers"] is not None:
        words.append("layers " + layers_text(data["layers"]))
    if data["modules"] is not None:
        words.append("modules " + names_text(data["modules"], "modules"))
    return " ".join(words)


def stitch_text(stitch):
    words = ["stitch"]
    if "kind" in stitch:
        words.append(id_text(stitch["kind"], "stitch kind"))
    else:
        # The parser reads a kind name right after stitch as the kind.
        for name in stitch:
            if name in STITCH_KINDS:
                raise ValueError(f"the stitch option {name!r} is also a stitch kind, so text cannot hold it "
                                 "without a kind")
            break
    for name in stitch:
        if name == "kind":
            continue
        words.append(id_text(name, "stitch option") + " " + value_text(stitch[name], f"stitch option {name}"))
    return " ".join(words)


def graft_text(data):
    check_keys(data, GRAFT_KEYS, "graft")
    words = ["graft", id_text(data["id"], "graft id")]
    if data.get("at") is not None:
        words.append("at " + int_text(data["at"], "at"))
    if data.get("control") is not None:
        words.append("control " + id_text(data["control"], "control"))
    if data.get("control_seed", 0) != 0:
        words.append("seed " + int_text(data["control_seed"], "seed"))
    if data.get("glue_rank") is not None:
        words.append("rank " + int_text(data["glue_rank"], "rank"))
    if data.get("sink", True) is False:
        words.append("sink off")
    if data.get("rms_match", False) is True:
        words.append("rms_match on")
    if data.get("part_lora_rank") is not None:
        words.append("part_lora_rank " + int_text(data["part_lora_rank"], "part_lora_rank"))
    # stitch reads to the end of the line, so it goes last.
    if data.get("stitch") is not None:
        words.append(stitch_text(data["stitch"]))
    return " ".join(words)


def lora_text(data):
    check_keys(data, LORA_KEYS, "lora")
    words = ["lora"]
    if data.get("rank") is not None:
        words.append("rank " + int_text(data["rank"], "rank"))
    if data.get("budget_params") is not None:
        words.append("budget_params " + int_text(data["budget_params"], "budget_params"))
    if data.get("targets") is not None:
        words.append("targets " + names_text(data["targets"], "targets"))
    if data.get("alpha") is not None:
        words.append("alpha " + number_text(data["alpha"], "alpha"))
    if data.get("layers") is not None:
        words.append("layers " + layers_text(data["layers"]))
    return " ".join(words)


def heal_text(data):
    check_keys(data, HEAL_KEYS, "heal")
    words = ["heal"]
    words.append("steps " + int_text(data["steps"], "steps"))
    words.append("data " + id_text(data["data"], "data"))
    words.append("lr " + number_text(data["lr"], "lr"))
    words.append("batch_size " + int_text(data["batch_size"], "batch_size"))
    words.append("seq_len " + int_text(data["seq_len"], "seq_len"))
    return " ".join(words)


def format_program(program):
    """Write a Program (or its to_dict() form) as text, one statement per line."""
    if isinstance(program, dict):
        program = program_from_dict(program)
    data = program.to_dict()
    check_keys(data, ["host", "seed", "masks", "task_vectors", "grafts", "heal", "lora"], "program")
    lines = ["host " + id_text(data["host"], "host"), "seed " + int_text(data["seed"], "seed")]
    if data["masks"] is not None:
        lines.append("mask " + mask_text(data["masks"]))
    for tv in data["task_vectors"]:
        lines.append(tv_text(tv))
    if data.get("lora") is not None:
        lines.append(lora_text(data["lora"]))
    for graft in data["grafts"]:
        lines.append(graft_text(graft))
    if data["heal"] is not None:
        lines.append(heal_text(data["heal"]))
    return "\n".join(lines) + "\n"


# Command line


def compile_file(program, out_dir, catalog_path, align, device):
    """Compile a parsed program. Return the compile report."""
    from wsp.catalog import Catalog
    from wsp.compiler import compile_program

    catalog = Catalog.load(catalog_path)
    align_texts = None
    heal_texts = None
    if len(program.grafts) > 0 and align > 0:
        from wsp.data import task_texts

        name = "gsm8k_train"
        if program.heal_spec is not None:
            name = program.heal_spec.data
        texts = task_texts(name, seed=0)
        align_texts = texts[:align]
        if program.heal_spec is not None:
            heal_texts = texts
    # Write the text form before the compile, so a program from JSON that
    # text cannot hold gets a warning now and not a crash after the work.
    text = None
    try:
        text = format_program(program)
    except ValueError as error:
        print(f"warning: no program.wsp, since the program cannot be written as text: {error}", file=sys.stderr)
    compile_program(program, catalog, out_dir, align_texts=align_texts, heal_texts=heal_texts, device=device)
    wsp_path = os.path.join(out_dir, "program.wsp")
    if text is not None:
        # Keep the text form next to program.json.
        with open(wsp_path, "w") as f:
            f.write(text)
    elif os.path.exists(wsp_path):
        # A program.wsp from an older compile would not match this program.
        os.remove(wsp_path)
    with open(os.path.join(out_dir, "compile_report.json")) as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m wsp.lang", description="Read, check, and compile wsp programs.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="parse a program file and print its JSON")
    check.add_argument("path")
    show = commands.add_parser("format", help="print a program file (text or .json) as text")
    show.add_argument("path")
    build = commands.add_parser("compile", help="compile a program file into out_dir")
    build.add_argument("path")
    build.add_argument("out_dir")
    build.add_argument("--catalog", required=True, help="catalog.jsonl")
    build.add_argument("--align", type=int, default=200,
                       help="number of heal texts for the W_in fit when the program has grafts (0 skips the fit)")
    build.add_argument("--device", default=None, help="cpu or cuda (default: cuda if present)")
    args = parser.parse_args(argv)

    try:
        program = load_program(args.path)
    except OSError as error:
        print(f"{args.path}: {error.strerror}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"{args.path}: {error}", file=sys.stderr)
        return 1

    if args.command == "check":
        print(json.dumps(program.to_dict(), indent=2))
        return 0
    if args.command == "format":
        try:
            text = format_program(program)
        except ValueError as error:
            print(f"{args.path}: {error}", file=sys.stderr)
            return 1
        sys.stdout.write(text)
        return 0
    report = compile_file(program, args.out_dir, args.catalog, args.align, args.device)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
