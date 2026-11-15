"""Program text: render a program dict to wsp text, parse text back, and validate."""

from wsp import lang
from wsp.swarm import program_schema as ps
from wsp.swarm import space


class ProgramError(ValueError):
    """A program or its text could not be read. line is the 1-based line, or None."""

    def __init__(self, message, line=None):
        ValueError.__init__(self, message)
        self.message = message
        self.line = line


def render(program):
    """Return the program as wsp text. Raises ProgramError when it cannot be written."""
    try:
        return lang.format_program(program)
    except lang.LangError as error:
        raise ProgramError(error.message, error.line)
    except (ValueError, KeyError, TypeError) as error:
        raise ProgramError(str(error))


def render_or_empty(program):
    """Like render, but returns an empty string when the program cannot be written."""
    try:
        return render(program)
    except ProgramError:
        return ""


def parse(text):
    """Parse wsp text into a program dict. Raises ProgramError with a line number."""
    try:
        program = lang.parse(text)
    except lang.LangError as error:
        raise ProgramError(error.message, error.line)
    except (ValueError, KeyError, TypeError) as error:
        raise ProgramError(str(error))
    return program.to_dict()


def normalize(program):
    """Return the canonical program dict. Raises ProgramError when it is not a program."""
    try:
        return ps.normalize(program)
    except ValueError as error:
        raise ProgramError(str(error))


def validate(program, catalog, config):
    """Return a list of error strings. An empty list means the program is valid."""
    return space.validate(program, catalog, config)


def readable(program, catalog=None):
    """A short name a person can read, for example "Math block L8-19 at L6 + Qwen2.5-7B tap at L10, heal 300".

    Parts use their palette label when the catalog has them, else the swarm's
    short name. A random-part control reads as "random part".
    """
    from studio.server import catalog as catalog_mod

    pieces = []
    for graft in ps.grafts(program or {}):
        part_id = ps.graft_id(graft)
        if ps.graft_control(graft) is not None:
            label = "random part"
        elif catalog is not None and part_id in catalog.records:
            label = catalog_mod.part_label(catalog.records[part_id])
        else:
            label = space.short_name(part_id)
        at = ps.graft_at(graft)
        if at is not None:
            label = label + " at L" + str(at)
        pieces.append(label)
    for tap in (program or {}).get("taps") or []:
        pieces.append(catalog_mod.donor_short(tap.get("donor")) + " tap at L" + str(tap.get("at")))
    for mask_id in ps.mask_ids(program or {}):
        pieces.append("minus " + catalog_mod.mask_label(mask_id).lower())
    lora = (program or {}).get("lora")
    if lora is not None and lora.get("rank") is not None:
        pieces.append("LoRA r" + str(lora["rank"]))
    if len(pieces) == 0:
        name = "the host alone"
    else:
        name = " + ".join(pieces)
    steps = ps.heal_steps(program or {})
    if steps:
        name = name + ", heal " + str(steps)
    return name
