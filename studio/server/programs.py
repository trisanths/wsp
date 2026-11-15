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


def heal_data_problem(program):
    """A plain message when the program heals on a benchmark train set or an unknown name, else None."""
    from wsp import data as wdata

    name = ps.heal_data(program)
    if name is None:
        return None
    try:
        if wdata.is_pool_data(name):
            return None
    except ValueError as error:
        return str(error)
    source = wdata.benchmark_source(name)
    if source is not None:
        return ("Heal data " + name + " is the train set of the benchmark " + source + ". Heal and fit text may not "
                "come from a benchmark. Use a pool: " + ", ".join(wdata.POOL_NAMES + ["mix"]) + ".")
    return "Heal data " + str(name) + " is not an approved pool. Use one of: " + ", ".join(wdata.POOL_NAMES + ["mix"]) + "."


def validate(program, catalog, config):
    """Return a list of error strings. An empty list means the program is valid."""
    problem = heal_data_problem(program)
    if problem is not None:
        return [problem]
    return space.validate(program, catalog, config)


def readable(program, catalog=None):
    """A short name a person can read, for example "Math block L8-19 at L6, heal 300".

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
