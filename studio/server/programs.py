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
