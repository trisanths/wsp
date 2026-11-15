"""Which donor each skill is probed against, and which probe path a donor needs. Pure Python.

The practice feedback is the benchmark-free probe of wsp/probe.py. For a
skill, transfer is measured against that skill's donor: the source model of
the components the program uses for the skill. A donor with the base's
tokenizer is compared by next-token KL ("kl"). A donor with another
tokenizer is compared on text the donor generated ("text").
"""

from composer import sizes
from composer.skills import SKILLS

# Record keys whose text can tag a mined part for a skill.
TAG_KEYS = ["skill", "skills", "domain", "domains", "tags", "label", "trained_on"]
KL = "kl"
TEXT = "text"


def tag_words(record):
    """The lower-case words a catalog record carries in its skill tag fields."""
    words = []
    for key in TAG_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            words.append(value.lower())
        elif isinstance(value, (list, tuple)):
            for item in value:
                words.append(str(item).lower())
        elif isinstance(value, dict):
            for item in value:
                words.append(str(item).lower())
    return words


def tagged_skills(record, skills):
    """The skills a record is tagged for, by the tag words of composer.skills.SKILLS. May be empty."""
    words = tag_words(record)
    found = []
    for skill in skills:
        for tag in SKILLS[skill]["tags"]:
            if any(tag in word for word in words):
                found.append(skill)
                break
    return found


def program_donors(program_dict, records):
    """Return [(model, record or None)] for each non-host source the program uses, in program order.

    records: {catalog id: record}. A block id has no catalog row, so it is read from the id.
    A distillation teacher counts as an untagged donor.
    """
    host = program_dict.get("host") or program_dict.get("base")
    out = []

    def add(model, record):
        if model and model != host:
            out.append((model, record))

    for graft in program_dict.get("grafts", []):
        record = records.get(graft["id"])
        if record is None:
            record = sizes.block_record(graft["id"])
        if record is not None:
            add(record.get("model"), record)
    experts = program_dict.get("experts")
    if isinstance(experts, dict):
        for donor in experts.get("donors", []):
            add(donor, None)
    for insert in program_dict.get("inserts", []):
        add(insert.get("donor"), None)
    splice = program_dict.get("splice")
    if isinstance(splice, dict):
        for key in splice.get("sources", {}):
            add(splice["sources"][key], None)
    for tv in program_dict.get("task_vectors", []):
        record = records.get(tv["id"])
        if record is not None:
            add(record.get("model"), record)
    base_source = program_dict.get("base_source")
    if isinstance(base_source, dict):
        add(base_source.get("model"), None)
    heal = program_dict.get("heal")
    if isinstance(heal, dict) and heal.get("distill"):
        add(heal["distill"], None)
    return out


def skill_donors(program_dict, skills, records):
    """Return {skill: [donor model, ...]}: the source models of the components used for each skill.

    A component tagged for a skill serves that skill only. A component with no skill
    tag at all serves every skill. A skill that no component serves has an empty
    list, and its transfer is reported as 0.
    """
    out = {}
    for skill in skills:
        out[skill] = []
    for model, record in program_donors(program_dict, records):
        tagged = []
        if record is not None:
            tagged = tagged_skills(record, list(SKILLS))
        for skill in skills:
            if len(tagged) == 0 or skill in tagged:
                if model not in out[skill]:
                    out[skill].append(model)
    return out


def probe_path(models, host, donor):
    """"kl" when the donor shares the host's tokenizer, else "text" (also when a family is unknown)."""
    if sizes.same_tokenizer(models, host, donor) is True:
        return KL
    return TEXT


def probe_plan(program_dict, skills, records, models):
    """Return {skill: [{"donor": model, "path": "kl" or "text"}, ...]} for the skills."""
    host = program_dict.get("host") or program_dict.get("base")
    plan = {}
    donors = skill_donors(program_dict, skills, records)
    for skill in skills:
        plan[skill] = []
        for donor in donors[skill]:
            plan[skill].append({"donor": donor, "path": probe_path(models, host, donor)})
    return plan
