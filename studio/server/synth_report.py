"""The synthesis pipeline in the studio: which programs use it, what it refuses, and the shape of its results.

A program with grafts or inserted layers compiles with wsp/synth.py (pipeline "synth") unless the pipeline is
set to "classic". A program with only masks, task vectors or a LoRA uses the classic compile. The synth path
returns one verdict per component: "contributes", "no better than random" or "harmful".

The stub backend returns made-up verdicts, labeled stub, so the web app can be built against it.
"""

import zlib

from wsp.swarm import program_schema as ps

from studio.server import catalog as catalog_mod
from studio.server import settings

PIPELINES = ["synth", "classic"]
VERDICTS = ["contributes", "no better than random", "harmful"]
BAD_VERDICTS = ["no better than random", "harmful"]
DETAIL_KEYS = ["skill", "contribution", "text_gain", "content_gain", "transfer", "donor_kl_base",
               "donor_kl_written", "heldout_loss_on", "heldout_loss_off", "gate_domain", "gate_general",
               "fit_error", "fit_r2", "part_params", "closed", "scores_without"]
STAGE_NAMES = ["place", "translate in", "translate out", "heal 1", "heal 2", "verify"]


def pipeline_setting():
    """The pipeline the server asks for: WSP_STUDIO_PIPELINE, synth by default."""
    return settings.pipeline()


def has_parts(program):
    return len(ps.grafts(program)) > 0 or len(program.get("inserts") or []) > 0


def pipeline_for(program, requested=None):
    """"synth" or "classic" for a program. A program with no grafts and no inserts is always classic."""
    if requested is None:
        requested = pipeline_setting()
    if requested == "classic" or not has_parts(program):
        return "classic"
    return "synth"


def refusal_problems(program, catalog):
    """Plain messages for what the synthesis pipeline cannot build in this program. [] when it can, or on the stub."""
    if settings.is_stub() or pipeline_for(program) != "synth":
        return []
    from studio import modal_studio

    return modal_studio.refusals(program, catalog.records)


def public_components(raw, catalog, stub):
    """The contract shape of the components: id, label, kind, donor, host_layer, verdict, stub, detail."""
    out = []
    for row in raw or []:
        part_id = row.get("part_id") or row.get("id")
        record = catalog.records.get(part_id)
        label = row.get("label")
        if label is None:
            if record is not None:
                label = catalog_mod.part_label(record)
            else:
                label = str(part_id)
        detail = {}
        for key in DETAIL_KEYS:
            detail[key] = row.get(key)
        evidence = row.get("evidence")
        if evidence is None:
            evidence = "practice" if row.get("contribution") is not None else "text"
        out.append({"id": row["id"], "label": label, "kind": row.get("kind"), "donor": row.get("donor"),
                    "host_layer": row.get("host_layer"), "verdict": row.get("verdict"), "evidence": evidence,
                    "stub": bool(stub), "detail": detail})
    return out


def public_stages(raw, stub):
    out = []
    for row in raw or []:
        item = dict(row)
        item["stub"] = bool(stub)
        out.append(item)
    return out


def public(raw, catalog):
    """The synth part of a result from a real synth_remote result dict."""
    return {"pipeline": "synth", "components": public_components(raw.get("components"), catalog, False),
            "stages": public_stages(raw.get("stages"), False), "drift": raw.get("drift"),
            "sizes": raw.get("sizes"), "stub": False}


def verdict_summary(components):
    """{verdict: count} over a components list."""
    counts = {}
    for item in components or []:
        counts[item["verdict"]] = counts.get(item["verdict"], 0) + 1
    return counts


def none_contribute(components):
    """True when some component rests on a practice score and every such component is bad."""
    if not components:
        return False
    found = False
    for item in components:
        if item.get("evidence") != "practice":
            continue
        found = True
        if item.get("verdict") not in BAD_VERDICTS:
            return False
    return found


# The stub.

def unit(text, salt):
    """A deterministic number in [0, 1) from text and a salt."""
    return (zlib.crc32((salt + ":" + str(text)).encode("utf-8")) % 10000) / 10000.0


def stub_verdict(part_id, control):
    if control is not None:
        return "no better than random"
    return VERDICTS[zlib.crc32(str(part_id).encode("utf-8")) % 3]


def stub_detail(part_id, verdict):
    if verdict == "contributes":
        contribution = round(0.03 + 0.05 * unit(part_id, "c"), 4)
        gain = round(0.02 + 0.04 * unit(part_id, "g"), 4)
    elif verdict == "harmful":
        contribution = round(-0.02 - 0.04 * unit(part_id, "c"), 4)
        gain = round(-0.01 - 0.03 * unit(part_id, "g"), 4)
    else:
        contribution = round(0.01 * (unit(part_id, "c") - 0.5), 4)
        gain = round(0.01 * (unit(part_id, "g") - 0.5), 4)
    return {"skill": None, "contribution": contribution, "text_gain": round(contribution * 0.8, 4),
            "content_gain": gain, "transfer": round(0.2 + 0.5 * unit(part_id, "t"), 4), "donor_kl_base": None,
            "donor_kl_written": None, "heldout_loss_on": None, "heldout_loss_off": None,
            "gate_domain": round(0.5 + 0.4 * unit(part_id, "gd"), 4),
            "gate_general": round(0.1 + 0.3 * unit(part_id, "gg"), 4), "fit_error": round(0.3 * unit(part_id, "f"), 4),
            "fit_r2": round(0.6 + 0.3 * unit(part_id, "r"), 4), "part_params": None, "closed": False, "scores_without": None}


def stub_component_rows(program, catalog):
    """One stub component per graft and per inserted layer, in program order."""
    rows = []
    used = []
    for graft in ps.grafts(program):
        part_id = ps.graft_id(graft)
        record = catalog.records.get(part_id) or {}
        name = part_id
        count = 2
        while name in used:
            name = part_id + "#" + str(count)
            count += 1
        used.append(name)
        verdict = stub_verdict(part_id, ps.graft_control(graft))
        row = {"id": name, "part_id": part_id, "kind": record.get("type"), "donor": record.get("model"),
               "host_layer": ps.graft_at(graft), "verdict": verdict}
        row.update(stub_detail(part_id, verdict))
        rows.append(row)
    for item in program.get("inserts") or []:
        label = "insert " + str(item["donor"]).split("/")[-1] + " L" + str(item["layers"][0]) + "-" + \
            str(item["layers"][1])
        verdict = stub_verdict(label, item.get("control"))
        row = {"id": label, "part_id": label, "label": label, "kind": "inserted_layer", "donor": item["donor"],
               "host_layer": int(item["at"]) - 1, "verdict": verdict}
        row.update(stub_detail(label, verdict))
        rows.append(row)
    return rows


def stub_stage_rows(program, tasks):
    """Six stub stages with made-up seconds and practice scores for the scored tasks."""
    key = str(sorted(str(item) for item in program.get("grafts") or []))
    base = {}
    for task in tasks:
        base[task] = round(0.4 + 0.3 * unit(key + task, "base"), 4)
    heal1 = {task: round(base[task] + 0.02 * unit(key + task, "h1"), 4) for task in tasks}
    heal2 = {task: round(heal1[task] + 0.03 * unit(key + task, "h2"), 4) for task in tasks}
    seconds = [None, round(20 + 20 * unit(key, "s1"), 1), round(10 + 10 * unit(key, "s2"), 1),
               round(60 + 60 * unit(key, "s3"), 1), round(90 + 90 * unit(key, "s4"), 1),
               round(120 + 60 * unit(key, "s5"), 1)]
    before = [base, base, base, base, heal1, heal2]
    after = [base, base, base, heal1, heal2, heal2]
    details = ["Parts attached as closed branches, so the new network equals the host.",
               "Ridge fit, then match each part's native output.", "Output map from the donor's logits.",
               "Steps of the translations and gates, donor KL on skill text.",
               "Steps with a LoRA after the first seam, KL to the frozen host.",
               "Practice scores, rollback on a guard failure, per-part test and a random control."]
    rows = []
    for index in range(len(STAGE_NAMES)):
        rows.append({"name": STAGE_NAMES[index], "seconds": seconds[index], "practice_before": before[index],
                     "practice_after": after[index], "rolled_back": False, "detail": details[index],
                     "stub": True})
    return rows


def stub_public(program, catalog, tasks):
    """The synth part of a stub result: made-up verdicts that are the same every time for the same program."""
    if pipeline_for(program) != "synth":
        return None
    rows = stub_component_rows(program, catalog)
    key = str(program.get("grafts"))
    return {"pipeline": "synth", "components": public_components(rows, catalog, True),
            "stages": stub_stage_rows(program, tasks),
            "drift": {"base_kl": round(0.004 + 0.03 * unit(key, "drift"), 5),
                      "agree": round(0.9 + 0.08 * unit(key, "agree"), 4), "tokens": 0,
                      "text": "stub, not measured"},
            "sizes": None, "stub": True}
