# Agent-defined skills: the agent hypothesises a label from the weights (e.g. "calendar arithmetic",
# "Python indentation") and builds a test for it. Rules enforced here:
#   1. generator code (run in the network-blocked sandbox) is preferred; hand-written items are allowed
#      but flagged agent_items=True
#   2. a matched control set (same format, doesn't need the skill) is required
#   3. the unablated model must clearly beat chance (prefers each item's own answer over a shuffled one)
#   4. the test is frozen - hashed and saved under mine/skills/<model>/<name>.json with a held-out half the
#      agent never sees - BEFORE any ablation result on it is shown
#   5. once accepted it joins the one-pass scoring for this model and the gate treats it like any target
import hashlib
import json
import os
import random

import common

MIN_ITEMS = 40
MAX_ITEMS = 160
MIN_MARGIN = 0.5  # nats: mean log-prob of real answers minus shuffled answers

GENERATOR_CONTRACT = (
    "Generator contract: your code must write /work/skill.json containing "
    "{\"items\": [{\"prompt\": str, \"answer\": str}, ...], \"control\": [{\"prompt\": str, \"answer\": str}, ...]}. "
    "40-160 items each. Compute answers in code (don't hand-type them). Use random.seed(0). The prompt should "
    "end where the answer starts (include a leading space in the answer if natural). Controls must use the same "
    "format but not need the skill (e.g. same calendar sentences asking which word is capitalised)."
)


def normalise(rows):
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        prompt = str(r.get("prompt", r.get("context", "")))
        answer = str(r.get("answer", r.get("continuation", "")))
        if prompt.strip() == "" or answer.strip() == "":
            continue
        out.append({"context": prompt[:3000], "continuation": answer[:600], "digits_only": False})
    return out


def split_half(rows, seed):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    half = len(rows) // 2
    return {"search": rows[:half], "heldout": rows[half:]}


def define(miner, sandbox_getter, name, description, generator_code=None, items=None, control_items=None):
    name = common.clean_name(name)
    existing = miner.agent_skills()
    if name in existing or name in common.BUILTIN_TARGETS + common.CONTROL_ONLY:
        return {"error": "skill '" + name + "' already exists"}
    if len(existing) >= miner.state.get("max_agent_skills", 8):
        return {"error": "agent-defined skill cap reached"}

    agent_items = False
    if generator_code:
        box = sandbox_getter()
        out = box.run(generator_code)
        raw = box.read_file("/work/skill.json")
        if raw is None:
            return {"error": "generator did not write /work/skill.json", "output": out[-2000:]}
        try:
            data = json.loads(raw)
        except Exception as e:
            return {"error": "skill.json is not valid JSON: " + repr(e)[:200]}
        box.run("import os\nos.remove('/work/skill.json')")
        rows = normalise(data.get("items", []))
        ctrl = normalise(data.get("control", []))
    else:
        agent_items = True
        rows = normalise(items or [])
        ctrl = normalise(control_items or [])
    rows = rows[:MAX_ITEMS]
    ctrl = ctrl[:MAX_ITEMS]
    if len(rows) < MIN_ITEMS or len(ctrl) < MIN_ITEMS:
        return {"error": "need at least " + str(MIN_ITEMS) + " items and " + str(MIN_ITEMS) + " control items; got "
                         + str(len(rows)) + " and " + str(len(ctrl))}

    body = {"items": split_half(rows, 0), "control": split_half(ctrl, 1)}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

    # chance check on the unablated model (only this skill + its control are scored)
    payload = {"skills": {name: {"hash": digest, "items": body["items"], "control": body["control"]}}}
    base = miner.call("base", payload)
    margin = base["margin_vs_shuffled"].get(name)
    score = base["scores"].get(name + "|search", {}).get("score")
    if margin is None or margin < MIN_MARGIN:
        common.log_event(miner.short, "define_skill_rejected", {"name": name}, {"margin": margin, "score": score})
        return {"rejected": True, "reason": "model is near chance on this test (margin " + str(margin)
                + " nats < " + str(MIN_MARGIN) + "); make it easier or pick another hypothesis",
                "base_score_search": score}

    # freeze BEFORE any ablation result on it exists
    record = {"name": name, "description": description, "hash": digest, "agent_items": agent_items,
              "generator_code": generator_code, "items": body["items"], "control": body["control"],
              "base_margin": margin}
    path = os.path.join(common.skills_dir(miner.short), name + ".json")
    common.write_json_atomic(path, record)
    skills = miner.agent_skills()
    skills[name] = {"path": os.path.relpath(path, common.ROOT), "hash": digest, "description": description,
                    "agent_items": agent_items}
    miner.state["agent_skills"] = skills
    miner.save()
    common.log_event(miner.short, "define_skill", {"name": name, "hash": digest},
                     {"n_items": len(rows), "n_control": len(ctrl), "margin": margin, "score": score})
    return {"accepted": True, "name": name, "hash": digest, "n_items": len(rows), "n_control": len(ctrl),
            "base_score_search": score, "margin_nats": margin,
            "note": "frozen; the held-out half is hidden. It now joins every ablation's scoring."}
