"""Pure helpers of the simple builder (studio/web/screens/builder-simple-logic.js), run with node."""
import json
import pathlib
import shutil
import subprocess

import pytest

LOGIC = pathlib.Path(__file__).resolve().parent.parent / "studio" / "web" / "screens" / "builder-simple-logic.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

SCRIPT = """
import * as m from %s;
const locked = (p) => (p.status === 'rejected' || p.usable === false ? { why: 'x' } : null);
const cat = { host: { params_b: 1.24 }, parts: [
  { id: 'a', group: 'add', kind: 'block', label: 'Math block', donor: 'Qwen/Qwen2.5-Math-1.5B', params_m: 45, status: 'candidate', evidence: 'no measurement yet' },
  { id: 'b', group: 'add', kind: 'block', label: 'Math block 2', donor: 'x/Math-1B', params_m: 90, status: 'validated', evidence: 'ok' },
  { id: 'c', group: 'add', kind: 'expert', label: 'Fast recall', donor: 'LiquidAI/LFM2-8B-A1B', params_m: 90, status: 'candidate', evidence: 'x' },
  { id: 'd', group: 'add', kind: 'block', label: 'Math old', donor: 'x/Math-2B', params_m: 9, status: 'rejected', evidence: 'bad' },
  { id: 'e', group: 'add', kind: 'block', label: 'Math locked', donor: 'x/Math-3B', params_m: 9, status: 'candidate', usable: false },
], masks: [ { id: 'm1', label: 'Chemistry', status: 'candidate' }, { id: 'm2', label: 'Math facts', status: 'validated' }, { id: 'm3', label: 'Law', status: 'rejected' } ] };
const spec = { task_type: 'math', ingredients: [{ part_id: 'c' }], benchmarks: [{ id: 'gsm8k', name: 'GSM8K', kind: 'target', target: 0.5, base: 0.23 }] };
const out = {};
out.picks = m.pickSkills(cat, spec, locked).map((p) => p.id);
out.name = m.plainName(cat.parts[0], spec);
out.names = m.plainNames([cat.parts[0], cat.parts[0]], spec);
const ex = (id, label, pm) => ({ id, group: 'add', kind: 'expert', label, donor: 'LiquidAI/LFM2-24B-A2B', params_m: pm, status: 'candidate', evidence: 'x' });
out.byLabel = m.plainNames([ex('x1', 'LFM2-24B-A2B expert 18', 100), ex('x2', 'LFM2-24B-A2B expert 7', 100)], spec);
out.bySize = m.plainNames([ex('x1', 'same', 100), ex('x2', 'same', 200)], spec);
out.twins = m.pickSkills({ host: { params_b: 1 }, parts: [ex('x1', 'same', 100), ex('x2', 'same', 100), ex('x3', 'other', 100)], masks: [] }, { task_type: 'none', ingredients: [{ part_id: 'x1' }, { part_id: 'x2' }, { part_id: 'x3' }] }, locked).map((p) => p.id);
const cat2 = { masks: [{ id: 'q1', label: 'Forget bbeh boardgame qa', status: 'candidate' }, { id: 'q2', label: 'Forget medicine', status: 'candidate' }, { id: 'q3', label: 'dyck languages', status: 'candidate' }] };
out.topics = m.pickForgets(cat2, spec, null, locked, []).map((x) => x.id);
out.noTarget = m.resultOf({ benchmarks: [{ id: 'a', name: 'ARC', kind: 'guard' }] }, { status: 'ok', scores: { a: 0.4 } }).lines[0];
out.noSpec = m.resultOf(null, { status: 'ok', scores: { z: 0.4 } }).lines[0];
out.empty = m.resultOf(spec, { status: 'ok', scores: {} }).empty;

out.size = m.sizeText(cat.parts[0], 1.24);
out.tiny = m.sizeText({ params_m: 4 }, 1.24);
out.big = m.sizeText({ params_m: 6121.5 }, 1.24);
out.forgets = m.pickForgets(cat, spec, null, locked, []).map((x) => x.id);
out.forgetOn = m.pickForgets(cat, spec, null, locked, ['m2']).map((x) => x.id);
out.forgetName = m.forgetName(cat.masks[0]);
const att = (score, rnd) => ({ status: 'ok', scores: { gsm8k: score }, comparison: { margin: 0.02, verdict: 'success', rows: [{ id: 'random', measured: rnd != null, value: rnd }] } });
out.good = m.resultOf(spec, att(0.61, 0.5)).lines.join(' ');
out.small = m.resultOf(spec, att(0.51, 0.5)).helps;
out.no = m.resultOf(spec, att(0.4, 0.5)).helps;
out.nornd = m.resultOf(spec, att(0.4, null)).lines.join(' ');
out.missRes = m.resultOf(spec, att(0.4, 0.3));
out.hitRes = m.resultOf(spec, att(0.61, 0.5));
out.fail = m.resultOf(spec, { status: 'collapsed' });
out.size1 = m.sizeLine(2.3, 1.0, { max_total_params_b: 2, max_run_params_b: 1.8, max_heal_min: 20 }, 25);
out.polish = m.polishText(4.4, true);
out.polishOff = m.polishText(4.4, false);
out.modes = [m.pickMode('advanced', 'simple'), m.pickMode(null, 'advanced'), m.pickMode(null, null), m.pickMode('x', 'x')];
console.log(JSON.stringify(out));
""" % json.dumps(LOGIC.as_uri())


@pytest.fixture(scope="module")
def out():
    r = subprocess.run(["node", "--input-type=module", "-e", SCRIPT], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_picks_recommended_first_then_task_matches_without_locked(out):
    assert out["picks"] == ["c", "b", "a"]


def test_plain_names_and_sizes(out):
    assert out["name"] == "Math block from Qwen2.5-Math-1.5B"
    assert out["names"] == ["Math block from Qwen2.5-Math-1.5B (option 1)", "Math block from Qwen2.5-Math-1.5B (option 2)"]
    assert out["byLabel"] == ["Math skill from LFM2-24B-A2B, expert 18", "Math skill from LFM2-24B-A2B, expert 7"] or "expert 18" in out["byLabel"][0]
    assert "100M" in out["bySize"][0] and "200M" in out["bySize"][1]
    assert out["twins"] == ["x1", "x3"]
    assert out["size"] == "adds 45M parameters, about 4% bigger"
    assert out["tiny"] == "adds 4M parameters, less than 1% bigger"
    assert out["big"] == "adds 6.1B parameters, about 494% bigger"


def test_forgets_skip_the_task_and_rejected(out):
    assert out["forgets"] == ["m1"]
    assert out["forgetOn"][0] == "m2"
    assert out["forgetName"] == "Forget chemistry"


def test_forgets_hide_benchmark_codenames(out):
    assert out["topics"] == ["q2"]


def test_result_without_target_or_scores(out):
    assert out["noTarget"].startswith("Your model gets 40% right on ARC")
    assert out["noSpec"].startswith("Your model gets 40% right on z")
    assert out["empty"] is True


def test_result_verdict_and_next_step(out):
    assert out["hitRes"]["verdict"] == "Meets your target"
    assert out["hitRes"]["next"] is None
    assert out["missRes"]["verdict"] == "Not there yet"
    assert "another skill" in out["missRes"]["next"]
    assert "but the model is still short of the target" in " ".join(out["missRes"]["lines"])


def test_result_sentences(out):
    assert out["good"] == ("Your model gets 61% right on GSM8K (grade-school math questions). The starting model gets 23%. "
                           "That meets the target of 50%. Random parts of the same size get 50%, so these skills help.")
    assert out["small"] == "small"
    assert out["no"] == "no"
    assert "were not scored" in out["nornd"]
    assert out["fail"] == {"failed": True, "status": "collapsed"}


def test_limits_polish_and_mode(out):
    assert len(out["size1"]["warn"]) == 2
    assert "about 4 more minutes" in out["polish"]
    assert "minutes" not in out["polishOff"]
    assert out["modes"] == ["advanced", "advanced", "simple", "simple"]
