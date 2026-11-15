// Pure helpers for the simple builder. No imports, so node can test them.
// Every sentence a person reads in Simple mode is built here or in builder-simple.js.

const TASK_WORDS = {
  math: /math|gsm8k/,
  code: /code|coder/,
  legal: /law|legal|saul|civil|contract|court/,
  medicine: /medic|health|clinic|bio/,
};
const SKILL_NAMES = { math: 'Math', code: 'Coding', legal: 'Legal', medicine: 'Medicine', biology: 'Biology', chemistry: 'Chemistry', law: 'Legal', reasoning: 'Reasoning', general: 'General', physics: 'Physics', history: 'History' };
const STRONG = /against|beats|over random|above the control|better than/i;
const WEAK = /no measurement|one seed|not run|no control|mined from the library|not measured/i;

/** The words in a part's label and donor, lowercase. */
const hayOf = (x) => ((x.label || '') + ' ' + (x.donor || '')).toLowerCase();

/** The task regex for a spec's task type, or null. */
export function taskWords(spec) {
  return (spec && TASK_WORDS[spec.task_type]) || null;
}

/** 0 validated, 1 candidate with strong evidence, 2 other candidates, 3 anything else. */
export function evidenceRank(p) {
  if (p.status === 'validated') return 0;
  const ev = String(p.evidence || '');
  if (p.status === 'candidate' && STRONG.test(ev) && !WEAK.test(ev)) return 1;
  if (p.status === 'candidate') return 2;
  return 3;
}

/** Up to `max` parts to offer as skills: the spec's recommended parts, then parts named for its task.
 *  Only parts you can place. `locked` is the lock test (lockOf in the app). */
export function pickSkills(cat, spec, locked, max = 6) {
  if (!cat || !spec) return [];
  const placeable = (cat.parts || []).filter((p) => (!p.group || p.group === 'add') && !locked(p));
  const byId = new Map(placeable.map((p) => [p.id, p]));
  const out = [];
  const seen = new Set();
  const take = (p) => { if (p && !seen.has(p.id) && out.length < max) { seen.add(p.id); out.push(p); } };
  const rec = (spec.ingredients || []).map((g) => byId.get(g.part_id)).filter(Boolean);
  rec.map((p, i) => [p, i]).sort((a, b) => evidenceRank(a[0]) - evidenceRank(b[0]) || a[1] - b[1]).forEach((x) => take(x[0]));
  const words = taskWords(spec);
  if (words) {
    placeable.map((p, i) => [p, i]).filter((x) => words.test(hayOf(x[0]))).sort((a, b) => evidenceRank(a[0]) - evidenceRank(b[0]) || a[1] - b[1]).forEach((x) => take(x[0]));
  }
  return dropTwins(out);
}

/** Drop parts that read exactly like an earlier one (same name, sentence, size and trust). The first stays. */
function dropTwins(parts) {
  const seen = new Set();
  return parts.filter((p) => {
    const k = [plainName(p), plainSentence(p), Math.round(p.params_m || 0), trustText(p), labelDetail(p)].join('|');
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** The donor model name without its owner and a trailing -Instruct. */
export function donorName(donor) {
  return String(donor || '').split('/').pop().replace(/[-_](instruct|it|chat)$/i, '');
}

/** The skill word for a part: from its own skill list, else from its name, else from the spec's task. */
export function skillWord(part, spec) {
  const h = hayOf(part);
  for (const k of Object.keys(TASK_WORDS)) if (TASK_WORDS[k].test(h)) return SKILL_NAMES[k];
  if (part.task && SKILL_NAMES[part.task] && part.task !== 'general') return SKILL_NAMES[part.task];
  for (const s of part.skills || []) { const k = String(s).toLowerCase(); if (SKILL_NAMES[k]) return SKILL_NAMES[k]; }
  return '';
}

/** "Math skill from Qwen2.5-Math-1.5B". */
export function plainName(part, spec) {
  const w = skillWord(part, spec);
  const d = donorName(part.donor);
  if (w && d) return w + ' skill from ' + d;
  if (d) return (part.label || 'Skill') + ' from ' + d;
  return part.label || part.id;
}

const KIND_SENTENCES = {
  block: 'Copies in a stretch of layers from a model that is already good at this.',
  attn_heads: 'Copies in small pieces that help a model focus on the right details.',
  mlp_cluster: 'Copies in a small group of neurons that fire on this skill.',
  expert: 'Copies in a specialist from a larger model.',
  inserted_layer: 'Adds one new layer that steers the model toward this skill.',
};
export function plainSentence(part) {
  return KIND_SENTENCES[part.kind] || 'Adds a piece of another model that is good at this.';
}

/** "45M parameters" or "1.2B parameters". */
export function paramsText(m) {
  if (m == null || isNaN(m)) return null;
  if (m >= 1000) return (Math.round(m / 100) / 10) + 'B parameters';
  return (m >= 10 ? Math.round(m) : Math.round(m * 10) / 10) + 'M parameters';
}

/** "adds 45M parameters, about 4% bigger". hostB is the host size in billions. */
export function sizeText(part, hostB) {
  const t = paramsText(part.params_m);
  if (!t) return 'size not known';
  let s = 'adds ' + t;
  if (hostB > 0) {
    const p = (part.params_m / 1000 / hostB) * 100;
    s += p < 1 ? ', less than 1% bigger' : ', about ' + Math.round(p) + '% bigger';
  }
  return s;
}

/** A part's evidence in one plain word group. */
export function trustText(p) {
  if (p.status === 'validated') return 'Tested and it works';
  if (p.status === 'candidate') return evidenceRank(p) === 1 ? 'Early tests look good' : 'Not fully tested';
  return '';
}

const TOPIC = /\b(medicine|medical|health|clinical|biology|chemistry|physics|law|legal|history|geography|finance|economics|politics|religion|philosophy|psychology|literature|music|sports|cooking|astronomy|anatomy|nutrition|marketing|sociology|accounting|business|ethics|security)\b/i;
const CODENAME = /bbeh|dyck|\bqa\b|_qa\b|boolean|buggy|disambiguation|causal|boardgame|\w_\w/i;

/** True when a mask has a plain topic name a person can read, such as "Chemistry". */
export function hasTopicName(m) {
  const t = String(m.label || '');
  return TOPIC.test(t) && !CODENAME.test(t);
}

/** Deletion masks to offer: plain topics only, not rejected, not locked, and not about the skill this model is for. */
export function pickForgets(cat, spec, program, locked, maskIds, max = 6) {
  if (!cat) return [];
  const on = new Set(maskIds || []);
  const words = taskWords(spec);
  const ok = (m) => on.has(m.id) || (hasTopicName(m) && m.status !== 'rejected' && !locked(m) && !(words && words.test(hayOf(m))));
  const list = (cat.masks || []).filter(ok);
  list.sort((a, b) => (on.has(b.id) ? 1 : 0) - (on.has(a.id) ? 1 : 0) || evidenceRank(a) - evidenceRank(b));
  return list.slice(0, max);
}

/** "Forget medicine". */
export function forgetName(m) {
  return 'Forget ' + String(m.label || m.id).replace(/^(remove|forget)\s+/i, '').toLowerCase();
}

const n1 = (x) => String(Math.round(x * 1000) / 10);

const WHAT = [[/gsm8k|math/i, 'grade-school math questions'], [/mmlu/i, 'general knowledge questions'], [/humaneval|mbpp|code/i, 'coding problems'], [/arc/i, 'science questions'], [/hellaswag/i, 'common-sense questions'], [/legal|law/i, 'legal questions'], [/med|health/i, 'medical questions']];
const TASK_WHAT = { math: 'math questions', code: 'coding problems', legal: 'legal questions', medicine: 'medical questions' };

/** A plain description of what a benchmark asks, or ''. */
export function whatIs(name, spec) {
  for (const [re, w] of WHAT) if (re.test(String(name))) return w;
  return (spec && TASK_WHAT[spec.task_type]) || '';
}

/** The plain result of one scored attempt, or null. Reads the same fields builder.js reads.
 *  The main score is the target benchmark, else the first benchmark with a score, else the first score. */
export function resultOf(spec, attempt) {
  if (!attempt) return null;
  if (attempt.status && attempt.status !== 'ok') return { failed: true, status: attempt.status };
  const sc = attempt.scores || {};
  const bs = (spec && spec.benchmarks) || [];
  let t = bs.find((b) => b.kind === 'target' && sc[b.id] != null) || bs.find((b) => sc[b.id] != null);
  if (!t) {
    const k = Object.keys(sc).find((x) => sc[x] != null);
    if (!k) return { failed: false, empty: true, lines: ['The test finished, but it gave no scores. Open Advanced mode to see the details.'] };
    t = { id: k, name: k };
  }
  const score = sc[t.id];
  const rows = (attempt.comparison && attempt.comparison.rows) || [];
  const rnd = rows.find((r) => r.id === 'random');
  const baseRow = rows.find((r) => r.id === 'base');
  const base = t.base != null ? t.base : baseRow && baseRow.measured ? baseRow.value : null;
  const random = rnd && rnd.measured && rnd.value != null ? rnd.value : attempt.control && attempt.control.scores && attempt.control.scores[t.id] != null ? attempt.control.scores[t.id] : null;
  const margin = attempt.comparison && attempt.comparison.margin != null ? attempt.comparison.margin : 0.02;
  const what = whatIs(t.name, spec);
  const lines = ['Your model gets ' + n1(score) + '% right on ' + t.name + (what ? ' (' + what + ')' : '') + '.'];
  if (base != null) lines.push('The starting model gets ' + n1(base) + '%.');
  let meets = null;
  if (t.target != null) {
    meets = score >= t.target;
    lines.push(meets ? 'That meets the target of ' + n1(t.target) + '%.' : 'That is below the target of ' + n1(t.target) + '%.');
  }
  let helps = null;
  if (random != null) {
    const d = score - random;
    helps = d > margin ? 'yes' : d > 0 ? 'small' : 'no';
    let tail = helps === 'yes' ? 'so these skills help' : helps === 'small' ? 'so the gain is small, and may not come from these skills' : 'so these skills do not help yet';
    if (helps === 'yes' && meets === false) tail += ', but the model is still short of the target';
    lines.push('Random parts of the same size get ' + n1(random) + '%, ' + tail + '.');
  } else if (rnd) {
    lines.push('Random parts of the same size were not scored, so we cannot say yet that these skills help.');
  }
  const verdict = meets === true ? 'Meets your target' : meets === false ? 'Not there yet' : 'Test finished';
  const next = meets === false ? 'Next, try adding another skill, or hand your model to the agents in the last step.' : null;
  return { failed: false, name: t.name, what, score, base, target: t.target, meets, random, helps, verdict, next, lines };
}

const FAIL_TEXT = {
  collapsed: 'The model lost a skill it had before, so it was not scored. Try fewer skills or turn on smoothing.',
  infra_error: 'The test could not run. Nothing was scored. Try again.',
  compile_error: 'These skills could not be put together. Try removing one.',
  rejected_fit: 'One skill does not fit this model well enough to test. Try removing it.',
  invalid: 'This combination is not valid. Try removing a skill.',
  budget_exhausted: 'There are no tests left for this session.',
};
export function failText(status) { return FAIL_TEXT[status] || 'The test did not finish (' + status + ').'; }

/** Progress text while a test runs. */
export function progressText(job) {
  if (!job) return '';
  if (job.status === 'queued') return 'Getting ready to test your model.';
  return 'Testing your model. This can take a few minutes.';
}

const fmtB = (x) => (x == null ? null : (x < 0.01 && x > 0 ? Math.round(x * 1000) + 'M' : Number(x.toFixed(2)) + 'B'));

/** The size line and the over-the-limit warnings. */
export function sizeLine(totalB, runB, lim, healMin) {
  const parts = [];
  const warn = [];
  parts.push('Your model is ' + fmtB(totalB) + ' parameters' + (lim && lim.max_total_params_b != null ? ', and the limit is ' + fmtB(lim.max_total_params_b) : '') + '.');
  if (lim && lim.max_total_params_b != null && totalB > lim.max_total_params_b + 1e-6) warn.push('It is over the size limit. Remove a skill to fit.');
  if (lim && lim.max_run_params_b != null && runB != null && runB > lim.max_run_params_b + 1e-6) warn.push('It uses more compute per word than the limit allows. Remove a skill.');
  if (lim && lim.max_heal_min != null && healMin != null && healMin > lim.max_heal_min + 1e-6) warn.push('Smoothing would take longer than the limit of ' + lim.max_heal_min + ' minutes. Turn it off or remove a skill.');
  return { text: parts.join(' '), warn };
}

/** The smoothing sentence. The time shows only when smoothing is on. */
export function polishText(healMin, on) {
  if (!on) return 'Off. Testing is faster, but the added skills may score lower. We suggest turning it on when you add skills.';
  const t = healMin == null ? '' : healMin < 1 ? ' It takes under a minute more.' : ' It takes about ' + Math.round(healMin) + ' more minute' + (Math.round(healMin) === 1 ? '' : 's') + '.';
  return 'The model trains briefly so the new skills work well together. This usually raises the score.' + t;
}

/** The builder mode from the URL query, the saved choice, then the default. */
export function pickMode(queryMode, saved) {
  if (queryMode === 'advanced' || queryMode === 'simple') return queryMode;
  if (saved === 'advanced' || saved === 'simple') return saved;
  return 'simple';
}

/** The part's own label with the donor name taken out, short, or ''. */
export function labelDetail(part) {
  const d = donorName(part.donor);
  let t = String(part.label || '');
  if (d) t = t.split(d).join('');
  t = t.replace(/\bL\d+(\s*-\s*\d+)?\b/gi, '').replace(/\blayers?\s*\d+(\s*-\s*\d+)?\b/gi, '');
  t = t.replace(/^[\s,:;\-_/]+|[\s,:;\-_/]+$/g, '').replace(/\s+/g, ' ');
  return t.length > 30 ? t.slice(0, 29).trim() : t;
}

/** Plain names for a list of parts. Parts that would share a name get a detail from their label or size,
 *  and "(option 2)" only when nothing else tells them apart. */
export function plainNames(parts, spec) {
  const base = parts.map((p) => plainName(p, spec));
  const groups = {};
  base.forEach((n, i) => { (groups[n] = groups[n] || []).push(i); });
  const out = base.slice();
  Object.keys(groups).forEach((n) => {
    const ix = groups[n];
    if (ix.length < 2) return;
    const tries = [(p) => labelDetail(p), (p) => (labelDetail(p) ? labelDetail(p) + ', ' : '') + (paramsText(p.params_m) || '').replace(' parameters', ''), (p) => paramsText(p.params_m) && paramsText(p.params_m).replace(' parameters', '')];
    for (const f of tries) {
      const d = ix.map((i) => f(parts[i]));
      if (d.every(Boolean) && new Set(d).size === d.length) { ix.forEach((i, k) => { out[i] = n + ', ' + d[k]; }); return; }
    }
    ix.forEach((i, k) => { out[i] = n + ' (option ' + (k + 1) + ')'; });
  });
  return out;
}
