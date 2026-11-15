// Seed data and agent narration for demo mode. mock.js builds the fixed demo objects from this and
// plays the live run. Nothing here talks to a network. The numbers are sample numbers for a
// 1.24B host (Llama-3.2-1B-Instruct); the one Demo pill in the top bar labels the whole app.

import { DEMO } from './demo-flag.js';
import { addRecent, getRecent, patchRecent } from './util.js';

export const DEMO_IDS = {
  specMath: 'sp_demo_math',
  specLegal: 'sp_demo_legal',
  specShrink: 'sp_demo_code',
  runLive: 'run_demo_live',
  runDone: 'run_demo_done',
  runEmpty: 'run_demo_empty',
  builder: 'b_demo',
  shrink: 'sh_demo',
  agent: 'planner-1',
};
export const DEMO_SHRINK_JOB = 'j_demo_code';

/* ---------- spec looks ---------- */
const GUARD_IF = { id: 'if_guard', name: 'IFEval', kind: 'guard', floor: 0.45, base: 0.55, recommended: true };
const GUARD_MMLU = { id: 'mmlu_guard', name: 'MMLU', kind: 'guard', floor: 0.4, base: 0.46, recommended: true };

const sc = (model, params_b, score, sourced, url) => ({ model, params_b, score, sourced, source_url: sourced ? url : null });

export const MATH_LOOK = {
  title: 'Phone math tutor',
  summary: 'Solves grade-school word problems and shows each step. Small enough to run on a phone.',
  budget: { evals: 80, compute_min: 90, min_evals: 5, max_evals: 120 },
  size: { max_params_b: 2.0, min_b: 0.5, max_b: 8.0, host_params_b: 1.24 },
  limits: { max_total_params_b: 2.0, max_run_params_b: 1.9, max_heal_min: 20 },
  benchmarks: [
    { id: 'gsm8k', name: 'GSM8K', kind: 'target', target: 0.68, base: 0.58, lora: 0.63, recommended: true, research: true,
      reference_scores: [
        sc('Qwen3-1.7B', 1.7, 0.75, true, 'https://arxiv.org/abs/2505.09388'),
        sc('Llama-3.2-3B', 3, 0.777, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct'),
        sc('Gemma-3-1B', 1, 0.628, true, 'https://arxiv.org/abs/2503.19786'),
        sc('SmolLM2-1.7B', 1.7, 0.48, false),
      ] },
    { ...GUARD_IF, research: true, reference_scores: [sc('Llama-3.2-3B', 3, 0.77, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct'), sc('Gemma-3-1B', 1, 0.8, true, 'https://arxiv.org/abs/2503.19786')] },
    { ...GUARD_MMLU, research: true },
  ],
  extraIngredients: ['qmath_l18_contrast512'],
};

export const LEGAL_LOOK = {
  title: 'Contract reader',
  summary: 'Reads contract clauses and answers questions about them. Keeps its general skills.',
  budget: { evals: 60, compute_min: 75, min_evals: 5, max_evals: 120 },
  size: { max_params_b: 2.0, min_b: 0.5, max_b: 8.0, host_params_b: 1.24 },
  limits: { max_total_params_b: 2.0, max_run_params_b: 1.9, max_heal_min: 20 },
  benchmarks: [
    { id: 'mmlupro:law', name: 'MMLU-Pro law', kind: 'target', target: 0.22, base: 0.12, lora: 0.17, recommended: true },
    { ...GUARD_IF },
    { ...GUARD_MMLU },
  ],
  extraIngredients: [],
};

export const CODE_SHRINK_SPEC = {
  title: 'Code-only Qwen3 model',
  prompt: 'Keep the code skill of Qwen/Qwen3-30B-A3B-Instruct-2507, under 17B',
  summary: 'Keeps the code skill of Qwen3-30B-A3B-Instruct-2507 and removes experts the code skill does not use. The result has to fit under 17B stored.',
  keep_experts: 64,
  heal_steps: 300,
  limits: { max_total_params_b: 17, max_run_params_b: 3.5, max_heal_min: 20 },
  benchmarks: [
    { id: 'humaneval', name: 'HumanEval', kind: 'target', target: 0.85, base: 0.915, recommended: true },
    { id: 'mbpp', name: 'MBPP', kind: 'target', target: 0.75, base: 0.844, recommended: true },
  ],
};

/* ---------- benchmark research ---------- */
export function mathResearch() {
  const steps = [
    { t: 0.8, kind: 'think', text: 'I will look for benchmarks that test step-by-step arithmetic, and check which ones a 1B model can run here.' },
    { t: 2.2, kind: 'search', text: 'Searched: grade-school math word problem benchmark small language models', provider: 'exa' },
    { t: 3.8, kind: 'read', text: 'GSM8K: Training Verifiers to Solve Math Word Problems', url: 'https://arxiv.org/abs/2110.14168', provider: 'exa' },
    { t: 5.6, kind: 'search', text: 'Searched: GSM8K scores for instruct models under 4B parameters', provider: 'exa' },
    { t: 7.4, kind: 'read', text: 'Llama 3.2 model card', url: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct', provider: 'exa' },
    { t: 9.0, kind: 'read', text: 'Qwen3 technical report', url: 'https://arxiv.org/abs/2505.09388', provider: 'exa' },
    { t: 10.8, kind: 'read', text: 'Gemma 3 technical report', url: 'https://arxiv.org/abs/2503.19786', provider: 'exa' },
    { t: 12.6, kind: 'think', text: 'GSM8K is the closest test I can run. MATH-500 is the usual paper benchmark, but it is far too hard for a 2B limit and Sapien has no runner for it.' },
    { t: 14.4, kind: 'search', text: 'Searched: instruction following benchmark to guard against losing general skills', provider: 'exa' },
    { t: 16.2, kind: 'read', text: 'Instruction-Following Evaluation for Large Language Models', url: 'https://arxiv.org/abs/2311.07911', provider: 'exa' },
    { t: 18.0, kind: 'pick', text: 'Picked GSM8K: it matches the request directly' },
    { t: 19.2, kind: 'pick', text: 'Picked IFEval: catches a build that stops following instructions' },
    { t: 20.4, kind: 'pick', text: 'Picked MMLU: catches a build that loses general knowledge' },
  ];
  const proposals = [
    { name: 'gsm8k', label: 'GSM8K', measures: '8,500 grade-school word problems that need two to eight steps.', why: 'The closest runnable test of a tutor that shows its work. Public scores exist for small models, so the target has a fair reference.', runnable_id: 'gsm8k', kind: 'target', suggested_target: 0.68, in_spec: true,
      reference_scores: MATH_LOOK.benchmarks[0].reference_scores },
    { name: 'if_guard', label: 'IFEval', measures: 'Short instructions with checkable rules, such as length and format.', why: 'A math tune can break instruction following. This guard shows if that happened.', runnable_id: 'if_guard', kind: 'guard', suggested_target: null, in_spec: true,
      reference_scores: MATH_LOOK.benchmarks[1].reference_scores },
    { name: 'mmlu_guard', label: 'MMLU', measures: 'Multiple-choice questions across 57 school and college subjects.', why: 'A cheap check that general knowledge is still there after the new parts go in.', runnable_id: 'mmlu_guard', kind: 'guard', suggested_target: null, in_spec: true,
      reference_scores: [sc('Llama-3.2-3B', 3, 0.63, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct'), sc('Qwen3-1.7B', 1.7, 0.62, false)] },
    { name: 'math_500', label: 'MATH-500', measures: 'Competition problems, much harder than grade school.', why: 'Used in most papers on this skill, but Sapien has no runner for it and a 2B model scores near the floor.', runnable_id: null, kind: 'target', suggested_target: null, in_spec: false,
      reference_scores: [sc('Qwen3-1.7B', 1.7, 0.73, true, 'https://arxiv.org/abs/2505.09388'), sc('Llama-3.2-3B', 3, 0.48, false)] },
  ];
  const sources = steps.filter((x) => x.kind === 'read').map((x) => ({ title: x.text, url: x.url }));
  return { dur: 21.5, steps, proposals, sources,
    reasoning: 'The request is a math tutor that runs on a phone and shows its work, so the main benchmark should test multi-step word problems directly. GSM8K does, and published scores exist for models of 1B to 3B parameters, which gives the target slider a fair reference.\n\nA math tune can cost general skills, so IFEval and MMLU go in as guards with floors just under the host score. MATH-500 is the usual paper benchmark, but it is too hard at this size and has no runner yet, so it stays as reference.' };
}

export function legalResearch() {
  const steps = [
    { t: 0.8, kind: 'think', text: 'I will look for legal reasoning benchmarks that a 1B model can run, and for scores of other small models.' },
    { t: 2.4, kind: 'search', text: 'Searched: legal reasoning benchmark small language models', provider: 'exa' },
    { t: 4.6, kind: 'read', text: 'LegalBench: a collaboratively built benchmark for legal reasoning', url: 'https://arxiv.org/abs/2308.11462', provider: 'exa' },
    { t: 7.0, kind: 'read', text: 'CUAD: an expert-annotated NLP dataset for legal contract review', url: 'https://arxiv.org/abs/2103.06268', provider: 'exa' },
    { t: 9.4, kind: 'search', text: 'Searched: MMLU-Pro law accuracy for models under 4B parameters', provider: 'exa' },
    { t: 11.8, kind: 'read', text: 'MMLU-Pro: a more robust and challenging multi-task benchmark', url: 'https://arxiv.org/abs/2406.01574', provider: 'exa' },
    { t: 14.2, kind: 'think', text: 'LegalBench and CUAD match contract reading best, but Sapien has no runner for either. The law subset of MMLU-Pro is the closest test I can run today.' },
    { t: 16.6, kind: 'search', text: 'Searched: instruction following guard for small instruct models', provider: 'exa' },
    { t: 18.8, kind: 'read', text: 'Instruction-Following Evaluation for Large Language Models', url: 'https://arxiv.org/abs/2311.07911', provider: 'exa' },
    { t: 21.2, kind: 'pick', text: 'Picked MMLU-Pro law: the closest runnable legal test' },
    { t: 22.6, kind: 'pick', text: 'Picked IFEval: guards instruction following' },
    { t: 23.8, kind: 'pick', text: 'Picked MMLU: guards general knowledge' },
  ];
  const proposals = [
    { name: 'mmlupro_law', label: 'MMLU-Pro law', measures: 'Hard multiple-choice law questions with ten options each.', why: 'The closest test Sapien can run for reading legal text. Ten options make guessing less useful.', runnable_id: 'mmlupro:law', kind: 'target', suggested_target: 0.22, in_spec: true,
      reference_scores: [sc('Qwen3-1.7B', 1.7, 0.2, false), sc('Llama-3.2-3B', 3, 0.16, false), sc('Gemma-3-1B', 1, 0.11, false)] },
    { name: 'if_guard', label: 'IFEval', measures: 'Short instructions with checkable rules.', why: 'A legal tune can break instruction following. This guard shows if that happened.', runnable_id: 'if_guard', kind: 'guard', suggested_target: null, in_spec: true,
      reference_scores: [sc('Llama-3.2-3B', 3, 0.77, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct')] },
    { name: 'mmlu_guard', label: 'MMLU', measures: 'Multiple-choice questions across 57 subjects.', why: 'A cheap check that general knowledge survives.', runnable_id: 'mmlu_guard', kind: 'guard', suggested_target: null, in_spec: true, reference_scores: [] },
    { name: 'legalbench', label: 'LegalBench', measures: '162 legal reasoning tasks written by lawyers.', why: 'The standard legal benchmark, but Sapien has no runner for it yet.', runnable_id: null, kind: 'target', suggested_target: null, in_spec: false,
      reference_scores: [sc('Llama-3.2-3B', 3, 0.5, false)] },
    { name: 'cuad', label: 'CUAD', measures: 'Finds 41 kinds of clauses in real contracts.', why: 'Matches contract review well, but it needs span extraction that Sapien cannot score yet.', runnable_id: null, kind: 'target', suggested_target: null, in_spec: false, reference_scores: [] },
  ];
  const sources = steps.filter((x) => x.kind === 'read').map((x) => ({ title: x.text, url: x.url }));
  return { dur: 25, steps, proposals, sources,
    reasoning: 'Reading a contract is closest to legal reasoning questions, so the main benchmark should be a law test. MMLU-Pro law is the nearest one Sapien can run, and its ten answer options make a lucky guess less likely.\n\nLegalBench and CUAD are better matches in spirit, but neither has a runner here, so they stay as reference. IFEval and MMLU guard against a build that learns law and forgets everything else.' };
}

/* ---------- narration ---------- */
const recentPick = new Map();
// A random item that was not one of the last few picked for this key, so lines do not repeat.
export function choose(key, list) {
  if (!list.length) return null;
  const last = recentPick.get(key) || [];
  const room = Math.min(last.length, Math.max(0, list.length - 1), 3);
  const ban = last.slice(-room);
  let i = Math.floor(Math.random() * list.length);
  for (let k = 0; k < 8 && ban.includes(i); k++) i = Math.floor(Math.random() * list.length);
  last.push(i);
  if (last.length > 4) last.shift();
  recentPick.set(key, last);
  return list[i];
}
const short = (donor) => String(donor || '').split('/').pop();
const layerText = (part) => (part.layers[0] === part.layers[1] ? 'L' + part.layers[0] : 'L' + part.layers[0] + '-' + part.layers[1]);
const p1 = (x) => (x * 100).toFixed(1);
const p0 = (x) => String(Math.round(x * 100));

const TASK_WORD = { math: 'math', code: 'code', legal: 'legal reasoning', general: 'general' };

/** What a planner writes when it starts a new attempt. o: { style, part, at, rank, steps, data, best, host, vs, fitHere, fitAlt, altAt, lastName, lastScore, mask, floor, task } */
export function planThought(o) {
  const { part } = o;
  if (!part) return 'Host-only floor with a ' + o.steps + ' step heal on ' + o.data + ' text. ' + p0(o.host) + '% is the number every part has to beat, so I want it measured on the same items.';
  const D = short(part.donor), P = part.label, DL = layerText(part), task = TASK_WORD[part.task] || 'general';
  const lead = o.best != null ? 'The leader is at ' + p1(o.best) + '% on GSM8K, ' + ((o.best - o.host) * 100).toFixed(1) + ' points over the host. ' : '';
  const fit = o.fitAlt != null ? 'The free fit probe gave ' + o.fitHere.toFixed(2) + ' at L' + o.at + ' against ' + o.fitAlt.toFixed(2) + ' at L' + o.altAt + '. ' : '';
  const by = {
    blocks: [
      lead + P + ' (' + D + ' ' + DL + ') is a whole donor block. ' + fit + 'I place it at L' + o.at + ' with glue rank ' + o.rank + ' and heal ' + o.steps + ' steps on ' + o.data + ' text.',
      P + ' from ' + D + ' scored ' + o.vs + ' points over the random control on practice, so I keep it and try a heal on ' + o.data + ' text. ' + fit.trim(),
      'Two blocks within 2 layers of each other put the IFEval guard at risk. This time ' + P + ' goes in alone at L' + o.at + ', and the saved parameters pay for a ' + o.steps + ' step heal.',
      lead + 'A whole donor block should carry more of the ' + task + ' skill than a cluster of neurons. ' + D + ' ' + DL + ' at L' + o.at + ' is my next test, glue rank ' + o.rank + '.',
    ],
    small: [
      P + ' adds only ' + part.params_m + 'M parameters, so it fits the size limit with room to spare. ' + D + ' ' + DL + ' reads like ' + task + ' circuitry, so I probe L' + o.at + ' first. The probe is free.',
      (o.lastName ? 'The last small part (' + o.lastName + ') scored ' + p1(o.lastScore) + '%. ' : '') + P + ' targets similar ' + task + ' units at L' + o.at + '. ' + fit.trim(),
      lead + P + ' scored ' + o.vs + ' points over the random control, which is the margin I need. I pair it with a ' + o.steps + ' step heal on ' + o.data + ' text so the glue can settle.',
      'Small parts are cheap to stack. ' + P + ' (' + D + ' ' + DL + ') goes at L' + o.at + ', two layers from the default, to see whether the fit curve is flat there.',
    ],
    masks: [
      o.mask ? 'Removing ' + o.mask + ' knowledge should free capacity for ' + task + '. I pair that deletion mask with ' + P + ' at L' + o.at + ' and check that IFEval stays above its ' + p0(o.floor) + '% floor.' : lead + P + ' at L' + o.at + ' with no mask, to see what the deletion was costing.',
      lead + P + ' from ' + D + ' has the best control margin so far (' + o.vs + ' points). I try it again at L' + o.at + ' ' + (o.mask ? 'with the ' + o.mask + ' mask on top.' : 'with a longer heal.'),
      'The last masked run kept the guards but gained nothing. ' + P + ' at L' + o.at + ' with glue rank ' + o.rank + ' is a cleaner test of the part alone.',
    ],
  };
  const list = by[o.style] || by.blocks;
  return choose('plan.' + o.style, list);
}

/** A planner's reaction to a finished eval. o: { name, score, leader, collapsed, guard, guardScore, floor, at, healSteps, otherAt } */
export function evalThought(o) {
  if (o.collapsed) {
    return choose('collapse', [
      o.guard + ' fell to ' + p1(o.guardScore) + '%, under its ' + p0(o.floor) + '% floor. The graft at L' + o.at + ' sits too near another part, so I move it three layers away and lower the glue rank.',
      'That one collapsed: ' + o.guard + ' dropped to ' + p1(o.guardScore) + '%. I read it as a placement problem at L' + o.at + ' and not a part problem, so the same part gets another layer.',
      'Guards failed (' + o.guard + ' ' + p1(o.guardScore) + '%, floor ' + p0(o.floor) + '%). The lesson is saved for the analyst. Next I try one part, no mask, heal ' + o.healSteps + ' steps.',
    ]);
  }
  const d = (o.score - o.leader) * 100;
  if (d >= 0) {
    return choose('win', [
      o.name + ' reached ' + p1(o.score) + '%, ' + d.toFixed(1) + ' points over the old leader. The heal loss was still falling, so I will try ' + (o.healSteps * 2) + ' steps next.',
      'New best at ' + p1(o.score) + '%. The guards barely moved, so the part is not hurting general skills. I send the program to the judge for a random-part control.',
    ]);
  }
  return choose('miss', [
    o.name + ' scored ' + p1(o.score) + '%, ' + Math.abs(d).toFixed(1) + ' points under the leader. Not worth a control run, but the placement at L' + o.at + ' goes on the map.',
    p1(o.score) + '% is below the leader by ' + Math.abs(d).toFixed(1) + ' points. The guards held, so I keep the part and change only the layer.',
    o.name + ' trails by ' + Math.abs(d).toFixed(1) + ' points (' + p1(o.score) + '%). A heal of ' + o.healSteps + ' steps may be too short for this part, so I raise it.',
  ]);
}

export function planLine(o) {
  return choose('planline', [
    'Proposed ' + o.count + ' programs for this round',
    o.count + ' new programs queued, ' + (o.refine ? 'one refines the current leader' : 'all from untried placements'),
    'Round ' + o.round + ': ' + o.count + ' programs planned, ' + o.parts + ' distinct parts',
  ]);
}

export function probeLine(o) {
  const t = o.fit >= 0.7 ? 'a good fit' : o.fit >= 0.55 ? 'an average fit' : 'a weak fit';
  return choose('probeline', [
    'Checked fit of ' + o.name + ': ' + o.fit.toFixed(2),
    'Fit of ' + o.name + ' is ' + o.fit.toFixed(2) + ', ' + t,
    'Probed ' + o.name + ' at L' + o.at + ': ' + o.fit.toFixed(2),
  ]);
}

export function evalLine(o) {
  const g = o.target;
  if (o.collapsed) return o.name + ' collapsed: ' + o.guard + ' fell to ' + p1(o.guardScore) + '%, under the ' + p0(o.floor) + '% floor. The lesson is saved.';
  if (o.isBest) {
    return choose('evalbest', [
      o.name + ' scored ' + p1(g) + '% on GSM8K' + (o.prev != null ? ', ' + ((g - o.prev) * 100).toFixed(1) + ' points above the old leader' : '') + '. New best.',
      'Scored ' + o.name + ' at ' + p1(g) + '% on GSM8K. It leads the board.',
    ]);
  }
  return choose('evalmiss', [
    'Scored ' + o.name + ' at ' + p1(g) + '% on GSM8K.',
    o.name + ' reached ' + p1(g) + '%, ' + (((o.best || g) - g) * 100).toFixed(1) + ' points under the leader.',
    o.name + ' came in at ' + p1(g) + '% on GSM8K. Guards held.',
  ]);
}

export function mapLine(o) {
  const list = [
    'Map v' + o.v + ': ' + o.donor + ' parts near L' + o.at + ' lead, ' + o.good + ' of ' + o.total + ' scored above ' + p0(o.thr) + '%. ' + o.bad + (o.bad === 1 ? ' collapse' : ' collapses') + ' so far, mostly stacked parts.',
    'Placement note: ' + o.partName + ' fits best at L' + o.def + '. Three or more layers away it loses about ' + o.loss + ' points of fit.',
  ];
  if (o.heal600 != null && o.heal100 != null) list.push('Heal length matters: runs with 600 steps average ' + p1(o.heal600) + '% on GSM8K against ' + p1(o.heal100) + '% for runs with 100 steps.');
  if (o.pair != null) list.push('Pairs beat singles: the two best programs each use a block plus a neuron cluster, and they average ' + p1(o.pair) + '%. Single parts average ' + p1(o.single) + '%.');
  return choose('map', list);
}

export function controlLine(o) {
  return choose('control', [
    'Control for ' + o.name + ': random parts of the same size reach ' + p1(o.control) + '%, the real build ' + p1(o.score) + '%. The parts add ' + o.pts + ' points.',
    'The random-part control for ' + o.name + ' says the part adds ' + o.pts + ' points (' + p1(o.score) + '% against ' + p1(o.control) + '%).',
  ]);
}

export function nudgeThought(text) {
  return 'Nudge from you: "' + text + '". I will use it for this attempt.';
}

/* ---------- recents ---------- */
// The sidebar and the Runs page list what was opened in this browser. Demo mode fills them with the demo objects.
export function seedDemoRecents() {
  if (!DEMO) return;
  const have = getRecent();
  if (have.some((r) => r.id === DEMO_IDS.runLive || r.id === DEMO_IDS.specMath)) return;
  const now = Date.now();
  const min = 60 * 1000;
  // oldest first: addRecent puts each new item at the top
  const items = [
    { kind: 'spec', id: DEMO_IDS.specShrink, title: CODE_SHRINK_SPEC.title, ago: 26 * 60 },
    { kind: 'run', id: DEMO_IDS.runEmpty, title: 'Contract reader', status: 'empty', evals: 0, total: 60, ago: 22 * 60 },
    { kind: 'spec', id: DEMO_IDS.specMath, title: MATH_LOOK.title, ago: 5 * 60 + 40 },
    { kind: 'run', id: DEMO_IDS.runDone, title: MATH_LOOK.title, status: 'done', evals: 80, total: 80, ago: 4 * 60 + 12 },
    { kind: 'shrink', id: DEMO_IDS.shrink, spec: DEMO_IDS.specShrink, title: 'Shrink: ' + CODE_SHRINK_SPEC.title, status: 'done', job: DEMO_SHRINK_JOB, ago: 60 + 8 },
    { kind: 'spec', id: DEMO_IDS.specLegal, title: LEGAL_LOOK.title, ago: 11 },
    { kind: 'run', id: DEMO_IDS.runLive, title: MATH_LOOK.title, status: 'running', evals: 32, total: 80, ago: 3 },
  ];
  for (const { ago, ...it } of items) {
    addRecent(it);
    patchRecent(it.kind, it.id, { ts: now - ago * min });
  }
}
