// Mock data for ?mock=1: measured results and the compiler front door. Same shapes as the server (CONTRACT.md).
// The measured numbers are the ones in studio/server/measured.py. A test keeps the two in step.

import { DEMO, HOSTED, demoNow, wallOf, storageKey } from './demo-flag.js';
import { LANG } from './demo-data.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clone = (x) => JSON.parse(JSON.stringify(x));

const CAVEATS = [
  'Each score is from one run, so a gap of a few points may be noise.',
  'Two rows got a 500-step fine-tune after the cut, learning to match the full model on code text. The other rows got no training.',
  'Even fine-tuned, the 32-expert model (73.8 and 61.9) scores below Qwen2.5-Coder-7B (85.4 and 86.0), a coding model of about the same size.',
  'Cutting experts makes the model smaller to store but not faster. Each token still runs through about 3.0B parameters, against 7.6B for Qwen2.5-Coder-7B.',
  'Other skills drop. General knowledge (MMLU) falls from 77.8 to 44.9 with 64 experts kept.',
  'All 32-expert rows are the same size, 8.78B, because size depends only on how many experts are kept.',
];
const row = (label, experts_kept, rule, total_params_b, scores, is_control = false, is_full = false, heal_steps = 0) => ({ label, experts_kept, rule, total_params_b, scores, is_control, is_full, heal_steps });

const MEASURED = [{
  id: 'run19_qwen3_30b_a3b_code', title: 'Skill removal on Qwen3-30B-A3B, code only', kind: 'skill_removal', skill: 'code',
  model: 'Qwen/Qwen3-30B-A3B-Instruct-2507', experts_per_layer: 128, params_run_per_token_b: 3.03, date: '2026-11-14',
  scorer: 'wsp.skills_eval', runs_per_cell: 1, heal: '500 steps on two rows', source: 'compiler session, results/README.md run 19, 2026-11-14', caveats: CAVEATS,
  rows: [
    row('Full model', 128, 'none', 30.5, { humaneval: 91.5, mbpp: 84.4, gsm8k: 91.5, mmlu: 77.8 }, false, true),
    row('Keep 64 by code saliency', 64, 'saliency', 16.0, { humaneval: 87.8, mbpp: 79.0, gsm8k: 80.5, mmlu: 44.9 }),
    row('Keep 32 by code saliency', 32, 'saliency', 8.78, { humaneval: 57.9, mbpp: 55.3, gsm8k: 2.5, mmlu: 26.1 }),
    row('Keep 32 by code saliency, 500-step heal', 32, 'saliency', 8.78, { humaneval: 73.8, mbpp: 61.9, gsm8k: 2.0, mmlu: 25.6 }, false, false, 500),
    row('Keep 16 by code saliency', 16, 'saliency', 5.15, { humaneval: 3.7, mbpp: 7.8, gsm8k: 0.5, mmlu: 23.6 }),
    row('Keep 32 by routing frequency', 32, 'freq', 8.78, { humaneval: 7.9, mbpp: 11.7, gsm8k: 0.0, mmlu: 24.4 }),
    row('Keep 32 random (three seeds)', 32, 'random', 8.78, { humaneval: 0.0, mbpp: 0.0 }, true),
    row('Keep 32 random, 500-step heal (seed 1)', 32, 'random', 8.78, { humaneval: 11.6, mbpp: 23.3, gsm8k: 2.0, mmlu: 24.1 }, true, false, 500),
    row('Keep 32 by skill-to-general ratio', 32, 'selective', 8.78, { humaneval: 0.0, mbpp: 0.0, gsm8k: 0.0, mmlu: 23.9 }, true),
  ],
}];

const LROW = (label, params, arr, extra = {}) => ({ label, params_per_token_b: params, scores: Object.fromEntries(LANG.codes.map((c, i) => [c, arr[i]])), ...extra });
MEASURED.push({
  id: 'language_writein_2026_11_15', title: 'Five languages from Qwen2.5-7B into Llama-3.2-1B', kind: 'language_writein', skill: 'multilingual',
  model: 'unsloth/Llama-3.2-1B-Instruct', donor: 'Qwen/Qwen2.5-7B-Instruct', donor_layers: [4, 20], date: '2026-11-15',
  scorer: 'Belebele reading comprehension, 900 test items per language, never trained on', runs_per_cell: 3, source: 'results/README.md, language write-in, 2026-11-15',
  columns: LANG.codes.map((c) => ({ id: c, label: LANG.names[c] })),
  try_text: 'Teach Llama-3.2-1B to understand Arabic, Japanese, Chinese, Korean and Hindi',
  caveats: [
    'The graft is the mean of 3 seeds. The scrambled-donor control is the mean of 2 seeds.',
    'The LoRA and full fine-tune arms are still being scored, so there is no row for them yet.',
    'Qwen2.5-3B stays 2 to 6 points ahead of the graft in four languages, at fewer parameters per token.',
    'Languages with no training text rose too: Spanish 51.2 to 85.0, French 52.2 to 86.7, Bengali 34.0 to 54.9, Swahili 36.0 to 61.2.',
  ],
  rows: [
    LROW('Llama-3.2-1B base', 1.24, LANG.base, { is_full: true }),
    LROW('Same graft, donor weights scrambled (2 seeds)', LANG.graft_b, LANG.scrambled, { is_control: true }),
    LROW('Base + Qwen2.5-7B layers 4 to 20 (3 seeds)', LANG.graft_b, LANG.graft),
    LROW('Native Qwen2.5-1.5B', 1.54, LANG.qwen15),
    LROW('Native Qwen2.5-3B', 3.09, LANG.qwen3),
    LROW('Qwen2.5-7B donor alone', 7.62, LANG.donor),
  ],
});

const DATASET = { gsm8k: 'openai/gsm8k', mmlu: 'cais/mmlu', mmlu_guard: 'cais/mmlu', mmlu_es: 'cais/mmlu', humaneval: 'openai/openai_humaneval', mbpp: 'google-research-datasets/mbpp', if_guard: 'google/IFEval', arc: 'allenai/ai2_arc', hellaswag: 'Rowan/hellaswag', truthfulqa: 'truthfulqa/truthful_qa' };
const datasetOf = (id) => DATASET[id] || (id.startsWith('mmlupro:') ? 'TIGER-Lab/MMLU-Pro' : id.startsWith('belebele:') ? 'facebook/belebele' : id.startsWith('bbeh') ? 'BBEH/bbeh' : null);
const SKILL = { gsm8k: 'math', 'mmlupro:math': 'math', humaneval: 'code', mbpp: 'code', mmlu: 'knowledge', mmlu_guard: 'knowledge', 'mmlupro:health': 'medicine', 'mmlupro:law': 'law', 'mmlupro:chemistry': 'science', 'mmlupro:biology': 'science', arc: 'knowledge', mmlu_es: 'multilingual', bbeh_mini: 'reasoning', if_guard: 'general', hellaswag: 'general', truthfulqa: 'general' };
const skillOfBench = (id) => SKILL[id] || (id.startsWith('belebele:') ? 'multilingual' : null);
const TASK_SKILL = { math: 'math', code: 'code', legal: 'law', medicine: 'medicine', multilingual: 'multilingual', science: 'science', general: 'knowledge', knowledge: 'knowledge', support: 'instruction following', creative: 'writing', compact: 'knowledge' };
const slug = (t) => (String(t || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'sapien_request');

function requestOf(spec) {
  const reported = [];
  const unmapped = [];
  for (const b of spec.benchmarks || []) {
    const d = datasetOf(b.id);
    if (d == null) { if (!unmapped.includes(b.name || b.id)) unmapped.push(b.name || b.id); } else if (!reported.includes(d)) reported.push(d);
  }
  const lim = spec.limits || {};
  const req = { name: slug(spec.title) };
  if (spec.request_type === 'shrink' && spec.shrink) {
    req.base = spec.shrink.source_model; req.add = []; req.remove = [{ skill: '*' }]; req.keep = [spec.shrink.keep_skill];
  } else {
    req.base = spec.host;
    const skills = [];
    for (const b of spec.benchmarks || []) if (b.kind === 'target') { const sk = skillOfBench(b.id) || 'general'; if (!skills.includes(sk)) skills.push(sk); }
    if (!skills.length) skills.push(TASK_SKILL[spec.task_type] || 'knowledge');
    const maskOnly = (spec.ingredients || []).length > 0 && (spec.ingredients || []).every((g) => g.kind === 'deletion mask');
    req.add = maskOnly ? [] : skills.map((skill) => {
      const word = skill === 'code' ? 'code' : skill === 'law' ? 'law' : skill;
      const found = (spec.ingredients || []).filter((g) => g.donor !== spec.host && (g.label + ' ' + g.donor).toLowerCase().includes(word));
      return { skill, sources: found.length ? [...new Set(found.map((g) => g.donor))] : 'auto', components: null };
    });
    req.remove = maskOnly ? (spec.ingredients || []).map((g) => ({ skill: g.label.replace(/^Remove /, '').toLowerCase() })) : [];
    req.keep = [...new Set((spec.benchmarks || []).filter((b) => b.kind === 'guard' && skillOfBench(b.id)).map((b) => skillOfBench(b.id)))];
  }
  req.limits = { max_total_params_b: lim.max_total_params_b ?? spec.size.max_params_b, max_active_params_b: lim.max_run_params_b ?? null, max_heal_minutes: lim.max_heal_min ?? null };
  if (reported.length) req.reported = reported;
  req.seed = 0;
  return { request: req, unmapped };
}

function errorsOf(req) {
  if (!req.add.length && !req.remove.length) return ['the request needs something to do: add or remove must not both be empty'];
  return [];
}

const fam = (m) => String(m || '').split('/').pop().split(/[-_.\d]/)[0].toLowerCase();
const r2 = (x) => Math.round(x * 100) / 100;
function planOf(req, spec, parts) {
  const sh = spec.request_type === 'shrink' ? spec.shrink : null;
  if (sh) {
    const total = sh.target_params_b != null ? sh.target_params_b : Math.min(req.limits.max_total_params_b || 16.9, 16.9);
    const run = sh.source_active_params_b != null ? sh.source_active_params_b : 3.04;
    const problems = req.limits.max_total_params_b != null && total > req.limits.max_total_params_b + 1e-6 ? ['Over the total size limit. The shrunk model is estimated at ' + r2(total) + 'B parameters. The limit is ' + r2(req.limits.max_total_params_b) + 'B.'] : [];
    return { name: req.name, base: { id: req.base, params_b: sh.source_params_b, active_params_b: run, moe: sh.family === 'moe' }, limits: req.limits, keep: req.keep, remove: req.remove, sources: [],
      stages: [{ order: 1, method: 'slim', kind: 'plain', params_total_after_b: r2(total), params_run_after_b: run, limit_problems: problems }],
      arms: [{ arm: 'base', what: 'the base model alone' }, { arm: 'base_lora', what: 'the base plus a LoRA' }],
      warnings: [], refusals: [], feasible: !problems.length };
  }
  const hostB = (spec.size && spec.size.host_params_b) || 1.24;
  const base = { id: req.base, params_b: hostB, active_params_b: hostB, moe: false };
  const byId = new Map(parts.map((p) => [p.id, p]));
  const ings = (spec.ingredients || []).map((g) => ({ g, p: byId.get(g.part_id) })).filter((x) => x.p && x.g.kind !== 'deletion mask');
  const room = Math.max(0, (req.limits.max_total_params_b || 99) - hostB);
  let added = 0;
  const sources = req.add.map((a) => {
    const mine = ings.filter((x) => (Array.isArray(a.sources) ? a.sources.includes(x.p.donor) : false)).slice(0, 1);
    const pick = mine.length ? mine : ings.slice(0, 1);
    const donor = pick.length ? pick[0].p.donor : (parts.find((p) => (p.skills || []).includes(a.skill)) || {}).donor || 'the library';
    const adds = r2(pick.reduce((x, y) => x + (y.p.params_m || 0), 0) / 1000);
    added += adds;
    const same = fam(donor) === fam(req.base);
    return { skill: a.skill, source: donor, relation: same ? 'same_family' : 'other_family', chosen: 'synth',
      methods: [{ method: 'synth', fits: adds <= room + 1e-6, adds_total_b: adds, reason: same ? 'same family: donor layers go in with a short glue layer' : 'other family: donor layers go through a seam with translation layers' }] };
  });
  const total = r2(hostB + added);
  const problems = [];
  if (total > (req.limits.max_total_params_b || 99) + 1e-6) problems.push('Over the total size limit. This build is estimated at ' + total + 'B parameters (the host is ' + hostB + 'B and the added parts are ' + r2(added) + 'B). The limit is ' + r2(req.limits.max_total_params_b) + 'B.');
  if (req.remove.length) return { name: req.name, base, limits: req.limits, keep: req.keep, remove: req.remove, sources: [],
    stages: [{ order: 1, method: 'mask', kind: 'plain', params_total_after_b: hostB, params_run_after_b: hostB, limit_problems: [] }],
    arms: [{ arm: 'base', what: 'the base model alone' }, { arm: 'base_lora', what: 'the base plus a LoRA' }], warnings: [], refusals: [], feasible: true };
  return { name: req.name, base, limits: req.limits, keep: req.keep, remove: [], sources,
    stages: [{ order: 1, method: 'synth', kind: 'seam', params_total_after_b: total, params_run_after_b: total, limit_problems: problems }],
    arms: [{ arm: 'base', what: 'the base model alone' }, { arm: 'base_lora', what: 'the base plus a LoRA' }, { arm: 'base_distill', what: 'the base plus distillation' }],
    warnings: [], refusals: [], feasible: !problems.length };
}

export function makeExtra({ getSpec, getParts, getRecipes, getMasks }) {
  const mine = makeMine({ getParts, getRecipes, getMasks });
  return {
    ...mine,
    async measured() { await sleep(150); return clone(MEASURED); },
    async specRequest(id) {
      await sleep(150);
      const { request, unmapped } = requestOf(getSpec(id));
      return { request, unmapped, errors: errorsOf(request) };
    },
    async specPlan(id) {
      await sleep(900);
      const { request, unmapped } = requestOf(getSpec(id));
      return { plan: planOf(request, getSpec(id), getParts()), seconds: 0.9, offline: !DEMO, offline_reason: DEMO ? null : 'mock data', verified: false, unmapped,
        note: DEMO ? '' : "The compiler's plan is new and its lineage check is still being fixed. Treat it as a draft." };
    },
  };
}

/* ---------- mining: the queue, the mined models and the part details (demo and mock mode) ---------- */
// Models and parts come from the parts library (the real one in demo mode). The queue is a simulation:
// a request has a start time and a script, and its state is a pure function of the time since it started.

const hashStr = (s) => { let h = 2166136261; for (const c of String(s)) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; };
const rngOf = (seed) => { let a = hashStr(seed); return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
const encPath = (s) => String(s).split('/').map(encodeURIComponent).join('/');
const short = (m) => String(m || '').split('/').pop();

// facts that are public knowledge about common open models: layers, parameters in billions, experts per layer
const K = (layers, params_b, experts = 0) => ({ layers, params_b, experts, moe: experts > 0 });
const KNOWN_RAW = {
  'LiquidAI/LFM2-8B-A1B': K(24, 8.3, 32), 'LiquidAI/LFM2.5-8B-A1B': K(24, 8.3, 32), 'LiquidAI/LFM2-24B-A2B': K(40, 24, 64), 'LiquidAI/LFM2-1.2B': K(16, 1.17), 'LiquidAI/LFM2-700M': K(16, 0.74), 'LiquidAI/LFM2-350M': K(16, 0.35), 'LiquidAI/LFM2-350M-Math': K(16, 0.35),
  'LiquidAI/LFM2.5-1.2B-Instruct': K(16, 1.17), 'LiquidAI/LFM2.5-1.2B-Thinking': K(16, 1.17), 'LiquidAI/LFM2.5-2.6B': K(30, 2.6), 'LiquidAI/LFM2-2.6B': K(30, 2.6), 'LiquidAI/LFM3-1.7B': K(24, 1.7),
  'Qwen/Qwen3-30B-A3B': K(48, 30.5, 128), 'Qwen/Qwen3-30B-A3B-Instruct-2507': K(48, 30.5, 128), 'Qwen/Qwen3-235B-A22B': K(94, 235, 128), 'Qwen/Qwen3.5-35B-A3B': K(40, 35, 256),
  'Qwen/Qwen3-0.6B': K(28, 0.6), 'Qwen/Qwen3-1.7B': K(28, 1.7), 'Qwen/Qwen3-4B': K(36, 4.0), 'Qwen/Qwen3-8B': K(36, 8.2), 'Qwen/Qwen3-14B': K(40, 14.8), 'Qwen/Qwen3-32B': K(64, 32.8),
  'Qwen/Qwen2.5-0.5B-Instruct': K(24, 0.49), 'Qwen/Qwen2.5-1.5B-Instruct': K(28, 1.54), 'Qwen/Qwen2.5-3B-Instruct': K(36, 3.09), 'Qwen/Qwen2.5-7B-Instruct': K(28, 7.62), 'Qwen/Qwen2.5-14B-Instruct': K(48, 14.77), 'Qwen/Qwen2.5-32B-Instruct': K(64, 32.76), 'Qwen/Qwen2.5-72B-Instruct': K(80, 72.7),
  'Qwen/Qwen2.5-Coder-0.5B-Instruct': K(24, 0.49), 'Qwen/Qwen2.5-Coder-1.5B-Instruct': K(28, 1.54), 'Qwen/Qwen2.5-Coder-3B-Instruct': K(36, 3.09), 'Qwen/Qwen2.5-Coder-7B-Instruct': K(28, 7.62), 'Qwen/Qwen2.5-Coder-14B-Instruct': K(48, 14.77),
  'Qwen/Qwen2.5-Math-1.5B': K(28, 1.54), 'Qwen/Qwen2.5-Math-1.5B-Instruct': K(28, 1.54), 'Qwen/Qwen2.5-Math-7B-Instruct': K(28, 7.62),
  'Qwen/Qwen3.5-0.8B': K(24, 0.8), 'Qwen/Qwen3.5-2B': K(24, 2), 'Qwen/Qwen3.5-4B': K(32, 4), 'Qwen/Qwen3.5-9B': K(32, 9), 'Qwen/Qwen3-VL-8B-Instruct': K(36, 8.8),
  'meta-llama/Llama-3.2-1B-Instruct': K(16, 1.24), 'meta-llama/Llama-3.2-3B-Instruct': K(28, 3.21), 'meta-llama/Llama-3.1-8B-Instruct': K(32, 8.03), 'meta-llama/Llama-3.1-70B-Instruct': K(80, 70.6), 'meta-llama/Llama-3.3-70B-Instruct': K(80, 70.6), 'unsloth/Meta-Llama-3.1-8B-Instruct': K(32, 8.03),
  'mistralai/Mistral-7B-Instruct-v0.3': K(32, 7.25), 'mistralai/Mixtral-8x7B-Instruct-v0.1': K(32, 46.7, 8),
  'google/gemma-3-1b-it': K(26, 1.0), 'google/gemma-3-4b-it': K(34, 4.3), 'google/gemma-3-12b-it': K(48, 12.2), 'google/gemma-3-27b-it': K(62, 27.4), 'unsloth/gemma-3-1b-it': K(26, 1.0), 'google/gemma-2-2b-it': K(26, 2.6), 'unsloth/gemma-2-2b-it': K(26, 2.6), 'google/gemma-2-9b-it': K(42, 9.24), 'unsloth/gemma-2-9b-it': K(42, 9.24),
  'google/gemma-4-26B-A4B-it': K(30, 26, 128), 'google/gemma-4-31B-it': K(60, 31), 'google/gemma-4-E2B-it': K(35, 5.1), 'unsloth/medgemma-4b-it': K(34, 4.3),
  'microsoft/Phi-4': K(40, 14.7), 'microsoft/phi-4': K(40, 14.7), 'microsoft/Phi-3.5-mini-instruct': K(32, 3.82), 'microsoft/Phi-3-mini-4k-instruct': K(32, 3.82),
  'deepseek-ai/DeepSeek-V3': K(61, 671, 256), 'deepseek-ai/DeepSeek-R1': K(61, 671, 256), 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': K(28, 1.78), 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': K(28, 7.62), 'deepseek-ai/deepseek-coder-6.7b-instruct': K(32, 6.74),
  'openai/gpt-oss-20b': K(24, 20.9, 32), 'openai/gpt-oss-120b': K(36, 116.8, 128),
  'allenai/OLMo-2-1124-7B-Instruct': K(32, 7.3), 'allenai/OLMoE-1B-7B-0125-Instruct': K(16, 6.92, 64),
  'ibm-granite/granite-3.3-2b-instruct': K(40, 2.5), 'ibm-granite/granite-3.3-8b-instruct': K(40, 8.17),
  'HuggingFaceTB/SmolLM2-1.7B-Instruct': K(24, 1.71), 'Equall/Saul-7B-Instruct-v1': K(32, 7.24), 'AdaptLLM/law-chat': K(32, 6.74), 'Intelligent-Internet/II-Medical-8B': K(36, 8.2), 'm42-health/Llama3-Med42-8B': K(32, 8.03), 'FreedomIntelligence/Apollo2-7B': K(28, 7.62),
  'Nanbeige/Nanbeige4.2-3B': K(24, 3), 'Nanbeige/Nanbeige4.3-33B': K(56, 33),
};
const KNOWN = {};
for (const [k, v] of Object.entries(KNOWN_RAW)) KNOWN[k.toLowerCase()] = v;
const knownOf = (m) => KNOWN[String(m || '').toLowerCase()] || null;
const LICENSE = [[/^qwen\//i, 'Apache 2.0'], [/^liquidai\//i, 'LFM Open License'], [/^mistralai\//i, 'Apache 2.0'], [/^ibm-granite\//i, 'Apache 2.0'], [/^deepseek-ai\//i, 'MIT'], [/^microsoft\//i, 'MIT'], [/^google\/gemma|^unsloth\/gemma/i, 'Gemma terms'], [/^(meta-llama|unsloth)\/.*llama/i, 'Llama community license'], [/^openai\/gpt-oss/i, 'Apache 2.0'], [/^allenai\//i, 'Apache 2.0']];
const licenseOf = (m) => { const r = LICENSE.find((x) => x[0].test(m)); return r ? r[1] : 'Open weights'; };

/** Parameters in billions from a name such as "Qwen3-30B-A3B" or "LFM2-350M". null when the name has no size. */
function paramsOf(model) {
  const k = knownOf(model);
  if (k) return k.params_b;
  const n = short(model);
  const nx = n.match(/(\d+)x(\d+(?:\.\d+)?)B/i);
  if (nx) return +(nx[1] * nx[2]).toFixed(1);
  const b = n.match(/(\d+(?:\.\d+)?)B(?![a-z])/i);
  if (b) return +b[1];
  const m = n.match(/(\d+(?:\.\d+)?)M(?![a-z])/i);
  return m ? +(m[1] / 1000).toFixed(3) : null;
}
const isMoe = (model) => { const k = knownOf(model); return k ? k.moe : /\d+B-A\d/i.test(short(model)) || /\d+x\d+B/i.test(short(model)); };

const KIND_TECH = { block: 'skill-map', inserted_layer: 'skill-map', expert: 'saliency', attn_heads: 'ablation', mlp_cluster: 'ablation' };
function techOf(p) {
  const e = String(p.evidence || '');
  if (/zeroing the part drops/i.test(e)) return 'ablation';
  if (/router pick rate/i.test(e)) return 'saliency';
  if (/weight|delta/i.test(e) && /fine-tune|base/i.test(e)) return 'weight-delta';
  if (/activation|contrast/i.test(e) || /activation|contrast/i.test(p.id)) return 'activation-scan';
  if (/layer map|skipping layers/i.test(e)) return 'skill-map';
  return KIND_TECH[p.kind] || 'skill-map';
}
const layersOf = (p) => (p.layers && p.layers.length ? p.layers : [0, 0]);

const KNOWN_MAKERS = new Set(['qwen', 'meta-llama', 'mistralai', 'google', 'microsoft', 'deepseek-ai', 'allenai', 'ibm-granite', 'huggingfacetb', 'liquidai', 'nvidia', 'openai', 'tiiuae', '01-ai', 'cohereforai', 'unsloth', 'nanbeige', 'xiaomimimo', 'baichuan-inc', 'internlm', 'thudm', 'zai-org']);
const SKILL_FROM_LABEL = (p) => {
  const w = String(p.label || '').toLowerCase();
  const first = w.split(' ')[0];
  if (/^(english to|russian|german|french|chinese|japanese|spanish|hindi|arabic|korean|italian|portuguese)/.test(w)) return 'multilingual';
  if (['math', 'code', 'law', 'medicine', 'knowledge', 'multilingual', 'science', 'reasoning', 'biology', 'chemistry', 'physics', 'facts'].includes(first)) return first;
  return (p.skills || []).find((x) => x !== 'math') || (p.skills || [])[0] || 'general';
};
const KIND_WORD = { block: 'block', mlp_cluster: 'neuron cluster', attn_heads: 'attention heads', expert: 'expert' };
const cap1 = (t) => String(t).charAt(0).toUpperCase() + String(t).slice(1);

export function makeMine({ getParts, getRecipes, getMasks }) {
  const sleepMs = (ms = 140) => sleep(ms);
  const extra = new Map();      // model -> parts that finished mining requests added to the library
  const synth = new Map();      // part id -> a candidate from a mining script
  let cache = null, cacheLen = -1;
  const byModel = () => {
    const parts = getParts();
    const n = parts.length + [...extra.values()].reduce((a, l) => a + l.length, 0);
    if (cache && cacheLen === n) return cache;
    const map = new Map();
    for (const p of parts) { if (!p.donor) continue; if (!map.has(p.donor)) map.set(p.donor, []); map.get(p.donor).push(p); }
    for (const [m, l] of extra) { if (!map.has(m)) map.set(m, []); map.get(m).push(...l); }
    cache = map; cacheLen = n;
    return map;
  };
  const infoOf = (model, ps = null) => {
    const parts = ps || byModel().get(model) || [];
    const known = knownOf(model) || {};
    const maxL = parts.reduce((a, p) => Math.max(a, layersOf(p)[1]), 0);
    const pb = paramsOf(model);
    const layers = known.layers || (parts.length ? maxL + 1 : 0) || (pb > 60 ? 80 : pb > 20 ? 48 : pb > 10 ? 40 : pb > 5 ? 32 : pb > 2 ? 28 : 24);
    const moe = isMoe(model);
    return { model, short: short(model), params_b: pb, layers, experts: known.experts || (moe ? 64 : 0), kind: moe ? 'moe' : 'dense', license: licenseOf(model), maker: String(model).split('/')[0] };
  };
  const skillCounts = (ps) => { const c = {}; for (const p of ps) for (const s of p.skills || []) c[s] = (c[s] || 0) + 1; return c; };
  const kindCounts = (ps) => { const c = {}; for (const p of ps) c[p.kind] = (c[p.kind] || 0) + 1; return c; };
  const summary = (model) => {
    const ps = byModel().get(model) || [];
    const sk = skillCounts(ps);
    return { ...infoOf(model, ps), in_library: ps.length > 0, parts: ps.length, kinds: kindCounts(ps), skills: sk, top_skills: Object.entries(sk).sort((a, b) => b[1] - a[1]).slice(0, 3).map((x) => x[0]) };
  };
  const num = (p) => {
    const m = /(?:mlp|top|act\w*?)(\d{2,4})/.exec(p.id);
    if (p.kind === 'mlp_cluster') return m ? +m[1] : 512;
    if (p.kind === 'attn_heads') return 4 + (hashStr(p.id) % 5);
    if (p.kind === 'expert') { const k = /_k(\d+)_/.exec(p.id); return k ? +k[1] : 1 + (hashStr(p.id) % 4); }
    const [a, b] = layersOf(p); return b - a + 1;
  };
  // fields the detail window needs, derived from the part and its evidence (the same part always gets the same values)
  const rawOf = (p, ev) => {
    const tested = !/^not tested|^no measurement/i.test(ev);
    const sk = SKILL_FROM_LABEL(p);
    if (!tested) return { signature: {}, z: null, z_heldout: null, random_mean: null, random_std: null, random_n: 0 };
    const m = ev.match(/'([a-z0-9_]+)' skill[^:]*:\s*(-?[\d.]+) against (-?[\d.]+)/i) || ev.match(/on '([a-z0-9_]+)' text[^:]*:\s*(-?[\d.]+) against (-?[\d.]+)/i);
    const h = hashStr(p.id) % 1000 / 1000;
    const rejected = p.status === 'rejected';
    const sig = m ? +m[2] : +(0.08 + h * 0.3).toFixed(3);
    const rnd = m ? +m[3] : +(0.01 + h * 0.05).toFixed(3);
    const z = rejected ? +(0.4 + h * 1.8).toFixed(1) : +(3.3 + h * 6).toFixed(1);
    return { signature: { [m ? m[1] : sk]: sig }, z, z_heldout: +(z * (0.7 + h * 0.2)).toFixed(1), random_mean: rnd, random_std: +(0.012 + h * 0.02).toFixed(3), random_n: 5 };
  };
  const recipeIds = (ps) => new Set(ps.map((p) => p.id));
  const detail = (model) => {
    const ps = byModel().get(model) || [];
    const info = infoOf(model, ps);
    const lm = {};
    for (const p of ps) { const [a, b] = layersOf(p); for (let l = a; l <= Math.min(b, info.layers - 1); l++) { const k = l + ':' + p.kind; lm[k] = (lm[k] || 0) + 1; } }
    const layer_map = Object.entries(lm).map(([k, count]) => { const [l, kind] = k.split(':'); return { layer: +l, kind, count }; }).sort((a, b) => a.layer - b.layer);
    const ids = recipeIds(ps);
    const recipes = getRecipes().filter((r) => (r.components || []).some((c) => ids.has(c)) || (r.models || []).includes(model)).slice(0, 12);
    const findings = [];
    for (const p of ps) for (const f of p.findings || []) if (!findings.some((x) => x.id === f.id)) findings.push(f);
    const parts = ps.map((p) => { const ev = String(p.evidence || ''); const raw = rawOf(p, ev); return { ...p, z: raw.z, num_units: num(p) }; });
    return { ...info, in_library: ps.length > 0, parts, skills: skillCounts(ps), layer_map, techniques: [...new Set(ps.map(techOf))], findings: findings.slice(0, 20), recipes };
  };

  /* ----- the simulated queue ----- */
  const PHASE = { fetch: [0, 0.05], read: [0.05, 0.17], scan: [0.17, 0.57], ablate: [0.57, 0.82], gate: [0.82, 0.97], publish: [0.97, 1] };
  const STAGE_DEF = [['fetch', 'Fetch weights', 'modal'], ['read', 'Read the weights', 'openai'], ['scan', 'Scan layers', 'modal'], ['ablate', 'Switch parts off', 'modal'], ['gate', 'Random-part gate', 'modal'], ['publish', 'Publish to the library', 'pastdev']];
  const EFFORT = { quick: { sec: 120, mult: 0.6 }, standard: { sec: 200, mult: 1 }, deep: { sec: 320, mult: 1.8 } };
  EFFORT.smoke = EFFORT.quick; EFFORT.medium = EFFORT.standard; EFFORT.full = EFFORT.deep;
  // the same formula as the form on the Mine page, so the estimate it shows is the one the request reports
  const EST = { smoke: [0.6, 4], quick: [0.6, 4], medium: [2.5, 12], standard: [2.5, 12], full: [6, 30], deep: [6, 30] };
  const estimateOf = (params_b, effort) => { const e = EST[effort] || EST.medium; const gpu = Math.round((params_b || 8) * e[0] + e[1]); return { gpu_min: gpu, cost_usd: +(gpu / 60 * ((params_b || 8) > 10 ? 4 : 2.5)).toFixed(2) }; };
  const SKILL_LINES = { math: 'arithmetic and word problems', code: 'writing and fixing code', law: 'legal terms', medicine: 'medical text', reasoning: 'multi-step reasoning', knowledge: 'facts and recall', multilingual: 'non-English text', science: 'science questions' };

  const makeScript = (req) => {
    const rnd = rngOf(req.id + req.model);
    const all = byModel().get(req.model) || [];
    const info = infoOf(req.model, all);
    const L = info.layers;
    const want = new Set(req.skills || []);
    const isReq = (sk) => want.has(sk);
    const skillList = [...want].length ? [...want] : ['math', 'code'];
    const fresh = all.length === 0;
    const kinds = info.kind === 'moe' ? ['mlp_cluster', 'attn_heads', 'block', 'expert'] : ['mlp_cluster', 'attn_heads', 'block'];
    // parts already in the library: re-check the best ones, requested skills first, other skills at most a fifth
    const cap = Math.min(all.length, req.effort === 'deep' ? 36 : 22);
    const ranked = all.map((p) => [(isReq(SKILL_FROM_LABEL(p)) ? 0 : 1) + rnd() * 0.5, p]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
    const wantedPart = ranked.filter((p) => isReq(SKILL_FROM_LABEL(p)));
    const otherPart = ranked.filter((p) => !isReq(SKILL_FROM_LABEL(p)));
    const kept = [...wantedPart.slice(0, cap), ...otherPart.slice(0, Math.max(0, Math.floor(cap * 0.2)))].slice(0, cap);
    const dropOf = (kind) => (kind === 'block' ? 8 + rnd() * 10 : kind === 'mlp_cluster' ? 4 + rnd() * 10 : kind === 'attn_heads' ? 3 + rnd() * 8 : 3 + rnd() * 9);
    const items = kept.map((p) => { const kind = p.kind === 'inserted_layer' ? 'block' : p.kind; return { id: p.id, label: p.label, kind: p.kind, layer: layersOf(p)[0], real: true, ok: true, skill: SKILL_FROM_LABEL(p), drop: +dropOf(kind).toFixed(1), params_m: p.params_m }; });
    // new candidates: about one per layer for a model never mined, fewer for a model already in the library
    const nNew = fresh ? Math.max(18, Math.round(L * (0.8 + rnd() * 0.4))) : Math.max(5, Math.round((kept.length || 8) * 0.4));
    const keepRate = fresh ? 0.55 + rnd() * 0.15 : 0.3;
    for (let i = 0; i < nNew; i++) {
      const layer = Math.floor(rnd() * L);
      const kind = kinds[Math.floor(rnd() * kinds.length)];
      const skill = skillList[i % skillList.length];
      const span = kind === 'block' ? 2 + Math.floor(rnd() * 5) : 1;
      const layer2 = Math.min(L - 1, layer + span - 1);
      const label = cap1(skill) + ' ' + KIND_WORD[kind] + ' L' + layer + (layer2 > layer ? '-' + layer2 : '');
      const okx = rnd() < keepRate;
      const params_m = kind === 'block' ? Math.round((layer2 - layer + 1) * 38 * (1 + (info.params_b || 4) / 12)) : kind === 'mlp_cluster' ? Math.round(2 + rnd() * 40) : kind === 'attn_heads' ? Math.round(10 + rnd() * 50) : 100;
      items.push({ id: 'mined_' + req.id + '_' + i, label, kind, layer, layer2, real: false, ok: okx, skill, drop: okx ? +dropOf(kind).toFixed(1) : +(0.5 + rnd() * 2.5).toFixed(1), params_m });
    }
    items.sort((a, b) => a.layer - b.layer || (a.real ? 0 : 1) - (b.real ? 0 : 1));
    const N = items.length;
    const [s0, s1] = PHASE.scan, [a0, a1] = PHASE.ablate, [g0, g1] = PHASE.gate;
    items.forEach((it, i) => {
      it.tFound = s0 + (s1 - s0) * ((it.layer + 0.6) / L);
      const order = i / N;
      it.tAblate = a0 + (a1 - a0) * (0.04 + order * 0.9);
      it.tGate = g0 + (g1 - g0) * (0.04 + order * 0.9);
      it.z = it.ok ? +(3.2 + rnd() * 6.6).toFixed(1) : +(-0.6 + rnd() * 2.1).toFixed(1);
      if (!it.real) synth.set(it.id, { ...it, donor: req.model, req: req.id });
    });
    const newKept = items.filter((i) => !i.real && i.ok).length;
    const log = [];
    const L1 = (t, agent, kind, text) => log.push({ t, agent, kind, text });
    L1(0.002, 'Fetcher', 'info', 'Pulling ' + short(req.model) + ' from the model hub onto a Modal volume.');
    L1(0.03, 'Fetcher', 'info', info.layers + ' layers' + (info.experts ? ', ' + info.experts + ' experts per layer' : '') + (info.params_b ? ', ' + info.params_b + 'B parameters' : '') + '. Weights ready.');
    L1(0.06, 'Reader', 'guess', 'Reading weight statistics in a locked sandbox. No network, read-only.');
    if (req.note) L1(0.065, 'Reader', 'guess', 'Note from you: "' + req.note + '". ' + (/later|late|high|top|end/i.test(req.note) ? 'Guessing the later layers first.' : /early|first|low|start/i.test(req.note) ? 'Guessing the early layers first.' : 'I will weigh it when I rank the guesses.'));
    const spans = /later|late|high|top|end/i.test(req.note || '') ? [[0.6, 0.95, 'high'], [0.35, 0.6, 'middle'], [0.07, 0.3, 'low']] : [[0.07, 0.25, 'low'], [0.4, 0.62, 'middle'], [0.7, 0.95, 'high']];
    spans.forEach(([a, b, nm], i) => { const sk = skillList[i % skillList.length]; L1(0.08 + i * 0.035, 'Reader', 'guess', 'Guess: layers ' + Math.round(a * L) + ' to ' + Math.round(b * L) + ' (' + nm + ') may hold ' + (SKILL_LINES[sk] || sk) + '.'); });
    L1(0.15, 'Reader', 'memory', fresh ? 'Memory check: no earlier notes for ' + short(req.model) + '. Starting a fresh map.' : 'Memory check: ' + all.length + ' parts are already mapped for ' + short(req.model) + '. Re-checking the top ' + kept.length + ' and scanning for new ones.');
    L1(0.165, 'Scanner', 'scan', 'Scan beam starts at layer 0. Ranking units by how much more they fire on ' + skillList.join(' and ') + ' text than on general text.');
    const seenLayer = new Set();
    items.forEach((it) => {
      const tl = PHASE.scan[0] + (PHASE.scan[1] - PHASE.scan[0]) * (it.layer / L);
      L1(it.tFound, 'Scanner', 'scan', 'Layer ' + it.layer + ': ' + it.label + ' fires ' + (2 + rnd() * 5).toFixed(1) + 'x more on ' + it.skill.replace(/_/g, ' ') + ' text. Candidate.');
      if (it.layer % 6 === 0 && !seenLayer.has(it.layer)) { seenLayer.add(it.layer); L1(tl + 0.002, 'Scanner', 'scan', 'Layer ' + it.layer + ' of ' + L + ' scanned.'); }
      L1(it.tAblate, 'Ablator', 'ablate', 'Switched off ' + it.label + ': ' + it.skill.replace(/_/g, ' ') + ' drops ' + it.drop + ' points, other skills move ' + (rnd() * 1.4).toFixed(1) + '.');
      L1(it.tGate, 'Gate', it.ok ? 'keep' : 'reject', it.ok ? it.label + ': z ' + it.z + ' against 5 random parts of the same size. Kept.' : it.label + ': z ' + it.z + ', no better than random parts of the same size. Rejected.');
      if (it.ok && rnd() < 0.3) L1(it.tGate + 0.001, 'Memory', 'memory', 'Wrote to shared memory: ' + short(req.model) + ' keeps ' + it.skill.replace(/_/g, ' ') + ' near layer ' + it.layer + '.');
    });
    L1(0.575, 'Ablator', 'ablate', 'Testing ' + N + ' candidates against the skills one by one.');
    L1(0.83, 'Gate', 'info', 'Drawing 5 random parts of the same size for each candidate.');
    L1(0.975, 'Publisher', 'info', fresh ? 'Writing ' + newKept + ' kept parts and their evidence to the library.' : 'Writing ' + newKept + ' new parts to the library. ' + kept.length + ' were already there.');
    L1(0.995, 'Memory', 'memory', 'Posted the run summary to shared memory for the swarm.');
    log.sort((a, b) => a.t - b.t);
    return { items, log, L, N, info, newKept };
  };

  const reqs = new Map();
  const scripts = new Map();
  const now = demoNow();
  const KEY = storageKey('wsp.mine.reqs');
  const add = (r) => { reqs.set(r.id, r); return r; };
  const mkReq = (id, model, skills, effort, startAt, createdAgo = 0, note = '') => {
    const info = infoOf(model);
    return add({ id, model, skills, effort, startAt, note, total: (EFFORT[effort] || EFFORT.standard).sec * 1000, created: new Date(demoNow() - createdAgo).toISOString(), estimate: estimateOf(info.params_b, effort) });
  };
  const saved = () => { try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { return []; } };
  const persist = () => { try { localStorage.setItem(KEY, JSON.stringify([...reqs.values()].filter((r) => r.custom).slice(-20).map((r) => ({ id: r.id, model: r.model, skills: r.skills, effort: r.effort, startAt: r.startAt, note: r.note, created: r.created })))); } catch (e) { /* storage can be off */ } };
  const LIVE_MODEL = byModel().has('LiquidAI/LFM2-8B-A1B') ? 'LiquidAI/LFM2-8B-A1B' : [...byModel().keys()][0] || 'LiquidAI/LFM2-8B-A1B';
  const liveTotal = EFFORT.standard.sec * 1000;
  mkReq('mr_demo_live', LIVE_MODEL, ['code', 'math'], 'standard', now - 85000, 90000);
  const doneModels = ['Qwen/Qwen3-30B-A3B', 'google/gemma-4-26B-A4B-it'].filter((m) => byModel().has(m));
  doneModels.forEach((m, i) => mkReq('mr_demo_done' + (i + 1), m, i ? ['reasoning', 'knowledge'] : ['code', 'math', 'law'], i ? 'quick' : 'deep', now - 86400000 * (i + 1) - 4000000, 86400000 * (i + 1) + 4000000));
  mkReq('mr_demo_queued', 'mistralai/Mixtral-8x7B-Instruct-v0.1', ['code'], 'standard', now - 85000 + liveTotal + 4000, 20000);
  for (const r of saved()) { if (r && r.id && r.model && !reqs.has(r.id)) { const x = mkReq(r.id, r.model, r.skills || ['math', 'code'], r.effort || 'standard', r.startAt, demoNow() - Date.parse(r.created || new Date(demoNow()).toISOString()), r.note || ''); x.custom = true; } }

  const scriptOf = (r) => { if (!scripts.has(r.id)) scripts.set(r.id, makeScript(r)); return scripts.get(r.id); };
  // a finished request adds its kept new parts to the library, once
  const absorb = () => {
    for (const r of reqs.values()) {
      if (r.absorbed || demoNow() < r.startAt + r.total) continue;
      r.absorbed = true;
      const sc = scriptOf(r);
      const list = extra.get(r.model) || [];
      for (const it of sc.items) {
        if (it.real || !it.ok) continue;
        const h = hashStr(it.id);
        list.push({ id: it.id, group: 'add', kind: it.kind, label: it.label, donor: r.model, layers: [it.layer, it.layer2 || it.layer], params_m: it.params_m, default_at: Math.min(14, Math.round(it.layer / sc.L * 15)), summary: short(r.model) + ' layer ' + it.layer, status: 'candidate', evidence: 'Zeroing the part drops the ' + it.skill.replace(/_/g, ' ') + ' skill by ' + it.drop + ' points; 5 random parts of the same size drop it by ' + (0.4 + (h % 30) / 10).toFixed(1) + '.', usable: true, skills: [it.skill], findings: [], stitches: [] });
      }
      if (list.length) extra.set(r.model, list);
    }
  };
  const stateOf = (r, withLog = true) => {
    const t = demoNow() - r.startAt;
    const p = Math.max(0, Math.min(1, t / r.total));
    const status = t < 0 ? 'queued' : p >= 1 ? 'done' : 'running';
    const sc = scriptOf(r);
    const stages = STAGE_DEF.map(([id, name, sponsor]) => ({ id, name, sponsor, status: p >= PHASE[id][1] ? 'done' : p >= PHASE[id][0] && status === 'running' ? 'running' : 'pending', progress: Math.max(0, Math.min(1, (p - PHASE[id][0]) / (PHASE[id][1] - PHASE[id][0]))) }));
    const [s0, s1] = PHASE.scan;
    const layer = Math.max(0, Math.min(sc.L - 1, Math.floor(((p - s0) / (s1 - s0)) * sc.L)));
    const layers_scanned = p < s0 ? 0 : p >= s1 ? sc.L : layer;
    const found = sc.items.filter((i) => p >= i.tFound).map((i) => ({ id: i.id, label: i.label, kind: i.kind, layer: i.layer, skill: i.skill, z: p >= i.tGate ? i.z : null, drop: p >= i.tAblate ? i.drop : null, verdict: p >= i.tGate ? (i.ok ? 'kept' : 'rejected') : p >= i.tAblate ? 'testing' : 'candidate' }));
    const counters = { layers_scanned, layers_total: sc.L, candidates: found.length, ablations: found.filter((f) => f.drop != null).length, kept: found.filter((f) => f.verdict === 'kept').length, rejected: found.filter((f) => f.verdict === 'rejected').length };
    const out = { id: r.id, model: r.model, skills: r.skills, effort: r.effort, status, created: new Date(wallOf(Date.parse(r.created))).toISOString(), stages, estimate: r.estimate, progress: p, started: status === 'queued' ? null : new Date(wallOf(r.startAt)).toISOString(), finished: status === 'done' ? new Date(wallOf(r.startAt + r.total)).toISOString() : null, note: r.note || null, params_b: sc.info.params_b, experts: sc.info.experts || null, layers: sc.L, layer, counters, found };
    if (withLog) out.log = sc.log.filter((l) => p >= l.t).map((l) => ({ t: wallOf(r.startAt + l.t * r.total), agent: l.agent, kind: l.kind, text: l.text }));
    return out;
  };

  return {
    async mine() {
      await sleepMs(120);
      absorb();
      const requests = [...reqs.values()].map((r) => stateOf(r, false)).sort((a, b) => (b.status === 'running') - (a.status === 'running') || (b.status === 'queued') - (a.status === 'queued') || new Date(b.created) - new Date(a.created));
      const done = new Set(requests.filter((r) => r.status === 'done').map((r) => r.model));
      const models = [...byModel().keys()].filter((m) => byModel().get(m).length >= 1).map((m) => ({ ...summary(m), mined_by: done.has(m) ? 'request' : null }));
      return clone({ requests, models });
    },
    async mineLookup(model) {
      await sleepMs(220);
      absorb();
      const m = String(model || '').trim().replace(/^https?:\/\/(www\.)?huggingface\.co\//i, '').replace(/\/(tree|blob|resolve)\/.*$/, '').replace(/[?#].*$/, '').replace(/\/+$/, '');
      if (!/^[\w.-]+\/[\w.-]+$/.test(m)) return { model: m, valid: false, error: 'Use the form maker/name, for example Qwen/Qwen3-30B-A3B.' };
      const ps = byModel().get(m) || [];
      const lib = ps.length > 0 || !!knownOf(m);
      const knownMaker = KNOWN_MAKERS.has(m.split('/')[0].toLowerCase()) && paramsOf(m) != null;
      if (!lib && !knownMaker) {
        const sugg = [...byModel().keys()].filter((x) => byModel().get(x).length >= 6).slice(0, 3);
        return { model: m, valid: false, error: 'We could not find this model on Hugging Face. Check the spelling, or pick one from the list below.', suggestions: sugg };
      }
      return clone({ valid: true, ...summary(m), known: true, mining: [...reqs.values()].some((r) => r.model === m && stateOf(r, false).status !== 'done') });
    },
    async mineSubmit(body) {
      await sleepMs(300);
      const model = String(body.model || '').trim().replace(/^https?:\/\/(www\.)?huggingface\.co\//i, '').replace(/[?#].*$/, '').replace(/\/+$/, '');
      if (!model) { const e = new Error('Enter a model id first.'); throw e; }
      if (!(body.skills && body.skills.length) && body.skills) { const e = new Error('Pick at least one skill to look for.'); e.status = 400; throw e; }
      const live = [...reqs.values()].filter((r) => stateOf(r, false).status !== 'done');
      const id = 'mr_' + Math.random().toString(36).slice(2, 8);
      const r = mkReq(id, model, body.skills && body.skills.length ? body.skills : ['math', 'code'], body.effort || 'standard', demoNow() + (live.length ? 6000 : 1500), 0, String(body.note || '').trim().slice(0, 300));
      r.custom = true; persist();
      return clone(stateOf(r));
    },
    async mineReq(id) {
      await sleepMs(100);
      absorb();
      const r = reqs.get(id);
      if (!r) { const e = new Error('That mining request is not in this browser any more. Start a new one from the Mine page.'); e.status = 404; throw e; }
      return clone(stateOf(r));
    },
    async modelDetail(model) {
      await sleepMs(180);
      absorb();
      if (!HOSTED) {
        try {
          const res = await fetch('/api/models/' + encPath(model), { cache: 'no-store' });
          if (res.ok) { const d = await res.json(); if (d && d.model && Array.isArray(d.parts)) return d; }
        } catch (e) { /* fall back to the library in the page */ }
      }
      const k = knownOf(model);
      if (!(byModel().get(model) || []).length && !k && !(KNOWN_MAKERS.has(String(model).split('/')[0].toLowerCase()) && paramsOf(model) != null)) { const e = new Error('This model is not in the library and is not a known open model.'); e.status = 404; throw e; }
      return clone(detail(model));
    },
    async partDetail(id) {
      await sleepMs(160);
      absorb();
      if (!HOSTED) {
        try {
          const res = await fetch('/api/parts/' + encPath(id), { cache: 'no-store' });
          if (res.ok) { const d = await res.json(); if (d && d.id) return d; }
        } catch (e) { /* derive from the library item */ }
      }
      const mask = (getMasks ? getMasks() : []).find((x) => x.id === id);
      if (mask) return clone({ id: mask.id, group: 'remove', kind: 'deletion_mask', label: mask.label, summary: mask.summary, status: mask.status, evidence: mask.evidence, donor: null, layers: null, params_m: 0, usable: mask.usable !== false, skills: [], raw: { signature: {}, z: null, z_heldout: null, random_mean: null, random_std: null, random_n: 0 }, num_units: null, tier: mask.status, skill: null, selection: null, units_file: null, technique: 'deletion mask', findings: mask.findings || [], stitches: [], recipes: [] });
      let p = getParts().find((x) => x.id === id);
      if (!p) { const sp = synth.get(id); if (sp) p = { id: sp.id, group: 'add', kind: sp.kind, label: sp.label, donor: sp.donor, layers: [sp.layer, sp.layer2 || sp.layer], params_m: sp.params_m, default_at: 6, summary: short(sp.donor) + ' layer ' + sp.layer, status: sp.ok ? 'candidate' : 'rejected', evidence: sp.ok ? 'Zeroing the part drops the ' + sp.skill + ' skill by ' + sp.drop + ' points more than random parts of the same size.' : 'No better than random parts of the same size (z ' + sp.z + ').', usable: sp.ok, skills: [sp.skill], findings: [], stitches: [] }; }
      if (!p) for (const l of extra.values()) { const f = l.find((x) => x.id === id); if (f) p = f; }
      if (!p) { const e = new Error('No such part.'); e.status = 404; throw e; }
      const ev = String(p.evidence || '');
      const raw = rawOf(p, ev);
      const recipes = getRecipes().filter((r) => (r.components || []).includes(id)).slice(0, 8);
      return clone({ ...p, raw, num_units: num(p), tier: p.status, skill: SKILL_FROM_LABEL(p), selection: null, units_file: null, technique: techOf(p), evidence: ev, findings: p.findings || [], stitches: p.stitches || [], recipes });
    },
  };
}
