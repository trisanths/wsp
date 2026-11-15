// Mock data for ?mock=1: measured results and the compiler front door. Same shapes as the server (CONTRACT.md).
// The measured numbers are the ones in studio/server/measured.py. A test keeps the two in step.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clone = (x) => JSON.parse(JSON.stringify(x));

const CAVEATS = [
  'One run per cell, so a gap of a few points may be noise.',
  'No heal has run yet. These are the raw cuts.',
  'Removing experts keeps the parameters run per token at about 3.0B, so it shrinks storage and memory, not speed.',
  'Other skills fall: MMLU goes from 77.8 to 44.9 at 64 experts.',
  'The 32 expert rows share one size, 8.78B, because the same number of experts per layer gives the same size.',
];
const row = (label, experts_kept, rule, total_params_b, scores, is_control = false, is_full = false) => ({ label, experts_kept, rule, total_params_b, scores, is_control, is_full });

const MEASURED = [{
  id: 'run19_qwen3_30b_a3b_code', title: 'Skill removal on Qwen3-30B-A3B, code only', kind: 'skill_removal', skill: 'code',
  model: 'Qwen/Qwen3-30B-A3B-Instruct-2507', experts_per_layer: 128, params_run_per_token_b: 3.03, date: '2026-11-14',
  scorer: 'wsp.skills_eval', runs_per_cell: 1, heal: 'none', source: 'compiler session, team memory, 2026-11-14', caveats: CAVEATS,
  rows: [
    row('Full model', 128, 'none', 30.5, { humaneval: 91.5, mbpp: 84.4, gsm8k: 91.5, mmlu: 77.8 }, false, true),
    row('Keep 64 by code saliency', 64, 'saliency', 16.0, { humaneval: 87.8, mbpp: 79.0, gsm8k: 80.5, mmlu: 44.9 }),
    row('Keep 32 by code saliency', 32, 'saliency', 8.78, { humaneval: 57.9, mbpp: 55.3 }),
    row('Keep 16 by code saliency', 16, 'saliency', 5.15, { humaneval: 3.7, mbpp: 7.8 }),
    row('Keep 32 by routing frequency', 32, 'freq', 8.78, { humaneval: 7.9, mbpp: 11.7 }),
    row('Keep 32 random (two seeds)', 32, 'random', 8.78, { humaneval: 0.0, mbpp: 0.0 }, true),
    row('Keep 32 by skill-to-general ratio', 32, 'selective', 8.78, { humaneval: 0.0, mbpp: 0.0 }, true),
  ],
}];

const DATASET = { gsm8k: 'openai/gsm8k', mmlu: 'cais/mmlu', humaneval: 'openai/openai_humaneval', mbpp: 'google-research-datasets/mbpp', if_guard: 'google/IFEval' };
const datasetOf = (id) => DATASET[id] || (id.startsWith('mmlupro:') ? 'TIGER-Lab/MMLU-Pro' : id.startsWith('bbeh') ? 'BBEH/bbeh' : null);
const SKILL = { gsm8k: 'math', 'mmlupro:math': 'math', humaneval: 'code', mbpp: 'code', mmlu: 'knowledge', 'mmlupro:health': 'medicine', bbeh_mini: 'reasoning', if_guard: 'general' };
const slug = (t) => (String(t || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'sapien_request');

function requestOf(spec) {
  const reported = [];
  const unmapped = [];
  for (const b of spec.benchmarks || []) {
    const d = datasetOf(b.id);
    if (d == null) { if (!unmapped.includes(b.id)) unmapped.push(b.id); } else if (!reported.includes(d)) reported.push(d);
  }
  const lim = spec.limits || {};
  const req = { name: slug(spec.title) };
  if (spec.request_type === 'shrink' && spec.shrink) {
    req.base = spec.shrink.source_model; req.add = []; req.remove = [{ skill: '*' }]; req.keep = [spec.shrink.keep_skill];
  } else {
    req.base = spec.host;
    const skills = [];
    for (const b of spec.benchmarks || []) if (b.kind === 'target' && SKILL[b.id] && !skills.includes(SKILL[b.id])) skills.push(SKILL[b.id]);
    req.add = skills.map((skill) => {
      const found = (spec.ingredients || []).filter((g) => g.donor !== spec.host && (g.label + ' ' + g.donor).toLowerCase().includes(skill === 'code' ? 'code' : skill));
      return { skill, sources: found.length ? [...new Set(found.map((g) => g.donor))] : 'auto', components: null };
    });
    req.remove = [];
    req.keep = [...new Set((spec.benchmarks || []).filter((b) => b.kind === 'guard' && SKILL[b.id]).map((b) => SKILL[b.id]))];
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

function planOf(req) {
  const base = { id: req.base, params_b: 1.24, active_params_b: 1.24, moe: false };
  if (req.remove.length) {
    const total = Math.min(req.limits.max_total_params_b || 16.9, 16.9);
    return { name: req.name, base: { ...base, params_b: 30.53, active_params_b: 3.04, moe: true }, limits: req.limits, keep: req.keep, remove: req.remove, sources: [],
      stages: [{ order: 1, method: 'slim', kind: 'plain', params_total_after_b: total, params_run_after_b: 3.04, limit_problems: [] }],
      arms: [{ arm: 'base', what: 'the base model alone' }, { arm: 'base_lora', what: 'the base plus a LoRA' }],
      warnings: ['wsp/slim.py has one studio run (Qwen2.5-1.5B dense) with no trusted quality number'], refusals: [], feasible: true };
  }
  const src = req.add.map((a) => ({ skill: a.skill, source: Array.isArray(a.sources) ? a.sources[0] : 'Qwen/Qwen2.5-Coder-7B-Instruct', relation: 'same_family_other_size', chosen: 'synth',
    methods: [{ method: 'synth', fits: true, adds_total_b: 0.72, reason: 'other width or depth: donor layers go through a seam with translation layers' }] }));
  return { name: req.name, base, limits: req.limits, keep: req.keep, remove: [], sources: src,
    stages: [{ order: 1, method: 'synth', kind: 'seam', params_total_after_b: 1.96, params_run_after_b: 1.96, limit_problems: [] }],
    arms: [{ arm: 'base', what: 'the base model alone' }, { arm: 'base_lora', what: 'the base plus a LoRA' }, { arm: 'base_distill', what: 'the base plus distillation' }],
    warnings: ['wsp/synth.py has passed CPU tests on tiny models only; no GPU run on real weights has finished'], refusals: [], feasible: true };
}

export function makeExtra({ getSpec }) {
  return {
    async measured() { await sleep(150); return clone(MEASURED); },
    async specRequest(id) {
      await sleep(150);
      const { request, unmapped } = requestOf(getSpec(id));
      return { request, unmapped, errors: errorsOf(request) };
    },
    async specPlan(id) {
      await sleep(900);
      const { request, unmapped } = requestOf(getSpec(id));
      return { plan: planOf(request), seconds: 0.9, offline: true, offline_reason: 'mock data', verified: false, unmapped,
        note: "The compiler's plan is new and its lineage check is still being fixed. Treat it as a draft." };
    },
  };
}
