// Mock data for ?mock=1: measured results and the compiler front door. Same shapes as the server (CONTRACT.md).
// The measured numbers are the ones in studio/server/measured.py. A test keeps the two in step.

import { DEMO } from './demo-flag.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clone = (x) => JSON.parse(JSON.stringify(x));

const CAVEATS = [
  'One run per cell, so a gap of a few points may be noise.',
  'Two rows have a 500-step distillation heal on code text from the healdata pools. The other rows are raw cuts with no training.',
  'The healed 32 expert model (73.8, 61.9) does not beat Qwen2.5-Coder-7B (85.4, 86.0) at a similar total size. It runs 3.0B params per token against 7.6B.',
  'Removing experts keeps the parameters run per token at about 3.0B, so it shrinks storage and memory, not speed.',
  'Other skills fall: MMLU goes from 77.8 to 44.9 at 64 experts.',
  'The 32 expert rows share one size, 8.78B, because the same number of experts per layer gives the same size.',
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

export function makeExtra({ getSpec, getParts, getRecipes }) {
  const mine = makeMine({ getParts, getRecipes });
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
      return { plan: planOf(request), seconds: 0.9, offline: !DEMO, offline_reason: DEMO ? null : 'mock data', verified: false, unmapped,
        note: "The compiler's plan is new and its lineage check is still being fixed. Treat it as a draft." };
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

// facts that are public knowledge about a few models. Everything else is read off the parts library.
const KNOWN = {
  'LiquidAI/LFM2-8B-A1B': { layers: 24, experts: 32 },
  'LiquidAI/LFM2-24B-A2B': { layers: 40, experts: 64 },
  'Qwen/Qwen3-30B-A3B': { layers: 48, experts: 128 },
  'Qwen/Qwen3-30B-A3B-Instruct-2507': { layers: 48, experts: 128 },
  'mistralai/Mixtral-8x7B-Instruct-v0.1': { layers: 32, experts: 8 },
  'Qwen/Qwen2.5-Math-1.5B': { layers: 28 },
  'Qwen/Qwen2.5-Coder-7B-Instruct': { layers: 28 },
  'Qwen/Qwen2.5-Math-7B-Instruct': { layers: 28 },
};
const LICENSE = [[/^qwen\//i, 'Apache 2.0'], [/^liquidai\//i, 'LFM Open License'], [/^mistralai\//i, 'Apache 2.0'], [/^ibm-granite\//i, 'Apache 2.0'], [/^deepseek-ai\//i, 'MIT'], [/^microsoft\//i, 'MIT'], [/^google\/gemma/i, 'Gemma terms'], [/^meta-llama\//i, 'Llama community license']];
const licenseOf = (m) => { const r = LICENSE.find((x) => x[0].test(m)); return r ? r[1] : 'see the model card'; };

/** Parameters in billions from a name such as "Qwen3-30B-A3B" or "LFM2-350M". null when the name has no size. */
function paramsOf(model) {
  const n = short(model);
  const b = n.match(/(\d+(?:\.\d+)?)B(?![a-z])/i);
  if (b) return +b[1];
  const m = n.match(/(\d+(?:\.\d+)?)M(?![a-z])/i);
  return m ? +(m[1] / 1000).toFixed(3) : null;
}
const isMoe = (model) => /\d+B-A\d/i.test(short(model)) || /8x7B/i.test(short(model)) || !!(KNOWN[model] && KNOWN[model].experts);

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

export function makeMine({ getParts, getRecipes }) {
  const sleepMs = (ms = 140) => sleep(ms);
  let cache = null, cacheLen = -1;
  const byModel = () => {
    const parts = getParts();
    if (cache && cacheLen === parts.length) return cache;
    const map = new Map();
    for (const p of parts) { if (!p.donor) continue; if (!map.has(p.donor)) map.set(p.donor, []); map.get(p.donor).push(p); }
    cache = map; cacheLen = parts.length;
    return map;
  };
  const infoOf = (model, ps = null) => {
    const parts = ps || byModel().get(model) || [];
    const known = KNOWN[model] || {};
    const maxL = parts.reduce((a, p) => Math.max(a, layersOf(p)[1]), 0);
    const layers = Math.max(known.layers || 0, parts.length ? maxL + 1 : 0) || (paramsOf(model) > 20 ? 48 : paramsOf(model) > 5 ? 32 : 24);
    const moe = isMoe(model);
    return { model, short: short(model), params_b: paramsOf(model), layers, experts: known.experts || (moe ? null : 0), kind: moe ? 'moe' : 'dense', license: licenseOf(model), maker: String(model).split('/')[0] };
  };
  const skillCounts = (ps) => { const c = {}; for (const p of ps) for (const s of p.skills || []) c[s] = (c[s] || 0) + 1; return c; };
  const kindCounts = (ps) => { const c = {}; for (const p of ps) c[p.kind] = (c[p.kind] || 0) + 1; return c; };
  const summary = (model) => {
    const ps = byModel().get(model) || [];
    const sk = skillCounts(ps);
    return { ...infoOf(model, ps), in_library: ps.length > 0, parts: ps.length, kinds: kindCounts(ps), skills: sk, top_skills: Object.entries(sk).sort((a, b) => b[1] - a[1]).slice(0, 3).map((x) => x[0]) };
  };
  const detail = (model) => {
    const ps = byModel().get(model) || [];
    const info = infoOf(model, ps);
    const lm = {};
    for (const p of ps) { const [a, b] = layersOf(p); for (let l = a; l <= Math.min(b, info.layers - 1); l++) { const k = l + ':' + p.kind; lm[k] = (lm[k] || 0) + 1; } }
    const layer_map = Object.entries(lm).map(([k, count]) => { const [l, kind] = k.split(':'); return { layer: +l, kind, count }; }).sort((a, b) => a.layer - b.layer);
    const ids = new Set(ps.map((p) => p.id));
    const recipes = getRecipes().filter((r) => (r.components || []).some((c) => ids.has(c)) || (r.models || []).includes(model)).slice(0, 12);
    const findings = [];
    for (const p of ps) for (const f of p.findings || []) if (!findings.some((x) => x.id === f.id)) findings.push(f);
    return { ...info, in_library: ps.length > 0, parts: ps, skills: skillCounts(ps), layer_map, techniques: [...new Set(ps.map(techOf))], findings: findings.slice(0, 20), recipes };
  };

  /* ----- the simulated queue ----- */
  const PHASE = { fetch: [0, 0.05], read: [0.05, 0.17], scan: [0.17, 0.57], ablate: [0.57, 0.82], gate: [0.82, 0.97], publish: [0.97, 1] };
  const STAGE_DEF = [['fetch', 'Fetch weights', 'modal'], ['read', 'Read the weights', 'openai'], ['scan', 'Scan layers', 'modal'], ['ablate', 'Switch parts off', 'modal'], ['gate', 'Random-part gate', 'modal'], ['publish', 'Publish to the library', 'pastdev']];
  const EFFORT = { quick: { sec: 120, mult: 0.6 }, standard: { sec: 200, mult: 1 }, deep: { sec: 320, mult: 1.8 } };
  EFFORT.smoke = EFFORT.quick; EFFORT.medium = EFFORT.standard; EFFORT.full = EFFORT.deep;
  const estimateOf = (params_b, effort) => { const m = (EFFORT[effort] || EFFORT.standard).mult; const base = Math.max(6, (params_b || 8) * 1.1); const gpu_min = Math.round(base * m); return { gpu_min, cost_usd: +(gpu_min * 0.055).toFixed(2) }; };
  const SKILL_LINES = { math: 'arithmetic and word problems', code: 'writing and fixing code', law: 'legal terms', medicine: 'medical text', reasoning: 'multi-step reasoning', knowledge: 'facts and recall', multilingual: 'non-English text', science: 'science questions' };

  const makeScript = (req) => {
    const rnd = rngOf(req.id + req.model);
    const all = byModel().get(req.model) || [];
    const info = infoOf(req.model, all);
    const L = info.layers;
    const want = new Set(req.skills || []);
    const rank = (p) => ((p.skills || []).some((s) => want.has(s)) ? 0 : 1) + rnd() * 0.5;
    const cap = Math.min(all.length, req.effort === 'deep' ? 36 : 22);
    let kept = all.map((p) => [rank(p), p]).sort((a, b) => a[0] - b[0]).slice(0, cap).map((x) => x[1]);
    const synth = Math.max(5, Math.round((kept.length || 8) * 0.4));
    const kinds = ['mlp_cluster', 'attn_heads', 'expert', 'block'];
    const items = kept.map((p) => ({ id: p.id, label: p.label, kind: p.kind, layer: layersOf(p)[0], real: true, ok: true, skill: (p.skills || []).find((s) => want.has(s)) || (p.skills || [])[0] || 'general' }));
    const skillList = [...want].length ? [...want] : ['math', 'code'];
    for (let i = 0; i < synth; i++) { const layer = Math.floor(rnd() * L); const kind = kinds[Math.floor(rnd() * kinds.length)]; const skill = skillList[i % skillList.length]; items.push({ id: null, label: skill[0].toUpperCase() + skill.slice(1) + ' ' + (kind === 'expert' ? 'expert' : kind === 'attn_heads' ? 'heads' : 'neurons') + ' L' + layer, kind, layer, real: false, ok: false, skill }); }
    items.sort((a, b) => a.layer - b.layer || (a.real ? 0 : 1) - (b.real ? 0 : 1));
    const N = items.length;
    const [s0, s1] = PHASE.scan, [a0, a1] = PHASE.ablate, [g0, g1] = PHASE.gate;
    items.forEach((it, i) => {
      it.tFound = s0 + (s1 - s0) * ((it.layer + 0.6) / L);
      const order = i / N;
      it.tAblate = a0 + (a1 - a0) * (0.04 + order * 0.9);
      it.tGate = g0 + (g1 - g0) * (0.04 + order * 0.9);
      it.z = it.ok ? +(3.2 + rnd() * 6.6).toFixed(1) : +(-0.6 + rnd() * 2.1).toFixed(1);
      it.drop = it.ok ? +(18 + rnd() * 48).toFixed(1) : +(1 + rnd() * 5).toFixed(1);
    });
    const log = [];
    const L1 = (t, agent, kind, text) => log.push({ t, agent, kind, text });
    L1(0.002, 'Fetcher', 'info', 'Pulling ' + short(req.model) + ' from the model hub onto a Modal volume.');
    L1(0.03, 'Fetcher', 'info', info.layers + ' layers' + (info.experts ? ', ' + info.experts + ' experts per layer' : '') + (info.params_b ? ', ' + info.params_b + 'B parameters' : '') + '. Weights ready.');
    L1(0.06, 'Reader', 'guess', 'Reading weight statistics in a locked sandbox. No network, read-only.');
    const spans = [[0.07, 0.25, 'low'], [0.4, 0.62, 'middle'], [0.7, 0.95, 'high']];
    spans.forEach(([a, b, nm], i) => { const sk = skillList[i % skillList.length]; L1(0.08 + i * 0.035, 'Reader', 'guess', 'Guess: layers ' + Math.round(a * L) + ' to ' + Math.round(b * L) + ' (' + nm + ') may hold ' + (SKILL_LINES[sk] || sk) + '.'); });
    L1(0.15, 'Reader', 'memory', 'Memory check: no earlier notes for ' + short(req.model) + '. Starting a fresh map.');
    L1(0.165, 'Scanner', 'scan', 'Scan beam starts at layer 0. Ranking units by how much more they fire on ' + skillList.join(' and ') + ' text than on general text.');
    const seenLayer = new Set();
    items.forEach((it) => {
      const tl = PHASE.scan[0] + (PHASE.scan[1] - PHASE.scan[0]) * (it.layer / L);
      L1(it.tFound, 'Scanner', 'scan', 'Layer ' + it.layer + ': ' + it.label + ' fires ' + (2 + rnd() * 5).toFixed(1) + 'x more on ' + it.skill + ' text. Candidate.');
      if (it.layer % 6 === 0 && !seenLayer.has(it.layer)) { seenLayer.add(it.layer); L1(tl + 0.002, 'Scanner', 'scan', 'Layer ' + it.layer + ' of ' + L + ' scanned.'); }
      L1(it.tAblate, 'Ablator', 'ablate', 'Switched off ' + it.label + ': ' + it.skill + ' drops ' + it.drop + ' points, other skills move ' + (rnd() * 1.4).toFixed(1) + '.');
      L1(it.tGate, 'Gate', it.ok ? 'keep' : 'reject', it.ok ? it.label + ': z ' + it.z + ' against 5 random parts of the same size. Kept.' : it.label + ': z ' + it.z + ', no better than random parts of the same size. Rejected.');
      if (it.ok && rnd() < 0.3) L1(it.tGate + 0.001, 'Memory', 'memory', 'Wrote to shared memory: ' + short(req.model) + ' keeps ' + it.skill + ' near layer ' + it.layer + '.');
    });
    L1(0.575, 'Ablator', 'ablate', 'Testing ' + N + ' candidates against the skills one by one.');
    L1(0.83, 'Gate', 'info', 'Drawing 5 random parts of the same size for each candidate.');
    L1(0.975, 'Publisher', 'info', 'Writing ' + items.filter((i) => i.ok).length + ' kept parts and their evidence to the library.');
    L1(0.995, 'Memory', 'memory', 'Posted the run summary to shared memory for the swarm.');
    log.sort((a, b) => a.t - b.t);
    return { items, log, L, N, info };
  };

  const reqs = new Map();
  const scripts = new Map();
  const now = Date.now();
  const add = (r) => { reqs.set(r.id, r); return r; };
  const mkReq = (id, model, skills, effort, startAt, createdAgo = 0) => {
    const info = infoOf(model);
    return add({ id, model, skills, effort, startAt, total: (EFFORT[effort] || EFFORT.standard).sec * 1000, created: new Date(now - createdAgo).toISOString(), estimate: estimateOf(info.params_b, effort) });
  };
  const LIVE_MODEL = byModel().has('LiquidAI/LFM2-8B-A1B') ? 'LiquidAI/LFM2-8B-A1B' : [...byModel().keys()][0] || 'LiquidAI/LFM2-8B-A1B';
  const liveTotal = EFFORT.standard.sec * 1000;
  mkReq('mr_demo_live', LIVE_MODEL, ['code', 'math'], 'standard', now - 85000, 90000);
  const doneModels = ['Qwen/Qwen3-30B-A3B', 'google/gemma-4-26B-A4B-it'].filter((m) => byModel().has(m));
  doneModels.forEach((m, i) => mkReq('mr_demo_done' + (i + 1), m, i ? ['reasoning', 'knowledge'] : ['code', 'math', 'law'], i ? 'quick' : 'deep', now - 86400000 * (i + 1) - 4000000, 86400000 * (i + 1) + 4000000));
  mkReq('mr_demo_queued', 'mistralai/Mixtral-8x7B-Instruct-v0.1', ['code'], 'standard', now - 85000 + liveTotal + 4000, 20000);

  const scriptOf = (r) => { if (!scripts.has(r.id)) scripts.set(r.id, makeScript(r)); return scripts.get(r.id); };
  const stateOf = (r, withLog = true) => {
    const t = Date.now() - r.startAt;
    const p = Math.max(0, Math.min(1, t / r.total));
    const status = t < 0 ? 'queued' : p >= 1 ? 'done' : 'running';
    const sc = scriptOf(r);
    const stages = STAGE_DEF.map(([id, name, sponsor]) => ({ id, name, sponsor, status: p >= PHASE[id][1] ? 'done' : p >= PHASE[id][0] && status === 'running' ? 'running' : 'pending', progress: Math.max(0, Math.min(1, (p - PHASE[id][0]) / (PHASE[id][1] - PHASE[id][0]))) }));
    const [s0, s1] = PHASE.scan;
    const layer = Math.max(0, Math.min(sc.L - 1, Math.floor(((p - s0) / (s1 - s0)) * sc.L)));
    const layers_scanned = p < s0 ? 0 : p >= s1 ? sc.L : layer;
    const found = sc.items.filter((i) => p >= i.tFound).map((i) => ({ id: i.id, label: i.label, kind: i.kind, layer: i.layer, skill: i.skill, z: p >= i.tGate ? i.z : null, drop: p >= i.tAblate ? i.drop : null, verdict: p >= i.tGate ? (i.ok ? 'kept' : 'rejected') : p >= i.tAblate ? 'testing' : 'candidate' }));
    const counters = { layers_scanned, layers_total: sc.L, candidates: found.length, ablations: found.filter((f) => f.drop != null).length, kept: found.filter((f) => f.verdict === 'kept').length, rejected: found.filter((f) => f.verdict === 'rejected').length };
    const out = { id: r.id, model: r.model, skills: r.skills, effort: r.effort, status, created: r.created, stages, estimate: r.estimate, progress: p, started: status === 'queued' ? null : new Date(r.startAt).toISOString(), params_b: sc.info.params_b, experts: sc.info.experts || null, layers: sc.L, layer, counters, found };
    if (withLog) out.log = sc.log.filter((l) => p >= l.t).map((l) => ({ t: r.startAt + l.t * r.total, agent: l.agent, kind: l.kind, text: l.text }));
    return out;
  };

  return {
    async mine() {
      await sleepMs(120);
      const requests = [...reqs.values()].map((r) => stateOf(r, false)).sort((a, b) => (b.status === 'running') - (a.status === 'running') || (b.status === 'queued') - (a.status === 'queued') || new Date(b.created) - new Date(a.created));
      const done = new Set(requests.filter((r) => r.status === 'done').map((r) => r.model));
      const models = [...byModel().keys()].filter((m) => byModel().get(m).length >= 2).map((m) => ({ ...summary(m), mined_by: done.has(m) ? 'request' : null }));
      return clone({ requests, models });
    },
    async mineLookup(model) {
      await sleepMs(220);
      const m = String(model || '').trim();
      if (!/^[\w.-]+\/[\w.-]+$/.test(m)) return { model: m, valid: false, error: 'Use the form maker/name, for example Qwen/Qwen3-30B-A3B.' };
      const ps = byModel().get(m) || [];
      return clone({ valid: true, ...summary(m), known: ps.length > 0 || !!KNOWN[m] || paramsOf(m) != null, mining: [...reqs.values()].some((r) => r.model === m && stateOf(r, false).status !== 'done') });
    },
    async mineSubmit(body) {
      await sleepMs(300);
      const model = String(body.model || '').trim();
      if (!model) { const e = new Error('Enter a model id first.'); throw e; }
      const live = [...reqs.values()].filter((r) => stateOf(r, false).status !== 'done');
      const id = 'mr_' + Math.random().toString(36).slice(2, 8);
      const r = mkReq(id, model, body.skills && body.skills.length ? body.skills : ['math', 'code'], body.effort || 'standard', Date.now() + (live.length ? 6000 : 1500), 0);
      return clone(stateOf(r));
    },
    async mineReq(id) {
      await sleepMs(100);
      const r = reqs.get(id);
      if (!r) { const e = new Error('No such mining request.'); e.status = 404; throw e; }
      return clone(stateOf(r));
    },
    async modelDetail(model) {
      await sleepMs(180);
      try {
        const res = await fetch('/api/models/' + encPath(model), { cache: 'no-store' });
        if (res.ok) { const d = await res.json(); if (d && d.model && Array.isArray(d.parts)) return d; }
      } catch (e) { /* fall back to the library in the page */ }
      return clone(detail(model));
    },
    async partDetail(id) {
      await sleepMs(160);
      try {
        const res = await fetch('/api/parts/' + encPath(id), { cache: 'no-store' });
        if (res.ok) { const d = await res.json(); if (d && d.id) return d; }
      } catch (e) { /* derive from the library item */ }
      const p = getParts().find((x) => x.id === id);
      if (!p) { const e = new Error('No such part.'); e.status = 404; throw e; }
      const ev = String(p.evidence || '');
      const m = ev.match(/'([a-z0-9_]+)' skill[^:]*:\s*(-?[\d.]+) against (-?[\d.]+)/i) || ev.match(/on '([a-z0-9_]+)' text[^:]*:\s*(-?[\d.]+) against (-?[\d.]+)/i);
      const raw = m ? { signature: { [m[1]]: +m[2] }, z: null, z_heldout: null, random_mean: +m[3], random_std: null, random_n: 5 } : { signature: {}, z: null, z_heldout: null, random_mean: null, random_std: null, random_n: 0 };
      const recipes = getRecipes().filter((r) => (r.components || []).includes(id)).slice(0, 8);
      return clone({ ...p, raw, num_units: null, tier: p.source === 'atlas' ? 'map' : 'library', skill: (p.skills || [])[0] || null, selection: null, units_file: null, technique: techOf(p), evidence: ev, findings: p.findings || [], stitches: p.stitches || [], recipes });
    },
  };
}
