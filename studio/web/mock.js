// Fixture data and a simulated server, shaped like studio/CONTRACT.md.
// Used when the page is opened with ?mock=1. Nothing here talks to a network.

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const lat = () => sleep(120 + Math.random() * 200);
const now = () => Date.now() / 1000;
const rid = (p) => p + Math.random().toString(16).slice(2, 8);
const clone = (x) => JSON.parse(JSON.stringify(x));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const pick = (a) => a[Math.floor(Math.random() * a.length)];
function hash(s) { let x = 11; for (let i = 0; i < s.length; i++) x = (x * 33 + s.charCodeAt(i)) >>> 0; return x; }
const noise = (s) => ((hash(s) % 1000) / 1000 - 0.5);

const HOST = 'unsloth/Llama-3.2-1B-Instruct';
const HOST_LAYERS = 16;
const HOST_B = 1.24;

/* ---------- catalog ---------- */
const EV_OK = 'block written into the base beat a rank-16 LoRA on the base (held-out accuracy, mean of 3 seeds), same heal data and steps';
const EV_NEW = 'zeroing the part drops its skill more than random same-size parts do (relative drop, mean of 5 random parts)';
const NOT_MINED = 'Not in the compiler catalog yet. Mine it into a catalog record before a program can use it.';
const PARTS = [
  { id: 'qmath_block_8_19', group: 'add', kind: 'block', label: 'Math block', donor: 'Qwen/Qwen2.5-Math-1.5B', layers: [8, 19], params_m: 570, default_at: 6, summary: 'Qwen2.5-Math layers 8-19', task: 'math', power: 0.62, short: 'qmath.l8-19', status: 'validated', evidence: EV_OK + ': 0.51 against 0.47.', usable: true },
  { id: 'qmath_l18_contrast512', group: 'add', kind: 'mlp_cluster', label: 'Math neurons', donor: 'Qwen/Qwen2.5-Math-1.5B', layers: [18, 18], params_m: 6, default_at: 9, summary: '512 MLP units, layer 18', task: 'math', power: 0.42, short: 'qmath.n512', status: 'candidate', evidence: EV_NEW + ': 0.18 against -0.06.', usable: true },
  { id: 'r1_block_10_21', group: 'add', kind: 'block', label: 'Step-by-step reasoning', donor: 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', layers: [10, 21], params_m: 540, default_at: 7, summary: 'R1-Distill layers 10-21', task: 'math', power: 0.5, short: 'r1.l10-21', status: 'validated', evidence: EV_OK + ': 0.49 against 0.46.', usable: true },
  { id: 'lfm_experts_2', group: 'add', kind: 'expert', label: 'Fast recall', donor: 'LiquidAI/LFM2-8B-A1B', layers: [6, 9], params_m: 90, default_at: 10, summary: '2 experts, layers 6-9', task: 'general', power: 0.3, short: 'lfm.e2', status: 'candidate', evidence: 'no measurement yet: no graft run in results/ measures it', usable: true },
  { id: 'qcoder_block_6_17', group: 'add', kind: 'block', label: 'Code block', donor: 'Qwen/Qwen2.5-Coder-1.5B', layers: [6, 17], params_m: 560, default_at: 5, summary: 'Qwen2.5-Coder layers 6-17', task: 'code', power: 0.6, short: 'qcode.l6-17', status: 'candidate', evidence: 'one seed only, no control yet', usable: true },
  { id: 'saul_block_4_13', group: 'add', kind: 'block', label: 'Legal reasoning', donor: 'Equall/Saul-1B', layers: [4, 13], params_m: 410, default_at: 4, summary: 'Saul layers 4-13', task: 'legal', power: 0.55, short: 'saul.l4-13', status: 'candidate', evidence: 'one seed only, no control yet', usable: true },
  { id: 'gemma_block_5_12', group: 'add', kind: 'block', label: 'General block', donor: 'google/gemma-2-2b-it', layers: [5, 12], params_m: 380, default_at: 5, summary: 'Gemma-2-2B layers 5-12', task: 'general', power: 0.4, short: 'gemma.l5-12', status: 'validated', evidence: EV_OK + ': 0.44 against 0.41.', usable: true },
  { id: 'mistral_heads_8', group: 'add', kind: 'attn_heads', label: 'Math attention heads', donor: 'mistralai/Mistral-7B-Instruct-v0.3', layers: [8, 8], params_m: 50, default_at: 6, summary: 'Mistral-7B layer 8', task: 'math', power: 0.35, short: 'mis.h8', status: 'candidate', evidence: EV_NEW + ': 0.21 against 0.02.', usable: true },
  { id: 'phi_block_4_11', group: 'add', kind: 'block', label: 'Reasoning block', donor: 'microsoft/Phi-3-mini-4k-instruct', layers: [4, 11], params_m: 360, default_at: 4, summary: 'Phi-3-mini layers 4-11', task: 'math', power: 0.48, short: 'phi.l4-11', status: 'candidate', evidence: 'one seed only, no control yet', usable: true },
  { id: 'bio_block_8_15', group: 'add', kind: 'block', label: 'Biology block', donor: 'BioMistral/BioMistral-7B', layers: [8, 15], params_m: 420, default_at: 6, summary: 'BioMistral layers 8-15', task: 'general', power: 0.2, short: 'bio.l8-15', status: 'rejected', evidence: 'block lost to a rank-16 LoRA on the base: 0.43 against 0.46, same heal data and steps', usable: false, usable_reason: 'Rejected. Its measured result is below the control.' },
  { id: 'qwen14_block_14_34', group: 'add', kind: 'block', label: 'General block L14-34', donor: 'Qwen/Qwen2.5-14B-Instruct', layers: [14, 34], params_m: 6121.5, default_at: 6, summary: 'Qwen2.5-14B layers 14-34', task: 'general', power: 0.5, short: 'q14.l14-34', status: 'candidate', evidence: 'mined from the library, not run yet', usable: false, usable_reason: NOT_MINED },
];
const MASKS = [
  { id: 'mask_mmlupro_chemistry_llama1b', group: 'remove', label: 'Chemistry', summary: 'deletion mask, 0.17% of weights', status: 'candidate', evidence: 'no measurement yet', usable: true },
  { id: 'mask_mmlupro_law_llama1b', group: 'remove', label: 'Law', summary: 'deletion mask, 0.15% of weights', status: 'validated', evidence: 'removes the law skill: 0.11 to 0.03 on held-out items', usable: true },
  { id: 'mask_mmlupro_biology_llama1b', group: 'remove', label: 'Biology', summary: 'deletion mask, 0.12% of weights', status: 'candidate', evidence: 'no measurement yet', usable: true },
];
const TRAIN = [
  { id: 'lora', group: 'train', label: 'LoRA adapter', summary: 'rank 1 to 64' },
  { id: 'heal', group: 'train', label: 'Heal', summary: 'trains glue only' },
];
const PART = Object.fromEntries(PARTS.map((p) => [p.id, p]));
const GRAFT_PARTS = PARTS.filter((p) => p.usable !== false);
const MASK_IDS = new Set(MASKS.map((m) => m.id));

/* ---------- wsp text ---------- */
function programText(p) {
  const L = [];
  L.push('host ' + (p.host || HOST));
  L.push('seed ' + (p.seed || 0));
  const ids = [];
  (function walk(n) { if (!n) return; if (n.op === 'use') ids.push(n.id); else { walk(n.a); walk(n.b); } })(p.masks);
  if (ids.length) L.push('mask ' + ids.join(' | '));
  for (const g of p.grafts || []) {
    let s = 'graft ' + g.id + ' at ' + g.at;
    if (g.glue_rank) s += ' rank ' + g.glue_rank;
    L.push(s);
  }
  if (p.lora) L.push('lora rank ' + (p.lora.rank || 16));
  if (p.heal) L.push('heal steps ' + p.heal.steps + ' data ' + (p.heal.data || 'gsm8k_train'));
  return L.join('\n') + '\n';
}
function parseProgram(text) {
  const prog = { host: null, seed: 0, masks: null, task_vectors: [], grafts: [], heal: null, lora: null };
  const lines = text.split('\n');
  const fail = (line, msg) => { const e = new Error(msg); e.line = line; e.status = 400; throw e; };
  lines.forEach((raw, i) => {
    const ln = i + 1;
    const s = raw.replace(/#.*/, '').trim();
    if (!s) return;
    const t = s.split(/\s+/);
    const kw = t[0];
    const opts = (from, allowed) => {
      const o = {};
      for (let k = from; k < t.length; k += 2) {
        if (!allowed.includes(t[k])) fail(ln, "unknown option '" + t[k] + "' for " + kw);
        if (t[k + 1] == null) fail(ln, "option '" + t[k] + "' needs a value");
        o[t[k]] = t[k + 1];
      }
      return o;
    };
    const int = (v, name) => { if (!/^-?\d+$/.test(v)) fail(ln, name + ' must be a whole number, got ' + v); return parseInt(v, 10); };
    if (kw === 'host') { if (!t[1]) fail(ln, 'host needs a model name'); prog.host = t[1]; }
    else if (kw === 'seed') prog.seed = int(t[1] || '', 'seed');
    else if (kw === 'mask') {
      const ids = s.slice(4).split('|').map((x) => x.trim()).filter(Boolean);
      if (!ids.length) fail(ln, 'mask needs an id');
      ids.forEach((id) => { if (!MASK_IDS.has(id)) fail(ln, "unknown mask '" + id + "'"); });
      let e = { op: 'use', id: ids[0] };
      for (let k = 1; k < ids.length; k++) e = { op: 'union', a: e, b: { op: 'use', id: ids[k] } };
      prog.masks = e;
    } else if (kw === 'graft') {
      if (!t[1]) fail(ln, 'graft needs a part id');
      if (!PART[t[1]]) fail(ln, "unknown part '" + t[1] + "'");
      const o = opts(2, ['at', 'rank', 'glue_rank', 'control', 'seed']);
      const g = { id: t[1], at: PART[t[1]].default_at, control: null, control_seed: 0, glue_rank: null };
      if (o.at != null) { g.at = int(o.at, 'at'); if (g.at < 0 || g.at >= HOST_LAYERS) fail(ln, 'at must be 0 to ' + (HOST_LAYERS - 1)); }
      if (o.rank != null || o.glue_rank != null) g.glue_rank = int(o.rank ?? o.glue_rank, 'rank');
      prog.grafts.push(g);
    } else if (kw === 'tap' || kw === 'tap_train') {
      fail(ln, 'taps are not allowed: Sapien builds one network');
    } else if (kw === 'lora') {
      const o = opts(1, ['rank', 'targets', 'alpha', 'layers', 'budget']);
      prog.lora = { rank: o.rank != null ? int(o.rank, 'rank') : 16 };
    } else if (kw === 'heal') {
      const o = opts(1, ['steps', 'data', 'lr', 'batch_size', 'seq_len']);
      prog.heal = { steps: o.steps != null ? int(o.steps, 'steps') : 300, data: o.data || 'gsm8k_train', lr: 0.0003, batch_size: 8, seq_len: 384 };
    } else fail(ln, "unknown statement '" + kw + "'");
  });
  if (!prog.host) fail(1, 'a program needs a host line');
  return prog;
}
function validateProgram(p) {
  const errors = [];
  (p.grafts || []).forEach((g, i) => {
    if (!PART[g.id]) errors.push('graft ' + (i + 1) + ': unknown part ' + g.id);
    if (g.at == null || g.at < 0 || g.at >= HOST_LAYERS) errors.push('graft ' + (i + 1) + ': at is out of range');
  });
  return errors;
}

/* ---------- scoring model ---------- */
function relevance(spec, part) {
  if (part.task === spec.task_type) return 1;
  if (part.task === 'general') return 0.45;
  return 0.28;
}
function fitOf(part, at) {
  const d = Math.abs(at - part.default_at);
  return clamp(0.78 - 0.045 * d + noise(part.id + '@' + at) * 0.08, 0.2, 0.95);
}
function paramsOf(p) {
  let b = HOST_B;
  for (const g of p.grafts || []) b += (PART[g.id] ? PART[g.id].params_m : 0) / 1000;
  return Math.round(b * 100) / 100;
}
function evalProgram(spec, p) {
  const fit = {};
  let gain = 0;
  let penalty = 0;
  const gs = p.grafts || [];
  gs.forEach((g, i) => {
    const part = PART[g.id];
    if (!part) return;
    const f = fitOf(part, g.at);
    fit[g.id + '@' + g.at] = Math.round(f * 100) / 100;
    const glue = g.glue_rank ? 1 - Math.abs(Math.log2(g.glue_rank / 16)) * 0.05 : 0.95;
    gain += part.power * (f - 0.4) * relevance(spec, part) * glue;
    for (let j = 0; j < i; j++) if (Math.abs(gs[j].at - g.at) <= 2) penalty += 0.12;
  });
  if (gs.length >= 3) penalty += 0.1;
  const healBonus = p.heal ? 0.02 + 0.025 * Math.min(p.heal.steps, 900) / 300 : -0.03 * (gs.length ? 1 : 0);
  const loraBonus = p.lora ? 0.05 + 0.02 * Math.log2((p.lora.rank || 16) / 8) : 0;
  const maskCount = (() => { let n = 0; (function w(x) { if (!x) return; if (x.op === 'use') n++; else { w(x.a); w(x.b); } })(p.masks); return n; })();
  const scores = {};
  let tsum = 0, tn = 0;
  for (const b of spec.benchmarks) {
    const nz = noise(JSON.stringify(p) + b.id) * 0.012;
    if (b.kind === 'target') {
      const lift = (gain + healBonus + loraBonus) * 0.66 * (b.id === 'yours' ? 1.05 : 1);
      scores[b.id] = clamp(b.base + lift - penalty * 0.6 + nz, 0.01, 0.98);
      tsum += scores[b.id]; tn++;
    } else {
      scores[b.id] = clamp(b.base - 0.006 * gs.length - 0.004 * maskCount - penalty * 1.3 + (p.heal ? 0.004 : 0) + nz, 0.01, 0.98);
    }
    scores[b.id] = Math.round(scores[b.id] * 1000) / 1000;
  }
  const collapsed = spec.benchmarks.some((b) => b.kind === 'guard' && scores[b.id] < b.floor * 0.7);
  const combined = clamp((tn ? tsum / tn : 0) * 0.6 - (collapsed ? 0.12 : 0), 0, 1);
  const first = spec.benchmarks.find((b) => b.kind === 'target');
  let control = null;
  let vs = null;
  if (gs.length && first) {
    const cs = Math.round(clamp(first.base + (scores[first.id] - first.base) * (0.35 + Math.random() * 0.15), 0, 1) * 1000) / 1000;
    control = { scores: { [first.id]: cs } };
    vs = Math.round((scores[first.id] - cs) * 1000) / 1000;
  }
  return { fit, scores, combined: Math.round(combined * 1000) / 1000, control, vs_control: vs, params_b: paramsOf(p), status: collapsed ? 'collapsed' : 'ok' };
}
function describe(p) {
  const bits = [];
  const gs = (p.grafts || []).map((g) => (PART[g.id] ? PART[g.id].short : g.id) + '@' + g.at + (g.glue_rank ? ' r' + g.glue_rank : ''));
  let s = 'host' + (gs.length ? ' + ' + gs.join(' + ') : '');
  const ids = [];
  (function w(x) { if (!x) return; if (x.op === 'use') ids.push(x.id); else { w(x.a); w(x.b); } })(p.masks);
  ids.forEach((id) => bits.push('- ' + id.replace('mask_mmlupro_', '').replace('_llama1b', '')));
  if (p.lora) bits.push('lora r' + (p.lora.rank || 16));
  if (p.heal) bits.push('heal ' + p.heal.steps);
  return s + (bits.length ? ', ' + bits.join(', ') : '');
}
const progHash = (p) => hash(JSON.stringify(p)).toString(16).padStart(8, '0') + hash(describe(p)).toString(16).padStart(8, '0');

/* ---------- limits and comparison ---------- */
function estimateOf(p) {
  const total = paramsOf(p);
  const run = Math.round((HOST_B + 0.72 * (total - HOST_B)) * 100) / 100;
  const steps = (p.heal ? p.heal.steps : 0);
  return { total_params_b: total, run_params_b: run, heal_min: Math.round(steps * 0.3 * Math.max(run, 0.5) / 60 * 100) / 100, host_params_b: HOST_B, added_params_b: Math.round((total - HOST_B) * 100) / 100, unknown: [], parts: [], estimated: true };
}
const f2 = (x) => ('' + Math.round(x * 100) / 100);
function limitProblems(lim, e) {
  const out = [];
  if (!lim) return out;
  if (e.total_params_b > lim.max_total_params_b + 1e-6) out.push('Over the total size limit. This build is estimated at ' + f2(e.total_params_b) + 'B parameters (the host is ' + f2(HOST_B) + 'B and the added parts are ' + f2(e.added_params_b) + 'B). The limit is ' + f2(lim.max_total_params_b) + 'B.');
  if (e.run_params_b > lim.max_run_params_b + 1e-6) out.push('Over the per-token limit. This build runs an estimated ' + f2(e.run_params_b) + 'B parameters per token. The limit is ' + f2(lim.max_run_params_b) + 'B.');
  if (e.heal_min > lim.max_heal_min + 1e-6) out.push('Over the heal time limit. The heal and stitch training is estimated at ' + f2(e.heal_min) + ' minutes. The limit is ' + lim.max_heal_min + ' minutes.');
  return out;
}
function mockComparison(spec, ev, p) {
  const ts = spec.benchmarks.filter((b) => b.kind === 'target');
  const mean = (f) => { const v = ts.map(f); return v.some((x) => x == null) ? null : v.reduce((a, b) => a + b, 0) / v.length; };
  const result = mean((b) => ev.scores[b.id]);
  const rows = [];
  const add = (id, label, value, note, stub) => rows.push({ id, label, split: 'search', per_task: {}, value, measured: value != null, note, stub: !!stub, result_value: result, delta: value == null ? null : Math.round((result - value) * 1000) / 1000, beaten: value == null ? null : result - value > 0.01 });
  add('base', 'Base', mean((b) => b.base), 'the host alone, same items and same scorer', true);
  add('lora', 'Base + LoRA', mean((b) => b.lora), ts.every((b) => b.lora != null) ? 'a rank-16 LoRA on the same data' : 'not measured for every target', false);
  add('distill', 'Base + distillation', mean((b) => b.base + ((b.lora != null ? b.lora : b.base + 0.1) - b.base) * 0.8), 'stub data, not a measurement', true);
  add('random', 'Random components', ev.control ? ev.control.scores[ts[0].id] : null, ev.control ? 'the same build with random components of the same size' : 'not measured yet. The control runs a build with random parts of the same size.', true);
  const limB = spec.limits && spec.limits.max_total_params_b;
  const far = limB && limB >= 2 * HOST_B;
  if (far) add('best_model', 'Best model at ' + f2(limB) + 'B', mean((b) => b.base + 0.5 * ((b.target || b.base + 0.15) - b.base)), 'the best library model within the limit', true);
  const bad = ev.status !== 'ok' || limitProblems(spec.limits, estimateOf(p)).length > 0;
  const failedRows = rows.filter((r) => r.measured && !r.beaten);
  const open = rows.filter((r) => !r.measured);
  let verdict = 'success';
  if (bad || failedRows.length) verdict = 'failed'; else if (open.length) verdict = 'unproven';
  const headline = verdict === 'failed' ? 'Failed attempt. ' + (bad ? 'The build collapsed or went over a limit.' : 'It does not beat ' + failedRows.map((r) => r.label).join(', ') + '.') : verdict === 'unproven' ? 'Not proven yet: ' + open.map((r) => r.label).join(', ') + ' not measured.' : 'Beats every reference by more than 1 point.';
  return { verdict, headline, margin: 0.01, result: { label: 'This build', value: result, per_task: {}, params_b: ev.params_b, heldout: null }, rows, beaten: rows.filter((r) => r.beaten).map((r) => r.id), not_beaten: failedRows.map((r) => r.id), unmeasured: open.map((r) => r.id), best_model_shown: !!far, stub: true };
}

/* ---------- specs ---------- */
const BENCH_LIB = {
  gsm8k: { id: 'gsm8k', name: 'GSM8K', kind: 'target', target: 0.5, base: 0.23, lora: 0.467, recommended: true },
  humaneval: { id: 'humaneval', name: 'HumanEval', kind: 'target', target: 0.35, base: 0.19, lora: 0.31, recommended: true },
  'mmlupro:law': { id: 'mmlupro:law', name: 'MMLU-Pro law', kind: 'target', target: 0.25, base: 0.11, lora: 0.21, recommended: true },
  'mmlupro:chemistry': { id: 'mmlupro:chemistry', name: 'MMLU-Pro chemistry', kind: 'target', target: 0.25, base: 0.09, lora: 0.18, recommended: true },
  if_guard: { id: 'if_guard', name: 'IFEval', kind: 'guard', floor: 0.4, base: 0.42, recommended: true },
  mmlu_guard: { id: 'mmlu_guard', name: 'MMLU', kind: 'guard', floor: 0.3, base: 0.32, recommended: false },
  'mmlupro:biology': { id: 'mmlupro:biology', name: 'MMLU-Pro biology', kind: 'target', base: 0.107, target: 0.25, lora: null },
  arc: { id: 'arc', name: 'ARC Challenge', kind: 'target', base: 0.38, target: 0.5, lora: null },
  hellaswag: { id: 'hellaswag', name: 'HellaSwag', kind: 'guard', base: 0.6, floor: 0.55 },
  truthfulqa: { id: 'truthfulqa', name: 'TruthfulQA', kind: 'guard', base: 0.36, floor: 0.32 },
};
const TYPES = {
  math: { title: 'Laptop math tutor', summary: 'Answers grade-school word problems with worked steps. CPU only.', bench: ['gsm8k', 'if_guard'], ing: ['qmath_block_8_19', 'phi_block_4_11', 'mistral_heads_8', 'r1_block_10_21'] },
  code: { title: 'Small code helper', summary: 'Writes short Python functions from a plain description.', bench: ['humaneval', 'if_guard'], ing: ['qcoder_block_6_17', 'lfm_experts_2'] },
  legal: { title: 'Contract reader', summary: 'Reads clauses and answers questions about them. No chemistry or biology.', bench: ['mmlupro:law', 'if_guard'], ing: ['saul_block_4_13', 'lfm_experts_2'] },
  general: { title: 'Compact assistant', summary: 'A small general model tuned for the task you described.', bench: ['arc', 'if_guard'], ing: ['gemma_block_5_12', 'lfm_experts_2', 'r1_block_10_21'] },
};
function makeSpec(prompt, files, id) {
  const low = (prompt || '').toLowerCase();
  const type = /math|gsm|arith|algebra|word problem/.test(low) ? 'math' : /code|python|sql|function|program/.test(low) ? 'code' : /legal|law|contract|clause/.test(low) ? 'legal' : 'math';
  const T = TYPES[type];
  const bench = T.bench.map((k) => clone(BENCH_LIB[k]));
  const frows = (files || []).map((f) => {
    const rows = clamp(Math.round((f.size || 6000) / 55), 30, 400);
    const sr = Math.round(rows / 3);
    return { name: f.name, rows, score_rows: sr, heal_rows: rows - sr };
  });
  if (frows.length) bench.push({ id: 'yours', name: 'Your problems', kind: 'target', target: 0.6, base: 0.35, lora: 0.52, recommended: true });
  const used = new Set(bench.map((b) => b.id));
  const available = Object.values(BENCH_LIB).filter((b) => !used.has(b.id)).map((b) => { const c = clone(b); delete c.recommended; delete c.lora; return c; });
  const maxp = /500m|0\.5b/.test(low) ? 1.3 : /laptop|cpu|phone/.test(low) ? 2.0 : 2.5;
  return {
    id: id || rid('sp_'), title: T.title, summary: T.summary, prompt: prompt || '', files: frows, task_type: type, host: HOST,
    size: { max_params_b: maxp, min_b: 0.5, max_b: 8.0, host_params_b: HOST_B },
    budget: { evals: 30, compute_min: 45, min_evals: 5, max_evals: 120 },
    limits: { max_total_params_b: maxp, max_run_params_b: Math.min(maxp, 1.8), max_heal_min: 20 }, comparators: { distill: {} },
    benchmarks: bench, available_benchmarks: available,
    ingredients: T.ing.map((pid) => ({ part_id: pid, label: PART[pid].label, donor: PART[pid].donor, kind: PART[pid].kind })),
    drafted_by: 'rules', stub: true,
  };
}
const specs = new Map();
function getSpec(id) {
  if (!specs.has(id)) specs.set(id, makeSpec('A small model that solves grade-school math word problems step by step. It has to run on a laptop with no GPU.', [{ name: 'homework.csv', size: 6600 }], id));
  return specs.get(id);
}

/* ---------- run simulation ---------- */
const runs = new Map();
const PHASES = [['thinking', 1], ['probing', 1], ['compiling', 1], ['scoring', 3]];
const PLANNERS = [
  { id: 'planner-1', name: 'Planner 1', blurb: 'explores whole donor blocks', style: 'blocks' },
  { id: 'planner-2', name: 'Planner 2', blurb: 'explores small parts', style: 'small' },
  { id: 'planner-3', name: 'Planner 3', blurb: 'explores parts plus masks', style: 'masks' },
];
function newAgent(a) {
  return { id: a.id, name: a.name, role: a.id.split('-')[0], blurb: a.blurb, style: a.style, status: 'thinking', now: 'Reading the spec', best: null, attempts: 0, collapses: 0, lessons: 0, nudges_pending: 0, phase: 0, ticks: 0, cur: null, thinking: [], tools: [], history: [], nudges: [], bias: {} };
}
// a readable program name, like the server sends: "Math block L8-19 at L6, heal 300"
function readable(p) {
  const bits = (p.grafts || []).map((g) => (PART[g.id] ? PART[g.id].label : 'Part') + ' at L' + g.at);
  const ids = [];
  (function w(x) { if (!x) return; if (x.op === 'use') ids.push(x.id); else { w(x.a); w(x.b); } })(p.masks);
  if (ids.length) bits.push('remove ' + ids.length + (ids.length > 1 ? ' skills' : ' skill'));
  let out = bits.length ? bits.join(' + ') : 'Host only';
  if (p.lora) out += ', LoRA r' + (p.lora.rank || 16);
  if (p.heal) out += ', heal ' + p.heal.steps;
  return out;
}
function logAct(run, agent, kind, status, text, data = null) {
  run.actId = (run.actId || 0) + 1;
  run.activity.push({ id: run.actId, ts: now(), agent, kind, status, text, data, round: run.round || 1 });
  if (run.activity.length > 200) run.activity.shift();
}
function think(a, text) { a.thinking.push({ ts: now(), text }); if (a.thinking.length > 40) a.thinking.shift(); }
function tool(a, call, result, status = 'ok') { a.tools.push({ ts: now(), call, result, status }); if (a.tools.length > 40) a.tools.shift(); }

function newRun(spec, seed) {
  const run = {
    id: 's0-swarm-' + Math.floor(now()), spec: clone(spec), status: 'running', started: now(), elapsed: 0,
    evals_used: 0, evals_total: spec.budget.evals, compute_used: 0, compute_total: spec.budget.compute_min * 60,
    base_scores: Object.fromEntries(spec.benchmarks.map((b) => [b.id, b.base])), best: null, board: [], activity: [], curve: [], uses: {},
    agents: [newAgent({ id: 'analyst', name: 'Analyst', blurb: 'reads every result, writes the map' }), ...PLANNERS.map(newAgent), newAgent({ id: 'judge', name: 'Judge', blurb: 'ranks, runs controls, guards the budget' })],
    seed: seed || null, tickN: 0, round: 1, actId: 0,
  };
  const an = run.agents[0], ju = run.agents[4];
  an.now = 'Waiting for the first results'; an.status = 'idle';
  ju.now = 'Waiting for the first results'; ju.status = 'idle';
  logAct(run, 'judge', 'system', 'info', 'Run started on the stub backend: ' + run.evals_total + ' evals, ' + spec.budget.compute_min + ' compute minutes.');
  if (seed) logAct(run, 'judge', 'seed', 'info', 'Seeded with a program from the builder: ' + readable(seed));
  runs.set(run.id, run);
  ensureClock();
  return run;
}
let clock = 0;
function ensureClock() { if (!clock) clock = setInterval(() => runs.forEach(tickRun), 1000); }

function genProgram(run, a) {
  const spec = run.spec;
  const rel = GRAFT_PARTS.filter((p) => p.task === spec.task_type || Math.random() < 0.25);
  let pool = rel.length ? rel : GRAFT_PARTS;
  if (a.bias.small) pool = [...pool].sort((x, y) => x.params_m - y.params_m).slice(0, 3);
  if (a.bias.donor && a.lastDonor) pool = pool.filter((p) => p.donor !== a.lastDonor) || pool;
  if (!pool.length) pool = GRAFT_PARTS;
  const prog = { host: HOST, seed: 0, masks: null, task_vectors: [], grafts: [], heal: null, lora: null };
  if (run.seed && run.tickN < 3 && a.id === 'planner-1') return clone(run.seed);
  const n = a.style === 'blocks' ? (Math.random() < 0.25 ? 2 : 1) : 1;
  for (let i = 0; i < n; i++) {
    let part = a.style === 'small' ? pool.slice().sort((x, y) => x.params_m - y.params_m)[Math.floor(Math.random() * Math.min(2, pool.length))] : pick(pool);
    if (a.style === 'blocks') part = pick(pool.filter((p) => p.kind === 'block').length ? pool.filter((p) => p.kind === 'block') : pool);
    const at = clamp(part.default_at + pick([-3, -2, -1, -1, 0, 0, 1, 2, 4]), 1, HOST_LAYERS - 2);
    prog.grafts.push({ id: part.id, at, control: null, control_seed: 0, glue_rank: pick([null, 8, 16, 16, 32, 64]) });
    a.lastDonor = part.donor;
  }
  while (prog.grafts.length > 1 && paramsOf(prog) > spec.size.max_params_b) { prog.grafts.pop(); run.refused = (run.refused || 0) + 1; }
  if (prog.grafts.length && paramsOf(prog) > spec.size.max_params_b) { prog.grafts = []; run.refused = (run.refused || 0) + 1; }
  if (Math.random() < 0.08) run.refused = (run.refused || 0) + 1;
  if (a.style === 'masks' && Math.random() < 0.7) {
    const m = pick(MASKS);
    prog.masks = { op: 'use', id: m.id };
  }
  if (Math.random() < 0.85) prog.heal = { steps: pick([100, 300, 300, 600]), data: spec.task_type === 'math' ? 'gsm8k_train' : 'task_train', lr: 0.0003, batch_size: 8, seq_len: 384 };
  if (Math.random() < 0.2) prog.lora = { rank: pick([8, 16, 32]) };
  return prog;
}

function stepAgent(run, a) {
  if (run.status !== 'running') return;
  a.ticks++;
  const [pname, plen] = PHASES[a.phase];
  const spec = run.spec;
  if (a.ticks === 1) {
    a.status = pname;
    if (pname === 'thinking') {
      a.cur = genProgram(run, a);
      const pend = a.nudges.find((n) => !n.applied);
      if (pend) {
        pend.applied = true; a.nudges_pending = Math.max(0, a.nudges_pending - 1);
        const low = pend.text.toLowerCase();
        if (/small|tiny|fewer param/.test(low)) a.bias.small = true;
        if (/donor|different/.test(low)) a.bias.donor = true;
        think(a, 'Nudge from you: "' + pend.text + '". I will use it for this attempt.');
        if (/explain|plan/.test(low)) think(a, 'Plan: probe fit for two placements near the default layer, compile the better one, heal 300 steps, then score on the search split.');
        logAct(run, a.id, 'nudge_applied', 'ok', 'Read your nudge and will use it next round', { nudge: pend.text });
      }
      const g = a.cur.grafts[0];
      const part = g && PART[g.id];
      const best = run.best ? ' Best so far is ' + (run.best.scores[run.spec.benchmarks.find((b) => b.kind === 'target').id] * 100).toFixed(1) + '%.' : '';
      think(a, part ? 'Try ' + part.label + ' at layer ' + g.at + ' with glue rank ' + (g.glue_rank || 'full') + '.' + best + ' The fit probe is free, so I check it before I spend an eval.' : 'Try a host-only program with heal, as a floor.' + best);
      a.now = 'Planning ' + readable(a.cur);
      if (a.id === 'planner-1' || Math.random() < 0.5) logAct(run, a.id, 'plan', 'info', 'Proposed 3 programs to try this round', { count: 3 });
    } else if (pname === 'probing') {
      const g = a.cur.grafts[0];
      a.now = g ? 'Probing fit at L' + g.at + ' (free, no eval used)' : 'Checking the host (free)';
      if (g) {
        const f = fitOf(PART[g.id], g.at);
        tool(a, 'probe(' + g.id + '@' + g.at + ')', 'fit ' + f.toFixed(2) + ', free');
        logAct(run, a.id, 'probe', 'ok', 'Checked fit of ' + readable(a.cur) + ': ' + f.toFixed(2), { name: readable(a.cur), fits: [Math.round(f * 100) / 100], program_hash: progHash(a.cur) });
      }
    } else if (pname === 'compiling') {
      a.now = 'Compiling ' + readable(a.cur);
      tool(a, 'compile(' + describe(a.cur) + ')', (30 + Math.floor(Math.random() * 40)) + ' s, ' + paramsOf(a.cur).toFixed(2) + 'B params');
      if (a.cur.heal) tool(a, 'heal(' + a.cur.heal.steps + ' steps)', 'loss 2.10 to ' + (1.2 + Math.random() * 0.3).toFixed(2));
    } else {
      a.now = 'Scoring ' + readable(a.cur);
      tool(a, 'eval(search split)', 'running', 'info');
    }
  }
  if (a.ticks >= plen) {
    a.ticks = 0;
    a.phase++;
    if (a.phase >= PHASES.length) { a.phase = 0; finishEval(run, a); }
  }
}

function finishEval(run, a) {
  if (run.evals_used >= run.evals_total) return;
  const spec = run.spec;
  const ev = evalProgram(spec, a.cur);
  run.evals_used++;
  run.compute_used += 35 + Math.floor(Math.random() * 25);
  a.attempts++;
  const tgt = spec.benchmarks.find((b) => b.kind === 'target');
  const score = ev.scores[tgt.id];
  const last = a.tools[a.tools.length - 1];
  if (last && last.result === 'running') { last.result = 'done: ' + (score * 100).toFixed(1) + '%'; last.status = ev.status === 'ok' ? 'ok' : 'bad'; }
  const h = progHash(a.cur);
  a.history.push({ program_hash: h, describe: describe(a.cur), name: readable(a.cur), scores: ev.scores, combined: ev.combined, status: ev.status });
  if (a.history.length > 30) a.history.shift();
  if (ev.status === 'collapsed') { a.collapses++; a.lessons++; }
  if (a.best == null || (ev.status === 'ok' && score > a.best)) a.best = ev.status === 'ok' ? score : a.best;
  for (const g of a.cur.grafts) { const d = PART[g.id] && PART[g.id].donor; if (d) run.uses[d] = (run.uses[d] || 0) + 1; }
  run.uses[HOST] = (run.uses[HOST] || 0) + 1;
  const entry = { rank: 0, program_hash: h, describe: describe(a.cur), name: readable(a.cur), agent: a.id, scores: ev.scores, combined: ev.combined, status: ev.status, vs_control: null, params_b: ev.params_b, program: clone(a.cur), _pending: ev.vs_control, _age: 0 };
  run.board.push(entry);
  const curBest = run.curve.length ? run.curve[run.curve.length - 1].best : tgt.base;
  const isBest = ev.status === 'ok' && ev.params_b <= spec.size.max_params_b && (!run.best || score > run.best.scores[tgt.id]);
  run.curve.push({ eval: run.evals_used, best: isBest ? score : curBest, score });
  const guardsOut = spec.benchmarks.filter((b) => b.kind === 'guard').map((g) => ({ id: g.id, name: g.name, score: ev.scores[g.id], floor: g.floor, ok: ev.scores[g.id] >= g.floor }));
  if (isBest) run.best = { program_hash: h, describe: entry.describe, name: entry.name, scores: ev.scores, combined: ev.combined, agent: a.id, program: clone(a.cur) };
  const data = { name: entry.name, program_hash: h, target: Math.round(score * 10000) / 10000, best: run.best ? run.best.scores[tgt.id] : null, is_best: isBest, guards: guardsOut, scores: ev.scores, result: ev.status };
  if (isBest) logAct(run, a.id, 'eval', 'ok', 'Scored ' + entry.name + ' at ' + (score * 100).toFixed(1) + '% on ' + tgt.name + '. It is the best so far.', data);
  else if (ev.status === 'collapsed') {
    logAct(run, a.id, 'eval', 'bad', 'Scored ' + entry.name + ' and it collapsed. The lesson is saved.', data);
    think(a, 'That collapsed. Guards fell below their floor, so I will move the graft away from its neighbour and try again.');
  } else logAct(run, a.id, 'eval', 'info', 'Scored ' + entry.name + ' at ' + (score * 100).toFixed(1) + '% on ' + tgt.name + '.', data);
  run.board.sort((x, y) => (y.status === 'ok') - (x.status === 'ok') || y.combined - x.combined);
  run.board = run.board.slice(0, 14);
  run.board.forEach((e, i) => { e.rank = i + 1; });
}

function tickRun(run) {
  if (run.status === 'paused' || run.status === 'done' || run.status === 'error') return;
  if (run.status === 'stopping') {
    run.status = 'done';
    run.agents.forEach((a) => { a.status = 'done'; a.now = 'Stopped. Best program kept.'; });
    logAct(run, 'judge', 'milestone', 'info', 'Search stopped. The best program is kept.');
    return;
  }
  run.tickN++;
  run.elapsed += 14;
  run.round = Math.floor((run.tickN - 1) / 4) + 1;
  const ready = run.board.length;
  run.agents.forEach((a) => {
    if (a.id.startsWith('planner')) stepAgent(run, a);
  });
  // analyst and judge
  const an = run.agents[0], ju = run.agents[4];
  if (ready >= 2 && run.tickN % 5 === 0) {
    an.status = 'thinking';
    const ok = run.board.filter((e) => e.status === 'ok');
    const avgAt = ok.length ? ok.map((e) => e.program.grafts[0] ? e.program.grafts[0].at : 6).reduce((x, y) => x + y, 0) / ok.length : 6;
    const txt = 'Map v' + Math.ceil(run.tickN / 5) + ': grafts near L' + Math.round(avgAt) + ' win. ' + (() => { const n = run.board.filter((e) => e.status !== 'ok').length; return n + (n === 1 ? ' collapse' : ' collapses') + ' so far, mostly stacked grafts.'; })();
    an.now = txt; an.lessons++;
    think(an, txt);
    tool(an, 'read_results(' + ready + ' programs)', 'ok');
    logAct(run, 'analyst', 'map', 'info', txt, { detail: 'part                     fits   wins\n' + run.board.slice(0, 4).map((e) => (e.name + '                              ').slice(0, 24) + ' ' + (e.status === 'ok' ? 'yes' : 'no') + '   ' + (e.combined * 100).toFixed(0)).join('\n') });
  } else if (run.tickN % 5 === 2) an.status = 'idle';
  const top = run.board.slice(0, 3).find((e) => e.vs_control == null && e._pending != null);
  if (top) {
    top._age++;
    ju.status = 'scoring';
    ju.now = 'Running the random-part control for ' + top.name;
    if (top._age >= 2) {
      top.vs_control = top._pending;
      ju.status = 'idle'; ju.attempts++;
      ju.now = 'Control done for rank ' + top.rank + ': ' + (top.vs_control * 100).toFixed(1) + ' points above a random part';
      logAct(run, 'judge', 'verdict', top.vs_control > 0.02 ? 'ok' : 'info', 'The random-part control for ' + top.name + ' says the part adds ' + (top.vs_control * 100).toFixed(1) + ' points.', { frontier: run.board.slice(0, 3).map((e) => ({ program_hash: e.program_hash, name: e.name, target: e.scores[run.spec.benchmarks.find((b) => b.kind === 'target').id] })), retired: [], killed: [] });
      think(ju, ju.now);
      tool(ju, 'control(random part)', '+' + (top.vs_control * 100).toFixed(1));
    }
  } else if (ju.status === 'scoring') ju.status = 'idle';
  if (run.evals_used >= run.evals_total || run.compute_used >= run.compute_total) {
    run.status = 'done';
    run.agents.forEach((a) => { a.status = 'done'; a.now = 'Budget used. Finished.'; });
    const tg = run.spec.benchmarks.find((b) => b.kind === 'target');
    run.board.slice(0, 3).forEach((e) => logAct(run, 'judge', 'heldout', 'info', 'Held-out check of ' + e.name, { name: e.name, program_hash: e.program_hash, target: Math.max(0, e.scores[tg.id] - 0.02), scores: e.scores }));
    logAct(run, 'judge', 'milestone', 'ok', 'Search finished. The best program is ready to open in the builder.');
  }
}

function snapshot(run) {
  const spec = run.spec;
  const srcs = [{ model: HOST, role: 'host', uses: run.uses[HOST] || 0 }];
  Object.keys(run.uses).filter((k) => k !== HOST).forEach((k) => srcs.push({ model: k, role: 'donor', uses: run.uses[k] }));
  const s = {
    run_id: run.id, spec, status: run.status, stub: true, backend: 'stub', started_at: run.started, elapsed_s: run.elapsed,
    budget: { evals_used: run.evals_used, evals_total: run.evals_total, compute_used_s: run.compute_used, compute_total_s: run.compute_total },
    base_scores: run.base_scores, best: run.best,
    agents: run.agents.map((a) => ({ id: a.id, name: a.name, role: a.role, blurb: a.blurb, status: run.status === 'paused' ? 'paused' : a.status, now: a.now, best: a.best, attempts: a.attempts, collapses: a.collapses, lessons: a.lessons, nudges_pending: a.nudges_pending })),
    refused: run.refused || 0, limits: spec.limits, comparison: null,
    leaderboard: run.board.map((e) => { const c = { ...e }; delete c._pending; delete c._age; c.comparison = mockComparison(spec, { scores: e.scores, status: e.status, params_b: e.params_b, control: null, ...(e.vs_control != null ? { control: { scores: { [spec.benchmarks.find((b) => b.kind === 'target').id]: e.scores[spec.benchmarks.find((b) => b.kind === 'target').id] - e.vs_control } } } : {}) }, e.program); return c; }),
    activity: run.activity, curve: run.curve, sources: srcs,
  };
  if (s.best) { const e = s.leaderboard.find((x) => x.program_hash === s.best.program_hash); s.comparison = e ? e.comparison : null; }
  return clone(s);
}
function getRunObj(id) {
  if (!runs.has(id)) {
    const r = newRun(getSpec('sp_demo'));
    runs.delete(r.id); r.id = id; runs.set(id, r);
  }
  return runs.get(id);
}

/* ---------- builders and jobs ---------- */
const builders = new Map();
const jobs = new Map();
function getBuilder(id, specId) {
  if (!builders.has(id)) builders.set(id, { id, spec_id: specId || 'sp_demo', budget: { evals_used: 0, evals_total: 12, compute_used_s: 0, compute_total_s: 2700 } });
  return builders.get(id);
}

/* ---------- the api ---------- */
export const mockApi = {
  async health() {
    await lat();
    const heal = new URLSearchParams(location.search).get('heal') === '1';
    return { ok: true, product: 'Sapien', backend: 'stub', llm: 'none', llm_model: 'gpt-5.4-mini', stub: true, heal_note: heal ? 'The heal recipe is being fixed. Healing on dataset text currently damages the base model, so scores after a heal are not reliable yet.' : null };
  },
  async draftSpec(prompt, files) {
    await sleep(2200);
    if (!prompt || !prompt.trim()) { const e = new Error('Say what the model should do first.'); throw e; }
    const s = makeSpec(prompt, files);
    specs.set(s.id, s);
    return clone(s);
  },
  async getSpec(id) { await lat(); return clone(getSpec(id)); },
  async putSpec(id, spec) { await lat(); specs.set(id, clone({ ...spec, stub: true })); return clone(specs.get(id)); },
  async createRun(spec_id, program) { await lat(); const r = newRun(getSpec(spec_id), program); return { run_id: r.id }; },
  async getRun(id) { await lat(); return snapshot(getRunObj(id)); },
  openStream(id, onSnap) {
    const run = getRunObj(id);
    onSnap(snapshot(run));
    const t = setInterval(() => onSnap(snapshot(run)), 1000);
    return () => clearInterval(t);
  },
  async pause(id) { await lat(); const r = getRunObj(id); if (r.status === 'running') r.status = 'paused'; return { ok: true }; },
  async resume(id) { await lat(); const r = getRunObj(id); if (r.status === 'paused') r.status = 'running'; return { ok: true }; },
  async stop(id) { await lat(); const r = getRunObj(id); if (r.status === 'running' || r.status === 'paused') r.status = 'stopping'; return { ok: true }; },
  async getAgent(id, aid) {
    await lat();
    const run = getRunObj(id);
    const a = run.agents.find((x) => x.id === aid);
    if (!a) { const e = new Error('no such agent'); e.status = 404; throw e; }
    const snap = snapshot(run).agents.find((x) => x.id === aid);
    const cur = a.cur ? { program: clone(a.cur), text: programText(a.cur), describe: describe(a.cur), name: readable(a.cur) } : null;
    return clone({ agent: snap, current: cur, thinking: a.thinking, tools: a.tools, history: a.history, nudges: a.nudges });
  },
  async nudge(id, aid, text) {
    await lat();
    const run = getRunObj(id);
    const a = run.agents.find((x) => x.id === aid);
    if (!text || !text.trim()) { throw new Error('Write something first.'); }
    a.nudges.push({ ts: now(), text, applied: false });
    a.nudges_pending++;
    logAct(run, 'you', 'nudge', 'info', text, { to: aid, to_name: a.name });
    return { ok: true };
  },
  async catalog() { await lat(); return clone({ stub: true, host: { model: HOST, layers: HOST_LAYERS, params_b: HOST_B }, parts: PARTS.map(({ task, power, short, ...p }) => p), masks: MASKS, train: TRAIN, product: 'Sapien', kinds: ['block', 'attn_heads', 'mlp_cluster', 'expert', 'inserted_layer'].map((k) => ({ kind: k, name: k, count: PARTS.filter((p) => p.kind === k).length })), library: { atlas: false, counts: { candidate: PARTS.filter((p) => p.status === 'candidate').length, validated: PARTS.filter((p) => p.status === 'validated').length, rejected: PARTS.filter((p) => p.status === 'rejected').length } } }); },
  async programText(program) { await sleep(60); return { text: programText(program) }; },
  async programParse(text) { await sleep(80); return { program: parseProgram(text) }; },
  async programValidate(program) { await sleep(60); return { errors: validateProgram(program) }; },
  async createBuilder(spec_id) { await lat(); const id = rid('b_'); const b = getBuilder(id, spec_id); return { builder_id: id, budget: clone(b.budget) }; },
  async builderAction(id, program, action) {
    await lat();
    const b = getBuilder(id);
    const spec = getSpec(b.spec_id);
    const jid = rid('j_');
    {
      const q = clone(program);
      if (action === 'heal' && !q.heal) q.heal = { steps: 300 };
      if (action === 'train') { if (!q.lora) q.lora = { rank: 16 }; if (!q.heal) q.heal = { steps: 300 }; }
      const problems = limitProblems(spec.limits, estimateOf(q));
      if (problems.length) { const e = new Error(problems.join(' ')); e.status = 400; throw e; }
    }
    jobs.set(jid, { id: jid, bid: id, action, program: clone(program), spec, readyAt: Date.now() + 1000 + Math.random() * 1000, done: null });
    return { job_id: jid };
  },
  async getJob(jid) {
    await sleep(80);
    const j = jobs.get(jid);
    if (!j) { const e = new Error('no such job'); e.status = 404; throw e; }
    const base = { job_id: jid, action: j.action, stub: true, error: null };
    if (Date.now() < j.readyAt) return { ...base, status: Date.now() < j.readyAt - 1500 ? 'queued' : 'running', result: null, budget: clone(getBuilder(j.bid).budget) };
    if (!j.done) {
      const b = getBuilder(j.bid);
      let p = clone(j.program);
      if (j.action === 'heal' && !p.heal) p.heal = { steps: 300, data: 'gsm8k_train', lr: 0.0003, batch_size: 8, seq_len: 384 };
      if (j.action === 'train') { if (!p.lora) p.lora = { rank: 16 }; if (!p.heal) p.heal = { steps: 300, data: 'gsm8k_train', lr: 0.0003, batch_size: 8, seq_len: 384 }; }
      const ev = evalProgram(j.spec, p);
      let result;
      if (j.action === 'compile') {
        result = { fit: ev.fit, params_b: ev.params_b, compute_s: 25 };
        b.budget.compute_used_s += 25;
      } else {
        if (b.budget.evals_used >= b.budget.evals_total) { j.done = { ...base, status: 'error', result: null, budget: clone(b.budget), error: 'No evals left in this session.' }; return j.done; }
        const cs = 90 + Math.floor(Math.random() * 120);
        result = { fit: ev.fit, scores: ev.scores, combined: ev.combined, control: ev.control, params_b: ev.params_b, compute_s: cs, status: ev.status, program: p, size: estimateOf(p), limit_problems: [], comparison: mockComparison(j.spec, ev, p) };
        b.budget.evals_used++; b.budget.compute_used_s += cs;
      }
      j.done = { ...base, status: 'done', result, budget: clone(b.budget) };
    }
    return clone(j.done);
  },
  async builderSize(id, program) {
    await sleep(80);
    const spec = getSpec(getBuilder(id).spec_id);
    const estimate = estimateOf(program);
    const problems = limitProblems(spec.limits, estimate);
    return { estimate, limits: spec.limits, problems, ok: problems.length === 0 };
  },
  async suggest(id, program) {
    await sleep(350);
    const b = getBuilder(id);
    const spec = getSpec(b.spec_id);
    const out = [];
    const clonep = () => clone(program);
    const best = GRAFT_PARTS.filter((p) => p.task === spec.task_type).sort((x, y) => y.power - x.power)[0] || GRAFT_PARTS[0];
    if (!program.grafts.length) {
      const p = clonep(); p.grafts.push({ id: best.id, at: best.default_at, control: null, control_seed: 0, glue_rank: 16 });
      out.push({ title: 'Add ' + best.label + ' at L' + best.default_at + '.', detail: 'It is the best-fitting ' + spec.task_type + ' part in the catalog. Free probe shows fit ' + fitOf(best, best.default_at).toFixed(2) + '.', tag: 'likely +8 to 14', program: p });
    }
    program.grafts.forEach((g, i) => {
      const part = PART[g.id];
      if (part && g.at !== part.default_at) {
        const p = clonep(); p.grafts[i].at = part.default_at;
        out.push({ title: 'Move ' + part.label + ' to L' + part.default_at + '.', detail: 'Free probe: fit ' + fitOf(part, part.default_at).toFixed(2) + ' vs ' + fitOf(part, g.at).toFixed(2) + ' at L' + g.at + '.', tag: 'likely +1 to 2', program: p });
      }
    });
    if (program.grafts.length && !program.heal) {
      const p = clonep(); p.heal = { steps: 300, data: 'gsm8k_train', lr: 0.0003, batch_size: 8, seq_len: 384 };
      out.push({ title: 'Add a heal pass of 300 steps.', detail: 'Grafts without a heal usually lose a few points. Heal trains only the glue.', tag: 'likely +2 to 3', program: p });
    }
    if (program.grafts.length && !program.lora) {
      const p = clonep(); p.lora = { rank: 16 };
      if (!p.heal) p.heal = { steps: 300, data: 'gsm8k_train', lr: 0.0003, batch_size: 8, seq_len: 384 };
      out.push({ title: 'Add a rank 16 LoRA on top.', detail: 'It closes part of the gap to the fine-tuning bar.', tag: 'bigger change', program: p });
    }
    if (program.masks) {
      const p = clonep(); p.masks = null;
      out.push({ title: 'Drop the deletion mask.', detail: 'It saves no size and costs a little on the guards.', tag: 'cleanup', program: p });
    }
    if (program.grafts.length > 2) {
      const p = clonep(); p.grafts = p.grafts.slice(0, 2);
      out.push({ title: 'Keep only two grafts.', detail: 'Three or more close together tend to collapse the guards.', tag: 'cleanup', program: p });
    }
    return { suggestions: out.slice(0, 4) };
  },
  async handoff(id, program) {
    await lat();
    const b = getBuilder(id);
    const r = newRun(getSpec(b.spec_id), program);
    return { run_id: r.id };
  },
};
