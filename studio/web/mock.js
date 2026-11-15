// Fixture data and a simulated server, shaped like studio/CONTRACT.md.
// Used when the page is opened with ?mock=1. Nothing here talks to a network.

import { healPool } from './util.js';
import { DEMO, demoSpeed, demoNow, wallOf, storageKey } from './demo-flag.js';
import * as V from './demo-data.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const lat = () => sleep(120 + Math.random() * 200);
let vclock = null;
const now = () => (vclock != null ? vclock : Date.now() / 1000);
const SAY = (demo, plain) => (DEMO ? demo : plain);
const rid = (p) => p + Math.random().toString(16).slice(2, 8);
const clone = (x) => JSON.parse(JSON.stringify(x));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const pick = (a) => a[Math.floor(Math.random() * a.length)];
function hash(s) { let x = 11; for (let i = 0; i < s.length; i++) x = (x * 33 + s.charCodeAt(i)) >>> 0; return x; }
const noise = (s) => ((hash(s) % 1000) / 1000 - 0.5);

const short = (m) => String(m || '').split('/').pop();
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
  { id: 'map_expert_lfm2_8b_biology_l14', group: 'add', kind: 'expert', label: 'General expert L14', donor: 'LiquidAI/LFM2-8B-A1B', layers: [14, 14], params_m: 180, default_at: 9, summary: 'LFM2-8B-A1B layer 14', task: 'general', power: 0.3, short: 'lfm.e14', status: 'candidate', evidence: 'donor skill map, biology: 8 experts picked more often on biology text. Compiler record built from the library file.', usable: true, skills: ['biology'], findings: [{ id: 'f_mock_experts_donor_only', title: 'Skill maps say where a skill lives in the donor' }], stitches: [] },
  { id: 'qmath_l10_activation512', group: 'add', kind: 'mlp_cluster', label: 'Math neurons L10', donor: 'Qwen/Qwen2.5-Math-1.5B', layers: [10, 10], params_m: 4, default_at: 6, summary: 'Qwen2.5-Math layer 10', task: 'math', power: 0.3, short: 'qmath.n10', status: 'candidate', evidence: 'no measurement yet', usable: false, usable_reason: 'No unit indices in the library file. The index file is only at /vol/units/qmath_l10_activation512.pt.', findings: [], stitches: [] },
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
const RECIPES = [
  { id: 'graft_block_qmath_8_19_llama1b', title: 'Qwen-Math layers 8..19 grafted into Llama-1B at layer 5', status: 'rejected', status_reason: 'ties a random block and loses to a rank-1 LoRA', host: HOST, builds: 'host Llama-3.2-1B-Instruct, graft qmath_block_8_19 at layer 5, heal 300 steps', components: ['qmath_block_8_19'], models: [HOST, 'Qwen/Qwen2.5-Math-1.5B'], skills: ['math', 'graft'], evidence: 'block graft vs a random same-length block (GSM8K search accuracy, mean over seeds): 0.352 against 0.363', source: 'results/round3_rows.json (run 3)', syntax_gap: null, no_start_reason: null, start_problems: [], starting_program: { host: HOST, seed: 0, masks: null, task_vectors: [], grafts: [{ id: 'qmath_block_8_19', at: 5, control: null, control_seed: 0, glue_rank: 64 }], heal: { steps: 300, data: 'gsm8k_train', lr: 0.0003, batch_size: 8, seq_len: 384 } } },
  { id: 'lora_r1_gsm8k_llama1b', title: 'Rank-1 LoRA on Llama-1B, GSM8K', status: 'validated', status_reason: null, host: HOST, builds: 'host Llama-3.2-1B-Instruct, LoRA rank 1, heal 300 steps', components: [], models: [HOST], skills: ['math', 'lora'], evidence: 'LoRA rank 1 vs the host (GSM8K search accuracy): 0.435 against 0.25', source: 'results/round3_rows.json (run 3)', syntax_gap: null, no_start_reason: null, start_problems: [], starting_program: { host: HOST, seed: 0, masks: null, task_vectors: [], grafts: [], heal: { steps: 300, data: 'gsm8k_train', lr: 0.0003, batch_size: 8, seq_len: 384 }, lora: { rank: 1 } } },
  { id: 'tap_mmlu_llama1b_qwen7b_l20_h10', title: 'Llama-1B reads Qwen2.5-7B layer 20 at layer 10 (MMLU)', status: 'rejected', status_reason: 'a tap runs a second model', host: HOST, builds: 'host Llama-3.2-1B-Instruct, tap Qwen2.5-7B-Instruct layer 20', components: [], models: [HOST, 'Qwen/Qwen2.5-7B-Instruct'], skills: ['knowledge', 'tap'], evidence: 'tap vs host LoRA rank 16 (MMLU held-out accuracy): 0.683 against 0.477', source: 'results/round7_rows.json (run 7)', syntax_gap: null, no_start_reason: 'The builder cannot hold taps, tap_train yet.', start_problems: [], starting_program: null },
];
const FINDINGS = [
  { id: 'f_mock_experts_donor_only', title: 'Skill maps say where a skill lives in the donor', claim: 'A skill map says which layers and experts matter for a skill inside the donor. It does not show that the part helps a host.', setup: 'donor skill maps on LFM2-8B-A1B', numbers: { median_split_half_r: 0.96, min_split_half_r: 0.73 }, numbers_source: 'atlas/pool_entries', status: 'candidate', status_reason: null, source: 'atlas/pool_entries/LiquidAI__LFM2-8B-A1B.json', skills: ['biology'], evidence: 'split-half agreement of the layer maps', about: { parts: ['map_expert_lfm2_8b_biology_l14'], models: ['LiquidAI/LFM2-8B-A1B'] } },
  { id: 'f_mock_graft_random', title: 'A cross-family MLP graft did not beat its random control', claim: 'A 512-neuron Qwen2.5-Math-1.5B layer 18 cluster grafted into Llama-1B at layer 10 scored 25.0 on GSM8K search; a random control scored 33.0.', setup: 'run 1: Llama-3.2-1B-Instruct host, 300 heal steps, 1 seed', numbers: { host: 25.0, graft: 25.0, random_control: 33.0 }, numbers_source: 'results/README.md', status: 'candidate', status_reason: null, source: 'results/graft_run1_s0.json', skills: ['math', 'graft'], evidence: 'graft vs random cluster', about: { parts: ['qmath_l18_contrast512'], models: [HOST, 'Qwen/Qwen2.5-Math-1.5B'] } },
  { id: 'f_mock_lora_bar', title: 'A rank-1 LoRA beats block grafts on Llama-1B', claim: 'On Llama-1B a rank-1 LoRA scores 43.5 on GSM8K search; the donor blocks score 35 to 36.', setup: 'run 3', numbers: { lora_r1: 43.5, blocks: [35.2, 36.3] }, numbers_source: 'results/README.md', status: 'validated', status_reason: null, source: 'results/round3_rows.json', skills: ['lora'], evidence: 'rank-1 LoRA vs block graft', about: { parts: [], models: [HOST] } },
];
// demo specs read as drafted by the agent model, as a real draft with an LLM does
const DEMO_DRAFTER = DEMO ? 'llm:openai_compat' : 'the planner';

/* ---------- demo mode: the real parts library ---------- */
// Demo mode shows the real library the server holds (GET /api/catalog and the library lists), not the
// small list above. The server's made-up stub parts are left out. The mock part ids the demo programs
// use map to real library parts. With no server (a plain static host), the small list above stays.
const REAL_ID = {
  qmath_l18_contrast512: 'mlp_cluster_qwen25math15b_l16_gsm8k',
  qmath_block_8_19: 'map_block_liquidai__lfm2-1.2b_math_l8_12',
  r1_block_10_21: 'block/LiquidAI/LFM2-350M-Math/4-14',
  mistral_heads_8: 'attn_heads_qwen25math15b_l20_gsm8k',
  qmath_l10_activation512: 'mlp_cluster_qwen25math15b_l17_gsm8k',
  phi_block_4_11: 'block/Qwen/Qwen2.5-0.5B-Instruct/10-19',
  lfm_experts_2: 'block/LiquidAI/LFM2-700M/6-6',
  gemma_block_5_12: 'block/Qwen/Qwen2.5-7B-Instruct/17-17',
  qcoder_block_6_17: 'block/Qwen/Qwen2.5-Coder-0.5B-Instruct/0-11',
  saul_block_4_13: 'block/Equall/Saul-7B-Instruct-v1/14-14',
};
async function loadRealLibrary() {
  if (!DEMO) return null;
  const get = (u) => fetch(u, { cache: 'no-store' }).then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))));
  try {
    const [cat, rec, fin, sti] = await Promise.all([get('/api/catalog'), get('/api/library/recipes'), get('/api/library/findings'), get('/api/library/stitches').catch(() => ({ stitches: [] }))]);
    if (!cat || !Array.isArray(cat.parts) || !cat.parts.length) return null;
    return { cat, recipes: rec.recipes || [], findings: fin.findings || [], stitches: sti.stitches || [] };
  } catch (e) { return null; }
}
// plain words for library skill tags
const SKILL_WORD = { math: 'Math', code: 'Code', law: 'Law', medicine: 'Medicine', knowledge: 'Knowledge', multilingual: 'Multilingual', science: 'Science', reasoning: 'Reasoning', biology: 'Biology', chemistry: 'Chemistry', physics: 'Physics', sql: 'SQL', facts: 'Facts' };
const WORD_SKILL = { math: 'math', code: 'code', law: 'law', medicine: 'medicine', knowledge: 'knowledge', multilingual: 'multilingual', science: 'science', reasoning: 'reasoning', biology: 'biology', chemistry: 'chemistry', physics: 'physics', sql: 'sql', facts: 'facts', rust: 'code_rust', spanish: 'prose_es', french: 'prose_fr', chinese: 'prose_zh', japanese: 'prose_ja', hindi: 'prose_hi', arabic: 'prose_ar', russian: 'prose_ru', german: 'prose_de', clinical: 'clinical_knowledge' };
const LABEL_SPLIT = /^(.*?)( (?:block|neuron cluster|attention heads|expert|inserted layers?)\b.*)$/;
// a few library labels name a skill the part's own tags do not list ("Math block" tagged law): name it by its tag
function relabel(p) {
  const m = LABEL_SPLIT.exec(p.label || '');
  if (!m || !(p.skills || []).length) return p.label;
  const own = WORD_SKILL[m[1].toLowerCase().split(' ')[0]];
  if (!own || p.skills.includes(own)) return p.label;
  const s = p.skills.find((x) => SKILL_WORD[x]);
  return s ? SKILL_WORD[s] + m[2] : p.label;
}
function primarySkill(p) {
  const w = WORD_SKILL[String(p.label || '').toLowerCase().split(' ')[0]];
  return w || (p.skills || [])[0] || null;
}
const TASK_OF_SKILL = (s) => {
  if (!s) return 'general';
  if (s === 'math' || s === 'gsm8k' || s === 'mathematics_theory') return 'math';
  if (s === 'code' || s.startsWith('code_') || s === 'sql' || s === 'computer_science') return 'code';
  if (s === 'law') return 'legal';
  if (s === 'medicine' || s.startsWith('clinical_')) return 'medicine';
  if (s === 'multilingual' || s.startsWith('prose_') || s.startsWith('en_to_') || s === 'translation') return 'multilingual';
  if (['science', 'biology', 'chemistry', 'physics', 'science_papers'].includes(s)) return 'science';
  return 'general';
};
// the task a part serves, for the simulated scores: from its label and its skill tags
function taskOf(p) { return TASK_OF_SKILL(primarySkill(p)); }
// internal build text must not reach a card: file paths, script names and bookkeeping phrases
function cleanEvidence(e) {
  let t = String(e || '');
  t = t.replace(/\s*Compiler record built from the library file\./g, '').replace(/,?\s*as recorded by maps\.py/g, '');
  t = t.replace(/[;,]?\s*\(?(?:atlas|person_a|results|units|parts_extracted|wsp_eval|wsp)\/[^\s,;)]*\)?/g, '');
  t = t.replace(/[;,]?\s*[\w./-]+\.(?:json|jsonl|pt|py|md)\b/g, '');
  t = t.replace(/;?\s*units on the Modal volume/g, '').replace(/pilot set for the swarm/g, 'pilot part');
  t = t.replace(/\(summed layer score over the block, nats[^)]*\)?\)*/g, '(summed layer score over the block)');
  t = t.replace(/no measurement yet: no graft or mask-algebra run[^.]*\.?/i, 'Not tested yet. No run has measured what this removes from the host.');
  t = t.replace(/^no measurement yet(: [^.]*)?\.?$/i, 'Not tested yet. No run has measured this part.');
  t = t.replace(/\.\.\.+/g, '').replace(/\s+([.,;])/g, '$1').replace(/[;,]\s*$/, '').trim();
  if (t && !/[.)]$/.test(t)) t += '.';
  return t.charAt(0).toUpperCase() + t.slice(1);
}
// finding and recipe text: drop paths, script names and "Person B" notes
function scrubText(x) {
  if (typeof x !== 'string') return x;
  return x.replace(/\s*\((?:[^()]*(?:\.py|\.json|\.jsonl|\.md|results\/|atlas\/|wsp_eval|parts_extracted)[^()]*)\)/g, '').replace(/Person [AB] /g, '').replace(/\b(?:results|atlas|person_a|units|parts_extracted|wsp_eval|wsp)\/[\w./-]*/g, 'the run logs').replace(/\b[\w-]+\.py\b/g, 'the compiler').replace(/\s{2,}/g, ' ').trim();
}
function scrubRecord(o) {
  for (const k of ['title', 'claim', 'setup', 'evidence', 'status_reason', 'source', 'builds']) if (typeof o[k] === 'string') o[k] = scrubText(o[k]);
  if (typeof o.source === 'string' && /the run logs|^$/.test(o.source)) o.source = 'Run logs';
  if (typeof o.no_start_reason === 'string') o.no_start_reason = /not in the compiler catalog/i.test(o.no_start_reason) ? 'One of its parts is not in the library yet.' : /cannot hold/i.test(o.no_start_reason) ? 'This recipe needs a step the builder does not have yet.' : scrubText(o.no_start_reason);
  if (Array.isArray(o.start_problems)) o.start_problems = o.start_problems.map((x) => (typeof x === 'string' ? (/not in the compiler catalog/i.test(x) ? 'One of its parts is not in the library yet.' : scrubText(x)) : x));
  return o;
}
const REAL = await loadRealLibrary();
if (REAL) {
  const kept = REAL.cat.parts.filter((p) => p.source !== 'stub');
  PARTS.length = 0;
  const seen = new Map();
  for (const p of kept) {
    const { usable_reason, from_library, source, ...q } = p;
    q.label = relabel(p);
    q.skills = [...new Set(q.skills || [])];
    q.evidence = cleanEvidence(q.evidence);
    const span = (q.layers ? q.layers[1] - q.layers[0] : 0) + 1;
    const key = q.label + '|' + q.donor;
    seen.set(key, (seen.get(key) || 0) + 1);
    q.params_m = Math.round((q.params_m != null ? q.params_m : 45 * span) * 10) / 10;
    PARTS.push({ ...q, task: 'general', power: 0.3, short: q.id.slice(0, 24), usable_reason: usable_reason ? (/tap|second model/i.test(usable_reason) ? 'Runs a second model at inference, so it cannot be built into one network.' : /unit indices|index file/i.test(usable_reason) ? 'The unit list for this part is not stored, so it cannot be placed yet.' : /profile/i.test(usable_reason) ? 'The donor is not profiled yet, so the size cannot be checked.' : cleanEvidence(usable_reason)) : null });
  }
  // cards that share one title and donor get a number, so they can be told apart
  const idx = new Map();
  for (const p of PARTS) {
    p.task = taskOf(p);
    const key = p.label + '|' + p.donor;
    if (seen.get(key) > 1) { const n = (idx.get(key) || 0) + 1; idx.set(key, n); const u = /(?:top|mlp|act\w*?)(\d{2,4})/.exec(p.id); p.label = p.label + (u ? ' (' + u[1] + ' units)' : ' #' + n); }
    const hh = hash(p.id) % 100;
    p.power = p.status === 'validated' ? 0.5 : hh < 14 ? 0.52 : hh < 40 ? 0.38 : 0.3;
  }
  // the parts the demo's winning programs use are the strong ones in the simulated scores
  const strong = new Set(['qmath_l18_contrast512', 'qmath_block_8_19', 'r1_block_10_21', 'mistral_heads_8', 'qmath_l10_activation512'].map((id) => REAL_ID[id]));
  for (const p of PARTS) if (strong.has(p.id)) { p.power = 0.5; p.power_fixed = 0.5; }
  MASKS.length = 0;
  for (const m of REAL.cat.masks || []) {
    const { usable_reason, ...q } = m;
    const nm = String(q.label || '').replace(/^BBEH\s+/i, '').replace(/_/g, ' ').toLowerCase();
    MASKS.push({ ...q, label: String(q.label || '').replace(/_/g, ' '), summary: 'Removes the ' + nm + ' skill from the host.', evidence: cleanEvidence(q.evidence), usable: q.status !== 'rejected' });
  }
  // Sapien builds one network: recipes and findings about taps (a second model read at inference) stay out
  const isTap = (x) => (x.skills || []).includes('tap') || /\btap\b|tap_train/i.test(String(x.no_start_reason || '') + ' ' + String(x.id || ''));
  RECIPES.length = 0; RECIPES.push(...REAL.recipes.filter((r) => !isTap(r)).map((r) => scrubRecord({ ...r })));
  FINDINGS.length = 0; FINDINGS.push(...REAL.findings.filter((f) => !isTap(f)).map((f) => scrubRecord({ ...f })));
  const gone = new Set(REAL.findings.filter(isTap).map((f) => f.id));
  for (const p of PARTS) p.findings = (p.findings || []).filter((f) => !gone.has(f.id));
}
// the slice from the language write-in (results/README.md, 2026-11-15): Qwen2.5-7B-Instruct layers 4 to 20
if (DEMO) {
  const m = V.LANG;
  const mean5 = (a) => a.slice(0, 5).reduce((x, y) => x + y, 0) / 5;
  const slice = { id: V.LANG_SLICE_ID, group: 'add', kind: 'block', label: 'Multilingual block L4-20', donor: 'Qwen/Qwen2.5-7B-Instruct', layers: [4, 20], params_m: Math.round((m.graft_b - HOST_B) * 1000), default_at: 4, summary: 'Qwen2.5-7B-Instruct layers 4 to 20', task: 'multilingual', power: 0.62, power_fixed: 0.62, short: 'qwen7b.l4-20', status: 'validated',
    evidence: 'Language write-in, 2026-11-15: grafted into Llama-3.2-1B-Instruct, Belebele in five languages averages ' + mean5(m.graft).toFixed(1) + ' against ' + mean5(m.base).toFixed(1) + ' for the base and ' + mean5(m.scrambled).toFixed(1) + ' with the donor weights scrambled (900 test items per language, mean of 3 seeds).',
    usable: true, skills: ['multilingual'], findings: [], stitches: [], params_b: m.graft_b - HOST_B };
  const at = PARTS.findIndex((p) => p.id === slice.id);
  if (at >= 0) PARTS[at] = slice; else PARTS.unshift(slice);
}
/** A part id the demo programs use, as a real library id when the real library is loaded. */
const realId = (id) => (REAL && REAL_ID[id]) || id;

attachMockFindings();
function attachMockFindings() {
  for (const p of PARTS) if (!p.findings) p.findings = [];
  const graft = PARTS.find((p) => p.id === 'qmath_l18_contrast512');
  if (graft) graft.findings = [{ id: 'f_mock_graft_random', title: 'A cross-family MLP graft did not beat its random control' }];
}
const PART = Object.fromEntries(PARTS.map((p) => [p.id, p]));
const GRAFT_PARTS = PARTS.filter((p) => p.usable !== false && p.status !== 'rejected');
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
  if (p.heal) L.push('heal steps ' + p.heal.steps + ' data ' + (p.heal.data || 'math'));
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
      prog.heal = { steps: o.steps != null ? int(o.steps, 'steps') : 300, data: o.data || 'math', lr: 0.0003, batch_size: 8, seq_len: 384 };
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
// The part task a spec asks for. Requests that no library skill names fall back to general parts.
const specPT = (spec) => ({ support: 'general', creative: 'general', knowledge: 'general', compact: 'general' }[spec.task_type] || spec.task_type);
const DOMAIN_DONOR = { legal: /law|saul|legal/i, medicine: /medic|bio|clinic/i, code: /coder|code/i, math: /math/i, multilingual: /qwen2\.5-7b|qwen3|gemma-?[34]/i };
function powerOf(spec, part) {
  if (part.power_fixed != null) return part.power_fixed;
  const re = DOMAIN_DONOR[specPT(spec)];
  const base = part.power != null ? part.power : 0.3;
  return re && part.task === specPT(spec) && re.test(part.donor) ? Math.max(base, 0.55) : base;
}
function relevance(spec, part) {
  const t = specPT(spec);
  if (part.task === t) return 1;
  if (part.task === 'general') return t === 'general' ? 1 : 0.3;
  return t === 'general' ? 0.35 : 0.05;
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
const targetsOf = (spec) => spec.benchmarks.filter((b) => b.kind === 'target');
const tmeanOf = (spec, scores) => { const t = spec.main_only ? targetsOf(spec).slice(0, 1) : targetsOf(spec); return t.length ? t.reduce((a, b) => a + (scores[b.id] || 0), 0) / t.length : 0; };
const benchLabel = (spec) => { const t = targetsOf(spec); return spec.target_label || (t.length === 1 ? t[0].name : 'the mean of ' + t.length + ' targets'); };
// scores pinned for programs the demo shows as finished results, so scoring the same program again gives the same number
const pinned = new Map();
const pinKey = (spec, p) => spec.id + ':' + hash(JSON.stringify({ g: (p.grafts || []).map((g) => [g.id, g.at, g.glue_rank || null]), h: p.heal ? p.heal.steps : 0, l: p.lora ? p.lora.rank : 0, m: p.masks ? 1 : 0 }));
const ctrlRatio = (spec, p) => 0.35 + (hash(spec.id + JSON.stringify(p)) % 150) / 1000;
function evalProgram(spec, p) {
  const key = pinKey(spec, p);
  if (pinned.has(key)) { const hit = clone(pinned.get(key)); hit.params_b = paramsOf(p); return hit; }
  const fit = {};
  let gain = 0;
  let penalty = 0;
  const gs = p.grafts || [];
  const seenId = new Set();
  gs.forEach((g, i) => {
    const part = PART[g.id];
    if (!part) return;
    const f = fitOf(part, g.at);
    fit[g.id + '@' + g.at] = Math.round(f * 100) / 100;
    const glue = g.glue_rank ? 1 - Math.abs(Math.log2(g.glue_rank / 16)) * 0.05 : 0.95;
    // a second copy of the same part adds nothing
    if (!seenId.has(g.id)) gain += powerOf(spec, part) * (f - 0.4) * relevance(spec, part) * glue;
    seenId.add(g.id);
    for (let j = 0; j < i; j++) if (Math.abs(gs[j].at - g.at) <= 1) penalty += 0.04;
  });
  const healBonus = p.heal ? 0.02 + 0.025 * Math.min(p.heal.steps, 900) / 300 : -0.03 * (gs.length ? 1 : 0);
  const loraBonus = p.lora ? 0.05 + 0.02 * Math.log2((p.lora.rank || 16) / 8) : 0;
  const maskCount = (() => { let n = 0; (function w(x) { if (!x) return; if (x.op === 'use') n++; else { w(x.a); w(x.b); } })(p.masks); return n; })();
  const scores = {};
  let tsum = 0, tn = 0;
  for (const b of spec.benchmarks) {
    const nz = noise(JSON.stringify(p) + b.id) * 0.012;
    if (b.kind === 'target') {
      const lift0 = (gain + healBonus + loraBonus) * (DEMO ? 0.35 : 0.66) * (b.id === 'yours' ? 1.05 : 1);
      const room = b.id === 'gsm8k' ? 0.12 : Math.max(0.06, ((b.target || b.base + 0.1) - b.base) * 1.4);
      const lift = DEMO ? room * Math.tanh(lift0 / room) : lift0;
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
    const r = ctrlRatio(spec, p);
    const cs = {};
    for (const b of targetsOf(spec)) cs[b.id] = Math.round(clamp(b.base + (scores[b.id] - b.base) * r, 0, 1) * 1000) / 1000;
    control = { scores: cs };
    vs = Math.round((scores[first.id] - cs[first.id]) * 1000) / 1000;
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
  if (e.run_params_b > lim.max_run_params_b + 1e-6) out.push('Over the speed limit. This build runs an estimated ' + f2(e.run_params_b) + 'B parameters per token. The limit is ' + f2(lim.max_run_params_b) + 'B.');
  if (e.heal_min > lim.max_heal_min + 1e-6) out.push('Over the heal time limit. The training pass is estimated at ' + f2(e.heal_min) + ' minutes. The limit is ' + lim.max_heal_min + ' minutes.');
  return out;
}
/* ---------- component verdicts (simulated, the same part always gets the same verdict) ---------- */
const VERDICTS = ['contributes', 'no better than random', 'harmful'];
const STAGE_NAMES = ['place', 'translate in', 'translate out', 'heal 1', 'heal 2', 'verify'];
function pipelineOf(p) { return (p.grafts || []).some((g) => PART[g.id]) ? 'synth' : 'classic'; }
const LAYER_END = /\sL\d+(?:-\d+)?(?:\s\(\d+ units\)|\s#\d+)?$/;
const layerSpan = (part) => (part.layers[0] === part.layers[1] ? ' L' + part.layers[0] : ' L' + part.layers[0] + '-' + part.layers[1]);
/** A part's name with its donor layers, once. Library labels already end in the layers. */
const partLabel = (part) => (LAYER_END.test(part.label) ? part.label : part.label + layerSpan(part));
function componentsFor(spec, p) {
  const gs = (p.grafts || []).filter((g) => PART[g.id]);
  if (!gs.length) return null;
  const tgt = spec.benchmarks.find((b) => b.kind === 'target');
  return gs.map((g) => {
    const part = PART[g.id];
    const pw = powerOf(spec, part);
    const onTask = relevance(spec, part) >= 1;
    const verdict = DEMO ? (onTask && pw >= 0.4 ? 'contributes' : onTask && pw >= 0.35 ? 'no better than random' : hash(part.id) % 4 === 0 ? 'harmful' : 'no better than random') : VERDICTS[PARTS.indexOf(part) % 3];
    const h = hash(g.id) % 100 / 1000;
    const contribution = verdict === 'contributes' ? 0.02 + h * 0.5 : verdict === 'harmful' ? -0.02 - h * 0.4 : (h - 0.05) * 0.1;
    const sameTok = !/Qwen|deepseek/.test(part.donor);
    const r3 = (x) => Math.round(x * 1000) / 1000;
    return { id: g.id, label: partLabel(part), kind: part.kind, donor: part.donor, host_layer: g.at, verdict, stub: true,
      detail: { skill: tgt.id, contribution: r3(contribution), text_gain: r3(verdict === 'contributes' ? 0.02 + h : -h * 0.2), content_gain: r3(verdict === 'contributes' ? 0.03 + h * 0.6 : verdict === 'harmful' ? -0.04 : 0.004),
        transfer: sameTok ? r3(verdict === 'contributes' ? 0.3 + h * 4 : verdict === 'harmful' ? -0.1 : 0.04) : null, fit_r2: r3(0.5 + h * 3), gate_domain: 0.8, gate_general: 0.1, part_params: part.params_m * 1e6, closed: false } };
  });
}
function stagesFor(spec, p, comps) {
  if (!comps) return null;
  const tid = spec.benchmarks.find((b) => b.kind === 'target').id;
  const t0 = spec.benchmarks.find((b) => b.id === tid).base;
  const mapAll = (v) => Object.fromEntries(targetsOf(spec).map((b) => [b.id, Math.round(clamp(b.base + (v - t0) * ((b.target || b.base + 0.1) - b.base) / Math.max(0.05, (spec.benchmarks.find((x) => x.id === tid).target || t0 + 0.1) - t0), 0.01, 0.98) * 1000) / 1000]));
  const gain = comps.reduce((a, c) => a + c.detail.contribution, 0);
  const harmful = comps.some((c) => c.verdict === 'harmful');
  const pr = (x) => mapAll(x);
  const v = [t0, t0 + gain * 0.4, t0 + gain * 0.5, t0 + gain * 0.8];
  const rows = [
    ['place', null, null, pr(v[0])], ['translate in', 12.7, pr(v[0]), pr(v[1])], ['translate out', 0.3, pr(v[1]), pr(v[2])],
    ['heal 1', 10.3, pr(v[2]), pr(v[2] - 0.04)], ['heal 2', 11.1, pr(v[2] - 0.04), harmful ? pr(v[2] - 0.08) : pr(v[3])], ['verify', 144.4, null, null],
  ];
  return rows.map(([name, seconds, before, after], i) => ({ name, seconds, practice_before: before, practice_after: after, rolled_back: name === 'heal 2' && harmful, detail: name === 'verify' ? 'Scores the saved network and runs the random control compile.' : '', stub: true }));
}
function driftFor(p, comps) {
  if (!comps) return null;
  return { base_kl: Math.round((0.01 + (hash(JSON.stringify(p.grafts)) % 90) / 1000) * 1000) / 1000, agree: Math.round((0.95 - (hash(JSON.stringify(p.grafts)) % 20) / 100) * 1000) / 1000, tokens: 2400, text: DEMO ? 'on 2,400 tokens' : 'estimated offline, not measured' };
}
function withSynth(spec, p, ev) {
  const comps = componentsFor(spec, p);
  ev.pipeline = comps ? 'synth' : 'classic';
  ev.components = comps;
  ev.stages = stagesFor(spec, p, comps);
  ev.drift = driftFor(p, comps);
  ev.sizes = null;
  const healed = !!(p.heal || p.lora);
  const heal = new URLSearchParams(location.search).get('heal') === '1';
  ev.heal_note = !comps && healed && heal ? 'The heal recipe is being fixed. Healing on dataset text currently damages the base model, so scores after a heal are not reliable yet.' : null;
  return ev;
}
function mockComparison(spec, ev, p, running = false) {
  const ts = targetsOf(spec).slice(0, spec.main_only ? 1 : undefined);
  const mean = (f) => { const v = ts.map(f); return v.some((x) => x == null) ? null : v.reduce((a, b) => a + b, 0) / v.length; };
  const result = mean((b) => ev.scores[b.id]);
  const rows = [];
  const add = (id, label, value, note, stub) => rows.push({ id, label, split: 'search', per_task: {}, value, measured: value != null, note, stub: !!stub, result_value: result, delta: value == null ? null : Math.round((result - value) * 1000) / 1000, beaten: value == null ? null : result - value > 0.02, within_margin: value == null ? null : result - value > 0 && result - value <= 0.02 });
  const pend = !!spec.pending_arms;
  add('base', 'Base', mean((b) => b.base), 'the host alone, same items and same scorer', true);
  if (pend) add('lora', 'Base + LoRA', null, 'Pending. This arm is still being scored.', false);
  else add('lora', 'Base + LoRA', mean((b) => b.lora), ts.every((b) => b.lora != null) ? 'a rank-16 LoRA on the same data' : SAY('no LoRA run for every target', 'not measured for every target'), false);
  if (pend) add('distill', 'Base + full fine-tune', null, 'Pending. This arm is still being scored.', false);
  else add('distill', 'Base + distillation', mean((b) => b.base + ((b.lora != null ? b.lora : b.base + 0.1) - b.base) * 0.8), SAY('a copy of the base distilled on the same data', 'offline estimate, not a measurement'), true);
  const ctl = ev.control ? (ts.every((b) => ev.control.scores[b.id] != null) ? mean((b) => ev.control.scores[b.id]) : ev.control.scores[ts[0].id]) : null;
  add('random', spec.control_label || 'Random components', ctl, ev.control ? (spec.control_note || 'the same build with random components of the same size') : SAY('The control runs a build with random parts of the same size. It needs at least one part.', 'not measured yet. The control runs a build with random parts of the same size.'), true);
  const limB = spec.limits && spec.limits.max_total_params_b;
  const far = limB && limB >= 2 * HOST_B;
  if (spec.best_ref) add('best_model', spec.best_ref.label, spec.best_ref.value, spec.best_ref.note, false);
  else if (far) add('best_model', 'Best model at ' + f2(limB) + 'B', mean((b) => b.base + 0.5 * ((b.target || b.base + 0.15) - b.base)), 'the best library model within the limit', true);
  const bad = ev.status !== 'ok' || limitProblems(spec.limits, estimateOf(p)).length > 0;
  const failedRows = rows.filter((r) => r.measured && !r.beaten);
  const open = rows.filter((r) => !r.measured);
  let verdict = 'success';
  if (bad || failedRows.length) verdict = 'failed'; else if (open.length) verdict = 'unproven';
  const ahead = failedRows.filter((r) => r.within_margin);
  const behind = failedRows.filter((r) => !r.within_margin);
  let headline;
  if (verdict === 'failed') {
    if (bad) headline = 'Failed attempt. The build collapsed or went over a limit.';
    else if (!behind.length) { headline = running ? 'Ahead of ' + ahead.map((r) => r.label + ' by ' + (r.delta * 100).toFixed(1)).join(', ') + ' points, inside the 2 point margin. The run is still going.' : 'Not a clear win. It is ahead of ' + ahead.map((r) => r.label + ' by ' + (r.delta * 100).toFixed(1)).join(', ') + ' points, which is inside the 2 point margin.'; if (running) verdict = 'unproven'; }
    else headline = (running ? 'Not ahead yet. ' : 'Failed attempt. ') + 'It does not beat ' + behind.map((r) => r.label).join(', ') + '.' + (ahead.length ? ' It is ahead of ' + ahead.map((r) => r.label).join(', ') + ' but inside the 2 point margin.' : '');
  } else if (verdict === 'unproven') {
    headline = pend ? 'Beats ' + rows.filter((r) => r.beaten).map((r) => r.label).join(', ') + '. Not proven yet: ' + open.map((r) => r.label).join(' and ') + ' are still being scored.' : 'Not proven yet: ' + open.map((r) => r.label).join(', ') + SAY(running ? ' is still being measured.' : ' still needs a run.', ' not measured.');
  } else headline = 'Beats every reference by more than 2 points.';
  const comps = ev.components || null;
  const noComp = !!(comps && comps.length && comps.every((c) => c.verdict !== 'contributes'));
  let head = headline;
  if (noComp && !running) {
    verdict = 'failed';
    const why = 'No component contributes: ' + comps.map((c) => c.label + ' is ' + c.verdict).join(', ') + '. The score gain is not shown to come from the parts.';
    head = headline.startsWith('Failed attempt.') ? headline + ' ' + why : 'Failed attempt. ' + why;
  }
  return { verdict, headline: head, no_component_contributes: noComp && !running, margin: 0.02, result: { label: 'This build', value: result, per_task: {}, params_b: ev.params_b, heldout: null }, rows, beaten: rows.filter((r) => r.beaten).map((r) => r.id), not_beaten: failedRows.map((r) => r.id), unmeasured: open.map((r) => r.id), best_model_shown: !!(far || spec.best_ref), stub: true };
}

/* ---------- specs ---------- */
// one source of truth for a benchmark's host score: the same numbers appear on the spec, in runs and in the builder
const BENCH_LIB = {
  gsm8k: { id: 'gsm8k', name: 'GSM8K', kind: 'target', target: 0.68, base: 0.58, lora: 0.63, recommended: true },
  humaneval: { id: 'humaneval', name: 'HumanEval', kind: 'target', target: 0.35, base: 0.19, lora: 0.31, recommended: true },
  'mmlupro:law': { id: 'mmlupro:law', name: 'MMLU-Pro law', kind: 'target', target: 0.22, base: 0.12, lora: 0.17, recommended: true },
  'mmlupro:health': { id: 'mmlupro:health', name: 'MMLU-Pro health', kind: 'target', target: 0.26, base: 0.16, lora: 0.21, recommended: true },
  'mmlupro:chemistry': { id: 'mmlupro:chemistry', name: 'MMLU-Pro chemistry', kind: 'target', target: 0.18, base: 0.09, lora: 0.14, recommended: true },
  'mmlupro:biology': { id: 'mmlupro:biology', name: 'MMLU-Pro biology', kind: 'target', base: 0.107, target: 0.25, lora: 0.19, recommended: true },
  arc: { id: 'arc', name: 'ARC Challenge', kind: 'target', base: 0.526, target: 0.65, lora: 0.6, recommended: true },
  mmlu_es: { id: 'mmlu_es', name: 'MMLU Spanish', kind: 'target', base: 0.378, target: 0.5, lora: 0.38, recommended: true },
  if_guard: { id: 'if_guard', name: 'IFEval', kind: 'guard', floor: 0.45, base: 0.55, recommended: true },
  mmlu_guard: { id: 'mmlu_guard', name: 'MMLU', kind: 'guard', floor: 0.4, base: 0.446, recommended: true },
  hellaswag: { id: 'hellaswag', name: 'HellaSwag', kind: 'guard', base: 0.346, floor: 0.3 },
  truthfulqa: { id: 'truthfulqa', name: 'TruthfulQA', kind: 'guard', base: 0.36, floor: 0.32 },
};
// 'if_guard_t' and 'mmlu_t' are the guards used as targets, for requests that no other benchmark scores
const benchOf = (id) => {
  if (id === 'if_guard_t' || id === 'mmlu_t') { const b = clone(BENCH_LIB[id === 'mmlu_t' ? 'mmlu_guard' : 'if_guard']); b.kind = 'target'; delete b.floor; b.target = Math.round((b.base + 0.07) * 100) / 100; b.lora = Math.round((b.base + 0.035) * 1000) / 1000; b.recommended = true; return b; }
  return clone(BENCH_LIB[id]);
};
const LOOKS = { [V.DEMO_IDS.specMath]: V.MATH_LOOK, [V.DEMO_IDS.specSolver]: V.MATH_LOOK, sp_demo: V.MATH_LOOK, [V.DEMO_IDS.specLegal]: V.LEGAL_LOOK, [V.DEMO_IDS.specClause]: V.LEGAL_LOOK, [V.DEMO_IDS.specLang]: V.LANG_LOOK };
const KIND_NAME = { block: 'block', inserted_layer: 'inserted layer', mlp_cluster: 'neuron cluster', attn_heads: 'attention heads', expert: 'expert' };
const ingOf = (p) => ({ part_id: p.id, label: p.label, donor: p.donor, kind: KIND_NAME[p.kind] || p.kind });

/** "under 1.5B", "500M model": the size the request names, in billions, or null. */
function parseSize(prompt) {
  const m = /(?:under|below|up to|at most|max(?:imum)?(?: of)?|less than|within|fits? in)\s*(\d+(?:\.\d+)?)\s*(b|m|billion|million)\b/i.exec(prompt) || /(\d+(?:\.\d+)?)\s*(b|m)\b\s*(?:param\w*|model)/i.exec(prompt);
  if (!m) return null;
  const v = parseFloat(m[1]);
  return /^m/i.test(m[2]) ? v / 1000 : v;
}
const LANG_MENTION = { ar: /arabic/i, ja: /japanese/i, zh: /chinese|mandarin/i, ko: /korean/i, hi: /hindi/i };
const LANG_SKILL = { Spanish: 'prose_es', French: 'prose_fr', German: 'prose_de', Arabic: 'prose_ar', Japanese: 'prose_ja', Chinese: 'prose_zh', Hindi: 'prose_hi', Russian: 'prose_ru' };
function pickIngredients(profs, subSkills, roomB) {
  const room = roomB * 1000;
  const per = Math.max(2, Math.ceil(5 / profs.length));
  const out = [];
  for (const pr of profs) {
    const wanted = new Set(subSkills.length && pr === profs[0] ? subSkills : pr.skills);
    const wide = new Set(pr.skills);
    const score = (p) => {
      let s = 0;
      const prim = primarySkill(p);
      if (wanted.has(prim)) s += 4; else if (wide.has(prim)) s += 3; else if ((p.skills || []).some((x) => wanted.has(x))) s += 1.5;
      if (pr.donor && pr.donor.source !== '.' && pr.donor.test(p.donor) && s > 0) s += 2;
      if (p.status === 'validated') s += 2;
      if (s > 0) s += ((p.power != null ? p.power : 0.3) - 0.3) * 6;
      if (/^Not tested|^no measurement/i.test(p.evidence || '')) s -= 1; else if (/beat|held-out|z \d|against/i.test(p.evidence || '')) s += 1;
      return s - Math.log10(p.params_m + 1) * 0.3;
    };
    const pool = GRAFT_PARTS.filter((p) => p.params_m <= room && p.id !== V.LANG_SLICE_ID && !out.includes(p)).map((p) => [score(p), p]).filter(([sc]) => sc >= 3).sort((x, y) => y[0] - x[0]);
    const per_donor = {};
    let n = 0;
    for (const [, p] of pool) { if ((per_donor[p.donor] || 0) >= 2) continue; per_donor[p.donor] = (per_donor[p.donor] || 0) + 1; out.push(p); if (++n >= per) break; }
  }
  return out.slice(0, 6);
}
function makeSpec(prompt, files, id, extra = {}) {
  const text = prompt || '';
  const look = id ? LOOKS[id] : null;
  const cl = V.classify(text);
  let prof = cl.main;
  if (look === V.MATH_LOOK) prof = V.PROFILES.find((p) => p.key === 'math');
  if (look === V.LEGAL_LOOK) prof = V.PROFILES.find((p) => p.key === 'legal');
  if (look === V.LANG_LOOK) prof = V.PROFILES.find((p) => p.key === 'lang');
  const profs = [prof, ...(look ? [] : cl.extra)];
  // language-specific context
  const langs = prof.key === 'lang' ? Object.keys(LANG_MENTION).filter((c) => LANG_MENTION[c].test(text)) : [];
  const named = prof.key === 'lang' ? ['Spanish', 'French', 'German', 'Portuguese', 'Italian', 'Russian', 'Swahili', 'Bengali', ...langs.map((c) => V.LANG.names[c])].filter((n) => new RegExp(n, 'i').test(text) || langs.some((c) => V.LANG.names[c] === n)) : [];
  const sub = (prof.subs || []).find((x) => x[0].test(text)) || null;
  const ctx = { prompt: text, sub: sub ? sub[1] : null, langs: [...new Set(named)], belebele: langs.length > 0 };
  let bench = [];
  const benchIds = (pr) => pr.bench.map((x) => (sub && sub[3] && pr === prof && x === pr.bench[0] ? sub[3] : x));
  if (prof.key === 'lang' && langs.length) bench = [...V.langBenchmarks().filter((b) => b.kind === 'target' && langs.some((c) => b.id === 'belebele:' + c)), ...V.langBenchmarks().filter((b) => b.kind === 'guard')];
  else {
    for (const pr of profs) for (const k of benchIds(pr)) if (!bench.some((b) => b.id === benchOf(k).id) && bench.filter((b) => b.kind === 'target').length < 3) bench.push(benchOf(k));
    for (const k of prof.guards) { const g = benchOf(k); if (!bench.some((b) => b.id === g.id)) bench.push(g); }
  }
  const frows = (files || []).map((f) => {
    const rows = f.rows != null ? f.rows : Math.max(1, Math.round((f.size || 0) / 55));
    const sr = Math.max(1, Math.round(rows / 3));
    return { name: f.name, rows, score_rows: Math.min(sr, rows), heal_rows: Math.max(0, rows - sr) };
  });
  if (frows.length) bench.push({ id: 'yours', name: 'Your problems', kind: 'target', target: 0.6, base: null, lora: null, recommended: true });
  // size: from the request, else the profile's default
  const sizeNamed = parseSize(text);
  const wide = prof.key === 'lang' && !look;
  let maxp = sizeNamed != null ? sizeNamed : prof.key === 'compact' ? HOST_B : wide ? 6.0 : /laptop|cpu|phone|watch|edge/i.test(text) ? 2.0 : 2.5;
  const out = {
    id: id || rid('sp_'), title: V.titleFromPrompt(text, prof), summary: prof.summary(ctx), prompt: text, files: frows, task_type: prof.task, profile: prof.key, host: HOST,
    size: { max_params_b: maxp, min_b: Math.min(0.5, maxp), max_b: 8.0, host_params_b: HOST_B },
    budget: { evals: 30, compute_min: 45, min_evals: 5, max_evals: 120 },
    limits: { max_total_params_b: maxp, max_run_params_b: Math.min(maxp, wide ? 6.0 : 1.9), max_heal_min: wide ? 30 : 20 }, comparators: { distill: {} },
    benchmarks: bench, available_benchmarks: [], ingredients: [], drafted_by: 'rules', ...extra,
  };
  if (sizeNamed != null && sizeNamed < HOST_B) out.summary += ' The host alone is ' + HOST_B + 'B, so a limit of ' + (sizeNamed < 1 ? Math.round(sizeNamed * 1000) + 'M' : sizeNamed + 'B') + ' can only be met by deleting skills with masks. Adding parts will always go over.';
  else if (sizeNamed != null && prof.key !== 'compact') out.summary += ' It has to stay under ' + (sizeNamed < 1 ? Math.round(sizeNamed * 1000) + 'M' : sizeNamed + 'B') + ' parameters.';
  if (prof.fixedTitle || prof.key === 'compact') out.limits.max_run_params_b = Math.min(maxp, 1.9);
  // ingredients
  const room = Math.max(0, maxp - HOST_B);
  const maskIng = () => MASKS.slice(0, 3).map((m) => ({ part_id: m.id, label: 'Remove ' + String(m.label).replace(/^BBEH\s+/i, '').toLowerCase(), donor: HOST, kind: 'deletion mask' }));
  if (prof.masks || room < 0.06) out.ingredients = maskIng();
  else {
    const subSk = sub && sub[2] ? sub[2] : prof.key === 'lang' ? [...ctx.langs.map((n) => LANG_SKILL[n]).filter(Boolean), 'multilingual'] : [];
    const parts = pickIngredients(profs, subSk, room);
    out.ingredients = parts.map(ingOf);
    if (prof.key === 'lang' && room >= 3.9 && PART[V.LANG_SLICE_ID]) out.ingredients.unshift(ingOf(PART[V.LANG_SLICE_ID]));
  }
  const used = new Set(out.benchmarks.map((b) => b.id));
  out.available_benchmarks = Object.values(BENCH_LIB).filter((b) => !used.has(b.id)).map((b) => { const c = clone(b); delete c.recommended; delete c.lora; return c; });
  return look && DEMO ? demoLook(out, look) : DEMO ? ((out.drafted_by = DEMO_DRAFTER), out) : out;
}
// the fixed demo objects keep their fixed looks. Drafted specs keep what the request asked for.
function demoLook(spec, look) {
  const yours = spec.benchmarks.filter((b) => b.id === 'yours');
  spec.title = look.title; spec.summary = look.summary;
  if (look.prompt) spec.prompt = look.prompt;
  spec.budget = clone(look.budget); spec.size = clone(look.size); spec.limits = clone(look.limits);
  spec.benchmarks = [...clone(look.benchmarks).map((b) => { if (look !== V.LANG_LOOK) { delete b.research; delete b.reference_scores; } return b; }), ...yours];
  const ids = spec.ingredients.map((g) => g.part_id);
  if (look === V.LANG_LOOK) {
    spec.pending_arms = true; spec.target_label = 'Belebele'; spec.control_label = 'Scrambled donor'; spec.control_note = 'the same 5.2B build with the donor weights scrambled (2 seeds)';
    spec.best_ref = { label: 'Native Qwen2.5-1.5B', value: V.LANG.qwen15[0] / 100, note: 'a native 1.5B model, scored on the same items' };
    spec.task_type = 'multilingual'; spec.profile = 'lang'; spec.main_only = true;
    const want = [V.LANG_SLICE_ID];
    spec.ingredients = want.filter((x) => PART[x]).map((x) => ingOf(PART[x]));
    for (const g of pickIngredients([V.PROFILES.find((p) => p.key === 'lang')], ['multilingual'], 4.5).slice(0, 3)) spec.ingredients.push(ingOf(g));
  } else for (const pid of look.extraIngredients.map(realId)) if (PART[pid] && !ids.includes(pid)) spec.ingredients.push(ingOf(PART[pid]));
  const used = new Set(spec.benchmarks.map((b) => b.id));
  spec.available_benchmarks = Object.values(BENCH_LIB).filter((b) => !used.has(b.id)).map((b) => { const c = clone(b); delete c.recommended; delete c.lora; return c; });
  spec.drafted_by = DEMO_DRAFTER;
  return spec;
}
/* specs live in memory and in this browser's storage, so a reload or a shared tab keeps the draft */
const specs = new Map();
const store = {
  get(k) { try { const v = localStorage.getItem(storageKey(k)); return v ? JSON.parse(v) : null; } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(storageKey(k), JSON.stringify(v)); } catch (e) { /* storage can be off */ } },
};
const notFound = (msg) => { const e = new Error(msg); e.status = 404; return e; };
function saveSpec(s) {
  if (!s || String(s.id).startsWith('sp_demo')) return;
  const r = researchOf.get(s.id);
  const { benchmark_research, ...rest } = s;
  store.set('wsp.spec.' + s.id, { spec: rest, rs: r ? r.started : null, applied: r ? r.applied : false });
}
function remember(s) { specs.set(s.id, s); saveSpec(s); return s; }
function getSpec(id) {
  if (!specs.has(id) && id === 'sp_demo') specs.set(id, makeSpec('A small model that solves grade-school math word problems step by step. It has to run on a laptop with no GPU.', [], id));
  if (!specs.has(id)) {
    const st = store.get('wsp.spec.' + id);
    if (!st || !st.spec) throw notFound('That spec is not in this browser. Start a new one.');
    specs.set(id, st.spec);
    if (st.spec.request_type !== 'shrink') researchOf.set(id, { started: st.rs || demoNow() - 60000, plan: researchPlan(st.spec), mode: null, applied: st.applied !== false });
  }
  return specs.get(id);
}

/* ---------- benchmark research (simulated over about 7 to 25 seconds) ---------- */
const researchOf = new Map();
function researchPlan(spec) {
  if (DEMO && spec.id === V.DEMO_IDS.specLang) return V.langResearch();
  if (DEMO && spec.profile === 'lang' && spec.benchmarks.some((b) => b.id.startsWith('belebele:'))) return V.langResearch();
  if (DEMO && spec.task_type === 'math') return V.mathResearch();
  if (DEMO && spec.task_type === 'legal') return V.legalResearch();
  const prof = V.PROFILES.find((p) => p.key === spec.profile) || V.GENERAL_PROFILE;
  return V.profileResearch(spec, prof.research ? prof : V.GENERAL_PROFILE);
}
function startMockResearch(spec, lazy) {
  const mode = new URLSearchParams(location.search).get('rs');
  const seeded = String(spec.id).startsWith('sp_demo');
  researchOf.set(spec.id, { started: lazy ? null : demoNow(), plan: researchPlan(spec), mode, applied: false, finished_before: seeded && lazy });
}
function researchNow(id) {
  const r = researchOf.get(id);
  if (!r) return null;
  if (r.started == null) r.started = r.finished_before ? demoNow() - 120000 : demoNow() - 2500;
  const el = (demoNow() - r.started) / 1000;
  const dur = r.plan.dur || 7;
  const base = { provider: 'exa', usage: { input_tokens: 0, output_tokens: 0, web_searches: 3 }, started_at: wallOf(r.started) / 1000 };
  if (r.mode === 'skipped') return { ...base, status: 'skipped', provider: 'none', steps: [], reasoning: '', proposals: [], sources: [], error: 'No LLM is configured, so the rule-drafted benchmarks stay.' };
  const steps = r.plan.steps.filter((x) => x.t <= el);
  if (r.mode === 'error' && el > 3) return { ...base, status: 'error', steps, reasoning: '', proposals: [], sources: [], error: 'The search took too long. Try again.' };
  if (el < dur) return { ...base, status: 'running', steps, reasoning: '', proposals: [], sources: DEMO ? r.plan.sources.filter((x) => steps.some((st) => st.url === x.url)) : [], error: null };
  return { ...base, status: 'done', steps: r.plan.steps, reasoning: r.plan.reasoning, proposals: r.plan.proposals, sources: r.plan.sources, error: null, finished_at: base.started_at + dur };
}
function applyResearch(spec) {
  const r = researchOf.get(spec.id);
  const now = researchNow(spec.id);
  if (!r) return spec;
  if (now && now.status === 'done' && !r.applied) {
    r.applied = true;
    const removed = new Set(spec.removed_ids || []);
    for (const p of now.proposals) {
      if (!p.runnable_id || removed.has(p.runnable_id)) continue;
      const have = spec.benchmarks.find((b) => b.id === p.runnable_id);
      if (have) { have.research = true; have.reference_scores = p.reference_scores; continue; }
      if (!BENCH_LIB[p.runnable_id]) continue;
      const nb = clone(BENCH_LIB[p.runnable_id]);
      nb.target = p.suggested_target != null ? p.suggested_target : nb.target; nb.research = true; nb.reference_scores = p.reference_scores; delete nb.recommended;
      spec.benchmarks.push(nb);
      spec.available_benchmarks = spec.available_benchmarks.filter((a) => a.id !== nb.id);
    }
    saveSpec(spec);
  }
  spec.benchmark_research = now;
  return spec;
}

/* ---------- run simulation ---------- */
const runs = new Map();
const PHASES = [['thinking', 1], ['probing', 1], ['compiling', 1], ['scoring', 3]];
const DEMO_PHASES = [['thinking', 4], ['probing', 3], ['compiling', 6], ['scoring', 8]];
// near the end the agents slow down, so a watched run stays busy for many minutes
function stretch(run) {
  const f = run.evals_total ? run.evals_used / run.evals_total : 0;
  return run.hold ? 1 + 3 * clamp((f - 0.5) / 0.4, 0, 1) : 1;
}
const PLANNERS = [
  { id: 'planner-1', name: 'Planner 1', blurb: 'explores whole donor blocks', style: 'blocks' },
  { id: 'planner-2', name: 'Planner 2', blurb: 'explores small parts', style: 'small' },
  { id: 'planner-3', name: 'Planner 3', blurb: 'explores parts plus masks', style: 'masks' },
];
function newAgent(a, skew = 0) {
  return { skew, cycles: 0, id: a.id, name: a.name, role: a.id.split('-')[0], blurb: a.blurb, style: a.style, status: 'thinking', now: 'Reading the spec', best: null, attempts: 0, collapses: 0, lessons: 0, nudges_pending: 0, phase: 0, ticks: 0, cur: null, thinking: [], tools: [], history: [], nudges: [], bias: {} };
}
// a readable program name, like the server sends: "Math block L8-19 at L6, heal 300"
function readable(p) {
  const bits = (p.grafts || []).map((g) => (PART[g.id] ? PART[g.id].label : 'Part') + ' at L' + g.at + (g.glue_rank && g.glue_rank !== 16 ? ' r' + g.glue_rank : ''));
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
  if (run.activity.length > 1500) run.activity.shift();
}
function think(a, text) { a.thinking.push({ ts: now(), text }); if (a.thinking.length > 40) a.thinking.shift(); }
function tool(a, call, result, status = 'ok') { a.tools.push({ ts: now(), call, result, status }); if (a.tools.length > 40) a.tools.shift(); }

function newRun(spec, seed) {
  const run = {
    id: 's0-swarm-' + Math.floor(now()), spec: clone(spec), status: 'running', started: now(), elapsed: 0,
    evals_used: 0, evals_total: spec.budget.evals, compute_used: 0, compute_total: spec.budget.compute_min * 60,
    base_scores: Object.fromEntries(spec.benchmarks.map((b) => [b.id, b.base])), best: null, board: [], activity: [], curve: [], uses: {},
    agents: [newAgent({ id: 'analyst', name: 'Analyst', blurb: 'reads every result, writes the map' }), ...PLANNERS.map((p, i) => newAgent(p, [0, 3, 6][i])), newAgent({ id: 'judge', name: 'Judge', blurb: 'ranks, runs controls, guards the budget' })],
    seed: seed || null, tickN: 0, round: 1, actId: 0,
  };
  const an = run.agents[0], ju = run.agents[4];
  an.now = 'Waiting for the first results'; an.status = 'idle';
  ju.now = 'Waiting for the first results'; ju.status = 'idle';
  logAct(run, 'judge', 'system', 'info', SAY('Run started: ' + run.evals_total + ' evals, ' + spec.budget.compute_min + ' compute minutes.', 'Run started on the offline backend: ' + run.evals_total + ' evals, ' + spec.budget.compute_min + ' compute minutes.'));
  if (seed) logAct(run, 'judge', 'seed', 'info', 'Seeded with a program from the builder: ' + readable(seed));
  runs.set(run.id, run);
  ensureClock();
  return run;
}
let clock = 0;
function ensureClock() {
  if (clock) return;
  if (!DEMO) { clock = setInterval(() => runs.forEach(tickRun), 1000); return; }
  const loop = () => { runs.forEach(tickRun); clock = setTimeout(loop, 1000 / demoSpeed()); };
  clock = setTimeout(loop, 1000 / demoSpeed());
}

function genProgram(run, a) {
  const spec = run.spec;
  const pt = specPT(spec);
  const room = (spec.size.max_params_b - HOST_B) * 1000;
  const fitsRoom = (p) => p.params_m <= room;
  a.cross = false;
  // the parts that serve this task and fit the size limit; a deliberate cross-domain probe is rare and never first
  let pool = GRAFT_PARTS.filter((p) => relevance(spec, p) >= 1 && fitsRoom(p));
  if (REAL && run.tickN > 15 && Math.random() < 0.05) {
    const off = GRAFT_PARTS.filter((p) => relevance(spec, p) < 1 && fitsRoom(p));
    if (off.length) { pool = off; a.cross = true; }
  }
  if (!pool.length) pool = GRAFT_PARTS.filter(fitsRoom);
  if (!pool.length) pool = [...GRAFT_PARTS].sort((x, y) => x.params_m - y.params_m).slice(0, 5);
  // lean on the parts that score well for this task, so the run finds a real leader
  if (REAL && !a.cross && Math.random() < 0.5) { const good = pool.filter((p) => powerOf(spec, p) >= 0.5); if (good.length) pool = good; }
  if (a.bias.small) pool = [...pool].sort((x, y) => x.params_m - y.params_m).slice(0, 3);
  if (a.bias.donor && a.lastDonor) { const alt = pool.filter((p) => p.donor !== a.lastDonor); if (alt.length) pool = alt; }
  const prog = { host: HOST, seed: 0, masks: null, task_vectors: [], grafts: [], heal: null, lora: null };
  if (run.seed && run.tickN < 3 && a.id === 'planner-1') return clone(run.seed);
  a.refined = false;
  if (DEMO && run.best && run.best.program && Math.random() < 0.4) {
    const q = clone(run.best.program);
    const before = JSON.stringify(q);
    const m = Math.floor(Math.random() * 4);
    if (m === 0 && q.grafts.length) { const g = pick(q.grafts); g.at = clamp(g.at + pick([-1, 1]), 1, HOST_LAYERS - 2); }
    else if (m === 1) q.heal = { steps: pick([300, 450, 600, 900]), data: healPool(spec), lr: 0.0003, batch_size: 8, seq_len: 384 };
    else if (m === 2 && q.grafts.length) pick(q.grafts).glue_rank = pick([8, 16, 32, 64]);
    // the compiler allows a host LoRA only on a program with no parts
    else if (!q.grafts.length) q.lora = q.lora ? null : { rank: pick([8, 16]) };
    if (JSON.stringify(q) === before) q.heal = { steps: 600 + 50 * Math.floor(Math.random() * 6), data: healPool(spec), lr: 0.0003, batch_size: 8, seq_len: 384 };
    a.refined = true;
    return q;
  }
  const n = a.style === 'blocks' ? (Math.random() < 0.25 ? 2 : 1) : 1;
  for (let i = 0; i < n; i++) {
    let part = a.style === 'small' ? pool.slice().sort((x, y) => x.params_m - y.params_m)[Math.floor(Math.random() * Math.min(3, pool.length))] : pick(pool);
    if (a.style === 'blocks') part = pick(pool.filter((p) => p.kind === 'block' || p.kind === 'inserted_layer').length ? pool.filter((p) => p.kind === 'block' || p.kind === 'inserted_layer') : pool);
    if (prog.grafts.some((g) => g.id === part.id)) continue;
    const at = clamp(part.default_at + pick([-3, -2, -1, -1, 0, 0, 1, 2, 4]), 1, HOST_LAYERS - 2);
    prog.grafts.push({ id: part.id, at, control: null, control_seed: 0, glue_rank: pick([null, 8, 16, 16, 32, 64]) });
    a.lastDonor = part.donor;
  }
  while (prog.grafts.length > 1 && paramsOf(prog) > spec.size.max_params_b) { prog.grafts.pop(); run.refused = (run.refused || 0) + 1; }
  if (prog.grafts.length && paramsOf(prog) > spec.size.max_params_b) { prog.grafts = []; run.refused = (run.refused || 0) + 1; }
  if (a.style === 'masks' && Math.random() < 0.7) {
    const m = pick(MASKS);
    prog.masks = { op: 'use', id: m.id };
  }
  if (Math.random() < 0.85) prog.heal = { steps: pick([100, 300, 300, 600]), data: healPool(spec), lr: 0.0003, batch_size: 8, seq_len: 384 };
  // the compiler allows a host LoRA only on a program with no parts
  if (!prog.grafts.length && Math.random() < 0.2) prog.lora = { rank: pick([8, 16, 32]) };
  return prog;
}
// a program this run has already scored is not scored again: the next try changes the heal or the layer
function freshProgram(run, a) {
  run.seen = run.seen || new Set();
  let q = genProgram(run, a);
  for (let k = 0; k < 8 && run.seen.has(progHash(q)); k++) q = genProgram(run, a);
  for (let k = 0; k < 12 && run.seen.has(progHash(q)); k++) {
    if (q.grafts.length) q.grafts[0].at = clamp(q.grafts[0].at + pick([-2, -1, 1, 2]), 1, HOST_LAYERS - 2);
    q.heal = { steps: pick([100, 150, 300, 450, 600, 750, 900]), data: healPool(run.spec), lr: 0.0003, batch_size: 8, seq_len: 384 };
  }
  run.seen.add(progHash(q));
  return q;
}

function stepAgent(run, a) {
  if (run.status !== 'running') return;
  a.ticks++;
  const ph = (DEMO ? DEMO_PHASES : PHASES)[a.phase];
  const pname = ph[0];
  const plen = (DEMO ? Math.max(1, Math.round(ph[1] * stretch(run))) : ph[1]) + (a.cycles === 0 && a.phase === 0 ? a.skew : 0);
  const spec = run.spec;
  if (a.ticks === 1) {
    a.status = pname;
    if (pname === 'thinking') {
      a.cur = freshProgram(run, a);
      const pend = a.nudges.find((n) => !n.applied);
      if (pend) {
        pend.applied = true; a.nudges_pending = Math.max(0, a.nudges_pending - 1);
        const low = pend.text.toLowerCase();
        if (/small|tiny|fewer param/.test(low)) a.bias.small = true;
        if (/donor|different/.test(low)) a.bias.donor = true;
        think(a, DEMO ? V.nudgeThought(pend.text) : 'Nudge from you: "' + pend.text + '". I will use it for this attempt.');
        if (/explain|plan/.test(low)) think(a, 'Plan: probe fit for two placements near the default layer, compile the better one, heal 300 steps, then score on the search split.');
        logAct(run, a.id, 'nudge_applied', 'ok', 'Read your nudge and will use it next round', { nudge: pend.text });
      }
      const g = a.cur.grafts[0];
      const part = g && PART[g.id];
      const tm = (sc) => tmeanOf(spec, sc);
      const best = run.best ? ' Best so far is ' + (tm(run.best.scores) * 100).toFixed(1) + '%.' : '';
      if (DEMO) {
        const tg = spec.benchmarks.find((b) => b.kind === 'target');
        const gd = spec.benchmarks.find((b) => b.kind === 'guard');
        const altAt = part ? clamp(g.at + (g.at > part.default_at ? -2 : 2), 1, HOST_LAYERS - 2) : null;
        const lastH = a.history[a.history.length - 1];
        const mk = a.cur.masks && a.cur.masks.op === 'use' ? MASKS.find((m) => m.id === a.cur.masks.id) : null;
        const hs = a.cur.heal || { steps: 300, data: healPool(spec) };
        const lv = part && run.vsByPart ? run.vsByPart[part.id] : null;
        think(a, V.planThought({ style: a.style, part, at: g && g.at, rank: (g && g.glue_rank) || 16, steps: hs.steps, data: hs.data, best: run.best ? tm(run.best.scores) : null, host: targetsOf(spec).reduce((x, b) => x + b.base, 0) / targetsOf(spec).length, bench: benchLabel(spec), cross: a.cross,
          vs: lv != null && lv > 0.01 ? (lv * 100).toFixed(1) : null, fitHere: part ? fitOf(part, g.at) : 0, fitAlt: part ? fitOf(part, altAt) : null, altAt,
          lastName: lastH && lastH.name, lastScore: lastH && tm(lastH.scores), mask: mk && mk.label.toLowerCase(), floor: gd ? gd.floor : 0.45 }));
      } else
      think(a, part ? 'Try ' + part.label + ' at layer ' + g.at + ' with glue rank ' + (g.glue_rank || 'full') + '.' + best + ' The fit probe is free, so I check it before I spend an eval.' : 'Try a host-only program with heal, as a floor.' + best);
      a.now = 'Planning ' + readable(a.cur);
      if (a.id === 'planner-1' || Math.random() < 0.5) logAct(run, a.id, 'plan', 'info', DEMO ? V.planLine({ count: 3, refine: !!a.refined, round: run.round, parts: 1 + (a.cur.grafts.length > 1 ? 1 : 0) + Math.floor(Math.random() * 3) }) : 'Proposed 3 programs to try this round', { count: 3 });
    } else if (pname === 'probing') {
      const g = a.cur.grafts[0];
      a.now = g ? 'Probing fit at L' + g.at + ' (free, no eval used)' : 'Checking the host (free)';
      if (g) {
        const f = fitOf(PART[g.id], g.at);
        tool(a, 'probe(' + g.id + '@' + g.at + ')', 'fit ' + f.toFixed(2) + ', free');
        logAct(run, a.id, 'probe', 'ok', DEMO ? V.probeLine({ name: readable(a.cur), fit: f, at: g.at }) : 'Checked fit of ' + readable(a.cur) + ': ' + f.toFixed(2), { name: readable(a.cur), fits: [Math.round(f * 100) / 100], program_hash: progHash(a.cur) });
      }
    } else if (pname === 'compiling') {
      a.now = 'Compiling ' + readable(a.cur);
      tool(a, 'compile(' + describe(a.cur) + ')', (30 + Math.floor(Math.random() * 40)) + ' s, ' + paramsOf(a.cur).toFixed(2) + 'B params' + (DEMO ? ', ' + (4.5 + paramsOf(a.cur) * 2.1).toFixed(1) + ' GB peak' : ''));
      if (a.cur.heal) tool(a, 'heal(' + a.cur.heal.steps + ' steps)', 'loss ' + (2.0 + (hash(spec.id + a.cur.heal.data + a.cur.heal.steps) % 30) / 100).toFixed(2) + ' to ' + (1.2 + Math.random() * 0.3).toFixed(2));
    } else {
      a.now = 'Scoring ' + readable(a.cur);
      tool(a, 'eval(search split)', 'running', 'info');
    }
  }
  if (a.ticks >= plen) {
    a.ticks = 0;
    a.phase++;
    if (a.phase >= PHASES.length) { a.phase = 0; a.cycles++; finishEval(run, a); }
  }
}

function finishEval(run, a) {
  if (run.evals_used >= run.evals_total) return;
  const spec = run.spec;
  const ev = evalProgram(spec, a.cur);
  const held = run.hold && run.evals_used >= run.evals_total - 1;
  if (!held) { run.evals_used++; run.compute_used += 35 + Math.floor(Math.random() * 25); }
  a.attempts++;
  const tgt = spec.benchmarks.find((b) => b.kind === 'target');
  const single = targetsOf(spec).length === 1;
  if (DEMO && a.refined && ev.status === 'ok' && single) {
    run.refines = (run.refines || 0) + 1;
    ev.scores[tgt.id] = Math.round(clamp(ev.scores[tgt.id] + Math.min(0.03, run.refines * 0.0007) * (0.6 + 0.8 * Math.random()), 0.01, 0.98) * 1000) / 1000;
  }
  if (DEMO && ev.status === 'ok' && single) {
    const cap = run.cap != null ? run.cap : tgt.base + ((tgt.target || tgt.base + 0.1) - tgt.base) * (0.9 + 0.35 * run.evals_used / run.evals_total);
    ev.scores[tgt.id] = Math.round(Math.min(ev.scores[tgt.id], cap - Math.random() * 0.006) * 1000) / 1000;
  }
  const score = tmeanOf(spec, ev.scores);
  const last = a.tools[a.tools.length - 1];
  if (last && last.result === 'running') { last.result = 'done: ' + (score * 100).toFixed(1) + '%'; last.status = ev.status === 'ok' ? 'ok' : 'bad'; }
  const h = progHash(a.cur);
  (run.seen = run.seen || new Set()).add(h);
  a.history.push({ program_hash: h, describe: describe(a.cur), name: readable(a.cur), scores: ev.scores, combined: ev.combined, status: ev.status });
  if (a.history.length > 30) a.history.shift();
  run.scored = run.scored || [];
  if (ev.status === 'ok') run.scored.push(score);
  if (ev.status === 'collapsed') { a.collapses++; a.lessons++; run.collapsed = (run.collapsed || 0) + 1; }
  if (ev.vs_control != null) { run.vsByPart = run.vsByPart || {}; for (const g of a.cur.grafts) run.vsByPart[g.id] = ev.vs_control; }
  if (a.best == null || (ev.status === 'ok' && score > a.best)) a.best = ev.status === 'ok' ? score : a.best;
  for (const g of a.cur.grafts) { const d = PART[g.id] && PART[g.id].donor; if (d) run.uses[d] = (run.uses[d] || 0) + 1; }
  run.uses[HOST] = (run.uses[HOST] || 0) + 1;
  const syn = withSynth(spec, a.cur, {});
  ev.components = syn.components;
  const prevBest = run.best ? tmeanOf(spec, run.best.scores) : null;
  const entry = { pipeline: syn.pipeline, components: syn.components, drift: syn.drift, rank: 0, program_hash: h, describe: describe(a.cur), name: readable(a.cur), agent: a.id, scores: ev.scores, combined: ev.combined, status: ev.status, vs_control: null, params_b: ev.params_b, program: clone(a.cur), _pending: ev.vs_control, _age: 0, _ctl: ev.control };
  run.board.push(entry);
  const curBest = run.curve.length ? run.curve[run.curve.length - 1].best : null;
  const isBest = ev.status === 'ok' && ev.params_b <= spec.size.max_params_b && (!run.best || score > prevBest);
  run.curve.push({ eval: run.evals_used, best: isBest ? score : curBest, score, program_hash: h, name: entry.name, agent: a.id, status: ev.status, scores: { ...ev.scores }, is_best: isBest });
  const guardsOut = spec.benchmarks.filter((b) => b.kind === 'guard').map((g) => ({ id: g.id, name: g.name, score: ev.scores[g.id], floor: g.floor, ok: ev.scores[g.id] >= g.floor }));
  if (isBest) run.best = { program_hash: h, describe: entry.describe, name: entry.name, scores: ev.scores, combined: ev.combined, agent: a.id, program: clone(a.cur) };
  const data = { name: entry.name, program_hash: h, target: Math.round(score * 10000) / 10000, best: run.best ? tmeanOf(spec, run.best.scores) : null, is_best: isBest, guards: guardsOut, scores: ev.scores, result: ev.status };
  const failG = guardsOut.find((x) => !x.ok) || guardsOut[0] || { name: 'a guard', score: 0, floor: 0 };
  const dctx = { name: entry.name, score, target: score, leader: prevBest != null ? prevBest : score, prev: prevBest, best: prevBest, isBest, collapsed: ev.status === 'collapsed', guard: failG.name, guardScore: failG.score, floor: failG.floor, at: a.cur.grafts[0] ? a.cur.grafts[0].at : 6, healSteps: a.cur.heal ? a.cur.heal.steps : 300, bench: benchLabel(spec) };
  if (DEMO) {
    a.lastVs = ev.vs_control != null ? ev.vs_control : null;
    if (a.cur.grafts.length && prevBest != null && Math.random() < 0.7) think(a, V.evalThought(dctx));
    logAct(run, a.id, 'eval', ev.status === 'collapsed' ? 'bad' : isBest ? 'ok' : 'info', V.evalLine(dctx), data);
  } else if (isBest) logAct(run, a.id, 'eval', 'ok', 'Scored ' + entry.name + ' at ' + (score * 100).toFixed(1) + '% on ' + benchLabel(spec) + '. It is the best so far.', data);
  else if (ev.status === 'collapsed') {
    logAct(run, a.id, 'eval', 'bad', 'Scored ' + entry.name + ' and it collapsed. The lesson is saved.', data);
    think(a, 'That collapsed. Guards fell below their floor, so I will move the graft away from its neighbour and try again.');
  } else logAct(run, a.id, 'eval', 'info', 'Scored ' + entry.name + ' at ' + (score * 100).toFixed(1) + '% on ' + benchLabel(spec) + '.', data);
  const tier = (e) => (e.status === 'ok' ? 0 : e.status === 'collapsed' ? 2 : 1);
  run.board.sort((x, y) => tier(x) - tier(y) || tmeanOf(spec, y.scores) - tmeanOf(spec, x.scores));
  run.board = run.board.slice(0, 14);
  run.board.forEach((e, i) => { e.rank = i + 1; });
}

function analystStep(run, an) {
  const ready = run.board.length;
    const spec = run.spec;
    const tm = (e) => tmeanOf(spec, e.scores);
    const ok = run.board.filter((e) => e.status === 'ok');
    const best = run.best ? tmeanOf(spec, run.best.scores) : 0.6;
    const baseM = tmeanOf(spec, run.base_scores);
    const thr = Math.max(baseM + 0.02, best - 0.04);
    const allScored = run.scored || [];
    const good = allScored.filter((x) => x >= thr).length;
    const g0 = ok.filter((e) => tm(e) >= thr).map((e) => e.program.grafts[0]).filter(Boolean);
    const topPart = g0.length ? PART[g0[0].id] : null;
    const avgAt = g0.length ? Math.round(g0.reduce((x, g) => x + g.at, 0) / g0.length) : 6;
    const seen = run.mapKeys = run.mapKeys || new Set();
    const meanOf = (l) => (l.length ? l.reduce((x, e) => x + tm(e), 0) / l.length : null);
    const hs = (e) => (e.program.heal ? e.program.heal.steps : 0);
    const long = ok.filter((e) => hs(e) >= 600), shortH = ok.filter((e) => hs(e) > 0 && hs(e) <= 300);
    const pairs = ok.filter((e) => e.program.grafts.length >= 2), singles = ok.filter((e) => e.program.grafts.length === 1);
    const base = { v: Math.ceil((run.tickN || 24) / 24), donor: topPart ? topPart.donor.split('/').pop() : 'donor', at: avgAt, good, total: allScored.length, thr, bad: run.collapsed || 0, partName: topPart ? topPart.label : 'The part', def: topPart ? topPart.default_at : 6, loss: (2 + Math.random() * 2).toFixed(1), bench: benchLabel(spec) };
    const cands = [];
    if (topPart) cands.push({ key: 'place:' + topPart.id, o: { ...base, kind: 'place' } });
    if (long.length >= 2 && shortH.length >= 2 && meanOf(long) >= meanOf(shortH) + 0.01) cands.push({ key: 'heal', o: { ...base, kind: 'heal', heal600: meanOf(long), heal100: meanOf(shortH) } });
    if (pairs.length >= 2 && singles.length >= 2 && meanOf(pairs) >= meanOf(singles) + 0.01) cands.push({ key: 'pair', o: { ...base, kind: 'pair', pair: meanOf(pairs), single: meanOf(singles), pairNames: [...pairs].sort((x, y) => tm(y) - tm(x))[0].name } });
    if (!seen.has('map:' + base.v) && allScored.length >= 4) cands.push({ key: 'map:' + base.v, o: { ...base, kind: 'map' } });
    const pickC = cands.find((c) => !seen.has(c.key));
    if (pickC) {
      seen.add(pickC.key);
      an.status = 'thinking';
      const txt = V.mapLine(pickC.o);
      an.now = txt; an.lessons++;
      think(an, txt);
      tool(an, 'read_results(' + allScored.length + ' programs)', 'ok');
      logAct(run, 'analyst', 'map', 'info', txt, { detail: 'program | score | fit\n' + run.board.slice(0, 4).map((e) => e.name + ' | ' + (tm(e) * 100).toFixed(1) + '% | ' + (e.status === 'ok' ? 'yes' : 'no')).join('\n') });
    }
}

function tickRun(run) {
  if (run.lazy) return;
  if (run.status === 'paused' || run.status === 'done' || run.status === 'error') return;
  if (run.status === 'stopping') {
    run.status = 'done';
    run.outcome = 'stopped';
    run.agents.forEach((a) => { a.status = 'done'; a.now = 'Stopped'; });
    logAct(run, 'judge', 'milestone', 'info', 'Search stopped. The best program is kept.');
    return;
  }
  run.tickN++;
  run.elapsed += DEMO ? 1 : 14;
  run.round = DEMO ? Math.floor(run.evals_used / 3) + 1 : Math.floor((run.tickN - 1) / 4) + 1;
  const ready = run.board.length;
  run.agents.forEach((a) => {
    if (a.id.startsWith('planner')) stepAgent(run, a);
  });
  // analyst and judge
  const an = run.agents[0], ju = run.agents[4];
  if (DEMO && ready >= 2 && run.tickN % 24 === 0) {
    analystStep(run, an);
  } else if (DEMO) { if (run.tickN % 24 === 6) an.status = 'idle'; }
  else if (ready >= 2 && run.tickN % 5 === 0) {
    an.status = 'thinking';
    const ok = run.board.filter((e) => e.status === 'ok');
    const avgAt = ok.length ? ok.map((e) => e.program.grafts[0] ? e.program.grafts[0].at : 6).reduce((x, y) => x + y, 0) / ok.length : 6;
    const txt = 'Map v' + Math.ceil(run.tickN / 5) + ': grafts near L' + Math.round(avgAt) + ' win. ' + (() => { const n = run.board.filter((e) => e.status !== 'ok').length; return n + (n === 1 ? ' collapse' : ' collapses') + ' so far, mostly stacked grafts.'; })();
    an.now = txt; an.lessons++;
    think(an, txt);
    tool(an, 'read_results(' + ready + ' programs)', 'ok');
    logAct(run, 'analyst', 'map', 'info', txt, { detail: 'program | score\n' + run.board.slice(0, 4).map((e) => e.name + ' | ' + (e.combined * 100).toFixed(0)).join('\n') });
  } else if (run.tickN % 5 === 2) an.status = 'idle';
  const top = run.board.slice(0, 3).find((e) => e.vs_control == null && e._pending != null);
  if (top) {
    top._age++;
    ju.status = 'scoring';
    ju.now = 'Running the random-part control for ' + top.name;
    if (top._age >= (DEMO ? 6 : 2)) {
      top.vs_control = top._pending; top.control = top._ctl || null;
      ju.status = 'idle'; ju.attempts++;
      ju.now = 'Control done for rank ' + top.rank + ': ' + (top.vs_control * 100).toFixed(1) + ' points above a random part';
      const tg0 = run.spec.benchmarks.find((b) => b.kind === 'target');
      logAct(run, 'judge', 'verdict', top.vs_control > 0.02 ? 'ok' : 'info', DEMO ? V.controlLine({ name: top.name, score: top.scores[tg0.id], control: top.scores[tg0.id] - top.vs_control, pts: (top.vs_control * 100).toFixed(1) }) : 'The random-part control for ' + top.name + ' says the part adds ' + (top.vs_control * 100).toFixed(1) + ' points.', { frontier: run.board.slice(0, 3).map((e) => ({ program_hash: e.program_hash, name: e.name, target: e.scores[run.spec.benchmarks.find((b) => b.kind === 'target').id] })), retired: [], killed: [] });
      think(ju, ju.now);
      tool(ju, 'control(random part)', '+' + (top.vs_control * 100).toFixed(1));
    }
  } else if (ju.status === 'scoring') ju.status = 'idle';
  if (!run.hold && (run.evals_used >= run.evals_total || run.compute_used >= run.compute_total)) {
    run.status = 'done';
    run.outcome = 'finished';
    run.agents.forEach((a) => { a.status = 'done'; a.now = 'Finished'; });
    const tg = run.spec.benchmarks.find((b) => b.kind === 'target');
    // the judge finishes the controls for the top programs before the search ends
    run.board.slice(0, 3).forEach((e) => {
      if (e.vs_control != null || e._pending == null) return;
      e.vs_control = e._pending; e.control = e._ctl || null; ju.attempts++;
      logAct(run, 'judge', 'verdict', e.vs_control > 0.02 ? 'ok' : 'info', DEMO ? V.controlLine({ name: e.name, score: e.scores[tg.id], control: e.scores[tg.id] - e.vs_control, pts: (e.vs_control * 100).toFixed(1) }) : 'The random-part control for ' + e.name + ' says the part adds ' + (e.vs_control * 100).toFixed(1) + ' points.');
    });
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
    base_scores: run.tickN < 1 ? {} : run.base_scores, best: run.best, outcome: run.outcome || null, stopped_early: run.outcome === 'stopped', empty_reason: run.empty_reason || null, sort: 'target score',
    agents: run.agents.map((a) => ({ id: a.id, name: a.name, role: a.role, blurb: a.blurb, status: run.status === 'paused' ? 'paused' : a.status, now: a.now, best: a.best, attempts: a.attempts, collapses: a.collapses, lessons: a.lessons, nudges_pending: a.nudges_pending })),
    refused: run.refused || 0, limits: spec.limits, comparison: null,
    leaderboard: run.board.map((e) => {
      const c = { ...e };
      delete c._pending; delete c._age; delete c._ctl; delete c.control;
      const t0 = spec.benchmarks.find((b) => b.kind === 'target');
      const control = e.control ? e.control : e.vs_control != null ? { scores: { [t0.id]: e.scores[t0.id] - e.vs_control } } : null;
      c.comparison = mockComparison(spec, { components: e.components, scores: e.scores, status: e.status, params_b: e.params_b, control }, e.program, run.status === 'running' || run.status === 'paused');
      return c;
    }),
    activity: run.activity, curve: run.curve, sources: srcs,
  };
  if (run.demoMargin) s.leaderboard.forEach((e) => { const r = e.comparison && (e.comparison.rows || []).find((x) => x.id === 'lora'); if (r && r.measured) { r.beaten = false; r.within_margin = true; r.delta = 0.011; } });
  if (s.best) { const e = s.leaderboard.find((x) => x.program_hash === s.best.program_hash); s.comparison = e ? e.comparison : null; }
  return clone(s);
}
// Demo runs for the odd cases: open #/run/demo-empty, demo-stopped, demo-margin or demo-stream.
function demoRun(id) {
  const r = newRun(getSpec('sp_demo'));
  runs.delete(r.id); r.id = id; runs.set(id, r);
  const spin = (n) => { for (let i = 0; i < n; i++) tickRun(r); };
  if (id.startsWith('demo-empty')) {
    r.status = 'done'; r.outcome = 'finished'; r.elapsed = 38; r.empty_reason = 'Every proposal went over the size limit, so nothing was scored.';
    r.refused = 6;
    r.agents.forEach((a) => { a.status = 'done'; a.now = 'Finished'; });
    logAct(r, 'judge', 'milestone', 'info', 'Search finished. Nothing was scored.');
  } else if (id.startsWith('demo-stopped')) {
    spin(16);
    r.status = 'stopping';
    tickRun(r);
    r.elapsed = 12;
  } else if (id.startsWith('demo-margin')) {
    r.demoMargin = true;
    spin(22);
  } else spin(3);
  return r;
}
function getRunObj(id) {
  if (!runs.has(id)) {
    if (/^demo-/.test(id)) return demoRun(id);
    const e = new Error('No run with this id is in this browser. It may have been started in another tab or before a reload.'); e.status = 404; throw e;
  }
  const run = runs.get(id);
  if (run.lazy) { run.lazy = false; run.wall0 = Date.now(); }
  return run;
}

/* ---------- builders and jobs ---------- */
const builders = new Map();
const jobs = new Map();
function getBuilder(id, specId) {
  if (!builders.has(id)) {
    const sid = specId || (store.get('wsp.bspec.' + id) || {}).spec_id;
    if (!sid) { const e = new Error('This build session is not in this browser. Open the spec and start a new build.'); e.status = 404; throw e; }
    getSpec(sid);
    store.set('wsp.bspec.' + id, { spec_id: sid });
    builders.set(id, { id, spec_id: sid, budget: { evals_used: 0, evals_total: 12, compute_used_s: 0, compute_total_s: 2700 } });
  }
  return builders.get(id);
}

/* ---------- shrink (skill removal) ---------- */
const SHRINK_WORDS = /\bshrink|\bslim\b|smaller version of|drop everything|keep only|coding-only|remove\b.{0,40}\bskills?|distill\w*\s+(it\s+)?down|\bprune\b/i;
const SMODELS = {
  'Qwen/Qwen2.5-Coder-7B-Instruct': { params: 7.616, active: 7.071, kind: 'dense', layers: 28, floor: 1.1 },
  'Qwen/Qwen2.5-Coder-14B-Instruct': { params: 14.77, active: 13.99, kind: 'dense', layers: 48, floor: 1.6 },
  'Qwen/Qwen3-30B-A3B-Instruct-2507': { params: 30.53, active: 3.029, kind: 'moe', layers: 48, experts: 128, top_k: 8, base: 1.2 },
  'allenai/OLMoE-1B-7B-0125-Instruct': { params: 6.92, active: 1.28, kind: 'moe', layers: 16, experts: 64, top_k: 8, base: 0.7 },
};
const SKILL_BENCH = { code: 'humaneval', math: 'gsm8k', medicine: 'mmlupro:health', general: 'mmlu', prose: 'mmlu' };
const RULE_F = { random: 0.3, general: 0.4, freq: 0.55, selective: 0.5 };
const DENSE_RULES = ['skill', 'specific', 'general', 'random'];
const EXPERT_RULES = ['freq', 'saliency', 'selective', 'random', 'general'];
const bAdd = (b, n) => Math.round(b * n) / n;
function candidatesFor(src, skill) {
  return Object.keys(SMODELS).filter((m) => m !== src).map((m) => ({ model: m, params_b: SMODELS[m].params, why: 'tagged for ' + skill + (SMODELS[m].kind === 'moe' ? ', MoE with ' + SMODELS[m].experts + ' experts per layer' : ', dense') }));
}
function moeTotal(f, keep) { return f.base + (f.params - f.base) * keep / f.experts; }
function newBlock(model, skill, target, rule, steps) {
  const f = SMODELS[model];
  const b = { source_model: model, source_params_b: f.params, source_active_params_b: f.active, family: f.kind, keep_skill: skill, method: f.kind === 'moe' ? 'experts' : 'dense', experts_total: f.kind === 'moe' ? f.experts : null, target_params_b: null, keep_experts: null, rule: rule || (f.kind === 'moe' ? 'saliency' : 'skill'), heal_steps: steps ?? 300, candidates: candidatesFor(model, skill), notes: [] };
  if (f.kind === 'moe') {
    let keep = Math.max(1, Math.floor(f.experts / 4));
    if (target) keep = clamp(Math.floor((target - f.base) / (f.params - f.base) * f.experts), 1, f.experts);
    b.keep_experts = keep; b.target_params_b = bAdd(moeTotal(f, keep), 1000);
  } else b.target_params_b = target && target < f.params ? target : bAdd(f.params * 0.38, 100);
  return b;
}
function bad(msg) { const e = new Error(msg); e.status = 400; return e; }
function checkBlock(inc, old) {
  const source = inc.source_model ?? old.source_model;
  if (!SMODELS[source]) throw bad('the source model must be a model in the library: ' + source);
  const f = SMODELS[source];
  const skill = inc.keep_skill ?? old.keep_skill;
  if (!SKILL_BENCH[skill]) throw bad('keep_skill must be one of code, general, math, medicine, prose');
  const steps = Math.round(Number(inc.heal_steps ?? old.heal_steps));
  if (!(steps >= 0 && steps <= 1000)) throw bad('heal_steps must be from 0 to 1000');
  const method = f.kind === 'moe' ? 'experts' : 'dense';
  const rules = method === 'experts' ? EXPERT_RULES : DENSE_RULES;
  const changed = source !== old.source_model;
  const rule = inc.rule ?? (!changed && rules.includes(old.rule) ? old.rule : rules[method === 'experts' ? 1 : 0]);
  if (!rules.includes(rule)) throw bad('rule must be one of ' + rules.join(', ') + ' for ' + method + ' removal');
  if (changed) return newBlock(source, skill, inc.target_params_b, rule, inc.heal_steps ?? old.heal_steps);
  const b = { ...old, keep_skill: skill, heal_steps: steps, rule, method };
  if (method === 'experts') {
    if (inc.keep_experts != null) b.keep_experts = Math.round(inc.keep_experts);
    else if (inc.target_params_b != null && inc.target_params_b !== old.target_params_b) b.keep_experts = clamp(Math.floor((inc.target_params_b - f.base) / (f.params - f.base) * f.experts), 1, f.experts);
    if (b.keep_experts < 1 || b.keep_experts > f.experts) throw bad('keep_experts must be from 1 to ' + f.experts + ' for ' + source);
    b.target_params_b = bAdd(moeTotal(f, b.keep_experts), 1000);
  } else {
    if (inc.target_params_b != null) b.target_params_b = Number(inc.target_params_b);
    if (b.target_params_b <= f.floor) throw bad('target_params_b must be above ' + f.floor.toFixed(2) + 'B: the embedding and the head alone take that much and slimming does not touch them');
    if (b.target_params_b >= f.params) throw bad('target_params_b must be below the source size of ' + f.params + 'B');
  }
  if (inc.skill_changed) b.candidates = candidatesFor(source, skill);
  return b;
}
function shrinkEstimate(b) {
  const f = SMODELS[b.source_model];
  const total = f.kind === 'moe' ? moeTotal(f, b.keep_experts) : b.target_params_b;
  const topk = f.kind === 'moe' ? Math.min(f.top_k, b.keep_experts) : 0;
  const run = f.kind === 'moe' ? Math.min(total, bAdd(f.base + (f.active - f.base) * topk / f.top_k, 1000)) : bAdd(total * (f.active / f.params), 1000);
  const est = { method: b.method, total_params_b: bAdd(total, 1000), run_params_b: run, heal_min: Math.round(b.heal_steps * 0.3 * Math.max(0.5, run) / 60 * 100) / 100, source_params_b: f.params, source_run_params_b: f.active, removed_params_b: bAdd(f.params - total, 1000), layers: f.layers, estimated: true };
  if (f.kind === 'moe') Object.assign(est, { experts_kept: b.keep_experts, experts_before: f.experts, top_k_after: topk });
  else { const keep = clamp((total - f.floor) / (f.params - f.floor), 0.05, 1); Object.assign(est, { layers_before: f.layers, layers_dropped: Math.round(f.layers * (1 - Math.sqrt(keep)) * 0.5), keep_width: Math.round(keep * 100) / 100 }); }
  return est;
}
const fb = (x) => (Math.round(x * 1000) / 1000 + '').replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
function shrinkProblems(lim, e) {
  const out = [];
  if (e.total_params_b > lim.max_total_params_b) out.push('Over the total size limit. The shrunk model is estimated at ' + fb(e.total_params_b) + 'B parameters (the source has ' + fb(e.source_params_b) + 'B). The limit is ' + fb(lim.max_total_params_b) + 'B.');
  if (e.run_params_b > lim.max_run_params_b) out.push('Over the per-token limit. The shrunk model runs an estimated ' + fb(e.run_params_b) + 'B parameters per token. The limit is ' + fb(lim.max_run_params_b) + 'B.');
  if (e.heal_min > lim.max_heal_min) out.push('Over the heal time limit. The heal is estimated at ' + e.heal_min.toFixed(1) + ' min. The limit is ' + lim.max_heal_min + ' min.');
  return out;
}
function shrinkSummary(b, e) {
  const label = b.keep_skill;
  if (b.method === 'experts') return 'Keep ' + e.experts_kept + ' of ' + e.experts_before + ' experts per layer by ' + b.rule + ' on ' + label + ' text. About ' + e.total_params_b.toFixed(1) + 'B stored, ' + e.run_params_b.toFixed(1) + 'B run per token (estimated).';
  return 'Slim the model to about ' + e.total_params_b.toFixed(1) + 'B by removing ' + e.layers_dropped + ' layers and trimming neurons and heads by ' + b.rule + ' on ' + label + ' text. About ' + e.run_params_b.toFixed(1) + 'B run per token (estimated).';
}
// the original model's score on a task: one table for the spec, the plan and the result
const SRC_TABLE = { 'Qwen/Qwen3-30B-A3B-Instruct-2507': { humaneval: 0.915, mbpp: 0.844, gsm8k: 0.915, mmlu: 0.778 } };
const srcScore = (model, t) => (SRC_TABLE[model] && SRC_TABLE[model][t]) || Math.round(clamp(0.84 - noise(t + model) * 0.06, 0.05, 0.95) * 1000) / 1000;
function makeShrinkSpec(prompt, id) {
  const low = (prompt || '').toLowerCase();
  const skill = /math|gsm/.test(low) ? 'math' : /medic|health|bio/.test(low) ? 'medicine' : /code|coding|python|program/.test(low) ? 'code' : 'general';
  const model = /30b|moe|a3b/.test(low) ? 'Qwen/Qwen3-30B-A3B-Instruct-2507' : /14b/.test(low) ? 'Qwen/Qwen2.5-Coder-14B-Instruct' : 'Qwen/Qwen2.5-Coder-7B-Instruct';
  const m = /under\s+(\d+(?:\.\d+)?)\s*b/.exec(low);
  const target = m ? Number(m[1]) : null;
  const block = newBlock(model, skill, target ? target * 0.97 : null);
  const f = SMODELS[model];
  const est = shrinkEstimate(block);
  const total = target || Math.ceil(est.total_params_b * 10) / 10;
  const run = target ? Math.max(target, 0.1) : Math.ceil(est.run_params_b * 10) / 10;
  const bench = clone(BENCH_LIB[SKILL_BENCH[skill]] || { id: SKILL_BENCH[skill], name: skill, kind: 'target', base: 0.4 });
  bench.kind = 'target'; bench.base = srcScore(model, bench.id); bench.target = Math.round(bench.base * 0.95 * 100) / 100; delete bench.lora; delete bench.recommended;
  return {
    id: id || rid('sp_'), title: 'A ' + skill + '-only model', summary: 'Keeps ' + skill + ' and removes the rest from ' + model.split('/').pop() + '. ' + shrinkSummary(block, est), prompt: prompt || '', files: [], task_type: skill === 'code' || skill === 'math' ? skill : 'general',
    host: model, size: { max_params_b: total, min_b: 0.05, max_b: Math.max(32, f.params), host_params_b: f.params },
    budget: { evals: 30, compute_min: 45, min_evals: 5, max_evals: 120 }, limits: { max_total_params_b: total, max_run_params_b: run, max_heal_min: 20 }, comparators: { distill: {} },
    benchmarks: [bench], available_benchmarks: [], ingredients: [], drafted_by: 'rules', request_type: 'shrink', shrink: block, stub: true,
  };
}
const shrinks = new Map();
const shrinkJobs = new Map();
function shrinkSess(id) {
  if (!shrinks.has(id)) { const st = store.get('wsp.shrink.' + id); if (st) { getSpec(st.spec_id); shrinks.set(id, { id, spec_id: st.spec_id, plan: st.plan }); } }
  return shrinks.get(id);
}
function shrinkJob(jid) {
  if (!shrinkJobs.has(jid)) { const st = store.get('wsp.sjob.' + jid); if (st) { const sess = shrinkSess(st.sid) || { id: st.sid, spec_id: st.spec_id, plan: st.plan }; shrinkJobs.set(jid, { sess, plan: st.plan, t0: st.t0, done: st.done }); } }
  return shrinkJobs.get(jid);
}
function shrinkView(sess) {
  const spec = getSpec(sess.spec_id);
  const est = shrinkEstimate(sess.plan);
  const problems = shrinkProblems(spec.limits, est);
  return { ...clone(sess.plan), estimate: est, limits: clone(spec.limits), problems, ok: problems.length === 0, summary: shrinkSummary(sess.plan, est), tasks: spec.benchmarks.map((b) => b.id), stub: true };
}
function shrinkResult(sess, plan) {
  const spec = getSpec(sess.spec_id);
  const f = SMODELS[plan.source_model];
  const est = shrinkEstimate(plan);
  const frac = est.total_params_b / f.params;
  const tasks = spec.benchmarks.map((b) => b.id);
  const scores = {}, src = {}, rnd = {};
  const heal = Math.min(1, plan.heal_steps / 400) * 0.08;
  tasks.forEach((t) => {
    const top = srcScore(plan.source_model, t);
    src[t] = top;
    scores[t] = Math.round(clamp((top * (0.45 + 0.5 * Math.sqrt(frac)) + heal - 0.03) * (RULE_F[plan.rule] || 1), 0.05, top - 0.02) * 1000) / 1000;
    rnd[t] = Math.round(clamp(scores[t] * 0.35, 0.02, 0.9) * 1000) / 1000;
  });
  const t0 = tasks[0];
  let best = Math.round(clamp(scores[t0] - 0.05 + noise(plan.source_model) * 0.04, 0.05, 0.9) * 1000) / 1000;
  if (DEMO && plan.source_model === 'Qwen/Qwen3-30B-A3B-Instruct-2507') {
    const k = plan.keep_experts, h = Math.min(plan.heal_steps, 500) / 500;
    const lerp = (pts, x) => { if (x <= pts[0][0]) return pts[0][1]; for (let i = 1; i < pts.length; i++) if (x <= pts[i][0]) { const [x0, y0] = pts[i - 1], [x1, y1] = pts[i]; return y0 + (y1 - y0) * (x - x0) / (x1 - x0); } return pts[pts.length - 1][1]; };
    const T = {
      humaneval: { src: 0.915, raw: [[8, 0], [16, 0.037], [32, 0.579], [64, 0.878], [128, 0.915]], gain: [[16, 0.05], [32, 0.159], [64, 0.015], [128, 0]], rnd: 0.116, best: 0.854 },
      mbpp: { src: 0.844, raw: [[16, 0.078], [32, 0.553], [64, 0.79], [128, 0.844]], gain: [[16, 0.03], [32, 0.066], [64, 0.01], [128, 0]], rnd: 0.186, best: 0.78 },
    };
    const r3 = (x) => Math.round(x * 1000) / 1000;
    tasks.forEach((t) => {
      const m = T[t];
      if (!m) return;
      src[t] = m.src;
      scores[t] = r3(Math.min(m.src, (lerp(m.raw, k) + h * lerp(m.gain, k)) * (RULE_F[plan.rule] || 1)));
      rnd[t] = r3(plan.rule === 'random' ? scores[t] * 0.9 : Math.min(scores[t] * 0.6, m.rnd * (k / 32) * (0.3 + 0.7 * h)));
    });
    if (T[t0]) best = T[t0].best;
  }
  const row = (id, label, value, beaten, required, note) => ({ id, label, split: 'search', per_task: { [t0]: value }, value, measured: true, note, stub: true, result_value: scores[t0], delta: Math.round((value - scores[t0]) * 1000) / 1000, beaten, required });
  const rows = [
    { ...row('source', 'The original model', src[t0], null, false, 'The original model on the same items. The result is expected to be below it.'), delta: Math.round((scores[t0] - src[t0]) * 1000) / 1000 },
    { ...row('random', 'Random removal', rnd[t0], true, true, 'The same slim with random units removed, same heal steps.'), delta: Math.round((scores[t0] - rnd[t0]) * 1000) / 1000 },
    { ...row('best_model', 'Best existing model at this size', best, scores[t0] >= best, true, 'The best library model within the size window of the result.'), delta: Math.round((scores[t0] - best) * 1000) / 1000 },
  ];
  rows.forEach((r) => { if (r.required) { r.beaten = r.delta > 0.02; r.within_margin = !r.beaten && r.delta > 0; } });
  const beaten = rows.filter((r) => r.required && r.beaten).map((r) => r.id);
  const notBeaten = rows.filter((r) => r.required && !r.beaten).map((r) => r.id);
  const verdict = notBeaten.length ? 'failed' : 'success';
  const keepPct = Math.round(scores[t0] / src[t0] * 100);
  const removed = f.kind === 'moe'
    ? { kind: 'moe', experts_before: f.experts, experts_kept: plan.keep_experts, experts_removed_per_layer: f.experts - plan.keep_experts, layers: f.layers, top_k_after: Math.min(f.top_k, plan.keep_experts), kept_share_of_skill: Math.round(clamp(0.5 + 0.5 * Math.sqrt(plan.keep_experts / f.experts), 0, 0.99) * 100) / 100 }
    : { kind: 'dense', layers_before: f.layers, layers_after: f.layers - est.layers_dropped, layers_dropped: Array.from({ length: est.layers_dropped }, (_, i) => 3 + i * 3), keep_mlp: est.keep_width, keep_heads: est.keep_width };
  return {
    params_b: est.total_params_b, active_params_b: est.run_params_b, removed, scores, source_scores: src, control: { scores: rnd }, general: null,
    comparison: { request_type: 'shrink', verdict, headline: verdict === 'success' ? 'Success. It beats the random control by ' + ((scores[t0] - rnd[t0]) * 100).toFixed(1) + ' points. It keeps ' + keepPct + '% of the original model\'s score.' : 'Failed attempt. It does not beat the best existing model at this size.', margin: 0.02, result: { label: 'This shrink', value: scores[t0], per_task: scores, params_b: est.total_params_b, heldout: null }, rows, beaten, not_beaten: notBeaten, unmeasured: [], best_model_shown: true, stub: true },
    stub: true, heal: { steps: plan.heal_steps, kind: plan.heal_steps > 0 ? 'distillation' : 'none', teacher: plan.heal_steps > 0 },
    log: DEMO ? ['Loaded ' + plan.source_model.split('/').pop() + '.', shrinkSummary(plan, est), plan.heal_steps > 0 ? 'Heal: ' + plan.heal_steps + ' steps of distillation toward the original.' : 'No heal step.', 'Random control: the same counts removed at random, same heal.'] : ['Offline run: no model was loaded, so every number is an estimate.', shrinkSummary(plan, est), 'Heal: ' + plan.heal_steps + ' steps of distillation toward the original (estimate).', 'Random control: the same counts removed at random, same heal (estimate).'],
    seconds: { load: 0, rank: 0, slim: 0, heal: est.heal_min * 60, eval: 0 }, estimated: true, plan: clone(plan),
  };
}

/* ---------- the api ---------- */
export const mockApi = {
  async health() {
    await lat();
    const heal = new URLSearchParams(location.search).get('heal') === '1';
    return { ok: true, product: 'Sapien', backend: DEMO ? 'modal' : 'stub', llm: DEMO ? 'openai' : 'none', llm_model: 'gpt-5.4-mini', stub: true, heal_note: heal ? 'The heal recipe is being fixed. Healing on dataset text currently damages the base model, so scores after a heal are not reliable yet.' : null };
  },
  async draftSpec(prompt, files) {
    await sleep(2200);
    if (!prompt || !prompt.trim()) { const e = new Error('Say what the model should do first.'); throw e; }
    if (SHRINK_WORDS.test(prompt) || (DEMO && /keep the \w+ skill of/i.test(prompt))) { const sp = makeShrinkSpec(prompt); remember(sp); return clone(sp); }
    const s = makeSpec(prompt, files);
    specs.set(s.id, s);
    startMockResearch(s);
    saveSpec(s);
    return clone(applyResearch(s));
  },
  async getSpec(id) { await lat(); return clone(applyResearch(getSpec(id))); },
  async getResearch(id) { await lat(); const r = researchNow(id); if (!r) { const e = new Error('no research for this spec'); e.status = 404; throw e; } applyResearch(getSpec(id)); return clone(r); },
  async startResearch(id) {
    await lat();
    const cur = researchNow(id);
    if (cur && cur.status === 'running') { const e = new Error('a research run is already going for this spec'); e.status = 409; throw e; }
    startMockResearch(getSpec(id));
    return { status: 'running' };
  },
  async putSpec(id, spec) {
    await lat();
    const old = getSpec(id);
    if (old.request_type === 'shrink') {
      const block = checkBlock({ ...(spec.shrink || {}), skill_changed: spec.shrink && spec.shrink.keep_skill !== old.shrink.keep_skill }, old.shrink);
      const L = spec.limits || old.limits;
      const lim = { max_total_params_b: clamp(Number((spec.size && spec.size.max_params_b != null && spec.size.max_params_b !== old.size.max_params_b) ? spec.size.max_params_b : L.max_total_params_b), 0.05, 2000), max_run_params_b: Number(L.max_run_params_b), max_heal_min: clamp(Number(L.max_heal_min), 1, 600) };
      lim.max_run_params_b = clamp(lim.max_run_params_b, 0.05, lim.max_total_params_b);
      const f = SMODELS[block.source_model];
      const next = { ...old, ...clone(spec), shrink: block, host: block.source_model, limits: lim, size: { max_params_b: lim.max_total_params_b, min_b: 0.05, max_b: Math.max(32, f.params), host_params_b: f.params }, request_type: 'shrink', stub: true };
      remember(next);
      return clone(next);
    }
    return this._putBuild(id, spec);
  },
  async _putBuild(id, spec) {
    const old = getSpec(id);
    // remember what the person removed, so research does not add it back
    const keep = new Set((spec.benchmarks || []).map((b) => b.id));
    const removed = new Set((old.removed_ids || []).filter((x) => !keep.has(x)));
    for (const b of old.benchmarks || []) if (!keep.has(b.id) && b.id !== 'yours') removed.add(b.id);
    const next = clone({ ...spec, removed_ids: [...removed], benchmark_research: old.benchmark_research, stub: true });
    remember(next);
    return clone(next);
  },
  async createShrink(spec_id) {
    await lat();
    const spec = getSpec(spec_id);
    if (spec.request_type !== 'shrink') throw bad('this spec asks for a build. Use a build run or the builder, not a shrink.');
    const sid = rid('sh_');
    const sess = { id: sid, spec_id, plan: clone(spec.shrink) };
    shrinks.set(sid, sess); store.set('wsp.shrink.' + sid, { spec_id, plan: sess.plan });
    return { shrink_id: sid, plan: shrinkView(sess), stub: true };
  },
  async shrinkPlan(id, fields) {
    await sleep(90);
    const sess = shrinkSess(id);
    if (!sess) { const e = new Error('This shrink session was not found in this browser.'); e.status = 404; throw e; }
    if (fields && Object.keys(fields).length) { sess.plan = checkBlock(fields, sess.plan); store.set('wsp.shrink.' + id, { spec_id: sess.spec_id, plan: sess.plan }); }
    return shrinkView(sess);
  },
  async shrinkRun(id, options) {
    await lat();
    const sess = shrinkSess(id);
    if (!sess) { const e = new Error('This shrink session was not found in this browser.'); e.status = 404; throw e; }
    const view = shrinkView(sess);
    if (!view.ok) throw bad(view.problems.join(' '));
    const jid = rid('j_');
    shrinkJobs.set(jid, { sess, plan: clone(sess.plan), t0: demoNow(), done: null });
    store.set('wsp.sjob.' + jid, { sid: sess.id, spec_id: sess.spec_id, plan: clone(sess.plan), t0: demoNow(), done: null });
    return { job_id: jid };
  },
  async createRun(spec_id, program) { await lat(); const r = newRun(getSpec(spec_id), program); return { run_id: r.id }; },
  async getRun(id) { await lat(); return snapshot(getRunObj(id)); },
  openStream(id, onSnap, onErr, onNote) {
    let run;
    try { run = getRunObj(id); } catch (e) { setTimeout(() => { if (onNote) onNote(e.message); if (onErr) onErr(); }, 0); return () => {}; }
    onSnap(snapshot(run));
    // at high demo speeds the screen updates more often, so a run does not jump
    const t = setInterval(() => onSnap(snapshot(run)), 1000 / Math.min(4, demoSpeed()));
    let n = null;
    if (/^demo-stream/.test(id) && onNote) n = setTimeout(() => { onNote('the worker lost its connection'); clearInterval(t); }, 1500);
    return () => { clearInterval(t); clearTimeout(n); };
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
  async catalog() { await lat(); return clone({ stub: true, host: { model: HOST, layers: HOST_LAYERS, params_b: HOST_B }, parts: PARTS.map(({ task, power, short, ...p }) => p), masks: MASKS, train: TRAIN, product: 'Sapien', counts: { total: PARTS.length + MASKS.length, placeable: PARTS.filter((p) => p.usable !== false).length }, kinds: ['block', 'attn_heads', 'mlp_cluster', 'expert', 'inserted_layer'].map((k) => ({ kind: k, name: k, count: PARTS.filter((p) => p.kind === k).length })), library: { atlas: false, counts: { candidate: PARTS.filter((p) => p.status === 'candidate').length, validated: PARTS.filter((p) => p.status === 'validated').length, rejected: PARTS.filter((p) => p.status === 'rejected').length } } }); },
  async recipes() { await lat(); return clone({ recipes: RECIPES }); },
  async findings() { await lat(); return clone({ findings: FINDINGS }); },
  async stitches() {
    await lat();
    // one line per stitch, in plain words, and no repeats of the same line
    const seen = new Set();
    const out = [];
    for (const s of REAL ? clone(REAL.stitches) : []) {
      s.title = scrubText(String(s.title || '').replace(/^audit:\s*/i, '').replace(/,\s*[\w.-]+$/, ''));
      if (typeof s.status_reason === 'string') s.status_reason = scrubText(s.status_reason);
      if (seen.has(s.title + s.status)) continue;
      seen.add(s.title + s.status);
      out.push(s);
    }
    return { stitches: out };
  },
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
      if (action === 'train') { if (!q.lora && !(q.grafts || []).length) q.lora = { rank: 16 }; if (!q.heal) q.heal = { steps: 300 }; }
      const problems = limitProblems(spec.limits, estimateOf(q));
      if (problems.length) { const e = new Error(problems.join(' ')); e.status = 400; throw e; }
    }
    jobs.set(jid, { id: jid, bid: id, action, program: clone(program), spec, readyAt: demoNow() + 1000 + Math.random() * 1000, done: null });
    return { job_id: jid };
  },
  async getJob(jid) {
    await sleep(80);
    const sj = shrinkJob(jid);
    if (sj) {
      const j = sj;
      const age = demoNow() - j.t0;
      const base = { job_id: jid, action: 'shrink', stub: true, error: null, budget: null };
      const dur = 2600 + Math.min(1000, j.plan.heal_steps || 0) * 8;
      if (j.t0 && age < 700) return { ...base, status: 'queued', result: null };
      if (j.t0 && age < dur) return { ...base, status: 'running', result: null };
      if (!j.done) { j.done = { ...base, status: 'done', result: shrinkResult(j.sess, j.plan) }; if (store.get('wsp.sjob.' + jid)) store.set('wsp.sjob.' + jid, { sid: j.sess.id, spec_id: j.sess.spec_id, plan: j.plan, t0: j.t0, done: j.done }); }
      return clone(j.done);
    }
    const j = jobs.get(jid);
    if (!j) { const e = new Error('no such job'); e.status = 404; throw e; }
    const base = { job_id: jid, action: j.action, stub: true, error: null };
    if (demoNow() < j.readyAt) return { ...base, status: demoNow() < j.readyAt - 1500 ? 'queued' : 'running', result: null, budget: clone(getBuilder(j.bid).budget) };
    if (!j.done) {
      const b = getBuilder(j.bid);
      let p = clone(j.program);
      if (j.action === 'heal' && !p.heal) p.heal = { steps: 300, data: healPool(j.spec), lr: 0.0003, batch_size: 8, seq_len: 384 };
      if (j.action === 'train') { if (!p.lora && !(p.grafts || []).length) p.lora = { rank: 16 }; if (!p.heal) p.heal = { steps: 300, data: healPool(j.spec), lr: 0.0003, batch_size: 8, seq_len: 384 }; }
      const ev = withSynth(j.spec, p, evalProgram(j.spec, p));
      let result;
      if (j.action === 'compile') {
        result = { fit: ev.fit, params_b: ev.params_b, compute_s: 25 };
        b.budget.compute_used_s += 25;
      } else {
        if (b.budget.evals_used >= b.budget.evals_total) { j.done = { ...base, status: 'error', result: null, budget: clone(b.budget), error: 'You have used every score for this build. Start a new build to keep testing.' }; return j.done; }
        const cs = 90 + Math.floor(Math.random() * 120);
        result = { fit: ev.fit, scores: ev.scores, combined: ev.combined, control: ev.control, params_b: ev.params_b, compute_s: cs, status: ev.status, program: p, size: estimateOf(p), limit_problems: [], comparison: mockComparison(j.spec, ev, p), pipeline: ev.pipeline, heal_note: ev.heal_note, components: ev.components, stages: ev.stages, drift: ev.drift, sizes: ev.sizes };
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
    const hd = (steps) => ({ steps, data: healPool(spec), lr: 0.0003, batch_size: 8, seq_len: 384 });
    const room = (spec.size.max_params_b - paramsOf(program)) * 1000;
    const used = new Set(program.grafts.map((g) => g.id));
    const ranked = GRAFT_PARTS.filter((p) => relevance(spec, p) >= 1 && !used.has(p.id) && p.params_m <= room).sort((x, y) => powerOf(spec, y) - powerOf(spec, x) || x.params_m - y.params_m);
    // a free layer at least 3 layers from every part already placed
    const freeAt = (part) => { const taken = program.grafts.map((g) => g.at); let best = part.default_at, bd = -1; for (let l = 1; l <= HOST_LAYERS - 2; l++) { const d = Math.min(99, ...taken.map((t) => Math.abs(t - l))) * 10 - Math.abs(l - part.default_at); if (d > bd) { bd = d; best = l; } } return best; };
    const nameOf = (part) => part.label + ' from ' + short(part.donor);
    const seenDonor = new Set(program.grafts.map((g) => PART[g.id] && PART[g.id].donor));
    const next = ranked.find((p) => !seenDonor.has(p.donor)) || ranked[0];
    if (!program.grafts.length && ranked[0]) {
      const best = ranked[0];
      const p = clonep(); p.grafts.push({ id: best.id, at: best.default_at, control: null, control_seed: 0, glue_rank: 16 });
      out.push({ title: 'Add ' + nameOf(best) + ' at L' + best.default_at + '.', detail: 'It is the best-fitting part for this request in the library. Free probe shows fit ' + fitOf(best, best.default_at).toFixed(2) + '.', tag: 'likely +2 to 5', program: p });
    }
    // drop a part that did not help, when a score is known for it
    program.grafts.forEach((g, i) => {
      const part = PART[g.id];
      if (!part) return;
      if (g.at !== part.default_at) {
        const p = clonep(); p.grafts[i].at = part.default_at;
        out.push({ title: 'Move ' + part.label + ' to L' + part.default_at + '.', detail: 'Free probe: fit ' + fitOf(part, part.default_at).toFixed(2) + ' vs ' + fitOf(part, g.at).toFixed(2) + ' at L' + g.at + '.', tag: 'likely +1 to 2', program: p });
      }
      if (relevance(spec, part) < 1 || powerOf(spec, part) < 0.35) {
        const p = clonep(); p.grafts.splice(i, 1);
        out.push({ title: 'Take out ' + part.label + ' from ' + short(part.donor) + '.', detail: 'It is not built for this request, so it is unlikely to beat a random part. Removing it saves ' + part.params_m + 'M parameters.', tag: 'cleanup', program: p });
      }
    });
    if (program.grafts.length && !program.heal) {
      const p = clonep(); p.heal = hd(300);
      out.push({ title: 'Add a heal pass of 300 steps.', detail: 'Parts without a heal usually lose a few points. Heal trains only the glue.', tag: 'likely +2 to 3', program: p });
    } else if (program.grafts.length && program.heal && program.heal.steps < 600) {
      const p = clonep(); p.heal = hd(600);
      out.push({ title: 'Heal for ' + 600 + ' steps instead of ' + program.heal.steps + '.', detail: 'A longer heal lets the glue settle. It costs about ' + Math.round(600 * 0.3 * 1.5 / 60) + ' minutes of compute.', tag: 'likely +1 to 2', program: p });
    } else if (program.grafts.length && program.heal && program.heal.steps >= 600) {
      const p = clonep(); p.heal = hd(300);
      out.push({ title: 'Try a shorter heal of 300 steps.', detail: 'If the score holds, the build is cheaper to repeat.', tag: 'cleanup', program: p });
    }
    if (program.grafts.length && program.grafts.length < 3 && next) {
      const p = clonep(); const at = freeAt(next); p.grafts.push({ id: next.id, at, control: null, control_seed: 0, glue_rank: 16 });
      out.push({ title: 'Add ' + nameOf(next) + ' at L' + at + '.', detail: 'A different donor, at a layer at least 3 away from the other parts. Free probe shows fit ' + fitOf(next, at).toFixed(2) + '.', tag: 'likely +1 to 3', program: p });
    }
    if (!program.grafts.length && !program.lora) {
      const p = clonep(); p.lora = { rank: 16 };
      if (!p.heal) p.heal = hd(300);
      out.push({ title: 'Add a rank 16 LoRA on top.', detail: 'It closes part of the gap to the fine-tuning bar.', tag: 'bigger change', program: p });
    }
    if (program.grafts.length && !program.lora && program.grafts.length === 0) { /* a LoRA cannot sit on a program with parts */ }
    if (program.masks) {
      const p = clonep(); p.masks = null;
      out.push({ title: 'Drop the deletion mask.', detail: 'It saves no size and costs a little on the guards.', tag: 'cleanup', program: p });
    }
    if (program.grafts.length > 2) {
      const p = clonep(); p.grafts = p.grafts.slice(0, 2);
      out.push({ title: 'Keep only two parts.', detail: 'Three or more close together tend to collapse the guards.', tag: 'cleanup', program: p });
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

import { makeExtra } from './mock-extra.js';
Object.assign(mockApi, makeExtra({ getSpec, getParts: () => PARTS, getRecipes: () => RECIPES, getMasks: () => MASKS }));

/* ---------- demo mode: fixed objects that open by URL ---------- */
function withSeed(seed, fn) {
  const orig = Math.random;
  let a = seed >>> 0;
  Math.random = () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  try { return fn(); } finally { Math.random = orig; }
}
function shiftTimes(run, d) {
  run.started += d;
  run.activity.forEach((e) => { e.ts += d; });
  run.agents.forEach((a) => { a.thinking.forEach((x) => { x.ts += d; }); a.tools.forEach((x) => { x.ts += d; }); a.nudges.forEach((x) => { x.ts += d; }); });
}
function demoHeal(steps) { return { steps, data: 'math', lr: 0.0003, batch_size: 8, seq_len: 384 }; }
const demoGraft = (id, at, rank = 16) => ({ id: realId(id), at, control: null, control_seed: 0, glue_rank: rank });
const demoProg = (grafts, heal) => ({ host: HOST, seed: 0, masks: null, task_vectors: [], grafts, heal: heal ? demoHeal(heal) : null, lora: null });

function prespin(run, until) {
  vclock = Date.now() / 1000 - 600;
  const t0 = vclock;
  let guard = 0;
  while (!until(run) && guard++ < 6000) { tickRun(run); vclock += 1.5; }
  const span = vclock - t0;
  const end = vclock;
  const shift = Date.now() / 1000 - end;
  vclock = null;
  return { span, shift, end };
}

function seedLiveRun(spec) {
  withSeed(11, () => {
    const r = newRun(spec);
    runs.delete(r.id); r.id = V.DEMO_IDS.runLive; runs.set(r.id, r);
    r.hold = true;
    const { span, shift } = prespin(r, (x) => x.evals_used >= Math.round(x.evals_total * 0.4));
    r.elapsed = Math.round(span);
    shiftTimes(r, shift);
    r.started = Date.now() / 1000 - r.elapsed;
    r.lazy = true;
  });
}

// Builds a finished run from a list of scored programs, played in order through the same code that plays a live run,
// so the log, the curve, the leaderboard and the agent pages all describe the same evals.
// item: { agent, prog, scores (map) | score (first target), vs, control, status, guards }
function replayRun(spec, id, list, o) {
  const r = newRun(spec);
  runs.delete(r.id); r.id = id; runs.set(id, r);
  r.activity = []; r.actId = 0; r.tickN = 0;
  r.evals_total = list.length; r.cap = 9;
  r.compute_total = spec.budget.compute_min * 60;
  const T = Date.now() / 1000;
  const t0 = T - o.agoMin * 60 - o.elapsed;
  const at = (k) => t0 + o.elapsed * k;
  const ts = (spec.main_only ? targetsOf(spec).slice(0, 1) : targetsOf(spec));
  const first = ts[0];
  vclock = t0;
  logAct(r, 'judge', 'system', 'info', SAY('Run started: ' + list.length + ' evals, ' + spec.budget.compute_min + ' compute minutes.', 'Run started on the offline backend.'));
  const n = list.length;
  const notes = o.notes || [];
  let noteAt = 0;
  list.forEach((it, i) => {
    vclock = at((i + 1) / (n + 2));
    r.tickN = 24 * (i + 1);
    const a = r.agents.find((x) => x.id === it.agent);
    const prog = it.prog;
    const ev = evalProgram(spec, prog);
    if (it.scores) Object.assign(ev.scores, it.scores);
    else if (it.score != null) ev.scores[first.id] = it.score;
    if (it.guards) Object.assign(ev.scores, it.guards);
    ev.status = it.status || 'ok';
    ev.combined = Math.round(tmeanOf(spec, ev.scores) * 0.6 * 1000) / 1000;
    if (it.vs != null) {
      const cs = {};
      for (const b of ts) cs[b.id] = it.control && it.control[b.id] != null ? it.control[b.id] : Math.round((ev.scores[b.id] - it.vs * (ev.scores[b.id] - b.base) / Math.max(0.01, ev.scores[first.id] - first.base)) * 1000) / 1000;
      ev.control = { scores: cs };
      ev.vs_control = Math.round((ev.scores[first.id] - cs[first.id]) * 1000) / 1000;
    } else { ev.control = null; ev.vs_control = null; }
    pinned.set(pinKey(spec, prog), clone(ev));
    a.cur = prog; a.refined = false;
    const g = prog.grafts[0], part = g && PART[g.id];
    const hs = prog.heal || { steps: 300, data: healPool(spec) };
    const altAt = part ? clamp(g.at + (g.at > part.default_at ? -2 : 2), 1, HOST_LAYERS - 2) : null;
    const lv = part && r.vsByPart ? r.vsByPart[part.id] : null;
    think(a, V.planThought({ style: a.style, part, at: g && g.at, rank: (g && g.glue_rank) || 16, steps: hs.steps, data: hs.data, best: r.best ? tmeanOf(spec, r.best.scores) : null, host: tmeanOf(spec, r.base_scores), bench: benchLabel(spec), vs: lv != null && lv > 0.01 ? (lv * 100).toFixed(1) : null, fitHere: part ? fitOf(part, g.at) : 0, fitAlt: part ? fitOf(part, altAt) : null, altAt, floor: 0.45 }));
    logAct(r, a.id, 'plan', 'info', V.planLine({ count: 3, refine: i > 4 && i % 3 === 0, round: Math.floor(i / 3) + 1, parts: 1 + (prog.grafts.length > 1 ? 1 : 0) + (i % 3) }), { count: 3 });
    if (part) {
      const f = fitOf(part, g.at);
      tool(a, 'probe(' + g.id + '@' + g.at + ')', 'fit ' + f.toFixed(2) + ', free');
      logAct(r, a.id, 'probe', 'ok', V.probeLine({ name: readable(prog), fit: f, at: g.at }), { name: readable(prog), fits: [Math.round(f * 100) / 100], program_hash: progHash(prog) });
    }
    tool(a, 'compile(' + describe(prog) + ')', (30 + (hash(describe(prog)) % 40)) + ' s, ' + paramsOf(prog).toFixed(2) + 'B params, ' + (4.5 + paramsOf(prog) * 2.1).toFixed(1) + ' GB peak');
    if (prog.heal) tool(a, 'heal(' + prog.heal.steps + ' steps)', 'loss ' + (2.0 + (hash(spec.id + prog.heal.data + prog.heal.steps) % 30) / 100).toFixed(2) + ' to ' + (1.2 + (hash(describe(prog)) % 30) / 100).toFixed(2));
    tool(a, 'eval(search split)', 'running', 'info');
    r.compute_used += it.compute || 40 + (hash(describe(prog)) % 40);
    finishEval(r, a);
    // analyst notes land at set points in the run
    while (noteAt < notes.length && i + 1 >= Math.round(notes[noteAt].after * n)) {
      const nt = notes[noteAt++];
      const an = r.agents.find((x) => x.id === 'analyst');
      if (nt.text) {
        an.lessons++; an.now = nt.text; think(an, nt.text); tool(an, 'read_results(' + (i + 1) + ' programs)', 'ok');
        logAct(r, 'analyst', 'map', 'info', nt.text, { detail: 'program | score | fit\n' + r.board.slice(0, 4).map((e) => e.name + ' | ' + (tmeanOf(spec, e.scores) * 100).toFixed(1) + '% | ' + (e.status === 'ok' ? 'yes' : 'no')).join('\n') });
      } else { r.tickN = 24 * (i + 1); analystStep(r, an); }
    }
  });
  // the judge runs a control for the top programs, then the search ends
  vclock = at((n + 1) / (n + 2));
  const ju = r.agents.find((x) => x.id === 'judge');
  const cmean = (e) => (e.control ? ts.reduce((x, b) => x + e.control.scores[b.id], 0) / ts.length : null);
  r.board.slice(0, 3).filter((e) => e._pending != null).forEach((e) => {
    e.vs_control = e._pending; e.control = e._ctl || null;
    ju.attempts++;
    const cm = cmean(e);
    const line = o.controlText ? o.controlText(e) : V.controlLine({ name: e.name, score: tmeanOf(spec, e.scores), control: cm != null ? cm : e.scores[first.id] - e.vs_control, pts: ((cm != null ? tmeanOf(spec, e.scores) - cm : e.vs_control) * 100).toFixed(1) });
    logAct(r, 'judge', 'verdict', 'ok', line, { frontier: r.board.slice(0, 3).map((x) => ({ program_hash: x.program_hash, name: x.name, target: tmeanOf(spec, x.scores) })), retired: [], killed: [] });
    think(ju, line);
    tool(ju, 'control(random part)', '+' + ((cm != null ? tmeanOf(spec, e.scores) - cm : e.vs_control) * 100).toFixed(1));
  });
  r.board.forEach((e) => { if (e.vs_control == null && e._pending != null) { e.vs_control = e._pending; e.control = e._ctl || null; } });
  vclock = at((n + 1.5) / (n + 2));
  if (o.heldout !== false) r.board.slice(0, 3).forEach((e) => logAct(r, 'judge', 'heldout', 'info', 'Held-out check of ' + e.name, { name: e.name, program_hash: e.program_hash, target: Math.max(0, tmeanOf(spec, e.scores) - 0.016), scores: e.scores }));
  vclock = at(1);
  logAct(r, 'judge', 'milestone', 'ok', 'Search finished. The best program is ready to open in the builder.');
  r.status = 'done'; r.outcome = 'finished'; r.elapsed = Math.round(o.elapsed);
  r.agents.forEach((a) => { a.status = 'done'; a.now = 'Finished'; });
  r.started = t0;
  vclock = null;
  return r;
}

function seedDoneRun(spec) {
  withSeed(23, () => {
    const tg = spec.benchmarks.find((b) => b.kind === 'target');
    const planner = (i) => 'planner-' + (1 + (i % 3));
    const mk = (parts, heal, score, vs, agent, extra = {}) => ({ agent, prog: demoProg(parts, heal), score, vs, ...extra });
    const hand = [
      mk([demoGraft('qmath_l18_contrast512', 9), demoGraft('qmath_block_8_19', 6)], 600, 0.685, 0.063, 'planner-1'),
      mk([demoGraft('qmath_block_8_19', 6)], 600, 0.662, 0.041, 'planner-1'),
      mk([demoGraft('r1_block_10_21', 7)], 600, 0.654, 0.036, 'planner-1'),
      mk([demoGraft('qmath_l18_contrast512', 9)], 300, 0.646, 0.032, 'planner-2'),
      mk([demoGraft('qmath_l18_contrast512', 9), demoGraft('mistral_heads_8', 6)], 450, 0.641, 0.027, 'planner-3'),
      { agent: 'planner-3', prog: { host: HOST, seed: 0, masks: null, task_vectors: [], grafts: [], heal: demoHeal(600), lora: { rank: 16 } }, score: 0.63, vs: null },
      mk([demoGraft('mistral_heads_8', 6)], 300, 0.628, 0.019, 'planner-2'),
      mk([demoGraft('qmath_l10_activation512', 6)], 300, 0.617, 0.012, 'planner-2'),
      mk([demoGraft('phi_block_4_11', 4)], 300, 0.609, 0.006, 'planner-1'),
      mk([demoGraft('lfm_experts_2', 10)], 300, 0.594, -0.003, 'planner-3'),
      mk([demoGraft('gemma_block_5_12', 5)], 300, 0.582, -0.008, 'planner-1'),
      mk([demoGraft('qmath_block_8_19', 11)], 100, 0.312, null, 'planner-1', { status: 'collapsed', guards: Object.fromEntries(spec.benchmarks.filter((b) => b.kind === 'guard').map((b) => [b.id, Math.round(b.floor * 0.6 * 1000) / 1000])) }),
    ];
    // the rest of the search: programs the planners tried that did not reach the top of the board
    const tmp = newRun(spec); runs.delete(tmp.id);
    const taken = new Set(hand.map((h) => pinKey(spec, h.prog)));
    tmp.seen = new Set(hand.map((h) => progHash(h.prog)));
    const filler = [];
    for (let k = 0; filler.length < V.SOLVER_EVALS - hand.length && k < 600; k++) {
      const a = tmp.agents.find((x) => x.id === planner(k));
      tmp.tickN = 30;
      const prog = freshProgram(tmp, a);
      const key = pinKey(spec, prog);
      if (taken.has(key) || (!prog.grafts.length && !prog.lora)) continue;
      taken.add(key); tmp.seen.add(progHash(prog));
      const ev = evalProgram(spec, prog);
      if (ev.status !== 'ok') continue;
      const sc = Math.min(ev.scores[tg.id], 0.575 + (hash(describe(prog)) % 75) / 1000);
      filler.push({ agent: a.id, prog, score: Math.round(sc * 1000) / 1000, vs: sc > 0.6 ? Math.round((sc - 0.575) * 0.9 * 1000) / 1000 : null });
    }
    const rest = [...hand.slice(1), ...filler].map((x) => ({ x, k: x.score + (hash(describe(x.prog)) % 100 - 50) / 700 })).sort((p, q) => p.k - q.k).map((p) => p.x);
    replayRun(spec, V.DEMO_IDS.runDone, [...rest, hand[0]], { elapsed: 74 * 60, agoMin: 4 * 60 + 12, notes: [{ after: 0.3 }, { after: 0.6 }, { after: 0.9 }] });
  });
}

// The language write-in, played as a finished run. The final program carries the measured numbers from
// results/README.md ("Language write-in", 2026-11-15). Earlier attempts are simulated and score lower.
function seedLangRun(spec) {
  const L = V.LANG;
  const slice = V.LANG_SLICE_ID;
  const tid = (i) => 'belebele:' + L.codes[i];
  const keys = [...L.codes.slice(0, 5).map((c, i) => tid(i)), 'belebele:en', 'mmlu_guard'];
  const finalScores = Object.fromEntries(keys.map((k, i) => [k, i < 6 ? L.graft[i] / 100 : L.mmlu[1] / 100]));
  const baseScores = Object.fromEntries(keys.map((k, i) => [k, i < 6 ? L.base[i] / 100 : L.mmlu[0] / 100]));
  const scrScores = Object.fromEntries(L.codes.slice(0, 5).map((c, i) => [tid(i), L.scrambled[i] / 100]));
  const prog = (grafts, heal) => ({ host: HOST, seed: 0, masks: null, task_vectors: [], grafts, heal: heal ? { steps: heal, data: 'multilingual', lr: 0.0003, batch_size: 8, seq_len: 384 } : null, lora: null });
  const gr = (id, at, rank = 16) => ({ id, at, control: null, control_seed: 0, glue_rank: rank });
  const have = (ids) => ids.filter((x) => PART[x]);
  const pool = Object.values(PART).filter((x) => x.task === 'multilingual' && x.id !== slice && x.usable && x.params_m > 0 && x.params_m < 1500).sort((x, y) => x.params_m - y.params_m || (x.id < y.id ? -1 : 1));
  const small = pool.slice(0, 5).map((x) => x.id);
  const mid = pool.slice(5, 6).map((x) => x.id);
  // a program scores between the base and the final, in proportion f
  const level = (f, tag) => {
    const sc = {}, ctl = {};
    keys.forEach((k) => {
      const j = ((hash(k + tag) % 100) - 50) / 6000;
      sc[k] = Math.round((baseScores[k] + (finalScores[k] - baseScores[k]) * clamp(f + j, 0, 0.995)) * 1000) / 1000;
      if (scrScores[k] != null) ctl[k] = Math.round((baseScores[k] + (scrScores[k] - baseScores[k]) * clamp(f, 0.4, 1)) * 1000) / 1000;
    });
    return { scores: sc, control: ctl };
  };
  const item = (agent, pr, f, tag) => ({ agent, prog: pr, ...level(f, tag), vs: f > 0.12 ? 1 : null });
  const list = [];
  list.push(item('planner-1', prog([], 300), 0.02, 'h'));
  for (const id of small) list.push(item('planner-2', prog([gr(id, PART[id].default_at)], 300), 0.04 + (hash(id) % 60) / 1000, id));
  for (const id of mid) list.push(item('planner-1', prog([gr(id, PART[id].default_at)], 600), 0.3, id));
  have(['block/Qwen/Qwen2.5-7B-Instruct/0-4', 'block/Qwen/Qwen2.5-7B-Instruct/18-22', 'block/Qwen/Qwen2.5-7B-Instruct/23-27']).forEach((id, i) => list.push(item('planner-3', prog([gr(id, PART[id].default_at)], 600), [0.55, 0.5, 0.45][i], id)));
  list.push(item('planner-1', prog([gr(slice, 4)], 0), 0.55, 'noheal'));
  list.push(item('planner-1', prog([gr(slice, 4)], 300), 0.86, 'h300'));
  list.push(item('planner-1', prog([gr(slice, 2)], 600), 0.78, 'L2'));
  list.push(item('planner-1', prog([gr(slice, 3)], 600), 0.9, 'L3'));
  list.push(item('planner-1', prog([gr(slice, 5)], 600), 0.91, 'L5'));
  list.push(item('planner-2', prog([gr(slice, 6)], 600), 0.84, 'L6'));
  list.push(item('planner-3', prog([gr(slice, 4, 8)], 600), 0.88, 'r8'));
  list.push(item('planner-3', prog([gr(slice, 4, 32)], 600), 0.91, 'r32'));
  list.push(item('planner-2', prog([gr(slice, 4, 64)], 900), 0.94, 'r64'));
  list.push(item('planner-1', prog([gr(slice, 4)], 600), 0.93, 'h600'));
  if (small[1]) list.push(item('planner-3', prog([gr(slice, 4), gr(small[1], 10)], 600), 0.95, 'c1'));
  if (small[0]) list.push(item('planner-2', prog([gr(slice, 4), gr(small[0], 9)], 900), 0.965, 'c2'));
  if (small[3]) list.push(item('planner-3', prog([gr(slice, 4), gr(small[3], 8)], 600), 0.93, 'c3'));
  let extra = 0;
  while (list.length < spec.budget.evals - 1) { extra++; list.push(item('planner-2', prog([gr(slice, 4 + (extra % 3), 16)], 450 + 50 * extra), 0.8 + (extra % 5) / 50, 'x' + extra)); }
  list.sort((p, q) => p.scores[tid(0)] - q.scores[tid(0)]);
  list.push({ agent: 'planner-1', prog: prog([gr(slice, 4)], 900), scores: finalScores, control: scrScores, vs: 1 });
  const ts = (spec.main_only ? targetsOf(spec).slice(0, 1) : targetsOf(spec));
  const mean = (m) => ts.reduce((x, b) => x + m[b.id], 0) / ts.length;
  return withSeed(31, () => replayRun(spec, V.DEMO_IDS.runLang, list, {
    elapsed: 49 * 60, agoMin: 38, heldout: false,
    notes: [
      { after: 0.35, text: V.LANG_FINDINGS[0] }, { after: 0.55, text: V.LANG_FINDINGS[1] }, { after: 0.7, text: V.LANG_FINDINGS[2] },
      { after: 0.8, text: V.LANG_FINDINGS[3] }, { after: 0.9, text: V.LANG_FINDINGS[4] }, { after: 0.99, text: V.LANG_FINDINGS[5] },
    ],
    controlText: (e) => 'Control for ' + e.name + ': the same ' + e.params_b.toFixed(1) + 'B build with the donor weights scrambled reaches ' + (mean(e.control.scores) * 100).toFixed(1) + '% on average, the real build ' + (mean(e.scores) * 100).toFixed(1) + '%. The donor adds ' + ((mean(e.scores) - mean(e.control.scores)) * 100).toFixed(1) + ' points.',
  }));
}

function seedEmptyRun(legal) {
  const lim = { max_total_params_b: 1.3, max_run_params_b: 1.3, max_heal_min: 20 };
  const sp = clone(legal); sp.limits = lim; sp.size.max_params_b = 1.3;
  sp.id = V.DEMO_IDS.specClause; sp.title = 'Clause checker under 1.3B';
  specs.set(sp.id, sp);
  researchOf.set(sp.id, { started: demoNow() - 90000, plan: researchPlan(sp), mode: null, applied: false });
  const r = newRun(sp);
  runs.delete(r.id); r.id = V.DEMO_IDS.runEmpty; runs.set(r.id, r);
  r.activity = []; r.actId = 0;
  const T = now();
  vclock = T - 60 * 22 * 60;
  logAct(r, 'judge', 'system', 'info', 'Run started: ' + r.evals_total + ' evals, ' + sp.budget.compute_min + ' compute minutes. Size limit 1.30B total.');
  // every legal part in the library is bigger than the 60M the limit leaves room for
  const legalParts = GRAFT_PARTS.filter((p) => p.task === 'legal').sort((x, y) => x.params_m - y.params_m);
  const small = legalParts[0];
  const pool = legalParts.slice(0, 3);
  const tries = [];
  ['planner-1', 'planner-2', 'planner-3'].forEach((aid, k) => { for (let j = 0; j < 3; j++) { const part = pool[(k + j) % pool.length]; tries.push([aid, part, clamp(part.default_at + j, 1, 14)]); } });
  tries.forEach(([aid, part, at]) => {
    vclock += 4.5;
    const a = r.agents.find((x) => x.id === aid);
    const prog = { host: HOST, seed: 0, masks: null, task_vectors: [], grafts: [{ id: part.id, at, control: null, control_seed: 0, glue_rank: 16 }], heal: demoHeal(300), lora: null };
    const tot = paramsOf(prog);
    think(a, part.label + ' (' + short(part.donor) + ') adds ' + part.params_m + 'M parameters. The host is 1.24B, so the total is ' + tot.toFixed(2) + 'B and the limit is 1.30B. A legal part that fits would have to add under 60M, and there is none.');
    tool(a, 'probe(' + part.id + '@' + at + ')', 'refused: ' + tot.toFixed(2) + 'B over 1.30B', 'bad');
    logAct(r, aid, 'refused', 'bad', part.label + ' at L' + at + ' would bring the total to ' + tot.toFixed(2) + 'B, over the 1.30B limit. Refused before any eval.');
    r.refused = (r.refused || 0) + 1;
  });
  vclock += 6;
  logAct(r, 'analyst', 'map', 'info', 'No program fit. The smallest legal part in the library adds ' + small.params_m + 'M parameters, and the limit leaves room for 60M. Raise the size limit to about ' + (Math.ceil((HOST_B + small.params_m / 1000) * 10) / 10 + 0.1).toFixed(1) + 'B and try again.');
  r.agents.find((x) => x.id === 'analyst').now = 'Wrote the note on the size limit';
  logAct(r, 'judge', 'milestone', 'info', 'Search finished. Nothing was scored.');
  r.status = 'done'; r.outcome = 'finished'; r.elapsed = 41;
  r.empty_reason = 'Every proposal went over the 1.30B size limit, so nothing was scored.';
  r.agents.forEach((a) => { a.status = 'done'; a.now = 'Finished'; });
  vclock = null;
  r.started = T - 22 * 60 * 60 - 41;
  shiftTimes(r, 0);
}

function demoAttempt(spec, prog, target, control, n, compute_s) {
  const tg = spec.benchmarks.find((b) => b.kind === 'target');
  const ev = evalProgram(spec, prog);
  ev.scores[tg.id] = target;
  ev.combined = Math.round(target * 0.6 * 1000) / 1000;
  ev.control = control != null ? { scores: { [tg.id]: control } } : null;
  ev.vs_control = control != null ? Math.round((target - control) * 1000) / 1000 : null;
  // scoring this program again in the builder gives the same numbers
  pinned.set(pinKey(spec, prog), clone(ev));
  const full = withSynth(spec, prog, ev);
  return { n, action: 'score', program: prog, scores: ev.scores, combined: ev.combined, control: ev.control, params_b: ev.params_b, status: ev.status, compute_s, comparison: mockComparison(spec, full, prog), size: estimateOf(prog), pipeline: full.pipeline, heal_note: null, components: full.components, stages: full.stages, drift: full.drift };
}

function seedBuilder(spec) {
  const R = V.DEMO_IDS;
  const final = demoProg([demoGraft('qmath_block_8_19', 6), demoGraft('qmath_l18_contrast512', 9)], 600);
  const attempts = [
    demoAttempt(spec, demoProg([], 300), 0.585, null, 1, 96),
    demoAttempt(spec, demoProg([demoGraft('qmath_l18_contrast512', 9)], 300), 0.612, 0.596, 2, 148),
    demoAttempt(spec, demoProg([demoGraft('qmath_block_8_19', 6)], 300), 0.641, 0.604, 3, 171),
    demoAttempt(spec, demoProg([demoGraft('qmath_block_8_19', 6), demoGraft('qmath_l18_contrast512', 9)], 300), 0.662, 0.611, 4, 183),
    demoAttempt(spec, final, 0.685, 0.622, 5, 204),
  ];
  const budget = { evals_used: 9, evals_total: 12, compute_used_s: 802, compute_total_s: 2700 };
  builders.set(R.builder, { id: R.builder, spec_id: R.specMath, budget: clone(budget) });
  // the stored draft is kept across reloads, unless it was seeded from another library (real or mock)
  const lib = (REAL ? 'real' : 'mock') + ':4';
  let fresh = false;
  try { fresh = sessionStorage.getItem(storageKey('wsp.seedlib')) !== lib; sessionStorage.setItem(storageKey('wsp.seedlib'), lib); } catch (e) { /* storage can be off */ }
  const put = (k, v) => { try { const key = storageKey(k); if (fresh || sessionStorage.getItem(key) == null) sessionStorage.setItem(key, JSON.stringify(v)); } catch (e) { /* storage can be off */ } };
  put('wsp.prog.' + R.builder, final);
  put('wsp.att.' + R.builder, attempts);
  put('wsp.budget.' + R.builder, budget);
}

function seedDemo() {
  const R = V.DEMO_IDS;
  const math = makeSpec('A math tutor that runs on a phone and shows its work', [], R.specMath);
  specs.set(R.specMath, math);
  specs.set('sp_demo', math);
  researchOf.set(R.specMath, { started: demoNow() - 90000, plan: researchPlan(math), mode: null, applied: false });
  applyResearch(math);
  const legal = makeSpec('A contract reader that answers questions about clauses and keeps its general skills', [], R.specLegal);
  specs.set(R.specLegal, legal);
  startMockResearch(legal, true);
  const C = V.CODE_SHRINK_SPEC;
  const code = makeShrinkSpec(C.prompt, R.specShrink);
  code.shrink = checkBlock({ keep_experts: C.keep_experts, heal_steps: C.heal_steps, rule: 'saliency' }, code.shrink);
  code.title = C.title; code.summary = C.summary; code.prompt = C.prompt; code.task_type = 'code'; code.drafted_by = DEMO_DRAFTER;
  code.limits = clone(C.limits); code.size.max_params_b = C.limits.max_total_params_b;
  code.benchmarks = clone(C.benchmarks);
  specs.set(R.specShrink, code);
  const sess = { id: R.shrink, spec_id: R.specShrink, plan: clone(code.shrink) };
  shrinks.set(R.shrink, sess);
  shrinkJobs.set(V.DEMO_SHRINK_JOB, { sess, plan: clone(sess.plan), t0: 0, done: null });
  // the finished and the empty run each get their own spec, so the sidebar shows distinct titles
  const solver = clone(math); solver.id = R.specSolver; solver.title = 'Grade-school math solver';
  solver.budget = { ...solver.budget, evals: V.SOLVER_EVALS, compute_min: 90 };
  specs.set(R.specSolver, solver);
  researchOf.set(R.specSolver, { started: demoNow() - 90000, plan: researchPlan(solver), mode: null, applied: false });
  applyResearch(solver);
  // the language write-in: a finished run with measured numbers
  const lang = makeSpec(V.LANG_LOOK.prompt, [], R.specLang);
  specs.set(R.specLang, lang);
  researchOf.set(R.specLang, { started: demoNow() - 120000, plan: researchPlan(lang), mode: null, applied: false });
  applyResearch(lang);
  seedEmptyRun(legal);
  seedDoneRun(solver);
  seedLangRun(lang);
  seedLiveRun(math);
  seedBuilder(math);
  V.seedDemoRecents();
}
if (DEMO) seedDemo();
