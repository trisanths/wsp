import { useState, useEffect, useRef } from './lib.js';
import { DEMO, storageKey } from './demo-flag.js';

/** True when data carries stub: true and the app is not in demo mode (demo shows sample data as real). */
export const showStub = (x) => !!(x && x.stub) && !DEMO;

/* ---------- formatting ---------- */
export const pct = (x, d = 1) => (x == null || isNaN(x) ? '-' : (x * 100).toFixed(d));
export const pctS = (x, d = 1) => (x == null || isNaN(x) ? '-' : (x * 100).toFixed(d) + '%');
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const bsize = (x) => (x == null ? '-' : Number(x).toFixed(2) + 'B');
export function dur(s) {
  s = Math.max(0, Math.round(s || 0));
  const m = Math.floor(s / 60);
  if (m >= 60) return Math.floor(m / 60) + 'h ' + String(m % 60).padStart(2, '0') + 'm';
  if (m >= 1) return m + ' min' + (s % 60 ? ' ' + (s % 60) + 's' : '');
  return s + 's';
}
export function clock(ts) {
  const d = new Date(ts * 1000);
  return String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0') + ':' + String(d.getSeconds()).padStart(2, '0');
}
export const sgn = (x, d = 1) => (x >= 0 ? '+' : '-') + Math.abs(x * 100).toFixed(d);

// "5 min ago", from a time in milliseconds
export function ago(ts) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return 'just now';
  if (s < 3600) return Math.round(s / 60) + ' min ago';
  if (s < 86400) return Math.round(s / 3600) + ' h ago';
  return Math.round(s / 86400) + ' d ago';
}
// A quiet suffix for rows that share a title: the kind and when it was opened, like "run, 5 min ago".
export function rowSuffixes(rows) {
  const count = {};
  rows.forEach((r) => { const k = (r.title || r.id) + ''; count[k] = (count[k] || 0) + 1; });
  const out = {};
  rows.forEach((r) => { if (count[(r.title || r.id) + ''] > 1) out[r.kind + r.id] = r.kind + ', ' + ago(r.ts || Date.now()); });
  return out;
}
// the pool name the heal data comes from, by the kind of task
export function healPool(spec) {
  const t = spec && spec.task_type;
  if (t === 'math') return 'math';
  if (t === 'knowledge') return 'general';
  if (t === 'code') return 'code';
  return 'general';
}
// A readable name for a benchmark id, used only when the server sends no label.
const BENCH_NAMES = {
  gsm8k: 'GSM8K', humaneval: 'HumanEval', mbpp: 'MBPP', mmlu: 'MMLU', mmlu_guard: 'MMLU', ifeval: 'IFEval', if_guard: 'IFEval',
  arc: 'ARC Challenge', hellaswag: 'HellaSwag', truthfulqa: 'TruthfulQA', winogrande: 'WinoGrande', math: 'MATH', bbh: 'BIG-Bench Hard', gpqa: 'GPQA',
};
export function benchName(id) {
  const k = String(id || '');
  if (BENCH_NAMES[k]) return BENCH_NAMES[k];
  const m = /^mmlupro:(.+)$/.exec(k);
  if (m) return 'MMLU-Pro ' + m[1].replace(/_/g, ' ');
  return k.replace(/[_:]+/g, ' ').replace(/^./, (c) => c.toUpperCase());
}
// how a run ended: running, finished, stopped or error. A stopped run still has status "done".
export function runEnd(snap) {
  const s = snap.status;
  if (s === 'error' || (s === 'done' && snap.outcome === 'error')) return 'error';
  if (s === 'done') return snap.outcome === 'stopped' || (snap.outcome == null && snap.stopped_early === true) ? 'stopped' : 'finished';
  return 'running';
}
// the status word kept in Recent: a stopped run shows as "stopped", not "done"
export function recentStatus(snap) {
  const e = runEnd(snap);
  return e === 'stopped' ? 'stopped' : snap.status;
}

/* ---------- colors ---------- */
// Data colours come from the theme tokens, so they follow light and dark.
const PALETTE = ['var(--purple)', 'var(--amber)', 'var(--green)', 'var(--red)', 'var(--text-3)'];
function hash(s) {
  let x = 7;
  for (let i = 0; i < s.length; i++) x = (x * 31 + s.charCodeAt(i)) >>> 0;
  return x;
}
export function agentColor(id) {
  if (id === 'analyst') return 'var(--blue)';
  if (id === 'judge') return 'var(--red)';
  const m = /^planner-(\d+)$/.exec(id || '');
  if (m) return ['var(--purple)', 'var(--green)', 'var(--amber)', 'var(--blue)', 'var(--red)', 'var(--text-3)'][(Number(m[1]) - 1) % 6];
  return PALETTE[hash(id || '') % PALETTE.length];
}
export function agentInitials(a) {
  const id = a.id || '';
  const m = /^planner-(\d+)$/.exec(id);
  if (m) return 'P' + m[1];
  const name = a.name || id;
  return name.slice(0, 2).replace(/^./, (c) => c.toUpperCase());
}
export function donorColor(donor) {
  if (!donor) return 'var(--text-3)';
  return PALETTE[hash(donor.split('/').pop().toLowerCase().replace(/[^a-z]/g, '').slice(0, 5)) % PALETTE.length];
}
export function donorLetter(donor, label) {
  const name = (donor || label || '?').split('/').pop();
  return name.charAt(0).toUpperCase();
}
export const STATUS_COLOR = {
  thinking: 'var(--blue)', probing: 'var(--amber)', compiling: 'var(--amber)', scoring: 'var(--green)',
  idle: 'var(--text-4)', paused: 'var(--text-4)', done: 'var(--text-4)',
};
export const STATUS_ACTIVE = { thinking: 1, probing: 1, compiling: 1, scoring: 1 };

/* ---------- benchmark helpers ---------- */
export function primaryTarget(spec) {
  return (spec.benchmarks || []).find((b) => b.kind === 'target') || null;
}
// progress of a score toward its spec, 0..1
export function benchProgress(b, score) {
  if (score == null) return 0;
  if (b.kind === 'guard') return b.floor ? clamp(score / b.floor, 0, 1) : 1;
  const base = b.base ?? 0;
  const span = (b.target ?? 1) - base;
  if (span <= 0) return score >= (b.target ?? 1) ? 1 : 0;
  return clamp((score - base) / span, 0, 1);
}

/* ---------- program helpers ---------- */
export function emptyProgram(host) {
  return { host, seed: 0, masks: null, task_vectors: [], grafts: [], heal: null, lora: null };
}
// Sapien builds one network: drop any tap fields from saved programs and server replies
export function hasTaps(p) {
  return !!p && ((p.taps && p.taps.length > 0) || p.tap_train != null);
}
export function normProgram(p) {
  if (!p) return p;
  const { taps, tap_train, ...rest } = p;
  return rest;
}
// "Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B"
export function donorShort(d) {
  return String(d || '').split('/').pop().replace(/[-_](instruct|it|chat)$/i, '');
}
export function maskIds(expr) {
  const out = [];
  (function walk(n) {
    if (!n) return;
    if (n.op === 'use') out.push(n.id);
    else if (n.op === 'restrict') walk(n.inner);
    else { walk(n.a); walk(n.b); }
  })(expr);
  return out;
}
export function buildMask(ids) {
  if (!ids.length) return null;
  let e = { op: 'use', id: ids[0] };
  for (let i = 1; i < ids.length; i++) e = { op: 'union', a: e, b: { op: 'use', id: ids[i] } };
  return e;
}
export const clone = (x) => JSON.parse(JSON.stringify(x));

/* ---------- wsp text highlight ---------- */
const STATEMENTS = new Set(['host', 'seed', 'mask', 'tv', 'graft', 'lora', 'heal']);
const OPTIONS = new Set(['at', 'rank', 'glue_rank', 'control', 'seed', 'steps', 'data', 'lr', 'batch_size', 'seq_len', 'weight', 'layers', 'modules', 'targets', 'budget', 'alpha', 'stitch', 'objective', 'sink', 'restrict']);
export function tokenizeLine(line) {
  const ci = line.indexOf('#');
  const code = ci >= 0 ? line.slice(0, ci) : line;
  const comment = ci >= 0 ? line.slice(ci) : '';
  const out = [];
  const re = /(\s+|[^\s]+)/g;
  let m, first = true;
  while ((m = re.exec(code))) {
    const t = m[0];
    if (/^\s+$/.test(t)) { out.push({ t, c: '' }); continue; }
    let c = 's';
    if (first && STATEMENTS.has(t)) c = 'k';
    else if (OPTIONS.has(t)) c = 'f';
    else if (/^-?[0-9][0-9.e\-]*(\.\.[0-9]+)?$/.test(t)) c = 'n';
    else if (/^[|&()\-]$/.test(t)) c = 'op';
    first = false;
    out.push({ t, c });
  }
  if (comment) out.push({ t: comment, c: 'c' });
  return out;
}

/* ---------- hash router ---------- */
export function parseHash() {
  const raw = location.hash.replace(/^#/, '') || '/';
  const [path, qs] = raw.split('?');
  const q = new URLSearchParams(qs || '');
  const parts = path.split('/').filter(Boolean);
  if (!parts.length) return { name: 'prompt', q };
  if (parts[0] === 'spec' && parts[1]) return { name: 'spec', id: parts[1], q };
  if (parts[0] === 'run' && parts[1]) {
    const id = parts[1];
    // the old single-agent route moves to the agent page
    if (parts[2] === 'agent' && parts[3]) {
      try { history.replaceState(null, '', '#/run/' + id + '/agents/' + parts[3]); } catch (e) { /* ignore */ }
      return { name: 'run', id, page: 'agent', agent: parts[3], q };
    }
    if (parts[2] === 'agents' && parts[3]) return { name: 'run', id, page: 'agent', agent: parts[3], q };
    if (parts[2] === 'agents') return { name: 'run', id, page: 'agents', q };
    if (parts[2] === 'activity') return { name: 'run', id, page: 'activity', q };
    if (parts[2] === 'results') return { name: 'run', id, page: 'results', q };
    return { name: 'run', id, page: 'overview', q };
  }
  if (parts[0] === 'build' && parts[1]) return { name: 'build', id: parts[1], q };
  if (parts[0] === 'shrink' && parts[1]) return { name: 'shrink', id: parts[1], q };
  if (parts[0] === 'runs') return { name: 'runs', q };
  if (parts[0] === 'parts') return { name: 'parts', q };
  if (parts[0] === 'demo') return { name: 'demo', q };
  return { name: 'prompt', q };
}
export function go(path) { location.hash = '#' + path; }
export function useRoute() {
  const [r, setR] = useState(parseHash());
  useEffect(() => {
    const f = () => setR(parseHash());
    window.addEventListener('hashchange', f);
    return () => window.removeEventListener('hashchange', f);
  }, []);
  return r;
}

/* ---------- small stores ---------- */
const healthListeners = new Set();
let healthInfo = null;
export function setHealth(h) { healthInfo = h || null; healthListeners.forEach((f) => f(healthInfo)); }
export function useHealth() {
  const [h, setH] = useState(healthInfo);
  useEffect(() => { healthListeners.add(setH); setH(healthInfo); return () => healthListeners.delete(setH); }, []);
  return h;
}

const stubListeners = new Set();
let stubFlag = false;
export function markStub(v) {
  if (DEMO) return;
  if (v && !stubFlag) { stubFlag = true; stubListeners.forEach((f) => f(true)); }
}
export function useStubFlag() {
  const [s, setS] = useState(stubFlag);
  useEffect(() => { stubListeners.add(setS); setS(stubFlag); return () => stubListeners.delete(setS); }, []);
  return s;
}

const toastListeners = new Set();
export function toast(text, kind = 'info') {
  toastListeners.forEach((f) => f({ id: Math.random(), text, kind }));
}
// The toast stack in components.js listens here and owns the timers and the exit motion.
export function subscribeToasts(fn) {
  toastListeners.add(fn);
  return () => toastListeners.delete(fn);
}

/* ---------- hooks ---------- */
export function useInterval(fn, ms) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (ms == null) return;
    const id = setInterval(() => ref.current(), ms);
    return () => clearInterval(id);
  }, [ms]);
}
export function useDebouncedCallback(fn, ms) {
  const ref = useRef(fn);
  ref.current = fn;
  const t = useRef(0);
  const call = (...a) => {
    clearTimeout(t.current);
    t.current = setTimeout(() => ref.current(...a), ms);
  };
  call.flush = (...a) => { clearTimeout(t.current); return ref.current(...a); };
  call.cancel = () => clearTimeout(t.current);
  return call;
}
// keeps the last good value of a number, so counters do not jump
export function usePrev(v) {
  const r = useRef(v);
  useEffect(() => { r.current = v; });
  return r.current;
}

/* ---------- theme ---------- */
const themeListeners = new Set();
export function currentTheme() { return document.documentElement.classList.contains('dark') ? 'dark' : 'light'; }
export function setTheme(t, save = true) {
  const el = document.documentElement;
  el.classList.remove('light', 'dark');
  el.classList.add(t);
  if (save) { try { localStorage.setItem('wsp.theme', t); } catch (e) { /* storage can be off */ } }
  themeListeners.forEach((f) => f(t));
}
export function useTheme() {
  const [t, setT] = useState(currentTheme());
  useEffect(() => {
    themeListeners.add(setT);
    let mq = null, f = null;
    let saved = null;
    try { saved = localStorage.getItem('wsp.theme'); } catch (e) { /* ignore */ }
    if (!saved && window.matchMedia) {
      mq = window.matchMedia('(prefers-color-scheme: dark)');
      f = (e) => setTheme(e.matches ? 'dark' : 'light', false);
      if (mq.addEventListener) mq.addEventListener('change', f);
    }
    return () => { themeListeners.delete(setT); if (mq && f && mq.removeEventListener) mq.removeEventListener('change', f); };
  }, []);
  return [t, () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark')];
}

/* ---------- recent specs, runs and the last builder ---------- */
const recentListeners = new Set();
export function getRecent() {
  try { const a = JSON.parse(localStorage.getItem(storageKey('wsp.recent')) || '[]'); return Array.isArray(a) ? a : []; } catch (e) { return []; }
}
export function addRecent(item) {
  const list = getRecent();
  const old = list.find((x) => x.kind === item.kind && x.id === item.id);
  const next = { ...(old || {}), ...item, ts: old && !item.touch ? old.ts : Date.now() };
  delete next.touch;
  const out = [next, ...list.filter((x) => !(x.kind === item.kind && x.id === item.id))].slice(0, 16);
  try { localStorage.setItem(storageKey('wsp.recent'), JSON.stringify(out)); } catch (e) { /* storage can be off */ }
  recentListeners.forEach((f) => f(out));
}
export function patchRecent(kind, id, patch) {
  const list = getRecent();
  const i = list.findIndex((x) => x.kind === kind && x.id === id);
  if (i < 0) return;
  list[i] = { ...list[i], ...patch };
  try { localStorage.setItem(storageKey('wsp.recent'), JSON.stringify(list)); } catch (e) { /* ignore */ }
  recentListeners.forEach((f) => f(list));
}
export function useRecent() {
  const [l, setL] = useState(getRecent());
  useEffect(() => { recentListeners.add(setL); setL(getRecent()); return () => recentListeners.delete(setL); }, []);
  return l;
}
export function getLastBuild() { try { return JSON.parse(localStorage.getItem(storageKey('wsp.lastBuild'))); } catch (e) { return null; } }
export function setLastBuild(v) { try { localStorage.setItem(storageKey('wsp.lastBuild'), JSON.stringify(v)); } catch (e) { /* ignore */ } }
