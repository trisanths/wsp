import { useState, useEffect, useRef } from './lib.js';

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

/* ---------- colors ---------- */
const PALETTE = ['#b38bff', '#4fd1ff', '#ffb547', '#3ddc97', '#ff8fb1', '#7c8cff', '#f2e35c', '#8bd3ff'];
function hash(s) {
  let x = 7;
  for (let i = 0; i < s.length; i++) x = (x * 31 + s.charCodeAt(i)) >>> 0;
  return x;
}
export function agentColor(id) {
  if (id === 'analyst') return '#7c8cff';
  if (id === 'judge') return '#ff6b7a';
  const m = /^planner-(\d+)$/.exec(id || '');
  if (m) return ['#b38bff', '#4fd1ff', '#ffb547', '#3ddc97', '#ff8fb1', '#f2e35c'][(Number(m[1]) - 1) % 6];
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
  if (!donor) return '#8b93a7';
  return PALETTE[hash(donor.split('/').pop().toLowerCase().replace(/[^a-z]/g, '').slice(0, 5)) % PALETTE.length];
}
export function donorLetter(donor, label) {
  const name = (donor || label || '?').split('/').pop();
  return name.charAt(0).toUpperCase();
}
export const STATUS_COLOR = {
  thinking: 'var(--cyan)', probing: 'var(--warn)', compiling: 'var(--warn)', scoring: 'var(--good)',
  idle: 'var(--faint)', paused: 'var(--faint)', done: 'var(--faint)',
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
export function specProgress(spec, scores) {
  const bs = spec.benchmarks || [];
  if (!bs.length || !scores) return 0;
  let sum = 0;
  for (const b of bs) sum += benchProgress(b, scores[b.id]);
  return sum / bs.length;
}

/* ---------- program helpers ---------- */
export function emptyProgram(host) {
  return { host, seed: 0, masks: null, task_vectors: [], grafts: [], heal: null, lora: null, taps: [], tap_train: null };
}
// older saved programs and server replies may lack the tap fields
export function normProgram(p) {
  if (!p) return p;
  return { ...p, taps: p.taps || [], tap_train: p.tap_train == null ? null : p.tap_train };
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
const STATEMENTS = new Set(['host', 'seed', 'mask', 'tv', 'graft', 'lora', 'heal', 'tap', 'tap_train']);
const OPTIONS = new Set(['at', 'rank', 'glue_rank', 'control', 'seed', 'steps', 'data', 'lr', 'batch_size', 'seq_len', 'weight', 'layers', 'modules', 'targets', 'budget', 'alpha', 'stitch', 'objective', 'sink', 'restrict', 'layer', 'attn_dim', 'heads', 'branch_lr', 'gate_lr']);
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
    if (parts[2] === 'agent' && parts[3]) return { name: 'run', id: parts[1], agent: parts[3], q };
    return { name: 'run', id: parts[1], q };
  }
  if (parts[0] === 'build' && parts[1]) return { name: 'build', id: parts[1], q };
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
export function useToasts() {
  const [list, setList] = useState([]);
  useEffect(() => {
    const f = (t) => {
      setList((l) => [...l, t]);
      setTimeout(() => setList((l) => l.filter((x) => x.id !== t.id)), 3800);
    };
    toastListeners.add(f);
    return () => toastListeners.delete(f);
  }, []);
  return list;
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
