// Helpers and small parts shared by the run pages.
import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { pctS, primaryTarget, recentStatus } from '../util.js';
import { Dot } from '../components.js';
import { SubtleNum, ThinkLine } from '../motion.js';

/* ---------- layout ---------- */
// true on a phone-width screen, so wide tables can become cards
export function useNarrow(px = 640) {
  const q = '(max-width: ' + px + 'px)';
  const get = () => typeof matchMedia === 'function' && matchMedia(q).matches;
  const [n, setN] = useState(get);
  useEffect(() => {
    if (typeof matchMedia !== 'function') return undefined;
    const m = matchMedia(q);
    const on = () => setN(m.matches);
    on();
    m.addEventListener ? m.addEventListener('change', on) : m.addListener(on);
    return () => (m.removeEventListener ? m.removeEventListener('change', on) : m.removeListener(on));
  }, [q]);
  return n;
}

/* ---------- names ---------- */
let catalog = null;
let catalogLoad = null;
// The parts catalog, loaded once. Used to build a readable name when the server did not send one.
export function useCatalog() {
  const [cat, setCat] = useState(catalog);
  useEffect(() => {
    if (catalog) return;
    if (!catalogLoad) catalogLoad = api.catalog().then((c) => { catalog = c; return c; }).catch(() => null);
    catalogLoad.then((c) => { if (c) setCat(c); });
  }, []);
  return cat;
}
// "Math block L8-10 at L5, heal 300" from a program, using catalog labels
export function progName(p, cat) {
  if (!p) return null;
  const bits = (p.grafts || []).map((g) => {
    const part = cat && cat.parts.find((x) => x.id === g.id);
    return (part ? part.label : 'Part') + ' at L' + g.at;
  });
  let masks = 0;
  (function walk(n) { if (!n) return; if (n.op === 'use') masks++; else if (n.op === 'restrict') walk(n.inner); else { walk(n.a); walk(n.b); } })(p.masks);
  if (masks) bits.push('remove ' + masks + (masks > 1 ? ' skills' : ' skill'));
  let s = bits.length ? bits.join(' + ') : 'Host only';
  if (p.lora) s += ', LoRA r' + (p.lora.rank || 16);
  if (p.heal) s += ', heal ' + p.heal.steps;
  return s;
}
// the readable name of a leaderboard row, best entry or history entry
export function nameOf(row, cat) {
  if (!row) return '';
  return row.name || progName(row.program, cat) || row.describe || '';
}

/* ---------- status ---------- */
export const ACTIVE = { thinking: 1, probing: 1, compiling: 1, scoring: 1 };
// A run that ended on its own with nothing scored. Runs from before the server sent outcome "empty" have no best.
export function isEmptyEnd(snap) {
  if (snap.status !== 'done' || snap.outcome === 'stopped' || snap.outcome === 'error' || snap.stopped_early === true) return false;
  return snap.outcome === 'empty' || !snap.best;
}
// the status word kept in Recent: "empty" for a finished run with nothing scored
export function runRecent(snap) {
  const s = recentStatus(snap);
  return s === 'done' && isEmptyEnd(snap) ? 'empty' : s;
}
// Once the run has ended, every agent gets the same grey dot, amber when nothing was scored.
export function agentDot(status, ended, empty) {
  if (ended && empty) return 'var(--amber)';
  if (ended) return 'var(--text-3)';
  if (ACTIVE[status]) return 'var(--blue)';
  if (status === 'done') return 'var(--text-3)';
  return 'var(--text-4)';
}
export const STATUS_WORD = { thinking: 'Thinking', probing: 'Checking fit', compiling: 'Compiling', scoring: 'Scoring', idle: 'Idle', paused: 'Paused', done: 'Done' };

// the score the run is aiming at, as a fraction, or null
export function bestTarget(snap) {
  const t = primaryTarget(snap.spec);
  if (!t || !snap.best) return null;
  const v = snap.best.scores[t.id];
  return v == null ? null : v;
}

/* ---------- agent list ---------- */
// A plain list: dot, name, what it is doing, best score. No cards.
// Dots that travel between the agents, drawn in a lane right after their names while a run is live.
// Each dot leaves one agent and arrives at another along a curve, to show the agents passing notes.
const SVGNS = 'http://www.w3.org/2000/svg';
function AgentFlow({ list, count }) {
  const svg = useRef(null);
  useEffect(() => {
    const el = svg.current, box = list.current;
    if (!el || !box) return undefined;
    let raf = 0, last = 0, rows = [], x0 = 0;
    const flights = [];
    const measure = () => {
      const top = box.getBoundingClientRect();
      const rs = [...box.querySelectorAll('.arow')];
      rows = rs.map((r) => { const b = r.getBoundingClientRect(); return b.top - top.top + b.height / 2; });
      x0 = Math.max(0, ...rs.map((r) => { const n = r.querySelector('.an'); if (!n) return 0; const g = document.createRange(); g.selectNodeContents(n); return g.getBoundingClientRect().right - top.left; })) + 8;
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(box);
    const spawn = () => {
      if (rows.length < 2) return;
      const i = Math.floor(Math.random() * rows.length);
      let j = Math.floor(Math.random() * (rows.length - 1));
      if (j >= i) j += 1;
      const y1 = rows[i], y2 = rows[j], bend = 10 + 7 * Math.abs(i - j);
      const d = `M ${x0} ${y1} Q ${x0 + bend * 2} ${(y1 + y2) / 2} ${x0} ${y2}`;
      const path = document.createElementNS(SVGNS, 'path');
      path.setAttribute('d', d);
      path.setAttribute('class', 'af-path');
      const dot = document.createElementNS(SVGNS, 'circle');
      dot.setAttribute('r', '2.6');
      dot.setAttribute('class', 'af-dot');
      el.append(path, dot);
      flights.push({ path, dot, len: path.getTotalLength(), t0: performance.now(), dur: 900 + 160 * Math.abs(i - j) });
    };
    const tick = (now) => {
      if (now - last > 420) { last = now; spawn(); }
      for (let k = flights.length - 1; k >= 0; k--) {
        const f = flights[k];
        const u = (now - f.t0) / f.dur;
        if (u >= 1) { f.path.remove(); f.dot.remove(); flights.splice(k, 1); continue; }
        const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        const p = f.path.getPointAtLength(e * f.len);
        f.dot.setAttribute('cx', p.x);
        f.dot.setAttribute('cy', p.y);
        const fade = Math.min(1, u * 6, (1 - u) * 6);
        f.dot.style.opacity = fade;
        f.path.style.opacity = 0.35 * fade;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); flights.forEach((f) => { f.path.remove(); f.dot.remove(); }); };
  }, [count]);
  return html`<svg class="aflow" ref=${svg} aria-hidden="true"></svg>`;
}

export function AgentList({ snap }) {
  const ended = snap.status === 'done' || snap.status === 'error';
  const empty = isEmptyEnd(snap);
  const list = useRef(null);
  return html`<div class=${'alist' + (ended ? '' : ' flowing')} ref=${list}>${!ended && html`<${AgentFlow} list=${list} count=${snap.agents.length} />`}${snap.agents.map((a) => html`<a class="arow" key=${a.id} href=${'#/run/' + snap.run_id + '/agents/' + a.id}>
    <${Dot} color=${agentDot(a.status, ended, empty)} pulse=${!ended && !!ACTIVE[a.status]} />
    <span class="an">${a.name}</span>
    <span class="anow"><${ThinkLine} text=${a.now || STATUS_WORD[a.status] || a.status} still=${ended || !ACTIVE[a.status]} fluid /></span>
    <span class="asc num">${a.best != null ? html`<${SubtleNum} text=${pctS(a.best)} />` : '-'}</span>
  </a>`)}</div>`;
}
