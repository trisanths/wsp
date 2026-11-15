// Helpers and small parts shared by the run pages.
import { html, useState, useEffect } from '../lib.js';
import { api } from '../api.js';
import { pctS, primaryTarget } from '../util.js';
import { Dot } from '../components.js';
import { SubtleNum, ThinkLine } from '../motion.js';

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
// Once the run has ended, every agent gets the same grey dot.
export function agentDot(status, ended) {
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
export function AgentList({ snap }) {
  const ended = snap.status === 'done' || snap.status === 'error';
  return html`<div class="alist">${snap.agents.map((a) => html`<a class="arow" key=${a.id} href=${'#/run/' + snap.run_id + '/agents/' + a.id}>
    <${Dot} color=${agentDot(a.status, ended)} pulse=${!ended && !!ACTIVE[a.status]} />
    <span class="an">${a.name}</span>
    <span class="anow"><${ThinkLine} text=${a.now || STATUS_WORD[a.status] || a.status} still=${ended || !ACTIVE[a.status]} fluid /></span>
    <span class="asc num">${a.best != null ? html`<${SubtleNum} text=${pctS(a.best)} />` : '-'}</span>
  </a>`)}</div>`;
}
