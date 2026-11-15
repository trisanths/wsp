// Small shared parts for the comparison set, part status and model logos.
import { html } from './lib.js';
import { sgn, pct, useHealth } from './util.js';
import { Icon, StubBadge } from './components.js';
import { ModelLogo } from './logos.js';

/** The models a program takes parts from (unique, in order), read through the catalog. */
export function donorsOf(program, cat) {
  const out = [];
  if (!program || !cat) return out;
  for (const g of program.grafts || []) {
    const part = cat.parts.find((x) => x.id === g.id);
    if (part && part.donor && !out.includes(part.donor)) out.push(part.donor);
  }
  return out;
}

/** A row of model logos, at most `max`, then "+N". */
export function Logos({ models, max = 3, size = 18 }) {
  if (!models || !models.length) return null;
  const shown = models.slice(0, max);
  const more = models.length - shown.length;
  return html`<span class="mlogos">${shown.map((m) => html`<${ModelLogo} model=${m} size=${size} alt="" />`)}${more > 0 && html`<span class="mlogo-more">+${more}</span>`}</span>`;
}

/** The one-line verdict of a comparison, and its colour class. */
export function verdictOf(c) {
  if (!c || !c.verdict || c.verdict === 'none') return null;
  const label = (r) => r.label;
  const rows = c.rows || [];
  const named = (ids) => rows.filter((r) => (ids || []).includes(r.id)).map(label);
  let text;
  if (c.verdict === 'success' || c.verdict === 'beats') text = 'Beats every reference';
  else if (c.verdict === 'failed') {
    const names = rows.filter((r) => r.measured && r.beaten === false).map(label);
    text = names.length ? 'Does not beat ' + names.join(', ') : 'Failed attempt';
  } else {
    const names = rows.filter((r) => !r.measured).map(label);
    text = 'Not proven yet' + (names.length ? ': ' + names.join(', ') + ' not measured' : '');
  }
  const kind = c.verdict === 'success' || c.verdict === 'beats' ? 'up' : c.verdict === 'failed' ? 'down' : 'faint';
  return { text: c.headline || text, kind, stub: !!c.stub };
}

/** The "Compared with" list: one row per reference. */
export function Compare({ c, compact, heal }) {
  if (!c || !c.rows || !c.rows.length) return null;
  const v = verdictOf(c);
  return html`<div class=${'cmp' + (compact ? ' compact' : '')}>
    <${HealNote} show=${!!heal} />
    ${v && html`<div class=${'cmp-verdict ' + v.kind}>${v.text}</div>`}
    <div class="cmp-list">${c.rows.map((r) => html`<div class="cmp-row" key=${r.id} title=${r.note || null}>
      <span class="cl">${r.label}${r.stub && html` <${StubBadge} small />`}</span>
      <span class="cv num">${r.measured && r.value != null ? pct(r.value) + '%' : ''}</span>
      <span class=${'cd num ' + (!r.measured || r.delta == null ? 'faint' : r.beaten == null ? 'faint' : r.beaten ? 'up' : 'down')}>${!r.measured || r.delta == null ? 'not measured' : r.beaten == null ? html`${sgn(r.delta)} <span class="cref">reference</span>` : html`${sgn(r.delta)} <${Icon} n=${r.beaten ? 'check' : 'x'} size=${13} sw=${2.25} />`}</span>
    </div>`)}</div>
  </div>`;
}

/** A part's status as a small tag. */
export function StatusTag({ status }) {
  if (!status) return null;
  const k = status === 'validated' ? 'g' : status === 'rejected' ? 'r' : 'c';
  return html`<span class=${'pst ' + k}>${status}</span>`;
}

export const KIND_NAMES = { block: 'Blocks', attn_heads: 'Attention heads', mlp_cluster: 'Neuron clusters', expert: 'Experts', inserted_layer: 'Inserted layers' };
export const KIND_ORDER = ['block', 'attn_heads', 'mlp_cluster', 'expert', 'inserted_layer'];

/** A part that cannot go in a program, with the reason to show. */
export function lockOf(p) {
  if (p.status === 'rejected') return { why: 'Rejected. ' + (p.evidence || 'It lost to its control.'), rejected: true };
  if (p.locked) return { why: typeof p.locked === 'string' ? p.locked : p.usable_reason || 'Locked' };
  if (p.usable === false) return { why: p.usable_reason || 'Not usable yet' };
  return null;
}

/** Parts grouped by kind, in a fixed order, usable ones first. Parts without a kind go in one group. */
export function groupByKind(parts) {
  const groups = [];
  const kinds = [...KIND_ORDER, ...parts.map((p) => p.kind).filter((k) => k && !KIND_ORDER.includes(k))];
  for (const k of [...new Set(kinds)]) {
    const items = parts.filter((p) => p.kind === k);
    if (!items.length) continue;
    items.sort((a, b) => (lockOf(a) ? 1 : 0) - (lockOf(b) ? 1 : 0));
    groups.push({ kind: k, name: KIND_NAMES[k] || k, items });
  }
  return groups;
}

/** True when a program trains the model, so its score depends on the heal recipe. */
export const usesHeal = (p) => !!(p && (p.heal || p.lora));

/** The server's heal warning, or null when it sends none. */
export function useHealNote() {
  const h = useHealth();
  return h && typeof h.heal_note === 'string' && h.heal_note ? h.heal_note : null;
}

/** One quiet warning line next to a score that used a heal. With `mark`, only the small icon with the text as a tip. */
export function HealNote({ show = true, mark }) {
  const note = useHealNote();
  if (!show || !note) return null;
  if (mark) return html`<span class="healmark" data-tip=${note} data-tip-side="bottom" aria-label=${note}><${Icon} n="warn" size=${13} /></span>`;
  return html`<div class="healnote"><${Icon} n="warn" size=${13} /><span>${note}</span></div>`;
}
