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

/* ---------- filters for the parts lists (library page and builder palette) ---------- */

export const KIND_CHIPS = [['block', 'Blocks'], ['attn_heads', 'Attention heads'], ['mlp_cluster', 'Neuron clusters'], ['expert', 'Experts'], ['inserted_layer', 'Inserted layers'], ['deletion_mask', 'Masks']];
export const STATUS_CHIPS = ['validated', 'candidate', 'rejected'];
export const NO_FILTER = { kind: null, status: null, placeable: false };
export const PAGE_SIZE = 60;

/** The kind of a palette item. Masks have no kind of their own, they sit in the "remove" group. */
export const kindOf = (x) => x.kind || (x.group === 'remove' ? 'deletion_mask' : null);

/** The lowercase text a search looks in. Built once per item. */
function hay(x) {
  if (x._hay === undefined) x._hay = [x.label, x.id, x.donor, x.summary, x.kind, (x.skills || []).join(' ')].join(' ').toLowerCase();
  return x._hay;
}

/** Items that pass the kind, status, placeable and search filters. Placeable items come first. */
export function filterItems(items, f, q) {
  const text = (q || '').trim().toLowerCase();
  const words = text ? text.split(/\s+/) : [];
  const keep = [];
  items.forEach((x, i) => {
    if (f.kind && kindOf(x) !== f.kind) return;
    if (f.status && x.status !== f.status) return;
    const locked = lockOf(x) ? 1 : 0;
    if (f.placeable && locked) return;
    if (words.length) { const h = hay(x); for (const w of words) if (!h.includes(w)) return; }
    const rank = KIND_CHIPS.findIndex((c) => c[0] === kindOf(x));
    keep.push([x, locked, rank < 0 ? 99 : rank, i]);
  });
  return keep.sort((a, b) => (a[1] - b[1]) || (a[2] - b[2]) || (a[3] - b[3])).map((p) => p[0]);
}

/** Counts per kind and per status over a list of items. */
export function countItems(items) {
  const kinds = {}, status = {};
  for (const x of items) {
    const k = kindOf(x);
    if (k) kinds[k] = (kinds[k] || 0) + 1;
    if (x.status) status[x.status] = (status[x.status] || 0) + 1;
  }
  return { kinds, status };
}

/** The chip rows: kind chips with counts, status chips, and the placeable toggle. */
export function FilterRow({ counts, f, setF, compact, noKinds }) {
  const chip = (on, label, n, onClick, key) => html`<button key=${key} class=${'fchip2' + (on ? ' on' : '')} aria-pressed=${on ? 'true' : 'false'} onClick=${onClick}>${label}${n != null && html`<span class="n">${n}</span>`}</button>`;
  const kindTotal = Object.values(counts.kinds).reduce((a, b) => a + b, 0);
  return html`<div class=${'fbar' + (compact ? ' compact' : '')}>
    ${!noKinds && html`<div class="frow2" role="group" aria-label="Kind">
      ${chip(!f.kind, 'All', kindTotal, () => setF({ ...f, kind: null }), 'all')}
      ${KIND_CHIPS.filter(([k]) => counts.kinds[k]).map(([k, name]) => chip(f.kind === k, name, counts.kinds[k], () => setF({ ...f, kind: f.kind === k ? null : k }), k))}
    </div>`}
    <div class="frow2" role="group" aria-label="Status">
      ${STATUS_CHIPS.map((s) => chip(f.status === s, s[0].toUpperCase() + s.slice(1), counts.status[s] || 0, () => setF({ ...f, status: f.status === s ? null : s }), s))}
      ${!noKinds && html`<button class=${'fchip2 tog' + (f.placeable ? ' on' : '')} role="switch" aria-checked=${f.placeable ? 'true' : 'false'} onClick=${() => setF({ ...f, placeable: !f.placeable })}><span class="knob"></span>Placeable only</button>`}
    </div>
  </div>`;
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
