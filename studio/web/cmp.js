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
export function Compare({ c, compact, heal, note }) {
  if (!c || !c.rows || !c.rows.length) return null;
  const v = verdictOf(c);
  return html`<div class=${'cmp' + (compact ? ' compact' : '')}>
    <${HealNote} show=${!!heal} note=${note} />
    ${v && html`<div class=${'cmp-verdict ' + v.kind}>${v.text}</div>`}
    <div class="cmp-list">${c.rows.map((r) => html`<div class="cmp-row" key=${r.id} title=${r.note || null}>
      <span class="cl">${r.label}${r.stub && html` <${StubBadge} small />`}</span>
      <span class="cv num">${r.measured && r.value != null ? pct(r.value) + '%' : ''}</span>
      <span class=${'cd num ' + (!r.measured || r.delta == null ? 'faint' : r.beaten == null ? 'faint' : r.beaten ? 'up' : r.within_margin === true ? 'warn' : 'down')}>${!r.measured || r.delta == null ? 'not measured' : r.beaten == null ? html`${sgn(r.delta)} <span class="cref">reference</span>` : r.beaten === false && r.within_margin === true ? html`${sgn(r.delta)} <span class="wmark">within margin</span>` : html`${sgn(r.delta)} <${Icon} n=${r.beaten ? 'check' : 'x'} size=${13} sw=${2.25} />`}</span>
    </div>`)}</div>
    <div class="cmp-rule">A result must beat each reference by ${c.margin != null && c.margin > 0 ? Math.round(c.margin * 1000) / 10 : 2} points.</div>
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

/** The heal note for one result. A synth result has none. A classic builder result carries its own `heal_note`.
 *  Returns undefined when the result does not say (older servers, swarm entries), so the health note is used. */
export function healNoteOf(r) {
  if (!r) return undefined;
  if (r.pipeline === 'synth') return null;
  if (r.pipeline === 'classic' && 'heal_note' in r) return r.heal_note || null;
  return undefined;
}

/** One quiet warning line next to a score that used a heal. With `mark`, only the small icon with the text as a tip. */
export function HealNote({ show = true, mark, note: own }) {
  const health = useHealNote();
  const note = own === undefined ? health : own;
  if (!show || !note) return null;
  if (mark) return html`<span class="healmark" data-tip=${note} data-tip-side="bottom" aria-label=${note}><${Icon} n="warn" size=${13} /></span>`;
  return html`<div class="healnote"><${Icon} n="warn" size=${13} /><span>${note}</span></div>`;
}


/* ---------- component verdicts (the synthesis pipeline) ---------- */
const VERDICT_KIND = { contributes: 'g', 'no better than random': 'n', harmful: 'r' };
const fpts = (v) => (v >= 0 ? '+' : '-') + Math.abs(v * 100).toFixed(1) + ' points';

/** How many components contribute, as { n, of }, or null when there are none. */
export function partsCount(components) {
  if (!components || !components.length) return null;
  return { n: components.filter((c) => c.verdict === 'contributes').length, of: components.length };
}

/** The one number shown beside a component: transfer first, else the practice change, else the gain over a random part. */
export function mainNumber(c) {
  const d = (c && c.detail) || {};
  if (d.transfer != null) return Math.round(d.transfer * 100) + '% transfer';
  if (d.contribution != null) return fpts(d.contribution) + ' practice';
  if (d.content_gain != null) return fpts(d.content_gain) + ' vs random';
  if (d.text_gain != null) return d.text_gain.toFixed(3) + ' text gain';
  return '';
}

/** Every number a component has, one per line, for a tooltip. */
export function partTip(c) {
  const d = (c && c.detail) || {};
  const out = [c.label + ': ' + c.verdict + (c.stub ? ' (stub, not measured)' : '')];
  if (d.transfer != null) out.push('Transfer: ' + Math.round(d.transfer * 100) + '% of the gap from the host to the donor on held-out skill text');
  if (d.contribution != null) out.push('Practice change with the part on: ' + fpts(d.contribution));
  if (d.content_gain != null) out.push('Gain over a random part of the same size: ' + fpts(d.content_gain));
  if (d.text_gain != null) out.push('Text loss gain: ' + d.text_gain.toFixed(3));
  if (d.fit_r2 != null) out.push('Fit R squared: ' + d.fit_r2.toFixed(2));
  if (d.gate_domain != null && d.gate_general != null) out.push('Gate on skill text ' + d.gate_domain.toFixed(2) + ', on general text ' + d.gate_general.toFixed(2));
  if (d.part_params != null) out.push('Part size: ' + (d.part_params / 1e6).toFixed(2) + 'M parameters');
  if (out.length === 1) out.push('No numbers for this part.');
  return out.join('\n');
}

/** A small verdict tag. Only "contributes" is good. */
export function VerdictTag({ c }) {
  if (!c) return null;
  const k = VERDICT_KIND[c.verdict] || 'x';
  return html`<span class=${'vtag ' + k} title=${partTip(c)}>${c.verdict}</span>`;
}

/** The component of a program's graft: same part id, and the same host layer when the result names it. */
export function componentFor(components, g) {
  if (!components || !g) return null;
  return components.find((c) => c.id === g.id && (c.host_layer == null || c.host_layer === g.at)) || null;
}

/** The "Parts" list: one row per component. */
export function PartsList({ components, compact }) {
  if (!components || !components.length) return null;
  return html`<div class=${'parts' + (compact ? ' compact' : '')}>${components.map((c, i) => html`<div class="part-row" key=${c.id + '@' + i} title=${partTip(c)}>
    <${ModelLogo} model=${c.donor || ''} size=${16} alt="" />
    <span class="pl">${c.label}</span>
    <span class="pmeta"><${VerdictTag} c=${c} />${mainNumber(c) && html`<span class="pn2 num faint">${mainNumber(c)}</span>`}</span>
  </div>`)}${components.some((c) => c.stub) && html`<div class="part-stub faint"><${StubBadge} small /> Made-up verdicts, not measured.</div>`}</div>`;
}

const STAGE_NAMES = { place: 'Place', 'translate in': 'Translate in', 'translate out': 'Translate out', 'heal 1': 'Heal 1', 'heal 2': 'Heal 2', verify: 'Verify' };
const meanOf = (o) => {
  const v = Object.values(o || {}).filter((x) => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
const secs = (s) => (s == null ? '' : s >= 10 ? Math.round(s) + 's' : s.toFixed(1) + 's');

/** The drift from the base, in plain words. */
export function driftText(d) {
  if (!d) return null;
  const bits = [];
  if (d.agree != null) bits.push('The build picks the same top word as the base on ' + Math.round(d.agree * 100) + '% of tokens');
  if (d.base_kl != null) bits.push('KL from the base is ' + d.base_kl.toFixed(3) + ' (0 means the base is kept)');
  return bits.length ? bits.join('. ') + '.' : null;
}

/** The "Stages" disclosure: each pipeline stage, then the drift. */
export function Stages({ stages, drift, stub }) {
  if (!stages || !stages.length) return null;
  const dt = driftText(drift);
  return html`<details class="stages"><summary><span>Stages</span><span class="faint">${stages.length}</span></summary>
    <div class="st-list">${stages.map((s) => {
      const a = meanOf(s.practice_before), b = meanOf(s.practice_after);
      return html`<div class="st-row" key=${s.name} title=${s.detail || null}>
        <span class="sn">${STAGE_NAMES[s.name] || s.name}${s.rolled_back && html`<span class="sr">rolled back</span>`}</span>
        <span class="ss num faint">${secs(s.seconds)}</span>
        <span class="sp num">${a != null && b != null ? html`${pct(a)} to ${pct(b)}` : a != null ? pct(a) : b != null ? pct(b) : ''}</span>
      </div>`;
    })}</div>
    <div class="st-note faint">Practice score before and after each stage, in percent.</div>
    ${dt && html`<div class="st-drift">${dt}${drift.text ? html` <span class="faint">(${drift.text})</span>` : ''}${stub && html` <${StubBadge} small />`}</div>`}
  </details>`;
}
