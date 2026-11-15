import { html, useState, useEffect } from '../lib.js';
import { api } from '../api.js';
import { ModelLogo } from '../logos.js';
import { Icon } from '../components.js';

// Measured results: real scores from the compiler, shown with a small "measured" tag (never the STUB badge).
// Scores from the server are percent points (0 to 100).

export const TRY_TEXT = 'A coding-only model: keep the code experts of Qwen3-30B-A3B and get it under 17B';
const RULE_NAMES = { saliency: 'saliency', freq: 'routing frequency', random: 'random', selective: 'skill-to-general ratio', none: 'none' };

const short = (model) => String(model || '').split('/').pop();
const num = (x, d = 1) => (x == null ? '-' : Number(x).toFixed(d));
const sizeText = (b) => (b == null ? '-' : Number(b).toFixed(b < 10 ? 2 : 1) + 'B');

// one fetch for the whole page load, shared by every card
let cache = null;
function loadMeasured() {
  if (!cache) cache = api.measured().catch((e) => { cache = null; throw e; });
  return cache;
}

function useMeasured() {
  const [list, setList] = useState(null);
  useEffect(() => {
    let alive = true;
    loadMeasured().then((r) => alive && setList(Array.isArray(r) ? r : [])).catch(() => alive && setList([]));
    return () => { alive = false; };
  }, []);
  return list;
}

/** One measured fine-tune after a cut, as a sentence that gives heal steps a scale, or null before the data loads. */
export function useHealExample() {
  const list = useMeasured();
  for (const r of list || []) {
    const healed = (r.rows || []).find((x) => x.heal_steps && !x.is_control);
    const raw = healed && r.rows.find((x) => !x.heal_steps && !x.is_control && x.rule === healed.rule && x.experts_kept === healed.experts_kept);
    if (raw) return 'Measured: ' + healed.heal_steps + ' steps took a ' + healed.experts_kept + '-expert cut of ' + short(r.model).replace(/-Instruct.*$/, '') + ' from ' + num(raw.scores.humaneval) + ' to ' + num(healed.scores.humaneval) + ' on HumanEval.';
  }
  return null;
}

export function MeasuredTag() {
  return html`<span class="tag g mtag" title="Scored by the compiler on real weights">measured</span>`;
}

function Bar({ value, control }) {
  const w = Math.max(0, Math.min(100, value || 0));
  return html`<span class=${'mbar' + (control ? ' ctl' : '')} aria-hidden="true"><i style=${{ width: w + '%' }}></i></span>`;
}

// plain names for the rows; the compiler's own label stays as the hover title
function plainLabel(row, result) {
  const skill = result.skill || 'the skill';
  if (result.columns) return row.label;
  if (row.is_full) return 'Full model, all ' + (result.experts_per_layer || row.experts_kept) + ' experts';
  const what = { saliency: 'experts chosen for ' + skill, freq: 'most-used experts', random: 'random experts', selective: 'experts used more on ' + skill + ' than elsewhere' }[row.rule];
  if (!what) return row.label;
  return 'Keep ' + row.experts_kept + ' ' + what + (row.heal_steps ? ', then fine-tuned' : '');
}

// each control sits right under the row it checks: same experts kept, same fine-tune
function ordered(rows) {
  return rows.map((r, i) => [r, i]).sort(([a, i], [b, j]) => (b.is_full - a.is_full) || (b.experts_kept - a.experts_kept)
    || ((a.heal_steps || 0) - (b.heal_steps || 0)) || (a.is_control - b.is_control) || (i - j)).map(([r]) => r);
}

function RowLabel({ row, result }) {
  return html`<span class="ml" title=${row.label}>${plainLabel(row, result)}${row.is_control && html` <span class="tag mctl" title="A check: the same setup without the method. A real result has to beat it.">control</span>`}</span>`;
}

// the table of one result: HumanEval against total size, with a bar
export function MeasuredTable({ result, mark, only }) {
  const cols = result.columns;
  if (cols) {
    const rowsL = (result.rows || []).filter((row) => !only || only(row));
    const best = (id) => Math.max(...rowsL.map((r) => r.scores[id] ?? 0));
    return html`<div class="mtable mlang" role="table" aria-label=${result.title} style=${{ '--mcols': cols.length }}>
      <div class="mrow mhead" role="row"><span role="columnheader">Model</span><span role="columnheader" class="r">Size per token</span>${cols.map((c) => html`<span role="columnheader" class="r" key=${c.id}>${c.label}</span>`)}</div>
      ${rowsL.map((row) => html`<div class=${'mrow' + (mark && mark(row) ? ' on' : '') + (row.is_control ? ' mrow-ctl' : '')} role="row" key=${row.label}>
        <${RowLabel} row=${row} result=${result} />
        <span class="r num" role="cell">${sizeText(row.params_per_token_b)}</span>
        ${cols.map((c) => html`<span class="r num" role="cell" key=${c.id}>${num(row.scores[c.id])}</span>`)}
      </div>`)}
    </div>`;
  }
  const rows = ordered((result.rows || []).filter((row) => !only || only(row)));
  return html`<div class="mtable" role="table" aria-label=${result.title}>
    <div class="mrow mhead" role="row"><span role="columnheader">Model</span><span role="columnheader" class="r">Size</span><span role="columnheader" class="r" title="HumanEval: Python coding problems, percent solved">HumanEval</span><span role="columnheader" class="mbarcol"></span><span role="columnheader" class="r" title="MBPP: short Python problems, percent solved">MBPP</span></div>
    ${rows.map((row) => html`<div class=${'mrow' + (mark && mark(row) ? ' on' : '') + (row.is_control ? ' mrow-ctl' : '')} role="row" key=${row.label}>
      <${RowLabel} row=${row} result=${result} />
      <span class="r num" role="cell">${sizeText(row.total_params_b)}</span>
      <span class="r num" role="cell">${num(row.scores.humaneval)}</span>
      <${Bar} value=${row.scores.humaneval} control=${row.is_control} />
      <span class="r num" role="cell">${num(row.scores.mbpp)}</span>
    </div>`)}
  </div>`;
}

// the notes, as one footnote under the card
export function Caveats({ result }) {
  const notes = result.caveats || [];
  if (!notes.length) return null;
  return html`<p class="mnote"><span class="mnote-k">Notes.</span> ${notes.join(' ')}</p>`;
}

function Meta({ result }) {
  return html`<div class="mmeta faint">${short(result.model)} · ${result.date}</div>`;
}

export function MeasuredCard({ result, onTry, only }) {
  return html`<div class="card mcard">
    <div class="mhd"><${ModelLogo} model=${result.model} size=${20} /><h3 class="mtitle">${result.title}</h3><${MeasuredTag} /></div>
    <${Meta} result=${result} />
    <${MeasuredTable} result=${result} only=${only} />
    ${onTry && html`<div class="mfoot"><span class="faint small">Source: ${result.source}, scored with ${result.scorer}</span><button class="btn sm" onClick=${() => onTry(result.try_text || TRY_TEXT)}>Try this request</button></div>`}
    <${Caveats} result=${result} />
  </div>`;
}

// the home card shows the method against its random controls; the Results page keeps every row
const homeRow = (row) => row.is_full || row.rule === 'saliency' || row.rule === 'random';

// home screen: one card under the entry cards, collapsed until opened
export function MeasuredHome({ onTry }) {
  const list = useMeasured();
  const [open, setOpen] = useState(false);
  if (!list || !list.length) return null;
  const result = list[0];
  return html`<section class="mhome">
    <h2 class="mh"><button class="mtoggle" aria-expanded=${open ? 'true' : 'false'} onClick=${() => setOpen((o) => !o)}>
      <span class="mtk">Measured so far</span><span class="mtsub">${result.title}</span><span class=${'mchev' + (open ? ' open' : '')}><${Icon} n="chevron" size=${14} /></span>
    </button></h2>
    ${open && html`<${MeasuredCard} result=${result} onTry=${onTry} only=${homeRow} />`}
  </section>`;
}

// parts library page: a Results section
export function MeasuredResults() {
  const list = useMeasured();
  if (!list || !list.length) return null;
  return html`<section class="sect" id="results"><h2 class="sec"><span>Results</span><span class="faint cnt">${list.length}</span></h2>
    ${list.map((r) => html`<${MeasuredCard} key=${r.id} result=${r} />`)}</section>`;
}

// shrink page: the measured rows beside the plan's estimate, for the model and skill that were measured
export function MeasuredForPlan({ plan }) {
  const list = useMeasured();
  if (!list || !plan) return null;
  const result = list.find((r) => r.skill === plan.keep_skill && short(plan.source_model).startsWith('Qwen3-30B-A3B') && short(r.model).startsWith('Qwen3-30B-A3B'));
  if (!result || plan.method !== 'experts') return null;
  const est = plan.estimate || {};
  const kept = plan.keep_experts;
  const same = (row) => row.experts_kept === kept && row.rule === plan.rule;
  // among the rows that match, prefer the one healed like the plan (healed when the plan heals, raw when it does not)
  const planHeals = (plan.heal_steps || 0) > 0;
  const matches = result.rows.filter(same);
  const hit = matches.find((row) => ((row.heal_steps || 0) > 0) === planHeals) || matches[0];
  const healText = (row) => (row.heal_steps ? 'with a ' + row.heal_steps + '-step heal' : 'with no heal');
  const other = plan.source_model !== result.model;
  return html`<section class="rsect mplan">
    <h2 class="sec"><span>Measured for this model</span><${MeasuredTag} /></h2>
    <div class="mpgrid">
      <div class="mpcol">
        <div class="mpt">Your plan <span class="tag">estimated</span></div>
        <div class="mpl">Keep ${kept} of ${result.experts_per_layer} experts per layer by ${RULE_NAMES[plan.rule] || plan.rule}</div>
        <div class="mpv num">${sizeText(est.total_params_b)} stored, ${sizeText(est.run_params_b)} per token</div>
        <div class="faint small">Sizes are estimates. The plan does not estimate scores.</div>
        ${hit ? html`<div class="mpmeas"><span class="tag g">measured</span> At ${kept} experts by ${RULE_NAMES[hit.rule]} ${healText(hit)}: size ${sizeText(hit.total_params_b)}, HumanEval ${num(hit.scores.humaneval)}, MBPP ${num(hit.scores.mbpp)}.</div>`
          : html`<div class="faint small mpnone">No run at ${kept} experts by ${RULE_NAMES[plan.rule] || plan.rule}. The nearest measured rows are in the table.</div>`}
      </div>
      <div class="mpcol">
        <div class="mpt">Measured runs</div>
        <${MeasuredTable} result=${result} mark=${(row) => row === hit} />
      </div>
    </div>
    <${Caveats} result=${result} />
    ${other && html`<div class="faint small">These runs used ${short(result.model)}. Your source is ${short(plan.source_model)}.</div>`}
  </section>`;
}
