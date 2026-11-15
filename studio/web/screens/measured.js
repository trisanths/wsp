import { html, useState, useEffect } from '../lib.js';
import { api } from '../api.js';
import { ModelLogo } from '../logos.js';

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

export function MeasuredTag() {
  return html`<span class="tag g mtag" title="Scored by the compiler on real weights">measured</span>`;
}

function Bar({ value, control }) {
  const w = Math.max(0, Math.min(100, value || 0));
  return html`<span class=${'mbar' + (control ? ' ctl' : '')} aria-hidden="true"><i style=${{ width: w + '%' }}></i></span>`;
}

function RowLabel({ row }) {
  return html`<span class="ml">${row.label}${row.is_control && html` <span class="tag mctl" title="A control: it picks experts by a rule that should not work, so it shows what chance looks like">control</span>`}</span>`;
}

// the table of one result: HumanEval against total size, with a bar
export function MeasuredTable({ result, mark }) {
  const rows = result.rows || [];
  return html`<div class="mtable" role="table" aria-label=${result.title}>
    <div class="mrow mhead" role="row"><span role="columnheader">Row</span><span role="columnheader" class="r">Size</span><span role="columnheader" class="r">HumanEval</span><span role="columnheader" class="mbarcol"></span><span role="columnheader" class="r">MBPP</span></div>
    ${rows.map((row) => html`<div class=${'mrow' + (mark && mark(row) ? ' on' : '')} role="row" key=${row.label}>
      <${RowLabel} row=${row} />
      <span class="r num" role="cell">${sizeText(row.total_params_b)}</span>
      <span class="r num" role="cell">${num(row.scores.humaneval)}</span>
      <${Bar} value=${row.scores.humaneval} control=${row.is_control} />
      <span class="r num" role="cell">${num(row.scores.mbpp)}</span>
    </div>`)}
  </div>`;
}

export function Caveats({ result }) {
  return html`<ul class="mcav">${(result.caveats || []).map((c) => html`<li key=${c}>${c}</li>`)}</ul>`;
}

function Meta({ result }) {
  return html`<div class="mmeta faint">${short(result.model)} · ${result.runs_per_cell} run per cell · heal: ${result.heal} · scored with ${result.scorer} · ${result.date}</div>`;
}

export function MeasuredCard({ result, onTry }) {
  return html`<div class="card mcard">
    <div class="mhd"><${ModelLogo} model=${result.model} size=${20} /><h3 class="mtitle">${result.title}</h3><${MeasuredTag} /></div>
    <${Meta} result=${result} />
    <${MeasuredTable} result=${result} />
    <${Caveats} result=${result} />
    ${onTry && html`<div class="mfoot"><span class="faint small">${result.source}</span><button class="btn sm" onClick=${() => onTry(TRY_TEXT)}>Try this request</button></div>`}
  </div>`;
}

// home screen: one calm card under the entry cards
export function MeasuredHome({ onTry }) {
  const list = useMeasured();
  if (!list || !list.length) return null;
  return html`<section class="mhome"><h2 class="mh">Measured so far</h2><${MeasuredCard} result=${list[0]} onTry=${onTry} /></section>`;
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
  const hit = result.rows.find(same);
  const other = plan.source_model !== result.model;
  return html`<section class="rsect mplan">
    <h2 class="sec"><span>Measured for this model</span><${MeasuredTag} /></h2>
    <div class="mpgrid">
      <div class="mpcol">
        <div class="mpt">Your plan <span class="tag">estimated</span></div>
        <div class="mpl">Keep ${kept} of ${result.experts_per_layer} experts per layer by ${RULE_NAMES[plan.rule] || plan.rule}</div>
        <div class="mpv num">${sizeText(est.total_params_b)} stored, ${sizeText(est.run_params_b)} per token</div>
        <div class="faint small">Sizes are estimates. The plan does not estimate scores.</div>
        ${hit ? html`<div class="mpmeas"><span class="tag g">measured</span> At ${kept} experts by ${RULE_NAMES[hit.rule]}: size ${sizeText(hit.total_params_b)}, HumanEval ${num(hit.scores.humaneval)}, MBPP ${num(hit.scores.mbpp)}.</div>`
          : html`<div class="faint small mpnone">No run at ${kept} experts by ${RULE_NAMES[plan.rule] || plan.rule}. The nearest measured rows are in the table.</div>`}
      </div>
      <div class="mpcol">
        <div class="mpt">Measured runs</div>
        <${MeasuredTable} result=${result} mark=${same} />
      </div>
    </div>
    <${Caveats} result=${result} />
    ${other && html`<div class="faint small">These runs used ${short(result.model)}. Your source is ${short(plan.source_model)}.</div>`}
  </section>`;
}
