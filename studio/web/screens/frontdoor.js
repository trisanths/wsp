import { html, useState, useEffect, useMemo } from '../lib.js';
import { api } from '../api.js';
import { toast } from '../util.js';
import { Icon, Spinner } from '../components.js';

// The compiler front door on the spec page: turn the spec into a request, ask the compiler for a plan.

const RELATION = {
  same: 'is the same model as the base',
  fine_tune: 'is a fine-tune of the base',
  fine_tune_of_ancestor: 'is a fine-tune of an ancestor of the base',
  diverged_same_shape: 'has the same shape as the base but different weights',
  same_family_other_size: 'is the same family at another size',
  moe_same_width: 'is a mixture of experts with the same width as the base',
  foreign: 'is a different family',
};
const METHOD = {
  weave: 'a weave of its weights into the base',
  delta: 'a low-rank delta added to the base',
  synth: 'a seam with translation layers',
  feedback: 'a block whose output is written into the next token',
  transplant: 'a transplant of experts',
  slim: 'removing experts or layers',
  whole_model: 'routing to the whole source model',
};
const ARM = { base: 'the base alone', base_lora: 'base plus a LoRA on the same text', base_distill: 'base plus distillation from the source', best_existing: 'the best existing model' };

const modelName = (m) => String(m || '').split('/').pop();
const gb = (x) => (x == null ? '-' : Number(x).toFixed(x < 10 ? 2 : 1) + 'B');

function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text).catch(() => fallbackCopy(text));
  }
  return Promise.resolve(fallbackCopy(text));
}
function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
  document.body.removeChild(ta);
  if (!ok) throw new Error('Copy is blocked in this browser. Use Download instead.');
}

function PlanView({ out }) {
  const p = out.plan;
  const sources = p.sources || [];
  const removes = p.remove || [];
  const stages = p.stages || [];
  const arms = p.arms || [];
  const b = p.base || {};
  return html`<div class="fdplan">
    <p class="fdsrc">Base: ${modelName(b.id)}, ${gb(b.params_b)} total, ${gb(b.active_params_b)} per token.</p>
    ${sources.map((s) => {
      const chosen = (s.methods || []).find((m) => m.method === s.chosen);
      return html`<p class="fdsrc" key=${s.skill + s.source}>For ${s.skill}: ${modelName(s.source)} ${RELATION[s.relation] || s.relation}.${s.chosen ? ' Method: ' + (METHOD[s.chosen] || s.chosen) + '.' : ' No method fits.'}${chosen && chosen.adds_total_b != null ? ' It adds about ' + gb(chosen.adds_total_b) + '.' : ''}</p>`;
    })}
    ${removes.length > 0 && html`<p class="fdsrc">${removes.map((r) => r.skill === '*' ? 'Remove every skill except ' + ((p.keep || []).join(', ') || 'none') : 'Remove ' + r.skill).join('. ')}.</p>`}
    ${stages.length > 0 && html`<ol class="fdstages">${stages.map((st) => html`<li key=${st.order}><span class="mono">${st.method}</span><span>${gb(st.params_total_after_b)} total, ${gb(st.params_run_after_b)} per token after this stage</span>${(st.limit_problems || []).map((x) => html`<span class="down" key=${x}>over a limit: ${x}</span>`)}</li>`)}</ol>`}
    ${(p.refusals || []).map((r) => html`<div class="prob" key=${r}><${Icon} n="warn" size=${14} /><span>${r}</span></div>`)}
    ${(p.warnings || []).length > 0 && html`<ul class="fdwarn">${p.warnings.map((w) => html`<li key=${w}>${w}</li>`)}</ul>`}
    ${arms.length > 0 && html`<p class="fdarms faint">Compared against: ${arms.map((a) => ARM[a.arm] || a.arm).join(', ')}.</p>`}
    <p class="fdnote faint small">${out.note} Planned in ${out.seconds} s${out.offline ? ' from the offline facts table, because the Hugging Face Hub could not be read' + (out.offline_reason ? ' (' + out.offline_reason + ')' : '') : ' from the Hugging Face Hub'}. ${p.feasible ? 'Feasible.' : 'Not feasible.'}</p>
  </div>`;
}

export function CompilerPlan({ spec }) {
  const [view, setView] = useState(null);
  const [out, setOut] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const id = spec.id;
  // reload the request a moment after the spec changes (the spec saves on a short delay)
  const dep = JSON.stringify([spec.title, spec.host, spec.limits, spec.size && spec.size.max_params_b, (spec.benchmarks || []).map((b) => b.id), spec.shrink && [spec.shrink.source_model, spec.shrink.keep_skill]]);
  useEffect(() => {
    let alive = true;
    const t = setTimeout(() => { api.specRequest(id).then((v) => alive && setView(v)).catch((e) => alive && setErr(e.message)); }, view ? 1500 : 0);
    return () => { alive = false; clearTimeout(t); };
  }, [id, dep]);

  const json = useMemo(() => (view ? JSON.stringify(view.request, null, 2) : ''), [view]);
  const name = view ? view.request.name : 'request';
  const href = useMemo(() => 'data:application/json;charset=utf-8,' + encodeURIComponent(json), [json]);
  const plan = async () => {
    setBusy(true);
    setErr(null);
    try { setOut(await api.specPlan(id)); } catch (e) { setErr(e.message); setOut(null); }
    setBusy(false);
  };
  const copy = async () => {
    try { await copyText(json); toast('Request copied'); } catch (e) { toast(e.message, 'err'); }
  };
  const errors = (view && view.errors) || [];
  return html`<section class="sect fd"><h2 class="sec"><span>Compiler plan</span><span class="tag">draft</span></h2>
    <p class="faint small fdlede">The compiler reads how each source model relates to the base and picks a method. It reads model files from the Hugging Face Hub. No GPU, no weights.</p>
    <div class="fdbtns">
      <button class="btn sm" onClick=${plan} disabled=${busy || !view || errors.length > 0}>${busy ? html`<${Spinner} size=${12} />` : html`<${Icon} n="play" size=${12} />`} Plan with the compiler</button>
      <button class="btn sm" onClick=${copy} disabled=${!view}>Copy request JSON</button>
      <a class=${'btn sm' + (view ? '' : ' off')} href=${view ? href : null} download=${name + '.json'}>Download</a>
    </div>
    ${busy && html`<p class="faint small fdwait">Reading model files from the Hugging Face Hub. A large model can take about a minute.</p>`}
    <div class="fdcmd mono">sapien plan ${name}.json <span class="faint">then</span> sapien make ${name}.json -o out</div>
    ${errors.map((e) => html`<div class="prob" key=${e}><${Icon} n="warn" size=${14} /><span>${e}</span></div>`)}
    ${view && view.unmapped && view.unmapped.length > 0 && html`<p class="faint small">No dataset id is known for ${view.unmapped.join(', ')}, so ${view.unmapped.length === 1 ? 'it is' : 'they are'} not in the request.</p>`}
    ${err && html`<div class="prob" role="alert"><${Icon} n="warn" size=${14} /><span>${err}</span></div>`}
    ${out && out.plan && html`<${PlanView} out=${out} />`}
  </section>`;
}
