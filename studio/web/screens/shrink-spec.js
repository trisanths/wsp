import { html, useState, useRef } from '../lib.js';
import { clamp } from '../util.js';
import { Icon, Menu } from '../components.js';
import { SubtleNum } from '../motion.js';
import { ModelLogo } from '../logos.js';
import { Slider } from './spec.js';

// Shared by the spec page ("What to remove") and the shrink page.

export const SKILLS = [
  { id: 'code', name: 'Code', why: 'Programming text and Python functions' },
  { id: 'math', name: 'Math', why: 'Word problems and worked steps' },
  { id: 'medicine', name: 'Medicine', why: 'Health and biology questions' },
  { id: 'general', name: 'General knowledge', why: 'Broad facts and everyday questions' },
  { id: 'prose', name: 'Prose', why: 'Stories and plain writing' },
];
export const skillName = (id) => (SKILLS.find((s) => s.id === id) || { name: id }).name;

export const RULES = {
  experts: [
    { id: 'saliency', name: 'Saliency', why: 'Keep the experts the skill text leans on most. The default.' },
    { id: 'freq', name: 'Frequency', why: 'Keep the experts the skill text picks most often.' },
    { id: 'selective', name: 'Selective', why: 'Keep the experts that the skill uses and other text does not.' },
    { id: 'general', name: 'General text', why: 'Keep what general text uses. A control, it should score lower.' },
    { id: 'random', name: 'Random', why: 'Remove experts at random. A control, it should score lower.' },
  ],
  dense: [
    { id: 'skill', name: 'Skill cost', why: 'Remove the layers, neurons and heads that cost the skill the least. The default.' },
    { id: 'specific', name: 'Skill specific', why: 'Keep the units that matter to the skill and not to other text.' },
    { id: 'general', name: 'General text', why: 'Keep what general text uses. A control, it should score lower.' },
    { id: 'random', name: 'Random', why: 'Remove units at random. A control, it should score lower.' },
  ],
};
export const ruleName = (method, id) => { const r = (RULES[method] || []).find((x) => x.id === id || (id === 'skill_cost' && x.id === 'skill')); return r ? r.name : id; };
export const METHOD_TEXT = {
  experts: ['Expert removal', 'The model is a mixture of experts. Whole experts the skill rarely uses are removed from every layer.'],
  dense: ['Dense slimming', 'The model has no experts. Layers, neurons and attention heads the skill rarely uses are removed.'],
};
export const modelName = (m) => String(m || '').split('/').pop();
export const fmtB = (x, d = 2) => (x == null || isNaN(x) ? '-' : Number(x).toFixed(d) + 'B');

/** The experts per layer a slider can reach when the spec does not say. The shrink page knows the real number. */
const EXPERTS_SEEN = {};
export function expertsGuess(sh) {
  if (sh.experts_total) return sh.experts_total;
  // a candidate's reason names its experts ("MoE with 64 experts per layer"); remember it for when the model becomes the source
  for (const c of sh.candidates || []) { const m = /(\d+) experts per layer/.exec(c.why || ''); if (m) EXPERTS_SEEN[c.model] = Number(m[1]); }
  if (EXPERTS_SEEN[sh.source_model]) return EXPERTS_SEEN[sh.source_model];
  let n = 128;
  while (n < (sh.keep_experts || 1) * 4 && n < 1024) n *= 2;
  return n;
}

/** A quiet menu button: current value, a chevron, and a menu of choices. */
export function Pick({ label, value, items, onPick, width = 300 }) {
  const [open, setOpen] = useState(false);
  const [side, setSide] = useState({ origin: 'top-left', w: width });
  const wrap = useRef();
  // Measure on open: cap the menu to the viewport minus a 12px margin, and align its right edge with the
  // button's when the menu would run past the right side of the viewport.
  const toggle = () => {
    if (!open && wrap.current) {
      const r = wrap.current.getBoundingClientRect();
      const w = Math.min(width, Math.max(160, window.innerWidth - 24));
      setSide({ origin: r.left + w > window.innerWidth - 12 ? 'top-right' : 'top-left', w });
    }
    setOpen(!open);
  };
  return html`<span class="addwrap pickwrap" ref=${wrap}>
    <button class="btn sm" onClick=${toggle} aria-haspopup="menu" aria-expanded=${open} aria-label=${label}>${value}<${Icon} n="down" size=${14} /></button>
    <${Menu} open=${open} class="pickmenu" origin=${side.origin} onClose=${() => setOpen(false)} style=${{ minWidth: side.w + 'px' }}>
      ${items.map((it) => html`<button class=${'mi two' + (it.on ? ' on' : '')} key=${it.id} onClick=${() => { setOpen(false); onPick(it.id); }}>
        <span class="mi-main">${it.lead}<span class="mi-t"><span>${it.name}</span><span class="faint mi-why">${it.why}</span></span></span>${it.right && html`<span class="faint">${it.right}</span>`}
      </button>`)}
    <//>
  </span>`;
}

export function ShrinkSection({ spec, edit, onChangeSource }) {
  const sh = spec.shrink;
  if (!sh) return null;
  const moe = sh.method === 'experts';
  const src = sh.source_params_b;
  const cands = sh.candidates || [];
  const [m1, m2] = METHOD_TEXT[sh.method] || METHOD_TEXT.dense;
  const setSh = (patch) => edit((s) => { s.shrink = { ...s.shrink, ...patch }; });
  const tMin = Math.max(0.1, Math.round(src * 0.05 * 10) / 10);
  const tMax = Math.max(tMin + 0.1, Math.floor((src - 0.05) * 10) / 10);
  const target = clamp(sh.target_params_b != null ? sh.target_params_b : tMax, tMin, tMax);
  const nExp = expertsGuess(sh);
  const keep = clamp(sh.keep_experts || 1, 1, nExp);
  const steps = sh.heal_steps ?? 300;
  return html`<section class="sect"><h2 class="sec">What to remove</h2>
    <div class="srow">
      <div class="top"><span class="nm">Source model</span>
        ${cands.length > 0 && html`<${Pick} label="Change the source model" value="Change" onPick=${onChangeSource} width=${360}
          items=${cands.map((c) => ({ id: c.model, name: modelName(c.model), why: c.why, right: fmtB(c.params_b, 1), lead: html`<${ModelLogo} model=${c.model} size=${18} />` }))} />`}
      </div>
      <div class="card row-card srccard"><${ModelLogo} model=${sh.source_model} size=${24} />
        <div class="srcmain"><div class="srcname">${modelName(sh.source_model)}</div><div class="faint small">${fmtB(src)} stored, ${fmtB(sh.source_active_params_b)} run per token</div></div>
      </div>
      ${(sh.notes || []).map((n) => html`<div class="note"><span>${n}</span></div>`)}
    </div>
    <div class="srow">
      <div class="top"><span class="nm">Skill to keep</span>
        <${Pick} label="Skill to keep" value=${skillName(sh.keep_skill)} onPick=${(id) => setSh({ keep_skill: id })} width=${300}
          items=${SKILLS.map((k) => ({ ...k, on: k.id === sh.keep_skill }))} />
      </div>
      <div class="note"><span>Everything the model knows that this skill does not use is a candidate for removal.</span></div>
    </div>
    <div class="srow">
      <div class="top"><span class="nm">Method</span><span class="val">${m1}</span></div>
      <div class="note"><span>${m2} It follows from the model, so it is not a choice.</span></div>
    </div>
    <div class="srow">
      <div class="top"><span class="nm">Target size</span><span class="val"><${SubtleNum} text=${target.toFixed(1)} />B stored</span></div>
      <${Slider} value=${target} min=${tMin} max=${tMax} step=${0.1} ticks=${[{ v: src, kind: 'host', label: 'source ' + src.toFixed(2) + 'B' }]} ariaLabel="Target size in billions of parameters"
        onChange=${(v) => setSh(moe ? { target_params_b: Math.round(v * 10) / 10, keep_experts: null } : { target_params_b: Math.round(v * 10) / 10 })} />
      <div class="note"><span>${moe ? 'Moving this sets how many experts stay. The server picks the closest count that fits.' : 'The size after layers, neurons and heads are removed.'}</span><span>${tMin}B to ${tMax}B</span></div>
    </div>
    ${moe && html`<div class="srow">
      <div class="top"><span class="nm">Experts kept per layer</span><span class="val"><${SubtleNum} text=${String(keep)} /> of ${nExp}</span></div>
      <${Slider} value=${keep} min=${1} max=${nExp} step=${1} ariaLabel="Experts kept per layer" onChange=${(v) => setSh({ keep_experts: v })} />
      <div class="note"><span>Fewer experts means a smaller model. The plan page shows the size this gives.</span><span>1 to ${nExp}</span></div>
    </div>`}
    <div class="srow">
      <div class="top"><span class="nm">Heal steps</span><span class="val"><${SubtleNum} text=${String(steps)} /> steps</span></div>
      <${Slider} value=${steps} min=${0} max=${1000} step=${50} ariaLabel="Heal steps" onChange=${(v) => setSh({ heal_steps: v })} />
      <div class="note"><span>After removal the model trains to copy the original. More steps cost more time and usually recover more.</span><span>0 to 1000</span></div>
    </div>
  </section>`;
}
