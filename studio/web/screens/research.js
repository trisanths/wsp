import { html, useState, useEffect, useRef } from '../lib.js';
import { ModelLogo } from '../logos.js';
import { StreamText, ThinkLine, SuccessCheck, AutoHeight } from '../motion.js';
import { Icon, Spinner } from '../components.js';

const PROVIDERS = { exa: 'Exa', openai_web: 'OpenAI web search', none: 'the model\'s own knowledge' };
export const providerName = (p) => PROVIDERS[p] || 'web search';

// the server sends bare model names; the logo rules want an org prefix
const MAKERS = [
  [/^qwen/i, 'Qwen/'], [/^llama/i, 'meta-llama/'], [/^gemma/i, 'google/'], [/^phi/i, 'microsoft/'],
  [/^(mistral|ministral)/i, 'mistralai/'], [/^deepseek/i, 'deepseek-ai/'], [/^granite/i, 'ibm-granite/'],
  [/^(glm|chatglm)/i, 'zai-org/'], [/^kimi/i, 'moonshotai/'], [/^lfm/i, 'liquidai/'],
];
export function logoId(name) {
  const n = String(name || '');
  if (n.includes('/')) return n;
  for (const [re, pre] of MAKERS) if (re.test(n)) return pre + n;
  return n;
}

const safeUrl = (u) => { try { const x = new URL(u); return x.protocol === 'http:' || x.protocol === 'https:' ? x.href : null; } catch (e) { return null; } };
export const hostOf = (u) => { try { return new URL(u).hostname.replace(/^www\./, ''); } catch (e) { return ''; } };
const score = (s) => (s * 100).toFixed(1).replace(/\.0$/, '') + '%';

// what the line above the timeline says while the agent works
export function stepLine(s) {
  if (!s) return 'Starting research';
  const t = String(s.text || '');
  if (s.kind === 'search') return 'Searching: ' + t.replace(/^searched:?\s*/i, '');
  if (s.kind === 'read') return 'Reading: ' + t.replace(/^read(ing)?:?\s*/i, '');
  return t;
}

const STEP_ICON = {
  think: 'M9 18h6M10 21h4M12 3a6 6 0 00-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0012 3z',
  search: 'M11 4a7 7 0 100 14 7 7 0 000-14zM16.5 16.5L21 21',
  read: 'M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6',
  pick: 'M5 12.5l4.5 4.5L19 7.5',
};
function StepIcon({ kind }) {
  return html`<svg class="ic-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d=${STEP_ICON[kind] || STEP_ICON.think}/></svg>`;
}

function Steps({ steps, animateFrom = 0, offset = 0, clamp }) {
  if (!steps.length) return null;
  return html`<ol class=${'rs-steps' + (clamp ? ' clamp' : '')}>${steps.map((s, i) => html`<li key=${i + offset} class=${'rs-step k-' + s.kind}><span class="rs-ic"><${StepIcon} kind=${s.kind} /></span><${StreamText} tag="span" class="rs-tx" text=${s.kind === 'read' ? 'Read: ' + s.text : s.text} play=${i + offset >= animateFrom} /></li>`)}</ol>`;
}

function Live({ r }) {
  const steps = r.steps || [];
  const first = useRef(steps.length);
  const last = steps[steps.length - 1];
  const before = Math.max(0, steps.length - 1 - 6);
  const shown = steps.slice(before, -1);
  return html`<div class="rs-live">
    <div class="rs-now"><${ThinkLine} fluid=${true} text=${stepLine(last)} /><span class="tag rs-prov">${providerName(r.provider)}</span></div>
    <${AutoHeight}>${before > 0 && html`<div class="rs-more">${before} earlier ${before === 1 ? 'step' : 'steps'}</div>`}<${Steps} steps=${shown} offset=${before} animateFrom=${first.current} clamp=${true} /><//>
  </div>`;
}

function Sources({ sources }) {
  const list = (sources || []).map((s) => ({ ...s, href: safeUrl(s.url) })).filter((s) => s.href);
  if (!list.length) return null;
  return html`<div class="rs-sub">Sources</div><ul class="rs-src">${list.map((s, i) => html`<li key=${i}><a href=${s.href} target="_blank" rel="noopener noreferrer">${s.title || hostOf(s.href)}</a><span class="rs-host">${hostOf(s.href)}</span></li>`)}</ul>`;
}

function Done({ r }) {
  const [open, setOpen] = useState(false);
  const n = (r.sources || []).length, m = (r.proposals || []).length;
  const via = r.provider === 'none' ? '' : ' with ' + providerName(r.provider);
  const text = (n ? 'Researched ' + n + (n === 1 ? ' source' : ' sources') + via + ', picked ' : 'Picked ') + m + (m === 1 ? ' benchmark' : ' benchmarks') + (r.provider === 'none' ? ' from the model\'s own knowledge' : '');
  return html`<div class="rs-done">
    <div class="rs-sumrow"><${SuccessCheck} size=${16} /><span>${text}</span><button class="lnk" aria-expanded=${open} onClick=${() => setOpen(!open)}>${open ? 'Hide the work' : 'Show the work'}</button></div>
    <${AutoHeight}>${open && html`<div class="rs-work">
      ${r.reasoning && String(r.reasoning).split(/\n\s*\n/).map((p, i) => html`<p class="rs-reason" key=${i}>${p}</p>`)}
      ${(r.steps || []).length > 0 && html`<div class="rs-sub">Steps</div><${Steps} steps=${r.steps} animateFrom=${1e9} />`}
      <${Sources} sources=${r.sources} />
    </div>`}<//>
  </div>`;
}

export function ResearchBlock({ r, starting, onStart }) {
  const running = r && r.status === 'running';
  const btn = html`<button class="lnk" disabled=${running || starting} onClick=${onStart}>${starting ? html`<${Spinner} size=${11} /> ` : ''}${r ? 'Research again' : 'Research benchmarks'}</button>`;
  if (!r) return html`<div class="rs rs-bare">${btn}</div>`;
  return html`<section class="sect rs">
    <h2 class="sec"><span>Benchmark research</span>${!running && btn}</h2>
    ${running && html`<${Live} r=${r} />`}
    ${r.status === 'done' && html`<${Done} r=${r} />`}
    ${r.status === 'error' && html`<div class="rs-line err">${r.error || 'The research did not finish.'}</div>`}
    ${r.status === 'skipped' && html`<div class="rs-line">${r.error || 'Research was skipped.'}</div>`}
  </section>`;
}

// reference scores as slider ticks
export function refTicks(scores) {
  return (scores || []).filter((s) => s && typeof s.score === 'number').map((s) => ({ v: Math.min(1, Math.max(0, s.score)), kind: 'ref' + (s.sourced ? '' : ' est') }));
}

export function RefList({ scores }) {
  const list = (scores || []).filter((s) => s && typeof s.score === 'number');
  if (!list.length) return null;
  return html`<ul class="refs">${list.map((s, i) => {
    const href = s.sourced ? safeUrl(s.source_url) : null;
    return html`<li key=${i}><${ModelLogo} model=${logoId(s.model)} size=${14} /><span class="rf-nm">${s.model}</span>${s.params_b != null && html`<span class="rf-sz">${s.params_b}B</span>`}<span class="rf-sc">${score(s.score)}</span>${href ? html`<a class="rf-src" href=${href} target="_blank" rel="noopener noreferrer" data-tip=${s.quote ? String(s.quote).slice(0, 240) : null}>source</a>` : html`<span class="rf-est" data-tip="The agent could not link this number to a page it read">estimate</span>`}</li>`;
  })}</ul>`;
}

export function AlsoConsidered({ proposals }) {
  const list = (proposals || []).filter((p) => p.in_spec === false);
  if (!list.length) return null;
  return html`<section class="sect"><h2 class="sec"><span>Also considered</span></h2>
    <div class="rs-faint">Sapien cannot run these yet. Shown for reference.</div>
    ${list.map((p, i) => html`<div class="alsorow" key=${i}>
      <div class="top"><span class="nm">${p.name}</span><span class="tag">reference only</span></div>
      ${p.measures && html`<div class="rs-why">${p.measures}</div>`}
      <${RefList} scores=${p.reference_scores} />
    </div>`)}
  </section>`;
}
