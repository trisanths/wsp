// Model maker logos. A model gets its maker's mark only when the maker itself
// published it (or it is an exact re-upload, like unsloth's copies of Llama and
// Gemma). Fine-tunes by other people get a plain letter. Files and license:
// logos/README.md.
import { html } from './lib.js';

// Each rule: a test on the lowercased model id, the file in logos/, the maker's
// name for alt text, and whether the file is single colour (drawn in black, so it
// is inverted in the dark theme).
const RULES = [
  { test: (m) => m.startsWith('qwen/'), file: 'qwen-color', name: 'Qwen' },
  { test: (m) => m.startsWith('liquidai/'), file: 'liquid', name: 'Liquid AI', mono: true },
  { test: (m) => m.startsWith('meta-llama/') || /^unsloth\/(meta-)?llama/.test(m), file: 'meta-color', name: 'Meta' },
  { test: (m) => m.startsWith('google/gemma') || m.startsWith('unsloth/gemma'), file: 'gemma-color', name: 'Gemma' },
  { test: (m) => m.startsWith('mistralai/') || m.startsWith('unsloth/mistral'), file: 'mistral-color', name: 'Mistral AI' },
  { test: (m) => m.startsWith('deepseek-ai/'), file: 'deepseek-color', name: 'DeepSeek' },
  { test: (m) => m.startsWith('microsoft/'), file: 'microsoft-color', name: 'Microsoft' },
  { test: (m) => m.startsWith('xiaomimimo/'), file: 'xiaomimimo', name: 'Xiaomi MiMo', mono: true },
  { test: (m) => m.startsWith('zai-org/') || m.startsWith('thudm/'), file: 'zai', name: 'Z.ai', mono: true },
  { test: (m) => m.startsWith('moonshotai/'), file: 'kimi-color', name: 'Moonshot AI' },
  { test: (m) => m.startsWith('inclusionai/'), file: 'ling-color', name: 'inclusionAI' },
  { test: (m) => m.startsWith('ibm-granite/') || m.startsWith('ibm/'), file: 'ibm', name: 'IBM', mono: true },
  { test: (m) => m.startsWith('allenai/'), file: 'ai2-color', name: 'Ai2' },
  { test: (m) => m.startsWith('minimaxai/'), file: 'minimax-color', name: 'MiniMax' },
];

/** The logo for a model id: {src, name, mono} or null when the maker has no mark here. */
export function logoFor(model) {
  const m = String(model || '').toLowerCase();
  for (const r of RULES) if (r.test(m)) return { src: 'logos/' + r.file + '.svg', name: r.name, mono: !!r.mono };
  return null;
}

/** The first letter of the model's own name (after the org), for the fallback badge. */
function letterFor(model) {
  const name = String(model || '').split('/').pop() || '?';
  const ch = name.replace(/^[^a-z0-9]+/i, '').charAt(0);
  return (ch || '?').toUpperCase();
}

/**
 * A small square with the maker's logo, or a plain letter when there is none.
 * size is in pixels. The image is decorative when a label sits next to it, so pass
 * alt="" in that case (the default); pass a name when the logo stands alone.
 */
export function ModelLogo({ model, size = 18, alt = '', class: cls = '' }) {
  const logo = logoFor(model);
  const style = { width: size + 'px', height: size + 'px' };
  if (!logo) {
    return html`<span class=${'mlogo mlogo-letter ' + cls} style=${{ ...style, fontSize: Math.round(size * 0.55) + 'px' }} aria-hidden=${alt ? null : 'true'} title=${alt || null}>${letterFor(model)}</span>`;
  }
  return html`<span class=${'mlogo ' + cls} style=${style} title=${alt || logo.name}><img src=${logo.src} alt=${alt} width=${size} height=${size} class=${logo.mono ? 'mlogo-mono' : ''} loading="lazy" decoding="async" /></span>`;
}

/* ---------- sponsors: the services that run parts of Sapien ---------- */
// OpenAI models draft specs, think for the agents and write suggestions, Exa searches the web for
// benchmarks and published scores, Modal runs the GPU work and the agent sandboxes, and past.dev
// holds the memory the agents share. Their marks are in logos/ (see README).
export const SPONSORS = {
  openai: { src: 'logos/openai.svg', name: 'OpenAI', url: 'https://openai.com', mono: true },
  exa: { src: 'logos/exa.svg', name: 'Exa', url: 'https://exa.ai' },
  modal: { src: 'logos/modal.svg', name: 'Modal', url: 'https://modal.com' },
  pastdev: { src: 'logos/pastdev.svg', name: 'past.dev', url: 'https://past.dev' },
};

/** A sponsor's mark alone, for tight rows. */
export function SponsorMark({ who, size = 14, title }) {
  const s = SPONSORS[who];
  if (!s) return null;
  return html`<img class=${'spmark' + (s.mono ? ' mlogo-mono' : '')} src=${s.src} width=${size} height=${size} alt=${title || s.name} title=${title || s.name} loading="lazy" decoding="async" />`;
}

/**
 * A small chip that credits a sponsor where its service is at work, for example
 * "Search by Exa" or "GPUs on Modal". It links to the sponsor's site.
 */
export function PoweredBy({ who, label, size = 14, class: cls = '' }) {
  const s = SPONSORS[who];
  if (!s) return null;
  return html`<a class=${'powered ' + cls} href=${s.url} target="_blank" rel="noopener noreferrer" title=${(label ? label + ' ' : '') + s.name}><img class=${s.mono ? 'mlogo-mono' : null} src=${s.src} width=${size} height=${size} alt="" aria-hidden="true" /><span>${label ? label + ' ' : ''}<b>${s.name}</b></span></a>`;
}

/** "Built with" and the sponsors, for the home and tour pages. */
export function SponsorStrip() {
  return html`<div class="spstrip"><span class="faint">Built with</span>${['openai', 'exa', 'modal', 'pastdev'].map((w) => html`<${PoweredBy} key=${w} who=${w} size=${16} />`)}</div>`;
}
