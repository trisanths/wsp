import { html, useState, useEffect, useRef } from './lib.js';
import { agentColor, agentInitials, tokenizeLine, useStubFlag, useToasts, clamp, pct, go } from './util.js';

const ICONS = {
  plus: 'M12 5v14M5 12h14',
  x: 'M6 6l12 12M18 6L6 18',
  pause: 'M8 5v14M16 5v14',
  play: 'M7 4l13 8-13 8z',
  stop: 'M6 6h12v12H6z',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  undo: 'M9 14L4 9l5-5M4 9h10a6 6 0 010 12h-3',
  chevron: 'M9 6l6 6-6 6',
  down: 'M6 9l6 6 6-6',
  search: 'M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-4.5-4.5',
  clip: 'M20 11.5l-8 8a5 5 0 01-7-7l8.5-8.5a3.3 3.3 0 014.7 4.7L9.5 17.2a1.7 1.7 0 01-2.4-2.4L15 7',
  grip: 'M9 6h.01M9 12h.01M9 18h.01M15 6h.01M15 12h.01M15 18h.01',
  more: 'M5 12h.01M12 12h.01M19 12h.01',
  arrow: 'M5 12h14M13 6l6 6-6 6',
  bolt: 'M13 3L5 14h6l-1 7 8-11h-6z',
  code: 'M8 8l-5 4 5 4M16 8l5 4-5 4M14 5l-4 14',
  layers: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5',
  trash: 'M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13',
  gear: 'M12 9a3 3 0 100 6 3 3 0 000-6zM12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M18.4 5.6l-2.1 2.1M7.7 16.3l-2.1 2.1',
  heal: 'M12 5v14M5 12h14',
  up: 'M12 19V5M6 11l6-6 6 6',
};
export function Icon({ n, size = 16, sw = 2 }) {
  return html`<svg class="ic-svg" width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width=${sw} stroke-linecap="round" stroke-linejoin="round"><path d=${ICONS[n] || ''}/></svg>`;
}

export function StubBadge({ small, title }) {
  return html`<span class=${'stubtag' + (small ? ' sm' : '')} title=${title || 'Stub backend: these numbers are not real results'}>STUB</span>`;
}

export function Logo() {
  return html`<a class="logo" href="#/"><span class="logomark">w</span><span><b>wsp</b> studio</span></a>`;
}

export function TopBar({ children, right, crumb }) {
  const stub = useStubFlag();
  return html`<header class="topbar">
    <${Logo} />
    ${crumb != null && html`<span class="faint">/</span><span class="crumb">${crumb}</span>`}
    ${children}
    <span class="grow"></span>
    ${stub && html`<${StubBadge} />`}
    ${right}
  </header>`;
}

export function Dot({ color, pulse }) {
  return html`<span class=${'dot' + (pulse ? ' pulse' : '')} style=${{ background: color, color }}></span>`;
}

export function Avatar({ agent, size = 32 }) {
  const c = agentColor(agent.id);
  return html`<div class="av" style=${{ background: c, width: size + 'px', height: size + 'px', fontSize: size * 0.4 + 'px' }}>${agentInitials(agent)}</div>`;
}

export function Tag({ kind, children, title }) {
  return html`<span class=${'tag ' + (kind || '')} title=${title}>${children}</span>`;
}

// fixed scale per benchmark so bars do not jump when a score moves
export function benchScale(b) {
  const m = Math.max(b.target || 0, b.lora || 0, b.floor || 0, b.base || 0);
  return clamp(Math.ceil((m * 1.35) / 0.05) * 0.05, 0.4, 1);
}
export function BenchBar({ b, value, tall }) {
  const sc = benchScale(b);
  const at = (x) => clamp((x / sc) * 100, 0, 100) + '%';
  const guardBad = b.kind === 'guard' && value != null && value < b.floor;
  return html`<div class=${'btrack' + (tall ? ' tall' : '')}>
    <i style=${{ width: value == null ? '0%' : at(value) }} class=${guardBad ? 'bad' : b.kind === 'guard' ? 'good' : ''}></i>
    ${b.base != null && html`<u class="t-base" style=${{ left: at(b.base) }} title=${'Host score ' + pct(b.base) + '%'}></u>`}
    ${b.lora != null && html`<u class="t-lora" style=${{ left: at(b.lora) }} title=${'LoRA ' + pct(b.lora) + '%'}></u>`}
    ${b.kind === 'target' && b.target != null && html`<u class="t-target" style=${{ left: at(b.target) }} title=${'Target ' + pct(b.target) + '%'}></u>`}
    ${b.kind === 'guard' && b.floor != null && html`<u class="t-floor" style=${{ left: at(b.floor) }} title=${'Floor ' + pct(b.floor) + '%'}></u>`}
  </div>`;
}
export function BenchLegend({ guard }) {
  return html`<span class="legend"><span class="lg base">host</span><span class="lg lora">LoRA</span><span class="lg target">target</span>${guard && html`<span class="lg floor">floor</span>`}</span>`;
}

export function Ring({ value, label = 'of the way to spec', size = 104 }) {
  const C = 2 * Math.PI * 42;
  const v = clamp(value || 0, 0, 1);
  return html`<svg class="ring" width=${size} height=${size} viewBox="0 0 100 100">
    <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7c8cff"/><stop offset="1" stop-color="#b38bff"/></linearGradient></defs>
    <circle cx="50" cy="50" r="42" stroke="#222837" stroke-width="9" fill="none"/>
    <circle cx="50" cy="50" r="42" stroke="url(#rg)" stroke-width="9" fill="none" stroke-linecap="round" stroke-dasharray=${C} stroke-dashoffset=${C * (1 - v)} transform="rotate(-90 50 50)" style="transition: stroke-dashoffset .6s cubic-bezier(.2,.8,.2,1)"/>
    <text x="50" y="50" text-anchor="middle" fill="#e8ecf4" font-size="20" font-weight="700">${Math.round(v * 100)}%</text>
    <text x="50" y="64" text-anchor="middle" fill="#8b93a7" font-size="7.5">${label}</text>
  </svg>`;
}

export function Tabs({ tabs, value, onChange, right }) {
  return html`<div class="tabs">${tabs.map((t) => html`<button class=${'tab' + (value === t.id ? ' on' : '')} onClick=${() => onChange(t.id)}>${t.label}${t.n != null && html`<span class="tabn">${t.n}</span>`}</button>`)}<span class="grow"></span>${right}</div>`;
}

// wsp text with syntax colors. prev is the older text, changed lines get a highlight
export function CodeView({ text, prev }) {
  const lines = (text || '').replace(/\n$/, '').split('\n');
  const old = prev != null ? new Set(prev.split('\n')) : null;
  return html`<div class="code">${lines.map((ln, i) => html`<div class=${'cl' + (old && !old.has(ln) ? ' hl' : '')}><span class="ln">${i + 1}</span><span class="lt">${tokenizeLine(ln).map((t) => (t.c ? html`<span class=${'tk-' + t.c}>${t.t}</span>` : t.t))}${ln === '' ? ' ' : ''}</span></div>`)}</div>`;
}

export function Toasts() {
  const list = useToasts();
  return html`<div class="toasts">${list.map((t) => html`<div class=${'toast ' + t.kind} key=${t.id}>${t.text}</div>`)}</div>`;
}

export function Spinner({ size = 14 }) {
  return html`<span class="spin" style=${{ width: size + 'px', height: size + 'px' }}></span>`;
}

export function ErrorBox({ error, onRetry }) {
  return html`<div class="errbox card"><div class="label">Something went wrong</div><div style="margin:6px 0 10px">${error.message || String(error)}</div>${onRetry && html`<button class="btn" onClick=${onRetry}>Try again</button>`}<a class="btn ghost" href="#/">Start over</a></div>`;
}

// a tiny menu that closes on outside click
export function Menu({ open, onClose, children, class: cls, style }) {
  const ref = useRef();
  useEffect(() => {
    if (!open) return;
    const f = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    const k = (e) => { if (e.key === 'Escape') onClose(); };
    setTimeout(() => document.addEventListener('mousedown', f), 0);
    document.addEventListener('keydown', k);
    return () => { document.removeEventListener('mousedown', f); document.removeEventListener('keydown', k); };
  }, [open]);
  if (!open) return null;
  return html`<div class=${'menu ' + (cls || '')} style=${style} ref=${ref}>${children}</div>`;
}
