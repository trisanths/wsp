import { html, useState, useEffect, useRef } from './lib.js';
import { agentColor, agentInitials, tokenizeLine, useStubFlag, useHealth, subscribeToasts, clamp, pct, go } from './util.js';
import { usePresence, presenceClass, tokenMs, reduced, PopNum, SlidingBar, dissolveText } from './motion.js';

const ICONS = {
  plus: 'M12 5v14M5 12h14',
  x: 'M6 6l12 12M18 6L6 18',
  warn: 'M12 4l9 16H3zM12 10v4M12 17h.01',
  lock: 'M7 11V8a5 5 0 0110 0v3M6 11h12v9H6z',
  pause: 'M8 5v14M16 5v14',
  play: 'M7 4l13 8-13 8z',
  stop: 'M6 6h12v12H6z',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  undo: 'M9 14L4 9l5-5M4 9h10a6 6 0 010 12h-3',
  chevron: 'M9 6l6 6-6 6',
  left: 'M15 6l-6 6 6 6',
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
  sun: 'M12 8a4 4 0 100 8 4 4 0 000-8zM12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M18.7 5.3l-1.4 1.4M6.7 17.3l-1.4 1.4',
  moon: 'M20 14.5A8 8 0 019.5 4a8 8 0 1010.5 10.5z',
  panel: 'M4 5h16v14H4zM9.5 5v14',
  grid: 'M4 4h6.5v6.5H4zM13.5 4H20v6.5h-6.5zM4 13.5h6.5V20H4zM13.5 13.5H20V20h-6.5z',
  runs: 'M3 12h4l3-8 4 16 3-8h4',
  cube: 'M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5',
  book: 'M5 4h11a3 3 0 013 3v13H8a3 3 0 01-3-3zM5 17a3 3 0 013-3h11',
  sigma: 'M18 5H6l6.5 7L6 19h12',
  spark: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z',
  file: 'M7 3h7l5 5v13H7zM14 3v5h5',
  scissors: 'M5 5l14 14M5 19L19 5',
  server: 'M4 5h16v6H4zM4 13h16v6H4zM8 8h.01M8 16h.01',
};
export function Icon({ n, size = 16, sw = 1.75 }) {
  return html`<svg class="ic-svg" width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width=${sw} stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d=${ICONS[n] || ''}/></svg>`;
}

export function StubBadge({ small, title }) {
  return html`<span class=${'stubtag' + (small ? ' sm' : '')} title=${title || 'Stub backend: these numbers are not real results'}>STUB</span>`;
}

// the studio mark: three bars, the middle one cut, like a deletion mask
export function Mark({ size = 24 }) {
  return html`<svg class="mark" width=${size} height=${size} viewBox="0 0 24 24" aria-hidden="true"><rect width="24" height="24" rx="7" fill="var(--btn-bg)"/><path d="M7.5 6.5v11M12 6.5v11M16.5 6.5v11" stroke="var(--btn-fg)" stroke-width="2" stroke-linecap="round"/><path d="M12 10.8v2.4" stroke="var(--btn-bg)" stroke-width="4"/></svg>`;
}
export function Logo() {
  return html`<a class="logo" href="#/"><${Mark} /><span class="logotext">Sapien</span></a>`;
}

export function TopBar({ children, right }) {
  const stub = useStubFlag();
  const health = useHealth();
  const [open, setOpen] = useState(false);
  const modal = health && health.backend === 'modal';
  const label = health && health.llm_model ? health.llm_model + ' agents' : 'Sapien agents';
  return html`<header class="topbar">
    <div class="picker-wrap">
      <button class="picker" onClick=${() => setOpen(!open)} aria-haspopup="menu" aria-expanded=${open}><span>${label}</span><${Icon} n="down" size=${14} /></button>
      <${Menu} open=${open} origin="top-left" onClose=${() => setOpen(false)} style=${{ top: '36px', left: 0, minWidth: '260px' }}>
        <div class="mrow"><span class="mk">Agent model</span><span class="mv">${(health && health.llm_model) || 'not set'}</span></div>
        <div class="mrow"><span class="mk">Agent provider</span><span class="mv">${(health && health.llm) || 'none'}</span></div>
        <div class="mrow"><span class="mk">Backend</span><span class="mv">${health ? (modal ? 'Modal GPU' : health.backend === 'stub' ? 'Stub' : health.backend) : 'unknown'}</span></div>
        <div class="mnote">Read only. The server sets these.</div>
      <//>
    </div>
    ${children}
    <span class="grow"></span>
    ${right}
    ${modal ? html`<span class="pill" title="Jobs run on Modal GPUs">Modal GPU</span>` : stub && html`<${StubBadge} />`}
  </header>`;
}

export function Dot({ color, pulse }) {
  return html`<span class=${'dot' + (pulse ? ' pulse' : '')} style=${{ background: color, color }}></span>`;
}

export function Avatar({ agent, size = 32 }) {
  const c = agentColor(agent.id);
  return html`<div class="av" style=${{ '--c': c, width: size + 'px', height: size + 'px', fontSize: size * 0.4 + 'px' }}>${agentInitials(agent)}</div>`;
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
  return html`<div class="ringwrap" style=${{ width: size + 'px', height: size + 'px' }}>
    <svg class="ring" width=${size} height=${size} viewBox="0 0 100 100">
      <circle class="ring-bg" cx="50" cy="50" r="42" stroke-width="7" fill="none"/>
      <circle class="ring-fg" cx="50" cy="50" r="42" stroke-width="7" fill="none" stroke-linecap="round" stroke-dasharray=${C} stroke-dashoffset=${C * (1 - v)} transform="rotate(-90 50 50)"/>
    </svg>
    <div class="ring-c"><div class="ring-v"><${PopNum} text=${Math.round(v * 100) + '%'} /></div><div class="ring-l">${label}</div></div>
  </div>`;
}

export function Tabs({ tabs, value, onChange, right }) {
  return html`<${SlidingBar} class="tabs">${tabs.map((t) => html`<button class=${'tab' + (value === t.id ? ' on' : '')} onClick=${() => onChange(t.id)}>${t.label}${t.n != null && html`<span class="tabn">${t.n}</span>`}</button>`)}<span class="grow"></span>${right}<//>`;
}

// wsp text with syntax colors. prev is the older text, changed lines get a highlight
export function CodeView({ text, prev }) {
  const lines = (text || '').replace(/\n$/, '').split('\n');
  const old = prev != null ? new Set(prev.split('\n')) : null;
  return html`<div class="code">${lines.map((ln, i) => html`<div class=${'cl' + (old && !old.has(ln) ? ' hl' : '')}><span class="ln">${i + 1}</span><span class="lt">${tokenizeLine(ln).map((t) => (t.c ? html`<span class=${'tk-' + t.c}>${t.t}</span>` : t.t))}${ln === '' ? ' ' : ''}</span></div>`)}</div>`;
}

// Up to three toasts stack with the newest in front. A fourth pushes the oldest out.
export function Toasts() {
  const [items, setItems] = useState([]); // newest first
  const stack = useRef();
  const timers = useRef(new Set());
  const removeLater = (id, ms) => {
    if (timers.current.has(id)) return;
    timers.current.add(id);
    setTimeout(() => { timers.current.delete(id); setItems((l) => l.filter((x) => x.id !== id)); }, reduced() ? 0 : ms);
  };
  const dismiss = (id) => setItems((l) => l.map((x) => (x.id === id && x.mode === 'in' ? { ...x, mode: 'dismiss' } : x)));
  useEffect(() => subscribeToasts((t) => {
    setItems((l) => {
      let n = 0;
      return [{ ...t, mode: 'in' }, ...l].map((x) => (x.mode === 'in' && n++ >= 3 ? { ...x, mode: 'leaving' } : x));
    });
    setTimeout(() => dismiss(t.id), 3800);
  }), []);
  useEffect(() => {
    items.forEach((x) => { if (x.mode === 'leaving') removeLater(x.id, tokenMs('--stack-close', 250) + 60); else if (x.mode === 'dismiss') removeLater(x.id, tokenMs('--toast-close', 250) + 60); });
  }, [items]);
  // hovering the stack fans it out. This is geometry, because the gaps belong to no element.
  useEffect(() => {
    if (items.length < 2) return;
    const gap = () => parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--stack-spread-gap')) || 8;
    const within = (e, below) => {
      const el = stack.current;
      if (!el) return false;
      const r = el.getBoundingClientRect();
      return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom + below;
    };
    const move = (e) => {
      const el = stack.current;
      if (!el) return;
      if (el.classList.contains('is-spread')) { if (!within(e, (el.offsetHeight + gap()) * 2)) el.classList.remove('is-spread'); }
      else if (within(e, 0)) el.classList.add('is-spread');
    };
    window.addEventListener('pointermove', move);
    return () => { window.removeEventListener('pointermove', move); if (stack.current) stack.current.classList.remove('is-spread'); };
  }, [items.length > 1]);
  let depth = 0;
  return html`<div class="toasts"><div class="t-stack" ref=${stack}>${items.map((t) => {
    const d = t.mode === 'in' ? depth++ : 0;
    return html`<div key=${t.id} class=${'toast t-stack-banner ' + t.kind + (t.mode === 'leaving' ? ' is-leaving' : t.mode === 'dismiss' ? ' is-dismiss' : '')} data-depth=${d}
      ref=${(el) => { if (el && !el._ent) { el._ent = 1; if (!reduced()) { el.classList.add('is-enter'); void el.offsetWidth; el.classList.remove('is-enter'); } } }}>${t.text}</div>`;
  })}</div></div>`;
}

export function Spinner({ size = 14 }) {
  return html`<span class="spin" style=${{ width: size + 'px', height: size + 'px' }}></span>`;
}

export function ErrorBox({ error, onRetry }) {
  return html`<div class="errbox card"><div class="label">Something went wrong</div><div style="margin:6px 0 10px">${error.message || String(error)}</div>${onRetry && html`<button class="btn" onClick=${onRetry}>Try again</button>`}<a class="btn ghost" href="#/">Start over</a></div>`;
}

// A tiny menu that closes on outside click. It opens from the corner named by origin.
export function Menu({ open, onClose, children, class: cls, style, origin = 'top-left' }) {
  const ref = useRef();
  const st = usePresence(open, tokenMs('--dropdown-close-dur', 150));
  useEffect(() => {
    if (!open) return;
    const f = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    const k = (e) => { if (e.key === 'Escape') onClose(); };
    const t = setTimeout(() => document.addEventListener('mousedown', f), 0);
    document.addEventListener('keydown', k);
    return () => { clearTimeout(t); document.removeEventListener('mousedown', f); document.removeEventListener('keydown', k); };
  }, [open]);
  if (st === 'out') return null;
  return html`<div class=${'menu t-dropdown ' + presenceClass(st) + ' ' + (cls || '')} data-origin=${cls && /\bup\b/.test(cls) ? 'bottom-left' : origin} style=${style} ref=${ref}>${children}</div>`;
}

// A composer: rounded box, optional chips on top, a growing textarea, attach and send buttons.
export function Composer({ value, onInput, onSubmit, placeholder, chips, onAttach, disabled, over, taRef, autoFocus, footLeft, label, minRows = 2, extra, clearRef }) {
  const own = useRef();
  const ta = taRef || own;
  const wrap = useRef(), mirror = useRef(), phold = useRef(), glow = useRef();
  const has = !!(value && value.trim());
  // the parent calls clearRef.current() just before it empties the text, so the sent text dissolves
  if (clearRef) clearRef.current = () => dissolveText({ wrap: wrap.current, ta: ta.current, mirror: mirror.current, phold: phold.current, glow: glow.current });
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(220, Math.max(el.scrollHeight, minRows * 24)) + 'px';
  }, [value]);
  useEffect(() => { if (autoFocus && ta.current) ta.current.focus(); }, []);
  return html`<div class=${'composer' + (over ? ' over' : '')}>
    ${chips}
    <div class="t-clear" ref=${wrap}>
      <textarea ref=${ta} rows=${minRows} value=${value} placeholder=${placeholder} aria-label=${label || 'Message'} disabled=${disabled} onInput=${(e) => onInput(e.target.value)} onKeyDown=${(e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); onSubmit(); } }}></textarea>
      <div class="t-clear-mirror" aria-hidden="true" ref=${mirror}></div>
      <div class="t-clear-placeholder" aria-hidden="true" ref=${phold}>${placeholder}</div>
      <div class="t-clear-glow" aria-hidden="true" ref=${glow}></div>
    </div>
    <div class="comp-foot">
      <div class="comp-left">${onAttach && html`<button class="rbtn" type="button" data-tip="Attach files" aria-label="Attach files" onClick=${onAttach}><${Icon} n="clip" size=${16} /></button>`}${footLeft}</div>
      <button class=${'send' + (has ? ' on' : '')} type="button" data-tip="Send" aria-label="Send" disabled=${!has || disabled} onClick=${onSubmit}><${Icon} n="up" size=${16} sw=${2} /></button>
    </div>
    ${extra}
  </div>`;
}
