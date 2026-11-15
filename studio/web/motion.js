// Motion helpers for the studio. The CSS lives in motion.css.
// Each helper adapts a free transition from transitions.dev: the transition's
// name is in the comment above it. Helpers are generic so any screen can reuse them.
import { html, useState, useEffect, useLayoutEffect, useRef } from './lib.js';

/* ---------- shared helpers ---------- */

// True when the user asked for less motion. The html.reduce-motion class mirrors the media query.
export function reduced() {
  if (document.documentElement.classList.contains('reduce-motion')) return true;
  return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

// Reads a duration token such as --toast-close in ms. A probe element resolves var() and calc().
let probe = null;
export function tokenMs(name, fallback) {
  if (!probe) {
    probe = document.createElement('span');
    probe.style.cssText = 'position:absolute;visibility:hidden;pointer-events:none';
    document.body.appendChild(probe);
  }
  probe.style.transitionDuration = 'var(' + name + ')';
  const v = getComputedStyle(probe).transitionDuration || '';
  const n = parseFloat(v);
  if (!Number.isFinite(n)) return fallback;
  return v.endsWith('ms') ? n : n * 1000;
}
function tokenNum(name, fallback) {
  const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name));
  return Number.isFinite(v) ? v : fallback;
}
const reflow = () => void document.body.offsetWidth;

// Adds a class, lets its animation run, then removes it. Safe to call again while it runs.
export function playOnce(el, cls, ms) {
  if (!el || reduced()) return;
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
  clearTimeout(el._playTimer);
  el._playTimer = setTimeout(() => el.classList.remove(cls), ms);
}

/* ---------- presence: mount, open, close, unmount ---------- */
// Returns 'out' (unmounted), 'pre' (mounted at rest), 'open', or 'closing'.
// Used by menus, the workspace panel and anything that needs an exit animation.
export function usePresence(open, closeMs) {
  const [st, setSt] = useState(open ? 'open' : 'out');
  useLayoutEffect(() => {
    if (open && st === 'out') setSt(reduced() ? 'open' : 'pre');
    else if (open && st === 'closing') setSt('open');
    else if (!open && (st === 'pre' || st === 'open')) setSt(reduced() ? 'out' : 'closing');
    else if (st === 'pre') { reflow(); setSt('open'); }
  }, [open, st]);
  useEffect(() => {
    if (st !== 'closing') return;
    const t = setTimeout(() => setSt('out'), closeMs);
    return () => clearTimeout(t);
  }, [st]);
  return st;
}
export const presenceClass = (st) => (st === 'open' ? 'is-open' : st === 'closing' ? 'is-closing' : '');

/* ---------- transitions.dev: text states swap ---------- */
// Swaps the text of an element: old text exits up, new text enters from below.
export function swapText(el, next) {
  if (!el) return;
  if (el.textContent === next) return;
  if (reduced()) { el.textContent = next; return; }
  const dur = tokenMs('--text-swap-dur', 150);
  el.classList.add('is-exit');
  clearTimeout(el._swapTimer);
  el._swapTimer = setTimeout(() => {
    el.textContent = next;
    el.classList.remove('is-exit');
    el.classList.add('is-enter-start');
    void el.offsetHeight;
    el.classList.remove('is-enter-start');
  }, dur);
}
export function SwapText({ text, class: cls }) {
  const ref = useRef();
  const first = useRef(true);
  useLayoutEffect(() => {
    if (first.current) { first.current = false; ref.current.textContent = text; return; }
    swapText(ref.current, text);
  }, [text]);
  return html`<span class=${'t-text-swap ' + (cls || '')} ref=${ref}></span>`;
}

/* ---------- transitions.dev: icon swap ---------- */
export function IconSwap({ state, a, b }) {
  return html`<span class="t-icon-swap" data-state=${state}><span class="t-icon" data-icon="a">${a}</span><span class="t-icon" data-icon="b">${b}</span></span>`;
}

/* ---------- transitions.dev: number subtle change ---------- */
function lastNumber(s) {
  const m = /-?\d+(\.\d+)?/.exec(String(s));
  return m ? parseFloat(m[0]) : NaN;
}
// Keeps the last text it saw and builds a change record when the text differs.
function useChange(text) {
  const st = useRef({ cur: text, prev: null, up: true, seq: 0 });
  const s = st.current;
  if (s.cur !== text) {
    const a = lastNumber(s.cur), b = lastNumber(text);
    st.current = { cur: text, prev: s.cur, up: Number.isNaN(a) || Number.isNaN(b) ? true : b >= a, seq: s.seq + 1 };
  }
  return st.current;
}
// A small number that changes often. Only the changed digits cross-fade.
export function SubtleNum({ text, class: cls }) {
  const c = useChange(String(text));
  const str = c.cur, prev = c.prev;
  const same = prev != null && prev.length === str.length;
  const parts = [...str].map((ch, i) => {
    if (c.seq === 0 || !same || prev[i] === ch) return ch;
    return html`<span class="t-subtle-slot"><span class="t-subtle-old" aria-hidden="true">${prev[i]}</span><span class="t-subtle-new">${ch}</span></span>`;
  });
  if (c.seq > 0 && !same) {
    return html`<span class=${'t-subtle-number is-animating ' + (cls || '')} key=${c.seq} style=${{ '--subtle-dir': c.up ? 1 : -1 }} aria-label=${str}><span class="t-subtle-new">${str}</span></span>`;
  }
  return html`<span class=${'t-subtle-number' + (c.seq > 0 ? ' is-animating ' : ' ') + (cls || '')} key=${c.seq} style=${{ '--subtle-dir': c.up ? 1 : -1 }} aria-label=${str}>${parts}</span>`;
}

/* ---------- transitions.dev: number pop-in ---------- */
// A number that should be noticed when it changes. Shared leading characters stay still.
export function PopNum({ text, class: cls }) {
  const c = useChange(String(text));
  const str = c.cur, old = c.prev;
  if (c.seq === 0) return html`<span class=${'t-digit-group ' + (cls || '')} aria-label=${str}>${[...str].map((ch) => html`<span class="t-digit" data-same="">${ch}</span>`)}</span>`;
  let same = 0;
  while (same < old.length && same < str.length && old[same] === str[same]) same++;
  const kids = [...str].map((ch, i) => {
    if (i < same || !/\d/.test(ch)) return html`<span class="t-digit" data-same="">${ch}</span>`;
    const fromEnd = str.length - 1 - i;
    const oldCh = old[old.length - (str.length - i)];
    return html`<span class="t-digit-slot" data-stagger=${fromEnd === 1 ? '1' : fromEnd === 0 ? '2' : null}>${oldCh && /\d/.test(oldCh) ? html`<span class="t-digit-old" aria-hidden="true">${oldCh}</span>` : null}<span class="t-digit">${ch}</span></span>`;
  });
  return html`<span class=${'t-digit-group is-animating ' + (cls || '')} key=${c.seq} aria-label=${str}>${kids}</span>`;
}

/* ---------- transitions.dev: spinning counter ---------- */
const STEP = (50 * Math.PI) / 180;
const FADE_END = 1.8, LAND_AT = 0.6, ARRIVE_AT = 0.29, CLEAR_MS = 200, ENTER_MS = 120;
function bezier([x1, y1, x2, y2]) {
  const at = (a, b, t) => 3 * a * t * (1 - t) * (1 - t) + 3 * b * t * t * (1 - t) + t * t * t;
  return (x) => {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    let lo = 0, hi = 1, t = x;
    for (let i = 0; i < 20; i++) { if (at(x1, x2, t) < x) lo = t; else hi = t; t = (lo + hi) / 2; }
    return at(y1, y2, t);
  };
}
function emLength(el, name, fallbackEm) {
  const p = document.createElement('span');
  p.style.cssText = 'position:absolute;visibility:hidden;width:var(' + name + ')';
  el.appendChild(p);
  const px = p.getBoundingClientRect().width;
  p.remove();
  return px || fallbackEm * (parseFloat(getComputedStyle(el).fontSize) || 24);
}
function reelTokens(el) {
  const css = getComputedStyle(document.documentElement);
  const m = css.getPropertyValue('--reel-ease').match(/cubic-bezier\(([^)]+)\)/);
  return {
    stepDur: tokenNum('--reel-step-dur', 130), stagger: tokenNum('--reel-stagger', 90),
    ease: bezier(m ? m[1].split(',').map(parseFloat) : [0.35, -0.02, 0.43, 1]),
    offset: emLength(el, '--reel-side-offset', 0.52), sideScale: tokenNum('--reel-side-scale', 0.7),
    sideBlur: emLength(el, '--reel-side-blur', 0.083), sideOpacity: tokenNum('--reel-side-opacity', 0.17),
    centerScale: tokenNum('--reel-center-scale', 0.9), centerBlur: emLength(el, '--reel-center-blur', 0.042),
    anticDur: tokenNum('--reel-antic-dur', 100), anticScale: tokenNum('--reel-antic-scale', 0.94),
    anticLift: emLength(el, '--reel-antic-lift', 0.125), landDur: tokenNum('--reel-land-dur', 740), landPeak: tokenNum('--reel-land-peak', 1.03),
  };
}
const smooth = (x) => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };
const wave = (s) => (s >= 1 ? 0 : Math.exp(-3 * s) * Math.cos(3 * Math.PI * s) * (1 - s));
const WAVE_TROUGH = (() => { let low = 0; for (let i = 1; i < 400; i++) low = Math.min(low, wave(i / 400)); return -low; })();
function buildColumns(el, str) {
  el.replaceChildren();
  const cols = [];
  for (const ch of str) {
    if (ch < '0' || ch > '9') {
      const sep = document.createElement('span');
      sep.className = 't-reel-sep';
      sep.textContent = ch;
      el.appendChild(sep);
      continue;
    }
    const col = document.createElement('span');
    col.className = 't-reel-col';
    const sizer = document.createElement('span');
    sizer.className = 't-reel-sizer';
    sizer.textContent = '0';
    col.appendChild(sizer);
    const cells = [];
    for (let d = 0; d < 10; d++) {
      const cell = document.createElement('span');
      cell.className = 't-reel-digit';
      cell.textContent = String(d);
      col.appendChild(cell);
      cells.push(cell);
    }
    el.appendChild(col);
    cols.push({ cells, digit: +ch });
  }
  return cols;
}
function reelRest(col, digit) {
  col.cells.forEach((cell, d) => {
    cell.style.transform = 'none'; cell.style.filter = 'none'; cell.style.zIndex = '';
    cell.style.opacity = d === digit ? '1' : '0';
  });
}
function reelLook(dist, middle, o) {
  const angle = Math.min(dist * STEP, Math.PI / 2);
  const face = (Math.cos(angle) - Math.cos(STEP)) / (1 - Math.cos(STEP));
  return { angle, scale: o.sideScale + (middle - o.sideScale) * face, back: dist <= 1 ? 1 + (o.sideOpacity - 1) * dist : o.sideOpacity * (1 - (dist - 1) / (FADE_END - 1)) };
}
function reelPlace(col, pos, spread, middle, o, land) {
  const radius = o.offset / Math.sin(STEP);
  col.cells.forEach((cell, d) => {
    let p = (((d - pos) % 10) + 10) % 10;
    if (p > 5) p -= 10;
    const dist = Math.abs(p);
    if (dist >= FADE_END) { cell.style.opacity = '0'; return; }
    const l = reelLook(dist, middle, o);
    let y = (p < 0 ? 1 : -1) * radius * Math.sin(l.angle);
    const blur = (o.centerBlur + (o.sideBlur - o.centerBlur) * Math.min(dist, 1)) * spread;
    const away = Math.max(0, Math.min(1, (dist - 0.3) / 0.5));
    let opacity = l.back * (1 - (1 - spread) * away);
    let scale = l.scale;
    let z = 100 - Math.round(dist * 10);
    if (land && d === col.digit) { y = y * land.arrive + land.lift; scale = land.scale; opacity = land.opacity; z = 200; }
    cell.style.transform = 'translateY(' + y.toFixed(2) + 'px) scale(' + scale.toFixed(4) + ')';
    cell.style.filter = blur > 0.01 ? 'blur(' + blur.toFixed(2) + 'px)' : 'none';
    cell.style.opacity = opacity.toFixed(3);
    cell.style.zIndex = String(z);
  });
}
// Rolls the digits of el from one value to another. A value is a whole number or a text such
// as "41.1%"; only the digits roll. Returns a function that cancels it.
export function countTo(el, from, to) {
  const str = typeof to === 'number' ? to.toLocaleString('en-US') : String(to);
  el.setAttribute('aria-label', str);
  const cols = buildColumns(el, str);
  if (reduced() || from === to) { cols.forEach((c) => reelRest(c, c.digit)); return () => {}; }
  const o = reelTokens(el);
  const fromDigits = (typeof from === 'number' ? String(from) : String(from).replace(/\D/g, '')).padStart(cols.length, '0').slice(-cols.length);
  cols.forEach((c, i) => { c.from = +fromDigits[i]; reelRest(c, c.from); if (c.from === c.digit) c.done = true; });
  const t0 = performance.now();
  let raf = 0;
  const tick = () => {
    const t = performance.now() - t0;
    let running = false;
    cols.forEach((c, i) => {
      if (c.done) return;
      const local = t - i * o.stagger;
      if (local <= 0) { running = true; return; }
      const total = (c.digit - c.from + 10) % 10;
      const pos = c.from + total * o.ease(local / (total * o.stepDur));
      const enter = Math.min(1, local / ENTER_MS);
      if (c.landAt === undefined && c.from + total - pos < LAND_AT) c.landAt = t;
      const since = c.landAt === undefined ? -1 : t - c.landAt;
      if (since >= o.anticDur + o.landDur) { reelRest(c, c.digit); c.done = true; return; }
      running = true;
      const middle = 1 + (o.centerScale - 1) * enter;
      let spread = enter, land = null;
      if (since >= 0) {
        if (!c.landFrom) c.landFrom = reelLook(c.from + total - pos, middle, o);
        const arrive = 1 - smooth(since / (o.anticDur + o.landDur * ARRIVE_AT));
        if (since < o.anticDur) {
          const a = smooth(since / o.anticDur);
          land = { scale: c.landFrom.scale + (o.anticScale - c.landFrom.scale) * a, lift: -o.anticLift * a, opacity: c.landFrom.back + (1 - c.landFrom.back) * a, arrive };
        } else {
          const sp = since - o.anticDur, sn = sp / o.landDur, w = wave(sn);
          land = { scale: sn < 1 / 6 ? 1 + (o.anticScale - 1) * w : 1 - ((o.landPeak - 1) * w) / WAVE_TROUGH, lift: -o.anticLift * w, opacity: 1, arrive };
          spread = 1 - Math.min(1, sp / CLEAR_MS);
        }
      }
      reelPlace(c, pos, spread, middle, o, land);
    });
    if (running) raf = requestAnimationFrame(tick);
  };
  tick();
  return () => cancelAnimationFrame(raf);
}
// A number or text such as "41.1%" that rolls when it changes. Digits are built by countTo, so Preact renders no children.
export function Reel({ value, class: cls }) {
  const ref = useRef();
  const last = useRef(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (last.current == null) {
      const str = typeof value === 'number' ? value.toLocaleString('en-US') : String(value);
      buildColumns(el, str).forEach((c) => reelRest(c, c.digit));
      el.setAttribute('aria-label', str);
    } else if (last.current !== value) {
      if (el._cancel) el._cancel();
      el._cancel = countTo(el, last.current, value);
    }
    last.current = value;
  }, [value]);
  useEffect(() => () => { if (ref.current && ref.current._cancel) ref.current._cancel(); }, []);
  return html`<span class=${'t-reel ' + (cls || '')} ref=${ref}></span>`;
}

/* ---------- transitions.dev: streaming text ---------- */
// Wraps the words of el in spans and resolves them one by one. Returns a cancel function.
export function streamText(el, maxMs = 1800) {
  const words = el.textContent.trim().split(/\s+/);
  if (reduced() || words.length < 2) return () => {};
  el.textContent = '';
  const spans = words.map((w, i) => {
    const s = document.createElement('span');
    s.className = 't-stream-w';
    s.textContent = w;
    el.appendChild(s);
    if (i < words.length - 1) el.appendChild(document.createTextNode(' '));
    return s;
  });
  const gap = Math.min(tokenNum('--stream-gap', 60), maxMs / spans.length);
  let n = 0, timer = 0;
  reflow();
  (function next() {
    if (n >= spans.length) return;
    spans[n++].classList.add('is-in');
    timer = setTimeout(next, gap);
  })();
  return () => clearTimeout(timer);
}
// Text that streams in when it first appears. Set play to false for text that was already there.
export function StreamText({ text, play = true, class: cls, tag = 'div' }) {
  const ref = useRef();
  useLayoutEffect(() => {
    if (!play || !ref.current) return;
    return streamText(ref.current);
  }, []);
  return html`<${tag} class=${'t-stream ' + (cls || '')} ref=${ref}>${text}<//>`;
}

/* ---------- transitions.dev: shimmer text ---------- */
export function Shimmer({ text, tag = 'span', class: cls }) {
  return html`<${tag} class=${'t-shimmer ' + (cls || '')} data-text=${text}>${text}<//>`;
}

/* ---------- transitions.dev: thinking states ---------- */
// A status line that shimmers while it holds and swaps when the text prop changes.
// Set still when the work is idle, so the line stops shimmering. Set fluid to truncate in a flex row.
export function ThinkLine({ text, class: cls, still, fluid }) {
  const box = useRef();
  const live = useRef(null);
  const sizer = useRef();
  const make = (t, cls2) => {
    const el = document.createElement('span');
    el.className = 't-think-text' + (cls2 ? ' ' + cls2 : '');
    el.textContent = t;
    el.setAttribute('data-text', t);
    box.current.appendChild(el);
    return el;
  };
  useLayoutEffect(() => {
    sizer.current.textContent = text;
    if (!live.current) { live.current = make(text); return; }
    if (live.current.textContent === text) return;
    if (reduced()) { live.current.textContent = text; live.current.setAttribute('data-text', text); return; }
    const swap = tokenMs('--think-swap', 150), gap = tokenMs('--think-gap', 50);
    const leaving = live.current;
    leaving.classList.add('is-exit');
    const next = make(text, 'is-enter-start');
    live.current = next;
    const release = () => { void next.offsetWidth; next.classList.remove('is-enter-start'); };
    if (gap > 0) setTimeout(release, gap); else release();
    setTimeout(() => leaving.remove(), swap + gap + 20);
  }, [text]);
  return html`<span class=${'t-think ' + (still ? 'is-still ' : '') + (fluid ? 'fluid ' : '') + (cls || '')} role="status" ref=${box}><span class="t-think-sizer" aria-hidden="true" ref=${sizer}></span></span>`;
}

/* ---------- transitions.dev: error state shake ---------- */
export function shake(el) {
  if (!el || reduced()) return;
  const ms = tokenNum('--shake-dur-a', 80) * 2 + tokenNum('--shake-dur-b', 60) * 2;
  el.classList.remove('is-shaking');
  void el.offsetWidth;
  el.classList.add('is-shaking');
  clearTimeout(el._shakeTimer);
  el._shakeTimer = setTimeout(() => el.classList.remove('is-shaking'), ms + 20);
}
// Returns a ref. The element shakes each time key changes to a truthy value (not on first render).
export function useShake(key) {
  const ref = useRef();
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    if (key) shake(ref.current);
  }, [key]);
  return ref;
}

/* ---------- transitions.dev: success check ---------- */
export function SuccessCheck({ play = true, size = 18, class: cls }) {
  return html`<span class=${'t-success-check ' + (play ? '' : 'is-static ') + (cls || '')} data-state="in" aria-hidden="true"><svg width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg></span>`;
}

/* ---------- transitions.dev: matrix loader ---------- */
const CORNERS = [0, 3, 12, 15], RING = [1, 2, 7, 11, 14, 13, 8, 4], INNER = [5, 6, 9, 10];
const TWINKLE = [7, 2, 11, 5, 14, 9, 0, 12, 3, 15, 6, 10, 13, 1, 8, 4];
export function Matrix({ variant = 'scan', rounded = true }) {
  const cycle = tokenNum('--matrix-cycle', 1200);
  const dots = [];
  for (let idx = 0; idx < 16; idx++) {
    const col = idx % 4;
    if (rounded && CORNERS.includes(idx)) { dots.push(html`<i class="is-gap"></i>`); continue; }
    let d = 0, still = false;
    if (variant === 'scan') d = col * (cycle / 10);
    else if (variant === 'twinkle') d = TWINKLE[idx] * (cycle / 16);
    else if (variant === 'orbit') { const k = RING.indexOf(idx); if (k === -1) still = true; else d = k * (cycle / 8); }
    else if (variant === 'pulse') d = INNER.includes(idx) ? 0 : cycle * 0.16;
    dots.push(html`<i style=${{ '--d': Math.round(d), animation: still ? 'none' : null }}></i>`);
  }
  return html`<span class="t-matrix" data-variant=${variant} aria-hidden="true">${dots}</span>`;
}

/* ---------- transitions.dev: tabs sliding ---------- */
// A bar whose active child (.on or aria-selected) gets a pill that slides to it.
export function SlidingBar({ class: cls, children, ...rest }) {
  const box = useRef();
  const pill = useRef();
  const ready = useRef(false);
  const place = (animate) => {
    const b = box.current, p = pill.current;
    if (!b || !p) return;
    const a = b.querySelector('.on, [aria-selected="true"]');
    if (!a) { p.style.opacity = '0'; return; }
    const apply = () => {
      p.style.opacity = '1';
      p.style.transform = 'translate(' + a.offsetLeft + 'px,' + a.offsetTop + 'px)';
      p.style.width = a.offsetWidth + 'px';
      p.style.height = a.offsetHeight + 'px';
    };
    if (animate && ready.current) { apply(); return; }
    const prev = p.style.transition;
    p.style.transition = 'none';
    apply();
    void p.offsetWidth;
    p.style.transition = prev;
    ready.current = true;
  };
  useLayoutEffect(() => { place(true); });
  useEffect(() => {
    if (!window.ResizeObserver) return;
    const ro = new ResizeObserver(() => place(false));
    ro.observe(box.current);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => place(false));
    return () => ro.disconnect();
  }, []);
  return html`<div class=${'t-slide ' + (cls || '')} ref=${box} ...${rest}><span class="t-tabs-pill" ref=${pill} aria-hidden="true"></span>${children}</div>`;
}

/* ---------- transitions.dev: card resize ---------- */
// Wraps content and tweens its height when the content grows or shrinks.
export function AutoHeight({ children, class: cls }) {
  const wrap = useRef();
  const inner = useRef();
  useLayoutEffect(() => {
    const w = wrap.current, i = inner.current;
    w.style.height = i.offsetHeight + 'px';
    if (!window.ResizeObserver) return;
    let first = true;
    const ro = new ResizeObserver(() => {
      if (first) { first = false; w.classList.add('t-resize'); }
      w.style.height = i.offsetHeight + 'px';
    });
    ro.observe(i);
    return () => ro.disconnect();
  }, []);
  return html`<div class=${'t-autoh ' + (cls || '')} ref=${wrap}><div ref=${inner}>${children}</div></div>`;
}

/* ---------- transitions.dev: skeleton reveal ---------- */
// Shows the skeleton until ready, then cross-fades into the children.
export function SkelReveal({ ready, skeleton, children, fill }) {
  const [ph, setPh] = useState(ready ? 'done' : 'load');
  // keep the last skeleton so it can fade out after the caller stops passing it
  const sk = useRef(null);
  if (skeleton) sk.current = skeleton;
  // the first ready render already shows the content at rest, so it can be measured
  const eff = ready && ph === 'load' ? 'a' : ph;
  useLayoutEffect(() => {
    if (!ready || ph !== 'load') return;
    if (reduced()) setPh('done');
    else { reflow(); setPh('b'); }
  }, [ready, ph]);
  useEffect(() => {
    if (ph !== 'b') return;
    const t = setTimeout(() => setPh('done'), tokenMs('--reveal-dur', 400) + 60);
    return () => clearTimeout(t);
  }, [ph]);
  return html`<div class=${'t-skel ph-' + eff + (fill ? ' fill' : '')}>
    <div class="t-skel-content">${ready ? children : null}</div>
    ${eff !== 'done' && html`<div class=${'t-skel-skeleton' + (eff === 'load' ? ' is-pulsing' : '')}>${sk.current}</div>`}
  </div>`;
}

/* ---------- transitions.dev: page side-by-side ---------- */
// Keeps the old page on screen while the new one slides in. depth decides the direction.
export function PageSwap({ pageKey, depth, children }) {
  const st = useRef({ key: pageKey, depth, pages: [{ key: pageKey, vnode: children, state: 'active', settled: true, dir: 0 }], timer: 0 });
  const [, tick] = useState(0);
  const s = st.current;
  if (s.key !== pageKey) {
    const fwd = depth > s.depth;
    const old = s.pages.find((p) => p.key === s.key);
    if (reduced()) s.pages = [{ key: pageKey, vnode: children, state: 'active', settled: true, dir: 0 }];
    else s.pages = [{ ...old, state: 'exit', settled: false, dir: fwd ? -1 : 1 }, { key: pageKey, vnode: children, state: 'enter', settled: false, dir: fwd ? 1 : -1 }];
    s.key = pageKey; s.depth = depth;
    clearTimeout(s.timer);
  } else {
    const cur = s.pages.find((p) => p.key === pageKey);
    if (cur) cur.vnode = children;
  }
  useLayoutEffect(() => {
    const e = s.pages.find((p) => p.state === 'enter');
    if (!e) return;
    reflow();
    e.state = 'active';
    tick((n) => n + 1);
    s.timer = setTimeout(() => {
      s.pages = s.pages.filter((p) => p.state !== 'exit');
      s.pages.forEach((p) => { p.settled = true; });
      tick((n) => n + 1);
    }, Math.max(tokenMs('--page-slide-dur', 250), tokenMs('--page-fade-dur', 250)) + 60);
  }, [pageKey]);
  return html`<div class="t-page-slide">${s.pages.map((p) => html`<div key=${p.key} class=${'t-page' + (p.state === 'active' ? ' is-active' : '') + (p.state === 'exit' ? ' is-exit' : '') + (p.settled ? ' is-settled' : '')} style=${{ '--t-page-dir': p.dir }}>${p.vnode}</div>`)}</div>`;
}

/* ---------- transitions.dev: tooltip ---------- */
// One shared tooltip. Any element with data-tip="text" (and data-tip-side="top|right|bottom") gets it.
export function initTooltips() {
  if (document.getElementById('t-tt-shared')) return;
  const tt = document.createElement('div');
  tt.id = 't-tt-shared';
  tt.className = 't-tt';
  tt.setAttribute('role', 'tooltip');
  document.body.appendChild(tt);
  let cur = null;
  const hide = () => { tt.classList.remove('is-on'); cur = null; };
  const show = (el) => {
    const text = el.getAttribute('data-tip');
    if (!text) return;
    cur = el;
    const side = el.getAttribute('data-tip-side') || 'top';
    const r = el.getBoundingClientRect();
    // long evidence strings are cut to a readable tip; the full text stays on the part's page
    tt.textContent = text.length > 160 ? text.slice(0, 160).replace(/\s+\S*$/, '') + '…' : text;
    tt.setAttribute('data-side', side);
    const x = side === 'right' ? r.right + 8 : r.left + r.width / 2;
    const y = side === 'right' ? r.top + r.height / 2 : side === 'bottom' ? r.bottom + 8 : r.top - 8;
    tt.style.setProperty('--tt-x', x + 'px');
    tt.style.setProperty('--tt-y', y + 'px');
    tt.classList.add('is-on');
  };
  document.addEventListener('pointerover', (e) => {
    const el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (el === cur) return;
    hide();
    if (el && e.pointerType !== 'touch') show(el);
  });
  document.addEventListener('focusin', (e) => {
    const el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (el && e.target.matches && e.target.matches(':focus-visible')) show(el); else hide();
  });
  document.addEventListener('focusout', hide);
  document.addEventListener('pointerdown', hide);
  document.addEventListener('scroll', hide, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
}

/* ---------- transitions.dev: input clear dissolve ---------- */
// The typed text flies down and fades while a soft streak ignites under each word,
// and the placeholder falls in from above. Resolves when the animation ends.
export function dissolveText({ wrap, ta, mirror, phold, glow }) {
  const text = ta.value;
  if (!text || reduced()) return Promise.resolve();
  const root = document.documentElement;
  const num = (n, fb) => tokenNum(n, fb);
  const total = num('--clear-dur', 1000), outDur = num('--clear-out-dur', 400), inDur = num('--clear-in-dur', 400);
  const outFly = num('--clear-out-fly', 12), inFly = num('--clear-in-fly', 12), blur = num('--clear-blur', 2);
  const delay = num('--glow-delay', 50), peakAt = num('--glow-peak-at', 0.15), spread = num('--glow-spread', 1.5);
  const dark = root.classList.contains('dark');
  const gOp = num('--glow-opacity', 0.42) * (dark ? 2 : 1);
  const ease = (name) => { const m = getComputedStyle(root).getPropertyValue(name).match(/cubic-bezier\(([^)]+)\)/); return bezier(m ? m[1].split(',').map(parseFloat) : [0.22, 1, 0.36, 1]); };
  const easeOut = ease('--clear-out-ease'), easeIn = ease('--clear-in-ease');
  const keepFocus = document.activeElement === ta;
  wrap.style.minHeight = wrap.offsetHeight + 'px';
  mirror.textContent = text;
  wrap.classList.add('is-clearing');
  // one glow streak per word, placed from the word's real position in the mirror
  const wr = wrap.getBoundingClientRect();
  const rgb = dark ? '255,255,255' : '0,0,0';
  const layers = [];
  const node = mirror.firstChild;
  const range = document.createRange();
  const re = /\S+/g;
  let m, count = 0;
  while ((m = re.exec(text)) && count++ < 60 && node) {
    range.setStart(node, m.index);
    range.setEnd(node, m.index + m[0].length);
    const rects = range.getClientRects();
    const r = rects[rects.length - 1];
    if (!r) continue;
    const cx = r.left + r.width / 2 - wr.left, by = r.bottom - wr.top - 2;
    const hw = Math.max(r.width * 0.45, 8) * spread;
    [[0, 0.8, 7, 0.22], [hw * 0.45, 0.55, 8, 0.18], [-hw * 0.4, 0.65, 6, 0.16], [hw * 0.15, 0.9, 5, 0.14]].forEach(([dx, rw, rh, a]) => {
      layers.push('radial-gradient(ellipse ' + Math.max(hw * rw, 2).toFixed(1) + 'px ' + rh + 'px at ' + (cx + dx).toFixed(1) + 'px ' + by.toFixed(1) + 'px, rgba(' + rgb + ',' + a + '), transparent)');
    });
  }
  glow.style.background = layers.join(', ');
  glow.style.mixBlendMode = dark ? 'screen' : 'multiply';
  glow.style.opacity = '0';
  phold.style.transform = 'translateY(' + -inFly + 'px)';
  phold.style.opacity = '0.9';
  phold.style.filter = 'blur(' + blur + 'px)';
  return new Promise((resolve) => {
    const t0 = performance.now();
    (function tick(now) {
      const el = now - t0;
      const eo = easeOut(Math.min(1, el / outDur));
      mirror.style.transform = 'translateY(' + (eo * outFly).toFixed(1) + 'px)';
      mirror.style.opacity = (1 - eo).toFixed(3);
      mirror.style.filter = 'blur(' + (eo * blur).toFixed(1) + 'px)';
      const ei = easeIn(Math.min(1, el / inDur));
      phold.style.transform = 'translateY(' + (-inFly + ei * inFly).toFixed(1) + 'px)';
      phold.style.opacity = (0.9 + ei * 0.1).toFixed(3);
      phold.style.filter = 'blur(' + (blur - ei * blur).toFixed(1) + 'px)';
      let g = 0;
      if (el > delay) {
        const gp = Math.min(1, (el - delay) / Math.max(1, total - delay));
        g = gp < peakAt ? gp / peakAt : 1 - (gp - peakAt) / (1 - peakAt);
      }
      glow.style.opacity = (g * gOp).toFixed(3);
      if (el < total) { requestAnimationFrame(tick); return; }
      wrap.classList.remove('is-clearing');
      wrap.style.minHeight = '';
      [mirror, phold].forEach((x) => { x.style.cssText = ''; });
      mirror.textContent = '';
      glow.style.cssText = '';
      if (keepFocus && ta.isConnected && !ta.disabled) ta.focus({ preventScroll: true });
      resolve();
    })(performance.now());
  });
}
