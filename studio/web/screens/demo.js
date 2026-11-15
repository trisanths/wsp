import { html, useState, useEffect, useLayoutEffect, useRef } from '../lib.js';
import { DEMO, setDemo, demoSpeed, setDemoSpeed } from '../demo-flag.js';
import { go, useTheme } from '../util.js';
import { TopBar, Icon } from '../components.js';
import { SponsorStrip } from '../logos.js';
import { reduced } from '../motion.js';

// The guided tour. Each step opens a screen (hash), finds one element on it (sel, with an
// optional first-line text match), and puts a white spotlight on it with a caption. `sec` widens
// the target to its section. `click` steps show an animated pointer: clicking the highlighted
// element moves the tour on (the element's own action is held back, so the tour stays on its
// sample data; `pass` lets it through, for tabs that switch the view in place). Look steps move
// on with a click anywhere in the highlight, Next, or the right arrow. In auto mode the tour
// plays itself: it waits on each step for a time set by the caption length (plus `hold`), and on
// click steps the pointer taps before it moves on.
export const STEPS = [
  { ch: 'Ask for a model', hash: '#/?type=1', sel: '.entry-row', say: 'Start from a kind of model, or describe your own in the box below.' },
  { ch: 'Ask for a model', hash: '#/?type=1', sel: '.dock-in .send', click: true, hold: 2500, say: 'The request is plain words: what the model should be good at, and where it runs. Send it, and Sapien drafts a spec.' },

  { ch: 'The spec', hash: '#/spec/sp_demo_legal', sel: 'h2.sec', text: 'Benchmark research', sec: true, say: 'A research agent finds the benchmarks that measure the skill and reads their sources.' },
  { ch: 'The spec', hash: '#/spec/sp_demo_math', sel: 'h2.sec', text: 'Limits', sec: true, say: 'Set the limits on size, speed and heal time. Below them sit the targets from the research and the guards: skills the model must keep.' },
  { ch: 'The spec', hash: '#/spec/sp_demo_math', sel: '.btn.primary.big', text: 'Let agents build it', click: true, say: 'Hand it to a swarm of agents.' },

  { ch: 'Agents build it', hash: '#/run/run_demo_live', sel: 'h2.rh2', text: 'Best model so far', sec: true, say: 'The best program so far. It changes as each eval is scored.' },
  { ch: 'Agents build it', hash: '#/run/run_demo_live', sel: 'a.arow', text: 'Planner 1', click: true, say: 'Each agent has one job: the analyst reads results, planners propose programs, the judge runs the controls. Open one.' },
  { ch: 'Agents build it', hash: '#/run/run_demo_live/agents/planner-1', sel: 'h2.rh2', text: 'Now', sec: true, say: 'What this planner is doing now: the attempt it is on, why it chose it, and the part being tested on the host.' },
  { ch: 'Agents build it', hash: '#/run/run_demo_live/agents/planner-1', sel: 'a.tab', text: 'Activity', click: true, say: 'Open the activity feed.' },
  { ch: 'Agents build it', hash: '#/run/run_demo_live/activity', sel: '.alog', hold: 1500, say: 'Every agent at once, round by round: what each one is thinking, the programs it proposes, and the scores that come back. New lines arrive live.' },
  { ch: 'Agents build it', hash: '#/run/run_demo_live/activity', sel: '.adock', say: 'Nudge the agents in plain words. A planner reads the note before its next attempt.' },
  { ch: 'Agents build it', hash: '#/run/run_demo_live/activity', sel: 'a.tab', text: 'Results', click: true, hold: 1200, say: 'Now the results of a finished run.' },
  { ch: 'Agents build it', hash: '#/run/run_demo_done/results', sel: 'h2.rh2', text: 'Leaderboard', sec: true, say: 'Ranked against the host, a LoRA, distillation and random parts of the same size. The winner has to beat all of them.' },

  { ch: 'Autoresearch', hash: '#/run/run_demo_live/pipeline', sel: '.pl-obj', say: 'The autoresearch view of a live run: the goal in one sentence, how close the best model is to the target, and how much of the budget is used.' },
  { ch: 'Autoresearch', hash: '#/run/run_demo_live/pipeline', sel: '.pl-sh h2', text: 'Now running', sec: true, hold: 2500, say: 'Each planner is testing one program. The 3D view shows the part docking into the host model, and the bar shows how far along the experiment is.' },
  { ch: 'Autoresearch', hash: '#/run/run_demo_live/pipeline', sel: '.pl-sh h2', text: 'Experiment log', sec: true, say: 'Every attempt so far: what changed, the score, and whether it was kept as the new best or thrown away.' },
  { ch: 'Autoresearch', hash: '#/run/run_demo_live/agents', sel: 'section.collab', say: 'The agents hand work to each other: the analyst maps parts, planners propose programs, the judge runs the controls, and they share what they learn.' },
  { ch: 'Autoresearch', hash: '#/run/run_demo_live/findings', sel: 'h2.rh2', text: 'What the agents found', sec: true, say: 'Everything the swarm has learned so far: parts that help, dead ends, and open questions, each with its evidence.' },

  { ch: 'Mine a model', hash: '#/mine', sel: '.mn-submit', say: 'Parts come from mining open models. Add any model by its Hugging Face id and Sapien queues it.' },
  { ch: 'Mine a model', hash: '#/mine/req/mr_demo_live', sel: 'h2.sec', text: 'The donor scan', sec: true, hold: 2500, say: 'An agent reads the weights layer by layer and names candidate parts.' },
  { ch: 'Mine a model', hash: '#/mine/req/mr_demo_live', sel: 'h2.sec', text: 'Candidate parts', sec: true, say: 'Each candidate is switched off and scored on every skill. It is kept only if it hurts its own skill more than random parts of the same size.' },
  { ch: 'Mine a model', hash: '#/mine/model/Qwen/Qwen3-30B-A3B', sel: 'h2.sec', text: 'Where the parts are', sec: true, say: 'Everything mined from one model: where its parts sit, the skills they hold, and how they were found.' },

  { ch: 'Build it yourself', hash: '#/build/b_demo?spec=sp_demo_math', sel: '.builder .canvas', say: 'Or build it yourself: drag parts from the library on the left onto the host model. Scores against the spec sit on the right.' },
  { ch: 'Build it yourself', hash: '#/build/b_demo?spec=sp_demo_math', sel: '.builder button', text: 'Code', click: true, pass: true, say: 'Switch to the code view.' },
  { ch: 'Build it yourself', hash: '#/build/b_demo?spec=sp_demo_math&view=code', sel: '.codepane', say: 'The same program as text, with the result of each stage.' },

  { ch: 'Shrink a model', hash: '#/spec/sp_demo_code', sel: 'h2.sec', text: 'What to remove', sec: true, say: 'Ask for a smaller model that keeps one skill. Sapien picks what to remove.' },
  { ch: 'Shrink a model', hash: '#/spec/sp_demo_code', sel: '.btn.primary.big', click: true, say: 'Plan the shrink.' },
  { ch: 'Shrink a model', hash: '#/shrink/sh_demo?spec=sp_demo_code', sel: 'h2.sec', text: 'Result', sec: true, say: 'What was removed, scored against the original and against a random removal of the same size.' },

  { ch: 'The library', hash: '#/parts', sel: 'button.fchip2', text: 'Deletions', click: true, pass: true, say: 'Every part mined from open models. Filter to deletions.' },
  { ch: 'The library', hash: '#/parts?kind=deletion_mask', sel: 'button.ptab', text: 'Results', click: true, pass: true, hold: 1200, say: 'Each deletion mask removes one skill from a model. Next, the measured results.' },
  { ch: 'The library', hash: '#/parts?tab=results', sel: ['.screen .mcard', '.mtable'], say: 'Numbers measured by the compiler on real weights, kept apart from the sample data.' },
  { ch: 'The library', hash: '#/runs', sel: '.card.list', say: 'Every run, builder session and shrink, in one list. That is the tour.' },
];
const CHAPTERS = [...new Set(STEPS.map((s) => s.ch))];
const chapterStart = (ch) => STEPS.findIndex((s) => s.ch === ch);
// how long auto mode stays on a step once its target is in place
const dwell = (s) => (s.click ? 2400 : 2600 + s.say.length * 32) + (s.hold || 0);

/* ---------- tour state: one small store, kept in session storage so a reload keeps the place ---------- */
const TKEY = 'sapien.demo.tour';
function loadTour() {
  try { const v = JSON.parse(sessionStorage.getItem(TKEY)); if (v && Number.isInteger(v.i)) return { on: !!v.on, auto: !!v.auto, i: Math.max(0, Math.min(STEPS.length - 1, v.i)) }; } catch (e) { /* storage can be off */ }
  return { on: false, auto: false, i: 0 };
}
let tour = loadTour();
const subs = new Set();
function setTour(next) {
  tour = { ...tour, ...next };
  try { sessionStorage.setItem(TKEY, JSON.stringify(tour)); } catch (e) { /* storage can be off */ }
  subs.forEach((f) => f(tour));
}
function useTour() {
  const [t, set] = useState(tour);
  useEffect(() => { subs.add(set); return () => subs.delete(set); }, []);
  return t;
}
export function startTour(i = 0, auto = false) { setTour({ on: true, i, auto }); }
const stepTo = (i) => setTour({ on: true, i: Math.max(0, Math.min(STEPS.length - 1, i)) });
const endTour = () => setTour({ on: false, auto: false });
const setAuto = (auto) => setTour({ auto });

function norm(h) {
  const s = (h || '').replace(/^#/, '') || '/';
  const [p, q] = s.split('?');
  const path = p.length > 1 ? p.replace(/\/+$/, '') : '/';
  const params = new URLSearchParams(q || '');
  const keys = [...params.keys()].sort();
  return path + (keys.length ? '?' + keys.map((k) => k + '=' + params.get(k)).join('&') : '');
}
const loose = (h) => norm(h).replace(/[?&]type=1/, '').replace(/\?$/, '');

/* ---------- finding the target ---------- */
const firstLine = (e) => (e.innerText || '').trim().split('\n')[0].trim();
function shown(e) {
  if (e.closest('.tour, .demo-ctl') || e.closest('.t-page:not(.is-active)')) return false;
  const r = e.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}
function findTarget(s) {
  for (const sel of [].concat(s.sel)) {
    let els = [...document.querySelectorAll(sel)].filter(shown);
    // exact first line, then any line (a chip can lead with an icon glyph), then a prefix
    if (s.text) {
      const lines = (e) => (e.innerText || '').split('\n').map((x) => x.trim());
      const rank = (e) => (firstLine(e) === s.text ? 0 : lines(e).includes(s.text) ? 1 : firstLine(e).startsWith(s.text) ? 2 : 3);
      els = els.map((e) => [rank(e), e]).filter(([r]) => r < 3).sort((x, y) => x[0] - y[0]).map(([, e]) => e);
    }
    if (!els.length) continue;
    let e = els[0];
    if (s.sec) e = e.closest('section') || e;
    if (s.up) e = e.closest(s.up) || e;
    return e;
  }
  return null;
}
const PAD = 8;
function boxOf(el) {
  const r = el.getBoundingClientRect();
  const vw = window.innerWidth, vh = window.innerHeight;
  // keep the ring on screen when a section is taller than the window
  const top = Math.max(r.top - PAD, 6), bottom = Math.min(r.bottom + PAD, vh - 6);
  const left = Math.max(r.left - PAD, 6), right = Math.min(r.right + PAD, vw - 6);
  return { x: left, y: top, w: Math.max(right - left, 24), h: Math.max(bottom - top, 24), cx: r.left + r.width / 2, cy: r.top + r.height / 2 };
}
const inside = (b, x, y) => b && x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h;
function offscreen(el) {
  const r = el.getBoundingClientRect();
  return r.top < 60 || r.bottom > window.innerHeight - 24;
}
const playIcon = (on) => on
  ? html`<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>`
  : html`<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5l12 7-12 7z" fill="currentColor"/></svg>`;

/* ---------- the overlay ---------- */
function TourLayer({ i, auto }) {
  const step = STEPS[i];
  const [box, setBox] = useState(() => ({ x: 0, y: 0, w: window.innerWidth, h: window.innerHeight, cx: window.innerWidth / 2, cy: window.innerHeight / 2, none: true }));
  const [lost, setLost] = useState(false);
  const [ready, setReady] = useState(false);
  const [nudge, setNudge] = useState(0);
  const [tip, setTip] = useState({ x: 0, y: 0, below: true });
  const tipRef = useRef();
  const elRef = useRef(null);
  const boxRef = useRef(box);
  boxRef.current = box;
  const last = i === STEPS.length - 1;

  // open the step's screen
  useEffect(() => {
    if (norm(location.hash) !== norm(step.hash)) location.hash = step.hash;
  }, [i]);

  // find the element and follow it every frame (pages slide in, live runs move)
  useEffect(() => {
    let el = null, raf = 0, alive = true, scrolled = false, prev = '';
    const t0 = performance.now();
    setLost(false); setReady(false); elRef.current = null;
    const frame = (now) => {
      if (!alive) return;
      if (!el || !el.isConnected || !shown(el)) {
        el = findTarget(step);
        elRef.current = el;
        scrolled = false;
      }
      if (el) {
        if (!scrolled) { scrolled = true; if (offscreen(el)) el.scrollIntoView({ block: 'center', behavior: reduced() ? 'auto' : 'smooth' }); }
        const b = boxOf(el);
        const key = [b.x, b.y, b.w, b.h].map(Math.round).join(',');
        if (key !== prev) { prev = key; setBox(b); }
        setLost(false); setReady(true);
      } else if (now - t0 > 4000) {
        setLost(true); setReady(true);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [i]);

  // auto mode: wait out the glide and the caption, tap on click steps, then move on
  useEffect(() => {
    if (!auto || !ready) return;
    const t = setTimeout(() => {
      if (step.pass && elRef.current) elRef.current.click();
      if (last) setAuto(false); else stepTo(i + 1);
    }, dwell(step));
    return () => clearTimeout(t);
  }, [auto, ready, i]);

  // place the caption under the highlight, or over it when there is no room
  useLayoutEffect(() => {
    const t = tipRef.current;
    if (!t) return;
    const vw = window.innerWidth, vh = window.innerHeight;
    const w = t.offsetWidth, h = t.offsetHeight, gap = 14;
    const b = lost ? { x: vw / 2, y: vh / 2, w: 0, h: 0 } : box;
    let x = Math.max(12, Math.min(vw - w - 12, b.x + b.w / 2 - w / 2)), y, below = true, clip = null;
    if (b.y + b.h + gap + h < vh - 12) y = b.y + b.h + gap;
    else if (b.y - gap - h > 12) { y = b.y - gap - h; below = false; }
    else if (vw - (b.x + b.w) - gap - 12 >= w || b.x - gap - 12 >= w) {
      // a tall highlight leaves no room above or below: put the caption beside it
      y = Math.min(vh - h - 64, Math.max(12, b.y + 12));
      x = vw - (b.x + b.w) - gap - 12 >= w ? b.x + b.w + gap : b.x - gap - w;
    } else {
      // no room beside it either: end the highlight above the caption, so the caption never covers it
      y = vh - h - 60;
      clip = y - gap;
    }
    setTip({ x, y, below, clip });
  }, [box, lost, i, auto]);

  // clicks: on the highlight they move the tour on, elsewhere they are held so the tour stays put
  useEffect(() => {
    const ours = (e) => e.target && e.target.closest && e.target.closest('.tour-tip, .demo-ctl');
    const hold = (e) => {
      if (ours(e) || !e.isTrusted) return;
      if (step.pass && !lost && inside(boxRef.current, e.clientX, e.clientY)) return;
      e.preventDefault(); e.stopPropagation();
    };
    const onClick = (e) => {
      // auto mode's own tap on a pass-through tab is a synthetic click: let it reach the tab
      if (ours(e) || !e.isTrusted) return;
      const hit = !lost && inside(boxRef.current, e.clientX, e.clientY);
      if (hit && step.pass) { setTimeout(() => stepTo(i + 1), 0); return; }
      e.preventDefault(); e.stopPropagation();
      if (hit) stepTo(i + 1 < STEPS.length ? i + 1 : i);
      else setNudge((n) => n + 1);
    };
    const onKey = (e) => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const stop = () => { e.preventDefault(); e.stopPropagation(); };
      if (e.key === 'Escape') { stop(); endTour(); }
      else if (e.key === ' ') { stop(); setAuto(!tour.auto); }
      else if (e.key === 'ArrowRight' || e.key === 'Enter') { stop(); if (i + 1 < STEPS.length) stepTo(i + 1); else endTour(); }
      else if (e.key === 'ArrowLeft') { stop(); stepTo(i - 1); }
    };
    document.addEventListener('pointerdown', hold, true);
    document.addEventListener('mousedown', hold, true);
    document.addEventListener('click', onClick, true);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('pointerdown', hold, true);
      document.removeEventListener('mousedown', hold, true);
      document.removeEventListener('click', onClick, true);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [i, lost]);

  // a wrong click flashes the ring
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    if (!nudge) return;
    setFlash(true);
    const t = setTimeout(() => setFlash(false), 420);
    return () => clearTimeout(t);
  }, [nudge]);

  const chIdx = CHAPTERS.indexOf(step.ch);
  const pointer = step.click && !lost && !box.none;
  const hint = auto ? 'Playing · Space to pause' : step.click ? 'Click where the pointer shows' : 'Click the highlight or press →';
  const spot = lost ? { x: window.innerWidth / 2 - 30, y: window.innerHeight / 2 - 30, w: 60, h: 60 }
    : tip.clip != null && !box.none ? { ...box, h: Math.max(24, Math.min(box.h, tip.clip - box.y)) } : box;
  const radius = Math.min(14, spot.h / 2);

  return html`<div class="tour" role="dialog" aria-modal="false" aria-label="Guided tour">
    <div class=${'tour-spot' + (flash ? ' flash' : '') + (box.none && !lost ? ' start' : '')} style=${{ transform: `translate(${spot.x}px, ${spot.y}px)`, width: spot.w + 'px', height: spot.h + 'px', borderRadius: radius + 'px' }}></div>
    <div class=${'tour-hand' + (pointer ? ' on' : '')} style=${{ transform: `translate(${box.cx}px, ${box.cy}px)` }} aria-hidden="true">
      <span class="tour-ripple" key=${'r' + i}></span>
      <svg class="tour-cursor" key=${'c' + i} width="26" height="26" viewBox="0 0 24 24"><path d="M5 3l14 7.5-6.2 1.6L10 18.5z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>
    </div>
    <div class=${'tour-tip' + (tip.below ? ' below' : ' above')} ref=${tipRef} style=${{ transform: `translate(${tip.x}px, ${tip.y}px)` }}>
      <div class="tour-top">
        <span class="tour-ch">${step.ch}</span>
        <span class="tour-n num">${i + 1} / ${STEPS.length}</span>
      </div>
      <div class="tour-say" key=${'s' + i}>${lost ? 'This part of the screen is not showing yet. Press Next to go on.' : step.say}</div>
      <div class="tour-foot">
        <span class="tour-hint">${pointer && !auto ? html`<span class="tour-dot"></span>` : null}${lost ? '' : hint}</span>
        <span class="tour-btns">
          <button class="tour-btn" disabled=${i === 0} onClick=${() => stepTo(i - 1)} aria-label="Back"><${Icon} n="left" size=${14} /></button>
          <button class="tour-btn" onClick=${() => setAuto(!auto)} aria-label=${auto ? 'Pause' : 'Play automatically'} aria-pressed=${auto}>${playIcon(auto)}</button>
          <button class="tour-btn primary" onClick=${() => (last ? endTour() : stepTo(i + 1))}>${last ? 'Finish' : 'Next'}</button>
        </span>
      </div>
      <div class="tour-prog" aria-hidden="true">${CHAPTERS.map((c, k) => html`<i key=${c} class=${k < chIdx ? 'done' : k === chIdx ? 'on' : ''}>${k === chIdx && auto && ready && html`<b key=${'a' + i} style=${{ animationDuration: dwell(step) + 'ms' }}></b>`}</i>`)}</div>
    </div>
  </div>`;
}

/* ---------- the tour page ---------- */
export function DemoScreen() {
  if (!DEMO) {
    return html`<${TopBar} />
    <main class="screen"><div class="doc wide demo-page">
      <h1 class="page-title">Demo tour</h1>
      <p class="demo-lede">Demo mode fills every screen with sample data and runs without a server, so you can show the whole product from start to finish.</p>
      <div class="demo-sp"><${SponsorStrip} /></div>
      <div class="demo-cta"><button class="btn primary" onClick=${() => setDemo(true, '#/demo')}>Turn on demo mode</button></div>
      <p class="demo-note">A small Demo label stays in the top bar while it is on, so sample numbers are never read as measured results. You can also press Ctrl+Shift+D (Cmd+Shift+D on a Mac) on any screen.</p>
    </div></main>`;
  }
  return html`<${TopBar} />
  <main class="screen"><div class="doc wide demo-page">
    <h1 class="page-title">Demo tour</h1>
    <p class="demo-lede">A guided walk through the product on sample data, ${STEPS.length} steps. A white highlight moves to each part of the screen and shows where to click. Click the highlight or press the right arrow to go on, or let it play by itself. Space pauses, Escape leaves the tour.</p>
    <div class="demo-sp"><${SponsorStrip} /></div>
    <div class="demo-cta">
      <button class="btn primary" onClick=${() => startTour(0, false)}>Start the tour</button>
      <button class="btn" onClick=${() => startTour(0, true)}>Play automatically</button>
      <button class="btn" onClick=${() => setDemo(false, '#/')}>Turn off demo mode</button>
    </div>
    <div class="demo-grid demo-chapters">${CHAPTERS.map((ch, k) => {
      const steps = STEPS.filter((s) => s.ch === ch);
      return html`<button class="card demo-card" key=${ch} onClick=${() => startTour(chapterStart(ch), false)}>
        <div class="demo-ct"><span class="num demo-k">${k + 1}</span>${ch}</div>
        <div class="demo-cl">${steps[0].say}</div>
        <span class="demo-open">${steps.length} steps<${Icon} n="arrow" size=${14} /></span>
      </button>`;
    })}</div>
  </div></main>`;
}

/* ---------- the floating control, and the tour layer when the tour is on ---------- */
export function DemoControl() {
  const t = useTour();
  const [hidden, setHidden] = useState(false);
  const [list, setList] = useState(false);
  const [more, setMore] = useState(false);
  const [speed, setSpeed] = useState(demoSpeed());
  const [theme, flip] = useTheme();
  const root = useRef();
  const st = useRef(t);
  st.current = t;

  // when someone leaves the tour's screen another way (back button, address bar), follow if a
  // step lives there, otherwise pause the tour
  useEffect(() => {
    const f = () => {
      setList(false); setMore(false);
      const cur = st.current;
      if (!cur.on) return;
      if (norm(location.hash) === norm(STEPS[cur.i].hash) || loose(location.hash) === loose(STEPS[cur.i].hash)) return;
      // a pass-through tab moves the hash to the next step's screen a moment before the tour does
      if (STEPS[cur.i + 1] && loose(location.hash) === loose(STEPS[cur.i + 1].hash)) return;
      const j = STEPS.findIndex((s) => loose(s.hash) === loose(location.hash));
      if (j >= 0) stepTo(j); else endTour();
    };
    window.addEventListener('hashchange', f);
    return () => window.removeEventListener('hashchange', f);
  }, []);
  useEffect(() => {
    const onKey = (e) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      const cur = st.current;
      if (e.key === 'ArrowRight') { e.preventDefault(); stepTo(cur.on ? cur.i + 1 : cur.i); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); stepTo(cur.on ? cur.i - 1 : cur.i); }
      else if (e.code === 'KeyH') { e.preventDefault(); setHidden((h) => !h); setList(false); setMore(false); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  useEffect(() => {
    if (!list && !more) return;
    const onDown = (e) => { if (root.current && !root.current.contains(e.target)) { setList(false); setMore(false); } };
    const onEsc = (e) => { if (e.key === 'Escape') { setList(false); setMore(false); } };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onEsc);
    return () => { document.removeEventListener('pointerdown', onDown); document.removeEventListener('keydown', onEsc); };
  }, [list, more]);

  const layer = t.on && html`<${TourLayer} i=${t.i} auto=${t.auto} />`;
  if (hidden) return layer;
  const pickSpeed = (v) => { setSpeed(v); setDemoSpeed(v); };
  const curCh = t.on ? STEPS[t.i].ch : null;
  const resumable = t.i > 0 && t.i < STEPS.length - 1;
  return html`${layer}<div class="demo-ctl" ref=${root} role="region" aria-label="Demo tour">
    ${list && html`<div class="demo-pop demo-list" role="menu">
      <div class="demo-pop-h">Chapters</div>
      ${CHAPTERS.map((ch, k) => html`<button key=${ch} role="menuitem" class=${'demo-item' + (ch === curCh ? ' on' : '')} onClick=${() => { setList(false); startTour(chapterStart(ch), t.auto); }}><span class="num demo-n">${k + 1}</span><span>${ch}</span></button>`)}
      <div class="demo-pop-h">Tour</div>
      ${t.on && html`<button role="menuitem" class="demo-item" onClick=${() => { setList(false); endTour(); }}><span class="demo-n"></span><span>Leave the tour (Esc)</span></button>`}
      <button role="menuitem" class="demo-item" onClick=${() => { setList(false); endTour(); go('/demo'); }}><span class="demo-n"></span><span>Tour overview</span></button>
    </div>`}
    ${more && html`<div class="demo-pop demo-more">
      <div class="demo-row"><span class="demo-rl">Speed</span><span class="demo-seg" role="group" aria-label="Demo speed">${[1, 2, 4].map((v) => html`<button key=${v} class=${speed === v ? 'on' : ''} aria-pressed=${speed === v} onClick=${() => pickSpeed(v)}>${v}x</button>`)}</span></div>
      ${speed !== demoSpeed() && html`<div class="demo-rn">Applies on the next reload.</div>`}
      <div class="demo-row"><span class="demo-rl">Theme</span><button class="demo-link" onClick=${flip}>${theme === 'dark' ? 'Light' : 'Dark'}</button></div>
      <div class="demo-row"><span class="demo-rl">Control</span><button class="demo-link" onClick=${() => { setHidden(true); setMore(false); }}>Hide (Alt+H)</button></div>
      <div class="demo-row"><span class="demo-rl">Demo mode</span><button class="demo-link" onClick=${() => { endTour(); setDemo(false, '#/'); }}>Turn off</button></div>
    </div>`}
    <div class="demo-bar">
      ${t.on
        ? html`<button class="demo-b" aria-label="Previous step" data-tip="Back (←)" disabled=${t.i === 0} onClick=${() => stepTo(t.i - 1)}><${Icon} n="left" size=${14} /></button>
          <button class="demo-title" aria-haspopup="menu" aria-expanded=${list} onClick=${() => { setList(!list); setMore(false); }}><span class="demo-lab">${t.i + 1} of ${STEPS.length} · ${curCh}</span></button>
          <button class="demo-b" aria-label=${t.auto ? 'Pause' : 'Play automatically'} data-tip=${t.auto ? 'Pause (Space)' : 'Play (Space)'} onClick=${() => setAuto(!t.auto)}>${playIcon(t.auto)}</button>
          <button class="demo-b" aria-label="Next step" data-tip="Next (→)" disabled=${t.i === STEPS.length - 1} onClick=${() => stepTo(t.i + 1)}><${Icon} n="arrow" size=${14} /></button>`
        : html`<button class="demo-title demo-start" onClick=${() => startTour(resumable ? t.i : 0, false)}><span class="demo-lab">${resumable ? 'Resume the tour' : 'Start the tour'}</span></button>
          <button class="demo-b" aria-label="Play automatically" data-tip="Play automatically" onClick=${() => startTour(resumable ? t.i : 0, true)}>${playIcon(false)}</button>
          <button class="demo-b" aria-label="Chapters" aria-haspopup="menu" aria-expanded=${list} onClick=${() => { setList(!list); setMore(false); }}><${Icon} n="chevron" size=${14} /></button>`}
      <button class="demo-b demo-mb" aria-label="Demo options" aria-expanded=${more} onClick=${() => { setMore(!more); setList(false); }}>${speed}x</button>
    </div>
  </div>`;
}

export function useDemoShortcut() {
  useEffect(() => {
    const f = (e) => {
      if (e.shiftKey && (e.ctrlKey || e.metaKey) && !e.altKey && e.code === 'KeyD') { e.preventDefault(); setDemo(!DEMO); }
    };
    window.addEventListener('keydown', f);
    return () => window.removeEventListener('keydown', f);
  }, []);
}
