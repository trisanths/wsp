import { html, useState, useEffect, useRef } from '../lib.js';
import { DEMO, setDemo, demoSpeed, setDemoSpeed } from '../demo-flag.js';
import { go, useTheme } from '../util.js';
import { TopBar, Icon } from '../components.js';

export const SCENES = [
  { id: 'home', section: 'Start', title: 'Home with the prompt typed out', line: 'The home screen types a sample request into the box, the way a first visit would.', hash: '#/?type=1' },
  { id: 'spec-research', section: 'Start', title: 'Spec with research working', line: 'A new spec while the benchmark research agent is still searching and reading sources.', hash: '#/spec/sp_demo_legal' },
  { id: 'spec-ready', section: 'Start', title: 'Spec ready with sliders and benchmarks', line: 'A finished spec with the limits set and recommended benchmarks with their sources.', hash: '#/spec/sp_demo_math' },
  { id: 'run-live', section: 'Agents build it', title: 'Live run overview', line: 'A run partway through its evals, with new leaders appearing as agents score proposals.', hash: '#/run/run_demo_live' },
  { id: 'run-activity', section: 'Agents build it', title: 'Live run activity', line: 'The running feed of what every agent is thinking, proposing and scoring.', hash: '#/run/run_demo_live/activity' },
  { id: 'run-agent', section: 'Agents build it', title: 'One agent at work', line: 'A single agent: its reasoning, the parts it picked and the attempts it scored.', hash: '#/run/run_demo_live/agents/planner-1' },
  { id: 'run-results', section: 'Agents build it', title: 'Results leaderboard', line: 'The finished run ranked against the comparison rows: base, LoRA, distillation and random parts.', hash: '#/run/run_demo_done/results' },
  { id: 'run-done', section: 'Agents build it', title: 'Finished run overview', line: 'A completed run with a clear winner that beats every comparison row.', hash: '#/run/run_demo_done' },
  { id: 'run-empty', section: 'Agents build it', title: 'Run that scored nothing', line: 'What you see when every proposal was refused by the limits, and what to try next.', hash: '#/run/run_demo_empty' },
  { id: 'build-canvas', section: 'Build it yourself', title: 'Builder canvas', line: 'A placed program of parts on the canvas, with scored attempts and the comparison.', hash: '#/build/b_demo?spec=sp_demo_math' },
  { id: 'build-code', section: 'Build it yourself', title: 'Builder code view', line: 'The same program as code, with the result of each stage.', hash: '#/build/b_demo?spec=sp_demo_math&view=code' },
  { id: 'shrink-spec', section: 'Shrink a model', title: 'Shrink spec', line: 'Keep the code skill of a larger model and cut it under a size limit.', hash: '#/spec/sp_demo_code' },
  { id: 'shrink-result', section: 'Shrink a model', title: 'Shrink result', line: 'A finished shrink session with a successful result.', hash: '#/shrink/sh_demo?spec=sp_demo_code' },
  { id: 'parts', section: 'Library', title: 'Parts library', line: 'Every part mined from open models, ready to place in a program.', hash: '#/parts' },
  { id: 'deletions', section: 'Library', title: 'Deletions', line: 'Deletion masks that remove one skill from a model.', hash: '#/parts?kind=deletion_mask' },
  { id: 'measured', section: 'Library', title: 'Measured results', line: 'Numbers measured by the compiler, kept apart from the sample data.', hash: '#/parts?tab=results' },
  { id: 'runs', section: 'Library', title: 'Runs page', line: 'Every run and builder session in one list.', hash: '#/runs' },
];
const SECTIONS = ['Start', 'Agents build it', 'Build it yourself', 'Shrink a model', 'Library'];

function norm(h) {
  const s = (h || '').replace(/^#/, '') || '/';
  const [p, q] = s.split('?');
  const path = p.length > 1 ? p.replace(/\/+$/, '') : '/';
  const params = new URLSearchParams(q || '');
  const keys = [...params.keys()].filter((k) => k !== 'type').sort();
  return path + (keys.length ? '?' + keys.map((k) => k + '=' + params.get(k)).join('&') : '');
}
function sceneIndex(hash) {
  const n = norm(hash);
  return SCENES.findIndex((s) => norm(s.hash) === n);
}
const open = (sc) => { location.hash = sc.hash; };

export function DemoScreen() {
  if (!DEMO) {
    return html`<${TopBar} />
    <main class="screen"><div class="doc wide demo-page">
      <h1 class="page-title">Demo tour</h1>
      <p class="demo-lede">Demo mode fills every screen with sample data and runs without a server, so you can show the whole product from start to finish.</p>
      <div class="demo-cta"><button class="btn primary" onClick=${() => setDemo(true, '#/demo')}>Turn on demo mode</button></div>
      <p class="demo-note">A small Demo label stays in the top bar while it is on, so sample numbers are never read as measured results. You can also press Ctrl+Shift+D (Cmd+Shift+D on a Mac) on any screen.</p>
    </div></main>`;
  }
  return html`<${TopBar} />
  <main class="screen"><div class="doc wide demo-page">
    <h1 class="page-title">Demo tour</h1>
    <p class="demo-lede">Every screen below runs on sample data. Start the tour to walk through them in order, or open any scene on its own. A small control at the bottom right moves between scenes.</p>
    <div class="demo-cta">
      <button class="btn primary" onClick=${() => open(SCENES[0])}>Start the tour</button>
      <button class="btn" onClick=${() => setDemo(false, '#/')}>Turn off demo mode</button>
    </div>
    ${SECTIONS.map((sec) => html`<section class="demo-sec" key=${sec}>
      <h2 class="demo-h">${sec}</h2>
      <div class="demo-grid">${SCENES.filter((s) => s.section === sec).map((s) => html`<div class="card demo-card" key=${s.id}>
        <div class="demo-ct">${s.title}</div>
        <div class="demo-cl">${s.line}</div>
        <a class="demo-open" href=${s.hash}>Open<${Icon} n="arrow" size=${14} /></a>
      </div>`)}</div>
    </section>`)}
  </div></main>`;
}

export function DemoControl() {
  const [hash, setHash] = useState(location.hash);
  const [hidden, setHidden] = useState(false);
  const [list, setList] = useState(false);
  const [more, setMore] = useState(false);
  const [speed, setSpeed] = useState(demoSpeed());
  const [theme, flip] = useTheme();
  const root = useRef();
  const idx = sceneIndex(hash);
  const cur = idx >= 0 ? SCENES[idx] : null;
  const st = useRef({});
  st.current = { idx };

  useEffect(() => {
    const f = () => { setHash(location.hash); setList(false); setMore(false); };
    window.addEventListener('hashchange', f);
    return () => window.removeEventListener('hashchange', f);
  }, []);
  const step = (d) => {
    const i = st.current.idx;
    const next = i < 0 ? (d > 0 ? 0 : SCENES.length - 1) : Math.max(0, Math.min(SCENES.length - 1, i + d));
    if (next !== i) open(SCENES[next]);
  };
  useEffect(() => {
    const onKey = (e) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      const t = e.target;
      const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
      if (e.key === 'ArrowRight' && !typing) { e.preventDefault(); step(1); }
      else if (e.key === 'ArrowLeft' && !typing) { e.preventDefault(); step(-1); }
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

  if (hidden) return null;
  const label = cur ? (idx + 1) + ' of ' + SCENES.length + ' · ' + cur.title : 'Demo tour';
  const pickSpeed = (v) => { setSpeed(v); setDemoSpeed(v); };
  return html`<div class="demo-ctl" ref=${root} role="region" aria-label="Demo tour">
    ${list && html`<div class="demo-pop demo-list" role="menu">${SECTIONS.map((sec) => html`<div key=${sec}>
      <div class="demo-pop-h">${sec}</div>
      ${SCENES.map((s, i) => s.section === sec && html`<button key=${s.id} role="menuitem" class=${'demo-item' + (i === idx ? ' on' : '')} onClick=${() => { setList(false); open(s); }}><span class="num demo-n">${i + 1}</span><span>${s.title}</span></button>`)}
    </div>`)}
      <div class="demo-pop-h">Tour</div>
      <button role="menuitem" class="demo-item" onClick=${() => { setList(false); go('/demo'); }}><span class="demo-n"></span><span>Tour overview</span></button>
    </div>`}
    ${more && html`<div class="demo-pop demo-more">
      <div class="demo-row"><span class="demo-rl">Speed</span><span class="demo-seg" role="group" aria-label="Demo speed">${[1, 2, 4].map((v) => html`<button key=${v} class=${speed === v ? 'on' : ''} aria-pressed=${speed === v} onClick=${() => pickSpeed(v)}>${v}x</button>`)}</span></div>
      ${speed !== demoSpeed() && html`<div class="demo-rn">Applies on the next reload.</div>`}
      <div class="demo-row"><span class="demo-rl">Theme</span><button class="demo-link" onClick=${flip}>${theme === 'dark' ? 'Light' : 'Dark'}</button></div>
      <div class="demo-row"><span class="demo-rl">Control</span><button class="demo-link" onClick=${() => { setHidden(true); setMore(false); }}>Hide (Alt+H)</button></div>
      <div class="demo-row"><span class="demo-rl">Demo mode</span><button class="demo-link" onClick=${() => setDemo(false, '#/')}>Turn off</button></div>
    </div>`}
    <div class="demo-bar">
      <button class="demo-b" aria-label="Previous scene" data-tip="Previous (Alt+Left)" disabled=${idx === 0} onClick=${() => step(-1)}><${Icon} n="left" size=${14} /></button>
      <button class="demo-title" aria-haspopup="menu" aria-expanded=${list} onClick=${() => { setList(!list); setMore(false); }}><span class="demo-lab">${label}</span></button>
      <button class="demo-b" aria-label="Next scene" data-tip="Next (Alt+Right)" disabled=${idx === SCENES.length - 1} onClick=${() => step(1)}><${Icon} n="arrow" size=${14} /></button>
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
