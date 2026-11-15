import { html, useState, useEffect, useRef } from './lib.js';
import { api } from './api.js';
import { go, toast, useRoute, useRecent, useTheme, useHealth, useStubFlag, getLastBuild } from './util.js';
import { Icon, Mark, StubBadge, Dot } from './components.js';

const RUN_COLOR = { running: 'var(--blue)', starting: 'var(--blue)', paused: 'var(--amber)', stopping: 'var(--amber)', done: 'var(--green)', error: 'var(--red)' };

function Row({ icon, label, href, onClick, on, rail, badge }) {
  const inner = html`<${Icon} n=${icon} size=${18} /><span class="sb-text">${label}</span>${badge}`;
  return href
    ? html`<a class=${'sb-row' + (on ? ' on' : '')} href=${href} title=${rail ? label : null} aria-label=${label} aria-current=${on ? 'page' : null}>${inner}</a>`
    : html`<button class=${'sb-row' + (on ? ' on' : '')} onClick=${onClick} title=${rail ? label : null} aria-label=${label}>${inner}</button>`;
}

export function Sidebar({ rail, onToggle, route }) {
  const recent = useRecent();
  const [theme, flip] = useTheme();
  const health = useHealth();
  const stub = useStubFlag();
  const modal = health && health.backend === 'modal';
  const openBuilder = async () => {
    const last = getLastBuild();
    if (last && last.id) { go('/build/' + last.id + (last.specId ? '?spec=' + last.specId : '')); return; }
    const spec = recent.find((r) => r.kind === 'spec');
    if (!spec) { toast('Draft a spec first, then pick how to build it.', 'info'); go('/'); return; }
    try {
      const r = await api.createBuilder(spec.id);
      try { sessionStorage.setItem('wsp.budget.' + r.builder_id, JSON.stringify(r.budget)); } catch (e) { /* storage can be off */ }
      go('/build/' + r.builder_id + '?spec=' + spec.id);
    } catch (e) { toast('Could not open the builder: ' + e.message, 'err'); }
  };
  const isRecent = (r) => (r.kind === 'spec' && route.name === 'spec' && route.id === r.id) || (r.kind === 'run' && route.name === 'run' && route.id === r.id);
  return html`<aside class=${'sidebar' + (rail ? ' rail' : '')} aria-label="Sidebar">
    <div class="sb-head">
      <a class="logo" href="#/" title="wsp studio" aria-label="wsp studio home"><${Mark} size=${24} /><span class="logotext sb-text">wsp studio</span></a>
      <button class="rbtn sb-toggle" onClick=${onToggle} title=${rail ? 'Expand sidebar' : 'Collapse sidebar'} aria-label=${rail ? 'Expand sidebar' : 'Collapse sidebar'}><${Icon} n="panel" size=${18} /></button>
    </div>
    <nav class="sb-nav">
      <${Row} icon="plus" label="New model" href="#/" on=${route.name === 'prompt'} rail=${rail} />
      <${Row} icon="runs" label="Runs" href="#/runs" on=${route.name === 'runs'} rail=${rail} />
      <${Row} icon="cube" label="Builder" onClick=${openBuilder} on=${route.name === 'build'} rail=${rail} />
      <${Row} icon="grid" label="Parts library" href="#/parts" on=${route.name === 'parts'} rail=${rail} />
    </nav>
    <div class="sb-label sb-text">Recent</div>
    <div class="sb-recent sb-text">
      ${recent.length === 0 && html`<div class="sb-empty">Specs and runs you open show up here.</div>`}
      ${recent.slice(0, 12).map((r) => html`<a class=${'sb-row sub' + (isRecent(r) ? ' on' : '')} key=${r.kind + r.id} href=${'#/' + r.kind + '/' + r.id} title=${r.title}>
        <${Icon} n=${r.kind === 'run' ? 'runs' : 'file'} size=${16} /><span class="sb-title">${r.title || r.id}</span>
        ${r.kind === 'run' && r.status && html`<${Dot} color=${RUN_COLOR[r.status] || 'var(--text-4)'} />`}</a>`)}
    </div>
    <div class="sb-foot">
      <button class="sb-row" onClick=${flip} title=${theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} aria-label=${theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}><${Icon} n=${theme === 'dark' ? 'sun' : 'moon'} size=${18} /><span class="sb-text">${theme === 'dark' ? 'Light theme' : 'Dark theme'}</span></button>
      <div class="sb-status" title=${modal ? 'Jobs run on Modal GPUs' : stub ? 'Stub backend: numbers are not real results' : 'Backend'}>
        <${Icon} n="server" size=${18} />
        <span class="sb-text">${modal ? 'Modal GPU' : stub ? html`Stub backend <${StubBadge} small />` : health ? health.backend : 'Connecting'}</span>
      </div>
    </div>
  </aside>`;
}

// the sidebar plus the main area. The builder starts with a rail, narrow windows too.
export function Shell({ children }) {
  const route = useRoute();
  const wide = () => window.innerWidth >= 1100;
  const [rail, setRail] = useState(() => route.name === 'build' || !wide());
  const prev = useRef(route.name);
  useEffect(() => {
    if (prev.current !== route.name) {
      if (route.name === 'build') setRail(true);
      else if (prev.current === 'build' && wide()) setRail(false);
      prev.current = route.name;
    }
  }, [route.name]);
  return html`<div class="shell">
    <${Sidebar} rail=${rail} onToggle=${() => setRail(!rail)} route=${route} />
    <div class="main-area">${children}</div>
  </div>`;
}
