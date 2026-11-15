import { html, render, useEffect } from './lib.js';
import { useRoute, markStub, setHealth } from './util.js';
import { api } from './api.js';
import { Toasts } from './components.js';
import { Shell } from './shell.js';
import { PromptScreen } from './screens/prompt.js';
import { SpecScreen } from './screens/spec.js';
import { RunScreen } from './screens/run.js';
import { BuilderScreen } from './screens/builder.js';
import { ShrinkScreen } from './screens/shrink.js';
import { RunsScreen, PartsScreen } from './screens/pages.js';
import { PageSwap, initTooltips } from './motion.js';

// Deeper screens slide in from the right. Equal or shallower depth slides back.
const DEPTH = { prompt: 0, runs: 1, parts: 1, spec: 1, run: 2, build: 2, shrink: 2 };

function App() {
  const r = useRoute();
  useEffect(() => { api.health().then((h) => { setHealth(h); if (h && h.stub) markStub(true); }).catch(() => {}); }, []);
  let screen;
  if (r.name === 'spec') screen = html`<${SpecScreen} key=${r.id} id=${r.id} />`;
  else if (r.name === 'run') screen = html`<${RunScreen} key=${r.id} id=${r.id} page=${r.page} agent=${r.agent} />`;
  else if (r.name === 'build') screen = html`<${BuilderScreen} key=${r.id} id=${r.id} specId=${r.q.get('spec')} />`;
  else if (r.name === 'shrink') screen = html`<${ShrinkScreen} key=${r.id} id=${r.id} specId=${r.q.get('spec')} />`;
  else if (r.name === 'runs') screen = html`<${RunsScreen} />`;
  else if (r.name === 'parts') screen = html`<${PartsScreen} />`;
  else screen = html`<${PromptScreen} />`;
  return html`<${Shell}><${PageSwap} pageKey=${r.name + ':' + (r.id || '')} depth=${DEPTH[r.name] ?? 0}>${screen}<//><//><${Toasts} />`;
}

initTooltips();
render(html`<${App} />`, document.getElementById('app'));
