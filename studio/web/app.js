import { html, render, useEffect } from './lib.js';
import { useRoute, markStub, setHealth } from './util.js';
import { api } from './api.js';
import { Toasts } from './components.js';
import { PromptScreen } from './screens/prompt.js';
import { SpecScreen } from './screens/spec.js';
import { RunScreen } from './screens/run.js';
import { BuilderScreen } from './screens/builder.js';

function App() {
  const r = useRoute();
  useEffect(() => { api.health().then((h) => { setHealth(h); if (h && h.stub) markStub(true); }).catch(() => {}); }, []);
  let screen;
  if (r.name === 'spec') screen = html`<${SpecScreen} key=${r.id} id=${r.id} />`;
  else if (r.name === 'run') screen = html`<${RunScreen} key=${r.id} id=${r.id} agent=${r.agent} />`;
  else if (r.name === 'build') screen = html`<${BuilderScreen} key=${r.id} id=${r.id} specId=${r.q.get('spec')} />`;
  else screen = html`<${PromptScreen} />`;
  return html`${screen}<${Toasts} />`;
}

render(html`<${App} />`, document.getElementById('app'));
