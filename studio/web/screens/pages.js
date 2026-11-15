import { html, useState, useEffect } from '../lib.js';
import { api } from '../api.js';
import { pct, donorColor, donorLetter, donorShort, useRecent, patchRecent, dur } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Dot, StubBadge } from '../components.js';

const RUN_LABEL = { starting: 'Starting', running: 'Running', paused: 'Paused', stopping: 'Stopping', done: 'Finished', error: 'Error', gone: 'Not on the server' };
const RUN_COLOR = { starting: 'var(--blue)', running: 'var(--blue)', paused: 'var(--amber)', stopping: 'var(--amber)', done: 'var(--green)', error: 'var(--red)', gone: 'var(--text-4)' };
const LIVE = { starting: 1, running: 1, paused: 1, stopping: 1 };

function ago(ts) {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return 'just now';
  if (s < 3600) return Math.round(s / 60) + ' min ago';
  if (s < 86400) return Math.round(s / 3600) + ' h ago';
  return Math.round(s / 86400) + ' d ago';
}

export function RunsScreen() {
  const recent = useRecent();
  const runs = recent.filter((r) => r.kind === 'run');
  const ids = runs.map((r) => r.id).join(',');
  // refresh the status of runs that were still live the last time we saw them
  useEffect(() => {
    let dead = false;
    runs.filter((r) => !r.status || LIVE[r.status]).forEach((r) => {
      api.getRun(r.id).then((s) => { if (!dead && s) patchRecent('run', r.id, { status: s.status, evals: s.budget && s.budget.evals_used, total: s.budget && s.budget.evals_total }); }).catch((e) => { if (!dead && e.status === 404) patchRecent('run', r.id, { status: 'gone' }); });
    });
    return () => { dead = true; };
  }, [ids]);
  return html`<${TopBar} />
  <main class="screen"><div class="doc wide">
    <h1 class="page-title">Runs</h1>
    <p class="lede">Runs you opened in this browser, newest first. The status is the last one this page saw.</p>
    ${runs.length === 0
      ? html`<div class="card emptycard"><div class="emptyt">No runs yet</div><div class="faint">Draft a spec and let the agents build it. The run shows up here.</div><a class="btn primary" href="#/">New model</a></div>`
      : html`<div class="card list">${runs.map((r) => html`<a class="lrow" key=${r.id} href=${'#/run/' + r.id}>
          <${Icon} n="runs" size=${18} />
          <div class="lmain"><div class="ltitle ell">${r.title || r.id}</div><div class="faint lsub"><span class="mono">${r.id}</span>${r.total ? ' · ' + r.evals + ' of ' + r.total + ' evals' : ''} · opened ${ago(r.ts)}</div></div>
          <span class="pill"><${Dot} color=${RUN_COLOR[r.status] || 'var(--text-4)'} pulse=${r.status === 'running'} />${RUN_LABEL[r.status] || 'Unknown'}</span>
          <${Icon} n="chevron" size=${16} />
        </a>`)}</div>`}
  </div></main>`;
}

function measuredLine(m) {
  if (!m || m.score == null) return null;
  return String(m.task || '').toUpperCase() + ' ' + pct(m.score) + '%' + (m.lora != null ? ', LoRA ' + pct(m.lora) + '%' : '') + (m.host != null ? ', host ' + pct(m.host) + '%' : '');
}

function PartCard({ ic, color, title, sub, meta, measured }) {
  return html`<div class="card pcard">
    <div class="pic" style=${{ '--c': color }}>${ic}</div>
    <div class="pc-main"><div class="pc-t">${title}</div><div class="pc-s">${sub}</div>
      ${meta && html`<div class="pc-m">${meta}</div>`}
      ${measured && html`<div class="pc-meas"><${Icon} n="check" size=${13} sw=${2} />Measured: ${measured}</div>`}</div>
  </div>`;
}

export function PartsScreen() {
  const [cat, setCat] = useState(null);
  const [err, setErr] = useState(null);
  const [q, setQ] = useState('');
  const load = () => { setErr(null); api.catalog().then(setCat).catch(setErr); };
  useEffect(load, []);
  if (err) return html`<${TopBar} /><${ErrorBox} error=${err} onRetry=${load} />`;
  if (!cat) return html`<${TopBar} /><main class="screen"><div class="doc wide"><div class="skeleton" style="height:40px;width:280px"></div><div class="grid3" style="margin-top:24px">${[1, 2, 3, 4, 5, 6].map(() => html`<div class="skeleton" style="height:96px"></div>`)}</div></div></main>`;
  const m = (x) => !q || (x.label + ' ' + (x.donor || '') + ' ' + (x.summary || '') + ' ' + (x.kind || '')).toLowerCase().includes(q.toLowerCase());
  const taps = cat.parts.filter((x) => x.kind === 'tap' && m(x));
  const adds = cat.parts.filter((x) => x.kind !== 'tap' && m(x));
  const rms = cat.masks.filter(m);
  const trs = cat.train.filter(m);
  const none = !taps.length && !adds.length && !rms.length && !trs.length;
  const layers = (p) => 'L' + p.layers[0] + (p.layers[1] !== p.layers[0] ? ' to L' + p.layers[1] : '');
  return html`<${TopBar} />
  <main class="screen"><div class="doc wide">
    <div class="ph"><div><h1 class="page-title">Parts library</h1>
      <p class="lede">Host model ${cat.host.model.split('/').pop()}, ${cat.host.layers} layers, ${cat.host.params_b.toFixed(2)}B parameters. Drag these onto it in the builder.${cat.stub ? ' ' : ''}${cat.stub && html`<${StubBadge} small />`}</p></div>
      <div class="searchbox pq"><${Icon} n="search" size=${16} /><input placeholder="Search parts" aria-label="Search parts" value=${q} onInput=${(e) => setQ(e.target.value)} /></div></div>
    ${taps.length > 0 && html`<section class="sect"><h2 class="sec"><span>Tap a bigger model</span><span class="faint cnt">${taps.length}</span></h2><div class="grid3">${taps.map((p) => html`<${PartCard} key=${p.id} ic=${donorLetter(p.donor, p.label)} color="var(--blue)" title=${donorShort(p.donor)} sub=${'reads layer ' + p.donor_layer + ', joins at L' + p.default_at} meta=${'+' + p.params_m + 'M trained'} measured=${measuredLine(p.measured)} />`)}</div></section>`}
    ${adds.length > 0 && html`<section class="sect"><h2 class="sec"><span>Add a skill</span><span class="faint cnt">${adds.length}</span></h2><div class="grid3">${adds.map((p) => html`<${PartCard} key=${p.id} ic=${donorLetter(p.donor, p.label)} color=${donorColor(p.donor)} title=${p.label} sub=${donorShort(p.donor) + ' · ' + p.kind + ' · ' + layers(p)} meta=${'+' + p.params_m + 'M parameters, joins at L' + p.default_at} measured=${measuredLine(p.measured)} />`)}</div></section>`}
    ${rms.length > 0 && html`<section class="sect"><h2 class="sec"><span>Remove a skill</span><span class="faint cnt">${rms.length}</span></h2><div class="grid3">${rms.map((p) => html`<${PartCard} key=${p.id} ic="-" color="var(--red)" title=${p.label} sub=${p.summary} measured=${measuredLine(p.measured)} />`)}</div></section>`}
    ${trs.length > 0 && html`<section class="sect"><h2 class="sec"><span>Train</span><span class="faint cnt">${trs.length}</span></h2><div class="grid3">${trs.map((p) => html`<${PartCard} key=${p.id} ic=${p.id === 'lora' ? 'R' : 'H'} color=${p.id === 'lora' ? 'var(--amber)' : 'var(--green)'} title=${p.label} sub=${p.summary} />`)}</div></section>`}
    ${none && html`<div class="empty" style="margin-top:30px">Nothing matches "${q}".</div>`}
  </div></main>`;
}
