import { html, useState, useEffect } from '../lib.js';
import { api } from '../api.js';
import { pct, donorColor, donorLetter, donorShort, useRecent, patchRecent, dur } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Dot, StubBadge } from '../components.js';
import { SkelReveal } from '../motion.js';
import { ModelLogo } from '../logos.js';
import { StatusTag, groupByKind, lockOf } from '../cmp.js';
import { MeasuredResults } from './measured.js';

const RUN_LABEL = { planned: 'Planned', starting: 'Starting', running: 'Running', paused: 'Paused', stopping: 'Stopping', done: 'Finished', error: 'Error', gone: 'Not on the server' };
const RUN_COLOR = { planned: 'var(--text-4)', starting: 'var(--blue)', running: 'var(--blue)', paused: 'var(--amber)', stopping: 'var(--amber)', done: 'var(--green)', error: 'var(--red)', gone: 'var(--text-4)' };
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
  const runs = recent.filter((r) => r.kind === 'run' || r.kind === 'shrink');
  const ids = runs.map((r) => r.id + (r.status || '')).join(',');
  // refresh the status of runs that were still live the last time we saw them
  useEffect(() => {
    let dead = false;
    runs.filter((r) => r.kind === 'shrink' && r.status === 'running' && r.job).forEach((r) => {
      api.getJob(r.job).then((j) => { if (!dead && j && (j.status === 'done' || j.status === 'error')) patchRecent('shrink', r.id, { status: j.status }); }).catch((e) => { if (!dead && e.status === 404) patchRecent('shrink', r.id, { status: 'gone' }); });
    });
    runs.filter((r) => r.kind === 'run' && (!r.status || LIVE[r.status])).forEach((r) => {
      api.getRun(r.id).then((s) => { if (!dead && s) patchRecent('run', r.id, { status: s.status, evals: s.budget && s.budget.evals_used, total: s.budget && s.budget.evals_total }); }).catch((e) => { if (!dead && e.status === 404) patchRecent('run', r.id, { status: 'gone' }); });
    });
    return () => { dead = true; };
  }, [ids]);
  return html`<${TopBar} />
  <main class="screen"><div class="doc wide">
    <h1 class="page-title">Runs</h1>
    <p class="lede">Runs and shrinks you opened in this browser, newest first. The status is the last one this page saw.</p>
    ${runs.length === 0
      ? html`<div class="card emptycard"><div class="emptyt">No runs yet</div><div class="faint">Draft a spec and let the agents build it. The run shows up here.</div><a class="btn primary" href="#/">New model</a></div>`
      : html`<div class="card list">${runs.map((r) => html`<a class="lrow" key=${r.kind + r.id} href=${r.kind === 'shrink' ? '#/shrink/' + r.id + (r.spec ? '?spec=' + r.spec : '') : '#/run/' + r.id}>
          <${Icon} n=${r.kind === 'shrink' ? 'shrink' : 'runs'} size=${18} />
          <div class="lmain"><div class="ltitle ell">${r.title || r.id}</div><div class="faint lsub"><span class="mono">${r.id}</span>${r.kind === 'shrink' ? ' · shrink' : ''}${r.total ? ' · ' + r.evals + ' of ' + r.total + ' evals' : ''} · opened ${ago(r.ts)}</div></div>
          <span class="pill"><${Dot} color=${RUN_COLOR[r.status] || 'var(--text-4)'} pulse=${r.status === 'running'} />${RUN_LABEL[r.status] || 'Unknown'}</span>
          <${Icon} n="chevron" size=${16} />
        </a>`)}</div>`}
  </div></main>`;
}

function measuredLine(m) {
  if (!m || m.score == null) return null;
  return String(m.task || '').toUpperCase() + ' ' + pct(m.score) + '%' + (m.lora != null ? ', LoRA ' + pct(m.lora) + '%' : '') + (m.host != null ? ', host ' + pct(m.host) + '%' : '');
}

function PartCard({ lead, title, sub, meta, measured, status, evidence, lock }) {
  return html`<div class=${'card pcard' + (lock ? ' locked' : '')} data-tip=${lock ? lock.why : null} data-tip-side="bottom">
    ${lead}
    <div class="pc-main"><div class="pc-top"><div class="pc-t">${title}</div>${lock && !lock.rejected ? html`<${Icon} n="lock" size=${13} />` : html`<${StatusTag} status=${status} />`}</div><div class="pc-s">${sub}</div>
      ${meta && html`<div class="pc-m">${meta}</div>`}
      ${evidence && html`<div class="pc-ev">${evidence}</div>`}
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
  if (!cat) return html`<${TopBar} /><main class="screen"><div class="doc wide"><${SkelReveal} ready=${false} skeleton=${html`<div class="skeleton" style="height:40px;width:280px"></div><div class="grid3" style="margin-top:24px">${[1, 2, 3, 4, 5, 6].map(() => html`<div class="skeleton" style="height:96px"></div>`)}</div>`} /></div></main>`;
  const m = (x) => !q || (x.label + ' ' + (x.donor || '') + ' ' + (x.summary || '') + ' ' + (x.kind || '')).toLowerCase().includes(q.toLowerCase());
  const adds = cat.parts.filter((x) => m(x));
  const rms = cat.masks.filter(m);
  const trs = cat.train.filter(m);
  const grouped = cat.kinds && cat.kinds.length;
  const partCard = (p) => html`<${PartCard} key=${p.id} lead=${html`<${ModelLogo} model=${p.donor} size=${20} />`} title=${p.label} sub=${donorShort(p.donor) + ' · ' + p.kind + ' · ' + layers(p)} meta=${'+' + p.params_m + 'M parameters, joins at L' + p.default_at} status=${p.status} evidence=${p.evidence} lock=${lockOf(p)} measured=${measuredLine(p.measured)} />`;
  const none = !adds.length && !rms.length && !trs.length;
  const layers = (p) => 'L' + p.layers[0] + (p.layers[1] !== p.layers[0] ? ' to L' + p.layers[1] : '');
  return html`<${TopBar} />
  <main class="screen"><div class="doc wide"><${SkelReveal} ready=${true} skeleton=${null}>
    <div class="ph"><div><h1 class="page-title">Parts library</h1>
      <p class="lede"><${ModelLogo} model=${cat.host.model} size=${16} /> Host model ${cat.host.model.split('/').pop()}, ${cat.host.layers} layers, ${cat.host.params_b.toFixed(2)}B parameters. Drag these onto it in the builder.${cat.stub ? ' ' : ''}${cat.stub && html`<${StubBadge} small />`}</p></div>
      <div class="searchbox pq"><${Icon} n="search" size=${16} /><input placeholder="Search parts" aria-label="Search parts" value=${q} onInput=${(e) => setQ(e.target.value)} /></div></div>
    ${grouped ? groupByKind(adds).map((g) => html`<section class="sect" key=${g.kind}><h2 class="sec"><span>${g.name}</span><span class="faint cnt">${g.items.length}</span></h2><div class="grid3">${g.items.map(partCard)}</div></section>`) : adds.length > 0 && html`<section class="sect"><h2 class="sec"><span>Add a skill</span><span class="faint cnt">${adds.length}</span></h2><div class="grid3">${adds.map(partCard)}</div></section>`}
    ${rms.length > 0 && html`<section class="sect"><h2 class="sec"><span>Remove a skill</span><span class="faint cnt">${rms.length}</span></h2><div class="grid3">${rms.map((p) => html`<${PartCard} key=${p.id} lead=${html`<div class="pic" style=${{ '--c': 'var(--red)' }}>-</div>`} title=${p.label} sub=${p.summary} status=${p.status} evidence=${p.evidence} lock=${p.status === 'rejected' ? lockOf(p) : null} measured=${measuredLine(p.measured)} />`)}</div></section>`}
    ${trs.length > 0 && html`<section class="sect"><h2 class="sec"><span>Train</span><span class="faint cnt">${trs.length}</span></h2><div class="grid3">${trs.map((p) => html`<${PartCard} key=${p.id} lead=${html`<div class="pic" style=${{ '--c': p.id === 'lora' ? 'var(--amber)' : 'var(--green)' }}>${p.id === 'lora' ? 'R' : 'H'}</div>`} title=${p.label} sub=${p.summary} />`)}</div></section>`}
    ${none && html`<div class="empty" style="margin-top:30px">Nothing matches "${q}".</div>`}
    <${MeasuredResults} />
  <//></div></main>`;
}
