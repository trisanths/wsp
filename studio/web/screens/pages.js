import { html, useState, useEffect, useMemo, useRef } from '../lib.js';
import { api } from '../api.js';
import { DEMO } from '../demo-flag.js';
import { pct, donorColor, donorLetter, donorShort, useRecent, patchRecent, dur, ago, rowSuffixes, recentStatus, recentSub, sizeM, go, toast, parseHash, showStub } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Dot, StubBadge } from '../components.js';
import { SkelReveal } from '../motion.js';
import { ModelLogo } from '../logos.js';
import { StatusTag, lockOf, filterItems, countItems, FilterRow, NO_FILTER, PAGE_SIZE, Pager, DELETION } from '../cmp.js';
import { openProgramInBuilder } from './shared.js';
import { runRecent } from './run-shared.js';
import { MeasuredResults } from './measured.js';
import { openPart } from './part-detail.js';

const RUN_LABEL = { planned: 'Planned', starting: 'Starting', running: 'Running', paused: 'Paused', stopping: 'Stopping', done: 'Finished', empty: 'Ended, nothing scored', stopped: 'Stopped', error: 'Error', gone: 'Not on the server' };
const RUN_COLOR = { planned: 'var(--text-4)', starting: 'var(--blue)', running: 'var(--blue)', paused: 'var(--amber)', stopping: 'var(--amber)', done: 'var(--green)', empty: 'var(--amber)', stopped: 'var(--text-3)', error: 'var(--red)', gone: 'var(--text-4)' };
const LIVE = { starting: 1, running: 1, paused: 1, stopping: 1 };

export function RunsScreen() {
  const recent = useRecent();
  const runs = recent.filter((r) => r.kind === 'run' || r.kind === 'shrink');
  const suffix = rowSuffixes(runs);
  const ids = runs.map((r) => r.id + (r.status || '')).join(',');
  // refresh the status of runs that were still live the last time we saw them
  useEffect(() => {
    let dead = false;
    runs.filter((r) => r.kind === 'shrink' && r.status === 'running' && r.job).forEach((r) => {
      api.getJob(r.job).then((j) => { if (!dead && j && (j.status === 'done' || j.status === 'error')) patchRecent('shrink', r.id, { status: j.status }); }).catch((e) => { if (!dead && e.status === 404) patchRecent('shrink', r.id, { status: 'gone' }); });
    });
    runs.filter((r) => r.kind === 'run' && (!r.status || LIVE[r.status] || r.status === 'done')).forEach((r) => {
      api.getRun(r.id).then((s) => { if (!dead && s) patchRecent('run', r.id, { sub: recentSub(s.spec), status: runRecent(s), evals: s.budget && s.budget.evals_used, total: s.budget && s.budget.evals_total }); }).catch((e) => { if (!dead && e.status === 404) patchRecent('run', r.id, { status: 'gone' }); });
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
          <div class="lmain"><div class="ltitle ell">${r.title || r.id}${suffix[r.kind + r.id] && html` <span class="sb-sfx">${suffix[r.kind + r.id]}</span>`}</div><div class="faint lsub">${r.sub || (r.kind === 'shrink' ? 'Shrink plan' : 'Agent run')}${r.total ? ' · ' + r.evals + ' of ' + r.total + ' evals' : ''} · opened ${ago(r.ts)}</div></div>
          <span class="pill"><${Dot} color=${RUN_COLOR[r.status] || 'var(--text-4)'} pulse=${r.status === 'running'} />${RUN_LABEL[r.status] || 'Unknown'}</span>
          <${Icon} n="chevron" size=${16} />
        </a>`)}</div>`}
  </div></main>`;
}

function measuredLine(m) {
  if (!m || m.score == null) return null;
  return String(m.task || '').toUpperCase() + ' ' + pct(m.score) + '%' + (m.lora != null ? ', LoRA ' + pct(m.lora) + '%' : '') + (m.host != null ? ', host ' + pct(m.host) + '%' : '');
}

const plural = (n, one, many) => n + ' ' + (n === 1 ? one : many);

function PartCard({ pid, lead, title, sub, meta, measured, status, evidence, lock, findings, onFinding }) {
  const notes = findings || [];
  const open = (e) => { if (pid && !(e.target.closest && e.target.closest('button, summary, a, details'))) openPart(pid); };
  const key = (e) => { if (pid && e.target === e.currentTarget && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openPart(pid); } };
  return html`<div class=${'card pcard' + (pid ? ' openable' : '') + (lock ? ' locked' : '')} data-tip=${lock ? lock.why : null} data-tip-side="bottom" role=${pid ? 'button' : null} tabindex=${pid ? 0 : null} onClick=${open} onKeyDown=${key} aria-label=${pid ? 'Open ' + title : null}>
    ${lead}
    <div class="pc-main"><div class="pc-top"><div class="pc-t">${title}</div>${lock && !lock.rejected ? html`<${Icon} n="lock" size=${13} />` : html`<${StatusTag} status=${status} />`}</div><div class="pc-s">${sub}</div>
      ${meta && html`<div class="pc-m">${meta}</div>`}
      ${evidence && html`<div class="pc-ev">${evidence}</div>`}
      ${lock && !lock.rejected && html`<div class="pc-ev">${lock.why}</div>`}
      ${measured && html`<div class="pc-meas"><${Icon} n="check" size=${13} sw=${2} />Measured: ${measured}</div>`}
      ${notes.length > 0 && html`<details class="pc-note"><summary>${plural(notes.length, 'finding', 'findings')}</summary><ul>${notes.map((f) => html`<li key=${f.id}><button class="lnk" onClick=${() => onFinding(f.id)}>${f.title}</button></li>`)}</ul></details>`}
    </div>
  </div>`;
}

const KIND_ONE = { block: 'block', attn_heads: 'attention heads', mlp_cluster: 'neuron cluster', expert: 'expert', inserted_layer: 'inserted layers' };
const layerText =(p) => (p.layers ? 'L' + p.layers[0] + (p.layers[1] !== p.layers[0] ? ' to L' + p.layers[1] : '') : '');

function PartsTab({ cat, q, onFinding }) {
  const [f, setF] = useState(() => (parseHash().q.get('kind') === DELETION ? { ...NO_FILTER, kind: DELETION } : NO_FILTER));
  const [page, setPage] = useState(1);
  const top = useRef(null);
  useEffect(() => {
    const on = () => { const r = parseHash(); if (r.name !== 'parts') return; setPage(1); setF(r.q.get('kind') === DELETION ? { ...NO_FILTER, kind: DELETION } : NO_FILTER); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const all = useMemo(() => [...cat.parts, ...cat.masks], [cat]);
  const isDel = f.kind === DELETION;
  const counts = useMemo(() => countItems(all, f), [all, isDel]);
  // rejected parts stay out of the list unless the Rejected chip asks for them
  const shown = useMemo(() => filterItems(all, f, q).filter((x) => f.status === 'rejected' || x.status !== 'rejected'), [all, f, q]);
  const inCat = (isDel ? cat.masks : all.filter((x) => !cat.masks.includes(x))).filter((x) => f.status === 'rejected' || x.status !== 'rejected').length;
  useEffect(() => { setPage(1); }, [f, q]);
  const pages = Math.max(1, Math.ceil(shown.length / PAGE_SIZE));
  const at = Math.min(page, pages);
  const goPage = (n) => { setPage(n); if (top.current) top.current.scrollIntoView({ block: 'start', behavior: 'smooth' }); };
  const trains = !f.kind && !f.status && !f.placeable ? cat.train.filter((x) => !q || (x.label + ' ' + x.summary).toLowerCase().includes(q.toLowerCase())) : [];
  const card = (p) => p.group === 'remove'
    ? html`<${PartCard} key=${p.id} pid=${p.id} lead=${html`<div class="pic" style=${{ '--c': 'var(--red)' }}>-</div>`} title=${p.label} sub=${p.summary} status=${p.status} evidence=${p.evidence} lock=${p.status === 'rejected' ? lockOf(p) : null} measured=${measuredLine(p.measured)} findings=${p.findings} onFinding=${onFinding} />`
    : html`<${PartCard} key=${p.id} pid=${p.id} lead=${html`<${ModelLogo} model=${p.donor} size=${20} />`} title=${p.label} sub=${donorShort(p.donor) + ' · ' + (KIND_ONE[p.kind] || p.kind) + ' · ' + layerText(p)} meta=${p.params_m != null ? '+' + sizeM(p.params_m) + ' parameters, joins at L' + p.default_at + (cat.host && p.params_m / 1000 > cat.host.params_b * 2 ? '. Larger than this host.' : '') : null} status=${p.status} evidence=${p.evidence} lock=${lockOf(p)} measured=${measuredLine(p.measured)} findings=${p.findings} onFinding=${onFinding} />`;
  const rows = shown.slice((at - 1) * PAGE_SIZE, at * PAGE_SIZE);
  const what = isDel ? ['deletion', 'deletions'] : ['part', 'parts'];
  return html`<div ref=${top} class="pq-top"></div><${FilterRow} counts=${counts} f=${f} setF=${setF} />
    <div class="faint" style="margin-top:14px;font-size:12px" aria-live="polite">${shown.length === inCat ? plural(inCat, what[0], what[1]) : shown.length + ' of ' + plural(inCat, what[0], what[1])}${isDel ? '. A deletion removes a skill from the host and adds no parameters.' : cat.counts && !DEMO ? ', ' + cat.counts.placeable + ' placeable in the compiler catalog' : ''}</div>
    ${trains.length > 0 && at === 1 && html`<section class="sect"><h2 class="sec"><span>Training steps</span><span class="faint cnt">${trains.length}</span></h2><div class="grid3">${trains.map((p) => html`<${PartCard} key=${p.id} lead=${html`<div class="pic" style=${{ '--c': p.id === 'lora' ? 'var(--amber)' : 'var(--green)' }}>${p.id === 'lora' ? 'R' : 'H'}</div>`} title=${p.label} sub=${p.summary} />`)}</div></section>`}
    <div class="grid3">${rows.map(card)}</div>
    <${Pager} page=${at} total=${shown.length} size=${PAGE_SIZE} onPage=${goPage} label=${isDel ? 'Deletion pages' : 'Part pages'} />
    ${shown.length === 0 && html`<div class="empty" style="margin-top:30px">${q ? 'Nothing matches "' + q + '".' : 'Nothing matches these filters.'}</div>`}`;
}

/** Status chips and the list filter that recipes and findings share. */
function useStatusFilter(items) {
  const [f, setF] = useState(NO_FILTER);
  return [f, setF, (q, text) => items.filter((x) => (!f.status || x.status === f.status) && (!q || text(x).toLowerCase().includes(q.toLowerCase())))];
}

function PartLinks({ ids, names }) {
  const known = (ids || []).filter((id) => names.has(id));
  if (!known.length) return null;
  return known.map((id) => html`<button class="lnk" key=${id} title=${id} onClick=${() => openPart(id)}>${names.get(id)}</button>`);
}

function RecipesTab({ q, names }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const recent = useRecent();
  useEffect(() => { api.recipes().then((d) => setData(d.recipes)).catch(setErr); }, []);
  const [f, setF, pick] = useStatusFilter(data || []);
  if (err) return html`<${ErrorBox} error=${err} onRetry=${() => { setErr(null); api.recipes().then((d) => setData(d.recipes)).catch(setErr); }} />`;
  if (!data) return html`<div class="skeleton" style="height:96px;margin-top:16px"></div>`;
  const counts = countItems(data);
  const rows = pick(q, (r) => r.title + ' ' + r.builds + ' ' + (r.models || []).join(' ') + ' ' + (r.skills || []).join(' '))
    .map((r, i) => [r, r.starting_program ? 0 : 1, i]).sort((a, b) => (a[1] - b[1]) || (a[2] - b[2])).map((p) => p[0]);
  const open = (r) => {
    const spec = recent.find((x) => x.kind === 'spec');
    if (!spec) { toast('Draft a spec first, then open a recipe in the builder.', 'info'); go('/'); return; }
    openProgramInBuilder(spec.id, r.starting_program);
  };
  return html`<${FilterRow} counts=${counts} f=${f} setF=${setF} noKinds />
    <div class="faint" style="margin-top:14px;font-size:12px">Programs that were run, with what was measured. Open one in the builder to change it and run it again.</div>
    <div class="plist">${rows.map((r) => html`<div class="card rcard" key=${r.id}>
      <div class="rc-top"><div><div class="rc-t">${r.title}</div><div class="rc-m">${[donorShort(r.host), (r.skills || []).join(', ')].filter(Boolean).join(' · ')}</div></div><${StatusTag} status=${r.status} /></div>
      <div class="rc-s">${r.builds}</div>
      <div class="rc-m">${r.evidence}</div>
      ${r.status_reason && html`<div class="rc-m">${r.status_reason}</div>`}
      ${(r.components || []).some((id) => names.has(id)) && html`<div class="about">Parts: <${PartLinks} ids=${r.components} names=${names} /></div>`}
      <div class="rc-act">
        ${r.starting_program ? html`<button class="btn sm primary" onClick=${() => open(r)}>Open in builder</button>` : html`<span class="faint" style="font-size:12px">${r.no_start_reason || 'No starting program.'}</span>`}
        ${(r.start_problems || []).length > 0 && html`<span class="faint" style="font-size:12px">${r.start_problems.join(' ')}</span>`}
      </div>
    </div>`)}</div>
    ${rows.length === 0 && html`<div class="empty" style="margin-top:30px">No recipes match.</div>`}`;
}

function NumberList({ numbers }) {
  const keys = Object.keys(numbers || {});
  if (!keys.length) return null;
  const show = (v) => (Array.isArray(v) ? v.join(' to ') : typeof v === 'object' && v !== null ? Object.entries(v).map(([k, x]) => k.replace(/_/g, ' ') + ' ' + show(x)).join(', ') : String(v));
  return html`<div class="nums">${keys.map((k) => html`<span key=${k}>${k.replace(/_/g, ' ')} <b>${show(numbers[k])}</b></span>`)}</div>`;
}

function FindingsTab({ q, focus, names }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => { api.findings().then((d) => setData(d.findings)).catch(setErr); }, []);
  const [f, setF, pick] = useStatusFilter(data || []);
  useEffect(() => {
    if (!data || !focus) return;
    const el = document.getElementById('fd-' + focus);
    if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }, [data, focus]);
  if (err) return html`<${ErrorBox} error=${err} onRetry=${() => { setErr(null); api.findings().then((d) => setData(d.findings)).catch(setErr); }} />`;
  if (!data) return html`<div class="skeleton" style="height:96px;margin-top:16px"></div>`;
  const counts = countItems(data);
  const rows = pick(focus ? '' : q, (r) => r.title + ' ' + r.claim + ' ' + (r.skills || []).join(' ') + ' ' + (r.about.models || []).join(' '));
  return html`<${FilterRow} counts=${counts} f=${f} setF=${setF} noKinds />
    <div class="faint" style="margin-top:14px;font-size:12px">What earlier runs found, with the numbers as printed in the results log.</div>
    <div class="plist">${rows.map((r) => html`<div id=${'fd-' + r.id} class=${'card fcard' + (focus === r.id ? ' flash' : '')} key=${r.id}>
      <div class="rc-top"><div class="rc-t">${r.title}</div><${StatusTag} status=${r.status} /></div>
      <div class="rc-s">${r.claim}</div>
      <${NumberList} numbers=${r.numbers} />
      <div class="rc-m">${r.setup ? r.setup + '. ' : ''}${r.source ? 'Source: ' + r.source : ''}</div>
      ${r.status_reason && html`<div class="rc-m">${r.status_reason}</div>`}
      ${((r.about.models || []).length > 0 || (r.about.parts || []).some((id) => names.has(id))) && html`<div class="about">
        ${(r.about.models || []).map((m) => html`<span key=${m} style="display:inline-flex;gap:5px;align-items:center"><${ModelLogo} model=${m} size=${14} />${m.split('/').pop()}</span>`)}
        ${(r.about.parts || []).some((id) => names.has(id)) && html`<span>Parts:</span>`}
        <${PartLinks} ids=${(r.about.parts || []).filter((id) => names.has(id)).slice(0, 6)} names=${names} />
        ${(r.about.parts || []).filter((id) => names.has(id)).length > 6 && html`<span>and ${(r.about.parts || []).filter((id) => names.has(id)).length - 6} more</span>`}
      </div>`}
    </div>`)}</div>
    ${rows.length === 0 && html`<div class="empty" style="margin-top:30px">No findings match.</div>`}`;
}

const TABS = [['parts', 'Parts', 'Search parts'], ['recipes', 'Recipes', 'Search recipes'], ['findings', 'Findings', 'Search findings'], ['results', 'Results', '']];

export function PartsScreen() {
  const [cat, setCat] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState(() => { const t = parseHash().q.get('tab'); return TABS.some((x) => x[0] === t) ? t : 'parts'; });
  const [qs, setQs] = useState({ parts: '', recipes: '', findings: '' });
  const [focus, setFocus] = useState(null);
  const [tabCounts, setTabCounts] = useState({});
  useEffect(() => {
    const on = () => { const r = parseHash(); if (r.name !== 'parts') return; const t = r.q.get('tab'); setTab(TABS.some((x) => x[0] === t) ? t : 'parts'); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const names = useMemo(() => new Map(cat ? [...cat.parts, ...cat.masks].map((x) => [x.id, x.label || x.id]) : []), [cat]);
  const load = () => { setErr(null); api.catalog().then(setCat).catch(setErr); };
  useEffect(load, []);
  useEffect(() => {
    api.recipes().then((d) => setTabCounts((c) => ({ ...c, recipes: d.recipes.length }))).catch(() => {});
    api.findings().then((d) => setTabCounts((c) => ({ ...c, findings: d.findings.length }))).catch(() => {});
    api.measured().then((d) => setTabCounts((c) => ({ ...c, results: Array.isArray(d) ? d.length : 0 }))).catch(() => {});
  }, []);
  if (err) return html`<${TopBar} /><${ErrorBox} error=${err} onRetry=${load} />`;
  if (!cat) return html`<${TopBar} /><main class="screen"><div class="doc wide"><${SkelReveal} ready=${false} skeleton=${html`<div class="skeleton" style="height:40px;width:280px"></div><div class="grid3" style="margin-top:24px">${[1, 2, 3, 4, 5, 6].map(() => html`<div class="skeleton" style="height:96px"></div>`)}</div>`} /></div></main>`;
  const q = qs[tab];
  const setQ = (v) => setQs({ ...qs, [tab]: v });
  const openFinding = (id) => { setFocus(id); setTab('findings'); };
  const place = TABS.find((t) => t[0] === tab)[2];
  return html`<${TopBar} />
  <main class="screen"><div class="doc wide"><${SkelReveal} ready=${true} skeleton=${null}>
    <div class="ph"><div><h1 class="page-title">Parts library</h1>
      <p class="lede"><${ModelLogo} model=${cat.host.model} size=${16} /> Host model ${cat.host.model.split('/').pop()}, ${cat.host.layers} layers, ${cat.host.params_b.toFixed(2)}B parameters. Open a part to see where it came from. Add it to your model from its window or in the builder.</p></div>
      ${tab !== 'results' && html`<div class="searchbox pq"><${Icon} n="search" size=${16} /><input placeholder=${place} aria-label=${place} value=${q} onInput=${(e) => setQ(e.target.value)} /></div>`}</div>
    <div class="ptabs" role="tablist">${TABS.map(([id, name]) => html`<button key=${id} class=${'ptab' + (tab === id ? ' on' : '')} role="tab" aria-selected=${tab === id ? 'true' : 'false'} onClick=${() => { setFocus(null); setTab(id); }}>${name}<span class="n">${id === 'parts' ? cat.parts.length + cat.masks.length : tabCounts[id] != null ? tabCounts[id] : ''}</span></button>`)}</div>
    ${tab === 'parts' && html`<${PartsTab} cat=${cat} q=${q} onFinding=${openFinding} />`}
    ${tab === 'recipes' && html`<${RecipesTab} q=${q} names=${names} />`}
    ${tab === 'findings' && html`<${FindingsTab} q=${q} focus=${focus} names=${names} />`}
    ${tab === 'results' && html`<${MeasuredResults} />`}
  <//></div></main>`;
}
