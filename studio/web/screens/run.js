// The run, split into pages: Overview, Activity, Agents (and one agent), Results.
// This screen owns the one stream of snapshots. The pages read it, so switching pages never reconnects.
import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { toast, pct, pctS, sgn, dur, primaryTarget, addRecent, patchRecent } from '../util.js';
import { TopBar, Icon, ErrorBox } from '../components.js';
import { Shimmer, SubtleNum, Reel, SuccessCheck, SkelReveal, SlidingBar, PageSwap, IconSwap, SwapText } from '../motion.js';
import { openProgramInBuilder, bestProgram } from './shared.js';
import { useCatalog, nameOf, AgentList, bestTarget } from './run-shared.js';
import { ActivityPage } from './run-activity.js';
import { AgentPage } from './run-agent.js';
import { ResultsPage } from './run-results.js';

const TABS = [['overview', 'Overview', ''], ['activity', 'Activity', '/activity'], ['agents', 'Agents', '/agents'], ['results', 'Results', '/results']];
// the order of pages sets the slide direction
const DEPTH = { overview: 0, activity: 1, agents: 2, agent: 2.5, results: 3 };

function StatusLine({ snap, justDone }) {
  const b = snap.budget;
  const s = snap.status;
  const per = b.evals_used ? snap.elapsed_s / b.evals_used : 0;
  const left = per * (b.evals_total - b.evals_used);
  const evalPart = html`eval <${SubtleNum} text=${String(b.evals_used)} /> of ${b.evals_total}`;
  if (s === 'done') return html`<span class="rstat"><${SuccessCheck} play=${justDone} size=${16} class="ok" /> Finished in ${dur(snap.elapsed_s)}</span>`;
  if (s === 'error') return html`<span class="rstat down">Stopped by an error</span>`;
  if (s === 'stopping') return html`<span class="rstat">Stopping, ${evalPart}. The best program is kept.</span>`;
  if (s === 'paused') return html`<span class="rstat">Paused, ${evalPart}</span>`;
  return html`<span class="rstat"><${Shimmer} text=${s === 'starting' ? 'Starting' : 'Running'} />, ${evalPart}${left > 0 && s === 'running' ? ', about ' + dur(left) + ' left' : ''}</span>`;
}

function RunHead({ snap, justDone, onPause, onResume, onStop }) {
  const b = snap.budget;
  const s = snap.status;
  const frac = b.evals_total ? Math.min(1, b.evals_used / b.evals_total) : 0;
  const live = s === 'running' || s === 'starting' || s === 'paused' || s === 'stopping';
  const paused = s === 'paused';
  return html`<header class="rhead"><div class="col">
    <div class="rh-row">
      <h1 class="page-title ell">${snap.spec.title}</h1>
      <span class="grow"></span>
      ${live && html`<div class="rh-btns">
        <button class="btn sm" disabled=${s === 'stopping' || s === 'starting'} onClick=${paused ? onResume : onPause}><${IconSwap} state=${paused ? 'b' : 'a'} a=${html`<${Icon} n="pause" size=${14} />`} b=${html`<${Icon} n="play" size=${14} />`} /><${SwapText} text=${paused ? 'Resume' : 'Pause'} /></button>
        <button class="btn sm" disabled=${s === 'stopping'} onClick=${onStop}>${s === 'stopping' ? 'Stopping' : 'Stop'}</button>
      </div>`}
    </div>
    <div class="rh-status"><${StatusLine} snap=${snap} justDone=${justDone} /></div>
    <div class=${'rline' + (s === 'done' ? ' ok' : '')} aria-hidden="true"><i style=${{ width: (s === 'done' ? 1 : frac) * 100 + '%' }}></i></div>
  </div></header>`;
}

function SubNav({ id, page }) {
  const cur = page === 'agent' ? 'agents' : page;
  return html`<nav class="rnav" aria-label="Run pages"><div class="col"><${SlidingBar} class="tabs rtabs">${TABS.map(([k, label, path]) => html`<a class=${'tab' + (cur === k ? ' on' : '')} href=${'#/run/' + id + path} aria-current=${cur === k ? 'page' : null}>${label}</a>`)}<//></div></nav>`;
}

/* ---------- overview ---------- */
function BestBlock({ snap, cat, done, busy, onBuild }) {
  const t = primaryTarget(snap.spec);
  const v = bestTarget(snap);
  const start = t ? (snap.base_scores[t.id] ?? t.base) : null;
  const dStart = v != null && start != null ? v - start : null;
  const dLora = v != null && t && t.lora != null ? v - t.lora : null;
  const cmp = (d, label) => html`<span class=${d >= 0 ? 'up' : 'down'}>${sgn(d)}</span> ${label}`;
  return html`<section class="rsec">
    <h2 class="rh2">Best model so far</h2>
    <${SkelReveal} ready=${v != null} skeleton=${html`<div class="skeleton" style="height:20px;width:60%"></div><div class="skeleton" style="height:48px;width:140px;margin-top:14px"></div>`}>
      ${v != null && html`<div class="best">
        <div class="bname">${nameOf(snap.best, cat)}</div>
        <div class="bscore"><span class="bnum"><${Reel} value=${pctS(v)} /></span><span class="faint">on ${t.name}</span></div>
        <div class="bcmp">${dStart != null && html`<span>${cmp(dStart, 'vs start')}</span>`}${dLora != null && html`<span>${cmp(dLora, 'vs fine-tuning (LoRA)')}</span>`}</div>
        <div class="bact">${done
          ? html`<button class="btn primary" disabled=${busy} onClick=${onBuild}>Open in builder <${Icon} n="arrow" size=${14} /></button>`
          : html`<button class="linkbtn" disabled=${busy} onClick=${onBuild}>Open in builder</button>`}</div>
      </div>`}
    <//>
  </section>`;
}

function TopPrograms({ snap, cat, id }) {
  const t = primaryTarget(snap.spec);
  const rows = snap.leaderboard.slice(0, 5);
  return html`<section class="rsec">
    <h2 class="rh2">Top programs</h2>
    ${rows.length ? html`<div class="plist">${rows.map((r) => html`<div class="prow" key=${r.program_hash}><span class="rk num">${r.rank}</span><span class="pn ell" title=${nameOf(r, cat)}>${nameOf(r, cat)}</span><span class="ps num">${t && r.scores[t.id] != null ? html`<${SubtleNum} text=${pctS(r.scores[t.id])} />` : '-'}</span></div>`)}</div>
      <a class="linkbtn" href=${'#/run/' + id + '/results'}>See all results</a>` : html`<div class="faint">No scored programs yet.</div>`}
  </section>`;
}

function Overview({ snap, cat, id, done, busy, onBuild }) {
  return html`<div class="rpage"><div class="col">
    <${BestBlock} snap=${snap} cat=${cat} done=${done} busy=${busy} onBuild=${onBuild} />
    <section class="rsec"><h2 class="rh2">Agents</h2><${AgentList} snap=${snap} /></section>
    <${TopPrograms} snap=${snap} cat=${cat} id=${id} />
  </div></div>`;
}

function AgentsPage({ snap }) {
  return html`<div class="rpage"><div class="col"><section class="rsec"><h2 class="rh2">Agents</h2><${AgentList} snap=${snap} /></section></div></div>`;
}

/* ---------- the run ---------- */
export function RunScreen({ id, page, agent }) {
  const [snap, setSnap] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [justDone, setJustDone] = useState(false);
  const prev = useRef(null);
  const cat = useCatalog();

  useEffect(() => {
    let closed = false, poll = null;
    const on = (s) => { if (!closed) { setSnap(s); setErr(null); } };
    // Runs live in server memory, so a server restart ends them. On a 404, stop polling and say so.
    let close = () => {};
    const gone = (e) => {
      if (closed || !e || e.status !== 404) return false;
      if (poll) { clearInterval(poll); poll = null; }
      close();
      setErr(new Error('This run is no longer on the server. Runs are kept in server memory, so a server restart ends them. Start it again from its spec.'));
      return true;
    };
    const startPoll = () => { if (!poll && !closed) poll = setInterval(() => api.getRun(id).then(on).catch(gone), 2000); };
    api.getRun(id).then(on).catch((e) => { if (!gone(e) && !closed) setErr(e); });
    close = api.openStream(id, on, startPoll);
    return () => { closed = true; close(); if (poll) clearInterval(poll); };
  }, [id]);
  useEffect(() => {
    if (!snap) return;
    addRecent({ kind: 'run', id, title: snap.spec.title, status: snap.status });
  }, [id, !!snap]);
  useEffect(() => { if (snap) patchRecent('run', id, { status: snap.status }); }, [snap && snap.status]);
  // the check draws only when the run ends while the page is open
  useEffect(() => {
    if (!snap) return;
    if (prev.current && prev.current !== 'done' && snap.status === 'done') setJustDone(true);
    prev.current = snap.status;
  }, [snap && snap.status]);

  if (err && !snap) return html`<${TopBar} /><${ErrorBox} error=${err} onRetry=${() => location.reload()} />`;
  const loading = html`<div class="col"><div class="skeleton" style="height:32px;width:320px;margin-top:8px"></div><div class="skeleton" style="height:140px;margin-top:32px"></div></div>`;
  let body = null;
  if (snap) {
    const spec = snap.spec;
    const act = async (fn) => { try { await fn(id); } catch (e) { toast(e.message, 'err'); } };
    const openBest = async () => {
      setBusy(true);
      const p = await bestProgram(id, snap);
      if (!p) toast('The server did not send the best program. Opening an empty builder.', 'info');
      await openProgramInBuilder(spec.id, p);
      setBusy(false);
    };
    const done = snap.status === 'done';
    let pageEl;
    if (page === 'activity') pageEl = html`<${ActivityPage} snap=${snap} id=${id} />`;
    else if (page === 'agents') pageEl = html`<${AgentsPage} snap=${snap} />`;
    else if (page === 'agent') pageEl = html`<${AgentPage} key=${agent} snap=${snap} id=${id} agentId=${agent} cat=${cat} />`;
    else if (page === 'results') pageEl = html`<${ResultsPage} snap=${snap} id=${id} cat=${cat} onBuild=${(p) => openProgramInBuilder(spec.id, p)} />`;
    else pageEl = html`<${Overview} snap=${snap} cat=${cat} id=${id} done=${done} busy=${busy} onBuild=${openBest} />`;
    body = html`<${RunHead} snap=${snap} justDone=${justDone} onPause=${() => act(api.pause)} onResume=${() => act(api.resume)} onStop=${() => act(api.stop)} />
      <${SubNav} id=${id} page=${page} />
      <${PageSwap} pageKey=${page + ':' + (agent || '')} depth=${DEPTH[page] ?? 0}>${pageEl}<//>`;
  }
  return html`<div class="runview"><${TopBar} /><${SkelReveal} fill ready=${!!snap} skeleton=${loading}>${body}<//></div>`;
}
