// The run, split into pages: Overview, Activity, Agents (and one agent), Results.
// This screen owns the one stream of snapshots. The pages read it, so switching pages never reconnects.
import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { toast, pct, pctS, sgn, dur, primaryTarget, addRecent, patchRecent, runEnd, recentStatus } from '../util.js';
import { TopBar, Icon, ErrorBox, StopMark } from '../components.js';
import { Shimmer, SubtleNum, Reel, SuccessCheck, SkelReveal, SlidingBar, PageSwap, IconSwap, SwapText } from '../motion.js';
import { openProgramInBuilder, bestProgram } from './shared.js';
import { useCatalog, nameOf, AgentList, bestTarget, isEmptyEnd, runRecent } from './run-shared.js';
import { PipelinePage } from './run-pipeline.js';
import { ActivityPage } from './run-activity.js';
import { AgentPage, AgentsBoard } from './run-agent.js';
import { FindingsPage } from './run-findings.js';
import { ResultsPage } from './run-results.js';
import { PoweredBy } from '../logos.js';
import { Logos, donorsOf, verdictOf, HealNote, usesHeal, healNoteOf, partsCount } from '../cmp.js';

const TABS = [['overview', 'Overview', ''], ['pipeline', 'Pipeline', '/pipeline'], ['activity', 'Activity', '/activity'], ['agents', 'Agents', '/agents'], ['findings', 'Found so far', '/findings'], ['results', 'Results', '/results']];
// the order of pages sets the slide direction
const DEPTH = { overview: 0, pipeline: 0.5, activity: 1, agents: 2, agent: 2.5, findings: 2.8, results: 3 };

function StatusLine({ snap, justDone }) {
  const b = snap.budget;
  const s = snap.status;
  const per = b.evals_used ? snap.elapsed_s / b.evals_used : 0;
  const left = per * (b.evals_total - b.evals_used);
  const evalPart = html`eval <${SubtleNum} text=${String(b.evals_used)} /> of ${b.evals_total}`;
  const end = runEnd(snap);
  if (end === 'finished' && isEmptyEnd(snap)) return html`<span class="rstat"><${StopMark} /> Ended, nothing scored, after ${dur(snap.elapsed_s)}</span>`;
  if (end === 'finished') return html`<span class="rstat"><${SuccessCheck} play=${justDone} size=${16} class="ok" /> Finished in ${dur(snap.elapsed_s)}</span>`;
  if (end === 'stopped') return html`<span class="rstat"><${StopMark} /> Stopped after ${dur(snap.elapsed_s)}</span>`;
  if (end === 'error') return html`<span class="rstat down">Stopped by an error</span>`;
  if (s === 'stopping') return html`<span class="rstat">Stopping, ${evalPart}. The best program is kept.</span>`;
  if (s === 'paused') return html`<span class="rstat">Paused, ${evalPart}</span>`;
  return html`<span class="rstat"><${Shimmer} text=${s === 'starting' ? 'Starting' : 'Running'} />, ${evalPart}${left > 0 && s === 'running' ? ', about ' + dur(left) + ' left' : ''}</span>`;
}

function RunHead({ snap, justDone, note, onPause, onResume, onStop }) {
  const b = snap.budget;
  const s = snap.status;
  const frac = b.evals_total ? Math.min(1, b.evals_used / b.evals_total) : 0;
  const live = s === 'running' || s === 'starting' || s === 'paused' || s === 'stopping';
  const paused = s === 'paused';
  const end = runEnd(snap) === 'finished' && isEmptyEnd(snap) ? 'empty' : runEnd(snap);
  return html`<header class="rhead"><div class="col">
    <div class="rh-row">
      <h1 class="page-title ell">${snap.spec.title}</h1>
      <span class="grow"></span>
      ${live && html`<div class="rh-btns">
        <button class="btn sm" disabled=${s === 'stopping' || s === 'starting'} onClick=${paused ? onResume : onPause}><${IconSwap} state=${paused ? 'b' : 'a'} a=${html`<${Icon} n="pause" size=${14} />`} b=${html`<${Icon} n="play" size=${14} />`} /><${SwapText} text=${paused ? 'Resume' : 'Pause'} /></button>
        <button class="btn sm" disabled=${s === 'stopping'} onClick=${onStop}>${s === 'stopping' ? 'Stopping' : 'Stop'}</button>
      </div>`}
    </div>
    <div class="rh-status"><${StatusLine} snap=${snap} justDone=${justDone} /><${PoweredBy} who="modal" label="GPUs on" class="rstatchip" /></div>
    ${note && html`<div class="rh-note" role="status">${note}</div>`}
    <div class=${'rline' + (end === 'finished' ? ' ok' : end === 'stopped' ? ' stopped' : '')} aria-hidden="true"><i style=${{ width: (end === 'finished' ? 1 : frac) * 100 + '%' }}></i></div>
  </div></header>`;
}

function SubNav({ id, page }) {
  const cur = page === 'agent' ? 'agents' : page;
  return html`<nav class="rnav" aria-label="Run pages"><div class="col"><${SlidingBar} class="tabs rtabs">${TABS.map(([k, label, path]) => html`<a class=${'tab' + (cur === k ? ' on' : '')} href=${'#/run/' + id + path} aria-current=${cur === k ? 'page' : null}>${label}</a>`)}<//></div></nav>`;
}

/* ---------- overview ---------- */
function BestBlock({ snap, cat, done, busy, onBuild }) {
  const ended = snap.status === 'done' || snap.status === 'error';
  const t = primaryTarget(snap.spec);
  const v = bestTarget(snap);
  const start = t ? ((snap.base_scores || {})[t.id] ?? t.base) : null;
  const dStart = v != null && start != null ? v - start : null;
  const dLora = v != null && t && t.lora != null ? v - t.lora : null;
  const comp = snap.comparison || (snap.best && (snap.leaderboard.find((r) => r.program_hash === snap.best.program_hash) || {}).comparison) || null;
  const lora = comp && (comp.rows || []).find((r) => r.id === 'lora' && r.measured && r.delta != null);
  const dLoraC = lora ? lora.delta : dLora;
  const bestRow = snap.best && snap.leaderboard.find((r) => r.program_hash === snap.best.program_hash);
  const pc = partsCount(bestRow && bestRow.components);
  const vd = verdictOf(comp);
  const donors = donorsOf(snap.best && snap.best.program, cat);
  const cmp = (d, label) => html`<span class=${d >= 0 ? 'up' : 'down'}>${sgn(d)}</span> ${label}`;
  return html`<section class="rsec">
    <h2 class="rh2">Best model so far</h2>
    ${v == null && ended ? html`<div class="rempty"><div>${snap.empty_reason || 'Nothing was scored.'}</div><a class="linkbtn" href=${'#/spec/' + snap.spec.id}>Edit spec</a></div>` : html`<${SkelReveal} ready=${v != null} skeleton=${html`<div class="skeleton" style="height:20px;width:60%"></div><div class="skeleton" style="height:48px;width:140px;margin-top:14px"></div>`}>
      ${v != null && html`<div class="best">
        <div class="bname"><${Logos} models=${donors.length ? donors : [snap.spec.host]} max=${4} size=${20} /><span>${nameOf(snap.best, cat)}</span></div>
        <div class="bscore"><span class="bnum"><${Reel} value=${pctS(v)} /></span><span class="faint">on ${t.name}</span></div>
        <div class="bcmp">${dStart != null && html`<span>${cmp(dStart, 'vs start')}</span>`}${dLoraC != null && html`<span>${cmp(dLoraC, 'vs fine-tuning (LoRA)')}</span>`}</div>
        <${HealNote} show=${usesHeal(snap.best && snap.best.program)} note=${healNoteOf(bestRow)} />
        ${pc && html`<div class="bparts">${pc.n} of ${pc.of} ${pc.of === 1 ? 'part contributes' : 'parts contribute'}</div>`}
        ${vd && html`<div class=${'bverdict ' + vd.kind}>${vd.text}</div>`}
        <div class="bact">${done
          ? html`<button class="btn primary" disabled=${busy} onClick=${onBuild}>Open in builder <${Icon} n="arrow" size=${14} /></button>`
          : html`<button class="linkbtn" disabled=${busy} onClick=${onBuild}>Open in builder</button>`}</div>
      </div>`}
    <//>`}
  </section>`;
}

function TopPrograms({ snap, cat, id }) {
  const t = primaryTarget(snap.spec);
  const rows = snap.leaderboard.slice(0, 5);
  return html`<section class="rsec">
    <h2 class="rh2">Top programs</h2>
    ${rows.length ? html`<div class="plist">${rows.map((r) => html`<div class="prow" key=${r.program_hash}><span class="rk num">${r.rank}</span><span class="pn"><${Logos} models=${donorsOf(r.program, cat)} max=${3} size=${16} /><span title=${nameOf(r, cat)}>${nameOf(r, cat)}</span></span><span class="ps num">${t && r.scores[t.id] != null ? html`<${SubtleNum} text=${pctS(r.scores[t.id])} />` : '-'}</span></div>`)}</div>
      <a class="linkbtn" href=${'#/run/' + id + '/results'}>See all results</a>` : html`<div class="faint">No scored programs yet.</div>`}
  </section>`;
}

function Overview({ snap, cat, id, done, busy, onBuild }) {
  return html`<div class="rpage"><div class="col">
    <${BestBlock} snap=${snap} cat=${cat} done=${done} busy=${busy} onBuild=${onBuild} />
    ${snap.refused > 0 && !/refused/.test(snap.empty_reason || '') && html`<div class="refused">${snap.refused} ${snap.refused === 1 ? 'proposal' : 'proposals'} refused for going over your limits</div>`}
    <section class="rsec"><div class="rsechead"><h2 class="rh2">Agents</h2><${PoweredBy} who="pastdev" label="Notes shared through" /></div><${AgentList} snap=${snap} /></section>
    <${TopPrograms} snap=${snap} cat=${cat} id=${id} />
  </div></div>`;
}

function AgentsPage({ snap, id, cat }) {
  return html`<${AgentsBoard} snap=${snap} id=${id} cat=${cat} />`;
}

/* ---------- the run ---------- */
export function RunScreen({ id, page, agent }) {
  const [snap, setSnap] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [justDone, setJustDone] = useState(false);
  const [note, setNote] = useState(null);
  const prev = useRef(null);
  const cat = useCatalog();

  useEffect(() => {
    let closed = false, poll = null;
    const on = (s) => { if (!closed) { setSnap(s); setErr(null); } };
    const onStream = (s) => { if (!closed) { setNote(null); on(s); } };
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
    close = api.openStream(id, onStream, startPoll, (msg) => { if (!closed) { setNote('The live stream reported: ' + msg + '. Polling instead.'); startPoll(); } });
    return () => { closed = true; close(); if (poll) clearInterval(poll); };
  }, [id]);
  useEffect(() => {
    if (!snap) return;
    addRecent({ kind: 'run', id, title: snap.spec.title, status: runRecent(snap) });
  }, [id, !!snap]);
  useEffect(() => { if (snap) patchRecent('run', id, { status: runRecent(snap) }); }, [snap && runRecent(snap)]);
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
    else if (page === 'agents') pageEl = html`<${AgentsPage} snap=${snap} id=${id} cat=${cat} />`;
    else if (page === 'agent') pageEl = html`<${AgentPage} key=${agent} snap=${snap} id=${id} agentId=${agent} cat=${cat} />`;
    else if (page === 'findings') pageEl = html`<${FindingsPage} snap=${snap} id=${id} cat=${cat} />`;
    else if (page === 'pipeline') pageEl = html`<${PipelinePage} snap=${snap} id=${id} cat=${cat} />`;
    else if (page === 'results') pageEl = html`<${ResultsPage} snap=${snap} id=${id} cat=${cat} onBuild=${(p) => openProgramInBuilder(spec.id, p)} />`;
    else pageEl = html`<${Overview} snap=${snap} cat=${cat} id=${id} done=${done} busy=${busy} onBuild=${openBest} />`;
    body = html`<${RunHead} snap=${snap} justDone=${justDone} note=${note} onPause=${() => act(api.pause)} onResume=${() => act(api.resume)} onStop=${() => act(api.stop)} />
      <${SubNav} id=${id} page=${page} />
      <${PageSwap} pageKey=${page + ':' + (agent || '')} depth=${DEPTH[page] ?? 0}>${pageEl}<//>`;
  }
  return html`<div class="runview"><${TopBar} /><${SkelReveal} fill ready=${!!snap} skeleton=${loading}>${body}<//></div>`;
}
