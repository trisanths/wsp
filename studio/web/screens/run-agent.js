// One agent: what it is doing, its program, its recent steps, its reasoning and its history.
import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { toast, clock, pctS, primaryTarget, useInterval } from '../util.js';
import { Dot, CodeView, Composer } from '../components.js';
import { ThinkLine, StreamText, SubtleNum } from '../motion.js';
import { openProgramInBuilder } from './shared.js';
import { nameOf, progName, ACTIVE, agentDot, STATUS_WORD } from './run-shared.js';

const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const VERBS = { probe: 'Fit check', compile: 'Compile', heal: 'Heal', eval: 'Score', read_results: 'Read results', control: 'Random-part control' };
// A tool call in plain words. The part names inside the call are dropped.
function plainCall(call) {
  const m = /^(\w+)\((.*)\)$/s.exec(call || '');
  if (!m) return call;
  const label = VERBS[m[1]] || cap(m[1].replace(/_/g, ' '));
  return ['heal', 'eval', 'read_results'].includes(m[1]) && m[2] ? label + ', ' + m[2] : label;
}

export function AgentPage({ snap, id, agentId, cat }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const [more, setMore] = useState(false);
  const prev = useRef({ text: null, prev: null });
  const seen = useRef(null);
  const clearRef = useRef();

  const load = () => api.getAgent(id, agentId).then((x) => {
    const t = x.current ? x.current.text : null;
    if (t !== prev.current.text) prev.current = { text: t, prev: prev.current.text };
    setD(x); setErr(null);
  }).catch(setErr);
  useEffect(() => { load(); }, [id, agentId]);
  useInterval(load, 1500);

  const send = async () => {
    const msg = text.trim();
    if (!msg) return;
    setSending(true);
    try {
      await api.nudge(id, agentId, msg);
      if (clearRef.current) clearRef.current();
      setText('');
      toast('Nudge sent to ' + a.name, 'ok');
      load();
    } catch (e) { toast(e.message, 'err'); }
    setSending(false);
  };

  const a = d ? d.agent : (snap.agents.find((x) => x.id === agentId) || { id: agentId, name: agentId, status: 'idle', attempts: 0, collapses: 0 });
  const live = snap.agents.find((x) => x.id === agentId) || a;
  const t = primaryTarget(snap.spec);
  const active = !!ACTIVE[live.status];
  const cur = d && d.current;
  const thinking = d ? d.thinking.slice().reverse() : [];
  // the first load is the baseline: only later reasoning streams in
  if (d && !seen.current) seen.current = new Set(d.thinking.map((x) => x.ts));
  useEffect(() => { if (d && seen.current) d.thinking.forEach((x) => seen.current.add(x.ts)); });
  const shown = more ? thinking : thinking.slice(0, 4);
  const bits = [live.attempts + (live.attempts === 1 ? ' attempt' : ' attempts'), live.collapses + (live.collapses === 1 ? ' collapse' : ' collapses')];
  if (live.best != null) bits.push('best ' + pctS(live.best));

  return html`<div class="rpage agentp">
    <div class="ascroll"><div class="col">
      <a class="back" href=${'#/run/' + id + '/agents'}>Agents</a>
      <div class="ahd">
        <h1 class="page-title">${a.name}</h1>
        <span class="astat"><${Dot} color=${agentDot(live.status)} pulse=${active} />${STATUS_WORD[live.status] || live.status}</span>
      </div>
      <div class="ameta">${bits.join(', ')}</div>
      ${err && !d && html`<div class="down" style="margin-top:24px">${err.message}</div>`}
      ${!d && !err && html`<div class="skeleton" style="height:120px;margin-top:32px"></div>`}
      ${d && html`
        <section class="rsec"><h2 class="rh2">Now</h2><div class="anowline"><${ThinkLine} text=${live.now || STATUS_WORD[live.status] || ''} still=${!active} fluid /></div></section>
        <section class="rsec"><h2 class="rh2">Program</h2>
          ${cur ? html`<div class="pname">${cur.name || progName(cur.program, cat) || cur.describe}</div>
            <div class="well"><${CodeView} text=${cur.text} prev=${prev.current.prev} /></div>
            <div class="bact"><button class="linkbtn" onClick=${() => openProgramInBuilder(snap.spec.id, cur.program)}>Open in builder</button></div>`
          : html`<div class="faint">No program in flight right now.</div>`}
        </section>
        <section class="rsec"><h2 class="rh2">Recent steps</h2>
          ${d.tools.length ? html`<div class="tlist">${d.tools.slice(-8).reverse().map((x, i) => html`<div class="trow" key=${x.ts + x.call + i}><span class="tc mono">${plainCall(x.call)}</span><span class=${'tr ' + (x.status === 'ok' ? 'up' : x.status === 'bad' ? 'down' : 'dim3')}>${x.result}</span></div>`)}</div>` : html`<div class="faint">None yet.</div>`}
        </section>
        <section class="rsec"><h2 class="rh2">Reasoning</h2>
          ${shown.length ? shown.map((x, i) => html`<div class="rline2" key=${x.ts + ':' + i}><span class="ts num">${clock(x.ts)}</span><${StreamText} class="rtxt" text=${x.text} play=${!!seen.current && !seen.current.has(x.ts)} /></div>`) : html`<div class="faint">Nothing yet.</div>`}
          ${thinking.length > 4 && html`<button class="linkbtn" onClick=${() => setMore(!more)}>${more ? 'Show less' : 'Show ' + (thinking.length - 4) + ' earlier'}</button>`}
        </section>
        <section class="rsec"><h2 class="rh2">History</h2>
          ${d.history.length ? html`<div class="plist">${d.history.slice().reverse().slice(0, 15).map((h, i) => {
            const bad = h.status !== 'ok';
            const v = t ? h.scores[t.id] : null;
            return html`<div class="prow" key=${h.program_hash + i}><span class=${'pn ell' + (bad ? ' down' : '')} title=${nameOf(h, cat)}>${nameOf(h, cat)}</span><span class=${'ps num' + (bad ? ' down' : '')}>${bad ? (h.status === 'collapsed' ? 'collapsed' : h.status.replace(/_/g, ' ')) : v != null ? pctS(v) : '-'}</span></div>`;
          })}</div>` : html`<div class="faint">No scored programs yet.</div>`}
        </section>
        ${d.nudges.length > 0 && html`<section class="rsec"><h2 class="rh2">Your nudges</h2><div class="plist">${d.nudges.slice().reverse().map((n) => html`<div class="prow"><span class="pn">${n.text}</span><span class="ps dim3">${n.applied ? 'read' : 'waiting'}</span></div>`)}</div></section>`}
      `}
    </div></div>
    <div class="adock"><div class="col">
      <${Composer} value=${text} onInput=${setText} onSubmit=${send} disabled=${sending} minRows=${1} clearRef=${clearRef} label=${'Nudge ' + a.name} placeholder=${'Tell ' + a.name + ' what to try next'} />
    </div></div>
  </div>`;
}
