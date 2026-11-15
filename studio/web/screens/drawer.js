import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { go, toast, clock, pct, pctS, primaryTarget, STATUS_COLOR, STATUS_ACTIVE, useInterval } from '../util.js';
import { Icon, Avatar, Dot, Tag, CodeView, Spinner, Composer } from '../components.js';
import { openProgramInBuilder } from './shared.js';

const QUICK = ['Prefer smaller models', 'Try a different donor', 'Spend fewer evals', 'Explain your plan'];

export function AgentView({ runId, agentId, snap }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const prev = useRef({ text: null, prev: null });

  const load = () => api.getAgent(runId, agentId).then((x) => {
    const t = x.current ? x.current.text : null;
    if (t !== prev.current.text) prev.current = { text: t, prev: prev.current.text };
    setD(x); setErr(null);
  }).catch(setErr);
  useEffect(() => { setD(null); prev.current = { text: null, prev: null }; load(); }, [runId, agentId]);
  useInterval(load, 1500);

  const send = async (t) => {
    const msg = (t ?? text).trim();
    if (!msg) return;
    setSending(true);
    try { await api.nudge(runId, agentId, msg); setText(''); toast('Nudge sent to ' + (d ? d.agent.name : agentId), 'ok'); load(); } catch (e) { toast(e.message, 'err'); }
    setSending(false);
  };

  const a = d ? d.agent : (snap.agents.find((x) => x.id === agentId) || { id: agentId, name: agentId, status: 'idle' });
  const target = primaryTarget(snap.spec);
  const active = !!STATUS_ACTIVE[a.status];
  const cur = d && d.current;
  const chips = html`<div class="qchips">${QUICK.map((q) => html`<button class="chip" disabled=${sending} onClick=${() => send(q)}>${q}</button>`)}</div>`;
  return html`<div class="agentview">
    <div class="dr-hd">
      <${Avatar} agent=${a} size=${30} /><h2 class="dr-name">${a.name}</h2>
      ${snap.best && snap.best.agent === a.id && html`<${Tag} kind="b">best program<//>`}
      <span class="pill"><${Dot} color=${STATUS_COLOR[a.status] || 'var(--text-4)'} pulse=${active} /><b style="text-transform:capitalize">${a.status}</b></span>
    </div>
    ${err && !d && html`<div class="down">${err.message}</div>`}
    ${!d && !err && html`<div class="skeleton" style="height:180px"></div>`}
    ${d && html`
      <div class="dr-blurb">${a.blurb}. <b>${a.now}</b></div>
      ${cur && cur.program && html`<div class="dr-actions"><button class="btn sm" onClick=${() => openProgramInBuilder(snap.spec.id, cur.program)}>Open its program in builder</button></div>`}
      <div class="well">
        <h3 class="sec-sm">Program it is writing</h3>
        ${cur ? html`<${CodeView} text=${cur.text} prev=${prev.current.prev} /><div class="faint" style="margin-top:6px;font-size:12px">${cur.describe}</div>` : html`<div class="faint" style="padding:6px 0">No program in flight right now.</div>`}
        <h3 class="sec-sm">Tool calls</h3>
        ${d.tools.length ? d.tools.slice(-10).map((t, i) => html`<div class=${'tool ' + t.status} key=${t.ts + t.call + i}><span class="cl2" title=${t.call}>${t.call}</span><span class="rs">${t.result}</span></div>`) : html`<div class="faint">None yet.</div>`}
      </div>
      <h3 class="sec-sm">Thinking <span class="faint" style="font-weight:400">newest first</span></h3>
      ${d.thinking.length ? d.thinking.slice(-8).reverse().map((t, i) => html`<div class="think" key=${t.ts + i}><span class="ts">${clock(t.ts)}</span>${t.text}</div>`) : html`<div class="faint">Nothing yet.</div>`}
      <h3 class="sec-sm">History · ${d.history.length}</h3>
      ${d.history.length ? d.history.slice().reverse().slice(0, 12).map((h) => { const v = target ? h.scores[target.id] : null; const sc = target ? Math.max(target.target, target.lora || 0) * 1.3 : 1; return html`<div class="hist"><span class="mono ell" title=${h.describe}>${h.describe}</span><div class="mini"><i style=${{ width: Math.min(100, ((v || 0) / sc) * 100) + '%', background: h.status === 'ok' ? 'var(--blue)' : 'var(--red)' }}></i></div><span class=${'num ' + (h.status === 'ok' ? '' : 'down')} style="text-align:right">${h.status === 'ok' ? pct(v) : h.status}</span></div>`; }) : html`<div class="faint">No scored programs yet.</div>`}
      ${d.nudges.length > 0 && html`<h3 class="sec-sm">Your nudges</h3>${d.nudges.slice().reverse().map((n) => html`<div class="nrow"><${Icon} n=${n.applied ? 'check' : 'chevron'} size=${14} /><span>${n.text}</span><span class="grow"></span><${Tag} kind=${n.applied ? 'g' : 'w'}>${n.applied ? 'read' : 'waiting'}<//></div>`)}`}
    `}
    <div class="nudgebox">
      <${Composer} value=${text} onInput=${setText} onSubmit=${() => send()} disabled=${sending} minRows=${1} chips=${chips} label=${'Nudge ' + a.name} placeholder=${'Tell ' + a.name + ' what to try next'} />
      <div class="dock-note">A nudge goes into the agent's next turn and shows in the conversation.</div>
    </div>
  </div>`;
}
