import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { go, toast, clock, pct, pctS, primaryTarget, STATUS_COLOR, STATUS_ACTIVE, useInterval } from '../util.js';
import { Icon, Avatar, Dot, Tag, CodeView, Spinner } from '../components.js';
import { openProgramInBuilder } from './shared.js';

const QUICK = ['Prefer smaller models', 'Try a different donor', 'Spend fewer evals', 'Explain your plan'];

export function AgentDrawer({ runId, agentId, snap }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const prev = useRef({ text: null, prev: null });
  const close = () => go('/run/' + runId);

  const load = () => api.getAgent(runId, agentId).then((x) => {
    const t = x.current ? x.current.text : null;
    if (t !== prev.current.text) prev.current = { text: t, prev: prev.current.text };
    setD(x); setErr(null);
  }).catch(setErr);
  useEffect(() => { setD(null); prev.current = { text: null, prev: null }; load(); }, [runId, agentId]);
  useInterval(load, 1500);
  useEffect(() => {
    const k = (e) => { if (e.key === 'Escape' && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) close(); };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [runId]);

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
  return html`<div class="scrim" onClick=${close}></div>
  <aside class="drawer" role="dialog" aria-label=${a.name}>
    <div class="dr-hd">
      <${Avatar} agent=${a} size=${30} /><b style="font-size:15px">${a.name}</b>
      ${snap.best && snap.best.agent === a.id && html`<${Tag} kind="a">best program<//>`}
      <span class="pill"><${Dot} color=${STATUS_COLOR[a.status] || 'var(--faint)'} pulse=${active} /><b style="text-transform:capitalize">${a.status}</b></span>
      <span class="grow"></span>
      ${cur && cur.program && html`<button class="btn ghost sm" onClick=${() => openProgramInBuilder(snap.spec.id, cur.program)}>Open its program in builder</button>`}
      <button class="iconbtn" title="Close (Esc)" onClick=${close}><${Icon} n="x" /></button>
    </div>
    <div class="dr-body">
      ${err && !d && html`<div class="down">${err.message}</div>`}
      ${!d && !err && html`<div class="skeleton" style="height:180px"></div>`}
      ${d && html`
        <div class="dim" style="margin-bottom:14px">${a.blurb}. <span style="color:var(--text)">${a.now}</span></div>
        <h4 class="label">Program it is writing</h4>
        ${cur ? html`<${CodeView} text=${cur.text} prev=${prev.current.prev} /><div class="faint" style="margin-top:6px;font-size:11.5px">${cur.describe}</div>` : html`<div class="faint card" style="padding:14px">No program in flight right now.</div>`}
        <h4 class="label">Tool calls</h4>
        ${d.tools.length ? d.tools.slice(-10).map((t, i) => html`<div class=${'tool ' + t.status} key=${t.ts + t.call + i}><span class="cl2" title=${t.call}>${t.call}</span><span class="rs">${t.result}</span></div>`) : html`<div class="faint">None yet.</div>`}
        <h4 class="label">Thinking <span class="faint" style="text-transform:none;letter-spacing:0;font-weight:400">newest first</span></h4>
        ${d.thinking.length ? d.thinking.slice(-8).reverse().map((t, i) => html`<div class="think" key=${t.ts + i}><span class="ts">${clock(t.ts)}</span>${t.text}</div>`) : html`<div class="faint">Nothing yet.</div>`}
        <h4 class="label">History · ${d.history.length}</h4>
        ${d.history.length ? d.history.slice().reverse().slice(0, 12).map((h) => { const v = target ? h.scores[target.id] : null; const sc = target ? Math.max(target.target, target.lora || 0) * 1.3 : 1; return html`<div class="hist"><span class="mono" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title=${h.describe}>${h.describe}</span><div class="mini"><i style=${{ width: Math.min(100, ((v || 0) / sc) * 100) + '%', background: h.status === 'ok' ? 'var(--accent)' : 'var(--bad)' }}></i></div><span class=${'num ' + (h.status === 'ok' ? '' : 'down')} style="text-align:right">${h.status === 'ok' ? pct(v) : h.status}</span></div>`; }) : html`<div class="faint">No scored programs yet.</div>`}
        ${d.nudges.length > 0 && html`<h4 class="label">Your nudges</h4>${d.nudges.slice().reverse().map((n) => html`<div class="nrow"><${Icon} n=${n.applied ? 'check' : 'chevron'} size=${13} /><span>${n.text}</span><span class="grow"></span><${Tag} kind=${n.applied ? 'g' : 'w'}>${n.applied ? 'read' : 'waiting'}<//></div>`)}`}
      `}
    </div>
    <div class="nudgebox">
      <div class="label">Nudge ${a.name}</div>
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">${QUICK.map((q) => html`<button class="pill" disabled=${sending} onClick=${() => send(q)}>${q}</button>`)}</div>
      <div class="row"><input class="input" value=${text} placeholder=${'Tell ' + a.name + ' something, for example "try DeepSeek-R1-Distill as the donor next"'} onInput=${(e) => setText(e.target.value)} onKeyDown=${(e) => { if (e.key === 'Enter') send(); }} />
        <button class="btn primary" disabled=${sending || !text.trim()} onClick=${() => send()}>${sending ? html`<${Spinner} />` : 'Send'}</button></div>
      <div class="faint" style="margin-top:8px;font-size:11.5px">A nudge goes into the agent's next turn and shows in the run activity.</div>
    </div>
  </aside>`;
}
