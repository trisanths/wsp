// The Activity page: a readable log of the run, grouped by round, with a composer to nudge agents.
import { html, useState, useEffect, useLayoutEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { toast, pctS } from '../util.js';
import { Icon, Composer, Menu } from '../components.js';
import { StreamText, SubtleNum, usePresence, presenceClass } from '../motion.js';
import { nameOf } from './run-shared.js';

// Older servers use other kind names and send no id, round or data. These map them to the new ones.
const ALIAS = { llm_plan: 'plan', analyst_map: 'map', judge_verdict: 'verdict', nudge_sent: 'nudge', stop: 'milestone', done: 'milestone' };
const RESULT_LABEL = { rejected_fit: 'fit too low', compile_error: 'did not compile', invalid: 'not valid', infra_error: 'GPU error', budget_exhausted: 'no budget', collapsed: 'collapsed' };

function norm(e) {
  let kind = ALIAS[e.kind] || e.kind;
  if (e.kind === 'nudge' && e.agent !== 'you' && !e.data) kind = 'nudge_applied';
  const key = e.id != null ? 'i' + e.id : e.ts + '|' + e.agent + '|' + e.kind + '|' + String(e.text).slice(0, 24);
  return { ...e, kind, key, data: e.data || null, round: e.round == null ? null : e.round };
}
function nudgeText(e) {
  const m = /^(?:Nudge for [^:]+|You sent a nudge): (.*)$/s.exec(e.text);
  return m ? m[1] : e.text;
}
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

// Turns the activity list into groups: { round, rows }, plus the held-out rows for the end.
function buildLog(snap) {
  const planners = snap.agents.filter((a) => a.role === 'planner').length;
  const nameById = Object.fromEntries(snap.agents.map((a) => [a.id, a.name]));
  const groups = [];
  const held = [];
  let cur = null;
  for (const raw of snap.activity) {
    const e = norm(raw);
    if (e.kind === 'heldout') { held.push(e); continue; }
    if (!cur || (e.round != null && e.round !== cur.round)) { cur = { round: e.round, rows: [], key: 'g' + (e.round == null ? 'x' : e.round) + groups.length, probes: null }; groups.push(cur); }
    if (e.kind === 'probe') {
      if (!cur.probes) { cur.probes = { type: 'probes', key: 'p' + e.key, items: [] }; cur.rows.push(cur.probes); }
      cur.probes.items.push(e);
      continue;
    }
    if (e.kind === 'nudge') {
      const t = nudgeText(e);
      // one nudge to every planner arrives as several items: merge them, even with other lines between
      const last = cur.rows.slice(-6).reverse().find((r) => r.type === 'nudge' && r.text === t && Math.abs(e.ts - r.ts) < 3);
      if (last) { last.n++; last.to.push(e.data && e.data.to ? e.data.to : e.agent); last.toName = null; continue; }
      const to = e.data && e.data.to_name ? e.data.to_name : null;
      cur.rows.push({ type: 'nudge', key: e.key, ts: e.ts, text: t, n: 1, to: [e.data && e.data.to ? e.data.to : e.agent], toName: to, toAll: e.data && e.data.to === 'all' });
      continue;
    }
    cur.rows.push({ type: 'line', key: e.key, e });
  }
  return { groups, held, planners, nameById };
}

function sentence(e, nameById) {
  const name = nameById[e.agent];
  let t = String(e.text || '');
  if (name && t.startsWith(name + ' ')) t = t.slice(name.length + 1);
  return cap(t);
}

function Fold({ open, onToggle, label, children }) {
  return html`<div class="t-acc" data-open=${open ? 'true' : 'false'}>
    <button class="foldbtn" onClick=${onToggle} aria-expanded=${open}>${label}<span class="t-acc-chevron"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5L8 10.5L12 6.5"/></svg></span></button>
    <div class="t-acc-panel"><div class="t-acc-panel-inner">${children}</div></div>
  </div>`;
}

function Line({ e, showName, fresh, ctx }) {
  const [open, setOpen] = useState(false);
  const d = e.data;
  const text = sentence(e, ctx.nameById);
  const who = ctx.nameById[e.agent] || e.agent;
  if (e.kind === 'milestone') return html`<div class="lmile">${e.text}</div>`;
  if (e.kind === 'nudge_applied') return html`<div class="al"><span class="ln"></span><span class="lt dim3">${(ctx.nameById[e.agent] || 'An agent') + ' read your nudge'}</span><span></span></div>`;
  let cls = 'lt';
  let right = null;
  let sub = null;
  if (e.kind === 'eval') {
    if (d) {
      const result = d.result || 'ok';
      if (result === 'collapsed') {
        cls += ' bad';
        const failed = (d.guards || []).filter((g) => !g.ok).map((g) => g.name);
        if (failed.length) sub = html`<div class="lsub bad">Failed: ${failed.join(', ')}</div>`;
      } else if (result !== 'ok') {
        cls += ' dim3';
        right = html`<span class="lr dim3">${RESULT_LABEL[result] || result}</span>`;
      } else {
        right = html`<span class="lr">${d.is_best && html`<span class="tag g">new best</span>`}<span class="num lscore"><${SubtleNum} text=${d.target != null ? pctS(d.target) : '-'} /></span></span>`;
      }
    } else if (e.status === 'bad') cls += ' bad';
  } else if (e.kind === 'plan') cls += ' dim3';
  else if (e.kind === 'system' || e.kind === 'handoff') cls += ' dim3';
  else if (e.kind === 'error' || e.status === 'bad') cls += ' bad';
  const hasDetail = e.kind === 'map' && d && d.detail;
  return html`<div class="al">
    <span class="ln">${showName ? who : ''}</span>
    <div class="lbody"><${StreamText} class=${cls} text=${text} play=${fresh} />${sub}
      ${hasDetail && html`<${Fold} open=${open} onToggle=${() => setOpen(!open)} label=${open ? 'Hide details' : 'Show details'}><pre class="detwell mono">${d.detail}</pre><//>`}
    </div>
    ${right || html`<span></span>`}
  </div>`;
}

function ProbeFold({ row, showName, fresh, ctx }) {
  const [open, setOpen] = useState(false);
  const n = row.items.length;
  const who = ctx.nameById[row.items[0].agent] || '';
  return html`<div class="al">
    <span class="ln">${showName ? who : ''}</span>
    <div class="lbody"><${Fold} open=${open} onToggle=${() => setOpen(!open)} label=${html`<span class="dim3"><${SubtleNum} text=${String(n)} /> fit ${n === 1 ? 'check' : 'checks'}</span>`}>
      <div class="plines">${row.items.map((p) => html`<div class="pl" key=${p.key}>${sentence(p, ctx.nameById)}</div>`)}</div>
    <//></div><span></span>
  </div>`;
}

function Nudge({ row, ctx }) {
  const all = row.toAll || (!row.toName && row.n > 1 && row.n >= ctx.planners);
  const names = row.to.map((x) => ctx.nameById[x] || x);
  const label = all ? 'To all planners' : 'To ' + (row.toName || names.join(', '));
  return html`<div class="unudge"><div class="ulabel">${label}</div><div class="ubub">${row.text}</div></div>`;
}

function NewPill({ unread, onClick }) {
  const st = usePresence(unread > 0, 200);
  const last = useRef(unread);
  if (unread > 0) last.current = unread;
  if (st === 'out') return null;
  return html`<button class=${'newpill t-toast ' + presenceClass(st)} onClick=${onClick}>New updates<span class="t-badge" data-open=${unread > 0 ? 'true' : 'false'}><span class="t-badge-dot">${last.current}</span></span></button>`;
}

export function ActivityPage({ snap, id }) {
  const log = buildLog(snap);
  const scroller = useRef();
  const stick = useRef(true);
  const seen = useRef(null);
  const [unread, setUnread] = useState(0);
  const [text, setText] = useState('');
  const [to, setTo] = useState('all');
  const [pick, setPick] = useState(false);
  const [sending, setSending] = useState(false);
  const clearRef = useRef();
  const keys = [];
  log.groups.forEach((g) => g.rows.forEach((r) => keys.push(r.key)));
  log.held.forEach((h) => keys.push(h.key));
  // the first render is the baseline: only items that arrive later stream in
  if (!seen.current) seen.current = new Set(keys);
  const fresh = (k) => !seen.current.has(k);
  const total = keys.length;
  const prevTotal = useRef(total);
  const scrollEnd = () => { const el = scroller.current; if (el) el.scrollTop = el.scrollHeight; };
  useLayoutEffect(() => { scrollEnd(); }, []);
  useEffect(() => {
    keys.forEach((k) => seen.current.add(k));
    const added = total - prevTotal.current;
    prevTotal.current = total;
    if (added <= 0) return;
    if (stick.current) scrollEnd(); else setUnread((n) => n + added);
  }, [total, snap.activity.length, snap.activity.length ? snap.activity[snap.activity.length - 1].id || snap.activity[snap.activity.length - 1].ts : 0]);
  const onScroll = () => {
    const el = scroller.current;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (stick.current) setUnread(0);
  };

  const planners = snap.agents.filter((a) => a.role === 'planner');
  const target = to === 'all' ? null : snap.agents.find((a) => a.id === to);
  const send = async () => {
    const t = text.trim();
    if (!t) return;
    setSending(true);
    try {
      const ids = target ? [target.id] : planners.map((a) => a.id);
      await Promise.all(ids.map((aid) => api.nudge(id, aid, t)));
      if (clearRef.current) clearRef.current();
      setText('');
      stick.current = true;
      toast(target ? 'Nudge sent to ' + target.name : 'Nudge sent to all planners', 'ok');
    } catch (e) { toast(e.message, 'err'); }
    setSending(false);
  };
  const picker = html`<span class="pickwrap"><button class="pickbtn" type="button" onClick=${() => setPick(!pick)} aria-haspopup="menu"><${Icon} n="spark" size=${14} />${target ? target.name : 'All planners'}<${Icon} n="down" size=${12} /></button>
    <${Menu} open=${pick} onClose=${() => setPick(false)} class="up" style=${{ bottom: '36px', left: 0, minWidth: '190px' }}>
      <button class=${'mi' + (to === 'all' ? ' on' : '')} onClick=${() => { setTo('all'); setPick(false); }}><span>All planners</span>${to === 'all' && html`<${Icon} n="check" size=${14} />`}</button>
      ${snap.agents.map((a) => html`<button class=${'mi' + (to === a.id ? ' on' : '')} key=${a.id} onClick=${() => { setTo(a.id); setPick(false); }}><span>${a.name}</span>${to === a.id && html`<${Icon} n="check" size=${14} />`}</button>`)}
    <//></span>`;

  const ctx = { nameById: log.nameById, planners: log.planners };
  const cat = null;
  return html`<div class="rpage act">
    <div class="alog" ref=${scroller} onScroll=${onScroll}><div class="col narrow">
      ${log.groups.length === 0 && html`<div class="faint">Nothing has happened yet.</div>`}
      ${log.groups.map((g) => {
        let prev = null;
        return html`<section class="rgroup" key=${g.key}>
          ${g.round != null && html`<div class="rdiv">Round ${g.round}</div>`}
          ${g.rows.map((r) => {
            if (r.type === 'nudge') { prev = null; return html`<${Nudge} key=${r.key} row=${r} ctx=${ctx} />`; }
            const agent = r.type === 'probes' ? r.items[0].agent : r.e.agent;
            const solo = r.type === 'line' && (r.e.kind === 'milestone' || r.e.kind === 'nudge_applied');
            const showName = !solo && agent !== prev;
            if (!solo) prev = agent;
            return r.type === 'probes'
              ? html`<${ProbeFold} key=${r.key} row=${r} showName=${showName} ctx=${ctx} />`
              : html`<${Line} key=${r.key} e=${r.e} showName=${showName} fresh=${fresh(r.key)} ctx=${ctx} />`;
          })}
        </section>`;
      })}
      ${log.held.length > 0 && html`<section class="rgroup">
        <div class="rdiv">Held-out check</div>
        ${log.held.map((h) => html`<div class="al held" key=${h.key}><span class="ln"></span><span class="lt ell">${nameOf(h.data, cat) || h.text}</span><span class="lr num lscore">${h.data && h.data.target != null ? pctS(h.data.target) : ''}</span></div>`)}
      </section>`}
    </div></div>
    <${NewPill} unread=${unread} onClick=${() => { scrollEnd(); stick.current = true; setUnread(0); }} />
    <div class="adock"><div class="col narrow">
      <${Composer} value=${text} onInput=${setText} onSubmit=${send} disabled=${sending} minRows=${1} clearRef=${clearRef} label="Nudge the agents" footLeft=${picker} placeholder=${target ? 'Tell ' + target.name + ' what to try next' : 'Tell the planners what to try next'} />
      <div class="dock-note">Agents can be wrong. Check results on the held-out split before you use them.</div>
    </div></div>
  </div>`;
}
