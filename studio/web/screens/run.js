import { html, useState, useEffect, useRef, useMemo } from '../lib.js';
import { api } from '../api.js';
import { go, toast, pct, pctS, sgn, dur, clock, clamp, bsize, primaryTarget, agentColor, STATUS_COLOR, STATUS_ACTIVE, useInterval, addRecent, patchRecent } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Avatar, Dot, Tag, Tabs, BenchBar, BenchLegend, StubBadge, benchScale, Composer, Menu } from '../components.js';
import { AgentView } from './drawer.js';
import { openProgramInBuilder, bestProgram } from './shared.js';

const STATUS_LABEL = { starting: 'Starting', running: 'Running', paused: 'Paused', stopping: 'Stopping', done: 'Finished', error: 'Error' };

function StatusPill({ snap }) {
  const s = snap.status;
  const col = s === 'running' || s === 'starting' ? 'var(--blue)' : s === 'done' ? 'var(--green)' : s === 'error' ? 'var(--red)' : 'var(--amber)';
  return html`<span class="pill"><${Dot} color=${col} pulse=${s === 'running' || s === 'starting'} /><b>${STATUS_LABEL[s] || s}</b> <span class="num">${dur(snap.elapsed_s)}</span></span>`;
}

function AgentCard({ a, sel, target, onClick }) {
  const active = !!STATUS_ACTIVE[a.status];
  const prog = a.best != null && target ? clamp((a.best - target.base) / Math.max(0.01, target.target - target.base), 0, 1) : 0;
  return html`<div class=${'agent' + (sel ? ' sel' : '')} onClick=${onClick} role="button" tabindex="0" onKeyDown=${(e) => { if (e.key === 'Enter') onClick(); }}>
    <div class="hd"><${Avatar} agent=${a} />
      <div style="flex:1;min-width:0"><div class="nm">${a.name}</div><div class="bl">${a.blurb}</div></div>
      <span class="st" title=${a.status}><${Dot} color=${STATUS_COLOR[a.status] || 'var(--text-3)'} pulse=${active} /></span></div>
    <div class="now" title=${a.now}><b>${a.status}</b> ${a.now ? '· ' + a.now : ''}</div>
    ${a.role === 'planner' && html`<div class="mini"><i style=${{ width: prog * 100 + '%', background: 'var(--blue)' }}></i></div>`}
    <div class="ft"><span>${a.best != null ? 'best ' + pct(a.best) + '%' : a.lessons ? a.lessons + ' lessons' : 'no score yet'}</span><span>${a.attempts} attempts${a.collapses ? ' · ' + a.collapses + (a.collapses > 1 ? ' collapses' : ' collapse') : ''}${a.nudges_pending ? ' · ' + a.nudges_pending + ' nudge' : ''}</span></div>
  </div>`;
}

function Kpi({ b, snap, hero }) {
  const v = snap.best ? snap.best.scores[b.id] : null;
  const isT = b.kind === 'target';
  const hit = isT ? v != null && v >= b.target : v != null && v >= b.floor;
  return html`<div class=${'card kpi' + (hero ? ' hero' : '')}>
    <div class="label">${b.name} ${isT ? 'best' : 'guard'}${snap.stub && html`<${StubBadge} small />`}</div>
    <div class=${'v' + (!isT && v != null ? (hit ? ' up' : ' down') : '')}>${v != null ? pctS(v) : pctS(b.base)}</div>
    <div class="s">${isT ? 'target ' + pct(b.target, 0) + ' · start ' + pct(b.base, 0) : 'floor ' + pct(b.floor, 0) + ' · start ' + pct(b.base, 0)}</div>
    <${BenchBar} b=${b} value=${v ?? b.base} tall />
  </div>`;
}

export function LineChart({ curve, bench, total, compact }) {
  const W = 520, H = compact ? 150 : 230, L = 40, R = 14, T = 16, B = 24;
  const vals = [bench.base, bench.target, bench.lora, ...curve.map((c) => c.best), ...curve.map((c) => c.score)].filter((x) => x != null);
  let lo = Math.floor((Math.min(...vals) - 0.03) * 20) / 20;
  let hi = Math.ceil((Math.max(...vals) + 0.04) * 20) / 20;
  lo = Math.max(0, lo); hi = Math.min(1, hi);
  const X = (i) => L + (i / Math.max(1, total)) * (W - L - R);
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks = [];
  const n = compact ? 3 : 4;
  for (let i = 0; i <= n; i++) ticks.push(lo + ((hi - lo) * i) / n);
  const pts = curve.map((c, i) => [X(c.eval), Y(c.best)]);
  const line = pts.length ? 'M' + X(0) + ',' + Y(bench.base) + ' ' + pts.map((p) => 'L' + p[0] + ',' + p[1]).join(' ') : '';
  const area = pts.length ? line + ' L' + pts[pts.length - 1][0] + ',' + (H - B) + ' L' + X(0) + ',' + (H - B) + ' Z' : '';
  const gid = 'ga' + (compact ? 'c' : '');
  return html`<svg class="lchart" viewBox=${'0 0 ' + W + ' ' + H} style=${{ width: '100%', height: H + 'px', display: 'block' }}>
    <defs><linearGradient id=${gid} x1="0" x2="0" y1="0" y2="1"><stop offset="0" style="stop-color:var(--blue);stop-opacity:.18"/><stop offset="1" style="stop-color:var(--blue);stop-opacity:0"/></linearGradient></defs>
    ${ticks.map((t) => html`<g><line class="lc-grid" x1=${L} x2=${W - R} y1=${Y(t)} y2=${Y(t)}/><text class="lc-txt" x=${L - 6} y=${Y(t) + 3} font-size="9.5" text-anchor="end">${Math.round(t * 100)}%</text></g>`)}
    <text class="lc-txt" x=${L} y=${H - 6} font-size="9.5">0</text>
    <text class="lc-txt" x=${W - R} y=${H - 6} font-size="9.5" text-anchor="end">eval ${total}</text>
    ${bench.target != null && html`<g><line class="lc-target" x1=${L} x2=${W - R} y1=${Y(bench.target)} y2=${Y(bench.target)} stroke-dasharray="3 4"/><text class="lc-target-t" x=${W - R} y=${Y(bench.target) - 4} font-size="10" text-anchor="end">target ${pct(bench.target, 0)}%</text></g>`}
    ${bench.lora != null && html`<g><line class="lc-lora" x1=${L} x2=${W - R} y1=${Y(bench.lora)} y2=${Y(bench.lora)} stroke-dasharray="7 4"/><text class="lc-lora-t" x=${W - R} y=${Y(bench.lora) + 11} font-size="10" text-anchor="end">LoRA ${pct(bench.lora)}%</text></g>`}
    <line class="lc-base" x1=${L} x2=${W - R} y1=${Y(bench.base)} y2=${Y(bench.base)} stroke-dasharray="1 4"/>
    ${curve.map((c) => html`<circle class="lc-dot" cx=${X(c.eval)} cy=${Y(c.score)} r="2.2"/>`)}
    ${pts.length > 0 && html`<path d=${area} fill=${'url(#' + gid + ')'} /><path class="lc-line" d=${line} fill="none" stroke-width="2" stroke-linejoin="round"/><circle class="lc-end" cx=${pts[pts.length - 1][0]} cy=${pts[pts.length - 1][1]} r="4.5" stroke-width="2"/>`}
    ${!pts.length && html`<text class="lc-txt" x=${W / 2} y=${H / 2} font-size="12" text-anchor="middle">Waiting for the first score</text>`}
  </svg>`;
}

function Leaderboard({ snap, onBuild }) {
  const [open, setOpen] = useState(null);
  const bs = snap.spec.benchmarks;
  const rows = snap.leaderboard;
  const name = (id) => (snap.agents.find((a) => a.id === id) || { name: id }).name;
  const short = (id) => (id.startsWith('planner-') ? 'P' + id.split('-')[1] : id.slice(0, 2).replace(/^./, (c) => c.toUpperCase()));
  if (!rows.length) return html`<div class="faint" style="padding:30px;text-align:center"><${Spinner} /> <span style="margin-left:8px">Waiting for the first scored program</span></div>`;
  return html`<div class="tscroll"><table class="lb" style="table-layout:fixed"><thead><tr><th style="width:36px">#</th><th>Program</th><th style="width:54px">By</th>${bs.map((b) => html`<th style="width:74px">${b.name.length > 9 ? b.name.slice(0, 8) + '.' : b.name}</th>`)}<th style="width:92px">vs random</th><th style="width:68px">Size</th></tr></thead>
    <tbody>${rows.map((r) => {
      const isBest = snap.best && r.program_hash === snap.best.program_hash;
      const ok = r.status === 'ok';
      return html`<tr class=${'row' + (isBest ? ' best' : '')} key=${r.program_hash} onClick=${() => setOpen(open === r.program_hash ? null : r.program_hash)}>
        <td class="num">${r.rank}</td><td><div class="desc" title=${r.describe}>${r.describe}</div></td>
        <td><span class=${'tag' + (isBest ? ' b' : '')} title=${name(r.agent)}>${short(r.agent)}</span></td>
        ${bs.map((b) => { const v = r.scores[b.id]; const bad = b.kind === 'guard' && v != null && v < b.floor; return html`<td class=${'num' + (bad ? ' down' : '')}>${v != null ? pct(v) : '-'}</td>`; })}
        <td>${!ok ? html`<${Tag} kind="r">${r.status}<//>` : r.vs_control == null ? html`<span class="faint">pending</span>` : html`<${Tag} kind=${r.vs_control > 0.02 ? 'g' : ''}>${sgn(r.vs_control)}<//>`}</td>
        <td class="num">${bsize(r.params_b)}</td></tr>
        ${open === r.program_hash && html`<tr class="det"><td colspan=${bs.length + 5}><div class="detgrid">
          ${bs.map((b) => html`<div><div style="display:flex;justify-content:space-between;margin-bottom:6px"><span class="dim">${b.name}</span><b class="num">${pct(r.scores[b.id])}%</b></div><${BenchBar} b=${b} value=${r.scores[b.id]} tall /></div>`)}
          <div style="grid-column:1/-1;display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-top:6px" class="dim">
            <span>by <b class="t1">${name(r.agent)}</b></span><span>${bsize(r.params_b)} parameters</span>
            <span>${r.vs_control == null ? 'random-part control not run yet' : 'target score is ' + (r.vs_control * 100).toFixed(1) + ' points above its random-part control'}</span>
            <span class="mono faint">${r.program_hash.slice(0, 8)}</span><span class="grow"></span>
            <button class="btn sm" onClick=${(e) => { e.stopPropagation(); onBuild(r.program); }}>Open in builder</button>
            <button class="btn sm ghost" onClick=${(e) => { e.stopPropagation(); go('/run/' + snap.run_id + '/agent/' + r.agent); }}>Open ${name(r.agent)}</button></div>
        </div></td></tr>`}`;
    })}</tbody></table></div>`;
}

function BenchmarksTab({ snap }) {
  const bs = snap.spec.benchmarks;
  const t = primaryTarget(snap.spec);
  return html`<div style="padding:6px 16px 16px">
    <div style="display:flex;justify-content:space-between;align-items:center;margin:8px 0 2px"><span class="label">Best per benchmark</span><${BenchLegend} guard /></div>
    ${bs.map((b) => { const v = snap.best ? snap.best.scores[b.id] : null; return html`<div class="brow" style="grid-template-columns:130px 1fr 54px"><span class="nm" title=${b.name}>${b.name}</span><${BenchBar} b=${b} value=${v ?? b.base} tall /><span class="num" style="text-align:right">${pct(v ?? b.base)}%</span></div>`; })}
    ${t && html`<div class="label" style="margin:18px 0 4px">${t.name} per eval</div><${LineChart} curve=${snap.curve} bench=${t} total=${snap.budget.evals_total} compact />`}
  </div>`;
}

function SourcesTab({ snap }) {
  const mx = Math.max(1, ...snap.sources.map((s) => s.uses));
  return html`<div>${snap.sources.map((s) => html`<div class="src"><div style="min-width:0"><div class="mono" style="font-size:12.5px;overflow:hidden;text-overflow:ellipsis">${s.model}</div></div><div class="btrack"><i style=${{ width: (s.uses / mx) * 100 + '%' }}></i></div><span class="num dim" style="text-align:right">${s.uses}</span></div>`)}
    ${snap.sources.length < 2 && html`<div class="faint" style="padding:14px 16px">Donors appear here as agents use them.</div>`}</div>`;
}

function FloatBar({ snap, onPause, onResume, onStop, onOpen, busy }) {
  const b = snap.budget;
  const t = primaryTarget(snap.spec);
  const score = snap.best && t ? snap.best.scores[t.id] : null;
  const done = snap.status === 'done';
  const errd = snap.status === 'error';
  const frac = b.evals_total ? b.evals_used / b.evals_total : 0;
  const cfrac = b.compute_total_s ? b.compute_used_s / b.compute_total_s : 0;
  const per = b.evals_used ? snap.elapsed_s / b.evals_used : 0;
  const left = per * (b.evals_total - b.evals_used);
  const segs = [];
  const n = Math.min(b.evals_total, 30);
  for (let i = 1; i < n; i++) segs.push(i / n);
  const hit = t && score != null && score >= t.target;
  return html`<div class="floatwrap"><div class=${'floatbar' + (done ? ' done' : '')}>
    <div class="fb-row">
      <div class="fb-l"><div class="fb-t">${done ? 'Run finished' : errd ? 'Run stopped by an error' : 'Eval ' + Math.min(b.evals_used + (snap.status === 'running' ? 1 : 0), b.evals_total) + ' of ' + b.evals_total}</div>
        <div class="fb-s">${dur(b.compute_used_s)} of ${dur(b.compute_total_s)} compute${done || errd ? '' : snap.status === 'running' && left > 0 ? ' · about ' + dur(left) + ' left' : ''}</div></div>
      <div class="fb-r"><div class="fb-t">${score != null ? pctS(score) : '-'} ${t && html`<span class="fb-to">to ${pct(t.target, 0)}%</span>`}${snap.stub && html`<${StubBadge} small />`}</div><div class="fb-s">${done ? (hit ? 'target reached' : 'best found') : 'best so far'}</div></div>
      <div class="fb-btns">${done
        ? html`<button class="btn primary" disabled=${busy} onClick=${onOpen}>${busy ? html`<${Spinner} />` : null} Open best in builder <${Icon} n="arrow" size=${14} /></button>`
        : errd ? html`<button class="btn" onClick=${onOpen}>Open best in builder</button>`
        : html`${snap.status === 'paused' ? html`<button class="btn" onClick=${onResume}><${Icon} n="play" size=${14} /> Resume</button>` : html`<button class="btn" disabled=${snap.status === 'stopping'} onClick=${onPause}><${Icon} n="pause" size=${14} /> Pause</button>`}
          <button class="btn primary" disabled=${snap.status === 'stopping'} onClick=${onStop}>${snap.status === 'stopping' ? 'Stopping' : 'Stop and keep best'}</button>`}</div>
    </div>
    <div class=${'bigbar' + (done ? ' ok' : '')}><div class="fill" style=${{ width: frac * 100 + '%' }}></div>${segs.map((s) => html`<span class="seg" style=${{ left: s * 100 + '%' }}></span>`)}<div class="cfill" style=${{ width: cfrac * 100 + '%' }} title="compute used"></div></div>
  </div></div>`;
}


const USER_KINDS = { nudge_sent: 1 };
// the text a user nudge carries, without the log prefix
function nudgeText(e) {
  const m = /^(?:Nudge for [^:]+|You sent a nudge): (.*)$/s.exec(e.text);
  return m ? m[1] : e.text;
}
function isUserNudge(e) { return USER_KINDS[e.kind] || (e.kind === 'nudge' && /^Nudge for /.test(e.text)); }

// turns the activity list into display blocks: user bubbles and agent groups
function buildBlocks(snap) {
  const blocks = [];
  const planners = snap.agents.filter((a) => a.role === 'planner').length;
  for (const e of snap.activity) {
    if (isUserNudge(e)) {
      const t = nudgeText(e);
      const last = blocks[blocks.length - 1];
      if (last && last.type === 'user' && last.text === t && Math.abs(e.ts - last.ts) < 3) { last.n++; last.agents.push(e.agent); continue; }
      blocks.push({ type: 'user', ts: e.ts, text: t, n: 1, agents: [e.agent], key: 'u' + e.ts + e.agent });
      continue;
    }
    const last = blocks[blocks.length - 1];
    if (last && last.type === 'agent' && last.agent === e.agent) last.items.push(e);
    else blocks.push({ type: 'agent', agent: e.agent, items: [e], key: 'a' + e.ts + e.agent });
  }
  blocks.forEach((b) => { if (b.type === 'user') b.all = b.n > 1 && b.n >= planners; });
  return blocks;
}

function EvalLine({ e }) {
  const m = /^.*? scored (.*?): (.*)$/s.exec(e.text);
  if (!m) return html`<div class=${'msg-line ' + e.status}>${e.text}</div>`;
  const scores = m[2].split(', ');
  return html`<div class=${'evline ' + e.status}><div class="evh"><${Dot} color=${e.status === 'bad' ? 'var(--red)' : 'var(--green)'} /><span class="evl">${e.status === 'bad' ? 'Collapsed' : 'Scored'}</span><span class="mono evp ell" title=${m[1]}>${m[1]}</span></div><div class="evs">${scores.map((x) => html`<span class="evc">${x}</span>`)}</div></div>`;
}

function AgentMsgs({ block, agents, onOpen }) {
  const a = agents.find((x) => x.id === block.agent) || { id: block.agent, name: block.agent, role: block.agent };
  return html`<div class="amsg">
    <div class="ahead"><${Avatar} agent=${a} size=${24} /><button class="aname" onClick=${() => onOpen(a.id)} title=${'Open ' + a.name}>${a.name}</button><span class="rtag">${a.role}</span></div>
    ${block.items.map((e) => e.kind === 'eval' || e.kind === 'collapse' ? html`<${EvalLine} e=${e} key=${e.ts + e.text} />`
      : html`<div class=${'msg-line ' + e.status} key=${e.ts + e.text}>${e.text}</div>`)}
  </div>`;
}

function specLine(spec, snap) {
  const bits = spec.benchmarks.map((b) => b.name + (b.kind === 'target' ? ' to ' + pct(b.target, 0) + '%' : ' over ' + pct(b.floor, 0) + '%'));
  return bits.join(', ') + '. Up to ' + spec.size.max_params_b + 'B parameters, ' + snap.budget.evals_total + ' evals.';
}

function Conversation({ snap, id, onOpen }) {
  const ref = useRef();
  const stick = useRef(true);
  const [fresh, setFresh] = useState(0);
  const [text, setText] = useState('');
  const [to, setTo] = useState('all');
  const [pick, setPick] = useState(false);
  const [sending, setSending] = useState(false);
  const blocks = buildBlocks(snap);
  const thinking = snap.status === 'running' || snap.status === 'starting' ? snap.agents.filter((a) => a.status === 'thinking') : [];
  const count = snap.activity.length;
  const prevCount = useRef(count);
  const scrollEnd = () => { const el = ref.current; if (el) el.scrollTop = el.scrollHeight; };
  useEffect(() => { scrollEnd(); }, []);
  useEffect(() => {
    if (count === prevCount.current) return;
    const added = count - prevCount.current;
    prevCount.current = count;
    if (stick.current) scrollEnd(); else if (added > 0) setFresh((n) => n + added);
  }, [count]);
  const onScroll = () => {
    const el = ref.current;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (stick.current) setFresh(0);
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
  return html`<section class="convo">
    <div class="msgs" ref=${ref} onScroll=${onScroll}><div class="msgs-in">
      <div class="umsg"><div class="bubble"><div class="bt1">${snap.spec.title}</div><div class="bt2">${specLine(snap.spec, snap)}</div></div></div>
      ${blocks.map((b) => b.type === 'user'
        ? html`<div class="umsg" key=${b.key}><div class="bubble"><div class="bto">${b.all ? 'To all planners' : 'To ' + ((snap.agents.find((a) => a.id === b.agents[0]) || {}).name || b.agents[0])}</div>${b.text}</div></div>`
        : html`<${AgentMsgs} key=${b.key} block=${b} agents=${snap.agents} onOpen=${onOpen} />`)}
      ${thinking.map((a) => html`<div class="thinkrow" key=${'t' + a.id}><${Avatar} agent=${a} size=${24} /><span>${a.name} is thinking</span><span class="dots"><i></i><i></i><i></i></span></div>`)}
      ${snap.status === 'done' && html`<div class="endrow">Run finished. Open the best program in the builder from the panel.</div>`}
    </div></div>
    ${fresh > 0 && html`<button class="newpill" onClick=${() => { scrollEnd(); stick.current = true; setFresh(0); }}><${Icon} n="down" size=${14} /> ${fresh} new message${fresh > 1 ? 's' : ''}</button>`}
    <div class="convo-dock"><div class="dock-in">
      <${Composer} value=${text} onInput=${setText} onSubmit=${send} disabled=${sending} minRows=${1} label="Nudge the agents" footLeft=${picker} placeholder=${target ? 'Tell ' + target.name + ' what to try next' : 'Tell the planners what to try next'} />
      <div class="dock-note">Agents can be wrong. Check results on the held-out split before you use them.</div>
    </div></div>
  </section>`;
}

function Overview({ snap, id, agent, onOpen, onBuild }) {
  const [tab, setTab] = useState('leaderboard');
  const spec = snap.spec;
  const t = primaryTarget(spec);
  const bestRow = snap.best ? snap.leaderboard.find((r) => r.program_hash === snap.best.program_hash) : null;
  const targets = spec.benchmarks.filter((b) => b.kind === 'target');
  const guards = spec.benchmarks.filter((b) => b.kind === 'guard');
  const bestScore = snap.best && t ? snap.best.scores[t.id] : null;
  const gap = bestScore != null && t && t.lora != null ? bestScore - t.lora : null;
  const tabs = [{ id: 'leaderboard', label: 'Leaderboard', n: snap.leaderboard.length }, { id: 'benchmarks', label: 'Benchmarks' }, { id: 'sources', label: 'Source models', n: snap.sources.length }];
  return html`<div class="ov">
    <div class="sec-sm">Agents</div>
    <div class="agrid">${snap.agents.map((a) => html`<${AgentCard} key=${a.id} a=${a} sel=${agent === a.id} target=${t} onClick=${() => onOpen(a.id)} />`)}</div>
    <div class="kpis">
      ${targets.map((b) => html`<${Kpi} key=${b.id} b=${b} snap=${snap} />`)}
      ${guards.map((b) => html`<${Kpi} key=${b.id} b=${b} snap=${snap} />`)}
      <div class="card kpi"><div class="label">Size${snap.stub && html`<${StubBadge} small />`}</div><div class="v">${bestRow ? bsize(bestRow.params_b) : bsize(spec.size.host_params_b)}</div><div class="s">limit ${spec.size.max_params_b}B · host ${bsize(spec.size.host_params_b)}</div>
        <div class="mini" style="margin-top:12px"><i style=${{ width: clamp(((bestRow ? bestRow.params_b : spec.size.host_params_b) / spec.size.max_params_b) * 100, 0, 100) + '%', background: (bestRow ? bestRow.params_b : spec.size.host_params_b) > spec.size.max_params_b ? 'var(--red)' : 'var(--blue)' }}></i></div></div>
      <div class="card kpi"><div class="label">vs fine-tuning${snap.stub && html`<${StubBadge} small />`}</div>
        <div class=${'v ' + (gap == null ? '' : gap >= 0 ? 'up' : 'down')}>${gap == null ? 'n/a' : sgn(gap)}</div>
        <div class="s">${t && t.lora != null ? 'LoRA bar ' + pctS(t.lora) : 'no LoRA bar known'}</div></div>
    </div>
    <div class="card panel"><${Tabs} tabs=${tabs} value=${tab} onChange=${setTab} right=${tab === 'leaderboard' ? html`<span class="faint" style="font-size:12px">click a row for detail</span>` : null} />
      <div style="min-height:160px">
        ${tab === 'leaderboard' && html`<${Leaderboard} snap=${snap} onBuild=${onBuild} />`}
        ${tab === 'benchmarks' && html`<${BenchmarksTab} snap=${snap} />`}
        ${tab === 'sources' && html`<${SourcesTab} snap=${snap} />`}
      </div></div>
    <div class="card chartcard"><div style="display:flex;justify-content:space-between;align-items:center"><span class="sec-sm">${t ? 'Best ' + t.name + ' vs evals' : 'Best score vs evals'}</span><${BenchLegend} /></div>
      <div style="margin-top:6px">${t && html`<${LineChart} curve=${snap.curve} bench=${t} total=${snap.budget.evals_total} />`}</div></div>
  </div>`;
}

export function RunScreen({ id, agent }) {
  const [snap, setSnap] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(() => !!agent || window.innerWidth >= 1400);
  const [exp, setExp] = useState(false);
  const [tabs, setTabs] = useState(agent ? [agent] : []);

  useEffect(() => {
    let closed = false, poll = null;
    const on = (s) => { if (!closed) { setSnap(s); setErr(null); } };
    const startPoll = () => { if (!poll && !closed) poll = setInterval(() => api.getRun(id).then(on).catch(() => {}), 2000); };
    api.getRun(id).then(on).catch((e) => { if (!closed) setErr(e); });
    const close = api.openStream(id, on, startPoll);
    return () => { closed = true; close(); if (poll) clearInterval(poll); };
  }, [id]);
  useEffect(() => {
    if (!snap) return;
    addRecent({ kind: 'run', id, title: snap.spec.title, status: snap.status });
  }, [id, !!snap]);
  useEffect(() => { if (snap) patchRecent('run', id, { status: snap.status }); }, [snap && snap.status]);
  useEffect(() => { if (agent) { setTabs((t) => (t.includes(agent) ? t : [...t, agent])); setOpen(true); } }, [agent]);

  const toggle = html`<button class=${'btn sm' + (open ? ' ghost' : '')} onClick=${() => setOpen(!open)} title=${open ? 'Close the workspace' : 'Open the workspace'} aria-pressed=${open}><${Icon} n="panel" size=${16} /> Workspace</button>`;
  if (err && !snap) return html`<${TopBar} /><${ErrorBox} error=${err} onRetry=${() => location.reload()} />`;
  if (!snap) return html`<div class="runview"><${TopBar} /><div class="run-head"><div class="skeleton" style="height:32px;width:320px"></div></div><div class="task"><div class="convo"><div class="msgs"><div class="msgs-in"><div class="skeleton" style="height:90px"></div><div class="skeleton" style="height:140px"></div></div></div></div></div></div>`;

  const spec = snap.spec;
  const act = async (fn) => { try { await fn(id); } catch (e) { toast(e.message, 'err'); } };
  const openBest = async (prog) => {
    setBusy(true);
    const p = prog || (await bestProgram(id, snap));
    if (!p) toast('The server did not send the best program. Opening an empty builder.', 'info');
    await openProgramInBuilder(spec.id, p);
    setBusy(false);
  };
  const openAgent = (aid) => go('/run/' + id + '/agent/' + aid);
  const closeTab = (aid) => { setTabs((t) => t.filter((x) => x !== aid)); if (agent === aid) go('/run/' + id); };
  const names = Object.fromEntries(snap.agents.map((a) => [a.id, a]));
  const active = agent && tabs.includes(agent) ? agent : 'overview';

  return html`<div class="runview"><${TopBar} right=${html`${spec.benchmarks.slice(0, 2).map((b) => html`<span class="pill hide-sm">${b.name} <b>${b.kind === 'target' ? 'to ' + pct(b.target, 0) : 'over ' + pct(b.floor, 0)}%</b></span>`)}<span class="pill hide-sm">${snap.budget.evals_total - snap.budget.evals_used} evals left</span>${toggle}`} />
    <div class="run-head">
      <div class="rh-main"><h1 class="page-title ell">${spec.title}</h1><${StatusPill} snap=${snap} /></div>
      <a class="btn" href=${'#/spec/' + spec.id}>Edit spec</a>
    </div>
    <div class=${'task' + (open && exp ? ' exp' : '')}>
      <${Conversation} snap=${snap} id=${id} onOpen=${openAgent} />
      ${open && html`<aside class="wspanel" aria-label="Workspace">
        <div class="wstabs">
          <div class="chips32">
            <button class=${'tchip' + (active === 'overview' ? ' on' : '')} onClick=${() => go('/run/' + id)}>Overview</button>
            ${tabs.map((aid) => html`<span class=${'tchip' + (active === aid ? ' on' : '')} key=${aid}><button class="tlabel" onClick=${() => openAgent(aid)}>${(names[aid] || { name: aid }).name}</button><button class="tx" title="Close tab" aria-label=${'Close ' + ((names[aid] || { name: aid }).name)} onClick=${() => closeTab(aid)}><${Icon} n="x" size=${12} /></button></span>`)}
          </div>
          <span class="grow"></span>
          <button class="iconbtn sq hide-narrow" title=${exp ? 'Shrink the panel' : 'Expand the panel'} aria-label="Expand" onClick=${() => setExp(!exp)}><${Icon} n=${exp ? 'left' : 'arrow'} size=${16} /></button>
          <button class="iconbtn sq" title="Close the panel" aria-label="Close panel" onClick=${() => setOpen(false)}><${Icon} n="x" size=${16} /></button>
        </div>
        <div class="wsbody">
          ${active === 'overview' ? html`<${Overview} snap=${snap} id=${id} agent=${agent} onOpen=${openAgent} onBuild=${openBest} />` : html`<${AgentView} key=${active} runId=${id} agentId=${active} snap=${snap} />`}
        </div>
        <${FloatBar} snap=${snap} busy=${busy} onPause=${() => act(api.pause)} onResume=${() => act(api.resume)} onStop=${() => act(api.stop)} onOpen=${() => openBest()} />
      </aside>`}
    </div>
  </div>`;
}
