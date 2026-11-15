// The agents of a run: the board (all agents side by side, with how they work together) and one agent's mind view.
import { html, useState, useEffect, useRef, useMemo } from '../lib.js';
import { api } from '../api.js';
import { toast, clock, ago, pctS, primaryTarget, useInterval } from '../util.js';
import { Dot, Composer, Avatar } from '../components.js';
import { ThinkLine, StreamText, SubtleNum } from '../motion.js';
import { PoweredBy, SponsorMark, ModelLogo } from '../logos.js';
import { Brain3D } from '../brain3d.js';
import { openProgramInBuilder } from './shared.js';
import { nameOf, progName, ACTIVE, agentDot, STATUS_WORD, isEmptyEnd } from './run-shared.js';
import { deriveFindings, ProgramStrip, PartChip, VDot, kindOf, actKey, shortDonor, partLabel } from './run-findings.js';
import { openPart } from './part-detail.js';

const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const plural = (n, one, many) => n + ' ' + (n === 1 ? one : many || one + 's');
const pts = (x, d = 1) => (x >= 0 ? '+' : '-') + Math.abs(x * 100).toFixed(d);
const ONMODAL = /^(compile|heal|eval)\b/;
const VERBS = { probe: 'Fit check', compile: 'Compile', heal: 'Heal', eval: 'Score', read_results: 'Read results', control: 'Random-part control' };
const RESULT_LABEL = { rejected_fit: 'Fit too low', compile_error: 'Did not compile', invalid: 'Not valid', infra_error: 'GPU error', budget_exhausted: 'No budget' };
const ROLE = {
  analyst: { label: 'Analyst', does: 'Maps which parts of the donor models fit where, and writes the map for the planners.' },
  planner: { label: 'Planner', does: 'Proposes programs to try: which parts to graft, at which layer, with how much heal.' },
  judge: { label: 'Judge', does: 'Scores every program, runs the random-part controls and keeps the best ones.' },
};
const roleOf = (a) => (a && (a.role || String(a.id).replace(/-\d+$/, ''))) || 'planner';

function plainCall(call) {
  const m = /^(\w+)\((.*)\)$/s.exec(call || '');
  if (!m) return call;
  const label = VERBS[m[1]] || cap(m[1].replace(/_/g, ' '));
  return ['heal', 'eval', 'read_results'].includes(m[1]) && m[2] ? label + ' (' + m[2] + ')' : label;
}

/* ---------- data ---------- */
// The detail of every agent, kept fresh while the run is live.
function useAgentDetails(id, agents, live) {
  const [map, setMap] = useState({});
  const ids = agents.map((a) => a.id).join(',');
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);
  const load = () => Promise.all(agents.map((a) => api.getAgent(id, a.id).then((x) => [a.id, x]).catch(() => null))).then((rs) => {
    if (!alive.current) return;
    setMap((m) => { const n = { ...m }; rs.forEach((r) => { if (r) n[r[0]] = r[1]; }); return n; });
  });
  useEffect(() => { load(); }, [id, ids]);
  useInterval(load, live ? 2500 : null);
  return map;
}

// where a score landed against the best before it, read from the curve
function deltaOf(snap, hash, score) {
  const c = snap.curve || [];
  const i = c.findIndex((x) => x.program_hash === hash);
  if (i < 0 || score == null) return null;
  const prev = i > 0 ? c[i - 1].best : null;
  if (prev == null) {
    const t = primaryTarget(snap.spec);
    const base = t ? (snap.base_scores || {})[t.id] ?? t.base : null;
    return base != null ? { d: score - base, vs: 'host' } : null;
  }
  return { d: score - prev, vs: 'best' };
}

function rowOf(snap, hash) { return (snap.leaderboard || []).find((r) => r.program_hash === hash) || null; }

/* ---------- small parts ---------- */
function Spark({ pts: P, w = 240, h = 32, label }) {
  if (!P.length) return html`<div class="spark none faint">No scores yet</div>`;
  const vals = P.map((p) => p.v);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;
  const x = (i) => (P.length === 1 ? w / 2 : 4 + (i * (w - 8)) / (P.length - 1));
  const y = (v) => h - 4 - ((v - lo) / span) * (h - 8);
  const d = P.map((p, i) => (i ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(p.v).toFixed(1)).join(' ');
  return html`<svg class="spark" viewBox=${'0 0 ' + w + ' ' + h} width="100%" height=${h} role="img" aria-label=${label || 'Scores over time'} preserveAspectRatio="none">
    ${P.length > 1 && html`<path class="sp-line" d=${d} pathLength="1" />`}
    ${P.map((p, i) => html`<circle key=${i} class=${'sp-dot' + (p.bad ? ' bad' : p.best ? ' best' : '') + (i === P.length - 1 ? ' last' : '')} cx=${x(i)} cy=${y(p.v)} r=${i === P.length - 1 ? 3 : 2} />`)}
  </svg>`;
}

function sparkFor(snap, a) {
  const c = snap.curve || [];
  if (a.id === 'judge') return c.map((p) => (p.best != null ? { v: p.best, best: p.is_best } : null)).filter(Boolean);
  if (roleOf(a) !== 'planner') return [];
  return c.filter((p) => p.agent === a.id && p.score != null).map((p) => ({ v: p.score, bad: p.status === 'collapsed', best: p.is_best }));
}

function StepTrail({ status, ended }) {
  const steps = [['thinking', 'Think'], ['probing', 'Probe fit'], ['compiling', 'Compile and heal'], ['scoring', 'Score']];
  const at = ended ? -1 : steps.findIndex((s) => s[0] === status);
  return html`<ol class="steptrail" aria-label="Where it is in an attempt">${steps.map(([k, l], i) => html`<li class=${i === at ? 'now' : at >= 0 && i < at ? 'past' : ''} key=${k}><span class="st-dot"></span><span>${l}</span>${i === 2 && html`<${SponsorMark} who="modal" size=${12} title="Compile, heal and score run on Modal" />`}${i === 0 && html`<${SponsorMark} who="openai" size=${12} title="Thinking by OpenAI" />`}</li>`)}</ol>`;
}

/* ---------- how the agents work together ---------- */
const LINKS = { map: ['a2p', 'memw'], plan: ['p2j', 'memw'], eval: ['p2j', 'j2p'], verdict: ['j2p', 'memw'], heldout: ['j2p'], nudge_applied: ['memr'], nudge: ['memr'], probe: ['p2j'], handoff: ['memr', 'memw'] };

function Arrow({ dir = 'h', rev, hot, label, n }) {
  return html`<div class=${'carrow ' + dir + (rev ? ' rev' : '') + (hot ? ' hot' : '')} aria-hidden=${label ? null : 'true'}><span class="cline"></span><span class="chead"></span>${hot && html`<span class="cdot" key=${hot}></span>`}${label && html`<span class="clab">${label}${n != null && html` <b class="num">${n}</b>`}</span>`}</div>`;
}

function Collab({ snap, running }) {
  const acts = useMemo(() => (snap.activity || []).map((e) => ({ ...e, kind: kindOf(e), key: actKey(e) })), [snap.activity]);
  const [hot, setHot] = useState({});
  const [nodes, setNodes] = useState({});
  const seen = useRef(null);
  const timer = useRef();
  useEffect(() => {
    if (seen.current === null) { seen.current = new Set(acts.map((e) => e.key)); return; }
    const fresh = acts.filter((e) => !seen.current.has(e.key));
    if (!fresh.length) return;
    fresh.forEach((e) => seen.current.add(e.key));
    const k = Date.now();
    const h = {}, nd = {};
    for (const e of fresh) {
      for (const l of LINKS[e.kind] || []) h[l] = k + Math.random();
      const r = e.agent === 'you' ? null : roleOf(snap.agents.find((a) => a.id === e.agent) || { id: e.agent });
      if (r) nd[r] = k + Math.random();
    }
    setHot((p) => ({ ...p, ...h }));
    setNodes((p) => ({ ...p, ...nd }));
    clearTimeout(timer.current);
    timer.current = setTimeout(() => { setHot({}); setNodes({}); }, 1500);
  }, [acts.length, acts.length ? acts[acts.length - 1].key : '']);
  useEffect(() => () => clearTimeout(timer.current), []);

  const cnt = (k) => acts.filter((e) => e.kind === k).length;
  const planners = snap.agents.filter((a) => roleOf(a) === 'planner');
  const analyst = snap.agents.find((a) => roleOf(a) === 'analyst');
  const judge = snap.agents.find((a) => roleOf(a) === 'judge');
  const scored = (snap.leaderboard || []).length;
  const shared = acts.filter((e) => ['plan', 'map', 'verdict', 'handoff', 'nudge_applied'].includes(e.kind)).length;
  const lessons = snap.agents.reduce((s, a) => s + (a.lessons || 0), 0);
  const act = (a) => a && running && !!ACTIVE[a.status];
  const frontier = (() => { const v = acts.filter((e) => e.kind === 'verdict' && e.data && e.data.frontier).pop(); return v ? v.data.frontier.length : null; })();
  return html`<section class="collab" aria-label="How the agents work together">
    <div class="cgrid">
      <div class=${'cnode' + (nodes.analyst ? ' hotn' : '') + (act(analyst) ? ' working' : '')}><div class="cn-h">Analyst</div><div class="cn-d">Maps the parts</div><div class="cn-s"><b class="num"><${SubtleNum} text=${String(cnt('map'))} /></b> maps written</div></div>
      <${Arrow} hot=${hot.a2p} label="map" />
      <div class=${'cnode planners' + (nodes.planner ? ' hotn' : '') + (planners.some(act) ? ' working' : '')}><div class="cn-h">${plural(planners.length, 'planner')}</div><div class="cn-d">Propose programs</div><div class="cn-s"><b class="num"><${SubtleNum} text=${String(cnt('plan'))} /></b> plans, <b class="num"><${SubtleNum} text=${String(scored)} /></b> scored</div><div class="cn-dots">${planners.map((p) => html`<${Dot} key=${p.id} color=${agentDot(p.status, !running, false)} pulse=${act(p)} />`)}</div></div>
      <${Arrow} hot=${hot.p2j} label="programs" />
      <div class=${'cnode' + (nodes.judge ? ' hotn' : '') + (act(judge) ? ' working' : '')}><div class="cn-h">Judge</div><div class="cn-d">Scores and keeps the frontier</div><div class="cn-s"><b class="num"><${SubtleNum} text=${String(cnt('verdict'))} /></b> verdicts${frontier != null ? ', ' + frontier + ' on the frontier' : ''}</div></div>
    </div>
    <div class="cgrid tethers">
      <div class="tcell"><${Arrow} dir="v" rev hot=${hot.memw} /><${Arrow} dir="v" hot=${hot.memr} /></div><span></span>
      <div class="tcell"><${Arrow} dir="v" rev hot=${hot.memw} /><${Arrow} dir="v" hot=${hot.memr} /></div><span></span>
      <div class="tcell"><${Arrow} dir="v" rev hot=${hot.memw} /><${Arrow} dir="v" hot=${hot.memr} /></div>
    </div>
    <div class=${'cmem' + (hot.memw || hot.memr ? ' hotn' : '')}>
      <${PoweredBy} who="pastdev" label="Shared memory on" />
      <span class="cm-t">Everyone reads it before acting and writes what they learn.</span>
      <span class="cm-n"><b class="num"><${SubtleNum} text=${String(shared)} /></b> notes shared, <b class="num"><${SubtleNum} text=${String(lessons)} /></b> lessons saved</span>
    </div>
    <div class="cinfo faint"><${SponsorMark} who="openai" size=${13} /> OpenAI thinks for every agent. <${SponsorMark} who="modal" size=${13} /> Modal runs their sandboxes and the GPU jobs.</div>
  </section>`;
}

/* ---------- the board ---------- */
function ProgramBlock({ det, cat, vmap }) {
  const cur = det && det.current;
  if (!cur || !cur.program) return html`<div class="pblock faint">No program in flight.</div>`;
  return html`<div class="pblock"><div class="pb-n ell" title=${cur.name || cur.describe}>${cur.name || progName(cur.program, cat) || cur.describe}</div><${ProgramStrip} program=${cur.program} cat=${cat} vmap=${vmap} stop=${true} /></div>`;
}

function AgentCard({ a, det, snap, cat, vmap, i, ended, empty, nav }) {
  const role = roleOf(a);
  const active = !ended && !!ACTIVE[a.status];
  const think = det && det.thinking && det.thinking.length ? det.thinking[det.thinking.length - 1] : null;
  const sp = sparkFor(snap, a);
  const t = primaryTarget(snap.spec);
  const memN = (snap.activity || []).filter((e) => e.agent === a.id && ['plan', 'map', 'verdict', 'nudge_applied'].includes(kindOf(e))).length;
  const modal = active && (a.status === 'compiling' || a.status === 'scoring' || a.status === 'probing');
  const onClick = (e) => { if (e.target.closest('button, a')) return; nav(); };
  return html`<article class=${'acard reveal' + (active ? ' live' : '')} style=${{ '--i': i }} tabindex="0" role="link" aria-label=${'Open ' + a.name} onClick=${onClick} onKeyDown=${(e) => { if (e.key === 'Enter' && e.target === e.currentTarget) nav(); }}>
    <header class="ac-h"><${Avatar} agent=${a} size=${34} />
      <div class="ac-who"><a class="ac-name" href=${'#/run/' + snap.run_id + '/agents/' + a.id}>${a.name}</a><span class="ac-role">${ROLE[role] ? ROLE[role].label : cap(role)}</span></div>
      <span class="grow"></span>
      ${active ? html`<span class="livechip">${STATUS_WORD[a.status] || a.status}</span>` : html`<span class="astat"><${Dot} color=${agentDot(a.status, ended, empty)} />${ended && empty ? 'Ended' : STATUS_WORD[a.status] || a.status}</span>`}
    </header>
    <p class="ac-does">${a.blurb ? cap(a.blurb) + '. ' : ''}${ROLE[role] ? ROLE[role].does : ''}</p>
    <div class="ac-think"><${SponsorMark} who="openai" size=${13} title="Thinking by OpenAI" /><${ThinkLine} text=${a.now || STATUS_WORD[a.status] || ''} still=${!active} fluid /></div>
    ${think && html`<blockquote class="ac-quote" title=${think.text}>${think.text}</blockquote>`}
    ${role === 'planner' && html`<div class="ac-sec"><span class="ac-k">Trying now</span><${ProgramBlock} det=${det} cat=${cat} vmap=${vmap} /></div>`}
    <div class="ac-stats">
      <div><b class="num"><${SubtleNum} text=${String(a.attempts)} /></b><span>attempts</span></div>
      <div><b class="num">${a.best != null ? html`<${SubtleNum} text=${pctS(a.best)} />` : '-'}</b><span>best${t ? ' on ' + t.name : ''}</span></div>
      <div><b class=${'num' + (a.collapses ? ' down' : '')}><${SubtleNum} text=${String(a.collapses)} /></b><span>collapses</span></div>
      <div><b class="num"><${SubtleNum} text=${String(a.lessons || 0)} /></b><span>lessons</span></div>
    </div>
    <div class="ac-spark"><${Spark} pts=${sp} label=${a.name + ' scores over time'} /><span class="faint ac-sl">${a.id === 'judge' ? 'best so far' : role === 'planner' ? 'its scores' : ''}</span></div>
    <footer class="ac-f">${modal && html`<span class="powered sm"><${SponsorMark} who="modal" size=${12} /> Modal</span>`}${memN > 0 && html`<span class="powered sm"><${SponsorMark} who="pastdev" size=${12} /> ${memN} shared</span>`}<span class="grow"></span><a class="linkbtn" href=${'#/run/' + snap.run_id + '/agents/' + a.id}>Open mind view</a></footer>
  </article>`;
}

export function AgentsBoard({ snap, id, cat }) {
  const ended = snap.status === 'done' || snap.status === 'error';
  const empty = isEmptyEnd(snap);
  const live = !ended;
  const dets = useAgentDetails(id, snap.agents, live);
  const d = useMemo(() => deriveFindings(snap, cat), [snap, cat]);
  const order = [...snap.agents].sort((a, b) => ['analyst', 'planner', 'judge'].indexOf(roleOf(a)) - ['analyst', 'planner', 'judge'].indexOf(roleOf(b)));
  return html`<div class="rpage"><div class="col agwide">
    <div class="rsechead agtop"><h2 class="rh2">How the agents work together</h2><div class="agpow"><${PoweredBy} who="openai" label="Thinking with" /><${PoweredBy} who="modal" label="Sandboxes on" /></div></div>
    <${Collab} snap=${snap} running=${live} />
    <div class="rsechead agtop"><h2 class="rh2">Each agent</h2><span class="faint">Open a card to see its reasoning, attempts and what it learned.</span></div>
    <div class="acards">${order.map((a, i) => html`<${AgentCard} key=${a.id} a=${a} det=${dets[a.id]} snap=${snap} cat=${cat} vmap=${d.vmap} i=${i} ended=${ended} empty=${empty} nav=${() => { location.hash = '#/run/' + id + '/agents/' + a.id; }} />`)}</div>
  </div></div>`;
}

/* ---------- one agent: the mind view ---------- */
// thinking, tool calls and activity rows in time order, cut into attempts
function buildAttempts(det, snap, agentId, role) {
  const items = [];
  (det.thinking || []).forEach((x) => items.push({ t: 'think', ts: x.ts, text: x.text }));
  (det.tools || []).forEach((x) => items.push({ t: 'tool', ts: x.ts, call: x.call, result: x.result, status: x.status }));
  (snap.activity || []).forEach((e) => { if (e.agent === agentId) items.push({ t: 'act', ts: e.ts, e: { ...e, kind: kindOf(e), key: actKey(e) } }); });
  items.sort((a, b) => a.ts - b.ts);
  const groups = [];
  let cur = null;
  for (const it of items) {
    const fresh = !cur || (it.t === 'think' && (cur.done || (role !== 'planner' && cur.items.some((x) => x.t !== 'think'))));
    if (fresh) { cur = { items: [], done: false, key: 'g' + it.ts }; groups.push(cur); }
    cur.items.push(it);
    if (it.t === 'act' && ((it.e.kind === 'eval' && it.e.data) || it.e.kind === 'refused')) cur.done = true;
  }
  return groups;
}

function describeGroup(g, snap, cat, det) {
  const ev = g.items.find((x) => x.t === 'act' && x.e.kind === 'eval' && x.e.data);
  const ref = g.items.find((x) => x.t === 'act' && x.e.kind === 'refused');
  const probe = g.items.find((x) => x.t === 'act' && x.e.kind === 'probe');
  const data = ev ? ev.e.data : null;
  const row = data ? rowOf(snap, data.program_hash) : null;
  const badges = [];
  const lessons = [];
  let name = (data && data.name) || null;
  if (!name) { const pl = g.items.find((x) => x.t === 'act' && x.e.kind === 'probe' && x.e.data); name = pl && pl.e.data.name; }
  const score = data ? data.target : null;
  const dl = data ? deltaOf(snap, data.program_hash, score) : null;
  const comps = (row && row.components) || [];
  if (data) {
    if (data.is_best) { badges.push(['g', 'New best']); lessons.push('Kept as the new best' + (dl ? ', ' + pts(dl.d) + ' points over the ' + (dl.vs === 'host' ? 'host' : 'previous best') : '') + '.'); }
    if (data.result === 'collapsed') {
      badges.push(['r', 'Collapsed']);
      const failed = (data.guards || []).filter((x) => !x.ok).map((x) => x.name);
      lessons.push('It collapsed' + (failed.length ? ': ' + failed.join(', ') + ' fell below the floor.' : '.'));
    } else if (data.result && data.result !== 'ok') { badges.push(['w', RESULT_LABEL[data.result] || data.result]); lessons.push((RESULT_LABEL[data.result] || data.result) + ', so nothing was scored.'); }
    else if (!data.is_best && dl) lessons.push('Scored ' + pctS(score) + ', ' + pts(dl.d) + ' points ' + (dl.d >= 0 ? 'above' : 'below') + ' the best at the time.');
    if (data.result === 'ok' && comps.length) {
      const bad = comps.filter((c) => c.verdict !== 'contributes');
      if (bad.length === comps.length) { badges.push([comps.some((c) => c.verdict === 'harmful') ? 'r' : 'w', comps.some((c) => c.verdict === 'harmful') ? 'Harmful part' : 'No better than random']); }
      bad.forEach((c) => lessons.push(c.label + ' was ' + c.verdict + (c.detail && c.detail.contribution != null ? ' (' + pts(c.detail.contribution) + ' points).' : '.')));
    }
  }
  if (ref) { badges.push(['w', 'Refused over a limit']); lessons.push(ref.e.text); }
  const fits = probe && probe.e.data && probe.e.data.fits ? probe.e.data.fits : [];
  const fitTool = g.items.find((x) => x.t === 'tool' && /^probe/.test(x.call || ''));
  const fit = fits.length ? fits[0] : fitTool ? (/fit\s+([\d.]+)/.exec(fitTool.result || '') || [])[1] : null;
  if (!g.done && g.items.length) badges.push(['b', 'In progress']);
  const grafted = row && row.program ? row.program.grafts || [] : [];
  return { data, row, badges, lessons, name, score, dl, comps, fit, grafted, ref };
}

function stageOf(it) {
  if (it.t === 'think') return ['Thinking', 'openai'];
  if (it.t === 'tool') {
    const c = it.call || '';
    if (/^probe/.test(c)) return ['Probe', null];
    if (/^(compile|heal)/.test(c)) return ['Compile', 'modal'];
    if (/^eval|^control/.test(c)) return ['Score', 'modal'];
    return ['Step', null];
  }
  const k = it.e.kind;
  return { eval: ['Result', null], probe: ['Probe', null], plan: ['Plan', 'pastdev'], map: ['Map', 'pastdev'], verdict: ['Verdict', 'pastdev'], nudge_applied: ['Nudge', 'pastdev'], refused: ['Refused', null], handoff: ['Handoff', 'pastdev'] }[k] || [cap(k), null];
}

function Step({ it, fresh, ctx }) {
  const [st, sp] = stageOf(it);
  const body = it.t === 'think' ? html`<${StreamText} class="tl-x" text=${it.text} play=${fresh} />`
    : it.t === 'tool' ? html`<span class="tl-x mono">${plainCall(it.call)}</span>`
    : html`<span class=${'tl-x' + (it.e.status === 'bad' ? ' down' : '')}>${it.e.text}</span>`;
  const right = it.t === 'tool' ? html`<span class=${'tl-r ' + (it.status === 'ok' ? 'up' : it.status === 'bad' ? 'down' : 'dim3')}>${it.result}</span>`
    : it.t === 'act' && it.e.kind === 'eval' && it.e.data && it.e.data.target != null ? html`<span class="tl-r num">${pctS(it.e.data.target)}</span>` : html`<span></span>`;
  return html`<div class=${'tl-step ' + it.t}>
    <span class="tl-s">${sp && html`<${SponsorMark} who=${sp} size=${11} />`}${st}</span>${body}${right}<span class="tl-t num faint">${clock(it.ts)}</span>
  </div>`;
}

function AttemptCard({ g, n, snap, cat, det, vmap, seenTs, live }) {
  const x = describeGroup(g, snap, cat, det);
  const title = x.name || (g.items.find((i) => i.t === 'think') ? 'Planning the next program' : 'Working');
  const prog = x.row && x.row.program;
  return html`<article class=${'attempt reveal' + (!g.done ? ' open' : '')} style=${{ '--i': 0 }}>
    <header class="at-h">
      <span class="at-n faint num">Attempt ${n}</span><span class="at-t ell" title=${title}>${title}</span>
      <span class="grow"></span>
      ${x.badges.map(([k, l]) => html`<span class=${'tag ' + k} key=${l}>${l}</span>`)}
      ${x.score != null && html`<span class="at-sc num">${pctS(x.score)}${x.dl && html` <span class=${x.dl.d >= 0 ? 'up' : 'down'}>${pts(x.dl.d)}</span>`}</span>`}
    </header>
    <div class="tl-steps">${g.items.map((it, i) => html`<${Step} key=${it.t + it.ts + i} it=${it} fresh=${!seenTs.has(it.ts)} ctx=${{}} />`)}</div>
    ${(x.fit != null || x.comps.length > 0 || x.grafted.length > 0) && html`<div class="at-ev"><span class="faint">Evidence</span>
      ${x.fit != null && html`<span class="fchip">Fit <b class="num">${x.fit}</b></span>`}
      ${x.score != null && html`<span class="fchip">Score <b class="num">${pctS(x.score)}</b></span>`}
      ${x.comps.length ? x.comps.map((c) => html`<${PartChip} key=${c.id} id=${c.id} label=${c.label} at=${c.host_layer} donor=${c.donor} verdict=${c.verdict} />`) : html`<${ProgramStrip} program=${prog} cat=${cat} vmap=${vmap} />`}
      ${prog && html`<button class="linkbtn" onClick=${() => openProgramInBuilder(snap.spec.id, prog)}>Open in builder</button>`}
    </div>`}
    ${x.lessons.length > 0 && html`<div class="at-lesson"><span class="tl-s">Lesson</span><span>${x.lessons.join(' ')}</span></div>`}
  </article>`;
}

function LearnCard({ f, snap }) {
  return html`<article class=${'lcard t-' + f.tone}>
    <div class="lc-top"><span class=${'tag ' + ({ good: 'g', warn: 'w', bad: 'r', info: 'b' }[f.tone] || '')}>${{ works: 'Works', dead: 'Dead end', fact: f.map ? 'Analyst map' : 'Fact', open: 'Open question', thought: 'From its thinking' }[f.type]}</span>${f.ts && html`<span class="faint">${ago(f.ts * 1000)}</span>`}</div>
    <div class="lc-t">${f.title}</div>${f.body && html`<div class="lc-b">${f.body}</div>`}
    <div class="lc-f">${f.part && html`<button class="linkbtn" onClick=${() => openPart(f.part)}>Part window</button>`}<a class="linkbtn" href=${'#/run/' + snap.run_id + '/findings'}>All findings</a></div>
  </article>`;
}

const LEARN_RE = /lesson|learn|worked|dead end|no better|collapse|avoid|did not|does not|better than|beat|noted/i;
function thoughtLessons(thinking) {
  const out = [];
  for (const x of thinking.slice().reverse()) {
    const s = String(x.text).split(/(?<=[.!?])\s+/).find((q) => LEARN_RE.test(q));
    if (s && !out.some((o) => o.title === s)) out.push({ id: 'th' + x.ts, type: 'thought', tone: 'info', title: s.length > 150 ? s.slice(0, 147) + '...' : s, body: null, ts: x.ts });
    if (out.length >= 4) break;
  }
  return out;
}

const NEXT_RE = /\b(next|will (try|move|add|use|test|raise|lower)|plan:|going to|i'll|should (try|move|add))\b/i;
function nextFrom(thinking, nudges) {
  const out = [];
  const pend = (nudges || []).filter((n) => !n.applied);
  pend.forEach((n) => out.push('Read your nudge: ' + n.text));
  const last = thinking.length ? thinking[thinking.length - 1].text : '';
  String(last).split(/(?<=[.!?])\s+/).filter((s) => NEXT_RE.test(s)).slice(-2).forEach((s) => out.push(s));
  return out;
}

function suggestNudges(snap, d, a, det, cat) {
  const out = [];
  const role = roleOf(a);
  const t = primaryTarget(snap.spec);
  if (role === 'planner') {
    const hist = (det && det.history) || [];
    const own = hist.filter((h) => h.status === 'ok' && t && h.scores[t.id] != null).sort((x, y) => y.scores[t.id] - x.scores[t.id])[0] || (snap.best && snap.best.agent === a.id ? snap.best : null);
    const row = own && (rowOf(snap, own.program_hash) || own);
    const prog = row && row.program;
    if (prog) {
      const nm = nameOf(own, cat);
      const hs = prog.heal ? prog.heal.steps : 0;
      out.push(hs ? 'Try heal ' + (hs >= 900 ? hs + 300 : hs * 2) + ' on ' + nm : 'Add a heal of 300 steps to ' + nm);
      const g = (prog.grafts || [])[0];
      if (g) {
        const s = d.parts[g.id];
        const lab = s ? s.label : partLabel(cat, g.id);
        const to = s && s.bestLayer != null && s.bestLayer !== g.at ? s.bestLayer : g.at > 1 ? g.at - 1 : g.at + 1;
        out.push(s && s.bestLayer === to ? 'Move ' + lab + ' to L' + to + ', its best layer so far' : 'Move ' + lab + ' up a layer, to L' + to);
      }
    }
    const badDonor = Object.values(d.donors).filter((x) => x.collapsed >= 2 || x.noise >= 2).sort((x, y) => y.collapsed + y.noise - (x.collapsed + x.noise))[0];
    if (badDonor) out.push('Stop trying ' + shortDonor(badDonor.donor) + ': its programs collapsed or tied random ' + (badDonor.collapsed + badDonor.noise) + ' times');
    const fresh = d.items.find((f) => f.id.startsWith('untried:'));
    if (fresh) out.push('Try ' + fresh.title.replace(' has not been tried', '') + ', no program has used it yet');
  } else if (role === 'judge') {
    out.push('Keep only programs that beat random parts by 3 points or more');
    out.push('Re-score the top 3 on the held-out items now');
  } else {
    const s = Object.values(d.parts).find((x) => x.bestLayer != null);
    if (s) out.push('Map the layers next to ' + s.label + ' at L' + s.bestLayer);
    out.push('Look at the donor with the fewest scored programs next');
  }
  return out.slice(0, 4);
}

export function AgentPage({ snap, id, agentId, cat }) {
  const [det, setDet] = useState(null);
  const [err, setErr] = useState(null);
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const [more, setMore] = useState(false);
  const clearRef = useRef();
  const seen = useRef(null);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);

  const load = () => api.getAgent(id, agentId).then((x) => { if (alive.current) { setDet(x); setErr(null); } }).catch((e) => alive.current && setErr(e));
  useEffect(() => { load(); }, [id, agentId]);
  const ended = snap.status === 'done' || snap.status === 'error';
  useInterval(load, ended ? null : 1500);

  const live = snap.agents.find((x) => x.id === agentId) || { id: agentId, name: agentId, status: 'idle', attempts: 0, collapses: 0 };
  const a = det ? { ...live, ...det.agent } : live;
  const role = roleOf(a);
  const active = !ended && !!ACTIVE[live.status];
  const t = primaryTarget(snap.spec);
  const d = useMemo(() => deriveFindings(snap, cat), [snap, cat]);
  const thinking = det ? det.thinking : [];
  const tools = det ? det.tools : [];
  if (det && !seen.current) seen.current = new Set([...thinking.map((x) => x.ts), ...tools.map((x) => x.ts)]);
  const seenSet = seen.current || new Set();
  useEffect(() => { if (det && seen.current) { thinking.forEach((x) => seen.current.add(x.ts)); tools.forEach((x) => seen.current.add(x.ts)); } });

  const groups = useMemo(() => (det ? buildAttempts(det, snap, agentId, role).reverse() : []), [det, snap.activity, agentId]);
  const total = groups.length;
  const shownG = more ? groups : groups.slice(0, 4);
  const cur = det && det.current;
  const lastThought = thinking.length ? thinking[thinking.length - 1] : null;
  const nexts = det ? nextFrom(thinking, det.nudges) : [];
  const lastTool = tools.length ? tools[tools.length - 1] : null;
  const pulse = lastTool && cur && cur.program ? { kind: /^heal/.test(lastTool.call) ? 'heal' : /^eval/.test(lastTool.call) ? 'score' : 'graft', at: ((cur.program.grafts || [])[0] || {}).at, ok: lastTool.status !== 'bad', key: lastTool.ts + lastTool.call } : null;
  const brainParts = cur && cur.program ? (cur.program.grafts || []).map((g) => { const p = cat && cat.parts.find((x) => x.id === g.id); return { id: g.id, label: p ? p.label : partLabel(cat, g.id), kind: p && p.kind, at: g.at, donor: p && p.donor, verdict: d.vmap[g.id] }; }) : [];

  // learned: this agent's findings, the analyst's maps, and lines from its thinking
  const mine = d.items.filter((f) => (f.agents || []).includes(agentId) && !f.map).sort((x, y) => (y.ts || 0) - (x.ts || 0));
  const maps = role === 'planner' ? d.items.filter((f) => f.map).sort((x, y) => (y.ts || 0) - (x.ts || 0)).slice(0, 2) : [];
  const own = role === 'analyst' ? d.items.filter((f) => f.map).sort((x, y) => (y.ts || 0) - (x.ts || 0)).slice(0, 6) : [];
  const th = thoughtLessons(thinking);
  const learned = [...mine.slice(0, 5), ...maps, ...own, ...th];

  // attempts table, newest first
  const hist = det ? det.history.map((h, i) => ({ ...h, n: i + 1 })).reverse() : [];

  const sugg = suggestNudges(snap, d, a, det, cat);
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
  const bits = [plural(live.attempts, 'attempt'), plural(live.collapses, 'collapse')];
  const modalCalls = tools.filter((x) => ONMODAL.test(x.call || '')).length;
  const memRows = (snap.activity || []).filter((e) => e.agent === agentId && ['plan', 'map', 'verdict', 'nudge_applied', 'handoff'].includes(kindOf(e))).length;

  return html`<div class="rpage agentp mind">
    <div class="ascroll"><div class="col agwide">
      <a class="back" href=${'#/run/' + id + '/agents'}>Agents</a>
      <div class="mind-hd">
        <${Avatar} agent=${a} size=${44} />
        <div class="mh-t"><div class="ahd"><h1 class="page-title">${a.name}</h1><span class="tag">${ROLE[role] ? ROLE[role].label : cap(role)}</span>
          ${active ? html`<span class="livechip">${STATUS_WORD[live.status] || live.status}</span>` : html`<span class="astat"><${Dot} color=${agentDot(live.status, ended, isEmptyEnd(snap))} />${ended && isEmptyEnd(snap) ? 'Ended' : STATUS_WORD[live.status] || live.status}</span>`}</div>
          <div class="ameta">${ROLE[role] ? ROLE[role].does : ''}</div></div>
        <div class="mh-stats"><div><b class="num"><${SubtleNum} text=${String(live.attempts)} /></b><span>attempts</span></div><div><b class="num">${live.best != null ? html`<${SubtleNum} text=${pctS(live.best)} />` : '-'}</b><span>best</span></div><div><b class=${'num' + (live.collapses ? ' down' : '')}>${live.collapses}</b><span>collapses</span></div><div><b class="num">${live.lessons || 0}</b><span>lessons</span></div></div>
      </div>
      ${err && !det && html`<div class="down" style="margin-top:24px">${err.message}</div>`}
      ${!det && !err && html`<div class="skeleton" style="height:180px;margin-top:32px"></div>`}
      ${det && html`
      <section class="rsec">
        <div class="rsechead"><h2 class="rh2">Now</h2><${PoweredBy} who="openai" label="Thinking with" /></div>
        <div class=${'nowcard' + (active ? ' working' : '')}>
          <div class="now-l">
            <div class="now-line"><${ThinkLine} text=${live.now || STATUS_WORD[live.status] || ''} still=${!active} fluid /></div>
            ${role === 'planner' && html`<${StepTrail} status=${live.status} ended=${ended} />`}
            ${lastThought ? html`<div class="now-why"><span class="k">Why</span><${StreamText} class="now-txt" text=${lastThought.text} play=${!seenSet.has(lastThought.ts)} /><span class="faint now-ts">${clock(lastThought.ts)}</span></div>` : html`<div class="faint">No thoughts yet.</div>`}
            ${nexts.length > 0 && html`<div class="now-next"><span class="k">Plans next</span><ul>${nexts.map((s, i) => html`<li key=${i}>${s}</li>`)}</ul></div>`}
          </div>
          ${role === 'planner' && html`<div class="now-r">
            ${cur && cur.program ? html`<div class="stage3d mind3d"><${Brain3D} layers=${(cat && cat.host && cat.host.layers) || 16} parts=${brainParts} cuts=${[]} pulse=${pulse} height=${200} /></div>
              <div class="pb-n ell" title=${cur.name || cur.describe}>${cur.name || progName(cur.program, cat) || cur.describe}</div>
              <${ProgramStrip} program=${cur.program} cat=${cat} vmap=${d.vmap} />
              <div class="bact"><button class="linkbtn" onClick=${() => openProgramInBuilder(snap.spec.id, cur.program)}>Open in builder</button></div>` : html`<div class="faint">No program in flight right now.</div>`}
          </div>`}
        </div>
      </section>

      <section class="rsec">
        <div class="rsechead"><h2 class="rh2">Reasoning, attempt by attempt</h2><span class="faint">${plural(total, 'attempt')}, newest first</span></div>
        ${total ? html`<div class="attempts">${shownG.map((g, i) => html`<${AttemptCard} key=${g.key} g=${g} n=${total - i} snap=${snap} cat=${cat} det=${det} vmap=${d.vmap} seenTs=${seenSet} live=${active} />`)}</div>
          ${total > 4 && html`<button class="linkbtn" onClick=${() => setMore(!more)}>${more ? 'Show fewer' : 'Show ' + (total - 4) + ' earlier'}</button>`}` : html`<div class="faint">Nothing yet.</div>`}
      </section>

      <section class="rsec">
        <div class="rsechead"><h2 class="rh2">What it learned</h2><a class="linkbtn" href=${'#/run/' + id + '/findings'}>Everything the swarm found</a></div>
        ${learned.length ? html`<div class="lgrid">${learned.map((f) => html`<${LearnCard} key=${f.id} f=${f} snap=${snap} />`)}</div>` : html`<div class="faint">Nothing learned yet. Findings appear when this agent's programs are scored.</div>`}
      </section>

      <section class="rsec">
        <h2 class="rh2">Its attempts</h2>
        ${hist.length ? html`<div class="ptwrap"><table class="ptable atab"><thead><tr><th>#</th><th>Program</th><th class="r">Score</th><th class="r">Change</th><th>Verdict</th></tr></thead><tbody>
          ${hist.slice(0, 20).map((h) => {
            const row = rowOf(snap, h.program_hash);
            const prog = (row && row.program) || (cur && cur.program && cur.program_hash === h.program_hash ? cur.program : null);
            const v = t ? h.scores[t.id] : null;
            const dl = deltaOf(snap, h.program_hash, v);
            const comps = (row && row.components) || [];
            const donors = [...new Set(((prog && prog.grafts) || []).map((g) => { const p = cat && cat.parts.find((x) => x.id === g.id); return p && p.donor; }).filter(Boolean))];
            let verdict = html`<span class="faint">-</span>`;
            if (h.status === 'collapsed') verdict = html`<span class="tag r">Collapsed</span>`;
            else if (h.status !== 'ok') verdict = html`<span class="tag w">${RESULT_LABEL[h.status] || h.status.replace(/_/g, ' ')}</span>`;
            else if (comps.length) verdict = comps.some((c) => c.verdict === 'contributes') ? html`<span class="tag g">Contributes</span>` : html`<span class=${'tag ' + (comps.some((c) => c.verdict === 'harmful') ? 'r' : 'w')}>${comps.some((c) => c.verdict === 'harmful') ? 'Harmful' : 'No better than random'}</span>`;
            return html`<tr key=${h.program_hash + h.n} class=${'ptr' + (prog ? '' : ' nolink')} onClick=${() => prog && openProgramInBuilder(snap.spec.id, prog)} title=${prog ? 'Open this program in the builder' : ''}>
              <td class="num faint">${h.n}</td>
              <td><div class="at-prog"><span class="mlogos">${donors.slice(0, 3).map((m) => html`<${ModelLogo} key=${m} model=${m} size=${16} />`)}</span><span class="ell">${nameOf(h, cat)}</span></div>
                ${comps.length > 0 && html`<div class="at-chips">${comps.map((c) => html`<${PartChip} key=${c.id} id=${c.id} label=${c.label} at=${c.host_layer} donor=${c.donor} verdict=${c.verdict} stop=${true} />`)}</div>`}</td>
              <td class="r num">${h.status === 'ok' && v != null ? pctS(v) : '-'}</td>
              <td class=${'r num ' + (dl ? (dl.d >= 0 ? 'up' : 'down') : 'faint')}>${dl ? pts(dl.d) : '-'}</td>
              <td>${verdict}</td></tr>`;
          })}</tbody></table></div>` : html`<div class="faint">No scored programs yet.</div>`}
      </section>

      <section class="rsec"><h2 class="rh2">What runs this agent</h2>
        <div class="sp3">
          <div class="spcard"><${PoweredBy} who="openai" label="Thinks with" /><span class="faint">${thinking.length} thoughts so far. Every line above came from the model.</span></div>
          <div class="spcard"><${PoweredBy} who="modal" label="Runs on" /><span class="faint">${modalCalls} compile, heal and score steps ran in Modal sandboxes and on GPUs.</span></div>
          <div class="spcard"><${PoweredBy} who="pastdev" label="Remembers with" /><span class="faint">${memRows} notes written to or read from the shared memory.</span></div>
        </div>
      </section>`}
    </div></div>
    <div class="adock nudgedock"><div class="col agwide">
      <div class="nd-h"><span class="nd-t">Nudge ${a.name}</span><span class="faint">${live.nudges_pending ? plural(live.nudges_pending, 'nudge') + ' waiting to be read' : 'It reads your note before its next attempt.'}</span><span class="grow"></span><${PoweredBy} who="pastdev" label="Shared through" /></div>
      ${sugg.length > 0 && html`<div class="sugs" role="group" aria-label="Suggested nudges">${sugg.map((s) => html`<button class="sug" key=${s} onClick=${() => setText(s)}>${s}</button>`)}</div>`}
      <${Composer} value=${text} onInput=${setText} onSubmit=${send} disabled=${sending} minRows=${1} clearRef=${clearRef} label=${'Nudge ' + a.name} placeholder=${'Tell ' + a.name + ' what to try next'} />
      ${det && det.nudges.length > 0 && html`<div class="nd-list">${det.nudges.slice(-3).reverse().map((n, i) => html`<span class="nd-i" key=${i}><${VDot} v=${n.applied ? 'contributes' : null} /> ${n.text} <span class="faint">${n.applied ? 'read' : 'waiting'}</span></span>`)}</div>`}
    </div></div>
  </div>`;
}
