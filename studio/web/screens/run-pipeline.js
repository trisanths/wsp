// The autoresearch pipeline of one run: a 3D hero of the best program, the seven stages, the agent lanes,
// a live event ticker, the techniques the agents use and the verdict against the references.
import { html, useState, useEffect, useRef, useMemo } from '../lib.js';
import { dur, pct, pctS, primaryTarget } from '../util.js';
import { Dot, Avatar, Tag } from '../components.js';
import { Reel, PopNum, SubtleNum, ThinkLine } from '../motion.js';
import { Brain3D } from '../brain3d.js';
import { ModelLogo, PoweredBy, SponsorMark } from '../logos.js';
import { STAGES, TECHNIQUES } from '../techniques.js';
import { Logos, donorsOf, verdictOf } from '../cmp.js';
import { openPart } from './part-detail.js';
import { ACTIVE, STATUS_WORD, agentDot, isEmptyEnd, nameOf, bestTarget } from './run-shared.js';

const SHARED = new Set(['plan', 'map', 'verdict', 'handoff', 'nudge_applied']);
const ALIAS = { llm_plan: 'plan', analyst_map: 'map', judge_verdict: 'verdict', nudge_sent: 'nudge' };
const kindOf = (e) => ALIAS[e.kind] || e.kind;
const SPONSOR_FOR = { eval: 'modal', probe: 'modal', heldout: 'modal', plan: 'openai', map: 'openai', verdict: 'pastdev', handoff: 'pastdev', nudge: 'pastdev', nudge_applied: 'pastdev' };
const SP_LABEL = { openai: 'Drafted by', exa: 'Search by', modal: 'GPUs on', pastdev: 'Shared memory on' };
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const hashOf = (s) => { let h = 0; for (const c of String(s || '')) h = (h * 31 + c.charCodeAt(0)) >>> 0; return h; };

/* ---------- the program of the best build, as Brain3D wants it ---------- */
function maskIds(m, out = []) {
  if (!m) return out;
  if (m.op === 'use') out.push(m.id);
  else if (m.op === 'restrict') maskIds(m.inner, out);
  else { maskIds(m.a, out); maskIds(m.b, out); }
  return out;
}
function brainOf(snap, cat, entry) {
  const prog = (entry && entry.program) || (snap.best && snap.best.program) || null;
  const layers = (cat && cat.host && cat.host.layers) || 16;
  const comps = (entry && entry.components) || [];
  const parts = prog
    ? (prog.grafts || []).map((g) => {
      const c = comps.find((x) => x.id === g.id);
      const p = cat && cat.parts.find((x) => x.id === g.id);
      return { id: g.id, label: c ? c.label : p ? p.label : 'Part', kind: (c && c.kind) || (p && p.kind) || 'block', at: Math.min(layers - 1, Math.max(0, g.at)), donor: (c && c.donor) || (p && p.donor) || '', verdict: c ? c.verdict : null };
    })
    : [];
  const ids = prog ? maskIds(prog.masks) : [];
  const cuts = ids.map((id, i) => {
    const m = cat && (cat.masks || []).find((x) => x.id === id);
    return { layer: Math.round(((i + 1) / (ids.length + 1)) * (layers - 1)), frac: 0.22, label: m ? m.label : 'Skill removed' };
  });
  return { layers, parts, cuts, heals: !!(prog && prog.heal) };
}
function layerFrom(text, layers, seed) {
  const m = /\bat L(\d+)/.exec(text || '');
  if (m) return Math.min(layers - 1, +m[1]);
  return hashOf(seed) % layers;
}
// The animation steps one event plays, in order, each with a caption.
function stepsFor(e, snap, cat) {
  const k = kindOf(e);
  const d = e.data || {};
  const layers = (cat && cat.host && cat.host.layers) || 16;
  if (k === 'probe') {
    return [{ kind: 'probe', at: layerFrom(d.name || e.text, layers, d.program_hash || e.text), ok: null, cap: 'Checking how well a part fits: ' + (d.name || 'a candidate') }];
  }
  if (k === 'eval') {
    const entry = snap.leaderboard.find((r) => r.program_hash === d.program_hash);
    const prog = entry && entry.program;
    const out = [];
    const nm = d.name || (entry && nameOf(entry, cat)) || 'a program';
    if (prog && prog.grafts && prog.grafts.length) {
      const g = prog.grafts[0];
      const c = (entry.components || []).find((x) => x.id === g.id);
      const p = cat && cat.parts.find((x) => x.id === g.id);
      out.push({ kind: 'graft', at: g.at, partId: g.id, ok: null, cap: 'Grafting ' + ((c && c.label) || (p && p.label) || 'a part') + ' into layer ' + g.at });
    }
    if (prog && maskIds(prog.masks).length) out.push({ kind: 'cut', at: Math.floor(layers / 2), ok: null, cap: 'Cutting out a skill with a deletion mask' });
    if (prog && prog.heal) out.push({ kind: 'heal', at: Math.floor(layers / 2), ok: null, cap: 'Healing the glue for ' + prog.heal.steps + ' steps' });
    const ok = d.result === 'ok' || d.result == null ? d.result !== 'collapsed' && e.status !== 'bad' : false;
    out.push({ kind: 'score', at: layers - 1, ok, partId: null, cap: (ok ? 'Scored ' + nm : 'Failed: ' + nm) + (d.target != null ? ', ' + pctS(d.target) : '') + (d.is_best ? ', a new best' : '') });
    return out;
  }
  if (k === 'plan') return [{ kind: 'scan', at: hashOf(e.text) % layers, ok: null, cap: 'A planner is writing programs' }];
  if (k === 'map') return [{ kind: 'scan', at: hashOf(e.text) % layers, ok: null, cap: 'The analyst is mapping where skills live' }];
  if (k === 'verdict') return [{ kind: 'score', at: layers - 1, ok: null, partId: null, cap: 'The judge is ranking the frontier' }];
  return [];
}
function chainFor(snap, cat, entry) {
  const prog = entry && entry.program;
  if (!prog) return [];
  const layers = (cat && cat.host && cat.host.layers) || 16;
  const e = { kind: 'eval', status: 'ok', data: { program_hash: entry.program_hash, name: nameOf(entry, cat), target: entry.target_score, result: 'ok' }, text: '' };
  const out = stepsFor(e, snap, cat);
  return out.length ? out : [{ kind: 'scan', at: layers >> 1, ok: null, cap: 'Reading the best program' }];
}

/* ---------- numbers ---------- */
function statsOf(snap, cat) {
  const act = snap.activity.map((e) => ({ ...e, kind: kindOf(e) }));
  const lb = snap.leaderboard;
  const partIds = new Set();
  lb.forEach((r) => ((r.program && r.program.grafts) || []).forEach((g) => partIds.add(g.id)));
  const count = (k) => act.filter((e) => e.kind === k).length;
  const comps = lb.reduce((n, r) => n + ((r.components || []).length), 0);
  const bs = (snap.spec && snap.spec.benchmarks) || [];
  const sourced = bs.reduce((n, b) => n + (b.reference_scores || []).filter((x) => x.sourced).length, 0);
  const donors = (snap.sources || []).filter((s) => s.role === 'donor').map((s) => s.model);
  return {
    act,
    evals: snap.budget.evals_used, total: snap.budget.evals_total,
    programs: lb.length, parts: partIds.size,
    probes: count('probe'), plans: count('plan'), maps: count('map'), verdicts: count('verdict'), held: count('heldout'),
    shared: act.filter((e) => SHARED.has(e.kind)).length,
    withGraft: lb.filter((r) => r.program && (r.program.grafts || []).length).length,
    withMask: lb.filter((r) => r.program && maskIds(r.program.masks).length).length,
    withHeal: lb.filter((r) => r.program && r.program.heal).length,
    controls: lb.filter((r) => r.vs_control != null).length,
    drifts: lb.filter((r) => r.drift).length,
    comps, benchmarks: bs.length, sourced, donors, libParts: cat ? cat.parts.length : null,
  };
}

/* ---------- stages ---------- */
function activeStages(snap) {
  if (snap.status === 'done') return { done: 7, on: new Set([6]) };
  if (snap.status === 'error') return { done: 7, on: new Set() };
  const on = new Set();
  if (snap.status === 'starting' || !snap.agents.length) { on.add(1); return { done: 1, on }; }
  for (const a of snap.agents) {
    if (a.status === 'thinking') on.add(a.role === 'analyst' ? 2 : 3);
    else if (a.status === 'probing' || a.status === 'compiling') on.add(4);
    else if (a.status === 'scoring') on.add(5);
  }
  if (!on.size) on.add(3);
  return { done: Math.min(...on), on };
}
function stageStats(id, s, snap, bt) {
  const best = snap.best;
  switch (id) {
    case 'request': return [[snap.spec && snap.spec.size ? snap.spec.size.max_params_b + 'B' : '-', 'size limit'], [s.benchmarks, 'targets']];
    case 'research': return [[s.benchmarks, 'benchmarks'], [s.sourced, 'sourced scores']];
    case 'mine': return [[s.libParts == null ? '-' : s.libParts, 'library parts'], [s.donors.length, 'donors in play']];
    case 'swarm': return [[snap.agents.length, 'agents'], [s.shared, 'shared notes']];
    case 'build': return [[s.withGraft, 'grafted'], [s.withHeal, 'healed']];
    case 'test': return [[s.evals, 'scores'], [s.controls, 'controls']];
    default: return [[best && bt != null ? pctS(bt) : '-', 'best score'], [best ? 'ready' : 'none yet', 'build']];
  }
}

function StageRail({ snap, s, bt, sel, setSel }) {
  const st = activeStages(snap);
  return html`<div class="pl-railwrap"><div class="pl-rail" role="list">${STAGES.map((g, i) => {
    const on = st.on.has(i);
    const done = i < st.done || (snap.status === 'done');
    const stats = stageStats(g.id, s, snap, bt);
    return html`<div class=${'pl-st' + (on ? ' on' : '') + (done && !on ? ' done' : '') + (sel === i ? ' sel' : '')} key=${g.id} role="listitem">
      <div class="pl-node">
        <span class="pl-dot">${done && !on ? html`<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg>` : i + 1}</span>
        ${i < STAGES.length - 1 && html`<svg class="pl-seg" width="100%" height="6" preserveAspectRatio="none"><line class=${done || on ? 'flowline' : ''} x1="0" y1="3" x2="100%" y2="3"/></svg>`}
      </div>
      <button class=${'pl-card reveal' + (on ? ' glow-on' : '')} style=${{ '--i': i }} onClick=${() => setSel(i)} aria-pressed=${sel === i}>
        <span class="pl-cn">${g.name}</span>
        <span class="pl-sp">${g.sponsor ? html`<${SponsorMark} who=${g.sponsor} size=${14} />` : html`<span class="pl-spx"></span>`}${on && html`<span class="livechip"></span>`}</span>
        <span class="pl-stats">${stats.map(([v, l]) => html`<span class="pl-stat"><b class="num"><${SubtleNum} text=${String(v)} /></b><span>${l}</span></span>`)}</span>
      </button>
    </div>`;
  })}</div></div>`;
}

function StageDetail({ i, s, snap, cat }) {
  const g = STAGES[i];
  return html`<div class="pl-detail card" key=${g.id}>
    <div class="pl-dh"><span class="pl-dn">${g.name}</span>${g.sponsor && html`<${PoweredBy} who=${g.sponsor} label=${SP_LABEL[g.sponsor]} />`}</div>
    <p class="pl-dl">${g.line}</p>
    ${g.id === 'research' && html`<div class="pl-dx">${(snap.spec.benchmarks || []).slice(0, 4).map((b) => html`<${Tag} kind=${b.research ? 'b' : ''} key=${b.id}>${b.name}${b.research ? ' · found by search' : ''}<//>`)}</div>`}
    ${g.id === 'mine' && html`<div class="pl-dx">${s.donors.length ? s.donors.map((m) => html`<span class="pl-donor" key=${m}><${ModelLogo} model=${m} size=${18} /><span>${m.split('/').pop()}</span></span>`) : html`<span class="faint">Donors appear as agents use them.</span>`}</div>`}
    ${g.id === 'swarm' && html`<div class="pl-dx">${snap.agents.map((a) => html`<span class="pl-donor" key=${a.id}><${Dot} color=${agentDot(a.status, snap.status === 'done', false)} pulse=${!!ACTIVE[a.status]} /><span>${a.name}</span></span>`)}</div>`}
    ${g.id === 'build' && html`<div class="pl-dx faint">${s.probes} fit checks, ${s.withGraft} programs with grafts, ${s.withMask} with deletion masks, ${s.withHeal} healed.</div>`}
    ${g.id === 'test' && html`<div class="pl-dx faint">${s.evals} scored, ${s.controls} with a random-part control, ${s.comps} parts judged.</div>`}
    ${g.id === 'request' && html`<div class="pl-dx faint">${snap.spec.title || ''}</div>`}
  </div>`;
}

/* ---------- agent lanes ---------- */
function Lane({ a, snap, cat, i }) {
  const ended = snap.status === 'done' || snap.status === 'error';
  const act = !ended && !!ACTIVE[a.status];
  const chips = snap.curve.filter((c) => c.agent === a.id).slice(-4);
  return html`<a class=${'pl-lane card reveal' + (act ? ' working' : '')} style=${{ '--i': i }} href=${'#/run/' + snap.run_id + '/agents/' + a.id}>
    <div class="pl-lh"><${Avatar} agent=${a} size=${28} /><div class="pl-lt"><span class="pl-ln">${a.name}</span><span class="faint pl-lr">${a.blurb || a.role}</span></div>
      <span class="pl-lb num">${a.best != null ? html`<${SubtleNum} text=${pctS(a.best)} />` : '-'}</span></div>
    <div class="pl-now"><${SponsorMark} who="openai" size=${13} title="Thinking by OpenAI" /><${ThinkLine} text=${a.now || STATUS_WORD[a.status] || a.status} still=${!act} fluid /></div>
    <div class="pl-chips">${chips.length ? chips.map((c) => {
      const entry = snap.leaderboard.find((r) => r.program_hash === c.program_hash);
      const donors = entry ? donorsOf(entry.program, cat) : [];
      const bad = c.status && c.status !== 'ok';
      return html`<span class=${'pl-chip' + (bad ? ' bad' : '') + (c.is_best ? ' best' : '')} key=${c.eval} title=${c.name || ''}><${Logos} models=${donors} max=${2} size=${14} /><b class="num">${bad ? 'x' : c.score != null ? pct(c.score) : '-'}</b></span>`;
    }) : html`<span class="faint pl-none">${a.role === 'planner' ? 'No scored attempts yet' : a.role === 'analyst' ? 'Maps the donors' : 'Ranks the frontier'}</span>`}</div>
  </a>`;
}

/* ---------- ticker ---------- */
function Ticker({ s, snap }) {
  const names = Object.fromEntries(snap.agents.map((a) => [a.id, a.name]));
  const rows = s.act.slice(-9).reverse();
  return html`<div class="pl-ticker">${rows.map((e) => {
    const sp = SPONSOR_FOR[e.kind];
    const d = e.data || {};
    return html`<div class=${'pl-ev reveal' + (e.status === 'bad' ? ' bad' : '')} key=${e.id != null ? 'i' + e.id : e.ts + e.agent} style=${{ '--i': 0 }}>
      <span class="pl-evm">${sp ? html`<${SponsorMark} who=${sp} size=${14} />` : html`<span class="pl-evd"></span>`}</span>
      <span class="pl-evt"><b>${names[e.agent] || e.agent}</b> ${(() => { const t = String(e.text || ''), n = names[e.agent]; return n && t.startsWith(n + ' ') ? t.slice(n.length + 1) : t; })()}</span>
      ${e.kind === 'eval' && d.is_best && html`<span class="tag g">new best</span>`}
      ${e.kind === 'eval' && d.target != null && html`<span class="num pl-evs">${pctS(d.target)}</span>`}
    </div>`;
  })}</div>`;
}

/* ---------- looping glyphs for techniques ---------- */
function Glyph({ kind }) {
  const k = kind || 'search';
  let body;
  if (k === 'mine') body = html`<g>${[8, 16, 24, 32].map((y, i) => html`<rect class="pg-bar" style=${{ '--d': i * 0.35 + 's' }} x="6" y=${y} width="52" height="4" rx="2"/>`)}<rect class="pg-beam" x="2" y="4" width="60" height="3" rx="1.5"/></g>`;
  else if (k === 'delete') body = html`<g>${Array.from({ length: 12 }, (_, i) => html`<rect class=${'pg-cell' + (i % 5 === 2 ? ' gone' : '')} style=${{ '--d': (i % 4) * 0.2 + 's' }} x=${6 + (i % 6) * 9} y=${i < 6 ? 10 : 24} width="7" height="9" rx="2"/>`)}</g>`;
  else if (k === 'graft') body = html`<g><rect class="pg-host" x="6" y="22" width="52" height="8" rx="3"/><rect class="pg-part" x="26" y="8" width="14" height="10" rx="3"/><line class="pg-link" x1="33" y1="18" x2="33" y2="22"/></g>`;
  else if (k === 'heal') body = html`<g><path class="pg-wave" d="M4 20 C 14 6, 22 34, 32 20 S 50 6, 60 20" fill="none"/><path class="pg-wave2" d="M4 20 C 14 6, 22 34, 32 20 S 50 6, 60 20" fill="none"/></g>`;
  else if (k === 'score') body = html`<g>${[14, 24, 18, 30].map((h, i) => html`<rect class="pg-rise" style=${{ '--d': i * 0.18 + 's' }} x=${10 + i * 13} y=${36 - h} width="9" height=${h} rx="2"/>`)}</g>`;
  else if (k === 'swarm') body = html`<g><line class="pg-edge" x1="14" y1="28" x2="32" y2="10"/><line class="pg-edge" x1="32" y1="10" x2="50" y2="28"/><line class="pg-edge" x1="14" y1="28" x2="50" y2="28"/>${[[14, 28], [32, 10], [50, 28]].map(([x, y], i) => html`<circle class="pg-node" style=${{ '--d': i * 0.4 + 's' }} cx=${x} cy=${y} r="4.5"/>`)}</g>`;
  else body = html`<g>${[10, 20, 30].map((y, i) => html`<rect class="pg-bar" style=${{ '--d': i * 0.3 + 's' }} x="6" y=${y - 2} width=${[40, 30, 36][i]} height="4" rx="2"/>`)}<circle class="pg-lens" cx="48" cy="18" r="7" fill="none"/></g>`;
  return html`<svg class="pg" width="64" height="40" viewBox="0 0 64 40" aria-hidden="true">${body}</svg>`;
}

function countFor(id, s) {
  switch (id) {
    case 'weight-read': case 'skill-map': return s.maps;
    case 'shared-memory': return s.shared;
    case 'benchmark-search': return s.benchmarks;
    case 'program': return s.plans;
    case 'fit-probe': return s.probes;
    case 'graft': return s.withGraft;
    case 'deletion-mask': return s.withMask;
    case 'synthesis-heal': return s.withHeal;
    case 'controls': return s.controls;
    case 'verdicts': return s.comps;
    case 'drift': return s.drifts;
    case 'comparisons': return s.evals;
    case 'heldout': return s.held;
    default: return null;
  }
}

function Techniques({ s }) {
  const order = ['research', 'mine', 'swarm', 'build', 'test'];
  const name = Object.fromEntries(STAGES.map((g) => [g.id, g.name]));
  return html`<div class="pl-tech">${order.map((sid) => {
    const list = TECHNIQUES.filter((t) => t.stage === sid);
    if (!list.length) return null;
    return html`<div class="pl-tg" key=${sid}>
      <div class="pl-tgh">${name[sid]}</div>
      <div class="pl-tgrid">${list.map((t, i) => {
        const n = countFor(t.id, s);
        return html`<div class="pl-tc card reveal" key=${t.id} style=${{ '--i': i }}>
          <div class="pl-tt"><${Glyph} kind=${t.anim} /><div class="pl-tn"><span>${t.name}</span><span class="pl-tm">${t.sponsor && html`<${SponsorMark} who=${t.sponsor} size=${13} />`}${n != null ? html`<span class=${'num ' + (n ? 'pl-ran' : 'faint')}>${n ? 'ran ' + n + (n === 1 ? ' time' : ' times') : 'not run yet'}</span>` : html`<span class="faint">runs while mining</span>`}</span></div></div>
          <p class="pl-tw">${t.what}</p>
          <p class="pl-ty">${t.why}</p>
        </div>`;
      })}</div>
    </div>`;
  })}</div>`;
}

/* ---------- verdict ---------- */
function Verdict({ snap, bt, cat }) {
  const c = snap.comparison;
  const t = primaryTarget(snap.spec);
  const [go, setGo] = useState(false);
  useEffect(() => { const r = requestAnimationFrame(() => setGo(true)); return () => cancelAnimationFrame(r); }, []);
  const rows = [];
  if (bt != null) rows.push({ id: 'ours', label: 'This build', value: bt, ours: true });
  if (c) (c.rows || []).filter((r) => r.measured && r.value != null).forEach((r) => rows.push({ id: r.id, label: r.label, value: r.value, beaten: r.beaten, delta: r.delta }));
  if (!rows.length) return html`<div class="faint">The verdict appears once a build is scored and its references have run.</div>`;
  const mx = Math.max(...rows.map((r) => r.value), 0.01);
  const v = verdictOf(c);
  const entry = snap.leaderboard.find((r) => snap.best && r.program_hash === snap.best.program_hash);
  const comps = (entry && entry.components) || [];
  return html`<div class="pl-verdict">
    ${v && html`<div class=${'pl-vh ' + v.kind}>${v.text}</div>`}
    <div class="pl-bars">${rows.map((r, i) => {
      const col = r.ours ? (c && (c.verdict === 'failed') ? 'var(--amber)' : 'var(--green)') : /lora/.test(r.id) ? 'var(--amber)' : /random/.test(r.id) ? 'var(--text-4)' : 'var(--text-3)';
      return html`<div class=${'pl-bar' + (r.ours ? ' ours' : '')} key=${r.id}><span class="pl-bl">${r.label}</span>
        <span class="pl-bt"><span class="pl-bf" style=${{ width: go ? (r.value / mx) * 100 + '%' : '0%', background: col, transitionDelay: i * 90 + 'ms' }}></span></span>
        <b class="num pl-bv">${pct(r.value)}%</b></div>`;
    })}</div>
    ${t && html`<div class="faint pl-vn">${t.name}, the main target.</div>`}
    ${comps.length > 0 && html`<div class="pl-parts"><span class="label">Parts in the best build</span>${comps.map((p) => html`<button class=${'pl-part ' + (p.verdict === 'contributes' ? 'g' : p.verdict === 'harmful' ? 'r' : 'n')} key=${p.id} onClick=${() => openPart(p.id)}><${ModelLogo} model=${p.donor} size=${14} /><span>${p.label}</span><em>${p.verdict}</em></button>`)}</div>`}
  </div>`;
}

/* ---------- the page ---------- */
export function PipelinePage({ snap: raw, id, cat }) {
  const snap = useMemo(() => ({ ...raw, activity: raw.activity || [], leaderboard: raw.leaderboard || [], agents: raw.agents || [], curve: raw.curve || [], budget: raw.budget || { evals_used: 0, evals_total: 0 }, spec: raw.spec || {} }), [raw]);
  const s = useMemo(() => statsOf(snap, cat), [snap, cat]);
  const bt = bestTarget(snap);
  const entry = snap.best ? snap.leaderboard.find((r) => r.program_hash === snap.best.program_hash) : null;
  const st = activeStages(snap);
  const [sel, setSel] = useState(null);
  const shownStage = sel != null ? sel : Math.min(...(st.on.size ? st.on : [6]));
  const brain = useMemo(() => brainOf(snap, cat, entry), [snap, cat, entry]);
  const [pulse, setPulse] = useState(null);
  const [caption, setCaption] = useState('');
  const q = useRef({ queue: [], seen: null, timer: 0, last: 0, n: 0, snap, cat, entry });
  q.current.snap = snap; q.current.cat = cat; q.current.entry = entry;

  // new events become animation steps
  useEffect(() => {
    const r = q.current;
    const evs = snap.activity;
    const maxId = evs.reduce((m, e) => (e.id != null && e.id > m ? e.id : m), 0);
    if (r.seen == null) {
      r.seen = maxId;
      const last = [...evs].reverse().find((e) => stepsFor({ ...e, kind: kindOf(e) }, snap, cat).length);
      if (last) r.queue.push(...stepsFor({ ...last, kind: kindOf(last) }, snap, cat));
      return;
    }
    const fresh = evs.filter((e) => e.id != null && e.id > r.seen);
    r.seen = maxId;
    for (const e of fresh) r.queue.push(...stepsFor({ ...e, kind: kindOf(e) }, snap, cat));
    if (r.queue.length > 8) r.queue.splice(0, r.queue.length - 8);
  }, [snap, cat]);

  // play one step at a time; with nothing new, replay the best build now and then
  useEffect(() => {
    const r = q.current;
    const play = () => {
      let step = r.queue.shift();
      if (!step) {
        if (Date.now() - r.last < 6500) return;
        const chain = chainFor(r.snap, r.cat, r.entry);
        if (!chain.length) return;
        r.queue.push(...chain.slice(1));
        step = chain[0];
      }
      r.last = Date.now();
      r.n += 1;
      setPulse({ kind: step.kind, at: step.at, partId: step.partId || null, ok: step.ok, key: r.n });
      setCaption(step.cap);
    };
    const t = setInterval(play, 1300);
    play();
    return () => clearInterval(t);
  }, []);

  const lastEv = snap.activity[snap.activity.length - 1];
  const idleCap = snap.status === 'done' ? 'The run is finished. This is the best build.' : lastEv ? cap(String(lastEv.text || '')) : 'Starting the agents';
  const running = snap.status !== 'done' && snap.status !== 'error';
  const ended = !running;

  return html`<div class="rpage"><div class="pl-wrap">
    <section class="pl-hero stage3d">
      <${Brain3D} height=${440} layers=${brain.layers} parts=${brain.parts} cuts=${brain.cuts} pulse=${pulse} mode="host" label=${snap.best ? nameOf(snap.best, cat) : 'The host model'} onPick=${(pid) => openPart(pid)} />
      <div class="stage-hud"><span class=${running ? 'livechip' : 'faint'}>${running ? 'Live' : isEmptyEnd(snap) ? 'Ended, nothing scored' : 'Finished'}</span>
        <span class="pl-pp"><${PoweredBy} who="modal" label="GPUs on" /><${PoweredBy} who="openai" label="Agents by" /></span></div>
      <div class="pl-cap"><span class="pl-capk">${pulse ? { graft: 'Graft', cut: 'Cut', heal: 'Heal', score: 'Score', probe: 'Probe', scan: 'Scan' }[pulse.kind] : 'Ready'}</span><span class="pl-capt">${caption || idleCap}</span></div>
    </section>

    <section class="pl-counters">
      ${[
        ['Evals', html`<${Reel} value=${s.evals} /><span class="pl-of">of ${s.total}</span>`],
        ['Programs tried', html`<${Reel} value=${s.programs} />`],
        ['Parts tried', html`<${Reel} value=${s.parts} />`],
        ['Best score', html`<${PopNum} text=${bt != null ? pctS(bt) : '-'} />`],
        ['Time', html`<${SubtleNum} text=${dur(snap.elapsed_s)} />`],
      ].map(([l, v], i) => html`<div class="pl-k reveal" style=${{ '--i': i }} key=${l}><div class="pl-kv num">${v}</div><div class="pl-kl">${l}</div></div>`)}
    </section>

    <section class="rsec"><h2 class="rh2">The pipeline</h2>
      <${StageRail} snap=${snap} s=${s} bt=${bt} sel=${shownStage} setSel=${setSel} />
      <${StageDetail} i=${shownStage} s=${s} snap=${snap} cat=${cat} />
    </section>

    <section class="rsec pl-two">
      <div><h2 class="rh2">Agents</h2><div class="pl-lanes">${snap.agents.map((a, i) => html`<${Lane} key=${a.id} a=${a} snap=${snap} cat=${cat} i=${i} />`)}</div></div>
      <div><h2 class="rh2 pl-rh">Live events <${PoweredBy} who="pastdev" label="Shared on" /></h2><${Ticker} s=${s} snap=${snap} /></div>
    </section>

    <section class="rsec"><h2 class="rh2">Verdict</h2><${Verdict} snap=${snap} bt=${bt} cat=${cat} /></section>

    <section class="rsec"><h2 class="rh2">Techniques the agents use</h2><${Techniques} s=${s} /></section>
  </div></div>`;
}
