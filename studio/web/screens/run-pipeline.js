// The autoresearch dashboard of one run (#/run/<id>/pipeline).
// Top to bottom: the objective with real progress, the planners that are working now (each with its
// program, hypothesis and step progress), the experiment log, a right column with the best build,
// the score chart, the top findings and the agents, and a collapsible "How it works" section.
// Every number comes from the run snapshot, the agent detail endpoint or timestamps in the activity feed.
import { html, useState, useEffect, useRef, useMemo, Fragment } from '../lib.js';
import { api } from '../api.js';
import { useHealth, useModalChip, dur, pct, pctS, sgn, primaryTarget, clamp } from '../util.js';
import { Avatar } from '../components.js';
import { SubtleNum, PopNum } from '../motion.js';
import { Brain3D } from '../brain3d.js';
import { ModelLogo, PoweredBy, SponsorMark } from '../logos.js';
import { STAGES, TECHNIQUES } from '../techniques.js';
import { Logos, donorsOf, verdictOf, Compare } from '../cmp.js';
import { openPart } from './part-detail.js';
import { openProgramInBuilder } from './shared.js';
import { LineChart } from './run-results.js';
import { deriveFindings, ProgramStrip, kindOf } from './run-findings.js';
import { ACTIVE, STATUS_WORD, AgentList, isEmptyEnd, nameOf, bestTarget } from './run-shared.js';

const SHARED = new Set(['plan', 'map', 'verdict', 'handoff', 'nudge_applied']);
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const hashOf = (s) => { let h = 0; for (const c of String(s || '')) h = (h * 31 + c.charCodeAt(0)) >>> 0; return h; };
const med = (xs) => {
  const v = xs.filter((x) => x != null && isFinite(x) && x > 0).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
};
const fmtPct = (v) => (v == null ? '-' : (v * 100).toFixed(Math.abs(v * 100 - Math.round(v * 100)) < 0.05 ? 0 : 1) + '%');
const secs = (s) => (s == null ? '' : s < 1.5 ? 'a moment' : dur(s));
const ended = (snap) => snap.status === 'done' || snap.status === 'error';

function useNow(ms = 1000) {
  const [n, setN] = useState(Date.now() / 1000);
  useEffect(() => { const t = setInterval(() => setN(Date.now() / 1000), ms); return () => clearInterval(t); }, [ms]);
  return n;
}
function useWide(px) {
  const q = '(min-width: ' + px + 'px)';
  const [w, setW] = useState(() => (window.matchMedia ? window.matchMedia(q).matches : true));
  useEffect(() => {
    if (!window.matchMedia) return;
    const m = window.matchMedia(q);
    const f = () => setW(m.matches);
    f();
    m.addEventListener('change', f);
    return () => m.removeEventListener('change', f);
  }, [q]);
  return w;
}

/* ---------- the program as Brain3D wants it ---------- */
function maskIds(m, out = []) {
  if (!m) return out;
  if (m.op === 'use') out.push(m.id);
  else if (m.op === 'restrict') maskIds(m.inner, out);
  else { maskIds(m.a, out); maskIds(m.b, out); }
  return out;
}
function brainOf(prog, cat, comps) {
  const layers = (cat && cat.host && cat.host.layers) || 16;
  const list = comps || [];
  const parts = prog
    ? (prog.grafts || []).map((g) => {
      const c = list.find((x) => x.id === g.id);
      const p = cat && cat.parts.find((x) => x.id === g.id);
      return { id: g.id, label: c ? c.label : p ? p.label : 'Part', kind: (c && c.kind) || (p && p.kind) || 'block', at: Math.min(layers - 1, Math.max(0, g.at)), donor: (c && c.donor) || (p && p.donor) || '', verdict: c ? c.verdict : null };
    })
    : [];
  const ids = prog ? maskIds(prog.masks) : [];
  const cuts = ids.map((id, i) => {
    const m = cat && (cat.masks || []).find((x) => x.id === id);
    return { layer: Math.round(((i + 1) / (ids.length + 1)) * (layers - 1)), frac: 0.22, label: m ? m.label : 'Skill removed' };
  });
  return { layers, parts, cuts };
}

/* ---------- timing: how long each step took earlier in this run ---------- */
// Two sources. (1) The activity feed has a timestamp for each probe row and each eval row, so the time from one
// eval to the next probe is planning, and the time from a probe to its eval is the rest of the attempt.
// (2) This page watches agent status change, and keeps how long each status lasted. Compile, heal and score are not
// logged apart in the feed, so when only (1) is known they share the measured time in proportion to the program's work.
function useStepLog(snap) {
  const ref = useRef({ rec: {}, obs: { thinking: [], probing: [], compiling: [], scoring: [], control: [] } });
  const r = ref.current;
  useMemo(() => {
    const t = Date.now() / 1000;
    for (const a of snap.agents) {
      const prev = r.rec[a.id];
      if (!prev) { r.rec[a.id] = { status: a.status, since: null, partial: false }; continue; }
      if (prev.status === a.status) continue;
      if (ACTIVE[prev.status] && prev.since != null && !prev.partial) {
        const arr = r.obs[a.id === 'judge' ? 'control' : prev.status];
        if (arr) { arr.push(t - prev.since); if (arr.length > 8) arr.shift(); }
      }
      prev.partial = prev.status === 'paused';
      prev.status = a.status;
      prev.since = a.status === 'paused' ? null : t;
    }
  }, [snap]);
  return r;
}

function timingOf(snap, obs) {
  const seg1 = [], seg2 = [], cyc = [];
  const st = {};
  for (const raw of snap.activity) {
    const e = { ...raw, kind: kindOf(raw) };
    if (e.agent === 'you') continue;
    const s = (st[e.agent] = st[e.agent] || { end: null, probe: null });
    if (e.kind === 'probe') { if (s.end != null && e.ts > s.end) seg1.push(e.ts - s.end); s.probe = e.ts; }
    else if (e.kind === 'eval') {
      if (s.probe != null && s.end != null && s.probe > s.end) seg2.push(e.ts - s.probe);
      if (s.end != null) cyc.push(e.ts - s.end);
      s.end = e.ts; s.probe = null;
    }
  }
  const last = (a) => a.slice(-8);
  const cy = med(last(cyc));
  const actPlan = med(last(seg1)) ?? (cy != null ? cy * 0.25 : null);
  const actRest = med(last(seg2)) ?? (cy != null ? cy * 0.75 : null);
  const o = { P: med(obs.thinking), PR: med(obs.probing), C: med(obs.compiling), S: med(obs.scoring) };
  const ctrl = med(obs.control);
  // the time each step is expected to take, for a program with `hs` heal steps (null when heal is unknown)
  const est = (hs) => {
    const wp = 0.1, wc = 0.25, wh = hs == null ? 0.2 : hs > 0 ? Math.min(0.4, 0.1 + hs / 2500) : 0, ws = 0.35;
    const wt = wp + wc + wh + ws;
    const rest = actRest;
    const hw = wh / (wc + wh);
    return {
      plan: o.P ?? actPlan,
      probe: o.PR ?? (rest != null ? rest * (wp / wt) : null),
      compile: o.C != null ? o.C * (1 - hw) : rest != null ? rest * (wc / wt) : null,
      heal: hs === 0 ? 0 : o.C != null ? o.C * hw : rest != null ? rest * (wh / wt) : null,
      score: o.S ?? (rest != null ? rest * (ws / wt) : null),
      verdict: 3,
    };
  };
  return { est, ctrl, cycle: cy };
}

const STEPS = [['plan', 'Plan'], ['probe', 'Probe'], ['compile', 'Compile'], ['heal', 'Heal'], ['score', 'Score'], ['verdict', 'Verdict']];
const JSTEPS = [['compile', 'Compile'], ['heal', 'Heal'], ['score', 'Score'], ['verdict', 'Verdict']];
const JW = [0.25, 0.3, 0.35, 0.1];
const STATUS_STEP = { thinking: 'plan', probing: 'probe', compiling: 'compile', scoring: 'score' };
const STEP_CAP = {
  plan: 'Writing the next program', probe: 'Checking how well the part fits the host', compile: 'Compiling the program into one network',
  heal: 'Healing: training the glue layers on general text', score: 'Scoring the network on the benchmarks', verdict: 'Deciding whether to keep the result',
};
const JSTEP_CAP = { compile: 'Compiling the same build with random parts', heal: 'Healing the random build', score: 'Scoring the random build', verdict: 'Comparing it with the real parts' };

function lastEvalTs(snap, id) {
  for (let i = snap.activity.length - 1; i >= 0; i--) { const e = snap.activity[i]; if (e.agent === id && kindOf(e) === 'eval') return e.ts; }
  return null;
}
function lastRow(snap, id) {
  for (let i = snap.activity.length - 1; i >= 0; i--) if (snap.activity[i].agent === id) return snap.activity[i];
  return null;
}

// where one planner is in its attempt, from its status, its latest rows and the step durations
function plannerModel(a, snap, log, tm, prog, now) {
  const rec = log.rec[a.id];
  const hs = prog ? (prog.heal ? prog.heal.steps : 0) : null;
  const E = tm.est(hs);
  const t0 = lastEvalTs(snap, a.id) ?? snap.started_at;
  const row = lastRow(snap, a.id);
  let key = STATUS_STEP[a.status] || 'plan';
  let flash = false;
  if (row && kindOf(row) === 'eval' && now - row.ts < 3.5 && ACTIVE[a.status]) { key = 'verdict'; flash = true; }
  const known = STEPS.every(([k]) => E[k] != null);
  const order = STEPS.map(([k]) => k);
  const start = {};
  let acc = 0;
  for (const k of order) { start[k] = acc; acc += E[k] || 0; }
  const total = acc;
  let since = rec && rec.since != null ? rec.since : known ? t0 + (start[key] || 0) : null;
  if (flash) since = row.ts;
  if (since != null && since > now) since = now - 1;
  if (key === 'compile' && hs > 0 && E.compile != null && since != null && now - since > E.compile) { since += E.compile; key = 'heal'; }
  const idx = order.indexOf(key);
  let frac = null, pctNow = null, left = null, over = false;
  if (known && since != null && total > 0) {
    const d = Math.max(0, now - since);
    over = d > (E[key] || 0) * 1.25 && !flash;
    frac = E[key] ? clamp(d / E[key], 0, 0.95) : 0.5;
    pctNow = clamp((start[key] + frac * (E[key] || 0)) / total, 0, 0.99);
    left = Math.max(0, total * (1 - pctNow));
  }
  const steps = STEPS.map(([k, label], i) => ({ key: k, label, state: k === 'heal' && hs === 0 ? 'skip' : i < idx ? 'done' : i === idx ? 'cur' : 'next', fill: i < idx ? 100 : i === idx && frac != null ? frac * 100 : 0 }));
  return { steps, key, idx, pct: pctNow, left, over, elapsed: Math.max(0, now - t0), caption: STEP_CAP[key], n: order.length, frac, flash };
}

function judgeModel(a, snap, log, tm, entry, now) {
  const rec = log.rec[a.id];
  const total = tm.ctrl;
  const ev = entry ? lastEvalForHash(snap, entry.program_hash) : null;
  let vts = null;
  for (let i = snap.activity.length - 1; i >= 0; i--) { const e = snap.activity[i]; if (e.agent === a.id && kindOf(e) === 'verdict') { vts = e.ts; break; } }
  const t0 = Math.max(ev ? ev.ts : 0, vts || 0) || null;
  let since = rec && rec.since != null ? rec.since : t0;
  if (since != null && since > now) since = now - 1;
  const d = since != null ? Math.max(0, now - since) : null;
  let p = null, idx = 0;
  if (total != null && d != null) {
    p = clamp(d / total, 0, 0.97);
    let c = 0;
    idx = JW.findIndex((w) => { c += w; return p < c; });
    if (idx < 0) idx = 2;
  }
  const over = total != null && d != null && d > total * 1.25;
  const cum = [0, 0.25, 0.55, 0.9];
  const steps = JSTEPS.map(([k, label], i) => ({ key: k, label, state: p == null ? (i === 0 ? 'cur' : 'next') : i < idx ? 'done' : i === idx ? 'cur' : 'next', fill: p == null ? 0 : i < idx ? 100 : i === idx ? clamp((p - cum[i]) / JW[i], 0, 1) * 100 : 0 }));
  return { steps, key: JSTEPS[idx][0], idx, pct: p, left: p != null && total != null ? Math.max(0, total * (1 - p)) : null, over, elapsed: d, caption: JSTEP_CAP[JSTEPS[idx][0]], n: JSTEPS.length, frac: null };
}
function lastEvalForHash(snap, h) {
  for (let i = snap.activity.length - 1; i >= 0; i--) { const e = snap.activity[i]; if (kindOf(e) === 'eval' && e.data && e.data.program_hash === h) return e; }
  return null;
}

/* ---------- the objective ---------- */
function runWord(snap) {
  if (snap.status === 'starting') return ['Starting', 'b'];
  if (snap.status === 'paused') return ['Paused', 'w'];
  if (snap.status === 'stopping') return ['Stopping', 'w'];
  if (snap.status === 'error') return ['Stopped by an error', 'r'];
  if (snap.status === 'done') return snap.outcome === 'stopped' ? ['Stopped', 'w'] : isEmptyEnd(snap) ? ['Ended, nothing scored', 'w'] : ['Finished', 'g'];
  return ['Running', 'b'];
}

function Objective({ snap, cat }) {
  const health = useHealth();
  const modalChip = useModalChip();
  const spec = snap.spec;
  const t = primaryTarget(spec);
  const host = String(spec.host || '').split('/').pop() || 'the host model';
  const hostScore = t ? ((snap.base_scores || {})[t.id] ?? t.base) : null;
  const lim = snap.limits || spec.limits || {};
  const maxB = lim.max_total_params_b ?? (spec.size && spec.size.max_params_b);
  const guards = (spec.benchmarks || []).filter((b) => b.kind === 'guard' && b.floor != null);
  const others = (spec.benchmarks || []).filter((b) => b.kind === 'target' && (!t || b.id !== t.id));
  const best = bestTarget(snap);
  const [word, tone] = runWord(snap);
  const live = !ended(snap) && snap.status !== 'paused';
  const research = (spec.benchmarks || []).some((b) => b.research);
  const sentence = t
    ? 'Raise ' + host + ' from ' + fmtPct(hostScore) + ' to ' + fmtPct(t.target) + ' on ' + t.name + ' by grafting in parts from other models'
      + (maxB != null ? ', in at most ' + maxB + 'B parameters' : '') + (lim.max_heal_min ? ' and ' + lim.max_heal_min + ' minutes of healing' : '')
      + (guards.length ? ', while keeping ' + guards.map((g) => g.name + ' above ' + fmtPct(g.floor)).join(' and ') : '') + '.'
    : 'Improve ' + host + ' on the targets of this spec.';
  return html`<section class="pl-obj card">
    <div class="pl-obj-top">
      <span class=${'pl-chipst ' + tone}>${live ? html`<span class="pl-livedot"></span>` : null}${word}</span>
      <span class="pl-obj-eyebrow">Objective</span>
      <span class="grow"></span>
      <span class="pl-sponsors" aria-label="Services this run uses">
        <${PoweredBy} who="openai" label="Agents by" size=${12} />
        ${modalChip && html`<${PoweredBy} who="modal" label="GPUs on" size=${12} />`}
        <${PoweredBy} who="pastdev" label="Memory on" size=${12} />
        ${research && html`<${PoweredBy} who="exa" label="Search by" size=${12} />`}
      </span>
    </div>
    <p class="pl-sentence">${sentence}</p>
    ${others.length > 0 && html`<p class="pl-also">Also scored on ${others.map((b) => b.name).join(', ')}. The bar below follows ${t ? t.name : 'the main target'}.</p>`}
    ${t && html`<${GoalBar} t=${t} host=${hostScore} best=${best} done=${ended(snap)} />`}
    <${Meters} snap=${snap} />
  </section>`;
}

// host score at the left, target at the right, the best score so far as a marker
function GoalBar({ t, host, best, done }) {
  const lo = host ?? t.base ?? 0, hi = t.target ?? Math.max(best ?? 0, lo + 0.1);
  const span = Math.max(0.0001, hi - lo);
  const at = (v) => clamp(((v - lo) / span) * 100, 0, 100);
  const beat = best != null && best >= hi;
  const fill = best != null ? at(best) : 0;
  const lora = t.lora != null && t.lora > lo && t.lora < hi ? t.lora : null;
  const gain = best != null ? (best - lo) * 100 : null;
  const left = best != null && hi > lo ? clamp(((hi - best) / (hi - lo)) * 100, 0, 100) : null;
  return html`<div class="pl-goal" role="img" aria-label=${'Best ' + t.name + ' score ' + (best != null ? fmtPct(best) : 'not scored yet') + ', from ' + fmtPct(lo) + ' to a target of ' + fmtPct(hi)}>
    <div class="pl-goal-head">
      <span class="pl-goal-now">${best != null ? html`<span class="pl-big num"><${PopNum} text=${fmtPct(best)} /></span> <span class="pl-goal-on">on ${t.name}, target ${fmtPct(hi)}</span>` : html`<span class="pl-goal-on">No valid build scored yet. The host scores ${fmtPct(lo)} on ${t.name}, the target is ${fmtPct(hi)}.</span>`}</span>
      <span class="pl-goal-gain num">${best == null ? '' : beat ? 'Target beaten by ' + ((best - hi) * 100).toFixed(1) + ' points' : left != null ? ((hi - best) * 100).toFixed(1) + ' points to go' : ''}${best != null && gain != null && !beat ? '' : ''}</span>
    </div>
    <div class="pl-track">
      <i class=${'pl-fill' + (beat ? ' beat' : '')} style=${{ width: fill + '%' }}></i>
      ${lora != null && html`<span class="pl-tick lora" style=${{ left: at(lora) + '%' }} title=${'A rank-16 LoRA on the host scores ' + fmtPct(lora)}></span>`}
      ${best != null && html`<span class="pl-mark" style=${{ left: fill + '%' }} title=${'Best so far ' + fmtPct(best)}></span>`}
    </div>
    <div class="pl-goal-scale">
      <span><b class="num">${fmtPct(lo)}</b> host, where it starts</span>
      ${lora != null && html`<span class="pl-lorat" style=${{ left: at(lora) + '%' }}>LoRA ${fmtPct(lora)}</span>`}
      <span class="r"><b class="num">${fmtPct(hi)}</b> target</span>
    </div>
  </div>`;
}

function Meters({ snap }) {
  const now = useNow(1000);
  const recv = useMemo(() => Date.now() / 1000, [snap]);
  const b = snap.budget;
  const run = snap.status === 'running' || snap.status === 'starting';
  const el = snap.elapsed_s + (run ? Math.max(0, now - recv) : 0);
  const evalFrac = b.evals_total ? clamp(b.evals_used / b.evals_total, 0, 1) : 0;
  const gpuU = (b.compute_used_s || 0) / 60, gpuT = (b.compute_total_s || 0) / 60;
  const gpuFrac = gpuT ? clamp(gpuU / gpuT, 0, 1) : 0;
  const m1 = (v) => (v >= 10 ? Math.round(v) : Math.round(v * 10) / 10);
  // when the run ends: whichever budget runs out first, at the pace so far
  const c = [];
  if (b.evals_used > 0 && snap.elapsed_s > 0) c.push((snap.elapsed_s / b.evals_used) * (b.evals_total - b.evals_used));
  if (b.compute_used_s > 0 && snap.elapsed_s > 0) c.push((snap.elapsed_s / b.compute_used_s) * (b.compute_total_s - b.compute_used_s));
  const left = c.length ? Math.max(0, Math.min(...c) - (run ? Math.max(0, now - recv) : 0)) : null;
  const done = ended(snap);
  const tile = (label, big, of, frac, note, key) => html`<div class="pl-meter" key=${key}>
    <div class="pl-mlab">${label}</div>
    <div class="pl-mval"><span class="num"><${SubtleNum} text=${big} /></span><span class="pl-mof">${of}</span></div>
    <div class="pl-mbar"><i style=${{ width: frac * 100 + '%' }}></i></div>
    <div class="pl-mnote">${note}</div>
  </div>`;
  return html`<div class="pl-meters">
    ${tile('Evals used', String(b.evals_used), ' of ' + b.evals_total + ' evals', evalFrac, evalFrac >= 1 ? 'Eval budget used' : b.evals_total - b.evals_used + ' evals left', 'e')}
    ${tile('GPU time used', String(m1(gpuU)), ' of ' + m1(gpuT) + ' GPU minutes', gpuFrac, gpuFrac >= 1 ? 'GPU budget used' : m1(Math.max(0, gpuT - gpuU)) + ' GPU minutes left', 'g')}
    ${tile('Time', dur(el), done ? ' in total' : ' elapsed', done ? 1 : evalFrac, done ? (snap.stopped_early ? 'Stopped before the budget ran out' : 'The run has ended') : snap.status === 'paused' ? 'Paused, the clock is stopped' : left != null ? 'About ' + dur(left) + ' left, at the pace so far' : 'Time left shows after the first eval', 't')}
  </div>`;
}

/* ---------- now running ---------- */
function oneSentence(text, max = 190) {
  const s = String(text || '').replace(/\s+/g, ' ').trim();
  if (!s) return '';
  const parts = s.split(/(?<=[.!?])\s+/);
  let out = parts[0];
  if (out.length < 70 && parts[1]) out += ' ' + parts[1];
  if (out.length > max) { out = out.slice(0, max); out = out.slice(0, out.lastIndexOf(' ') > 60 ? out.lastIndexOf(' ') : max) + '...'; }
  return out;
}

function useAgentDetail(runId, a, on) {
  const [det, setDet] = useState(null);
  const key = a.status + '|' + a.attempts + '|' + a.now;
  useEffect(() => {
    if (!on) return;
    let alive = true;
    api.getAgent(runId, a.id).then((d) => { if (alive) setDet(d); }).catch(() => {});
    return () => { alive = false; };
  }, [runId, a.id, key, on]);
  return det;
}

function StepBar({ m, paused, kind }) {
  const unknown = m.pct == null;
  return html`<div class=${'pl-stepbox' + (paused ? ' paused' : '')}>
    <div class="pl-steps" role="list" aria-label=${kind === 'judge' ? 'Steps of the control run' : 'Steps of this attempt'}>${m.steps.map((s) => html`<div class=${'pl-stp ' + s.state + (s.state === 'cur' && unknown ? ' unk' : '')} role="listitem" key=${s.key} aria-current=${s.state === 'cur' ? 'step' : null}>
      <span class="pl-stp-bar"><i style=${{ width: (s.state === 'cur' && unknown ? 100 : s.fill) + '%' }}></i></span>
      <span class="pl-stp-lab">${s.label}${s.state === 'skip' ? ' (none)' : ''}</span>
    </div>`)}</div>
    <div class="pl-stepline">
      <span class="pl-stepcap"><span class="pl-stepn num">Step ${m.idx + 1} of ${m.n}</span>${paused ? 'Paused' : m.caption}</span>
      <span class="pl-steptime num">${unknown ? 'Time estimate shows after one full attempt' : Math.round(m.pct * 100) + '%'}${m.elapsed != null && !unknown ? ', ' + secs(m.elapsed) + ' in' : ''}${m.left != null && !paused ? (m.over ? ', taking longer than usual' : ', about ' + secs(m.left) + ' left') : ''}</span>
    </div>
  </div>`;
}

// the clock lives here so the 3D scene beside it does not re-render each second
function Steps({ kind, a, snap, log, prog, entry, onStep }) {
  const now = useNow(1000);
  const paused = snap.status === 'paused' || a.status === 'paused';
  const recv = useMemo(() => Date.now() / 1000, [snap]);
  const tm = useMemo(() => timingOf(snap, log.obs), [snap]);
  const t = paused ? recv : now;
  const m = kind === 'judge' ? judgeModel(a, snap, log, tm, entry, t) : plannerModel(a, snap, log, tm, prog, t);
  useEffect(() => { if (onStep) onStep(m.key, m.flash); }, [m.key]);
  return html`<${StepBar} m=${m} paused=${paused} kind=${kind} />`;
}

function lastResult(snap, id, t) {
  const c = [...snap.curve].reverse().find((x) => x.agent === id);
  if (!c) return null;
  const entry = snap.leaderboard.find((r) => r.program_hash === c.program_hash);
  const act = lastEvalForHash(snap, c.program_hash);
  const d = decisionOf(c, entry, act);
  return { c, d };
}

function BrainBox({ prog, cat, comps, pulse, label, wide, height }) {
  const b = useMemo(() => brainOf(prog, cat, comps), [prog, cat, comps]);
  if (!wide) return null;
  return html`<div class="pl-brain" style=${{ height: height + 'px' }}><${Brain3D} height=${height} layers=${b.layers} parts=${b.parts} cuts=${b.cuts} pulse=${pulse} mode="host" steps=${false} onPick=${(pid) => openPart(pid)} /></div>`;
}

function PlannerCard({ a, snap, cat, log, wide, idx }) {
  const live = !!ACTIVE[a.status] || a.status === 'paused';
  const det = useAgentDetail(snap.run_id, a, live);
  const prog = det && det.current ? det.current.program : null;
  const pname = det && det.current ? det.current.name || det.current.describe : '';
  const entry = prog ? snap.leaderboard.find((r) => JSON.stringify(r.program) === JSON.stringify(prog)) : null;
  const think = det && det.thinking && det.thinking.length ? det.thinking[det.thinking.length - 1].text : null;
  const planRow = [...snap.activity].reverse().find((e) => e.agent === a.id && kindOf(e) === 'plan');
  const hyp = oneSentence(think || (planRow && planRow.text) || a.now);
  const [pulse, setPulse] = useState(null);
  const n = useRef(0);
  const layers = (cat && cat.host && cat.host.layers) || 16;
  const onStep = (key) => {
    n.current += 1;
    const g = prog && prog.grafts && prog.grafts[0];
    const at = g ? Math.min(layers - 1, g.at) : layers >> 1;
    const kind = { plan: 'scan', probe: 'probe', compile: 'graft', heal: 'heal', score: 'score', verdict: 'score' }[key];
    const ok = key === 'verdict' ? (lastResult(snap, a.id) || { c: {} }).c.status === 'ok' : null;
    setPulse({ kind, at: kind === 'scan' ? hashOf(a.id + a.attempts) % layers : at, partId: kind === 'graft' && g ? g.id : null, ok, key: n.current });
  };
  const lr = lastResult(snap, a.id);
  const t = primaryTarget(snap.spec);
  const comps = (entry && entry.components) || null;
  const vmap = {};
  (comps || []).forEach((c) => { vmap[c.id] = c.verdict; });
  return html`<article class=${'pl-now card reveal' + (a.status === 'paused' ? '' : ' working')} style=${{ '--i': idx }}>
    <header class="pl-nh">
      <${Avatar} agent=${a} size=${28} />
      <div class="pl-nt"><a class="pl-nn" href=${'#/run/' + snap.run_id + '/agents/' + a.id}>${a.name}</a><span class="pl-nm">${a.blurb || 'Planner'}${a.attempts ? ', ' + a.attempts + (a.attempts === 1 ? ' attempt scored' : ' attempts scored') : ''}</span></div>
      <span class=${'pl-stat ' + (a.status === 'paused' ? 'w' : 'b')}>${STATUS_WORD[a.status] || a.status}</span>
    </header>
    <div class=${'pl-nbody' + (wide ? ' wide' : '')}>
      <${BrainBox} prog=${prog} cat=${cat} comps=${comps} pulse=${pulse} label=${pname || 'Program being tested'} wide=${wide} height=${250} />
      <div class="pl-ninfo">
        <div class="pl-lab">Hypothesis</div>
        <p class="pl-hyp">${hyp || 'Waiting for the first idea.'}</p>
        <div class="pl-lab">Program being tested${prog ? '' : ' (loading)'}</div>
        ${prog ? html`<${ProgramStrip} program=${prog} cat=${cat} vmap=${vmap} />` : html`<div class="faint pl-small">${a.now}</div>`}
        ${pname && html`<div class="pl-pname">${pname}</div>`}
      </div>
    </div>
    <${Steps} kind="planner" a=${a} snap=${snap} log=${log} prog=${prog} onStep=${onStep} />
    ${lr && html`<div class="pl-last"><span class="pl-lab">Last result</span><span class="pl-lastt">${lr.c.name || 'A program'}: <b class="num">${lr.c.score != null ? fmtPct(lr.c.score) : '-'}</b>${t ? ' on ' + t.name : ''}, ${lr.d.label.toLowerCase()}.</span></div>`}
  </article>`;
}

function JudgeCard({ a, snap, cat, log, idx }) {
  const nm = String(a.now || '');
  const entry = snap.leaderboard.find((r) => r.name && nm.includes(r.name)) || null;
  return html`<article class="pl-now pl-judge card reveal working" style=${{ '--i': idx }}>
    <header class="pl-nh">
      <${Avatar} agent=${a} size=${28} />
      <div class="pl-nt"><a class="pl-nn" href=${'#/run/' + snap.run_id + '/agents/' + a.id}>${a.name}</a><span class="pl-nm">Control run: the same build with random parts</span></div>
      <span class="pl-stat b">Checking</span>
    </header>
    <div class="pl-ninfo solo">
      <div class="pl-lab">What it is checking</div>
      <p class="pl-hyp">${oneSentence(a.now)}</p>
      ${entry && html`<${ProgramStrip} program=${entry.program} cat=${cat} />`}
      <p class="pl-small faint">A part counts only if the build beats the same build with random parts of the same size.</p>
    </div>
    <${Steps} kind="judge" a=${a} snap=${snap} log=${log} entry=${entry} />
  </article>`;
}

function NowRunning({ snap, cat, log, best }) {
  const wide = useWide(760);
  const planners = snap.agents.filter((a) => a.role === 'planner');
  const done = ended(snap);
  const working = done ? [] : planners.filter((a) => ACTIVE[a.status] || a.status === 'paused');
  const idle = done ? [] : planners.filter((a) => !working.includes(a));
  const judge = snap.agents.find((a) => a.role === 'judge');
  const judging = !done && judge && judge.status === 'scoring';
  const bestRow = best ? snap.leaderboard.find((r) => r.program_hash === best.program_hash) : null;
  const sub = done ? 'The run has ended. This is the program that scored best.' : working.length ? working.length + (working.length === 1 ? ' planner is' : ' planners are') + ' testing a program each. The 3D view shows the program on the host model.' : 'No planner is working at this moment.';
  return html`<section class="pl-sec">
    <div class="pl-sh"><h2>${done ? 'Final result' : 'Now running'}</h2><p>${sub}</p></div>
    <div class="pl-nowgrid">
      ${working.map((a, i) => html`<${PlannerCard} key=${a.id} a=${a} snap=${snap} cat=${cat} log=${log} wide=${wide} idx=${i} />`)}
      ${judging && html`<${JudgeCard} key="judge" a=${judge} snap=${snap} cat=${cat} log=${log} idx=${working.length} />`}
      ${done && html`<article class="pl-now card">
        <header class="pl-nh"><div class="pl-nt"><span class="pl-nn">${best ? 'The final best program' : 'Nothing was scored'}</span><span class="pl-nm">${best ? nameOf(best, cat) : snap.empty_reason || 'The run ended before a program was scored.'}</span></div></header>
        ${best && html`<div class=${'pl-nbody' + (wide ? ' wide' : '')}>
          <${BrainBox} prog=${best.program} cat=${cat} comps=${bestRow && bestRow.components} pulse=${null} label=${nameOf(best, cat)} wide=${wide} height=${250} />
          <div class="pl-ninfo"><div class="pl-lab">Parts</div><${ProgramStrip} program=${best.program} cat=${cat} vmap=${Object.fromEntries(((bestRow && bestRow.components) || []).map((c) => [c.id, c.verdict]))} /></div>
        </div>`}
      </article>`}
    </div>
    ${!done && !working.length && !judging && html`<div class="pl-empty card">${snap.status === 'starting' ? 'The agents are starting. The first program shows up in a few seconds.' : 'The agents are between attempts.'}</div>`}
    ${idle.length > 0 && html`<p class="pl-idle">${idle.map((a) => a.name).join(', ')} ${idle.length === 1 ? 'is' : 'are'} idle right now.</p>`}
  </section>`;
}

/* ---------- experiment log ---------- */
const FAIL = { rejected_fit: 'fit too low', compile_error: 'did not compile', invalid: 'not a valid program', infra_error: 'GPU job failed' };
function decisionOf(c, entry, act) {
  if (c.is_best) return { k: 'best', label: 'Kept as new best' };
  if (c.status === 'collapsed') {
    const gs = (act && act.data && act.data.guards) || [];
    const g = gs.find((x) => x.ok === false);
    return { k: 'collapsed', label: 'Collapsed', note: g ? g.name + ' fell to ' + fmtPct(g.score) + ', floor ' + fmtPct(g.floor) : 'a guarded skill fell below its floor' };
  }
  if (c.status && c.status !== 'ok') return { k: 'failed', label: 'Failed: ' + (FAIL[c.status] || String(c.status).replace(/_/g, ' ')) };
  if (entry && entry.vs_control != null && entry.vs_control <= 0.02) return { k: 'random', label: 'No better than random', note: 'only ' + sgn(entry.vs_control) + ' points above random parts' };
  return { k: 'discarded', label: 'Discarded', note: 'did not beat the best so far' };
}

function buildLog(snap) {
  const t = primaryTarget(snap.spec);
  const hostScore = t ? ((snap.base_scores || {})[t.id] ?? t.base) : null;
  const acts = snap.activity.map((e) => ({ ...e, kind: kindOf(e) }));
  const byHash = {};
  acts.filter((e) => e.kind === 'eval').forEach((e) => { const h = e.data && e.data.program_hash; if (h) (byHash[h] = byHash[h] || []).push(e); });
  const lb = {};
  snap.leaderboard.forEach((r) => { lb[r.program_hash] = r; });
  const items = [];
  let lastTs = -Infinity;
  snap.curve.forEach((c, i) => {
    const act = c.program_hash && byHash[c.program_hash] ? byHash[c.program_hash].shift() : null;
    const ts = act ? act.ts : lastTs;
    lastTs = ts;
    const prevBest = i > 0 ? snap.curve[i - 1].best : null;
    const ref = prevBest != null ? prevBest : hostScore;
    const entry = lb[c.program_hash] || null;
    items.push({ key: 'e' + c.eval + ':' + (c.program_hash || i), type: 'eval', n: c.eval, ts, c, entry, act, ref, refHost: prevBest == null, delta: c.score != null && ref != null ? c.score - ref : null, dec: decisionOf(c, entry, act), ord: i });
  });
  acts.filter((e) => e.kind === 'refused').forEach((e, i) => items.push({ key: 'r' + (e.id != null ? e.id : e.ts + ':' + i), type: 'refused', ts: e.ts, e, ord: 1e6 + i, dec: { k: 'refused', label: 'Refused over a limit' } }));
  items.sort((x, y) => x.ts - y.ts || x.ord - y.ord);
  return items.reverse();
}

function Guards({ act, snap }) {
  const gs = (act && act.data && act.data.guards) || [];
  if (!gs.length) return null;
  return html`<div class="pl-guards"><span class="pl-lab">Guarded skills</span>${gs.map((g) => html`<span class=${'pl-guard ' + (g.ok ? 'ok' : 'bad')} key=${g.id}>${g.name} <b class="num">${fmtPct(g.score)}</b>, floor ${fmtPct(g.floor)}</span>`)}</div>`;
}

function LogDetail({ it, snap, cat }) {
  const { entry, act, c } = it;
  const comps = (entry && entry.components) || [];
  const vmap = Object.fromEntries(comps.map((x) => [x.id, x.verdict]));
  const nContrib = comps.filter((x) => x.verdict === 'contributes').length;
  const agent = snap.agents.find((a) => a.id === c.agent);
  return html`<div class="pl-ld">
    <div class="pl-ldc">
      <div class="pl-lab">Program and its parts</div>
      ${entry ? html`<${ProgramStrip} program=${entry.program} cat=${cat} vmap=${vmap} stop />` : html`<div class="faint pl-small">This program has left the top of the leaderboard, so its parts are no longer listed.</div>`}
      ${comps.length > 0 && html`<p class="pl-small">${nContrib} of ${comps.length} ${comps.length === 1 ? 'part contributes' : 'parts contribute'}. Click a part to open its window.</p>`}
      <${Guards} act=${act} snap=${snap} />
      ${it.dec.note && html`<p class="pl-small faint">${cap(it.dec.note)}.</p>`}
    </div>
    <div class="pl-ldc">
      ${entry && entry.comparison ? html`<div class="pl-lab">Compared with the references</div><${Compare} c=${entry.comparison} compact />` : html`<div class="pl-lab">Compared with the references</div><div class="faint pl-small">No comparison for this program yet.</div>`}
    </div>
    <div class="pl-lda">
      ${entry && entry.program && html`<button class="btn sm" onClick=${() => openProgramInBuilder(snap.spec.id, entry.program)}>Open in builder</button>`}
      ${agent && html`<a class="btn sm" href=${'#/run/' + snap.run_id + '/agents/' + agent.id}>Open ${agent.name}</a>`}
    </div>
  </div>`;
}

function ExperimentLog({ snap, cat }) {
  const items = useMemo(() => buildLog(snap), [snap]);
  const t = primaryTarget(snap.spec);
  const [open, setOpen] = useState(null);
  const [limit, setLimit] = useState(12);
  const first = useRef(null);
  if (first.current == null) first.current = new Set(items.map((x) => x.key));
  const names = Object.fromEntries(snap.agents.map((a) => [a.id, a.name]));
  const shown = items.slice(0, limit);
  const nEval = items.filter((x) => x.type === 'eval').length;
  const refusedShown = items.filter((x) => x.type === 'refused').length;
  return html`<section class="pl-sec">
    <div class="pl-sh"><h2>Experiment log</h2><p>${nEval ? 'Every scored attempt, newest first. ' + (t ? 'The score is ' + t.name + '; the change is against the best score before that attempt. ' : '') + 'Click a row for its parts and comparison.' : 'Every scored attempt appears here, newest first.'}</p></div>
    <div class="pl-log card" role="table" aria-label="Experiment log">
      <div class="pl-lhead" role="row"><span role="columnheader">#</span><span role="columnheader">Agent</span><span role="columnheader">What changed</span><span role="columnheader" class="r">${t ? t.name : 'Score'}</span><span role="columnheader" class="r">Change</span><span role="columnheader">Decision</span><span></span></div>
      ${shown.length === 0 && html`<div class="pl-lempty">Nothing is scored yet. The first row appears when the first program finishes.</div>`}
      ${shown.map((it) => {
        const isOpen = open === it.key;
        const isNew = !first.current.has(it.key);
        if (it.type === 'refused') {
          return html`<div class=${'pl-lrow refused' + (isNew ? ' enter' : '')} key=${it.key} role="row">
            <span class="num faint">-</span><span class="pl-lag">${names[it.e.agent] || it.e.agent}</span>
            <span class="pl-lwhat">${it.e.text}</span><span class="r faint">not scored</span><span class="r faint">-</span>
            <span><span class="pl-dec refused">${it.dec.label}</span></span><span></span></div>`;
        }
        const c = it.c;
        const donors = it.entry ? donorsOf(it.entry.program, cat) : [];
        const bad = c.status && c.status !== 'ok';
        return html`<${Fragment} key=${it.key}>
          <div class=${'pl-lrow ' + it.dec.k + (isOpen ? ' open' : '') + (isNew ? ' enter' : '')} role="row" tabindex="0" aria-expanded=${isOpen} onClick=${() => setOpen(isOpen ? null : it.key)} onKeyDown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(isOpen ? null : it.key); } }}>
            <span class="num pl-ln">${it.n}</span>
            <span class="pl-lag">${names[c.agent] || c.agent || ''}</span>
            <span class="pl-lwhat">${donors.length > 0 && html`<${Logos} models=${donors} max=${2} size=${16} />`}<span class="pl-lname" title=${c.name}>${c.name || (it.entry && nameOf(it.entry, cat)) || 'A program'}</span></span>
            <span class=${'r num pl-lscore' + (bad ? ' bad' : '')}>${c.score != null ? fmtPct(c.score) : '-'}</span>
            <span class=${'r num pl-ldelta ' + (it.delta == null ? '' : it.delta > 0.0005 ? 'up' : it.delta < -0.0005 ? 'down' : 'flat')} title=${it.ref != null ? 'Against ' + fmtPct(it.ref) + (it.refHost ? ', the host score (no earlier best)' : ', the best score before this attempt') : ''}>${it.delta == null ? '-' : sgn(it.delta) + ' pts'}</span>
            <span><span class=${'pl-dec ' + it.dec.k}>${it.dec.label}</span></span>
            <span class="pl-chev" aria-hidden="true"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5L8 10.5L12 6.5"/></svg></span>
          </div>
          ${isOpen && html`<div class="pl-ldwrap"><${LogDetail} it=${it} snap=${snap} cat=${cat} /></div>`}
        <//>`;
      })}
    </div>
    <div class="pl-logfoot">
      ${items.length > limit && html`<button class="btn sm" onClick=${() => setLimit(limit + 15)}>Show ${Math.min(15, items.length - limit)} more</button>`}
      ${snap.refused > refusedShown && html`<span class="faint pl-small">${snap.refused} ${snap.refused === 1 ? 'proposal was' : 'proposals were'} refused in total for going over your limits, before any GPU time was spent.</span>`}
    </div>
  </section>`;
}

/* ---------- right column ---------- */
function BestCard({ snap, cat }) {
  const t = primaryTarget(snap.spec);
  const bt = bestTarget(snap);
  const row = snap.best ? snap.leaderboard.find((r) => r.program_hash === snap.best.program_hash) : null;
  const c = snap.comparison || (row && row.comparison) || null;
  const v = verdictOf(c);
  const rows = [];
  if (bt != null) rows.push({ id: 'ours', label: 'This build', value: bt, ours: true });
  if (c) (c.rows || []).filter((r) => r.measured && r.value != null).forEach((r) => rows.push({ id: r.id, label: r.label, value: r.value, delta: r.delta, beaten: r.beaten }));
  const mx = Math.max(...rows.map((r) => r.value), 0.01);
  const [go, setGo] = useState(false);
  useEffect(() => { const r = requestAnimationFrame(() => setGo(true)); return () => cancelAnimationFrame(r); }, []);
  const comps = (row && row.components) || [];
  const vmap = Object.fromEntries(comps.map((x) => [x.id, x.verdict]));
  const hostScore = t ? ((snap.base_scores || {})[t.id] ?? t.base) : null;
  if (bt == null) {
    return html`<section class="pl-side-sec card pl-bestc"><h2 class="pl-h3">Best so far</h2><p class="pl-small faint">${ended(snap) ? snap.empty_reason || 'Nothing was scored.' : 'No valid build is scored yet. The best one shows here with its comparison against the references.'}</p></section>`;
  }
  return html`<section class="pl-side-sec card pl-bestc">
    <h2 class="pl-h3">Best so far</h2>
    <div class="pl-bscore"><span class="pl-big num"><${PopNum} text=${fmtPct(bt)} /></span><span class="pl-small faint">${t ? t.name : 'target score'}${hostScore != null ? ', ' + sgn(bt - hostScore) + ' points over the host' : ''}</span></div>
    <div class="pl-bname"><${Logos} models=${donorsOf(snap.best.program, cat)} max=${3} size=${18} /><span>${nameOf(snap.best, cat)}</span></div>
    ${v && html`<div class=${'pl-bverdict ' + v.kind}>${v.text}</div>`}
    ${comps.length > 0 && html`<div class="pl-lab">Parts and their verdicts</div><${ProgramStrip} program=${snap.best.program} cat=${cat} vmap=${vmap} />`}
    <div class="pl-lab">Against the references, ${t ? t.name : 'target'} score</div>
    <div class="pl-bars">${rows.map((r, i) => {
      const col = r.ours ? (c && c.verdict === 'failed' ? 'var(--amber)' : 'var(--green)') : /lora/.test(r.id) ? 'var(--amber)' : /random/.test(r.id) ? 'var(--text-4)' : 'var(--text-3)';
      return html`<div class=${'pl-bar' + (r.ours ? ' ours' : '')} key=${r.id}><span class="pl-bl" title=${r.label}>${r.label}</span>
        <span class="pl-bt"><i style=${{ width: go ? (r.value / mx) * 100 + '%' : '0%', background: col, transitionDelay: i * 80 + 'ms' }}></i></span>
        <b class="num pl-bv">${fmtPct(r.value)}</b></div>`;
    })}</div>
    ${rows.length < 2 && html`<p class="pl-small faint">The references appear once the best build has been compared.</p>`}
    ${snap.best.program && html`<button class="btn sm pl-bbtn" onClick=${() => openProgramInBuilder(snap.spec.id, snap.best.program)}>Open in builder</button>`}
  </section>`;
}

function ChartCard({ snap }) {
  const t = primaryTarget(snap.spec);
  if (!t) return null;
  return html`<section class="pl-side-sec card">
    <h2 class="pl-h3">Score after each eval</h2>
    <p class="pl-small faint">The line is the best ${t.name} score so far. Dots are single attempts. Dashed lines are the target and the LoRA reference.</p>
    <${LineChart} curve=${snap.curve} bench=${t} total=${snap.budget.evals_total} snap=${snap} />
  </section>`;
}

function FindingsCard({ snap, cat }) {
  const d = useMemo(() => deriveFindings(snap, cat), [snap, cat]);
  const items = d.items.filter((x) => !x.map).sort((a, b) => b.strength - a.strength);
  const top = items.slice(0, 3);
  return html`<section class="pl-side-sec card">
    <h2 class="pl-h3">What the agents have learned</h2>
    ${top.length ? html`<ul class="pl-finds">${top.map((f) => html`<li key=${f.id} class=${'pl-find ' + (f.tone || 'info')}><span class="pl-fdot"></span><span><b>${f.title}</b>${f.body ? html`<span class="pl-fbody">${oneSentence(f.body, 120)}</span>` : null}</span></li>`)}</ul>`
      : html`<p class="pl-small faint">Findings appear after the first few scored programs.</p>`}
    <a class="linkbtn" href=${'#/run/' + snap.run_id + '/findings'}>${items.length > 3 ? 'See all ' + items.length + ' findings' : 'Open Found so far'}</a>
  </section>`;
}

function AgentsCard({ snap }) {
  return html`<section class="pl-side-sec card">
    <h2 class="pl-h3">Agents</h2>
    <p class="pl-small faint">What each agent is doing now, and the best score it found. Click one for its page.</p>
    <${AgentList} snap=${snap} />
  </section>`;
}

/* ---------- how it works ---------- */
function activeStages(snap) {
  if (snap.status === 'done') return { done: 7, on: new Set() };
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
function statsOf(snap, cat) {
  const act = snap.activity.map((e) => ({ ...e, kind: kindOf(e) }));
  const lb = snap.leaderboard;
  const count = (k) => act.filter((e) => e.kind === k).length;
  const bs = (snap.spec && snap.spec.benchmarks) || [];
  return {
    evals: snap.budget.evals_used, probes: count('probe'), plans: count('plan'), maps: count('map'), held: count('heldout'),
    shared: act.filter((e) => SHARED.has(e.kind)).length,
    withGraft: lb.filter((r) => r.program && (r.program.grafts || []).length).length,
    withMask: lb.filter((r) => r.program && maskIds(r.program.masks).length).length,
    withHeal: lb.filter((r) => r.program && r.program.heal).length,
    controls: lb.filter((r) => r.vs_control != null).length,
    drifts: lb.filter((r) => r.drift).length,
    comps: lb.reduce((n, r) => n + (r.components || []).length, 0),
    benchmarks: bs.length, sourced: bs.reduce((n, b) => n + (b.reference_scores || []).filter((x) => x.sourced).length, 0),
    donors: (snap.sources || []).filter((s) => s.role === 'donor').length, libParts: cat ? cat.parts.length : null,
  };
}
function stageStats(id, s, snap) {
  const best = bestTarget(snap);
  switch (id) {
    case 'request': return [[snap.spec && snap.spec.size ? snap.spec.size.max_params_b + 'B' : '-', 'size limit']];
    case 'research': return [[s.benchmarks, 'benchmarks'], [s.sourced, 'sourced scores']];
    case 'mine': return [[s.libParts == null ? '-' : s.libParts, 'library parts'], [s.donors, 'donors used']];
    case 'swarm': return [[snap.agents.length, 'agents'], [s.shared, 'shared notes']];
    case 'build': return [[s.withGraft, 'with grafts'], [s.withHeal, 'healed']];
    case 'test': return [[s.evals, 'scored'], [s.controls, 'controls']];
    default: return [[best != null ? fmtPct(best) : '-', 'best score']];
  }
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

function HowItWorks({ snap, cat }) {
  const [open, setOpen] = useState(() => { try { return localStorage.getItem('sapien.pipeline.how') === '1'; } catch (e) { return false; } });
  const toggle = () => { const n = !open; setOpen(n); try { localStorage.setItem('sapien.pipeline.how', n ? '1' : '0'); } catch (e) { /* storage can be off */ } };
  const s = useMemo(() => statsOf(snap, cat), [snap, cat]);
  const st = activeStages(snap);
  const [sel, setSel] = useState(null);
  const order = ['research', 'mine', 'swarm', 'build', 'test'];
  const name = Object.fromEntries(STAGES.map((g) => [g.id, g.name]));
  const shown = sel != null ? sel : st.on.size ? Math.min(...st.on) : 6;
  const g = STAGES[shown];
  return html`<section class="pl-sec pl-how">
    <button class="pl-howbtn" onClick=${toggle} aria-expanded=${open}>
      <span class=${'pl-howchev' + (open ? ' open' : '')} aria-hidden="true"><svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M6 4l4 4-4 4"/></svg></span>
      <span class="pl-howt">How it works</span>
      <span class="pl-small faint">The seven stages from request to result, and the techniques the agents use.</span>
    </button>
    ${open && html`<div class="pl-howbody">
      <div class="pl-rail" role="list" aria-label="The seven stages">${STAGES.map((x, i) => {
        const on = st.on.has(i);
        const done = i < st.done || snap.status === 'done';
        return html`<button class=${'pl-rs' + (on ? ' on' : done ? ' done' : '') + (shown === i ? ' sel' : '')} key=${x.id} role="listitem" onClick=${() => setSel(i)} aria-pressed=${shown === i}>
          <span class="pl-rn"><span class="pl-rdot">${done && !on ? html`<svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg>` : i + 1}</span><i class="pl-rline"></i></span>
          <span class="pl-rname">${x.name}</span>
          <span class="pl-rsp">${x.sponsor ? html`<${SponsorMark} who=${x.sponsor} size=${13} />` : null}${on && html`<span class="livechip"></span>`}</span>
          <span class="pl-rstats">${stageStats(x.id, s, snap).map(([v, l]) => html`<span><b class="num">${v}</b> ${l}</span>`)}</span>
        </button>`;
      })}</div>
      <p class="pl-rdesc"><b>${g.name}.</b> ${g.line}</p>
      <div class="pl-tech">${order.map((sid) => {
        const list = TECHNIQUES.filter((t) => t.stage === sid);
        if (!list.length) return null;
        return html`<div class="pl-tg" key=${sid}><div class="pl-lab">${name[sid]}</div>${list.map((tq) => {
          const n = countFor(tq.id, s);
          return html`<details class="pl-tq" key=${tq.id}>
            <summary><span class="pl-tqn">${tq.name}</span>${tq.sponsor && html`<${SponsorMark} who=${tq.sponsor} size=${12} />`}<span class=${'pl-tqc num' + (n ? ' ran' : '')}>${n != null ? (n ? 'ran ' + n + (n === 1 ? ' time' : ' times') : 'not run yet') : 'runs while mining'}</span></summary>
            <p>${tq.what}</p><p class="faint">${tq.why}</p>
          </details>`;
        })}</div>`;
      })}</div>
    </div>`}
  </section>`;
}

/* ---------- the page ---------- */
export function PipelinePage({ snap: raw, id, cat }) {
  const snap = useMemo(() => ({
    ...raw, run_id: raw.run_id || id, activity: raw.activity || [], leaderboard: raw.leaderboard || [], agents: raw.agents || [], curve: raw.curve || [],
    sources: raw.sources || [], budget: raw.budget || { evals_used: 0, evals_total: 0 }, spec: raw.spec || {},
  }), [raw]);
  const log = useStepLog(snap);
  return html`<div class="rpage"><div class="pl-wrap">
    <${Objective} snap=${snap} cat=${cat} />
    <div class="pl-grid">
      <div class="pl-main">
        <${NowRunning} snap=${snap} cat=${cat} log=${log} best=${snap.best} />
        <${ExperimentLog} snap=${snap} cat=${cat} />
      </div>
      <aside class="pl-side" aria-label="Summary">
        <${BestCard} snap=${snap} cat=${cat} />
        <${ChartCard} snap=${snap} />
        <${FindingsCard} snap=${snap} cat=${cat} />
        <${AgentsCard} snap=${snap} />
      </aside>
    </div>
    <${HowItWorks} snap=${snap} cat=${cat} />
  </div></div>`;
}
