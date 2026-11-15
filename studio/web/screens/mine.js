import { html, useState, useEffect, useMemo, useRef } from '../lib.js';
import { api } from '../api.js';
import { go, toast, useInterval, ago, donorShort } from '../util.js';
import { TopBar, Icon, ErrorBox, Spinner } from '../components.js';
import { Reel, reduced, SkelReveal } from '../motion.js';
import { ModelLogo, SponsorMark, PoweredBy } from '../logos.js';
import { StatusTag, KIND_NAMES, KIND_ORDER, Pager } from '../cmp.js';
import { Brain3D } from '../brain3d.js';
import { STAGES, TECHNIQUES, techniqueById } from '../techniques.js';
import { openPart, inferTechnique } from './part-detail.js';

// The mining pages: the queue and the mined models (#/mine), one request (#/mine/req/<id>), one model in full (#/mine/model/<hf id>).

const KIND_COLOR = { block: 'var(--blue)', attn_heads: 'var(--purple)', mlp_cluster: 'var(--green)', expert: 'var(--amber)', inserted_layer: 'var(--red)' };
const kindColor = (k) => KIND_COLOR[k] || 'var(--text-3)';
const SKILLS = ['math', 'code', 'law', 'medicine', 'reasoning', 'knowledge', 'multilingual', 'science'];
const EFFORTS = [
  ['smoke', 'Quick', 'Scan fewer candidates. Good for a first look.', 0.6, 4],
  ['medium', 'Standard', 'The usual depth: every layer, five random draws per part.', 2.5, 12],
  ['full', 'Deep', 'More candidates and held-out checks.', 6, 30],
];
// the mining stages, with the techniques the agents use in each (ids in techniques.js)
const MINE_STAGES = [
  ['fetch', 'Fetch weights', 'modal', []],
  ['read', 'Read the weights', 'openai', ['weight-read']],
  ['scan', 'Scan layers', 'modal', ['activation-scan', 'attribution', 'saliency', 'skill-map', 'weight-delta']],
  ['ablate', 'Switch parts off', 'modal', ['ablation']],
  ['gate', 'Random-part gate', 'modal', ['gate']],
  ['publish', 'Publish to the library', 'pastdev', ['shared-memory']],
];
const STAGE_NOTE = {
  fetch: 'The model is pulled from the model hub onto a Modal volume, so the later stages read it from fast disk.',
  read: 'An agent reads weight statistics in a locked sandbox and guesses where skills may live. It checks shared memory first.',
  scan: 'A beam moves through the layers. Units that fire more on skill text than on general text become candidates.',
  ablate: 'Each candidate is switched off and the model is scored on every skill at once.',
  gate: 'A candidate stays only if it beats random parts of the same size. Most do not.',
  publish: 'Kept parts and their evidence go to the library, and a note goes to the agents\' shared memory.',
};
const AGENT_SPONSOR = { Fetcher: 'modal', Reader: 'openai', Scanner: 'modal', Ablator: 'modal', Gate: 'modal', Memory: 'pastdev', Publisher: 'pastdev' };
const KIND_LINE = { info: 'var(--text-3)', guess: 'var(--blue)', scan: 'var(--blue)', ablate: 'var(--amber)', keep: 'var(--green)', reject: 'var(--red)', memory: 'var(--purple)' };

const bText = (b) => (b == null ? '-' : b >= 1 ? +(+b).toFixed(1) + 'B' : Math.round(b * 1000) + 'M');
const skillName = (s) => String(s || '').replace(/_/g, ' ');
const isLive = (r) => r && (r.status === 'running' || r.status === 'queued' || r.status === 'starting');
const estimate = (params_b, effort) => { const e = EFFORTS.find((x) => x[0] === effort) || EFFORTS[1]; const gpu = Math.round((params_b || 8) * e[3] + e[4]); return { gpu_min: gpu, cost_usd: +(gpu / 60 * ((params_b || 8) > 10 ? 4 : 2.5)).toFixed(2) }; };
const totalOf = (m) => (typeof m.parts === 'number' ? m.parts : Array.isArray(m.parts) ? m.parts.length : m.parts_count || 0);
const kindsOf = (m) => m.kinds || m.counts || (Array.isArray(m.parts) ? m.parts.reduce((a, p) => { a[p.kind] = (a[p.kind] || 0) + 1; return a; }, {}) : {});
const skillsOf = (m) => (m.top_skills && m.top_skills.length ? m.top_skills : Object.entries(m.skills || {}).sort((a, b) => b[1] - a[1]).slice(0, 3).map((x) => x[0]));
const timeOf = (iso) => { const t = Date.parse(iso); return isNaN(t) ? Date.now() : t; };

/* ---------- progress, derived from the request itself ---------- */

const STAGE_WEIGHT = { fetch: 0.05, read: 0.12, scan: 0.4, ablate: 0.25, gate: 0.15, publish: 0.03 };
const EFFORT_NAME = { smoke: 'Quick', medium: 'Standard', full: 'Deep', quick: 'Quick', standard: 'Standard', deep: 'Deep' };
const effortName = (e) => EFFORT_NAME[e] || 'Standard';
const stOf = (r, id) => ((r.stages || []).find((s) => s.id === id)) || {};
const stStatus = (s) => (s.status === 'done' || s.status === 'running' ? s.status : 'pending');
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const pct = (x) => Math.round(clamp01(x) * 100);
const cnt = (r) => r.counters || {};

/** Per-stage fraction (0 to 1) or null when the stage runs but reports no fraction. */
function stageFrac(r, id) {
  const s = stOf(r, id);
  const st = stStatus(s);
  if (st === 'done') return 1;
  if (st === 'pending') return 0;
  if (typeof s.progress === 'number') return clamp01(s.progress);
  const c = cnt(r);
  if (id === 'scan' && c.layers_total) return clamp01((c.layers_scanned || 0) / c.layers_total);
  if (id === 'ablate' && c.candidates) return clamp01((c.ablations || 0) / c.candidates);
  if (id === 'gate' && c.ablations) return clamp01(((c.kept || 0) + (c.rejected || 0)) / c.ablations);
  return null;
}

/** Overall progress, 0 to 1. A queued request is 0 and a done request is 1. */
function reqProgress(r) {
  if (r.status === 'done') return 1;
  if (r.status === 'queued') return 0;
  if (typeof r.progress === 'number') return clamp01(r.progress);
  let sum = 0;
  for (const id of Object.keys(STAGE_WEIGHT)) { const f = stageFrac(r, id); sum += STAGE_WEIGHT[id] * (f == null ? 0.5 : f); }
  return clamp01(sum);
}

const curStageId = (r) => { const s = (r.stages || []).find((x) => x.status === 'running'); return s ? s.id : null; };
const stageName = (id) => (MINE_STAGES.find((s) => s[0] === id) || [])[1] || '';

/** The count line under each stage, in plain words. */
function stageLine(r, id) {
  const c = cnt(r);
  const st = stStatus(stOf(r, id));
  const n = (x) => (x == null ? 0 : x);
  if (st === 'pending') {
    if (id === 'scan') return c.layers_total ? '0 of ' + c.layers_total + ' layers' : 'Not started';
    return 'Not started';
  }
  if (id === 'fetch') return st === 'done' ? 'Weights are on disk' : 'Downloading the weights';
  if (id === 'read') return st === 'done' ? 'Guesses made' : 'Reading weight statistics';
  if (id === 'scan') return n(c.layers_scanned) + (c.layers_total ? ' of ' + c.layers_total : '') + ' layers, ' + n(c.candidates) + ' named';
  if (id === 'ablate') return n(c.ablations) + ' of ' + n(c.candidates) + ' switched off';
  if (id === 'gate') return n(c.kept) + ' kept, ' + n(c.rejected) + ' rejected';
  return st === 'done' ? n(c.kept) + ' parts published' : 'Writing to the library';
}

/** One sentence that says what is happening now. */
function narrate(r) {
  const c = cnt(r);
  const n = (x) => (x == null ? 0 : x);
  if (r.status === 'queued') return 'Waiting for a free GPU. Nothing has run yet.';
  if (r.status === 'failed' || r.status === 'error') return 'This request stopped with an error. ' + (r.error || 'The log may say why.');
  if (r.status === 'done') return 'Mining is finished. ' + n(c.kept) + ' of ' + n(c.candidates) + ' candidates beat random parts and went to the library.';
  const id = curStageId(r);
  if (id === 'fetch') return 'Downloading the weights to a Modal volume.';
  if (id === 'read') return 'An agent is reading weight statistics to guess where the skills live.';
  if (id === 'scan') return 'Scanning layer ' + Math.min(n(c.layers_scanned) + 1, c.layers_total || 9999) + (c.layers_total ? ' of ' + c.layers_total : '') + '. ' + n(c.candidates) + ' candidates named so far.';
  if (id === 'ablate') return 'Switching off candidate ' + Math.min(n(c.ablations) + 1, Math.max(n(c.candidates), 1)) + ' of ' + n(c.candidates) + ' and scoring the model without it.';
  if (id === 'gate') return 'Comparing each candidate with random parts of the same size. ' + n(c.kept) + ' kept, ' + n(c.rejected) + ' rejected so far.';
  if (id === 'publish') return 'Writing the kept parts and their evidence to the library.';
  return 'Starting up.';
}

/** The stage in words for a queue row. */
function stageWords(r) {
  const c = cnt(r);
  if (r.status === 'queued') return 'Queued, waiting for a free GPU';
  if (r.status === 'done') return 'Done: ' + (c.kept != null ? c.kept + ' parts kept' : 'finished') + (c.candidates ? ' of ' + c.candidates + ' candidates' : '');
  if (r.status === 'failed' || r.status === 'error') return 'Stopped with an error';
  const id = curStageId(r);
  if (!id) return 'Starting up';
  return stageName(id) + ': ' + stageLine(r, id).toLowerCase();
}

const dur = (ms) => {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 90) return s + ' sec';
  const m = Math.round(s / 60);
  return m < 90 ? m + ' min' : (s / 3600).toFixed(1) + ' h';
};
/** Elapsed and time-left lines. The estimate needs 5 percent of real progress first. */
function timing(r, now) {
  const p = reqProgress(r);
  if (r.status === 'queued') return { a: 'Not started', b: r.estimate ? 'Estimate once it starts: about ' + r.estimate.gpu_min + ' GPU min' : '' };
  const t0 = r.started ? timeOf(r.started) : null;
  if (r.status === 'done') return { a: t0 && r.finished ? 'Took ' + dur(timeOf(r.finished) - t0) : 'Finished', b: '' };
  if (r.status !== 'running' || !t0) return { a: '', b: '' };
  const el = Math.max(0, now - t0);
  if (p < 0.05) return { a: 'Running for ' + dur(el), b: 'Time left: working it out' };
  return { a: 'Running for ' + dur(el), b: 'About ' + dur(el * (1 - p) / p) + ' left' };
}

const modelSpecs = (r) => [r.params_b != null ? bText(r.params_b) + ' parameters' : null, r.layers ? r.layers + ' layers' : null, r.experts ? r.experts + ' experts per layer' : r.experts === 0 ? 'dense' : null].filter(Boolean);

function Bar({ value, tone, indeterminate }) {
  return html`<span class=${'mn-bar' + (tone ? ' ' + tone : '') + (indeterminate ? ' ind' : '')} role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${indeterminate ? undefined : pct(value)}><i style=${{ width: indeterminate ? '35%' : pct(value) + '%' }}></i></span>`;
}

function useNow(live) {
  const [now, setNow] = useState(Date.now());
  useInterval(() => setNow(Date.now()), live ? 1000 : null);
  return now;
}

/* ---------- small pieces ---------- */

function KindBar({ kinds, height = 6 }) {
  const total = Object.values(kinds).reduce((a, b) => a + b, 0) || 1;
  const order = [...KIND_ORDER, ...Object.keys(kinds).filter((k) => !KIND_ORDER.includes(k))].filter((k) => kinds[k]);
  return html`<div class="mn-kindbar" style=${{ height: height + 'px' }} role="img" aria-label=${order.map((k) => (KIND_NAMES[k] || k) + ' ' + kinds[k]).join(', ')}>${order.map((k, i) => html`<span key=${k} style=${{ flex: kinds[k] / total, background: kindColor(k), '--i': i }} title=${(KIND_NAMES[k] || k) + ': ' + kinds[k]}></span>`)}</div>`;
}

function KindLegend({ kinds }) {
  const order = [...KIND_ORDER, ...Object.keys(kinds).filter((k) => !KIND_ORDER.includes(k))].filter((k) => kinds[k]);
  return html`<div class="mn-legend">${order.map((k) => html`<span key=${k}><i style=${{ background: kindColor(k) }}></i>${KIND_NAMES[k] || k} <b class="num">${kinds[k]}</b></span>`)}</div>`;
}

/** A row of the six stages: sponsor mark, a fill, and the stage name. */
function StageStrip({ stages, onPick, active, big }) {
  return html`<ol class=${'mn-strip' + (big ? ' big' : '')}>${MINE_STAGES.map(([id, name, sp], i) => {
    const s = (stages || []).find((x) => x.id === id) || { status: 'pending' };
    const fill = s.status === 'done' ? 100 : s.status === 'running' ? Math.max(8, Math.round((s.progress != null ? s.progress : 0.5) * 100)) : 0;
    const el = html`<span class=${'mn-seg ' + s.status + (active === id ? ' sel' : '')}>
      <span class="mn-seg-fill" style=${{ width: fill + '%' }}></span>
      <span class="mn-seg-in"><${SponsorMark} who=${s.sponsor || sp} size=${big ? 15 : 12} /><span class="mn-seg-n ell">${s.name || name}</span>${s.status === 'done' && html`<${Icon} n="check" size=${12} sw=${2.4} />`}</span>
    </span>`;
    return html`<li key=${id} style=${{ '--i': i }}>${onPick ? html`<button class="mn-segbtn" onClick=${() => onPick(id)} aria-pressed=${active === id ? 'true' : 'false'}>${el}</button>` : el}</li>`;
  })}</ol>`;
}

function StatusChip({ status }) {
  if (status === 'running') return html`<span class="livechip">Mining</span>`;
  if (status === 'queued') return html`<span class="tag">Queued</span>`;
  if (status === 'done') return html`<span class="tag g">Done</span>`;
  if (status === 'error' || status === 'failed') return html`<span class="tag r">Failed</span>`;
  return html`<span class="tag">${status || 'Unknown'}</span>`;
}

function Count({ n }) { return html`<${Reel} value=${n == null ? 0 : n} />`; }

/* ---------- #/mine: submit panel, queue, mined models ---------- */

/** The preview scene: its own timer, so the form does not re-render on every beam step, and it idles when the tab is hidden. */
function BeamPreview({ layers, label }) {
  const [beam, setBeam] = useState(0);
  useEffect(() => {
    if (reduced()) return undefined;
    const id = setInterval(() => { if (!document.hidden) setBeam((b) => (b + 1) % layers); }, 240);
    return () => clearInterval(id);
  }, [layers]);
  return html`<${Brain3D} height=${330} layers=${layers} mode="donor" scan=${{ layer: beam % layers, total: layers, found: [] }} highlight=${beam % layers} label=${label} />`;
}

function SubmitPanel({ onSubmitted }) {
  const [model, setModel] = useState('');
  const [look, setLook] = useState(null);
  const [looking, setLooking] = useState(false);
  const [skills, setSkills] = useState(['math', 'code']);
  const [effort, setEffort] = useState('medium');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const layers = (look && look.valid !== false && look.layers) || 28;
  useEffect(() => {
    const m = model.trim();
    if (!m) { setLook(null); setLooking(false); return; }
    setLooking(true);
    let dead = false;
    const t = setTimeout(() => {
      api.mineLookup(m).then((r) => { if (!dead) { setLook(r); setLooking(false); } }).catch((e) => { if (!dead) { setLook({ valid: false, error: e.message }); setLooking(false); } });
    }, 350);
    return () => { dead = true; clearTimeout(t); };
  }, [model]);
  const ok = look && look.valid !== false && look.model;
  const est = ok ? estimate(look.params_b, effort) : null;
  const toggle = (s) => setSkills(skills.includes(s) ? skills.filter((x) => x !== s) : [...skills, s]);
  const submit = async () => {
    if (!ok || busy) return;
    setBusy(true);
    try {
      const r = await api.mineSubmit({ model: look.model, skills, effort, note: note.trim() });
      toast('Mining started for ' + String(r.model || look.model).split('/').pop() + '.', 'ok');
      onSubmitted(r);
      go('/mine/req/' + r.id);
    } catch (e) { toast('Could not start mining: ' + e.message, 'err'); setBusy(false); }
  };
  const specs = ok ? [['Parameters', bText(look.params_b)], ['Layers', look.layers], ['Experts', look.experts ? look.experts : look.kind === 'moe' ? 'mixture' : 'dense'], ['License', look.license]] : [];
  return html`<section class="card mn-submit stage3d">
    <div class="mn-submit-l">
      <h2 class="mn-h">Pick a model to mine</h2>
      <p class="faint mn-sub">Agents read its weights, find the parts that hold a skill, and keep only the parts that beat random ones.</p>
      <label class="mn-field"><span>Model id</span>
        <span class="mn-inp"><${Icon} n="search" size=${16} /><input placeholder="Qwen/Qwen3-30B-A3B-Instruct-2507" value=${model} onInput=${(e) => setModel(e.target.value)} spellcheck="false" autocomplete="off" aria-label="Model id" />${looking && html`<${Spinner} />`}</span></label>
      ${look && look.valid === false && model.trim() && html`<div class="mn-bad reveal">${look.error || 'That does not look like a model id.'}</div>`}
      ${ok && html`<div class="mn-look reveal" key=${look.model}>
        <${ModelLogo} model=${look.model} size=${34} />
        <div class="mn-look-main">
          <div class="mn-look-n">${look.model}</div>
          <div class="mn-specs">${specs.map(([k, v], i) => html`<span class="reveal" style=${{ '--i': i }} key=${k}><em>${k}</em><b class="num">${v == null ? '-' : v}</b></span>`)}</div>
        </div>
        ${look.in_library ? html`<span class="tag g" title=${look.parts + ' parts are already in the library'}>In library, ${look.parts} parts</span>` : html`<span class="tag w">Not in library</span>`}
      </div>`}
      <div class="mn-field"><span>Skills to look for</span>
        <div class="mn-chips">${SKILLS.map((s) => html`<button key=${s} class=${'fchip2' + (skills.includes(s) ? ' on' : '')} aria-pressed=${skills.includes(s) ? 'true' : 'false'} onClick=${() => toggle(s)}>${s}</button>`)}</div></div>
      <div class="mn-field"><span>Effort</span>
        <div class="mn-efforts" role="radiogroup" aria-label="Effort">${EFFORTS.map(([id, name, line]) => { const e = estimate(ok ? look.params_b : 8, id); return html`<button key=${id} role="radio" aria-checked=${effort === id ? 'true' : 'false'} class=${'mn-eff' + (effort === id ? ' on' : '')} onClick=${() => setEffort(id)}>
          <b>${name}</b><small>${line}</small><span class="num">about ${e.gpu_min} GPU min, $${e.cost_usd.toFixed(2)}</span></button>`; })}</div></div>
      <label class="mn-field"><span>Note for the agents <i class="faint">(optional)</i></span>
        <input class="mn-note" placeholder="For example: focus on the later layers" value=${note} onInput=${(e) => setNote(e.target.value)} /></label>
      <div class="mn-actions">
        <button class="btn primary" disabled=${!ok || busy || !skills.length} onClick=${submit}>${busy ? 'Starting' : 'Start mining'}</button>
        <span class="faint num">${est ? 'Estimate: about ' + est.gpu_min + ' GPU minutes, $' + est.cost_usd.toFixed(2) : 'Enter a model id to see the estimate.'}</span>
        <span class="grow"></span>
        <${PoweredBy} who="modal" label="GPUs on" size=${14} />
      </div>
    </div>
    <div class="mn-submit-r">
      <${BeamPreview} layers=${layers} label=${ok ? 'Preview of the scan of ' + donorShort(look.model) + '. The beam moves up one layer at a time.' : 'Preview. Enter a model id and the scene shows its layers.'} />
      <div class="stage-hud"><span>${ok ? look.layers + ' layers' : 'Preview'}</span><span class="livechip">Scan preview</span></div>
    </div>
  </section>`;
}

function ReqRow({ r, i }) {
  const p = reqProgress(r);
  const failed = r.status === 'failed' || r.status === 'error';
  return html`<a class=${'mn-req reveal' + (r.status === 'running' ? ' working' : '')} style=${{ '--i': i }} href=${'#/mine/req/' + r.id}>
    <${ModelLogo} model=${r.model} size=${30} />
    <div class="mn-req-main">
      <div class="mn-req-top"><span class="mn-req-n ell">${r.model}</span><${StatusChip} status=${r.status} /></div>
      <div class="mn-req-bar"><${Bar} value=${p} tone=${r.status === 'done' ? 'g' : failed ? 'r' : ''} /><b class="num">${pct(p)}%</b></div>
      <div class="mn-req-s ell"><span class=${r.status === 'running' ? 'mn-now' : 'faint'}>${stageWords(r)}</span><span class="faint"> · ${(r.skills || []).map(skillName).join(', ') || 'all skills'} · ${effortName(r.effort)} · ${r.status === 'queued' ? 'queued ' : 'started '}${ago(timeOf(r.created))}</span></div>
    </div>
    <${Icon} n="chevron" size=${16} />
  </a>`;
}

function ModelCard({ m, i }) {
  const kinds = kindsOf(m);
  const total = totalOf(m);
  return html`<a class="card mn-card reveal" style=${{ '--i': Math.min(i, 14) }} href=${'#/mine/model/' + m.model}>
    <div class="mn-card-top"><${ModelLogo} model=${m.model} size=${30} /><div class="mn-card-t"><div class="mn-card-n ell">${m.short || donorShort(m.model)}</div><div class="faint ell" style="font-size:12px">${m.model.split('/')[0]}</div></div>
      ${m.kind === 'moe' && html`<span class="tag b">MoE</span>`}</div>
    <div class="mn-card-specs num"><span>${bText(m.params_b)}</span><span>${m.layers || '-'} layers</span>${m.experts ? html`<span>${m.experts} experts</span>` : null}</div>
    <${KindBar} kinds=${kinds} />
    <div class="mn-card-foot"><span class="num"><b>${total.toLocaleString('en-US')}</b> parts</span><span class="mn-skills">${skillsOf(m).map((s) => html`<span class="tag" key=${s}>${skillName(s)}</span>`)}</span></div>
  </a>`;
}

const SORTS = [['parts', 'Most parts'], ['size', 'Largest'], ['name', 'Name']];
const CARD_PAGE = 18;

function MineHome() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [q, setQ] = useState('');
  const [sort, setSort] = useState('parts');
  const [page, setPage] = useState(1);
  const top = useRef(null);
  const load = () => api.mine().then((d) => { setData(d); setErr(null); }).catch(setErr);
  useEffect(() => { load(); }, []);
  const live = data && (data.requests || []).some(isLive);
  useInterval(load, live ? 2500 : null);
  useEffect(() => { setPage(1); }, [q, sort]);
  const models = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const l = ((data && data.models) || []).filter((m) => !ql || (m.model + ' ' + skillsOf(m).join(' ')).toLowerCase().includes(ql));
    l.sort((a, b) => (sort === 'name' ? a.model.localeCompare(b.model) : sort === 'size' ? (b.params_b || 0) - (a.params_b || 0) : totalOf(b) - totalOf(a)));
    return l;
  }, [data, q, sort]);
  if (err && !data) return html`<${ErrorBox} error=${err} onRetry=${load} />`;
  const reqs = ((data && data.requests) || []).slice().sort((a, b) => (isLive(b) ? 1 : 0) - (isLive(a) ? 1 : 0) || timeOf(b.created) - timeOf(a.created));
  const allParts = ((data && data.models) || []).reduce((a, m) => a + totalOf(m), 0);
  const pages = Math.max(1, Math.ceil(models.length / CARD_PAGE));
  const at = Math.min(page, pages);
  return html`<div class="doc wide mn">
    <div class="ph"><div><h1 class="page-title">Mine a model</h1>
      <p class="lede">Open models hold skills in small parts. Agents find those parts, test them against random ones, and add the survivors to the library.</p></div>
      <div class="mn-powered"><${PoweredBy} who="modal" label="GPUs and sandboxes on" /><${PoweredBy} who="openai" label="Reading agents by" /><${PoweredBy} who="pastdev" label="Memory on" /></div></div>
    <div class="mn-hero num">
      <div><b><${Count} n=${(data && data.models || []).length} /></b><span>models mined</span></div>
      <div><b><${Count} n=${allParts} /></b><span>parts in the library</span></div>
      <div><b><${Count} n=${reqs.filter(isLive).length} /></b><span>requests running or queued</span></div>
    </div>
    <${SubmitPanel} onSubmitted=${load} />
    <section class="sect"><h2 class="sec"><span>Requests</span><span class="faint cnt">${reqs.filter((r) => r.status === 'running').length} running, ${reqs.filter((r) => r.status === 'queued').length} waiting, ${reqs.filter((r) => r.status === 'done').length} done</span></h2>
      ${!data ? html`<div class="skeleton" style="height:96px"></div>` : reqs.length === 0 ? html`<div class="empty">No requests yet. Pick a model above to start one.</div>` : html`<div class="card mn-queue">${reqs.map((r, i) => html`<${ReqRow} key=${r.id} r=${r} i=${i} />`)}</div>`}
    </section>
    <section class="sect"><div ref=${top}></div><h2 class="sec"><span>Mined models</span><span class="faint cnt">${models.length}</span></h2>
      <div class="mn-toolbar">
        <div class="searchbox mn-q"><${Icon} n="search" size=${16} /><input placeholder="Search models or skills" aria-label="Search mined models" value=${q} onInput=${(e) => setQ(e.target.value)} /></div>
        <div class="mn-sorts" role="group" aria-label="Sort">${SORTS.map(([id, name]) => html`<button key=${id} class=${'fchip2' + (sort === id ? ' on' : '')} aria-pressed=${sort === id ? 'true' : 'false'} onClick=${() => setSort(id)}>${name}</button>`)}</div>
      </div>
      ${!data ? html`<div class="grid3"><div class="skeleton" style="height:140px"></div><div class="skeleton" style="height:140px"></div><div class="skeleton" style="height:140px"></div></div>`
        : html`<div class="mn-grid">${models.slice((at - 1) * CARD_PAGE, at * CARD_PAGE).map((m, i) => html`<${ModelCard} key=${m.model} m=${m} i=${i} />`)}</div>
          ${models.length === 0 && html`<div class="empty">No mined model matches.</div>`}
          <${Pager} page=${at} total=${models.length} size=${CARD_PAGE} onPage=${(n) => { setPage(n); if (top.current) top.current.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} label="Model pages" />`}
    </section>
  </div>`;
}

/* ---------- #/mine/req/<id>: one request, live ---------- */

const mmss = (ms) => { const s = Math.max(0, Math.round(ms / 1000)); return String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0'); };
const STATE = { kept: ['g', 'Kept'], rejected: ['r', 'Rejected'], testing: ['b', 'Testing'], candidate: ['', 'Named'] };
const SPONSOR_NAME = { modal: 'Modal', openai: 'OpenAI', pastdev: 'past.dev', exa: 'Exa' };

function LogView({ log }) {
  const box = useRef(null);
  const stick = useRef(true);
  const onScroll = () => { const b = box.current; if (b) stick.current = b.scrollHeight - b.scrollTop - b.clientHeight < 40; };
  useEffect(() => { const b = box.current; if (b && stick.current) b.scrollTop = b.scrollHeight; }, [log.length]);
  const tms = (t) => (typeof t === 'number' ? t : timeOf(t));
  const t0 = log.length ? tms(log[0].t) : 0;
  const shown = log.slice(-120);
  return html`<div class="mn-log" ref=${box} onScroll=${onScroll} role="log" aria-live="off">
    ${shown.length === 0 && html`<div class="faint">Waiting for the first agent note.</div>`}
    ${shown.map((l, i) => html`<div class="mn-ll" key=${l.t + ':' + i + String(l.text || '').slice(0, 12)}>
      <span class="mn-lt num">${mmss(tms(l.t) - t0)}</span>
      <span class="mn-la"><${SponsorMark} who=${AGENT_SPONSOR[l.agent]} size=${12} />${l.agent}</span>
      <span class="mn-lx" style=${{ '--k': KIND_LINE[l.kind] || 'var(--text-3)' }}>${l.text}</span>
    </div>`)}
  </div>`;
}

/** The six stages as a stepper. Each step has its status, a bar, and its counts. */
function Stepper({ r, active, onPick }) {
  return html`<ol class="mn-steps">${MINE_STAGES.map(([id, name, sp], i) => {
    const s = stOf(r, id);
    const st = stStatus(s);
    const f = stageFrac(r, id);
    return html`<li key=${id} style=${{ '--i': i }}><button class=${'mn-step ' + st + (active === id ? ' sel' : '')} onClick=${() => onPick(id)} aria-pressed=${active === id ? 'true' : 'false'}>
      <span class="mn-step-top"><span class="mn-step-no num">${st === 'done' ? html`<${Icon} n="check" size=${12} sw=${2.6} />` : i + 1}</span><span class="mn-step-sp"><${SponsorMark} who=${s.sponsor || sp} size=${14} /><em>${SPONSOR_NAME[s.sponsor || sp] || ''}</em></span></span>
      <span class="mn-step-n">${s.name || name}</span>
      <span class="mn-step-bar"><${Bar} value=${f == null ? 0 : f} tone=${st === 'done' ? 'g' : ''} indeterminate=${st === 'running' && f == null} />${st === 'running' && f != null && html`<b class="num">${pct(f)}%</b>`}</span>
      <span class="mn-step-c num">${stageLine(r, id)}</span>
    </button></li>`;
  })}</ol>`;
}

const PART_CAP = 60;

function Candidates({ found, live }) {
  const [tab, setTab] = useState('all');
  const [more, setMore] = useState(false);
  const group = (f) => (f.verdict === 'kept' ? 'kept' : f.verdict === 'rejected' ? 'rejected' : 'testing');
  const n = { all: found.length, testing: found.filter((f) => group(f) === 'testing').length, kept: found.filter((f) => group(f) === 'kept').length, rejected: found.filter((f) => group(f) === 'rejected').length };
  const rows = found.filter((f) => tab === 'all' || group(f) === tab).slice().reverse();
  const shown = more ? rows : rows.slice(0, PART_CAP);
  const tabs = [['all', 'All'], ['testing', 'Testing'], ['kept', 'Kept'], ['rejected', 'Rejected by the gate']];
  return html`<section class="sect"><h2 class="sec"><span>Candidate parts</span><span class="faint cnt">${found.length} named${live ? ', newest first' : ''}</span></h2>
    <p class="faint mn-cap">A candidate is a group of units that fires more on the skill than on general text. Each is switched off to see how much the skill drops, then compared with random parts of the same size.</p>
    <div class="mn-tabs" role="tablist">${tabs.map(([k, name]) => html`<button key=${k} role="tab" aria-selected=${tab === k ? 'true' : 'false'} class=${'fchip2' + (tab === k ? ' on' : '')} onClick=${() => setTab(k)}>${name}<span class="n">${n[k]}</span></button>`)}</div>
    <div class="card mn-table mn-cands"><div class="mn-tscroll"><table>
      <thead><tr><th>State</th><th>Part</th><th class="num">Layer</th><th>Kind</th><th>Skill</th><th class="num" title="How far the skill score falls when the part is switched off">Score drop</th><th class="num" title="Standard deviations above random parts of the same size. 3 or more passes the gate.">z-score</th></tr></thead>
      <tbody>${shown.map((f, i) => { const [tone, name] = STATE[f.verdict] || ['', f.verdict || 'Named']; const open = f.id ? () => openPart(f.id) : null;
        return html`<tr key=${(f.id || f.label) + ':' + f.layer} class=${'mn-tr' + (open ? '' : ' static') + (f.verdict === 'testing' ? ' working' : '')} tabindex=${open ? 0 : undefined} onClick=${open} onKeyDown=${open && ((e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } })}>
          <td><span class=${'tag ' + tone}>${name}</span></td>
          <td class="mn-tl"><span class="ell">${f.label}</span></td>
          <td class="num">${f.layer != null ? 'L' + f.layer : '-'}</td>
          <td><span class="mn-kk"><i class="mn-kd" style=${{ background: kindColor(f.kind) }}></i>${(KIND_NAMES[f.kind] || f.kind || '-').replace(/s$/, '')}</span></td>
          <td>${f.skill ? skillName(f.skill) : html`<span class="faint">-</span>`}</td>
          <td class="num">${f.drop != null ? '-' + (+f.drop).toFixed(1) + ' pts' : html`<span class="faint">not tested</span>`}</td>
          <td class="num">${f.z != null ? html`<span class=${f.z >= 3 ? 'up' : 'down'}>${(+f.z).toFixed(1)}</span>` : html`<span class="faint">-</span>`}</td></tr>`; })}</tbody></table>
      ${rows.length === 0 && html`<div class="faint mn-none">${found.length === 0 ? (live ? 'The scan has not named a candidate yet.' : 'No candidates were named.') : 'Nothing in this group yet.'}</div>`}</div>
      ${rows.length > PART_CAP && !more && html`<button class="mn-more" onClick=${() => setMore(true)}>Show all ${rows.length}</button>`}</div>
  </section>`;
}

function ReqView({ id }) {
  const [r, setR] = useState(null);
  const [err, setErr] = useState(null);
  const [pick, setPick] = useState(null);
  const [pulse, setPulse] = useState(null);
  const [showLog, setShowLog] = useState(false);
  const prev = useRef(null);
  const take = (n) => {
    const p = prev.current;
    if (p && n.counters) {
      const c = n.counters;
      if ((c.rejected || 0) > (p.rejected || 0)) { const f = [...(n.found || [])].reverse().find((x) => x.verdict === 'rejected'); setPulse({ kind: 'score', at: f ? f.layer : n.layer, partId: f && f.id, ok: false, key: 'r' + c.rejected }); }
      else if ((c.kept || 0) > (p.kept || 0)) { const f = [...(n.found || [])].reverse().find((x) => x.verdict === 'kept'); setPulse({ kind: 'score', at: f ? f.layer : n.layer, partId: f && f.id, ok: true, key: 'k' + c.kept }); }
      else if (n.layer !== (p.layer)) setPulse({ kind: 'probe', at: n.layer, ok: null, key: 'p' + n.layer });
    }
    prev.current = { ...(n.counters || {}), layer: n.layer };
    setR(n);
  };
  const load = () => api.mineReq(id).then((n) => { take(n); setErr(null); }).catch(setErr);
  useEffect(() => { load(); }, [id]);
  const live = r && isLive(r);
  useInterval(load, live ? 1200 : null);
  const now = useNow(live);
  if (err && !r) return html`<${ErrorBox} error=${err} onRetry=${load} />`;
  if (!r) return html`<div class="doc wide"><div class="skeleton" style="height:360px"></div></div>`;
  const found = r.found || [];
  const c = r.counters || {};
  const layers = r.layers || c.layers_total || 24;
  const cur = curStageId(r);
  const stageId = pick || cur || (r.status === 'done' ? 'publish' : 'fetch');
  const stageDef = MINE_STAGES.find((s) => s[0] === stageId);
  const kept = found.filter((f) => f.verdict === 'kept').length;
  const scanLayer = r.layer != null ? r.layer : c.layers_scanned || 0;
  const p = reqProgress(r);
  const tm = timing(r, now);
  const story = narrate(r);
  const specs = modelSpecs(r);
  const failed = r.status === 'failed' || r.status === 'error';
  const doneStages = (r.stages || []).filter((s) => s.status === 'done').length;
  return html`<div class="doc wide mn">
    <a class="mn-back lnk" href="#/mine"><${Icon} n="chevron" size=${13} />Mine a model</a>
    <div class="ph mn-rhead"><div>
      <div class="mn-rtitle"><${ModelLogo} model=${r.model} size=${42} /><h1 class="page-title ell">${donorShort(r.model)}</h1><${StatusChip} status=${r.status} /></div>
      <p class="lede">${[r.model].concat(specs).join(' · ')}</p>
      <p class="faint mn-ask">Looking for ${(r.skills || []).map(skillName).join(', ') || 'every skill'}. ${effortName(r.effort)} effort${r.estimate ? ', estimated at about ' + r.estimate.gpu_min + ' GPU minutes and $' + (+r.estimate.cost_usd).toFixed(2) : ''}.</p></div>
      <div class="mn-powered"><${PoweredBy} who="modal" label="GPUs on" /><${PoweredBy} who="openai" label="Reading agent by" /><${PoweredBy} who="pastdev" label="Memory on" /></div></div>

    <section class=${'card mn-over' + (failed ? ' bad' : '')}>
      <div class="mn-pct"><b class="num"><${Count} n=${pct(p)} /><small>%</small></b><span>${r.status === 'done' ? 'complete' : 'done'}</span></div>
      <div class="mn-over-main">
        <${Bar} value=${p} tone=${r.status === 'done' ? 'g' : failed ? 'r' : ''} />
        <div class="mn-story">${story}</div>
        ${r.status === 'done' && html`<div><a class="btn sm primary" href=${'#/mine/model/' + r.model}>View parts in the library</a></div>`}
        <div class="mn-over-meta faint num"><span>${doneStages} of ${MINE_STAGES.length} stages done</span>${tm.a && html`<span>${tm.a}</span>`}${tm.b && html`<span>${tm.b}</span>`}${c.candidates != null && html`<span>${c.candidates} candidates, ${kept} kept</span>`}</div>
      </div>
    </section>

    <section class="sect"><h2 class="sec"><span>Stages</span><span class="faint cnt">click a stage to read what it does</span></h2>
      <${Stepper} r=${r} active=${stageId} onPick=${(s) => setPick(s === pick ? null : s)} />
      <div class="card mn-stagebox" key=${stageId}>
        <div class="mn-stagehead"><${SponsorMark} who=${stageDef[2]} size=${16} /><b>${stageDef[1]}</b><span class="faint">${STAGE_NOTE[stageId]}</span></div>
        ${stageDef[3].length > 0 && html`<details class="mn-tdet"><summary>Techniques used here (${stageDef[3].length})</summary>
          <div class="mn-techs">${stageDef[3].map((tid) => { const t = techniqueById(tid); return t && html`<div class="mn-tech" key=${tid}><div class="mn-tech-top"><b>${t.name}</b><${SponsorMark} who=${t.sponsor} size=${13} /></div><div class="mn-tech-what">${t.what}</div><div class="faint mn-tech-why">${t.why}</div></div>`; })}</div></details>`}
      </div></section>

    <section class="sect"><h2 class="sec"><span>The donor scan</span><span class="faint cnt">${layers} layers</span></h2>
      <div class="mn-live">
        <div class=${'stage3d mn-stage' + (r.status === 'running' ? ' working' : '')}>
          <${Brain3D} height=${400} layers=${layers} mode="donor" scan=${{ layer: scanLayer, total: layers, found: found.filter((f) => f.verdict !== 'rejected').map((f) => ({ layer: f.layer, kind: f.kind, label: f.label, id: f.id })) }} pulse=${pulse} highlight=${stageId === 'scan' && live ? scanLayer : null} label=${story} onPick=${(pid) => pid && openPart(pid)} />
          <div class="stage-hud"><span class="mn-hud-l">${live ? html`<span class="livechip">${cur ? stageName(cur) : 'Working'}</span>` : html`<${StatusChip} status=${r.status} />`}</span><span class="num">${found.length ? kept + ' kept of ' + found.length + ' named' : layers + ' layers'}</span></div>
        </div>
        <aside class="card mn-read">
          <h3>How to read this</h3>
          <ul>
            <li><b>Each flat sheet</b> is one layer of the model. Layer 0 is at the bottom.</li>
            <li><b>The beam</b> is the scan. It moves up one layer at a time.</li>
            <li><b>A glowing mark</b> is a candidate part the scan named. Click one to open it.</li>
            <li><b>A flash</b> means a part was judged: green kept, red rejected.</li>
          </ul>
          <div class="mn-read-n num">
            <div><b><${Count} n=${c.candidates || 0} /></b><span>candidates</span></div>
            <div class="g"><b><${Count} n=${c.kept || 0} /></b><span>kept</span></div>
            <div class="r"><b><${Count} n=${c.rejected || 0} /></b><span>rejected</span></div>
          </div>
        </aside>
      </div></section>

    <${Candidates} found=${found} live=${live} />

    <section class="sect"><button class=${'mn-logtoggle' + (showLog ? ' open' : '')} onClick=${() => setShowLog(!showLog)} aria-expanded=${showLog ? 'true' : 'false'}><${Icon} n="chevron" size=${14} /><span>${showLog ? 'Hide the log' : 'Show the log'}</span><span class="faint num">${(r.log || []).length} lines${live ? ', live' : ''}</span></button>
      ${showLog && html`<div class="card mn-logcard reveal"><${LogView} log=${r.log || []} /><div class="faint mn-lognote">Each line names the agent that wrote it. The OpenAI mark is the reading agent.</div></div>`}</section>
  </div>`;
}

/* ---------- #/mine/model/<id>: one mined model in full ---------- */

const zOf = (p) => (p.z != null ? p.z : p.raw && p.raw.z != null ? p.raw.z : null);
const TABLE_PAGE = 15;

function Heat({ detail, onHover, hot }) {
  const L = detail.layers || 1;
  const kinds = KIND_ORDER.filter((k) => (detail.layer_map || []).some((x) => x.kind === k));
  const grid = useMemo(() => { const g = {}; let mx = 1; for (const x of detail.layer_map || []) { g[x.kind + ':' + x.layer] = x.count; mx = Math.max(mx, x.count); } return { g, mx }; }, [detail]);
  if (!kinds.length) return html`<div class="faint">No layer map for this model.</div>`;
  return html`<div class="mn-heat" onMouseLeave=${() => onHover(null)}>
    ${kinds.map((k, ri) => html`<div class="mn-hrow" key=${k}><span class="mn-hk"><i style=${{ background: kindColor(k) }}></i>${KIND_NAMES[k] || k}</span>
      <span class="mn-hcells" style=${{ gridTemplateColumns: 'repeat(' + L + ', minmax(7px, 1fr))' }}>${Array.from({ length: L }, (_, l) => { const n = grid.g[k + ':' + l] || 0; return html`<span key=${l} class=${'mn-hc' + (n ? ' on' : '') + (hot === l ? ' hot' : '')} style=${{ '--c': kindColor(k), '--a': n ? 14 + Math.round((n / grid.mx) * 86) : 0, '--i': l + ri * 3 }} title=${'Layer ' + l + ': ' + n + ' ' + (KIND_NAMES[k] || k).toLowerCase()} onMouseEnter=${() => onHover(l)}></span>`; })}</span></div>`)}
    <div class="mn-haxis"><span>L0</span><span>L${Math.round(L / 2)}</span><span>L${L - 1}</span></div>
  </div>`;
}

function ModelView({ id }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [hot, setHot] = useState(null);
  const [f, setF] = useState({ kind: null, skill: null, status: null });
  const [sort, setSort] = useState({ k: 'layer', dir: 1 });
  const [q, setQ] = useState('');
  const [page, setPage] = useState(1);
  const tbl = useRef(null);
  const load = () => { setErr(null); api.modelDetail(id).then(setD).catch(setErr); };
  useEffect(load, [id]);
  useEffect(() => { setPage(1); }, [f, q, sort]);
  const parts = (d && d.parts) || [];
  const rows = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const l = parts.filter((p) => (!f.kind || p.kind === f.kind) && (!f.skill || (p.skills || []).includes(f.skill)) && (!f.status || p.status === f.status) && (!ql || (p.label + ' ' + p.id).toLowerCase().includes(ql)));
    const val = { label: (p) => p.label, kind: (p) => p.kind, layer: (p) => (p.layers ? p.layers[0] : 0), size: (p) => p.params_m || 0, z: (p) => { const z = zOf(p); return z == null ? -1e9 : z; }, status: (p) => p.status || '' }[sort.k];
    return l.sort((a, b) => { const x = val(a), y = val(b); return (x < y ? -1 : x > y ? 1 : 0) * sort.dir; });
  }, [parts, f, q, sort]);
  const byTech = useMemo(() => { const m = {}; for (const p of parts) { const t = inferTechnique(p); m[t] = (m[t] || 0) + 1; } return m; }, [parts]);
  if (err && !d) return html`<${ErrorBox} error=${err} onRetry=${load} />`;
  if (!d) return html`<div class="doc wide"><div class="skeleton" style="height:420px"></div></div>`;
  const kinds = parts.reduce((a, p) => { a[p.kind] = (a[p.kind] || 0) + 1; return a; }, {});
  const skills = Object.entries(d.skills || {}).sort((a, b) => b[1] - a[1]);
  const maxSk = skills.length ? skills[0][1] : 1;
  const L = d.layers || 24;
  const statuses = [...new Set(parts.map((p) => p.status).filter(Boolean))];
  const techIds = [...new Set([...(d.techniques || []), ...Object.keys(byTech)])].filter((t) => techniqueById(t)).sort((a, b) => (byTech[b] || 0) - (byTech[a] || 0));
  const pages = Math.max(1, Math.ceil(rows.length / TABLE_PAGE));
  const at = Math.min(page, pages);
  const th = (k, name, cls) => html`<th class=${cls || ''} aria-sort=${sort.k === k ? (sort.dir > 0 ? 'ascending' : 'descending') : 'none'}><button class=${'mn-th' + (sort.k === k ? ' on' : '')} onClick=${() => setSort({ k, dir: sort.k === k ? -sort.dir : 1 })}>${name}<span class="mn-arrow">${sort.k === k ? (sort.dir > 0 ? '↑' : '↓') : ''}</span></button></th>`;
  const sizeText = (m) => (m == null ? '-' : m >= 1000 ? (m / 1000).toFixed(2) + 'B' : m + 'M');
  const specLine = [d.params_b != null ? bText(d.params_b) + ' parameters' : null, L + ' layers', d.experts ? d.experts + ' experts per layer' : d.kind === 'moe' ? 'mixture of experts' : 'dense'].filter(Boolean).join(' · ');
  const kindWords = KIND_ORDER.filter((k) => kinds[k]).map((k) => kinds[k] + ' ' + (KIND_NAMES[k] || k).toLowerCase());
  const topNames = skills.slice(0, 3).map(([k]) => skillName(k));
  const glanceText = parts.length === 0 ? 'Nothing from this model is in the library yet.' : 'Mining found ' + parts.length + ' parts in ' + (d.short || donorShort(d.model)) + ': ' + kindWords.join(', ') + '.' + (topNames.length ? ' They cover ' + topNames.join(', ').replace(/, ([^,]*)$/, ' and $1') + ' most.' : '');
  return html`<div class="doc wide mn">
    <a class="mn-back lnk" href="#/mine"><${Icon} n="chevron" size=${13} />Mine a model</a>
    <div class="ph mn-rhead"><div>
      <div class="mn-rtitle"><${ModelLogo} model=${d.model} size=${40} /><h1 class="page-title ell">${d.short || donorShort(d.model)}</h1>${d.kind === 'moe' && html`<span class="tag b">Mixture of experts</span>`}${d.in_library === false && html`<span class="tag w">Not in library</span>`}</div>
      <p class="lede">${d.model} · ${specLine}${d.license ? ' · ' + d.license : ''}</p></div>
      <div class="mn-powered"><${PoweredBy} who="modal" label="Mined on" /><${PoweredBy} who="pastdev" label="Shared through" /></div></div>
    <section class="card mn-glance">
      <div class="mn-g-n"><b class="num"><${Count} n=${parts.length} /></b><span>parts mined</span></div>
      <div class="mn-g-main">
        <p class="mn-g-s">${glanceText}</p>
        <${KindBar} kinds=${kinds} height=${8} />
        <${KindLegend} kinds=${kinds} />
      </div>
      <div class="mn-g-sk"><span class="mn-g-h">Top skills</span>
        ${skills.slice(0, 3).map(([k, n]) => html`<button class="mn-g-chip" key=${k} onClick=${() => { setF({ ...f, skill: k }); if (tbl.current) tbl.current.scrollIntoView({ behavior: 'smooth', block: 'start' }); }}><span class="ell">${skillName(k)}</span><b class="num">${n}</b></button>`)}
        ${skills.length === 0 && html`<span class="faint">None recorded</span>`}</div>
    </section>
    ${parts.length === 0 ? html`<section class="card mn-pad mn-empty3d"><b>This model has not been mined yet</b><p class="faint">No scan has run on it, so there is nothing to draw. Start a request on the Mine a model page to scan its ${L} layers and publish parts.</p><a class="btn" href="#/mine">Go to Mine a model</a></section>` : html`<div class="stage3d mn-stage">
      <${Brain3D} height=${440} layers=${L} mode="donor" scan=${{ layer: L, total: L, found: parts.slice(0, 240).map((p) => ({ layer: p.layers ? p.layers[0] : 0, kind: p.kind, label: p.label, id: p.id })) }} highlight=${hot} label=${parts.length + ' parts across ' + L + ' layers'} onPick=${(pid) => pid && openPart(pid)} />
      <div class="stage-hud"><span>${hot != null ? 'Layer ' + hot : 'Every part glows at its layer. Click one to open it.'}</span><span class="mn-hud-r faint">${parts.length} parts, ${L} layers</span></div>
    </div>`}
    <section class="sect"><h2 class="sec"><span>Where the parts are</span><span class="faint cnt">layers by kind</span></h2>
      <div class="card mn-pad"><${Heat} detail=${d} onHover=${setHot} hot=${hot} /></div></section>
    <section class="sect"><h2 class="sec"><span>Skills it covers</span><span class="faint cnt">${skills.length}</span></h2>
      <div class="card mn-pad mn-skillgrid">${skills.length === 0 && html`<span class="faint">No skills recorded.</span>`}
        ${skills.slice(0, 18).map(([s, n], i) => html`<button key=${s} class=${'mn-skill reveal' + (f.skill === s ? ' on' : '')} style=${{ '--i': i }} onClick=${() => { setF({ ...f, skill: f.skill === s ? null : s }); if (tbl.current) tbl.current.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} aria-pressed=${f.skill === s ? 'true' : 'false'}>
          <span class="mn-skill-n ell">${skillName(s)}</span><span class="mn-skill-bar"><i style=${{ width: Math.max(4, (n / maxSk) * 100) + '%', '--i': i }}></i></span><b class="num"><${Count} n=${n} /></b></button>`)}</div></section>
    <section class="sect"><h2 class="sec"><span>How its parts were found</span><span class="faint cnt">${techIds.length} techniques</span></h2>
      <div class="mn-techgrid">${techIds.map((tid, i) => { const t = techniqueById(tid); return html`<div class="card mn-techcard reveal" style=${{ '--i': i }} key=${tid}>
        <div class="mn-tech-top"><b>${t.name}</b><${SponsorMark} who=${t.sponsor} size=${14} /><span class="grow"></span><span class="tag num">${byTech[tid] || 0} parts</span></div>
        <div class="mn-tech-what">${t.what}</div><div class="faint mn-tech-why">${t.why}</div></div>`; })}</div></section>
    <section class="sect" ref=${tbl}><h2 class="sec"><span>All parts</span><span class="faint cnt">${rows.length === parts.length ? parts.length : rows.length + ' of ' + parts.length}</span></h2>
      <div class="fbar"><div class="frow2" role="group" aria-label="Kind">
        <button class=${'fchip2' + (!f.kind ? ' on' : '')} onClick=${() => setF({ ...f, kind: null })}>All<span class="n">${parts.length}</span></button>
        ${KIND_ORDER.filter((k) => kinds[k]).map((k) => html`<button key=${k} class=${'fchip2' + (f.kind === k ? ' on' : '')} aria-pressed=${f.kind === k ? 'true' : 'false'} onClick=${() => setF({ ...f, kind: f.kind === k ? null : k })}><i class="mn-kd" style=${{ background: kindColor(k) }}></i>${KIND_NAMES[k]}<span class="n">${kinds[k]}</span></button>`)}</div>
        <div class="frow2" role="group" aria-label="Status">${statuses.map((s) => html`<button key=${s} class=${'fchip2' + (f.status === s ? ' on' : '')} aria-pressed=${f.status === s ? 'true' : 'false'} onClick=${() => setF({ ...f, status: f.status === s ? null : s })}>${s}</button>`)}
          ${f.skill && html`<button class="fchip2 on" onClick=${() => setF({ ...f, skill: null })}>Skill: ${skillName(f.skill)} <${Icon} n="x" size=${12} /></button>`}
          <div class="searchbox mn-q"><${Icon} n="search" size=${16} /><input placeholder="Search parts" aria-label="Search parts" value=${q} onInput=${(e) => setQ(e.target.value)} /></div></div></div>
      <div class="card mn-table"><div class="mn-tscroll"><table><thead><tr>${th('label', 'Part')}${th('kind', 'Kind')}${th('layer', 'Layers', 'num')}${th('size', 'Size', 'num')}${th('z', 'z-score', 'num')}${th('status', 'Status')}</tr></thead>
        <tbody>${rows.slice((at - 1) * TABLE_PAGE, at * TABLE_PAGE).map((p, i) => { const z = zOf(p); return html`<tr key=${p.id} class="mn-tr reveal" style=${{ '--i': Math.min(i, 10) }} tabindex="0" onClick=${() => openPart(p.id)} onKeyDown=${(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPart(p.id); } }}>
          <td class="mn-tl"><span class="ell">${p.label}</span></td><td><span class="mn-kk"><i class="mn-kd" style=${{ background: kindColor(p.kind) }}></i>${(KIND_NAMES[p.kind] || p.kind)}</span></td>
          <td class="num">${p.layers ? 'L' + p.layers[0] + (p.layers[1] !== p.layers[0] ? ' to ' + p.layers[1] : '') : '-'}</td><td class="num">${sizeText(p.params_m)}</td>
          <td class="num">${z != null ? html`<span class=${z >= 3 ? 'up' : ''}>${(+z).toFixed(1)}</span>` : html`<span class="faint">-</span>`}</td><td><${StatusTag} status=${p.status} /></td></tr>`; })}</tbody></table>
        ${rows.length === 0 && html`<div class="empty mn-none">No part matches.</div>`}</div></div>
      <${Pager} page=${at} total=${rows.length} size=${TABLE_PAGE} onPage=${(n) => { setPage(n); if (tbl.current) tbl.current.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} label="Part pages" />
    </section>
    ${(d.findings || []).length > 0 && html`<section class="sect"><h2 class="sec"><span>Findings</span><span class="faint cnt">${d.findings.length}</span></h2>
      <div class="card mn-pad"><ul class="pd-list">${d.findings.map((x) => html`<li key=${x.id || x}><a class="lnk" href="#/parts?tab=findings">${x.title || x.id || x}</a></li>`)}</ul></div></section>`}
    ${(d.recipes || []).length > 0 && html`<section class="sect"><h2 class="sec"><span>Recipes</span><span class="faint cnt">${d.recipes.length}</span></h2>
      <div class="card mn-pad"><ul class="pd-list">${d.recipes.map((x) => html`<li key=${x.id || x}><a class="lnk" href="#/parts?tab=recipes">${x.title || x.id || x}</a>${x.builds ? html` <span class="faint">${x.builds}</span>` : null}</li>`)}</ul></div></section>`}
  </div>`;
}

export function MineScreen({ sub, id }) {
  return html`<${TopBar} /><main class="screen">
    ${sub === 'req' && id ? html`<${ReqView} key=${id} id=${id} />` : sub === 'model' && id ? html`<${ModelView} key=${id} id=${id} />` : html`<${MineHome} />`}
  </main>`;
}
