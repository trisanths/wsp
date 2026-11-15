import { html, useState, useEffect, useLayoutEffect, useRef, useMemo } from '../lib.js';
import { api } from '../api.js';
import { go, toast, pct, pctS, sgn, bsize, clamp, clone, emptyProgram, normProgram, donorShort, maskIds, buildMask, donorColor, donorLetter, specProgress, primaryTarget, tokenizeLine, setLastBuild } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Menu, BenchBar, BenchLegend, Ring, Tag, StubBadge } from '../components.js';
import { PopNum, SlidingBar, SkelReveal, Matrix, SuccessCheck, useShake, shake, SwapText } from '../motion.js';

// Short labels for score statuses other than ok. Only "collapsed" means the model lost a skill.
const STATUS_LABELS = { collapsed: 'collapsed', infra_error: 'GPU error', compile_error: 'compile error', rejected_fit: 'fit too low', invalid: 'invalid', budget_exhausted: 'no budget' };
const STATUS_MESSAGES = {
  collapsed: 'This program collapsed a guard.',
  infra_error: 'The GPU job failed before scoring. No score was recorded. Try again, or try a smaller program.',
  compile_error: 'The program did not compile.',
  rejected_fit: 'A part fits the host too poorly to score (below the fit gate).',
  invalid: 'The program is not valid.',
  budget_exhausted: 'This session has no evals left.',
};
function statusLabel(status) { return STATUS_LABELS[status] || status; }
function statusMessage(status) { return STATUS_MESSAGES[status] || 'Score status: ' + status; }

const GLUE = [null, 4, 8, 16, 32, 64];
const HEAL_STEPS = [0, 100, 200, 300, 400, 600, 800, 1000];
const TAP_TRAIN_FALLBACK = { data: 'mmlu_aux', steps: 500, batch_size: 8, seq_len: 256, branch_lr: 0.0003, gate_lr: 0.03 };
const TAP_STEP = 100, TAP_MIN = 100, TAP_MAX = 2000;
const tapKey = (t) => 'tap:' + t.donor + '@' + t.at;
// every graft and tap as one list of canvas nodes
function nodesOf(p) {
  return [...p.grafts.map((g, i) => ({ key: 'g' + i, t: 'g', i, at: g.at })), ...(p.taps || []).map((t, i) => ({ key: 't' + i, t: 't', i, at: t.at }))];
}
function measuredLine(m) {
  if (!m || m.score == null) return null;
  return String(m.task || '').toUpperCase() + ' ' + pct(m.score) + (m.lora != null ? ' vs LoRA ' + pct(m.lora) : '') + ' (measured)';
}
let DRAG = null; // what is being dragged right now

const ls = {
  get(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* storage can be off */ } },
};

function dragImage(e) {
  const src = e.currentTarget;
  const g = src.cloneNode(true);
  g.style.cssText = 'position:fixed;left:-1000px;top:0;width:' + src.offsetWidth + 'px;transform:rotate(-2deg);background:var(--bg-raised);border:1px solid var(--blue);border-radius:12px;box-shadow:var(--shadow-float);opacity:1;';
  document.body.appendChild(g);
  try { e.dataTransfer.setDragImage(g, 24, 20); } catch (x) { /* ignore */ }
  setTimeout(() => g.remove(), 0);
}

function Piece({ ic, color, t1, t2, t3, used, dragging, onDragStart, onDragEnd, onClick }) {
  return html`<div class=${'piece' + (dragging ? ' dragging' : '') + (used ? ' used' : '')} draggable="true" onDragStart=${onDragStart} onDragEnd=${onDragEnd} onClick=${onClick} title="Drag onto the model, or click to add">
    <div class="pic" style=${{ '--c': color }}>${ic}</div><div style="min-width:0"><div class="t1">${t1}</div><div class="t2">${t2}</div>${t3 && html`<div class="t3">${t3}</div>`}</div>
    <span class="add">${used ? html`<${Icon} n="check" size=${14} />` : html`<${Icon} n="plus" size=${14} />`}</span></div>`;
}

function briefProgram(p, cat) {
  const bits = (p.grafts || []).map((g) => { const x = cat && cat.parts.find((q) => q.id === g.id); return (x ? x.label : g.id) + ' L' + g.at; });
  const ids = maskIds(p.masks);
  if (ids.length) bits.push('- ' + ids.length + ' mask' + (ids.length > 1 ? 's' : ''));
  for (const t of p.taps || []) bits.push(donorShort(t.donor) + ' tap L' + t.at);
  if (p.lora) bits.push('LoRA r' + (p.lora.rank || 16));
  if (p.heal) bits.push('heal ' + p.heal.steps);
  return bits.length ? bits.join(' + ') : 'host only';
}

export function BuilderScreen({ id, specId }) {
  const [cat, setCat] = useState(null);
  const [spec, setSpec] = useState(null);
  const [err, setErr] = useState(null);
  const [program, setProgramRaw] = useState(null);
  const [undo, setUndo] = useState([]);
  const [view, setView] = useState('canvas');
  const [layerMode, setLayerMode] = useState('simple');
  const [q, setQ] = useState('');
  const [over, setOver] = useState(null);
  const [dragId, setDragId] = useState(null);
  const [menuAt, setMenuAt] = useState(null);
  const [sel, setSel] = useState(null);
  const [geo, setGeo] = useState({ y: {}, wires: [], h: 0 });
  const [budget, setBudget] = useState(() => { try { return JSON.parse(ls.get('wsp.budget.' + id)); } catch (e) { return null; } });
  const [job, setJob] = useState(null);
  const [fits, setFits] = useState({});
  const [attempts, setAttempts] = useState(() => { try { return JSON.parse(ls.get('wsp.att.' + id)) || []; } catch (e) { return []; } });
  const [sugs, setSugs] = useState(null);
  const [sugBusy, setSugBusy] = useState(false);
  const [codeText, setCodeText] = useState('');
  const [codeErr, setCodeErr] = useState(null);
  const [scroll, setScroll] = useState({ x: 0, y: 0 });
  const [handing, setHanding] = useState(false);
  const [cw, setCw] = useState(1000);
  const [flash, setFlash] = useState(null);
  const flashTimer = useRef(0);
  const btnRefs = useRef({});
  const known = useRef(null);
  const newKeys = useRef(new Set());
  const errRef = useShake(codeErr ? codeErr.message + ':' + codeErr.line : null);
  const compact = cw < 780;
  const nodeX = compact ? 276 : 480;

  const innerRef = useRef();
  const scrollRef = useRef();
  const nodeRefs = useRef({});
  const fromCode = useRef(false);
  const parseSeq = useRef(0);
  const parseTimer = useRef(0);
  const alive = useRef(true);
  const programRef = useRef(null);
  programRef.current = program;
  useEffect(() => () => { alive.current = false; clearTimeout(parseTimer.current); clearTimeout(flashTimer.current); }, []);

  /* ---- load ---- */
  const load = async () => {
    setErr(null);
    try {
      const [c, s] = await Promise.all([api.catalog(), specId ? api.getSpec(specId) : Promise.resolve(null)]);
      setCat(c); setSpec(s);
      let p = null;
      try { p = JSON.parse(ls.get('wsp.prog.' + id)); } catch (e) { p = null; }
      if (!p) { try { p = JSON.parse(ls.get('wsp.seed.' + id)); } catch (e) { p = null; } }
      setProgramRaw(normProgram(p || emptyProgram(c.host.model)));
    } catch (e) { setErr(e); }
  };
  useEffect(() => { load(); setLastBuild({ id, specId }); }, [id]);
  useEffect(() => { if (program) ls.set('wsp.prog.' + id, JSON.stringify(program)); }, [program]);
  useEffect(() => { ls.set('wsp.att.' + id, JSON.stringify(attempts.map((a) => ({ ...a })))); }, [attempts]);

  /* ---- edit helpers ---- */
  const setProgram = (p, keepUndo) => {
    p = normProgram(p);
    if (!keepUndo) setUndo((u) => [...u.slice(-40), programRef.current]);
    programRef.current = p;
    setProgramRaw(p);
  };
  const edit = (fn) => { const p = clone(programRef.current); fn(p); setProgram(p); };
  const doUndo = () => {
    setUndo((u) => {
      if (!u.length) return u;
      const prev = u[u.length - 1];
      programRef.current = prev; setProgramRaw(prev);
      return u.slice(0, -1);
    });
  };
  useEffect(() => {
    const k = (e) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === 'z' && !/INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || '')) { e.preventDefault(); doUndo(); }
    };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, []);

  const healDefault = () => ({ steps: 300, data: spec && spec.task_type === 'math' ? 'gsm8k_train' : 'task_train', lr: 0.0003, batch_size: 8, seq_len: 384 });
  const addPart = (part, at) => { edit((p) => { p.grafts.push({ id: part.id, at: at ?? part.default_at ?? 6, control: null, control_seed: 0, glue_rank: null }); }); setSel(null); };
  const addTap = (part, at) => {
    edit((p) => {
      p.taps.push({ donor: part.donor, donor_layer: part.donor_layer, at: at ?? part.default_at ?? 6, attn_dim: 256, heads: 8, tap_id: null });
      if (!p.tap_train) p.tap_train = clone((cat && cat.tap_train_default) || TAP_TRAIN_FALLBACK);
    });
    setSel(null);
  };
  const addAny = (part, at) => (part.kind === 'tap' ? addTap(part, at) : addPart(part, at));
  const removeTap = (i) => edit((p) => { p.taps.splice(i, 1); if (!p.taps.length) p.tap_train = null; });
  const addMask = (m) => edit((p) => { const ids = maskIds(p.masks); if (!ids.includes(m.id)) p.masks = buildMask([...ids, m.id]); });
  const addTrain = (t) => edit((p) => { if (t.id === 'lora' && !p.lora) p.lora = { rank: 16 }; if (t.id === 'heal' && !p.heal) p.heal = healDefault(); });
  const trainUsed = (t) => program && (t.id === 'lora' ? !!program.lora : !!program.heal);

  /* ---- layers ---- */
  const N = cat ? cat.host.layers : 16;
  const slabs = useMemo(() => {
    if (!program) return [];
    const hot = new Set(nodesOf(program).map((n) => n.at));
    const out = [];
    if (layerMode === 'layers') {
      for (let l = 0; l < N; l++) out.push({ first: l, n: 1, hot: hot.has(l) });
      return out;
    }
    let l = 0;
    while (l < N) {
      if (hot.has(l)) { out.push({ first: l, n: 1, hot: true }); l++; continue; }
      let n = 0;
      while (l + n < N && !hot.has(l + n) && n < 4) n++;
      out.push({ first: l, n, hot: false });
      l += n;
    }
    return out;
  }, [program, layerMode, N]);

  const layerAt = (e) => {
    const el = e.target.closest ? e.target.closest('[data-first]') : null;
    if (!el) return null;
    const first = +el.dataset.first, n = +el.dataset.n;
    const r = el.getBoundingClientRect();
    return first + clamp(Math.floor(((e.clientY - r.top) / r.height) * n), 0, n - 1);
  };
  const onDragOver = (e) => {
    if (!DRAG) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = DRAG.kind === 'node' ? 'move' : 'copy';
    const layer = layerAt(e);
    const want = layer != null ? 'L' + layer : 'tail';
    setOver((o) => (o === want ? o : want));
  };
  const onDragLeave = (e) => { if (!e.currentTarget.contains(e.relatedTarget)) setOver(null); };
  const endDrag = () => { DRAG = null; setOver(null); setDragId(null); };
  const onDrop = (e) => {
    if (!DRAG) return;
    e.preventDefault();
    const d = DRAG;
    const layer = layerAt(e);
    endDrag();
    if (d.kind === 'part') { addAny(d.part, layer ?? undefined); if (layer == null) toast('Added at its default layer, L' + d.part.default_at + '. Drag the card to move it.'); }
    else if (d.kind === 'node') { if (layer != null) edit((p) => { (d.t === 't' ? p.taps : p.grafts)[d.i].at = layer; }); }
    else if (d.kind === 'mask') addMask(d.mask);
    else if (d.kind === 'train') addTrain(d.train);
  };
  const startDrag = (e, d, dragKey) => { DRAG = d; e.dataTransfer.effectAllowed = 'copyMove'; try { e.dataTransfer.setData('text/plain', dragKey); } catch (x) { /* ignore */ } dragImage(e); setTimeout(() => setDragId(dragKey), 0); };

  /* ---- geometry: node positions and wires from the real DOM ---- */
  useLayoutEffect(() => {
    const inner = innerRef.current;
    if (!inner || !program) return;
    const ir = inner.getBoundingClientRect();
    const items = nodesOf(program).map((n) => {
      const slab = inner.querySelector('[data-hot="' + n.at + '"]');
      if (!slab) return null;
      const r = slab.getBoundingClientRect();
      const node = nodeRefs.current[n.key];
      return { n, cy: r.top - ir.top + r.height / 2, x1: r.right - ir.left, h: node ? node.offsetHeight : 120 };
    }).filter(Boolean).sort((a, b) => a.cy - b.cy);
    const y = {};
    const wires = [];
    let bottom = 8;
    for (const it of items) {
      const top = Math.max(it.cy - 24, bottom + 12);
      y[it.n.key] = Math.round(top);
      bottom = top + it.h;
      let color, tap = false;
      if (it.n.t === 't') { color = 'var(--blue)'; tap = true; }
      else {
        const g = program.grafts[it.n.i];
        const part = cat && cat.parts.find((p) => p.id === g.id);
        color = donorColor(part && part.donor);
      }
      wires.push({ k: it.n.key, tap, x1: it.x1, y1: it.cy, x2: nodeX, y2: top + 24, color });
    }
    const stackH = inner.querySelector('.stack') ? inner.querySelector('.stack').offsetHeight + 40 : 0;
    const next = { y, wires, h: Math.max(bottom + 160, stackH + 160) };
    setGeo((g) => (JSON.stringify(g) === JSON.stringify(next) ? g : next));
  });
  useEffect(() => {
    const inner = innerRef.current;
    if (!inner || !window.ResizeObserver) return;
    const ro = new ResizeObserver(() => { if (scrollRef.current) setCw(scrollRef.current.clientWidth); setGeo((g) => ({ ...g })); });
    ro.observe(inner);
    if (scrollRef.current) ro.observe(scrollRef.current);
    return () => ro.disconnect();
  }, [view, cat]);

  /* ---- code view ---- */
  useEffect(() => {
    if (!program || view === 'canvas') return;
    if (fromCode.current) { fromCode.current = false; return; }
    const seq = ++parseSeq.current;
    api.programText(program).then((r) => { if (alive.current && seq === parseSeq.current) { setCodeText(r.text); setCodeErr(null); } }).catch(() => {});
  }, [program, view]);
  const onCode = (text) => {
    setCodeText(text);
    clearTimeout(parseTimer.current);
    const seq = ++parseSeq.current;
    parseTimer.current = setTimeout(async () => {
      try {
        const r = await api.programParse(text);
        if (!alive.current || seq !== parseSeq.current) return;
        setCodeErr(null);
        if (JSON.stringify(normProgram(r.program)) !== JSON.stringify(programRef.current)) { fromCode.current = true; setProgram(r.program); }
      } catch (e) {
        if (alive.current && seq === parseSeq.current) setCodeErr({ message: e.message, line: e.line });
      }
    }, 450);
  };

  /* ---- jobs ---- */
  const run = async (action) => {
    if (job) return;
    let p = clone(programRef.current);
    if (action === 'heal' && !p.heal) p.heal = healDefault();
    if (action === 'train') { if (!p.lora) p.lora = { rank: 16 }; if (!p.heal) p.heal = healDefault(); }
    if (action !== 'compile' && JSON.stringify(p) !== JSON.stringify(programRef.current)) setProgram(p);
    setJob({ action, status: 'queued', t0: Date.now() });
    try {
      const { job_id } = await api.builderAction(id, p, action);
      let j;
      for (;;) {
        await new Promise((r) => setTimeout(r, 650));
        if (!alive.current) return;
        j = await api.getJob(job_id);
        if (j.budget) { setBudget(j.budget); ls.set('wsp.budget.' + id, JSON.stringify(j.budget)); }
        if (j.status === 'done' || j.status === 'error') break;
        setJob((x) => (x ? { ...x, status: j.status } : x));
      }
      if (j.status === 'error') { toast(j.error || 'The job failed', 'err'); shake(btnRefs.current[action]); setJob(null); return; }
      const r = j.result || {};
      if (r.fit) setFits(r.fit);
      if (action === 'compile') {
        toast('Fit check done' + (r.params_b ? ': ' + bsize(r.params_b) + ' parameters' : ''), 'ok');
      } else {
        const at = { n: 0, action, program: p, scores: r.scores, combined: r.combined, control: r.control, params_b: r.params_b, status: r.status, compute_s: r.compute_s };
        setAttempts((a) => [...a, { ...at, n: a.length + 1 }]);
        if (r.status && r.status !== 'ok') { toast(statusMessage(r.status), 'err'); shake(btnRefs.current[action]); }
        else if (action === 'score') { setFlash(action); clearTimeout(flashTimer.current); flashTimer.current = setTimeout(() => setFlash(null), 1900); }
        refreshSugs(p);
      }
    } catch (e) { toast(e.message, 'err'); shake(btnRefs.current[action]); }
    if (alive.current) setJob(null);
  };

  const refreshSugs = async (p) => {
    setSugBusy(true);
    try { const r = await api.suggest(id, p || programRef.current); if (alive.current) setSugs(r.suggestions || []); } catch (e) { if (alive.current) setSugs([]); }
    if (alive.current) setSugBusy(false);
  };
  useEffect(() => { if (program && sugs == null && !sugBusy) refreshSugs(program); }, [program == null]);

  const handoff = async () => {
    setHanding(true);
    try { const r = await api.handoff(id, programRef.current); go('/run/' + r.run_id); } catch (e) { toast(e.message, 'err'); setHanding(false); }
  };

  if (err) return html`<${TopBar} /><${ErrorBox} error=${err} onRetry=${load} />`;
  if (!cat || !program) return html`<${TopBar} /><${SkelReveal} fill ready=${false} skeleton=${html`<div class="builder"><div class="palette"><div class="skeleton" style="height:420px"></div></div><div class="canvas"></div><div class="side"><div class="skeleton" style="height:240px"></div></div></div>`} />`;
  // nodes that appear after the first render are new, and fade in with their wire
  {
    const keys = nodesOf(program).map((n) => n.key);
    if (known.current === null) known.current = new Set(keys);
    else {
      keys.forEach((k) => { if (!known.current.has(k)) { known.current.add(k); newKeys.current.add(k); } });
      [...known.current].forEach((k) => { if (!keys.includes(k)) { known.current.delete(k); newKeys.current.delete(k); } });
    }
  }

  /* ---- derived ---- */
  const partById = (pid) => cat.parts.find((p) => p.id === pid);
  const maskById = (mid) => cat.masks.find((m) => m.id === mid);
  const lastOk = [...attempts].reverse().find((a) => a.scores);
  const prevOk = lastOk ? [...attempts].slice(0, attempts.indexOf(lastOk)).reverse().find((a) => a.scores) : null;
  const benches = spec ? spec.benchmarks : [];
  const avg = (a) => { const v = Object.values(a.scores || {}); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : null; };
  const prog = spec && lastOk ? specProgress(spec, lastOk.scores) : 0;
  const t1 = spec ? primaryTarget(spec) : null;
  const vsCtl = lastOk && lastOk.control && t1 ? lastOk.scores[t1.id] - lastOk.control.scores[t1.id] : null;
  const tapPart = (t) => cat.parts.find((p) => p.kind === 'tap' && p.donor === t.donor && p.donor_layer === t.donor_layer) || cat.parts.find((p) => p.kind === 'tap' && p.donor === t.donor);
  const tapParamsM = program.taps.reduce((s, t) => s + ((tapPart(t) || {}).params_m || 3), 0);
  // A tap keeps its donor in memory and runs it on the prompt, so each distinct donor counts toward size.
  const tapDonors = [...new Set(program.taps.map((t) => t.donor))];
  const donorParamsB = tapDonors.reduce((s, d) => { const p = (cat.parts || []).find((x) => x.kind === 'tap' && x.donor === d); return s + ((p && p.donor_params_b) || 0); }, 0);
  const estSize = cat.host.params_b + program.grafts.reduce((s, g) => s + ((partById(g.id) || {}).params_m || 0) / 1000, 0) + tapParamsM / 1000 + donorParamsB;
  const matches = (x) => !q || (x.label + ' ' + (x.donor || '') + ' ' + (x.summary || '') + ' ' + x.group).toLowerCase().includes(q.toLowerCase());
  const tapParts = cat.parts.filter((x) => x.kind === 'tap' && matches(x)), addParts = cat.parts.filter((x) => x.kind !== 'tap' && matches(x)), rmParts = cat.masks.filter(matches), trParts = cat.train.filter(matches);
  const used = (pid) => program.grafts.some((g) => g.id === pid);
  const tapUsed = (x) => program.taps.some((t) => t.donor === x.donor && t.donor_layer === x.donor_layer);
  const mIds = maskIds(program.masks);
  const healLabel = program.heal ? program.heal.steps + ' steps' : '300 steps';
  const cvClass = 'cv-body' + (view === 'split' ? ' split' : '');
  const showCanvas = view !== 'code';
  const showCode = view !== 'canvas';
  const hotAt = (l) => program.grafts.filter((g) => g.at === l);
  const tapsAt = (l) => program.taps.filter((t) => t.at === l);
  const dropping = (s) => over && over !== 'tail' && DRAG && DRAG.kind !== 'mask' && DRAG.kind !== 'train' && +over.slice(1) >= s.first && +over.slice(1) < s.first + s.n;
  const jobPct = job ? Math.min(94, ((Date.now() - job.t0) / 2600) * 100) : 0;
  const codeLines = codeText.replace(/\n$/, '').split('\n');

  const slabEl = (s) => {
    const label = s.n === 1 ? 'L' + s.first : 'L' + s.first + ' to L' + (s.first + s.n - 1);
    const hint = s.first === 0 ? 'embed + early' : s.first + s.n === N ? 'late + head' : '';
    const dr = dropping(s);
    const gs = hotAt(s.first), ts = tapsAt(s.first);
    const dl = dr ? +over.slice(1) : null;
    return html`<div class=${'slab' + (s.hot ? ' port' : '') + (dr ? ' drop' : '') + (s.n > 1 && dr ? ' tall' : '')} data-first=${s.first} data-n=${s.n} data-hot=${s.hot ? s.first : null} key=${'s' + s.first}>
      <span class="nm">${dr ? 'Drop here: L' + dl : label}</span>
      ${s.hot && !dr ? html`<span class="reads">reads ${[...gs.map((g) => (partById(g.id) || { label: g.id }).label), ...ts.map((t) => donorShort(t.donor))].join(', ')}</span>` : dr ? html`<span class="faint">release to graft</span>` : html`<span class="faint">${hint}</span>`}
    </div>`;
  };

  return html`<${TopBar} right=${html`<span class="pill hide-sm">evals left <b class="num">${budget ? budget.evals_total - budget.evals_used + ' / ' + budget.evals_total : '-'}</b></span>
      <span class="pill hide-md">compute <b class="num">${budget ? (budget.compute_used_s / 60).toFixed(budget.compute_used_s < 600 ? 1 : 0) + ' / ' + Math.round(budget.compute_total_s / 60) + ' min' : '-'}</b></span>
      <button class="btn ghost sm" disabled=${!undo.length} onClick=${doUndo} title="Undo (Ctrl or Cmd + Z)"><${Icon} n="undo" size=${14} /> Undo</button><button class="btn sm" disabled=${handing} onClick=${handoff}>${handing ? html`<${Spinner} size=${12} />` : null} Hand to agents</button>`}>
      <h1 class="page-title bt-title ell" title=${spec ? spec.title : 'Builder'}>${spec ? spec.title : 'Builder'}</h1><span class="tag">draft ${attempts.length + 1}</span>
    <//>
  <${SkelReveal} fill ready=${true} skeleton=${null}><div class="builder">
    <aside class="palette">
      <div class="searchbox"><${Icon} n="search" size=${15} /><input placeholder="Search parts" value=${q} onInput=${(e) => setQ(e.target.value)} /></div>
      ${tapParts.length > 0 && html`<div class="pgroup">Tap a bigger model</div>`}
      ${tapParts.map((p) => html`<${Piece} key=${p.id} ic=${donorLetter(p.donor, p.label)} color="var(--blue)" t1=${donorShort(p.donor)} t2=${'reads layer ' + p.donor_layer} t3=${measuredLine(p.measured)} used=${tapUsed(p)} dragging=${dragId === p.id} onDragStart=${(e) => startDrag(e, { kind: 'part', part: p }, p.id)} onDragEnd=${endDrag} onClick=${() => addTap(p)} />`)}
      ${addParts.length > 0 && html`<div class="pgroup">Add a skill</div>`}
      ${addParts.map((p) => html`<${Piece} key=${p.id} ic=${donorLetter(p.donor, p.label)} color=${donorColor(p.donor)} t1=${p.label} t2=${p.donor.split('/').pop() + ' · ' + p.kind + ' · L' + p.layers[0] + (p.layers[1] !== p.layers[0] ? '-' + p.layers[1] : '')} used=${used(p.id)} dragging=${dragId === p.id} onDragStart=${(e) => startDrag(e, { kind: 'part', part: p }, p.id)} onDragEnd=${endDrag} onClick=${() => addPart(p)} />`)}
      ${rmParts.length > 0 && html`<div class="pgroup">Remove a skill</div>`}
      ${rmParts.map((m) => html`<${Piece} key=${m.id} ic="-" color="var(--red)" t1=${m.label} t2=${m.summary} used=${mIds.includes(m.id)} dragging=${dragId === m.id} onDragStart=${(e) => startDrag(e, { kind: 'mask', mask: m }, m.id)} onDragEnd=${endDrag} onClick=${() => addMask(m)} />`)}
      ${trParts.length > 0 && html`<div class="pgroup">Train</div>`}
      ${trParts.map((t) => html`<${Piece} key=${t.id} ic=${t.id === 'lora' ? 'R' : 'H'} color=${t.id === 'lora' ? 'var(--amber)' : 'var(--green)'} t1=${t.label} t2=${t.summary} used=${trainUsed(t)} dragging=${dragId === t.id} onDragStart=${(e) => startDrag(e, { kind: 'train', train: t }, t.id)} onDragEnd=${endDrag} onClick=${() => addTrain(t)} />`)}
      ${!tapParts.length && !addParts.length && !rmParts.length && !trParts.length && html`<div class="faint" style="margin-top:20px">Nothing matches "${q}".</div>`}
    </aside>

    <section class="canvas">
      <div class="ctool">
        <${SlidingBar} class="seg2">${[['canvas', 'Canvas', 'grid'], ['code', 'Code', 'code'], ['split', 'Split', 'layers']].map(([k, l]) => html`<button class=${view === k ? 'on' : ''} onClick=${() => setView(k)}>${l}</button>`)}<//>
        ${showCanvas && html`<${SlidingBar} class="seg2"><button class=${layerMode === 'simple' ? 'on' : ''} onClick=${() => setLayerMode('simple')}>Simple</button><button class=${layerMode === 'layers' ? 'on' : ''} onClick=${() => setLayerMode('layers')}>Layers</button><//>`}
      </div>
      <div class=${cvClass}>
        ${showCanvas && html`<div class="cv-scroll" ref=${scrollRef} onDragOver=${onDragOver} onDragLeave=${onDragLeave} onDrop=${onDrop} onClick=${() => { setSel(null); setMenuAt(null); }}>
          <div class=${'cv-inner' + (compact ? ' compact' : '')} ref=${innerRef} style=${{ minHeight: geo.h + 'px' }}>
            <svg class="wires" width=${nodeX + 260} height=${Math.max(geo.h, 300)}>
              <defs>${geo.wires.map((w) => html`<linearGradient id=${'wg' + w.k} gradientUnits="userSpaceOnUse" x1=${w.x1} y1="0" x2=${w.x2} y2="0"><stop offset="0" style="stop-color:var(--text-3)"/><stop offset="1" style=${{ stopColor: w.color }}/></linearGradient>`)}</defs>
              ${geo.wires.map((w) => { const dx = (w.x2 - w.x1) * 0.55; const d = 'M' + w.x1 + ',' + w.y1 + ' C' + (w.x1 + dx) + ',' + w.y1 + ' ' + (w.x2 - dx) + ',' + w.y2 + ' ' + w.x2 + ',' + w.y2; return html`<g key=${w.k} class=${newKeys.current.has(w.k) ? 'is-new' : null}><path d=${d} style=${{ d: "path('" + d + "')" }} stroke=${w.tap ? 'var(--blue)' : 'url(#wg' + w.k + ')'} stroke-dasharray=${w.tap ? '7 6' : null} /><circle cx=${w.x1} cy=${w.y1} r="4" style="fill:var(--text-3)"/><circle cx=${w.x2} cy=${w.y2} r="4" style=${{ fill: w.color }}/></g>`; })}
            </svg>
            <div class="stack" style=${{ marginLeft: '0px' }}>
              <span class="label hostlabel">Host · ${cat.host.model.split('/').pop()}</span>
              ${slabs.map(slabEl)}
              ${mIds.map((mid) => html`<div class="slab mask" key=${mid}><span class="nm">- ${(maskById(mid) || { label: mid }).label} mask</span><span class="ctl"><span class="faint">all layers</span><button class="x" data-tip="Remove mask" aria-label="Remove mask" onClick=${(e) => { e.stopPropagation(); edit((p) => { p.masks = buildMask(maskIds(p.masks).filter((x) => x !== mid)); }); }}>×</button></span></div>`)}
              ${program.tap_train && html`<div class="slab taptrain"><span class="nm">Tap training<span class="sub" title="Training data">data ${program.tap_train.data}</span></span><span class="ctl"><button data-tip="Fewer steps" aria-label="Fewer steps" disabled=${program.tap_train.steps <= TAP_MIN} onClick=${() => edit((p) => { p.tap_train.steps = clamp((p.tap_train.steps || 500) - TAP_STEP, TAP_MIN, TAP_MAX); })}>-</button><span class="num" style="min-width:56px;text-align:center">${program.tap_train.steps} steps</span><button data-tip="More steps" aria-label="More steps" disabled=${program.tap_train.steps >= TAP_MAX} onClick=${() => edit((p) => { p.tap_train.steps = clamp((p.tap_train.steps || 500) + TAP_STEP, TAP_MIN, TAP_MAX); })}>+</button></span></div>`}
              ${program.lora && html`<div class="slab lora"><span class="nm">LoRA</span><span class="ctl">rank <select value=${program.lora.rank || 16} onChange=${(e) => edit((p) => { p.lora.rank = +e.target.value; })}>${[1, 2, 4, 8, 16, 32, 64].map((r) => html`<option value=${r}>${r}</option>`)}</select><button class="x" data-tip="Remove LoRA" aria-label="Remove LoRA" onClick=${() => edit((p) => { p.lora = null; })}>×</button></span></div>`}
              ${program.heal && html`<div class="slab heal"><span class="nm">Heal</span><span class="ctl"><button data-tip="Fewer steps" aria-label="Fewer steps" onClick=${() => edit((p) => { const i = HEAL_STEPS.indexOf(p.heal.steps); p.heal.steps = HEAL_STEPS[Math.max(1, (i < 0 ? 3 : i) - 1)]; })}>-</button><span class="num" style="min-width:62px;text-align:center">${program.heal.steps} steps</span><button data-tip="More steps" aria-label="More steps" onClick=${() => edit((p) => { const i = HEAL_STEPS.indexOf(p.heal.steps); p.heal.steps = HEAL_STEPS[Math.min(HEAL_STEPS.length - 1, (i < 0 ? 3 : i) + 1)]; })}>+</button><button class="x" data-tip="Remove heal" aria-label="Remove heal" onClick=${() => edit((p) => { p.heal = null; })}>×</button></span></div>`}
              <div class=${'slab tail' + (over === 'tail' && DRAG ? ' drop' : '')}>${over === 'tail' && DRAG ? 'Release to add' : 'Drop a mask, heal or LoRA here'}</div>
            </div>
            ${program.grafts.map((g, i) => { const part = partById(g.id) || { label: g.id, donor: '', kind: 'part', layers: [0, 0], params_m: 0 }; const fit = fits[g.id + '@' + g.at]; const col = donorColor(part.donor); const nk = 'g' + i;
              return html`<div class=${'node' + (newKeys.current.has(nk) ? ' is-new' : '') + (dragId === nk ? ' dragging' : '') + (sel === nk ? ' sel' : '') + (menuAt === nk ? ' menu-open' : '')} key=${nk} ref=${(el) => { nodeRefs.current[nk] = el; }} style=${{ left: nodeX + 'px', top: (geo.y[nk] ?? 20) + 'px', zIndex: menuAt === nk ? 9 : 2 }} draggable="true"
                onDragStart=${(e) => startDrag(e, { kind: 'node', t: 'g', i }, nk)} onDragEnd=${endDrag} onClick=${(e) => { e.stopPropagation(); setSel(nk); }}>
                <div class="nh"><div class="pic" style=${{ '--c': col }}>${donorLetter(part.donor, part.label)}</div><span class="t">${part.label}</span>
                  <button class="iconbtn" style="width:24px;height:24px" data-tip="Options" aria-label="Options" onClick=${(e) => { e.stopPropagation(); setMenuAt(menuAt === nk ? null : nk); }}><${Icon} n="more" size=${16} /></button></div>
                <${Menu} open=${menuAt === nk} origin="top-right" onClose=${() => setMenuAt(null)} style=${{ top: '38px', right: '8px', minWidth: '170px' }}>
                  <div class="mlabel">Glue rank</div>
                  ${GLUE.map((r) => html`<button class=${'mi' + ((g.glue_rank || null) === r ? ' on' : '')} onClick=${() => { edit((p) => { p.grafts[i].glue_rank = r; }); setMenuAt(null); }}><span>${r == null ? 'Full rank' : 'Rank ' + r}</span>${(g.glue_rank || null) === r && html`<${Icon} n="check" size=${14} />`}</button>`)}
                  <div class="msep"></div>
                  <button class="mi" style="color:var(--red)" onClick=${() => { edit((p) => { p.grafts.splice(i, 1); }); setMenuAt(null); setSel(null); }}><span>Remove</span><${Icon} n="trash" size=${14} /></button>
                <//>
                <div class="ds">${part.summary || part.donor}</div>
                <div class="tags"><span class="tag">at L${g.at}</span><span class="tag">glue ${g.glue_rank ? 'r' + g.glue_rank : 'full'}</span>${part.layers && html`<span class="tag c">donor L${part.layers[0]}${part.layers[1] !== part.layers[0] ? '-' + part.layers[1] : ''}</span>`}</div>
                <div class="nf"><span>+${((part.params_m || 0) / 1000).toFixed(2)}B params</span>${fit != null ? html`<span class=${'fitchip' + (fit < 0.45 ? ' low' : fit < 0.65 ? ' mid' : '')}>fit ${fit.toFixed(2)}</span>` : html`<span class="faint">not checked</span>`}</div>
              </div>`; })}
            ${program.taps.map((t, i) => { const nk = 't' + i; const part = tapPart(t); const fit = fits[tapKey(t)]; const ml = part && part.donor_layer === t.donor_layer ? measuredLine(part.measured) : null;
              return html`<div class=${'node tap' + (newKeys.current.has(nk) ? ' is-new' : '') + (dragId === nk ? ' dragging' : '') + (sel === nk ? ' sel' : '') + (menuAt === nk ? ' menu-open' : '')} key=${nk} ref=${(el) => { nodeRefs.current[nk] = el; }} style=${{ left: nodeX + 'px', top: (geo.y[nk] ?? 20) + 'px', zIndex: menuAt === nk ? 9 : 2 }} draggable="true"
                onDragStart=${(e) => startDrag(e, { kind: 'node', t: 't', i }, nk)} onDragEnd=${endDrag} onClick=${(e) => { e.stopPropagation(); setSel(nk); }}>
                <div class="nh"><div class="pic" style=${{ '--c': 'var(--blue)' }}>${donorLetter(t.donor)}</div><span class="t">Reads ${donorShort(t.donor)} L${t.donor_layer}</span>
                  <button class="iconbtn" style="width:24px;height:24px" data-tip="Options" aria-label="Options" onClick=${(e) => { e.stopPropagation(); setMenuAt(menuAt === nk ? null : nk); }}><${Icon} n="more" size=${16} /></button></div>
                <${Menu} open=${menuAt === nk} origin="top-right" onClose=${() => setMenuAt(null)} style=${{ top: '38px', right: '8px', minWidth: '190px' }}>
                  <div class="mlabel">Donor layer</div>
                  <div style="padding:2px 10px 6px"><input class="numin" type="number" min="0" max="200" step="1" value=${t.donor_layer} onChange=${(e) => { const v = Math.round(Number(e.target.value)); if (Number.isFinite(v) && v >= 0 && v <= 200 && v !== t.donor_layer) edit((p) => { p.taps[i].donor_layer = v; }); else e.target.value = t.donor_layer; }} /></div>
                  <div class="msep"></div>
                  <button class="mi" style="color:var(--red)" onClick=${() => { removeTap(i); setMenuAt(null); setSel(null); }}><span>Remove</span><${Icon} n="trash" size=${14} /></button>
                <//>
                <div class="ds">${ml || 'A small trained branch reads the donor hidden states.'}</div>
                <div class="tags"><span class="taptag">TAP</span><span class="tag">at L${t.at}</span><span class="tag c">donor L${t.donor_layer}</span></div>
                <div class="nf"><span>+${part && part.params_m != null ? part.params_m : 3}M trained</span>${typeof fit === 'number' ? html`<span class=${'fitchip' + (fit < 0.45 ? ' low' : fit < 0.65 ? ' mid' : '')}>fit ${fit.toFixed(2)}</span>` : html`<span class="faint">trains in Heal</span>`}</div>
              </div>`; })}
            ${!program.grafts.length && !program.taps.length && html`<div class="empty-hint" style=${{ left: nodeX - 60 + 'px' }}><b>Drag a skill onto a layer</b>It grafts a part of a donor model into the host at that layer. A tap reads a bigger model instead. A wire shows the link.</div>`}
          </div>
        </div>`}
        ${showCode && html`<div class="codepane">
          <div class="editor">
            <div class="gut"><div style=${{ transform: 'translateY(' + -scroll.y + 'px)' }}>${codeLines.map((_, i) => html`<div class=${codeErr && codeErr.line === i + 1 ? 'err' : ''}>${i + 1}</div>`)}</div></div>
            <div class="hlayer"><div style=${{ transform: 'translate(' + -scroll.x + 'px,' + -scroll.y + 'px)' }}>${codeLines.map((ln, i) => html`<div style=${{ height: '21.25px', background: codeErr && codeErr.line === i + 1 ? 'rgba(255,107,122,.14)' : '' }}>${tokenizeLine(ln).map((t) => (t.c ? html`<span class=${'tk-' + t.c}>${t.t}</span>` : t.t))}${ln === '' ? ' ' : ''}</div>`)}</div></div>
            <textarea class="over" spellcheck="false" wrap="off" value=${codeText} onInput=${(e) => onCode(e.target.value)} onScroll=${(e) => setScroll({ x: e.target.scrollLeft, y: e.target.scrollTop })}></textarea>
          </div>
          ${codeErr ? html`<div class="errline" ref=${errRef}>${codeErr.line != null ? 'Line ' + codeErr.line + ': ' : ''}${codeErr.message}</div>` : html`<div class="okline"><span>wsp text. Edits update the canvas.</span><span class="up">valid</span></div>`}
        </div>`}
      </div>
      <div class="cv-actions" style=${view === 'code' ? { display: 'none' } : {}}>
        ${job && html`<div class="jobline"><${Spinner} size=${13} /><span style="text-transform:capitalize">${job.action === 'compile' ? 'Checking fit' : job.action === 'heal' ? 'Healing' : job.action === 'train' ? 'Training' : 'Scoring'}${job.status === 'queued' ? ', queued' : ''}</span><div class="bigbar"><div class="fill" style=${{ width: jobPct + '%', transition: 'width 1s linear' }}></div></div></div>`}
        <div class="row">
          <button class=${'btn' + (job && job.action === 'compile' ? ' is-busy' : '')} ref=${(el) => { btnRefs.current.compile = el; }} disabled=${!!job} onClick=${() => run('compile')}>${job && job.action === 'compile' ? html`<${Matrix} variant="scan" />` : html`<${Icon} n="gear" size=${14} />`} Compile <span class="k">free fit check</span></button>
          <button class=${'btn' + (job && job.action === 'heal' ? ' is-busy' : '')} ref=${(el) => { btnRefs.current.heal = el; }} disabled=${!!job} onClick=${() => run('heal')}>${job && job.action === 'heal' ? html`<${Matrix} variant="pulse" />` : html`<${Icon} n="heal" size=${14} />`} Heal <span class="k">${healLabel}</span></button>
          <button class=${'btn' + (job && job.action === 'train' ? ' is-busy' : '')} ref=${(el) => { btnRefs.current.train = el; }} disabled=${!!job} onClick=${() => run('train')}>${job && job.action === 'train' ? html`<${Matrix} variant="orbit" />` : html`<${Icon} n="up" size=${14} />`} Train <span class="k">LoRA r${program.lora ? program.lora.rank : 16}</span></button>
          <button class=${'btn primary' + (job && job.action === 'score' ? ' is-busy' : '')} ref=${(el) => { btnRefs.current.score = el; }} disabled=${!!job} onClick=${() => run('score')}>${job && job.action === 'score' ? html`<${Matrix} variant="twinkle" />` : flash === 'score' ? html`<${SuccessCheck} size=${14} />` : html`<${Icon} n="play" size=${13} />`} Score <span class="k">1 eval</span></button>
        </div>
      </div>
    </section>

    <aside class="side">
      <div class="card ringcard">
        <${Ring} value=${prog} />
        <div style="min-width:0"><div class="label">${lastOk ? 'Attempt ' + lastOk.n + ' · average' : 'No score yet'}</div>
          <div class="num bigv">${html`<${PopNum} text=${lastOk ? pctS(avg(lastOk)) : '-'} />`}${lastOk && stubOf() && html`<${StubBadge} small />`}</div>
          <div class="dim" style="font-size:12px">${lastOk ? 'across ' + Object.keys(lastOk.scores).length + ' benchmarks' : 'Press Score to run one eval'}${prevOk ? html`<br /><span class=${avg(lastOk) >= avg(prevOk) ? 'up' : 'down'}>${sgn(avg(lastOk) - avg(prevOk))}</span> vs attempt ${prevOk.n}` : ''}</div></div>
      </div>

      <div class="card benchcard">
        <div class="bhead"><span class="label">Benchmarks</span><${BenchLegend} guard /></div>
        ${benches.map((b) => html`<div class="brow"><span class="nm" title=${b.name}>${b.name}</span><${BenchBar} b=${b} value=${lastOk ? lastOk.scores[b.id] : null} /><span class="num" style="text-align:right"><${PopNum} text=${lastOk && lastOk.scores[b.id] != null ? pct(lastOk.scores[b.id]) : '-'} /></span></div>`)}
        ${!benches.length && html`<div class="faint" style="margin:10px 0">Open the builder from a spec to see its benchmarks.</div>`}
        <div class="statgrid">
          <div class="stat"><div class="l">SIZE</div><b><${PopNum} text=${bsize(program.taps.length > 0 ? estSize : (lastOk && lastOk.params_b ? lastOk.params_b : estSize))} /></b></div>
          <div class="stat"><div class="l">SPEED</div><b class="faint">n/a</b></div>
          <div class="stat"><div class="l">VS RANDOM</div><b class=${vsCtl == null ? 'faint' : vsCtl > 0.02 ? 'up' : ''}><${PopNum} text=${vsCtl == null ? 'n/a' : sgn(vsCtl)} /></b></div>
        </div>
        ${program.taps.length > 0 && html`<div class="infoline">Taps train first (+${tapParamsM.toFixed(1)}M trained params), then LoRA or heal. Size counts the donors (${donorParamsB.toFixed(1)}B): they run on the prompt only, so decoding stays near host speed (run 7: 88 to 121 tok/s with a 7B tap, host 97).</div>`}
      </div>

      ${attempts.length > 0 && html`<div><div class="label" style="margin:2px 0 6px">Attempts this session</div>
        ${[...attempts].reverse().slice(0, 8).map((a) => html`<div class="attempt" title="Load this program" onClick=${() => { setProgram(clone(a.program)); setFits({}); }}><span class="n">${a.n}</span><span class="d">${briefProgram(a.program, cat)}</span><span class=${'num ' + (a.status && a.status !== 'ok' ? 'down' : '')}>${a.status && a.status !== 'ok' ? statusLabel(a.status) : t1 && a.scores ? pct(a.scores[t1.id]) : pctS(a.combined)}</span></div>`)}</div>`}

      <div style="display:flex;justify-content:space-between;align-items:center"><span class="label">Suggestions</span><button class="btn ghost sm" disabled=${sugBusy} onClick=${() => refreshSugs()}>${sugBusy ? html`<${Spinner} size=${11} />` : 'Refresh'}</button></div>
      ${sugBusy && !sugs && html`<div class="sugload"><${Spinner} size=${13} /> Thinking</div>`}
      ${(sugs || []).map((s) => html`<div class="sug"><b>${s.title}</b> <span class="dim">${s.detail}</span><div class="row"><${Tag} kind=${/likely/.test(s.tag) ? 'g' : /bigger/.test(s.tag) ? 'b' : ''}>${s.tag}<//><button class="btn sm" onClick=${() => { setProgram(clone(s.program)); setFits({}); toast('Applied. Undo with Ctrl or Cmd + Z.'); setSugs((x) => (x || []).filter((y) => y !== s)); }}>Apply</button></div></div>`)}
      ${sugs && !sugs.length && !sugBusy && html`<div class="faint">No suggestions right now.</div>`}
      <div class="sug"><b>Let the agents search from this draft.</b> <span class="dim">They start from your program and try many variations.</span><div class="row"><${Tag} kind="b">bigger change<//><button class="btn sm" disabled=${handing} onClick=${handoff}>Hand to agents</button></div></div>
    </aside>
  </div><//>`;

  function stubOf() { return !!(cat && cat.stub); }
}
