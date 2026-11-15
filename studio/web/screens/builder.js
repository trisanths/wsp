import { html, useState, useEffect, useLayoutEffect, useRef, useMemo } from '../lib.js';
import { api } from '../api.js';
import { go, toast, pct, pctS, sgn, bsize, clamp, clone, emptyProgram, maskIds, buildMask, donorColor, donorLetter, specProgress, primaryTarget, tokenizeLine } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Menu, BenchBar, BenchLegend, Ring, Tag, StubBadge } from '../components.js';

const GLUE = [null, 4, 8, 16, 32, 64];
const HEAL_STEPS = [0, 100, 200, 300, 400, 600, 800, 1000];
let DRAG = null; // what is being dragged right now

const ls = {
  get(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* storage can be off */ } },
};

function dragImage(e) {
  const src = e.currentTarget;
  const g = src.cloneNode(true);
  g.style.cssText = 'position:fixed;left:-1000px;top:0;width:' + src.offsetWidth + 'px;transform:rotate(-3deg);background:#151923;border:1px solid #7c8cff;border-radius:12px;box-shadow:0 18px 40px rgba(0,0,0,.55),0 0 0 4px rgba(124,140,255,.18);opacity:1;';
  document.body.appendChild(g);
  try { e.dataTransfer.setDragImage(g, 24, 20); } catch (x) { /* ignore */ }
  setTimeout(() => g.remove(), 0);
}

function Piece({ ic, color, t1, t2, used, dragging, onDragStart, onDragEnd, onClick }) {
  return html`<div class=${'piece' + (dragging ? ' dragging' : '') + (used ? ' used' : '')} draggable="true" onDragStart=${onDragStart} onDragEnd=${onDragEnd} onClick=${onClick} title="Drag onto the model, or click to add">
    <div class="pic" style=${{ background: color }}>${ic}</div><div style="min-width:0"><div class="t1">${t1}</div><div class="t2">${t2}</div></div>
    <span class="add">${used ? html`<${Icon} n="check" size=${14} />` : html`<${Icon} n="plus" size=${14} />`}</span></div>`;
}

function briefProgram(p, cat) {
  const bits = (p.grafts || []).map((g) => { const x = cat && cat.parts.find((q) => q.id === g.id); return (x ? x.label : g.id) + ' L' + g.at; });
  const ids = maskIds(p.masks);
  if (ids.length) bits.push('- ' + ids.length + ' mask' + (ids.length > 1 ? 's' : ''));
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
  const [geo, setGeo] = useState({ y: [], wires: [], h: 0 });
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
  useEffect(() => () => { alive.current = false; clearTimeout(parseTimer.current); }, []);

  /* ---- load ---- */
  const load = async () => {
    setErr(null);
    try {
      const [c, s] = await Promise.all([api.catalog(), specId ? api.getSpec(specId) : Promise.resolve(null)]);
      setCat(c); setSpec(s);
      let p = null;
      try { p = JSON.parse(ls.get('wsp.prog.' + id)); } catch (e) { p = null; }
      if (!p) { try { p = JSON.parse(ls.get('wsp.seed.' + id)); } catch (e) { p = null; } }
      setProgramRaw(p || emptyProgram(c.host.model));
    } catch (e) { setErr(e); }
  };
  useEffect(() => { load(); }, [id]);
  useEffect(() => { if (program) ls.set('wsp.prog.' + id, JSON.stringify(program)); }, [program]);
  useEffect(() => { ls.set('wsp.att.' + id, JSON.stringify(attempts.map((a) => ({ ...a })))); }, [attempts]);

  /* ---- edit helpers ---- */
  const setProgram = (p, keepUndo) => {
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
  const addMask = (m) => edit((p) => { const ids = maskIds(p.masks); if (!ids.includes(m.id)) p.masks = buildMask([...ids, m.id]); });
  const addTrain = (t) => edit((p) => { if (t.id === 'lora' && !p.lora) p.lora = { rank: 16 }; if (t.id === 'heal' && !p.heal) p.heal = healDefault(); });
  const trainUsed = (t) => program && (t.id === 'lora' ? !!program.lora : !!program.heal);

  /* ---- layers ---- */
  const N = cat ? cat.host.layers : 16;
  const slabs = useMemo(() => {
    if (!program) return [];
    const hot = new Set(program.grafts.map((g) => g.at));
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
    if (d.kind === 'part') { addPart(d.part, layer ?? undefined); if (layer == null) toast('Added at its default layer, L' + d.part.default_at + '. Drag the card to move it.'); }
    else if (d.kind === 'node') { if (layer != null) edit((p) => { p.grafts[d.i].at = layer; }); }
    else if (d.kind === 'mask') addMask(d.mask);
    else if (d.kind === 'train') addTrain(d.train);
  };
  const startDrag = (e, d, dragKey) => { DRAG = d; e.dataTransfer.effectAllowed = 'copyMove'; try { e.dataTransfer.setData('text/plain', dragKey); } catch (x) { /* ignore */ } dragImage(e); setTimeout(() => setDragId(dragKey), 0); };

  /* ---- geometry: node positions and wires from the real DOM ---- */
  useLayoutEffect(() => {
    const inner = innerRef.current;
    if (!inner || !program) return;
    const ir = inner.getBoundingClientRect();
    const items = program.grafts.map((g, i) => {
      const slab = inner.querySelector('[data-hot="' + g.at + '"]');
      if (!slab) return null;
      const r = slab.getBoundingClientRect();
      const node = nodeRefs.current[i];
      return { i, cy: r.top - ir.top + r.height / 2, x1: r.right - ir.left, h: node ? node.offsetHeight : 120 };
    }).filter(Boolean).sort((a, b) => a.cy - b.cy);
    const y = [];
    const wires = [];
    let bottom = 8;
    for (const it of items) {
      const top = Math.max(it.cy - 24, bottom + 12);
      y[it.i] = Math.round(top);
      bottom = top + it.h;
      const g = program.grafts[it.i];
      const part = cat && cat.parts.find((p) => p.id === g.id);
      wires.push({ i: it.i, x1: it.x1, y1: it.cy, x2: nodeX, y2: top + 24, color: donorColor(part && part.donor) });
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
        if (JSON.stringify(r.program) !== JSON.stringify(programRef.current)) { fromCode.current = true; setProgram(r.program); }
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
      if (j.status === 'error') { toast(j.error || 'The job failed', 'err'); setJob(null); return; }
      const r = j.result || {};
      if (r.fit) setFits(r.fit);
      if (action === 'compile') {
        toast('Fit check done' + (r.params_b ? ': ' + bsize(r.params_b) + ' parameters' : ''), 'ok');
      } else {
        const at = { n: 0, action, program: p, scores: r.scores, combined: r.combined, control: r.control, params_b: r.params_b, status: r.status, compute_s: r.compute_s };
        setAttempts((a) => [...a, { ...at, n: a.length + 1 }]);
        if (r.status && r.status !== 'ok') toast('This program collapsed a guard.', 'err');
        refreshSugs(p);
      }
    } catch (e) { toast(e.message, 'err'); }
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
  if (!cat || !program) return html`<${TopBar} /><div class="builder"><div class="palette"><div class="skeleton" style="height:420px"></div></div><div class="canvas"></div><div class="side"><div class="skeleton" style="height:240px"></div></div></div>`;

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
  const estSize = cat.host.params_b + program.grafts.reduce((s, g) => s + ((partById(g.id) || {}).params_m || 0) / 1000, 0);
  const matches = (x) => !q || (x.label + ' ' + (x.donor || '') + ' ' + (x.summary || '') + ' ' + x.group).toLowerCase().includes(q.toLowerCase());
  const addParts = cat.parts.filter(matches), rmParts = cat.masks.filter(matches), trParts = cat.train.filter(matches);
  const used = (pid) => program.grafts.some((g) => g.id === pid);
  const mIds = maskIds(program.masks);
  const healLabel = program.heal ? program.heal.steps + ' steps' : '300 steps';
  const cvClass = 'cv-body' + (view === 'split' ? ' split' : '');
  const showCanvas = view !== 'code';
  const showCode = view !== 'canvas';
  const hotAt = (l) => program.grafts.filter((g) => g.at === l);
  const dropping = (s) => over && over !== 'tail' && DRAG && DRAG.kind !== 'mask' && DRAG.kind !== 'train' && +over.slice(1) >= s.first && +over.slice(1) < s.first + s.n;
  const jobPct = job ? Math.min(94, ((Date.now() - job.t0) / 2600) * 100) : 0;
  const codeLines = codeText.replace(/\n$/, '').split('\n');

  const slabEl = (s) => {
    const label = s.n === 1 ? 'L' + s.first : 'L' + s.first + ' to L' + (s.first + s.n - 1);
    const hint = s.first === 0 ? 'embed + early' : s.first + s.n === N ? 'late + head' : '';
    const dr = dropping(s);
    const gs = hotAt(s.first);
    const dl = dr ? +over.slice(1) : null;
    return html`<div class=${'slab' + (s.hot ? ' port' : '') + (dr ? ' drop' : '') + (s.n > 1 && dr ? ' tall' : '')} data-first=${s.first} data-n=${s.n} data-hot=${s.hot ? s.first : null} key=${'s' + s.first}>
      <span class="nm">${dr ? 'Drop here: L' + dl : label}</span>
      ${s.hot && !dr ? html`<span style="color:var(--cyan);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">reads ${gs.map((g) => (partById(g.id) || { label: g.id }).label).join(', ')}</span>` : dr ? html`<span class="faint">release to graft</span>` : html`<span class="faint">${hint}</span>`}
    </div>`;
  };

  return html`<${TopBar} crumb=${spec ? spec.title : 'Builder'} right=${html`<button class="btn ghost sm" disabled=${!undo.length} onClick=${doUndo} title="Undo (Ctrl or Cmd + Z)"><${Icon} n="undo" size=${14} /> Undo</button><button class="btn sm" disabled=${handing} onClick=${handoff}>${handing ? html`<${Spinner} size=${12} />` : null} Hand to agents</button>`}>
      <span class="tag">draft ${attempts.length + 1}</span><span class="grow"></span>
      <span class="pill hide-sm">evals left <b class="num">${budget ? budget.evals_total - budget.evals_used + ' / ' + budget.evals_total : '-'}</b></span>
      <span class="pill hide-sm">compute <b class="num">${budget ? (budget.compute_used_s / 60).toFixed(budget.compute_used_s < 600 ? 1 : 0) + ' / ' + Math.round(budget.compute_total_s / 60) + ' min' : '-'}</b></span>
    <//>
  <div class="builder">
    <aside class="palette">
      <div class="searchbox"><${Icon} n="search" size=${15} /><input placeholder="Search skills, models, parts" value=${q} onInput=${(e) => setQ(e.target.value)} /></div>
      ${addParts.length > 0 && html`<div class="pgroup label">Add a skill</div>`}
      ${addParts.map((p) => html`<${Piece} key=${p.id} ic=${donorLetter(p.donor, p.label)} color=${donorColor(p.donor)} t1=${p.label} t2=${p.donor.split('/').pop() + ' · ' + p.kind + ' · L' + p.layers[0] + (p.layers[1] !== p.layers[0] ? '-' + p.layers[1] : '')} used=${used(p.id)} dragging=${dragId === p.id} onDragStart=${(e) => startDrag(e, { kind: 'part', part: p }, p.id)} onDragEnd=${endDrag} onClick=${() => addPart(p)} />`)}
      ${rmParts.length > 0 && html`<div class="pgroup label">Remove a skill</div>`}
      ${rmParts.map((m) => html`<${Piece} key=${m.id} ic="-" color="var(--bad)" t1=${m.label} t2=${m.summary} used=${mIds.includes(m.id)} dragging=${dragId === m.id} onDragStart=${(e) => startDrag(e, { kind: 'mask', mask: m }, m.id)} onDragEnd=${endDrag} onClick=${() => addMask(m)} />`)}
      ${trParts.length > 0 && html`<div class="pgroup label">Train</div>`}
      ${trParts.map((t) => html`<${Piece} key=${t.id} ic=${t.id === 'lora' ? 'R' : 'H'} color=${t.id === 'lora' ? 'var(--warn)' : 'var(--good)'} t1=${t.label} t2=${t.summary} used=${trainUsed(t)} dragging=${dragId === t.id} onDragStart=${(e) => startDrag(e, { kind: 'train', train: t }, t.id)} onDragEnd=${endDrag} onClick=${() => addTrain(t)} />`)}
      ${!addParts.length && !rmParts.length && !trParts.length && html`<div class="faint" style="margin-top:20px">Nothing matches "${q}".</div>`}
    </aside>

    <section class="canvas">
      <div class="ctool">
        <span class="seg2">${[['canvas', 'Canvas', 'grid'], ['code', 'Code', 'code'], ['split', 'Split', 'layers']].map(([k, l]) => html`<button class=${view === k ? 'on' : ''} onClick=${() => setView(k)}>${l}</button>`)}</span>
        ${showCanvas && html`<span class="seg2"><button class=${layerMode === 'simple' ? 'on' : ''} onClick=${() => setLayerMode('simple')}>Simple</button><button class=${layerMode === 'layers' ? 'on' : ''} onClick=${() => setLayerMode('layers')}>Layers</button></span>`}
      </div>
      <div class=${cvClass}>
        ${showCanvas && html`<div class="cv-scroll" ref=${scrollRef} onDragOver=${onDragOver} onDragLeave=${onDragLeave} onDrop=${onDrop} onClick=${() => { setSel(null); setMenuAt(null); }}>
          <div class=${'cv-inner' + (compact ? ' compact' : '')} ref=${innerRef} style=${{ minHeight: geo.h + 'px' }}>
            <svg class="wires" width=${nodeX + 260} height=${Math.max(geo.h, 300)}>
              <defs>${geo.wires.map((w) => html`<linearGradient id=${'wg' + w.i} gradientUnits="userSpaceOnUse" x1=${w.x1} y1="0" x2=${w.x2} y2="0"><stop offset="0" stop-color="#4fd1ff"/><stop offset="1" stop-color=${w.color}/></linearGradient>`)}</defs>
              ${geo.wires.map((w) => { const dx = (w.x2 - w.x1) * 0.55; const d = 'M' + w.x1 + ',' + w.y1 + ' C' + (w.x1 + dx) + ',' + w.y1 + ' ' + (w.x2 - dx) + ',' + w.y2 + ' ' + w.x2 + ',' + w.y2; return html`<g key=${w.i}><path d=${d} style=${{ d: "path('" + d + "')" }} stroke=${'url(#wg' + w.i + ')'} /><circle cx=${w.x1} cy=${w.y1} r="4.5" fill="#4fd1ff"/><circle cx=${w.x2} cy=${w.y2} r="4.5" fill=${w.color}/></g>`; })}
            </svg>
            <div class="stack" style=${{ marginLeft: '0px' }}>
              <span class="label">Host · ${cat.host.model.split('/').pop()}</span>
              ${slabs.map(slabEl)}
              ${mIds.map((mid) => html`<div class="slab mask" key=${mid}><span class="nm">- ${(maskById(mid) || { label: mid }).label} mask</span><span class="ctl"><span class="faint">all layers</span><button class="x" title="Remove mask" onClick=${(e) => { e.stopPropagation(); edit((p) => { p.masks = buildMask(maskIds(p.masks).filter((x) => x !== mid)); }); }}>×</button></span></div>`)}
              ${program.lora && html`<div class="slab lora"><span class="nm">LoRA</span><span class="ctl">rank <select value=${program.lora.rank || 16} onChange=${(e) => edit((p) => { p.lora.rank = +e.target.value; })}>${[1, 2, 4, 8, 16, 32, 64].map((r) => html`<option value=${r}>${r}</option>`)}</select><button class="x" title="Remove LoRA" onClick=${() => edit((p) => { p.lora = null; })}>×</button></span></div>`}
              ${program.heal && html`<div class="slab heal"><span class="nm">Heal</span><span class="ctl"><button title="Fewer steps" onClick=${() => edit((p) => { const i = HEAL_STEPS.indexOf(p.heal.steps); p.heal.steps = HEAL_STEPS[Math.max(1, (i < 0 ? 3 : i) - 1)]; })}>-</button><span class="num" style="min-width:62px;text-align:center">${program.heal.steps} steps</span><button title="More steps" onClick=${() => edit((p) => { const i = HEAL_STEPS.indexOf(p.heal.steps); p.heal.steps = HEAL_STEPS[Math.min(HEAL_STEPS.length - 1, (i < 0 ? 3 : i) + 1)]; })}>+</button><button class="x" title="Remove heal" onClick=${() => edit((p) => { p.heal = null; })}>×</button></span></div>`}
              <div class=${'slab tail' + (over === 'tail' && DRAG ? ' drop' : '')}>${over === 'tail' && DRAG ? 'Release to add' : 'Drop a mask, heal or LoRA here'}</div>
            </div>
            ${program.grafts.map((g, i) => { const part = partById(g.id) || { label: g.id, donor: '', kind: 'part', layers: [0, 0], params_m: 0 }; const key = g.id + '@' + g.at; const fit = fits[key]; const col = donorColor(part.donor);
              return html`<div class=${'node' + (dragId === 'n' + i ? ' dragging' : '') + (sel === i ? ' sel' : '') + (menuAt === i ? ' menu-open' : '')} key=${'n' + i} ref=${(el) => { nodeRefs.current[i] = el; }} style=${{ left: nodeX + 'px', top: (geo.y[i] ?? 20) + 'px', zIndex: menuAt === i ? 9 : 2 }} draggable="true"
                onDragStart=${(e) => startDrag(e, { kind: 'node', i }, 'n' + i)} onDragEnd=${endDrag} onClick=${(e) => { e.stopPropagation(); setSel(i); }}>
                <div class="nh"><div class="pic" style=${{ background: col }}>${donorLetter(part.donor, part.label)}</div><span class="t">${part.label}</span>
                  <button class="iconbtn" style="width:24px;height:24px" title="Options" onClick=${(e) => { e.stopPropagation(); setMenuAt(menuAt === i ? null : i); }}><${Icon} n="more" size=${16} /></button></div>
                <${Menu} open=${menuAt === i} onClose=${() => setMenuAt(null)} style=${{ top: '38px', right: '8px', minWidth: '170px' }}>
                  <div class="label" style="padding:4px 10px">Glue rank</div>
                  ${GLUE.map((r) => html`<button class=${'mi' + ((g.glue_rank || null) === r ? ' on' : '')} onClick=${() => { edit((p) => { p.grafts[i].glue_rank = r; }); setMenuAt(null); }}><span>${r == null ? 'Full rank' : 'Rank ' + r}</span>${(g.glue_rank || null) === r && html`<${Icon} n="check" size=${14} />`}</button>`)}
                  <div style="height:1px;background:var(--line);margin:4px 0"></div>
                  <button class="mi" style="color:var(--bad)" onClick=${() => { edit((p) => { p.grafts.splice(i, 1); }); setMenuAt(null); setSel(null); }}><span>Remove</span><${Icon} n="trash" size=${14} /></button>
                <//>
                <div class="ds">${part.summary || part.donor}</div>
                <div class="tags"><span class="tag">at L${g.at}</span><span class="tag">glue ${g.glue_rank ? 'r' + g.glue_rank : 'full'}</span>${part.layers && html`<span class="tag c">donor L${part.layers[0]}${part.layers[1] !== part.layers[0] ? '-' + part.layers[1] : ''}</span>`}</div>
                <div class="nf"><span>+${((part.params_m || 0) / 1000).toFixed(2)}B params</span>${fit != null ? html`<span class=${'fitchip' + (fit < 0.45 ? ' low' : fit < 0.65 ? ' mid' : '')}>fit ${fit.toFixed(2)}</span>` : html`<span class="faint">not checked</span>`}</div>
              </div>`; })}
            ${!program.grafts.length && html`<div class="empty-hint" style=${{ left: nodeX - 60 + 'px' }}><b>Drag a skill onto a layer</b>It grafts a part of a donor model into the host at that layer. A wire shows the link.</div>`}
          </div>
        </div>`}
        ${showCode && html`<div class="codepane">
          <div class="editor">
            <div class="gut"><div style=${{ transform: 'translateY(' + -scroll.y + 'px)' }}>${codeLines.map((_, i) => html`<div class=${codeErr && codeErr.line === i + 1 ? 'err' : ''}>${i + 1}</div>`)}</div></div>
            <div class="hlayer"><div style=${{ transform: 'translate(' + -scroll.x + 'px,' + -scroll.y + 'px)' }}>${codeLines.map((ln, i) => html`<div style=${{ height: '21.25px', background: codeErr && codeErr.line === i + 1 ? 'rgba(255,107,122,.14)' : '' }}>${tokenizeLine(ln).map((t) => (t.c ? html`<span class=${'tk-' + t.c}>${t.t}</span>` : t.t))}${ln === '' ? ' ' : ''}</div>`)}</div></div>
            <textarea class="over" spellcheck="false" wrap="off" value=${codeText} onInput=${(e) => onCode(e.target.value)} onScroll=${(e) => setScroll({ x: e.target.scrollLeft, y: e.target.scrollTop })}></textarea>
          </div>
          ${codeErr ? html`<div class="errline">${codeErr.line != null ? 'Line ' + codeErr.line + ': ' : ''}${codeErr.message}</div>` : html`<div class="okline"><span>wsp text. Edits update the canvas.</span><span class="up">valid</span></div>`}
        </div>`}
      </div>
      <div class="cv-actions" style=${view === 'code' ? { display: 'none' } : {}}>
        ${job && html`<div class="jobline"><${Spinner} size=${13} /><span style="text-transform:capitalize">${job.action === 'compile' ? 'Checking fit' : job.action === 'heal' ? 'Healing' : job.action === 'train' ? 'Training' : 'Scoring'}${job.status === 'queued' ? ', queued' : ''}</span><div class="bigbar"><div class="fill" style=${{ width: jobPct + '%', transition: 'width 1s linear' }}></div></div></div>`}
        <div class="row">
          <button class="btn" disabled=${!!job} onClick=${() => run('compile')}><${Icon} n="gear" size=${14} /> Compile <span class="k">free fit check</span></button>
          <button class="btn" disabled=${!!job} onClick=${() => run('heal')}><${Icon} n="heal" size=${14} /> Heal <span class="k">${healLabel}</span></button>
          <button class="btn" disabled=${!!job} onClick=${() => run('train')}><${Icon} n="up" size=${14} /> Train <span class="k">LoRA r${program.lora ? program.lora.rank : 16}</span></button>
          <button class="btn primary" disabled=${!!job} onClick=${() => run('score')}><${Icon} n="play" size=${13} /> Score <span class="k">1 eval</span></button>
        </div>
      </div>
    </section>

    <aside class="side">
      <div class="card" style="padding:14px;display:flex;gap:14px;align-items:center">
        <${Ring} value=${prog} />
        <div style="min-width:0"><div class="label">${lastOk ? 'Attempt ' + lastOk.n + ' · average' : 'No score yet'}</div>
          <div style="font-size:24px;font-weight:700;margin-top:2px" class="num">${lastOk ? pctS(avg(lastOk)) : '-'}${lastOk && stubOf() && html`<${StubBadge} small />`}</div>
          <div class="dim" style="font-size:12px">${lastOk ? 'across ' + Object.keys(lastOk.scores).length + ' benchmarks' : 'Press Score to run one eval'}${prevOk ? html`<br /><span class=${avg(lastOk) >= avg(prevOk) ? 'up' : 'down'}>${sgn(avg(lastOk) - avg(prevOk))}</span> vs attempt ${prevOk.n}` : ''}</div></div>
      </div>

      <div class="card" style="padding:12px 14px">
        <div style="display:flex;justify-content:space-between;align-items:center"><span class="label">Benchmarks</span><${BenchLegend} guard /></div>
        ${benches.map((b) => html`<div class="brow"><span class="nm" title=${b.name}>${b.name}</span><${BenchBar} b=${b} value=${lastOk ? lastOk.scores[b.id] : null} /><span class="num" style="text-align:right">${lastOk && lastOk.scores[b.id] != null ? pct(lastOk.scores[b.id]) : '-'}</span></div>`)}
        ${!benches.length && html`<div class="faint" style="margin:10px 0">Open the builder from a spec to see its benchmarks.</div>`}
        <div class="statgrid">
          <div class="stat"><div class="l">SIZE</div><b>${bsize(lastOk && lastOk.params_b ? lastOk.params_b : estSize)}</b></div>
          <div class="stat"><div class="l">SPEED</div><b class="faint">n/a</b></div>
          <div class="stat"><div class="l">VS RANDOM</div><b class=${vsCtl == null ? 'faint' : vsCtl > 0.02 ? 'up' : ''}>${vsCtl == null ? 'n/a' : sgn(vsCtl)}</b></div>
        </div>
      </div>

      ${attempts.length > 0 && html`<div><div class="label" style="margin:2px 0 6px">Attempts this session</div>
        ${[...attempts].reverse().slice(0, 8).map((a) => html`<div class="attempt" title="Load this program" onClick=${() => { setProgram(clone(a.program)); setFits({}); }}><span class="n">${a.n}</span><span class="d">${briefProgram(a.program, cat)}</span><span class=${'num ' + (a.status && a.status !== 'ok' ? 'down' : '')}>${a.status && a.status !== 'ok' ? 'collapsed' : t1 && a.scores ? pct(a.scores[t1.id]) : pctS(a.combined)}</span></div>`)}</div>`}

      <div style="display:flex;justify-content:space-between;align-items:center"><span class="label">Suggestions</span><button class="btn ghost sm" disabled=${sugBusy} onClick=${() => refreshSugs()}>${sugBusy ? html`<${Spinner} size=${11} />` : 'Refresh'}</button></div>
      ${sugBusy && !sugs && html`<div class="sugload"><${Spinner} size=${13} /> Thinking</div>`}
      ${(sugs || []).map((s) => html`<div class="sug"><b>${s.title}</b> <span class="dim">${s.detail}</span><div class="row"><${Tag} kind=${/likely/.test(s.tag) ? 'g' : /bigger/.test(s.tag) ? 'a' : ''}>${s.tag}<//><button class="btn sm" onClick=${() => { setProgram(clone(s.program)); setFits({}); toast('Applied. Undo with Ctrl or Cmd + Z.'); setSugs((x) => (x || []).filter((y) => y !== s)); }}>Apply</button></div></div>`)}
      ${sugs && !sugs.length && !sugBusy && html`<div class="faint">No suggestions right now.</div>`}
      <div class="sug"><b>Let the agents search from this draft.</b> <span class="dim">They start from your program and try many variations.</span><div class="row"><${Tag} kind="a">bigger change<//><button class="btn sm" disabled=${handing} onClick=${handoff}>Hand to agents</button></div></div>
    </aside>
  </div>`;

  function stubOf() { return !!(cat && cat.stub); }
}
