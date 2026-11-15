// Brain3D: a readable 3D picture of a network being edited. Uses three.js 0.170.0 (loaded on first use).
//
// The host is a stack of thin layer plates (layer 0 at the bottom). A grafted part is a solid block beside the
// stack with a label card, joined to its layer by one curved wire. A deletion mask is a red notch in its plate.
// Animations play as a queue of single steps (pick, probe, splice, heal, score, cut), one at a time, and a step
// bar under the scene names the current one in plain words.
//
// Brain3D({ height = 360, layers = 16, parts = [], cuts = [], pulse = null, scan = null, mode = 'host',
//           highlight = null, label = null, title = null, model = null, size = null, result = null,
//           steps = auto, idle = null, onPick = null, class })
//   layers     number of host layers. Many layers are grouped (for example 4 layers per plate).
//   parts      [{ id, label, kind, at, donor, verdict, size?, fit? }] grafted parts docked beside layer `at`.
//              kind: block | mlp_cluster | attn_heads | expert | inserted_layer.
//              verdict: 'contributes' (green) | 'no better than random' (grey) | 'harmful' (red) | null (donor tint).
//              size: a short text such as "2.1M params". fit: the probe number shown as "fit 0.81".
//   cuts       [{ layer, frac, label }] deletion masks: a red notch cut out of the plate (frac sets its width).
//   pulse      { kind, key, ... }. When key changes, steps are queued and play one at a time.
//              kind: pick | probe | splice | heal | score | cut (one step each), graft (pick, probe, splice),
//              compile (graft and cut), scan (a score climb without a readout), settle (jump to the finished
//              picture), seq (uses `steps`: an array of step ids).
//              partId limits part steps to one part, at names a layer, ok colors the score (true green, false red,
//              null blue), fit is the probe number for a pulse with no part, readout is the score text
//              (for example "GSM8K 66.0%, +3.8 over random parts").
//   result     { text, ok } the last score readout, shown at the top and kept until a new run starts.
//   mode       'host' or 'donor'. In donor mode `scan` = { layer, total, found: [{ layer, kind, label, id }] } moves
//              a highlight up the plates ("Scanning layer 12 of 48"); found parts pop out to the side.
//   title      the model name over the stack. model gives its logo, size a text like "7.6B parameters".
//   label      a small line under the title (or the title when none is given).
//   steps      true shows the step bar, false hides it (default: on for host mode at 300px or taller).
//   idle       the sentence shown in the step bar when nothing is playing.
//   highlight  a layer index to emphasize. onPick(partId): a part card was clicked.
// Colors come from the theme tokens and follow theme changes. The scene pauses offscreen and in hidden tabs,
// caps the pixel ratio at 2, shows a still labeled frame under prefers-reduced-motion, and falls back to a
// labeled 2D SVG drawing when WebGL or the library is unavailable. Drag to look around; it eases back.

import { html, useEffect, useRef, useState } from './lib.js';
import { logoFor, ModelLogo } from './logos.js';

const THREE_URL = 'https://esm.sh/three@0.170.0';
let threeP = null;
const loadThree = () => (threeP || (threeP = import(THREE_URL).catch((e) => { threeP = null; throw e; })));

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (u) => 1 - Math.pow(1 - clamp(u, 0, 1), 3);
const easeIO = (u) => { u = clamp(u, 0, 1); return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; };
const easeBack = (u) => { u = clamp(u, 0, 1); const c = 1.5; return 1 + (c + 1) * Math.pow(u - 1, 3) + c * Math.pow(u - 1, 2); };
const shortName = (s) => String(s || '').split('/').pop();

/* ---------- the scene's measurements (world units) ---------- */
const W = 3.9, D = 1.6, T = 0.12;
const MAXS = 24;            // most plates
const COLS = 15, ROWS = 5;  // neuron dots per plate

const KIND_NAME = { block: 'Block', mlp_cluster: 'MLP cluster', attn_heads: 'Attention heads', expert: 'Expert', inserted_layer: 'Inserted layer' };
const KIND_TOKEN = { block: 'blue', mlp_cluster: 'green', attn_heads: 'amber', expert: 'purple', inserted_layer: 'blue' };
const VERDICT_TOKEN = { contributes: 'green', 'no better than random': 'grey', harmful: 'red' };
const VERDICT_WORD = { contributes: 'helps', 'no better than random': 'no better than random', harmful: 'hurts' };

export const STEP_DEFS = [
  { id: 'pick', name: 'Pick part' }, { id: 'probe', name: 'Probe fit' }, { id: 'splice', name: 'Splice in' },
  { id: 'heal', name: 'Heal' }, { id: 'score', name: 'Score' }, { id: 'cut', name: 'Cut' },
];
const BASE_DUR = { pick: 1.4, probe: 1.5, splice: 1.7, heal: 1.8, score: 2.1, cut: 1.7 };

/* ---------- plates: many layers share a plate ---------- */
function plateLayout(layers) {
  const L = Math.max(1, Math.round(layers) || 1);
  const gs = Math.max(1, Math.ceil(L / 20));
  const n = Math.ceil(L / gs);
  const sp = n <= 6 ? 1.0 : n <= 10 ? 0.8 : 0.62;
  const plates = Array.from({ length: n }, (_, i) => ({ i, lo: i * gs, hi: Math.min(L - 1, (i + 1) * gs - 1), y: i * sp }));
  return { L, gs, n, sp, plates, Hs: (n - 1) * sp, of: (l) => clamp(Math.floor(clamp(l, 0, L - 1) / gs), 0, n - 1) };
}
const plateName = (p) => (p.lo === p.hi ? 'L' + p.lo : 'L' + p.lo + '-' + p.hi);

/* ---------- colors from the page ---------- */
function parseColor(str) {
  const nums = (str || '').match(/-?\d*\.?\d+(?:e-?\d+)?/g);
  if (!nums || nums.length < 3) return null;
  const n = nums.map(Number);
  if (/^color\(/.test(str)) return [n[0], n[1], n[2]];
  return [n[0] / 255, n[1] / 255, n[2] / 255];
}
function makeResolver(root) {
  const probe = document.createElement('span');
  probe.style.cssText = 'position:absolute;width:0;height:0;visibility:hidden;pointer-events:none;transition:none !important;animation:none !important';
  root.appendChild(probe);
  const resolve = (v, fb) => {
    probe.style.color = '';
    probe.style.color = v;
    let c = parseColor(getComputedStyle(probe).color);
    if (!c && fb) { probe.style.color = ''; probe.style.color = fb; c = parseColor(getComputedStyle(probe).color); }
    return c || [0.5, 0.5, 0.5];
  };
  return { resolve, dispose: () => probe.remove() };
}
const mix3 = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

/* ---------- the plan: a pulse becomes a list of single steps ---------- */
function partWord(p) { return p ? '"' + (p.label || p.id) + '"' : 'the part'; }
function planSteps(pulse, ctx) {
  const parts = ctx.parts || [], cuts = ctx.cuts || [];
  const k = pulse.kind;
  let ids;
  if (k === 'seq') ids = (pulse.steps || []).slice();
  else if (k === 'graft') ids = ['pick', 'probe', 'splice'];
  else if (k === 'compile') ids = ['pick', 'probe', 'splice', 'cut'];
  else if (k === 'scan') ids = ['score'];
  else ids = [k];
  const one = pulse.partId != null ? parts.filter((x) => x.id === pulse.partId) : [];
  const sel = one.length ? one : parts;
  const out = [];
  for (const id of ids) {
    if (id === 'settle') { out.push({ id: 'settle', dur: 0 }); continue; }
    if (!BASE_DUR[id]) continue;
    const st = { id, parts: [], cuts: [], at: pulse.at, ok: pulse.ok == null ? null : pulse.ok, fit: pulse.fit, readout: pulse.readout || null, silent: k === 'scan' };
    let n = 1;
    if (id === 'pick' || id === 'splice') {
      if (!sel.length) continue;
      st.parts = sel.map((x) => x.id); n = sel.length;
      st.text = n === 1 ? (id === 'pick' ? 'Picking ' + partWord(sel[0]) + (sel[0].donor ? ' from ' + shortName(sel[0].donor) : ' from a donor model') + '.' : 'Splicing ' + partWord(sel[0]) + ' into layer ' + sel[0].at + ' of the host.')
        : (id === 'pick' ? 'Picking ' + n + ' parts from their donor models.' : 'Splicing the ' + n + ' parts into the host, each at its own layer.');
    } else if (id === 'probe') {
      st.parts = sel.map((x) => x.id); n = Math.max(1, sel.length);
      if (sel.length === 1) st.text = 'Checking how well ' + partWord(sel[0]) + ' fits layer ' + sel[0].at + '.';
      else if (sel.length) st.text = 'Checking how well each part fits the layer it will go into.';
      else st.text = 'Checking how well a part fits layer ' + (pulse.at == null ? '' : pulse.at) + '.';
      if (!sel.length && pulse.at == null) st.at = Math.floor((ctx.layers || 16) / 2);
    } else if (id === 'heal') {
      st.parts = sel.map((x) => x.id); n = Math.max(1, sel.length);
      st.text = sel.length > 1 ? 'Healing the seams. A short training pass smooths each join.' : 'Healing the seam. A short training pass smooths the join' + (pulse.at != null || sel.length ? ' at layer ' + (sel.length ? sel[0].at : pulse.at) : '') + '.';
    } else if (id === 'score') {
      st.text = 'Running the benchmark through the whole stack, bottom to top.';
    } else if (id === 'cut') {
      let cs = cuts.map((c, i) => ({ c, i }));
      if (pulse.at != null) { const f = cs.filter((x) => Math.round(x.c.layer) === Math.round(pulse.at)); if (f.length) cs = f; }
      if (!cs.length) continue;
      st.cuts = cs.map((x) => x.i); n = cs.length;
      st.text = n === 1 ? 'Cutting out ' + (cs[0].c.label ? '"' + cs[0].c.label + '"' : 'these units') + ' from layer ' + Math.round(cs[0].c.layer) + '.' : 'Cutting out ' + n + ' sets of units, one notch per plate.';
    }
    st.dur = Math.min(2.7, BASE_DUR[id] * (1 + 0.12 * (n - 1)));
    out.push(st);
  }
  return out;
}

/* ---------- the sequencer: one step at a time, never overlapping ---------- */
function createSequencer(onInfo) {
  const q = [];
  const s = { cur: null, u: 0, busy: false, done: [], hooks: null, instant: false, getCtx: null };
  let doneSet = [];
  const emit = (text) => onInfo({ cur: s.cur ? s.cur.id : null, text: s.cur ? s.cur.text : (text || ''), done: doneSet.slice(), dur: s.cur ? s.cur.dur : 0, key: (s.cur ? s.cur.n : 0) });
  let counter = 0;
  const start = (st) => { s.cur = st; s.u = 0; st.n = ++counter; if (s.hooks) s.hooks.start(st); emit(); };
  const end = () => {
    const st = s.cur;
    if (st && st.id !== 'settle' && !st.silent && doneSet.indexOf(st.id) < 0) doneSet.push(st.id);
    if (s.hooks && st) s.hooks.end(st);
    s.cur = null;
    if (q.length) start(q.shift()); else { s.busy = false; emit(''); }
  };
  s.enqueue = (pulse) => {
    const steps = planSteps(pulse, s.getCtx());
    if (!steps.length) return false;
    if (!s.busy && !q.length && steps.some((st) => doneSet.indexOf(st.id) >= 0)) doneSet = [];
    for (const st of steps) q.push(st);
    while (q.length > 14) q.shift();
    if (s.instant) {
      while (q.length) { const st = q.shift(); s.cur = st; st.n = ++counter; if (s.hooks) { s.hooks.start(st); s.hooks.end(st); } if (st.id !== 'settle' && !st.silent && doneSet.indexOf(st.id) < 0) doneSet.push(st.id); s.cur = null; }
      s.busy = false; emit('');
      return true;
    }
    if (!s.busy) { s.busy = true; start(q.shift()); }
    return true;
  };
  s.tick = (dt) => {
    if (!s.cur) return false;
    if (s.cur.dur <= 0) { end(); return s.busy; }
    s.u += dt / s.cur.dur;
    if (s.u >= 1) { s.u = 1; end(); }
    return s.busy;
  };
  s.clear = () => { q.length = 0; s.cur = null; s.busy = false; };
  return s;
}
// the window of step progress in which item i of n is active
function win(i, n, u) {
  if (n <= 1) return clamp(u, 0, 1);
  const w = Math.max(0.4, 1 / (1 + 0.6 * (n - 1)));
  const off = (1 - w) / (n - 1);
  return clamp((u - i * off) / w, 0, 1);
}

/* ---------- shaders ---------- */
const DOT_VS = `
attribute float aSlab; attribute float aCutIn; attribute float aSeed;
uniform float uBase[${MAXS}]; uniform float uGlow[${MAXS}]; uniform float uCut[${MAXS}];
uniform float uTime, uPx, uLight;
uniform vec3 uCNeuron, uCBlue, uCGlow;
varying vec4 vCol;
void main() {
  int s = int(aSlab + 0.5);
  float b = uBase[s]; float g = uGlow[s]; float cut = uCut[s] * aCutIn;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 col = mix(uCNeuron, uCBlue, clamp(b, 0.0, 1.0) * 0.8);
  col = mix(col, uCGlow, clamp(g, 0.0, 1.0));
  float size = 0.098 * (1.0 + clamp(g, 0.0, 1.0) * 0.8 + b * 0.25);
  float a = (0.62 + 0.3 * clamp(g + b * 0.4, 0.0, 1.0)) * (1.0 - cut);
  gl_PointSize = max(2.2, size * uPx / -mv.z);
  gl_Position = projectionMatrix * mv;
  vCol = vec4(col, clamp(a, 0.0, 1.0));
}`;
const DOT_FS = `
varying vec4 vCol;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  if (d > 1.0) discard;
  gl_FragColor = vec4(vCol.rgb, vCol.a * smoothstep(1.0, 0.72, d));
}`;

/* ---------- small DOM helpers ---------- */
function mk(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function logoNode(donor, size) {
  const logo = logoFor(donor);
  const span = mk('span', 'mlogo' + (logo ? '' : ' mlogo-letter'));
  span.style.width = span.style.height = size + 'px';
  if (logo) {
    const img = document.createElement('img');
    img.src = logo.src; img.alt = ''; img.width = img.height = size; img.decoding = 'async';
    if (logo.mono) img.className = 'mlogo-mono';
    span.appendChild(img);
  } else {
    span.style.fontSize = Math.round(size * 0.55) + 'px';
    span.textContent = (shortName(donor).replace(/^[^a-z0-9]+/i, '').charAt(0) || '?').toUpperCase();
  }
  return span;
}
function declutter(items, gap, lo, hi) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) if (items[i].y - items[i - 1].y < gap) items[i].y = items[i - 1].y + gap;
  const over = items.length ? items[items.length - 1].y - hi : 0;
  if (over > 0) { for (let i = items.length - 1; i >= 0; i--) { items[i].y -= over; if (i > 0 && items[i].y - items[i - 1].y < gap) continue; } }
  if (items.length && items[0].y < lo) { const sh = lo - items[0].y; items.forEach((x) => { x.y += sh; }); }
}

/* ---------- the 3D engine ---------- */
function createEngine(THREE, els, propsRef, seq, reduced) {
  const { gl, ov } = els;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  renderer.setClearColor(0x000000, 0);
  gl.appendChild(renderer.domElement);
  const canvas = renderer.domElement;
  canvas.className = 'b3d-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.5, 200);
  const amb = new THREE.AmbientLight(0xffffff, 0.8 * Math.PI);
  const sun = new THREE.DirectionalLight(0xffffff, 0.45 * Math.PI);
  sun.position.set(0.32, 0.83, 0.46);
  scene.add(amb, sun);
  const rs = makeResolver(gl);
  let C = null;
  const world = new THREE.Group();
  scene.add(world);

  let L = null;                 // plate layout
  let mode = 'host';
  let sig = '';
  let cw = 100, ch = 100, dpr = 1;
  let time = 0, visible = true, raf = 0, last = 0, disposed = false;
  let az0 = 17, el0 = 19, dAz = 0, dEl = 0, dragging = false;
  const objs = [];              // things to dispose on rebuild
  let parts = [], cuts = [], found = [];
  const pst = new Map();        // part runtime state, kept across rebuilds
  const cutOpen = new Map();    // cut key -> 0..1 (1 is a finished notch)
  let beam = 0;                 // eased scan position (plate index)
  let readout = null, pendingResult = null, resultText = null, readoutShow = 0;
  let glowCol = [0, 0.5, 1];
  const dyn = { tags: [], foundCards: new Map(), counts: [], beam: null };
  const baseG = new Float32Array(MAXS), glowG = new Float32Array(MAXS), cutU = new Float32Array(MAXS);
  let instMesh = null, instColors = null, edgeLines = null, dotMat = null, dotPts = null, foundMesh = null;
  let ghostGeom = null;
  const vec = new THREE.Vector3();
  const tmpM = new THREE.Matrix4(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3();
  const tmpC = new THREE.Color();
  const col3 = (a) => tmpC.setRGB(a[0], a[1], a[2], THREE.SRGBColorSpace).clone();

  /* overlay elements */
  const ticksEl = mk('div', 'b3d-ticklayer');
  const cardsEl = mk('div', 'b3d-cardlayer');
  ov.append(ticksEl, cardsEl);
  const readEl = mk('div', 'b3d-readout'); readEl.style.opacity = '0';
  const countEl = mk('div', 'b3d-counters'); countEl.style.display = 'none';
  const legendEl = mk('div', 'b3d-legend');
  ov.append(readEl, countEl, legendEl);
  let ticks = [], cards = [], cutCards = [], extraEls = [];

  function resolveColors() {
    const r = (v, fb) => rs.resolve(v, fb);
    const bg = r('var(--bg)', '#fff');
    const lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2];
    C = {
      bg, light: lum > 0.5 ? 1 : 0, t1: r('var(--text-1)'), t3: r('var(--text-3)'), blue: r('var(--blue)'), green: r('var(--green)'), red: r('var(--red)'),
      amber: r('var(--amber)'), purple: r('var(--purple)'), grey: r('var(--text-3)'),
    };
    C.plate = mix3(bg, C.t1, lum > 0.5 ? 0.075 : 0.2);
    C.edge = mix3(bg, C.t1, lum > 0.5 ? 0.3 : 0.45);
    C.neuron = mix3(bg, C.t1, lum > 0.5 ? 0.5 : 0.62);
  }
  const tok = (name) => (C[name] || C.grey);
  function partColor(p) {
    if (p.verdict && VERDICT_TOKEN[p.verdict]) return tok(VERDICT_TOKEN[p.verdict]);
    return tok('purple');
  }
  const cssRgb = (c) => 'rgb(' + Math.round(c[0] * 255) + ',' + Math.round(c[1] * 255) + ',' + Math.round(c[2] * 255) + ')';

  /* ---------- build the structure for the current props ---------- */
  function disposeObjs() {
    for (const o of objs) {
      world.remove(o);
      o.traverse && o.traverse((x) => { if (x.geometry && !x.userData.shared) x.geometry.dispose(); if (x.material) { (Array.isArray(x.material) ? x.material : [x.material]).forEach((m) => m.dispose()); } });
    }
    objs.length = 0;
    ticks.forEach((t) => t.el.remove()); cards.forEach((c) => c.el.remove()); cutCards.forEach((c) => c.el.remove()); extraEls.forEach((e) => e.remove());
    ticks = []; cards = []; cutCards = []; extraEls = [];
    dyn.foundCards.forEach((c) => c.el.remove()); dyn.foundCards.clear();
    dyn.tags = []; dyn.counts.forEach((e) => e.remove()); dyn.counts = []; dyn.beam = null; generic.el = null;
    if (ghostGeom) { ghostGeom.dispose(); ghostGeom = null; }
    foundMesh = null; instMesh = null; edgeLines = null; dotPts = null; dotMat = null;
  }
  function roundedPoly(shape, pts, rs_) {
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const p = pts[i], a = pts[(i + n - 1) % n], b = pts[(i + 1) % n], r = rs_[i];
      const va = [a[0] - p[0], a[1] - p[1]], vb = [b[0] - p[0], b[1] - p[1]];
      const la = Math.hypot(va[0], va[1]), lb = Math.hypot(vb[0], vb[1]);
      const p1 = [p[0] + va[0] / la * r, p[1] + va[1] / la * r], p2 = [p[0] + vb[0] / lb * r, p[1] + vb[1] / lb * r];
      if (i === 0) shape.moveTo(p1[0], p1[1]); else shape.lineTo(p1[0], p1[1]);
      if (r > 0) shape.quadraticCurveTo(p[0], p[1], p2[0], p2[1]); else shape.lineTo(p2[0], p2[1]);
    }
    shape.closePath();
  }
  function plateGeom(notch) {
    const shape = new THREE.Shape();
    const x0 = -W / 2, x1 = W / 2, y0 = -D / 2, y1 = D / 2, R = 0.16;
    if (!notch) roundedPoly(shape, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], [R, R, R, R]);
    else {
      const nw = notch.w, nd = notch.d;
      roundedPoly(shape, [[x0 + nw, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0 + nd], [x0 + nw, y0 + nd]], [0.04, R, R, R, 0.04, 0.03]);
    }
    const g = new THREE.ExtrudeGeometry(shape, { depth: T, bevelEnabled: false, curveSegments: 5 });
    g.rotateX(-Math.PI / 2);
    g.translate(0, -T / 2, 0);
    return g;
  }
  const notchOf = (c) => ({ w: W * clamp(0.2 + 0.5 * (c.frac == null ? 0.35 : c.frac), 0.2, 0.62), d: D * 0.56 });

  function build(p) {
    disposeObjs();
    mode = p.mode === 'donor' ? 'donor' : 'host';
    L = plateLayout(p.layers || 16);
    parts = (mode === 'host' ? (p.parts || []) : []).slice(0, 14).map((x) => ({ ...x, at: clamp(Math.round(x.at || 0), 0, L.L - 1) }));
    cuts = (mode === 'host' ? (p.cuts || []) : []).slice(0, 8).map((c) => ({ ...c, layer: clamp(Math.round(c.layer || 0), 0, L.L - 1) }));
    const n = L.n;
    const cutBy = new Map(); cuts.forEach((c, i) => { const pi = L.of(c.layer); if (!cutBy.has(pi)) cutBy.set(pi, i); });

    /* plates: one instanced mesh for plain plates, separate meshes for notched ones */
    const baseG0 = plateGeom(null);
    baseG0.userData.shared = false;
    const plainIdx = []; for (let i = 0; i < n; i++) if (!cutBy.has(i)) plainIdx.push(i);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide });
    instMesh = new THREE.InstancedMesh(baseG0, mat, Math.max(1, plainIdx.length));
    instMesh.count = plainIdx.length;
    plainIdx.forEach((pi, k) => { tmpM.makeTranslation(0, L.plates[pi].y, 0); instMesh.setMatrixAt(k, tmpM); instMesh.setColorAt(k, col3(C.plate)); });
    instMesh.instanceMatrix.needsUpdate = true;
    instMesh.userData.plain = plainIdx;
    world.add(instMesh); objs.push(instMesh);
    // outlines of the plain plates in one draw call
    const eg = new THREE.EdgesGeometry(baseG0, 25);
    const ea = eg.getAttribute('position').array, ec = ea.length;
    const merged = new Float32Array(ec * plainIdx.length);
    plainIdx.forEach((pi, k) => { for (let j = 0; j < ec; j += 3) { merged[k * ec + j] = ea[j]; merged[k * ec + j + 1] = ea[j + 1] + L.plates[pi].y; merged[k * ec + j + 2] = ea[j + 2]; } });
    const mg = new THREE.BufferGeometry(); mg.setAttribute('position', new THREE.BufferAttribute(merged, 3));
    eg.dispose();
    edgeLines = new THREE.LineSegments(mg, new THREE.LineBasicMaterial({ color: col3(C.edge), transparent: true, opacity: 0.7 }));
    world.add(edgeLines); objs.push(edgeLines);

    cuts.forEach((c, i) => {
      const pi = L.of(c.layer);
      if (cutBy.get(pi) !== i) return;
      const nt = notchOf(c);
      c._pi = pi; c._nt = nt;
      c._closed = new THREE.Group(); c._open = new THREE.Group();
      [[c._closed, null], [c._open, nt]].forEach(([grp, nn]) => {
        const g = plateGeom(nn);
        const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: col3(C.plate), side: THREE.DoubleSide }));
        const e = new THREE.LineSegments(new THREE.EdgesGeometry(g, 25), new THREE.LineBasicMaterial({ color: col3(C.edge), transparent: true, opacity: 0.7 }));
        grp.add(m, e); grp.position.y = L.plates[pi].y; world.add(grp); objs.push(grp);
        grp.userData.mat = m.material;
      });
      const gg = new THREE.BoxGeometry(nt.w, T, nt.d);
      const gm = new THREE.Mesh(gg, new THREE.MeshBasicMaterial({ color: col3(C.red), transparent: true, opacity: 0.4, depthWrite: false }));
      const ge = new THREE.LineSegments(new THREE.EdgesGeometry(gg), new THREE.LineBasicMaterial({ color: col3(C.red), transparent: true, opacity: 0.9 }));
      c._ghost = new THREE.Group(); c._ghost.add(gm, ge); c._ghost.userData.fill = gm.material; c._ghost.userData.line = ge.material;
      c._gx = -W / 2 + nt.w / 2; c._gz = D / 2 - nt.d / 2;
      c._ghost.position.set(c._gx, L.plates[pi].y, c._gz);
      world.add(c._ghost); objs.push(c._ghost);
      if (!cutOpen.has(c.layer + ':' + c.label)) cutOpen.set(c.layer + ':' + c.label, 1);
      c._k = c.layer + ':' + c.label;
    });

    /* neuron dots: one Points object for every plate */
    const per = COLS * ROWS, total = per * n;
    const pos = new Float32Array(total * 3), aS = new Float32Array(total), aC = new Float32Array(total), aSeed = new Float32Array(total);
    for (let pi = 0; pi < n; pi++) {
      const c = cuts.find((x) => x._pi === pi);
      for (let r = 0; r < ROWS; r++) for (let q = 0; q < COLS; q++) {
        const k = pi * per + r * COLS + q;
        const x = lerp(-W / 2 + 0.34, W / 2 - 0.34, q / (COLS - 1)), z = lerp(-D / 2 + 0.3, D / 2 - 0.3, r / (ROWS - 1));
        pos[k * 3] = x; pos[k * 3 + 1] = L.plates[pi].y + T / 2 + 0.006; pos[k * 3 + 2] = z;
        aS[k] = pi; aSeed[k] = ((k * 2654435761) >>> 0) / 4294967296;
        aC[k] = c && x < -W / 2 + c._nt.w + 0.02 && z > D / 2 - c._nt.d - 0.02 ? 1 : 0;
      }
    }
    const dg = new THREE.BufferGeometry();
    dg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    dg.setAttribute('aSlab', new THREE.BufferAttribute(aS, 1));
    dg.setAttribute('aCutIn', new THREE.BufferAttribute(aC, 1));
    dg.setAttribute('aSeed', new THREE.BufferAttribute(aSeed, 1));
    dotMat = new THREE.ShaderMaterial({
      vertexShader: DOT_VS, fragmentShader: DOT_FS, transparent: true, depthWrite: false,
      uniforms: { uBase: { value: baseG }, uGlow: { value: glowG }, uCut: { value: cutU }, uTime: { value: 0 }, uPx: { value: 600 }, uLight: { value: 0 },
        uCNeuron: { value: new THREE.Vector3(...C.neuron) }, uCBlue: { value: new THREE.Vector3(...C.blue) }, uCGlow: { value: new THREE.Vector3(...C.blue) } },
    });
    dotPts = new THREE.Points(dg, dotMat); dotPts.frustumCulled = false;
    world.add(dotPts); objs.push(dotPts);

    /* labels at the left of each plate */
    L.plates.forEach((pl) => { const el = mk('div', 'b3d-tick', plateName(pl)); ticksEl.appendChild(el); ticks.push({ el, pl, on: false }); });

    /* parts: a block, a curved wire and a label card each */
    const order = parts.map((x, i) => i).sort((a, b) => L.plates[L.of(parts[a].at)].y - L.plates[L.of(parts[b].at)].y);
    const gap = parts.length > 1 ? Math.min(0.95, Math.max(0.5, (L.Hs + 0.6) / parts.length)) : 0;
    const ys = parts.map((x) => L.plates[L.of(x.at)].y);
    order.forEach((idx, j) => { if (j > 0) ys[idx] = Math.max(ys[idx], ys[order[j - 1]] + gap); });
    const over = order.length ? ys[order[order.length - 1]] - (L.Hs + 0.3) : 0;
    if (over > 0) for (let j = order.length - 1; j >= 0; j--) { ys[order[j]] -= over; if (j < order.length - 1) ys[order[j]] = Math.min(ys[order[j]], ys[order[j + 1]] - gap); }
    parts.forEach((pt, i) => {
      const col = partColor(pt);
      const pi = L.of(pt.at);
      const S = new THREE.Vector3(W / 2 - 0.05, L.plates[pi].y, 0.0);
      const H = new THREE.Vector3(W / 2 + 1.85, ys[i], 0.15);
      const E = new THREE.Vector3(H.x + 1.5, H.y + 0.35, 0.3);
      const M = new THREE.Vector3(W / 2 + 0.8, (S.y + H.y) / 2 + (H.y >= S.y ? -0.12 : 0.12), 0.55);
      const curve = new THREE.CatmullRomCurve3([S, M, H, E], false, 'centripetal');
      let uH = 0.7, best = 1e9;
      for (let k = 0; k <= 60; k++) { const u = k / 60; const d = curve.getPointAt(u).distanceTo(H); if (d < best) { best = d; uH = u; } }
      const tube = new THREE.TubeGeometry(curve, 40, 0.03, 5, false);
      const wmat = new THREE.MeshBasicMaterial({ color: col3(col), transparent: true, opacity: 0.9 });
      const wire = new THREE.Mesh(tube, wmat); wire.frustumCulled = false;
      const bg = new THREE.BoxGeometry(pt.kind === 'inserted_layer' ? 1.15 : 0.9, pt.kind === 'inserted_layer' ? 0.2 : 0.44, 0.66);
      const bm = new THREE.MeshLambertMaterial({ color: col3(col) });
      const block = new THREE.Mesh(bg, bm);
      const bl = new THREE.LineSegments(new THREE.EdgesGeometry(bg), new THREE.LineBasicMaterial({ color: col3(mix3(col, [0, 0, 0], 0.45)), transparent: true, opacity: 0.8 }));
      block.add(bl);
      const grp = new THREE.Group(); grp.add(wire, block); world.add(grp); objs.push(grp);
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.5, 0.58, 48), new THREE.MeshBasicMaterial({ color: col3(C.blue), transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
      ring.rotation.x = -Math.PI / 2; ring.position.set(0, L.plates[pi].y + T / 2 + 0.02, 0); ring.visible = false;
      world.add(ring); objs.push(ring);
      // the card
      const el = mk(onPickable() ? 'button' : 'div', 'b3d-card');
      el.style.setProperty('--c', cssRgb(col));
      el.appendChild(logoNode(pt.donor || pt.label, 18));
      const tx = mk('span', 'b3d-card-t');
      tx.append(mk('b', null, pt.label || pt.id));
      const sub = ['layer ' + pt.at, pt.size || '', pt.verdict ? VERDICT_WORD[pt.verdict] : ''].filter(Boolean).join(' · ');
      tx.append(mk('i', null, sub));
      el.appendChild(tx);
      if (pt.verdict) el.classList.add('v-' + (VERDICT_TOKEN[pt.verdict] || 'grey'));
      el.addEventListener('click', () => { const f = propsRef.current.onPick; if (f) f(pt.id); });
      cardsEl.appendChild(el);
      const tag = mk('div', 'b3d-tag'); tag.style.opacity = '0'; cardsEl.appendChild(tag);
      let st = pst.get(pt.id);
      if (!st) { st = { mode: 'docked', u: uH, vis: 0, pop: 0.001 }; pst.set(pt.id, st); }
      cards.push({ el, tag, pt, st, curve, uH, wire, wmat, block, grp, ring, S, H, tubeCount: tube.index.count, nameEl: tx });
    });

    /* cut cards */
    cuts.forEach((c) => {
      if (!c._ghost) return;
      const el = mk('div', 'b3d-cutcard'); el.append(mk('b', null, 'Removed'), document.createTextNode(c.label ? ': ' + c.label : ''));
      cardsEl.appendChild(el); cutCards.push({ el, c });
    });

    /* donor: found parts */
    if (mode === 'donor') {
      const g = new THREE.BoxGeometry(0.3, 0.2, 0.3);
      foundMesh = new THREE.InstancedMesh(g, new THREE.MeshLambertMaterial({ color: 0xffffff }), 260);
      foundMesh.count = 0; foundMesh.frustumCulled = false;
      world.add(foundMesh); objs.push(foundMesh);
      foundList.length = 0; foundSeen.clear();
    }
    legend();
    for (let i = 0; i < MAXS; i++) { baseG[i] = 0; glowG[i] = 0; cutU[i] = 0; }
    setupDone = true;
    layoutCamera();
  }
  const foundList = [], foundSeen = new Map();
  let setupDone = false;
  const onPickable = () => !!propsRef.current.onPick;

  function legend() {
    legendEl.textContent = '';
    const items = mode === 'host'
      ? [['dot', 'Host neurons', cssRgb(C.neuron)], ['sq', 'Donor part', cssRgb(C.purple)], ['cut', 'Removed', cssRgb(C.red)], ['dot', 'Signal', cssRgb(C.blue)]]
      : [['dot', 'Donor neurons', cssRgb(C.neuron)], ['sq', 'Part found', cssRgb(C.green)], ['dot', 'Scan', cssRgb(C.blue)]];
    items.forEach(([k, t, c]) => { const s = mk('span', 'b3d-lg'); const sw = mk('i', 'b3d-sw ' + k); sw.style.background = k === 'cut' ? 'transparent' : c; if (k === 'cut') { sw.style.borderColor = c; sw.style.background = c.replace('rgb', 'rgba').replace(')', ',0.3)'); } s.append(sw, document.createTextNode(t)); legendEl.appendChild(s); });
  }

  /* ---------- camera ---------- */
  let target = new THREE.Vector3(), dist = 20;
  function layoutCamera() {
    if (!L) return;
    const compact = ch < 250;
    const hasSide = parts.length > 0 || mode === 'donor';
    const leftPx = (compact ? 30 : 44) + (cuts.length ? (compact ? 78 : 96) : 0);
    const rightPx = hasSide ? Math.min(compact ? 150 : 262, cw * 0.42) : 24;
    const topPx = compact ? 30 : 62, botPx = compact ? 10 : 34;
    const elR = el0 * Math.PI / 180;
    const content = { w: W + (hasSide ? 2.4 : 0.3), h: L.Hs * Math.cos(elR) + D * Math.sin(elR) + T + 0.5 };
    const s = Math.max(6, Math.min((cw - leftPx - rightPx) / content.w, (ch - topPx - botPx) / content.h));
    dist = (ch / 2) / (s * Math.tan(camera.fov * Math.PI / 360));
    const cx = -W / 2 + content.w / 2 - 0.1, cy = L.Hs / 2;
    target.set(cx - ((leftPx - rightPx) / 2) / s, cy + ((topPx - botPx) / 2) / s, 0);
    camera.near = Math.max(0.5, dist * 0.3); camera.far = dist * 4;
    camera.aspect = cw / Math.max(1, ch);
    camera.updateProjectionMatrix();
    if (dotMat) dotMat.uniforms.uPx.value = ch * dpr / (2 * Math.tan(camera.fov * Math.PI / 360));
  }
  function placeCamera() {
    const sway = reduced ? 0 : 1;
    const az = (az0 + dAz + Math.sin(time * 0.45) * 2.6 * sway) * Math.PI / 180;
    const el = (el0 + dEl + Math.sin(time * 0.33) * 0.9 * sway) * Math.PI / 180;
    camera.position.set(target.x + dist * Math.sin(az) * Math.cos(el), target.y + dist * Math.sin(el), target.z + dist * Math.cos(az) * Math.cos(el));
    camera.lookAt(target);
    camera.updateMatrixWorld();
  }

  /* ---------- per-frame state ---------- */
  function stepEffects(dt) {
    glowG.fill(0);
    cutU.fill(0);
    const cur = seq.cur, u = seq.u;
    glowCol = C.blue;
    // plates that hold a docked part glow softly
    baseG.fill(0);
    cards.forEach((c) => { if (c.st.mode !== 'hidden') baseG[L.of(c.pt.at)] = Math.max(baseG[L.of(c.pt.at)], 0.55 * Math.min(1, c.st.vis)); });
    const hi = propsRef.current.highlight;
    if (hi != null && hi >= 0) baseG[L.of(hi)] = Math.max(baseG[L.of(hi)], 0.9);
    // parts
    cards.forEach((c, i) => {
      const st = c.st;
      if (st.pop > 0 && st.pop < 1) { st.pop = Math.min(1, st.pop + dt / 0.6); st.vis = easeBack(st.pop); }
      if (st.flash > 0) st.flash = Math.max(0, st.flash - dt * 0.7);
      if (cur && cur.parts && cur.parts.indexOf(c.pt.id) >= 0) {
        const j = cur.parts.indexOf(c.pt.id), n = cur.parts.length, w = win(j, n, u);
        if (cur.id === 'pick') { st.mode = w > 0 ? 'shelf' : 'hidden'; st.u = 1; st.vis = easeBack(w); st.tent = 1; }
        else if (cur.id === 'splice') { st.mode = 'moving'; st.vis = 1; st.u = lerp(1, c.uH, easeIO(w)); if (w >= 1) st.flash = 1; }
        else if (cur.id === 'probe') {
          const pl = L.of(c.pt.at), b = Math.sin(Math.PI * clamp(w, 0, 1));
          glowG[pl] = Math.max(glowG[pl], 0.9 * b);
        } else if (cur.id === 'heal') {
          const pl = L.of(c.pt.at);
          for (let k = -3; k <= 3; k++) { const q = pl + k; if (q < 0 || q >= L.n) continue; const reach = clamp((w * 1.4 - 0.4) * 4, 0, 4); const wv = Math.exp(-Math.pow((Math.abs(k) - reach * 0.75) / 0.9, 2)) * (w > 0.3 ? 1 : 0) * (1 - clamp((w - 0.85) / 0.15, 0, 1) * 0.7); glowG[q] = Math.max(glowG[q], wv * 0.85); }
        }
      }
      if (st.flash > 0) glowG[L.of(c.pt.at)] = Math.max(glowG[L.of(c.pt.at)], st.flash * 0.9);
    });
    if (cur && cur.id === 'score') {
      const climb = u * (L.n + 2) - 1;
      for (let i = 0; i < L.n; i++) glowG[i] = Math.max(glowG[i], Math.exp(-Math.pow((i - climb) / 1.1, 2)) * 0.95 + (u > 0.84 ? (1 - Math.abs(u - 0.92) / 0.08) * 0.0 : 0));
      if (u > 0.72) { const okc = cur.ok === true ? C.green : cur.ok === false ? C.red : C.blue; glowCol = mix3(C.blue, okc, clamp((u - 0.72) / 0.12, 0, 1)); const f = clamp((u - 0.72) / 0.14, 0, 1) * (1 - clamp((u - 0.9) / 0.1, 0, 1) * 0.6); for (let i = 0; i < L.n; i++) glowG[i] = Math.max(glowG[i], f * 0.8); }
    }
    if (cur && cur.id === 'probe' && !cur.parts.length && cur.at != null) { glowG[L.of(cur.at)] = Math.max(glowG[L.of(cur.at)], 0.9 * Math.sin(Math.PI * u)); }
    if (cur && cur.id === 'heal' && !cur.parts.length && cur.at != null) { const pl = L.of(cur.at); for (let k = -3; k <= 3; k++) { const q = pl + k; if (q >= 0 && q < L.n) glowG[q] = Math.max(glowG[q], Math.exp(-Math.pow((Math.abs(k) - u * 3) / 0.9, 2)) * 0.8); } }
    // cuts: closed -> notch opens
    cuts.forEach((c) => {
      if (!c._ghost) return;
      const o = cutOpen.get(c._k) == null ? 1 : cutOpen.get(c._k);
      const ci = cuts.indexOf(c);
      const active = cur && cur.id === 'cut' && cur.cuts.indexOf(ci) >= 0;
      const w = active ? win(cur.cuts.indexOf(ci), cur.cuts.length, u) : (o >= 1 ? 1 : 0);
      c._anim = w;
      const showOpen = active ? w >= 0.5 : o >= 1;
      c._open.visible = showOpen; c._closed.visible = !showOpen;
      let gy = L.plates[c._pi].y, fill = 0.4, sc = 1;
      if (active) {
        if (w < 0.5) { sc = lerp(0.35, 1, ease(w / 0.5)); fill = 0.35 + 0.5 * (w / 0.5); gy += T * 0.5 + 0.04; glowG[c._pi] = Math.max(glowG[c._pi], 0.5); }
        else { const d = ease((w - 0.5) / 0.5); gy += lerp(T * 0.5 + 0.04, -0.12, d); fill = lerp(0.85, 0.4, d); }
      }
      c._ghost.position.set(c._gx, gy, c._gz); c._ghost.scale.set(sc, 1, sc);
      c._ghost.userData.fill.opacity = fill;
      cutU[c._pi] = showOpen ? 1 : 0;
    });
    // donor scan
    if (mode === 'donor') {
      const sc = propsRef.current.scan || { layer: 0, total: 1, found: [] };
      const plate = sc.layer >= sc.total ? L.n + 1 : clamp(sc.layer / L.gs, 0, L.n - 1);
      beam += (plate - beam) * Math.min(1, dt * 3.2);
      if (reduced) beam = plate;
      const done = sc.layer >= sc.total;
      for (let i = 0; i < L.n; i++) {
        if (i < beam - 0.5) baseG[i] = Math.max(baseG[i], 0.28);
        glowG[i] = Math.max(glowG[i], done ? 0 : Math.exp(-Math.pow((i - beam) / 0.55, 2)));
      }
      // found items: pop in
      updateFound(dt, sc);
    }
    // wire, block, ring, tag
    const stp = cur && cur.id;
    cards.forEach((c) => {
      const st = c.st, visible = st.mode !== 'hidden' && st.vis > 0.01;
      c.grp.visible = visible;
      if (!visible) { c.ring.visible = false; return; }
      const uu = st.mode === 'docked' ? c.uH : st.u;
      const p = c.curve.getPointAt(clamp(uu, 0, 1));
      c.block.position.copy(p);
      const sc = Math.max(0.001, st.vis);
      c.block.scale.setScalar(sc);
      c.wire.geometry.setDrawRange(0, Math.max(0, Math.round(clamp(uu, 0, 1) * 40)) * 30);
      const tentative = st.mode === 'shelf' || (st.mode === 'moving' && stp === 'splice' && uu > c.uH + 0.05);
      c.wmat.opacity = tentative ? 0.35 : 0.92;
      // probe ring and tag
      let ping = 0;
      if (cur && cur.id === 'probe' && cur.parts.indexOf(c.pt.id) >= 0) ping = win(cur.parts.indexOf(c.pt.id), cur.parts.length, u);
      c.ping = ping;
      c.ring.visible = ping > 0 && ping < 1;
      if (c.ring.visible) { const r = 0.6 + ping * 1.9; c.ring.scale.set(r, r, r); c.ring.material.opacity = (1 - ping) * 0.85; }
    });
    // the generic ping for a probe with no part
    if (cur && cur.id === 'probe' && !cur.parts.length) generic.ping = u; else generic.ping = 0;
  }
  const generic = { ping: 0 };

  function updateFound(dt, sc) {
    const now = time;
    (sc.found || []).forEach((f) => {
      const key = f.id != null ? String(f.id) : f.layer + ':' + f.label + ':' + f.kind;
      if (!foundSeen.has(key)) {
        const pl = L.of(f.layer || 0);
        const slot = foundList.filter((x) => x.pl === pl).length;
        if (foundList.length >= 250) return;
        const idx = foundList.length;
        const born = reduced || foundSeen.size === 0 && (sc.found || []).length > 6 ? -10 : now;
        const rec = { key, f, pl, slot, idx, born, color: tok(KIND_TOKEN[f.kind] || 'green') };
        foundSeen.set(key, rec); foundList.push(rec);
        foundMesh.setColorAt(idx, col3(rec.color));
        foundMesh.count = foundList.length;
      }
    });
    let any = false;
    foundList.forEach((r) => {
      const t = clamp((now - r.born) / 0.9, 0, 1);
      if (t < 1) any = true;
      const x0 = W / 2, y = L.plates[r.pl].y;
      const col = r.slot % 5, row = Math.floor(r.slot / 5) % 3;
      const tx = W / 2 + 0.55 + col * 0.36, tz = -0.5 + row * 0.5;
      const e = ease(t);
      tmpP.set(lerp(x0 - 0.4, tx, e), y + 0.15 + Math.sin(Math.PI * t) * 0.5, lerp(0, tz, e));
      const s = t <= 0 ? 0.001 : easeBack(t);
      tmpS.set(s, s, s);
      tmpM.compose(tmpP, tmpQ.identity(), tmpS);
      foundMesh.setMatrixAt(r.idx, tmpM);
    });
    if (foundMesh.instanceColor) foundMesh.instanceColor.needsUpdate = true;
    foundMesh.instanceMatrix.needsUpdate = true;
    foundActive = any;
  }
  let foundActive = false;

  /* ---------- colors on the plates (follow glow) ---------- */
  function paintPlates() {
    const plain = instMesh.userData.plain;
    plain.forEach((pi, k) => {
      const g = clamp(glowG[pi] * 0.4 + baseG[pi] * 0.16, 0, 0.6);
      const c = mix3(C.plate, glowG[pi] > baseG[pi] * 0.5 ? glowCol : C.blue, g);
      instMesh.setColorAt(k, tmpC.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace));
    });
    if (instMesh.instanceColor) instMesh.instanceColor.needsUpdate = true;
    cuts.forEach((c) => {
      if (!c._ghost) return;
      const g = clamp(glowG[c._pi] * 0.4 + baseG[c._pi] * 0.16, 0, 0.6), cc = mix3(C.plate, C.blue, g);
      c._closed.userData.mat.color.setRGB(cc[0], cc[1], cc[2], THREE.SRGBColorSpace);
      c._open.userData.mat.color.setRGB(cc[0], cc[1], cc[2], THREE.SRGBColorSpace);
    });
    dotMat.uniforms.uCGlow.value.set(glowCol[0], glowCol[1], glowCol[2]);
    dotMat.uniforms.uTime.value = time;
  }

  /* ---------- the HTML labels follow the 3D positions ---------- */
  const proj = (v) => { vec.copy(v).project(camera); return { x: (vec.x * 0.5 + 0.5) * cw, y: (-vec.y * 0.5 + 0.5) * ch, z: vec.z }; };
  function layoutOverlay() {
    const compact = ch < 250;
    // plate ticks
    if (ticks.length) {
      const a = proj(vec.set(-W / 2, L.plates[0].y, 0)), b = ticks.length > 1 ? proj(vec.set(-W / 2, L.plates[1].y, 0)) : { y: a.y - 40 };
      const dy = Math.abs(a.y - b.y) || 40;
      const stride = Math.max(1, Math.ceil((compact ? 15 : 17) / dy));
      ticks.forEach((t, i) => {
        const show = (i % stride === 0 || i === ticks.length - 1) && !(i !== ticks.length - 1 && ticks.length - 1 - i < stride * 0.6) || t.on;
        const p = proj(vec.set(-W / 2 - 0.02, t.pl.y, D * 0.5));
        t.el.style.opacity = show ? '1' : '0';
        t.el.style.transform = 'translate3d(' + (p.x - 6).toFixed(1) + 'px,' + p.y.toFixed(1) + 'px,0) translate(-100%,-50%)';
        const on = glowG[t.pl.i] > 0.35 || (propsRef.current.highlight != null && L.of(propsRef.current.highlight) === t.pl.i) || (mode === 'donor' && Math.abs(beam - t.pl.i) < 0.5 && (propsRef.current.scan || {}).layer < (propsRef.current.scan || {}).total);
        if (on !== t.on) { t.on = on; t.el.classList.toggle('on', on); }
      });
    }
    // part cards
    const items = [];
    cards.forEach((c) => {
      const st = c.st;
      const vis = st.mode === 'hidden' ? 0 : clamp(st.vis, 0, 1);
      const bp = c.block.position;
      const p = proj(vec.set(bp.x + 0.5 * Math.max(0.2, st.vis), bp.y, bp.z));
      c.px = p.x + 12; c.py = p.y; c.vis = vis;
      if (vis > 0.02) items.push(c);
      // fit tag near the plate
      let tg = 0;
      if (c.ping > 0 && c.ping < 1.0) tg = clamp(Math.min(c.ping * 5, (1 - c.ping) * 6 + 0.25), 0, 1);
      if (seq.cur && seq.cur.id === 'probe' && seq.cur.parts.indexOf(c.pt.id) >= 0 && c.ping >= 1) tg = 1;
      c.tagV = tg;
    });
    if (items.length) {
      const arr = items.map((c) => ({ c, y: c.py }));
      declutter(arr, compact ? 34 : 50, 24, ch - 40);
      arr.forEach((o) => { o.c.py2 = o.y; });
    }
    cards.forEach((c) => {
      const on = c.vis > 0.02;
      c.el.style.opacity = on ? String(c.vis) : '0';
      c.el.style.pointerEvents = on && c.vis > 0.9 && onPickable() ? 'auto' : 'none';
      if (on) c.el.style.transform = 'translate3d(' + c.px.toFixed(1) + 'px,' + (c.py2 == null ? c.py : c.py2).toFixed(1) + 'px,0) translate(0,-50%)';
      const sp = proj(vec.set(W / 2 + 0.05, c.S.y, 0.1));
      const fit = c.pt.fit != null ? c.pt.fit : (seq.cur && seq.cur.id === 'probe' && seq.cur.fit != null ? seq.cur.fit : null);
      const txt = fit != null && !isNaN(fit) ? 'fit ' + Number(fit).toFixed(2) : 'checking fit';
      if (c.tag._t !== txt) { c.tag.textContent = txt; c.tag._t = txt; }
      c.tag.style.opacity = String(c.tagV || 0);
      c.tag.style.transform = 'translate3d(' + (sp.x + 12).toFixed(1) + 'px,' + (sp.y - 16).toFixed(1) + 'px,0) translate(0,-50%)';
    });
    // cut labels
    cutCards.forEach((cc) => {
      const c = cc.c;
      const p = proj(vec.set(c._gx - c._nt.w / 2 - 0.02, c._ghost.position.y, c._gz));
      const w = c._anim == null ? 1 : c._anim;
      const cutting = seq.cur && seq.cur.id === 'cut' && seq.cur.cuts.indexOf(cuts.indexOf(c)) >= 0;
      const v = cutting ? clamp((w - 0.4) / 0.3, 0, 1) : 1;
      cc.el.style.opacity = String(v);
      cc.el.style.transform = 'translate3d(' + (p.x - 54).toFixed(1) + 'px,' + (p.y + 4).toFixed(1) + 'px,0) translate(-100%,-50%)';
    });
    // the generic probe tag (no part)
    if (!generic.el) { generic.el = mk('div', 'b3d-tag'); generic.el.style.opacity = '0'; cardsEl.appendChild(generic.el); extraEls.push(generic.el); }
    if (generic.ping > 0 && seq.cur && seq.cur.at != null) {
      const sp = proj(vec.set(W / 2 + 0.05, L.plates[L.of(seq.cur.at)].y, 0.1));
      generic.el.textContent = seq.cur.fit != null ? 'fit ' + Number(seq.cur.fit).toFixed(2) : 'checking fit';
      generic.el.style.opacity = String(clamp(Math.min(generic.ping * 5, 1), 0, 1));
      generic.el.style.transform = 'translate3d(' + (sp.x + 12).toFixed(1) + 'px,' + (sp.y - 16).toFixed(1) + 'px,0) translate(0,-50%)';
    } else generic.el.style.opacity = '0';
    // readout
    readoutAnim();
    // donor counters and found cards
    if (mode === 'donor') donorOverlay(compact);
  }
  function readoutAnim() {
    const showing = readout && readoutShow > 0.01;
    readEl.style.opacity = showing ? String(readoutShow) : '0';
    if (readout && readEl._t !== readout.text) {
      readEl.textContent = ''; readEl.className = 'b3d-readout ' + (readout.ok === true ? 'ok' : readout.ok === false ? 'bad' : '');
      readEl.append(mk('i', 'b3d-readout-dot'), document.createTextNode(readout.text)); readEl._t = readout.text;
    }
  }
  function donorOverlay(compact) {
    const sc = propsRef.current.scan || { layer: 0, total: 1, found: [] };
    const total = sc.total || L.L;
    const scanned = clamp(Math.floor(sc.layer), 0, total);
    const nf = (sc.found || []).length;
    const txt = 'Layers scanned ' + scanned + ' of ' + total + '  ·  Parts found ' + nf;
    countEl.style.display = compact ? 'none' : '';
    if (countEl._t !== txt) { countEl.textContent = txt; countEl._t = txt; }
    // recent found cards
    const recent = foundList.slice(-4);
    const keep = new Set(recent.map((r) => r.key));
    dyn.foundCards.forEach((c, k) => { if (!keep.has(k)) { c.el.remove(); dyn.foundCards.delete(k); } });
    const arr = [];
    recent.forEach((r) => {
      let c = dyn.foundCards.get(r.key);
      if (!c) {
        const el = mk('div', 'b3d-card small'); el.style.setProperty('--c', cssRgb(r.color));
        const tx = mk('span', 'b3d-card-t'); tx.append(mk('b', null, r.f.label || 'Part'), mk('i', null, (KIND_NAME[r.f.kind] || 'Part') + ' · layer ' + (r.f.layer == null ? '?' : r.f.layer)));
        el.appendChild(tx); cardsEl.appendChild(el); c = { el }; dyn.foundCards.set(r.key, c);
      }
      const col = r.slot % 5, row = Math.floor(r.slot / 5) % 3;
      const t = clamp((time - r.born) / 0.9, 0, 1);
      const p = proj(vec.set(W / 2 + 0.55 + 4 * 0.36 + 0.4, L.plates[r.pl].y + 0.15, -0.5 + row * 0.5));
      arr.push({ c, y: p.y, x: p.x, t });
    });
    declutter(arr, compact ? 36 : 48, 20, ch - 36);
    arr.forEach((o) => { o.c.el.style.opacity = String(clamp((o.t - 0.5) * 2, 0, 1)); o.c.el.style.transform = 'translate3d(' + o.x.toFixed(1) + 'px,' + o.y.toFixed(1) + 'px,0) translate(0,-50%)'; });
    // a per-plate count when a plate holds many parts
    const per = new Map(); foundList.forEach((r) => per.set(r.pl, (per.get(r.pl) || 0) + 1));
    while (dyn.counts.length < per.size) { const e = mk('div', 'b3d-count'); cardsEl.appendChild(e); dyn.counts.push(e); }
    let k = 0;
    per.forEach((n, pl) => {
      const e = dyn.counts[k++];
      e.textContent = n > 5 ? '+' + (n - 5) + ' more' : '';
      const p = proj(vec.set(W / 2 + 0.55 + 5 * 0.36 + 0.1, L.plates[pl].y, 0));
      e.style.opacity = n > 5 && !arr.some((o) => Math.abs(o.y - p.y) < 28) ? '1' : '0';
      e.style.transform = 'translate3d(' + p.x.toFixed(1) + 'px,' + p.y.toFixed(1) + 'px,0) translate(0,-50%)';
    });
    // the scan label sits at the plate being scanned
    if (!dyn.beam) { dyn.beam = mk('div', 'b3d-tag'); cardsEl.appendChild(dyn.beam); }
    const live = sc.layer < total;
    const bi = clamp(Math.round(beam), 0, L.n - 1);
    const bp = proj(vec.set(-W / 2 - 0.02, L.plates[bi].y, D * 0.5));
    const bt = live ? 'Scanning layer ' + Math.min(total, Math.max(0, Math.floor(sc.layer)) + 1) + ' of ' + total : 'Scan done';
    if (dyn.beam._t !== bt) { dyn.beam.textContent = bt; dyn.beam._t = bt; }
    dyn.beam.style.opacity = compact && !live ? '0' : '1';
    dyn.beam.style.transform = 'translate3d(' + Math.max(bp.x - 52, (dyn.beam.offsetWidth || 120) + 6).toFixed(1) + 'px,' + bp.y.toFixed(1) + 'px,0) translate(-100%,-50%)';
  }

  /* ---------- sequence hooks ---------- */
  const getPartState = (id) => pst.get(id);
  seq.hooks = {
    start(st) {
      if (st.id === 'pick') { st.parts.forEach((id) => { const s = getPartState(id); if (s) { s.mode = 'hidden'; s.vis = 0; s.pop = 0; } }); }
      if (st.id === 'probe') st.parts.forEach((id) => { const s = getPartState(id); if (s && s.mode === 'hidden') { s.mode = 'shelf'; s.u = 1; s.vis = 1; } });
      if (st.id === 'splice') st.parts.forEach((id) => { const s = getPartState(id); if (s && s.mode !== 'shelf') { s.mode = 'shelf'; s.u = 1; s.vis = 1; } });
      if (st.id === 'cut') st.cuts.forEach((i) => { const c = cuts[i]; if (c && c._k) cutOpen.set(c._k, 0); });
      if (st.id !== 'settle' && st.id !== 'cut') { readoutShow = 0; if (st.id !== 'score') readout = null; }
      if (st.id === 'settle') settle();
      kick();
    },
    end(st) {
      if (st.id === 'pick') st.parts.forEach((id) => { const s = getPartState(id); if (s) { s.mode = 'shelf'; s.u = 1; s.vis = 1; } });
      if (st.id === 'splice') st.parts.forEach((id) => { const s = getPartState(id); if (s) { s.mode = 'docked'; s.vis = 1; s.flash = 1; } });
      if (st.id === 'cut') st.cuts.forEach((i) => { const c = cuts[i]; if (c && c._k) cutOpen.set(c._k, 1); });
      if (st.id === 'score') {
        if (st.readout) readout = { text: st.readout, ok: st.ok }; else if (pendingResult) readout = pendingResult;
        pendingResult = null;
        if (readout) readoutShow = 1;
      }
      kick();
    },
  };
  function settle() {
    pst.forEach((s) => { s.mode = 'docked'; s.vis = 1; });
    cuts.forEach((c) => { if (c._k) cutOpen.set(c._k, 1); });
  }

  /* ---------- props ---------- */
  function set(p) {
    if (!C) return;
    const s = JSON.stringify([p.layers, p.mode, (p.parts || []).map((x) => [x.id, x.label, x.kind, x.at, x.donor, x.verdict, x.size, x.fit]), (p.cuts || []).map((c) => [c.layer, c.frac, c.label]), onPickable()]);
    if (s !== sig) {
      sig = s;
      const hadParts = new Set(pst.keys());
      build(p);
      // parts that are new and not part of a queued sequence pop in; ones that went away are forgotten
      const ids = new Set(parts.map((x) => x.id));
      [...pst.keys()].forEach((k) => { if (!ids.has(k)) pst.delete(k); });
      cards.forEach((c) => { if (!hadParts.has(c.pt.id) && !reduced) { c.st.mode = 'docked'; c.st.pop = 0.001; c.st.vis = 0.001; } else if (c.st.pop === 0.001 || c.st.vis === 0) { c.st.vis = 1; } });
      if (reduced) cards.forEach((c) => { c.st.vis = 1; c.st.mode = c.st.mode === 'hidden' ? 'docked' : c.st.mode; });
    }
    const r = p.result;
    if (r && r.text !== resultText) {
      resultText = r.text;
      if (seq.busy) pendingResult = { text: r.text, ok: r.ok == null ? null : r.ok };
      else { readout = { text: r.text, ok: r.ok == null ? null : r.ok }; readoutShow = 1; }
    } else if (!r && resultText != null) { resultText = null; if (!seq.busy) { readout = null; readoutShow = 0; } }
    kick();
  }

  /* ---------- loop ---------- */
  function frame(now) {
    raf = 0;
    if (disposed) return;
    if (!running()) { last = 0; return; }
    const dt = last ? Math.min(0.1, (now - last) / 1000) : 0.016;
    last = now; time += dt;
    const busy = seq.tick(dt);
    if (!dragging) { dAz *= Math.exp(-dt * 2.4); dEl *= Math.exp(-dt * 2.4); if (Math.abs(dAz) < 0.02) dAz = 0; if (Math.abs(dEl) < 0.02) dEl = 0; }
    draw(dt);
    if (readout && !busy && readoutShow < 1 && !seq.busy) readoutShow = Math.min(1, readoutShow + dt * 3);
    raf = requestAnimationFrame(frame);
  }
  function draw(dt) {
    if (!C || !L || !instMesh) return;
    stepEffects(dt);
    paintPlates();
    placeCamera();
    renderer.render(scene, camera);
    layoutOverlay();
  }
  const running = () => visible && !document.hidden && !reduced;
  function kick() {
    if (disposed) return;
    if (reduced) { requestAnimationFrame(() => { if (!disposed) { seq.tick(100); drawOnce(); } }); return; }
    if (!raf && running()) { last = 0; raf = requestAnimationFrame(frame); }
  }
  function drawOnce() { if (!C || !L) return; time = 0; for (let i = 0; i < 3; i++) { stepEffects(1); } paintPlates(); placeCamera(); renderer.render(scene, camera); layoutOverlay(); if (mode === 'donor') { /* found items need one more pass */ stepEffects(1); paintPlates(); renderer.render(scene, camera); layoutOverlay(); } }

  /* ---------- resize, visibility, theme, drag ---------- */
  function resize() {
    const r = gl.getBoundingClientRect();
    cw = Math.max(10, Math.round(r.width)); ch = Math.max(10, Math.round(r.height));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(dpr); renderer.setSize(cw, ch, false);
    ov.style.setProperty('--b3d-h', ch + 'px');
    ov.classList.toggle('b3d-narrow', cw < 480);
    layoutCamera();
    kick();
    if (reduced) requestAnimationFrame(drawOnce);
  }
  const ro = new ResizeObserver(resize); ro.observe(gl);
  const io = new IntersectionObserver((es) => { visible = es[es.length - 1].isIntersecting; if (visible) kick(); }, { threshold: 0 });
  io.observe(gl);
  const onVis = () => { if (!document.hidden) kick(); };
  document.addEventListener('visibilitychange', onVis);
  function recolor() {
    resolveColors();
    if (L) { const p = propsRef.current; sig = ''; set(p); if (reduced) requestAnimationFrame(drawOnce); }
  }
  const mo = new MutationObserver(recolor);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
  const mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  if (mq && mq.addEventListener) mq.addEventListener('change', recolor);
  let px0 = 0, py0 = 0;
  const down = (e) => { dragging = true; px0 = e.clientX; py0 = e.clientY; canvas.classList.add('grab'); try { canvas.setPointerCapture(e.pointerId); } catch (x) { /* ok */ } };
  const move = (e) => { if (!dragging) return; dAz = clamp(dAz + (e.clientX - px0) * 0.28, -38, 38); dEl = clamp(dEl - (e.clientY - py0) * 0.2, -14, 22); px0 = e.clientX; py0 = e.clientY; if (reduced) drawOnce(); else kick(); };
  const up = () => { dragging = false; canvas.classList.remove('grab'); kick(); };
  canvas.addEventListener('pointerdown', down); canvas.addEventListener('pointermove', move); canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);

  resolveColors();
  resize();

  function dispose() {
    disposed = true;
    if (raf) cancelAnimationFrame(raf);
    ro.disconnect(); io.disconnect(); mo.disconnect();
    if (mq && mq.removeEventListener) mq.removeEventListener('change', recolor);
    document.removeEventListener('visibilitychange', onVis);
    canvas.removeEventListener('pointerdown', down); canvas.removeEventListener('pointermove', move); canvas.removeEventListener('pointerup', up); canvas.removeEventListener('pointercancel', up);
    seq.hooks = null;
    disposeObjs();
    [ticksEl, cardsEl, readEl, countEl, legendEl].forEach((e) => e.remove());
    renderer.dispose();
    try { renderer.forceContextLoss(); } catch (e) { /* ok */ }
    canvas.remove(); rs.dispose();
    pst.clear();
  }
  return { set, dispose, ctx: () => ({ parts, cuts, layers: L ? L.L : 16 }), resize };
}

/* ---------- 2D fallback: a labeled drawing, with the same steps highlighted ---------- */
function Fallback({ height, layers = 16, parts = [], cuts = [], highlight = null, onPick, mode = 'host', scan = null, info }) {
  const Wd = 520, Hh = Math.max(200, height), lay = plateLayout(layers);
  const top = 22, bot = Hh - 18, step = lay.n > 1 ? (bot - top) / (lay.n - 1) : 0;
  const yOf = (pi) => bot - pi * step;
  const th = Math.max(5, Math.min(16, step * 0.62));
  const cur = info && info.cur;
  const cutOf = (pi) => cuts.find((c) => lay.of(c.layer) === pi);
  const stride = Math.max(1, Math.ceil(15 / Math.max(step, 1)));
  const scanPl = mode === 'donor' && scan ? clamp(Math.floor(scan.layer / lay.gs), 0, lay.n - 1) : -1;
  const px = 70, pw = 220;
  const ps = parts.slice(0, 12).map((p, i) => ({ p, pi: lay.of(p.at || 0), i })).sort((a, b) => a.pi - b.pi);
  const ys = ps.map((o) => yOf(o.pi)); const gap = 34;
  for (let i = ps.length - 2; i >= 0; i--) ys[i] = Math.min(ys[i], ys[i + 1] - gap);
  for (let i = 1; i < ps.length; i++) ys[i] = Math.max(ys[i], ys[i - 1] + gap);
  return html`<svg class="b3d-fallback" viewBox=${'0 0 ' + Wd + ' ' + Hh} preserveAspectRatio="xMidYMid meet" role="img" aria-label=${mode === 'donor' ? 'The donor model drawn as a stack of layers' : 'The host model drawn as a stack of layers with grafted parts and removed units'}>
    ${Array.from({ length: lay.n }, (_, pi) => {
      const hl = highlight != null && lay.of(highlight) === pi, c = cutOf(pi), active = (cur === 'score') || (scanPl === pi);
      const has = ps.some((o) => o.pi === pi);
      return html`<g key=${pi}>
        <path d=${'M' + px + ',' + yOf(pi) + ' l' + (pw * 0.12) + ',' + (-th) + ' h' + pw + ' l' + (-pw * 0.12) + ',' + th + ' z'} fill=${hl || has || active ? 'var(--blue)' : 'var(--fill-2)'} fill-opacity=${hl || has || active ? 0.22 : 1} stroke="var(--text-3)" stroke-opacity="0.5" stroke-width="1" />
        ${c && html`<path d=${'M' + px + ',' + yOf(pi) + ' l' + (pw * 0.12 * 0.4) + ',' + (-th * 0.4) + ' h' + (pw * 0.3) + ' l' + (-pw * 0.12 * 0.4) + ',' + (th * 0.4) + ' z'} fill="var(--red)" fill-opacity="0.35" stroke="var(--red)" stroke-width="1.2" stroke-dasharray="3 2" />`}
        ${c && html`<text x=${px - 8} y=${yOf(pi) - th / 2 - 8} text-anchor="end" font-size="11" fill="var(--red)">Removed${c.label ? ': ' + c.label : ''}</text>`}
        ${(pi % stride === 0 || pi === lay.n - 1) && html`<text x=${px - 8} y=${yOf(pi) - th / 2 + 4} text-anchor="end" font-size="11" fill=${hl || active ? 'var(--blue)' : 'var(--text-3)'}>${plateName(lay.plates[pi])}</text>`}
      </g>`; })}
    ${ps.map((o, j) => { const col = o.p.verdict === 'contributes' ? 'var(--green)' : o.p.verdict === 'harmful' ? 'var(--red)' : o.p.verdict ? 'var(--text-3)' : 'var(--blue)'; const y = ys[j], y0 = yOf(o.pi) - th / 2, bx = px + pw + 40;
      const now = cur === 'pick' || cur === 'probe' || cur === 'splice' || cur === 'heal';
      return html`<g key=${o.p.id} style=${{ cursor: onPick ? 'pointer' : 'default' }} onClick=${() => onPick && onPick(o.p.id)}><title>${o.p.label || o.p.id}</title>
        <path d=${'M' + (px + pw + 4) + ',' + y0 + ' C' + (px + pw + 28) + ',' + y0 + ' ' + (bx - 24) + ',' + y + ' ' + bx + ',' + y} fill="none" stroke=${col} stroke-width=${now ? 2.4 : 1.6} />
        <rect x=${bx} y=${y - 11} width="30" height="22" rx="4" fill=${col} fill-opacity="0.3" stroke=${col} stroke-width="1.5" />
        <text x=${bx + 38} y=${y - 1} font-size="12" font-weight="500" fill="var(--text-1)">${String(o.p.label || o.p.id).slice(0, 26)}</text>
        <text x=${bx + 38} y=${y + 12} font-size="10.5" fill="var(--text-3)">${(KIND_NAME[o.p.kind] || 'Part') + ' · layer ' + o.p.at + (o.p.verdict ? ' · ' + VERDICT_WORD[o.p.verdict] : '')}</text></g>`; })}
    ${mode === 'donor' && scan && html`<text x=${px + pw + 40} y=${24} font-size="12" fill="var(--text-2)">${'Layers scanned ' + clamp(Math.floor(scan.layer), 0, scan.total) + ' of ' + scan.total + ', parts found ' + (scan.found || []).length}</text>`}
  </svg>`;
}

/* ---------- the step bar under the scene ---------- */
function StepBar({ info, idle, mode }) {
  const cur = info && info.cur;
  if (mode === 'donor') return null;
  return html`<div class="b3d-bar" aria-live="polite">
    <ol class="b3d-steps">
      ${STEP_DEFS.map((s, i) => html`<li key=${s.id} class=${'b3d-step' + (cur === s.id ? ' on' : '') + (info && info.done.indexOf(s.id) >= 0 && cur !== s.id ? ' done' : '')}>
        ${cur === s.id && html`<i class="b3d-fill" key=${info.key} style=${{ animationDuration: (info.dur || 1.5) + 's' }}></i>`}
        <span class="b3d-n">${info && info.done.indexOf(s.id) >= 0 && cur !== s.id ? '✓' : i + 1}</span><span>${s.name}</span></li>`)}
    </ol>
    <div class="b3d-sentence">${(info && info.text) || idle}</div>
  </div>`;
}

/* ---------- the component ---------- */
function idleText(p) {
  if (p.idle) return p.idle;
  if (p.mode === 'donor') return 'The donor model, one plate per layer.';
  const np = (p.parts || []).length, nc = (p.cuts || []).length;
  if (!np && !nc) return 'The host model as a stack of layers. Each plate is one layer.';
  const bits = [];
  if (np) bits.push(np + (np === 1 ? ' part is' : ' parts are') + ' spliced into the host');
  if (nc) bits.push(nc + (nc === 1 ? ' set of units is' : ' sets of units are') + ' removed');
  return bits.join(' and ') + '.';
}
export function Brain3D(props) {
  const { height = 360, label = null, class: cls, mode = 'host' } = props;
  const glRef = useRef(), ovRef = useRef();
  const eng = useRef(null), pr = useRef(props), pending = useRef([]), lastKey = useRef(props.pulse ? props.pulse.key : undefined);
  const seqRef = useRef(null);
  const [fb, setFb] = useState(false);
  const [info, setInfo] = useState(null);
  const reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  pr.current = props;
  if (!seqRef.current) {
    seqRef.current = createSequencer((i) => setInfo(i));
    seqRef.current.instant = reduced;
    seqRef.current.getCtx = () => {
      const p = pr.current;
      if (eng.current) return eng.current.ctx();
      return { parts: (p.parts || []).map((x) => ({ ...x, at: Math.round(x.at || 0) })), cuts: p.cuts || [], layers: p.layers || 16 };
    };
  }
  const seq = seqRef.current;
  const showBar = props.steps == null ? (height >= 300) : !!props.steps;
  const compact = height < 300;
  // a fallback and a 3D scene both read the same sequencer; the fallback needs its own clock
  useEffect(() => {
    if (!fb) return undefined;
    let raf = 0, last = 0;
    const loop = (t) => { const dt = last ? Math.min(0.1, (t - last) / 1000) : 0.016; last = t; if (seq.tick(dt)) raf = requestAnimationFrame(loop); else raf = 0; };
    seq.hooks = { start() { if (!raf && !seq.instant) { last = 0; raf = requestAnimationFrame(loop); } }, end() {} };
    return () => { if (raf) cancelAnimationFrame(raf); seq.hooks = null; };
  }, [fb]);
  useEffect(() => {
    if (fb) return undefined;
    let dead = false, e = null;
    (async () => {
      try {
        const THREE = await loadThree();
        if (dead) return;
        e = createEngine(THREE, { gl: glRef.current, ov: ovRef.current }, pr, seq, reduced);
        eng.current = e;
        e.set(pr.current);
        const q = pending.current; pending.current = [];
        q.forEach((p) => seq.enqueue(p));
      } catch (err) {
        console.warn('Brain3D falls back to 2D:', err && err.message);
        if (e) { try { e.dispose(); } catch (x) { /* ok */ } }
        eng.current = null;
        if (!dead) setFb(true);
      }
    })();
    return () => { dead = true; if (e) e.dispose(); eng.current = null; seq.clear(); };
  }, [fb]);
  useEffect(() => { if (eng.current) eng.current.set(props); });
  useEffect(() => {
    const p = props.pulse;
    if (!p || p.key == null || p.key === lastKey.current) return;
    lastKey.current = p.key;
    if (eng.current || fb) seq.enqueue(p); else pending.current.push(p);
  }, [props.pulse && props.pulse.key]);
  const title = props.title || (mode === 'host' ? label : null);
  const sub = props.title ? label : null;
  const sc = props.scan;
  const caption = mode === 'donor' && sc
    ? (sc.layer >= sc.total ? 'Scanned all ' + sc.total + ' layers. Found ' + (sc.found || []).length + ' parts.' : 'Scanning layer ' + Math.min(sc.total, Math.max(0, Math.floor(sc.layer)) + 1) + ' of ' + sc.total + '.')
    : null;
  const barText = caption || ((info && info.text) || idleText(props));
  const capText = compact && label && !(info && info.text) ? label : barText;
  return html`<div class=${'brain3d' + (cls ? ' ' + cls : '') + (compact ? ' compact' : '')} style=${{ height: height + 'px' }}>
    <div class="b3d-scene">
      ${fb ? html`<${Fallback} ...${props} height=${Math.max(160, height - (showBar ? 78 : 0))} info=${info} />` : html`<div class="b3d-gl" ref=${glRef}></div>`}
      <div class="b3d-overlay" ref=${ovRef}></div>
      ${(title || props.model) && html`<div class="b3d-title">${props.model && html`<${ModelLogo} model=${props.model} size=${compact ? 18 : 22} alt="" />`}<div class="b3d-title-t"><b>${title || shortName(props.model)}</b>${(sub || props.size) && html`<span>${[props.size, sub].filter(Boolean).join(' · ')}</span>`}</div></div>`}
      ${fb && html`<div class="b3d-legend"><span class="b3d-lg"><i class="b3d-sw dot" style="background:var(--text-3)"></i>Host neurons</span><span class="b3d-lg"><i class="b3d-sw sq" style="background:var(--purple)"></i>Donor part</span><span class="b3d-lg"><i class="b3d-sw cut" style="border-color:var(--red)"></i>Removed</span></div>`}
      ${!showBar && capText && html`<div class="b3d-caption" aria-live="polite">${capText}</div>`}
    </div>
    ${showBar && (mode === 'donor' ? html`<div class="b3d-bar"><div class="b3d-sentence">${barText}</div></div>` : html`<${StepBar} info=${info} idle=${idleText(props)} mode=${mode} />`)}
  </div>`;
}
