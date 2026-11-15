// Brain3D: a 3D view of a network being edited. Uses three.js 0.170.0 (loaded on first use).
//
// Brain3D({ height = 360, layers = 16, parts = [], cuts = [], pulse = null, scan = null,
//           mode = 'host', highlight = null, label = null, onPick = null, class })
//   layers     number of host layers, drawn as a stack of neuron sheets (layer 0 at the bottom).
//   parts      [{ id, label, kind, at, donor, verdict }] grafted parts docked beside layer `at`.
//              kind: block | mlp_cluster | attn_heads | expert | inserted_layer.
//              verdict: 'contributes' (green) | 'no better than random' (grey) | 'harmful' (red) | null (donor tint).
//   cuts       [{ layer, frac, label }] deletion masks: frac (0..1) of the layer's units fall away as a clean slice.
//   pulse      { kind, at, partId, ok, key }. When key changes, one animation plays:
//              graft (a part flies in and splices; no partId replays every part), cut (a slice at `at`),
//              heal (a wave along the glue links, then through the stack), score (a signal climbs the stack and
//              flashes green for ok === true, red for false, blue for null), probe (a ping at `at`),
//              scan (a beam sweeps the stack once).
//   mode       'host' or 'donor'. In donor mode `scan` = { layer, total, found: [{ layer, kind, label, id }] }
//              drives a beam through the layers; found parts lift out and glow.
//   highlight  a layer index to emphasize. label: a caption. onPick(partId): a part was clicked.
// Colors come from the theme tokens and follow theme changes. The scene pauses offscreen and in hidden tabs,
// caps the pixel ratio at 2, shows a still frame under prefers-reduced-motion, and falls back to a 2D SVG
// drawing when WebGL or the library is unavailable. Drag to orbit; hold Ctrl or Cmd and scroll to zoom.

import { html, useEffect, useRef, useState } from './lib.js';
import { donorColor } from './util.js';

const THREE_URL = 'https://esm.sh/three@0.170.0';
let threeP = null;
const loadThree = () => (threeP || (threeP = import(THREE_URL).catch((e) => { threeP = null; throw e; })));

const R = 3; // layer radius in scene units
const TAU = Math.PI * 2;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (u) => 1 - Math.pow(1 - clamp(u, 0, 1), 3);
const easeIO = (u) => { u = clamp(u, 0, 1); return u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2; };
function mulberry(a) { return () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function hashStr(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }

const VERDICT = { contributes: 'green', 'no better than random': 'grey', harmful: 'red' };
const KIND_TOKEN = { block: 'blue', mlp_cluster: 'green', attn_heads: 'amber', expert: 'purple', inserted_layer: 'blue' };

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
  probe.style.cssText = 'position:absolute;width:0;height:0;visibility:hidden;pointer-events:none';
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

/* ---------- shaders ---------- */
const SQ = 'float sq(float x) { return x * x; }\n';
const ENERGY = SQ + `
uniform float uSigY, uSigAmp, uHealY, uHealR, uHealAmp, uFlash, uBeamY, uBeamOn;
uniform vec3 uSigCol, uHealCol, uFlashCol, uBeamCol;
float energy(float y, out vec3 col) {
  float s = uSigAmp * exp(-sq((y - uSigY) / 0.8));
  float h = uHealAmp * exp(-sq((abs(y - uHealY) - uHealR) / 0.55));
  float b = uBeamOn * exp(-sq((y - uBeamY) / 0.4));
  float f = uFlash;
  float sum = s + h + b + f + 1e-4;
  col = (uSigCol * s + uHealCol * h + uBeamCol * b + uFlashCol * f) / sum;
  return s + h + b * 1.4 + f * 0.8;
}`;

const NEURON_VS = ENERGY + `
attribute float aLayer; attribute float aSeed; attribute float aSize; attribute float aCutT; attribute float aRim;
uniform float uTime, uPx, uCam, uHi, uHiOn, uPk, uPkAmp, uScanL, uScanOn, uLight;
uniform vec3 uColor, uRed;
varying vec4 vCol;
void main() {
  vec3 p = position;
  float ph = uTime * 0.7 + aSeed * 6.2831 + aLayer * 0.45;
  p.y += sin(ph) * 0.035;
  p.x += sin(ph * 0.7 + aSeed * 9.0) * 0.02 * (1.0 - aRim);
  p.z += cos(ph * 0.6 + aSeed * 7.0) * 0.02 * (1.0 - aRim);
  p.xz *= 1.0 + sin(uTime * 0.5 + aLayer * 0.3) * 0.012;
  float dt = uTime - aCutT;
  float cut = step(0.0, dt);
  float fall = clamp(dt / 1.4, 0.0, 1.0);
  float ghost = step(1.4, dt);
  float falling = cut * (1.0 - ghost);
  p.y -= falling * fall * fall * 1.3;
  p.x += falling * (aSeed - 0.5) * fall * 0.9;
  p.z += falling * (fract(aSeed * 7.3) - 0.5) * fall * 0.9;
  vec3 ec; float e = energy(p.y, ec);
  float hi = (uHiOn * exp(-sq(aLayer - uHi) * 4.0)) + uPkAmp * exp(-sq(aLayer - uPk) * 3.0);
  float dim = mix(1.0, 0.3, uScanOn * smoothstep(uScanL - 0.4, uScanL + 0.6, aLayer));
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  float depth = mix(1.0, 0.42, clamp((-mv.z - uCam) / 7.0, 0.0, 1.0));
  float a = (0.5 + 0.35 * aRim + 0.12 * sin(ph * 2.0)) * dim * depth;
  a = a * (1.0 - falling * fall) + hi * 0.45 + e;
  a = mix(a, 0.13, ghost * cut);
  vec3 col = mix(uColor, ec, clamp(e, 0.0, 1.0) * 0.85);
  col = mix(col, uRed, cut);
  col = mix(col, uColor * 1.15, clamp(hi, 0.0, 1.0) * 0.5);
  float size = aSize * (1.0 + hi * 0.5 + e * 0.9) * mix(1.0, 0.55, ghost * cut);
  gl_PointSize = max(1.5, size * uPx / -mv.z);
  gl_Position = projectionMatrix * mv;
  vCol = vec4(col * mix(1.0, 0.88, uLight), clamp(a * mix(1.0, 1.3, uLight), 0.0, 1.0));
}`;

const SPRITE_FS = SQ + `
varying vec4 vCol;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c) * 2.0;
  if (d > 1.0) discard;
  float core = exp(-d * d * 9.0);
  float halo = sq(1.0 - d) * 0.5;
  gl_FragColor = vec4(vCol.rgb, clamp((core + halo) * vCol.a, 0.0, 1.0));
}`;

const LINE_VS = ENERGY + `
attribute float aSeed; attribute float aT;
uniform float uTime, uCam, uLight;
uniform vec3 uColor;
varying vec4 vCol;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 ec; float e = energy(position.y, ec);
  float ph = fract(uTime * 0.18 + aSeed);
  float pulse = exp(-sq((aT - ph) / 0.1));
  float depth = mix(1.0, 0.4, clamp((-mv.z - uCam) / 7.0, 0.0, 1.0));
  float a = (0.06 + 0.3 * pulse) * depth + e * 0.25;
  vec3 col = mix(uColor, ec, clamp(e, 0.0, 1.0) * 0.8);
  vCol = vec4(col * mix(1.0, 0.88, uLight), clamp(a * mix(1.0, 1.5, uLight), 0.0, 1.0));
  gl_Position = projectionMatrix * mv;
}`;
const LINE_FS = `varying vec4 vCol; void main() { gl_FragColor = vCol; }`;

const SHEET_VS = ENERGY + `
attribute float aState;
uniform float uHiOn;
varying vec2 vP; varying float vE; varying vec3 vEC; varying float vWorldY;
void main() {
  vec4 wp = instanceMatrix * vec4(position, 1.0);
  vP = position.xz / ${R.toFixed(1)};
  vWorldY = wp.y;
  vE = energy(wp.y, vEC);
  vE = vE * 0.6 + aState;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * wp;
}`;
const SHEET_FS = `
uniform vec3 uColor; uniform float uA; uniform float uRim; uniform float uLight;
varying vec2 vP; varying float vE; varying vec3 vEC;
void main() {
  float r = length(vP);
  float a = uRim > 0.5 ? uA : uA * (0.25 + 0.75 * smoothstep(0.0, 1.0, r));
  a = a + vE * (uRim > 0.5 ? 0.4 : 0.14);
  vec3 col = mix(uColor, vEC, clamp(vE, 0.0, 1.0) * 0.7);
  gl_FragColor = vec4(col * mix(1.0, 0.88, uLight), clamp(a, 0.0, 1.0));
}`;

const PART_VS = `
attribute float aSeed; attribute float aSize;
uniform float uTime, uPx, uCam, uS, uAmp, uFlash, uHover, uLight;
uniform vec3 uColor;
varying vec4 vCol;
void main() {
  vec3 p = position;
  p += 0.02 * vec3(sin(uTime * 1.3 + aSeed * 20.0), cos(uTime * 1.1 + aSeed * 13.0), sin(uTime * 0.9 + aSeed * 17.0));
  vec4 mv = modelViewMatrix * vec4(p * uS, 1.0);
  float a = uAmp * (0.72 + 0.28 * sin(uTime * 2.0 + aSeed * 9.0)) + uFlash * 0.5 + uHover * 0.25;
  vec3 col = uColor * (1.0 + uFlash * 0.35 + uHover * 0.2);
  float size = aSize * uS * (1.0 + uFlash * 0.7 + uHover * 0.25);
  gl_PointSize = max(1.5, size * uPx / -mv.z);
  gl_Position = projectionMatrix * mv;
  vCol = vec4(clamp(col, 0.0, 1.4) * mix(1.0, 0.9, uLight), clamp(a * mix(1.0, 1.25, uLight), 0.0, 1.0));
}`;

const LINK_VS = SQ + `
attribute float aT; attribute float aBig;
uniform float uTime, uPx, uCam, uGrow, uHeal, uFlash, uAmp, uLight, uSeed;
uniform vec3 uColor;
varying vec4 vCol;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  float vis = 1.0 - smoothstep(uGrow - 0.06, uGrow + 0.001, aT);
  vis = max(vis, aBig * step(0.999, uGrow));
  float fl = pow(0.5 + 0.5 * sin(aT * 14.0 - uTime * 2.2 + uSeed * 6.0), 3.0);
  float h = exp(-sq((aT - uHeal) / 0.1)) * step(0.001, uHeal) * (1.0 - smoothstep(1.05, 1.25, uHeal));
  float size = (0.075 + 0.07 * fl + 0.22 * h + 0.14 * uFlash + aBig * 0.22) * (1.0 + 0.0 * aT);
  float a = (0.35 + 0.4 * fl + 0.9 * h + 0.5 * uFlash + aBig * 0.3) * vis * uAmp;
  gl_PointSize = max(1.2, size * uPx / -mv.z);
  gl_Position = projectionMatrix * mv;
  vCol = vec4(uColor * mix(1.0, 0.9, uLight) * (1.0 + h * 0.4), clamp(a * mix(1.0, 1.25, uLight), 0.0, 1.0));
}`;

const BURST_VS = `
attribute vec3 aCol; attribute float aA; attribute float aS;
uniform float uPx, uLight;
varying vec4 vCol;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = max(1.0, aS * uPx / -mv.z);
  gl_Position = projectionMatrix * mv;
  vCol = vec4(aCol * mix(1.0, 0.9, uLight), aA);
}`;

const DISC_VS = `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const DISC_FS = SQ + `
varying vec2 vUv;
uniform float uR, uW, uAlpha, uFill, uLight; uniform vec3 uColor;
void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float d = length(p);
  float ring = exp(-sq((d - uR) / uW));
  float fill = uFill * (1.0 - smoothstep(0.0, 1.0, d)) * smoothstep(0.0, 0.1, uR);
  float edge = 1.0 - smoothstep(0.82, 1.0, d);
  float a = (ring + fill) * uAlpha * edge;
  gl_FragColor = vec4(uColor * mix(1.0, 0.9, uLight), clamp(a * mix(1.0, 1.2, uLight), 0.0, 1.0));
}`;

const SLICE_VS = `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const SLICE_FS = SQ + `
varying vec2 vUv; uniform float uAlpha; uniform vec3 uColor; uniform float uLight;
void main() {
  float ex = 1.0 - pow(abs(vUv.x - 0.5) * 2.0, 3.0);
  float ey = exp(-sq((vUv.y - 0.5) * 5.0));
  gl_FragColor = vec4(uColor, clamp(ex * ey * uAlpha * mix(1.0, 1.2, uLight), 0.0, 1.0));
}`;

/* ---------- point clouds for parts ---------- */
function blobPoints(kind, rng) {
  const out = [];
  const gauss = () => (rng() + rng() + rng() - 1.5) / 1.5;
  if (kind === 'block') {
    for (let x = -1; x <= 1; x++) for (let y = -1; y <= 1; y++) for (let z = -1; z <= 1; z++) out.push([x * 0.17 + gauss() * 0.02, y * 0.17 + gauss() * 0.02, z * 0.17 + gauss() * 0.02]);
    for (let i = 0; i < 18; i++) out.push([gauss() * 0.28, gauss() * 0.28, gauss() * 0.28]);
  } else if (kind === 'attn_heads') {
    const cs = [[-0.2, 0.2], [0.2, 0.2], [-0.2, -0.2], [0.2, -0.2]];
    cs.forEach(([cx, cy]) => { for (let i = 0; i < 12; i++) { const a = (i / 12) * TAU; out.push([cx + Math.cos(a) * 0.13, cy + Math.sin(a) * 0.13, gauss() * 0.04]); } out.push([cx, cy, 0]); });
  } else if (kind === 'expert') {
    const v = [[0.4, 0, 0], [-0.4, 0, 0], [0, 0.4, 0], [0, -0.4, 0], [0, 0, 0.4], [0, 0, -0.4]];
    for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) { if (i >> 1 === j >> 1) continue; for (let k = 1; k < 5; k++) { const t = k / 5; out.push([lerp(v[i][0], v[j][0], t), lerp(v[i][1], v[j][1], t), lerp(v[i][2], v[j][2], t)]); } }
    v.forEach((q) => out.push(q));
  } else if (kind === 'inserted_layer') {
    for (let i = 0; i < 46; i++) { const r = 0.5 * Math.sqrt((i + 0.5) / 46); const a = i * 2.39996; out.push([Math.cos(a) * r, gauss() * 0.02, Math.sin(a) * r]); }
    for (let i = 0; i < 20; i++) { const a = (i / 20) * TAU; out.push([Math.cos(a) * 0.52, 0, Math.sin(a) * 0.52]); }
  } else {
    for (let i = 0; i < 64; i++) out.push([gauss() * 0.3, gauss() * 0.3, gauss() * 0.3]);
  }
  return out;
}

/* ---------- the engine ---------- */
function createEngine(THREE, els, propsRef) {
  const { gl: glHost, tip, ticks } = els;
  const V3 = THREE.Vector3;
  const reduced = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setClearColor(0x000000, 0);
  const canvas = renderer.domElement;
  canvas.className = 'b3d-canvas';
  glHost.appendChild(canvas);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
  const fovTan = Math.tan((38 * Math.PI) / 360);
  const { resolve, dispose: disposeProbe } = makeResolver(glHost);

  /* shared uniforms */
  const v3 = (c) => new V3(c[0], c[1], c[2]);
  const U = {
    uTime: { value: 0 }, uPx: { value: 400 }, uCam: { value: 16 }, uLight: { value: 0 },
    uSigY: { value: -99 }, uSigAmp: { value: 0 }, uHealY: { value: 0 }, uHealR: { value: 0 }, uHealAmp: { value: 0 },
    uFlash: { value: 0 }, uBeamY: { value: 0 }, uBeamOn: { value: 0 },
    uSigCol: { value: new V3() }, uHealCol: { value: new V3() }, uFlashCol: { value: new V3() }, uBeamCol: { value: new V3() },
    uColor: { value: new V3() }, uRed: { value: new V3() },
    uHi: { value: 0 }, uHiOn: { value: 0 }, uPk: { value: 0 }, uPkAmp: { value: 0 }, uScanL: { value: 0 }, uScanOn: { value: 0 },
  };
  let theme = null;
  const materials = []; // every material that blends by theme
  const blend = () => (theme && theme.light ? THREE.NormalBlending : THREE.AdditiveBlending);
  const track = (m) => { materials.push(m); return m; };

  /* state */
  let cfg = { layers: 0, mode: 'host', parts: [], cuts: [], highlight: null, scan: null };
  let sp = 0.5, H = 8, T = reduced ? 4 : 0;
  const layerY = (l) => (l - (cfg.layers - 1) / 2) * sp;
  const layerDock = (l, k, found) => { const a = 0.7 + l * 0.55 + k * 1.1 + (found ? 0.5 : 0); return a; };
  const rimPoint = (l, a) => new V3(Math.cos(a) * R, layerY(l), Math.sin(a) * R);
  const effects = [];
  const fx = (dur, upd, done, delay = 0) => { if (reduced) { upd(1); if (done) done(); return; } effects.push({ t0: T + delay, dur, upd, done, started: false }); };
  let dead = false, visible = true, inView = true, raf = 0, needsRender = true;

  /* theme */
  function readTheme() {
    const light = (() => { const c = resolve('var(--bg-sunk, var(--bg))'); return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] > 0.5; })();
    theme = {
      light, blue: resolve('var(--blue)'), green: resolve('var(--green)'), red: resolve('var(--red)'), amber: resolve('var(--amber)'),
      purple: resolve('var(--purple)', '#8269cb'), grey: resolve('var(--text-3)'),
    };
    U.uLight.value = light ? 1 : 0;
    const donor = cfg.mode === 'donor';
    const base = donor ? theme.blue.map((x, i) => lerp(x, theme.green[i], 0.45)) : theme.blue;
    U.uColor.value.copy(v3(base));
    U.uRed.value.copy(v3(theme.red));
    U.uSigCol.value.copy(v3(theme.blue.map((x) => Math.min(1, x * 1.15 + 0.1))));
    U.uHealCol.value.copy(v3(theme.green));
    U.uFlashCol.value.copy(v3(theme.blue));
    U.uBeamCol.value.copy(v3(theme.green.map((x, i) => lerp(x, theme.blue[i], 0.25))));
    materials.forEach((m) => { m.blending = blend(); m.needsUpdate = true; });
    if (stackReady) { sheetMat.uniforms.uColor.value.copy(v3(base)); rimMat.uniforms.uColor.value.copy(v3(base)); }
    parts.forEach((o) => paintPart(o));
    needsRender = true;
  }

  /* ---------- the stack ---------- */
  let stackReady = false;
  const stackGroup = new THREE.Group(); scene.add(stackGroup);
  let neurons, lines, sheets, rims, sheetMat, rimMat, aCutT, aStateArr, layerProj = [], layerDir = [];
  let perLayer = 0, rimN = 64, layerCutFrac = [], aULayer = [];
  function disposeObj(o) { o.traverse((c) => { if (c.geometry) c.geometry.dispose(); if (c.material) { const i = materials.indexOf(c.material); if (i >= 0) materials.splice(i, 1); c.material.dispose(); } }); }
  function buildStack(L) {
    while (stackGroup.children.length) { const c = stackGroup.children[0]; stackGroup.remove(c); disposeObj(c); }
    stackReady = false;
    cfg.layers = L;
    H = Math.min(8.4, Math.max(L * 0.55, 1.6)); sp = L > 1 ? H / (L - 1) : 0;
    perLayer = clamp(Math.round(9000 / Math.max(L, 1)), 170, 520);
    const per = perLayer + rimN, total = L * per;
    const pos = new Float32Array(total * 3), aLayer = new Float32Array(total), aSeed = new Float32Array(total), aSize = new Float32Array(total), aRim = new Float32Array(total), aU = new Float32Array(total);
    aCutT = new Float32Array(total).fill(1e9);
    layerProj = []; layerDir = [];
    for (let l = 0; l < L; l++) {
      const rng = mulberry(l * 7919 + 13);
      const da = rng() * TAU, dx = Math.cos(da), dz = Math.sin(da);
      layerDir.push([dx, dz]);
      const y0 = layerY(l);
      const proj = [];
      for (let i = 0; i < per; i++) {
        const k = l * per + i;
        let x, z, rim = 0;
        if (i < perLayer) {
          const r = R * Math.sqrt((i + 0.5 + (rng() - 0.5) * 0.6) / perLayer) * 0.98;
          const th = i * 2.39996 + rng() * 0.2;
          x = Math.cos(th) * r; z = Math.sin(th) * r;
        } else {
          const j = i - perLayer; const th = (j / rimN) * TAU + (rng() - 0.5) * 0.03; const r = R * (0.995 + rng() * 0.012);
          x = Math.cos(th) * r; z = Math.sin(th) * r; rim = 1;
        }
        pos[k * 3] = x; pos[k * 3 + 1] = y0 + (rng() - 0.5) * 0.07; pos[k * 3 + 2] = z;
        aLayer[k] = l; aSeed[k] = rng(); aRim[k] = rim;
        aSize[k] = rim ? 0.17 + rng() * 0.06 : 0.1 + rng() * 0.11;
        proj.push([x * dx + z * dz, i]);
      }
      proj.sort((a, b) => a[0] - b[0]);
      layerProj.push(proj.map((p) => p[0]));
      proj.forEach((p, rank) => { aU[l * per + p[1]] = rank / (per - 1); });
    }
    aULayer = aU;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aLayer', new THREE.BufferAttribute(aLayer, 1));
    g.setAttribute('aSeed', new THREE.BufferAttribute(aSeed, 1));
    g.setAttribute('aSize', new THREE.BufferAttribute(aSize, 1));
    g.setAttribute('aRim', new THREE.BufferAttribute(aRim, 1));
    g.setAttribute('aCutT', new THREE.BufferAttribute(aCutT, 1));
    g.boundingSphere = new THREE.Sphere(new V3(), 30);
    const nm = track(new THREE.ShaderMaterial({ vertexShader: NEURON_VS, fragmentShader: SPRITE_FS, uniforms: U, transparent: true, depthTest: false, depthWrite: false, blending: blend() }));
    neurons = new THREE.Points(g, nm); neurons.frustumCulled = false; neurons.renderOrder = 2;
    stackGroup.add(neurons);

    // synapse lines between neighbouring layers
    if (L > 1) {
      const segs = clamp(Math.round(1200 / (L - 1)), 28, 80);
      const lp = new Float32Array((L - 1) * segs * 6), ls = new Float32Array((L - 1) * segs * 2), lt = new Float32Array((L - 1) * segs * 2);
      const rng = mulberry(4242);
      let o = 0, so = 0;
      for (let l = 0; l < L - 1; l++) {
        for (let s = 0; s < segs; s++) {
          const th = rng() * TAU, r = R * Math.sqrt(rng()) * 0.96;
          const th2 = th + (rng() - 0.5) * 0.5, r2 = clamp(r + (rng() - 0.5) * 0.7, 0, R * 0.98);
          lp[o++] = Math.cos(th) * r; lp[o++] = layerY(l); lp[o++] = Math.sin(th) * r;
          lp[o++] = Math.cos(th2) * r2; lp[o++] = layerY(l + 1); lp[o++] = Math.sin(th2) * r2;
          const sd = rng(); ls[so] = sd; ls[so + 1] = sd; lt[so] = 0; lt[so + 1] = 1; so += 2;
        }
      }
      const lg = new THREE.BufferGeometry();
      lg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
      lg.setAttribute('aSeed', new THREE.BufferAttribute(ls, 1));
      lg.setAttribute('aT', new THREE.BufferAttribute(lt, 1));
      lg.boundingSphere = new THREE.Sphere(new V3(), 30);
      const lm = track(new THREE.ShaderMaterial({ vertexShader: LINE_VS, fragmentShader: LINE_FS, uniforms: U, transparent: true, depthTest: false, depthWrite: false, blending: blend() }));
      lines = new THREE.LineSegments(lg, lm); lines.frustumCulled = false; lines.renderOrder = 1;
      stackGroup.add(lines);
    }

    // translucent sheets and rims, one instanced draw call each
    aStateArr = new Float32Array(L);
    const mk = (geo, rim, a) => {
      geo.rotateX(-Math.PI / 2);
      geo.setAttribute('aState', new THREE.InstancedBufferAttribute(aStateArr, 1));
      const m = track(new THREE.ShaderMaterial({ vertexShader: SHEET_VS, fragmentShader: SHEET_FS, transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide, blending: blend(), uniforms: { ...U, uColor: { value: new V3() }, uA: { value: a }, uRim: { value: rim } } }));
      const im = new THREE.InstancedMesh(geo, m, L);
      const mat4 = new THREE.Matrix4();
      for (let l = 0; l < L; l++) { mat4.makeTranslation(0, layerY(l), 0); im.setMatrixAt(l, mat4); }
      im.frustumCulled = false;
      return im;
    };
    sheets = mk(new THREE.CircleGeometry(R * 1.01, 72), 0, 0.075); sheets.renderOrder = 0;
    rims = mk(new THREE.RingGeometry(R * 0.996, R * 1.012, 96), 1, 0.35); rims.renderOrder = 0;
    sheetMat = sheets.material; rimMat = rims.material;
    stackGroup.add(sheets, rims);
    stackReady = true;
    layerCutFrac = new Array(L).fill(0);
    readTheme();
  }

  /* ---------- cuts ---------- */
  function applyCuts(list, animate) {
    if (!stackReady) return;
    const want = new Array(cfg.layers).fill(0);
    const labels = {};
    (list || []).forEach((c) => { const l = clamp(Math.round(c.layer), 0, cfg.layers - 1); want[l] = clamp(Math.max(want[l], +c.frac || 0), 0, 1); labels[l] = c.label; });
    const per = perLayer + rimN, arr = neurons.geometry.attributes.aCutT;
    for (let l = 0; l < cfg.layers; l++) {
      const d = want[l], prev = layerCutFrac[l] || 0;
      if (d === prev) continue;
      for (let i = 0; i < per; i++) {
        const k = l * per + i, u = aULayer[k];
        const was = prev > 0 && u >= 1 - prev;
        const now = d > 0 && u >= 1 - d;
        if (!now) arr.array[k] = 1e9;
        else if (was) arr.array[k] = -100;
        else arr.array[k] = animate ? T + 0.05 + ((u - (1 - d)) / Math.max(d, 0.01)) * 0.6 : -100;
      }
      if (animate && d > prev && !reduced) sliceFx(l, d);
      layerCutFrac[l] = d;
    }
    arr.needsUpdate = true;
    needsRender = true;
  }
  function replayCut(l) {
    const d = layerCutFrac[l];
    if (!d) return false;
    const per = perLayer + rimN, arr = neurons.geometry.attributes.aCutT;
    for (let i = 0; i < per; i++) { const k = l * per + i, u = aULayer[k]; if (u >= 1 - d) arr.array[k] = T + 0.05 + ((u - (1 - d)) / Math.max(d, 0.01)) * 0.6; }
    arr.needsUpdate = true;
    sliceFx(l, d);
    return true;
  }

  /* ---------- effects: discs, slices, bursts ---------- */
  const discGeo = new THREE.PlaneGeometry(2, 2); discGeo.rotateX(-Math.PI / 2);
  const discs = [];
  for (let i = 0; i < 8; i++) {
    const m = track(new THREE.ShaderMaterial({ vertexShader: DISC_VS, fragmentShader: DISC_FS, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uR: { value: 0 }, uW: { value: 0.05 }, uAlpha: { value: 0 }, uFill: { value: 0 }, uLight: U.uLight, uColor: { value: new V3() } } }));
    const mesh = new THREE.Mesh(discGeo, m); mesh.visible = false; mesh.frustumCulled = false; mesh.renderOrder = 4;
    scene.add(mesh); discs.push({ mesh, busy: false });
  }
  function ring(y, col, o = {}) {
    const d = discs.find((x) => !x.busy); if (!d) return;
    d.busy = true; const m = d.mesh; m.visible = true; m.position.y = y; m.scale.setScalar(R * 1.5);
    const u = m.material.uniforms; u.uColor.value.copy(v3(col)); u.uW.value = o.w || 0.05; u.uFill.value = o.fill || 0;
    const r0 = o.r0 ?? 0.05, r1 = o.r1 ?? 0.72, a = o.alpha ?? 0.9;
    fx(o.dur || 1.2, (t) => { u.uR.value = lerp(r0, r1, ease(t)); u.uAlpha.value = a * Math.pow(1 - t, 1.5); needsRender = true; }, () => { m.visible = false; d.busy = false; });
  }
  // a steady ring for the highlighted layer, a soft disc under the scan beam
  const hiDisc = new THREE.Mesh(discGeo, track(new THREE.ShaderMaterial({ vertexShader: DISC_VS, fragmentShader: DISC_FS, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uR: { value: 0.66 }, uW: { value: 0.035 }, uAlpha: { value: 0 }, uFill: { value: 0 }, uLight: U.uLight, uColor: { value: new V3() } } })));
  hiDisc.scale.setScalar(R * 1.5); hiDisc.visible = false; hiDisc.frustumCulled = false; hiDisc.renderOrder = 4; scene.add(hiDisc);
  const beamDisc = new THREE.Mesh(discGeo, track(new THREE.ShaderMaterial({ vertexShader: DISC_VS, fragmentShader: DISC_FS, transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending, uniforms: { uR: { value: 0.67 }, uW: { value: 0.03 }, uAlpha: { value: 0 }, uFill: { value: 0.22 }, uLight: U.uLight, uColor: { value: new V3() } } })));
  beamDisc.scale.setScalar(R * 1.5); beamDisc.visible = false; beamDisc.frustumCulled = false; beamDisc.renderOrder = 4; scene.add(beamDisc);

  const sliceMesh = new THREE.Mesh(new THREE.PlaneGeometry(R * 2.5, 0.6), track(new THREE.ShaderMaterial({ vertexShader: SLICE_VS, fragmentShader: SLICE_FS, transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, uniforms: { uAlpha: { value: 0 }, uColor: { value: new V3() }, uLight: U.uLight } })));
  sliceMesh.visible = false; sliceMesh.frustumCulled = false; sliceMesh.renderOrder = 4; scene.add(sliceMesh);
  function sliceFx(l, frac) {
    const [dx, dz] = layerDir[l], pr = layerProj[l];
    const bound = pr[clamp(Math.floor((1 - frac) * (pr.length - 1)), 0, pr.length - 1)];
    const y = layerY(l);
    sliceMesh.rotation.y = Math.atan2(dx, dz);
    sliceMesh.material.uniforms.uColor.value.copy(v3(theme.red.map((x) => Math.min(1, x * 1.1 + 0.08))));
    sliceMesh.visible = true;
    fx(0.8, (t) => {
      const d = lerp(R * 1.25, bound, easeIO(t));
      sliceMesh.position.set(dx * d, y, dz * d);
      sliceMesh.material.uniforms.uAlpha.value = (t < 0.8 ? 1 : (1 - t) / 0.2) * 0.95;
      if (t > 0.15 && t < 0.9 && Math.random() < 0.9) { const p = new V3(dx * d + (-dz) * (Math.random() - 0.5) * R * 1.6, y, dz * d + dx * (Math.random() - 0.5) * R * 1.6); emit(p, theme.red, 2, 0.8, 0.9, 0.2); }
      needsRender = true;
    }, () => { sliceMesh.visible = false; });
    ring(y, theme.red, { r1: 0.8, dur: 1.1, alpha: 0.7 });
  }

  /* bursts */
  const NB = 420;
  const bPos = new Float32Array(NB * 3), bCol = new Float32Array(NB * 3), bA = new Float32Array(NB), bS = new Float32Array(NB);
  const bVel = new Float32Array(NB * 3), bAge = new Float32Array(NB).fill(9), bLife = new Float32Array(NB).fill(1), bAmp = new Float32Array(NB), bSize = new Float32Array(NB);
  const bg = new THREE.BufferGeometry();
  bg.setAttribute('position', new THREE.BufferAttribute(bPos, 3)); bg.setAttribute('aCol', new THREE.BufferAttribute(bCol, 3));
  bg.setAttribute('aA', new THREE.BufferAttribute(bA, 1)); bg.setAttribute('aS', new THREE.BufferAttribute(bS, 1));
  bg.boundingSphere = new THREE.Sphere(new V3(), 50);
  const burstPts = new THREE.Points(bg, track(new THREE.ShaderMaterial({ vertexShader: BURST_VS, fragmentShader: SPRITE_FS, uniforms: { uPx: U.uPx, uLight: U.uLight }, transparent: true, depthTest: false, depthWrite: false, blending: blend() })));
  burstPts.frustumCulled = false; burstPts.renderOrder = 5; scene.add(burstPts);
  let bNext = 0, bActive = 0;
  function emit(p, col, n, speed = 1.4, life = 1.1, size = 0.22) {
    if (reduced) return;
    for (let i = 0; i < n; i++) {
      const k = bNext; bNext = (bNext + 1) % NB;
      const th = Math.random() * TAU, ph = Math.acos(2 * Math.random() - 1), sp2 = speed * (0.4 + Math.random() * 0.8);
      bPos[k * 3] = p.x; bPos[k * 3 + 1] = p.y; bPos[k * 3 + 2] = p.z;
      bVel[k * 3] = Math.sin(ph) * Math.cos(th) * sp2; bVel[k * 3 + 1] = Math.cos(ph) * sp2; bVel[k * 3 + 2] = Math.sin(ph) * Math.sin(th) * sp2;
      bCol[k * 3] = col[0]; bCol[k * 3 + 1] = col[1]; bCol[k * 3 + 2] = col[2];
      bAge[k] = 0; bLife[k] = life * (0.6 + Math.random() * 0.7); bAmp[k] = 0.9; bSize[k] = size * (0.6 + Math.random() * 0.8);
    }
    bActive = NB;
  }
  function stepBursts(dt) {
    if (!bActive) return;
    let any = false;
    for (let k = 0; k < NB; k++) {
      if (bAge[k] >= bLife[k]) { bA[k] = 0; continue; }
      any = true; bAge[k] += dt;
      const f = Math.pow(0.12, dt);
      bVel[k * 3] *= f; bVel[k * 3 + 1] *= f; bVel[k * 3 + 2] *= f;
      bPos[k * 3] += bVel[k * 3] * dt; bPos[k * 3 + 1] += bVel[k * 3 + 1] * dt; bPos[k * 3 + 2] += bVel[k * 3 + 2] * dt;
      const u = bAge[k] / bLife[k];
      bA[k] = bAmp[k] * (1 - u) * (1 - u); bS[k] = bSize[k] * (1 - u * 0.5);
    }
    bg.attributes.position.needsUpdate = true; bg.attributes.aA.needsUpdate = true; bg.attributes.aS.needsUpdate = true; bg.attributes.aCol.needsUpdate = true;
    if (!any) bActive = 0;
    needsRender = true;
  }

  /* ---------- parts ---------- */
  const parts = new Map(); // id -> obj
  function partColor(o) {
    const d = o.def;
    if (o.found) return resolve('var(--' + (KIND_TOKEN[d.kind] || 'blue') + ', var(--blue))');
    const base = resolve(donorColor(d.donor));
    const v = VERDICT[d.verdict];
    if (!v) return base;
    const tint = v === 'green' ? theme.green : v === 'red' ? theme.red : theme.grey;
    return base.map((x, i) => lerp(x, tint[i], 0.62));
  }
  function paintPart(o) {
    const c = partColor(o);
    o.target = c;
    if (!o.col) o.col = c.slice();
  }
  function makePart(def, found, index) {
    const rng = mulberry(hashStr(String(def.id)) + 5);
    const pts = blobPoints(def.kind, rng);
    const n = pts.length + 1;
    const pos = new Float32Array(n * 3), seed = new Float32Array(n), size = new Float32Array(n);
    pts.forEach((p, i) => { pos.set(p, (i + 1) * 3); seed[i + 1] = rng(); size[i + 1] = 0.06 + rng() * 0.07; });
    size[0] = 0.62; seed[0] = 0.5; // the glowing core
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1)); g.setAttribute('aSize', new THREE.BufferAttribute(size, 1));
    g.boundingSphere = new THREE.Sphere(new V3(), 2);
    const uni = { uTime: U.uTime, uPx: U.uPx, uCam: U.uCam, uLight: U.uLight, uS: { value: 1 }, uAmp: { value: 0 }, uFlash: { value: 0 }, uHover: { value: 0 }, uColor: { value: new V3() } };
    const m = track(new THREE.ShaderMaterial({ vertexShader: PART_VS, fragmentShader: SPRITE_FS, uniforms: uni, transparent: true, depthTest: false, depthWrite: false, blending: blend() }));
    const mesh = new THREE.Points(g, m); mesh.frustumCulled = false; mesh.renderOrder = 6;
    scene.add(mesh);
    const o = { def, found, index, mesh, uni, link: null, dock: new V3(), anchor: new V3(), amp: 0, appear: 0, flash: 0, hover: 0, flying: false, wait: false, flyU: 0, from: new V3(), ctrl: new V3(), col: null, target: null, grow: 0, healU: 0, sx: -1e4, sy: -1e4, sd: 1, spin: 0.25 + rng() * 0.3 };
    paintPart(o);
    o.col = o.target.slice();
    parts.set(def.id, o);
    return o;
  }
  function layoutPart(o) {
    const d = o.def, l = clamp(Math.round(d.at ?? d.layer ?? 0), 0, cfg.layers - 1);
    o.layer = l;
    const a = layerDock(l, o.index, o.found);
    const rad = R + (o.found ? 2.1 : 1.7 + 0.3 * o.index);
    const y = layerY(l) + (o.found ? 0.55 : 0);
    o.dock.set(Math.cos(a) * rad, y, Math.sin(a) * rad);
    o.anchor.copy(rimPoint(l, a));
    o.angle = a;
    // the link: a curve of glow sprites from the part to the rim
    if (o.link) { scene.remove(o.link); disposeObj(o.link); }
    const N = 54, ps = new Float32Array((N + 1) * 3), aT = new Float32Array(N + 1), aBig = new Float32Array(N + 1);
    const c = new V3().addVectors(o.dock, o.anchor).multiplyScalar(0.5).add(new V3(0, 0.55, 0));
    for (let i = 0; i <= N; i++) {
      const t = i / N, q = Math.min(1, t) , w0 = (1 - q) * (1 - q), w1 = 2 * (1 - q) * q, w2 = q * q;
      ps[i * 3] = w0 * o.dock.x + w1 * c.x + w2 * o.anchor.x; ps[i * 3 + 1] = w0 * o.dock.y + w1 * c.y + w2 * o.anchor.y; ps[i * 3 + 2] = w0 * o.dock.z + w1 * c.z + w2 * o.anchor.z;
      aT[i] = t; aBig[i] = i === N ? 1 : 0;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(ps, 3)); g.setAttribute('aT', new THREE.BufferAttribute(aT, 1)); g.setAttribute('aBig', new THREE.BufferAttribute(aBig, 1));
    g.boundingSphere = new THREE.Sphere(new V3(), 40);
    const uni = { uTime: U.uTime, uPx: U.uPx, uCam: U.uCam, uLight: U.uLight, uGrow: { value: o.grow }, uHeal: { value: 0 }, uFlash: { value: 0 }, uAmp: { value: o.found ? 0.55 : 1 }, uSeed: { value: (hashStr(String(d.id)) % 100) / 100 }, uColor: { value: new V3() } };
    const m = track(new THREE.ShaderMaterial({ vertexShader: LINK_VS, fragmentShader: SPRITE_FS, uniforms: uni, transparent: true, depthTest: false, depthWrite: false, blending: blend() }));
    o.link = new THREE.Points(g, m); o.link.frustumCulled = false; o.link.renderOrder = 5; scene.add(o.link);
    o.linkU = uni;
    // flight path for fly-ins
    const out = new V3(Math.cos(a), 0, Math.sin(a));
    o.from.copy(o.dock).addScaledVector(out, 5.5).add(new V3(-Math.sin(a) * 3, 4.5, Math.cos(a) * 3));
    o.ctrl.copy(o.dock).addScaledVector(out, 2.5).add(new V3(0, 3.8, 0));
    o.mesh.position.copy(o.dock);
  }
  function removePart(o) {
    scene.remove(o.mesh); disposeObj(o.mesh);
    if (o.link) { scene.remove(o.link); disposeObj(o.link); }
    parts.delete(o.def.id);
  }
  function partSig(d) { return [d.id, d.kind, d.at ?? d.layer, d.donor, d.verdict, d.label].join('|'); }
  function syncParts(list, found, first) {
    if (!stackReady) return;
    const want = new Map();
    const byLayer = {};
    (list || []).forEach((d) => { const l = clamp(Math.round(d.at ?? 0), 0, cfg.layers - 1); const k = byLayer[l] = (byLayer[l] ?? -1) + 1; want.set(d.id, { d, found: false, index: k }); });
    const fl = {};
    (found || []).slice(0, 48).forEach((d) => { const key = 'found:' + (d.id ?? d.label); const l = clamp(Math.round(d.layer ?? 0), 0, cfg.layers - 1); const k = fl[l] = (fl[l] ?? -1) + 1; want.set(key, { d: { ...d, id: key, pickId: d.id, at: l }, found: true, index: k }); });
    [...parts.values()].forEach((o) => { if (!want.has(o.def.id)) removePart(o); });
    want.forEach((w, id) => {
      let o = parts.get(id);
      if (o && partSig(o.def) !== partSig(w.d)) { const wasAppear = o.appear; removePart(o); o = null; w.keep = wasAppear; }
      const isNew = !o;
      if (!o) o = makePart(w.d, w.found, w.index);
      o.def = w.d; o.index = w.index; o.found = w.found;
      layoutPart(o); paintPart(o);
      if (isNew) {
        if (first || w.keep != null || reduced) { o.appear = 1; o.grow = 1; o.linkU.uGrow.value = 1; }
        else if (w.found) flyIn(o, 0, { from: o.anchor.clone().multiplyScalar(0.55).setY(layerY(o.layer)), ctrl: o.anchor.clone().add(new V3(0, 1.6, 0)), burst: true, dur: 1.3 });
        else fx(1.0, (t) => { o.appear = ease(t); o.grow = ease(t); o.linkU.uGrow.value = ease(t); }, () => { emit(o.dock, o.col, 24, 1.0, 0.9); });
      } else { o.linkU.uGrow.value = o.grow; }
    });
    buildTicks();
    needsRender = true;
  }
  function flyIn(o, delay, opt = {}) {
    if (reduced) { o.appear = 1; o.grow = 1; o.linkU.uGrow.value = 1; return; }
    o.wait = true; o.grow = 0; o.linkU.uGrow.value = 0; o.appear = 1;
    const from = opt.from || o.from, ctrl = opt.ctrl || o.ctrl;
    let prev = null;
    fx(opt.dur || 1.5, (t) => {
      o.wait = false; o.flying = true; o.flyU = t;
      const e = ease(t), a = (1 - e) * (1 - e), b = 2 * (1 - e) * e, c = e * e;
      o.mesh.position.set(a * from.x + b * ctrl.x + c * o.dock.x, a * from.y + b * ctrl.y + c * o.dock.y, a * from.z + b * ctrl.z + c * o.dock.z);
      o.scaleFly = 0.45 + 0.55 * e;
      if (!prev || o.mesh.position.distanceTo(prev) > 0.18) { emit(o.mesh.position, o.col, 2, 0.35, 0.8, 0.14); prev = o.mesh.position.clone(); }
      needsRender = true;
    }, () => {
      o.flying = false; o.scaleFly = 1;
      land(o);
    }, delay);
  }
  function land(o) {
    o.flash = 1;
    emit(o.dock, o.col, 40, 1.6, 1.1, 0.24);
    emit(o.anchor, o.col, 70, 2.0, 1.2, 0.26);
    ring(layerY(o.layer), o.col, { r0: 0.35, r1: 0.78, dur: 1.2, alpha: 0.95, w: 0.045 });
    fx(0.8, (t) => { o.grow = ease(t); o.linkU.uGrow.value = ease(t) * 1.02; }, () => { o.grow = 1; o.linkU.uGrow.value = 1.05; });
    fx(1.2, (t) => { U.uPkAmp.value = (1 - t) * (1 - t); U.uPk.value = o.layer; }, () => { U.uPkAmp.value = 0; });
  }

  /* ---------- highlight, scan ---------- */
  function setHighlight(h) { cfg.highlight = h; }
  function mapLayer(l, total) { return total && total > 1 && total !== cfg.layers ? (l / (total - 1)) * (cfg.layers - 1) : l; }

  /* ---------- play ---------- */
  function play(p) {
    if (!p || !stackReady || dead) return;
    const L = cfg.layers, mid = Math.floor((L - 1) / 2);
    const at = p.at != null ? clamp(Math.round(p.at), 0, L - 1) : null;
    needsRender = true;
    if (reduced) { return; }
    switch (p.kind) {
      case 'graft': {
        const list = p.partId ? [parts.get(p.partId)].filter(Boolean) : [...parts.values()].filter((o) => !o.found);
        if (!list.length) { const l = at ?? mid; ring(layerY(l), theme.blue, { r1: 0.8 }); emit(rimPoint(l, 1.2), theme.blue, 50, 1.6); break; }
        list.forEach((o, i) => flyIn(o, Math.min(i, 8) * 0.4));
        break;
      }
      case 'cut': {
        const l = at ?? (cfg.cuts[0] ? clamp(Math.round(cfg.cuts[0].layer), 0, L - 1) : mid);
        if (!replayCut(l)) { sliceFx(l, 0.4); emit(rimPoint(l, 0.5), theme.red, 60, 1.8); }
        break;
      }
      case 'heal': {
        const ys = [...parts.values()].filter((o) => !o.found).map((o) => layerY(o.layer));
        const y0 = at != null ? layerY(at) : ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
        const ps = [...parts.values()].filter((o) => !o.found);
        fx(1.2, (t) => { ps.forEach((o) => { o.linkU.uHeal.value = t * 1.3; o.flash = Math.max(o.flash, Math.sin(t * Math.PI) * 0.5); }); }, () => { ps.forEach((o) => { o.linkU.uHeal.value = 0; }); });
        fx(2.0, (t) => { U.uHealY.value = y0; U.uHealR.value = ease(t) * (H * 0.9 + 1); U.uHealAmp.value = 0.95 * (1 - t) * Math.min(1, t * 8); }, () => { U.uHealAmp.value = 0; }, 0.9);
        ring(y0, theme.green, { r1: 0.78, dur: 1.4, alpha: 0.8, delay: 0 });
        break;
      }
      case 'score': {
        const y0 = layerY(0) - 1, y1 = layerY(L - 1) + 1;
        const flashCol = p.ok === true ? theme.green : p.ok === false ? theme.red : theme.blue;
        fx(2.1, (t) => { U.uSigY.value = lerp(y0, y1, easeIO(t)); U.uSigAmp.value = 1; if (Math.random() < 0.5) emit(new V3(Math.cos(t * 40) * R * 0.9, U.uSigY.value, Math.sin(t * 40) * R * 0.9), theme.blue, 1, 0.3, 0.7, 0.16); }, () => {
          U.uSigAmp.value = 0;
          U.uFlashCol.value.copy(v3(flashCol));
          const top = layerY(L - 1);
          ring(top, flashCol, { r1: 0.85, dur: 1.4, alpha: 1, w: 0.06 });
          for (let i = 0; i < 4; i++) emit(new V3(Math.cos(i * 1.57) * R * 0.9, top, Math.sin(i * 1.57) * R * 0.9), flashCol, 30, 1.6, 1.2);
          const t0 = T;
          fx(1.3, (u) => { U.uFlash.value = 0.9 * Math.pow(1 - u, 2); if (p.ok === false) camShake = (1 - u) * 0.12 * (u < 0.35 ? 1 : 0); }, () => { U.uFlash.value = 0; camShake = 0; });
        });
        break;
      }
      case 'probe': {
        const l = at ?? mid;
        ring(layerY(l), theme.blue, { r1: 0.8, dur: 1.2, alpha: 0.95 });
        fx(0.6, () => {}, () => ring(layerY(l), theme.blue, { r1: 0.8, dur: 1.2, alpha: 0.6 }));
        fx(1.4, (t) => { U.uPk.value = l; U.uPkAmp.value = (1 - t) * (1 - t); }, () => { U.uPkAmp.value = 0; });
        emit(rimPoint(l, T), theme.blue, 24, 1.0, 0.9);
        break;
      }
      case 'scan': {
        const y0 = layerY(0) - 0.5, y1 = layerY(L - 1) + 0.5;
        beamDisc.visible = true; beamDisc.material.uniforms.uColor.value.copy(v3(theme.green));
        fx(2.6, (t) => { U.uBeamY.value = lerp(y0, y1, easeIO(t)); U.uBeamOn.value = 1; beamDisc.position.y = U.uBeamY.value; beamDisc.material.uniforms.uAlpha.value = 0.9; }, () => { U.uBeamOn.value = 0; beamDisc.visible = false; });
        break;
      }
      default: ring(layerY(at ?? mid), theme.blue, {});
    }
  }

  /* ---------- ticks (layer labels) ---------- */
  let tickEls = [];
  function buildTicks() {
    ticks.textContent = ''; tickEls = [];
    if (!stackReady) return;
    const set = new Set([0, cfg.layers - 1]);
    if (typeof cfg.highlight === 'number') set.add(clamp(Math.round(cfg.highlight), 0, cfg.layers - 1));
    parts.forEach((o) => { if (!o.found) set.add(o.layer); });
    (cfg.cuts || []).forEach((c) => set.add(clamp(Math.round(c.layer), 0, cfg.layers - 1)));
    [...set].sort((a, b) => a - b).slice(0, 14).forEach((l) => {
      const d = document.createElement('div'); d.className = 'b3d-tick' + (l === cfg.highlight ? ' on' : ''); d.textContent = 'L' + l;
      ticks.appendChild(d); tickEls.push({ el: d, l });
    });
  }

  /* ---------- camera and interaction ---------- */
  let yaw = 0.6, pitch = 0.42, zoom = 1, dragging = false, hoverPart = null, camShake = 0, vyaw = 0;
  let W = 10, Hh = 10;
  function resize() {
    const r = glHost.getBoundingClientRect();
    W = Math.max(40, r.width); Hh = Math.max(40, r.height);
    const pr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pr); renderer.setSize(W, Hh, false);
    camera.aspect = W / Hh; camera.updateProjectionMatrix();
    U.uPx.value = (Hh * pr) / (2 * fovTan);
    needsRender = true;
  }
  const tmpV = new V3();
  function camDist() {
    const hv = Math.max(H, 4) / 2 + 1.7, hh = R + (parts.size ? 3.4 : 1.2);
    return Math.max(hv / fovTan, hh / (fovTan * camera.aspect)) * 1.06 * zoom;
  }
  function placeCamera(dt) {
    const d = camDist(); U.uCam.value = d;
    const cp = Math.cos(pitch);
    camera.position.set(Math.sin(yaw) * cp * d, Math.sin(pitch) * d, Math.cos(yaw) * cp * d);
    if (camShake > 0) camera.position.add(tmpV.set((Math.random() - 0.5) * camShake * 6, (Math.random() - 0.5) * camShake * 6, 0));
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
  }
  let downAt = null;
  const onDown = (e) => { dragging = true; vyaw = 0; downAt = { x: e.clientX, y: e.clientY, moved: 0 }; try { canvas.setPointerCapture(e.pointerId); } catch (x) { /* ok */ } canvas.classList.add('grab'); };
  const onMove = (e) => {
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (dragging && downAt) {
      const dx = e.clientX - downAt.x, dy = e.clientY - downAt.y;
      downAt.moved += Math.abs(dx) + Math.abs(dy); downAt.x = e.clientX; downAt.y = e.clientY;
      yaw -= dx * 0.008; vyaw = -dx * 0.008; pitch = clamp(pitch + dy * 0.006, -0.15, 1.3);
      needsRender = true;
    }
    pickAt(mx, my);
  };
  const onUp = (e) => {
    const wasClick = downAt && downAt.moved < 5;
    dragging = false; canvas.classList.remove('grab');
    try { canvas.releasePointerCapture(e.pointerId); } catch (x) { /* ok */ }
    if (wasClick) { const r = canvas.getBoundingClientRect(); pickAt(e.clientX - r.left, e.clientY - r.top); if (hoverPart) { const cb = propsRef.current.onPick; if (cb) cb(hoverPart.def.pickId ?? hoverPart.def.id); } }
    downAt = null;
  };
  const onLeave = () => { if (!dragging) { hoverPart = null; hideTip(); } };
  const onWheel = (e) => { if (!(e.ctrlKey || e.metaKey)) return; e.preventDefault(); zoom = clamp(zoom * (1 + e.deltaY * 0.0015), 0.55, 1.5); needsRender = true; };
  function pickAt(mx, my) {
    let best = null, bd = 30;
    parts.forEach((o) => { if (o.wait || o.appear < 0.3) return; const d = Math.hypot(o.sx - mx, o.sy - my); if (d < bd) { bd = d; best = o; } });
    if (best !== hoverPart) { hoverPart = best; needsRender = true; }
    if (best) showTip(best, mx, my); else hideTip();
    canvas.style.cursor = best ? 'pointer' : '';
  }
  function showTip(o, mx, my) {
    const d = o.def;
    const kindName = (d.kind || 'part').replace(/_/g, ' ');
    tip.textContent = '';
    const a = document.createElement('b'); a.textContent = d.label || d.id;
    const b = document.createElement('span'); b.textContent = (o.found ? kindName + ' · L' + o.layer : (d.donor ? String(d.donor).split('/').pop() + ' · ' : '') + kindName + ' · L' + o.layer) + (d.verdict ? ' · ' + d.verdict : '');
    tip.append(a, b);
    tip.style.opacity = '1';
    tip.style.transform = 'translate(' + Math.round(clamp(mx + 14, 4, W - 190)) + 'px,' + Math.round(clamp(my + 14, 4, Hh - 54)) + 'px)';
  }
  function hideTip() { tip.style.opacity = '0'; }
  canvas.addEventListener('pointerdown', onDown); canvas.addEventListener('pointermove', onMove); canvas.addEventListener('pointerup', onUp); canvas.addEventListener('pointercancel', onUp); canvas.addEventListener('pointerleave', onLeave); canvas.addEventListener('wheel', onWheel, { passive: false });

  /* ---------- frame ---------- */
  const proj = new V3();
  let last = performance.now();
  function frame(now) {
    raf = 0;
    if (dead) return;
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const run = !reduced && visible && inView;
    if (run) {
      T += dt; U.uTime.value = T;
      for (let i = effects.length - 1; i >= 0; i--) {
        const e = effects[i];
        if (T < e.t0) continue;
        const u = (T - e.t0) / e.dur;
        if (u >= 1) { e.upd(1); effects.splice(i, 1); if (e.done) e.done(); } else e.upd(Math.max(0, u));
      }
      // smoothing
      const hl = typeof cfg.highlight === 'number';
      if (hl) { U.uHi.value = U.uHiOn.value < 0.05 ? cfg.highlight : lerp(U.uHi.value, cfg.highlight, 1 - Math.exp(-dt * 8)); }
      U.uHiOn.value = lerp(U.uHiOn.value, hl ? 1 : 0, 1 - Math.exp(-dt * 6));
      const sc = cfg.scan, donor = cfg.mode === 'donor' && sc;
      if (donor) {
        const target = mapLayer(sc.layer, sc.total);
        U.uScanL.value = lerp(U.uScanL.value, target, 1 - Math.exp(-dt * 5));
        const scanning = sc.total ? sc.layer < sc.total - 1 : true;
        U.uScanOn.value = lerp(U.uScanOn.value, scanning ? 1 : 0, 1 - Math.exp(-dt * 3));
        const by = layerY(clamp(U.uScanL.value, 0, cfg.layers - 1));
        U.uBeamY.value = by; U.uBeamOn.value = U.uScanOn.value;
        beamDisc.visible = U.uScanOn.value > 0.02; beamDisc.position.y = by; beamDisc.material.uniforms.uAlpha.value = 0.8 * U.uScanOn.value; beamDisc.material.uniforms.uColor.value.copy(v3(theme.green));
      } else if (cfg.mode !== 'donor') { U.uScanOn.value = lerp(U.uScanOn.value, 0, 1 - Math.exp(-dt * 6)); }
      if (hiDisc) {
        hiDisc.visible = U.uHiOn.value > 0.03 && stackReady;
        if (hiDisc.visible) { hiDisc.position.y = layerY(clamp(U.uHi.value, 0, cfg.layers - 1)); hiDisc.material.uniforms.uAlpha.value = (0.55 + 0.25 * Math.sin(T * 3)) * U.uHiOn.value; hiDisc.material.uniforms.uColor.value.copy(v3(theme.blue)); }
      }
      // orbit
      if (!dragging) { vyaw *= Math.pow(0.04, dt); yaw += vyaw + (hoverPart ? 0 : dt * 0.06); }
      needsRender = true;
    }
    if (needsRender || run) {
      const L = cfg.layers;
      placeCamera(dt);
      // parts
      parts.forEach((o) => {
        const m = o.mesh, u = o.uni;
        o.flash = Math.max(0, o.flash - dt * 1.6);
        o.hover = lerp(o.hover, o === hoverPart ? 1 : 0, 1 - Math.exp(-dt * 10));
        for (let i = 0; i < 3; i++) o.col[i] = lerp(o.col[i], o.target[i], 1 - Math.exp(-dt * 6));
        u.uColor.value.set(o.col[0], o.col[1], o.col[2]);
        o.linkU.uColor.value.set(o.col[0], o.col[1], o.col[2]);
        u.uAmp.value = o.wait ? 0 : o.appear * (o.found ? 0.85 : 0.95);
        u.uFlash.value = o.flash; u.uHover.value = o.hover; o.linkU.uFlash.value = o.flash * 0.8;
        u.uS.value = (o.scaleFly ?? 1) * (o.found ? 0.8 : 1) * (0.6 + 0.4 * o.appear) * (1 + o.flash * 0.15);
        if (!o.flying) m.position.set(o.dock.x, o.dock.y + Math.sin(T * 1.1 + o.index * 1.7 + (o.layer || 0)) * 0.07, o.dock.z);
        m.rotation.y += dt * o.spin * (reduced ? 0 : 1); m.rotation.x = Math.sin(T * 0.4 + o.index) * 0.2;
        m.visible = !o.wait; o.link.visible = !o.wait;
        proj.copy(m.position).project(camera);
        o.sx = (proj.x * 0.5 + 0.5) * W; o.sy = (-proj.y * 0.5 + 0.5) * Hh; o.sd = proj.z;
      });
      // ticks: screen-left of each layer
      if (tickEls.length) {
        const right = tmpV.set(1, 0, 0).applyQuaternion(camera.quaternion);
        tickEls.forEach((t) => {
          proj.set(0, layerY(t.l), 0).addScaledVector(right, -R * 1.18).project(camera);
          t.el.style.transform = 'translate(' + Math.round((proj.x * 0.5 + 0.5) * W) + 'px,' + Math.round((-proj.y * 0.5 + 0.5) * Hh) + 'px) translate(-100%,-50%)';
        });
      }
      stepBursts(run ? dt : 0);
      for (let l = 0; l < L; l++) { const t = (typeof cfg.highlight === 'number' && Math.abs(cfg.highlight - l) < 0.5) ? 0.25 : 0; if (aStateArr[l] !== t) { aStateArr[l] = t; sheets.geometry.attributes.aState.needsUpdate = true; } }
      renderer.render(scene, camera);
      needsRender = false;
    }
    schedule();
  }
  function schedule() {
    if (dead || raf) return;
    if (reduced) { if (needsRender) raf = requestAnimationFrame(frame); return; }
    if (visible && inView) raf = requestAnimationFrame(frame);
  }
  const kick = () => { last = performance.now(); needsRender = true; schedule(); };

  /* observers */
  const ro = new ResizeObserver(() => { resize(); kick(); }); ro.observe(glHost);
  const io = new IntersectionObserver((es) => { inView = es.some((e) => e.isIntersecting); if (inView) kick(); });
  io.observe(glHost);
  const onVis = () => { visible = !document.hidden; if (visible) kick(); };
  document.addEventListener('visibilitychange', onVis);
  const mo = new MutationObserver(() => { readTheme(); kick(); });
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
  const mq = window.matchMedia ? matchMedia('(prefers-color-scheme: dark)') : null;
  const onMq = () => { readTheme(); kick(); };
  if (mq && mq.addEventListener) mq.addEventListener('change', onMq);

  /* ---------- public ---------- */
  let sigs = {}, firstSet = true;
  function set(p) {
    if (dead) return;
    const layers = Math.max(1, Math.round(p.layers || 16));
    const modeChanged = cfg.mode !== (p.mode || 'host');
    cfg.mode = p.mode || 'host';
    let rebuilt = false;
    if (!theme) readTheme();
    if (layers !== cfg.layers || !stackReady) { [...parts.values()].forEach(removePart); sigs = {}; buildStack(layers); rebuilt = true; }
    else if (modeChanged) readTheme();
    cfg.scan = p.scan || null;
    const hi = typeof p.highlight === 'number' ? clamp(Math.round(p.highlight), 0, layers - 1) : null;
    const hiChanged = hi !== cfg.highlight; setHighlight(hi);
    cfg.cuts = p.cuts || [];
    const cutsSig = JSON.stringify(cfg.cuts.map((c) => [c.layer, c.frac]));
    if (rebuilt || cutsSig !== sigs.cuts) { applyCuts(cfg.cuts, !rebuilt && !firstSet && sigs.cuts != null); sigs.cuts = cutsSig; }
    const partsSig = JSON.stringify((p.parts || []).map((d) => [d.id, d.kind, d.at, d.donor, d.verdict, d.label]));
    const foundSig = JSON.stringify(((p.scan && p.scan.found) || []).map((d) => [d.id, d.layer, d.kind, d.label]));
    if (rebuilt || partsSig !== sigs.parts || foundSig !== sigs.found) { syncParts(p.parts, cfg.mode === 'donor' && p.scan ? p.scan.found : null, firstSet || rebuilt); sigs.parts = partsSig; sigs.found = foundSig; }
    else if (hiChanged) buildTicks();
    if (hiChanged || rebuilt) buildTicks();
    firstSet = false;
    kick();
  }
  function dispose() {
    dead = true;
    if (raf) cancelAnimationFrame(raf);
    ro.disconnect(); io.disconnect(); mo.disconnect(); document.removeEventListener('visibilitychange', onVis);
    if (mq && mq.removeEventListener) mq.removeEventListener('change', onMq);
    canvas.removeEventListener('pointerdown', onDown); canvas.removeEventListener('pointermove', onMove); canvas.removeEventListener('pointerup', onUp); canvas.removeEventListener('pointercancel', onUp); canvas.removeEventListener('pointerleave', onLeave); canvas.removeEventListener('wheel', onWheel);
    scene.traverse((c) => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
    discGeo.dispose(); bg.dispose();
    renderer.dispose();
    try { renderer.forceContextLoss(); } catch (e) { /* ok */ }
    canvas.remove(); disposeProbe();
    effects.length = 0; parts.clear();
  }
  resize();
  return { set, play, dispose };
}

/* ---------- 2D fallback ---------- */
function Fallback({ height, layers = 16, parts = [], cuts = [], highlight = null, onPick, label, pulse }) {
  const W = 420, Hh = Math.max(200, height), L = Math.max(1, layers);
  const top = 26, bot = Hh - 26, step = L > 1 ? (bot - top) / (L - 1) : 0;
  const yOf = (l) => bot - l * step;
  const cutOf = (l) => (cuts.find((c) => Math.round(c.layer) === l) || {}).frac || 0;
  return html`<svg class="b3d-fallback" viewBox=${'0 0 ' + W + ' ' + Hh} preserveAspectRatio="xMidYMid meet" role="img" aria-label=${label || 'Layer stack'}>
    ${Array.from({ length: L }, (_, l) => html`<g key=${l}>
      <ellipse cx="170" cy=${yOf(l)} rx="110" ry=${Math.max(3, Math.min(14, step * 0.45))} fill="none" stroke="var(--blue)" stroke-opacity=${highlight === l ? 0.95 : 0.4} stroke-width=${highlight === l ? 2 : 1.2} />
      ${cutOf(l) > 0 && html`<path d=${'M' + (170 + 110 * (1 - 2 * cutOf(l))) + ',' + (yOf(l) - 8) + ' L' + (170 + 110 * (1 - 2 * cutOf(l))) + ',' + (yOf(l) + 8)} stroke="var(--red)" stroke-width="2" stroke-dasharray="2 3" />`}
    </g>`)}
    ${parts.map((p, i) => { const l = clamp(Math.round(p.at || 0), 0, L - 1); const col = p.verdict === 'contributes' ? 'var(--green)' : p.verdict === 'harmful' ? 'var(--red)' : p.verdict ? 'var(--text-3)' : 'var(--blue)'; const x = 310 + (i % 2) * 40;
      return html`<g key=${p.id} class="b3d-fp" onClick=${() => onPick && onPick(p.id)} style=${{ cursor: onPick ? 'pointer' : 'default' }}><title>${p.label || p.id}</title><path d=${'M280,' + yOf(l) + ' L' + x + ',' + yOf(l)} stroke=${col} stroke-width="1.5" /><circle cx=${x} cy=${yOf(l)} r="9" fill=${col} fill-opacity="0.25" stroke=${col} stroke-width="1.5" /></g>`; })}
    ${pulse && html`<circle key=${pulse.key} class="b3d-fping" cx="170" cy=${yOf(clamp(Math.round(pulse.at ?? Math.floor(L / 2)), 0, L - 1))} r="12" fill="none" stroke=${pulse.ok === false ? 'var(--red)' : pulse.ok === true ? 'var(--green)' : 'var(--blue)'} stroke-width="2" />`}
  </svg>`;
}

/* ---------- the component ---------- */
export function Brain3D(props) {
  const { height = 360, label = null, class: cls } = props;
  const glRef = useRef(), tipRef = useRef(), ticksRef = useRef();
  const eng = useRef(null), pr = useRef(props), pending = useRef(null), lastKey = useRef(props.pulse ? props.pulse.key : undefined);
  const [fb, setFb] = useState(false);
  pr.current = props;
  useEffect(() => {
    if (fb) return undefined;
    let dead = false, e = null;
    (async () => {
      try {
        const THREE = await loadThree();
        if (dead) return;
        e = createEngine(THREE, { gl: glRef.current, tip: tipRef.current, ticks: ticksRef.current }, pr);
        eng.current = e;
        e.set(pr.current);
        if (pending.current) { const p = pending.current; pending.current = null; e.play(p); }
      } catch (err) {
        console.warn('Brain3D falls back to 2D:', err && err.message);
        if (e) { try { e.dispose(); } catch (x) { /* ok */ } }
        eng.current = null;
        if (!dead) setFb(true);
      }
    })();
    return () => { dead = true; if (e) e.dispose(); eng.current = null; };
  }, [fb]);
  useEffect(() => { if (eng.current) eng.current.set(props); });
  useEffect(() => {
    const p = props.pulse;
    if (!p || p.key == null || p.key === lastKey.current) return;
    lastKey.current = p.key;
    if (eng.current) eng.current.play(p); else pending.current = p;
  }, [props.pulse && props.pulse.key]);
  return html`<div class=${'brain3d' + (cls ? ' ' + cls : '')} style=${{ height: height + 'px' }}>
    ${fb ? html`<${Fallback} ...${props} height=${height} />` : html`<div class="b3d-gl" ref=${glRef}></div>`}
    <div class="b3d-ticks" ref=${ticksRef}></div>
    <div class="b3d-tip" ref=${tipRef}></div>
    ${label && html`<div class="b3d-label">${label}</div>`}
  </div>`;
}
