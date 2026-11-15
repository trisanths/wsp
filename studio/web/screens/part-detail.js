import { html, useState, useEffect, useRef, useMemo } from '../lib.js';
import { api } from '../api.js';
import { go, toast, getRecent, parseHash, emptyProgram, buildMask, donorShort } from '../util.js';
import { Icon, Spinner } from '../components.js';
import { usePresence, presenceClass, Reel } from '../motion.js';
import { ModelLogo, SponsorMark, PoweredBy } from '../logos.js';
import { StatusTag, KIND_NAMES, lockOf } from '../cmp.js';
import { Brain3D } from '../brain3d.js';
import { TECHNIQUES, techniqueById } from '../techniques.js';
import { openProgramInBuilder } from './shared.js';

// The detail window for one part, opened from the library, the builder palette, the mining pages and the pipeline.
// openPart(id) shows it; PartDetailHost renders the open window once, from app.js.
let listeners = new Set();
let current = null;
export function openPart(id) { current = id || null; listeners.forEach((f) => f(current)); }
export function closePart() { openPart(null); }

const KIND_ONE = { block: 'Block', attn_heads: 'Attention heads', mlp_cluster: 'Neuron cluster', expert: 'Expert', inserted_layer: 'Inserted layers', deletion_mask: 'Deletion mask' };

/** The technique id that most likely found a part, read from its evidence text and kind. */
export function inferTechnique(p) {
  if (!p) return null;
  if (p.technique && techniqueById(p.technique)) return p.technique;
  const e = String(p.evidence || '');
  if (/zeroing the part drops/i.test(e)) return 'ablation';
  if (/router pick rate/i.test(e)) return 'saliency';
  if (/activation|contrast/i.test(e + ' ' + p.id)) return 'activation-scan';
  if (/layer map|skipping layers/i.test(e)) return 'skill-map';
  return { expert: 'saliency', attn_heads: 'ablation', mlp_cluster: 'ablation' }[p.kind] || 'skill-map';
}

let hostInfo = null;
async function getHost() {
  if (hostInfo) return hostInfo;
  try { const c = await api.catalog(); hostInfo = c && c.host ? c.host : { model: '', layers: 16 }; } catch (e) { hostInfo = { model: '', layers: 16 }; }
  return hostInfo;
}

const num = (x, d = 2) => (x == null || isNaN(x) ? '-' : (+x).toFixed(d));
const sizeText = (m) => (m == null ? '-' : m >= 1000 ? (m / 1000).toFixed(2) + 'B' : m + 'M');
const skillName = (s) => String(s || '').replace(/_/g, ' ');
const layerText = (p) => (p.layers ? 'L' + p.layers[0] + (p.layers[1] !== p.layers[0] ? ' to L' + p.layers[1] : '') : '-');

/** Bars: score drop per skill when the part is switched off, against the band of random parts of the same size. */
function SignatureChart({ raw, skill }) {
  const sig = Object.entries((raw && raw.signature) || {}).sort((a, b) => b[1] - a[1]);
  const [on, setOn] = useState(false);
  useEffect(() => { const t = setTimeout(() => setOn(true), 60); return () => clearTimeout(t); }, []);
  if (!sig.length) return html`<div class="faint pd-empty">No per-skill scores are recorded for this part.</div>`;
  const mean = raw.random_mean, sd = raw.random_std;
  const hi = Math.max(...sig.map((s) => s[1]), mean != null ? mean + 2 * (sd || 0) : 0, 0.0001) * 1.08;
  const lo = Math.min(0, ...sig.map((s) => s[1]));
  const span = hi - lo;
  const at = (v) => ((v - lo) / span) * 100;
  const bandL = mean != null ? at(Math.max(lo, mean - 2 * (sd || 0))) : null;
  const bandW = mean != null && sd ? at(mean + 2 * sd) - bandL : 0;
  return html`<div class="pd-sig" role="img" aria-label="Score drop per skill against random parts">
    <div class="pd-sig-rows">
      ${mean != null && html`<div class="pd-band" style=${{ left: 'calc(120px + (100% - 120px - 52px) * ' + (bandL / 100) + ')', width: 'calc((100% - 120px - 52px) * ' + (bandW / 100) + ')' }} title="Random parts of the same size"></div>
        <div class="pd-meanline" style=${{ left: 'calc(120px + (100% - 120px - 52px) * ' + (at(mean) / 100) + ')' }}></div>`}
      ${sig.map(([k, v], i) => html`<div class="pd-sig-row reveal" style=${{ '--i': i }} key=${k}>
        <span class=${'pd-sig-k ell' + (k === skill ? ' own' : '')} title=${skillName(k)}>${skillName(k)}</span>
        <span class="pd-sig-track"><span class=${'pd-sig-bar' + (k === skill ? ' own' : '')} style=${{ left: at(Math.min(0, v)) + '%', width: on ? Math.abs(at(v) - at(0)) + '%' : '0%', transitionDelay: i * 60 + 'ms' }}></span></span>
        <span class="pd-sig-v num">${num(v, 3)}</span>
      </div>`)}
    </div>
    <div class="pd-legend">
      <span><i class="lg own"></i>Score drop</span>
      ${mean != null && html`<span><i class="lg band"></i>Random parts of the same size${sd ? ' (mean plus or minus 2 sd, ' + (raw.random_n || 5) + ' draws)' : ' (mean of ' + (raw.random_n || 5) + ' draws: ' + num(mean, 3) + ')'}</span>`}
    </div>
  </div>`;
}

/** A half-circle gauge for a z-score: how many standard deviations above random parts. */
function ZGauge({ z, zHeld }) {
  const [on, setOn] = useState(false);
  useEffect(() => { const t = setTimeout(() => setOn(true), 80); return () => clearTimeout(t); }, []);
  const R = 62, C = Math.PI * R;
  const MAX = 10;
  const frac = (v) => Math.max(0, Math.min(1, v / MAX));
  const pt = (v) => { const a = Math.PI * (1 - frac(v)); return [80 + R * Math.cos(a), 78 - R * Math.sin(a)]; };
  const [hx, hy] = zHeld != null ? pt(zHeld) : [0, 0];
  if (z == null) return html`<div class="pd-gauge empty"><div class="faint">No z-score recorded.</div><div class="faint" style="font-size:12px">Only parts that went through the random-part gate have one.</div></div>`;
  const tone = z >= 3 ? 'var(--green)' : z > 0 ? 'var(--amber)' : 'var(--red)';
  return html`<div class="pd-gauge">
    <svg viewBox="0 0 160 92" width="180" height="104" role="img" aria-label=${'z-score ' + num(z, 1)}>
      <path d="M18 78 A62 62 0 0 1 142 78" fill="none" stroke="var(--fill-3)" stroke-width="10" stroke-linecap="round" />
      <path class="pd-garc" d="M18 78 A62 62 0 0 1 142 78" fill="none" stroke=${tone} stroke-width="10" stroke-linecap="round" stroke-dasharray=${C} stroke-dashoffset=${on ? C * (1 - frac(z)) : C} />
      ${zHeld != null && html`<circle cx=${hx} cy=${hy} r="5" fill="var(--bg-card)" stroke=${tone} stroke-width="2"><title>Held-out z ${num(zHeld, 1)}</title></circle>`}
      ${[0, 5, 10].map((t) => { const a = Math.PI * (1 - t / MAX); return html`<text key=${t} x=${80 + 80 * Math.cos(a)} y=${80 - 78 * Math.sin(a) + (t === 5 ? -4 : 12)} text-anchor="middle" font-size="9" fill="var(--text-3)">${t}</text>`; })}
    </svg>
    <div class="pd-gnum"><span class="pd-gz"><${Reel} value=${num(z, 1)} /></span><span class="faint">z-score</span></div>
    <div class="faint pd-gnote">Standard deviations above random parts of the same size.${zHeld != null ? ' Held-out items: ' + num(zHeld, 1) + '.' : ''}</div>
  </div>`;
}

function FactRow({ k, children }) {
  return html`<div class="pd-fact"><dt>${k}</dt><dd>${children}</dd></div>`;
}

function Body({ id, data, onClose }) {
  const [host, setHost] = useState(null);
  const [donorInfo, setDonorInfo] = useState(null);
  useEffect(() => { let dead = false; getHost().then((h) => { if (!dead) setHost(h); }); return () => { dead = true; }; }, []);
  useEffect(() => {
    let dead = false;
    if (data && data.donor) api.modelDetail(data.donor).then((m) => { if (!dead) setDonorInfo(m); }).catch(() => {});
    return () => { dead = true; };
  }, [data && data.donor]);
  if (!data) return null;
  const d = data;
  const raw = d.raw || {};
  const tech = techniqueById(inferTechnique(d)) || TECHNIQUES[0];
  const isMask = d.group === 'remove' || !d.donor;
  const lock = lockOf(d);
  const lay = d.layers || [0, 0];
  const donorLayers = Math.max((donorInfo && donorInfo.layers) || 0, lay[1] + 1, 8);
  const hostLayers = (host && host.layers) || 16;
  const at = d.default_at != null ? Math.min(d.default_at, hostLayers - 1) : Math.floor(hostLayers / 3);
  const verdict = raw.z == null ? null : raw.z >= 3 ? 'contributes' : 'no better than random';
  const found = [{ layer: lay[0], kind: d.kind, label: d.label, id: d.id }];
  const addToBuilder = async () => {
    const spec = parseHash().q.get('spec') || (getRecent().find((x) => x.kind === 'spec') || {}).id;
    if (!spec) {
      try { await navigator.clipboard.writeText(d.id); toast('Part id copied. Draft a spec to open the builder with it.', 'info'); } catch (e) { toast('Draft a spec first, then add the part in the builder.', 'info'); }
      return;
    }
    const h = await getHost();
    const prog = emptyProgram(h.model);
    if (isMask) prog.masks = buildMask([d.id]);
    else prog.grafts = [{ id: d.id, at, control: null, control_seed: 0, glue_rank: null }];
    onClose();
    openProgramInBuilder(spec, prog);
  };
  const openModel = () => { onClose(); go('/mine/model/' + d.donor); };
  const copyId = () => { try { navigator.clipboard.writeText(d.id); toast('Part id copied.', 'info'); } catch (e) { /* no clipboard */ } };
  const recipes = d.recipes || [];
  const findings = d.findings || [];
  const stitches = d.stitches || [];
  return html`<div class="pd-body">
    <div class="pd-where reveal" style=${{ '--i': 0 }}>
      ${!isMask && html`<div class="stage3d pd-stage"><${Brain3D} height=${240} layers=${donorLayers} mode="donor" scan=${{ layer: donorLayers, total: donorLayers, found }} highlight=${lay[0]} label=${'In the donor, L' + lay[0] + (lay[1] !== lay[0] ? ' to L' + lay[1] : '')} />
        <div class="stage-hud"><span><${ModelLogo} model=${d.donor} size=${14} /> ${donorShort(d.donor)}</span><span class="num">${donorLayers} layers</span></div></div>
        <div class="stage3d pd-stage"><${Brain3D} height=${240} layers=${hostLayers} mode="host" parts=${[{ id: d.id, label: d.label, kind: d.kind, at, donor: d.donor, verdict }]} pulse=${{ kind: 'graft', at, partId: d.id, ok: null, key: 'pd:' + d.id }} highlight=${at} label=${'Docks at host layer ' + at} />
        <div class="stage-hud"><span>Host${host && host.model ? ': ' + donorShort(host.model) : ''}</span><span class="num">${hostLayers} layers</span></div></div>`}
      ${isMask && html`<div class="stage3d pd-stage wide"><${Brain3D} height=${240} layers=${hostLayers} mode="host" cuts=${[{ layer: Math.floor(hostLayers / 2), frac: 0.18, label: d.label }]} pulse=${{ kind: 'cut', at: Math.floor(hostLayers / 2), ok: null, key: 'pdc:' + d.id }} label="A deletion mask cuts these units" /></div>`}
    </div>
    <dl class="pd-facts reveal" style=${{ '--i': 1 }}>
      ${!isMask && html`<${FactRow} k="Donor"><${ModelLogo} model=${d.donor} size=${16} /> <button class="lnk pd-donorlink" onClick=${openModel}>${d.donor}</button><//>`}
      ${!isMask && html`<${FactRow} k="Donor layers">${layerText(d)}<//>`}
      ${!isMask && html`<${FactRow} k="Host layer">L${at} <span class="faint">default</span><//>`}
      <${FactRow} k="Size">${sizeText(d.params_m)}<//>
      <${FactRow} k="Units">${d.num_units != null ? d.num_units.toLocaleString('en-US') : '-'}<//>
      <${FactRow} k="Tier">${d.tier || '-'}<//>
      <${FactRow} k="Skill">${d.skill ? skillName(d.skill) : (d.skills || []).map(skillName).join(', ') || '-'}<//>
      ${d.selection && html`<${FactRow} k="Selection">${d.selection}<//>`}
      <${FactRow} k="Status"><${StatusTag} status=${d.status} /><//>
    </dl>
    ${!isMask && html`<section class="pd-sect reveal" style=${{ '--i': 2 }}>
      <h3 class="pd-h">Signature against random parts</h3>
      <div class="pd-sigwrap">
        <div class="pd-sigmain"><${SignatureChart} raw=${raw} skill=${d.skill} /></div>
        <${ZGauge} z=${raw.z} zHeld=${raw.z_heldout} />
      </div>
    </section>`}
    <section class="pd-sect reveal" style=${{ '--i': 3 }}>
      <h3 class="pd-h">How it was found</h3>
      <div class="pd-tech">
        <div class="pd-tech-top"><b>${tech.name}</b><${PoweredBy} who=${tech.sponsor} label="runs on" size=${13} /></div>
        <div class="pd-tech-what">${tech.what}</div>
        <div class="faint pd-tech-why">${tech.why}</div>
      </div>
      <div class="pd-evid"><span class="faint">Evidence</span>${d.evidence || 'None recorded.'}</div>
      ${lock && html`<div class="pd-lock"><${Icon} n="lock" size=${14} /> ${lock.why}</div>`}
    </section>
    ${(findings.length > 0 || stitches.length > 0 || recipes.length > 0) && html`<section class="pd-sect reveal" style=${{ '--i': 4 }}>
      ${findings.length > 0 && html`<h3 class="pd-h">Findings</h3><ul class="pd-list">${findings.map((f) => html`<li key=${f.id || f}><a class="lnk" href="#/parts?tab=findings" onClick=${onClose}>${f.title || f.id || f}</a></li>`)}</ul>`}
      ${stitches.length > 0 && html`<h3 class="pd-h">Stitches</h3><ul class="pd-list">${stitches.slice(0, 8).map((s, i) => html`<li key=${i}>${typeof s === 'string' ? s : s.title || s.id || s.summary || JSON.stringify(s)}</li>`)}</ul>`}
      ${recipes.length > 0 && html`<h3 class="pd-h">Recipes that use it</h3><ul class="pd-list">${recipes.map((r) => html`<li key=${r.id || r}><a class="lnk" href="#/parts?tab=recipes" onClick=${onClose}>${r.title || r.id || r}</a></li>`)}</ul>`}
    </section>`}
    <div class="pd-foot">
      <button class="btn ghost sm" onClick=${copyId}>Copy id</button>
      <span class="grow"></span>
      ${!isMask && html`<button class="btn" onClick=${openModel}>Open model</button>`}
      <button class="btn primary" disabled=${!!(lock && lock.rejected)} data-tip=${lock && lock.rejected ? 'Rejected parts cannot go in a program.' : null} onClick=${addToBuilder}>Add to builder</button>
    </div>
  </div>`;
}

const FOCUSABLE = 'a[href], button:not(:disabled), input, select, textarea, [tabindex]:not([tabindex="-1"])';

export function PartDetailHost() {
  const [id, setId] = useState(current);
  const last = useRef(null);
  if (id) last.current = id;
  const st = usePresence(!!id, 170);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const win = useRef(null);
  const from = useRef(null);
  useEffect(() => { listeners.add(setId); return () => listeners.delete(setId); }, []);
  useEffect(() => {
    const h = () => { if (current) closePart(); };
    window.addEventListener('hashchange', h);
    return () => window.removeEventListener('hashchange', h);
  }, []);
  useEffect(() => {
    if (!id) return;
    let dead = false;
    setData(null); setErr(null);
    api.partDetail(id).then((d) => { if (!dead) setData(d); }).catch((e) => { if (!dead) setErr(e); });
    return () => { dead = true; };
  }, [id]);
  useEffect(() => {
    if (!id) return undefined;
    const esc = (e) => { if (e.key === 'Escape') closePart(); };
    document.addEventListener('keydown', esc);
    return () => document.removeEventListener('keydown', esc);
  }, [!!id]);
  useEffect(() => {
    if (!id) return;
    from.current = document.activeElement;
    const t = setTimeout(() => { const b = win.current && win.current.querySelector('.pd-x'); if (b) b.focus(); }, 30);
    document.body.classList.add('pd-open');
    return () => { clearTimeout(t); document.body.classList.remove('pd-open'); if (from.current && from.current.focus) { try { from.current.focus(); } catch (e) { /* gone */ } } };
  }, [!!id]);
  const onKey = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); closePart(); return; }
    if (e.key !== 'Tab' || !win.current) return;
    const f = [...win.current.querySelectorAll(FOCUSABLE)].filter((x) => x.offsetParent !== null);
    if (!f.length) return;
    const a = f[0], z = f[f.length - 1];
    if (e.shiftKey && document.activeElement === a) { e.preventDefault(); z.focus(); } else if (!e.shiftKey && document.activeElement === z) { e.preventDefault(); a.focus(); }
  };
  if (st === 'out') return null;
  const shown = id || last.current;
  const d = data && data.id === shown ? data : null;
  return html`<div class=${'pd-scrim ' + presenceClass(st)} onMouseDown=${(e) => { if (e.target === e.currentTarget) closePart(); }} onKeyDown=${onKey}>
    <div class="pd-win" ref=${win} role="dialog" aria-modal="true" aria-label=${d ? d.label : 'Part details'} tabindex="-1">
      <header class="pd-head">
        ${d ? html`<${ModelLogo} model=${d.donor || ''} size=${30} />` : html`<span class="pd-ph"></span>`}
        <div class="pd-title">
          <h2 class="pd-name ell">${d ? d.label : err ? 'Part not found' : 'Loading the part'}</h2>
          <div class="pd-sub">${d && html`<span class="tag">${KIND_ONE[d.kind] || KIND_NAMES[d.kind] || d.kind || 'Deletion mask'}</span><${StatusTag} status=${d.status} /><span class="mono faint ell">${d.id}</span>`}${!d && !err && html`<span class="faint"><${Spinner} /></span>`}</div>
        </div>
        <button class="pd-x btn ghost sm" aria-label="Close" onClick=${closePart}><${Icon} n="x" size=${16} /></button>
      </header>
      ${err && html`<div class="pd-err">${err.message}<div class="mono faint">${shown}</div></div>`}
      ${!d && !err && html`<div class="pd-skel"><div class="skeleton" style="height:210px"></div><div class="skeleton" style="height:96px;margin-top:12px"></div><div class="skeleton" style="height:140px;margin-top:12px"></div></div>`}
      ${d && html`<${Body} key=${d.id} id=${shown} data=${d} onClose=${closePart} />`}
    </div>
  </div>`;
}
