import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { go, toast, pct, clamp, donorColor, donorLetter, useDebouncedCallback, clone, addRecent, patchRecent, setLastBuild } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Menu, StubBadge } from '../components.js';
import { SubtleNum, AutoHeight, SkelReveal, reduced } from '../motion.js';
import { ModelLogo } from '../logos.js';

// slider with optional tick marks (fractions of the range are given as values)
export function Slider({ value, min, max, step, onChange, ticks = [], color, ariaLabel }) {
  const f = (v) => clamp((v - min) / (max - min), 0, 1);
  const pos = (v) => 'calc(8px + (100% - 16px) * ' + f(v) + ')';
  return html`<div class=${'sl' + (ticks.some((t) => t.label) ? ' has-ticks' : '')} style=${{ '--p': 'calc(8px + (100% - 16px) * ' + f(value) + ')', '--c1': color || 'var(--text-1)' }}>
    <input type="range" min=${min} max=${max} step=${step} value=${value} aria-label=${ariaLabel} onInput=${(e) => onChange(parseFloat(e.target.value))} />
    ${ticks.map((t) => html`<span class=${'tk ' + t.kind} style=${{ left: pos(t.v) }}>${t.label && html`<span>${t.label}</span>`}</span>`)}
  </div>`;
}

function BenchRow({ b, onChange, onRemove, late, leaving }) {
  const born = useRef(late);
  const isT = b.kind === 'target';
  const key = isT ? 'target' : 'floor';
  const ticks = [];
  if (b.base != null) ticks.push({ v: b.base, kind: 'base', label: 'host ' + pct(b.base, 0) });
  if (b.lora != null) ticks.push({ v: b.lora, kind: 'lora', label: 'LoRA ' + pct(b.lora) });
  const val = b[key] ?? b.base ?? 0.3;
  return html`<div class=${'srow' + (born.current ? ' t-row-in' : '') + (leaving ? ' t-row-out' : '')}>
    <div class="top">
      <span class="nm">${b.name}${b.recommended && html`<span class="tag b">recommended</span>`}</span>
      <span class="vwrap"><span class=${'val' + (isT ? '' : ' guard')}>${isT ? 'aim for ' : 'stay above '}<${SubtleNum} text=${pct(val, 0)} />%</span><button class="xbtn" data-tip="Remove" aria-label=${'Remove ' + b.name} onClick=${onRemove}><${Icon} n="x" size=${14} /></button></span>
    </div>
    <${Slider} value=${val} min=${0} max=${1} step=${0.01} ticks=${ticks} color=${isT ? 'var(--text-1)' : 'var(--red)'} ariaLabel=${b.name} onChange=${(v) => onChange({ [key]: Math.round(v * 100) / 100 })} />
    <div class="note"><span>${b.base == null ? 'Host score not measured yet. The first run measures it' + (isT ? '.' : '. A guard keeps the model from losing this skill.') : isT ? 'Host starts at ' + pct(b.base) + '%' : 'Host starts at ' + pct(b.base) + '%. A guard keeps the model from losing this skill.'}${b.lora != null ? '. Fine-tuning (LoRA) reaches ' + pct(b.lora) + '%.' : ''}</span></div>
  </div>`;
}

export function SpecScreen({ id }) {
  const [spec, setSpec] = useState(null);
  const [err, setErr] = useState(null);
  const [save, setSave] = useState('saved');
  const [menu, setMenu] = useState(false);
  const [busy, setBusy] = useState(null);
  const latest = useRef(null);
  const dirty = useRef(false);
  const first = useRef(true);
  const [gone, setGone] = useState(null);
  useEffect(() => { if (spec) first.current = false; }, [!!spec]);

  const load = () => { setErr(null); api.getSpec(id).then((s) => { latest.current = s; setSpec(s); addRecent({ kind: 'spec', id, title: s.title }); }).catch(setErr); };
  useEffect(load, [id]);

  const doSave = async () => {
    if (!dirty.current) return;
    dirty.current = false;
    setSave('saving');
    try { await api.putSpec(id, latest.current); patchRecent('spec', id, { title: latest.current.title }); setSave(dirty.current ? 'dirty' : 'saved'); } catch (e) { setSave('error'); toast('Could not save: ' + e.message, 'err'); dirty.current = true; }
  };
  const debounced = useDebouncedCallback(doSave, 700);
  const edit = (fn) => {
    const next = clone(latest.current);
    fn(next);
    latest.current = next;
    dirty.current = true;
    setSpec(next);
    setSave('dirty');
    debounced();
  };
  useEffect(() => () => { debounced.cancel(); if (dirty.current && latest.current) api.putSpec(id, latest.current).catch(() => {}); }, []);

  if (err) return html`<${TopBar} /><${ErrorBox} error=${err} onRetry=${load} />`;
  if (!spec) return html`<${TopBar} /><main class="screen"><div class="doc"><${SkelReveal} ready=${false} skeleton=${html`<div class="skeleton" style="height:48px"></div><div class="skeleton" style="height:420px;margin-top:20px"></div>`} /></div></main>`;

  const targets = spec.benchmarks.filter((b) => b.kind === 'target');
  const guards = spec.benchmarks.filter((b) => b.kind === 'guard');
  const upd = (bid, patch) => edit((s) => { Object.assign(s.benchmarks.find((b) => b.id === bid), patch); });
  const remove = (bid) => edit((s) => {
    const b = s.benchmarks.find((x) => x.id === bid);
    s.benchmarks = s.benchmarks.filter((x) => x.id !== bid);
    const back = { ...b }; delete back.target; delete back.floor; delete back.recommended;
    s.available_benchmarks = [...s.available_benchmarks, back];
  });
  // a removed row fades out first, then the targets card resizes
  const drop = (bid) => { setGone(bid); setTimeout(() => { remove(bid); setGone(null); }, reduced() ? 0 : 170); };
  const add = (a) => { edit((s) => {
    s.available_benchmarks = s.available_benchmarks.filter((x) => x.id !== a.id);
    const nb = { ...a };
    if (a.kind === 'guard') nb.floor = Math.round(Math.max(0, (a.base ?? 0.4) - 0.02) * 100) / 100;
    else nb.target = Math.round(Math.min(0.95, (a.base ?? 0.2) + 0.15) * 100) / 100;
    s.benchmarks = [...s.benchmarks, nb];
  }); setMenu(false); };

  const flush = async () => { debounced.cancel(); await doSave(); };
  const startRun = async () => {
    setBusy('run');
    try { await flush(); const r = await api.createRun(id, null); go('/run/' + r.run_id); } catch (e) { toast(e.message, 'err'); setBusy(null); }
  };
  const startBuilder = async () => {
    setBusy('build');
    try { await flush(); const r = await api.createBuilder(id); try { sessionStorage.setItem('wsp.budget.' + r.builder_id, JSON.stringify(r.budget)); } catch (e) { /* storage can be off */ } setLastBuild({ id: r.builder_id, specId: id }); go('/build/' + r.builder_id + '?spec=' + id); } catch (e) { toast(e.message, 'err'); setBusy(null); }
  };

  const hostB = spec.size.host_params_b;
  const lim = { max_total_params_b: spec.size.max_params_b, max_run_params_b: spec.size.max_params_b, max_heal_min: 20, ...(spec.limits || {}) };
  const totMax = Math.min(spec.size.max_b, Math.max(8, Math.ceil(lim.max_total_params_b * 2)));
  const healMax = Math.max(120, lim.max_heal_min * 2);
  const setLimit = (patch) => edit((s) => {
    s.limits = { ...lim, ...(s.limits || {}), ...patch };
    if (patch.max_total_params_b != null) { s.size.max_params_b = patch.max_total_params_b; s.limits.max_run_params_b = Math.min(s.limits.max_run_params_b, patch.max_total_params_b); }
  });
  const sizeTicks = hostB != null ? [{ v: hostB, kind: 'host', label: 'host ' + hostB.toFixed(2) + 'B' }] : [];
  const b = spec.budget;
  const compMax = Math.max(240, b.compute_min * 3);
  const addable = spec.available_benchmarks || [];

  return html`<${TopBar} right=${html`<span class="savestate">${save === 'saving' ? html`<${Spinner} size=${11} /> Saving` : save === 'error' ? html`<span class="down">Not saved</span>` : save === 'dirty' ? 'Unsaved changes' : html`<${Icon} n="check" size=${13} /> Saved`}</span>`} />
  <main class="screen"><div class="doc"><${SkelReveal} ready=${true} skeleton=${null}>
    <input class="doc-title" value=${spec.title} aria-label="Title" onInput=${(e) => edit((s) => { s.title = e.target.value; })} />
    <textarea class="doc-sum" rows="2" ref=${(el) => { if (el) { el.style.height = 'auto'; el.style.height = el.scrollHeight + 2 + 'px'; } }} value=${spec.summary} aria-label="Summary" onInput=${(e) => edit((s) => { s.summary = e.target.value; })}></textarea>

    <section class="sect"><h2 class="sec">Limits</h2>
      <div class="srow">
        <div class="top"><span class="nm">Total size</span><span class="val">up to <${SubtleNum} text=${lim.max_total_params_b.toFixed(1)} />B parameters</span></div>
        <${Slider} value=${lim.max_total_params_b} min=${Math.max(spec.size.min_b, hostB || 0)} max=${totMax} step=${0.1} ticks=${sizeTicks} ariaLabel="Maximum total parameters" onChange=${(v) => setLimit({ max_total_params_b: Math.round(v * 10) / 10 })} />
        <div class="note"><span>${lim.max_total_params_b < (hostB || 0) ? 'Below the host size. Only deletion masks can get there.' : 'Room for ' + Math.max(0, lim.max_total_params_b - (hostB || 0)).toFixed(2) + 'B of grafted parts.'}</span><span>${Math.max(spec.size.min_b, hostB || 0)}B to ${totMax}B</span></div>
      </div>
      <div class="srow">
        <div class="top"><span class="nm">Run per token</span><span class="val">up to <${SubtleNum} text=${lim.max_run_params_b.toFixed(1)} />B parameters</span></div>
        <${Slider} value=${Math.min(lim.max_run_params_b, lim.max_total_params_b)} min=${Math.min(hostB || 0.5, lim.max_total_params_b)} max=${lim.max_total_params_b} step=${0.1} ticks=${sizeTicks} ariaLabel="Maximum parameters run per token" onChange=${(v) => setLimit({ max_run_params_b: Math.round(v * 10) / 10 })} />
        <div class="note"><span>The parameters a single token passes through. It sets speed and energy.</span><span>up to the total size</span></div>
      </div>
      <div class="srow">
        <div class="top"><span class="nm">Heal time</span><span class="val"><${SubtleNum} text=${String(lim.max_heal_min)} /> min at most</span></div>
        <${Slider} value=${lim.max_heal_min} min=${1} max=${healMax} step=${1} ariaLabel="Maximum heal minutes" onChange=${(v) => setLimit({ max_heal_min: v })} />
        <div class="note"><span>Builds over a limit are refused, with the numbers.</span><span>1 to ${healMax} min</span></div>
      </div>
    </section>

    <section class="sect"><h2 class="sec">Budget</h2>
      <div class="srow">
        <div class="top"><span class="nm">Evals</span><span class="val"><${SubtleNum} text=${String(b.evals)} /> scored attempts</span></div>
        <${Slider} value=${b.evals} min=${b.min_evals} max=${b.max_evals} step=${1} ariaLabel="Evals" onChange=${(v) => edit((s) => { s.budget.evals = v; })} />
        <div class="note"><span>A free fit check does not use an eval.</span><span>${b.min_evals} to ${b.max_evals}</span></div>
      </div>
      <div class="srow">
        <div class="top"><span class="nm">Compute</span><span class="val"><${SubtleNum} text=${String(b.compute_min)} /> min</span></div>
        <${Slider} value=${b.compute_min} min=${5} max=${compMax} step=${5} ariaLabel="Compute minutes" onChange=${(v) => edit((s) => { s.budget.compute_min = v; })} />
        <div class="note"><span>The run stops at whichever limit it hits first.</span><span>5 to ${compMax} min</span></div>
      </div>
    </section>

    <section class="sect"><h2 class="sec"><span>Targets</span><span class="legend"><span class="lg base">host</span><span class="lg lora">LoRA</span></span></h2>
      <${AutoHeight}>${targets.length ? targets.map((x) => html`<${BenchRow} key=${x.id} b=${x} late=${!first.current} leaving=${gone === x.id} onChange=${(p) => upd(x.id, p)} onRemove=${() => drop(x.id)} />`) : html`<div class="empty">No targets. Add one below.</div>`}<//>
    </section>
    <section class="sect"><h2 class="sec"><span>Guards</span></h2>
      <${AutoHeight}>${guards.map((x) => html`<${BenchRow} key=${x.id} b=${x} late=${!first.current} leaving=${gone === x.id} onChange=${(p) => upd(x.id, p)} onRemove=${() => drop(x.id)} />`)}
      ${!guards.length && html`<div class="empty">No guards. A guard stops the model from losing a skill you want to keep.</div>`}<//>
    </section>

    <div class="addwrap">
      <button class="btn" onClick=${() => setMenu(!menu)} disabled=${!addable.length}><${Icon} n="plus" size=${16} /> Add benchmark</button>
      <${Menu} open=${menu} origin="top-left" onClose=${() => setMenu(false)} style=${{ top: '40px', left: 0, minWidth: '280px' }}>
        ${['target', 'guard'].map((k) => html`${addable.some((a) => a.kind === k) && html`<div class="mlabel">${k === 'target' ? 'Targets' : 'Guards'}</div>`}${addable.filter((a) => a.kind === k).map((a) => html`<button class="mi" onClick=${() => add(a)}><span>${a.name}</span><span class="faint">host ${pct(a.base, 0)}%</span></button>`)}`)}
      <//>
    </div>

    ${spec.files && spec.files.length > 0 && html`<section class="sect"><h2 class="sec">Your files</h2>
      ${spec.files.map((f) => html`<div class="filerow"><div><b>${f.name}</b> <span class="faint">${f.rows} rows</span><div class="faint" style="font-size:12px">${f.score_rows} rows held out for scoring, ${f.heal_rows} rows for heal training</div></div>
        <div><div class="splitbar"><i class="s1" style=${{ width: (f.score_rows / f.rows) * 100 + '%' }}></i><i class="s2" style=${{ width: (f.heal_rows / f.rows) * 100 + '%' }}></i></div><div class="splitnote"><span>score ${f.score_rows}</span><span>heal ${f.heal_rows}</span></div></div></div>`)}
    </section>`}

    <section class="sect"><h2 class="sec">Host model</h2>
      <div class="card row-card"><${ModelLogo} model=${spec.host} size=${24} /><div><div class="mono" style="font-size:13px;word-break:break-all">${spec.host}</div><div class="faint" style="font-size:12px">${hostB != null ? hostB.toFixed(2) + 'B parameters' : ''}</div></div></div>
    </section>

    <section class="sect"><h2 class="sec">Likely ingredients</h2>
      <div class="faint" style="font-size:12px;margin:-4px 0 8px">Ranked for this spec. You can change them in the builder.</div>
      <div class="card ings">
        ${(spec.ingredients || []).map((g) => html`<div class="ing"><${ModelLogo} model=${g.donor} size=${20} /><div style="min-width:0"><div style="font-weight:500">${g.label}</div><div class="faint ell" style="font-size:12px">${g.donor}</div></div><span class="tag" style="margin-left:auto">${g.kind}</span></div>`)}
        ${!(spec.ingredients || []).length && html`<div class="empty" style="padding:12px 16px">None picked yet.</div>`}
      </div>
    </section>

    <div class="buildcard">
      <div class="bc-head"><span class="bc-title">How do you want to build it?</span><span class="faint bc-meta">Drafted by ${spec.drafted_by || 'rules'}.${spec.stub ? ' Stub numbers are not real results.' : ''}</span></div>
      <div class="bc-row">
        <button class="btn primary big" disabled=${!!busy} onClick=${startRun}><span class="bt"><b>Let agents build it</b><small>A swarm searches and you watch</small></span>${busy === 'run' ? html`<${Spinner} />` : html`<${Icon} n="arrow" />`}</button>
        <button class="btn big" disabled=${!!busy} onClick=${startBuilder}><span class="bt"><b>Build it myself</b><small>Drag parts onto the host model</small></span>${busy === 'build' ? html`<${Spinner} />` : html`<${Icon} n="arrow" />`}</button>
      </div>
    </div>
  <//></div></main>`;
}
