import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { go, toast, pct, clamp, donorColor, donorLetter, useDebouncedCallback, clone } from '../util.js';
import { TopBar, Icon, Spinner, ErrorBox, Menu, StubBadge } from '../components.js';

// slider with optional tick marks (fractions of the range are given as values)
export function Slider({ value, min, max, step, onChange, ticks = [], color, ariaLabel }) {
  const f = (v) => clamp((v - min) / (max - min), 0, 1);
  const pos = (v) => 'calc(8px + (100% - 16px) * ' + f(v) + ')';
  return html`<div class=${'sl' + (ticks.some((t) => t.label) ? ' has-ticks' : '')} style=${{ '--p': 'calc(8px + (100% - 16px) * ' + f(value) + ')', '--c1': color || 'var(--accent)' }}>
    <input type="range" min=${min} max=${max} step=${step} value=${value} aria-label=${ariaLabel} onInput=${(e) => onChange(parseFloat(e.target.value))} />
    ${ticks.map((t) => html`<span class=${'tk ' + t.kind} style=${{ left: pos(t.v) }}>${t.label && html`<span>${t.label}</span>`}</span>`)}
  </div>`;
}

function BenchRow({ b, onChange, onRemove }) {
  const isT = b.kind === 'target';
  const key = isT ? 'target' : 'floor';
  const ticks = [];
  if (b.base != null) ticks.push({ v: b.base, kind: 'base', label: 'host ' + pct(b.base, 0) });
  if (b.lora != null) ticks.push({ v: b.lora, kind: 'lora', label: 'LoRA ' + pct(b.lora) });
  const val = b[key] ?? b.base ?? 0.3;
  return html`<div class="srow">
    <div class="top">
      <span class="nm">${b.name}${b.recommended && html`<span class="tag a">recommended</span>`}</span>
      <span style="display:flex;align-items:center;gap:8px"><span class="val" style=${{ color: isT ? 'var(--good)' : 'var(--bad)' }}>${isT ? 'aim for ' : 'stay above '}${pct(val, 0)}%</span><button class="xbtn" title="Remove" onClick=${onRemove}><${Icon} n="x" size=${14} /></button></span>
    </div>
    <${Slider} value=${val} min=${0} max=${1} step=${0.01} ticks=${ticks} color=${isT ? 'var(--good)' : 'var(--bad)'} ariaLabel=${b.name} onChange=${(v) => onChange({ [key]: Math.round(v * 100) / 100 })} />
    <div class="note"><span>${isT ? 'Host starts at ' + pct(b.base) + '%' : 'Host starts at ' + pct(b.base) + '%. A guard keeps the model from losing this skill.'}${b.lora != null ? '. Fine-tuning (LoRA) reaches ' + pct(b.lora) + '%.' : ''}</span></div>
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

  const load = () => { setErr(null); api.getSpec(id).then((s) => { latest.current = s; setSpec(s); }).catch(setErr); };
  useEffect(load, [id]);

  const doSave = async () => {
    if (!dirty.current) return;
    dirty.current = false;
    setSave('saving');
    try { await api.putSpec(id, latest.current); setSave(dirty.current ? 'dirty' : 'saved'); } catch (e) { setSave('error'); toast('Could not save: ' + e.message, 'err'); dirty.current = true; }
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
  if (!spec) return html`<${TopBar} /><div class="spec-wrap"><div class="skeleton" style="height:520px"></div><div class="skeleton" style="height:360px"></div></div>`;

  const targets = spec.benchmarks.filter((b) => b.kind === 'target');
  const guards = spec.benchmarks.filter((b) => b.kind === 'guard');
  const upd = (bid, patch) => edit((s) => { Object.assign(s.benchmarks.find((b) => b.id === bid), patch); });
  const remove = (bid) => edit((s) => {
    const b = s.benchmarks.find((x) => x.id === bid);
    s.benchmarks = s.benchmarks.filter((x) => x.id !== bid);
    const back = { ...b }; delete back.target; delete back.floor; delete back.recommended;
    s.available_benchmarks = [...s.available_benchmarks, back];
  });
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
    try { await flush(); const r = await api.createBuilder(id); try { sessionStorage.setItem('wsp.budget.' + r.builder_id, JSON.stringify(r.budget)); } catch (e) { /* storage can be off */ } go('/build/' + r.builder_id + '?spec=' + id); } catch (e) { toast(e.message, 'err'); setBusy(null); }
  };

  const hostB = spec.size.host_params_b;
  const sizeTicks = hostB != null ? [{ v: hostB, kind: 'host', label: 'host ' + hostB.toFixed(2) + 'B' }] : [];
  const b = spec.budget;
  const compMax = Math.max(240, b.compute_min * 3);
  const addable = spec.available_benchmarks || [];

  return html`<${TopBar} crumb=${spec.title} right=${html`<span class="savestate">${save === 'saving' ? html`<${Spinner} size=${11} /> Saving` : save === 'error' ? html`<span class="down">Not saved</span>` : save === 'dirty' ? 'Unsaved changes' : html`<${Icon} n="check" size=${13} /> Saved`}</span>`} />
  <main class="screen"><div class="spec-wrap">
    <div class="card spec-main">
      <input class="title-in" value=${spec.title} aria-label="Title" onInput=${(e) => edit((s) => { s.title = e.target.value; })} />
      <textarea class="sum-in" rows="2" value=${spec.summary} aria-label="Summary" onInput=${(e) => edit((s) => { s.summary = e.target.value; })}></textarea>

      <div class="sect"><div class="label">Size and budget</div>
        <div class="srow">
          <div class="top"><span class="nm">Size limit</span><span class="val">up to ${spec.size.max_params_b.toFixed(1)}B parameters</span></div>
          <${Slider} value=${spec.size.max_params_b} min=${spec.size.min_b} max=${spec.size.max_b} step=${0.1} ticks=${sizeTicks} ariaLabel="Maximum parameters" onChange=${(v) => edit((s) => { s.size.max_params_b = Math.round(v * 10) / 10; })} />
          <div class="note"><span>${spec.size.max_params_b < hostB ? 'Below the host size. Only deletion masks can get there.' : 'Room for ' + Math.max(0, (spec.size.max_params_b - hostB)).toFixed(2) + 'B of grafted parts.'}</span><span>${spec.size.min_b}B to ${spec.size.max_b}B</span></div>
        </div>
        <div class="srow">
          <div class="top"><span class="nm">Evals</span><span class="val">${b.evals} scored attempts</span></div>
          <${Slider} value=${b.evals} min=${b.min_evals} max=${b.max_evals} step=${1} ariaLabel="Evals" onChange=${(v) => edit((s) => { s.budget.evals = v; })} />
          <div class="note"><span>A free fit check does not use an eval.</span><span>${b.min_evals} to ${b.max_evals}</span></div>
        </div>
        <div class="srow">
          <div class="top"><span class="nm">Compute</span><span class="val">${b.compute_min} min</span></div>
          <${Slider} value=${b.compute_min} min=${5} max=${compMax} step=${5} ariaLabel="Compute minutes" onChange=${(v) => edit((s) => { s.budget.compute_min = v; })} />
          <div class="note"><span>The run stops at whichever limit it hits first.</span><span>5 to ${compMax} min</span></div>
        </div>
      </div>

      <div class="sect"><div class="label"><span>Targets</span><span class="legend"><span class="lg base">host</span><span class="lg lora">LoRA</span></span></div>
        ${targets.length ? targets.map((x) => html`<${BenchRow} key=${x.id} b=${x} onChange=${(p) => upd(x.id, p)} onRemove=${() => remove(x.id)} />`) : html`<div class="faint" style="padding:10px 0">No targets. Add one below.</div>`}
      </div>
      <div class="sect"><div class="label"><span>Guards</span></div>
        ${guards.map((x) => html`<${BenchRow} key=${x.id} b=${x} onChange=${(p) => upd(x.id, p)} onRemove=${() => remove(x.id)} />`)}
        ${!guards.length && html`<div class="faint" style="padding:10px 0">No guards. A guard stops the model from losing a skill you want to keep.</div>`}
      </div>

      <div class="addwrap">
        <button class="btn" onClick=${() => setMenu(!menu)} disabled=${!addable.length}><${Icon} n="plus" size=${14} /> Add benchmark</button>
        <${Menu} open=${menu} onClose=${() => setMenu(false)} style=${{ top: '40px', left: 0, minWidth: '280px' }}>
          ${['target', 'guard'].map((k) => html`${addable.some((a) => a.kind === k) && html`<div class="label" style="padding:6px 10px 2px">${k === 'target' ? 'Targets' : 'Guards'}</div>`}${addable.filter((a) => a.kind === k).map((a) => html`<button class="mi" onClick=${() => add(a)}><span>${a.name}</span><span class="faint">host ${pct(a.base, 0)}%</span></button>`)}`)}
        <//>
      </div>

      ${spec.files && spec.files.length > 0 && html`<div class="sect"><div class="label">Your files</div>
        ${spec.files.map((f) => html`<div class="filerow"><div><b>${f.name}</b> <span class="faint">${f.rows} rows</span><div class="faint" style="font-size:11.5px">${f.score_rows} rows held out for scoring, ${f.heal_rows} rows for heal training</div></div>
          <div><div class="splitbar"><i class="s1" style=${{ width: (f.score_rows / f.rows) * 100 + '%' }}></i><i class="s2" style=${{ width: (f.heal_rows / f.rows) * 100 + '%' }}></i></div><div class="note" style="display:flex;justify-content:space-between;font-size:11px;color:var(--faint);margin-top:4px"><span style="color:var(--accent)">score ${f.score_rows}</span><span style="color:var(--good)">heal ${f.heal_rows}</span></div></div></div>`)}
      </div>`}
    </div>

    <aside>
      <div class="card side-card"><h3>Host model</h3>
        <div style="display:flex;gap:10px;align-items:center"><div class="pic" style=${{ background: donorColor(spec.host) }}>${donorLetter(spec.host)}</div><div><div class="mono" style="font-size:12.5px;word-break:break-all">${spec.host}</div><div class="faint" style="font-size:11.5px">${hostB != null ? hostB.toFixed(2) + 'B parameters' : ''}</div></div></div>
      </div>
      <div class="card side-card"><h3>Likely ingredients</h3>
        <div class="faint" style="font-size:11.5px;margin-bottom:4px">Ranked for this spec. You can change them in the builder.</div>
        ${(spec.ingredients || []).map((g) => html`<div class="ing"><div class="pic" style=${{ background: donorColor(g.donor) }}>${donorLetter(g.donor, g.label)}</div><div style="min-width:0"><div style="font-weight:600">${g.label}</div><div class="faint" style="font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${g.donor}</div></div><span class="tag" style="margin-left:auto">${g.kind}</span></div>`)}
        ${!(spec.ingredients || []).length && html`<div class="faint">None picked yet.</div>`}
      </div>
      <div class="card side-card"><h3>How do you want to build it?</h3>
        <div class="cta">
          <button class="btn primary big" disabled=${!!busy} onClick=${startRun}><span><b>Let agents build it</b><small>A swarm searches and you watch</small></span>${busy === 'run' ? html`<${Spinner} />` : html`<${Icon} n="arrow" />`}</button>
          <button class="btn big" disabled=${!!busy} onClick=${startBuilder}><span><b>Build it myself</b><small>Drag parts onto the host model</small></span>${busy === 'build' ? html`<${Spinner} />` : html`<${Icon} n="arrow" />`}</button>
        </div>
        <div class="faint" style="margin-top:12px;font-size:11.5px">Drafted by ${spec.drafted_by || 'rules'}. ${spec.stub ? 'Stub numbers are not real results.' : ''}</div>
      </div>
    </aside>
  </div></main>`;
}
