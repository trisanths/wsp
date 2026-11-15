import { html, useState, useEffect, useRef } from '../lib.js';
import { api } from '../api.js';
import { toast, clamp, pct, dur, addRecent, patchRecent, getRecent, useDebouncedCallback, benchName, useModalChip } from '../util.js';
import { TopBar, Icon, ErrorBox, StubBadge } from '../components.js';
import { SubtleNum, SkelReveal, Matrix, SuccessCheck, shake, AutoHeight } from '../motion.js';
import { ModelLogo, PoweredBy } from '../logos.js';
import { Compare } from '../cmp.js';
import { Slider, sizeRef } from './spec.js';
import { MeasuredForPlan } from './measured.js';
import { RULES, METHOD_TEXT, skillName, ruleName, modelName, fmtB, Pick } from './shrink-spec.js';

const fmtMin = (m) => (m == null ? '-' : m < 1 ? Math.round(m * 60) + ' s' : m < 10 ? m.toFixed(1) + ' min' : Math.round(m) + ' min');

// which plan problem belongs to which number
const kindOf = (msg) => (/total size/i.test(msg) ? 'total' : /per-token/i.test(msg) ? 'run' : /heal time/i.test(msg) ? 'heal' : 'other');

function Figure({ label, before, now, limit, bad, tip }) {
  return html`<div class=${'fig' + (bad ? ' bad' : '')}>
    <div class="fig-l">${label}</div>
    <div class="fig-v">
      ${before && before !== now && html`<s class="fig-old" title="The source model">${before}</s>`}
      <span class="fig-new num"><${SubtleNum} text=${now} /></span>
    </div>
    <div class="fig-s"><span>estimated</span>${limit && html`<span>limit ${limit}</span>`}</div>
  </div>`;
}

function Removed({ r }) {
  if (!r) return null;
  if (r.kind === 'moe') {
    return html`<p class="rm">Kept ${r.experts_kept} of ${r.experts_before} experts in each of ${r.layers} layers, so ${r.experts_removed_per_layer} experts per layer are gone. Each token still uses ${r.top_k_after} experts.${r.kept_share_of_skill != null ? ' The kept experts take ' + Math.round(r.kept_share_of_skill * 100) + '% of the work the skill text gave them.' : ''}</p>`;
  }
  const dropped = r.layers_dropped || [];
  return html`<p class="rm">${dropped.length ? 'Dropped ' + dropped.length + ' of ' + r.layers_before + ' layers (' + dropped.map((x) => x + 1).join(', ') + '), ' : 'Kept all ' + r.layers_before + ' layers, '}and ${r.keep_mlp != null ? 'kept ' + Math.round(r.keep_mlp * 100) + '% of the MLP neurons' : 'trimmed the MLP neurons'}${r.keep_heads != null ? ' and ' + Math.round(r.keep_heads * 100) + '% of the attention heads' : ''}. ${r.layers_after != null ? r.layers_after + ' layers are left.' : ''}</p>`;
}

function Result({ job, spec, plan }) {
  const modalChip = useModalChip();
  const r = job.result;
  if (!r) return null;
  const healed = r.heal && r.heal.kind !== 'none' && r.heal.steps > 0;
  const kindText = !r.heal ? null : r.heal.kind === 'distillation' ? 'Healed by distillation toward the original model' : r.heal.kind === 'lm_only' ? 'Healed on language text only, because the original did not fit in memory beside the copy' : 'No heal step ran';
  const names = Object.fromEntries(((spec && spec.benchmarks) || []).map((b) => [b.id, b.name]));
  const tasks = Object.keys(r.scores || {});
  const before = r.params_before_b != null ? r.params_before_b : plan.source_params_b;
  const src = r.source_scores || {};
  const ctl = (r.control && r.control.scores) || {};
  return html`<section class="rsect result" id="result">
    <h2 class="sec"><span>Result</span><span class="sec-r">${r.stub && html`<${StubBadge} small />`}${modalChip && html`<${PoweredBy} who="modal" label="GPUs on" />`}</span></h2>
    <div class="figs">
      <${Figure} label="Stored" before=${fmtB(before, 2)} now=${fmtB(r.params_b, 2)} />
      <${Figure} label="Run per token" before=${fmtB(plan.source_active_params_b, 2)} now=${fmtB(r.active_params_b, 2)} />
      <${Figure} label="Heal time" now=${r.seconds && r.seconds.heal != null ? fmtMin(r.seconds.heal / 60) : '-'} />
    </div>
    ${r.estimated && html`<div class="faint small figs-note">The sizes are estimates. Only a real run counts the parameters.</div>`}
    <h3 class="sub">What was removed</h3>
    <${Removed} r=${r.removed} />
    <h3 class="sub">Scores</h3>
    <div class="scs">${tasks.map((t) => html`<div class="sc" key=${t}>
      <span class="sc-n">${names[t] || benchName(t)}</span>
      <span class="sc-v num">${pct(r.scores[t])}%</span>
      <span class="faint small sc-o">${src[t] != null ? 'original ' + pct(src[t]) + '%' : 'original not scored'}${ctl[t] != null ? ', random removal ' + pct(ctl[t]) + '%' : ''}</span>
    </div>`)}</div>
    <${Compare} c=${r.comparison} heal=${healed} />
    ${(r.limit_problems || []).filter((m) => !(r.comparison && r.comparison.headline && String(r.comparison.headline).includes(m))).map((m, i) => html`<div class="limprob" key=${i}>${m}</div>`)}
    ${healed && kindText && html`<div class="healline">${kindText}.</div>`}
    ${r.log && r.log.length > 0 && html`<h3 class="sub">Log</h3><${LogLines} lines=${r.log} />`}
  </section>`;
}

function LogLines({ lines }) {
  return html`<ol class="tl">${lines.map((l, i) => html`<li key=${i}><i></i><span>${l}</span></li>`)}</ol>`;
}

export function ShrinkScreen({ id, specId }) {
  const modalChip = useModalChip();
  const [plan, setPlan] = useState(null);
  const [spec, setSpec] = useState(null);
  const [err, setErr] = useState(null);
  const [draft, setDraft] = useState({});
  const [job, setJob] = useState(null);
  const [runErr, setRunErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const btn = useRef();
  const alive = useRef(true);
  const seq = useRef(0);
  const pending = useRef({});
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

  const adopt = (p) => {
    setPlan(p);
    setDraft({ keep_experts: p.keep_experts, target_params_b: p.target_params_b, heal_steps: p.heal_steps });
  };
  const load = () => {
    setErr(null);
    api.shrinkPlan(id, {}).then((p) => { if (alive.current) { adopt(p); } }).catch(setErr);
    if (specId) api.getSpec(specId).then((s) => alive.current && setSpec(s)).catch(() => {});
  };
  useEffect(load, [id]);

  // a finished or running job from an earlier visit
  useEffect(() => {
    const me = getRecent().find((r) => r.kind === 'shrink' && r.id === id);
    if (me && me.job) follow(me.job, true);
  }, [id]);

  useEffect(() => { if (plan && spec) addRecent({ kind: 'shrink', id, spec: specId, title: 'Shrink: ' + (spec.title || 'plan'), touch: false }); }, [!!plan, !!spec]);

  // plan edits: the numbers update after a short pause
  const send = useDebouncedCallback(async () => {
    const fields = pending.current;
    pending.current = {};
    const mine = ++seq.current;
    try {
      const p = await api.shrinkPlan(id, fields);
      if (!alive.current || mine !== seq.current) return;
      setPlan(p);
      setDraft((d) => ({ ...d, keep_experts: p.keep_experts, target_params_b: p.target_params_b, heal_steps: p.heal_steps }));
    } catch (e) {
      if (!alive.current) return;
      toast(e.message, 'err');
      api.shrinkPlan(id, {}).then((p) => alive.current && adopt(p)).catch(() => {});
    }
  }, 350);
  const edit = (fields) => {
    pending.current = { ...pending.current, ...fields };
    setDraft((d) => ({ ...d, ...fields }));
    send();
  };

  async function follow(jobId, quiet) {
    setJob({ id: jobId, status: 'queued', result: null });
    try {
      for (;;) {
        await new Promise((r) => setTimeout(r, 650));
        if (!alive.current) return;
        const j = await api.getJob(jobId);
        if (!alive.current) return;
        setJob({ id: jobId, status: j.status, result: j.result, error: j.error, log: j.log });
        if (j.status === 'done' || j.status === 'error') {
          patchRecent('shrink', id, { status: j.status });
          if (j.status === 'error') { setRunErr(j.error || 'The shrink failed'); shake(btn.current); }
          return;
        }
      }
    } catch (e) {
      if (e.status === 404 && quiet) { setJob(null); return; }
      setRunErr(e.message);
      setJob(null);
    }
  }
  const run = async () => {
    if (busy || (job && job.status !== 'done' && job.status !== 'error')) return;
    setRunErr(null);
    setBusy(true);
    send.cancel();
    if (Object.keys(pending.current).length) { await send.flush(); }
    try {
      const { job_id } = await api.shrinkRun(id, {});
      patchRecent('shrink', id, { status: 'running', job: job_id });
      setBusy(false);
      follow(job_id);
    } catch (e) {
      setBusy(false);
      setRunErr(e.message);
      shake(btn.current);
    }
  };

  if (err) return html`<${TopBar} /><${ErrorBox} error=${err} onRetry=${load} />`;
  if (!plan) return html`<${TopBar} /><main class="screen"><div class="doc shrink"><${SkelReveal} ready=${false} skeleton=${html`<div class="skeleton" style="height:36px;width:60%"></div><div class="skeleton" style="height:200px;margin-top:24px"></div>`} /></div></main>`;

  const est = plan.estimate || {};
  const lim = plan.limits || {};
  const probs = (plan.problems || []).map((m) => ({ m, k: kindOf(m) }));
  const has = (k) => probs.some((p) => p.k === k);
  const moe = plan.method === 'experts';
  const nExp = est.experts_before || 128;
  const src = plan.source_params_b;
  const tMin = Math.max(0.1, Math.round(src * 0.05 * 10) / 10);
  const tMax = Math.max(tMin + 0.1, Math.floor((src - 0.05) * 10) / 10);
  const running = !!job && job.status !== 'done' && job.status !== 'error';
  const done = !!job && job.status === 'done' && !!job.result;
  const rules = RULES[plan.method] || [];
  const reason = !plan.ok && probs.length ? probs[0].m.split('. ')[0] + '. Lower the plan, or raise the limit in the spec.' : null;
  const title = (spec && spec.title) || 'Shrink plan';
  const [m1] = METHOD_TEXT[plan.method] || ['Slimming'];

  return html`<${TopBar} />
  <main class="screen"><div class="doc shrink"><${SkelReveal} ready=${true} skeleton=${null}>
    <header class="sh-head">
      <h1 class="sh-title">${title}</h1>
      <div class="sh-sub">
        <span class="sh-src"><${ModelLogo} model=${plan.source_model} size=${20} /><span>${modelName(plan.source_model)}</span><span class="faint">${fmtB(src)}</span></span>
        ${plan.stub && html`<${StubBadge} small />`}
        <span class="grow"></span>
        ${specId && html`<a class="btn sm" href=${'#/spec/' + specId}>Edit spec</a>`}
      </div>
    </header>

    <section class="rsect plan">
      <h2 class="sec">Plan</h2>
      <p class="sh-sum">${plan.summary}</p>
      <div class="figs">
        <${Figure} label="Stored size" before=${fmtB(est.source_params_b != null ? est.source_params_b : src, 2)} now=${fmtB(est.total_params_b, 2)} limit=${lim.max_total_params_b != null ? fmtB(lim.max_total_params_b, 1) : null} bad=${has('total')} />
        <${Figure} label="Run per token" before=${fmtB(est.source_run_params_b != null ? est.source_run_params_b : plan.source_active_params_b, 2)} now=${fmtB(est.run_params_b, 2)} limit=${lim.max_run_params_b != null ? fmtB(lim.max_run_params_b, 1) : null} bad=${has('run')} />
        <${Figure} label="Heal time" now=${fmtMin(est.heal_min)} limit=${lim.max_heal_min != null ? fmtMin(lim.max_heal_min) : null} bad=${has('heal')} />
      </div>
      <${AutoHeight}>${probs.map((p) => html`<div class="prob" key=${p.m}><${Icon} n="warn" size=${14} /><span>${p.m}</span></div>`)}<//>

      <div class="sh-ctl">
        <div class="srow">
          <div class="top"><span class="nm">${moe ? 'Experts kept per layer' : 'Target size'}</span>
            <span class="val">${moe ? html`<${SubtleNum} text=${String(draft.keep_experts ?? plan.keep_experts)} /> of ${nExp}` : html`<${SubtleNum} text=${Number(draft.target_params_b ?? plan.target_params_b).toFixed(1)} />B stored`}</span></div>
          ${moe
            ? html`<${Slider} value=${clamp(draft.keep_experts ?? plan.keep_experts, 1, nExp)} min=${1} max=${nExp} step=${1} ariaLabel="Experts kept per layer" onChange=${(v) => edit({ keep_experts: v })} />`
            : html`<${Slider} value=${clamp(draft.target_params_b ?? plan.target_params_b, tMin, tMax)} min=${tMin} max=${tMax} step=${0.1} ticks=${[{ v: src, kind: 'host', label: 'source ' + src.toFixed(2) + 'B' }]} ariaLabel="Target size in billions of parameters" onChange=${(v) => edit({ target_params_b: Math.round(v * 10) / 10 })} />`}
          <div class="note"><span>${m1}. ${moe ? 'Fewer experts gives a smaller model.' : sizeRef(Number(draft.target_params_b ?? plan.target_params_b))}</span><span>${moe ? '1 to ' + nExp : tMin + 'B to ' + tMax + 'B'}</span></div>
        </div>
        <div class="srow">
          <div class="top"><span class="nm">Rule</span>
            <${Pick} label="Removal rule" value=${ruleName(plan.method, plan.rule)} width=${340} onPick=${(r) => edit({ rule: r })}
              items=${rules.map((r) => ({ ...r, on: r.id === plan.rule }))} /></div>
          <div class="note"><span>Keeping ${skillName(plan.keep_skill).toLowerCase()}. ${(rules.find((r) => r.id === plan.rule) || {}).why || ''}</span></div>
        </div>
        <div class="srow">
          <div class="top"><span class="nm">Heal steps</span><span class="val"><${SubtleNum} text=${String(draft.heal_steps ?? plan.heal_steps)} /> steps</span></div>
          <${Slider} value=${draft.heal_steps ?? plan.heal_steps} min=${0} max=${1000} step=${50} ariaLabel="Heal steps" onChange=${(v) => edit({ heal_steps: v })} />
          <div class="note"><span>The smaller model trains to copy the original. More steps take longer.</span><span>0 to 1000</span></div>
        </div>
      </div>

      <div class="runrow">
        <button ref=${btn} class=${'btn primary big1' + (running ? ' is-busy' : '')} disabled=${running || busy || !plan.ok} onClick=${run}>${running ? html`<${Matrix} variant="scan" />` : done ? html`<${SuccessCheck} size=${14} />` : html`<${Icon} n="play" size=${13} />`} ${running ? 'Shrinking' : done ? 'Run it again' : 'Run the shrink'}</button>
        ${!plan.ok && reason && html`<span class="why">Cannot run yet. ${reason}</span>`}
        ${plan.ok && !running && !runErr && html`<span class="faint small">Nothing runs until you press this. The numbers above are estimates.</span>`}
        <span class="grow"></span>${modalChip && html`<${PoweredBy} who="modal" label="GPUs on" />`}
      </div>
      ${runErr && html`<div class="prob run-err" role="alert"><${Icon} n="warn" size=${14} /><span>${runErr}</span></div>`}
      ${running && html`<div class="jobline2"><span>${job.status === 'queued' ? 'Queued' : 'Running'}</span>${job.log && job.log.length > 0 && html`<${LogLines} lines=${job.log} />`}</div>`}
    </section>

    <${MeasuredForPlan} plan=${plan} />

    ${done && html`<${Result} job=${job} spec=${spec} plan=${plan} />`}
  <//></div></main>`;
}
