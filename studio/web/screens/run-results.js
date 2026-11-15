// The Results page: the full leaderboard, the best-score chart, the benchmark targets and the sources.
import { html, useState, useEffect, useRef, Fragment } from '../lib.js';
import { api } from '../api.js';
import { go, pct, pctS, sgn, bsize, primaryTarget } from '../util.js';
import { BenchBar, BenchLegend, CodeView, Tag } from '../components.js';
import { SubtleNum } from '../motion.js';
import { nameOf } from './run-shared.js';

// a thin line chart: the best score so far, the LoRA and target bars as dashed hairlines
export function LineChart({ curve, bench, total }) {
  const W = 760, H = 220, L = 40, R = 14, T = 14, B = 24;
  const vals = [bench.base, bench.target, bench.lora, ...curve.map((c) => c.best), ...curve.map((c) => c.score)].filter((x) => x != null);
  let lo = Math.floor((Math.min(...vals) - 0.03) * 20) / 20;
  let hi = Math.ceil((Math.max(...vals) + 0.04) * 20) / 20;
  lo = Math.max(0, lo); hi = Math.min(1, hi);
  const X = (i) => L + (i / Math.max(1, total)) * (W - L - R);
  const Y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  const ticks = [];
  for (let i = 0; i <= 4; i++) ticks.push(lo + ((hi - lo) * i) / 4);
  const pts = curve.map((c) => [X(c.eval), Y(c.best)]);
  const line = pts.length ? 'M' + X(0) + ',' + Y(bench.base) + ' ' + pts.map((p) => 'L' + p[0] + ',' + p[1]).join(' ') : '';
  return html`<svg class="lchart" viewBox=${'0 0 ' + W + ' ' + H} style=${{ width: '100%', height: 'auto', display: 'block' }}>
    ${ticks.map((t) => html`<g><line class="lc-grid" x1=${L} x2=${W - R} y1=${Y(t)} y2=${Y(t)}/><text class="lc-txt" x=${L - 6} y=${Y(t) + 3} font-size="10" text-anchor="end">${Math.round(t * 100)}%</text></g>`)}
    <text class="lc-txt" x=${L} y=${H - 6} font-size="10">0</text>
    <text class="lc-txt" x=${W - R} y=${H - 6} font-size="10" text-anchor="end">eval ${total}</text>
    ${bench.target != null && html`<g><line class="lc-target" x1=${L} x2=${W - R} y1=${Y(bench.target)} y2=${Y(bench.target)} stroke-dasharray="3 4"/><text class="lc-txt" x=${W - R} y=${Y(bench.target) - 5} font-size="10" text-anchor="end">target ${pct(bench.target, 0)}%</text></g>`}
    ${bench.lora != null && html`<g><line class="lc-lora" x1=${L} x2=${W - R} y1=${Y(bench.lora)} y2=${Y(bench.lora)} stroke-dasharray="6 4"/><text class="lc-txt" x=${W - R} y=${Y(bench.lora) + 12} font-size="10" text-anchor="end">LoRA ${pct(bench.lora)}%</text></g>`}
    <line class="lc-base" x1=${L} x2=${W - R} y1=${Y(bench.base)} y2=${Y(bench.base)} stroke-dasharray="1 4"/>
    ${curve.map((c) => html`<circle class="lc-dot" cx=${X(c.eval)} cy=${Y(c.score)} r="1.8"/>`)}
    ${pts.length > 0 && html`<path class="lc-line" d=${line} fill="none" stroke-width="1.5" stroke-linejoin="round"/><circle class="lc-end" cx=${pts[pts.length - 1][0]} cy=${pts[pts.length - 1][1]} r="3.5"/>`}
    ${!pts.length && html`<text class="lc-txt" x=${W / 2} y=${H / 2} font-size="12" text-anchor="middle">Waiting for the first score</text>`}
  </svg>`;
}

function Detail({ r, snap, cat, text, onBuild }) {
  const bs = snap.spec.benchmarks;
  const name = (aid) => (snap.agents.find((a) => a.id === aid) || { name: aid }).name;
  return html`<div class="detin">
    <div class="detgrid">${bs.map((b) => html`<div><div class="detbar"><span class="dim">${b.name}</span><b class="num">${pct(r.scores[b.id])}%</b></div><${BenchBar} b=${b} value=${r.scores[b.id]} tall /></div>`)}</div>
    ${text != null ? html`<div class="well"><${CodeView} text=${text} /></div>` : html`<div class="faint">Loading the program text.</div>`}
    <div class="detfoot"><span class="faint">Found by ${name(r.agent)}. ${r.vs_control == null ? 'The random-part control has not run yet.' : 'It scores ' + (r.vs_control * 100).toFixed(1) + ' points above its random-part control.'}</span><span class="grow"></span>
      <button class="linkbtn" onClick=${() => onBuild(r.program)}>Open in builder</button>
      <a class="linkbtn" href=${'#/run/' + snap.run_id + '/agents/' + r.agent}>Open ${name(r.agent)}</a></div>
  </div>`;
}

function Leaderboard({ snap, cat, onBuild }) {
  const [open, setOpen] = useState(null);
  const [closing, setClosing] = useState(null);
  const [texts, setTexts] = useState({});
  const timer = useRef(0);
  const bs = snap.spec.benchmarks;
  const toggle = (r) => {
    clearTimeout(timer.current);
    if (open === r.program_hash) { setClosing(r.program_hash); setOpen(null); timer.current = setTimeout(() => setClosing(null), 300); return; }
    setClosing(open);
    setOpen(r.program_hash);
    timer.current = setTimeout(() => setClosing(null), 300);
    if (r.program && texts[r.program_hash] == null) api.programText(r.program).then((x) => setTexts((t) => ({ ...t, [r.program_hash]: x.text }))).catch(() => setTexts((t) => ({ ...t, [r.program_hash]: '' })));
  };
  useEffect(() => () => clearTimeout(timer.current), []);
  if (!snap.leaderboard.length) return html`<div class="faint">No scored programs yet.</div>`;
  const short = (n) => (n.length > 10 ? n.slice(0, 9) + '.' : n);
  return html`<div class="tscroll"><table class="lb rlb"><thead><tr><th style="width:32px">#</th><th>Program</th>${bs.map((b) => html`<th class="r" style="width:76px" title=${b.name}>${short(b.name)}</th>`)}<th class="r" style="width:84px">vs random</th><th class="r" style="width:64px">Size</th><th style="width:28px"></th></tr></thead>
    <tbody>${snap.leaderboard.map((r) => {
      const ok = r.status === 'ok';
      const isOpen = open === r.program_hash;
      const body = isOpen || closing === r.program_hash;
      return html`<${Fragment} key=${r.program_hash}>
        <tr class=${'row' + (isOpen ? ' open' : '')} onClick=${() => toggle(r)}>
          <td class="num dim3">${r.rank}</td><td><div class="desc ell" title=${nameOf(r, cat)}>${nameOf(r, cat)}</div></td>
          ${bs.map((b) => { const v = r.scores[b.id]; const bad = b.kind === 'guard' && v != null && v < b.floor; return html`<td class=${'num r' + (bad ? ' down' : '')}>${v != null ? html`<${SubtleNum} text=${pct(v)} />` : '-'}</td>`; })}
          <td class="r">${!ok ? html`<${Tag} kind="r">${r.status.replace(/_/g, ' ')}<//>` : r.vs_control == null ? html`<span class="faint">pending</span>` : html`<span class=${r.vs_control > 0.02 ? 'up' : 'dim3'}>${sgn(r.vs_control)}</span>`}</td>
          <td class="num r">${bsize(r.params_b)}</td>
          <td><span class="t-acc faint" data-open=${isOpen ? 'true' : 'false'}><span class="t-acc-chevron"><svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5L8 10.5L12 6.5"/></svg></span></span></td>
        </tr>
        <tr class="t-acc det" data-open=${isOpen ? 'true' : 'false'}><td colspan=${bs.length + 5}><div class="t-acc-panel"><div class="t-acc-panel-inner">${body && html`<${Detail} r=${r} snap=${snap} cat=${cat} text=${texts[r.program_hash]} onBuild=${onBuild} />`}</div></div></td></tr>
      <//>`;
    })}</tbody></table></div>`;
}

function Targets({ snap }) {
  const bs = snap.spec.benchmarks;
  return html`<div>
    <div class="tlegend"><${BenchLegend} guard /></div>
    ${bs.map((b) => { const v = snap.best ? snap.best.scores[b.id] : null; return html`<div class="brow2" key=${b.id}><span class="nm" title=${b.name}>${b.name}</span><${BenchBar} b=${b} value=${v ?? b.base} tall /><span class="num ps"><${SubtleNum} text=${pct(v ?? b.base)} />%</span></div>`; })}
  </div>`;
}

function Sources({ snap }) {
  const mx = Math.max(1, ...snap.sources.map((s) => s.uses));
  return html`<div>${snap.sources.map((s) => html`<div class="srcrow" key=${s.model}><span class="sm ell" title=${s.model}>${s.model}</span><span class="faint">${s.role}</span><div class="btrack"><i style=${{ width: (s.uses / mx) * 100 + '%' }}></i></div><span class="num dim ps">${s.uses}</span></div>`)}
    ${snap.sources.length < 2 && html`<div class="faint" style="margin-top:8px">Donors appear here as agents use them.</div>`}</div>`;
}

export function ResultsPage({ snap, id, cat, onBuild }) {
  const t = primaryTarget(snap.spec);
  return html`<div class="rpage"><div class="col wide">
    <section class="rsec"><h2 class="rh2">Leaderboard</h2><${Leaderboard} snap=${snap} cat=${cat} onBuild=${onBuild} /></section>
    <section class="rsec"><h2 class="rh2">${t ? 'Best ' + t.name + ' per eval' : 'Best score per eval'}</h2>${t && html`<${LineChart} curve=${snap.curve} bench=${t} total=${snap.budget.evals_total} />`}</section>
    <section class="rsec"><h2 class="rh2">Benchmarks</h2><${Targets} snap=${snap} /></section>
    <section class="rsec"><h2 class="rh2">Source models</h2><${Sources} snap=${snap} /></section>
  </div></div>`;
}
