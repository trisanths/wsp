// Found so far: everything the swarm has learned, derived from the run snapshot alone.
// deriveFindings() is shared with the agent pages, so a finding reads the same everywhere.
import { html, useState, useRef, useMemo } from '../lib.js';
import { pctS, sgn, ago, primaryTarget, clock } from '../util.js';
import { Avatar } from '../components.js';
import { SubtleNum } from '../motion.js';
import { PoweredBy, SponsorMark, ModelLogo } from '../logos.js';
import { openProgramInBuilder } from './shared.js';
import { openPart } from './part-detail.js';
import { nameOf, progName } from './run-shared.js';

const VKEY = { contributes: 'good', 'no better than random': 'noise', harmful: 'bad' };
const ALIAS = { llm_plan: 'plan', analyst_map: 'map', judge_verdict: 'verdict', nudge_sent: 'nudge' };
export const kindOf = (e) => ALIAS[e.kind] || e.kind;
export const actKey = (e) => (e.id != null ? 'i' + e.id : e.ts + '|' + e.agent + '|' + e.kind + '|' + String(e.text).slice(0, 24));
export const shortDonor = (d) => String(d || '').split('/').pop();
export const partLabel = (cat, id) => {
  const p = cat && cat.parts.find((x) => x.id === id);
  return p ? p.label : String(id || 'Part').replace(/_/g, ' ');
};
const plural = (n, one, many) => n + ' ' + (n === 1 ? one : many || one + 's');
const pts = (x, d = 1) => (x >= 0 ? '+' : '-') + Math.abs(x * 100).toFixed(d);

/* ---------- small shared parts ---------- */
export function VDot({ v }) {
  return html`<span class=${'vdot ' + (VKEY[v] || 'none')} title=${v || 'not tested yet'}></span>`;
}

// a part as a small button: donor mark, label, layer, verdict dot. It opens the part window.
export function PartChip({ id, label, at, donor, verdict, stop }) {
  return html`<button type="button" class="pchip" title=${(label || id) + (at != null ? ' at layer ' + at : '') + (verdict ? ', ' + verdict : '') + '. Open the part window.'} onClick=${(e) => { if (stop) e.stopPropagation(); openPart(id); }}>
    ${donor && html`<${ModelLogo} model=${donor} size=${14} />`}<span class="pl">${label || id}</span>${at != null && html`<span class="pat num">L${at}</span>`}<${VDot} v=${verdict} />
  </button>`;
}

function maskCount(p) {
  let n = 0;
  (function walk(x) { if (!x) return; if (x.op === 'use') n++; else if (x.op === 'restrict') walk(x.inner); else { walk(x.a); walk(x.b); } })(p && p.masks);
  return n;
}

// the parts of a program as chips: grafts with donors, skills removed, heal and LoRA
export function ProgramStrip({ program, cat, vmap, stop }) {
  if (!program) return null;
  const grafts = program.grafts || [];
  const masks = maskCount(program);
  return html`<div class="pstrip">
    ${grafts.map((g, i) => {
      const part = cat && cat.parts.find((x) => x.id === g.id);
      return html`<${PartChip} key=${g.id + i} id=${g.id} label=${part ? part.label : partLabel(cat, g.id)} at=${g.at} donor=${part && part.donor} verdict=${vmap && vmap[g.id]} stop=${stop} />`;
    })}
    ${masks > 0 && html`<span class="pchip static" title="Deletion masks switch off a skill"><span class="pl">Removes ${plural(masks, 'skill')}</span></span>`}
    ${program.heal && html`<span class="pchip static" title="Training on general text so the part fits the host"><span class="pl">Heal ${program.heal.steps}</span></span>`}
    ${program.lora && html`<span class="pchip static"><span class="pl">LoRA r${program.lora.rank || 16}</span></span>`}
    ${!grafts.length && !masks && !program.heal && !program.lora && html`<span class="pchip static"><span class="pl">Host only</span></span>`}
  </div>`;
}

/* ---------- derivation ---------- */
// Everything the swarm has learned, from the snapshot. No server fields are needed.
export function deriveFindings(snap, cat) {
  const t = primaryTarget(snap.spec);
  const rows = snap.leaderboard || [];
  const acts = (snap.activity || []).map((e) => ({ ...e, kind: kindOf(e), key: actKey(e) }));
  const nameById = Object.fromEntries((snap.agents || []).map((a) => [a.id, a.name]));
  const evalByHash = {};
  for (const e of acts) if (e.kind === 'eval' && e.data && e.data.program_hash) evalByHash[e.data.program_hash] = e;
  const score = (r) => (t && r.scores ? r.scores[t.id] : null);
  const hostScore = t ? (snap.base_scores || {})[t.id] ?? t.base : null;

  const parts = {};
  const tried = {};
  const donors = {};
  const heal = {};
  const stat = (c) => (parts[c.id] = parts[c.id] || { id: c.id, label: c.label || partLabel(cat, c.id), kind: c.kind, donor: c.donor, n: 0, tested: 0, good: 0, noise: 0, bad: 0, layers: {}, csum: 0, cn: 0, agents: new Set(), hashes: [], latest: null, latestTs: -1, best: null });
  for (const r of rows) {
    const ev = evalByHash[r.program_hash];
    const ts = ev ? ev.ts : null;
    const grafts = (r.program && r.program.grafts) || [];
    for (const g of grafts) {
      const rec = (tried[g.id] = tried[g.id] || { n: 0, layers: new Set() });
      rec.n++; rec.layers.add(g.at);
      const part = cat && cat.parts.find((x) => x.id === g.id);
      const d = part && part.donor;
      if (d) {
        const ds = (donors[d] = donors[d] || { donor: d, n: 0, collapsed: 0, ok: 0, noise: 0, agents: new Set(), hashes: [] });
        ds.n++; ds.agents.add(r.agent); ds.hashes.push(r.program_hash);
        if (r.status === 'collapsed') ds.collapsed++; else if (r.status === 'ok') ds.ok++;
      }
    }
    const hs = r.program && r.program.heal ? r.program.heal.steps : 0;
    if (r.status === 'ok' && score(r) != null) { const h = (heal[hs] = heal[hs] || { steps: hs, n: 0, sum: 0, hashes: [], agents: new Set() }); h.n++; h.sum += score(r); h.hashes.push(r.program_hash); h.agents.add(r.agent); }
    for (const c of r.components || []) {
      const s = stat(c);
      s.n++; s.agents.add(r.agent); s.hashes.push(r.program_hash);
      if (!s.donor) s.donor = c.donor;
      const L = c.host_layer;
      const lay = (s.layers[L] = s.layers[L] || { at: L, n: 0, good: 0, sum: 0, cn: 0 });
      lay.n++;
      const k = VKEY[c.verdict];
      if (k) { s.tested++; if (k === 'good') { s.good++; lay.good++; } else if (k === 'noise') s.noise++; else s.bad++; }
      const con = c.detail && c.detail.contribution;
      if (con != null) { s.csum += con; s.cn++; lay.sum += con; lay.cn++; if (!s.best || con > s.best.con) s.best = { con, hash: r.program_hash, at: L }; }
      if (ts != null && ts > s.latestTs) { s.latestTs = ts; s.latest = c.verdict; }
      else if (ts == null && s.latestTs < 0) s.latest = c.verdict;
      if (c.verdict !== 'contributes' && c.donor && donors[c.donor]) donors[c.donor].noise++;
    }
  }
  // parts only seen as grafts (classic pipeline, or no verdict yet)
  for (const id of Object.keys(tried)) if (!parts[id]) { const part = cat && cat.parts.find((x) => x.id === id); stat({ id, label: part && part.label, kind: part && part.kind, donor: part && part.donor }); parts[id].n = tried[id].n; parts[id].untested = true; }
  const vmap = {};
  for (const s of Object.values(parts)) { s.mean = s.cn ? s.csum / s.cn : null; s.bestLayer = bestLayerOf(s); s.layerSet = tried[s.id] ? [...tried[s.id].layers].sort((a, b) => a - b) : Object.keys(s.layers).map(Number).sort((a, b) => a - b); if (s.latest) vmap[s.id] = s.latest; }

  const items = [];
  const whenOfHashes = (hs) => Math.max(0, ...hs.map((h) => (evalByHash[h] ? evalByHash[h].ts : 0)));
  const evid = (hs) => [...new Set(hs)].map((h) => evalByHash[h]).filter(Boolean);
  const agentsOf = (set) => [...set].filter(Boolean);

  // parts: contributes, dead ends, tried once
  for (const s of Object.values(parts)) {
    const hs = [...new Set(s.hashes)];
    const base = { part: s.id, hashes: hs, agents: agentsOf(s.agents), ts: whenOfHashes(hs) || null, acts: evid(hs) };
    if (s.good > 0) {
      const rate = s.good / s.tested;
      const bl = s.bestLayer;
      items.push({ ...base, id: 'part:' + s.id, type: 'works', tone: rate >= 0.5 ? 'good' : 'warn',
        title: s.label + (rate >= 0.5 ? ' contributes' : ' contributes sometimes'),
        body: 'It contributed in ' + s.good + ' of ' + plural(s.tested, 'scored program') + (s.mean != null ? ', with a mean of ' + pts(s.mean) + ' points when switched off' : '') + (bl != null ? '. Best host layer so far is L' + bl + '.' : '.'),
        chips: [['Mean', s.mean != null ? pts(s.mean) : '-'], ['Layer', bl != null ? 'L' + bl : '-'], ['Seen in', plural(s.n, 'program')]],
        strength: s.good * 10 + (s.mean || 0) * 100 });
    } else if (s.tested >= 2 || s.bad > 0) {
      items.push({ ...base, id: 'dead:' + s.id, type: 'dead', tone: s.bad ? 'bad' : 'noise',
        title: s.label + (s.bad ? ' hurt the model' : ' never beat random'),
        body: (s.bad ? 'It was harmful in ' + s.bad + ' of ' + plural(s.tested, 'scored program') + '. ' : '') + (s.noise ? 'It was no better than a random part of the same size in ' + s.noise + '. ' : '') + 'Tried at ' + s.layerSet.map((l) => 'L' + l).join(', ') + '.',
        chips: [['Tested', String(s.tested)], ['Layers tried', String(s.layerSet.length)]],
        strength: s.tested * 5 + s.bad * 6 });
    } else {
      items.push({ ...base, id: 'once:' + s.id, type: 'open', tone: 'info',
        title: s.label + (s.n > 1 ? ' has no verdict yet' : ' was tried once'),
        body: s.tested ? 'The one scored program called it ' + s.latest + '. One program is not enough to rule it out.' : 'No verdict yet. It needs a scored program with a control.',
        chips: [['Tried', s.n + 'x']], strength: 1 });
    }
  }
  // donors that kept collapsing
  for (const d of Object.values(donors)) {
    if (d.collapsed >= 2 || (d.n >= 3 && d.collapsed / d.n >= 0.5)) {
      const hs = [...new Set(d.hashes)];
      items.push({ id: 'donor:' + d.donor, type: 'dead', tone: 'bad', donor: d.donor, hashes: hs, agents: agentsOf(d.agents), ts: whenOfHashes(hs) || null, acts: evid(hs).filter((e) => e.data && e.data.result === 'collapsed'),
        title: 'Programs with ' + shortDonor(d.donor) + ' keep collapsing',
        body: d.collapsed + ' of ' + plural(d.n, 'program') + ' that use it collapsed (a guard fell below its floor).',
        chips: [['Collapsed', d.collapsed + '/' + d.n]], strength: d.collapsed * 8 });
    }
  }
  // refused over limits
  const refused = acts.filter((e) => e.kind === 'refused');
  if (refused.length) {
    items.push({ id: 'refused', type: 'dead', tone: 'warn', hashes: [], agents: [...new Set(refused.map((e) => e.agent))], ts: refused[refused.length - 1].ts, acts: refused.slice(-6),
      title: plural(Math.max(refused.length, snap.refused || 0), 'program') + ' went over your limits',
      body: 'These proposals were dropped before scoring: ' + refused[refused.length - 1].text, chips: [['Refused', String(Math.max(refused.length, snap.refused || 0))]], strength: refused.length * 3 });
  }
  // the analyst's map rows
  for (const e of acts.filter((x) => x.kind === 'map')) {
    items.push({ id: 'map:' + e.key, type: 'fact', tone: 'info', hashes: [], agents: [e.agent], ts: e.ts, acts: [e], detail: e.data && e.data.detail,
      title: shortSentence(e.text), body: e.text.length > 120 ? e.text : null, chips: [], strength: 4, map: true });
  }
  // heal facts
  const hg = Object.values(heal).sort((a, b) => b.n - a.n);
  if (hg.length >= 2) {
    const [a, b] = hg;
    const ma = a.sum / a.n, mb = b.sum / b.n;
    const hi = ma >= mb ? a : b, lo = hi === a ? b : a;
    const hm = hi.sum / hi.n, lm = lo.sum / lo.n;
    const nm = (h) => (h.steps ? 'heal ' + h.steps : 'no heal');
    items.push({ id: 'heal', type: 'fact', tone: 'info', hashes: [...hi.hashes, ...lo.hashes], agents: [...new Set([...hi.agents, ...lo.agents])], ts: whenOfHashes([...hi.hashes, ...lo.hashes]) || null, acts: evid([...hi.hashes, ...lo.hashes]).slice(-6),
      title: 'Programs with ' + nm(hi) + ' score higher than with ' + nm(lo),
      body: nm(hi).replace(/^./, (c) => c.toUpperCase()) + ' averages ' + pctS(hm) + ' over ' + plural(hi.n, 'program') + ', against ' + pctS(lm) + ' over ' + plural(lo.n, 'program') + ' with ' + nm(lo) + '. A small sample: it is a lean, not a proof.',
      chips: [['Gap', pts(hm - lm)]], strength: Math.abs(hm - lm) * 100 });
  } else if (hg.length === 1 && hg[0].n >= 3) {
    items.push({ id: 'heal', type: 'fact', tone: 'info', hashes: hg[0].hashes, agents: [...hg[0].agents], ts: whenOfHashes(hg[0].hashes) || null, acts: [],
      title: 'Every scored program uses ' + (hg[0].steps ? 'heal ' + hg[0].steps : 'no heal'), body: 'Heal length has not been varied yet, so its effect is unknown.', chips: [], strength: 2 });
  }
  // placement facts
  for (const s of Object.values(parts)) {
    const ls = Object.values(s.layers).sort((a, b) => a.at - b.at);
    if (s.good > 0 && ls.length >= 2) {
      const goodAt = ls.filter((l) => l.good > 0).map((l) => 'L' + l.at);
      const badAt = ls.filter((l) => l.good === 0).map((l) => 'L' + l.at);
      if (goodAt.length && badAt.length) items.push({ id: 'place:' + s.id, type: 'fact', tone: 'info', part: s.id, hashes: [...new Set(s.hashes)], agents: [...s.agents].filter(Boolean), ts: whenOfHashes(s.hashes) || null, acts: evid(s.hashes).slice(-6),
        title: 'Where ' + s.label + ' sits matters', body: 'It contributes at ' + goodAt.join(', ') + ' and did not at ' + badAt.join(', ') + '.', chips: [['Best', s.bestLayer != null ? 'L' + s.bestLayer : '-']], strength: 6 });
    }
  }
  // open questions: layers not tried, untried validated parts, unmeasured references
  const nL = cat && cat.host ? cat.host.layers : null;
  for (const s of Object.values(parts)) {
    if (s.good > 0 && s.bestLayer != null) {
      const near = [s.bestLayer - 2, s.bestLayer - 1, s.bestLayer + 1, s.bestLayer + 2].filter((l) => l >= 1 && (nL == null || l <= nL - 2) && !s.layerSet.includes(l));
      if (near.length) items.push({ id: 'layers:' + s.id, type: 'open', tone: 'info', part: s.id, hashes: [], agents: [], ts: null, acts: [],
        title: s.label + ' is untried at ' + near.map((l) => 'L' + l).join(', '), body: 'It works at L' + s.bestLayer + '. Layers next to it have not been scored.', chips: [], strength: 5 });
    }
  }
  if (cat) {
    const fresh = cat.parts.filter((p) => p.status === 'validated' && p.usable !== false && !tried[p.id] && !parts[p.id]).slice(0, 3);
    for (const p of fresh) items.push({ id: 'untried:' + p.id, type: 'open', tone: 'info', part: p.id, hashes: [], agents: [], ts: null, acts: [], title: p.label + ' has not been tried', body: 'It is a validated part from ' + shortDonor(p.donor) + ' that no program has used yet.', chips: [], strength: 3 });
  }
  const bestRow = snap.best && rows.find((r) => r.program_hash === snap.best.program_hash);
  const cmp = (bestRow && bestRow.comparison) || snap.comparison || null;
  const unm = cmp && (cmp.rows || []).filter((r) => !r.measured);
  if (unm && unm.length) items.push({ id: 'unmeasured', type: 'open', tone: 'info', hashes: bestRow ? [bestRow.program_hash] : [], agents: [], ts: null, acts: [], title: 'The best program is not yet measured against ' + unm.map((r) => r.label).join(', '), body: 'Until those rows are measured the result is not proven.', chips: [], strength: 7 });

  const comps = (bestRow && bestRow.components) || [];
  const best = bestRow ? { row: bestRow, name: nameOf(bestRow, cat), score: score(bestRow), vs: bestRow.vs_control, cmp, comps, hostScore, ts: evalByHash[bestRow.program_hash] ? evalByHash[bestRow.program_hash].ts : null, program: bestRow.program } : null;
  return { parts, vmap, items, best, acts, nameById, evalByHash, donors, heal, tried, refused, t, score };
}

function bestLayerOf(s) {
  const ls = Object.values(s.layers).filter((l) => l.good > 0);
  if (!ls.length) return null;
  ls.sort((a, b) => (b.cn ? b.sum / b.cn : -1) - (a.cn ? a.sum / a.cn : -1) || b.good - a.good);
  return ls[0].at;
}
function shortSentence(t) {
  const s = String(t || '').trim();
  const m = /^(.{20,110}?[.:])(\s|$)/.exec(s);
  return m ? m[1].replace(/[.:]$/, '') : s.length > 110 ? s.slice(0, 107) + '...' : s;
}

/* ---------- page ---------- */
const TYPES = [['all', 'Everything'], ['works', 'What works'], ['dead', 'Dead ends'], ['fact', 'Placement and heal'], ['open', 'Open questions']];
const TYPE_NAME = { works: 'Works', dead: 'Dead end', fact: 'Fact', open: 'Open question' };
const TONE_TAG = { good: 'g', warn: 'w', bad: 'r', noise: '', info: 'b' };

function AgentPill({ id, name, run }) {
  return html`<a class="fag" href=${'#/run/' + run + '/agents/' + id} onClick=${(e) => e.stopPropagation()}><${Avatar} agent=${{ id, name }} size=${16} /><span>${name || id}</span></a>`;
}

function Evidence({ f, snap, d, cat }) {
  const [open, setOpen] = useState(false);
  const prog = f.hashes && f.hashes.length ? (snap.leaderboard.find((r) => r.program_hash === f.hashes[f.hashes.length - 1]) || {}).program : null;
  return html`<div class="fev">
    <div class="fev-btns">
      ${f.part && html`<button class="linkbtn" onClick=${() => openPart(f.part)}>Part window</button>`}
      ${prog && html`<button class="linkbtn" onClick=${() => openProgramInBuilder(snap.spec.id, prog)}>Open program in builder</button>`}
      ${f.acts && f.acts.length > 0 && html`<button class="linkbtn" aria-expanded=${open} onClick=${() => setOpen(!open)}>${open ? 'Hide evidence' : plural(f.acts.length, 'activity row') + ' as evidence'}</button>`}
      <a class="linkbtn" href=${'#/run/' + snap.run_id + '/activity'}>Activity</a>
    </div>
    ${open && html`<div class="fev-rows">${f.acts.slice(-6).map((e) => html`<div class="fev-row" key=${e.key}><span class="num fev-t">${clock(e.ts)}</span><span class="fev-a">${d.nameById[e.agent] || e.agent}</span><span class="fev-x">${e.text}</span></div>`)}${f.detail && html`<pre class="detwell mono">${f.detail}</pre>`}</div>`}
  </div>`;
}

function FindingCard({ f, i, snap, d, cat, fresh }) {
  return html`<article class=${'fcard t-' + f.tone + (fresh ? ' fresh' : ' reveal')} style=${{ '--i': Math.min(i, 12) }} data-type=${f.type}>
    <div class="fc-top"><span class=${'tag ' + (TONE_TAG[f.tone] || '')}>${TYPE_NAME[f.type]}</span>${fresh && html`<span class="tag b newtag">New</span>`}<span class="grow"></span>${f.ts ? html`<span class="faint fc-ago" title=${clock(f.ts)}>${ago(f.ts * 1000)}</span>` : html`<span class="faint fc-ago">no one yet</span>`}</div>
    <h3 class="fc-t">${f.title}</h3>
    ${f.body && html`<p class="fc-b">${f.body}</p>`}
    ${f.chips && f.chips.length > 0 && html`<div class="fc-chips">${f.chips.map(([k, v]) => html`<span class="fchip"><span class="faint">${k}</span> <b class="num">${v}</b></span>`)}</div>`}
    <div class="fc-foot">
      <div class="fc-ag">${f.agents && f.agents.length ? f.agents.slice(0, 3).map((a) => html`<${AgentPill} key=${a} id=${a} name=${d.nameById[a]} run=${snap.run_id} />`) : html`<span class="faint">Found by: nobody yet</span>`}${f.agents && f.agents.length > 3 && html`<span class="faint">+${f.agents.length - 3}</span>`}</div>
    </div>
    <${Evidence} f=${f} snap=${snap} d=${d} cat=${cat} />
  </article>`;
}

function Hero({ d, snap, cat }) {
  const b = d.best;
  if (!b) return html`<section class="rsec fhero empty"><h2 class="rh2">Best program so far</h2><div class="faint">Nothing has been scored yet. The best program and the reasons it wins appear here.</div></section>`;
  const rowsC = (b.cmp && b.cmp.rows) || [];
  const beaten = rowsC.filter((r) => r.measured && r.beaten);
  const lost = rowsC.filter((r) => r.measured && r.beaten === false);
  const gain = b.score != null && b.hostScore != null ? b.score - b.hostScore : null;
  return html`<section class="rsec fhero reveal">
    <div class="rsechead"><h2 class="rh2">Best program so far</h2><span class="faint">${b.ts ? 'found ' + ago(b.ts * 1000) : ''}${b.row.agent && d.nameById[b.row.agent] ? ' by ' + d.nameById[b.row.agent] : ''}</span></div>
    <div class="fh-card">
      <div class="fh-main">
        <div class="fh-name">${b.name}</div>
        <div class="fh-score"><span class="bnum"><${SubtleNum} text=${b.score != null ? pctS(b.score) : '-'} /></span><span class="faint">on ${d.t ? d.t.name : 'the target'}</span>${gain != null && html`<span class=${gain >= 0 ? 'up' : 'down'}>${pts(gain)} over the host</span>`}</div>
        <${ProgramStrip} program=${b.program} cat=${cat} vmap=${d.vmap} />
      </div>
      <div class="fh-why">
        <div class="fh-h">Why it wins</div>
        <ul class="fh-list">
          ${b.vs != null && html`<li class=${b.vs > 0.02 ? 'up' : 'warn'}>${pts(b.vs)} points over the same program with random parts</li>`}
          ${beaten.slice(0, 4).map((r) => html`<li class="up" key=${r.id}>Beats ${r.label}${r.delta != null ? ' (' + sgn(r.delta) + ')' : ''}</li>`)}
          ${lost.slice(0, 3).map((r) => html`<li class="down" key=${r.id}>Does not beat ${r.label}</li>`)}
          ${b.comps.map((c) => html`<li key=${c.id}><${VDot} v=${c.verdict} /> ${c.label}: ${c.verdict}${c.detail && c.detail.contribution != null ? ' (' + pts(c.detail.contribution) + ' points)' : ''}</li>`)}
          ${!b.vs && !rowsC.length && !b.comps.length && html`<li class="faint">The comparison against references runs after the search.</li>`}
        </ul>
      </div>
    </div>
    <div class="fh-act"><button class="btn sm" onClick=${() => openProgramInBuilder(snap.spec.id, b.program)}>Open in builder</button><a class="linkbtn" href=${'#/run/' + snap.run_id + '/results'}>All results</a></div>
  </section>`;
}

const PCOLS = [['label', 'Part'], ['n', 'Tried'], ['good', 'Verdicts'], ['mean', 'Mean effect'], ['layer', 'Best layer']];
function PartsTable({ d, cat }) {
  const [sort, setSort] = useState(['good', -1]);
  const list = Object.values(d.parts);
  const val = (s, k) => (k === 'label' ? s.label.toLowerCase() : k === 'layer' ? (s.bestLayer == null ? -1 : s.bestLayer) : k === 'mean' ? (s.mean == null ? -9 : s.mean) : k === 'good' ? s.good / Math.max(1, s.tested) + s.good / 100 : s[k]);
  const sorted = list.slice().sort((a, b) => { const x = val(a, sort[0]), y = val(b, sort[0]); return (x < y ? -1 : x > y ? 1 : 0) * sort[1]; });
  const set = (k) => setSort(sort[0] === k ? [k, -sort[1]] : [k, k === 'label' ? 1 : -1]);
  if (!list.length) return null;
  return html`<section class="rsec"><h2 class="rh2">Parts the swarm has scored</h2>
    <div class="ptwrap"><table class="ptab"><thead><tr>${PCOLS.map(([k, l]) => html`<th class=${k === 'label' ? '' : 'r'} key=${k}><button class=${'thbtn' + (sort[0] === k ? ' on' : '')} onClick=${() => set(k)} aria-label=${'Sort by ' + l}>${l}${sort[0] === k ? html`<span class="arr">${sort[1] > 0 ? '↑' : '↓'}</span>` : ''}</button></th>`)}</tr></thead>
    <tbody>${sorted.map((s) => html`<tr key=${s.id} class="ptr" onClick=${() => openPart(s.id)} tabindex="0" onKeyDown=${(e) => { if (e.key === 'Enter') openPart(s.id); }}>
      <td><span class="pt-n">${s.donor && html`<${ModelLogo} model=${s.donor} size=${16} />`}<span class="ell">${s.label}</span></span></td>
      <td class="r num">${s.n}</td>
      <td class="r"><span class="vbar" title=${s.good + ' contribute, ' + s.noise + ' no better than random, ' + s.bad + ' harmful'}>${s.tested ? html`<i class="g" style=${{ flex: s.good }}></i><i class="n" style=${{ flex: s.noise }}></i><i class="b" style=${{ flex: s.bad }}></i>` : html`<i class="u" style=${{ flex: 1 }}></i>`}</span><span class="num vcount">${s.tested ? s.good + '/' + s.tested : '-'}</span></td>
      <td class=${'r num ' + (s.mean == null ? 'faint' : s.mean > 0.01 ? 'up' : s.mean < 0 ? 'down' : '')}>${s.mean == null ? '-' : pts(s.mean)}</td>
      <td class="r num">${s.bestLayer != null ? 'L' + s.bestLayer : html`<span class="faint">-</span>`}</td>
    </tr>`)}</tbody></table></div></section>`;
}

export function FindingsPage({ snap, id, cat }) {
  const d = useMemo(() => deriveFindings(snap, cat), [snap, cat]);
  const [type, setType] = useState('all');
  const [who, setWho] = useState('all');
  const [sort, setSort] = useState('new');
  // a finding that was not there on the first render animates in as it arrives
  const seen = useRef(null);
  const freshIds = useRef(new Set());
  if (seen.current === null) seen.current = new Set(d.items.map((f) => f.id));
  else for (const f of d.items) if (!seen.current.has(f.id)) { seen.current.add(f.id); freshIds.current.add(f.id); }

  const counts = { all: d.items.length };
  for (const f of d.items) counts[f.type] = (counts[f.type] || 0) + 1;
  let list = d.items.filter((f) => (type === 'all' || f.type === type) && (who === 'all' || (f.agents || []).includes(who)));
  list = list.slice().sort((a, b) => {
    const fa = freshIds.current.has(a.id) ? 1 : 0, fb = freshIds.current.has(b.id) ? 1 : 0;
    if (fa !== fb) return fb - fa;
    if (sort === 'strong') return b.strength - a.strength;
    if (sort === 'name') return a.title.localeCompare(b.title);
    return (b.ts || 0) - (a.ts || 0) || b.strength - a.strength;
  });
  const finders = snap.agents.filter((a) => d.items.some((f) => (f.agents || []).includes(a.id)));
  const live = !['done', 'error'].includes(snap.status);
  return html`<div class="rpage"><div class="col agwide">
    <div class="fnd-head"><div><h2 class="rh2 fh2">Found so far</h2><div class="faint">${live ? 'Updates as the agents score programs.' : 'What the swarm learned in this run.'} Built from ${plural((snap.leaderboard || []).length, 'scored program')} and ${plural((snap.activity || []).length, 'activity row')}.</div></div><${PoweredBy} who="pastdev" label="Shared through" /></div>
    <${Hero} d=${d} snap=${snap} cat=${cat} />
    <div class="fstats">${[['works', 'Parts that work', 'g'], ['dead', 'Dead ends', 'r'], ['fact', 'Facts', 'b'], ['open', 'Open questions', '']].map(([k, l, c], i) => html`<button class=${'fstat reveal' + (type === k ? ' on' : '')} style=${{ '--i': i }} onClick=${() => setType(type === k ? 'all' : k)}><span class=${'fs-n ' + c}><${SubtleNum} text=${String(counts[k] || 0)} /></span><span class="fs-l">${l}</span></button>`)}</div>
    <${PartsTable} d=${d} cat=${cat} />
    <section class="rsec">
      <div class="rsechead"><h2 class="rh2">What the agents found</h2><span class="faint">${plural(list.length, 'finding')}</span></div>
      <div class="fbar">
        <div class="fchips" role="group" aria-label="Type">${TYPES.map(([k, l]) => html`<button class=${'fchipbtn' + (type === k ? ' on' : '')} onClick=${() => setType(k)}>${l}<span class="num"> ${counts[k] || 0}</span></button>`)}</div>
        <div class="fchips" role="group" aria-label="Found by">${[['all', 'Any agent'], ...finders.map((a) => [a.id, a.name])].map(([k, l]) => html`<button class=${'fchipbtn' + (who === k ? ' on' : '')} onClick=${() => setWho(k)}>${l}</button>`)}</div>
        <label class="fsort faint">Sort <select value=${sort} onChange=${(e) => setSort(e.target.value)}><option value="new">Newest evidence</option><option value="strong">Strongest first</option><option value="name">By name</option></select></label>
      </div>
      ${list.length ? html`<div class="fgrid">${list.map((f, i) => html`<${FindingCard} key=${f.id} f=${f} i=${i} snap=${snap} d=${d} cat=${cat} fresh=${freshIds.current.has(f.id)} />`)}</div>`
        : html`<div class="rempty"><div>${d.items.length ? 'Nothing matches these filters.' : 'No findings yet. They appear when the first programs are scored.'}</div>${d.items.length > 0 && html`<button class="linkbtn" onClick=${() => { setType('all'); setWho('all'); }}>Clear filters</button>`}</div>`}
    </section>
  </div></div>`;
}
