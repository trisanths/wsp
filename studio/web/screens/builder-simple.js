import { html, useState, useMemo, useEffect } from '../lib.js';
import { DEMO } from '../demo-flag.js';
import { Icon, Spinner } from '../components.js';
import { SuccessCheck } from '../motion.js';
import { ModelLogo } from '../logos.js';
import { openPart } from './part-detail.js';
import { lockOf, filterItems, Pager } from '../cmp.js';
import { maskIds as maskIdsOf } from '../util.js';
import { pickSkills, browseOrder, dropTwins, fitsRoom, plainName, plainNames, plainSentence, sizeText, trustText, pickForgets, forgetName, resultOf, failText, progressText, sizeLine, polishText } from './builder-simple-logic.js';

const BROWSE_SIZE = 8;

function Help({ children }) {
  return html`<details class="bs-help"><summary>What does this mean?</summary><p>${children}</p></details>`;
}

function Step({ n, id, title, tag, lead, children }) {
  return html`<section class="bs-step" aria-labelledby=${'bs-h-' + id} data-step=${id}>
    <div class="bs-head"><span class="bs-n" aria-hidden="true">${n}</span><h2 class="bs-title" id=${'bs-h-' + id}>${title}</h2>${tag && html`<span class="tag">${tag}</span>`}</div>
    ${lead && html`<p class="bs-lead">${lead}</p>`}
    ${children}
  </section>`;
}

function SkillCard({ part, spec, hostB, on, onToggle, tight, tooBig, name: given }) {
  const name = given || plainName(part, spec);
  const trust = trustText(part);
  return html`<div class=${'bs-card card' + (on ? ' on' : '') + (tight ? ' tight' : '')}>
    <div class="bs-card-top"><${ModelLogo} model=${part.donor} size=${28} alt="" />
      <div class="bs-card-text"><div class="bs-name">${name}</div>
        <div class="bs-sent">${plainSentence(part)}</div>
        <div class="bs-meta"><span>${sizeText(part, hostB)}</span>${tooBig && html`<span class="bs-big-note">Too big for this model</span>`}${trust && html`<span class=${'bs-trust' + (part.status === 'validated' ? ' g' : '')}>${trust}</span>`}</div></div></div>
    <div class="bs-card-act">
      <button class="btn ghost sm" type="button" onClick=${() => openPart(part.id)} aria-label=${'More about ' + name}>More</button>
      <button class=${'btn sm bs-add' + (on ? ' on' : tooBig ? '' : ' primary')} type="button" disabled=${tooBig && !on} aria-pressed=${on ? 'true' : 'false'} aria-label=${(on ? 'Remove ' : 'Add ') + name + (tooBig && !on ? ' (too big for this model)' : '')} onClick=${() => onToggle(part, !on)}>${tooBig && !on ? 'Too big' : on ? html`<${Icon} n="check" size=${14} /> Added` : html`<${Icon} n="plus" size=${14} /> Add`}</button>
    </div>
  </div>`;
}

function Browse({ cat, spec, hostB, usedIds, onToggle, roomB }) {
  const [q, setQ] = useState('');
  const [page, setPage] = useState(1);
  const [showBig, setShowBig] = useState(false);
  const all = useMemo(() => dropTwins(browseOrder(filterItems(cat.parts.filter((p) => !p.group || p.group === 'add'), { kind: null, status: null, placeable: true }, q), spec)), [cat, q, spec && spec.id]);
  // parts that fit the room left under the size limit; added parts stay so they can be removed
  const fits = (p) => usedIds.has(p.id) || fitsRoom(p, roomB);
  const small = all.filter(fits);
  const bigN = all.length - small.length;
  const items = showBig ? [...small, ...all.filter((p) => !fits(p))] : small;
  useEffect(() => { setPage(1); }, [q, showBig]);
  const shown = items.slice((page - 1) * BROWSE_SIZE, page * BROWSE_SIZE);
  const shownNames = plainNames(shown, spec);
  return html`<div class="bs-browse">
    <div class="searchbox bs-search"><${Icon} n="search" size=${15} /><input type="search" placeholder="Search skills, such as math or a model name" aria-label="Search skills" value=${q} onInput=${(e) => setQ(e.target.value)} /></div>
    ${bigN > 0 && html`<label class="bs-bigtoggle"><input type="checkbox" checked=${showBig} onChange=${(e) => setShowBig(e.target.checked)} /> Show parts that are too big (${bigN})</label>`}
    ${shown.map((p, i) => html`<${SkillCard} key=${p.id} name=${shownNames[i]} part=${p} spec=${spec} hostB=${hostB} on=${usedIds.has(p.id)} onToggle=${onToggle} tooBig=${!fits(p)} tight />`)}
    ${!shown.length && html`<div class="faint bs-none">${q ? 'Nothing matches "' + q + '".' : 'Every other skill is too big for this model.'}${!q && bigN > 0 && !showBig ? ' Turn on the box above to see them.' : ''}${q && bigN > 0 && !showBig ? ' ' + bigN + (bigN === 1 ? ' match is' : ' matches are') + ' too big. Turn on the box above to see them.' : ''}</div>`}
    <${Pager} page=${page} total=${items.length} size=${BROWSE_SIZE} onPage=${setPage} label="Skill pages" />
  </div>`;
}

/** The simple builder: one column of plain steps over the same program the advanced builder edits. */
export function SimpleBuilder({ gone, earlierN, cat, spec, program, brain, est, totalB, runB, lim, attempts, job, flash, stub, handing, onAdd, onRemove, onToggleMask, onPolish, onScore, onHandoff, onAdvanced }) {
  const [browse, setBrowse] = useState(false);
  const hostB = cat.host.params_b || 0;
  const usedIds = new Set(program.grafts.map((g) => g.id));
  const limB = lim && lim.max_total_params_b != null ? lim.max_total_params_b : null;
  const picked = useMemo(() => pickSkills(cat, spec, lockOf, 6, limB != null ? limB - hostB : null), [cat, spec && spec.id, limB, hostB]);
  // parts added some other way (Advanced mode, browse) stay on the list so they can be removed
  const extra = program.grafts.map((g) => g.id).filter((id, i, a) => a.indexOf(id) === i && !picked.some((p) => p.id === id)).map((id) => cat.parts.find((p) => p.id === id)).filter(Boolean);
  const skills = [...picked, ...extra];
  const names = plainNames(skills, spec);
  const mIds = maskIdsOf(program.masks);
  const forgets = useMemo(() => pickForgets(cat, spec, program, lockOf, mIds), [cat, spec && spec.id, mIds.join(',')]);
  const healOn = !!program.heal;
  const healMin = est ? est.heal_min : null;
  const scored = [...attempts].reverse().find((a) => a.scores || (a.status && a.status !== 'ok'));
  const res = resultOf(spec, scored);
  const earlier = !!scored && attempts.indexOf(scored) < earlierN;
  const sameAsLast = !!scored && JSON.stringify(scored.program) === JSON.stringify(program);
  const size = sizeLine(totalB, runB, lim, healOn ? healMin : null);
  const goal = spec ? spec.summary || spec.title : 'Pick the skills your model should have.';
  const running = !!job && job.action === 'score';
  const busy = !!job;
  const pctNow = job ? Math.min(94, ((Date.now() - job.t0) / 2600) * 100) : 0;

  const added = program.grafts.map((g) => cat.parts.find((p) => p.id === g.id)).filter(Boolean);
  const caption = running ? 'Testing your model...' : added.length ? added.length + (added.length === 1 ? ' skill is' : ' skills are') + ' added to your model.' : 'Your starting model. Add a skill below to see it join.';

  return html`<main class="screen bs-screen"><div class="bs-page">
    ${gone && html`<div class="bs-gone card" role="alert"><p>This build could not be loaded. Go back and start it again.</p><a class="btn sm" href="#/">Go to the home page</a></div>`}
    <div class="bs-hero">
      <h1 class="page-title bs-h1">Build your model</h1>
      <p class="lede">You start with ${(cat.host.model || '').split('/').pop()}. Add skills from other models, then test it.</p>
      <div class="bs-brain card" aria-label="A picture of your model with the skills you added">${brain}</div>
      <p class="bs-cap" role="status">${caption}</p>
      ${added.length > 0 && html`<ul class="bs-chips" aria-label="Skills you added">${added.map((p) => html`<li key=${p.id}><button type="button" class="bs-chip" onClick=${() => openPart(p.id)}><${ModelLogo} model=${p.donor} size=${16} alt="" />${plainName(p, spec)}</button></li>`)}</ul>`}
    </div>

    <${Step} n="1" id="skills" title="What it should be good at" lead=${goal}>
      ${skills.length > 0 ? html`<div class="bs-list">${skills.map((p, i) => html`<${SkillCard} key=${p.id} name=${names[i]} part=${p} spec=${spec} hostB=${hostB} on=${usedIds.has(p.id)} onToggle=${(part, on) => (on ? onAdd(part) : onRemove(part.id))} />`)}</div>` : html`<p class="faint">No ready-made skills match this goal yet. Browse all skills below.</p>`}
      <div class="bs-more">
        <button class="btn sm" type="button" aria-expanded=${browse ? 'true' : 'false'} onClick=${() => setBrowse(!browse)}>${browse ? 'Hide the full list' : 'Browse all skills'}</button>
        <button class="btn ghost sm" type="button" onClick=${onAdvanced}>Switch to Advanced mode</button>
      </div>
      ${browse && html`<${Browse} cat=${cat} spec=${spec} hostB=${hostB} usedIds=${usedIds} roomB=${limB != null ? limB - totalB : null} onToggle=${(part, on) => (on ? onAdd(part) : onRemove(part.id))} />`}
      <${Help}>A skill is a piece of another model that is already good at something, such as math. Adding it copies that piece into your model. Each skill makes the model a little bigger.<//>
    <//>

    ${forgets.length > 0 && html`<${Step} n="2" id="forget" title="Anything it should forget" tag="Optional" lead="Switching one on removes what the model knows about that topic. Leave all off if you are not sure.">
      <div class="bs-toggles" role="group" aria-label="Topics to forget">${forgets.map((m) => { const on = mIds.includes(m.id); return html`<button key=${m.id} type="button" class=${'bs-toggle' + (on ? ' on' : '')} role="switch" aria-checked=${on ? 'true' : 'false'} onClick=${() => onToggleMask(m)}><span class="bs-sw" aria-hidden="true"><span></span></span><span class="bs-tt">${forgetName(m)}</span></button>`; })}</div>
      <${Help}>Forgetting removes what the model knows about a topic. That can lower its scores on other tasks. It does not make the model bigger.<//>
    <//>`}

    <${Step} n=${forgets.length > 0 ? '3' : '2'} id="polish" title="Polish">
      <button type="button" class=${'bs-toggle bs-polish' + (healOn ? ' on' : '')} role="switch" aria-checked=${healOn ? 'true' : 'false'} onClick=${() => onPolish(!healOn)}><span class="bs-sw" aria-hidden="true"><span></span></span><span class="bs-tt">Smooth the joins</span></button>
      <p class="bs-lead">${polishText(healMin, healOn)}</p>
      ${program.lora && html`<p class="faint bs-note">Advanced mode also has an extra tuning step turned on.</p>`}
      <${Help}>New skills are copied in from other models, so they do not fit perfectly at first. Smoothing trains the model for a short while until they do.<//>
    <//>

    <${Step} n=${forgets.length > 0 ? '4' : '3'} id="test" title="Test it">
      <p class="bs-lead">${size.text}</p>
      ${size.warn.map((w) => html`<p class="bs-warn" role="alert"><${Icon} n="warn" size=${15} /> ${w}</p>`)}
      <button type="button" class=${'btn primary bs-test' + (running ? ' is-busy' : '')} disabled=${busy || gone} onClick=${onScore}>${running ? html`<${Spinner} size=${14} />` : flash === 'score' ? html`<${SuccessCheck} size=${15} />` : html`<${Icon} n="play" size=${14} />`} ${running ? 'Testing' : 'Test my model'}</button>
      ${running && html`<div class="bs-progress" role="status"><span>${progressText(job)}</span><div class="bigbar"><div class="fill" style=${{ width: pctNow + '%', transition: 'width 1s linear' }}></div></div></div>`}
      ${!running && res && !res.failed && html`<div class=${'bs-result card' + (res.meets ? ' good' : '')} aria-live="polite">
        <div class="bs-rtags">${earlier && html`<span class="tag">Last test (from earlier)</span>`}</div>
        ${res.empty ? html`<p>${res.lines.join(' ')}</p>` : html`
          <div class="bs-verdict">${res.verdict}</div>
          <div class="bs-big num">${Math.round(res.score * 1000) / 10}%<span class="bs-unit"> right on ${res.name}</span></div>
          <p>${res.lines.join(' ')}</p>
          ${res.next && html`<p>${res.next}</p>`}`}
        ${!sameAsLast && html`<p class="faint bs-note">You changed your model after this test. Test again to update the result.</p>`}

      </div>`}
      ${!running && res && res.failed && html`<div class="bs-result card bad" role="alert"><p>${failText(res.status)}</p></div>`}
      <${Help}>The test gives your model a set of questions it has not seen and counts how many it gets right. We also test the starting model and a model with random parts of the same size, so you can see if your skills are what helped.<//>
    <//>

    <${Step} n=${forgets.length > 0 ? '5' : '4'} id="agents" title="Let the agents improve it" lead="A team of AI agents starts from your model, tries many changes, and keeps the best one. It runs on its own.">
      <button type="button" class="btn bs-hand" disabled=${handing} onClick=${onHandoff}>${handing ? html`<${Spinner} size=${12} />` : html`<${Icon} n="spark" size=${14} />`} Hand to agents</button>
    <//>
  </div></main>`;
}
