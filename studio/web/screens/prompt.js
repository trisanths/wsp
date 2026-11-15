import { html, useState, useRef, useEffect, useLayoutEffect } from '../lib.js';
import { api } from '../api.js';
import { go, toast } from '../util.js';
import { TopBar, Icon, Composer } from '../components.js';
import { ThinkLine, reduced } from '../motion.js';

const CARDS = [
  { icon: 'book', color: 'var(--blue)', label: 'A knowledge model', text: 'A small model that answers general knowledge questions nearly as well as a much bigger one. It has to run on a laptop.' },
  { icon: 'sigma', color: 'var(--purple)', label: 'A math model', text: 'A small model that solves grade-school math word problems step by step. It has to run on a laptop.' },
  { icon: 'bolt', color: 'var(--amber)', label: 'A smaller, faster model', text: 'A smaller and faster version of the host model that keeps most of its accuracy on everyday questions.' },
  { icon: 'grid', color: 'var(--green)', label: 'Browse parts', href: '#/parts' },
];
const EXAMPLES = [
  'Drop everything but legal reasoning',
  'A 500M model for support-ticket routing',
  'The best Python helper under 2B parameters',
  'A model that knows medicine and still does math',
];
const STATES = ['Reading your files', 'Picking benchmarks', 'Drafting the spec'];
const HOLD_MS = 1100;
// the headline reveals once per page load
let revealed = false;

function Chevron() {
  return html`<span class="t-learn-chevron" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 16 16"><path class="t-learn-arm t-learn-arm-top" d="M6 4L10 8"/><path class="t-learn-arm t-learn-arm-bot" d="M10 8L6 12"/></svg></span>`;
}

const fsize = (n) => (n > 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB');

export function PromptScreen() {
  const [text, setText] = useState('');
  const [files, setFiles] = useState([]);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(0);
  const [sent, setSent] = useState('');
  const [shown, setShown] = useState(revealed);
  const [settled, setSettled] = useState(revealed);
  const clearRef = useRef();
  const fileIn = useRef();
  const ta = useRef();
  const depth = useRef(0);

  useLayoutEffect(() => {
    if (revealed) return;
    revealed = true;
    void document.body.offsetWidth;
    setShown(true);
    const t = setTimeout(() => setSettled(true), 900);
    return () => clearTimeout(t);
  }, []);
  useEffect(() => {
    if (!busy) return;
    setStep(0);
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STATES.length - 1)), HOLD_MS);
    return () => clearInterval(t);
  }, [busy]);

  const addFiles = (list) => {
    const arr = Array.from(list || []);
    if (!arr.length) return;
    setFiles((f) => [...f, ...arr.filter((n) => !f.some((x) => x.name === n.name && x.size === n.size))]);
  };
  const submit = async () => {
    if (busy) return;
    if (!text.trim()) { toast('Say what the model should do first.', 'err'); if (ta.current) ta.current.focus(); return; }
    const prompt = text.trim();
    setBusy(true);
    setSent(prompt);
    // the typed text dissolves while the request runs; the screen waits for both
    const dissolve = clearRef.current ? clearRef.current() : Promise.resolve();
    setText('');
    try {
      const [spec] = await Promise.all([api.draftSpec(prompt, files), dissolve, new Promise((r) => setTimeout(r, reduced() ? 0 : HOLD_MS))]);
      go('/spec/' + spec.id);
    } catch (e) {
      setBusy(false);
      setText(prompt);
      toast(e.message, 'err');
    }
  };
  const fill = (t) => { setText(t); if (ta.current) ta.current.focus(); };
  const onDrop = (e) => { e.preventDefault(); depth.current = 0; setOver(false); addFiles(e.dataTransfer.files); };
  const onEnter = (e) => { e.preventDefault(); depth.current++; setOver(true); };
  const onLeave = () => { depth.current = Math.max(0, depth.current - 1); if (!depth.current) setOver(false); };

  const chips = files.length > 0 && html`<div class="fchips">${files.map((f, i) => html`<span class="fchip" key=${f.name + i}><${Icon} n="file" size=${14} /><span class="fn">${f.name}</span><span class="sz">${fsize(f.size)}</span><button data-tip="Remove" aria-label=${'Remove ' + f.name} onClick=${() => setFiles(files.filter((_, j) => j !== i))}><${Icon} n="x" size=${12} /></button></span>`)}</div>`;

  return html`<${TopBar} />
  <main class="screen home">
    <div class="home-top">
      ${busy
        ? html`<div class="card drafting">
            <div class="dtitle"><${ThinkLine} text=${STATES[step]} /></div>
            <div class="dtext">${sent}</div>
            <div class="shimbar"><i></i></div>
          </div>`
        : html`<div class=${'t-stagger' + (shown ? ' is-shown' : '') + (settled ? ' is-settled' : '')}>
          <h1 class="display t-stagger-line t-stagger-line--1">What should your model be good at?</h1>
          <div class="entry-row t-stagger-line t-stagger-line--2">${CARDS.map((c) => html`<a class="entry t-learn" key=${c.label} href=${c.href || null} role=${c.href ? null : 'button'} tabindex="0" onClick=${c.href ? null : () => fill(c.text)} onKeyDown=${c.href ? null : (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fill(c.text); } }}><span class="eic" style=${{ color: c.color }}><${Icon} n=${c.icon} size=${20} /></span><span class="elabel">${c.label}</span><${Chevron} /></a>`)}</div>
          <div class="examples t-stagger-line t-stagger-line--3">${EXAMPLES.map((x) => html`<button class="exchip" key=${x} onClick=${() => fill(x)}>${x}</button>`)}</div>
          </div>`}
    </div>
    ${html`<div class="dock" onDragEnter=${onEnter} onDragOver=${(e) => e.preventDefault()} onDragLeave=${onLeave} onDrop=${onDrop}>
      <div class="dock-in">
        <${Composer} taRef=${ta} clearRef=${clearRef} disabled=${busy} autoFocus label="Describe your model" value=${text} onInput=${setText} onSubmit=${submit} over=${over} chips=${chips} extra=${over && html`<div class="dropveil">Drop files to attach them</div>`} onAttach=${() => fileIn.current.click()} placeholder="Describe the model you need and where it has to run" />
        <input ref=${fileIn} type="file" multiple hidden onChange=${(e) => { addFiles(e.target.files); e.target.value = ''; }} />
        <div class="dock-note">Files can be examples, questions with answers, or documents. Part is held out for scoring, part is used to heal.</div>
      </div>
    </div>`}
  </main>`;
}
