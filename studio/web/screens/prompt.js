import { html, useState, useRef, useEffect } from '../lib.js';
import { api } from '../api.js';
import { go, toast } from '../util.js';
import { TopBar, Icon, Spinner, Composer } from '../components.js';

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
const STEPS = ['Reading your request', 'Picking a host model', 'Choosing benchmarks and targets', 'Setting size and budget'];

const fsize = (n) => (n > 1048576 ? (n / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(n / 1024)) + ' KB');

export function PromptScreen() {
  const [text, setText] = useState('');
  const [files, setFiles] = useState([]);
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [step, setStep] = useState(0);
  const fileIn = useRef();
  const ta = useRef();
  const depth = useRef(0);

  useEffect(() => {
    if (!busy) return;
    setStep(0);
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 650);
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
    setBusy(true);
    try {
      const spec = await api.draftSpec(text.trim(), files);
      go('/spec/' + spec.id);
    } catch (e) {
      setBusy(false);
      toast(e.message, 'err');
    }
  };
  const fill = (t) => { setText(t); if (ta.current) ta.current.focus(); };
  const onDrop = (e) => { e.preventDefault(); depth.current = 0; setOver(false); addFiles(e.dataTransfer.files); };
  const onEnter = (e) => { e.preventDefault(); depth.current++; setOver(true); };
  const onLeave = () => { depth.current = Math.max(0, depth.current - 1); if (!depth.current) setOver(false); };

  const chips = files.length > 0 && html`<div class="fchips">${files.map((f, i) => html`<span class="fchip" key=${f.name + i}><${Icon} n="file" size=${14} /><span class="fn">${f.name}</span><span class="sz">${fsize(f.size)}</span><button title="Remove" aria-label=${'Remove ' + f.name} onClick=${() => setFiles(files.filter((_, j) => j !== i))}><${Icon} n="x" size=${12} /></button></span>`)}</div>`;

  return html`<${TopBar} />
  <main class="screen home">
    <div class="home-top">
      ${busy
        ? html`<div class="card drafting">
            <div class="dtitle"><${Spinner} size=${16} /> Drafting your spec</div>
            <div class="dtext">${text}</div>
            <div class="steps">${STEPS.map((s, i) => html`<div class=${'dstep' + (i === step ? ' on' : i < step ? ' done' : '')}><span class="ck">${i < step ? html`<${Icon} n="check" size=${14} />` : i === step ? html`<${Spinner} size=${11} />` : html`<span class="dot" style="background:var(--text-4)"></span>`}</span>${s}</div>`)}</div>
            <div class="shimbar"><i></i></div>
          </div>`
        : html`<h1 class="display">What should your model be good at?</h1>
          <div class="entry-row">${CARDS.map((c) => html`<a class="entry" key=${c.label} href=${c.href || null} role=${c.href ? null : 'button'} tabindex="0" onClick=${c.href ? null : () => fill(c.text)} onKeyDown=${c.href ? null : (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fill(c.text); } }}><span class="eic" style=${{ color: c.color }}><${Icon} n=${c.icon} size=${20} /></span><span class="elabel">${c.label}</span></a>`)}</div>
          <div class="examples">${EXAMPLES.map((x) => html`<button class="exchip" key=${x} onClick=${() => fill(x)}>${x}</button>`)}</div>`}
    </div>
    ${!busy && html`<div class="dock" onDragEnter=${onEnter} onDragOver=${(e) => e.preventDefault()} onDragLeave=${onLeave} onDrop=${onDrop}>
      <div class="dock-in">
        <${Composer} taRef=${ta} autoFocus label="Describe your model" value=${text} onInput=${setText} onSubmit=${submit} over=${over} chips=${chips} extra=${over && html`<div class="dropveil">Drop files to attach them</div>`} onAttach=${() => fileIn.current.click()} placeholder="Describe the model you need and where it has to run" />
        <input ref=${fileIn} type="file" multiple hidden onChange=${(e) => { addFiles(e.target.files); e.target.value = ''; }} />
        <div class="dock-note">Files can be examples, questions with answers, or documents. Part is held out for scoring, part is used to heal.</div>
      </div>
    </div>`}
  </main>`;
}
