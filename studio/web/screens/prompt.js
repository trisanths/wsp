import { html, useState, useRef, useEffect } from '../lib.js';
import { api } from '../api.js';
import { go, toast } from '../util.js';
import { TopBar, Icon, Spinner } from '../components.js';

const EXAMPLES = [
  'A small model that solves grade-school math word problems step by step. It has to run on a laptop.',
  'Drop everything but legal reasoning',
  'A 500M model for support-ticket routing',
  'The best Python helper under 2B parameters',
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

  useEffect(() => { if (ta.current) ta.current.focus(); }, []);
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
  const onDrop = (e) => { e.preventDefault(); depth.current = 0; setOver(false); addFiles(e.dataTransfer.files); };
  const onEnter = (e) => { e.preventDefault(); depth.current++; setOver(true); };
  const onLeave = () => { depth.current = Math.max(0, depth.current - 1); if (!depth.current) setOver(false); };

  return html`<${TopBar} />
  <main class="screen"><div class="prompt-wrap">
    <h1>What model do you need?</h1>
    <div class="sub">Say what it should do and where it has to run. Add examples if you have them.</div>
    ${busy
      ? html`<div class="card drafting">
          <div style="display:flex;align-items:center;gap:10px;font-weight:600;font-size:15px"><${Spinner} size=${16} /> Drafting your spec</div>
          <div class="dim" style="margin-top:6px;white-space:pre-wrap;max-height:60px;overflow:hidden">${text}</div>
          <div class="steps">${STEPS.map((s, i) => html`<div class=${'dstep' + (i === step ? ' on' : i < step ? ' done' : '')}><span class="ck">${i < step ? html`<${Icon} n="check" size=${14} />` : i === step ? html`<${Spinner} size=${11} />` : html`<span class="dot" style="background:var(--line2)"></span>`}</span>${s}</div>`)}</div>
          <div class="shimbar"><i></i></div>
        </div>`
      : html`<div class=${'pbox' + (over ? ' over' : '')} onDragEnter=${onEnter} onDragOver=${(e) => e.preventDefault()} onDragLeave=${onLeave} onDrop=${onDrop}>
          <textarea ref=${ta} value=${text} placeholder="A small model that solves grade-school math word problems step by step. It has to run on a laptop with no GPU." onInput=${(e) => setText(e.target.value)} onKeyDown=${(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit(); }}></textarea>
          <div class="pfoot">
            <div class="fchips">
              ${files.map((f, i) => html`<span class="fchip" key=${f.name + i}><${Icon} n="clip" size=${13} />${f.name}<span class="sz">${fsize(f.size)}</span><button title="Remove" onClick=${() => setFiles(files.filter((_, j) => j !== i))}><${Icon} n="x" size=${12} /></button></span>`)}
              <button class="btn ghost sm" onClick=${() => fileIn.current.click()}><${Icon} n="clip" size=${14} /> Attach files</button>
              <input ref=${fileIn} type="file" multiple hidden onChange=${(e) => { addFiles(e.target.files); e.target.value = ''; }} />
            </div>
            <button class="btn primary big" onClick=${submit}>Draft a spec <span class="k">Cmd+Enter</span></button>
          </div>
          ${over && html`<div class="dropveil">Drop files to attach them</div>`}
        </div>`}
    ${!busy && html`<div class="examples">${EXAMPLES.map((x) => html`<button class="exchip" onClick=${() => { setText(x); if (ta.current) ta.current.focus(); }}>${x.length > 62 ? x.slice(0, 60) + '...' : x}</button>`)}</div>
    <div class="faint" style="margin-top:18px;font-size:12px">Files can be examples, questions with answers, or documents. Part is held out for scoring, part is used to heal.</div>`}
  </div></main>`;
}
