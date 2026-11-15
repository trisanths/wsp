// Demo mode: the whole app runs on built-in sample data in the browser (the mock API), with no server
// calls, so a demo works with no backend or GPU. It is on when the URL has ?demo=1 or the saved
// switch is on. One small "Demo" pill marks every screen in place of the many STUB badges.
// Demo data uses its own browser storage keys, so it never mixes with real runs and recents.

const KEY = 'sapien.demo';
const SPEED_KEY = 'sapien.demo.speed';

function read(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
function write(key, value) { try { if (value == null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch (e) { /* storage can be off */ } }

const param = new URLSearchParams(location.search).get('demo');
if (param === '1') write(KEY, '1');
if (param === '0') write(KEY, null);

/** True on the hosted site (sapien.trisanth.com). It has no server, so demo mode is always on there. */
export const HOSTED = typeof window !== 'undefined' && window.SAPIEN_HOSTED === true;

/** True when demo mode is on for this page load. Switching it reloads the page. */
export const DEMO = HOSTED || read(KEY) === '1';

/** Turn demo mode on or off. Reloads the page at the same screen, or at `hash` when given. */
export function setDemo(on, hash) {
  if (HOSTED) return; // the hosted site has no server, so demo mode cannot change
  write(KEY, on ? '1' : null);
  const q = new URLSearchParams(location.search);
  q.delete('demo');
  q.delete('mock');
  const search = q.toString();
  const target = location.pathname + (search ? '?' + search : '') + (hash != null ? hash : location.hash);
  const sameDoc = location.pathname + (search ? '?' + search : '') === location.pathname + location.search;
  location.replace(target);
  // a change of fragment alone does not reload, so reload to switch the data source
  if (sameDoc) location.reload();
}

/** The speeds the demo offers. 1 is the real feel. */
export const DEMO_SPEEDS = [1, 2, 4, 10];
const speedOf = (v) => (DEMO_SPEEDS.includes(Number(v)) ? Number(v) : 1);
let speed = speedOf(read(SPEED_KEY));

// Every page load starts the demo fresh: new seeded runs, recents, specs, builds, mining requests
// and tour position. Only the demo switch and the chosen speed stay.
if (DEMO) {
  const keep = new Set([KEY, SPEED_KEY]);
  for (const box of ['localStorage', 'sessionStorage']) {
    try {
      const s = window[box];
      const old = [];
      for (let i = 0; i < s.length; i++) { const k = s.key(i); if (k && k.startsWith('sapien.demo.') && !keep.has(k)) old.push(k); }
      old.forEach((k) => s.removeItem(k));
    } catch (e) { /* storage can be off */ }
  }
}

// The demo clock. Simulated work (agent runs, research, mining, builder and shrink jobs) waits on
// this clock, and it runs `speed` times faster than the wall clock.
let wall0 = Date.now();
let demo0 = wall0;

/** The demo clock in milliseconds. Outside demo mode it is the wall clock. */
export function demoNow() { return DEMO ? demo0 + (Date.now() - wall0) * speed : Date.now(); }

/** The wall-clock time that shows a demo-clock time `t` at the right age on screen. */
export function wallOf(t) { return t - (demoNow() - Date.now()); }

const speedListeners = new Set();
/** Call `fn` with the new speed when it changes. Returns a function that stops the calls. */
export function onDemoSpeed(fn) { speedListeners.add(fn); return () => speedListeners.delete(fn); }

/** How fast simulated work moves in demo mode: 1 (real feel), 2, 4, or 10. */
export function demoSpeed() { return speed; }

/** Change the demo speed. It applies at once, and the demo clock does not jump. */
export function setDemoSpeed(v) {
  demo0 = demoNow();
  wall0 = Date.now();
  speed = speedOf(v);
  write(SPEED_KEY, speed === 1 ? null : String(speed));
  speedListeners.forEach((fn) => fn(speed));
}

/** The browser storage key to use: demo mode gets its own copy of every key. */
export function storageKey(key) { return DEMO ? 'sapien.demo.' + key : key; }
