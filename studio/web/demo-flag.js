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

/** How fast simulated runs move in demo mode: 1 (real feel), 2, or 4. */
export function demoSpeed() {
  const v = Number(read(SPEED_KEY));
  return v === 2 || v === 4 ? v : 1;
}
export function setDemoSpeed(v) { write(SPEED_KEY, v === 2 || v === 4 ? String(v) : null); }

/** The browser storage key to use: demo mode gets its own copy of every key. */
export function storageKey(key) { return DEMO ? 'sapien.demo.' + key : key; }
