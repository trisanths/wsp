import { markStub } from './util.js';

export const MOCK = new URLSearchParams(location.search).get('mock') === '1';

async function req(method, path, body, isForm) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opt.body = body;
    else { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  }
  let res;
  try { res = await fetch(path, opt); } catch (e) { throw new Error('Cannot reach the server'); }
  let data = null;
  try { data = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || res.status + ' ' + res.statusText);
    if (data && data.line != null) err.line = data.line;
    err.status = res.status;
    throw err;
  }
  if (data && data.stub) markStub(true);
  return data;
}

const real = {
  health: () => req('GET', '/api/health'),
  draftSpec(prompt, files) {
    const f = new FormData();
    f.append('prompt', prompt);
    for (const file of files) f.append('files', file, file.name);
    return req('POST', '/api/spec', f, true);
  },
  getSpec: (id) => req('GET', '/api/spec/' + id),
  putSpec: (id, spec) => req('PUT', '/api/spec/' + id, spec),
  getResearch: (id) => req('GET', '/api/spec/' + id + '/research'),
  startResearch: (id) => req('POST', '/api/spec/' + id + '/research', {}),
  createRun: (spec_id, program = null) => req('POST', '/api/runs', { spec_id, program }),
  getRun: (id) => req('GET', '/api/runs/' + id),
  openStream(id, onSnap, onErr, onNote) {
    const es = new EventSource('/api/runs/' + id + '/stream');
    es.addEventListener('snapshot', (e) => {
      try { const s = JSON.parse(e.data); if (s && s.stub) markStub(true); onSnap(s); } catch (x) { /* skip bad frame */ }
    });
    // "event: error" with {"error": "..."} is a server message. A bare error is a dropped connection.
    es.addEventListener('error', (e) => {
      if (!e || typeof e.data !== 'string') { es.close(); onErr(); return; }
      let msg = 'stream error';
      try { const d = JSON.parse(e.data); if (d && d.error) msg = String(d.error); } catch (x) { /* keep the default */ }
      if (onNote) onNote(msg);
      onErr();
    });
    return () => es.close();
  },
  pause: (id) => req('POST', '/api/runs/' + id + '/pause', {}),
  resume: (id) => req('POST', '/api/runs/' + id + '/resume', {}),
  stop: (id) => req('POST', '/api/runs/' + id + '/stop', {}),
  getAgent: (id, a) => req('GET', '/api/runs/' + id + '/agents/' + a),
  nudge: (id, a, text) => req('POST', '/api/runs/' + id + '/agents/' + a + '/nudge', { text }),
  // Sapien builds one network: tap parts from an older server never reach a screen
  catalog: () => req('GET', '/api/catalog').then((c) => {
    if (!c) return c;
    const { tap_train_default, ...rest } = c;
    return { ...rest, parts: (c.parts || []).filter((x) => x.kind !== 'tap') };
  }),
  recipes: () => req('GET', '/api/library/recipes'),
  findings: () => req('GET', '/api/library/findings'),
  stitches: () => req('GET', '/api/library/stitches'),
  programText: (program) => req('POST', '/api/program/text', { program }),
  programParse: (text) => req('POST', '/api/program/parse', { text }),
  programValidate: (program) => req('POST', '/api/program/validate', { program }),
  createBuilder: (spec_id) => req('POST', '/api/builder', { spec_id }),
  builderAction: (id, program, action) => req('POST', '/api/builder/' + id + '/action', { program, action }),
  getJob: (id) => req('GET', '/api/jobs/' + id),
  builderSize: (id, program) => req('POST', '/api/builder/' + id + '/size', { program }),
  suggest: (id, program) => req('POST', '/api/builder/' + id + '/suggest', { program }),
  handoff: (id, program) => req('POST', '/api/builder/' + id + '/handoff', { program }),
  createShrink: (spec_id) => req('POST', '/api/shrink', { spec_id }),
  shrinkPlan: (id, fields = {}) => req('POST', '/api/shrink/' + id + '/plan', fields),
  shrinkRun: (id, options = {}) => req('POST', '/api/shrink/' + id + '/run', options),
};

let impl = real;
if (MOCK) {
  const m = await import('./mock.js');
  impl = m.mockApi;
}
const wrapped = {};
for (const k of Object.keys(real)) {
  wrapped[k] = async (...a) => {
    const r = await impl[k](...a);
    if (r && r.stub && k !== 'openStream') markStub(true);
    return r;
  };
}
wrapped.openStream = impl.openStream;
export const api = wrapped;
