import { api } from '../api.js';
import { go, toast } from '../util.js';
import { storageKey } from '../demo-flag.js';

// Create a builder for a spec and open it. The program is passed to the builder
// through sessionStorage, because the contract has no route to load one.
export async function openProgramInBuilder(specId, program) {
  try {
    const r = await api.createBuilder(specId);
    if (program) {
      try { sessionStorage.setItem(storageKey('wsp.seed.' + r.builder_id), JSON.stringify(program)); } catch (e) { /* storage can be off */ }
    }
    try { sessionStorage.setItem(storageKey('wsp.budget.' + r.builder_id), JSON.stringify(r.budget)); } catch (e) { /* ignore */ }
    go('/build/' + r.builder_id + '?spec=' + specId);
  } catch (e) {
    toast('Could not open the builder: ' + e.message, 'err');
  }
}

export async function bestProgram(runId, snap) {
  if (snap.best && snap.best.program) return snap.best.program;
  const row = (snap.leaderboard || []).find((r) => snap.best && r.program_hash === snap.best.program_hash && r.program);
  if (row) return row.program;
  if (snap.best && snap.best.agent) {
    try {
      const d = await api.getAgent(runId, snap.best.agent);
      if (d.current && d.current.program) return d.current.program;
    } catch (e) { /* fall through */ }
  }
  return null;
}
