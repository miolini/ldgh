const STORAGE_KEY = 'fieldnote-agent-workflow-v1';
const MAX_TRANSCRIPT_ITEMS = 180;
const MAX_STORED_CHARS = 1_500_000;

function safeParse(value) {
  try { return JSON.parse(value); } catch { return null; }
}

/** Synchronous browser journal. Each checkpoint is one atomic localStorage value. */
export class WorkflowJournal {
  constructor(storage = globalThis.localStorage, key = STORAGE_KEY) {
    this.storage = storage;
    this.key = key;
    this.data = this.read();
  }

  read() {
    const saved = safeParse(this.storage?.getItem(this.key));
    if (!saved || saved.schemaVersion !== 1) return {schemaVersion:1, revision:0, history:[], transcript:[], activeRun:null};
    return {...saved, history:Array.isArray(saved.history)?saved.history:[], transcript:Array.isArray(saved.transcript)?saved.transcript:[], activeRun:saved.activeRun??null};
  }

  snapshot() { return structuredClone(this.data); }

  commit(patch = {}) {
    const next = {...this.data, ...patch, schemaVersion:1, revision:this.data.revision+1, updatedAt:new Date().toISOString()};
    next.history = Array.isArray(next.history) ? next.history.slice(-280) : [];
    next.transcript = Array.isArray(next.transcript) ? next.transcript.slice(-MAX_TRANSCRIPT_ITEMS) : [];
    const serialized = JSON.stringify(next);
    if (serialized.length > MAX_STORED_CHARS) throw new Error('The saved conversation is too large for browser storage. Clear some chat history before continuing.');
    this.storage.setItem(this.key, serialized);
    this.data = next;
    return this.snapshot();
  }

  startRun(run) { const {transcript,...active}=run; return this.commit({...(transcript?{transcript}:{}),activeRun:{...active, phase:'request', startedAt:active.startedAt??Date.now(), updatedAt:Date.now()}}); }
  checkpoint(run) { const {transcript,...active}=run; return this.commit({...(transcript?{transcript}:{}),activeRun:{...active, updatedAt:Date.now()}}); }
  finishRun(history, transcript) { return this.commit({history, transcript, activeRun:null}); }
  pauseRun(run, error) { const {transcript,...active}=run; return this.commit({...(transcript?{transcript}:{}),activeRun:{...active, phase:'paused', lastError:String(error??'Interrupted'), updatedAt:Date.now()}}); }
  clear() { return this.commit({history:[], transcript:[], activeRun:null}); }

  appendTranscript(entry) {
    return this.commit({transcript:[...this.data.transcript, {...entry, at:entry.at??Date.now()}]});
  }
}

export const workflowStorageKey = STORAGE_KEY;
