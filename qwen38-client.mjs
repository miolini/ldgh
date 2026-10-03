export async function createQwen38Client({onProgress = () => {}} = {}) {
  const worker = new Worker(new URL('./qwen38-worker.mjs', import.meta.url), {type:'module'});
  const requests = new Map(); let sequence = 0, disposed = false;
  worker.onmessage = ({data}) => {
    const request = requests.get(data.id); if (!request) return;
    if (data.type === 'progress') {onProgress(data.report); return;}
    if (data.type === 'chunk') {request.queue.push(data.chunk); request.wake?.();}
    else {
      request.done = true; request.error = data.type === 'error' ? new Error(data.error) : null;
      request.wake?.();
    }
  };
  worker.onerror = event => {
    for (const request of requests.values()) {request.done = true; request.error = new Error(event.message || 'Browser model worker failed.'); request.wake?.();}
  };
  async function* invoke(action, options) {
    if(disposed)throw new Error('The in-browser model was unloaded.');
    const id = ++sequence, request = {queue:[], done:false}; requests.set(id, request);
    worker.postMessage({id, action, options});
    try {
      while (true) {
        while (request.queue.length) yield request.queue.shift();
        if (request.done) {if (request.error) throw request.error; return;}
        await new Promise(resolve => {request.wake = resolve;}); request.wake = null;
      }
    } finally {requests.delete(id);}
  }
  const ready=(async()=>{try{for await (const unused of invoke('load', {maxTokens:4096})) void unused;}
    catch(error){if(!disposed)dispose();throw error;}})();
  function dispose(){
    if(disposed)return;disposed=true;
    for(const request of requests.values()){request.done=true;request.error=new Error('The in-browser model was unloaded.');request.wake?.();}
    worker.terminate();
  }
  return {ready,chat:{completions:{create:async options => invoke('generate', options)}}, interruptGenerate(){if(!disposed)worker.postMessage({action:'stop'});}, dispose};
}
