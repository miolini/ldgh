import {createQwen38Runtime} from './qwen38-runtime.mjs';
let runtime, progressTarget;
let cancellation = 0;
let generationTask = Promise.resolve();
self.onmessage = async ({data}) => {
  const {id, action, options} = data;
  if (action === 'stop') {cancellation++; runtime?.interruptGenerate(); return;}
  try {
    if (action === 'load') {
      runtime?.dispose(); runtime = undefined;
      progressTarget = id;
      runtime = await createQwen38Runtime({...options, onProgress:report => self.postMessage({id:progressTarget, type:'progress', report})});
    } else if (action === 'generate') {
      if (!runtime) throw new Error('The model is not loaded.');
      // A new chat may arrive while the previous stopped token is finishing.
      const requestedAt = cancellation;
      const next = generationTask.catch(() => {}).then(async () => {
        if (requestedAt !== cancellation) throw new Error('Response stopped.');
        progressTarget = id;
        for await (const chunk of await runtime.chat.completions.create(options)) self.postMessage({id, type:'chunk', chunk});
      });
      generationTask = next;
      await next;
    } else throw new Error('Unknown runtime action.');
    self.postMessage({id, type:'done'});
  } catch (error) {self.postMessage({id, type:'error', error:error.message ?? String(error)});}
};
