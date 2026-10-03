const result = document.querySelector('#result');
document.querySelector('#run').addEventListener('click', async event => {
  event.target.disabled = true;
  const worker = new Worker(new URL('./gpu-memory-worker.mjs',import.meta.url),{type:'module'});
  worker.onmessage = ({data}) => {
    result.textContent = data.text ?? JSON.stringify(data,null,2);
    if (data.done) {document.querySelector('#report-json').textContent = JSON.stringify(data,null,2); worker.terminate(); event.target.disabled=false;}
  };
  worker.onerror = event => {result.textContent=`Allocation check failed: ${event.message}`; worker.terminate(); document.querySelector('#run').disabled=false;};
  worker.postMessage({action:'run'});
});
