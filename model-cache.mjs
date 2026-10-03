// Files are downloaded to the local HTTP server's disk, not to browser RAM.
export function initializeModelCache() {
  const button = document.querySelector('#model-download');
  const status = document.querySelector('#model-file-status');
  const progress = document.querySelector('#model-file-progress');
  if (!button || !status || !progress) return;
  let timer, pending = false;
  const gb = bytes => `${(bytes / 1e9).toFixed(2)} GB`;
  function render(state) {
    progress.value = state.progress ?? 0;
    progress.hidden = state.downloaded === 0 && !state.active;
    button.disabled = state.active || state.status === 'verifying' || state.sha256Verified;
    button.textContent = state.sha256Verified ? 'Files cached' : state.active ? 'Downloading…' : state.downloaded ? 'Resume download' : 'Download files';
    status.textContent = state.error ? `Download paused: ${state.error}`
      : state.sha256Verified ? `${gb(state.total)} cached on this Mac · verified`
      : state.status === 'verifying' ? 'All files downloaded · checking integrity…'
      : state.active && state.axelBytesDone
        ? `At least ${gb(state.downloaded)} / ${gb(state.total)} present · ${gb(state.verifiedBytes || 0)} verified · Axel received ${gb(state.axelBytesDone)}`
        : `${gb(state.downloaded)} / ${gb(state.total)} cached on this Mac`;
  }
  async function refresh() {
    clearTimeout(timer);
    if (pending) return;
    pending = true;
    try {
      const response = await fetch('/api/model-cache/status', {cache:'no-store'});
      if (!response.ok) throw new Error('Local download route unavailable');
      const state = await response.json(); render(state);
      if (state.active || state.inflight) timer = setTimeout(refresh, 2000);
    } catch (error) { status.textContent = error.message; button.disabled = true; }
    finally { pending = false; }
  }
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      const response = await fetch('/api/model-cache/start', {method:'POST'});
      if (!response.ok) throw new Error('Could not start the local model download');
      render(await response.json()); timer = setTimeout(refresh, 1000);
    } catch (error) { status.textContent = error.message; button.disabled = false; }
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  refresh();
}
