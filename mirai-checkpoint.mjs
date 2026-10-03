// Bounded, pinned safetensors range loader for the exact requested checkpoint.
// It does not load the full model or start inference on its own.
export const MIRAI_REPOSITORY = 'trymirai/Qwen3.8-27B-S-experimental';
export const MIRAI_REVISION = '32d4c0422a14353808659f2f2333b207bc1dd12a';
const CHECKPOINT_URL = '/models/qwen38/vllm/trellis.mirai';
const CHUNK_BYTES = 8 * 1024 * 1024;

export async function fetchCheckpointRange(start, length, {signal, onProgress} = {}) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(length) || start < 0 || length <= 0 || length > CHUNK_BYTES) throw new Error('Invalid checkpoint range');
  const end = start + length - 1;
  const response = await fetch(`${CHECKPOINT_URL}?browser-range=${start}-${end}`, {headers:{Range:`bytes=${start}-${end}`}, signal});
  if (response.status !== 206) {
    await response.body?.cancel();
    throw new Error(`Checkpoint server must honor bounded byte ranges (received ${response.status}).`);
  }
  const contentRange = response.headers.get('content-range');
  if (contentRange && !contentRange.startsWith(`bytes ${start}-${end}/`)) {
    await response.body?.cancel(); throw new Error('Checkpoint server returned a different range.');
  }
  const reader = response.body.getReader(), output = new Uint8Array(length);
  let received = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      if (received + value.byteLength > length) throw new Error('Oversized checkpoint range');
      output.set(value, received); received += value.byteLength;
      onProgress?.({loaded:received, total:length});
    }
  } finally { reader.releaseLock(); }
  if (received !== length) throw new Error('Checkpoint range was truncated.');
  return output.buffer;
}

export async function openMiraiCheckpoint(options = {}) {
  const first = await fetchCheckpointRange(0, 8, options);
  const length64 = new DataView(first).getBigUint64(0, true);
  if (length64 > 4_000_000n) throw new Error('Unexpected checkpoint header size');
  const headerLength = Number(length64);
  const header = JSON.parse(new TextDecoder().decode(await fetchCheckpointRange(8, headerLength, options)));
  const base = headerLength + 8;
  const layers = JSON.parse(header.__metadata__.layers);
  const smallTensors = new Map();
  function tensorInfo(name) {
    const item = header[name];
    if (!item || !Array.isArray(item.data_offsets)) throw new Error(`Checkpoint tensor missing: ${name}`);
    const [start, end] = item.data_offsets;
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end <= start) throw new Error('Invalid tensor offsets');
    return {...item, byteLength:end - start};
  }
  async function tensor(name, {signal = options.signal, onProgress} = {}) {
    if (smallTensors.has(name)) return smallTensors.get(name);
    const info = tensorInfo(name);
    const output = new Uint8Array(info.byteLength);
    for (let offset = 0; offset < info.byteLength; offset += CHUNK_BYTES) {
      const length = Math.min(CHUNK_BYTES, info.byteLength - offset);
      output.set(new Uint8Array(await fetchCheckpointRange(base + info.data_offsets[0] + offset, length, {signal,
        onProgress:progress => onProgress?.({tensor:name, loaded:offset + progress.loaded, total:info.byteLength}),
      })), offset);
    }
    if (info.byteLength <= 128 * 1024) smallTensors.set(name, output.buffer);
    return output.buffer;
  }
  return {
    repository:MIRAI_REPOSITORY, revision:MIRAI_REVISION, header, layers, tensorInfo, tensor,
    async projectionTensors(name, settings = {}) {
      const meta = layers[name];
      if (!meta) throw new Error(`Projection missing: ${name}`);
      const result = {
        signs:await tensor(`rotation.signs_${meta.in_features}`, settings),
        small_q:await tensor(`rotation.q_${meta.in_features}`, settings),
      };
      if (meta.has_perm) result.perm = await tensor(`${name}.perm`, settings);
      for (let index = 0; index < meta.blocks.length; index++) {
        const block = meta.blocks[index];
        for (const role of ['packets', 'entries', 'rowscale']) result[`${index}.${role}`] = await tensor(`${name}.${index}.${role}`, settings);
        const codebook = `codebook.${block.format.slice(0, 2)}`;
        result[codebook] = await tensor(codebook, settings);
      }
      return result;
    },
    clearSmallTensorCache() { smallTensors.clear(); },
  };
}
