import { MIRAI_REVISION } from './mirai-checkpoint.mjs';

export function decodeDenseTensor(bytes, dtype) {
  if (dtype === 'F32') return new Float32Array(bytes.slice(0));
  if (dtype === 'BF16') {
    const half = new Uint16Array(bytes), output = new Float32Array(half.length), bits = new Uint32Array(output.buffer);
    for (let index = 0; index < half.length; index++) bits[index] = half[index] << 16;
    return output;
  }
  throw new Error(`Unsupported dense tensor dtype: ${dtype}`);
}

export async function openDenseWeights(base = '/models/qwen38/vllm/') {
  const response = await fetch(base + 'dense-manifest.json');
  if (!response.ok) throw new Error('Dense checkpoint manifest unavailable');
  const manifest = await response.json();
  if (manifest.revision !== MIRAI_REVISION) throw new Error('Dense checkpoint revision mismatch');
  const shards = new Map();
  return {
    manifest,
    async tensor(name) {
      const item = manifest.tensors[name];
      if (!item) throw new Error(`Dense tensor missing: ${name}`);
      if (!shards.has(item.file)) shards.set(item.file, (async () => {
        const result = await fetch(base + item.file);
        if (!result.ok) throw new Error(`Dense checkpoint shard unavailable: ${item.file}`);
        const buffer = await result.arrayBuffer();
        const expected = manifest.files[item.file];
        if (buffer.byteLength !== expected.bytes) throw new Error('Dense checkpoint length mismatch');
        const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))].map(byte => byte.toString(16).padStart(2, '0')).join('');
        if (hash !== expected.sha256) throw new Error('Dense checkpoint integrity mismatch');
        return buffer;
      })());
      const shard = await shards.get(item.file);
      return decodeDenseTensor(shard.slice(...item.offsets), item.dtype);
    },
    clearCache() { shards.clear(); },
  };
}
