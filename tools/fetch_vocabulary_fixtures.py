"""Fetch bounded vocabulary slices and produce independent NumPy references."""
import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fetch_mirai_fixtures import ROOT, REVISION, read_range
from mirai_reference import bf16

def hadamard32(x):
    x = np.asarray(x, np.float32).reshape(-1, 32).copy()
    for stride in (1, 2, 4, 8, 16):
        v = x.reshape(-1, 2, stride)
        a, b = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = a + b, a - b
    return x / np.float32(np.sqrt(32))

def main():
    header = json.loads((ROOT / 'model-source/mirai-sidecar-header.json').read_text())
    base = 8 + struct.unpack('<Q', read_range(0, 8))[0]
    rows, hidden = 96, 5120
    directory = ROOT / 'webgpu-fixtures/vocabulary'
    directory.mkdir(exist_ok=True)
    def fetch(name):
        item = header[name]
        start, end = item['data_offsets']
        length = end - start
        if name.endswith('codes'): length = rows * item['shape'][1]
        elif name.endswith('ladder_indices'): length = rows * 40
        elif name.endswith('row_scales'): length = rows * 4
        data = read_range(base + start, length)
        filename = name + '.bin'
        (directory / filename).write_bytes(data)
        return name, {'file': filename, 'dtype': item['dtype'], 'source_shape': item['shape'], 'bytes': length, 'sha256': hashlib.sha256(data).hexdigest()}
    names = [f'{kind}.{role}' for kind in ('embedding', 'head') for role in ('codes', 'row_scales', 'ladder', 'ladder_indices', 'signs')] + ['embedding.table']
    with ThreadPoolExecutor(max_workers=4) as pool: tensors = dict(pool.map(fetch, names))
    manifest = {'source_repository': 'trymirai/Qwen3.8-27B-S-experimental', 'source_revision': REVISION, 'rows': rows, 'hidden': hidden, 'tensors': tensors}
    (directory / 'fixture.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def tensor(name):
        dtype = {'U8':'u1', 'I8':'i1', 'F32':'<f4', 'F16':'<f2'}[tensors[name]['dtype']]
        return np.fromfile(directory / tensors[name]['file'], dtype=dtype)
    ids = np.array([0, 1, 42, 85, 95], '<u4')
    codes = tensor('embedding.codes').reshape(rows, hidden // 4)[ids]
    table = tensor('embedding.table').reshape(256, 4)
    values = table[codes].reshape(len(ids), hidden).astype(np.float32)
    indices = tensor('embedding.ladder_indices').reshape(rows, 40)[ids]
    unpacked = np.stack((indices & 15, indices >> 4), axis=-1).reshape(len(ids), 80)
    scale = tensor('embedding.row_scales')[ids, None] * tensor('embedding.ladder').astype(np.float32)[unpacked]
    weights = hadamard32(values * np.repeat(scale, 64, axis=1)).reshape(len(ids), hidden)
    expected = bf16(weights * tensor('embedding.signs'))
    ids.tofile(directory / 'ids.bin'); expected.tofile(directory / 'embedding-expected.bin')
    i = np.arange(hidden, dtype=np.float32)
    x = bf16(np.sin(i * np.float32(.017)) * np.float32(.1) + np.cos(i * np.float32(.031)) * np.float32(.05))
    rotated = hadamard32(x * tensor('head.signs')).reshape(hidden).astype(np.float16).astype(np.float32)
    codes = tensor('head.codes').reshape(rows, hidden * 3 // 8)
    bit = np.arange(hidden) * 3
    padded = np.pad(codes.astype(np.uint16), ((0, 0), (0, 1)))
    pairs = padded[:, bit // 8] | (padded[:, bit // 8 + 1] << 8)
    values = ((pairs >> (bit % 8)) & 7).astype(np.float32) * 2 - 7
    indices = tensor('head.ladder_indices').reshape(rows, 40)
    unpacked = np.stack((indices & 15, indices >> 4), axis=-1).reshape(rows, 80)
    scales = np.repeat(tensor('head.ladder').astype(np.float32)[unpacked], 64, axis=1)
    weights = (values * scales).astype(np.float16).astype(np.float32)
    logits = (weights * rotated).sum(axis=1, dtype=np.float32) * tensor('head.row_scales')
    x.tofile(directory / 'head-input.bin'); rotated.tofile(directory / 'head-rotated.bin'); logits.tofile(directory / 'head-expected.bin')
    print('Fetched vocabulary fixtures and generated references:', sum(t['bytes'] for t in tensors.values()), 'source bytes')

if __name__ == '__main__': main()
