"""Validate a complete, permuted two-block projection from the exact checkpoint."""
import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fetch_mirai_fixtures import ROOT, REVISION, read_range, FORMATS
from mirai_reference import reference, bf16

def main():
    header = json.loads((ROOT / 'model-source/mirai-sidecar-header.json').read_text())
    meta = json.loads(header['__metadata__']['layers'])['layers.3.self_attn.qkv_proj']
    base = 8 + struct.unpack('<Q', read_range(0, 8))[0]
    directory = ROOT / 'webgpu-fixtures/projection'
    directory.mkdir(exist_ok=True)
    names = {role:f'layers.3.self_attn.qkv_proj.{role}' for role in ['perm']}
    for index, block in enumerate(meta['blocks']):
        for role in ['packets', 'entries', 'rowscale']: names[f'{index}.{role}'] = f'layers.3.self_attn.qkv_proj.{index}.{role}'
        names[f"codebook.{block['format'][:2]}"] = f"codebook.{block['format'][:2]}"
    names['signs'] = 'rotation.signs_5120'; names['small_q'] = 'rotation.q_5120'
    def fetch(pair):
        role, name = pair
        info = header[name]; start, end = info['data_offsets']; size = end - start
        if size > 32 * 1024 * 1024: raise RuntimeError('Projection fixture exceeded bounded tensor size')
        content = read_range(base + start, size); filename = role + '.bin'
        (directory / filename).write_bytes(content)
        return role, {'file':filename, 'dtype':info['dtype'], 'source_tensor':name, 'bytes':size, 'sha256':hashlib.sha256(content).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool: tensors = dict(pool.map(fetch, names.items()))
    manifest = {'source_repository':'trymirai/Qwen3.8-27B-S-experimental', 'source_revision':REVISION, 'layer':'layers.3.self_attn.qkv_proj', 'metadata':meta, 'tensors':tensors}
    (directory / 'fixture.json').write_text(json.dumps(manifest, indent=2) + '\n')
    indices = np.arange(5120, dtype=np.float32)
    x = bf16(np.sin(indices * np.float32(.017)) * np.float32(.1) + np.cos(indices * np.float32(.031)) * np.float32(.05))
    internal = []
    for index, block in enumerate(meta['blocks']):
        v, _, steps, _ = FORMATS[block['format']]
        parts = {role:tensors[f'{index}.{role}'] for role in ['packets', 'entries', 'rowscale']}
        parts.update(signs=tensors['signs'], small_q=tensors['small_q'], codebook=tensors[f"codebook.{block['format'][:2]}"])
        block_manifest = {'format':block['format'], 'rows':block['rows'], 'columns':5120, 'rotation_order':5, 'packets_per_row':5120 // (v * steps), 'tensors':parts}
        result, _, _, _ = reference(directory, block_manifest, x)
        internal.append(result)
    permutation = np.fromfile(directory / tensors['perm']['file'], '<i4')
    expected = np.concatenate(internal)[permutation]
    x.tofile(directory / 'input.bin'); expected.tofile(directory / 'expected.bin')
    print('Complete projection:', len(expected), 'outputs;', sum(item['bytes'] for item in tensors.values()), 'source bytes', flush=True)

if __name__ == '__main__': main()
