"""Fetch only target-model dense shards and tokenizer from the pinned checkpoint."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
from fetch_mirai_fixtures import ROOT, REVISION

def main():
    directory = ROOT / 'model-source/vllm'
    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    names = {k:v for k,v in index.items() if k.startswith('model.language_model.')}
    files = sorted(set(names.values()))
    tree = {item['path'].split('/')[-1]:item['size'] for item in json.loads((ROOT / 'model-source/vllm-tree.json').read_text()) if item.get('type') == 'file'}
    if any(tree[f] > 1_000_000 for f in files): raise RuntimeError('Unexpected large target-model dense shard')
    url = f'https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental/resolve/{REVISION}/vllm/'
    def fetch(filename):
        expected = tree[filename]
        path = directory / filename
        if path.exists() and path.stat().st_size == expected: return filename
        with urllib.request.urlopen(url + filename, timeout=60) as response:
            data = response.read(expected + 1)
        if len(data) != expected: raise RuntimeError('Source file length mismatch')
        path.write_bytes(data); print(filename, len(data), flush=True); return filename
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(fetch, files + ['tokenizer.json']))
    headers = {}
    for filename in files:
        data = (directory / filename).read_bytes()
        length = struct.unpack('<Q', data[:8])[0]
        headers[filename] = (json.loads(data[8:8+length]), 8+length)
    tensors = {}
    for name, filename in names.items():
        header, base = headers[filename]
        info = header[name]
        tensors[name.replace('model.language_model.', 'model.', 1)] = {'file':filename, 'dtype':info['dtype'], 'shape':info['shape'], 'offsets':[base + v for v in info['data_offsets']]}
    manifest = {'revision':REVISION, 'tensors':tensors, 'files':{f:{'bytes':tree[f], 'sha256':hashlib.sha256((directory / f).read_bytes()).hexdigest()} for f in files + ['tokenizer.json']}}
    (directory / 'dense-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Target dense tensors:', len(tensors), 'bytes:', sum(tree[f] for f in files), flush=True)

if __name__ == '__main__': main()
