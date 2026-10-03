"""Populate the existing local range cache with bounded concurrent readers.

The running server owns download deduplication, disk writes and integrity checks.
This process never stores a second copy of the weights.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import urllib.request

BASE = 'http://localhost:8000'
CHUNK = 8 * 1024 * 1024
status = json.load(urllib.request.urlopen(BASE + '/api/model-cache/status'))
size = status['total']
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--first-index', type=int)
args = parser.parse_args()
if not 1 <= args.workers <= 24: parser.error('Workers must be between 1 and 24')
first = status['totalChunks'] - 1 if args.first_index is None else args.first_index
if not 0 <= first < status['totalChunks']: parser.error('First index outside checkpoint')
indices = iter(reversed(range(first + 1)))
lock = threading.Lock()
completed = 0
failed = 0
last_report = time.monotonic()

def reader():
    global completed, failed, last_report
    while True:
        with lock: index = next(indices, None)
        if index is None: return
        start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
        try:
            request = urllib.request.Request(BASE + '/models/qwen38/vllm/trellis.mirai', headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(request, timeout=3600) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}': raise RuntimeError('Range mismatch')
                received = 0
                while part := response.read(256 * 1024): received += len(part)
                if received != end - start + 1: raise RuntimeError('Incomplete range')
            with lock: completed += 1
        except Exception:
            with lock: failed += 1
        with lock:
            if time.monotonic() - last_report >= 30:
                print(json.dumps({'completedRanges':completed, 'failedRanges':failed}), flush=True)
                last_report = time.monotonic()

with ThreadPoolExecutor(max_workers=args.workers) as pool:
    for future in [pool.submit(reader) for _ in range(args.workers)]: future.result()
print(json.dumps({'done':True, 'completedRanges':completed, 'failedRanges':failed}), flush=True)
