"""Local dashboard server with resumable, verified model byte ranges.

The dashboard chat connects to its configured OpenAI-compatible inference service.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
REVISION = '32d4c0422a14353808659f2f2333b207bc1dd12a'
SIZE = 8449310484
SHA256 = '56009f4fc373c17ac0e9950224d9fe6123b416021c7c65317d5342d6d783d0e2'
CHUNK_SIZE = 8 * 1024 * 1024
MODEL_URL = f'https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental/resolve/{REVISION}/vllm/trellis.mirai'
LLM_PROVIDER_URL = os.environ.get('FIELDNOTE_LLM_BASE_URL', 'http://100.99.215.85:8082/v1').rstrip('/')
LLM_MODEL_ID = 'qwen3.8-27b-uncensored'
LLM_MAX_REQUEST_BYTES = 2 * 1024 * 1024

class ModelCache:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'trellis.mirai'
        self.state_path = directory / 'chunks.json'
        self.axel_path = directory / 'axel-trellis.mirai'
        self.axel_pid_path = directory / 'axel-pid.json'
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.inflight = set()
        self.partial = {}
        self.checked = set()
        self.hashes = {}
        self.verified = False
        self.error = None
        self.worker = None
        self.axel_worker = None
        self.state = 'idle'
        if self.state_path.exists() and self.path.exists():
            saved = json.loads(self.state_path.read_text())
            if saved.get('revision') == REVISION and saved.get('size') == SIZE and self.path.stat().st_size == SIZE:
                self.hashes = saved.get('chunks', {})
                self.verified = saved.get('verified') is True
                self.state = 'ready' if self.verified else 'cached-partial'
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.ftruncate(self.fd, SIZE)  # Sparse allocation: physical bytes only as chunks arrive.

    def status(self):
        with self.lock:
            completed = sum(min(CHUNK_SIZE, SIZE - int(index) * CHUNK_SIZE) for index in self.hashes)
            downloaded = completed + sum(self.partial.values())
            axel_bytes_done = 0
            axel_running = False
            try:
                marker = json.loads(self.axel_pid_path.read_text())
                pid = marker.get('pid')
                if marker.get('revision') == REVISION and marker.get('file') == self.axel_path.name and isinstance(pid, int) and pid > 1:
                    process = subprocess.run(['ps', '-p', str(pid), '-o', 'comm='], capture_output=True, text=True, timeout=2)
                    axel_running = process.returncode == 0 and process.stdout.strip().split('/')[-1] == 'axel'
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            try:
                state = self.axel_path.with_suffix('.mirai.st')
                raw = state.read_bytes()
                if len(raw) >= 10:
                    count = struct.unpack_from('<H', raw, 0)[0]
                    if 1 <= count <= 128 and len(raw) == 10 + 16 * count:
                        axel_bytes_done = min(SIZE, struct.unpack_from('<Q', raw, 2)[0])
            except OSError:
                pass
            intervals = [(int(index) * CHUNK_SIZE, min(SIZE, (int(index) + 1) * CHUNK_SIZE)) for index in self.hashes]
            intervals += [(int(index) * CHUNK_SIZE, min(SIZE, int(index) * CHUNK_SIZE + amount)) for index, amount in self.partial.items()]
            # Axel's state file stores total bytes received at offset 2. Worker
            # segments can be reassigned while downloading, so reconstructing
            # coverage from their current offsets can substantially overcount.
            # Use the larger independent byte count conservatively; this can
            # understate the union with cached ranges, but never claims 100% early.
            downloaded = max(downloaded, axel_bytes_done)
            intervals.sort()
            unique = 0
            if intervals:
                start, end = intervals[0]
                for first, last in intervals[1:]:
                    if first <= end: end = max(end, last)
                    else: unique += end - start; start, end = first, last
                unique += end - start
            active = (self.worker is not None and self.worker.is_alive()) or (self.axel_worker is not None and self.axel_worker.is_alive()) or axel_running
            if axel_running and self.state not in ('ready', 'verifying'):
                self.state = 'downloading'
            elif self.axel_worker is not None and self.axel_worker.is_alive():
                self.state = 'verifying'
            elif self.state == 'downloading' and not (self.worker is not None and self.worker.is_alive()):
                self.state = 'cached-partial'
            status = {'repository':'trymirai/Qwen3.8-27B-S-experimental', 'revision':REVISION,
                    'status':self.state, 'downloaded':downloaded, 'total':SIZE, 'progress':downloaded / SIZE,
                    'verifiedBytes':completed, 'axelBytesDone':axel_bytes_done,
                    'cachedChunks':len(self.hashes), 'totalChunks':(SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE,
                    'active':active, 'inflight':len(self.inflight), 'inflightChunks':sorted(self.inflight),
                    'sha256Verified':self.verified, 'expectedSha256':SHA256, 'error':self.error}
            if (not self.verified and not active and not self.inflight and not state.exists()
                    and self.axel_path.exists() and self.axel_path.stat().st_size == SIZE):
                if self.axel_worker is None or not self.axel_worker.is_alive():
                    self.state = 'verifying'
                    self.axel_worker = threading.Thread(target=self.verify_axel, daemon=True, name='mirai-axel-verify')
                    self.axel_worker.start()
                status.update({'status':'verifying', 'downloaded':SIZE, 'progress':1.0, 'active':True})
            return status

    def verify_axel(self):
        try:
            digest = hashlib.sha256()
            chunk_hashes = {}
            with self.axel_path.open('rb') as source:
                index = 0
                while data := source.read(CHUNK_SIZE):
                    digest.update(data)
                    chunk_hashes[str(index)] = hashlib.sha256(data).hexdigest()
                    index += 1
            if index != (SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE or digest.hexdigest() != SHA256:
                raise RuntimeError('Axel file does not match the pinned checkpoint SHA256')
            with self.changed:
                while self.inflight:
                    self.changed.wait(timeout=5)
                os.close(self.fd)
                os.replace(self.axel_path, self.path)
                self.fd = os.open(self.path, os.O_RDWR)
                self.hashes = chunk_hashes
                self.checked = set(range(len(chunk_hashes)))
                self.verified = True
                self.state = 'ready'
                self.error = None
                self.save()
            self.axel_pid_path.unlink(missing_ok=True)
            self.axel_path.with_suffix('.mirai.st').unlink(missing_ok=True)
            (self.directory / 'axel-source-url').unlink(missing_ok=True)
        except Exception as error:
            with self.lock:
                self.state = 'error'
                self.error = str(error)

    def save(self):
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'revision':REVISION, 'size':SIZE, 'verified':self.verified, 'chunks':self.hashes}))
        os.replace(temporary, self.state_path)

    def chunk(self, index):
        start = index * CHUNK_SIZE
        length = min(CHUNK_SIZE, SIZE - start)
        if index < 0 or length <= 0: raise ValueError('Chunk outside checkpoint')
        key = str(index)
        with self.changed:
            while index in self.inflight: self.changed.wait(timeout=30)
            if key in self.hashes:
                data = os.pread(self.fd, length, start)
                if len(data) == length and (index in self.checked or hashlib.sha256(data).hexdigest() == self.hashes[key]):
                    self.checked.add(index)
                    return data
                del self.hashes[key]
                self.verified = False
            self.inflight.add(index)
        try:
            end = start + length - 1
            request = urllib.request.Request(MODEL_URL + f'?local-range={start}-{end}', headers={'Range':f'bytes={start}-{end}'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        expected_range = f'bytes {start}-{end}/{SIZE}'
                        if response.status != 206 or response.headers.get('Content-Range') != expected_range: raise RuntimeError('Source did not honor the exact pinned range')
                        data = bytearray()
                        while len(data) < length:
                            part = response.read(min(256 * 1024, length - len(data)))
                            if not part: break
                            data.extend(part)
                            with self.lock: self.partial[index] = len(data)
                    if len(data) != length: raise RuntimeError('Source range length mismatch')
                    break
                except Exception:
                    if attempt == 2: raise
                    time.sleep(1 + attempt)
            digest = hashlib.sha256(data).hexdigest()
            written = 0
            while written < length: written += os.pwrite(self.fd, data[written:], start + written)
            os.fsync(self.fd)
            with self.changed:
                self.hashes[key] = digest
                self.partial.pop(index, None)
                self.checked.add(index)
                self.error = None
                self.save()
            return data
        except Exception as error:
            with self.lock: self.error = str(error)
            raise
        finally:
            with self.changed:
                self.inflight.discard(index)
                self.partial.pop(index, None)
                self.changed.notify_all()

    def fragment(self, start, end):
        """Small cold reads do not wait for an entire 8 MB cache chunk."""
        if not 0 <= start <= end < SIZE or end - start + 1 > 256 * 1024: raise ValueError('Fragment outside bound')
        directory = self.directory / 'ranges'
        directory.mkdir(exist_ok=True)
        destination = directory / f'{start}-{end}.bin'
        length = end - start + 1
        if destination.exists() and destination.stat().st_size == length: return destination.read_bytes()
        request = urllib.request.Request(MODEL_URL + f'?fixture={start}-{end}', headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}': raise RuntimeError('Invalid source fragment range')
            data = response.read(length + 1)
        if len(data) != length: raise RuntimeError('Truncated source fragment')
        temporary = destination.with_suffix(f'.{threading.get_ident()}.tmp')
        temporary.write_bytes(data); os.replace(temporary, destination)
        return data

    def start(self):
        with self.lock:
            if self.verified or (self.worker is not None and self.worker.is_alive()) or self.status()['active']: return self.status()
            self.state = 'downloading'
            self.error = None
            self.worker = threading.Thread(target=self.download, daemon=True, name='mirai-cache-download')
            self.worker.start()
            return self.status()

    def download(self):
        try:
            count = (SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
            indices = iter(range(count))
            failure = threading.Event()
            errors = []
            def cache_worker():
                while not failure.is_set():
                    with self.lock:
                        index = next(indices, None)
                    if index is None: return
                    try: self.chunk(index)
                    except Exception as error:
                        with self.lock:
                            errors.append(error)
                            self.state = 'error'
                            self.error = str(error)
                        failure.set()
                        return
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(cache_worker) for _ in range(4)]
                for future in futures: future.result()
            if errors: raise errors[0]
            with self.lock: self.state = 'verifying'
            digest = hashlib.sha256()
            with self.path.open('rb') as source:
                while data := source.read(CHUNK_SIZE): digest.update(data)
            if digest.hexdigest() != SHA256: raise RuntimeError('Full checkpoint SHA256 differs from Hugging Face LFS identity')
            with self.lock:
                self.verified = True
                self.state = 'ready'
                self.save()
        except Exception as error:
            with self.lock:
                self.state = 'error'
                self.error = str(error)


def byte_range(value, size):
    if not value: return 0, size - 1, False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
    if not match or not any(match.groups()): raise ValueError('One byte range required')
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0: raise ValueError('Invalid suffix')
        start, end = max(0, size - suffix), size - 1
    else:
        start, end = int(first), min(int(last), size - 1) if last else size - 1
    if start >= size or end < start: raise ValueError('Unsatisfiable range')
    return start, end, True

class Handler(SimpleHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(ROOT), **kwargs)

    def json_response(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if self.command != 'HEAD': self.wfile.write(body)

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and urllib.parse.urlparse(origin).netloc != self.headers.get('Host'):
            self.json_response({'error':'Origin must match the local server'}, 403); return
        path = urllib.parse.urlparse(self.path).path
        if path == '/api/model-cache/start': self.json_response(self.server.model_cache.start()); return
        if path == '/api/llm/chat/completions': self.proxy_llm_completion(); return
        self.json_response({'error':'Unknown route'}, 404)

    def do_GET(self): self.route()
    def do_HEAD(self): self.route()

    def route(self):
        path = urllib.parse.urlparse(self.path).path
        if path == '/api/model-cache/status': self.json_response(self.server.model_cache.status()); return
        if path == '/api/llm/models': self.proxy_llm_models(); return
        prefix = '/models/qwen38/vllm/'
        if path.startswith(prefix):
            filename = urllib.parse.unquote(path[len(prefix):])
            if filename == 'trellis.mirai': self.serve_model(); return
            if '/' in filename or '\\' in filename or filename in ('', '.', '..'):
                self.json_response({'error':'Invalid model asset'}, 404); return
            source = ROOT / 'model-source/vllm' / filename
            if not source.is_file(): self.json_response({'error':'Model asset not cached'}, 404); return
            self.serve_file(source); return
        if path.startswith('/model-cache/'):
            self.json_response({'error':'Use the model range route'}, 404); return
        source = Path(self.translate_path(self.path))
        if source.is_file(): self.serve_file(source); return
        super().do_HEAD() if self.command == 'HEAD' else super().do_GET()

    def proxy_llm_models(self):
        request = urllib.request.Request(f'{LLM_PROVIDER_URL}/models', headers={'Accept':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=20) as upstream:
                body = upstream.read(2 * 1024 * 1024 + 1)
                if len(body) > 2 * 1024 * 1024:
                    self.json_response({'error':'Model catalog response is too large'}, 502); return
                self.send_response(upstream.status)
                self.send_header('Content-Type', upstream.headers.get('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                if self.command != 'HEAD': self.wfile.write(body)
        except urllib.error.HTTPError as error:
            self.json_response({'error':f'LLM provider returned HTTP {error.code}'}, 502)
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            self.json_response({'error':f'LLM provider is unavailable: {error}'}, 502)

    def proxy_llm_completion(self):
        try: length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.json_response({'error':'Invalid request length'}, 400); return
        if length <= 0 or length > LLM_MAX_REQUEST_BYTES:
            self.json_response({'error':'Request body must be between 1 byte and 2 MiB'}, 413 if length > LLM_MAX_REQUEST_BYTES else 400); return
        body = self.rfile.read(length)
        try: payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.json_response({'error':'Request body must be valid JSON'}, 400); return
        if not isinstance(payload, dict) or payload.get('model') != LLM_MODEL_ID or not isinstance(payload.get('messages'), list):
            self.json_response({'error':f'Request must include model {LLM_MODEL_ID} and a messages array'}, 400); return
        if len(payload['messages']) > 500:
            self.json_response({'error':'Too many chat messages'}, 400); return
        streaming = payload.get('stream') is True
        headers = {'Content-Type':'application/json', 'Accept':'text/event-stream' if streaming else 'application/json'}
        request = urllib.request.Request(f'{LLM_PROVIDER_URL}/chat/completions', data=body, headers=headers, method='POST')
        try:
            upstream = urllib.request.urlopen(request, timeout=600)
        except urllib.error.HTTPError as error:
            body = error.read(LLM_MAX_REQUEST_BYTES + 1)
            self.send_response(error.code)
            self.send_header('Content-Type', error.headers.get('Content-Type', 'application/json'))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass
            return
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            self.json_response({'error':f'LLM provider is unavailable: {error}'}, 502); return

        with upstream:
            self.send_response(upstream.status)
            self.send_header('Content-Type', upstream.headers.get('Content-Type', 'application/json'))
            self.send_header('Cache-Control', 'no-store, no-transform')
            self.send_header('X-Accel-Buffering', 'no')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            try:
                reader = getattr(upstream, 'read1', upstream.read)
                while chunk := reader(64 * 1024):
                    self.wfile.write(f'{len(chunk):X}\r\n'.encode('ascii'))
                    self.wfile.write(chunk)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                self.wfile.write(b'0\r\n\r\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True

    def file_headers(self, size, content_type, etag):
        try: start, end, partial = byte_range(self.headers.get('Range'), size)
        except ValueError:
            self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.send_header('Content-Length', '0'); self.end_headers(); return None
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('ETag', f'"{etag}"')
        if partial: self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        return start, end

    def serve_file(self, source):
        size = source.stat().st_size
        if size == 0:
            self.send_response(200); self.send_header('Content-Length', '0'); self.end_headers(); return
        limits = self.file_headers(size, self.guess_type(str(source)), f'{source.stat().st_mtime_ns}-{size}')
        if limits is None or self.command == 'HEAD': return
        start, end = limits
        try:
            with source.open('rb') as data:
                data.seek(start)
                remaining = end - start + 1
                while remaining:
                    part = data.read(min(256 * 1024, remaining))
                    if not part: break
                    self.wfile.write(part); remaining -= len(part)
        except (BrokenPipeError, ConnectionResetError): pass

    def serve_model(self):
        limits = self.file_headers(SIZE, 'application/octet-stream', SHA256)
        if limits is None or self.command == 'HEAD': return
        start, end = limits
        try:
            first_chunk = start // CHUNK_SIZE
            if end - start + 1 <= 256 * 1024 and str(first_chunk) not in self.server.model_cache.hashes:
                self.wfile.write(self.server.model_cache.fragment(start, end)); self.wfile.flush(); return
            position = start
            while position <= end:
                index = position // CHUNK_SIZE
                data = self.server.model_cache.chunk(index)
                offset = position % CHUNK_SIZE
                part = data[offset:offset + min(len(data) - offset, end - position + 1)]
                self.wfile.write(part); self.wfile.flush(); position += len(part)
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as error:
            self.log_error('Model stream failed: %s', error)
            self.close_connection = True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'model-cache' / REVISION)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.model_cache = ModelCache(args.cache_dir)
    print(f'Dashboard and model cache: http://localhost:{args.port}', flush=True)
    server.serve_forever()

if __name__ == '__main__': main()
