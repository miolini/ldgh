"""Fetch small, unmodified tensor slices from the pinned Mirai checkpoint."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
REVISION = "32d4c0422a14353808659f2f2333b207bc1dd12a"
URL = f"https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental/resolve/{REVISION}/vllm/trellis.mirai"
FORMATS = {"v4t8": (4, 8, 16, 1), "v2t4": (2, 4, 32, 1), "v2t6": (2, 6, 64, 3)}


def read_range(start, length):
    end = start + length - 1
    request = urllib.request.Request(URL + f"?fixture={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError("The model server did not honor the bounded range request")
        data = response.read(length + 1)
        if len(data) != length:
            raise RuntimeError("Unexpected source range length")
        return data


def main():
    header_path = ROOT / "model-source/mirai-sidecar-header.json"
    if not header_path.exists():
        header_length = struct.unpack("<Q", read_range(0, 8))[0]
        if header_length > 4_000_000:
            raise RuntimeError("Unexpected header length")
        header = json.loads(read_range(8, header_length))
        header_path.write_text(json.dumps(header, indent=2))
    else:
        header = json.loads(header_path.read_text())
        header_length = struct.unpack("<Q", read_range(0, 8))[0]
    tensor_base = header_length + 8
    layers = json.loads(header["__metadata__"]["layers"])
    target = ROOT / "webgpu-fixtures"
    target.mkdir(exist_ok=True)
    manifests = []
    for fmt, (vector_width, _, steps, words) in FORMATS.items():
        candidates = [(meta["out_features"], name, meta) for name, meta in layers.items()
                      if len(meta["blocks"]) == 1 and meta["blocks"][0]["format"] == fmt and not meta["has_perm"]]
        _, key, meta = min(candidates, key=lambda item: item[0])
        rows = min(meta["out_features"], 96)
        columns = meta["in_features"]
        packets = columns // (vector_width * steps)
        tensors = {
            "packets": (f"{key}.0.packets", rows * packets * words * 16),
            "entries": (f"{key}.0.entries", rows * packets * (1 if fmt == "v4t8" else 2)),
            "rowscale": (f"{key}.0.rowscale", rows * 4),
            "codebook": (f"codebook.{fmt[:2]}", None),
            "signs": (f"rotation.signs_{columns}", None),
            "small_q": (f"rotation.q_{columns}", None),
        }
        directory = target / fmt
        directory.mkdir(exist_ok=True)

        def fetch_tensor(pair):
            role, (name, byte_count) = pair
            info = header[name]
            first, last = info["data_offsets"]
            length = byte_count if byte_count is not None else last - first
            if length > 2_000_000:
                raise RuntimeError("Fixture tensor unexpectedly exceeds the size bound")
            content = read_range(tensor_base + first, length)
            filename = role + ".bin"
            (directory / filename).write_bytes(content)
            return role, {"file": filename, "dtype": info["dtype"], "source_tensor": name,
                          "source_shape": info["shape"], "bytes": length, "sha256": hashlib.sha256(content).hexdigest()}

        with ThreadPoolExecutor(max_workers=4) as pool:
            fetched = dict(pool.map(fetch_tensor, tensors.items()))
        manifest = {"source_repository": "trymirai/Qwen3.8-27B-S-experimental", "source_revision": REVISION,
                    "source_file": "vllm/trellis.mirai", "layer": key, "format": fmt,
                    "rows": rows, "columns": columns, "packets_per_row": packets,
                    "rotation_order": header[f"rotation.q_{columns}"]["shape"][0], "tensors": fetched}
        (directory / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(fmt)
        print(fmt, key, rows, columns, sum(item["bytes"] for item in fetched.values()), "bytes", flush=True)
    (target / "index.json").write_text(json.dumps(manifests) + "\n")


if __name__ == "__main__":
    main()
