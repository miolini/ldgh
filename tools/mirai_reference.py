"""Independent NumPy reference for the supplied CUDA plugin's Mirai S GEMV math.

It preserves compressed checkpoint values and the plugin's two-plane activation
quantization. It validates layer operations, not complete model generation.
"""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORMATS = {"v4t8": (4, 8, 16, 1), "v2t4": (2, 4, 32, 1), "v2t6": (2, 6, 64, 3)}


def bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def read_tensor(directory, manifest, role):
    info = manifest["tensors"][role]
    dtype = {"F32": "<f4", "I16": "<i2", "U8": "u1"}[info["dtype"]]
    return np.fromfile(directory / info["file"], dtype=dtype)


def reference(directory, manifest, x):
    v, t, steps, words = FORMATS[manifest["format"]]
    rows, columns = manifest["rows"], manifest["columns"]
    order = manifest["rotation_order"]
    width = columns // order
    signs = read_tensor(directory, manifest, "signs").reshape(width, order)
    small_q = read_tensor(directory, manifest, "small_q").reshape(order, order)
    signed_x = bf16(x).reshape(width, order) * signs
    rotated = np.zeros((width, order), np.float32)
    for out in range(order):
        for source in range(order):
            rotated[:, out] += signed_x[:, source] * small_q[out, source]
    stride = 1
    while stride < width:
        view = rotated.reshape(-1, 2, stride, order)
        low, high = view[:, 0].copy(), view[:, 1].copy()
        view[:, 0] = low + high
        view[:, 1] = low - high
        stride *= 2
    rotated *= np.float32(1 / np.sqrt(width))
    rotated = rotated.reshape(-1)
    maximum = np.abs(rotated).max()
    scale = np.float32(maximum / np.float32(127)) if maximum else np.float32(1)
    scaled = rotated * np.float32(1 / scale)
    coarse = np.clip(np.rint(scaled), -127, 127).astype(np.int32)
    fine = np.clip(np.rint((scaled - coarse.astype(np.float32)) * np.float32(254)), -127, 127).astype(np.int32)
    residue_sums = scale * (coarse.reshape(-1, 4).sum(axis=0).astype(np.float32) +
                             fine.reshape(-1, 4).sum(axis=0).astype(np.float32) / np.float32(254))
    entries = read_tensor(directory, manifest, "entries").astype(np.uint32)
    packet_count = manifest["packets_per_row"]
    entries = entries.reshape(rows // 32, packet_count, 32).transpose(0, 2, 1).reshape(rows, packet_count)
    packets = np.fromfile(directory / manifest["tensors"]["packets"]["file"], dtype="<u4")
    packets = packets.reshape(rows // 32, packet_count, words, 32, 4).transpose(0, 3, 1, 2, 4).reshape(rows, packet_count, words * 4)
    acc0, acc1 = np.zeros(rows, np.int32), np.zeros(rows, np.int32)
    for packet in range(packet_count):
        state = entries[:, packet].copy()
        for step in range(steps):
            bit = step * t
            word, shift = bit // 32, bit % 32
            symbol = packets[:, packet, word] >> np.uint32(shift)
            if shift + t > 32:
                symbol |= packets[:, packet, word + 1] << np.uint32(32 - shift)
            state = ((state << np.uint32(t)) | (symbol & np.uint32((1 << t) - 1))) & np.uint32(0xffff)
            hashed = state * np.uint32(0xcfccb83f) + np.uint32(0x584b4aa3)
            hashed ^= hashed >> np.uint32(16)
            hashed *= np.uint32(0x85ebca6b)
            hashed ^= hashed >> np.uint32(16)
            for position in range(v):
                byte = ((hashed >> np.uint32(8 * position)) & np.uint32(255)).astype(np.int32)
                pairs = (byte & 3) + ((byte >> 2) & 3) + ((byte >> 4) & 3) + ((byte >> 6) & 3)
                level_plus_54 = 8 * pairs + ((3 * (byte & 15)) & 15)
                column = packet * steps * v + step * v + position
                acc0 += level_plus_54 * coarse[column]
                acc1 += level_plus_54 * fine[column]
    acc0 -= np.int32(54 * int(coarse.sum()))
    acc1 -= np.int32(54 * int(fine.sum()))
    codebook = read_tensor(directory, manifest, "codebook")
    offset = np.float32(0)
    for i in range(4):
        offset = np.float32(offset + np.float32(codebook[i + 1] * residue_sums[i]))
    dot = np.float32(codebook[0] * scale) * (acc0.astype(np.float32) + acc1.astype(np.float32) * np.float32(1 / 254))
    rowscale = read_tensor(directory, manifest, "rowscale")
    return bf16(rowscale * (dot + offset)), rotated, coarse, fine


def main():
    for fmt in json.loads((ROOT / "webgpu-fixtures/index.json").read_text()):
        directory = ROOT / "webgpu-fixtures" / fmt
        manifest = json.loads((directory / "fixture.json").read_text())
        indices = np.arange(manifest["columns"], dtype=np.float32)
        x = bf16(np.sin(indices * np.float32(.017)) * np.float32(.1) + np.cos(indices * np.float32(.031)) * np.float32(.05))
        expected, rotated, coarse, fine = reference(directory, manifest, x)
        x.tofile(directory / "input.bin")
        expected.tofile(directory / "expected.bin")
        rotated.tofile(directory / "rotated.bin")
        np.stack([coarse, fine], axis=1).astype("<i4").tofile(directory / "planes.bin")
        print(fmt, "rows", len(expected), "expected output range", float(expected.min()), float(expected.max()))


if __name__ == "__main__":
    main()
