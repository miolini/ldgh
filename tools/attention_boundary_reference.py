"""Independent attention references at context boundaries.

Earlier K/V entries are zero, as in a synthetic sequence with zero projected
keys and values. The final position contains the supplied original-weight norm
input. This exercises attention across every populated cache index without
requiring a full model or thousands of projection runs.
"""
import numpy as np
from qwen38_reference import DIRECTORY, pattern, rms, dense
from mirai_reference import bf16

wq = dense('model.layers.3.self_attn.q_norm.weight')
wk = dense('model.layers.3.self_attn.k_norm.weight')
for position in (255, 256, 4095):
    x = pattern(14336, position, .3)
    q_gate = x[:12288].reshape(24, 512)
    q = rms(q_gate[:, :256], wq)
    k = rms(x[12288:13312].reshape(4, 256), wk)
    v = x[13312:].reshape(4, 256)
    angles = np.array([np.float32(position / 10000000 ** (frequency / 32)) for frequency in range(32)], np.float32)
    cosine, sine = bf16(np.cos(angles)), bf16(np.sin(angles))
    def rotate(y):
        y = y.copy()
        low, high = y[:, :32].copy(), y[:, 32:64].copy()
        y[:, :32] = bf16(low * cosine - high * sine)
        y[:, 32:64] = bf16(high * cosine + low * sine)
        return y
    q, k = rotate(q), rotate(k)
    scores = np.sum(np.repeat(k, 6, axis=0) * q, axis=-1, dtype=np.float32) * np.float32(.0625)
    maximum = np.maximum(scores, np.float32(0))
    numerator = np.exp(scores - maximum)
    probability = numerator / (np.exp(-maximum) * np.float32(position) + numerator)
    output = bf16(np.repeat(v, 6, axis=0) * probability[:, None])
    gate = bf16(np.float32(1) / (np.float32(1) + np.exp(-q_gate[:, 256:])))
    result = bf16(output * gate)
    x.tofile(DIRECTORY / f'boundary-input-{position}.bin')
    q.tofile(DIRECTORY / f'boundary-query-{position}.bin')
    result.tofile(DIRECTORY / f'boundary-expected-{position}.bin')
print('Generated attention references at positions 255, 256 and 4095')
