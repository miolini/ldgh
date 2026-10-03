"""Independent sequential reference using original dense checkpoint tensors."""
import json
import numpy as np
from fetch_mirai_fixtures import ROOT
from mirai_reference import bf16

DIRECTORY = ROOT / 'webgpu-fixtures/attention'

def dense(name):
    info = json.loads((ROOT / 'model-source/vllm/dense-manifest.json').read_text())['tensors'][name]
    data = (ROOT / 'model-source/vllm' / info['file']).read_bytes()[slice(*info['offsets'])]
    if info['dtype'] == 'F32': return np.frombuffer(data, '<f4').copy()
    bits = np.frombuffer(data, '<u2').astype('<u4') << 16
    return bits.view('<f4').copy()

def pattern(width, time, scale=.2):
    i = np.arange(width, dtype=np.float32)
    return bf16(np.sin(i * np.float32(.017) + np.float32(time * .31)) * np.float32(scale) + np.cos(i * np.float32(.031) - np.float32(time * .17)) * np.float32(scale / 2))

def rms(x, w, offset=1, gate=None):
    inverse = np.float32(1) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
    y = (x * inverse) * (w + np.float32(offset))
    if gate is not None: y *= gate / (np.float32(1) + np.exp(-gate))
    return bf16(y)

def main():
    DIRECTORY.mkdir(exist_ok=True)
    norm_input = pattern(5120, 0)
    norm_input.tofile(DIRECTORY / 'norm-input.bin')
    rms(norm_input, dense('model.layers.0.input_layernorm.weight')).tofile(DIRECTORY / 'norm-expected.bin')
    other = pattern(5120, 1)
    other.tofile(DIRECTORY / 'add-input.bin'); bf16(norm_input + other).tofile(DIRECTORY / 'add-expected.bin')
    mlp = pattern(34816, 2)
    mlp.tofile(DIRECTORY / 'mlp-input.bin')
    bf16((mlp[:17408] / (np.float32(1) + np.exp(-mlp[:17408]))) * mlp[17408:]).tofile(DIRECTORY / 'mlp-expected.bin')
    prefix = 'model.layers.0.linear_attn.'
    conv = dense(prefix + 'conv1d.weight').reshape(10240, 4)
    A_log, bias, w = dense(prefix + 'A_log'), dense(prefix + 'dt_bias'), dense(prefix + 'norm.weight')
    history = np.zeros((3, 10240), np.float32); state = np.zeros((48, 128, 128), np.float32)
    for time in range(4):
        x, ba = pattern(16384, time), pattern(96, time + 3, .3)
        x.tofile(DIRECTORY / f'delta-input-{time}.bin'); ba.tofile(DIRECTORY / f'delta-ba-{time}.bin')
        values = x[:10240] * conv[:, 3]
        for old in range(3): values += history[old] * conv[:, old]
        history = np.concatenate((history[1:], x[None, :10240]))
        values = bf16(values / (np.float32(1) + np.exp(-values)))
        qk = values[:4096].reshape(32, 128)
        qk = bf16(qk / np.sqrt(np.sum(qk * qk, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6)))
        q, k = qk[:16], qk[16:]
        v = values[4096:].reshape(48, 128)
        alpha = ba[48:] + bias
        softplus = np.where(alpha > 20, alpha, np.log(np.float32(1) + np.exp(alpha)))
        decay = np.exp(-np.exp(A_log) * softplus)
        beta = np.float32(1) / (np.float32(1) + np.exp(-ba[:48]))
        state *= decay[:, None, None]
        prediction = np.zeros((48, 128), np.float32)
        for row in range(128): prediction += state[:, row] * np.repeat(k[:, row], 3)[:, None]
        delta = beta[:, None] * (v - prediction)
        output = np.zeros((48, 128), np.float32)
        for row in range(128):
            state[:, row] += np.repeat(k[:, row], 3)[:, None] * delta
            output += state[:, row] * np.repeat(q[:, row], 3)[:, None]
        output = bf16(output * np.float32(1 / np.sqrt(128)))
        result = rms(output, w, offset=0, gate=x[10240:].reshape(48, 128))
        values.tofile(DIRECTORY / f'delta-conv-{time}.bin'); qk.tofile(DIRECTORY / f'delta-qk-{time}.bin')
        result.tofile(DIRECTORY / f'delta-expected-{time}.bin')
    state.tofile(DIRECTORY / 'delta-state-expected.bin')
    keys, vals = [], []
    wq, wk = dense('model.layers.3.self_attn.q_norm.weight'), dense('model.layers.3.self_attn.k_norm.weight')
    for time in range(4):
        x = pattern(14336, time, .3); x.tofile(DIRECTORY / f'attention-input-{time}.bin')
        q_and_gate = x[:12288].reshape(24, 512)
        q = rms(q_and_gate[:, :256], wq); gate = q_and_gate[:, 256:]
        k = rms(x[12288:13312].reshape(4, 256), wk); v = x[13312:].reshape(4, 256)
        angles = np.array([np.float32(time / 10000000 ** (frequency / 32)) for frequency in range(32)], np.float32)
        cosine, sine = bf16(np.cos(angles)), bf16(np.sin(angles))
        def rotate(y):
            y = y.copy(); low, high = y[:, :32].copy(), y[:, 32:64].copy()
            y[:, :32] = bf16(low * cosine - high * sine); y[:, 32:64] = bf16(high * cosine + low * sine)
            return y
        q, k = rotate(q), rotate(k); keys.append(k); vals.append(v)
        repeated_k = np.repeat(np.stack(keys), 6, axis=1)
        scores = np.sum(repeated_k * q[None], axis=-1, dtype=np.float32) * np.float32(.0625)
        probability = np.exp(scores - scores.max(axis=0, keepdims=True))
        probability /= probability.sum(axis=0, keepdims=True, dtype=np.float32)
        value = bf16(np.sum(np.repeat(np.stack(vals), 6, axis=1) * probability[:, :, None], axis=0, dtype=np.float32))
        result = bf16(value * bf16(np.float32(1) / (np.float32(1) + np.exp(-gate))))
        q.tofile(DIRECTORY / f'attention-query-{time}.bin'); result.tofile(DIRECTORY / f'attention-expected-{time}.bin')
    print('Produced dense, four-step DeltaNet and four-step causal GQA references')

if __name__ == '__main__': main()
