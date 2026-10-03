// Text-only Qwen3.8 dense and attention operators for the exact Mirai checkpoint.
// F32 GPU storage represents BF16 values at the native model's rounding points.
export const roundingWGSL = `
fn bf16(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  return bitcast<f32>((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
}
fn sigmoid(value: f32) -> f32 { return 1.0 / (1.0 + exp(-value)); }
fn silu(value: f32) -> f32 { return value * sigmoid(value); }
`;

export function gpuStorage(device, data) {
  const size = typeof data === 'number' ? data : data.byteLength;
  const buffer = device.createBuffer({size:Math.max(4, Math.ceil(size / 4) * 4), usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
  if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}
const caches = new WeakMap();
export async function gpuPipeline(device, label, code) {
  if (!caches.has(device)) caches.set(device, new Map());
  const cache = caches.get(device);
  if (cache.has(code)) return cache.get(code);
  const pending = (async () => {
    const module = device.createShaderModule({label, code});
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(`${label}: ${errors.map(message => message.message).join('; ')}`);
    return device.createComputePipelineAsync({label, layout:'auto', compute:{module, entryPoint:'main'}});
  })();
  cache.set(code, pending);
  try { return await pending; } catch (error) { cache.delete(code); throw error; }
}
export function gpuBindings(device, pipeline, buffers) {
  return device.createBindGroup({layout:pipeline.getBindGroupLayout(0), entries:buffers.map((buffer, binding) => ({binding, resource:{buffer}}))});
}
export function gpuDispatch(encoder, pipeline, bindings, x, y = 1) {
  const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(x, y); pass.end();
}

export async function createRMSNorm(device, weights, {groups = 1, offset = 1, gated = false, gateOffset = 0, epsilon = 1e-6} = {}) {
  const width = weights.length;
  const output = gpuStorage(device, width * groups * 4), weightBuffer = gpuStorage(device, weights);
  const code = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${gated ? '@group(0) @binding(3) var<storage, read> gate: array<f32>;' : ''}
var<workgroup> sums: array<f32, 256>;
${roundingWGSL}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  var sum = 0.0;
  for (var column = local.x; column < ${width}u; column += 256u) { let x = input[group.x * ${width}u + column]; sum += x * x; }
  sums[local.x] = sum; workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { sums[local.x] += sums[local.x + stride]; } workgroupBarrier();
  }
  let inverse = inverseSqrt(sums[0] / ${width}.0 + ${epsilon});
  for (var column = local.x; column < ${width}u; column += 256u) {
    let index = group.x * ${width}u + column;
    var value = input[index] * inverse * (weights[column] + ${offset}.0);
    ${gated ? `value *= silu(gate[${gateOffset}u + index]);` : ''}
    output[index] = bf16(value);
  }
}`;
  const pipeline = await gpuPipeline(device, 'Qwen RMS norm', code);
  return {
    output,
    encode(encoder, input, gate) { gpuDispatch(encoder, pipeline, gpuBindings(device, pipeline, [input, weightBuffer, output, ...(gated ? [gate] : [])]), groups); return output; },
    dispose() { output.destroy(); weightBuffer.destroy(); },
  };
}

export async function createElementwise(device, width, mode) {
  if (!['add', 'silu-mul'].includes(mode)) throw new Error('Unknown elementwise operation');
  const output = gpuStorage(device, width * 4);
  const code = `
@group(0) @binding(0) var<storage, read> first: array<f32>;
${mode === 'add' ? '@group(0) @binding(1) var<storage, read> second: array<f32>;' : ''}
@group(0) @binding(${mode === 'add' ? 2 : 1}) var<storage, read_write> output: array<f32>;
${roundingWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let i = global.x;
  if (i < ${width}u) { output[i] = bf16(${mode === 'add' ? 'first[i] + second[i]' : `silu(first[i]) * first[${width}u + i]`}); }
}`;
  const pipeline = await gpuPipeline(device, `Qwen ${mode}`, code);
  return {
    output,
    encode(encoder, first, second) { gpuDispatch(encoder, pipeline, gpuBindings(device, pipeline, [first, ...(mode === 'add' ? [second] : []), output]), Math.ceil(width / 256)); return output; },
    dispose() { output.destroy(); },
  };
}

export async function createDeltaNet(device, weights) {
  const owned = [];
  const allocate = data => { const buffer = gpuStorage(device, data); owned.push(buffer); return buffer; };
  const convolution = allocate(weights.conv), decay = allocate(Float32Array.from([...weights.A_log, ...weights.dt_bias]));
  const history = allocate(10240 * 3 * 4), qkv = allocate(10240 * 4), normalized = allocate(4096 * 4);
  const state = allocate(48 * 128 * 128 * 4), recurrent = allocate(6144 * 4);
  const convCode = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> history: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
${roundingWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  let column = global.x;
  if (column >= 10240u) { return; }
  var value = input[column] * weights[column * 4u + 3u];
  for (var time = 0u; time < 3u; time += 1u) { value += history[time * 10240u + column] * weights[column * 4u + time]; }
  history[column] = history[10240u + column]; history[10240u + column] = history[20480u + column]; history[20480u + column] = input[column];
  output[column] = bf16(silu(value));
}`;
  const l2Code = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
var<workgroup> squares: array<f32, 128>;
${roundingWGSL}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let index = group.x * 128u + local.x; let value = input[index];
  squares[local.x] = value * value; workgroupBarrier();
  for (var stride = 64u; stride > 0u; stride /= 2u) { if (local.x < stride) { squares[local.x] += squares[local.x + stride]; } workgroupBarrier(); }
  output[index] = bf16(value * inverseSqrt(squares[0] + 0.000001));
}`;
  const recurrenceCode = `
@group(0) @binding(0) var<storage, read> qk: array<f32>;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> ba: array<f32>;
@group(0) @binding(3) var<storage, read> parameters: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
${roundingWGSL}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let head = group.x; let column = local.x; let key_head = head / 3u;
  let alpha = ba[48u + head] + parameters[48u + head];
  var softplus = alpha; if (alpha <= 20.0) { softplus = log(1.0 + exp(alpha)); }
  let decay = exp(-exp(parameters[head]) * softplus); let beta = sigmoid(ba[head]);
  var prediction = 0.0;
  for (var row = 0u; row < 128u; row += 1u) {
    let index = head * 16384u + row * 128u + column;
    state[index] *= decay; prediction += state[index] * qk[2048u + key_head * 128u + row];
  }
  let update = beta * (qkv[4096u + head * 128u + column] - prediction);
  var result = 0.0;
  for (var row = 0u; row < 128u; row += 1u) {
    let index = head * 16384u + row * 128u + column;
    state[index] += qk[2048u + key_head * 128u + row] * update;
    result += state[index] * qk[key_head * 128u + row];
  }
  output[head * 128u + column] = bf16(result * inverseSqrt(128.0));
}`;
  const [convPipeline, l2Pipeline, recurrencePipeline] = await Promise.all([
    gpuPipeline(device, 'Qwen causal convolution', convCode), gpuPipeline(device, 'Qwen QK L2 norm', l2Code), gpuPipeline(device, 'Qwen DeltaNet recurrence', recurrenceCode),
  ]);
  const norm = await createRMSNorm(device, weights.norm, {groups:48, offset:0, gated:true, gateOffset:10240});
  return {
    output:norm.output, buffers:{history, qkv, normalized, state, recurrent},
    encode(encoder, inputQkvz, inputBa) {
      gpuDispatch(encoder, convPipeline, gpuBindings(device, convPipeline, [inputQkvz, convolution, history, qkv]), 40);
      gpuDispatch(encoder, l2Pipeline, gpuBindings(device, l2Pipeline, [qkv, normalized]), 32);
      gpuDispatch(encoder, recurrencePipeline, gpuBindings(device, recurrencePipeline, [normalized, qkv, inputBa, decay, state, recurrent]), 48);
      return norm.encode(encoder, recurrent, inputQkvz);
    },
    reset() { device.queue.writeBuffer(history, 0, new Float32Array(10240 * 3)); device.queue.writeBuffer(state, 0, new Float32Array(48 * 128 * 128)); },
    dispose() { owned.forEach(buffer => buffer.destroy()); norm.dispose(); },
  };
}

export async function createFullAttention(device, weights, {maxTokens = 2048} = {}) {
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 4096) throw new Error('Attention context capacity must be between 1 and 4096 tokens');
  const owned = [], allocate = data => { const buffer = gpuStorage(device, data); owned.push(buffer); return buffer; };
  const normWeights = allocate(Float32Array.from([...weights.q_norm, ...weights.k_norm]));
  const query = allocate(6144 * 4), gate = allocate(6144 * 4), keyCache = allocate(maxTokens * 1024 * 4), valueCache = allocate(maxTokens * 1024 * 4);
  const parameters = allocate(new Uint32Array([0])), cosSin = allocate(64 * 4), output = allocate(6144 * 4);
  // The projected gate is BF16. A correctly rounded lookup avoids device exp()
  // approximations crossing a BF16 halfway point before the output multiply.
  // Every BF16 input is covered; no checkpoint-specific values are hardcoded.
  const sigmoidTable = new Float32Array(65536), sigmoidBits = new Uint32Array(sigmoidTable.buffer);
  const gateValue = new Float32Array(1), gateBits = new Uint32Array(gateValue.buffer);
  for (let index = 0; index < 65536; index++) {
    gateBits[0] = index << 16;
    sigmoidTable[index] = 1 / (1 + Math.exp(-gateValue[0]));
    sigmoidBits[index] = (sigmoidBits[index] + 0x7fff + ((sigmoidBits[index] >>> 16) & 1)) & 0xffff0000;
  }
  const roundedSigmoid = allocate(sigmoidTable);
  const prepareCode = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> query: array<f32>;
@group(0) @binding(3) var<storage, read_write> keys: array<f32>;
@group(0) @binding(4) var<storage, read_write> values: array<f32>;
@group(0) @binding(5) var<storage, read_write> gates: array<f32>;
@group(0) @binding(6) var<storage, read> parameters: array<u32>;
@group(0) @binding(7) var<storage, read> rotation: array<f32>;
var<workgroup> normed: array<f32, 256>;
var<workgroup> squares: array<f32, 256>;
${roundingWGSL}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let column = local.x; let is_query = group.x < 24u;
  var first = group.x * 512u; var weight_offset = 0u;
  if (!is_query) { first = 12288u + (group.x - 24u) * 256u; weight_offset = 256u; }
  let value = input[first + column]; squares[column] = value * value; workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) { if (column < stride) { squares[column] += squares[column + stride]; } workgroupBarrier(); }
  normed[column] = bf16(value * inverseSqrt(squares[0] / 256.0 + 0.000001) * (weights[weight_offset + column] + 1.0));
  workgroupBarrier();
  var rotated = normed[column];
  if (column < 64u) {
    let frequency = column % 32u;
    if (column < 32u) { rotated = bf16(normed[column] * rotation[frequency] - normed[column + 32u] * rotation[32u + frequency]); }
    else { rotated = bf16(normed[column] * rotation[frequency] + normed[column - 32u] * rotation[32u + frequency]); }
  }
  if (is_query) { query[group.x * 256u + column] = rotated; gates[group.x * 256u + column] = input[first + 256u + column]; }
  else {
    let index = parameters[0] * 1024u + (group.x - 24u) * 256u + column;
    keys[index] = rotated; values[index] = input[13312u + (group.x - 24u) * 256u + column];
  }
}`;
  const attentionCode = `
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> gates: array<f32>;
@group(0) @binding(4) var<storage, read> parameters: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read> rounded_sigmoid: array<f32>;
var<workgroup> scores: array<f32, ${maxTokens}>;
var<workgroup> reduction: array<f32, 256>;
${roundingWGSL}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let head = group.x; let key_head = head / 6u; let count = parameters[0] + 1u;
  var maximum = -3.402823e38;
  for (var token = local.x; token < count; token += 256u) {
    var dot = 0.0;
    for (var column = 0u; column < 256u; column += 1u) { dot += query[head * 256u + column] * keys[token * 1024u + key_head * 256u + column]; }
    let score = dot * 0.0625; scores[token] = score; maximum = max(maximum, score);
  }
  reduction[local.x] = maximum; workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) { if (local.x < stride) { reduction[local.x] = max(reduction[local.x], reduction[local.x + stride]); } workgroupBarrier(); }
  maximum = reduction[0]; workgroupBarrier();
  var total = 0.0;
  for (var token = local.x; token < count; token += 256u) { let weight = exp(scores[token] - maximum); scores[token] = weight; total += weight; }
  reduction[local.x] = total; workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) { if (local.x < stride) { reduction[local.x] += reduction[local.x + stride]; } workgroupBarrier(); }
  var value = 0.0;
  for (var token = 0u; token < count; token += 1u) { value += scores[token] / reduction[0] * values[token * 1024u + key_head * 256u + local.x]; }
  let index = head * 256u + local.x;
  output[index] = bf16(bf16(value) * rounded_sigmoid[bitcast<u32>(gates[index]) >> 16u]);
}`;
  const [prepare, attend] = await Promise.all([gpuPipeline(device, 'Qwen QK norm and rotary cache', prepareCode), gpuPipeline(device, 'Qwen causal GQA attention', attentionCode)]);
  return {
    output, buffers:{query, gate, keyCache, valueCache}, maxTokens,
    encode(encoder, input, position) {
      if (!Number.isInteger(position) || position < 0 || position >= maxTokens) throw new Error('Attention context capacity exceeded');
      device.queue.writeBuffer(parameters, 0, new Uint32Array([position]));
      const rotation = new Float32Array(64);
      const bits = new Uint32Array(rotation.buffer);
      for (let frequency = 0; frequency < 32; frequency++) {
        const angle = Math.fround(position / 10000000 ** (frequency / 32));
        rotation[frequency] = Math.cos(angle); rotation[32 + frequency] = Math.sin(angle);
      }
      for (let index = 0; index < 64; index++) bits[index] = (bits[index] + 0x7fff + ((bits[index] >>> 16) & 1)) & 0xffff0000;
      device.queue.writeBuffer(cosSin, 0, rotation);
      gpuDispatch(encoder, prepare, gpuBindings(device, prepare, [input, normWeights, query, keyCache, valueCache, gate, parameters, cosSin]), 28);
      gpuDispatch(encoder, attend, gpuBindings(device, attend, [query, keyCache, valueCache, gate, parameters, output, roundedSigmoid]), 24);
      return output;
    },
    reset() { device.queue.writeBuffer(keyCache, 0, new Float32Array(maxTokens * 1024)); device.queue.writeBuffer(valueCache, 0, new Float32Array(maxTokens * 1024)); },
    dispose() { owned.forEach(buffer => buffer.destroy()); },
  };
}
