// WebGPU port of the Mirai S tape decoder and two-plane GEMV described in the
// checkpoint's mirai_s/kernels.cu. The compressed checkpoint is preserved.
// This module implements a layer operator, not complete model generation.

const FORMATS = {
  v4t8: { vector: 4, bits: 8, steps: 16, words: 1 },
  v2t4: { vector: 2, bits: 4, steps: 32, words: 1 },
  v2t6: { vector: 2, bits: 6, steps: 64, words: 3 },
};

function rotationShader(columns, order) {
  const width = columns / order;
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> signs: array<f32>;
@group(0) @binding(2) var<storage, read> small_q: array<f32>;
@group(0) @binding(3) var<storage, read_write> rotated: array<f32>;
@group(0) @binding(4) var<storage, read_write> column_max: array<f32>;
var<workgroup> column: array<f32, ${width}>;
var<workgroup> maxima: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let tid = local.x;
  let out = group.x;
  for (var w = tid; w < ${width}u; w += 256u) {
    var value = 0.0;
    for (var c = 0u; c < ${order}u; c += 1u) {
      let index = w * ${order}u + c;
      value += input[index] * signs[index] * small_q[out * ${order}u + c];
    }
    column[w] = value;
  }
  workgroupBarrier();
  for (var stride = 1u; stride < ${width}u; stride *= 2u) {
    for (var pair = tid; pair < ${width / 2}u; pair += 256u) {
      let low = (pair / stride) * 2u * stride + pair % stride;
      let high = low + stride;
      let a = column[low];
      let b = column[high];
      column[low] = a + b;
      column[high] = a - b;
    }
    workgroupBarrier();
  }
  var maximum = 0.0;
  for (var w = tid; w < ${width}u; w += 256u) {
    let value = column[w] * inverseSqrt(${width}.0);
    rotated[w * ${order}u + out] = value;
    maximum = max(maximum, abs(value));
  }
  maxima[tid] = maximum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (tid < stride) { maxima[tid] = max(maxima[tid], maxima[tid + stride]); }
    workgroupBarrier();
  }
  if (tid == 0u) { column_max[out] = maxima[0]; }
}`;
}

function quantizationShader(columns, order) {
  return `
@group(0) @binding(0) var<storage, read> rotated: array<f32>;
@group(0) @binding(1) var<storage, read> column_max: array<f32>;
@group(0) @binding(2) var<storage, read_write> planes: array<u32>;
@group(0) @binding(3) var<storage, read_write> stats: array<f32>;
var<workgroup> sums: array<i32, 2048>;
fn round_even(value: f32) -> i32 {
  let low = floor(value);
  let fraction = value - low;
  if (fraction > 0.5 || (fraction == 0.5 && (i32(low) & 1) != 0)) { return i32(low) + 1; }
  return i32(low);
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let tid = local.x;
  var maximum = 0.0;
  for (var c = 0u; c < ${order}u; c += 1u) { maximum = max(maximum, column_max[c]); }
  var scale = 1.0;
  if (maximum > 0.0) { scale = maximum / 127.0; }
  let inverse = 1.0 / scale;
  for (var r = 0u; r < 8u; r += 1u) { sums[tid * 8u + r] = 0; }
  for (var g = tid; g < ${columns / 4}u; g += 256u) {
    var plane0 = 0u;
    var plane1 = 0u;
    for (var r = 0u; r < 4u; r += 1u) {
      let scaled = rotated[g * 4u + r] * inverse;
      let coarse = clamp(round_even(scaled), -127, 127);
      let fine = clamp(round_even((scaled - f32(coarse)) * 254.0), -127, 127);
      plane0 |= (bitcast<u32>(coarse) & 255u) << (8u * r);
      plane1 |= (bitcast<u32>(fine) & 255u) << (8u * r);
      sums[tid * 8u + r] += coarse;
      sums[tid * 8u + r + 4u] += fine;
    }
    planes[g * 2u] = plane0;
    planes[g * 2u + 1u] = plane1;
  }
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (tid < stride) {
      for (var r = 0u; r < 8u; r += 1u) { sums[tid * 8u + r] += sums[(tid + stride) * 8u + r]; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    stats[0] = scale;
    stats[1] = f32(sums[0] + sums[1] + sums[2] + sums[3]);
    stats[2] = f32(sums[4] + sums[5] + sums[6] + sums[7]);
    for (var r = 0u; r < 4u; r += 1u) { stats[3u + r] = scale * (f32(sums[r]) + f32(sums[4u + r]) / 254.0); }
    stats[7] = 0.0;
  }
}`;
}

function matvecShader(columns, format) {
  const f = FORMATS[format];
  const packetCount = columns / (f.vector * f.steps);
  return `
@group(0) @binding(0) var<storage, read> packets: array<u32>;
@group(0) @binding(1) var<storage, read> entries: array<u32>;
@group(0) @binding(2) var<storage, read> rowscale: array<f32>;
@group(0) @binding(3) var<storage, read> planes: array<u32>;
@group(0) @binding(4) var<storage, read> stats: array<f32>;
@group(0) @binding(5) var<storage, read> codebook: array<f32>;
@group(0) @binding(6) var<storage, read> row_map: array<i32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> partial0: array<i32, 128>;
var<workgroup> partial1: array<i32, 128>;
fn packet_word(slot: u32, lane: u32, index: u32) -> u32 {
  return packets[((slot * ${f.words}u + index / 4u) * 32u + lane) * 4u + index % 4u];
}
fn hash_state(state: u32) -> u32 {
  var hashed = state * 0xcfccb83fu + 0x584b4aa3u;
  hashed ^= hashed >> 16u;
  hashed *= 0x85ebca6bu;
  return hashed ^ (hashed >> 16u);
}
fn signed_byte(word: u32, index: u32) -> i32 {
  return bitcast<i32>((word >> (index * 8u)) << 24u) >> 24u;
}
fn rounded_bf16(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  return bitcast<f32>((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
}
@compute @workgroup_size(32, 4)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let lane = local.x;
  let slice = local.y;
  let row = group.x * 32u + lane;
  var acc0 = 0;
  var acc1 = 0;
  for (var packet = slice; packet < ${packetCount}u; packet += 4u) {
    let slot = group.x * ${packetCount}u + packet;
    var state = entries[slot * 32u + lane];
    for (var step = 0u; step < ${f.steps}u; step += 1u) {
      let bit = step * ${f.bits}u;
      let word = bit / 32u;
      let shift = bit % 32u;
      var symbol = packet_word(slot, lane, word) >> shift;
      if (shift + ${f.bits}u > 32u) { symbol |= packet_word(slot, lane, word + 1u) << (32u - shift); }
      state = ((state << ${f.bits}u) | (symbol & ${(1 << f.bits) - 1}u)) & 65535u;
      let hashed = hash_state(state);
      for (var position = 0u; position < ${f.vector}u; position += 1u) {
        let byte = (hashed >> (position * 8u)) & 255u;
        let pairs = (byte & 3u) + ((byte >> 2u) & 3u) + ((byte >> 4u) & 3u) + ((byte >> 6u) & 3u);
        let level_plus_54 = i32(8u * pairs + ((3u * (byte & 15u)) & 15u));
        let column = packet * ${f.steps * f.vector}u + step * ${f.vector}u + position;
        acc0 += level_plus_54 * signed_byte(planes[(column / 4u) * 2u], column % 4u);
        acc1 += level_plus_54 * signed_byte(planes[(column / 4u) * 2u + 1u], column % 4u);
      }
    }
  }
  partial0[slice * 32u + lane] = acc0;
  partial1[slice * 32u + lane] = acc1;
  workgroupBarrier();
  if (slice == 0u) {
    for (var s = 1u; s < 4u; s += 1u) { acc0 += partial0[s * 32u + lane]; acc1 += partial1[s * 32u + lane]; }
    acc0 -= 54 * i32(stats[1]);
    acc1 -= 54 * i32(stats[2]);
    var offset = codebook[1] * stats[3];
    for (var r = 1u; r < 4u; r += 1u) { offset += codebook[r + 1u] * stats[r + 3u]; }
    let dot = codebook[0] * stats[0] * (f32(acc0) + f32(acc1) * (1.0 / 254.0));
    let destination = row_map[row];
    if (destination >= 0) { output[u32(destination)] = rounded_bf16(rowscale[row] * (dot + offset)); }
  }
}`;
}

function storage(device, data) {
  const size = typeof data === 'number' ? data : data.byteLength;
  const buffer = device.createBuffer({ size: Math.max(4, Math.ceil(size / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

const pipelineCaches = new WeakMap();
async function pipeline(device, label, code) {
  if (!pipelineCaches.has(device)) pipelineCaches.set(device, new Map());
  const cache = pipelineCaches.get(device);
  if (cache.has(code)) return cache.get(code);
  const pending = compilePipeline(device, label, code);
  cache.set(code, pending);
  try { return await pending; } catch (error) { cache.delete(code); throw error; }
}

async function compilePipeline(device, label, code) {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length) throw new Error(`${label}: ${errors.map((message) => message.message).join('; ')}`);
  return device.createComputePipelineAsync({ label, layout: 'auto', compute: { module, entryPoint: 'main' } });
}

function bindings(device, computePipeline, buffers) {
  return device.createBindGroup({ layout: computePipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
}

export async function readGPUBuffer(device, source, ArrayType = Float32Array) {
  const readback = device.createBuffer({ size: source.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, source.size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const output = new ArrayType(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return output;
}

export async function createMiraiLayer(device, fixture, tensors, sharedOutput = null) {
  const { columns, rows, rotation_order: order, format } = fixture;
  if (!FORMATS[format]) throw new Error(`Unsupported Mirai S format ${format}`);
  if (rows % 32 || columns % (FORMATS[format].steps * FORMATS[format].vector)) throw new Error('Invalid Mirai tape dimensions');
  const width = columns / order;
  if (!Number.isInteger(width) || (width & (width - 1))) throw new Error('The rotation width must be a power of two');
  const [rotatePipeline, quantizePipeline, dotPipeline] = await Promise.all([
    pipeline(device, `Mirai rotation ${columns}`, rotationShader(columns, order)),
    pipeline(device, `Mirai input quantization ${columns}`, quantizationShader(columns, order)),
    pipeline(device, `Mirai GEMV ${format}`, matvecShader(columns, format)),
  ]);
  const entries = Uint32Array.from(format === 'v4t8' ? new Uint8Array(tensors.entries) : new Uint16Array(tensors.entries));
  const buffers = {
    input: storage(device, columns * 4), signs: storage(device, new Float32Array(tensors.signs)), small_q: storage(device, new Float32Array(tensors.small_q)),
    rotated: storage(device, columns * 4), maxima: storage(device, order * 4), planes: storage(device, columns * 2), stats: storage(device, 8 * 4),
    packets: storage(device, new Uint32Array(tensors.packets)), entries: storage(device, entries), rowscale: storage(device, new Float32Array(tensors.rowscale)),
    codebook: storage(device, new Float32Array(tensors.codebook)), row_map: storage(device, tensors.row_map ? new Int32Array(tensors.row_map) : Int32Array.from({ length: rows }, (_, index) => index)), output: sharedOutput ?? storage(device, (fixture.out_features ?? rows) * 4),
  };
  const rotateBindings = bindings(device, rotatePipeline, [buffers.input, buffers.signs, buffers.small_q, buffers.rotated, buffers.maxima]);
  const quantizeBindings = bindings(device, quantizePipeline, [buffers.rotated, buffers.maxima, buffers.planes, buffers.stats]);
  const dotBindings = bindings(device, dotPipeline, [buffers.packets, buffers.entries, buffers.rowscale, buffers.planes, buffers.stats, buffers.codebook, buffers.row_map, buffers.output]);
  function encode(encoder, input) {
      if (input instanceof Float32Array) {
        if (input.length !== columns) throw new Error('Input width mismatch');
        device.queue.writeBuffer(buffers.input, 0, input);
      } else {
        if (input.size !== columns * 4) throw new Error('GPU input width mismatch');
        if (input !== buffers.input) encoder.copyBufferToBuffer(input, 0, buffers.input, 0, columns * 4);
      }
      const pass = encoder.beginComputePass();
      pass.setPipeline(rotatePipeline); pass.setBindGroup(0, rotateBindings); pass.dispatchWorkgroups(order);
      pass.setPipeline(quantizePipeline); pass.setBindGroup(0, quantizeBindings); pass.dispatchWorkgroups(1);
      pass.setPipeline(dotPipeline); pass.setBindGroup(0, dotBindings); pass.dispatchWorkgroups(rows / 32);
      pass.end();
      return buffers.output;
  }
  return {
    buffers, encode,
    run(input) {
      const encoder = device.createCommandEncoder();
      encode(encoder, input);
      device.queue.submit([encoder.finish()]);
      return device.queue.onSubmittedWorkDone();
    },
    dispose() { Object.values(buffers).forEach((buffer) => { if (buffer !== sharedOutput) buffer.destroy(); }); },
  };
}

// A production projection can contain different tape formats and padded,
// permuted rows. The permutation maps output rows to internal rows.
export async function createMiraiProjection(device, metadata, tensorData) {
  const rowCounts = metadata.blocks.map((_, index) => new Float32Array(tensorData[`${index}.rowscale`]).length);
  const rowMap = new Int32Array(rowCounts.reduce((sum, count) => sum + count, 0)).fill(-1);
  if (metadata.has_perm && !tensorData.perm) throw new Error('Projection permutation missing');
  const permutation = metadata.has_perm ? new Int32Array(tensorData.perm) : null;
  if (permutation && permutation.length !== metadata.out_features) throw new Error('Projection permutation length mismatch');
  for (let outputRow = 0; outputRow < metadata.out_features; outputRow++) {
    const internalRow = permutation ? permutation[outputRow] : outputRow;
    if (internalRow < 0 || internalRow >= rowMap.length || rowMap[internalRow] !== -1) throw new Error('Invalid projection permutation');
    rowMap[internalRow] = outputRow;
  }
  const output = storage(device, metadata.out_features * 4), operators = [];
  try {
    for (let index = 0; index < metadata.blocks.length; index++) {
      const block = metadata.blocks[index], columns = metadata.in_features;
      const small_q = tensorData.small_q;
      const order = Math.sqrt(new Float32Array(small_q).length);
      const tensors = {
        packets:tensorData[`${index}.packets`], entries:tensorData[`${index}.entries`], rowscale:tensorData[`${index}.rowscale`],
        codebook:tensorData[`codebook.${block.format.slice(0, 2)}`], signs:tensorData.signs, small_q,
        row_map:rowMap.slice(block.offset, block.offset + rowCounts[index]).buffer,
      };
      operators.push(await createMiraiLayer(device, {columns, rows:rowCounts[index], rotation_order:order, format:block.format, out_features:metadata.out_features}, tensors, output));
    }
  } catch (error) {
    operators.forEach(operator => operator.dispose()); output.destroy(); throw error;
  }
  return {
    output, blocks:operators,
    encode(encoder, input) { for (const operator of operators) operator.encode(encoder, input); return output; },
    async run(input) {
      const encoder = device.createCommandEncoder();
      for (const operator of operators) operator.encode(encoder, input);
      device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); return output;
    },
    dispose() { operators.forEach(operator => operator.dispose()); output.destroy(); },
  };
}
