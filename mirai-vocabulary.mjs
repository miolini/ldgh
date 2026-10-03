// Exact D4 embedding and I3 head codecs from the requested Mirai S checkpoint.
// Raw row-major safetensors layout; no repacking or weight requantization.
import { readGPUBuffer } from './mirai-webgpu.mjs';
export { readGPUBuffer };

const utility = `
fn bf16(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  return bitcast<f32>((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
}
fn fp16(value: f32) -> f32 { return unpack2x16float(pack2x16float(vec2<f32>(value, 0.0))).x; }
`;

const embedding = `
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> row_scales: array<f32>;
@group(0) @binding(2) var<storage, read> ladder: array<f32>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> table: array<i32>;
@group(0) @binding(5) var<storage, read> signs: array<f32>;
@group(0) @binding(6) var<storage, read> ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> values: array<f32, 32>;
${utility}
fn code_byte(index: u32) -> u32 { return (codes[index / 4u] >> ((index % 4u) * 8u)) & 255u; }
fn ladder_byte(index: u32) -> u32 { return (indices[index / 4u] >> ((index % 4u) * 8u)) & 255u; }
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let column = group.x * 32u + local.x;
  let row = ids[group.y];
  let code = code_byte(row * 1280u + column / 4u);
  let packed = ladder_byte(row * 40u + column / 128u);
  let rung = (packed >> (((column / 64u) % 2u) * 4u)) & 15u;
  values[local.x] = f32(table[code * 4u + column % 4u]) * (row_scales[row] * ladder[rung]);
  workgroupBarrier();
  for (var stride = 1u; stride < 32u; stride *= 2u) {
    let a = values[local.x];
    let b = values[local.x ^ stride];
    workgroupBarrier();
    if ((local.x & stride) == 0u) { values[local.x] = a + b; }
    else { values[local.x] = b - a; }
    workgroupBarrier();
  }
  output[group.y * 5120u + column] = bf16(values[local.x] / sqrt(32.0) * signs[column]);
}`;

const headInput = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> signs: array<f32>;
@group(0) @binding(2) var<storage, read_write> rotated: array<f32>;
var<workgroup> values: array<f32, 32>;
${utility}
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let column = group.x * 32u + local.x;
  values[local.x] = input[column] * signs[column];
  workgroupBarrier();
  for (var stride = 1u; stride < 32u; stride *= 2u) {
    let a = values[local.x]; let b = values[local.x ^ stride];
    workgroupBarrier();
    if ((local.x & stride) == 0u) { values[local.x] = a + b; }
    else { values[local.x] = b - a; }
    workgroupBarrier();
  }
  rotated[column] = fp16(values[local.x] / sqrt(32.0));
}`;

const head = `
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> row_scales: array<f32>;
@group(0) @binding(2) var<storage, read> ladder: array<f32>;
@group(0) @binding(3) var<storage, read> indices: array<u32>;
@group(0) @binding(4) var<storage, read> rotated: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits: array<f32>;
var<workgroup> partial: array<f32, 256>;
${utility}
fn code_byte(index: u32) -> u32 { return (codes[index / 4u] >> ((index % 4u) * 8u)) & 255u; }
fn ladder_byte(index: u32) -> u32 { return (indices[index / 4u] >> ((index % 4u) * 8u)) & 255u; }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * 65535u;
  if (row >= HEAD_ROWS) { return; }
  var sum = 0.0;
  for (var column = local.x; column < 5120u; column += 256u) {
    let bit = column * 3u;
    let index = row * 1920u + bit / 8u;
    var packed = code_byte(index);
    if (bit % 8u > 5u) { packed |= code_byte(index + 1u) << 8u; }
    let code = (packed >> (bit % 8u)) & 7u;
    let packed_ladder = ladder_byte(row * 40u + column / 128u);
    let rung = (packed_ladder >> (((column / 64u) % 2u) * 4u)) & 15u;
    let value = fp16(f32(i32(code) * 2 - 7) * ladder[rung]);
    sum += value * rotated[column];
  }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { logits[row] = row_scales[row] * partial[0]; }
}`;

function storage(device, data) {
  const size = typeof data === 'number' ? data : data.byteLength;
  const buffer = device.createBuffer({size: Math.max(4, Math.ceil(size / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
  if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31, fraction = bits & 1023;
  return sign * (exponent === 0 ? fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : Infinity) : (1 + fraction / 1024) * 2 ** (exponent - 15));
}

async function pipeline(device, label, code) {
  const module = device.createShaderModule({label, code});
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(message => message.type === 'error');
  if (errors.length) throw new Error(`${label}: ${errors.map(message => message.message).join('; ')}`);
  return device.createComputePipelineAsync({label, layout:'auto', compute:{module, entryPoint:'main'}});
}

function bind(device, compute, buffers) {
  return device.createBindGroup({layout:compute.getBindGroupLayout(0), entries:buffers.map((buffer, binding) => ({binding, resource:{buffer}}))});
}

export async function createMiraiVocabulary(device, manifest, tensors, {maxEmbeddingTokens = 1} = {}) {
  const owned = [];
  const allocate = data => { const buffer = storage(device, data); owned.push(buffer); return buffer; };
  const surface = kind => {
    const raw = role => tensors[`${kind}.${role}`];
    return {
      codes: allocate(new Uint32Array(raw('codes'))), row_scales: allocate(new Float32Array(raw('row_scales'))),
      ladder: allocate(Float32Array.from(new Uint16Array(raw('ladder')), halfToFloat)),
      indices: allocate(new Uint32Array(raw('ladder_indices'))), signs: allocate(new Float32Array(raw('signs'))),
    };
  };
  const embed = surface('embedding'), outputHead = surface('head');
  const table = allocate(Int32Array.from(new Int8Array(tensors['embedding.table'])));
  const tokenIds = allocate(maxEmbeddingTokens * 4), embeddingOutput = allocate(maxEmbeddingTokens * 5120 * 4);
  const input = allocate(5120 * 4), rotated = allocate(5120 * 4), logits = allocate(manifest.rows * 4);
  const [embedPipeline, inputPipeline, headPipeline] = await Promise.all([
    pipeline(device, 'Mirai D4 embedding', embedding), pipeline(device, 'Mirai head rotation', headInput), pipeline(device, 'Mirai I3 output head', `const HEAD_ROWS = ${manifest.rows}u;\n${head}`),
  ]);
  const embedBindings = bind(device, embedPipeline, [embed.codes, embed.row_scales, embed.ladder, embed.indices, table, embed.signs, tokenIds, embeddingOutput]);
  const inputBindings = bind(device, inputPipeline, [input, outputHead.signs, rotated]);
  const headBindings = bind(device, headPipeline, [outputHead.codes, outputHead.row_scales, outputHead.ladder, outputHead.indices, rotated, logits]);
  return {
    buffers: {embedding:embeddingOutput, rotated, logits},
    encodeEmbedding(encoder, ids) {
      if (!(ids instanceof Uint32Array) || ids.length > maxEmbeddingTokens || ids.some(id => id >= manifest.rows)) throw new Error('Embedding token IDs outside vocabulary or capacity');
      device.queue.writeBuffer(tokenIds, 0, ids);
      const pass = encoder.beginComputePass();
      pass.setPipeline(embedPipeline); pass.setBindGroup(0, embedBindings); pass.dispatchWorkgroups(160, ids.length);
      pass.end(); return embeddingOutput;
    },
    encodeHead(encoder, hidden) {
      if (hidden instanceof Float32Array) {
        if (hidden.length !== 5120) throw new Error('Output head width mismatch');
        device.queue.writeBuffer(input, 0, hidden);
      } else {
        if (hidden.size !== 5120 * 4) throw new Error('GPU output head width mismatch');
        encoder.copyBufferToBuffer(hidden, 0, input, 0, 5120 * 4);
      }
      const pass = encoder.beginComputePass();
      pass.setPipeline(inputPipeline); pass.setBindGroup(0, inputBindings); pass.dispatchWorkgroups(160);
      pass.setPipeline(headPipeline); pass.setBindGroup(0, headBindings); pass.dispatchWorkgroups(Math.min(manifest.rows, 65535), Math.ceil(manifest.rows / 65535));
      pass.end(); return logits;
    },
    dispose() { owned.forEach(buffer => buffer.destroy()); },
  };
}

export async function createVocabularyFixture(device, manifest, tensors, ids) {
  const vocabulary = await createMiraiVocabulary(device, manifest, tensors, {maxEmbeddingTokens:ids.length});
  return {
    ...vocabulary,
    run(hidden) {
      const encoder = device.createCommandEncoder();
      vocabulary.encodeEmbedding(encoder, ids); vocabulary.encodeHead(encoder, hidden);
      device.queue.submit([encoder.finish()]); return device.queue.onSubmittedWorkDone();
    },
  };
}
