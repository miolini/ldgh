import { createMiraiLayer, createMiraiProjection, readGPUBuffer } from './mirai-webgpu.mjs';
import { createVocabularyFixture } from './mirai-vocabulary.mjs';
import { openMiraiCheckpoint, MIRAI_REVISION } from './mirai-checkpoint.mjs';

const output = document.querySelector('#result');
const button = document.querySelector('#run');
function report(value) { output.textContent += `${typeof value === 'string' ? value : JSON.stringify(value, null, 2)}\n`; }
async function binary(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Fixture download failed: ${url} (${response.status})`);
  return response.arrayBuffer();
}
button.addEventListener('click', async () => {
  button.disabled = true;
  output.textContent = '';
  let device;
  try {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter.');
    device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (event) => report(`GPU error: ${event.error.message}`));
    const capabilities = { maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize, adapterMaxBufferSize:adapter.limits.maxBufferSize, adapterMaxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize, features: [...adapter.features] };
    report(capabilities);
    const formats = await (await fetch('./webgpu-fixtures/index.json')).json();
    const results = [];
    for (const format of formats) {
      const base = `./webgpu-fixtures/${format}/`;
      const fixture = await (await fetch(base + 'fixture.json')).json();
      const tensors = Object.fromEntries(await Promise.all(Object.entries(fixture.tensors).map(async ([role, item]) => [role, await binary(base + item.file)])));
      const input = new Float32Array(await binary(base + 'input.bin'));
      const expected = new Float32Array(await binary(base + 'expected.bin'));
      const expectedRotated = new Float32Array(await binary(base + 'rotated.bin'));
      const expectedPlanes = new Int32Array(await binary(base + 'planes.bin'));
      const layer = await createMiraiLayer(device, fixture, tensors);
      const before = performance.now();
      await layer.run(input);
      const milliseconds = performance.now() - before;
      const actual = await readGPUBuffer(device, layer.buffers.output);
      const rotated = await readGPUBuffer(device, layer.buffers.rotated);
      const planes = await readGPUBuffer(device, layer.buffers.planes, Uint32Array);
      let maxAbsoluteError = 0, maxRotationError = 0, bf16Mismatches = 0, planeMismatches = 0, unexplainedPlaneMismatches = 0, maxBf16Ulps = 0;
      const actualBits = new Uint32Array(actual.buffer);
      const expectedBits = new Uint32Array(expected.buffer);
      const referenceScale = Math.fround(Math.max(...expectedRotated.map(Math.abs)) / 127);
      const referenceInverse = Math.fround(1 / referenceScale);
      const gpuScale = Math.fround(Math.max(...rotated.map(Math.abs)) / 127);
      const gpuInverse = Math.fround(1 / gpuScale);
      for (let index = 0; index < expected.length; index++) {
        maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(expected[index] - actual[index]));
        if (expected[index] !== actual[index]) bf16Mismatches++;
        const ordered = (bits) => bits & 0x80000000 ? 0x80000000 - (bits & 0x7fffffff) : 0x80000000 + bits;
        maxBf16Ulps = Math.max(maxBf16Ulps, Math.abs(ordered(expectedBits[index]) - ordered(actualBits[index])) / 65536);
      }
      for (let index = 0; index < rotated.length; index++) {
        maxRotationError = Math.max(maxRotationError, Math.abs(rotated[index] - expectedRotated[index]));
        for (let p = 0; p < 2; p++) {
          const word = planes[Math.floor(index / 4) * 2 + p];
          const byte = ((word >>> ((index % 4) * 8)) << 24) >> 24;
          if (byte !== expectedPlanes[index * 2 + p]) {
            planeMismatches++;
            // A different F32 reduction may move an activation across a half-integer.
            // Accept only an adjacent quantization bin whose rounding boundary lies
            // between the reference and GPU pre-quantization values.
            let referenceValue = Math.fround(expectedRotated[index] * referenceInverse);
            let gpuValue = Math.fround(rotated[index] * gpuInverse);
            if (p === 1) {
              const coarse = expectedPlanes[index * 2];
              referenceValue = Math.fround(Math.fround(referenceValue - coarse) * 254);
              gpuValue = Math.fround(Math.fround(gpuValue - coarse) * 254);
            }
            const boundary = (byte + expectedPlanes[index * 2 + p]) / 2;
            const nearBoundary = boundary >= Math.min(referenceValue, gpuValue) - 0.00002 && boundary <= Math.max(referenceValue, gpuValue) + 0.00002;
            if (Math.abs(byte - expectedPlanes[index * 2 + p]) !== 1 || !nearBoundary) unexplainedPlaneMismatches++;
          }
        }
      }
      const passed = maxBf16Ulps <= 1 && maxRotationError < 1e-5 && unexplainedPlaneMismatches === 0;
      const result = { format, layer: fixture.layer, rows: fixture.rows, columns: fixture.columns, milliseconds, maxAbsoluteError, maxRotationError, maxBf16Ulps, bf16Mismatches, planeMismatches, unexplainedPlaneMismatches, passed };
      results.push(result);
      report(result);
      await layer.run(new Float32Array(input.length));
      const zeros = await readGPUBuffer(device, layer.buffers.output);
      if (zeros.some((value) => value !== 0)) throw new Error(`${format}: zero input produced a nonzero output`);
      layer.dispose();
    }
    const vocabularyBase = './webgpu-fixtures/vocabulary/';
    const vocabularyManifest = await (await fetch(vocabularyBase + 'fixture.json')).json();
    const vocabularyTensors = Object.fromEntries(await Promise.all(Object.entries(vocabularyManifest.tensors).map(async ([role, item]) => [role, await binary(vocabularyBase + item.file)])));
    const ids = new Uint32Array(await binary(vocabularyBase + 'ids.bin'));
    const vocabulary = await createVocabularyFixture(device, vocabularyManifest, vocabularyTensors, ids);
    const headInputValues = new Float32Array(await binary(vocabularyBase + 'head-input.bin'));
    const headInputGPU = device.createBuffer({size:headInputValues.byteLength, usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    device.queue.writeBuffer(headInputGPU, 0, headInputValues);
    const vocabularyEncoder = device.createCommandEncoder();
    vocabulary.encodeEmbedding(vocabularyEncoder, ids); vocabulary.encodeHead(vocabularyEncoder, headInputGPU);
    device.queue.submit([vocabularyEncoder.finish()]); await device.queue.onSubmittedWorkDone();
    headInputGPU.destroy();
    const comparisons = [
      ['D4 embedding', 'embedding', 'embedding-expected.bin', 0],
      ['Head input rotation', 'rotated', 'head-rotated.bin', 'one-fp16-ulp'],
      ['I3 output head', 'logits', 'head-expected.bin', 0.00001],
    ];
    for (const [name, buffer, referenceFile, tolerance] of comparisons) {
      const actual = await readGPUBuffer(device, vocabulary.buffers[buffer]);
      const expected = new Float32Array(await binary(vocabularyBase + referenceFile));
      let maximumError = 0, mismatches = 0;
      for (let index = 0; index < expected.length; index++) {
        maximumError = Math.max(maximumError, Math.abs(actual[index] - expected[index]));
        const allowance = tolerance === 'one-fp16-ulp'
          ? Math.max(2 ** -24, 2 ** (Math.floor(Math.log2(Math.abs(expected[index]))) - 10))
          : tolerance * Math.max(1, Math.abs(expected[index]));
        if (!Number.isFinite(actual[index]) || Math.abs(actual[index] - expected[index]) > allowance) mismatches++;
      }
      const result = {operation: name, values: expected.length, maximumError, mismatches, tolerance, passed:mismatches === 0};
      report(result); results.push(result);
    }
    vocabulary.dispose();
    const projectionBase = './webgpu-fixtures/projection/';
    const projectionManifest = await (await fetch(projectionBase + 'fixture.json')).json();
    const projectionTensors = Object.fromEntries(await Promise.all(Object.entries(projectionManifest.tensors).map(async ([role, item]) => [role, await binary(projectionBase + item.file)])));
    const projection = await createMiraiProjection(device, projectionManifest.metadata, projectionTensors);
    const projectionStart = performance.now();
    const projectionInputValues = new Float32Array(await binary(projectionBase + 'input.bin'));
    const projectionInputGPU = device.createBuffer({size:projectionInputValues.byteLength, usage:GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    device.queue.writeBuffer(projectionInputGPU, 0, projectionInputValues);
    await projection.run(projectionInputGPU); projectionInputGPU.destroy();
    const projectionMilliseconds = performance.now() - projectionStart;
    const projectionActual = await readGPUBuffer(device, projection.output);
    const projectionExpected = new Float32Array(await binary(projectionBase + 'expected.bin'));
    let projectionError = 0, projectionMismatches = 0;
    for (let index = 0; index < projectionExpected.length; index++) {
      const error = Math.abs(projectionActual[index] - projectionExpected[index]);
      projectionError = Math.max(projectionError, error);
      const bf16Ulp = Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(projectionExpected[index]))) - 7));
      if (!Number.isFinite(projectionActual[index]) || error > bf16Ulp + 0.00002) projectionMismatches++;
    }
    const projectionResult = {operation:'Complete permuted QKV projection', layer:projectionManifest.layer, values:projectionExpected.length, blocks:projectionManifest.metadata.blocks.length, inputTransport:'GPU buffer copy', milliseconds:projectionMilliseconds, maximumError:projectionError, mismatches:projectionMismatches, tolerance:'one BF16 ULP plus 2e-5 absolute for activation rounding boundaries', passed:projectionMismatches === 0};
    report(projectionResult); results.push(projectionResult);
    projection.dispose();
    const passed = results.every((result) => result.passed);
    report({ operatorChecksPassed: passed, fullModelInferenceVerified: false });
    document.querySelector('#report-json').textContent = JSON.stringify({checkedAt:new Date().toISOString(), revision:MIRAI_REVISION, capabilities, results, operatorChecksPassed:passed, fullModelInferenceVerified:false}, null, 2);
    document.documentElement.dataset.operatorChecks = passed ? 'passed' : 'failed';
  } catch (error) {
    report(`FAILED: ${error.message}`);
    document.documentElement.dataset.operatorChecks = 'failed';
  } finally {
    device?.destroy();
    button.disabled = false;
  }
});

document.querySelector('#access').addEventListener('click', async (event) => {
  event.target.disabled = true;
  try {
    const checkpoint = await openMiraiCheckpoint();
    const data = await checkpoint.tensor('embedding.table');
    report({checkpointAccess:'passed', revision:checkpoint.revision, projections:Object.keys(checkpoint.layers).length, embeddingTableBytes:data.byteLength, scope:'Pinned header and one 1 KB tensor; full weights were not downloaded.'});
  } catch (error) { report(`Checkpoint access FAILED: ${error.message}`); }
  finally { event.target.disabled = false; }
});
