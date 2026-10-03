import { createRMSNorm, createElementwise, createDeltaNet, createFullAttention, gpuStorage } from './qwen38-ops.mjs';
import { openDenseWeights } from './qwen38-dense.mjs';
import { readGPUBuffer } from './mirai-webgpu.mjs';
import { MIRAI_REVISION } from './mirai-checkpoint.mjs';

const output = document.querySelector('#result');
function report(item) { output.textContent += (typeof item === 'string' ? item : JSON.stringify(item, null, 2)) + '\n'; }
async function fixture(name) {
  const response = await fetch(`./webgpu-fixtures/attention/${name}.bin`);
  if (!response.ok) throw new Error(`Missing reference ${name}`);
  return new Float32Array(await response.arrayBuffer());
}
async function compare(device, label, buffer, reference, {float32 = false} = {}) {
  const actual = await readGPUBuffer(device, buffer), expected = await fixture(reference);
  if (actual.length !== expected.length) throw new Error('Reference dimension mismatch');
  let maximumError = 0, mismatches = 0, exact = 0; const examples = [];
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]); maximumError = Math.max(maximumError, error);
    if (actual[i] === expected[i]) exact++;
    const ulp = Math.max(2 ** -133, 2 ** (Math.floor(Math.log2(Math.abs(expected[i]))) - 7));
    const allowance = float32 ? 5e-5 + 2e-5 * Math.abs(expected[i]) : ulp + 2e-5;
    if (!Number.isFinite(actual[i]) || error > allowance) {
      mismatches++;
      if (examples.length < 8) examples.push({index:i, expected:expected[i], actual:actual[i], error, allowance});
    }
  }
  const result = {operation:label, values:actual.length, maximumError, exact, mismatches, examples, tolerance:float32 ? '5e-5 absolute plus 2e-5 relative' : 'one BF16 ULP plus 2e-5 absolute', passed:mismatches === 0};
  report(result); return result;
}

document.querySelector('#run').addEventListener('click', async (event) => {
  event.target.disabled = true; output.textContent = '';
  let device;
  const owned = [], operators = [], results = [];
  try {
    if (!navigator.gpu) throw new Error('WebGPU unavailable');
    const adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice({requiredLimits:{maxComputeWorkgroupStorageSize:17408}});
    device.addEventListener('uncapturederror', e => report(`GPU error: ${e.error.message}`));
    const dense = await openDenseWeights();
    const allocate = data => { const buffer = gpuStorage(device, data); owned.push(buffer); return buffer; };
    const submit = async callback => { const encoder = device.createCommandEncoder(); callback(encoder); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone(); };
    const norm = await createRMSNorm(device, await dense.tensor('model.layers.0.input_layernorm.weight')); operators.push(norm);
    const input = allocate(await fixture('norm-input')), addInput = allocate(await fixture('add-input'));
    await submit(encoder => norm.encode(encoder, input)); results.push(await compare(device, 'Gemma RMS norm, original +1 weight offset', norm.output, 'norm-expected'));
    const add = await createElementwise(device, 5120, 'add'); operators.push(add);
    await submit(encoder => add.encode(encoder, input, addInput)); results.push(await compare(device, 'BF16 residual sum', add.output, 'add-expected'));
    const mlp = await createElementwise(device, 17408, 'silu-mul'); operators.push(mlp);
    const mlpInput = allocate(await fixture('mlp-input'));
    await submit(encoder => mlp.encode(encoder, mlpInput)); results.push(await compare(device, 'Fused SiLU gate times MLP up', mlp.output, 'mlp-expected'));
    const deltaWeights = Object.fromEntries(await Promise.all(Object.entries({conv:'conv1d.weight', A_log:'A_log', dt_bias:'dt_bias', norm:'norm.weight'}).map(async ([role, name]) => [role, await dense.tensor(`model.layers.0.linear_attn.${name}`)])));
    const delta = await createDeltaNet(device, deltaWeights); operators.push(delta);
    const qkvz = allocate(16384 * 4), ba = allocate(96 * 4);
    for (let time = 0; time < 4; time++) {
      device.queue.writeBuffer(qkvz, 0, await fixture(`delta-input-${time}`)); device.queue.writeBuffer(ba, 0, await fixture(`delta-ba-${time}`));
      await submit(encoder => delta.encode(encoder, qkvz, ba));
      results.push(await compare(device, `Causal convolution step ${time}`, delta.buffers.qkv, `delta-conv-${time}`));
      results.push(await compare(device, `DeltaNet QK normalization step ${time}`, delta.buffers.normalized, `delta-qk-${time}`));
      results.push(await compare(device, `DeltaNet gated output step ${time}`, delta.output, `delta-expected-${time}`));
    }
    results.push(await compare(device, 'DeltaNet state after four steps', delta.buffers.state, 'delta-state-expected', {float32:true}));
    delta.reset(); device.queue.writeBuffer(qkvz, 0, await fixture('delta-input-0')); device.queue.writeBuffer(ba, 0, await fixture('delta-ba-0'));
    await submit(encoder => delta.encode(encoder, qkvz, ba)); results.push(await compare(device, 'DeltaNet reset restores first-step result', delta.output, 'delta-expected-0'));
    const full = await createFullAttention(device, {q_norm:await dense.tensor('model.layers.3.self_attn.q_norm.weight'), k_norm:await dense.tensor('model.layers.3.self_attn.k_norm.weight')}, {maxTokens:4096}); operators.push(full);
    const qkv = allocate(14336 * 4);
    for (let time = 0; time < 4; time++) {
      device.queue.writeBuffer(qkv, 0, await fixture(`attention-input-${time}`));
      await submit(encoder => full.encode(encoder, qkv, time));
      results.push(await compare(device, `QK norm and partial RoPE step ${time}`, full.buffers.query, `attention-query-${time}`));
      results.push(await compare(device, `Causal GQA with sigmoid output gate step ${time}`, full.output, `attention-expected-${time}`));
    }
    for (const position of [255, 256, 4095]) {
      full.reset(); device.queue.writeBuffer(qkv, 0, await fixture(`boundary-input-${position}`));
      await submit(encoder => full.encode(encoder, qkv, position));
      results.push(await compare(device, `Boundary QK norm and RoPE position ${position}`, full.buffers.query, `boundary-query-${position}`));
      results.push(await compare(device, `Boundary causal GQA position ${position}`, full.output, `boundary-expected-${position}`));
    }
    const passed = results.every(result => result.passed);
    const summary = {checkedAt:new Date().toISOString(), revision:MIRAI_REVISION, results, attentionChecksPassed:passed, fullModelInferenceVerified:false};
    document.querySelector('#report-json').textContent = JSON.stringify(summary, null, 2);
    document.documentElement.dataset.attentionChecks = passed ? 'passed' : 'failed';
    report({attentionChecksPassed:passed, fullModelInferenceVerified:false});
  } catch (error) { report(`FAILED: ${error.message}`); }
  finally { operators.forEach(operator => operator.dispose()); owned.forEach(buffer => buffer.destroy()); device?.destroy(); event.target.disabled = false; }
});

document.querySelector('#tokenizer').addEventListener('click', async (event) => {
  event.target.disabled = true;
  try {
    const { openCaseTokenizer } = await import('./qwen38-tokenizer.mjs');
    const { buildCaseContext } = await import('./case-context.mjs');
    const tokenizer = await openCaseTokenizer();
    const data = await (await fetch('./sapini-clio-data.json')).json();
    const context = buildCaseContext(data);
    const messages = [context.systemMessage, {role:'user', content:'What needs attention first?'}];
    const {text, ids} = tokenizer.encodeChat(messages);
    const roundTrip = tokenizer.decode(ids);
    const passed = roundTrip === text && text.includes(context.systemMessage.content.trim()) && ids.every(id => Number.isInteger(id) && id >= 0 && id < 248320);
    report({tokenizerPassed:passed, promptTokens:ids.length, caseContextFirst:text.startsWith('<|im_start|>system\n'), roundTripExact:roundTrip === text, eos:tokenizer.eosTokenIds, fullModelInferenceVerified:false});
  } catch (error) { report(`Tokenizer FAILED: ${error.message}`); }
  finally { event.target.disabled = false; }
});
