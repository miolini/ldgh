// Full checkpoint orchestration. HTTP only transports weights; computation stays on WebGPU.
import {openMiraiCheckpoint} from './mirai-checkpoint.mjs';
import {openDenseWeights} from './qwen38-dense.mjs';
import {createMiraiProjection, readGPUBuffer} from './mirai-webgpu.mjs';
import {createMiraiVocabulary} from './mirai-vocabulary.mjs';
import {createRMSNorm, createElementwise, createDeltaNet, createFullAttention} from './qwen38-ops.mjs';
import {openCaseTokenizer} from './qwen38-tokenizer.mjs';

export async function createQwen38Runtime({maxTokens = 4096, onProgress = () => {}} = {}) {
  const cache = await fetch('/api/model-cache/status').then(r => r.json());
  if (!cache.sha256Verified) throw new Error('Finish downloading and verifying the local model files before loading the complete model.');
  if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.limits.maxStorageBufferBindingSize < 512 * 1024 * 1024) throw new Error('The GPU cannot bind the complete compressed vocabulary.');
  const workgroupBytes = Math.max(16384, maxTokens * 4 + 1024);
  if (adapter.limits.maxComputeWorkgroupStorageSize < workgroupBytes) throw new Error('GPU workgroup memory is too small for the requested attention context.');
  const device = await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:512 * 1024 * 1024, maxBufferSize:512 * 1024 * 1024, maxComputeWorkgroupStorageSize:workgroupBytes}});
  let deviceFailure;
  device.lost.then(info => {deviceFailure = new Error(`GPU device lost: ${info.message}`);});
  const owned = [], layers = [];
  const own = operator => {owned.push(operator); return operator;};
  const assertDevice = () => {if (deviceFailure) throw deviceFailure;};
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  try {
    const checkpoint = await openMiraiCheckpoint(), dense = await openDenseWeights(), tokenizer = await openCaseTokenizer();
    async function checkAllocation() {
      await device.queue.onSubmittedWorkDone();
      const validation = await device.popErrorScope(), memory = await device.popErrorScope();
      if (validation || memory) throw new Error((validation || memory).message);
      assertDevice(); device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
    }
    const tensors = {};
    for (const name of ['embedding.codes', 'embedding.row_scales', 'embedding.ladder', 'embedding.ladder_indices', 'embedding.signs', 'embedding.table', 'head.codes', 'head.row_scales', 'head.ladder', 'head.ladder_indices', 'head.signs']) {
      onProgress({progress:0, text:`Loading ${name} from this Mac…`});
      tensors[name] = await checkpoint.tensor(name);
      assertDevice();
    }
    const rows = checkpoint.header['embedding.row_scales'].shape[0];
    const vocabulary = own(await createMiraiVocabulary(device, {rows}, tensors));
    for (const name of Object.keys(tensors)) delete tensors[name];
    await checkAllocation();
    const projection = async name => {
      const data = await checkpoint.projectionTensors(name);
      const result = own(await createMiraiProjection(device, checkpoint.layers[name], data));
      await device.queue.onSubmittedWorkDone(); assertDevice();
      return result;
    };
    for (let index = 0; index < 64; index++) {
      onProgress({progress:(index + 1) / 65, text:`Loading case model: layer ${index + 1} of 64…`});
      const prefix = `model.layers.${index}.`, compact = `layers.${index}.`;
      const weights = suffix => dense.tensor(prefix + suffix);
      const layer = {
        inputNorm:own(await createRMSNorm(device, await weights('input_layernorm.weight'))),
        postNorm:own(await createRMSNorm(device, await weights('post_attention_layernorm.weight'))),
        attentionResidual:own(await createElementwise(device, 5120, 'add')),
        mlpResidual:own(await createElementwise(device, 5120, 'add')),
        activation:own(await createElementwise(device, 17408, 'silu-mul')),
      };
      layer.full = (index + 1) % 4 === 0;
      if (layer.full) {
        layer.qkv = await projection(compact + 'self_attn.qkv_proj');
        layer.attention = own(await createFullAttention(device, {q_norm:await weights('self_attn.q_norm.weight'), k_norm:await weights('self_attn.k_norm.weight')}, {maxTokens}));
        layer.out = await projection(compact + 'self_attn.o_proj');
      } else {
        layer.qkv = await projection(compact + 'linear_attn.in_proj_qkvz');
        layer.ba = await projection(compact + 'linear_attn.in_proj_ba');
        layer.attention = own(await createDeltaNet(device, {conv:await weights('linear_attn.conv1d.weight'), A_log:await weights('linear_attn.A_log'), dt_bias:await weights('linear_attn.dt_bias'), norm:await weights('linear_attn.norm.weight')}));
        layer.out = await projection(compact + 'linear_attn.out_proj');
      }
      layer.gateUp = await projection(compact + 'mlp.gate_up_proj');
      layer.down = await projection(compact + 'mlp.down_proj');
      layers.push(layer);
      await checkAllocation();
    }
    const finalNorm = own(await createRMSNorm(device, await dense.tensor('model.norm.weight')));
    dense.clearCache(); checkpoint.clearSmallTensorCache();
    await device.queue.onSubmittedWorkDone();
    const validation = await device.popErrorScope(), memory = await device.popErrorScope();
    if (validation || memory) throw new Error((validation || memory).message);
    assertDevice();
    let position = 0, interrupted = false, busy = false, cachedLogits = null;
    let consumedIds = [];
    async function forward(id, outputLogits) {
      assertDevice();
      if (position >= maxTokens) throw new Error(`Conversation exceeds the ${maxTokens}-token context capacity. Start a new chat.`);
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      let hidden = vocabulary.encodeEmbedding(encoder, new Uint32Array([id]));
      for (const layer of layers) {
        const normalized = layer.inputNorm.encode(encoder, hidden);
        const projected = layer.qkv.encode(encoder, normalized);
        const attended = layer.full ? layer.attention.encode(encoder, projected, position) : layer.attention.encode(encoder, projected, layer.ba.encode(encoder, normalized));
        const residual = layer.attentionResidual.encode(encoder, hidden, layer.out.encode(encoder, attended));
        const mlp = layer.gateUp.encode(encoder, layer.postNorm.encode(encoder, residual));
        hidden = layer.mlpResidual.encode(encoder, residual, layer.down.encode(encoder, layer.activation.encode(encoder, mlp)));
      }
      if (outputLogits) vocabulary.encodeHead(encoder, finalNorm.encode(encoder, hidden));
      device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
      const validation = await device.popErrorScope();
      if (validation) throw new Error(validation.message);
      assertDevice(); position++; consumedIds.push(id);
      cachedLogits = outputLogits ? await readGPUBuffer(device, vocabulary.buffers.logits) : null;
      return cachedLogits;
    }
    function reset() {position = 0; consumedIds = []; cachedLogits = null; layers.forEach(layer => layer.attention.reset());}
    function selectToken(logits, temperature, topP) {
      let best = 0;
      for (let i = 0; i < logits.length; i++) {
        if (!Number.isFinite(logits[i])) throw new Error('Model produced non-finite logits.');
        if (logits[i] > logits[best]) best = i;
      }
      if (!temperature || temperature <= 0) return best;
      const candidates = Array.from(logits, (value, id) => ({id, probability:Math.exp((value - logits[best]) / temperature)})).sort((a,b) => b.probability - a.probability);
      const total = candidates.reduce((sum,item) => sum + item.probability, 0);
      let mass = 0, count = 0;
      do {mass += candidates[count++].probability;} while (count < candidates.length && mass / total < topP);
      let random = Math.random() * mass;
      for (let i = 0; i < count; i++) {random -= candidates[i].probability; if (random <= 0) return candidates[i].id;}
      return candidates[count - 1].id;
    }
    async function* generate({messages, max_tokens = 1024, temperature = .3, top_p = .9}) {
      if (busy) throw new Error('A response is already running.');
      const {ids} = tokenizer.encodeChat(messages);
      if (ids.length + max_tokens > maxTokens) throw new Error(`Prompt uses ${ids.length} tokens; reduce the conversation to leave room for the response.`);
      // Reuse state only when the new rendered prompt exactly extends consumed IDs.
      // Any edit, new case or changed formatting causes a complete reset.
      const extendsState = consumedIds.length <= ids.length && consumedIds.every((id,index) => id === ids[index]);
      if (!extendsState) reset();
      const offset = consumedIds.length;
      busy = true; interrupted = false;
      let logits = cachedLogits, text = ''; const generated = [];
      let finishReason='length';
      try {
        for (let index = offset; index < ids.length; index++) {
          if (interrupted) return;
          logits = await forward(ids[index], index === ids.length - 1);
          onProgress({progress:(index + 1) / ids.length, text:`Reading case context: ${index + 1} of ${ids.length} tokens…`});
        }
        for (let index = 0; index < max_tokens && !interrupted; index++) {
          if (!logits) throw new Error('No next-token logits are available.');
          const token = selectToken(logits, temperature, top_p);
          if (tokenizer.eosTokenIds.includes(token)) {finishReason='stop';break;}
          generated.push(token);
          const decoded = tokenizer.decode(generated, true);
          // Wait for complete UTF-8 characters before emitting irreversible stream deltas.
          const stable = decoded.replace(/\uFFFD+$/, '');
          const content = stable.slice(text.length); text = stable;
          if (content) yield {choices:[{delta:{content}}]};
          logits = await forward(token, true);
        }
        if(!interrupted)yield {choices:[{delta:{},finish_reason:finishReason}]};
      } finally {busy = false;}
    }
    onProgress({progress:1, text:'Ready to run in this browser.'});
    return {chat:{completions:{create:async options => generate(options)}}, interruptGenerate(){interrupted = true;}, dispose(){owned.forEach(item => item.dispose()); device.destroy();}, forward, reset, tokenizer, device};
  } catch (error) {
    owned.forEach(item => item.dispose()); device.destroy(); throw error;
  }
}
