// Only tokenizer/template execution uses Transformers.js; model operators are custom WebGPU.
import { MIRAI_REVISION } from './mirai-checkpoint.mjs';
const TOKENIZERS_URL = 'https://esm.sh/@huggingface/transformers@3.8.1?target=es2022';

export async function openCaseTokenizer(base = '/models/qwen38/vllm/') {
  const [library, tokenizerJSON, config, generation, manifest] = await Promise.all([
    import(TOKENIZERS_URL),
    fetch(base + 'tokenizer.json').then(response => response.json()),
    fetch(base + 'tokenizer_config.json').then(response => response.json()),
    fetch(base + 'generation_config.json').then(response => response.json()),
    fetch(base + 'dense-manifest.json').then(response => response.json()),
  ]);
  if (manifest.revision !== MIRAI_REVISION) throw new Error('Tokenizer checkpoint revision mismatch');
  const tokenizer = new library.PreTrainedTokenizer(tokenizerJSON, config);
  return {
    eosTokenIds: Array.isArray(generation.eos_token_id) ? generation.eos_token_id : [generation.eos_token_id],
    encodeChat(messages) {
      if (!messages.length || messages[0].role !== 'system') throw new Error('The case context must be the first system message');
      // Apply the supplied template with reasoning disabled. This renders the
      // checkpoint's own assistant prefix, including the empty think block.
      const text = tokenizer.apply_chat_template(messages, {tokenize:false, add_generation_prompt:true, enable_thinking:false, reasoning_effort:'none'});
      const ids = tokenizer.encode(text, {add_special_tokens:false});
      return {text, ids};
    },
    decode(ids, skipSpecialTokens = false) { return tokenizer.decode(ids, {skip_special_tokens:skipSpecialTokens, clean_up_tokenization_spaces:false}); },
  };
}
