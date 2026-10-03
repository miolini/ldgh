# Requested model and current build state

The target remains `trymirai/Qwen3.8-27B-S-experimental`, revision `32d4c0422a14353808659f2f2333b207bc1dd12a`. Individual WebGPU operators now compile and execute in Chrome. The **complete model has not been loaded or run**, and the dashboard cannot generate answers yet.

The chat panel is implemented in `chat.mjs`. It keeps conversation history in memory and places `buildCaseContext()`'s system message first in every completion request. The initial context contains all 16 supplied matter fields, the six open tasks, and five source notes about coverage, conflicting liability accounts, surgery, ledgers, and discovery. It preserves conflicting source statements and includes source IDs. The prepared prefix is approximately 7,880 characters.

`browser-model.json` is the integration descriptor, currently `operators-verified`. The existing chat adapter accepts a completed WebLLM artifact through WebLLM 0.2.85. The custom WebGPU route now has its own worker/client adapter and is available through Load local model after the verified file download. It is labeled experimental until full-model execution is verified. Neither route may be marked `ready` until full generation works with the exact checkpoint and the case prefix is verified.

## Verified browser port progress

`mirai-webgpu.mjs` implements the original compressed tape decoder, rotation, two-plane activation quantization, and single-token matrix-vector products for all three stored formats: `v4t8`, `v2t4`, and `v2t6`. It also implements multiple blocks, original output-row permutations, and padding. Pipeline compilation is cached by device and shader source.

`mirai-vocabulary.mjs` implements the D4 embedding and I3 output-head codecs. The checkpoint's compressed weight values are preserved; no weight requantization is used. Activations retain the BF16/F16 rounding specified by the native plugin.

`mirai-checkpoint.mjs` fetches the pinned safetensors header and tensors through bounded, sequential 8 MiB ranges. It rejects a server that returns a full file instead of the requested range and caches only small tensors. Chrome successfully fetched the header and the 1 KB embedding table directly from Hugging Face. Larger transfers and full-model memory residency are unverified.

The browser harness is `webgpu-check.html`. Actual Chrome results are saved in `webgpu-operator-report.json`:

| Operation | Checked values | Result against independent NumPy reference |
| --- | ---: | --- |
| v4t8 matrix slice | 96 outputs | Exact BF16 outputs |
| v2t4 matrix slice | 96 outputs | Exact BF16 outputs |
| v2t6 matrix slice | 96 outputs | One output differs by one BF16 ULP |
| D4 token embeddings | 25,600 | Exact BF16 outputs |
| Output-head input rotation | 5,120 | Within one F16 ULP |
| I3 output-head slice | 96 logits | Maximum absolute error 1.24e-6 |
| Complete layer 3 QKV projection, two blocks and original permutation | 14,336 outputs | Within one BF16 ULP plus 2e-5 absolute; maximum error 4.88e-4 |

The matrix slices also pass zero-input checks. Rotation errors are at most 4.47e-8. A few intermediate activation bins differ only where that F32 error crosses a rounding boundary; the harness reports those differences separately. The complete projection took about 18 ms including input retrieval and GPU completion, not a whole-model throughput measurement.

Fixtures contain unmodified slices and one complete projection, with per-tensor SHA256 values and pinned revision provenance. Reference scripts are `tools/mirai_reference.py`, `tools/fetch_vocabulary_fixtures.py`, and `tools/fetch_projection_fixture.py`. These checks validate operators against an independent mathematical reference. They do **not** establish agreement with complete native uzu/vLLM inference, full vocabulary behavior, token generation, or full-model memory fit.

## Findings

- The checkpoint uses **Mirai S**, a custom 2.4-bit codec. The supplied runtimes are uzu for Apple Metal and a vLLM CUDA plugin. Its root config is an uzu `LanguageModelConfig`; its vLLM architecture is `MiraiSQwen3_5ForCausalLM` with `quant_method: mirai_s`.
- The inspected MLC quantization registry has no Mirai S implementation. Standard `convert_weight` does not provide this codec's loader or operators.
- The model specifies `output_gate_type: swish` for its **DeltaNet gated norm**. Its separate full-attention output gate is sigmoid. This is verified in the model card's prescribed vLLM 0.30.0 implementation. An earlier inspection incorrectly compared those two different gates; that mismatch finding is withdrawn.
- There is no prebuilt entry for the requested checkpoint in the inspected WebLLM registry.
- The root source weight file is 8,447,545,388 bytes; the vLLM compressed sidecar is 8,449,310,484 bytes. The current Mac has about 13 GiB available disk and 16 GiB total memory. The model card requires at least 24 GB memory for its native Mac runtime, which it says uses approximately 15 GB; the complete native download is approximately 12 GB. Browser memory requirements remain unmeasured.

Run `python3 tools/check_browser_model.py` to regenerate `browser-model-report.json`. Exit code 2 indicates unresolved findings. This is a metadata/source inspection, not a compiler or an inference test.

## Work required for execution entirely in the browser

1. Full orchestration is now implemented in `qwen38-runtime.mjs`, with a streaming worker/client adapter. Load and execute it after the local checkpoint finishes downloading; its full-model behavior remains unverified.
2. Browser attention checks pass all 25 comparisons against independent four-step references, including causal convolution, DeltaNet state, full attention, RoPE, norms, MLP activation and residuals. The checkpoint tokenizer renders the case prefix first, uses 2,030 tokens for the initial check and round-trips exactly. Generation and stopping are implemented but await full-model execution.
3. Load all 304 projections and full vocabulary surfaces with bounded memory, then measure actual memory and generation speed on the intended device.
4. Compare layer outputs and generated tokens against native uzu/vLLM for this exact checkpoint. The independent operator reference alone is insufficient.
5. Connect the completed worker to the chat panel, set `status: ready` after verification, and verify streamed answers with the system message containing the case prefix.

The model's existing OpenAI-compatible server is an alternative deployment route if chosen by the user. It would run inference outside the browser and would not satisfy a requirement for execution entirely inside the browser.

## Sources

- [Checkpoint model card](https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental/blob/main/README.md)
- [Checkpoint vLLM config](https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental/blob/main/vllm/config.json)
- [Checkpoint uzu runtime instructions](https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental/blob/main/uzu/README.md)
- [MLC Qwen3.5 implementation](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/qwen35/qwen35_model.py)
- [MLC WebLLM compilation guide](https://llm.mlc.ai/docs/deploy/webllm.html)

## Local file transport

`python3 server.py` serves the dashboard at localhost:8000. `/models/qwen38/vllm/trellis.mirai` supports HEAD, bounded and suffix byte ranges, 206/416 responses, and streamed responses. The pinned 8,449,310,484-byte checkpoint is cached in resumable 8 MiB chunks on disk. `/api/model-cache/status` reports partial progress; POST `/api/model-cache/start` starts or resumes downloading. Complete-file SHA256 must match the pinned LFS identity before the full runtime loads. The page shows download progress. Dense shards and tokenizer assets are served from the same local model route. No inference runs on this HTTP server.

Chrome rechecked the generalized vocabulary and projection APIs successfully after the runtime refactor. Browser access through the local HTTP route also passes: pinned header, all 304 projection descriptors, and a 1 KB embedding-table tensor. This does not imply the entire checkpoint is downloaded.

Full attention compiles and passes the same four-step reference with a 4,096-token KV allocation and a requested 17,408-byte workgroup storage limit. This verifies allocation and the first four positions, not all 4,096 positions or full-model generation. The runtime reuses recurrent/KV state only when the new rendered prompt is an exact extension of previously consumed token IDs; otherwise it resets. Worker progress is forwarded during generation as well as model loading.

Additional browser checks pass for synthetic attention at positions 255, 256 and 4095 with zero earlier K/V entries. The complete projection and vocabulary head also pass with GPU-buffer inputs, matching the transport used by the full runtime. Generation requests are serialized in the worker so a new chat waits for a stopped token to finish; canceled queued responses are discarded. Complete model loading and generation remain unverified while the checkpoint downloads.

## Full buffer allocation evidence

Chrome on this Mac successfully allocated and GPU-cleared all 5,186 planned full-runtime buffers, totaling 10,083,905,988 bytes (9.39 GiB), including 4,096-token full-attention caches. The plan comes from pinned compressed tensor shapes and the runtime allocation code. `gpu-memory-report.json` records the actual browser result. All buffers were released after the check. This establishes allocation feasibility for those buffers, not full loading with tokenizer/pipelines, native agreement, throughput, or generation. Complete-model execution remains the required next check after the file download.
