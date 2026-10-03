"""Inspect the requested checkpoint and host; this is not a model compiler."""

import datetime
import json
import pathlib
import platform
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
source = ROOT / "model-source"
config = json.loads((source / "vllm/config.json").read_text())
package = json.loads((source / "physical_package_report.json").read_text())
mlc_model = (source / "mlc-qwen35-model.py").read_text()
mlc_quantization = (source / "mlc-quantization.py").read_text()
webllm_registry = (source / "webllm-config.ts").read_text()
memory = None
if platform.system() == "Darwin":
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
free_disk = shutil.disk_usage(ROOT).free
issues = []
quantization = config["quantization_config"]["quant_method"]
if "mirai" not in mlc_quantization.lower():
    issues.append("The inspected MLC quantization registry does not implement the checkpoint's mirai_s weight format.")
gate = config["text_config"].get("output_gate_type")
# output_gate_type configures the DeltaNet output norm. Full attention has a
# separate sigmoid gate in vLLM 0.30.0; these must not be conflated.
if "Qwen3.8-27B-S" not in webllm_registry:
    issues.append("The requested model has no entry in the inspected WebLLM prebuilt model registry.")
model_bytes = package["model_safetensors_bytes"]
if free_disk < model_bytes:
    issues.append("Available local disk space is smaller than the single source model weight file.")
if memory is not None and memory < 24 * 1024**3:
    issues.append("This Mac has less than the 24 GB minimum specified by the checkpoint's native Mac runtime documentation.")
operator_path = ROOT / 'webgpu-operator-report.json'
operators = json.loads(operator_path.read_text()) if operator_path.exists() else {}
operator_verified = operators.get('revision') == '32d4c0422a14353808659f2f2333b207bc1dd12a' and operators.get('operatorChecksPassed') is True
report = {
    "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source_repository": "trymirai/Qwen3.8-27B-S-experimental",
    "source_revision": "32d4c0422a14353808659f2f2333b207bc1dd12a",
    "source_architecture": config["architectures"][0],
    "source_quantization": quantization,
    "source_output_gate": gate,
    "full_attention_output_gate": "sigmoid (vLLM 0.30.0 Qwen3NextAttention)",
    "source_model_bytes": model_bytes,
    "host_memory_bytes": memory,
    "host_free_disk_bytes": free_disk,
    "browser_build_produced": False,
    "model_inference_verified": False,
    "browser_operators_verified": operator_verified,
    "operator_report": "webgpu-operator-report.json" if operator_verified else None,
    "compatibility_findings": issues,
    "remaining_work": ["Hybrid Qwen3.8 attention and dense operators", "Tokenizer and generation loop", "Whole-model loading and memory measurement", "Full-model numerical and token-generation verification", "Connection of verified runtime to dashboard chat"],
    "inspection_scope": "Metadata inspection plus saved Chrome operator results. A custom WebGPU partial port exists; full-model inference has not been verified. Native Mac memory requirements do not establish browser requirements.",
}
(ROOT / "browser-model-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
sys.exit(2 if issues else 0)
