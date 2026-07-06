---
name: serving-llms-vllm
description: "vLLM v0.24.0: high-throughput LLM serving, OpenAI API, quantization, expert parallelism, KV offloading."
version: 2.0.0
author: Orchestra Research
license: MIT
dependencies: [vllm, torch, transformers]
platforms: [linux, macos]
metadata:
  hermes:
    tags: [vLLM, Inference Serving, PagedAttention, Continuous Batching, High Throughput, Production, OpenAI API, Quantization, Tensor Parallelism, FP8, KV Offloading, Expert Parallelism, GLM-5.2, DeepSeek-V4]

---

# vLLM v0.24.0 - High-Performance LLM Serving

**Current version: v0.24.0** (June 29, 2025 — 571 commits, 256 contributors)

Full release notes: [references/v0.24.0-release-notes.md](references/v0.24.0-release-notes.md)

## ⚠️ v0.24.0 Breaking Changes

### Device Selection (CRITICAL)
vLLM no longer sets `CUDA_VISIBLE_DEVICES` internally. Use `--device-ids` instead:
```bash
# NEW v0.24.0 — explicit GPU selection
vllm serve MODEL --tensor-parallel-size 2 --device-ids 0,1
```
`CUDA_VISIBLE_DEVICES` still works as a container-level env var, but vLLM no longer auto-sets it from `--tensor-parallel-size`.

### GGUF Quantization is Now a Plugin
GGUF is no longer built-in — it's a plugin. May need explicit installation for GGUF models.

### Removed Models
ERNIE, Xverse, Dots1, Bamba, Mono-InternVL, InternLM registry alias.

### Deprecated
First-gen Qwen/QwenVL, Transformers v4 support, `CUDA_VISIBLE_DEVICES` on ROCm.

## When to use

Use when deploying production LLM APIs, optimizing inference latency/throughput, or serving models with limited GPU memory. Supports OpenAI-compatible endpoints, quantization (GPTQ/AWQ/FP8/NVFP4), tensor parallelism, expert parallelism (DeepEP v2/NIXL), KV offloading, and speculative decoding.

## v0.24.0 Key Features

### New Serving Capabilities
- **API-key authentication** — `--api-key` or `VLLM_API_KEY` env var (Rust frontend)
- **CORS support** — built-in, no reverse proxy needed
- **`/pause` `/resume` `/is_paused`** — pause/resume inference without restarting
- **`/abort_requests`** — cancel in-flight requests
- **`/tokenize` + `/detokenize`** — token-level control
- **`thinking_token_budget`** — limit reasoning tokens for thinking models
- **`/get_world_size`** — query TP/PP configuration

### Engine Improvements
- **KV-cache watermark** — reduces preemptions under memory pressure
- **Marconi-style admission policy** — smarter cache admission for hybrid workloads
- **`fastsafetensors ParallelLoader`** — faster weight loading for large models (700GB+)
- **Cross-layer KV cache layout for MLA** — stride-aware kernels, better MLA throughput
- **Reduced scheduler copy overhead** — less CPU overhead per scheduling iteration
- **Async scheduling with prompt embeds** — better multimodal scheduling

### Quantization Advances
- **Online FP8 per-token-per-channel (PTPC)** — runtime FP8 without pre-quantized model
- **fp8_e5m2 KV cache for non-FP8 models** — use FP8 KV cache with BF16/FP16 checkpoints
- **NVFP4 MoE** — FlashInfer cutedsl NVFP4 GEMM backend
- **MXFP4 W4A4 MoE** — CUTLASS E8M0 scale support
- **Corrupt-output fix for MoE FP8 + LoRAs** — critical bug fix

### Model Support (v0.24.0)
- **GLM-4.7/5.1/5.2** — streaming parser engine, MLA optimizations
- **DeepSeek-V4** — FlashInfer sparse index cache (2-4% TTFT), prefill chunk planning (4% throughput)
- **MiniMax-M3** — full support with FP8 sparse GQA, MXFP4
- **Gemma 4** — unified FlashAttention (FA4) across all layers
- **Qwen3-VL** — video loader, multi-video processing
- **DiffusionGemma** — diffusion LLM with CPU path

### Distributed & Large-Scale
- **DeepEP v2** — expert parallelism with robustness fixes
- **NIXL EP** — elastic expert parallel communicator
- **KV push from prefill to decode via NIXL** — disaggregated serving
- **Mooncake PD support** — pipeline-parallel prefill/decode
- **Multi-tier async KV offloading** — CPU/GPU hybrid KV cache

### Speculative Decoding
- **Dynamic SD** — dynamic speculative decoding
- **DFlash with FlashInfer** — flash-based spec decoding
- **EAGLE3 for Qwen3** — improved draft model support
- **MTP support for DeepGEMM** — critical for GLM-5.2 MTP

## Quick start

vLLM achieves 24x higher throughput than standard transformers through PagedAttention (block-based KV cache) and continuous batching (mixing prefill/decode requests).

**Installation**:
```bash
pip install vllm
```

**Basic offline inference**:
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
sampling = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["Explain quantum computing"], sampling)
print(outputs[0].outputs[0].text)
```

**OpenAI-compatible server**:
```bash
vllm serve meta-llama/Llama-3-8B-Instruct

# Query with OpenAI SDK
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
print(client.chat.completions.create(
    model='meta-llama/Llama-3-8B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
).choices[0].message.content)
"
```

## Common workflows

### Workflow 1: Production API deployment

Copy this checklist and track progress:

```
Deployment Progress:
- [ ] Step 1: Configure server settings
- [ ] Step 2: Test with limited traffic
- [ ] Step 3: Enable monitoring
- [ ] Step 4: Deploy to production
- [ ] Step 5: Verify performance metrics
```

**Step 1: Configure server settings**

Choose configuration based on your model size:

```bash
# For 7B-13B models on single GPU
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000

# For 30B-70B models with tensor parallelism
vllm serve meta-llama/Llama-2-70b-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --port 8000

# For production with caching and metrics
vllm serve meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-metrics \
  --metrics-port 9090 \
  --port 8000 \
  --host 0.0.0.0
```

**Step 2: Test with limited traffic**

Run load test before production:

```bash
# Install load testing tool
pip install locust

# Create test_load.py with sample requests
# Run: locust -f test_load.py --host http://localhost:8000
```

Verify TTFT (time to first token) < 500ms and throughput > 100 req/sec.

**Step 3: Enable monitoring**

vLLM exposes Prometheus metrics on port 9090:

```bash
curl http://localhost:9090/metrics | grep vllm
```

Key metrics to monitor:
- `vllm:time_to_first_token_seconds` - Latency
- `vllm:num_requests_running` - Active requests
- `vllm:gpu_cache_usage_perc` - KV cache utilization

**Step 4: Deploy to production**

Use Docker for consistent deployment:

```bash
# Run vLLM in Docker
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching
```

**Step 5: Verify performance metrics**

Check that deployment meets targets:
- TTFT < 500ms (for short prompts)
- Throughput > target req/sec
- GPU utilization > 80%
- No OOM errors in logs

### Workflow 2: Offline batch inference

For processing large datasets without server overhead.

Copy this checklist:

```
Batch Processing:
- [ ] Step 1: Prepare input data
- [ ] Step 2: Configure LLM engine
- [ ] Step 3: Run batch inference
- [ ] Step 4: Process results
```

**Step 1: Prepare input data**

```python
# Load prompts from file
prompts = []
with open("prompts.txt") as f:
    prompts = [line.strip() for line in f]

print(f"Loaded {len(prompts)} prompts")
```

**Step 2: Configure LLM engine**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=2,  # Use 2 GPUs
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

sampling = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["</s>", "\n\n"]
)
```

**Step 3: Run batch inference**

vLLM automatically batches requests for efficiency:

```python
# Process all prompts in one call
outputs = llm.generate(prompts, sampling)

# vLLM handles batching internally
# No need to manually chunk prompts
```

**Step 4: Process results**

```python
# Extract generated text
results = []
for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    results.append({
        "prompt": prompt,
        "generated": generated,
        "tokens": len(output.outputs[0].token_ids)
    })

# Save to file
import json
with open("results.jsonl", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Processed {len(results)} prompts")
```

### Workflow 3: Quantized model serving

Fit large models in limited GPU memory.

```
Quantization Setup:
- [ ] Step 1: Choose quantization method
- [ ] Step 2: Find or create quantized model
- [ ] Step 3: Launch with quantization flag
- [ ] Step 4: Verify accuracy
```

**Step 1: Choose quantization method**

- **AWQ**: Best for 70B models, minimal accuracy loss
- **GPTQ**: Wide model support, good compression
- **FP8**: Fastest on H100 GPUs

**Step 2: Find or create quantized model**

Use pre-quantized models from HuggingFace:

```bash
# Search for AWQ models
# Example: TheBloke/Llama-2-70B-AWQ
```

**Step 3: Launch with quantization flag**

```bash
# Using pre-quantized model
vllm serve TheBloke/Llama-2-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95

# Results: 70B model in ~40GB VRAM
```

**Step 4: Verify accuracy**

Test outputs match expected quality:

```python
# Compare quantized vs non-quantized responses
# Verify task-specific performance unchanged
```

### Workflow 4: Large MoE Models (GLM-5.2, DeepSeek-V4) on Multi-GPU

For 700B+ MoE models on 8×H200 (1,128GB VRAM) or similar setups.

**Optimal v0.24.0 configuration for GLM-5.2 FP8 (440GB)**:
```bash
vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \
  --max-num-seqs 256 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key YOUR_KEY
```

**Key v0.24.0 optimizations for large MoE**:
- `--kv-cache-dtype fp8` — FP8 KV cache (now works with non-FP8 models too via fp8_e5m2)
- `--enable-prefix-caching` — KV-cache watermark reduces preemptions
- `--enable-chunked-prefill` — prefill chunk-planning optimization (4% throughput)
- `fastsafetensors ParallelLoader` — faster 440GB shard loading (automatic)
- GLM-5.2 streaming parser — built-in tool-call/reasoning parsing
- MLA stride-aware kernels — cross-layer KV cache layout for MLA models

**KV Offloading for extended context** (v0.24.0):
```bash
# CPU KV offloading — extends effective KV cache to VRAM + CPU RAM
vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 8 \
  --kv-offloading-backend native \
  --kv-cache-dtype fp8 \
  --max-model-len 200000 \
  --max-num-seqs 4
```

**INT4 quantized MoE** (fits on fewer GPUs):
```bash
vllm serve cyankiwi/GLM-5.2-AWQ-INT4 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8
```

**Cold start optimization tips**:
1. Pre-cache model weights on a network volume (don't download in-container)
2. Set `VLLM_DEEP_GEMM_WARMUP=skip` to skip 8-min DeepGEMM warmup (JIT instead)
3. Use `fastsafetensors ParallelLoader` (automatic in v0.24.0) for parallel shard loading
4. Set `HF_HUB_OFFLINE=1` if weights are cached — prevents HF Hub verification on every restart
5. Remove any `snapshot_download()` calls from container startup — they verify 700GB against HF Hub (30+ min)

**VRAM budget for GLM-5.2 FP8 on 8×H200 (1,128GB total)**:
```
Model weights (FP8):    440GB
KV cache (FP8, 200K):   ~480GB per concurrent request
Available for KV:       1,128 - 440 = 688GB
Max concurrent (200K):  1 (688GB / 480GB)
Max concurrent (32K):   ~4-6
With CPU offload (1TB): up to 8 concurrent at 200K with TQ-4bit
```

**Speculative decoding with MTP** (GLM-5.2):
```bash
# MTP draft model for 2-3x throughput
vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 8 \
  --speculative-model "dnhkng/GLM-5.2-AWQ-INT4-FP8-MTP-delta" \
  --num-speculative-tokens 5 \
  --speculative-method ngram
```

## When to use vs alternatives

**Use vLLM when:**
- Deploying production LLM APIs (100+ req/sec)
- Serving OpenAI-compatible endpoints
- Limited GPU memory but need large models
- Multi-user applications (chatbots, assistants)
- Need low latency with high throughput

**Use alternatives instead:**
- **llama.cpp**: CPU/edge inference, single-user
- **HuggingFace transformers**: Research, prototyping, one-off generation
- **TensorRT-LLM**: NVIDIA-only, need absolute maximum performance
- **Text-Generation-Inference**: Already in HuggingFace ecosystem

## Common issues

**Issue: Out of memory during model loading**

Reduce memory usage:
```bash
vllm serve MODEL \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096
```

Or use quantization:
```bash
vllm serve MODEL --quantization awq
```

**Issue: Slow first token (TTFT > 1 second)**

Enable prefix caching for repeated prompts:
```bash
vllm serve MODEL --enable-prefix-caching
```

For long prompts, enable chunked prefill:
```bash
vllm serve MODEL --enable-chunked-prefill
```

**Issue: Model not found error**

Use `--trust-remote-code` for custom models:
```bash
vllm serve MODEL --trust-remote-code
```

**Issue: Low throughput (<50 req/sec)**

Increase concurrent sequences:
```bash
vllm serve MODEL --max-num-seqs 512
```

Check GPU utilization with `nvidia-smi` - should be >80%.

**Issue: Inference slower than expected**

Verify tensor parallelism uses power of 2 GPUs:
```bash
vllm serve MODEL --tensor-parallel-size 4  # Not 3
```

Enable speculative decoding for faster generation:
```bash
vllm serve MODEL --speculative-model DRAFT_MODEL
```

### v0.24.0 Specific Issues

**Issue: `CUDA_VISIBLE_DEVICES` not working as expected**

v0.24.0 no longer sets it internally. Use `--device-ids`:
```bash
# OLD (may not work as expected in v0.24.0)
CUDA_VISIBLE_DEVICES=0,1 vllm serve MODEL --tensor-parallel-size 2

# NEW v0.24.0
vllm serve MODEL --tensor-parallel-size 2 --device-ids 0,1
```

**Issue: GGUF model not loading**

GGUF is now a plugin in v0.24.0. Install the plugin:
```bash
pip install vllm-gguf  # or check vLLM docs for current plugin name
```

**Issue: KV cache preemptions on large models**

Enable the new KV-cache watermark:
```bash
vllm serve MODEL \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --max-num-seqs 128  # Lower concurrency to reduce pressure
```

Or use CPU KV offloading:
```bash
vllm serve MODEL \
  --kv-offloading-backend native \
  --kv-cache-dtype fp8
```

**Issue: MoE FP8 + LoRA producing corrupt output**

Fixed in v0.24.0 — ensure you're on v0.24.0 or later. The corrupt-output bug with MoE FP8 models when LoRAs are loaded has been resolved.

**Issue: DeepGEMM warmup taking too long**

Skip warmup for faster cold starts (JIT at runtime):
```bash
export VLLM_DEEP_GEMM_WARMUP=skip
vllm serve zai-org/GLM-5.2-FP8 --tensor-parallel-size 8
```

**Issue: Triton autotuning taking too long**

Skip Triton autotuning:
```bash
export VLLM_TRITON_FORCE_FIRST_CONFIG=1
vllm serve MODEL
```

**Issue: Large model cold start >20 min**

1. Pre-cache weights on network volume — don't download in-container
2. Set `HF_HUB_OFFLINE=1` if weights cached
3. Remove `snapshot_download()` from startup code
4. Set `VLLM_DEEP_GEMM_WARMUP=skip`
5. `fastsafetensors ParallelLoader` loads shards in parallel (automatic in v0.24.0)

## Advanced topics

**v0.24.0 Full release notes**: See [references/v0.24.0-release-notes.md](references/v0.24.0-release-notes.md) for complete details on all 571 commits.

**Server deployment patterns**: See [references/server-deployment.md](references/server-deployment.md) for Docker, Kubernetes, Rust frontend, and load balancing configurations.

**Performance optimization**: See [references/optimization.md](references/optimization.md) for PagedAttention tuning, continuous batching, KV offloading, speculative decoding, and v0.24.0 benchmark results.

**Quantization guide**: See [references/quantization.md](references/quantization.md) for AWQ/GPTQ/FP8/NVFP4/MXFP4 setup, online FP8 PTPC, and accuracy comparisons.

**Troubleshooting**: See [references/troubleshooting.md](references/troubleshooting.md) for detailed error messages, debugging steps, and v0.24.0-specific issues.

## v0.24.0 Environment Variables

```bash
# Device selection (v0.24.0 — replaces internal CUDA_VISIBLE_DEVICES)
--device-ids 0,1,2,3

# API authentication (Rust frontend)
VLLM_API_KEY=your-secret-key

# DeepGEMM warmup skip (faster cold start, JIT at runtime)
VLLM_DEEP_GEMM_WARMUP=skip

# Skip Triton autotuning (faster startup)
VLLM_TRITON_FORCE_FIRST_CONFIG=1

# Debug logging
VLLM_LOGGING_LEVEL=DEBUG

# HF Hub offline mode (when weights are pre-cached)
HF_HUB_OFFLINE=1

# Fast HF transfers (for initial download)
HF_XET_HIGH_PERFORMANCE=1
```

## Hardware requirements

- **Small models (7B-13B)**: 1x A10 (24GB) or A100 (40GB)
- **Medium models (30B-40B)**: 2x A100 (40GB) with tensor parallelism
- **Large models (70B+)**: 4x A100 (40GB) or 2x A100 (80GB), use AWQ/GPTQ
- **MoE models (700B+)**: 8x H200 (141GB each, 1,128GB total), use FP8 + KV offloading

Supported platforms: NVIDIA (primary, SM80-SM100), AMD ROCm (gfx950, MI300X), Intel XPU, CPU, TPU

## Resources

- Official docs: https://docs.vllm.ai
- GitHub: https://github.com/vllm-project/vllm
- v0.24.0 release: https://github.com/vllm-project/vllm/releases/tag/v0.24.0
- Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)
- Community: https://discuss.vllm.ai



