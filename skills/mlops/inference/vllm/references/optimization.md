# Performance Optimization

## Contents
- PagedAttention explained
- Continuous batching mechanics
- Prefix caching strategies
- Speculative decoding setup
- Benchmark results and comparisons
- Performance tuning guide

## PagedAttention explained

**Traditional attention problem**:
- KV cache stored in contiguous memory
- Wastes ~50% GPU memory due to fragmentation
- Cannot dynamically reallocate for varying sequence lengths

**PagedAttention solution**:
- Divides KV cache into fixed-size blocks (like OS virtual memory)
- Dynamic allocation from free block queue
- Shares blocks across sequences (for prefix caching)

**Memory savings example**:
```
Traditional: 70B model needs 160GB KV cache → OOM on 8x A100
PagedAttention: 70B model needs 80GB KV cache → Fits on 4x A100
```

**Configuration**:
```bash
# Block size (default: 16 tokens)
vllm serve MODEL --block-size 16

# Number of GPU blocks (auto-calculated)
# Controlled by --gpu-memory-utilization
vllm serve MODEL --gpu-memory-utilization 0.9
```

## Continuous batching mechanics

**Traditional batching**:
- Wait for all sequences in batch to finish
- GPU idle while waiting for longest sequence
- Low GPU utilization (~40-60%)

**Continuous batching**:
- Add new requests as slots become available
- Mix prefill (new requests) and decode (ongoing) in same batch
- High GPU utilization (>90%)

**Throughput improvement**:
```
Traditional batching: 50 req/sec @ 50% GPU util
Continuous batching: 200 req/sec @ 90% GPU util
= 4x throughput improvement
```

**Tuning parameters**:
```bash
# Max concurrent sequences (higher = more batching)
vllm serve MODEL --max-num-seqs 256

# Prefill/decode schedule (auto-balanced by default)
# No manual tuning needed
```

## Prefix caching strategies

Reuse computed KV cache for common prompt prefixes.

**Use cases**:
- System prompts repeated across requests
- Few-shot examples in every prompt
- RAG contexts with overlapping chunks

**Example savings**:
```
Prompt: [System: 500 tokens] + [User: 100 tokens]

Without caching: Compute 600 tokens every request
With caching: Compute 500 tokens once, then 100 tokens/request
= 83% faster TTFT
```

**Enable prefix caching**:
```bash
vllm serve MODEL --enable-prefix-caching
```

**Automatic prefix detection**:
- vLLM detects common prefixes automatically
- No code changes required
- Works with OpenAI-compatible API

**Cache hit rate monitoring**:
```bash
curl http://localhost:9090/metrics | grep cache_hit
# vllm_cache_hit_rate: 0.75  (75% hit rate)
```

## Speculative decoding setup

Use smaller "draft" model to propose tokens, larger model to verify.

**Speed improvement**:
```
Standard: Generate 1 token per forward pass
Speculative: Generate 3-5 tokens per forward pass
= 2-3x faster generation
```

**How it works**:
1. Draft model proposes K tokens (fast)
2. Target model verifies all K tokens in parallel (one pass)
3. Accept verified tokens, restart from first rejection

**Setup with separate draft model**:
```bash
vllm serve meta-llama/Llama-3-70B-Instruct \
  --speculative-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-speculative-tokens 5
```

**Setup with n-gram draft** (no separate model):
```bash
vllm serve MODEL \
  --speculative-method ngram \
  --num-speculative-tokens 3
```

**When to use**:
- Output length > 100 tokens
- Draft model 5-10x smaller than target
- Acceptable 2-3% accuracy trade-off

## Benchmark results

**vLLM vs HuggingFace Transformers** (Llama 3 8B, A100):
```
Metric                  | HF Transformers | vLLM   | Improvement
------------------------|-----------------|--------|------------
Throughput (req/sec)    | 12              | 280    | 23x
TTFT (ms)              | 850             | 120    | 7x
Tokens/sec             | 45              | 2,100  | 47x
GPU Memory (GB)        | 28              | 16     | 1.75x less
```

**vLLM vs TensorRT-LLM** (Llama 2 70B, 4x A100):
```
Metric                  | TensorRT-LLM | vLLM   | Notes
------------------------|--------------|--------|------------------
Throughput (req/sec)    | 320          | 285    | TRT 12% faster
Setup complexity        | High         | Low    | vLLM much easier
NVIDIA-only            | Yes          | No     | vLLM multi-platform
Quantization support    | FP8, INT8    | AWQ/GPTQ/FP8 | vLLM more options
```

## Performance tuning guide

**Step 1: Measure baseline**

```bash
# Install benchmarking tool
pip install locust

# Run baseline benchmark
vllm bench throughput \
  --model MODEL \
  --input-tokens 128 \
  --output-tokens 256 \
  --num-prompts 1000

# Record: throughput, TTFT, tokens/sec
```

**Step 2: Tune memory utilization**

```bash
# Try different values: 0.7, 0.85, 0.9, 0.95
vllm serve MODEL --gpu-memory-utilization 0.9
```

Higher = more batch capacity = higher throughput, but risk OOM.

**Step 3: Tune concurrency**

```bash
# Try values: 128, 256, 512, 1024
vllm serve MODEL --max-num-seqs 256
```

Higher = more batching opportunity, but may increase latency.

**Step 4: Enable optimizations**

```bash
vllm serve MODEL \
  --enable-prefix-caching \     # For repeated prompts
  --enable-chunked-prefill \    # For long prompts
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 512
```

**Step 5: Re-benchmark and compare**

Target improvements:
- Throughput: +30-100%
- TTFT: -20-50%
- GPU utilization: >85%

**Common performance issues**:

**Low throughput (<50 req/sec)**:
- Increase `--max-num-seqs`
- Enable `--enable-prefix-caching`
- Check GPU utilization (should be >80%)

**High TTFT (>1 second)**:
- Enable `--enable-chunked-prefill`
- Reduce `--max-model-len` if possible
- Check if model is too large for GPU

**OOM errors**:
- Reduce `--gpu-memory-utilization` to 0.7
- Reduce `--max-model-len`
- Use quantization (`--quantization awq`)

## v0.24.0 Optimization Features

### 1. KV-cache watermark to reduce preemptions

`--enable-prefix-caching` now includes watermark logic that reserves a fraction of KV-cache blocks for in-flight requests. This prevents preemptions where active sequences are evicted (recomputed) because prefix-cache blocks consumed all available memory.

```bash
# Watermark is automatic when prefix caching is enabled.
# Control the reserved fraction via env var (default: 0.10 = 10% of blocks):
VLLM_PREFIX_CACHE_WATERMARK=0.15 vllm serve MODEL --enable-prefix-caching
```

**When to raise the watermark**: high preemption rates (`vllm:num_preemptions` metric > 0) under heavy prefix-cache workloads.

### 2. Marconi-style admission policy for hybrid cache

A Marconi-inspired admission policy decides which incoming requests are admitted to the hybrid (GPU + CPU) cache based on expected reuse value. Requests with low reuse probability are evicted sooner, improving effective cache hit rate under mixed workloads.

```bash
# Enable hybrid cache with the Marconi admission policy
vllm serve MODEL \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --cpu-swap-space 16

# The admission policy is active by default in v0.24.0 for hybrid cache setups.
```

### 3. fastsafetensors ParallelLoader

For very large models (700GB+), vLLM now uses `fastsafetensors.ParallelLoader` to load checkpoint shards in parallel across ranks. This reduces load time dramatically for multi-GPU setups and is **automatic** — no flag required.

```bash
# No explicit flag needed; parallel loading is automatic for multi-GPU.
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8

# To monitor shard load throughput, enable debug logging:
VLLM_LOGGING_LEVEL=DEBUG vllm serve MODEL --tensor-parallel-size 8
```

### 4. Cross-layer KV cache layout for MLA via stride-aware kernels

Multi-head Latent Attention (MLA) models benefit from a new cross-layer KV cache layout that uses stride-aware memory access kernels, improving memory bandwidth utilization for MLA-based architectures (DeepSeek-V2/V3/V4).

```bash
# MLA cross-layer layout is enabled automatically for MLA models.
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-prefix-caching
```

### 5. Reduced scheduler copy overhead

The scheduler's per-iteration copy overhead has been reduced via shallow-copy and in-place mutation optimizations. This lowers CPU time per scheduling iteration, which matters at high batch sizes.

No configuration required — applies automatically. Monitor with:
```bash
curl http://localhost:9090/metrics | grep vllm:scheduler_iteration_seconds
```

### 6. Async scheduling with prompt embeds for multimodal models

Multimodal models (LLaVA, Qwen2-VL, etc.) now support async scheduling where prompt embedding computation runs concurrently with the scheduler, reducing head-of-line blocking.

```bash
vllm serve Qwen/Qwen2-VL-72B-Instruct \
  --tensor-parallel-size 4 \
  --enable-chunked-prefill
# Async prompt-embed scheduling is enabled automatically for supported
# multimodal models in v0.24.0.
```

### 7. Prefill chunk-planning optimization

Improved prefill chunk planning reduces wasted compute during chunked prefill by coalescing small chunks. Yields ~4% end-to-end throughput improvement for DeepSeek-V4.

```bash
# Already enabled with chunked prefill in v0.24.0
vllm serve deepseek-ai/DeepSeek-V3 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

### 8. FlashInfer sparse index cache

When using the FlashInfer attention backend, a sparse attention index cache avoids recomputing index structures, giving 2-4% TTFT improvement on long-context workloads.

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve MODEL \
  --enable-prefix-caching \
  --max-model-len 128000
```

### 9. SM90 CUTLASS FP8 mm odd-M support

FP8 GEMM kernels (CUTLASS SM90 backend) now support odd-M matrix shapes, eliminating padding overhead. This gives 180-290% speedup on those kernel calls (common in decode and small-batch scenarios).

```bash
# Enable FP8 quantization (model must support FP8 weights or KV cache)
vllm serve MODEL \
  --quantization fp8 \
  --kv-cache-dtype fp8

# Ensure SM90 CUTLASS backend is used (default on H100/H200):
VLLM_USE_TRITON_FLASH_ATTN=0 vllm serve MODEL --quantization fp8
```

### 10. Tuned fused_moe FP8 for Qwen3-Next-80B on H100

The fused MoE FP8 kernel has been re-tuned for Qwen3-Next-80B on H100, delivering ~25% throughput improvement on MoE layers.

```bash
vllm serve Qwen/Qwen3-Next-80B \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --kv-cache-dtype fp8
```

### 11. `VLLM_TRITON_FORCE_FIRST_CONFIG` — skip Triton autotuning

Triton kernel autotuning adds startup latency and can select suboptimal configs on unusual shapes. This env var forces Triton to use the first tuning config without benchmarking alternatives.

```bash
# Skip Triton autotuning for faster startup (use when shapes are well-known)
VLLM_TRITON_FORCE_FIRST_CONFIG=1 vllm serve MODEL --quantization fp8

# Combine with pinned CUDA devices for reproducible perf:
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_TRITON_FORCE_FIRST_CONFIG=1 \
  vllm serve MODEL --tensor-parallel-size 4
```

### 12. CPU KV offloading with native backend

A new `native` CPU KV offloading backend provides multi-tier async batched lookups and a packed Host-Managed Array (HMA) layout, reducing CPU↔GPU transfer overhead for CPU-offloaded KV cache.

```bash
vllm serve MODEL \
  --kv-offloading-backend native \
  --cpu-swap-space 32 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.92
```

**Key internals**:
- Multi-tier: GPU VRAM → pinned host → pageable host
- Async batched lookups hide transfer latency behind compute
- Packed HMA layout reduces scatter/gather overhead

### 13. Speculative decoding enhancements

v0.24.0 adds multiple new speculative decoding paths:

**Dynamic Speculative Decoding (Dynamic SD)**: adjusts `num_speculative_tokens` at runtime based on observed acceptance rate.

```bash
vllm serve MODEL \
  --speculative-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num-speculative-tokens 5 \
  --speculative-method dynamic
# Acceptance rate is monitored; token count auto-adjusts 1–8.
```

**DFlash with FlashInfer**: draft-target attention via FlashInfer for lower overhead.

```bash
VLLM_ATTENTION_BACKEND=FLASHINFER vllm serve MODEL \
  --speculative-method dflash \
  --num-speculative-tokens 4
```

**EAGLE3 for Qwen3**: EAGLE3 draft model support for Qwen3 family.

```bash
vllm serve Qwen/Qwen3-32B \
  --speculative-model qwen3-eagle3-draft \
  --speculative-method eagle3 \
  --num-speculative-tokens 4
```

**MTP (Multi-Token Prediction) for DeepGEMM**: native MTP speculative path using DeepGEMM kernels, primarily for DeepSeek models with MTP heads.

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --speculative-method mtp \
  --num-speculative-tokens 3 \
  --quantization fp8
```
