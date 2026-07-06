# Troubleshooting Guide

## Contents
- Out of memory (OOM) errors
- Performance issues
- Model loading errors
- Network and connection issues
- Quantization problems
- Distributed serving issues
- Debugging tools and commands

## Out of memory (OOM) errors

### Symptom: `torch.cuda.OutOfMemoryError` during model loading

**Cause**: Model + KV cache exceeds available VRAM

**Solutions (try in order)**:

1. **Reduce GPU memory utilization**:
```bash
vllm serve MODEL --gpu-memory-utilization 0.7  # Try 0.7, 0.75, 0.8
```

2. **Reduce max sequence length**:
```bash
vllm serve MODEL --max-model-len 4096  # Instead of 8192
```

3. **Enable quantization**:
```bash
vllm serve MODEL --quantization awq  # 4x memory reduction
```

4. **Use tensor parallelism** (multiple GPUs):
```bash
vllm serve MODEL --tensor-parallel-size 2  # Split across 2 GPUs
```

5. **Reduce max concurrent sequences**:
```bash
vllm serve MODEL --max-num-seqs 128  # Default is 256
```

### Symptom: OOM during inference (not model loading)

**Cause**: KV cache fills up during generation

**Solutions**:

```bash
# Reduce KV cache allocation
vllm serve MODEL --gpu-memory-utilization 0.85

# Reduce batch size
vllm serve MODEL --max-num-seqs 64

# Reduce max tokens per request
# Set in client request: max_tokens=512
```

### Symptom: OOM with quantized model

**Cause**: Quantization overhead or incorrect configuration

**Solution**:
```bash
# Ensure quantization flag matches model
vllm serve TheBloke/Llama-2-70B-AWQ --quantization awq  # Must specify

# Try different dtype
vllm serve MODEL --quantization awq --dtype float16
```

## Performance issues

### Symptom: Low throughput (<50 req/sec expected >100)

**Diagnostic steps**:

1. **Check GPU utilization**:
```bash
watch -n 1 nvidia-smi
# GPU utilization should be >80%
```

If <80%, increase concurrent requests:
```bash
vllm serve MODEL --max-num-seqs 512  # Increase from 256
```

2. **Check if memory-bound**:
```bash
# If memory at 100% but GPU <80%, reduce sequence length
vllm serve MODEL --max-model-len 4096
```

3. **Enable optimizations**:
```bash
vllm serve MODEL \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 512
```

4. **Check tensor parallelism settings**:
```bash
# Must use power-of-2 GPUs
vllm serve MODEL --tensor-parallel-size 4  # Not 3 or 5
```

### Symptom: High TTFT (time to first token >1 second)

**Causes and solutions**:

**Long prompts**:
```bash
vllm serve MODEL --enable-chunked-prefill
```

**No prefix caching**:
```bash
vllm serve MODEL --enable-prefix-caching  # For repeated prompts
```

**Too many concurrent requests**:
```bash
vllm serve MODEL --max-num-seqs 64  # Reduce to prioritize latency
```

**Model too large for single GPU**:
```bash
vllm serve MODEL --tensor-parallel-size 2  # Parallelize prefill
```

### Symptom: Slow token generation (low tokens/sec)

**Diagnostic**:
```bash
# Check if model is correct size
vllm serve MODEL  # Should see model size in logs

# Check speculative decoding
vllm serve MODEL --speculative-model DRAFT_MODEL
```

**For H100 GPUs**, enable FP8:
```bash
vllm serve MODEL --quantization fp8
```

## Model loading errors

### Symptom: `OSError: MODEL not found`

**Causes**:

1. **Model name typo**:
```bash
# Check exact model name on HuggingFace
vllm serve meta-llama/Llama-3-8B-Instruct  # Correct capitalization
```

2. **Private/gated model**:
```bash
# Login to HuggingFace first
huggingface-cli login
# Then run vLLM
vllm serve meta-llama/Llama-3-70B-Instruct
```

3. **Custom model needs trust flag**:
```bash
vllm serve MODEL --trust-remote-code
```

### Symptom: `ValueError: Tokenizer not found`

**Solution**:
```bash
# Download model manually first
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('MODEL')"

# Then launch vLLM
vllm serve MODEL
```

### Symptom: `ImportError: No module named 'flash_attn'`

**Solution**:
```bash
# Install flash attention
pip install flash-attn --no-build-isolation

# Or disable flash attention
vllm serve MODEL --disable-flash-attn
```

## Network and connection issues

### Symptom: `Connection refused` when querying server

**Diagnostic**:

1. **Check server is running**:
```bash
curl http://localhost:8000/health
```

2. **Check port binding**:
```bash
# Bind to all interfaces for remote access
vllm serve MODEL --host 0.0.0.0 --port 8000

# Check if port is in use
lsof -i :8000
```

3. **Check firewall**:
```bash
# Allow port through firewall
sudo ufw allow 8000
```

### Symptom: Slow response times over network

**Solutions**:

1. **Increase timeout**:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    timeout=300.0  # 5 minute timeout
)
```

2. **Check network latency**:
```bash
ping SERVER_IP  # Should be <10ms for local network
```

3. **Use connection pooling**:
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=1)
session.mount('http://', HTTPAdapter(max_retries=retries))
```

## Quantization problems

### Symptom: `RuntimeError: Quantization format not supported`

**Solution**:
```bash
# Ensure correct quantization method
vllm serve MODEL --quantization awq  # For AWQ models
vllm serve MODEL --quantization gptq  # For GPTQ models

# Check model card for quantization type
```

### Symptom: Poor quality outputs after quantization

**Diagnostic**:

1. **Verify model is correctly quantized**:
```bash
# Check model config.json for quantization_config
cat ~/.cache/huggingface/hub/models--MODEL/config.json
```

2. **Try different quantization method**:
```bash
# If AWQ quality issues, try FP8 (H100 only)
vllm serve MODEL --quantization fp8

# Or use less aggressive quantization
vllm serve MODEL  # No quantization
```

3. **Increase temperature for better diversity**:
```python
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

## Distributed serving issues

### Symptom: `RuntimeError: Distributed init failed`

**Diagnostic**:

1. **Check environment variables**:
```bash
# On all nodes
echo $MASTER_ADDR  # Should be same
echo $MASTER_PORT  # Should be same
echo $RANK  # Should be unique per node (0, 1, 2, ...)
echo $WORLD_SIZE  # Should be same (total nodes)
```

2. **Check network connectivity**:
```bash
# From node 1 to node 2
ping NODE2_IP
nc -zv NODE2_IP 29500  # Check port accessibility
```

3. **Check NCCL settings**:
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Or your network interface
vllm serve MODEL --tensor-parallel-size 8
```

### Symptom: `NCCL error: unhandled cuda error`

**Solutions**:

```bash
# Set NCCL to use correct network interface
export NCCL_SOCKET_IFNAME=eth0  # Replace with your interface

# Increase timeout
export NCCL_TIMEOUT=1800  # 30 minutes

# Force P2P for debugging
export NCCL_P2P_DISABLE=1
```

## Debugging tools and commands

### Enable debug logging

```bash
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve MODEL
```

### Monitor GPU usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Memory breakdown
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

### Profile performance

```bash
# Built-in benchmarking
vllm bench throughput \
  --model MODEL \
  --input-tokens 128 \
  --output-tokens 256 \
  --num-prompts 100

vllm bench latency \
  --model MODEL \
  --input-tokens 128 \
  --output-tokens 256 \
  --batch-size 8
```

### Check metrics

```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# Filter for specific metrics
curl http://localhost:9090/metrics | grep vllm_time_to_first_token

# Key metrics to monitor:
# - vllm_time_to_first_token_seconds
# - vllm_time_per_output_token_seconds
# - vllm_num_requests_running
# - vllm_gpu_cache_usage_perc
# - vllm_request_success_total
```

### Test server health

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/v1/models

# Test completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL",
    "prompt": "Hello",
    "max_tokens": 10
  }'
```

### Common environment variables

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Limit to specific GPUs

# vLLM settings
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION=1  # Profile functions
export VLLM_USE_V1=1  # Use v1.0 engine (faster)

# NCCL settings (distributed)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0  # Enable InfiniBand
```

### Collect diagnostic info for bug reports

```bash
# System info
nvidia-smi
python --version
pip show vllm

# vLLM version and config
vllm --version
python -c "import vllm; print(vllm.__version__)"

# Run with debug logging
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve MODEL 2>&1 | tee vllm_debug.log

# Include in bug report:
# - vllm_debug.log
# - nvidia-smi output
# - Full command used
# - Expected vs actual behavior
```

## v0.24.0 Troubleshooting

### `CUDA_VISIBLE_DEVICES` no longer restricting GPUs

**Cause**: v0.24.0 no longer sets `CUDA_VISIBLE_DEVICES` internally based on the env var alone during server startup.

**Solution**: Use the `--device-ids` flag explicitly:

```bash
# Instead of relying on CUDA_VISIBLE_DEVICES alone:
CUDA_VISIBLE_DEVICES=0,1 vllm serve MODEL  # may not work as expected

# Use --device-ids (works correctly in v0.24.0):
vllm serve MODEL --device-ids 0,1
```

### GGUF model not loading

**Cause**: GGUF support has been moved out of core vLLM into a plugin package as of v0.24.0.

**Solution**: Install the GGUF plugin:

```bash
pip install vllm-gguf

# Then launch normally:
vllm serve /path/to/model.gguf
```

If you see `ValueError: Model type gguf not supported`, the plugin is missing or failed to import.

### KV cache preemptions (throughput drops, requests stall)

**Symptom**: Logs show `Preemption: X sequence groups preempted`, latency spikes, or throughput collapses under load.

**Solutions**:

```bash
# 1. Enable prefix caching (KV-cache watermark reuse):
vllm serve MODEL --enable-prefix-caching

# 2. Use FP8 KV cache dtype (halves KV memory):
vllm serve MODEL --kv-cache-dtype fp8

# 3. Reduce concurrent sequences:
vllm serve MODEL --max-num-seqs 128  # or lower

# Combined:
vllm serve MODEL \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --max-num-seqs 128
```

### MoE FP8 + LoRA producing corrupt / garbled output

**Cause**: Known bug in earlier versions where FP8 quantized MoE models combined with LoRA adapters produced corrupted tokens.

**Solution**: Fixed in v0.24.0. Ensure you are on the latest patch:

```bash
pip install -U vllm==0.24.0
# Verify:
python -c "import vllm; print(vllm.__version__)"
```

### DeepGEMM warmup takes too long

**Symptom**: Server startup stalls for several minutes during DeepGEMM kernel compilation/warmup.

**Solution**: Skip eager warmup; kernels will JIT-compile at runtime on first use:

```bash
export VLLM_DEEP_GEMM_WARMUP=skip
vllm serve MODEL
```

### Triton autotuning takes too long

**Symptom**: First request triggers prolonged Triton kernel autotuning, causing high TTFT or timeouts.

**Solution**: Force the first config found instead of exhaustive search:

```bash
export VLLM_TRITON_FORCE_FIRST_CONFIG=1
vllm serve MODEL
```

### Large model cold start >20 minutes

**Symptom**: `vllm serve` takes 20+ minutes to become ready for large models (70B+, dense or MoE).

**Solutions** (combine all that apply):

```bash
# 1. Pre-cache model weights locally:
huggingface-cli download MODEL_ID  # run once, ahead of time

# 2. Run fully offline to avoid any HF hub checks:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 3. Remove any manual snapshot_download() calls in your startup script;
#    let vLLM load directly from the HF cache path instead.

# 4. Skip DeepGEMM warmup:
export VLLM_DEEP_GEMM_WARMUP=skip

# 5. Use fastsafetensors ParallelLoader for faster weight loading:
pip install fastsafetensors
export VLLM_USE_FASTSAFETENSORS=1  # if applicable to your setup

# Combined launch:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_DEEP_GEMM_WARMUP=skip \
  vllm serve MODEL --tensor-parallel-size 8
```

### NCCL errors with expert parallelism (MoE)

**Symptom**: `NCCL error`, hangs, or crashes when using expert parallel (EP) with MoE models.

**Cause**: NCCL-based expert-parallel load balancing (EPLB) is fragile and can deadlock.

**Solution**: Use DeepEP v2 or NIXL EP instead of NCCL-based EPLB:

```bash
# DeepEP v2 (preferred for Hopper/Blackwell):
export VLLM_USE_DEEPEP=1
vllm serve MODEL --ep-num-token-experts N

# Or NIXL-based expert parallel:
export VLLM_USE_NIXL_EP=1
vllm serve MODEL --ep-num-token-experts N
```

Avoid relying on `NCCL_P2P_DISABLE` workarounds; they mask the underlying issue.

### OOM with multimodal models (vision/audio)

**Symptom**: `torch.cuda.OutOfMemoryError` when serving VLMs or audio models, especially with large batches of image/audio inputs.

**Cause**: Multimodal embeds consume significant KV cache and activation memory.

**Solution**: v0.24.0 enables async scheduling with prompt embeds automatically, reducing peak memory. Ensure you are on v0.24.0+:

```bash
pip install -U vllm==0.24.0
vllm serve MODEL  # async scheduling + prompt embeds enabled by default
```

If still OOM, also reduce concurrency:

```bash
vllm serve MODEL --max-num-seqs 64 --limit-mm-per-prompt image=2
```

### Audio decompression bomb (security)

**Cause**: Pre-v0.24.0, crafted audio uploads could trigger excessive decompression (zip-bomb-style attack on audio codecs).

**Solution**: Fixed in v0.24.0. Audio upload size limits are now enforced by default. Ensure you are on v0.24.0+:

```bash
pip install -U vllm==0.24.0
```

No additional configuration needed; the limit is applied server-side.

### int32 truncation in GGUF dequantization

**Cause**: Earlier versions had an int32 overflow during GGUF dequantize, causing silent weight corruption or crashes on large GGUF files.

**Solution**: Fixed in v0.24.0. Ensure latest version:

```bash
pip install -U vllm==0.24.0 vllm-gguf
```
