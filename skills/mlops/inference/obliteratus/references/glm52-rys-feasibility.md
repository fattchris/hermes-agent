# GLM-5.2 RYS Feasibility Analysis

**Date:** Jul 4, 2026
**Context:** Can RYS layer duplication be applied to GLM-5.2 (the model powering this Hermes instance)?

## GLM-5.2 Architecture

| Parameter | Value |
|---|---|
| Total params | 754B |
| Active params per token | ~18.5B |
| Architecture | `GlmMoeDsaForCausalLM` (glm_moe_dsa) |
| Hidden size | 6,144 |
| **Num hidden layers** | **78** |
| Num attention heads | 64 |
| KV heads | 64 (MLA-style with kv_lora_rank=512) |
| Head dim | 192 |
| Intermediate size | 12,288 |
| Vocab size | 154,880 |
| Max context | 1,048,576 (~1M) |
| Dtype | bfloat16 |

### MoE Configuration

| Parameter | Value |
|---|---|
| Routed experts | 256 |
| Shared experts | 1 |
| Active experts per token | 8 |
| MoE intermediate size | 2,048 |
| MoE layer frequency | 1 (every layer) |
| Scoring function | sigmoid |
| Topk method | noaux_tc (DeepSeek-style) |
| Routed scaling factor | 2.5 |

### Layer Structure

- **Layers 0-2 (3 layers):** Dense MLP (`first_k_dense_replace = 3`)
- **Layers 3-77 (75 layers):** Sparse MoE

### IndexShare DSA (Dynamic Sparse Attention)

GLM-5.2 introduces **IndexShare**: every 4 transformer layers share a lightweight indexer. The pattern is:

```
["full", "shared", "shared", "shared", "full", "shared", "shared", "shared", ...]
```

- **Full indexers:** at layers 0, 4, 8, 12, ..., 76 (every 4th layer, 20 total)
- **Shared indexers:** all remaining layers (58 total)
- `index_topk = 2048`, `index_topk_freq = 4`
- Reduces per-token FLOPs by 2.9× at 1M context

### MTP (Multi-Token Prediction)

- 1 next-N prediction layer for speculative decoding
- Increases acceptance length by ~20%

## Model Size & Quantization

| Format | Size | Fits Dr Teeth (128GB)? |
|---|---|---|
| BF16 (original) | 1.51 TB | ❌ No |
| Q8_0 (8-bit) | 801 GB | ❌ No |
| Q4_K_M (4-bit) | 466 GB | ❌ No |
| IQ4_XS (4-bit) | 365 GB | ❌ No |
| IQ3_XXS (3-bit) | 282 GB | ❌ No |
| IQ2_XXS (2-bit) | 238 GB | ❌ No |
| IQ1_S (1-bit) | 217 GB | ❌ No |

**Conclusion:** GLM-5.2 does not fit on any current Apple Silicon machine, even at 1-bit quantization. Dr Teeth (M5 Max, 128GB) cannot run it locally.

### AWQ-INT4 (Recommended for RYS Sweep, Jul 4 2026)

`cyankiwi/GLM-5.2-AWQ-INT4` — 440GB, AWQ 4-bit quantization, MIT license. Loads via `transformers.AutoModelForCausalLM.from_pretrained()` with `device_map="auto"` across multi-GPU.

**Minimum GPU config: 8× A100 80GB (640GB VRAM)** on RunPod = $11.92/hr. 6× A100 (480GB) is NOT enough — the model is 411GB on disk but loading overhead (weight conversion, device placement, fragmentation) pushes peak VRAM to ~500GB+. OOMs on 6× A100 with `CUDA out of memory. Tried to allocate 12.00 GiB. GPU 2 has 6.08 GiB free.`

**Must set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to reduce fragmentation during loading. Also use `low_cpu_mem_usage=True` in `from_pretrained()`.

**OOM traceback (6× A100, for reference):**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 GiB.
GPU 2 has a total capacity of 79.25 GiB of which 6.08 GiB is free.
→ _move_missing_keys_from_meta_to_device needs extra VRAM during weight conversion
```

**Fix:** Use 8× A100 80GB ($11.92/hr) or set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation (may help on 6 GPUs but not guaranteed).

**Download speed on RunPod:** ~266 MB/s with `HF_TRANSFER=1` and `max_workers=8` on a 3.6Gbps network. 440GB takes ~28 min. Run `huggingface_hub.snapshot_download()` to `/workspace/models` (network volume).

**Transformers version:** 5.13.0+ required (installed via `pip install transformers accelerate bitsandbytes`). PyTorch 2.12.1 installs alongside. Use `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` image as base.

**NVIDIA NVFP4 alternative:** `nvidia/GLM-5.2-NVFP4` (381GB) — requires Blackwell GPUs (B200/B300). NOT compatible with A100. Only use if you have Blackwell hardware.

**Why not bitsandbytes 4-bit:** AWQ-INT4 preserves quality better than bitsandbytes 4-bit (pre-quantized by cyankiwi with STEM calibration). bitsandbytes would require loading the full BF16 model first (1.5TB), which doesn't fit on 6× A100.

**Dependency issues encountered (Jul 4, 2026):**
- `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` ships with torch 2.1.0+cu118, torchaudio 2.1.0, torchvision 0.16.0 — all cu118. Installing transformers 5.13.0 pulls torch 2.12.1 (cu13), creating torchaudio/torchvision version mismatch. **Fix:** `pip install --upgrade torchaudio torchvision` after installing transformers.
- AWQ-INT4 models use `compressed-tensors` quantization format. **Fix:** `pip install compressed-tensors` (not installed by default).
- `huggingface_hub` may not be installed on the base image. **Fix:** `pip install huggingface_hub hf_transfer` before downloading.
- **transformers 5.x raises `RuntimeError` on weight conversion warnings** — AWQ-INT4 has minor weight conversion notes that transformers 5.13 treats as fatal. **Fix:** monkey-patch `log_state_dict_report` before calling `from_pretrained()`:
  ```python
  import transformers.utils.loading_report as lr
  original = lr.log_state_dict_report
  lr.log_state_dict_report = lambda *a, **kw: print("[suppressed]")
  model = AutoModelForCausalLM.from_pretrained(...)
  lr.log_state_dict_report = original  # restore
  ```
- **`torch_dtype` is deprecated in transformers 5.x** — use `dtype=` instead. Using `torch_dtype=` produces a deprecation warning but still works.
- **HF Xet I/O errors during 440GB download** — `RuntimeError: Task error: File reconstruction error: IO Error (Input/output error (os error 5))`. Just re-run `snapshot_download()` — it resumes from cache for completed shards. Happened at 68/83 shards on one run.
- **transformers 5.x raises `RuntimeError` on AWQ weight conversion warnings** — the `_finalize_model_loading` → `log_state_dict_report` path raises on any CONVERSION entries. **Fix:** Monkey-patch `log_state_dict_report` to suppress:
  ```python
  import transformers.utils.loading_report as lr
  original = lr.log_state_dict_report
  lr.log_state_dict_report = lambda *a, **kw: None  # suppress
  model = AutoModelForCausalLM.from_pretrained(...)
  lr.log_state_dict_report = original  # restore
  ```
- **RunPod pod volumes are ephemeral** — terminating a pod loses its `/workspace` volume. When switching from 6× to 8× A100 pods, the 440GB model had to be re-downloaded (~28 min). Use RunPod **network volumes** (persistent across pods) if you expect to switch pods.
- **RunPod REST API vs GraphQL:** Use REST API (`POST https://rest.runpod.io/v1/pods`) for pod creation — GraphQL field names are inconsistent (`gpuTypeIds` vs `gpuTypeIdList`, `gpuTypePriority` vs custom). REST API is cleaner and well-documented. Pass your local SSH public key via `env.PUBLIC_KEY` in the request body or SSH won't authenticate.

**Complete deps install command:**
```bash
pip install transformers accelerate bitsandbytes safetensors sentencepiece hf_transfer huggingface_hub compressed-tensors
# Fix version mismatches from base image
pip install --upgrade torchaudio torchvision
# ⛔ CRITICAL: reinstall torch for CUDA 12.8 (RunPod driver is 570.x = CUDA 12.8)
# Without this, torch.cuda.is_available() returns False and model loads to CPU only
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verify CUDA works before launching experiment:**
```bash
python3 -c "import torch; print(f'cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}')"
# Must print cuda_available=True. If False, torch was compiled for wrong CUDA version.
```

**Diagnostic:** If all GPUs show 0 MiB in `nvidia-smi` but the model "loaded" without errors, you have the CUDA mismatch. The model loaded to CPU RAM instead of GPU VRAM.

## Why GGUF Can't Do RYS Layer Surgery

RYS requires two things that GGUF/llama.cpp cannot provide:

1. **Hidden state access:** Phase 1 (anatomy mapping) needs `output_hidden_states=True` — the ability to collect intermediate layer activations. This is a transformers API; llama.cpp does not expose intermediate layer hidden states through its API.

2. **Layer list manipulation:** RYS duplicates layers by inserting copies into `model.model.layers` (a `nn.ModuleList`). GGUF is a flat binary format with tensors stored as sequential blocks. To duplicate a layer, you'd need to:
   - Manually duplicate the tensor block in the binary
   - Renumber all subsequent layer references
   - Update the metadata (offset table, tensor count)
   - Hope the architecture-specific loader handles the new layer count

   This is brittle and architecture-specific. The `gguf` Python library has no API for layer insertion.

## IndexShare DSA Complication

Even in transformers (where layer surgery is straightforward), GLM-5.2's IndexShare pattern creates a new challenge:

- **Duplicating a "full" indexer layer** breaks the 4-layer sharing pattern. The next 3 layers expect to share this indexer, but the duplicated layer would create a second "full" indexer in the same block.
- **Duplicating a "shared" indexer layer** may also break the pattern — the shared indexer is computed once and reused, so duplicating a shared-indexer layer means it would reference an indexer that may not be recomputed.
- **Qwen1.5-MoE (our validated testbed) does not have IndexShare** — it has standard attention, so duplicating any layer is safe.

**Key insight (Jul 4 2026):** At short context (<100 tokens, like our math probe), the DSA indexer's topk=2048 is effectively a no-op — you can't sparse-select 2048 tokens from 50. This means the sweep results at short context are likely valid even if the IndexShare pattern is disrupted. However, these results may NOT transfer to long-context performance where DSA actually matters.

**Mitigation strategies (implemented in sweep script):**
1. **Tier 1 (safest):** Duplicate entire 4-layer blocks (preserves the full→shared→shared→shared pattern)
2. **Tier 2 (moderate):** Duplicate individual shared-indexer layers (positions 1,2,3 within each 4-block) — these reference an indexer from the full layer, so duplicating them is less disruptive
3. **Tier 3 (risky):** Duplicate individual full-indexer layers (every 4th) — only test a few to see what happens. Duplicating creates two consecutive "full" indexer layers.

## NVFP4 Quantization (NVIDIA Official)

`nvidia/GLM-5.2-NVFP4` is NVIDIA's official 4-bit quantization of GLM-5.2, released Jun 2026. Key properties:

- **Size:** ~381GB (safetensors) — fits on 6× A100 80GB (480GB VRAM) with ~100GB headroom for KV cache
- **Shared expert NOT quantized** — only routed expert linear ops quantized to NVFP4
- **Near-lossless:** GPQA Diamond 89.39 (vs 89.52 FP8 baseline), all benchmarks within 1% of FP8
- **Hardware requirement:** NVIDIA Blackwell (B200, B300) for full NVFP4 support
- **Serving:** vLLM 0.23.0+ or SGLang with `--quantization modelopt_fp4`
- **RYS compatibility:** Not directly usable for RYS — NVFP4 quantizes expert weights, and the Blackwell requirement limits availability. For RYS sweep, use `bitsandbytes` 4-bit on the original BF16 model instead (works on A100, allows layer surgery in transformers).

**For RYS sweep:** Load `zai-org/GLM-5.2` (BF16) with `bitsandbytes` 4-bit quantization via `device_map="auto"` across 6-8 GPUs. This gives ~377GB model footprint with full transformers API access (hidden states, layer manipulation). NVFP4 is for production serving, not for surgery experiments.

## IndexShare DSA Short-Context Safety Argument

The RYS math probe uses very short contexts (~50 tokens per question). At this length, the DSA indexer's `index_topk=2048` is effectively a no-op — you cannot sparse-select 2048 tokens from a 50-token sequence. The attention pattern degenerates to dense attention regardless of indexer state.

**Implication for RYS sweep:** Duplicating individual layers — even "full" indexer layers at positions 0, 4, 8, 12... — should not crash or silently degrade at short context. The indexer reuse pattern only becomes meaningful at long context (thousands+ tokens). The sweep results will be valid for short-context math reasoning, but may not transfer to long-context performance where IndexShare matters.

**Caveat:** This is a theoretical argument, not empirically tested. The transformers implementation of `GlmMoeDsaForCausalLM` may still assert 4-layer periodicity regardless of context length. Start with 4-layer block duplication as the safe path, then try individual layers.

## Compute Requirements for Full GLM-5.2 RYS Sweep

### Approach 1: RunPod Multi-GPU (DEPLOYED Jul 4, 2026)

**Hardware:** 8× A100 80GB SXM on RunPod Secure Cloud (NOT 6× — 6× OOMs)
**Cost:** $11.92/hr ($1.49/GPU/hr × 8)
**Image:** `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
**Model:** `cyankiwi/GLM-5.2-AWQ-INT4` (440GB AWQ 4-bit, MIT license)
**Disk:** 1000GB volume for model cache + results
**Deps:** `pip install transformers accelerate bitsandbytes safetensors sentencepiece hf_transfer huggingface_hub`

**Timeline:**
- Model download: ~28 min (440GB at ~266 MB/s with HF_TRANSFER=1)
- Deps install: ~2 min
- Phase 1 (anatomy): ~5 min
- Phase 2 (baseline): ~5 min
- Phase 3 (sweep): 98 configs × ~2-3 min each = ~4-5 hours
- **Total: ~6 hours, ~$72 cost**

**Sweep config (98 total):**
- 17 × 4-layer blocks (Tier 1, IndexShare-safe)
- 54 × single shared-indexer layers (Tier 2)
- 8 × single full-indexer layers (Tier 3, risky)

**RunPod pod creation (REST API — tested and working Jul 4, 2026):**
```bash
# IMPORTANT: Use REST API (rest.runpod.io/v1/pods), NOT GraphQL — GraphQL field names
# are different (gpuTypeIds vs gpuTypeIdList) and inconsistent.
# Pass your local public key via env.PUBLIC_KEY or SSH won't work.
curl -s --request POST \
  --url https://rest.runpod.io/v1/pods \
  --header "Authorization: Bearer $RUNPOD_KEY" \
  --header "Content-Type: application/json" \
  --data '{
  "cloudType": "SECURE", "computeType": "GPU",
  "name": "rys-glm52-sweep",
  "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  "gpuCount": 6, "gpuTypeIds": ["NVIDIA A100-SXM4-80GB"],
  "gpuTypePriority": "availability",
  "containerDiskInGb": 100, "volumeInGb": 1000,
  "volumeMountPath": "/workspace",
  "ports": ["22/tcp"], "supportPublicIp": true,
  "minRAMPerGPU": 16, "minVCPUPerGPU": 4,
  "env": {"PUBLIC_KEY": "'"$(cat ~/.ssh/id_ed25519.pub)"'"}
}'
```

**Getting SSH access:** Pod takes ~2 min to boot. Poll `GET https://rest.runpod.io/v1/pods/$POD_ID` for `publicIp` and `portMappings.22`. SSH with `ssh -i ~/.ssh/id_ed25519 root@$IP -p $PORT`.

**Model download (tested Jul 4, 2026):**
```bash
# 440GB at ~266 MB/s on RunPod's 3.6Gbps network = ~28 min
# HF_TRANSFER=1 for max speed, max_workers=8 for parallelism
# May hit Xet I/O errors on individual shards — just re-run, resumes from cache
export HF_TRANSFER=1
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='cyankiwi/GLM-5.2-AWQ-INT4',
    cache_dir='/workspace/models',
    max_workers=8
)
print(f'Done: {path}')
"
```

**Sweep script:** See `templates/rys_glm52_experiment.py` — copy to pod and run with `PYTHONUNBUFFERED=1 python -u rys_glm52_experiment.py`. The script handles all 3 tiers of IndexShare safety (4-layer blocks, shared-indexer singles, full-indexer singles).

**Alternative (faster, more expensive):**
- 8× H200 141GB on RunPod = $35.12/hr
- BF16 (1.5TB) fits across 8× 141GB = 1,128GB
- ~3hr sweep, ~$105 total
- Avoids quantization concerns entirely

**Deployment steps (tested Jul 4, 2026):**
1. Create pod via RunPod REST API: `POST https://rest.runpod.io/v1/pods` with `gpuCount=8`, `gpuTypeIds=["NVIDIA A100-SXM4-80GB"]`, `containerDiskInGb=100`, `volumeInGb=1000`. Pass `env.PUBLIC_KEY` with your SSH public key.
2. Wait ~2 min, poll `GET /v1/pods/$POD_ID` for `publicIp` and `portMappings.22`. SSH: `ssh -i ~/.ssh/id_ed25519 root@$IP -p $PORT`
3. Install deps: `pip install transformers accelerate bitsandbytes safetensors sentencepiece hf_transfer huggingface_hub compressed-tensors && pip install --upgrade torchaudio torchvision`
4. Download model: `HF_TRANSFER=1 python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cyankiwi/GLM-5.2-AWQ-INT4', cache_dir='/workspace/models', max_workers=8)"` (~28 min, 440GB)
5. Copy `templates/rys_glm52_experiment.py` to pod. Run: `PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u rys_glm52_experiment.py`
6. **Pod volumes are ephemeral** — if you need to switch pods, use a RunPod network volume to persist the model download

### Approach 2: GGUF Surgery (Not Viable)

- Skip Phase 1 anatomy (we already know the pattern from Qwen1.5-MoE)
- Manually duplicate tensor blocks in a 1-bit GGUF file
- Run math probe via llama.cpp on a 256GB RAM cloud box
- **Risk:** IndexShare pattern likely breaks, 1-bit quantization may not preserve the subtle reasoning circuit differences RYS relies on, no hidden state access for anatomy mapping

### Approach 3: Z.ai API (Baseline Only)

- Run math probe baseline against the hosted GLM-5.2 API (free)
- Can't do surgery — API is a black box, no layer manipulation
- Useful for establishing baseline before cloud sweep

## Recommendation

**Qwen1.5-MoE sweep is COMPLETE (Jul 4, 2026)** — RYS on MoE is confirmed. 10/36 configs improved (+6.25%), encoding layers catastrophic (-50%), reasoning layers consistently help. See `references/rys-layer-duplication.md` → "Phase 2-3 Results" for full data.

1. **Go with Approach 1 (cloud multi-GPU)** — rent 8× A100 80GB, load in 4-bit, run full 78-layer sweep in transformers. Cost ~$75 for ~3hr sweep.
2. **Handle IndexShare carefully** — start by duplicating 4-layer blocks (preserves DSA pattern) rather than individual layers. If that works, try individual layers at positions 2-3 within a block (shared indexers, not block boundaries).
3. **Expected outcome:** Based on Qwen1.5-MoE results (24 layers → 4 winning single-layer positions + 6 winning blocks), GLM-5.2 (78 layers) should have ~15-20 winning single-layer positions and a broad winning block region in L25-L60. The reasoning region should start after the dense layers (L0-2) and end before the final decoding layers (~L70-77).
4. **Scale consideration:** GLM-5.2's 256 experts (vs Qwen's 60) and 8 active/tok (vs 4) mean each duplicated layer adds more compute. A +4-layer block (best config on Qwen) would add ~5% to inference cost on GLM-5.2 — still very efficient.

## Source Links

- Model: [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)
- GGUF: [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF)
- Blog: [z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2)
- Technical Report: [arXiv:2602.15763](https://arxiv.org/abs/2602.15763)
- IndexShare paper: [arXiv:2603.12201](https://arxiv.org/abs/2603.12201)
