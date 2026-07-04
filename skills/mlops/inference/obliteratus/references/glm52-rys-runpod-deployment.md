# GLM-5.2 RYS RunPod Deployment Guide

**Created:** Jul 4, 2026
**Model:** cyankiwi/GLM-5.2-AWQ-INT4 (440GB, AWQ 4-bit)
**Hardware:** 8× A100 80GB SXM on RunPod ($11.92/hr)

## Why AWQ-INT4 Instead of FP8 or Bitsandbytes?

| Format | Size | VRAM Needed (8×80=640GB) | Layer Surgery? |
|---|---|---|---|
| BF16 | 1.51 TB | ❌ No | ✅ |
| FP8 (zai-org) | 695 GB | ❌ No (needs 8× H200) | ✅ |
| AWQ-INT4 (cyankiwi) | 440 GB | ✅ Yes (440GB weights + headroom for activations) | ✅ |
| bitsandbytes 4-bit | ~377 GB | ✅ Yes | ⚠️ AWQ preferred — better quality, native support |
| GGUF (unsloth IQ1_S) | 217 GB | ✅ Yes | ❌ No hidden states, no layer manipulation |

**AWQ-INT4 is the sweet spot:** fits 8× A100 80GB with headroom for forward pass activations, preserves near-FP8 quality (GPQA 89.39 vs 89.52), and works with transformers `device_map="auto"` for layer surgery.

**⚠️ 6× A100 80GB (480GB) OOMs** — the 411GB model fits in VRAM but transformers needs ~30GB extra for weight conversion temp tensors during `from_pretrained()`. Loading reaches 100% but the process crashes silently on the first forward pass (no room for activation tensors). Use 8× A100 80GB (640GB) minimum. Rule of thumb: `model_size × 1.5 ≤ total_VRAM`.

## RunPod Pod Creation (REST API)

**Use the REST API, not GraphQL.** The REST endpoint at `rest.runpod.io/v1/pods` is simpler and handles SSH key injection via the `env.PUBLIC_KEY` field.

```bash
curl -s --request POST \
  --url https://rest.runpod.io/v1/pods \
  --header "Authorization: Bearer $RUNPOD_KEY" \
  --header "Content-Type: application/json" \
  --data '{
  "cloudType": "SECURE",
  "computeType": "GPU",
  "name": "rys-glm52-sweep",
  "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
  "gpuCount": 8,
  "gpuTypeIds": ["NVIDIA A100-SXM4-80GB"],
  "gpuTypePriority": "availability",
  "containerDiskInGb": 100,
  "volumeInGb": 1000,
  "volumeMountPath": "/workspace",
  "ports": ["22/tcp"],
  "supportPublicIp": true,
  "minRAMPerGPU": 16,
  "minVCPUPerGPU": 4,
  "env": {"PUBLIC_KEY": "<your-ed25519-pubkey>"}
}'
```

### SSH Key Gotcha

The `PUBLIC_KEY` env var must contain the **full public key string** including the key type prefix and comment:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA... zero@macmini
```

If the key doesn't match a local private key, SSH will fail with `Permission denied (publickey,password)`. Retrieve your local public key with `cat ~/.ssh/id_ed25519.pub` and pass it in the env var.

### Checking Pod Status

```bash
# REST API — returns publicIp and portMappings
curl -s --request GET \
  --url "https://rest.runpod.io/v1/pods/$POD_ID" \
  --header "Authorization: Bearer $RUNPOD_KEY"
```

The `portMappings` field maps container ports to public ports: `{"22": 10090}`. SSH with:
```bash
ssh -i ~/.ssh/id_ed25519 root@$PUBLIC_IP -p $PUBLIC_PORT
```

### Pod Specs (8× A100 80GB SXM)

| Spec | Value |
|---|---|
| GPU | 8× NVIDIA A100-SXM4-80GB (81920 MiB each) |
| Total VRAM | 640 GB |
| System RAM | 1501 GB |
| vCPUs | 96 |
| Disk | 100GB container + 1000GB /workspace volume |
| Cost | $11.92/hr |
| Data center | US-MD-1 |

## Dependency Installation

```bash
pip install --upgrade pip
pip install transformers accelerate bitsandbytes safetensors sentencepiece hf_transfer huggingface_hub compressed-tensors
# Fix version mismatches from base image (torch 2.1.0+cu118 → torch 2.12.1)
pip install --upgrade torchaudio torchvision
```

**Three dep issues encountered in production (Jul 4, 2026):**

1. **`huggingface_hub` not installed on base image** — the `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` image doesn't include it. If you run `snapshot_download()` before installing deps, you get `ModuleNotFoundError: No module named 'huggingface_hub'`. Install it explicitly.

2. **torchaudio/torchvision version mismatch** — the base image ships with torch 2.1.0+cu118, torchaudio 2.1.0, torchvision 0.16.0 (all cu118). Installing transformers 5.13.0 pulls torch 2.12.1 (cu13), but leaves old torchaudio/torchvision. When transformers tries to `import torchaudio` (for RNNT loss support), it crashes with `OSError: Could not load this library: libtorchaudio.so`. **Fix:** `pip install --upgrade torchaudio torchvision` after installing transformers.

3. **`compressed-tensors` not installed by default** — AWQ-INT4 models use the `compressed-tensors` quantization format. Without the package, `AutoModelForCausalLM.from_pretrained()` crashes with `ImportError: compressed-tensors>=0.15.0 is required`. **Fix:** `pip install compressed-tensors` (0.17.1+).

## Model Download

```bash
export HF_TRANSFER=1
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='cyankiwi/GLM-5.2-AWQ-INT4',
    cache_dir='/workspace/models',
    max_workers=8
)
print(f'Downloaded to: {path}')
"
```

**Download speed:** ~266 MB/s on RunPod's internal network. 440GB takes ~28 minutes.

**Xet I/O errors:** The HF Xet downloader may hit `RuntimeError: File reconstruction error: IO Error: Input/output error (os error 5)` on individual shards. This is transient. **Fix:** Just re-run `snapshot_download()` — it resumes from cache, only downloading the missing shards. The 440GB download completed successfully after one retry (68/83 shards on first attempt, 83/83 on retry).

**HF_TRANSFER rate limiting:** If you see a `Warning: You are sending unauthenticated requests to the HF Hub`, set `HF_TOKEN` to enable higher rate limits and faster downloads.

**Gotcha:** `huggingface_hub` is NOT installed by the base PyTorch template. Install it explicitly before running the download script. The `transformers` install pulls it as a dependency, but if you run the download in a separate command before pip finishes, it won't be available yet.

**Progress monitoring:** Check shard count:
```bash
ls /workspace/models/models--cyankiwi--GLM-5.2-AWQ-INT4/snapshots/*/model-*.safetensors | wc -l
# 83 total shards
```

## IndexShare-Aware Sweep Strategy

GLM-5.2's IndexShare DSA shares an attention indexer across every 4 layers:
```
L0(full) → L1(shared) → L2(shared) → L3(shared) → L4(full) → ...
```

### 3-Tier Safety Hierarchy

| Tier | What | Configs | Risk | Rationale |
|---|---|---|---|---|
| 1 | 4-layer blocks | 17 | ✅ Safe | Preserves full→shared→shared→shared pattern |
| 2 | Single shared-indexer layers | 54 | ⚠️ Low | Duplicating a shared-indexer layer doesn't create a second "full" indexer |
| 3 | Single full-indexer layers | 8 | 🔴 High | Duplicating a full-indexer layer creates back-to-back "full" indexers |

**Full-indexer layers:** 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76 (every 4th, 20 total)

**Shared-indexer layers:** all others (58 total)

### Short-Context Mitigation

At short context (~50 tokens, as used in math probes), the DSA indexer's `index_topk=2048` is almost a no-op — you can't sparse-select 2048 tokens from 50. So even Tier 3 (full-indexer duplication) will likely produce valid short-context results. However, these results may not transfer to long-context performance where IndexShare actually matters.

### Sweep Config Count

- Tier 1: 17 configs (4-layer blocks at L4-L72, step 4)
- Tier 2: 54 configs (shared-indexer layers L3-L74)
- Tier 3: 8 configs (full-indexer layers L8-L64, sample)
- **Total: 79 configs** × ~2-3 min each = ~4-5 hours

## Cost Estimate

| Component | Time | Cost |
|---|---|---|
| Pod setup + dep install | 10 min | $1.99 |
| Model download | 28 min | $5.57 |
| Phase 1 (anatomy) | 5 min | $0.99 |
| Phase 2 (baseline) | 10 min | $1.99 |
| Phase 3 (sweep, 79 configs) | ~4 hours | $47.68 |
| **Total** | **~5 hours** | **~$58** |

## AWQ + device_map="auto" Loading

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",          # spreads across all 8 GPUs
    trust_remote_code=True,
    dtype=torch.float16,        # AWQ compute dtype (NOT torch_dtype — deprecated in transformers 5.x)
)
```

AWQ models work with `device_map="auto"` out of the box — no special quantization config needed. The model is split across GPUs by layer count, with each GPU getting ~10 layers (78/8).

### ⚠️ Critical: Input tensors must go to the model's first device, NOT just "cuda"

When using `device_map="auto"`, the model is split across multiple GPUs. Input tensors placed with `.to("cuda")` map to `cuda:0`, which may NOT be the device of the first model layer. This causes a **silent crash** — the process dies with no traceback, no OOM in dmesg, and GPUs drop to 0 MiB.

**Always use `next(model.parameters()).device` to find the correct input device:**

```python
# WRONG — may crash silently if cuda:0 isn't where the model's first layer lives
inputs = tokenizer(text, return_tensors="pt").to("cuda")
outputs = model(**inputs)

# CORRECT — gets the device of the first model parameter
model_device = next(model.parameters()).device
inputs = tokenizer(text, return_tensors="pt").to(model_device)
outputs = model(**inputs)
```

This applies to ALL forward passes and `model.generate()` calls — both `collect_hidden_states()` and `math_probe()` must use `model_device`, not a hardcoded `DEVICE = "cuda"` constant.

### ⚠️ Silent crash after model loading (no traceback)

If the model loads to 100% but the process dies immediately with no error output, no OOM in dmesg, and GPUs drop to 0 MiB:

1. **Check device mapping** — the most common cause. `.to("cuda")` doesn't work with `device_map="auto"`. Use `next(model.parameters()).device`.
2. **Check VRAM headroom** — if GPUs are at 80/80 GB after loading, there's no room for forward pass activations (especially `output_hidden_states=True` which stores all 78 hidden state tensors). Use more GPUs or reduce `max_new_tokens`.
3. **Check stdout buffering** — use `PYTHONUNBUFFERED=1` and `python -u` to ensure errors are flushed before the process dies.
4. **Redirect output to file** — `> /workspace/rys_output.log 2>&1` ensures you can read the last output even if the SSH connection drops.

## Pointer-Based Layer Duplication (Zero Extra VRAM)

```python
def duplicate_layers(model, layer_indices):
    layers = list(model.model.layers)
    new_layers = []
    insert_set = set(layer_indices)
    for i, layer in enumerate(layers):
        new_layers.append(layer)       # original
        if i in insert_set:
            new_layers.append(layer)   # duplicate (same object = pointer-based)
    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
```

This inserts the **same nn.Module object** — no weight copy, zero extra VRAM. Only compute + KV cache cost increases. Same technique used by ExLlamaV3 for RYS.

## Model Reload Strategy

Each sweep config reloads the model from cache (~30s on A100 with local SSD). This is faster than trying to undo layer duplication in-place, and avoids any state contamination between configs.

```python
# For each config: reload fresh, duplicate, probe, delete
model, tokenizer = load_model()
extra = duplicate_layers(model, layers_to_dup)
correct, total, details = math_probe(model, tokenizer)
del model
gc.collect()
torch.cuda.empty_cache()
```
