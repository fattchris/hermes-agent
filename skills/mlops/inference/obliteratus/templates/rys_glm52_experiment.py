#!/usr/bin/env python3
"""
RYS Layer Duplication Experiment on GLM-5.2 MoE
=================================================
Target: GLM-5.2 (78 layers, 256 experts, 8 active/tok, 1 shared expert, IndexShare DSA)
Model: cyankiwi/GLM-5.2-AWQ-INT4 (440GB, 4-bit AWQ quantization)
Hardware: 8× A100 80GB (640GB VRAM total) on RunPod

Strategy:
- Phase 1: Anatomy mapping (centered cosine similarity) — 4 EN/ZH inputs
- Phase 2: Baseline math probe (16 questions)
- Phase 3: Layer duplication sweep
  - First: 4-layer blocks (preserves IndexShare pattern)
  - Then: Individual layers at shared-indexer positions (pos 2,3 in each 4-block)
  - Skip full-indexer layers (0,4,8,12,...) for individual duplication
- Phase 4: Results summary

Key: IndexShare DSA — every 4 layers share an indexer.
  full = layers 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76
  shared = all others (positions 1,2,3 within each 4-block)
  Safe to duplicate: shared-indexer layers (pos 1,2,3 in each block)
  Risky: full-indexer layers (every 4th) — duplicating breaks the pattern
  Safest: entire 4-layer blocks (preserves pattern completely)

Prerequisites on RunPod pod:
  pip install transformers accelerate bitsandbytes safetensors sentencepiece hf_transfer huggingface_hub compressed-tensors
  pip install --upgrade torchaudio torchvision
  pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  # Download model:
  export HF_TOKEN=hf_xxx HF_TRANSFER=1
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='cyankiwi/GLM-5.2-AWQ-INT4', cache_dir='/workspace/models', max_workers=8, token='$HF_TOKEN')"

Run:
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u rys_glm52_experiment.py
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
import sys
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "cyankiwi/GLM-5.2-AWQ-INT4"
MODEL_PATH = "/workspace/models/models--cyankiwi--GLM-5.2-AWQ-INT4/snapshots"
DEVICE = "cuda"
DTYPE = torch.float16  # AWQ uses float16 compute

# Find the actual snapshot path
def find_model_path():
    if os.path.exists(MODEL_PATH):
        for d in os.listdir(MODEL_PATH):
            full = os.path.join(MODEL_PATH, d)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
                return full
    return MODEL_ID

# ============================================================
# IndexShare DSA Configuration
# ============================================================
NUM_LAYERS = 78
FULL_INDEXER_LAYERS = set(range(0, NUM_LAYERS, 4))  # 0,4,8,...,76
SHARED_INDEXER_LAYERS = set(range(NUM_LAYERS)) - FULL_INDEXER_LAYERS

# ============================================================
# Phase 1: Centered Cosine Similarity (Three-Phase Anatomy)
# ============================================================

INPUTS = [
    "The process of photosynthesis converts light energy into chemical energy, which is stored in glucose molecules.",
    "光合作用将光能转化为化学能，储存在葡萄糖分子中。",
    "At dusk, the moon pours silver on the tide, and the wind carries a quiet song.",
    "黄昏时，月亮把银辉洒在潮汐上，风里带着一首安静的歌。",
]

COMPARISONS = [
    ("EN_fact↔EN_poem", 0, 2),
    ("ZH_fact↔ZH_poem", 1, 3),
    ("EN_fact↔ZH_fact", 0, 1),
    ("EN_poem↔ZH_poem", 2, 3),
    ("EN_fact↔ZH_poem", 0, 3),
    ("EN_poem↔ZH_fact", 2, 1),
]

def collect_hidden_states(model, tokenizer, texts):
    all_hidden = []
    # CRITICAL: with device_map="auto", the model's first layer may not be on cuda:0.
    # Use next(model.parameters()).device to find the correct input device.
    model_device = next(model.parameters()).device
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model_device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pooled = []
        for hs in hidden_states:
            pooled.append(hs.mean(dim=1).squeeze(0).float().cpu().numpy())
        all_hidden.append(np.stack(pooled))
        del outputs, hidden_states
        torch.cuda.empty_cache()
    return all_hidden

def compute_centered_cosine(all_hidden):
    num_inputs = len(all_hidden)
    num_layers = all_hidden[0].shape[0]
    stacked = np.stack(all_hidden)
    centered = np.zeros_like(stacked)
    for l in range(num_layers):
        mean_vec = stacked[:, l, :].mean(axis=0)
        for i in range(num_inputs):
            v = stacked[i, l, :] - mean_vec
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                centered[i, l, :] = v / norm
            else:
                centered[i, l, :] = v
    results = {}
    for name, i, j in COMPARISONS:
        sims = []
        for l in range(num_layers):
            cos_sim = np.dot(centered[i, l, :], centered[j, l, :])
            sims.append(cos_sim)
        results[name] = sims
    return results

def print_anatomy(results, num_layers):
    print("\n" + "=" * 70)
    print("PHASE 1: Three-Phase Anatomy (Centered Cosine Similarity)")
    print("=" * 70)
    
    cross_lang_same = []
    same_lang_diff = []
    for l in range(num_layers):
        cross_lang_same.append(np.mean([results["EN_fact↔ZH_fact"][l], results["EN_poem↔ZH_poem"][l]]))
        same_lang_diff.append(np.mean([results["EN_fact↔EN_poem"][l], results["ZH_fact↔ZH_poem"][l]]))
    
    print("\n--- Centered Similarity by Layer ---")
    for l in range(0, num_layers, 2):
        tag = "FULL" if l in FULL_INDEXER_LAYERS else "shrd"
        print(f"  L{l:2d} [{tag}]: cross-same={cross_lang_same[l]:+.3f}  same-diff={same_lang_diff[l]:+.3f}")
    
    reasoning_layers = [l for l in range(num_layers) if cross_lang_same[l] > same_lang_diff[l]]
    if reasoning_layers:
        print(f"\n→ Reasoning region: L{reasoning_layers[0]}-{reasoning_layers[-1]} ({len(reasoning_layers)} layers)")
    else:
        print("\n→ No clear reasoning region found")
    
    return reasoning_layers

# ============================================================
# Phase 2: Math Probe
# ============================================================

MATH_QUESTIONS = [
    ("What is the square root of 144?", "12"),
    ("What is the square root of 256?", "16"),
    ("What is 17 multiplied by 23?", "391"),
    ("What is 13 times 14?", "182"),
    ("What is the cube root of 27?", "3"),
    ("What is the cube root of 64?", "4"),
    ("What is 15% of 200?", "30"),
    ("What is 7 squared plus 8 squared?", "113"),
    ("What is 100 divided by 4?", "25"),
    ("What is 9 factorial divided by 8 factorial?", "9"),
    ("What is 3 to the power of 5?", "243"),
    ("What is the sum of the first 10 natural numbers?", "55"),
    ("What is 144 divided by 12?", "12"),
    ("What is 25% of 80?", "20"),
    ("What is 11 times 12?", "132"),
    ("What is the square root of 625?", "25"),
]

def math_probe(model, tokenizer, questions=MATH_QUESTIONS, max_new_tokens=100):
    correct = 0
    total = len(questions)
    details = []
    for q, expected in questions:
        prompt = f"Answer this math question with only the number, no explanation.\n\nQuestion: {q}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        is_correct = expected in response
        if is_correct:
            correct += 1
        details.append({"q": q, "expected": expected, "got": response[:100], "correct": is_correct})
        status = "✓" if is_correct else "✗"
        print(f"  {status} Q: {q[:50]:50s} → {response[:30]:30s} (expected: {expected})")
        del output
        torch.cuda.empty_cache()
    return correct, total, details

# ============================================================
# Phase 3: Layer Duplication (Pointer-Based, No Copy)
# ============================================================

def duplicate_layers(model, layer_indices):
    """Insert duplicate layers (pointer-based, zero extra VRAM) at their original position."""
    layers = list(model.model.layers)
    new_layers = []
    insert_set = set(layer_indices)
    
    for i, layer in enumerate(layers):
        new_layers.append(layer)  # original
        if i in insert_set:
            new_layers.append(layer)  # duplicate (same object = pointer-based)
    
    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    return len(new_layers) - len(layers)  # extra layers added

def load_model():
    """Load GLM-5.2 AWQ-INT4 with multi-GPU device map."""
    model_path = find_model_path()
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # transformers 5.x raises RuntimeError on weight conversion warnings — suppress
    import transformers.utils.loading_report as lr
    original_raise = lr.log_state_dict_report
    def patched_log(*args, **kwargs):
        print("  [suppressed weight conversion warning]")
    lr.log_state_dict_report = patched_log
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    
    lr.log_state_dict_report = original_raise
    model.eval()
    
    n_layers = len(model.model.layers)
    print(f"Model loaded: {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: {mem:.1f} GB used")
    
    return model, tokenizer

# ============================================================
# Main Experiment
# ============================================================

def main():
    results = {
        "model": MODEL_ID,
        "num_layers": NUM_LAYERS,
        "full_indexer_layers": sorted(FULL_INDEXER_LAYERS),
        "shared_indexer_layers": sorted(SHARED_INDEXER_LAYERS),
        "configs": [],
    }
    
    # ---- Phase 1: Anatomy ----
    print("=" * 70)
    print("LOADING MODEL FOR PHASE 1 (Anatomy)")
    print("=" * 70)
    model, tokenizer = load_model()
    
    print("\nCollecting hidden states...")
    all_hidden = collect_hidden_states(model, tokenizer, INPUTS)
    anatomy_results = compute_centered_cosine(all_hidden)
    reasoning_layers = print_anatomy(anatomy_results, NUM_LAYERS)
    results["anatomy_reasoning_layers"] = reasoning_layers
    results["anatomy_data"] = {k: v for k, v in anatomy_results.items()}
    
    # ---- Phase 2: Baseline ----
    print("\n" + "=" * 70)
    print("PHASE 2: Baseline Math Probe")
    print("=" * 70)
    baseline_correct, baseline_total, baseline_details = math_probe(model, tokenizer)
    baseline_score = baseline_correct / baseline_total
    print(f"\nBaseline: {baseline_correct}/{baseline_total} = {baseline_score:.4f}")
    results["baseline"] = baseline_score
    results["baseline_details"] = baseline_details
    
    with open("/workspace/rys_glm52_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ---- Phase 3: Sweep ----
    print("\n" + "=" * 70)
    print("PHASE 3: Layer Duplication Sweep")
    print("=" * 70)
    
    configs = []
    
    # TIER 1: 4-layer blocks (safest — preserves IndexShare pattern)
    for start in range(4, 72, 4):
        end = start + 4
        if end <= NUM_LAYERS:
            configs.append({
                "name": f"block4({start},{end})",
                "layers": list(range(start, end)),
                "tier": "4-layer-block"
            })
    
    # TIER 2: Individual shared-indexer layers (pos 1,2,3 in each 4-block)
    for l in sorted(SHARED_INDEXER_LAYERS):
        if 3 <= l <= 74:
            configs.append({
                "name": f"layer{l}_x2",
                "layers": [l],
                "tier": "single-shared"
            })
    
    # TIER 3: A few full-indexer layers (risky — breaks IndexShare pattern)
    for l in [8, 16, 24, 32, 40, 48, 56, 64]:
        configs.append({
            "name": f"layer{l}_x2_FULL",
            "layers": [l],
            "tier": "single-full-indexer"
        })
    
    print(f"Total configs to test: {len(configs)}")
    print(f"  Tier 1 (4-layer blocks): {sum(1 for c in configs if c['tier']=='4-layer-block')}")
    print(f"  Tier 2 (single shared): {sum(1 for c in configs if c['tier']=='single-shared')}")
    print(f"  Tier 3 (single full): {sum(1 for c in configs if c['tier']=='single-full-indexer')}")
    
    for i, config in enumerate(configs):
        name = config["name"]
        layers_to_dup = config["layers"]
        tier = config["tier"]
        
        print(f"\n--- Config {i+1}/{len(configs)}: {name} (tier: {tier}) ---")
        t0 = time.time()
        
        try:
            model, tokenizer = load_model()
            extra = duplicate_layers(model, layers_to_dup)
            new_count = len(model.model.layers)
            print(f"  Duplicated layers {layers_to_dup}, +{extra} layers → {new_count} total")
            
            correct, total, details = math_probe(model, tokenizer)
            score = correct / total
            delta = score - baseline_score
            t1 = time.time()
            
            mark = "★" if delta > 0 else ("-" if delta < 0 else "=")
            print(f"  {mark} {name:30s} +{extra}  {score:.4f}  {delta:+.4f}  ({int(t1-t0)}s)")
            
            results["configs"].append({
                "config": name, "tier": tier,
                "layers_duplicated": layers_to_dup,
                "extra_layers": extra, "total_layers": new_count,
                "score": score, "delta": delta,
                "time_seconds": int(t1 - t0), "details": details,
            })
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results["configs"].append({
                "config": name, "tier": tier,
                "layers_duplicated": layers_to_dup, "error": str(e),
            })
            torch.cuda.empty_cache()
        
        with open("/workspace/rys_glm52_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        if 'model' in dir():
            del model
            gc.collect()
            torch.cuda.empty_cache()
    
    # ---- Phase 4: Summary ----
    print("\n" + "=" * 70)
    print("PHASE 4: Results Summary")
    print("=" * 70)
    print(f"\nBaseline: {baseline_score:.4f}")
    
    valid = [c for c in results["configs"] if "score" in c]
    sorted_configs = sorted(valid, key=lambda x: x.get("delta", -999), reverse=True)
    
    print("\nTop 10:")
    for c in sorted_configs[:10]:
        print(f"  {c['config']:35s} +{c.get('extra_layers',0)}L | score={c['score']:.4f} | delta={c['delta']:+.4f} | {c['tier']}")
    
    print("\nBottom 10:")
    for c in sorted_configs[-10:]:
        print(f"  {c['config']:35s} +{c.get('extra_layers',0)}L | score={c['score']:.4f} | delta={c['delta']:+.4f} | {c['tier']}")
    
    improved = sum(1 for c in valid if c.get("delta", 0) > 0)
    degraded = sum(1 for c in valid if c.get("delta", 0) < 0)
    unchanged = sum(1 for c in valid if c.get("delta", 0) == 0)
    print(f"\n→ {improved} improved, {degraded} degraded, {unchanged} unchanged")
    
    if valid:
        best = max(valid, key=lambda x: x.get("delta", -999))
        print(f"\n🎯 Best: {best['config']} (+{best.get('extra_layers',0)} layers)")
        print(f"   Score: {best['score']:.4f} (base: {baseline_score:.4f})")
        print(f"   Delta: {best['delta']:+.4f}")
    
    with open("/workspace/rys_glm52_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /workspace/rys_glm52_results.json")

if __name__ == "__main__":
    main()
