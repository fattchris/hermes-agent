#!/usr/bin/env python3
"""
RYS Layer Duplication Experiment on MoE Model
=============================================
Target: Qwen1.5-MoE-A2.7B (24 layers, 60 experts, 4 per tok, shared experts)
Goal: Test if duplicating reasoning layers improves MoE model performance

Phase 1: Map three-phase anatomy via centered cosine similarity ✅ COMPLETE
Phase 2: Baseline math probe (BLOCKED — MPS too slow for MoE)
Phase 3: Duplicate mid-stack layers, re-probe (BLOCKED — needs CUDA)
Phase 4: Compare deltas

⚠️  WARNING: Do NOT run this on Apple Silicon MPS for MoE models!
    MPS lacks optimized MoE kernels — generate() takes 15+ min per question.
    Use CUDA GPU (RunPod/Modal) or llama.cpp GGUF instead.

    For dense models <7B on MPS, this script works fine.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import copy
import time
import sys
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "mps"  # Change to "cuda" for GPU servers
DTYPE = torch.float16

# ============================================================
# Phase 1: Centered Cosine Similarity (Three-Phase Anatomy)
# ============================================================

# Four inputs: EN fact, ZH fact, EN poem, ZH poem (same subjects)
INPUTS = [
    "The process of photosynthesis converts light energy into chemical energy, which is stored in glucose molecules.",
    "光合作用将光能转化为化学能，储存在葡萄糖分子中。",
    "At dusk, the moon pours silver on the tide, and the wind carries a quiet song.",
    "黄昏时，月亮把银辉洒在潮汐上，风里带着一首安静的歌。",
]

# Six pairwise comparisons
COMPARISONS = [
    ("EN_fact↔EN_poem", 0, 2),        # same lang, diff content
    ("ZH_fact↔ZH_poem", 1, 3),        # same lang, diff content
    ("EN_fact↔ZH_fact", 0, 1),        # cross-lang, same content
    ("EN_poem↔ZH_poem", 2, 3),        # cross-lang, same content
    ("EN_fact↔ZH_poem", 0, 3),        # cross-lang, diff content
    ("EN_poem↔ZH_fact", 2, 1),        # cross-lang, diff content
]

def collect_hidden_states(model, tokenizer, texts):
    """Collect pooled hidden states at every layer for each input."""
    all_hidden = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pooled = []
        for hs in hidden_states:
            pooled.append(hs.mean(dim=1).squeeze(0).float().cpu().numpy())
        all_hidden.append(np.stack(pooled))
    return all_hidden

def compute_centered_cosine(all_hidden):
    """Compute centered cosine similarity across layers.
    
    Centering: subtract mean vector across all inputs at each layer, re-normalize.
    This strips out the "I'm at layer N" component and reveals only how
    representations differ from each other.
    """
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
    """Print the centered similarity curves and identify phases."""
    print("\n" + "=" * 70)
    print("PHASE 1: Three-Phase Anatomy (Centered Cosine Similarity)")
    print("=" * 70)
    
    cross_lang_same = []
    same_lang_diff = []
    cross_lang_diff = []
    for l in range(num_layers):
        cross_lang_same.append(np.mean([results["EN_fact↔ZH_fact"][l], results["EN_poem↔ZH_poem"][l]]))
        same_lang_diff.append(np.mean([results["EN_fact↔EN_poem"][l], results["ZH_fact↔ZH_poem"][l]]))
        cross_lang_diff.append(np.mean([results["EN_fact↔ZH_poem"][l], results["EN_poem↔ZH_fact"][l]]))
    
    print("\n--- Centered Similarity by Layer ---")
    for l in range(num_layers):
        bar_cs = "█" * max(0, int(cross_lang_same[l] * 20))
        bar_sl = "▓" * max(0, int(same_lang_diff[l] * 20))
        print(f"  L{l:2d}: cross-same={cross_lang_same[l]:+.3f} {bar_cs}")
        print(f"        same-diff={same_lang_diff[l]:+.3f} {bar_sl}")
    
    reasoning_layers = [l for l in range(num_layers) if cross_lang_same[l] > same_lang_diff[l]]
    if reasoning_layers:
        print(f"\n→ Reasoning region (cross-lang same > same-lang diff): L{reasoning_layers[0]}-{reasoning_layers[-1]}")
    else:
        print("\n→ No clear reasoning region found")
    
    return reasoning_layers

# ============================================================
# Phase 2: Math Probe
# ============================================================

MATH_QUESTIONS = [
    ("What is the square root of 144?", "12"),
    ("What is the square root of 256?", "16"),
    ("What is the square root of 400?", "20"),
    ("What is the square root of 625?", "25"),
    ("What is the square root of 784?", "28"),
    ("What is the square root of 900?", "30"),
    ("What is the square root of 1024?", "32"),
    ("What is the square root of 1089?", "33"),
    ("What is 12 times 13?", "156"),
    ("What is 15 times 17?", "255"),
    ("What is 23 times 7?", "161"),
    ("What is 34 times 12?", "408"),
    ("What is 18 times 19?", "342"),
    ("What is the cube root of 27?", "3"),
    ("What is the cube root of 64?", "4"),
    ("What is the cube root of 125?", "5"),
]

def run_math_probe(model, tokenizer, questions=MATH_QUESTIONS, max_new=50, verbose=True):
    model.eval()
    correct = 0
    total = len(questions)
    for q, expected in questions:
        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        is_correct = expected in response
        if is_correct:
            correct += 1
        if verbose:
            marker = "✓" if is_correct else "✗"
            resp_short = response.strip().replace("\n", " ")[:50]
            print(f"    {marker} {q[:35]:35s} exp={expected:5s} got={resp_short}")
    return correct, total

# ============================================================
# Phase 3: Layer Duplication (pointer-based, zero extra VRAM)
# ============================================================

def load_model():
    """Load fresh model from cache."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    return model

def build_rys_model(layer_start, layer_end):
    """Load fresh model and duplicate layers [start, end).
    
    Uses pointer-based layer sharing: the duplicated layers reference
    the SAME weight tensors as the originals. Zero extra VRAM for
    parameters — only compute + KV cache cost. This matches the
    ExLlamaV3 pointer-based RYS approach.
    """
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    layers = model.model.layers
    # Reference the SAME layer objects (not copies)
    block_to_dup = [layers[i] for i in range(layer_start, layer_end)]
    
    insert_pos = layer_end
    new_layers = list(layers)[:insert_pos] + block_to_dup + list(layers)[insert_pos:]
    model.model.layers = nn.ModuleList(new_layers)
    model.config.num_hidden_layers = len(new_layers)
    
    extra = layer_end - layer_start
    total = len(new_layers)
    
    return model, tokenizer, extra, total

def free_model(model):
    """Force free model memory."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("RYS Layer Duplication Experiment — MoE Edition v2")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    # --- Phase 1: Anatomy ---
    print("\n--- Phase 1: Loading model for anatomy mapping ---")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = load_model()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Experts: {model.config.num_experts}")
    print(f"  Shared experts: {hasattr(model.config, 'shared_expert_intermediate_size')}")
    
    print("\n  Collecting hidden states...")
    hidden_states = collect_hidden_states(model, tokenizer, INPUTS)
    cosine_results = compute_centered_cosine(hidden_states)
    num_layers_model = model.config.num_hidden_layers
    reasoning_layers = print_anatomy(cosine_results, num_layers_model + 1)
    
    # --- Phase 2: Baseline ---
    print("\n" + "=" * 70)
    print("PHASE 2: Baseline Math Probe")
    print("=" * 70)
    baseline_correct, baseline_total = run_math_probe(model, tokenizer)
    baseline_score = baseline_correct / baseline_total
    print(f"\n→ Baseline: {baseline_correct}/{baseline_total} = {baseline_score:.4f}")
    
    free_model(model)
    
    # --- Phase 3: Sweep ---
    print("\n" + "=" * 70)
    print("PHASE 3: Layer Duplication Sweep")
    print("=" * 70)
    
    # Candidates: single layers + small blocks
    candidates = []
    for i in range(0, num_layers_model, 2):
        candidates.append((i, i+1, f"layer{i}_x2"))
    mid = num_layers_model // 2
    for start in range(max(0, mid-4), min(num_layers_model-2, mid+4)):
        for width in [2, 3, 4]:
            end = start + width
            if end <= num_layers_model:
                candidates.append((start, end, f"block({start},{end})"))
    
    # Dedup
    seen = set()
    unique = []
    for c in candidates:
        k = (c[0], c[1])
        if k not in seen:
            seen.add(k)
            unique.append(c)
    candidates = unique
    
    print(f"\nTesting {len(candidates)} configurations...")
    print(f"{'Config':<25} {'+L':>3} {'Score':>8} {'Delta':>8}")
    print("-" * 50)
    
    results = []
    for idx, (start, end, name) in enumerate(candidates):
        t0 = time.time()
        try:
            model, tok, extra, total = build_rys_model(start, end)
            correct, total_q = run_math_probe(model, tok, verbose=False)
            score = correct / total_q
            delta = score - baseline_score
        except Exception as e:
            print(f"  ✗ {name:<23} ERROR: {str(e)[:40]}")
            free_model(model)
            continue
        
        results.append({
            'config': name, 'start': start, 'end': end,
            'extra_layers': extra, 'score': score, 'delta': delta,
        })
        
        marker = "★" if delta > 0.02 else ("+" if delta > 0 else ("=" if delta == 0 else "-"))
        elapsed = time.time() - t0
        print(f"  {marker} {name:<23} +{extra:>2}  {score:.4f}   {delta:+.4f}  ({elapsed:.0f}s)")
        
        free_model(model)
        
        if (idx + 1) % 5 == 0:
            print(f"  --- {idx+1}/{len(candidates)} done ---")
    
    # --- Phase 4: Results ---
    print("\n" + "=" * 70)
    print("PHASE 4: Results Summary")
    print("=" * 70)
    
    results.sort(key=lambda x: x['delta'], reverse=True)
    
    print(f"\nBaseline: {baseline_score:.4f}")
    print(f"\nTop 5:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. {r['config']:<25} +{r['extra_layers']}L | "
              f"score={r['score']:.4f} | delta={r['delta']:+.4f}")
    print(f"\nBottom 5:")
    for r in results[-5:]:
        print(f"  {r['config']:<25} +{r['extra_layers']}L | "
              f"score={r['score']:.4f} | delta={r['delta']:+.4f}")
    
    winners = [r for r in results if r['delta'] > 0]
    losers = [r for r in results if r['delta'] < 0]
    print(f"\n→ {len(winners)} improved, {len(losers)} degraded, "
          f"{len(results)-len(winners)-len(losers)} unchanged")
    
    if winners:
        best = results[0]
        print(f"\n🎯 Best: {best['config']} (+{best['extra_layers']} layers)")
        print(f"   Score: {best['score']:.4f} (base: {baseline_score:.4f})")
        print(f"   Delta: {best['delta']:+.4f}")
    
    with open("rys_moe_results.json", "w") as f:
        json.dump({
            'model': MODEL_ID,
            'baseline': baseline_score,
            'num_layers': num_layers_model,
            'num_experts': 60,
            'has_shared_experts': True,
            'results': results,
            'anatomy_reasoning_layers': reasoning_layers,
        }, f, indent=2)
    print(f"\nResults saved to rys_moe_results.json")

if __name__ == "__main__":
    main()
