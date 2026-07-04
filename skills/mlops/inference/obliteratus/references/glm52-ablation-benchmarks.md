# GLM-5.2 PCA-Based Refusal Ablation (Project AESOP)

## Model Details
- **Base:** GLM-5.2 FP8 (744B MoE, 18.5B dense params)
- **HF Repo:** `cfontes/GLM-5.2-Ablated-Molt`
- **Method:** PCA-based activation steering (not weight orthogonalization)
- **LoRA/Training:** None — inference-time hooks only

## Methodology

### Refusal Direction Extraction
1. **Contrastive activation collection:** Forward passes on paired harmful/benign prompts, recording activations at `model.model.layers[L].mlp.shared_experts` for layers 25-65.
2. **PCA decomposition:** Compute difference (harmful - benign) activations per layer, perform PCA. First principal component = refusal direction.
3. **Storage:** Saved as `refusal_pca.pt` (2.9MB, 41 layers × 3 PCA components × 6144 hidden dim).

### Ablation Application
Refusal direction subtracted from shared expert outputs at inference time:
```python
def ablation_hook(module, input, output):
    hs = output[0]
    d = refusal_direction  # shape [6144]
    hs = hs - coeff * (hs @ d) / (d @ d) * d
    return (hs,) + output[1:]
```

- **Target layers:** 62-65 (top 4 layers, strongest refusal concentration)
- **Coefficient:** 0.1 (conservative)
- **PCA components:** Top 2 per layer

### Layer Selection Rationale
Refusal direction concentration measured across 78 layers:
- Layers 25-35: weak separation (norm 3-7)
- Layers 40-50: moderate (norm 9-16)
- Layers 55-64: strong (norm 23-34)

Layers 62-65 chosen as optimal: late enough for strongest signal, early enough to avoid disrupting final token prediction.

## Benchmark Results (Unified Harness v3.0.1)

| Benchmark | Metric | Score | 95% Wilson CI |
|---|---|---|---|
| AdvBench | refusal_rate ↓ | 18.0% | [11.4%, 27.2%] |
| Borderline | refusal_rate ↓ | 2.0% | [0.4%, 8.5%] |
| GPQA Diamond | accuracy ↑ | 92.0% | [81.5%, 97.1%] |
| MMLU-Pro | accuracy ↑ | 75.0% | [65.6%, 82.5%] |
| HumanEval | pass@1 ↑ | 77.4% | [70.4%, 83.2%] |
| GSM8K | accuracy ↑ | 93.0% | [85.8%, 96.7%] |
| HellaSwag | accuracy ↑ | 72.0% | [62.1%, 80.0%] |
| SimpleQA | accuracy ↑ | 56.0% | [41.7%, 69.3%] |
| IFEval (prompt) | accuracy ↑ | 41.2% | — |
| IFEval (instr) | accuracy ↑ | 54.9% | — |

## Key Observations
- **AdvBench at 18%** (baseline ~87%) = 82% of harmful prompts now get a response
- **No over-refusal:** Borderline at 2% means model doesn't refuse benign requests
- **Capability preserved:** GPQA 92%, GSM8K 93% indicate core reasoning intact
- **SimpleQA at 56%** is highest among all variants; ablated base retains strong factual knowledge

## Comparison with Qwable-3.6-27B Weight Orthogonalization

| Aspect | GLM-5.2 (PCA hooks) | Qwable-3.6-27B (weight surgery) |
|---|---|---|
| Method | PCA activation steering | Weight orthogonalization (diff-in-means) |
| Persistence | Inference-time only (hooks) | Permanent weight modification |
| Target | mlp.shared_experts (4 layers) | ffn_down (all 64 layers) |
| Coefficient/Alpha | 0.1 (very conservative) | 1.3 (minimum for 100% compliance) |
| Refusal removal | 87% → 18% (79% reduction) | 96.7% → 0% (100% removal) |
| Capability | GPQA 92%, GSM8K 93% | GSM8K 92.5%, MBPP 92.5% |
| Advantage | Reversible, no weight damage | Complete refusal removal |
| Disadvantage | 18% still refuse | Permanent, may have latent degradation |

The GLM-5.2 approach is more conservative — it trades complete refusal removal for better capability preservation and reversibility. The Qwable approach achieves 100% compliance but requires aggressive multi-layer weight surgery.

## What Still Refuses (18%)
The remaining 18% likely includes:
- Heavily RLHF-reinforced categories (CSAM, bioweapons) with deeper alignment
- Prompts triggering different refusal pathways not captured by PCA at layers 62-65
- Safety training hardcoded outside the shared expert refusal direction

To push toward 0%, would need to:
1. Increase coefficient (0.1 → 0.3+) or add more layers (62-65 → 55-65)
2. Switch to weight orthogonalization for permanent removal
3. Use OBLITERATUS `advanced` or `aggressive` method on the safetensors directly
