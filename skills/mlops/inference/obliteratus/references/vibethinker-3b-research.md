# VibeThinker-3B: Research Reference

## Paper
- **Title:** VibeThinker-3B: Exploring the Frontier of Verifiable Reasoning in Small Language Models
- **arXiv:** [2606.16140](https://arxiv.org/abs/2606.16140) (v1, June 15 2026)
- **Authors:** Sen Xu et al. (WeiboAI / Sina Weibo Inc.)
- **Base:** Qwen2.5-Coder-3B (3B dense, BF16)
- **GitHub:** https://github.com/WeiboAI/VibeThinker (MIT, eval/inference only — no training code)
- **HF:** https://huggingface.co/WeiboAI/VibeThinker-3B

## Core Theory: Parametric Compression-Coverage Hypothesis
- Verifiable reasoning (math, code, STEM) = **parameter-dense** → compressible into compact reasoning cores
- Open-domain knowledge = **parameter-expansive** → requires broad parameter coverage
- Explains GPQA-Diamond gap (70.2 vs 90+ for large models) — knowledge needs parameters, reasoning doesn't

## SSP Framework (Spectrum-to-Signal Principle)
Four-stage pipeline:
1. **Two-Stage Curriculum SFT** — Broad coverage (5 epochs) then hard reasoning (traces <5K removed, easy problems filtered). Diversity-exploring distillation merges per-domain specialists by most valid solutions (not highest Pass@1).
2. **Multi-Domain RL (MGPO)** — Sequential: Math → Code → STEM
3. **Offline Self-Distillation** — Learning Potential Score prioritizes not-yet-internalized correct traces
4. **Instruct RL** — Rule-based validators + rubric-based reward models

## MGPO (MaxEnt-Guided Policy Optimization)
- Dynamically weight prompts near capability boundary (p(q) ≈ 0.5)
- Weight: `w(q) = exp(-γ · D_ME(p(q) || p₀))`, p₀ = 0.5
- Prompts too hard (p≈0) or saturated (p≈1) get low weights
- Applied to GRPO-style clipped objective with group-relative advantage
- **Fully on-policy** to mitigate training-inference probability mismatch
- **Single 64K context from start** — progressive expansion weakens long-thinking
- **Long2Short Math RL** — reward redistribution favoring shorter correct solutions

## Key Findings for Ablation Work

### Non-Termination Problem (Our Novel Observation)
VT-3B exhibits **reasoning runaway** on hard problems:
- 96% of generated tokens are reasoning tokens
- Mean response: 18,887 tokens; 17.5% of rollouts truncated at 40,960 tokens
- On our benchmark: 19/26 (73%) vs Opus 4.6 at 26/26 (100%)
- Failure mode: spiraling analysis paralysis — the model generates 50K+ tokens of increasingly circular reasoning before hitting the token cap, producing truncated/empty output

**Root cause:** MGPO's entropy-preserving RL prevents entropy collapse (good for exploration during training) but at inference time this means the model never converges to a confident answer on hard problems. The broad exploration that helps during RL becomes analysis paralysis during generation.

### Dynamic `<think>` Logit Bias — EXPERIMENT COMPLETED (Jun 2026)

**Status: Results in. See `templates/think_bias_proxy.py` for the reusable proxy script.**

A stdlib-only HTTP proxy sits between client and llama-server, injecting `logit_bias` on the `</think>` token (ID 151666) to force the model to exit reasoning and commit to an answer. Three modes: none (baseline), static (constant bias), dynamic (ramp up after threshold).

#### Token IDs (Qwen2.5 tokenizer)
- `<think>` = 151665
- `</think>` = 151666
- `<|endoftext|>` = 151643

#### Results (2 problems, 9 unit tests, VibeThinker-3B Q4_K_M base model)

| Condition | Calculator | Word Ladder | Total | Tokens | Time |
|---|---|---|---|---|---|
| No bias | 0/6 | 3/3 | 3/9 | 11,329 | 167s |
| Static +1.0 | 0/6 | 3/3 | 3/9 | 12,750 | 192s |
| **Static +3.0** | 0/6 | **3/3** | 3/9 | **10,940** | **165s** |
| Dynamic +5-30 | 0/6 | 0/3 | 0/9 | 16,384 | 251s |

#### Key Findings

1. **Static +3.0 is the sweet spot for solvable problems.** Word ladder: 64% less reasoning (4,115 vs 11,296 chars), 31% fewer tokens, 33% faster, same accuracy. The model thinks just enough to solve the problem, then exits cleanly.

2. **Calculator is a capability wall, not a termination problem.** Even with optimal bias, the 3B model cannot implement a working calculator parser. The failure is model size, not reasoning runaway.

3. **High bias (>5.0) catastrophically breaks generation.** At +5.0 and above, the model exits reasoning prematurely before it has a solution, then produces 65K chars of garbage with no code. The model needs sufficient reasoning time; forcing it out too early is worse than letting it spiral.

4. **Dynamic mode (ramp) doesn't work with llama-server.** The OpenAI-compatible API applies `logit_bias` at request time (static), not per-token during generation. True dynamic biasing would require either modifying llama-server's C++ inference loop or using raw token-by-token completion with per-step bias updates. The "dynamic" mode in the proxy approximates this by computing a bias from max_tokens, but it's effectively static within a single generation.

5. **Bias doesn't help on unsolvable problems.** The +1.0 condition actually used MORE tokens than baseline on word ladder (4,558 vs 3,984) — a weak bias doesn't force exit, it just adds noise. The bias needs to be strong enough (3.0) to meaningfully shift the token distribution.

#### Limitations
- Only 2 problems tested; larger benchmark suite needed for statistical significance
- llama-server's request-time logit_bias can't do true per-token dynamic biasing
- Temperature was 0.7; results may differ at other temperatures
- Only tested base (non-ablated) VT-3B; ablated VT-3B may respond differently

#### Usage
```bash
# Start llama-server with VT-3B
llama-server -m VibeThinker-3B-Q4_K_M.gguf -ngl 99 -c 32768 --port 8081 --jinja --flash-attn on -ctk q8_0 -ctv q8_0 -np 1

# Start proxy with optimal static bias
python3 think_bias_proxy.py --port 8082 --upstream-port 8081 --mode static --static-bias 3.0

# Point clients at port 8082 instead of 8081
```

### Ablation Interaction
When VT-3B is abliterated (see pitfall #15 in SKILL.md), the non-termination problem worsens — the ablation damages the model's ability to self-terminate reasoning chains. This suggests the refusal direction overlaps with the "convergence" direction that MGPO trained. **Ablated VT-3B + dynamic think-bias is the logical combination to test.**

## Benchmark Scores (Paper)

### Math
| Benchmark | VT-3B | + CLR | Best Flagship |
|---|---|---|---|
| AIME 2025 | 91.4 | 96.7 | GLM-5: 96.7 |
| AIME 2026 | 94.3 | 97.1 | DeepSeek V3.2: 94.2 |
| HMMT 2025 | 89.3 | 95.4 | GLM-5: 97.9 |
| BruMO 2025 | 93.8 | 99.2 | Kimi K2.5: 98.3 |
| IMO-AnswerBench | 76.4 | 80.6 | GLM-5: 82.5 |

### Code
| Benchmark | Score | Comparison |
|---|---|---|
| LiveCodeBench v6 | 80.2 | Best <120B; GPT-OSS-120B: 81.9 |
| LeetCode OOD | 96.1% (123/128) | Beats GPT-5.2 (95.3%), Claude Opus 4.6 (86.7%) |

### Knowledge (Weakness)
| Benchmark | Score | Gap |
|---|---|---|
| GPQA-Diamond | 70.2 (72.9 +CLR) | GLM-5: 86.0, Gemini 3 Pro: 91.9 |

## CLR (Claim-Level Reliability) — Test-Time Scaling
- No weight updates needed
- Generate 32 candidate trajectories per query
- Extract 5 key claims from each
- Assess claim-level reliability, aggregate at claim granularity
- Boosts: AIME26 +2.8, HMMT25 +6.1, BruMO25 +5.4

## Training Cost
- 1.5B predecessor: **$7,800** post-training (30-60× cheaper than DeepSeek R1 $294K)
- 3B cost: not explicitly stated
- Training code: **NOT released** — only eval/inference

## Inference Config
- transformers >= 4.54.0 (pin 4.55.4 for vLLM)
- vLLM 0.10.1 or SGLang >= 0.4.9.post6
- Decoding: temp=1.0, top_p=0.95, top_k=-1, max_tokens=40960
- GGUF: https://huggingface.co/prithivMLmods/VibeThinker-3B-GGUF

## Independent Reproduction (AlphaXiv)
- AIME25 avg@8: Pass@1 = 85.0 (vs paper 91.4 at avg@64)
- Pass@8 = 93.3 — reproduced
- 17.5% of rollouts truncated at 40960 tokens (mean response: 18,887 tokens)
- Verdict: "3B reaches frontier-level AIME" — reproduced
