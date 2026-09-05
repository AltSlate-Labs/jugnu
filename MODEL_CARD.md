---
license: apache-2.0
language: en
library_name: transformers
tags:
  - tiny-lm
  - pretraining
  - llama
datasets:
  - HuggingFaceFW/fineweb-edu
---

# JugnuLM-53M

A ~53M parameter Llama-architecture language model pretrained from scratch on
~12B tokens of FineWeb-Edu. Built as an entry for the
[Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard)
(sub-150M parameter models).

## Architecture
| | |
|---|---|
| Params | ~53M |
| Layers | 8 |
| Hidden | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| Context | 2048 |
| Tokenizer | SmolLM2 (49152 BPE) |
| Position | RoPE |
| Norm / FFN | RMSNorm / SwiGLU |
| QK-Norm | yes (Qwen3-style per-head q/k RMSNorm) |
| Base arch | HF `Qwen3` (Llama + QK-Norm) |

## Training
- Data: `HuggingFaceFW/fineweb-edu` (sample-10BT), ~12B tokens seen
- Global batch: ~0.5M tokens, cosine LR 1.5e-3 → 1.5e-4, 2k warmup
- Stability: QK-Norm + z-loss (1e-4) to control logit magnitude
- Hardware: 4× NVIDIA RTX PRO 4500 Blackwell, bf16, DDP + torch.compile

## Evaluation (reproducible)
Evaluated with EleutherAI `lm-evaluation-harness`:
```bash
lm_eval --model hf \
  --model_args pretrained=<this-repo>,dtype=bfloat16 \
  --tasks blimp,arc_easy,wikitext --batch_size auto
```
| Metric | Score |
|---|---|
| BLiMP (acc) | __ |
| ARC-Easy (acc) | __ |
| WikiText-2 (byte_perplexity) | __ |

## Intended use
Research / leaderboard entry. A tiny base LM — not instruction-tuned, not for
production. Expect grammatical fluency but limited reasoning.
