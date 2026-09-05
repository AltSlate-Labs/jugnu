---
license: apache-2.0
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - tiny-lm
  - qwen3
  - pretraining
  - from-scratch
datasets:
  - HuggingFaceFW/fineweb-edu
---

# JugnuLM-53M

A **53.5M parameter** language model pretrained **from scratch** on ~12B tokens of
FineWeb-Edu. Built as an entry for the
[Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard)
(sub-150M parameter models). *Jugnu* (जुगनू) means "firefly" — small, but it glows.

It punches above its size: at ~53M it matches or beats models 1–2.7× larger on
grammar (BLiMP) and perplexity.

## Results

Evaluated with EleutherAI `lm-evaluation-harness`:

| Metric | Score |
|---|---|
| BLiMP (acc) | **78.14%** |
| ARC-Easy (acc) | **51.43%** |
| WikiText-2 (byte perplexity ↓) | **2.04** |
| WikiText-2 (bits/byte ↓) | 1.03 |
| Validation perplexity ↓ | 21.6 |

Reproduce:
```bash
lm_eval --model hf \
  --model_args pretrained=altslate/JugnuLM-53M,dtype=bfloat16 \
  --tasks blimp,arc_easy,wikitext --batch_size auto
```

## Architecture

| | |
|---|---|
| Params | 53.5M |
| Base architecture | Qwen3 (Llama-style + built-in QK-Norm) |
| Layers | 8 |
| Hidden size | 512 |
| Heads / KV heads | 8 / 4 (grouped-query attention) |
| Context | 2048 |
| Position | RoPE |
| Norm / FFN | RMSNorm / SwiGLU |
| Embeddings | tied |
| Tokenizer | SmolLM2 (49,152 vocab) |

## Training

- **Data:** `HuggingFaceFW/fineweb-edu` (sample-10BT), ~12B tokens seen, packed into 2048-token blocks
- **Optimizer:** AdamW (β 0.9/0.95, wd 0.1), grad clip 1.0
- **Schedule:** cosine, 2k warmup, peak LR 1.5e-3 → floor 1.5e-4
- **Batch:** ~0.5M tokens/step (global)
- **Stability:** QK-Norm + z-loss (1e-4) for logit control
- **Hardware:** 4× NVIDIA RTX PRO 4500 Blackwell, bf16, DDP + torch.compile

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("altslate/JugnuLM-53M")
model = AutoModelForCausalLM.from_pretrained("altslate/JugnuLM-53M")

ids = tok("The capital of France is", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=30, repetition_penalty=1.3)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Intended use & limitations

Research / leaderboard entry. This is a **base** model — **not** instruction-tuned
or aligned. Expect fluent English continuation and reasonable factual recall for its
size, but limited multi-step reasoning, occasional repetition, and hallucination.
Not for production use.

## Citation / attribution

Trained by AltSlate Labs. Recipe and training code:
https://github.com/AltSlate-Labs/jugnu
