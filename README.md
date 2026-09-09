# Jugnu 🪰✨

Tiny language models trained **from scratch**, by [AltSlate Labs](https://github.com/AltSlate-Labs).
*Jugnu* (जुगनू) means "firefly" — small, but it glows.

This repo is the **shared training recipe** for the whole Jugnu family: one set of
code, one config per model. Trained checkpoints live as separate Hugging Face
model repos.

## Models

| Model | Params | Geometry | BLiMP | ARC-Easy | WikiText-2 (byte-ppl) | Weights |
|---|---|---|---|---|---|---|
| **JugnuLM-53M** | 53.5M | 8L × 512 | 78.14% | 51.43% | 2.04 | [altslate/JugnuLM-53M](https://huggingface.co/altslate/JugnuLM-53M) |
| **JugnuLM-110M** | 109.7M | 23L × 576 (deep-thin) | **81.25%** | 52.48% | 1.95 | [altslate/JugnuLM-110M](https://huggingface.co/altslate/JugnuLM-110M) |
| **JugnuLM-110M-R1** | 109.7M | 23L × 576 + value residuals | 81.10% | 54.67% | 1.94 | [altslate/JugnuLM-110M-R1](https://huggingface.co/altslate/JugnuLM-110M-R1) |
| **JugnuLM-110M-R2** | 109.7M | 23L × 576 + value residuals + Muon | 80.78% | **56.10%** | 1.93 | [altslate/JugnuLM-110M-R2](https://huggingface.co/altslate/JugnuLM-110M-R2) |
| **JugnuLM-110M-R3** | 109.7M | R2 + data blend (FWEdu/DCLM/FineMath) | **81.79%** | 53.62% | **1.91** | [altslate/JugnuLM-110M-R3](https://huggingface.co/altslate/JugnuLM-110M-R3) |

JugnuLM-110M's 81.25% BLiMP ≈ GPT-X2-125M (81.28%) at ~12% fewer params and ~9× fewer
training tokens. Built for the [Tiny-ML Leaderboard](https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard) (sub-150M-param models).

### Ablation ladder

Starting from JugnuLM-110M (rung **R0**, an honest conventional baseline), we add one
lever at a time and keep only what beats the prior rung.

- **R1 — value residuals** ([ResFormer](https://arxiv.org/abs/2410.17897)): each layer's
  value gains a learned-gated residual from the first layer's value (`v_i += λ_i·v₀`),
  implemented in `value_residual.py`. **Result: ARC-Easy +2.2 (52.48→54.67) at a BLiMP tie
  and slightly lower perplexity → kept.**
  *Caveat:* value residuals are a custom attention pathway, so `eval.sh`'s stock
  `from_pretrained` silently drops them — evaluate by rebuilding the model and loading
  weights (incl. `vr_lambda`). To train R1: copy `config_110m.py`→`config.py` and point
  `train.py`'s `make_model` at `value_residual.py`.
- **R2 — Muon optimizer** ([`muon.py`](muon.py)): Newton-Schulz orthogonalized momentum on
  the 2D hidden matrices (attn + MLP), AdamW kept for embeddings/head/norms/`vr_lambda`.
  Stacked on the R1 value-residual model. **Result: ARC-Easy +1.43 (54.67→56.10) and lower
  perplexity at a small BLiMP dip (−0.32) → kept.** Muon's convergence lead was largest early
  (val-ppl −26% at step 1000) and compressed by end of the fixed token budget, but the
  downstream ARC gain persisted. Cumulative over R1+R2: **ARC-Easy +3.6** (52.48→56.10).
- **R3 — data blend** (FineWeb-Edu 55% / DCLM-baseline 35% / FineMath 10%, per-sequence mix;
  same optimizer/schedule/tokens as R2): **dropped.** It gave the **best BLiMP (81.79) and
  best perplexity (1.91)** of the family, but **ARC-Easy fell −2.48 (56.10→53.62)** — a loss
  on our binding metric, so it doesn't beat R2. *Finding:* FineWeb-Edu's educational filtering
  is what feeds ARC-Easy (grade-school science); diluting it with general web + math improves
  broad LM quality but removes ARC-relevant signal. **For ARC the lever is more educational
  data (and distillation), not more diversity.** Model published for the record; the ladder
  continues from R2.

## What's here

| File | Role |
|---|---|
| `config.py` | all hyperparameters for the current model |
| `model.py` | model + tokenizer builders (HF Qwen3 arch = Llama + QK-Norm) |
| `count_params.py` | assert the model is under the 150M limit |
| `prepare_data.py` | FineWeb-Edu → uint16 `.bin` memmaps |
| `train.py` | DDP + bf16 + torch.compile pretraining loop (CE + z-loss) |
| `value_residual.py` | R1 lever: value-residual (ResFormer) model builder |
| `muon.py` | R2 lever: Muon optimizer + 2D/embedding param split |
| `run_train.sh` | 4-GPU launcher (`torchrun`) |
| `eval.sh` | `lm-eval-harness` on `blimp,arc_easy,wikitext` |
| `predict_eff.py` | estimate the leaderboard efficiency score / rank |
| `watch_and_eval.sh` | wait for training to finish, then auto-run eval |
| `MODEL_CARD.md` | model card (also published on the HF page) |
| `SUBMIT.md` | how to submit to the Tiny-ML Leaderboard |

## Reproduce JugnuLM-53M

```bash
pip install -r requirements.txt
python count_params.py                        # confirm < 150M
python prepare_data.py                         # download + tokenize FineWeb-Edu
torchrun --standalone --nproc_per_node=4 train.py   # ~12B tokens
./eval.sh out/final                            # BLiMP / ARC-Easy / WikiText
```

Recipe in brief: **Qwen3 architecture** (Llama + built-in QK-Norm), 53.5M params,
GQA, tied embeddings, SmolLM2 tokenizer (49,152 vocab), **z-loss** for logit
stability, trained on ~12B tokens of `HuggingFaceFW/fineweb-edu` with a cosine
schedule on 4× NVIDIA RTX PRO 4500 Blackwell GPUs.

## Adding a new Jugnu model

The code is shared — a new family member is a new **config** + a new HF **weights**
repo, not a new codebase. Copy the desired config over `config.py`, retrain, and
publish the checkpoint as `altslate/JugnuLM-<size>`. Example: `config_110m.py` is the
deep-thin 110M recipe.

## License

Apache-2.0.
