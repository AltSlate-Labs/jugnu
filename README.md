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
| **JugnuLM-110M** | 109.7M | 23L × 576 (deep-thin) | **81.25%** | 52.48% | **1.95** | [altslate/JugnuLM-110M](https://huggingface.co/altslate/JugnuLM-110M) |
| **JugnuLM-110M-R1** | 109.7M | 23L × 576 + value residuals | 81.10% | **54.67%** | **1.94** | [altslate/JugnuLM-110M-R1](https://huggingface.co/altslate/JugnuLM-110M-R1) |

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

## What's here

| File | Role |
|---|---|
| `config.py` | all hyperparameters for the current model |
| `model.py` | model + tokenizer builders (HF Qwen3 arch = Llama + QK-Norm) |
| `count_params.py` | assert the model is under the 150M limit |
| `prepare_data.py` | FineWeb-Edu → uint16 `.bin` memmaps |
| `train.py` | DDP + bf16 + torch.compile pretraining loop (CE + z-loss) |
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
