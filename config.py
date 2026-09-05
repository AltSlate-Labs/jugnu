"""Central config for the tiny-LM leaderboard entry (53M / ~12B tokens)."""

# ---- Model (Llama-arch, ~53M params, well under the 150M board limit) ----
HIDDEN_SIZE          = 512
INTERMEDIATE_SIZE    = 1792
NUM_LAYERS           = 8
NUM_HEADS            = 8
NUM_KV_HEADS         = 4        # grouped-query attention
MAX_POSITION         = 2048
TIE_EMBEDDINGS       = True
TOKENIZER_NAME       = "HuggingFaceTB/SmolLM2-135M"  # reuse a good 49152-vocab BPE

# ---- Data ----
DATA_DIR             = "data"
HF_DATASET           = "HuggingFaceFW/fineweb-edu"
HF_DATASET_CONFIG    = "sample-10BT"   # ~10B tokens; downloads ~30GB, produces ~20GB of .bin
VAL_FRACTION         = 0.001

# ---- Training ----
BLOCK_SIZE           = 2048
MICRO_BATCH_SIZE     = 8       # per-GPU micro-batch (8 keeps fp32-logit loss in memory)
GRAD_ACCUM_STEPS     = 8       # -> global batch = 8 * 2048 * 8 * (num_gpus) tokens
                              #    with 4 GPUs = 524,288 tokens/step (~0.5M)
MAX_STEPS            = 23000   # 23000 * 524288 ~= 12.05B tokens
WARMUP_STEPS         = 2000
LEARNING_RATE        = 1.5e-3
MIN_LR               = 1.5e-4
WEIGHT_DECAY         = 0.1
BETA1, BETA2         = 0.9, 0.95
GRAD_CLIP            = 1.0

# ---- Logit magnitude control ----
# QK-Norm is built into the model architecture (see model.py).
# Z_LOSS: training-only auxiliary loss lambda * mean(logsumexp(logits)^2).
#   Prevents logit drift; what GPT-X2 (board #1) used. Set 0.0 to disable.
Z_LOSS_COEFF         = 1e-4
# LOGIT_SOFTCAP: optional cap*tanh(logits/cap) in the loss. Off by default
#   because it's training-only here and would mismatch lm-eval's raw logits.
#   Set to e.g. 30.0 to enable.
LOGIT_SOFTCAP        = 0.0

# ---- Logging / checkpointing ----
OUT_DIR              = "out"
LOG_INTERVAL         = 20
EVAL_INTERVAL        = 1000
EVAL_ITERS           = 50
SAVE_INTERVAL        = 5000
SEED                 = 1337

# ---- Metadata for the leaderboard entry ----
MODEL_NAME           = "JugnuLM-53M"
HF_ORG               = "altslate"
