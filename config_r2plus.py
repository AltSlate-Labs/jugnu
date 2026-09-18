"""R2+ final competitive run: 110M deep-thin + value residuals + Muon, WSD schedule,
data blend with decay-phase educational upweighting. ~25B tokens on 2 GPUs.
Consumed by value_residual.py, muon.py, and train_r2plus.py (all `import config as C`)."""

# ---- Model (~110M deep-thin + value residuals, = R2) ----
HIDDEN_SIZE          = 576
INTERMEDIATE_SIZE    = 1536
NUM_LAYERS           = 23
NUM_HEADS            = 9
NUM_KV_HEADS         = 3
MAX_POSITION         = 2048
TIE_EMBEDDINGS       = True
TOKENIZER_NAME       = "HuggingFaceTB/SmolLM2-135M"

# ---- Data (blend of existing bins) ----
DATA_DIR             = "/home/ubuntu/jugnu/data"
VAL_SPLIT            = "val"                 # FineWeb-Edu val, for comparability with R0-R2
# source bin (without .bin) -> educational? (upweighted in the decay phase)
SOURCES              = {"train": True, "dclm": False, "finemath": True}   # train.bin = FineWeb-Edu
STABLE_BLEND         = {"train": 0.55, "dclm": 0.35, "finemath": 0.10}    # = R3 blend (stable phase)
DECAY_BLEND          = {"train": 0.80, "dclm": 0.05, "finemath": 0.15}    # upweight educational in decay

# ---- Training (~25B tokens on 2 GPUs; global batch held at R2's 0.5M tok/step) ----
BLOCK_SIZE           = 2048
MICRO_BATCH_SIZE     = 4
GRAD_ACCUM_STEPS     = 32       # 4 * 2048 * 32 * 2 GPUs = 524,288 tokens/step
MAX_STEPS            = 48000    # 48000 * 524288 = 25.17B tokens
WARMUP_STEPS         = 2000
DECAY_STEPS          = 10000    # WSD decay over the last ~21% of steps

# ---- Optimizer: Muon (2D hidden) + AdamW (embeddings/head/norms/vr_lambda) ----
LEARNING_RATE        = 1.5e-3   # AdamW peak
MIN_LR               = 1.5e-4   # floor = 0.1 * peak (shared WSD multiplier)
MUON_LR              = 0.02     # Muon peak (modded-nanogpt value)
MUON_MOMENTUM        = 0.95
WEIGHT_DECAY         = 0.1
BETA1, BETA2         = 0.9, 0.95
GRAD_CLIP            = 1.0

# ---- Logit control ----
Z_LOSS_COEFF         = 1e-4
LOGIT_SOFTCAP        = 0.0

# ---- Logging / checkpointing ----
OUT_DIR              = "out_r2plus"
LOG_INTERVAL         = 20
EVAL_INTERVAL        = 1000
EVAL_ITERS           = 50
SAVE_INTERVAL        = 4000
SEED                 = 1337

# ---- Metadata ----
MODEL_NAME           = "JugnuLM-110M-R2-25B"
HF_ORG               = "altslate"
