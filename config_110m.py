"""JugnuLM-110M config — deep-thin (23L × 576), ~8B tokens.
To train this variant, copy these values over config.py (or import from here).
"""

# ---- Model (~110M, deep-thin per MobileLLM: depth over width) ----
HIDDEN_SIZE          = 576
INTERMEDIATE_SIZE    = 1536
NUM_LAYERS           = 23
NUM_HEADS            = 9
NUM_KV_HEADS         = 3        # grouped-query attention
MAX_POSITION         = 2048
TIE_EMBEDDINGS       = True
TOKENIZER_NAME       = "HuggingFaceTB/SmolLM2-135M"

# ---- Data ----
DATA_DIR             = "data"
HF_DATASET           = "HuggingFaceFW/fineweb-edu"
HF_DATASET_CONFIG    = "sample-10BT"
VAL_FRACTION         = 0.001

# ---- Training (~8B tokens) ----
BLOCK_SIZE           = 2048
MICRO_BATCH_SIZE     = 4        # 110M fits ~16GB/GPU at micro-batch 4 (bump to 8 if VRAM allows)
GRAD_ACCUM_STEPS     = 16       # global = 4 * 2048 * 16 * 4 GPUs = 524,288 tokens/step
MAX_STEPS            = 16000    # ~8.39B tokens
WARMUP_STEPS         = 2000
LEARNING_RATE        = 1.5e-3
MIN_LR               = 1.5e-4
WEIGHT_DECAY         = 0.1
BETA1, BETA2         = 0.9, 0.95
GRAD_CLIP            = 1.0

# ---- Logit control ----
Z_LOSS_COEFF         = 1e-4
LOGIT_SOFTCAP        = 0.0

# ---- Logging / checkpointing ----
OUT_DIR              = "out"
LOG_INTERVAL         = 20
EVAL_INTERVAL        = 1000
EVAL_ITERS           = 50
SAVE_INTERVAL        = 4000
SEED                 = 1337

# ---- Metadata ----
MODEL_NAME           = "JugnuLM-110M"
HF_ORG               = "altslate"
