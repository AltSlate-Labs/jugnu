"""JugnuLM value-residual config (Qwen3 + ResFormer value residuals)."""
from transformers import Qwen3Config

class JugnuVRConfig(Qwen3Config):
    model_type = "jugnu_vr"
