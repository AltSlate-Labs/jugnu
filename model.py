"""Model + tokenizer builders.

Uses the stock HF **Qwen3** architecture = Llama-style (GQA / RoPE / SwiGLU /
RMSNorm) **plus built-in QK-Norm** (per-head q_norm/k_norm). No custom modeling
code, and still directly loadable by lm-eval.
"""
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
import config as C


def make_tokenizer():
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    assert tok.eos_token_id is not None, "tokenizer needs an eos token"
    return tok


def make_config(tokenizer):
    return Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=C.HIDDEN_SIZE,
        intermediate_size=C.INTERMEDIATE_SIZE,
        num_hidden_layers=C.NUM_LAYERS,
        num_attention_heads=C.NUM_HEADS,
        num_key_value_heads=C.NUM_KV_HEADS,
        head_dim=C.HIDDEN_SIZE // C.NUM_HEADS,
        max_position_embeddings=C.MAX_POSITION,
        tie_word_embeddings=C.TIE_EMBEDDINGS,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation="sdpa",
        # Qwen3 applies RMSNorm to Q and K per-head (QK-Norm) automatically.
    )


def make_model():
    tok = make_tokenizer()
    model = Qwen3ForCausalLM(make_config(tok))
    return model, tok
