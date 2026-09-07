"""Phase-2 R1 model: R0 (110M deep-thin Qwen3) + value residuals (ResFormer).

Value residual: v_i = v_proj_i(x) + lambda_i * v0, where v0 is layer-0's value
projection. Implemented by swapping each layer's v_proj for a VResidualLinear that
shares a context dict; layer 0 writes v0, later layers add lambda_i * v0.
Adding at the v_proj output (flat, pre-reshape) is equivalent to adding on the
reshaped value_states, and avoids overriding the (version-specific) attention forward.
lambda init = 0 -> starts identical to R0, learns to use the residual.
"""
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
import config as C


def make_tokenizer():
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class VResidualLinear(nn.Linear):
    """v_proj replacement. Shares `ctx`; layer 0 stores v0, others add lambda*v0."""
    def __init__(self, in_f, out_f, ctx, is_first, bias=False):
        super().__init__(in_f, out_f, bias=bias)
        self.vr_ctx = ctx
        self.vr_is_first = is_first
        if not is_first:
            self.vr_lambda = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        v = super().forward(x)
        if self.vr_is_first:
            self.vr_ctx["v0"] = v
        else:
            v0 = self.vr_ctx.get("v0")
            if v0 is not None:
                v = v + self.vr_lambda * v0
        return v


def make_config(tok):
    return Qwen3Config(
        vocab_size=len(tok),
        hidden_size=C.HIDDEN_SIZE,
        intermediate_size=C.INTERMEDIATE_SIZE,
        num_hidden_layers=C.NUM_LAYERS,
        num_attention_heads=C.NUM_HEADS,
        num_key_value_heads=C.NUM_KV_HEADS,
        head_dim=C.HIDDEN_SIZE // C.NUM_HEADS,
        max_position_embeddings=C.MAX_POSITION,
        tie_word_embeddings=C.TIE_EMBEDDINGS,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        attn_implementation="sdpa",
    )


def make_model():
    tok = make_tokenizer()
    model = Qwen3ForCausalLM(make_config(tok))
    ctx = {}
    for i, layer in enumerate(model.model.layers):
        old = layer.self_attn.v_proj
        new = VResidualLinear(old.in_features, old.out_features, ctx,
                              is_first=(i == 0), bias=(old.bias is not None))
        with torch.no_grad():
            new.weight.copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        layer.self_attn.v_proj = new
    return model, tok
