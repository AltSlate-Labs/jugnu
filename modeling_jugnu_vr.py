"""JugnuLM value-residual model. Qwen3ForCausalLM with each layer v_proj replaced
by a value-residual linear: v_i = v_proj_i(x) + lambda_i * v0 (v0 = layer-0 value).
Loads correctly via AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True);
stock Qwen3 loading would silently drop the value-residual pathway."""
import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM
try:
    from .configuration_jugnu_vr import JugnuVRConfig  # HF dynamic-module (trust_remote_code) load
except ImportError:  # direct/script import (e.g. packaging) — importlib avoids check_imports flagging
    import importlib
    JugnuVRConfig = importlib.import_module("configuration_jugnu_vr").JugnuVRConfig


class VResidualLinear(nn.Linear):
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


class JugnuVRForCausalLM(Qwen3ForCausalLM):
    config_class = JugnuVRConfig

    def __init__(self, config):
        super().__init__(config)
        ctx = {}
        for i, layer in enumerate(self.model.layers):
            old = layer.self_attn.v_proj
            new = VResidualLinear(old.in_features, old.out_features, ctx,
                                  is_first=(i == 0), bias=(old.bias is not None))
            layer.self_attn.v_proj = new
        self.post_init()
