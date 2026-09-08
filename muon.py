"""Muon optimizer (Newton-Schulz orthogonalized momentum) for 2D hidden weights.

Reference: Keller Jordan et al. (modded-nanogpt). Use Muon for the 2D matrices of
the transformer (attn + MLP); keep AdamW for embeddings, norms, and scalars.
With DDP, gradients are all-reduced before step(), so each rank computes the same
orthogonalized update — consistent across ranks.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration to approximately orthogonalize G (2D)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, nesterov, ns = (group["lr"], group["momentum"],
                                          group["nesterov"], group["ns_steps"])
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                assert g.ndim == 2, "Muon is for 2D params only; route others to AdamW"
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                upd = g.add(buf, alpha=momentum) if nesterov else buf
                upd = zeropower_via_newtonschulz5(upd, steps=ns)
                # scale so update RMS matches across differently-shaped matrices
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(upd, alpha=-lr * scale)
        return loss


def split_params_for_muon(model):
    """Return (muon_params, adamw_params): 2D weights -> Muon; the rest -> AdamW.
    Embeddings and the LM head are kept on AdamW (Muon is for hidden matrices)."""
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_2d = p.ndim == 2
        is_embed = ("embed" in name) or ("lm_head" in name)
        if is_2d and not is_embed:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw
