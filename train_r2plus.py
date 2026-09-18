"""R2+ final competitive run: 110M VR + Muon, WSD schedule, blended data with
decay-phase educational upweighting. ~25B tokens.
  torchrun --standalone --nproc_per_node=2 train_r2plus.py     # (CUDA_VISIBLE_DEVICES picks the GPUs)
"""
import os, math, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from value_residual import make_model          # 110M deep-thin + value residuals
from muon import Muon, split_params_for_muon
import config as C

# ---------------- DDP / device ----------------
ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"]); local_rank = int(os.environ["LOCAL_RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"; torch.cuda.set_device(device); is_master = rank == 0
else:
    rank, local_rank, world_size = 0, 0, 1
    device = "cuda"; is_master = True
torch.manual_seed(C.SEED + rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
rng = np.random.default_rng(C.SEED + rank)

tokens_per_step = C.MICRO_BATCH_SIZE * C.BLOCK_SIZE * C.GRAD_ACCUM_STEPS * world_size
if is_master:
    print(f"world_size={world_size} tokens/step={tokens_per_step:,} total={tokens_per_step*C.MAX_STEPS/1e9:.2f}B", flush=True)

# ---------------- data (blended) ----------------
_MM = {}
def _mm(src):
    if src not in _MM:
        _MM[src] = np.memmap(os.path.join(C.DATA_DIR, f"{src}.bin"), dtype=np.uint16, mode="r")
    return _MM[src]
def blend_weights(step):
    ds = C.MAX_STEPS - C.DECAY_STEPS
    if step < ds: w = dict(C.STABLE_BLEND)
    else:
        r = (step - ds) / C.DECAY_STEPS
        w = {k: C.STABLE_BLEND[k]*(1-r) + C.DECAY_BLEND[k]*r for k in C.STABLE_BLEND}
    tot = sum(w.values()); return {k: v/tot for k, v in w.items()}
def get_micro(src):
    d = _mm(src); ix = rng.integers(0, len(d) - C.BLOCK_SIZE, size=C.MICRO_BATCH_SIZE)
    x = torch.stack([torch.from_numpy(d[i:i+C.BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True)
def get_val():
    d = _mm(C.VAL_SPLIT); ix = rng.integers(0, len(d) - C.BLOCK_SIZE, size=C.MICRO_BATCH_SIZE)
    x = torch.stack([torch.from_numpy(d[i:i+C.BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True)

# ---------------- model ----------------
raw_model, tok = make_model(); raw_model.to(device)
if is_master: print(f"params: {sum(p.numel() for p in raw_model.parameters())/1e6:.1f}M", flush=True)
model = DDP(raw_model, device_ids=[local_rank]) if ddp else raw_model   # COMPILE off: VR breaks torch.compile

# ---------------- optimizers: Muon (2D hidden) + AdamW (rest) ----------------
muon_params, adamw_params = split_params_for_muon(raw_model)
opt_muon = Muon(muon_params, lr=C.MUON_LR, momentum=C.MUON_MOMENTUM)
opt_adamw = torch.optim.AdamW(adamw_params, lr=C.LEARNING_RATE, betas=(C.BETA1, C.BETA2), weight_decay=C.WEIGHT_DECAY, fused=True)
if is_master: print(f"muon params: {sum(p.numel() for p in muon_params)/1e6:.1f}M | adamw params: {sum(p.numel() for p in adamw_params)/1e6:.1f}M", flush=True)

# ---------------- WSD schedule (shared multiplier) ----------------
def lr_mult(step):
    if step < C.WARMUP_STEPS: return (step + 1) / C.WARMUP_STEPS
    ds = C.MAX_STEPS - C.DECAY_STEPS
    if step < ds: return 1.0
    r = min(1.0, (step - ds) / C.DECAY_STEPS)
    floor = C.MIN_LR / C.LEARNING_RATE                 # 0.1
    return floor + 0.5 * (1 + math.cos(math.pi * r)) * (1 - floor)

def compute_losses(x):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
    logits = logits[:, :-1, :].float(); labels = x[:, 1:]
    if C.LOGIT_SOFTCAP: cap = C.LOGIT_SOFTCAP; logits = cap * torch.tanh(logits / cap)
    V = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1)); loss = ce
    if C.Z_LOSS_COEFF:
        lse = torch.logsumexp(logits, dim=-1); loss = ce + C.Z_LOSS_COEFF * (lse ** 2).mean()
    return loss, ce.detach()

@torch.no_grad()
def estimate_val_loss():
    model.eval(); losses = torch.zeros(C.EVAL_ITERS, device=device)
    for i in range(C.EVAL_ITERS): _, ce = compute_losses(get_val()); losses[i] = ce
    model.train()
    if ddp: dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    return losses.mean().item()

# ---------------- train loop ----------------
os.makedirs(C.OUT_DIR, exist_ok=True); model.train(); t0 = time.time()
srcs = list(C.STABLE_BLEND.keys())
for step in range(C.MAX_STEPS + 1):
    m = lr_mult(step)
    for g in opt_adamw.param_groups: g["lr"] = C.LEARNING_RATE * m
    for g in opt_muon.param_groups: g["lr"] = C.MUON_LR * m

    if step % C.EVAL_INTERVAL == 0 and step > 0:
        vl = estimate_val_loss()
        if is_master: print(f"[eval] step {step} val_loss {vl:.4f} val_ppl {math.exp(vl):.2f}", flush=True)
    if step % C.SAVE_INTERVAL == 0 and step > 0 and is_master:
        ckpt = os.path.join(C.OUT_DIR, f"step{step}"); raw_model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        print(f"[save] {ckpt}", flush=True)
    if step == C.MAX_STEPS: break

    w = blend_weights(step); probs = [w[s] for s in srcs]
    opt_muon.zero_grad(set_to_none=True); opt_adamw.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro in range(C.GRAD_ACCUM_STEPS):
        if ddp: model.require_backward_grad_sync = (micro == C.GRAD_ACCUM_STEPS - 1)
        src = srcs[int(rng.choice(len(srcs), p=probs))]
        loss, ce = compute_losses(get_micro(src)); (loss / C.GRAD_ACCUM_STEPS).backward()
        loss_accum += ce.item() / C.GRAD_ACCUM_STEPS
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), C.GRAD_CLIP)
    opt_muon.step(); opt_adamw.step()

    if step % C.LOG_INTERVAL == 0 and is_master:
        dt = time.time() - t0; tok_s = tokens_per_step * C.LOG_INTERVAL / dt if step > 0 else 0
        phase = "warmup" if step < C.WARMUP_STEPS else ("decay" if step >= C.MAX_STEPS - C.DECAY_STEPS else "stable")
        print(f"step {step:>6} | loss {loss_accum:.4f} | muon_lr {C.MUON_LR*m:.2e} adamw_lr {C.LEARNING_RATE*m:.2e} | "
              f"{phase} edu%={100*(w['train']+w['finemath']):.0f} | {tok_s/1e3:.1f}k tok/s", flush=True)
        t0 = time.time()

if is_master:
    final = os.path.join(C.OUT_DIR, "final"); raw_model.save_pretrained(final); tok.save_pretrained(final)
    print(f"[done] saved final model to {final}", flush=True)
    print("R2PLUS_DONE", flush=True)
if ddp: dist.destroy_process_group()
