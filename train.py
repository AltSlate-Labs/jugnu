"""Pretrain the tiny LM. DDP-ready for the 4x Blackwell box.

Single GPU:   python train.py
4 GPUs:       torchrun --standalone --nproc_per_node=4 train.py
"""
import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import make_model
import config as C

# ---------------- DDP / device setup ----------------
ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    is_master = rank == 0
else:
    rank, local_rank, world_size = 0, 0, 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_master = True

torch.manual_seed(C.SEED + rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

tokens_per_step = C.MICRO_BATCH_SIZE * C.BLOCK_SIZE * C.GRAD_ACCUM_STEPS * world_size
if is_master:
    print(f"world_size={world_size}  tokens/step={tokens_per_step:,}  "
          f"total tokens={tokens_per_step * C.MAX_STEPS/1e9:.2f}B")

# ---------------- data ----------------
def get_batch(split):
    # reopen memmap each call to avoid a slow memory leak (nanoGPT idiom)
    data = np.memmap(os.path.join(C.DATA_DIR, f"{split}.bin"),
                     dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - C.BLOCK_SIZE, (C.MICRO_BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i + C.BLOCK_SIZE].astype(np.int64))
                     for i in ix])
    return x.pin_memory().to(device, non_blocking=True)

# ---------------- model ----------------
raw_model, tok = make_model()
raw_model.to(device)
if is_master:
    n = sum(p.numel() for p in raw_model.parameters())
    print(f"params: {n/1e6:.1f}M")

if ddp:
    ddp_model = DDP(raw_model, device_ids=[local_rank])
else:
    ddp_model = raw_model

model = ddp_model
if os.environ.get("COMPILE", "1") == "1":
    try:
        model = torch.compile(ddp_model)
        if is_master:
            print("torch.compile: on")
    except Exception as e:
        if is_master:
            print(f"torch.compile failed, continuing eager: {e}")
        model = ddp_model

optimizer = torch.optim.AdamW(
    raw_model.parameters(), lr=C.LEARNING_RATE,
    betas=(C.BETA1, C.BETA2), weight_decay=C.WEIGHT_DECAY, fused=True)

# ---------------- lr schedule ----------------
def get_lr(step):
    if step < C.WARMUP_STEPS:
        return C.LEARNING_RATE * (step + 1) / C.WARMUP_STEPS
    if step >= C.MAX_STEPS:
        return C.MIN_LR
    ratio = (step - C.WARMUP_STEPS) / (C.MAX_STEPS - C.WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return C.MIN_LR + coeff * (C.LEARNING_RATE - C.MIN_LR)

def compute_losses(x):
    """Return (loss_for_backward, ce_for_logging).

    loss = cross-entropy [+ z-loss] [with optional logit soft-cap]. QK-Norm is
    handled inside the model. We compute logits explicitly so we can add z-loss.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
    logits = logits[:, :-1, :].float()          # predict token t+1 from <=t
    labels = x[:, 1:]
    if C.LOGIT_SOFTCAP:
        cap = C.LOGIT_SOFTCAP
        logits = cap * torch.tanh(logits / cap)
    V = logits.size(-1)
    ce = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
    loss = ce
    if C.Z_LOSS_COEFF:
        lse = torch.logsumexp(logits, dim=-1)
        loss = ce + C.Z_LOSS_COEFF * (lse ** 2).mean()
    return loss, ce.detach()


@torch.no_grad()
def estimate_val_loss():
    model.eval()
    losses = torch.zeros(C.EVAL_ITERS, device=device)
    for i in range(C.EVAL_ITERS):
        _, ce = compute_losses(get_batch("val"))
        losses[i] = ce
    model.train()
    if ddp:
        dist.all_reduce(losses, op=dist.ReduceOp.AVG)
    return losses.mean().item()

# ---------------- train loop ----------------
os.makedirs(C.OUT_DIR, exist_ok=True)
model.train()
t0 = time.time()
for step in range(C.MAX_STEPS + 1):
    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if step % C.EVAL_INTERVAL == 0 and step > 0:
        vl = estimate_val_loss()
        if is_master:
            print(f"[eval] step {step} val_loss {vl:.4f} val_ppl {math.exp(vl):.2f}")

    if step % C.SAVE_INTERVAL == 0 and step > 0 and is_master:
        ckpt = os.path.join(C.OUT_DIR, f"step{step}")
        raw_model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        print(f"[save] {ckpt}")

    if step == C.MAX_STEPS:
        break

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for micro in range(C.GRAD_ACCUM_STEPS):
        if ddp:
            ddp_model.require_backward_grad_sync = (micro == C.GRAD_ACCUM_STEPS - 1)
        x = get_batch("train")
        loss, ce = compute_losses(x)
        loss = loss / C.GRAD_ACCUM_STEPS
        loss.backward()
        loss_accum += ce.item() / C.GRAD_ACCUM_STEPS   # log CE, not CE+z-loss
    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), C.GRAD_CLIP)
    optimizer.step()

    if step % C.LOG_INTERVAL == 0 and is_master:
        dt = time.time() - t0
        tok_s = tokens_per_step * C.LOG_INTERVAL / dt if step > 0 else 0
        print(f"step {step:>6} | loss {loss_accum:.4f} | lr {lr:.2e} | "
              f"{tok_s/1e3:.1f}k tok/s")
        t0 = time.time()

# ---------------- final save ----------------
if is_master:
    final = os.path.join(C.OUT_DIR, "final")
    raw_model.save_pretrained(final)
    tok.save_pretrained(final)
    print(f"[done] saved final model to {final}")

if ddp:
    dist.destroy_process_group()
