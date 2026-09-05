#!/usr/bin/env bash
cd ~/jugnu
export OMP_NUM_THREADS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ./venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 train.py
