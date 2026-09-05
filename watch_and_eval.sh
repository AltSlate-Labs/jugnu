#!/usr/bin/env bash
# Runs ON THE BOX, detached. Waits for training to finish, then auto-runs eval.
cd ~/jugnu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

until grep -qaE "\[done\] saved final|Traceback|OutOfMemoryError|CUDA error|ChildFailedError|Killed" train.log; do
  sleep 120
done

if [ -d out/final ]; then
  echo "TRAINING_DONE $(date -u +%FT%TZ)" > eval_status.txt
  ./venv/bin/python -m lm_eval --model hf \
    --model_args pretrained=out/final,dtype=bfloat16 \
    --tasks blimp,arc_easy,wikitext \
    --device cuda:0 --batch_size auto \
    --output_path eval_results > eval.log 2>&1
  echo "EVAL_DONE $(date -u +%FT%TZ)" >> eval_status.txt
else
  echo "TRAINING_FAILED_NO_FINAL $(date -u +%FT%TZ)" > eval_status.txt
fi
