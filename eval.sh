#!/usr/bin/env bash
# Evaluate the trained model on the leaderboard's metrics.
# Usage: ./eval.sh [path-to-model]   (default: out/final)
set -euo pipefail

MODEL="${1:-out/final}"
OUT="eval_results"
mkdir -p "$OUT"

# blimp      -> grammar (BLiMP)
# arc_easy   -> ARC-Easy reasoning
# wikitext   -> report byte_perplexity  (== the board's WikiText-2 byte_ppl)
lm_eval --model hf \
  --model_args "pretrained=${MODEL},dtype=bfloat16" \
  --tasks blimp,arc_easy,wikitext \
  --device cuda:0 \
  --batch_size auto \
  --output_path "$OUT"

echo
echo "Look for these in the JSON under $OUT:"
echo "  blimp    -> acc"
echo "  arc_easy -> acc  (use acc, not acc_norm, to match the board)"
echo "  wikitext -> byte_perplexity"
