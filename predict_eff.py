"""Approximate the leaderboard's Efficiency (EFF) score and show where a new
entry would land.

The board's formula (from index.html):
    EFF = mean(BLiMP, ARC-Easy, normalized-WikiText2) * sizeBonus
    sizeBonus: log-scaled, 1.5x for the smallest model on the board, 1.0x for
               the largest.

We DON'T have the exact WikiText normalization or the exact size-bonus curve,
so this is an approximation for gut-check ranking only. To make it exact, copy
the getScore / getSizeMultiplier JS out of index.html and mirror it here.

Usage:
    python predict_eff.py --params 53e6 --blimp 72.0 --arc 40.0 --wiki 2.9
"""
import argparse
import math

# Snapshot of a few current rows (from the live board) for context.
# fields: name, params, blimp, arc, wiki
BOARD = [
    ("GPT-X2-125M",        125e6, 81.28, 57.07, 1.86),
    ("Haidass-143M-v1",    143e6, 79.05, 60.23, 1.89),
    ("Glint-1.3 (merged)", 0.982e6, 68.7, 32.5, 3.08),
    ("Supra-50M-Instruct", 51.8e6, 76.3, 52.2, 2.56),
    ("GPT-S-5M",           5.16e6, 72.27, 35.69, 2.57),
    ("Glint-2",            1.71e6, 66.36, 36.8, 3.09),
]

# size-bonus endpoints (approx smallest / largest params on the board)
MIN_P, MAX_P = 1e3, 143e6


def norm_wiki(ppl):
    # crude: map byte-perplexity to a 0-100 "goodness" score. Lower ppl -> higher.
    # Approximation only; replace with the board's real transform for exact ranks.
    return max(0.0, min(100.0, 100.0 * (1.0 - (ppl - 1.5) / 3.0)))


def size_bonus(params):
    p = max(MIN_P, min(MAX_P, params))
    frac = (math.log(MAX_P) - math.log(p)) / (math.log(MAX_P) - math.log(MIN_P))
    return 1.0 + 0.5 * frac  # 1.0 (largest) .. 1.5 (smallest)


def eff(params, blimp, arc, wiki):
    score = (blimp + arc + norm_wiki(wiki)) / 3.0
    return score * size_bonus(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=float, required=True)
    ap.add_argument("--blimp", type=float, required=True)
    ap.add_argument("--arc", type=float, required=True)
    ap.add_argument("--wiki", type=float, required=True)
    ap.add_argument("--name", default="YOUR MODEL")
    a = ap.parse_args()

    rows = [(n, eff(p, b, ar, w), p, b, ar, w) for (n, p, b, ar, w) in BOARD]
    mine = (a.name, eff(a.params, a.blimp, a.arc, a.wiki),
            a.params, a.blimp, a.arc, a.wiki)
    rows.append(mine)
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"{'#':>2}  {'model':<22}{'EFF~':>7}{'params':>10}"
          f"{'BLiMP':>8}{'ARC':>8}{'wiki':>7}")
    for i, (n, e, p, b, ar, w) in enumerate(rows, 1):
        mark = "  <-- you" if n == a.name else ""
        print(f"{i:>2}  {n:<22}{e:>7.2f}{p/1e6:>9.2f}M{b:>8.2f}{ar:>8.2f}"
              f"{w:>7.2f}{mark}")
    print("\n(EFF is APPROXIMATE — for exact ranks mirror the index.html JS.)")


if __name__ == "__main__":
    main()
