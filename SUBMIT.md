# Submitting to the Tiny-ML Leaderboard

The board is a **static HTML Space**. Entries live in a JS array inside
`index.html`. Submission = a PR that adds one object to that array (plus a public
HF model card with your eval command, so results are reproducible).

## 1. Publish the model
```bash
huggingface-cli login
huggingface-cli upload altslate/JugnuLM-53M out/final .
# then paste MODEL_CARD.md (with real numbers) as the repo README
```

## 2. Fork + edit the leaderboard Space
- Open https://huggingface.co/spaces/Glint-Research/Tiny-ML-Leaderboard
- Files → `index.html` → find the models array → add your object → open a PR.
  (Or open a Discussion asking them to add it, per their README.)

## 3. Your entry object (fill in the blanks from eval)
```javascript
{
  name: "JugnuLM-53M",
  org: "altslate",
  params: "53M",
  blimp: 0.0,      // BLiMP acc  (percent, e.g. 72.4)
  arc: 0.0,        // ARC-Easy acc (percent)
  aci: null,       // proprietary AxiomicLabs metric — leave null
  wiki: 0.0,       // WikiText-2 byte_perplexity (lower is better)
  tokens: "12B",
  releaseDate: "2026-09-05",
  links: { card: "https://huggingface.co/altslate/JugnuLM-53M" }
}
```

## Checklist before you PR
- [ ] `python count_params.py` shows < 150M
- [ ] `./eval.sh` produced blimp / arc_easy / wikitext numbers
- [ ] model card includes the exact `lm_eval` command (reproducibility bar)
- [ ] `python predict_eff.py --params 53e6 --blimp .. --arc .. --wiki ..`
      to see where you'd land
