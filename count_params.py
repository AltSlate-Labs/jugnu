"""Sanity-check the model size before spending GPU-days on it."""
from model import make_model
import config as C


def main():
    model, tok = make_model()
    total = sum(p.numel() for p in model.parameters())
    # tied embeddings are stored once; report the effective footprint too
    emb = model.get_input_embeddings().weight.numel()
    print(f"model:            {C.MODEL_NAME}")
    print(f"vocab size:       {len(tok)}")
    print(f"total params:     {total:,}  (~{total/1e6:.1f}M)")
    print(f"embedding params: {emb:,}  (tied={C.TIE_EMBEDDINGS})")
    print(f"non-embed params: {total-emb:,}  (~{(total-emb)/1e6:.1f}M)")
    assert total < 150_000_000, "OVER the 150M leaderboard limit!"
    print("OK: under the 150M leaderboard limit.")


if __name__ == "__main__":
    main()
