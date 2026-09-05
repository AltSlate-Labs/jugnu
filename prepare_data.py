"""Download FineWeb-Edu, tokenize with the SmolLM2 tokenizer, and write uint16
memmap .bin files (nanoGPT-style). Run once before training.

Note: downloads the full sample-10BT (~30GB) and writes ~20GB of .bin files.
Uses multiprocessing for tokenization; set NPROC to your core count.
"""
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from model import make_tokenizer
import config as C

NPROC = max(1, (os.cpu_count() or 8) - 2)


def main():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    enc = make_tokenizer()
    eos_id = enc.eos_token_id

    def process(example):
        ids = enc.encode(example["text"])
        ids.append(eos_id)
        return {"ids": ids, "len": len(ids)}

    print(f"loading {C.HF_DATASET}:{C.HF_DATASET_CONFIG} ...")
    ds = load_dataset(C.HF_DATASET, name=C.HF_DATASET_CONFIG,
                      split="train", num_proc=NPROC)

    split = ds.train_test_split(test_size=C.VAL_FRACTION, seed=C.SEED, shuffle=True)
    split["val"] = split.pop("test")

    tokenized = split.map(
        process,
        remove_columns=ds.column_names,
        desc="tokenizing",
        num_proc=NPROC,
    )

    for name, dset in tokenized.items():
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))
        path = os.path.join(C.DATA_DIR, f"{name}.bin")
        print(f"writing {path}: {arr_len:,} tokens ({arr_len*2/1e9:.1f} GB)")
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_shards = 1024
        idx = 0
        for s in tqdm(range(total_shards), desc=f"writing {name}"):
            batch = dset.shard(num_shards=total_shards, index=s,
                               contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    print("done.")


if __name__ == "__main__":
    main()
