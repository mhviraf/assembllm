"""Write random-weight GPT-2 checkpoints and a toy tokenizer in llm.c's file formats.

The real GPT-2 weights come from llm.c's dev/download_starter_pack.sh; these files
let the assembly be tested against the C reference without them.
"""
import sys
import numpy as np


def write_checkpoint(path, maxT, V, Vp, L, NH, C, seed=0, emb_std=0.3):
    rng = np.random.default_rng(seed)
    header = np.zeros(256, dtype=np.int32)
    header[:8] = [20240326, 3, maxT, V, L, NH, C, Vp]
    shapes = [
        ("wte", (Vp, C), emb_std), ("wpe", (maxT, C), emb_std),
        ("ln1w", (L, C), None), ("ln1b", (L, C), 0.02),
        ("qkvw", (L, 3 * C, C), 0.02 * 4), ("qkvb", (L, 3 * C), 0.02),
        ("attprojw", (L, C, C), 0.02 * 4), ("attprojb", (L, C), 0.02),
        ("ln2w", (L, C), None), ("ln2b", (L, C), 0.02),
        ("fcw", (L, 4 * C, C), 0.02 * 4), ("fcb", (L, 4 * C), 0.02),
        ("fcprojw", (L, C, 4 * C), 0.02 * 4), ("fcprojb", (L, C), 0.02),
        ("lnfw", (C,), None), ("lnfb", (C,), 0.02),
    ]
    with open(path, "wb") as f:
        f.write(header.tobytes())
        for name, shape, std in shapes:
            if std is None:  # layernorm gains around 1
                t = 1.0 + 0.1 * rng.standard_normal(shape)
            else:
                t = std * rng.standard_normal(shape)
            if name == "wte":
                t[V:] = 0.0  # padded rows, as in llm.c
            f.write(t.astype(np.float32).tobytes())


def write_tokenizer(path, V):
    header = np.zeros(256, dtype=np.uint32)
    header[:4] = [20240328, 2, V, V - 1]
    with open(path, "wb") as f:
        f.write(header.tobytes())
        for i in range(V):
            b = bytes([i]) if i < 256 else f" w{i}".encode()
            if i == V - 1:
                b = b"<|endoftext|>"
            f.write(bytes([len(b)]) + b)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    write_checkpoint(f"{out}/tiny.bin", maxT=64, V=1000, Vp=1024, L=2, NH=4, C=64)
    write_checkpoint(f"{out}/odd.bin", maxT=16, V=301, Vp=303, L=3, NH=3, C=30, seed=1)
    write_tokenizer(f"{out}/tiny_tok.bin", 1000)
    write_tokenizer(f"{out}/odd_tok.bin", 301)
    if "--124m" in sys.argv:
        write_checkpoint(f"{out}/gpt2_124M_random.bin", maxT=1024, V=50257, Vp=50304,
                         L=12, NH=12, C=768, seed=2, emb_std=0.1)
        write_tokenizer(f"{out}/gpt2_tok_fake.bin", 50257)
