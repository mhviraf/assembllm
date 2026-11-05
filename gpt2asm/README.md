# llm.c GPT-2 forward pass in ARM64 assembly

A hand-written AArch64 port of the inference half of Karpathy's [llm.c](https://github.com/karpathy/llm.c) (`train_gpt2.c`). It reads the same checkpoint and tokenizer files, runs the same kernels, and samples text with the same RNG, so its output matches llm.c's.

## Run it on an Apple Silicon Mac

```sh
git clone https://github.com/karpathy/llm.c
(cd llm.c && ./dev/download_starter_pack.sh)   # gpt2_124M.bin + gpt2_tokenizer.bin
make
./gpt2 llm.c/gpt2_124M.bin llm.c/gpt2_tokenizer.bin 64
```

The same source also builds on aarch64 Linux.

## What's in `gpt2_forward.S`

- **Kernels:** encoder, layernorm, matmul (weights repacked at load time into 16-channel panels, 5×16 NEON register tiles), causal attention, GELU, residual and softmax. `exp` and `tanh` are vectorized and hand-written.
- **Program:** checkpoint loader, decode-only tokenizer, xorshift sampler and the generation loop.
- **Threads:** a small pthreads pool splits the matmuls, attention, GELU and softmax across cores. Set `GPT2A_THREADS=n`; the default is all CPUs.
- **libc use:** limited to file I/O, `malloc`/`free`, `printf`/`putchar` and `pthread_create`/`pthread_join`.

## Check it against llm.c

```sh
make test LLMC=llm.c
```

This builds llm.c's `train_gpt2.c` next to the assembly and checks three things:

1. Every kernel, including odd sizes, against the C version.
2. The full forward pass on the real 124M weights.
3. The generated text, compared byte for byte.

## Limits

- Forward pass only; there's no training.
- Like llm.c, it recomputes the whole sequence for every new token (no KV cache).
- Generation starts from `<|endoftext|>` with no prompt, because llm.c's tokenizer only decodes.
