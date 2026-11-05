# assembllm

GPT-2 inference in hand-written ARM64 assembly. It's a port of the forward pass of Karpathy's [llm.c](https://github.com/karpathy/llm.c): it reads the same checkpoint and tokenizer files and produces the same output.

The code lives in [`gpt2asm/`](gpt2asm). That folder's README covers how to build it, run it and check it against llm.c.

## Speed vs llm.c (CPU)

| Threads | llm.c (ms/sample) | assembly (ms/sample) | assembly speedup |
|---|---|---|---|
| 1 | 408 | 264 | **1.54×** |
| 4 | 189 | 126 | **1.50×** |

**Setup:**
- **Machine:** MacBook Air (Apple silicon), Linux arm64 VM with 4 cores, gcc 11.4 with `-Ofast -march=native`.
- **Threading:** llm.c is multi-threaded with OpenMP; the assembly uses its own pthreads pool.
- **Workload:** 1,000 samples of 64 tokens, one forward pass each (B=1), at GPT-2 124M size. The two programs run interleaved and take turns going first, so warm-up and the Air's thermal throttling affect both equally.
- **Weights:** random, in GPT-2 124M's exact shapes. Speed doesn't depend on the weight values.
- **Agreement with llm.c:** max probability difference 9e-12, and the same top token at all 64,000 positions.

Short 30-sample runs, before the fanless Air heats up:

| Threads | llm.c | assembly | speedup |
|---|---|---|---|
| 1 | 367 ms | 231 ms | 1.59× |
| 2 | 213 ms | 140 ms | 1.52× |
| 3 | 184 ms | 119 ms | 1.55× |
| 4 | 149 ms | 97 ms | 1.54× |

## Where the speed comes from

- **Register-tiled matmul:** 4 rows × 4 output channels per tile, using 16 NEON accumulators. Weight blocks sit on the outer loop, so each weight is read once per call. A single core reaches 70–80 GFLOP/s.
- **Software prefetch:** the next weight block is fetched while the current one is in use. This mattered most for the final logits matmul, whose 154 MB weight table comes from main memory: it dropped from 111 ms to 65 ms on 1 thread, and from 58 ms to 20 ms on 4.
- **Thread pool:** workers start at the beginning of each forward pass and stop at the end. Matmuls are split by output channel, attention by head, GELU by element and softmax by row.

## Reproduce

```sh
cd gpt2asm && ./run_bench.sh          # downloads Karpathy's GPT-2 124M weights; uses all CPU threads
./run_bench.sh 1000 1                 # single-threaded
```

On macOS, llm.c needs `brew install libomp` to run multi-threaded.
