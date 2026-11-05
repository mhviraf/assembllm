# assembllm

GPT-2 inference in hand-written ARM64 assembly. It's a port of the forward pass of Karpathy's [llm.c](https://github.com/karpathy/llm.c): it reads the same checkpoint and tokenizer files and produces the same output.

The code lives in [`gpt2asm/`](gpt2asm). That folder's README covers how to build it, run it and check it against llm.c.

## Speed vs llm.c (CPU)

| Threads | llm.c (ms/sample) | assembly (ms/sample) | assembly speedup |
|---|---|---|---|
| 1 | 391 | 189 | **2.07×** |
| 4 | 174 | 78 | **2.22×** |

**Setup:**
- **Machine:** MacBook Air (Apple silicon), Linux arm64 VM with 4 cores, gcc 11.4 with `-Ofast -march=native`.
- **Threading:** llm.c is multi-threaded with OpenMP; the assembly uses its own pthreads pool.
- **Workload:** 1,000 samples of 64 tokens, one forward pass each (B=1), at GPT-2 124M size. The two programs run interleaved and take turns going first, so warm-up and the Air's thermal throttling affect both equally.
- **Weights:** random, in GPT-2 124M's exact shapes. Speed doesn't depend on the weight values.
- **Agreement with llm.c:** max probability difference 4e-10, and the same top token at all 64,000 positions.

## Optimization log

I measured each step with a short run: 30 samples of 64 tokens, interleaved with llm.c. Absolute times drift as the fanless Air heats up, but both programs drift together. So the speedup over llm.c is the number to compare between steps. **Step gain** is how much that speedup changed from the previous step.

| # | Change | 1 thread: ms · ×llm.c · step gain | 4 threads: ms · ×llm.c · step gain |
|---|---|---|---|
| 0 | Baseline: 4×4 dot-product NEON matmul, libc `expf`/`tanhf`, single-threaded | 280 · 1.31× | – |
| 1 | pthreads pool: matmul split by output channel, attention by head, GELU by element, softmax by row | 286 · 1.30× · −1% | 138 · 1.11× |
| 2 | Matmul loop order: weight blocks outside, rows inside | 291 · 1.27× · −2% | 128 · 1.15× · +4% |
| 3 | Software prefetch of the next weight block | 231 · 1.59× · **+25%** | 97 · 1.54× · **+34%** |
| 4 | Vectorized `exp`/`tanh`, plus weights repacked for an 8×8 kernel | 197 · 1.85× · **+16%** | 59 · 2.06× · **+34%** |
| 5 | 5×16 kernel (20 accumulators instead of 16) | 182 · 2.03× · **+10%** | 60 · 2.12× · +3% |

What each step taught me:

1. **Threads.** On 4 threads this was only 1.11× faster than llm.c. Profiling showed why: the final logits matmul barely got faster with more threads. Its 154 MB weight table comes from main memory on every pass.
2. **Loop order.** I swapped the loops so each weight is read from memory once per call instead of 16 times. It barely helped, which showed that memory bandwidth wasn't the limit.
3. **Prefetch.** This was the real fix. The cores were waiting on memory, because short bursts of weight reads are something the CPU's hardware prefetcher can't anticipate. Fetching the next block in software cut the logits matmul from 111 to 65 ms on 1 thread, and from 58 to 20 ms on 4.
4. **Vector exp/tanh and repacked weights.** I measured these two together. GELU dropped from 13 to 2 ms per forward pass and softmax from 7.5 to 2.5 ms. With weights rearranged once at load time, the kernel needs no summing-up step at the end and loads less data per multiply-add, reaching about 80 GFLOP/s per core.
5. **5×16 tile.** The 8×8 tile's 16 accumulators each received a new multiply-add exactly as fast as the previous one finished (4 cycles), so any hiccup stalled the core. A 5×16 tile has 20 accumulators, which gives the core slack, and loads less per multiply-add. It reaches about 94 GFLOP/s per core, close to peak. On 4 threads the gain is within noise.

The full 1,000-sample runs of the final version are in the table at the top: 2.07× on 1 thread and 2.22× on 4.

## Reproduce

```sh
cd gpt2asm && ./run_bench.sh          # downloads Karpathy's GPT-2 124M weights; uses all CPU threads
./run_bench.sh 1000 1                 # single-threaded
```

On macOS, llm.c needs `brew install libomp` to run multi-threaded.
