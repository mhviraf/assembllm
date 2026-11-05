# assembllm

GPT-2 inference in hand-written ARM64 assembly. It's a port of the forward pass of Karpathy's [llm.c](https://github.com/karpathy/llm.c): it reads the same checkpoint and tokenizer files and produces the same output.

The code lives in [`gpt2asm/`](gpt2asm). That folder's README covers how to build it, run it and check it against llm.c.

## Speed vs llm.c (CPU)

These runs use Karpathy's real GPT-2 124M weights on Tiny Shakespeare (val split), 64 tokens per sample, one forward pass each. Loss is the mean next-token cross-entropy; llm.c's fp32 is the reference.

| Precision | 4 threads: ms/sample · ×llm.c | 1 thread: ms/sample · ×llm.c | Loss change vs llm.c | Top token differs |
|---|---|---|---|---|
| llm.c (fp32, OpenMP) | 176–199 · 1× | 403–405 · 1× | – | – |
| **fp32** | 66 · **2.68×** | 183 · **2.20×** | +0.0001 | 1 of 32,000 |
| **fp16** | 38 · **4.78×** | 102 · **4.00×** | +0.0001 | 2.6% |
| **int8** (SmoothQuant) | 21 · **8.90×** | 54 · **7.62×** | +0.022 | 15% |

At 4 threads that's about 340 tokens/s for llm.c versus 1,090 (fp32), 1,670 (fp16) and 3,000 (int8) for the assembly.

- **fp32:** 500 samples on 4 threads, 200 on 1.
- **fp16 and int8:** 150 samples on 4 threads, 120 on 1, measured after the attention rewrite (step 12 in the log below).

- **fp32** is numerically equivalent to llm.c, differing only by rounding (total variation distance ≤ 1e-4 per position). Generated text matches llm.c token for token until the random draw falls in the long tail of tiny probabilities. There, rounding-level differences can pick a neighbouring token, just as two builds of llm.c with different compiler flags would.
- **fp16** is practically lossless.
- **int8** costs about 0.03 nats of loss. Without SmoothQuant it costs 0.69, so calibration is essential.

**Setup:**
- **Machine:** MacBook Air (Apple M1), Linux arm64 VM with 4 cores, gcc 11.4 with `-Ofast -march=native`.
- **Threading:** llm.c uses OpenMP; the assembly uses its own pthreads pool.
- **Measurement:** the two programs run interleaved and take turns going first, so warm-up and the fanless Air's throttling hit both equally. Absolute times drift between runs; compare the ×llm.c ratios.
- **OpenMP waiting:** run the 4-thread comparison with `OMP_WAIT_POLICY=passive`. By default, llm.c's OpenMP threads keep spinning for a while after each pass and steal cores from the assembly pass that follows. With `active` they never stop, and the assembly drops from 20 to 392 ms per pass. With `passive`, llm.c's own time barely changes (186 vs 189 ms), and int8 measures **9.53×** on 4 threads instead of 8.90×. `run_bench.sh` sets it.

## Optimization log

### Part 1: same math as llm.c (fp32)

Each step was measured with a short run: 30 samples of 64 tokens, using random weights in GPT-2 124M's exact shapes and interleaved with llm.c. **Step gain** is how much the speedup over llm.c changed from the previous step.

| # | Change | 1 thread: ms · ×llm.c · step gain | 4 threads: ms · ×llm.c · step gain |
|---|---|---|---|
| 0 | Baseline: 4×4 dot-product NEON matmul, libc `expf`/`tanhf`, single-threaded | 280 · 1.31× | – |
| 1 | pthreads pool: matmul split by output channel, attention by head, GELU by element, softmax by row | 286 · 1.30× · −1% | 138 · 1.11× |
| 2 | Matmul loop order: weight blocks outside, rows inside | 291 · 1.27× · −2% | 128 · 1.15× · +4% |
| 3 | Software prefetch of the next weight block | 231 · 1.59× · **+25%** | 97 · 1.54× · **+34%** |
| 4 | Vectorized `exp`/`tanh`, plus weights repacked for an 8×8 kernel | 197 · 1.85× · **+16%** | 59 · 2.06× · **+34%** |
| 5 | 5×16 kernel (20 accumulators instead of 16) | 182 · 2.03× · **+10%** | 60 · 2.12× · +3% |

1. **Threads.** On 4 threads this was only 1.11× faster than llm.c. Profiling showed why: the final logits matmul barely got faster with more threads. Its 154 MB weight table comes from main memory on every pass.
2. **Loop order.** Swapping the loops means each weight is read from memory once per call instead of 16 times. It barely helped, which showed that memory bandwidth wasn't the limit.
3. **Prefetch.** This was the real fix. The cores were waiting on memory, because short bursts of weight reads are something the CPU's hardware prefetcher can't anticipate. Fetching the next block in software cut the logits matmul from 111 to 65 ms on 1 thread, and from 58 to 20 ms on 4.
4. **Vector exp/tanh and repacked weights.** These two were measured together. GELU dropped from 13 to 2 ms per forward pass and softmax from 7.5 to 2.5 ms. With weights rearranged once at load time, the kernel needs no summing-up step at the end and loads less data per multiply-add, reaching about 80 GFLOP/s per core.
5. **5×16 tile.** The 8×8 tile's 16 accumulators each received a new multiply-add exactly as fast as the previous one finished (4 cycles), so any hiccup stalled the core. A 5×16 tile has 20 accumulators, which gives the core slack. It reaches about 94 GFLOP/s per core, close to the 100 GFLOP/s peak (4 FMA pipes × 4 lanes × 3.15 GHz).

### Part 2: lower precision

The fp32 kernel was at about 94% of the core's fp32 peak, so the next gains had to come from lower precision. A microbenchmark on the M1 core shows the same 4 vector instructions per cycle for each type, but each instruction does more work at lower precision:

| Instruction | Multiply-adds/s per core |
|---|---|
| fp32 `fmla` | 50 G |
| fp16 `fmla` | 101 G |
| int8 `sdot` | 203 G |

There is no bf16, i8mm or SME on the M1.

These steps were measured with the real weights: 100 samples, 4 threads, Tiny Shakespeare val.

| # | Change | ×llm.c (4 threads) | Loss change | Top token differs |
|---|---|---|---|---|
| 6 | fp32, plus multi-threaded layernorm, residual and input conversion | 2.61× → 2.76× | +0.0001 | 0% |
| 7 | fp16 weights and inputs, 5×32 kernel, fp16 sums folded into fp32 every 128 k | 4.81× | −0.010 | 4.4% |
| 8 | … folded every 64 k instead (kept) | 4.66× | **−0.0003** | 2.6% |
| 9 | int8 `sdot`, per-channel weight scales, per-row input scales | 8.27× | **+0.69** | 66% |
| 10 | + SmoothQuant calibration (α = 0.5) | 9.13× | +0.052 | 11% |
| 11 | + α = 0.25 (kept; α = 0.75 gave +0.128) | 9.26× | **+0.029** | 14% |

6. **Parallel small steps.** Layernorm, the residual add and the fp32→fp16/int8 input conversion had been single-threaded. That mattered once the matmuls got fast.
7. **fp16.** The products add up in fp16 and are folded into fp32 every 128 terms. That was fast, but the loss moved by 0.01.
8. **Folding every 64 terms** made fp16 essentially lossless for about 3% of speed (folding every 32 gave +0.0012).
9. **int8.** The `sdot` instruction adds int8 products into exact int32 sums, so the only error is rounding values to int8. But GPT-2 has a few input channels with values about 100× larger than the rest. With one scale per row, those channels flatten every other value to almost nothing.
10. **SmoothQuant.** A one-time calibration pass on 1,024 training tokens records each input channel's maximum. Then channel k of the activations is divided by s_k, and the matching weight column is multiplied by s_k: `s_k = max|x_k|^α / max|W_k|^(1−α)`. The math is unchanged, but the outliers move into the weights, which have one scale per output channel.
11. **α = 0.25** moves less into the weights. That works better for GPT-2, whose weights are already harder to quantize than OPT-style models.

### Part 3: what's left besides the matmuls

`GPT2A_PROFILE=1` prints a per-stage breakdown. It times each stage by reading the CPU's counter register (`cntvct_el0`) directly in assembly. On 1 thread with int8, the matmuls ran at about 90% of the `sdot` peak, and the other stages were about 17% of the time.

These numbers use the real weights. Stage times are per forward pass of 64 tokens, measured without llm.c in the loop. The ×llm.c figures come from interleaved runs on 4 threads (150 samples) and 1 thread (120 samples).

| # | Change | Effect |
|---|---|---|
| 12 | Attention for head size 64: query in 16 registers, 4 keys scored per pass, the 64-wide output kept in registers while summing over values. It replaces a dot-product call per query/key pair. | Attention 7.0 → 2.6 ms (1 thread), 1.6 → 0.6 ms (4 threads). int8 7.20× → **7.62×** (1 thread), 8.73× → **8.90×** (4 threads). fp16 3.88× → 4.00×, 4.63× → 4.78×. |
| 13 | GELU as x / (1 + e^(−2u)) using a reciprocal estimate plus Newton steps, and softmax multiplying by 1/sum instead of dividing | No measurable change: GELU stays at 2.2 ms, limited by the `exp` polynomial rather than the division. Kept because it's simpler. |

int8 on 1 thread now takes about 52 ms per forward pass, split as follows:

| Stage | ms |
|---|---|
| Matmuls (qkv, attention projection, MLP, logits) | 43 |
| Attention | 2.6 |
| Softmax | 2.3 |
| GELU | 2.2 |
| Layernorm | 0.9 |
| int8 input conversion | 0.8 |

## Reproduce

```sh
cd gpt2asm && ./run_bench.sh                    # downloads Karpathy's weights; fp32, all CPU threads
./run_bench.sh 1000 1                           # single-threaded
GPT2A_PRECISION=fp16 ./run_bench.sh 500         # fp16
GPT2A_PRECISION=int8 ./run_bench.sh 500         # int8: calibrates on the train split, evaluates on val
GPT2A_PROFILE=1 GPT2A_PRECISION=int8 ./run_bench.sh 100   # plus a per-stage time breakdown
```

On macOS, llm.c needs `brew install libomp` to run multi-threaded.
