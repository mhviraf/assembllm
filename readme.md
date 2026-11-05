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

At 4 threads that's about 340 tokens/s for llm.c versus 1,090 (fp32), 1,670 (fp16) and 3,000 (int8) for the assembly. These are scoring rates: one forward pass predicts all 64 positions at once. Generating text is sequential, one token per pass; see [Generating text](#generating-text).

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

## Native macOS, including the AMX coprocessor

These runs are native macOS on the same MacBook Air (M1), built without Xcode (see `gpt2asm/native/README.md`). They use 4 threads, the real weights and 40 forward passes of 64 tokens over Tiny Shakespeare val. Loss is the mean next-token loss over those passes; fp32 is the reference. llm.c isn't in this table because it can't be built natively here without Xcode.

| Precision | ms per pass | Tokens/s | vs NEON fp32 | Loss |
|---|---|---|---|---|
| fp32, NEON | 49.8 | 1,284 | 1× | 5.2520 |
| fp16, NEON | 28.0 | 2,286 | 1.78× | 5.2499 |
| **fp32, AMX** (`amx`) | 30.0 | 2,135 | **1.66×** | 5.2520 (identical) |
| **fp16 with fp32 sums, AMX** (`amx16`) | 18.3 | 3,501 | **2.72×** | 5.2519 |
| int8, NEON (SmoothQuant) | 14.4 | 4,434 | 3.45× | 5.2686 |

`amx16` is the fastest mode that keeps fp32's loss: int8 is about 25% faster but adds 0.017 nats.

## Generating text

`gpt2` samples text from `<|endoftext|>` with llm.c's RNG and prints exactly what llm.c's sampling loop prints. llm.c reruns the whole sequence for every new token, so token t costs a forward pass over t positions. The assembly keeps a KV cache instead: every layer's q, k, v rows stay in memory (about 9 MB per layer per 1,000 tokens, 113 MB for all 12 layers at the full 1,024), and each step runs only the newest token through the model (`gpt2a_forward_kv`). The math per position is unchanged, and the generated text is byte-identical.

Each token then costs one pass over the weights, so generation is limited by memory bandwidth, and smaller weights help most. These numbers are for 256 tokens on 4 threads in the Linux VM:

| Precision | Tokens/s | Weights read per token |
|---|---|---|
| fp32 | 82 | 500 MB |
| fp16 | 142 | 250 MB |
| int8 | 291 | 125 MB |

Before the cache, 63 tokens took about 2 s of wall time in fp32 (7.7 s of CPU time across 4 threads), and each new token got slower. Now they take 0.77 s, and every token costs the same.

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

### Part 4: Apple's AMX matrix coprocessor (native macOS)

The M1 has an undocumented matrix unit, AMX, next to the performance cores' shared L2. Each instruction does a 16 × 16 fp32 outer product (or 32 × 32 in fp16) into a 4 KB accumulator, which is far more math per instruction than NEON. The encoding comes from [corsix/amx](https://github.com/corsix/amx). AMX only works natively, not in a Linux VM, so step 14 first had to make native builds possible without Xcode.

| # | Change | Native, 4 threads, ms per pass | Loss |
|---|---|---|---|
| 14 | Native macOS build without Xcode: clang + `ld64.lld` against a hand-written `libSystem.tbd` | baseline: NEON fp32 49.8 (fp16 28.0, int8 14.4) | – |
| 15 | fp32 AMX kernel using only X0 / Y0 (the operand fields the first probe had confirmed), 2 AMX threads | 34.8 (1.40× NEON fp32) | identical |
| 16 | Operand fields fixed, 128-byte pair loads, CPU prefetch of the weight stream, threads pinned to the performance cores, 4 AMX threads | **30.0** (1.66×) | identical |
| 17 | fp16 weights and inputs with fp32 sums (`fma16` with fp32 accumulators), weights packed in the lane order AMX writes back | **18.3** (2.72×) | −0.0001 |

14. **Building without Xcode.** `clang -target arm64-apple-macos11` makes Mach-O objects on Linux. `ld64.lld` links them against a text file that lists the libSystem functions used, and signs the result ad hoc. macOS runs it as is.
15. **Finding out how AMX really behaves.** The first matmul got rows 0–15 right and everything after wrong. A register-by-register probe settled it: `fma32` takes the **Y** offset in bits 0–8 and the **X** offset in bits 10–18, the opposite of what the kernel assumed. Pair loads work, but only from 128-byte-aligned addresses; a 64-byte-aligned one is a bus error. Before the fix was known, a kernel that touches only X0 and Y0 avoided the question and already beat NEON fp32. `pmatmul` checks each AMX kernel against the NEON kernel on first use and keeps the first one that agrees.
16. **Corrected kernels.** The faster kernels load 128 bytes at once and use all four accumulators. In isolation they reach 1.1–1.4 TFLOP/s, against about 0.4 for NEON on 4 cores. In a real forward pass the weights come from DRAM, and AMX loads alone don't keep enough of them in flight. So each kernel also prefetches its weight stream from the CPU side, and 4 threads issuing AMX work beat 2 (30 vs 39 ms), even though they share one AMX unit.
17. **fp16 inputs, fp32 sums.** With bit 62 set, `fma16` multiplies 32 fp16 weights by 32 fp16 inputs into a 32 × 32 **fp32** tile. That halves the weight bytes and the load instructions, and each product is exact in fp32. Lane i of output row j lands in Z row 2j + i % 2, column i / 2. Packing each 32-channel panel in the order 0, 16, 1, 17, … makes each Z row hold 16 consecutive channels, so results store straight to memory. Unlike the NEON fp16 path, nothing is summed in fp16, and the loss matches fp32.

`amx16` per-stage times (ms per pass): qkv 2.5, attention projection 0.9, MLP 3.3 + 3.6, logits 5.0, everything else 2.8. The matmuls still run about 20% slower than in isolation, where the weights stay cached.

### Part 5: generation

| # | Change | Effect |
|---|---|---|
| 18 | KV cache: `gpt2a_forward_kv` runs only the new tokens and keeps each layer's q, k, v rows. Attention starts at the first new query (`G_ATT_T0`). The generator's timer uses wall-clock time instead of `clock()`, which added up CPU time across threads. | 63 tokens: 2 s → 0.77 s (fp32, Linux VM, 4 threads), identical text. With 256 tokens: 82 tokens/s fp32, 142 fp16, 291 int8. |

## Reproduce

```sh
cd gpt2asm && ./run_bench.sh                    # downloads Karpathy's weights; fp32, all CPU threads
./run_bench.sh 1000 1                           # single-threaded
GPT2A_PRECISION=fp16 ./run_bench.sh 500         # fp16
GPT2A_PRECISION=int8 ./run_bench.sh 500         # int8: calibrates on the train split, evaluates on val
GPT2A_PROFILE=1 GPT2A_PRECISION=int8 ./run_bench.sh 100   # plus a per-stage time breakdown
```

On macOS, llm.c needs `brew install libomp` to run multi-threaded.

Native macOS (no Xcode needed; the binaries are built on Linux by `gpt2asm/native/build.sh` and `gpt2asm/amx/build.sh`):

```sh
bash gpt2asm/amx/run_all.sh               # AMX kernel checks, then amx / amx16 / NEON timings -> gpt2asm/amx/results.txt
cd gpt2asm/native && GPT2A_PRECISION=amx16 ./gpt2_profile ../../gpt2_124M.bin ../../tiny_shakespeare_val.bin 4 40
```
