// gpt2_forward.S
//
// GPT-2 inference (forward pass + sampling) in hand-written ARM64 assembly.
// A port of the forward half of Andrej Karpathy's llm.c (train_gpt2.c):
// same checkpoint format, same tokenizer format, same kernels, same sampler.
//
// Builds on macOS (Apple Silicon) and on aarch64 Linux:
//     cc -O2 gpt2_forward.S -o gpt2 -lm
//     ./gpt2 [gpt2_124M.bin] [gpt2_tokenizer.bin] [num_tokens]
//
// Only libc is used: fopen/fread/fclose, malloc/free, printf/putchar/fflush,
// atoi, getenv, sysconf, clock, exit, and pthread_create/join. Everything else,
// including exp and tanh (vectorized, Cephes-style polynomial), is written by hand.
//
// Matmuls: at load time each weight matrix is repacked into panels of 16 output
// channels (k-major), so the kernel computes 5 rows x 16 channels with 20 vector
// accumulators and fmla-by-element: 80 multiply-adds per 9 loads, no horizontal
// sums, and enough independent accumulators to keep all 4 FMA pipes busy.
// Shapes that do not allow packing use the 4x4 dot-product kernel.
//
// GPT2A_PRECISION=fp16: weights are stored as fp16 in 32-channel panels and each
// matmul input is converted to fp16. The kernel computes 5 rows x 32 channels with
// 20 fp16 accumulators (2x the multiply-adds of fp32 per instruction) and folds
// them into fp32 outputs every 64 k, so the long sums still happen in fp32.
//
// GPT2A_PRECISION=int8: weights are int8 with one scale per output channel, and
// each matmul input is quantized to int8 with one scale per row. sdot adds 16
// int8 products per instruction into exact int32 sums (4x fp32's multiply-adds);
// the only error is the int8 rounding of weights and inputs.
// With GPT2A_CALIB=<llm.c token file>, int8 is calibrated SmoothQuant-style: a
// few fp32 forward passes record each input channel's max |x|, then channel k is
// divided by s_k = max|x_k|^a / max|W_k|^(1-a) in the activations and multiplied
// by s_k in the weights, moving GPT-2's outlier channels into the weights.
// a = 0.25 by default (best for GPT-2 124M); GPT2A_SQ_ALPHA=50 or 75 for 0.5, 0.75.
//
// Threads: the matmuls, attention, GELU and softmax are split across a small
// thread pool (GPT2A_THREADS=n, default: all CPUs). Workers are started at the
// beginning of each forward pass and joined at the end, so nothing spins between
// passes. Work is handed out with a generation counter and a done counter
// (load-acquire / store-release and an exclusive-monitor increment).
//
// Calling convention: standard AAPCS64. x18 is never touched (reserved on
// Apple). Varargs for printf go through helpers that put the arguments both
// in registers (Linux) and on the stack (Apple), so one source serves both.

// fp16 vector arithmetic (FEAT_FP16) and int8 dot products are used by the
// reduced-precision kernels; every Apple silicon core has both.
    .arch   armv8.2-a+fp16+dotprod

#if defined(__APPLE__)
#define SYM(x) _##x
#else
#define SYM(x) x
#endif

// ---------------------------------------------------------------------------
// macros

// define a global function
.macro FUNC name
    .text
    .p2align 4
    .globl \name
#if !defined(__APPLE__)
    .type \name, %function
#endif
\name:
.endm

// load the address of a local data symbol
.macro LOADADDR reg, sym
#if defined(__APPLE__)
    adrp    \reg, \sym@PAGE
    add     \reg, \reg, \sym@PAGEOFF
#else
    adrp    \reg, \sym
    add     \reg, \reg, :lo12:\sym
#endif
.endm

// load a 32-bit constant into a w register
.macro MOV32 wreg, imm
    movz    \wreg, #((\imm) & 0xffff)
    movk    \wreg, #(((\imm) >> 16) & 0xffff), lsl #16
.endm

// load a float constant (given as its IEEE-754 bit pattern) into an s register
.macro FLOADK sreg, wtmp, bits
    MOV32   \wtmp, \bits
    fmov    \sreg, \wtmp
.endm

// save / restore x19..x28 (frame must already hold x29, x30 at [sp])
.macro SAVE_X19_X28
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
.endm
.macro RESTORE_X19_X28
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
.endm

// --- vector exp: constants live in v20-v31 (load once with VEXP_CONSTS)
// v20 hi clamp, v21 lo clamp, v22 log2(e), v23/v24 ln2 split, v25-v30 poly, v31 1.0
.macro VEXP_CONSTS
    LOADADDR x9, vexp_consts
    ld1r    {v20.4s}, [x9], #4
    ld1r    {v21.4s}, [x9], #4
    ld1r    {v22.4s}, [x9], #4
    ld1r    {v23.4s}, [x9], #4
    ld1r    {v24.4s}, [x9], #4
    ld1r    {v25.4s}, [x9], #4
    ld1r    {v26.4s}, [x9], #4
    ld1r    {v27.4s}, [x9], #4
    ld1r    {v28.4s}, [x9], #4
    ld1r    {v29.4s}, [x9], #4
    ld1r    {v30.4s}, [x9], #4
    ld1r    {v31.4s}, [x9], #4
.endm

// x = exp(x), 4 lanes. exp(x) = 2^n * e^r, n = round(x / ln2), |r| <= ln2/2,
// e^r = 1 + r + r^2 * poly(r). Inputs are clamped to [-86, 88].
.macro VEXP x, n, p, q
    fmin    \x\().4s, \x\().4s, v20.4s
    fmax    \x\().4s, \x\().4s, v21.4s
    fmul    \n\().4s, \x\().4s, v22.4s
    frintn  \n\().4s, \n\().4s
    fmls    \x\().4s, \n\().4s, v23.4s
    fmls    \x\().4s, \n\().4s, v24.4s
    mov     \p\().16b, v26.16b
    fmla    \p\().4s, v25.4s, \x\().4s
    mov     \q\().16b, v27.16b
    fmla    \q\().4s, \p\().4s, \x\().4s
    mov     \p\().16b, v28.16b
    fmla    \p\().4s, \q\().4s, \x\().4s
    mov     \q\().16b, v29.16b
    fmla    \q\().4s, \p\().4s, \x\().4s
    mov     \p\().16b, v30.16b
    fmla    \p\().4s, \q\().4s, \x\().4s
    fmul    \q\().4s, \x\().4s, \x\().4s
    fadd    \x\().4s, \x\().4s, v31.4s
    fmla    \x\().4s, \p\().4s, \q\().4s
    fcvtzs  \n\().4s, \n\().4s
    shl     \n\().4s, \n\().4s, #23
    add     \x\().4s, \x\().4s, \n\().4s
.endm

// v4 = x  ->  v0 = 0.5 x (1 + tanh(u)) = x / (1 + e^(-2u)),  u = s (x + 0.044715 x^3)
// The reciprocal is an estimate plus two Newton steps (no divide).
// needs v16 = 0.044715, v17 = -2 sqrt(2/pi), exp constants. Uses v1-v3.
.macro GELU4
    fmul    v1.4s, v16.4s, v4.4s
    fmul    v1.4s, v1.4s, v4.4s
    fmul    v1.4s, v1.4s, v4.4s
    fadd    v1.4s, v4.4s, v1.4s
    fmul    v0.4s, v17.4s, v1.4s        // -2u
    VEXP    v0, v1, v2, v3
    fadd    v0.4s, v0.4s, v31.4s        // 1 + e^(-2u)
    frecpe  v1.4s, v0.4s
    frecps  v2.4s, v0.4s, v1.4s
    fmul    v1.4s, v1.4s, v2.4s
    frecps  v2.4s, v0.4s, v1.4s
    fmul    v1.4s, v1.4s, v2.4s
    fmul    v0.4s, v4.4s, v1.4s
.endm

// 5x16 kernel: one k of 5 rows (inputs v0-v4, lane i) times 16 channels (v5-v8)
// accumulators: row 0 -> v12-v15, row 1 -> v16-v19, ..., row 4 -> v28-v31
.macro FMA5x16 i
    fmla    v12.4s, v5.4s, v0.s[\i]
    fmla    v13.4s, v6.4s, v0.s[\i]
    fmla    v14.4s, v7.4s, v0.s[\i]
    fmla    v15.4s, v8.4s, v0.s[\i]
    fmla    v16.4s, v5.4s, v1.s[\i]
    fmla    v17.4s, v6.4s, v1.s[\i]
    fmla    v18.4s, v7.4s, v1.s[\i]
    fmla    v19.4s, v8.4s, v1.s[\i]
    fmla    v20.4s, v5.4s, v2.s[\i]
    fmla    v21.4s, v6.4s, v2.s[\i]
    fmla    v22.4s, v7.4s, v2.s[\i]
    fmla    v23.4s, v8.4s, v2.s[\i]
    fmla    v24.4s, v5.4s, v3.s[\i]
    fmla    v25.4s, v6.4s, v3.s[\i]
    fmla    v26.4s, v7.4s, v3.s[\i]
    fmla    v27.4s, v8.4s, v3.s[\i]
    fmla    v28.4s, v5.4s, v4.s[\i]
    fmla    v29.4s, v6.4s, v4.s[\i]
    fmla    v30.4s, v7.4s, v4.s[\i]
    fmla    v31.4s, v8.4s, v4.s[\i]
.endm

// four k steps: 5 input vectors (4 k each) and 4 x 16 packed weights
.macro KSTEP5
    ld1     {v0.4s}, [x0], #16
    ld1     {v1.4s}, [x1], #16
    ld1     {v2.4s}, [x2], #16
    ld1     {v3.4s}, [x3], #16
    ld1     {v4.4s}, [x4], #16
    ld1     {v5.4s, v6.4s, v7.4s, v8.4s}, [x8], #64
    FMA5x16 0
    ld1     {v5.4s, v6.4s, v7.4s, v8.4s}, [x8], #64
    FMA5x16 1
    ld1     {v5.4s, v6.4s, v7.4s, v8.4s}, [x8], #64
    FMA5x16 2
    ld1     {v5.4s, v6.4s, v7.4s, v8.4s}, [x8], #64
    FMA5x16 3
.endm

// fp16 5x32 kernel: one k of 5 rows (inputs v0-v4, lane i) times 32 channels (v8-v11)
.macro FMA5x32 i
    fmla    v12.8h, v8.8h, v0.h[\i]
    fmla    v13.8h, v9.8h, v0.h[\i]
    fmla    v14.8h, v10.8h, v0.h[\i]
    fmla    v15.8h, v11.8h, v0.h[\i]
    fmla    v16.8h, v8.8h, v1.h[\i]
    fmla    v17.8h, v9.8h, v1.h[\i]
    fmla    v18.8h, v10.8h, v1.h[\i]
    fmla    v19.8h, v11.8h, v1.h[\i]
    fmla    v20.8h, v8.8h, v2.h[\i]
    fmla    v21.8h, v9.8h, v2.h[\i]
    fmla    v22.8h, v10.8h, v2.h[\i]
    fmla    v23.8h, v11.8h, v2.h[\i]
    fmla    v24.8h, v8.8h, v3.h[\i]
    fmla    v25.8h, v9.8h, v3.h[\i]
    fmla    v26.8h, v10.8h, v3.h[\i]
    fmla    v27.8h, v11.8h, v3.h[\i]
    fmla    v28.8h, v8.8h, v4.h[\i]
    fmla    v29.8h, v9.8h, v4.h[\i]
    fmla    v30.8h, v10.8h, v4.h[\i]
    fmla    v31.8h, v11.8h, v4.h[\i]
.endm
.macro W32
    ld1     {v8.8h, v9.8h, v10.8h, v11.8h}, [x8], #64
.endm
// eight k steps: 5 input vectors (8 k each) and 8 x 32 packed fp16 weights
.macro KSTEP16
    ld1     {v0.8h}, [x0], #16
    ld1     {v1.8h}, [x1], #16
    ld1     {v2.8h}, [x2], #16
    ld1     {v3.8h}, [x3], #16
    ld1     {v4.8h}, [x4], #16
    W32
    FMA5x32 0
    W32
    FMA5x32 1
    W32
    FMA5x32 2
    W32
    FMA5x32 3
    W32
    FMA5x32 4
    W32
    FMA5x32 5
    W32
    FMA5x32 6
    W32
    FMA5x32 7
.endm
// add 4 fp16 accumulators (32 channels) into the fp32 output row at x9
.macro FOLD32 a, b, c, d
    fcvtl   v0.4s, \a\().4h
    fcvtl2  v1.4s, \a\().8h
    fcvtl   v2.4s, \b\().4h
    fcvtl2  v3.4s, \b\().8h
    ld1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x9]
    fadd    v8.4s, v8.4s, v0.4s
    fadd    v9.4s, v9.4s, v1.4s
    fadd    v10.4s, v10.4s, v2.4s
    fadd    v11.4s, v11.4s, v3.4s
    st1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x9], #64
    fcvtl   v0.4s, \c\().4h
    fcvtl2  v1.4s, \c\().8h
    fcvtl   v2.4s, \d\().4h
    fcvtl2  v3.4s, \d\().8h
    ld1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x9]
    fadd    v8.4s, v8.4s, v0.4s
    fadd    v9.4s, v9.4s, v1.4s
    fadd    v10.4s, v10.4s, v2.4s
    fadd    v11.4s, v11.4s, v3.4s
    st1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x9]
.endm

// int8 5x16 kernel: one group of 4 k (lane g of inputs v0-v4) times 16 channels (v8-v11)
.macro SDOT20 g
    sdot    v12.4s, v8.16b, v0.4b[\g]
    sdot    v13.4s, v9.16b, v0.4b[\g]
    sdot    v14.4s, v10.16b, v0.4b[\g]
    sdot    v15.4s, v11.16b, v0.4b[\g]
    sdot    v16.4s, v8.16b, v1.4b[\g]
    sdot    v17.4s, v9.16b, v1.4b[\g]
    sdot    v18.4s, v10.16b, v1.4b[\g]
    sdot    v19.4s, v11.16b, v1.4b[\g]
    sdot    v20.4s, v8.16b, v2.4b[\g]
    sdot    v21.4s, v9.16b, v2.4b[\g]
    sdot    v22.4s, v10.16b, v2.4b[\g]
    sdot    v23.4s, v11.16b, v2.4b[\g]
    sdot    v24.4s, v8.16b, v3.4b[\g]
    sdot    v25.4s, v9.16b, v3.4b[\g]
    sdot    v26.4s, v10.16b, v3.4b[\g]
    sdot    v27.4s, v11.16b, v3.4b[\g]
    sdot    v28.4s, v8.16b, v4.4b[\g]
    sdot    v29.4s, v9.16b, v4.4b[\g]
    sdot    v30.4s, v10.16b, v4.4b[\g]
    sdot    v31.4s, v11.16b, v4.4b[\g]
.endm
// sixteen k: 5 input vectors (16 int8 k each), 4 x (16 channels x 4 k) weights
.macro KSTEPQ
    ld1     {v0.16b}, [x0], #16
    ld1     {v1.16b}, [x1], #16
    ld1     {v2.16b}, [x2], #16
    ld1     {v3.16b}, [x3], #16
    ld1     {v4.16b}, [x4], #16
    ld1     {v8.16b, v9.16b, v10.16b, v11.16b}, [x8], #64
    SDOT20  0
    ld1     {v8.16b, v9.16b, v10.16b, v11.16b}, [x8], #64
    SDOT20  1
    ld1     {v8.16b, v9.16b, v10.16b, v11.16b}, [x8], #64
    SDOT20  2
    ld1     {v8.16b, v9.16b, v10.16b, v11.16b}, [x8], #64
    SDOT20  3
.endm
// int32 sums -> float, times the 16 channel scales (v8-v11) and the row scale
.macro DEQ a, b, c, d, rs
    scvtf   \a\().4s, \a\().4s
    scvtf   \b\().4s, \b\().4s
    scvtf   \c\().4s, \c\().4s
    scvtf   \d\().4s, \d\().4s
    fmul    \a\().4s, \a\().4s, v8.4s
    fmul    \b\().4s, \b\().4s, v9.4s
    fmul    \c\().4s, \c\().4s, v10.4s
    fmul    \d\().4s, \d\().4s, v11.4s
    fmul    \a\().4s, \a\().4s, \rs
    fmul    \b\().4s, \b\().4s, \rs
    fmul    \c\().4s, \c\().4s, \rs
    fmul    \d\().4s, \d\().4s, \rs
.endm
.macro BIAS16 a, b, c, d
    fadd    \a\().4s, \a\().4s, v0.4s
    fadd    \b\().4s, \b\().4s, v1.4s
    fadd    \c\().4s, \c\().4s, v2.4s
    fadd    \d\().4s, \d\().4s, v3.4s
.endm

// MMHOOK inp, rows, channels, stats_offset: before a matmul in the forward pass.
// While calibrating, records max |x| per input channel. After calibration (int8),
// sets x7 = this matmul's 1/s for the quantizer; otherwise x7 = 0. Uses x11, x12.
.macro MMHOOK inpreg, nreg, kreg, offreg
    mov     x7, #0
    ldr     x11, [x28, #G_STATS]
    cbz     x11, 98f
    add     x11, x11, \offreg, lsl #2
    ldr     x12, [x28, #G_CALIB]
    cbz     x12, 97f
    stp     x0, x1, [sp, #-64]!
    stp     x2, x3, [sp, #16]
    stp     x4, x5, [sp, #32]
    str     x6, [sp, #48]
    mov     x0, x11
    mov     x1, \inpreg
    mov     x2, \nreg
    mov     x3, \kreg
    bl      SYM(a_colmax)
    ldp     x2, x3, [sp, #16]
    ldp     x4, x5, [sp, #32]
    ldr     x6, [sp, #48]
    ldp     x0, x1, [sp], #64
    b       98f
97: mov     x7, x11
98:
.endm

// profiling: add the time since the last mark to slot, then mark (uses x9-x13)
.macro PROFTICK base, slot
    ldr     x9, [\base, #G_PROFON]
    cbz     x9, 96f
    mrs     x10, cntvct_el0
    ldr     x11, [\base, #G_PROF_LAST]
    sub     x12, x10, x11
    ldr     x13, [\base, #(G_PROF + 8 * \slot)]
    add     x13, x13, x12
    str     x13, [\base, #(G_PROF + 8 * \slot)]
    str     x10, [\base, #G_PROF_LAST]
96:
.endm

// float constants (bit patterns)
.equ F_NEG10000,   0xC61C4000      // -10000.0f, initial max in llm.c
.equ F_EPS,        0x3727C5AC      // 1e-5f, layernorm epsilon
.equ F_GELU_CUBE,  0x3D372713      // 0.044715f
.equ F_GELU_SCALE, 0x3F4C4229      // sqrtf(2.0f / M_PI)
.equ F_GELU_M2S,   0xBFCC4229      // -2 * sqrtf(2.0f / M_PI)
.equ F_2POW24,     0x4B800000      // 16777216.0f

.equ MODEL_MAGIC,  20240326
.equ TOK_MAGIC,    20240328

// layout of the global state block G (all 64-bit slots)
.equ G_MAXT,        0
.equ G_V,           8
.equ G_L,           16
.equ G_NH,          24
.equ G_C,           32
.equ G_VP,          40
.equ G_NPARAMS,     48
.equ G_PARAMS_MEM,  56
.equ G_ACT_CAP,     64       // activation capacity, in rows (B*T)
.equ G_ACT_X,       72       // residual stream      (N, C)
.equ G_ACT_LN,      80       // layernorm output     (N, C)
.equ G_ACT_QKV,     88       // q, k, v              (N, 3C)
.equ G_ACT_ATTY,    96       // attention output     (N, C)
.equ G_ACT_PROJ,    104      // projection output    (N, C)
.equ G_ACT_FCH,     112      // MLP hidden / gelu    (N, 4C)
.equ G_ACT_PROBS,   120      // logits, then probs   (N, Vp)
.equ G_ACT_SCRATCH, 128      // one attention row    (maxT)
.equ G_TOK_OK,      136
.equ G_TOK_VOCAB,   144
.equ G_TOK_EOT,     152
.equ G_TOK_TABLE,   160
.equ G_RNG,         168
.equ G_PARAMS,      176      // 16 tensor pointers, llm.c order
.equ G_PSIZES,      304      // 16 tensor sizes (floats)
.equ G_HEADER,      432      // 256 x int32 file header
.equ G_BYTE,        1456     // 1-byte read buffer
.equ G_POOL_GEN,    1536     // work generation counter (own cache line)
.equ G_POOL_DONE,   1600     // workers finished this generation (own cache line)
.equ G_NTHREADS,    1664     // threads used by the forward pass (0 = not chosen yet)
.equ G_POOL_FN,     1672     // part function for the current generation (0 = exit)
.equ G_TASK,        1680     // up to 8 arguments for the part function
.equ G_TIDS,        1744     // pthread_t of each worker (64 x 8 bytes)
.equ G_PACKED,      2304     // 1 when matmul weights are repacked for the 5x16 kernel
.equ G_WTE_PACKED,  2312     // packed copy of wte for the logits matmul
.equ G_ACT_H16,     2320     // fp16 copy of the current matmul input (N x 4C)
.equ G_ACT_Q8,      2328     // int8 copy of the current matmul input (N x 4C)
.equ G_ACT_XS,      2336     // its per-row scales (N + 8 floats)
.equ G_CALIB,       2344     // 1 while calibrating: record matmul input channel maxima
.equ G_STATS,       2352     // per-layer [C | C | C | 4C] + final [C]: max |x| (then 1/s)
.equ G_SCALES,      2360     // same layout: s (applied to the weights)
.equ G_ALPHA,       2368     // SmoothQuant alpha x 100: 25 (default), 50 or 75
.equ G_PROFON,      2376     // 1 when GPT2A_PROFILE is set
.equ G_PROF_LAST,   2384     // counter value at the last profiling mark
.equ G_PROF_N,      2392     // forward passes profiled
.equ G_PROF,        2400     // 12 per-stage tick totals
.equ G_SIZE,        2496
.equ NPROF,         12
.equ KC16,          64       // fp16 accumulation block before folding into fp32
.equ MAX_THREADS,   64

// parameter tensor slots (offsets into G_PARAMS), llm.c order
.equ P_WTE,      (G_PARAMS + 0)
.equ P_WPE,      (G_PARAMS + 8)
.equ P_LN1W,     (G_PARAMS + 16)
.equ P_LN1B,     (G_PARAMS + 24)
.equ P_QKVW,     (G_PARAMS + 32)
.equ P_QKVB,     (G_PARAMS + 40)
.equ P_ATTPROJW, (G_PARAMS + 48)
.equ P_ATTPROJB, (G_PARAMS + 56)
.equ P_LN2W,     (G_PARAMS + 64)
.equ P_LN2B,     (G_PARAMS + 72)
.equ P_FCW,      (G_PARAMS + 80)
.equ P_FCB,      (G_PARAMS + 88)
.equ P_FCPROJW,  (G_PARAMS + 96)
.equ P_FCPROJB,  (G_PARAMS + 104)
.equ P_LNFW,     (G_PARAMS + 112)
.equ P_LNFB,     (G_PARAMS + 120)

// ===========================================================================
// data

    .data
    .p2align 6
G:  .space G_SIZE

    .p2align 4
vexp_consts:
    .word   0x42B00000             // hi clamp
    .word   0xC2AC0000             // lo clamp
    .word   0x3FB8AA3B             // log2(e)
    .word   0x3F318000             // ln2 hi
    .word   0xB95E8083             // ln2 lo
    .word   0x39506967             // p0
    .word   0x3AB743CE             // p1
    .word   0x3C088908             // p2
    .word   0x3D2AA9C1             // p3
    .word   0x3E2AAAAA             // p4
    .word   0x3F000000             // p5
    .word   0x3F800000             // 1.0

s_rb:            .asciz "rb"
s_default_ckpt:  .asciz "gpt2_124M.bin"
s_default_tok:   .asciz "gpt2_tokenizer.bin"
s_cfg_head:      .asciz "[GPT-2 forward, ARM64 assembly]\n"
s_cfg_maxT:      .asciz "max_seq_len: %ld\n"
s_cfg_V:         .asciz "vocab_size: %ld\n"
s_cfg_Vp:        .asciz "padded_vocab_size: %ld\n"
s_cfg_L:         .asciz "num_layers: %ld\n"
s_cfg_NH:        .asciz "num_heads: %ld\n"
s_cfg_C:         .asciz "channels: %ld\n"
s_cfg_np:        .asciz "num_parameters: %ld\n"
s_generating:    .asciz "generating:\n---\n"
s_tokid:         .asciz "%ld "
s_timing:        .asciz "\n---\ngenerated %ld tokens in %.2f s (CPU time)\n"
s_err_open:      .asciz "Error: could not open file %s\n"
s_err_read:      .asciz "Error: short read from file\n"
s_err_magic:     .asciz "Bad magic model file\n"
s_err_version:   .asciz "Bad version in model file (need version 3, fp32)\n"
s_err_heads:     .asciz "Error: channels must be divisible by num_heads\n"
s_err_tmagic:    .asciz "Bad magic tokenizer file\n"
s_err_tversion:  .asciz "Bad version in tokenizer file\n"
s_err_tlen:      .asciz "Bad token length in tokenizer file\n"
s_err_malloc:    .asciz "Error: malloc failed\n"
s_err_init:      .asciz "Error: model was not initialized properly.\n"
s_err_T:         .asciz "Error: T is larger than max_seq_len\n"
s_err_cap:       .asciz "Error: B*T is larger than the allocated activations\n"
s_err_tok:       .asciz "Error: input token out of range [0, vocab_size)\n"
s_warn_tok:      .asciz "WARNING: could not open tokenizer file %s, printing token ids instead\n"
s_invalid_tok:   .asciz "invalid token id %ld!\n"
s_prof_names:
    .ascii "encoder        \0"
    .ascii "layernorm      \0"
    .ascii "qkv matmul     \0"
    .ascii "attention      \0"
    .ascii "attproj matmul \0"
    .ascii "residual       \0"
    .ascii "fc matmul      \0"
    .ascii "gelu           \0"
    .ascii "fcproj matmul  \0"
    .ascii "logits matmul  \0"
    .ascii "softmax        \0"
    .ascii "input convert  \0"
s_prof_fmt:      .asciz "  %s %8.3f ms\n"
s_prof_head:     .asciz "per forward pass (assembly, averaged over %ld passes):\n"
s_prof_total:    .asciz "  total           %8.3f ms\n"
s_env_profile:   .asciz "GPT2A_PROFILE"
s_env_threads:   .asciz "GPT2A_THREADS"
s_env_prec:      .asciz "GPT2A_PRECISION"
s_env_calib:     .asciz "GPT2A_CALIB"
s_env_alpha:     .asciz "GPT2A_SQ_ALPHA"
s_calib:         .asciz "calibrating int8 (SmoothQuant) on %ld tokens\n"
s_nocalib:       .asciz "note: int8 without GPT2A_CALIB=<token file> is uncalibrated and much less accurate\n"
s_err_data:      .asciz "Error: bad calibration token file\n"
s_prec16:        .asciz "precision: fp16 weights and matmul inputs, fp32 sums\n"
s_prec8:         .asciz "precision: int8 weights (per channel) and matmul inputs (per row), int32 sums\n"
s_threads:       .asciz "threads: %ld\n"
s_err_thread:    .asciz "Error: pthread_create failed\n"

// ===========================================================================
// small helpers

// printf with no varargs: x0 = format
    .text
    .p2align 4
pr0:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      SYM(printf)
    ldp     x29, x30, [sp], #16
    ret

// printf with one integer/pointer vararg: x0 = format, x1 = value
    .p2align 4
pr1:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #16
    str     x1, [sp]                // Apple passes varargs on the stack
    bl      SYM(printf)
    mov     sp, x29
    ldp     x29, x30, [sp], #16
    ret

// printf with an integer then a double: x0 = format, x1 = int, d0 = double
    .p2align 4
pr1f:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #16
    str     x1, [sp]
    str     d0, [sp, #8]
    bl      SYM(printf)
    mov     sp, x29
    ldp     x29, x30, [sp], #16
    ret

// print a message (x0 = format, x1 = optional arg) and exit(1). Never returns.
    .p2align 4
die:
    bl      pr1
    mov     w0, #1
    bl      SYM(exit)

// malloc that dies on failure: x0 = bytes -> x0 = pointer
    .p2align 4
xmalloc:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      SYM(malloc)
    cbz     x0, 1f
    ldp     x29, x30, [sp], #16
    ret
1:  LOADADDR x0, s_err_malloc
    b       die

// fread that dies on a short read: x0 = ptr, x1 = size, x2 = count, x3 = FILE*
    .p2align 4
xfread:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     x19, [sp, #16]
    mov     x19, x2
    bl      SYM(fread)
    cmp     x0, x19
    b.ne    1f
    ldr     x19, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret
1:  LOADADDR x0, s_err_read
    b       die

// ===========================================================================
// a_dot: s0 = dot(x9[0..x11), x10[0..x11))   (internal calling convention)
// clobbers x9, x10, x11, v0-v7, v16-v19. Preserves every other register.

FUNC SYM(a_dot)
    movi    v4.16b, #0
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.16b, #0
1:  cmp     x11, #16
    b.lt    2f
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x9], #64
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x10], #64
    fmla    v4.4s, v0.4s, v16.4s
    fmla    v5.4s, v1.4s, v17.4s
    fmla    v6.4s, v2.4s, v18.4s
    fmla    v7.4s, v3.4s, v19.4s
    sub     x11, x11, #16
    b       1b
2:  cmp     x11, #4
    b.lt    3f
    ld1     {v0.4s}, [x9], #16
    ld1     {v16.4s}, [x10], #16
    fmla    v4.4s, v0.4s, v16.4s
    sub     x11, x11, #4
    b       2b
3:  fadd    v4.4s, v4.4s, v5.4s
    fadd    v6.4s, v6.4s, v7.4s
    fadd    v4.4s, v4.4s, v6.4s
    faddp   v4.4s, v4.4s, v4.4s
    faddp   s0, v4.2s
4:  cbz     x11, 5f
    ldr     s1, [x9], #4
    ldr     s2, [x10], #4
    fmadd   s0, s1, s2, s0
    sub     x11, x11, #1
    b       4b
5:  ret

// ===========================================================================
// kernels (forward only). Shapes follow llm.c; N = B*T rows.

// void a_encoder_forward(float* out, const int* inp, const float* wte,
//                        const float* wpe, long B, long T, long C)
// out[b,t,:] = wte[inp[b,t],:] + wpe[t,:]
FUNC SYM(a_encoder_forward)
    lsl     x7, x6, #2                  // C*4 bytes
    mov     x8, #0                      // b
1:  cmp     x8, x4
    b.ge    9f
    mov     x9, #0                      // t
2:  cmp     x9, x5
    b.ge    8f
    madd    x10, x8, x5, x9             // b*T + t
    ldrsw   x11, [x1, x10, lsl #2]      // token id
    mul     x12, x11, x7
    add     x12, x2, x12                // wte row
    mul     x13, x9, x7
    add     x13, x3, x13                // wpe row
    mul     x14, x10, x7
    add     x14, x0, x14                // out row
    mov     x15, x6
3:  cmp     x15, #4
    b.lt    4f
    ld1     {v0.4s}, [x12], #16
    ld1     {v1.4s}, [x13], #16
    fadd    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x14], #16
    sub     x15, x15, #4
    b       3b
4:  cbz     x15, 5f
    ldr     s0, [x12], #4
    ldr     s1, [x13], #4
    fadd    s0, s0, s1
    str     s0, [x14], #4
    sub     x15, x15, #1
    b       4b
5:  add     x9, x9, #1
    b       2b
8:  add     x8, x8, #1
    b       1b
9:  ret

// void a_layernorm_forward(float* out, const float* inp, const float* weight,
//                          const float* bias, long N, long C)
FUNC SYM(a_layernorm_forward)
    FLOADK  s21, w9, F_EPS
    fmov    s22, #1.0
    scvtf   s20, x5                     // C as float
    lsl     x15, x5, #2                 // row bytes
    mov     x8, #0                      // row
ln_row:
    cmp     x8, x4
    b.ge    ln_done
    mul     x9, x8, x15
    add     x10, x1, x9                 // input row
    add     x11, x0, x9                 // output row
    // mean
    movi    v16.16b, #0
    mov     x12, x10
    mov     x13, x5
1:  cmp     x13, #4
    b.lt    2f
    ld1     {v0.4s}, [x12], #16
    fadd    v16.4s, v16.4s, v0.4s
    sub     x13, x13, #4
    b       1b
2:  faddp   v16.4s, v16.4s, v16.4s
    faddp   s16, v16.2s
3:  cbz     x13, 4f
    ldr     s0, [x12], #4
    fadd    s16, s16, s0
    sub     x13, x13, #1
    b       3b
4:  fdiv    s17, s16, s20               // m
    dup     v17.4s, v17.s[0]
    // variance (no bias correction)
    movi    v18.16b, #0
    mov     x12, x10
    mov     x13, x5
5:  cmp     x13, #4
    b.lt    6f
    ld1     {v0.4s}, [x12], #16
    fsub    v0.4s, v0.4s, v17.4s
    fmla    v18.4s, v0.4s, v0.4s
    sub     x13, x13, #4
    b       5b
6:  faddp   v18.4s, v18.4s, v18.4s
    faddp   s18, v18.2s
7:  cbz     x13, 8f
    ldr     s0, [x12], #4
    fsub    s0, s0, s17
    fmadd   s18, s0, s0, s18
    sub     x13, x13, #1
    b       7b
8:  fdiv    s18, s18, s20               // v
    fadd    s18, s18, s21               // v + eps
    fsqrt   s18, s18
    fdiv    s19, s22, s18               // rstd
    dup     v19.4s, v19.s[0]
    // out = (rstd * (x - m)) * w + b
    mov     x12, x10
    mov     x13, x5
    mov     x14, x2
    mov     x16, x3
    mov     x17, x11
9:  cmp     x13, #4
    b.lt    10f
    ld1     {v0.4s}, [x12], #16
    ld1     {v1.4s}, [x14], #16
    ld1     {v2.4s}, [x16], #16
    fsub    v0.4s, v0.4s, v17.4s
    fmul    v0.4s, v0.4s, v19.4s
    fmla    v2.4s, v0.4s, v1.4s
    st1     {v2.4s}, [x17], #16
    sub     x13, x13, #4
    b       9b
10: cbz     x13, 11f
    ldr     s0, [x12], #4
    ldr     s1, [x14], #4
    ldr     s2, [x16], #4
    fsub    s0, s0, s17
    fmul    s0, s0, s19
    fmadd   s2, s0, s1, s2
    str     s2, [x17], #4
    sub     x13, x13, #1
    b       10b
11: add     x8, x8, #1
    b       ln_row
ln_done:
    ret

// void a_matmul_forward(float* out, const float* inp, const float* weight,
//                       const float* bias, long N, long C, long OC)
// out[n,o] = bias[o] + sum_i inp[n,i] * weight[o,i]     (bias may be NULL)
FUNC SYM(a_matmul_forward)
    mov     x7, x6
    b       SYM(a_matmul_strided)

// void a_matmul_strided(float* out, const float* inp, const float* weight,
//                       const float* bias, long N, long C, long OC, long ldo)
// Same as above for OC output channels, but output rows are ldo floats apart,
// so a thread can compute a slice of the output channels in place.
// Main path: 4 rows x 4 output channels per tile, 16 vector accumulators.
// Leftover rows/channels (or C not a multiple of 4) use a single dot product.
FUNC SYM(a_matmul_strided)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    mov     x19, x0                     // out
    mov     x20, x1                     // inp
    mov     x21, x2                     // weight
    mov     x22, x3                     // bias
    mov     x23, x4                     // N
    mov     x24, x5                     // C
    mov     x25, x6                     // OC (channels computed here)
    mov     x16, x7                     // ldo (output row stride, floats)
    lsl     x26, x24, #2                // row bytes
    // Loop order: blocks of 4 output channels outside, blocks of 4 rows inside.
    // The 4 weight rows of a block stay in L1 while every input row streams
    // past them, so each weight is read from memory once per call.
    lsr     x14, x23, #2
    lsl     x14, x14, #2                // N4 = rows in full 4-row blocks
    lsr     x15, x25, #2
    lsl     x15, x15, #2                // OC4 = channels in full 4-channel blocks
    tst     x24, #3                     // tiles need C % 4 == 0
    b.eq    1f
    mov     x14, #0
    mov     x15, #0
1:  mov     x28, #0                     // o
mm_oblk:
    cmp     x28, x15
    b.ge    mm_otail
    mov     x27, #0                     // r
mm_rblk:
    cmp     x27, x14
    b.ge    mm_rtail
    // software prefetch: while this weight block is used for every row block,
    // pull in 1 KB of the next block per row block (64 rows cover 16 KB).
    // The hardware prefetcher cannot see these short bursts coming.
    add     x9, x28, #4
    mul     x9, x9, x26
    add     x9, x21, x9                 // next weight block
    add     x9, x9, x27, lsl #8         // + (r / 4) * 1 KB
    prfm    pldl1keep, [x9]
    prfm    pldl1keep, [x9, #64]
    prfm    pldl1keep, [x9, #128]
    prfm    pldl1keep, [x9, #192]
    prfm    pldl1keep, [x9, #256]
    prfm    pldl1keep, [x9, #320]
    prfm    pldl1keep, [x9, #384]
    prfm    pldl1keep, [x9, #448]
    prfm    pldl1keep, [x9, #512]
    prfm    pldl1keep, [x9, #576]
    prfm    pldl1keep, [x9, #640]
    prfm    pldl1keep, [x9, #704]
    prfm    pldl1keep, [x9, #768]
    prfm    pldl1keep, [x9, #832]
    prfm    pldl1keep, [x9, #896]
    prfm    pldl1keep, [x9, #960]
    mul     x9, x27, x26
    add     x0, x20, x9                 // input rows r .. r+3
    add     x1, x0, x26
    add     x2, x1, x26
    add     x3, x2, x26
    mul     x9, x28, x26
    add     x4, x21, x9                 // weight rows o .. o+3
    add     x5, x4, x26
    add     x6, x5, x26
    add     x7, x6, x26
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    movi    v24.16b, #0
    movi    v25.16b, #0
    movi    v26.16b, #0
    movi    v27.16b, #0
    movi    v28.16b, #0
    movi    v29.16b, #0
    movi    v30.16b, #0
    movi    v31.16b, #0
    mov     x8, x24
2:  ld1     {v0.4s}, [x0], #16
    ld1     {v1.4s}, [x1], #16
    ld1     {v2.4s}, [x2], #16
    ld1     {v3.4s}, [x3], #16
    ld1     {v4.4s}, [x4], #16
    ld1     {v5.4s}, [x5], #16
    ld1     {v6.4s}, [x6], #16
    ld1     {v7.4s}, [x7], #16
    fmla    v16.4s, v0.4s, v4.4s
    fmla    v17.4s, v0.4s, v5.4s
    fmla    v18.4s, v0.4s, v6.4s
    fmla    v19.4s, v0.4s, v7.4s
    fmla    v20.4s, v1.4s, v4.4s
    fmla    v21.4s, v1.4s, v5.4s
    fmla    v22.4s, v1.4s, v6.4s
    fmla    v23.4s, v1.4s, v7.4s
    fmla    v24.4s, v2.4s, v4.4s
    fmla    v25.4s, v2.4s, v5.4s
    fmla    v26.4s, v2.4s, v6.4s
    fmla    v27.4s, v2.4s, v7.4s
    fmla    v28.4s, v3.4s, v4.4s
    fmla    v29.4s, v3.4s, v5.4s
    fmla    v30.4s, v3.4s, v6.4s
    fmla    v31.4s, v3.4s, v7.4s
    subs    x8, x8, #4
    b.gt    2b
    // horizontal sums: each row ends up as one vector [o, o+1, o+2, o+3]
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v18.4s, v18.4s, v19.4s
    faddp   v16.4s, v16.4s, v18.4s
    faddp   v20.4s, v20.4s, v21.4s
    faddp   v22.4s, v22.4s, v23.4s
    faddp   v20.4s, v20.4s, v22.4s
    faddp   v24.4s, v24.4s, v25.4s
    faddp   v26.4s, v26.4s, v27.4s
    faddp   v24.4s, v24.4s, v26.4s
    faddp   v28.4s, v28.4s, v29.4s
    faddp   v30.4s, v30.4s, v31.4s
    faddp   v28.4s, v28.4s, v30.4s
    cbz     x22, 3f
    add     x9, x22, x28, lsl #2
    ld1     {v0.4s}, [x9]
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v20.4s, v20.4s, v0.4s
    fadd    v24.4s, v24.4s, v0.4s
    fadd    v28.4s, v28.4s, v0.4s
3:  mul     x9, x27, x16
    add     x9, x9, x28
    add     x9, x19, x9, lsl #2
    lsl     x10, x16, #2
    st1     {v16.4s}, [x9]
    add     x9, x9, x10
    st1     {v20.4s}, [x9]
    add     x9, x9, x10
    st1     {v24.4s}, [x9]
    add     x9, x9, x10
    st1     {v28.4s}, [x9]
    add     x27, x27, #4
    b       mm_rblk
mm_rtail:                               // leftover rows, channels o .. o+3
    cmp     x27, x23
    b.ge    mm_oblk_next
    mov     x12, x28
    mul     x9, x27, x26
    add     x13, x20, x9                // input row r
    mov     x0, x13
    mov     x1, x27
    bl      mm_one
    add     x28, x28, #1
    mov     x0, x13
    mov     x1, x27
    bl      mm_one
    add     x28, x28, #1
    mov     x0, x13
    mov     x1, x27
    bl      mm_one
    add     x28, x28, #1
    mov     x0, x13
    mov     x1, x27
    bl      mm_one
    mov     x28, x12
    add     x27, x27, #1
    b       mm_rtail
mm_oblk_next:
    add     x28, x28, #4
    b       mm_oblk
mm_otail:                               // leftover channels, every row
    cmp     x28, x25
    b.ge    mm_done
    mov     x27, #0
4:  cmp     x27, x23
    b.ge    5f
    mul     x9, x27, x26
    add     x0, x20, x9
    mov     x1, x27
    bl      mm_one
    add     x27, x27, #1
    b       4b
5:  add     x28, x28, #1
    b       mm_otail
mm_done:
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// mm_one: out[x1, o] = bias[o] + dot(x0, weight[o]), o = x28.
// Uses a_matmul_strided's registers x16, x19, x21, x22, x24, x26, x28.
    .p2align 4
mm_one:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    mov     x9, x0
    mul     x10, x28, x26
    add     x10, x21, x10
    mov     x11, x24
    bl      SYM(a_dot)
    cbz     x22, 1f
    ldr     s1, [x22, x28, lsl #2]
    fadd    s0, s0, s1
1:  mul     x9, x1, x16
    add     x9, x9, x28
    str     s0, [x19, x9, lsl #2]
    ldp     x29, x30, [sp], #16
    ret

// void a_attention_forward(float* out, const float* qkv, float* scratch,
//                          long B, long T, long C, long NH)
// qkv is (B,T,3C) holding q, k, v. out is (B,T,C). Causal softmax attention.
// scratch holds one row of attention weights (at least T floats).
FUNC SYM(a_attention_forward)
    lsl     x7, x6, #32                 // heads [0, NH)
    b       SYM(a_attention_heads)

// void a_attention_heads(float* out, const float* qkv, float* scratch,
//                        long B, long T, long C, long NH, long h0 | (h1 << 32))
// Attention for heads h0 .. h1-1 only (one thread's share).
FUNC SYM(a_attention_heads)
    stp     x29, x30, [sp, #-208]!
    mov     x29, sp
    SAVE_X19_X28
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #160]
    stp     d14, d15, [sp, #176]
    // locals: [sp, #144] first head offset, [sp, #152] end head offset,
    //         [sp, #192] scale (fp32 bits); x16 = 1 for the head-size-64 fast path
    mov     x19, x0                     // out
    mov     x20, x1                     // qkv
    mov     x21, x2                     // scratch
    mov     x22, x4                     // T
    mov     x23, x5                     // C
    mul     x24, x3, x4                 // B*T
    sdiv    x25, x5, x6                 // hs = C / NH
    mov     w9, w7
    mul     x9, x9, x25
    str     x9, [sp, #144]              // h0 * hs
    lsr     x9, x7, #32
    mul     x9, x9, x25
    str     x9, [sp, #152]              // h1 * hs
    VEXP_CONSTS                         // v20-v31 (a_dot leaves them alone)
    // scale = 1.0 / sqrtf(hs), float sqrt then double divide, like the C code
    scvtf   s0, x25
    fsqrt   s0, s0
    fcvt    d0, s0
    fmov    d1, #1.0
    fdiv    d0, d1, d0
    fcvt    s8, d0
    str     s8, [sp, #192]
    cmp     x25, #64
    cset    x16, eq                     // GPT-2: hs = 64 -> register-resident path
    mov     x26, #0                     // b*T
att_b:
    cmp     x26, x24
    b.ge    att_done
    mov     x27, #0                     // t
att_t:
    cmp     x27, x22
    b.ge    att_b_next
    ldr     x28, [sp, #144]             // h*hs
att_h:
    ldr     x9, [sp, #152]
    cmp     x28, x9
    b.ge    att_t_next
    // pass 1: score[t2] = (q . k_t2) * scale, track the max
    FLOADK  s9, w9, F_NEG10000
    add     x13, x23, x23, lsl #1       // C3 = 3C
    lsl     x14, x13, #2                // qkv row bytes
    add     x9, x26, x27
    mul     x9, x9, x13
    add     x9, x9, x28
    add     x15, x20, x9, lsl #2        // query
    mul     x9, x26, x13
    add     x9, x9, x23
    add     x9, x9, x28
    add     x13, x20, x9, lsl #2        // key of position 0
    mov     x12, #0                     // t2
    cbz     x16, 1f
    // fast path (hs = 64): q * scale in v0-v15, scores of 4 keys per pass. Keys past t
    // are clamped to key t (their scores are written past the row and never used).
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x15]
    add     x9, x15, #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x9]
    add     x9, x15, #128
    ld1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x9]
    add     x9, x15, #192
    ld1     {v12.4s, v13.4s, v14.4s, v15.4s}, [x9]
    add     x9, sp, #192
    ld1r    {v16.4s}, [x9]
    fmul    v0.4s, v0.4s, v16.4s
    fmul    v1.4s, v1.4s, v16.4s
    fmul    v2.4s, v2.4s, v16.4s
    fmul    v3.4s, v3.4s, v16.4s
    fmul    v4.4s, v4.4s, v16.4s
    fmul    v5.4s, v5.4s, v16.4s
    fmul    v6.4s, v6.4s, v16.4s
    fmul    v7.4s, v7.4s, v16.4s
    fmul    v8.4s, v8.4s, v16.4s
    fmul    v9.4s, v9.4s, v16.4s
    fmul    v10.4s, v10.4s, v16.4s
    fmul    v11.4s, v11.4s, v16.4s
    fmul    v12.4s, v12.4s, v16.4s
    fmul    v13.4s, v13.4s, v16.4s
    fmul    v14.4s, v14.4s, v16.4s
    fmul    v15.4s, v15.4s, v16.4s
    FLOADK  s24, w9, F_NEG10000
    dup     v24.4s, v24.s[0]
    mov     x10, #0                     // first key of the group
70: cmp     x10, x27
    b.gt    71f
    madd    x0, x10, x14, x13
    add     x9, x10, #1
    cmp     x9, x27
    csel    x9, x27, x9, gt
    madd    x1, x9, x14, x13
    add     x9, x10, #2
    cmp     x9, x27
    csel    x9, x27, x9, gt
    madd    x2, x9, x14, x13
    add     x9, x10, #3
    cmp     x9, x27
    csel    x9, x27, x9, gt
    madd    x3, x9, x14, x13
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v0.4s
    fmla    v17.4s, v21.4s, v0.4s
    fmla    v18.4s, v22.4s, v0.4s
    fmla    v19.4s, v23.4s, v0.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v1.4s
    fmla    v17.4s, v21.4s, v1.4s
    fmla    v18.4s, v22.4s, v1.4s
    fmla    v19.4s, v23.4s, v1.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v2.4s
    fmla    v17.4s, v21.4s, v2.4s
    fmla    v18.4s, v22.4s, v2.4s
    fmla    v19.4s, v23.4s, v2.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v3.4s
    fmla    v17.4s, v21.4s, v3.4s
    fmla    v18.4s, v22.4s, v3.4s
    fmla    v19.4s, v23.4s, v3.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v4.4s
    fmla    v17.4s, v21.4s, v4.4s
    fmla    v18.4s, v22.4s, v4.4s
    fmla    v19.4s, v23.4s, v4.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v5.4s
    fmla    v17.4s, v21.4s, v5.4s
    fmla    v18.4s, v22.4s, v5.4s
    fmla    v19.4s, v23.4s, v5.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v6.4s
    fmla    v17.4s, v21.4s, v6.4s
    fmla    v18.4s, v22.4s, v6.4s
    fmla    v19.4s, v23.4s, v6.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v7.4s
    fmla    v17.4s, v21.4s, v7.4s
    fmla    v18.4s, v22.4s, v7.4s
    fmla    v19.4s, v23.4s, v7.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v8.4s
    fmla    v17.4s, v21.4s, v8.4s
    fmla    v18.4s, v22.4s, v8.4s
    fmla    v19.4s, v23.4s, v8.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v9.4s
    fmla    v17.4s, v21.4s, v9.4s
    fmla    v18.4s, v22.4s, v9.4s
    fmla    v19.4s, v23.4s, v9.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v10.4s
    fmla    v17.4s, v21.4s, v10.4s
    fmla    v18.4s, v22.4s, v10.4s
    fmla    v19.4s, v23.4s, v10.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v11.4s
    fmla    v17.4s, v21.4s, v11.4s
    fmla    v18.4s, v22.4s, v11.4s
    fmla    v19.4s, v23.4s, v11.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v12.4s
    fmla    v17.4s, v21.4s, v12.4s
    fmla    v18.4s, v22.4s, v12.4s
    fmla    v19.4s, v23.4s, v12.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v13.4s
    fmla    v17.4s, v21.4s, v13.4s
    fmla    v18.4s, v22.4s, v13.4s
    fmla    v19.4s, v23.4s, v13.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v14.4s
    fmla    v17.4s, v21.4s, v14.4s
    fmla    v18.4s, v22.4s, v14.4s
    fmla    v19.4s, v23.4s, v14.4s
    ld1     {v20.4s}, [x0], #16
    ld1     {v21.4s}, [x1], #16
    ld1     {v22.4s}, [x2], #16
    ld1     {v23.4s}, [x3], #16
    fmla    v16.4s, v20.4s, v15.4s
    fmla    v17.4s, v21.4s, v15.4s
    fmla    v18.4s, v22.4s, v15.4s
    fmla    v19.4s, v23.4s, v15.4s
    faddp   v16.4s, v16.4s, v17.4s
    faddp   v18.4s, v18.4s, v19.4s
    faddp   v16.4s, v16.4s, v18.4s      // 4 scores
    add     x9, x21, x10, lsl #2
    st1     {v16.4s}, [x9]
    fmax    v24.4s, v24.4s, v16.4s
    add     x10, x10, #4
    b       70b
71: fmaxv   s9, v24.4s                  // max
    VEXP_CONSTS
    b       2f
1:  cmp     x12, x27
    b.gt    2f
    mov     x9, x15
    mov     x10, x13
    mov     x11, x25
    bl      SYM(a_dot)
    fmul    s0, s0, s8
    fcmp    s0, s9
    fcsel   s9, s0, s9, gt
    str     s0, [x21, x12, lsl #2]
    add     x13, x13, x14
    add     x12, x12, #1
    b       1b
2:  // pass 2: exp(score - max) and its sum, 4 at a time
    dup     v16.4s, v9.s[0]
    movi    v17.16b, #0
    movi    d18, #0
    mov     x9, x21
    add     x10, x27, #1
3:  cmp     x10, #4
    b.lt    4f
    ld1     {v0.4s}, [x9]
    fsub    v0.4s, v0.4s, v16.4s
    VEXP    v0, v1, v2, v3
    st1     {v0.4s}, [x9], #16
    fadd    v17.4s, v17.4s, v0.4s
    sub     x10, x10, #4
    b       3b
4:  cbz     x10, 5f
    ldr     s0, [x9]
    fsub    s0, s0, s9
    VEXP    v0, v1, v2, v3
    str     s0, [x9], #4
    fadd    s18, s18, s0
    sub     x10, x10, #1
    b       4b
5:  faddp   v17.4s, v17.4s, v17.4s
    faddp   s17, v17.2s
    fadd    s10, s17, s18               // expsum
    movi    d11, #0                     // inv = sum == 0 ? 0 : 1/sum
    fcmp    s10, #0.0
    b.eq    6f
    fmov    s0, #1.0
    fdiv    s11, s0, s10
6:  // pass 3: normalize
    dup     v1.4s, v11.s[0]
    mov     x9, x21
    add     x10, x27, #1
55: cmp     x10, #4
    b.lt    56f
    ld1     {v0.4s}, [x9]
    fmul    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x9], #16
    sub     x10, x10, #4
    b       55b
56: cbz     x10, 57f
    ldr     s0, [x9]
    fmul    s0, s0, s11
    str     s0, [x9], #4
    sub     x10, x10, #1
    b       56b
57:
    // pass 4: out[b,t,h,:] = sum_t2 att[t2] * v_t2
    add     x9, x26, x27
    mul     x9, x9, x23
    add     x9, x9, x28
    add     x10, x19, x9, lsl #2        // output slice
    mov     x11, x10
    mov     x12, x25
    cbnz    x16, 8f                     // fast path does not need the zeroing
7:  cbz     x12, 8f
    str     wzr, [x11], #4
    sub     x12, x12, #1
    b       7b
8:  add     x13, x23, x23, lsl #1       // C3
    lsl     x14, x13, #2                // row bytes (recomputed, a_dot calls above)
    mul     x9, x26, x13
    add     x9, x9, x23, lsl #1         // + 2C
    add     x9, x9, x28
    add     x13, x20, x9, lsl #2        // value of position 0
    cbz     x16, 79f
    // fast path: the 64-wide output lives in v0-v15 while summing over the values
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v4.16b, #0
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.16b, #0
    movi    v8.16b, #0
    movi    v9.16b, #0
    movi    v10.16b, #0
    movi    v11.16b, #0
    movi    v12.16b, #0
    movi    v13.16b, #0
    movi    v14.16b, #0
    movi    v15.16b, #0
    mov     x12, #0
80: cmp     x12, x27
    b.gt    81f
    add     x9, x21, x12, lsl #2
    ld1r    {v16.4s}, [x9]              // att[t2]
    mov     x9, x13
    ld1     {v17.4s, v18.4s, v19.4s, v20.4s}, [x9], #64
    ld1     {v21.4s, v22.4s, v23.4s, v24.4s}, [x9]
    fmla    v0.4s, v17.4s, v16.4s
    fmla    v1.4s, v18.4s, v16.4s
    fmla    v2.4s, v19.4s, v16.4s
    fmla    v3.4s, v20.4s, v16.4s
    fmla    v4.4s, v21.4s, v16.4s
    fmla    v5.4s, v22.4s, v16.4s
    fmla    v6.4s, v23.4s, v16.4s
    fmla    v7.4s, v24.4s, v16.4s
    add     x9, x13, #128
    ld1     {v17.4s, v18.4s, v19.4s, v20.4s}, [x9], #64
    ld1     {v21.4s, v22.4s, v23.4s, v24.4s}, [x9]
    fmla    v8.4s, v17.4s, v16.4s
    fmla    v9.4s, v18.4s, v16.4s
    fmla    v10.4s, v19.4s, v16.4s
    fmla    v11.4s, v20.4s, v16.4s
    fmla    v12.4s, v21.4s, v16.4s
    fmla    v13.4s, v22.4s, v16.4s
    fmla    v14.4s, v23.4s, v16.4s
    fmla    v15.4s, v24.4s, v16.4s
    add     x13, x13, x14
    add     x12, x12, #1
    b       80b
81:
    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x10]
    add     x9, x10, #64
    st1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x9]
    add     x9, x10, #128
    st1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x9]
    add     x9, x10, #192
    st1     {v12.4s, v13.4s, v14.4s, v15.4s}, [x9]
    b       13f
79: mov     x12, #0                     // t2
9:  cmp     x12, x27
    b.gt    13f
    ldr     s1, [x21, x12, lsl #2]
    dup     v1.4s, v1.s[0]
    mov     x9, x10
    mov     x11, x13
    mov     x15, x25
10: cmp     x15, #4
    b.lt    11f
    ld1     {v0.4s}, [x9]
    ld1     {v2.4s}, [x11], #16
    fmla    v0.4s, v2.4s, v1.4s
    st1     {v0.4s}, [x9], #16
    sub     x15, x15, #4
    b       10b
11: cbz     x15, 12f
    ldr     s0, [x9]
    ldr     s2, [x11], #4
    fmadd   s0, s2, s1, s0
    str     s0, [x9], #4
    sub     x15, x15, #1
    b       11b
12: add     x13, x13, x14
    add     x12, x12, #1
    b       9b
13: add     x28, x28, x25
    b       att_h
att_t_next:
    add     x27, x27, #1
    b       att_t
att_b_next:
    add     x26, x26, x22
    b       att_b
att_done:
    ldp     d8, d9, [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #160]
    ldp     d14, d15, [sp, #176]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #208
    ret

// void a_gelu_forward(float* out, const float* inp, long N)    (tanh GELU)
// out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))), 4 at a time,
// computed as x / (1 + exp(-2 * sqrt(2/pi) * (x + 0.044715 * x^3)))
FUNC SYM(a_gelu_forward)
    VEXP_CONSTS
    FLOADK  s16, w9, F_GELU_CUBE
    dup     v16.4s, v16.s[0]
    FLOADK  s17, w9, F_GELU_M2S
    dup     v17.4s, v17.s[0]
1:  cmp     x2, #4
    b.lt    2f
    ld1     {v4.4s}, [x1], #16
    GELU4
    st1     {v0.4s}, [x0], #16
    sub     x2, x2, #4
    b       1b
2:  cbz     x2, 3f
    ldr     s4, [x1], #4
    GELU4
    str     s0, [x0], #4
    sub     x2, x2, #1
    b       2b
3:  ret

// void a_residual_forward(float* out, const float* a, const float* b, long N)
FUNC SYM(a_residual_forward)
1:  cmp     x3, #16
    b.lt    2f
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x2], #64
    fadd    v0.4s, v0.4s, v4.4s
    fadd    v1.4s, v1.4s, v5.4s
    fadd    v2.4s, v2.4s, v6.4s
    fadd    v3.4s, v3.4s, v7.4s
    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    sub     x3, x3, #16
    b       1b
2:  cmp     x3, #4
    b.lt    3f
    ld1     {v0.4s}, [x1], #16
    ld1     {v1.4s}, [x2], #16
    fadd    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x0], #16
    sub     x3, x3, #4
    b       2b
3:  cbz     x3, 4f
    ldr     s0, [x1], #4
    ldr     s1, [x2], #4
    fadd    s0, s0, s1
    str     s0, [x0], #4
    sub     x3, x3, #1
    b       3b
4:  ret

// void a_softmax_forward(float* probs, const float* logits, long N, long V, long Vp)
// Softmax over the first V entries of each row; padding [V, Vp) is set to 0.
FUNC SYM(a_softmax_forward)
    VEXP_CONSTS
    mov     x8, #0                      // row
sm_row:
    cmp     x8, x2
    b.ge    sm_done
    mul     x9, x8, x4
    add     x10, x1, x9, lsl #2         // logits row
    add     x11, x0, x9, lsl #2         // probs row
    FLOADK  s16, w9, F_NEG10000         // max, starting from -10000 like llm.c
    dup     v16.4s, v16.s[0]
    mov     x12, x10
    mov     x13, x3
1:  cmp     x13, #4
    b.lt    2f
    ld1     {v0.4s}, [x12], #16
    fmax    v16.4s, v16.4s, v0.4s
    sub     x13, x13, #4
    b       1b
2:  fmaxv   s16, v16.4s
3:  cbz     x13, 4f
    ldr     s0, [x12], #4
    fmax    s16, s16, s0
    sub     x13, x13, #1
    b       3b
4:  dup     v17.4s, v16.s[0]
    movi    v18.16b, #0                 // vector sum
    movi    d19, #0                     // tail sum
    mov     x12, x10
    mov     x14, x11
    mov     x13, x3
5:  cmp     x13, #4
    b.lt    6f
    ld1     {v0.4s}, [x12], #16
    fsub    v0.4s, v0.4s, v17.4s
    VEXP    v0, v1, v2, v3
    st1     {v0.4s}, [x14], #16
    fadd    v18.4s, v18.4s, v0.4s
    sub     x13, x13, #4
    b       5b
6:  cbz     x13, 7f
    ldr     s0, [x12], #4
    fsub    s0, s0, s16
    VEXP    v0, v1, v2, v3
    str     s0, [x14], #4
    fadd    s19, s19, s0
    sub     x13, x13, #1
    b       6b
7:  faddp   v18.4s, v18.4s, v18.4s
    faddp   s18, v18.2s
    fadd    s18, s18, s19               // sum
    fmov    s1, #1.0
    fdiv    s18, s1, s18                // 1 / sum
    dup     v1.4s, v18.s[0]
    mov     x12, x11
    mov     x13, x3
8:  cmp     x13, #16
    b.lt    81f
    ld1     {v2.4s, v3.4s, v4.4s, v5.4s}, [x12]
    fmul    v2.4s, v2.4s, v1.4s
    fmul    v3.4s, v3.4s, v1.4s
    fmul    v4.4s, v4.4s, v1.4s
    fmul    v5.4s, v5.4s, v1.4s
    st1     {v2.4s, v3.4s, v4.4s, v5.4s}, [x12], #64
    sub     x13, x13, #16
    b       8b
81: cmp     x13, #4
    b.lt    9f
    ld1     {v0.4s}, [x12]
    fmul    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x12], #16
    sub     x13, x13, #4
    b       81b
9:  cbz     x13, 10f
    ldr     s0, [x12]
    fmul    s0, s0, s18
    str     s0, [x12], #4
    sub     x13, x13, #1
    b       9b
10: sub     x13, x4, x3                 // zero the padding
11: cbz     x13, 12f
    str     wzr, [x12], #4
    sub     x13, x13, #1
    b       11b
12: add     x8, x8, #1
    b       sm_row
sm_done:
    ret

// ===========================================================================
// packed matmul

// void a_pack_weights(float* dst, const float* src, long OC, long C)
// dst[(q*C + k)*16 + j] = src[(q*16 + j)*C + k]   (OC % 16 == 0)
FUNC SYM(a_pack_weights)
    lsl     x9, x3, #2                  // row bytes
    lsl     x10, x9, #4                 // 16 rows
    mov     x11, #0                     // first channel of the panel
1:  cmp     x11, x2
    b.ge    9f
    mov     x12, x1                     // column k of row 0 of this panel
    mov     x13, x3                     // k
2:  mov     x14, x12
    mov     x15, #16
3:  ldr     s0, [x14]
    str     s0, [x0], #4
    add     x14, x14, x9
    subs    x15, x15, #1
    b.gt    3b
    add     x12, x12, #4
    subs    x13, x13, #1
    b.gt    2b
    add     x1, x1, x10
    add     x11, x11, #16
    b       1b
9:  ret

// copy_f32(dst, src, n)
    .p2align 4
copy_f32:
1:  cmp     x2, #16
    b.lt    2f
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64
    sub     x2, x2, #16
    b       1b
2:  cbz     x2, 3f
    ldr     s0, [x1], #4
    str     s0, [x0], #4
    sub     x2, x2, #1
    b       2b
3:  ret

// pack_inplace(float* w, long OC, long C, float* tmp): repack an OC x C matrix in place
    .p2align 4
pack_inplace:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    mov     x19, x0
    mul     x20, x1, x2
    mov     x21, x3
    mov     x9, x1
    mov     x10, x2
    mov     x0, x21
    mov     x1, x19
    mov     x2, x9
    mov     x3, x10
    bl      SYM(a_pack_weights)
    mov     x0, x19
    mov     x1, x21
    mov     x2, x20
    bl      copy_f32
    ldp     x19, x20, [sp, #16]
    ldr     x21, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret

// pack_all: repack every matmul weight (in place) and make a packed copy of wte
    .p2align 4
pack_all:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    LOADADDR x19, G
    ldr     x20, [x19, #G_C]
    ldr     x21, [x19, #G_L]
    mul     x0, x20, x20
    lsl     x0, x0, #4                  // scratch: 4C*C floats
    bl      xmalloc
    mov     x22, x0
    add     x24, x20, x20, lsl #1       // 3C
    lsl     x25, x20, #2                // 4C
    mov     x23, #0                     // layer
1:  cmp     x23, x21
    b.ge    2f
    mul     x9, x23, x24
    mul     x9, x9, x20
    ldr     x0, [x19, #P_QKVW]
    add     x0, x0, x9, lsl #2
    mov     x1, x24
    mov     x2, x20
    mov     x3, x22
    bl      pack_inplace
    mul     x9, x23, x20
    mul     x9, x9, x20
    ldr     x0, [x19, #P_ATTPROJW]
    add     x0, x0, x9, lsl #2
    mov     x1, x20
    mov     x2, x20
    mov     x3, x22
    bl      pack_inplace
    mul     x9, x23, x25
    mul     x9, x9, x20
    ldr     x0, [x19, #P_FCW]
    add     x0, x0, x9, lsl #2
    mov     x1, x25
    mov     x2, x20
    mov     x3, x22
    bl      pack_inplace
    mul     x9, x23, x25
    mul     x9, x9, x20
    ldr     x0, [x19, #P_FCPROJW]
    add     x0, x0, x9, lsl #2
    mov     x1, x20
    mov     x2, x25
    mov     x3, x22
    bl      pack_inplace
    add     x23, x23, #1
    b       1b
2:  mov     x0, x22
    bl      SYM(free)
    ldr     x9, [x19, #G_VP]
    mul     x0, x9, x20
    lsl     x0, x0, #2
    bl      xmalloc
    str     x0, [x19, #G_WTE_PACKED]
    ldr     x1, [x19, #P_WTE]
    ldr     x2, [x19, #G_VP]
    mov     x3, x20
    bl      SYM(a_pack_weights)
    mov     x9, #1
    str     x9, [x19, #G_PACKED]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// void a_matmul_packed(float* out, const float* inp, const float* wp,
//                      const float* bias, long N, long C, long OC, long ldo)
// wp: OC x C weights repacked by a_pack_weights. Needs C % 4 == 0, OC % 16 == 0.
// Tiles of 5 rows x 16 channels. A last partial row block reuses the last row
// for the missing rows (so the weights are still read once) and stores only real rows.
// The first row block of each panel prefetches the next panel into L2.
FUNC SYM(a_matmul_packed)
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    SAVE_X19_X28
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]
    mov     x19, x0                     // out
    mov     x20, x1                     // inp
    mov     x21, x2                     // packed weights
    mov     x22, x3                     // bias
    mov     x23, x4                     // N
    mov     x24, x5                     // C
    mov     x25, x6                     // OC
    lsl     x26, x7, #2                 // output row bytes
    lsl     x12, x24, #2                // input row bytes
    lsl     x13, x24, #6                // panel bytes (C x 16 floats)
    add     x14, x13, #64
    add     x17, x13, #128
    add     x11, x13, #192
    sub     x16, x23, #1                // last row
    mov     x15, x21                    // current panel
    mov     x28, #0                     // o
pk_panel:
    cmp     x28, x25
    b.ge    pk_done
    mov     x27, #0                     // r
pk_rblk:
    cmp     x27, x23
    b.ge    pk_next
    // row pointers r .. r+4, clamped to the last row
    madd    x0, x27, x12, x20
    add     x9, x27, #1
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x1, x9, x12, x20
    add     x9, x27, #2
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x2, x9, x12, x20
    add     x9, x27, #3
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x3, x9, x12, x20
    add     x9, x27, #4
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x4, x9, x12, x20
    mov     x8, x15
    movi    v12.16b, #0
    movi    v13.16b, #0
    movi    v14.16b, #0
    movi    v15.16b, #0
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    movi    v24.16b, #0
    movi    v25.16b, #0
    movi    v26.16b, #0
    movi    v27.16b, #0
    movi    v28.16b, #0
    movi    v29.16b, #0
    movi    v30.16b, #0
    movi    v31.16b, #0
    mov     x9, x24
    cbnz    x27, 2f
1:  prfm    pldl2keep, [x8, x13]        // first row block: fetch the next panel
    prfm    pldl2keep, [x8, x14]
    prfm    pldl2keep, [x8, x17]
    prfm    pldl2keep, [x8, x11]
    KSTEP5
    subs    x9, x9, #4
    b.gt    1b
    b       3f
2:  KSTEP5
    subs    x9, x9, #4
    b.gt    2b
3:  cbz     x22, 4f
    add     x9, x22, x28, lsl #2
    ld1     {v5.4s, v6.4s, v7.4s, v8.4s}, [x9]
    fadd    v12.4s, v12.4s, v5.4s
    fadd    v13.4s, v13.4s, v6.4s
    fadd    v14.4s, v14.4s, v7.4s
    fadd    v15.4s, v15.4s, v8.4s
    fadd    v16.4s, v16.4s, v5.4s
    fadd    v17.4s, v17.4s, v6.4s
    fadd    v18.4s, v18.4s, v7.4s
    fadd    v19.4s, v19.4s, v8.4s
    fadd    v20.4s, v20.4s, v5.4s
    fadd    v21.4s, v21.4s, v6.4s
    fadd    v22.4s, v22.4s, v7.4s
    fadd    v23.4s, v23.4s, v8.4s
    fadd    v24.4s, v24.4s, v5.4s
    fadd    v25.4s, v25.4s, v6.4s
    fadd    v26.4s, v26.4s, v7.4s
    fadd    v27.4s, v27.4s, v8.4s
    fadd    v28.4s, v28.4s, v5.4s
    fadd    v29.4s, v29.4s, v6.4s
    fadd    v30.4s, v30.4s, v7.4s
    fadd    v31.4s, v31.4s, v8.4s
4:  sub     x10, x23, x27               // rows left (>= 1)
    mul     x9, x27, x26
    add     x9, x9, x28, lsl #2
    add     x9, x19, x9
    st1     {v12.4s, v13.4s, v14.4s, v15.4s}, [x9]
    cmp     x10, #1
    b.le    5f
    add     x9, x9, x26
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x9]
    cmp     x10, #2
    b.le    5f
    add     x9, x9, x26
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x9]
    cmp     x10, #3
    b.le    5f
    add     x9, x9, x26
    st1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x9]
    cmp     x10, #4
    b.le    5f
    add     x9, x9, x26
    st1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x9]
5:  add     x27, x27, #5
    b       pk_rblk
pk_next:
    add     x15, x15, x13
    add     x28, x28, #16
    b       pk_panel
pk_done:
    ldp     d8, d9, [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #160
    ret

// ===========================================================================
// fp16 matmul

// void a_pack_f16(uint16_t* dst, const float* src, long OC, long C)
// dst[(q*C + k)*32 + j] = fp16(src[(q*32 + j)*C + k])   (OC % 32 == 0)
FUNC SYM(a_pack_f16)
    lsl     x9, x3, #2                  // row bytes
    lsl     x10, x9, #5                 // 32 rows
    mov     x11, #0
1:  cmp     x11, x2
    b.ge    9f
    mov     x12, x1
    mov     x13, x3
2:  mov     x14, x12
    mov     x15, #32
3:  ldr     s0, [x14]
    fcvt    h0, s0
    str     h0, [x0], #2
    add     x14, x14, x9
    subs    x15, x15, #1
    b.gt    3b
    add     x12, x12, #4
    subs    x13, x13, #1
    b.gt    2b
    add     x1, x1, x10
    add     x11, x11, #32
    b       1b
9:  ret

// void a_f32_to_f16(uint16_t* dst, const float* src, long n)
FUNC SYM(a_f32_to_f16)
1:  cmp     x2, #16
    b.lt    2f
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x1], #64
    fcvtn   v4.4h, v0.4s
    fcvtn2  v4.8h, v1.4s
    fcvtn   v5.4h, v2.4s
    fcvtn2  v5.8h, v3.4s
    st1     {v4.8h, v5.8h}, [x0], #32
    sub     x2, x2, #16
    b       1b
2:  cbz     x2, 3f
    ldr     s0, [x1], #4
    fcvt    h0, s0
    str     h0, [x0], #2
    sub     x2, x2, #1
    b       2b
3:  ret

// void a_matmul_f16(float* out, const uint16_t* inp, const uint16_t* wp,
//                   const float* bias, long N, long C, long OC, long ldo)
// inp: N x C fp16. wp: packed by a_pack_f16. Needs C % 8 == 0, OC % 32 == 0.
// Tiles of 5 rows x 32 channels; fp16 sums over blocks of KC16 k are folded
// into the fp32 output, which starts as the bias.
FUNC SYM(a_matmul_f16)
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    SAVE_X19_X28
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]
    mov     x19, x0                     // out
    mov     x20, x1                     // inp (fp16)
    mov     x21, x2                     // packed fp16 weights
    mov     x22, x3                     // bias
    mov     x23, x4                     // N
    mov     x24, x5                     // C
    mov     x25, x6                     // OC
    lsl     x26, x7, #2                 // output row bytes
    lsl     x12, x24, #1                // input row bytes
    lsl     x13, x24, #6                // panel bytes (C x 32 halves)
    sub     x16, x23, #1                // last row
    mov     x15, x21                    // current panel
    mov     x28, #0                     // o
h_panel:
    cmp     x28, x25
    b.ge    h_done
    mov     x27, #0                     // r
h_rblk:
    cmp     x27, x23
    b.ge    h_next
    madd    x0, x27, x12, x20
    add     x9, x27, #1
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x1, x9, x12, x20
    add     x9, x27, #2
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x2, x9, x12, x20
    add     x9, x27, #3
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x3, x9, x12, x20
    add     x9, x27, #4
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x4, x9, x12, x20
    mov     x8, x15
    sub     x5, x23, x27                // rows left (>= 1)
    mul     x7, x27, x26
    add     x7, x7, x28, lsl #2
    add     x7, x19, x7                 // output tile
    // output rows start as the bias (or zero)
    movi    v0.16b, #0
    movi    v1.16b, #0
    movi    v2.16b, #0
    movi    v3.16b, #0
    movi    v4.16b, #0
    movi    v5.16b, #0
    movi    v6.16b, #0
    movi    v7.16b, #0
    cbz     x22, 1f
    add     x9, x22, x28, lsl #2
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x9], #64
    ld1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x9]
1:  mov     x9, x7
    mov     x10, #5                     // rows in this block: min(5, rows left)
    cmp     x5, x10
    csel    x10, x5, x10, lt
2:  st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x9]
    add     x11, x9, #64
    st1     {v4.4s, v5.4s, v6.4s, v7.4s}, [x11]
    add     x9, x9, x26
    subs    x10, x10, #1
    b.gt    2b
3:  mov     x6, x24                     // k left
h_chunk:
    movi    v12.16b, #0
    movi    v13.16b, #0
    movi    v14.16b, #0
    movi    v15.16b, #0
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    movi    v24.16b, #0
    movi    v25.16b, #0
    movi    v26.16b, #0
    movi    v27.16b, #0
    movi    v28.16b, #0
    movi    v29.16b, #0
    movi    v30.16b, #0
    movi    v31.16b, #0
    mov     x9, #KC16
    cmp     x6, x9
    csel    x9, x6, x9, lt              // k in this block
    sub     x6, x6, x9
    cbnz    x27, 5f
4:  add     x11, x8, x13                // first row block: fetch the next panel
    prfm    pldl2keep, [x11]
    prfm    pldl2keep, [x11, #64]
    prfm    pldl2keep, [x11, #128]
    prfm    pldl2keep, [x11, #192]
    prfm    pldl2keep, [x11, #256]
    prfm    pldl2keep, [x11, #320]
    prfm    pldl2keep, [x11, #384]
    prfm    pldl2keep, [x11, #448]
    KSTEP16
    subs    x9, x9, #8
    b.gt    4b
    b       6f
5:  KSTEP16
    subs    x9, x9, #8
    b.gt    5b
6:  // fold the fp16 block sums into the fp32 output rows
    mov     x9, x7
    FOLD32  v12, v13, v14, v15
    cmp     x5, #1
    b.le    7f
    add     x9, x7, x26
    FOLD32  v16, v17, v18, v19
    cmp     x5, #2
    b.le    7f
    add     x9, x7, x26, lsl #1
    FOLD32  v20, v21, v22, v23
    cmp     x5, #3
    b.le    7f
    add     x9, x7, x26, lsl #1
    add     x9, x9, x26
    FOLD32  v24, v25, v26, v27
    cmp     x5, #4
    b.le    7f
    add     x9, x7, x26, lsl #2
    FOLD32  v28, v29, v30, v31
7:  cbnz    x6, h_chunk
    add     x27, x27, #5
    b       h_rblk
h_next:
    add     x15, x15, x13
    add     x28, x28, #32
    b       h_panel
h_done:
    ldp     d8, d9, [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #160
    ret

// pack16_inplace(float* w, long OC, long C, void* tmp): fp16 panels written over w
    .p2align 4
pack16_inplace:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    mov     x19, x0
    mul     x20, x1, x2
    mov     x21, x3
    mov     x9, x1
    mov     x10, x2
    mov     x0, x21
    mov     x1, x19
    mov     x2, x9
    mov     x3, x10
    bl      SYM(a_pack_f16)
    mov     x0, x19                     // copy n/2 floats' worth of bytes (n halves)
    mov     x1, x21
    lsr     x2, x20, #1
    bl      copy_f32
    ldp     x19, x20, [sp, #16]
    ldr     x21, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret

// pack16_all: fp16-pack every matmul weight in place, fp16 copy of wte, fp16 input buffer
    .p2align 4
pack16_all:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    LOADADDR x19, G
    ldr     x20, [x19, #G_C]
    ldr     x21, [x19, #G_L]
    mul     x0, x20, x20
    lsl     x0, x0, #3                  // scratch: 4C*C halves
    bl      xmalloc
    mov     x22, x0
    add     x24, x20, x20, lsl #1       // 3C
    lsl     x25, x20, #2                // 4C
    mov     x23, #0
1:  cmp     x23, x21
    b.ge    2f
    mul     x9, x23, x24
    mul     x9, x9, x20
    ldr     x0, [x19, #P_QKVW]
    add     x0, x0, x9, lsl #2
    mov     x1, x24
    mov     x2, x20
    mov     x3, x22
    bl      pack16_inplace
    mul     x9, x23, x20
    mul     x9, x9, x20
    ldr     x0, [x19, #P_ATTPROJW]
    add     x0, x0, x9, lsl #2
    mov     x1, x20
    mov     x2, x20
    mov     x3, x22
    bl      pack16_inplace
    mul     x9, x23, x25
    mul     x9, x9, x20
    ldr     x0, [x19, #P_FCW]
    add     x0, x0, x9, lsl #2
    mov     x1, x25
    mov     x2, x20
    mov     x3, x22
    bl      pack16_inplace
    mul     x9, x23, x25
    mul     x9, x9, x20
    ldr     x0, [x19, #P_FCPROJW]
    add     x0, x0, x9, lsl #2
    mov     x1, x20
    mov     x2, x25
    mov     x3, x22
    bl      pack16_inplace
    add     x23, x23, #1
    b       1b
2:  mov     x0, x22
    bl      SYM(free)
    ldr     x9, [x19, #G_VP]
    mul     x0, x9, x20
    lsl     x0, x0, #1
    bl      xmalloc
    str     x0, [x19, #G_WTE_PACKED]
    ldr     x1, [x19, #P_WTE]
    ldr     x2, [x19, #G_VP]
    mov     x3, x20
    bl      SYM(a_pack_f16)
    mov     x9, #2
    str     x9, [x19, #G_PACKED]
    LOADADDR x0, s_prec16
    bl      pr0
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// ===========================================================================
// int8 matmul

// void a_qpack_w(int8_t* dst, const float* src, long OC, long C, const float* colscale)
// Per panel of 16 channels: 16 float scales (max|row| / 127), then for each group
// of 4 k, 16 channels x 4 int8. W[o][k] is multiplied by colscale[k] first (if given).
// Needs OC % 16 == 0, C % 4 == 0.
FUNC SYM(a_qpack_w)
    lsl     x9, x3, #2                  // row bytes
    mov     x10, #0                     // first channel of the panel
    MOV32   w15, 0x42FE0000             // 127.0f
    fmov    s7, w15
1:  cmp     x10, x2
    b.ge    9f
    mul     x11, x10, x9
    add     x11, x1, x11                // row 0 of this panel
    mov     x12, #0
2:  madd    x13, x12, x9, x11           // scales: max |W[o][k] * colscale[k]| / 127
    movi    d0, #0
    mov     x14, #0
3:  ldr     s1, [x13, x14, lsl #2]
    cbz     x4, 31f
    ldr     s5, [x4, x14, lsl #2]
    fmul    s1, s1, s5
31: fabs    s1, s1
    fmax    s0, s0, s1
    add     x14, x14, #1
    cmp     x14, x3
    b.lt    3b
    fdiv    s2, s0, s7
    fcmp    s0, #0.0
    b.ne    4f
    fmov    s2, #1.0
4:  str     s2, [x0], #4
    add     x12, x12, #1
    cmp     x12, #16
    b.lt    2b
    sub     x16, x0, #64                // the 16 scales just written
    mov     x12, #0                     // k group start
5:  cmp     x12, x3
    b.ge    8f
    mov     x13, #0                     // channel j
6:  madd    x14, x13, x9, x11
    ldr     s3, [x16, x13, lsl #2]
    mov     x15, #0
7:  add     x5, x12, x15
    ldr     s1, [x14, x5, lsl #2]
    cbz     x4, 71f
    ldr     s5, [x4, x5, lsl #2]
    fmul    s1, s1, s5
71: fdiv    s1, s1, s3
    fcvtns  w17, s1
    mov     w6, #127
    cmp     w17, w6
    csel    w17, w6, w17, gt
    mov     w6, #-127
    cmp     w17, w6
    csel    w17, w6, w17, lt
    strb    w17, [x0], #1
    add     x15, x15, #1
    cmp     x15, #4
    b.lt    7b
    add     x13, x13, #1
    cmp     x13, #16
    b.lt    6b
    add     x12, x12, #4
    b       5b
8:  add     x10, x10, #16
    b       1b
9:  ret

// void a_quant_rows(int8_t* q, float* scales, const float* x, long N, long C,
//                   const float* chanmul)
// Per row: y = x * chanmul (if given), scale = max|y| / 127, q = round(y / scale).
// Needs C % 16 == 0.
FUNC SYM(a_quant_rows)
    MOV32   w9, 0x42FE0000              // 127.0f
    fmov    s7, w9
1:  cbz     x3, 9f
    movi    v16.16b, #0
    mov     x10, x2
    mov     x11, x4
    mov     x12, x5
2:  ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x10], #64
    cbz     x5, 21f
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x12], #64
    fmul    v0.4s, v0.4s, v20.4s
    fmul    v1.4s, v1.4s, v21.4s
    fmul    v2.4s, v2.4s, v22.4s
    fmul    v3.4s, v3.4s, v23.4s
21: fabs    v0.4s, v0.4s
    fabs    v1.4s, v1.4s
    fabs    v2.4s, v2.4s
    fabs    v3.4s, v3.4s
    fmax    v0.4s, v0.4s, v1.4s
    fmax    v2.4s, v2.4s, v3.4s
    fmax    v0.4s, v0.4s, v2.4s
    fmax    v16.4s, v16.4s, v0.4s
    subs    x11, x11, #16
    b.gt    2b
    fmaxv   s16, v16.4s
    fdiv    s17, s16, s7                // scale
    fdiv    s18, s7, s16                // 1 / scale
    fcmp    s16, #0.0
    b.ne    3f
    fmov    s17, #1.0
    movi    d18, #0
3:  str     s17, [x1], #4
    dup     v18.4s, v18.s[0]
    mov     x11, x4
    mov     x12, x5
4:  ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x2], #64
    cbz     x5, 41f
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x12], #64
    fmul    v0.4s, v0.4s, v20.4s
    fmul    v1.4s, v1.4s, v21.4s
    fmul    v2.4s, v2.4s, v22.4s
    fmul    v3.4s, v3.4s, v23.4s
41: fmul    v0.4s, v0.4s, v18.4s
    fmul    v1.4s, v1.4s, v18.4s
    fmul    v2.4s, v2.4s, v18.4s
    fmul    v3.4s, v3.4s, v18.4s
    fcvtns  v0.4s, v0.4s
    fcvtns  v1.4s, v1.4s
    fcvtns  v2.4s, v2.4s
    fcvtns  v3.4s, v3.4s
    sqxtn   v4.4h, v0.4s
    sqxtn2  v4.8h, v1.4s
    sqxtn   v5.4h, v2.4s
    sqxtn2  v5.8h, v3.4s
    sqxtn   v6.8b, v4.8h
    sqxtn2  v6.16b, v5.8h
    st1     {v6.16b}, [x0], #16
    subs    x11, x11, #16
    b.gt    4b
    sub     x3, x3, #1
    b       1b
9:  ret

// void a_colmax(float* m, const float* x, long N, long C): m[k] = max(m[k], max_r |x[r][k]|)
FUNC SYM(a_colmax)
1:  cbz     x2, 9f
    mov     x9, x0
    mov     x10, x3
2:  cmp     x10, #4
    b.lt    3f
    ld1     {v0.4s}, [x1], #16
    ld1     {v1.4s}, [x9]
    fabs    v0.4s, v0.4s
    fmax    v1.4s, v1.4s, v0.4s
    st1     {v1.4s}, [x9], #16
    sub     x10, x10, #4
    b       2b
3:  cbz     x10, 4f
    ldr     s0, [x1], #4
    ldr     s1, [x9]
    fabs    s0, s0
    fmax    s1, s1, s0
    str     s1, [x9], #4
    sub     x10, x10, #1
    b       3b
4:  sub     x2, x2, #1
    b       1b
9:  ret

// void a_matmul_q8(float* out, const int8_t* xq, const int8_t* wq,
//                  const float* bias, long N, long C, long OC, long ldo)
// xq: N x C int8, row scales in G_ACT_XS. wq: packed by a_qpack_w.
// Tiles of 5 rows x 16 channels with int32 sdot sums. Needs C % 16 == 0, OC % 16 == 0.
FUNC SYM(a_matmul_q8)
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    SAVE_X19_X28
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    stp     d12, d13, [sp, #128]
    stp     d14, d15, [sp, #144]
    mov     x19, x0                     // out
    mov     x20, x1                     // xq
    mov     x21, x2                     // packed int8 weights
    mov     x22, x3                     // bias
    mov     x23, x4                     // N
    mov     x24, x5                     // C
    mov     x25, x6                     // OC
    lsl     x26, x7, #2                 // output row bytes
    mov     x12, x24                    // input row bytes
    lsl     x13, x24, #4
    add     x13, x13, #64               // panel bytes: 16 scales + 16 C int8
    LOADADDR x14, G
    ldr     x14, [x14, #G_ACT_XS]       // row scales
    sub     x16, x23, #1                // last row
    mov     x15, x21                    // current panel
    mov     x28, #0                     // o
q_panel:
    cmp     x28, x25
    b.ge    q_done
    mov     x27, #0                     // r
q_rblk:
    cmp     x27, x23
    b.ge    q_next
    madd    x0, x27, x12, x20
    add     x9, x27, #1
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x1, x9, x12, x20
    add     x9, x27, #2
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x2, x9, x12, x20
    add     x9, x27, #3
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x3, x9, x12, x20
    add     x9, x27, #4
    cmp     x9, x16
    csel    x9, x16, x9, gt
    madd    x4, x9, x12, x20
    add     x8, x15, #64                // past the 16 scales
    movi    v12.16b, #0
    movi    v13.16b, #0
    movi    v14.16b, #0
    movi    v15.16b, #0
    movi    v16.16b, #0
    movi    v17.16b, #0
    movi    v18.16b, #0
    movi    v19.16b, #0
    movi    v20.16b, #0
    movi    v21.16b, #0
    movi    v22.16b, #0
    movi    v23.16b, #0
    movi    v24.16b, #0
    movi    v25.16b, #0
    movi    v26.16b, #0
    movi    v27.16b, #0
    movi    v28.16b, #0
    movi    v29.16b, #0
    movi    v30.16b, #0
    movi    v31.16b, #0
    mov     x9, x24
    cbnz    x27, 2f
1:  add     x11, x8, x13                // first row block: fetch the next panel
    prfm    pldl2keep, [x11]
    prfm    pldl2keep, [x11, #64]
    prfm    pldl2keep, [x11, #128]
    prfm    pldl2keep, [x11, #192]
    KSTEPQ
    subs    x9, x9, #16
    b.gt    1b
    b       3f
2:  KSTEPQ
    subs    x9, x9, #16
    b.gt    2b
3:  // scales: 16 channel scales, 5 row scales (xs has padding past N)
    ld1     {v8.4s, v9.4s, v10.4s, v11.4s}, [x15]
    add     x9, x14, x27, lsl #2
    ld1     {v5.4s}, [x9], #16
    ld1     {v6.s}[0], [x9]
    DEQ     v12, v13, v14, v15, v5.s[0]
    DEQ     v16, v17, v18, v19, v5.s[1]
    DEQ     v20, v21, v22, v23, v5.s[2]
    DEQ     v24, v25, v26, v27, v5.s[3]
    DEQ     v28, v29, v30, v31, v6.s[0]
    cbz     x22, 4f
    add     x9, x22, x28, lsl #2
    ld1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x9]
    BIAS16  v12, v13, v14, v15
    BIAS16  v16, v17, v18, v19
    BIAS16  v20, v21, v22, v23
    BIAS16  v24, v25, v26, v27
    BIAS16  v28, v29, v30, v31
4:  sub     x10, x23, x27               // rows left (>= 1)
    mul     x9, x27, x26
    add     x9, x9, x28, lsl #2
    add     x9, x19, x9
    st1     {v12.4s, v13.4s, v14.4s, v15.4s}, [x9]
    cmp     x10, #1
    b.le    5f
    add     x9, x9, x26
    st1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x9]
    cmp     x10, #2
    b.le    5f
    add     x9, x9, x26
    st1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x9]
    cmp     x10, #3
    b.le    5f
    add     x9, x9, x26
    st1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x9]
    cmp     x10, #4
    b.le    5f
    add     x9, x9, x26
    st1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x9]
5:  add     x27, x27, #5
    b       q_rblk
q_next:
    add     x15, x15, x13
    add     x28, x28, #16
    b       q_panel
q_done:
    ldp     d8, d9, [sp, #96]
    ldp     d10, d11, [sp, #112]
    ldp     d12, d13, [sp, #128]
    ldp     d14, d15, [sp, #144]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #160
    ret

// q8_inplace(float* w, long OC, long C, void* tmp): int8 panels written over w
    .p2align 4
q8_inplace:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    str     x21, [sp, #32]
    mov     x19, x0
    mul     x20, x1, x2
    add     x20, x20, x1, lsl #2        // packed bytes: OC*C + 4*OC
    mov     x21, x3
    mov     x9, x1
    mov     x10, x2
    mov     x0, x21
    mov     x1, x19
    mov     x2, x9
    mov     x3, x10
    bl      SYM(a_qpack_w)
    mov     x0, x19
    mov     x1, x21
    lsr     x2, x20, #2                 // as floats
    bl      copy_f32
    ldp     x19, x20, [sp, #16]
    ldr     x21, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret

// q8_all: int8-pack every matmul weight in place, int8 copy of wte
    .p2align 4
q8_all:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    LOADADDR x19, G
    ldr     x20, [x19, #G_C]
    ldr     x21, [x19, #G_L]
    mul     x0, x20, x20
    lsl     x0, x0, #3                  // scratch, more than 4C*C + 4*4C bytes
    bl      xmalloc
    mov     x22, x0
    add     x24, x20, x20, lsl #1       // 3C
    lsl     x25, x20, #2                // 4C
    mov     x23, #0
    ldr     x26, [x19, #G_SCALES]       // column scales (0 if not calibrated)
    add     x27, x20, x20, lsl #1
    add     x27, x27, x25               // 7C floats per layer
1:  cmp     x23, x21
    b.ge    2f
    mul     x28, x23, x27               // this layer's scales: [C | C | C | 4C]
    mul     x9, x23, x24
    mul     x9, x9, x20
    ldr     x0, [x19, #P_QKVW]
    add     x0, x0, x9, lsl #2
    mov     x1, x24
    mov     x2, x20
    mov     x3, x22
    mov     x4, #0
    cbz     x26, 11f
    add     x4, x26, x28, lsl #2
11: bl      q8_inplace
    mul     x9, x23, x20
    mul     x9, x9, x20
    ldr     x0, [x19, #P_ATTPROJW]
    add     x0, x0, x9, lsl #2
    mov     x1, x20
    mov     x2, x20
    mov     x3, x22
    mov     x4, #0
    cbz     x26, 12f
    add     x9, x28, x20
    add     x4, x26, x9, lsl #2
12: bl      q8_inplace
    mul     x9, x23, x25
    mul     x9, x9, x20
    ldr     x0, [x19, #P_FCW]
    add     x0, x0, x9, lsl #2
    mov     x1, x25
    mov     x2, x20
    mov     x3, x22
    mov     x4, #0
    cbz     x26, 13f
    add     x9, x28, x20, lsl #1
    add     x4, x26, x9, lsl #2
13: bl      q8_inplace
    mul     x9, x23, x25
    mul     x9, x9, x20
    ldr     x0, [x19, #P_FCPROJW]
    add     x0, x0, x9, lsl #2
    mov     x1, x20
    mov     x2, x25
    mov     x3, x22
    mov     x4, #0
    cbz     x26, 14f
    add     x9, x28, x20, lsl #1
    add     x9, x9, x20
    add     x4, x26, x9, lsl #2
14: bl      q8_inplace
    add     x23, x23, #1
    b       1b
2:  mov     x0, x22
    bl      SYM(free)
    ldr     x9, [x19, #G_VP]
    mul     x0, x9, x20
    add     x0, x0, x9, lsl #2
    bl      xmalloc
    str     x0, [x19, #G_WTE_PACKED]
    ldr     x1, [x19, #P_WTE]
    ldr     x2, [x19, #G_VP]
    mov     x3, x20
    mov     x4, #0
    cbz     x26, 21f
    mul     x9, x21, x27
    add     x4, x26, x9, lsl #2         // final layernorm -> logits scales
21: bl      SYM(a_qpack_w)
    mov     x9, #3
    str     x9, [x19, #G_PACKED]
    LOADADDR x0, s_prec8
    bl      pr0
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// smooth_one(stats, scales, W, OC, K): for each input channel k of W (OC x K):
// s_k = sqrt(max|x_k| / max_o |W[o][k]|) (1 if either is 0), clamped to [1e-3, 1e3].
// scales <- s (for the weights), stats <- 1/s (for the activations).
    .p2align 4
smooth_one:
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    str     x23, [sp, #48]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x4
    mov     x23, x2
    mov     x22, x3
    mov     x9, #0
1:  cmp     x9, x21
    b.ge    2f
    str     wzr, [x20, x9, lsl #2]
    add     x9, x9, #1
    b       1b
2:  mov     x0, x20                     // scales <- column max of |W|
    mov     x1, x23
    mov     x2, x22
    mov     x3, x21
    bl      SYM(a_colmax)
    MOV32   w9, 0x3A83126F              // 1e-3
    fmov    s4, w9
    MOV32   w9, 0x447A0000              // 1e3
    fmov    s5, w9
    fmov    s6, #1.0
    mov     x9, #0
3:  cmp     x9, x21
    b.ge    6f
    ldr     s0, [x19, x9, lsl #2]
    ldr     s1, [x20, x9, lsl #2]
    fmov    s2, #1.0
    fcmp    s0, #0.0
    b.eq    4f
    fcmp    s1, #0.0
    b.eq    4f
    LOADADDR x10, G
    ldr     x10, [x10, #G_ALPHA]
    cmp     x10, #75
    b.eq    41f
    cmp     x10, #50
    b.ne    42f
    fdiv    s2, s0, s1                  // alpha 0.5: sqrt(ax / aw)
    fsqrt   s2, s2
    b       43f
41: fmul    s2, s0, s0                  // alpha 0.75: (ax^3 / aw)^(1/4)
    fmul    s2, s2, s0
    fdiv    s2, s2, s1
    fsqrt   s2, s2
    fsqrt   s2, s2
    b       43f
42: fmul    s3, s1, s1                  // alpha 0.25 (default): (ax / aw^3)^(1/4)
    fmul    s3, s3, s1
    fdiv    s2, s0, s3
    fsqrt   s2, s2
    fsqrt   s2, s2
43: fmax    s2, s2, s4
    fmin    s2, s2, s5
4:  str     s2, [x20, x9, lsl #2]
    fdiv    s3, s6, s2
    str     s3, [x19, x9, lsl #2]
    add     x9, x9, #1
    b       3b
6:  ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldr     x23, [sp, #48]
    ldp     x29, x30, [sp], #64
    ret

// calibrate(path): fp32 forward passes over up to 16 windows of 64 tokens from an
// llm.c token file, recording matmul input maxima, then smoothing factors.
    .p2align 4
calibrate:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    mov     x19, x0
    LOADADDR x20, G
    LOADADDR x1, s_rb
    bl      SYM(fopen)
    cbnz    x0, 1f
    LOADADDR x0, s_err_open
    mov     x1, x19
    b       die
1:  mov     x21, x0
    add     x0, x20, #G_HEADER
    mov     x1, #4
    mov     x2, #256
    mov     x3, x21
    bl      xfread
    ldr     w9, [x20, #G_HEADER]
    MOV32   w10, 20240520
    cmp     w9, w10
    b.eq    2f
    LOADADDR x0, s_err_data
    b       die
2:  ldrsw   x22, [x20, #(G_HEADER + 8)]
    mov     x9, #1024
    cmp     x22, x9
    csel    x22, x9, x22, gt
    lsr     x22, x22, #6
    lsl     x22, x22, #6
    cbnz    x22, 3f
    LOADADDR x0, s_err_data
    b       die
3:  LOADADDR x0, s_env_alpha
    bl      SYM(getenv)
    mov     x9, #25
    cbz     x0, 31f
    bl      SYM(atoi)
    sxtw    x9, w0
31: str     x9, [x20, #G_ALPHA]
    LOADADDR x0, s_calib
    mov     x1, x22
    bl      pr1
    lsl     x0, x22, #1
    bl      xmalloc
    mov     x23, x0
    mov     x1, #2
    mov     x2, x22
    mov     x3, x21
    bl      xfread
    mov     x0, x21
    bl      SYM(fclose)
    lsl     x0, x22, #2
    bl      xmalloc
    mov     x24, x0
    mov     x9, #0
4:  ldrh    w10, [x23, x9, lsl #1]
    str     w10, [x24, x9, lsl #2]
    add     x9, x9, #1
    cmp     x9, x22
    b.lt    4b
    ldr     x25, [x20, #G_C]
    ldr     x26, [x20, #G_L]
    add     x27, x25, x25, lsl #1
    add     x27, x27, x25, lsl #2       // 7C
    mul     x28, x26, x27
    add     x28, x28, x25               // stats floats
    lsl     x0, x28, #2
    bl      xmalloc
    str     x0, [x20, #G_STATS]
    mov     x9, #0
5:  str     wzr, [x0, x9, lsl #2]
    add     x9, x9, #1
    cmp     x9, x28
    b.lt    5b
    lsl     x0, x28, #2
    bl      xmalloc
    str     x0, [x20, #G_SCALES]
    mov     x0, #1
    mov     x1, #64
    bl      SYM(gpt2a_alloc_acts)
    mov     x9, #1
    str     x9, [x20, #G_CALIB]
    mov     x21, #0
6:  cmp     x21, x22
    b.ge    7f
    add     x0, x24, x21, lsl #2
    mov     x1, #1
    mov     x2, #64
    bl      SYM(gpt2a_forward)
    add     x21, x21, #64
    b       6b
7:  str     xzr, [x20, #G_CALIB]
    mov     x21, #0                     // layer
8:  cmp     x21, x26
    b.ge    9f
    mul     x22, x21, x27
    ldr     x0, [x20, #G_STATS]         // qkv: 3C x C
    add     x0, x0, x22, lsl #2
    ldr     x1, [x20, #G_SCALES]
    add     x1, x1, x22, lsl #2
    add     x9, x25, x25, lsl #1
    mul     x10, x21, x9
    mul     x10, x10, x25
    ldr     x2, [x20, #P_QKVW]
    add     x2, x2, x10, lsl #2
    mov     x3, x9
    mov     x4, x25
    bl      smooth_one
    add     x9, x22, x25                // attproj: C x C
    ldr     x0, [x20, #G_STATS]
    add     x0, x0, x9, lsl #2
    ldr     x1, [x20, #G_SCALES]
    add     x1, x1, x9, lsl #2
    mul     x10, x21, x25
    mul     x10, x10, x25
    ldr     x2, [x20, #P_ATTPROJW]
    add     x2, x2, x10, lsl #2
    mov     x3, x25
    mov     x4, x25
    bl      smooth_one
    add     x9, x22, x25, lsl #1        // fc: 4C x C
    ldr     x0, [x20, #G_STATS]
    add     x0, x0, x9, lsl #2
    ldr     x1, [x20, #G_SCALES]
    add     x1, x1, x9, lsl #2
    lsl     x11, x25, #2
    mul     x10, x21, x11
    mul     x10, x10, x25
    ldr     x2, [x20, #P_FCW]
    add     x2, x2, x10, lsl #2
    mov     x3, x11
    mov     x4, x25
    bl      smooth_one
    add     x9, x22, x25, lsl #1        // fcproj: C x 4C
    add     x9, x9, x25
    ldr     x0, [x20, #G_STATS]
    add     x0, x0, x9, lsl #2
    ldr     x1, [x20, #G_SCALES]
    add     x1, x1, x9, lsl #2
    lsl     x11, x25, #2
    mul     x10, x21, x11
    mul     x10, x10, x25
    ldr     x2, [x20, #P_FCPROJW]
    add     x2, x2, x10, lsl #2
    mov     x3, x25
    mov     x4, x11
    bl      smooth_one
    add     x21, x21, #1
    b       8b
9:  mul     x9, x26, x27                // logits: Vp x C (wte)
    ldr     x0, [x20, #G_STATS]
    add     x0, x0, x9, lsl #2
    ldr     x1, [x20, #G_SCALES]
    add     x1, x1, x9, lsl #2
    ldr     x2, [x20, #P_WTE]
    ldr     x3, [x20, #G_VP]
    mov     x4, x25
    bl      smooth_one
    str     xzr, [x20, #G_ACT_CAP]      // the caller allocates its own activations
    str     xzr, [x20, #G_PROF_N]       // calibration passes do not count in the profile
    add     x9, x20, #G_PROF
    mov     x10, #0
10: str     xzr, [x9, x10, lsl #3]
    add     x10, x10, #1
    cmp     x10, #NPROF
    b.lt    10b
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// ===========================================================================
// model// ===========================================================================
// model

// void gpt2a_build_from_checkpoint(const char* path)
// Reads an llm.c fp32 checkpoint (magic 20240326, version 3).
FUNC SYM(gpt2a_build_from_checkpoint)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    mov     x19, x0
    LOADADDR x20, G
    LOADADDR x1, s_rb
    bl      SYM(fopen)
    cbnz    x0, 1f
    LOADADDR x0, s_err_open
    mov     x1, x19
    b       die
1:  mov     x21, x0                     // FILE*
    add     x0, x20, #G_HEADER
    mov     x1, #4
    mov     x2, #256
    mov     x3, x21
    bl      xfread
    ldr     w9, [x20, #G_HEADER]
    MOV32   w10, MODEL_MAGIC
    cmp     w9, w10
    b.eq    2f
    LOADADDR x0, s_err_magic
    b       die
2:  ldr     w9, [x20, #(G_HEADER + 4)]
    cmp     w9, #3
    b.eq    3f
    LOADADDR x0, s_err_version
    b       die
3:  ldrsw   x9, [x20, #(G_HEADER + 8)]
    str     x9, [x20, #G_MAXT]
    ldrsw   x9, [x20, #(G_HEADER + 12)]
    str     x9, [x20, #G_V]
    ldrsw   x9, [x20, #(G_HEADER + 16)]
    str     x9, [x20, #G_L]
    ldrsw   x9, [x20, #(G_HEADER + 20)]
    str     x9, [x20, #G_NH]
    ldrsw   x9, [x20, #(G_HEADER + 24)]
    str     x9, [x20, #G_C]
    ldrsw   x9, [x20, #(G_HEADER + 28)]
    str     x9, [x20, #G_VP]
    // print the config
    LOADADDR x0, s_cfg_head
    bl      pr0
    LOADADDR x0, s_cfg_maxT
    ldr     x1, [x20, #G_MAXT]
    bl      pr1
    LOADADDR x0, s_cfg_V
    ldr     x1, [x20, #G_V]
    bl      pr1
    LOADADDR x0, s_cfg_Vp
    ldr     x1, [x20, #G_VP]
    bl      pr1
    LOADADDR x0, s_cfg_L
    ldr     x1, [x20, #G_L]
    bl      pr1
    LOADADDR x0, s_cfg_NH
    ldr     x1, [x20, #G_NH]
    bl      pr1
    LOADADDR x0, s_cfg_C
    ldr     x1, [x20, #G_C]
    bl      pr1
    // channels must split evenly into heads
    ldr     x9, [x20, #G_C]
    ldr     x10, [x20, #G_NH]
    cbz     x10, 4f
    sdiv    x11, x9, x10
    msub    x11, x11, x10, x9
    cbz     x11, 5f
4:  LOADADDR x0, s_err_heads
    b       die
5:  // parameter tensor sizes, in llm.c order
    ldr     x22, [x20, #G_C]            // C
    ldr     x23, [x20, #G_L]            // L
    mul     x24, x23, x22               // L*C
    add     x9, x20, #G_PSIZES
    ldr     x10, [x20, #G_VP]
    mul     x10, x10, x22
    str     x10, [x9, #0]               // wte      Vp*C
    ldr     x10, [x20, #G_MAXT]
    mul     x10, x10, x22
    str     x10, [x9, #8]               // wpe      maxT*C
    str     x24, [x9, #16]              // ln1w     L*C
    str     x24, [x9, #24]              // ln1b     L*C
    mul     x10, x24, x22
    add     x10, x10, x10, lsl #1
    str     x10, [x9, #32]              // qkvw     L*3C*C
    add     x10, x24, x24, lsl #1
    str     x10, [x9, #40]              // qkvb     L*3C
    mul     x10, x24, x22
    str     x10, [x9, #48]              // attprojw L*C*C
    str     x24, [x9, #56]              // attprojb L*C
    str     x24, [x9, #64]              // ln2w     L*C
    str     x24, [x9, #72]              // ln2b     L*C
    mul     x10, x24, x22
    lsl     x10, x10, #2
    str     x10, [x9, #80]              // fcw      L*4C*C
    lsl     x10, x24, #2
    str     x10, [x9, #88]              // fcb      L*4C
    mul     x10, x24, x22
    lsl     x10, x10, #2
    str     x10, [x9, #96]              // fcprojw  L*C*4C
    str     x24, [x9, #104]             // fcprojb  L*C
    str     x22, [x9, #112]             // lnfw     C
    str     x22, [x9, #120]             // lnfb     C
    // total
    mov     x25, #0
    mov     x10, #0
6:  ldr     x11, [x9, x10, lsl #3]
    add     x25, x25, x11
    add     x10, x10, #1
    cmp     x10, #16
    b.lt    6b
    str     x25, [x20, #G_NPARAMS]
    LOADADDR x0, s_cfg_np
    mov     x1, x25
    bl      pr1
    // read all parameters in one go
    lsl     x0, x25, #2
    bl      xmalloc
    str     x0, [x20, #G_PARAMS_MEM]
    mov     x26, x0
    mov     x1, #4
    mov     x2, x25
    mov     x3, x21
    bl      xfread
    // point the 16 tensors into the block
    add     x9, x20, #G_PSIZES
    add     x12, x20, #G_PARAMS
    mov     x10, #0
7:  str     x26, [x12, x10, lsl #3]
    ldr     x11, [x9, x10, lsl #3]
    add     x26, x26, x11, lsl #2
    add     x10, x10, #1
    cmp     x10, #16
    b.lt    7b
    mov     x0, x21
    bl      SYM(fclose)
    LOADADDR x0, s_env_profile
    bl      SYM(getenv)
    cmp     x0, #0
    cset    x9, ne
    str     x9, [x20, #G_PROFON]
    // repack matmul weights: fp16 5x32 kernel if asked for and C, Vp % 32 == 0,
    // else fp32 5x16 kernel if C, Vp % 16 == 0, else leave them as they are
    ldr     x9, [x20, #G_C]
    ldr     x10, [x20, #G_VP]
    orr     x21, x9, x10
    LOADADDR x0, s_env_prec
    bl      SYM(getenv)
    cbz     x0, 7f
    ldr     w9, [x0]                    // "fp16"
    MOV32   w10, 0x36317066
    cmp     w9, w10
    b.ne    6f
    tst     x21, #31
    b.ne    7f
    bl      pack16_all
    b       8f
6:  MOV32   w10, 0x38746e69             // "int8"
    cmp     w9, w10
    b.ne    7f
    tst     x21, #15
    b.ne    7f
    LOADADDR x0, s_env_calib
    bl      SYM(getenv)
    cbz     x0, 60f
    bl      calibrate
    b       61f
60: LOADADDR x0, s_nocalib
    bl      pr0
61: bl      q8_all
    b       8f
7:  tst     x21, #15
    b.ne    8f
    bl      pack_all
8:  RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// void gpt2a_alloc_acts(long B, long T): activation buffers for up to B*T rows
FUNC SYM(gpt2a_alloc_acts)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    LOADADDR x19, G
    mul     x20, x0, x1                 // N
    str     x20, [x19, #G_ACT_CAP]
    ldr     x21, [x19, #G_C]
    mul     x21, x21, x20               // N*C
    lsl     x0, x21, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_X]
    lsl     x0, x21, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_LN]
    add     x0, x21, x21, lsl #1
    lsl     x0, x0, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_QKV]
    lsl     x0, x21, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_ATTY]
    lsl     x0, x21, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_PROJ]
    lsl     x0, x21, #4                 // 4*N*C floats
    bl      xmalloc
    str     x0, [x19, #G_ACT_FCH]
    ldr     x9, [x19, #G_VP]
    mul     x0, x9, x20
    lsl     x0, x0, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_PROBS]
    ldr     x0, [x19, #G_MAXT]
    add     x0, x0, #8
    lsl     x0, x0, #8                  // maxT + 8 floats for each of 64 threads
    bl      xmalloc
    str     x0, [x19, #G_ACT_SCRATCH]
    lsl     x0, x21, #3                 // 4*N*C halves
    bl      xmalloc
    str     x0, [x19, #G_ACT_H16]
    lsl     x0, x21, #2                 // 4*N*C int8
    bl      xmalloc
    str     x0, [x19, #G_ACT_Q8]
    add     x0, x20, #8                 // N + 8 row scales
    lsl     x0, x0, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_XS]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// float* gpt2a_forward(const int* tokens, long B, long T)
// Returns probs, shape (B, T, Vp), same as llm.c's acts.probs.
FUNC SYM(gpt2a_forward)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    mov     x19, x0                     // tokens
    mov     x20, x1                     // B
    mov     x21, x2                     // T
    LOADADDR x28, G
    ldr     x9, [x28, #G_PARAMS_MEM]
    cbnz    x9, 1f
    LOADADDR x0, s_err_init
    b       die
1:  mul     x22, x20, x21               // N = B*T
    ldr     x9, [x28, #G_MAXT]
    cmp     x21, x9
    b.le    2f
    LOADADDR x0, s_err_T
    b       die
2:  ldr     x9, [x28, #G_ACT_CAP]
    cbnz    x9, 3f
    mov     x0, x20
    mov     x1, x21
    bl      SYM(gpt2a_alloc_acts)
    ldr     x9, [x28, #G_ACT_CAP]
3:  cmp     x22, x9
    b.le    4f
    LOADADDR x0, s_err_cap
    b       die
4:  ldr     x10, [x28, #G_V]            // every token must be in [0, V)
    mov     x11, #0
5:  cmp     x11, x22
    b.ge    7f
    ldrsw   x12, [x19, x11, lsl #2]
    cmp     x12, #0
    b.lt    6f
    cmp     x12, x10
    b.ge    6f
    add     x11, x11, #1
    b       5b
6:  LOADADDR x0, s_err_tok
    b       die
7:  bl      threads_init
    bl      pool_start
    ldr     x9, [x28, #G_PROFON]
    cbz     x9, 9f
    ldr     x10, [x28, #G_PROF_N]
    add     x10, x10, #1
    str     x10, [x28, #G_PROF_N]
    mrs     x10, cntvct_el0
    str     x10, [x28, #G_PROF_LAST]
9:  ldr     x23, [x28, #G_C]            // C
    add     x27, x23, x23, lsl #1
    add     x27, x27, x23, lsl #2       // 7C: per-layer stats stride
    // token + position embeddings
    ldr     x0, [x28, #G_ACT_X]
    mov     x1, x19
    ldr     x2, [x28, #P_WTE]
    ldr     x3, [x28, #P_WPE]
    mov     x4, x20
    mov     x5, x21
    mov     x6, x23
    bl      SYM(a_encoder_forward)
    PROFTICK x28, 0
    mov     x24, #0                     // layer l
fwd_layer:
    ldr     x9, [x28, #G_L]
    cmp     x24, x9
    b.ge    fwd_final
    mul     x25, x24, x23               // l*C
    // ln1
    ldr     x0, [x28, #G_ACT_LN]
    ldr     x1, [x28, #G_ACT_X]
    ldr     x2, [x28, #P_LN1W]
    add     x2, x2, x25, lsl #2
    ldr     x3, [x28, #P_LN1B]
    add     x3, x3, x25, lsl #2
    mov     x4, x22
    mov     x5, x23
    bl      playernorm
    PROFTICK x28, 1
    // qkv = ln1 @ qkvw^T + qkvb
    add     x26, x23, x23, lsl #1       // 3C
    mul     x9, x25, x26
    ldr     x0, [x28, #G_ACT_QKV]
    ldr     x1, [x28, #G_ACT_LN]
    ldr     x2, [x28, #P_QKVW]
    add     x2, x2, x9, lsl #2
    mul     x10, x24, x26
    ldr     x3, [x28, #P_QKVB]
    add     x3, x3, x10, lsl #2
    mov     x4, x22
    mov     x5, x23
    mov     x6, x26
    mul     x13, x24, x27
    MMHOOK  x1, x22, x23, x13
    bl      pmatmul
    PROFTICK x28, 2
    // attention
    ldr     x0, [x28, #G_ACT_ATTY]
    ldr     x1, [x28, #G_ACT_QKV]
    ldr     x2, [x28, #G_ACT_SCRATCH]
    mov     x3, x20
    mov     x4, x21
    mov     x5, x23
    ldr     x6, [x28, #G_NH]
    bl      pattention
    PROFTICK x28, 3
    // attention projection
    mul     x9, x25, x23
    ldr     x0, [x28, #G_ACT_PROJ]
    ldr     x1, [x28, #G_ACT_ATTY]
    ldr     x2, [x28, #P_ATTPROJW]
    add     x2, x2, x9, lsl #2
    ldr     x3, [x28, #P_ATTPROJB]
    add     x3, x3, x25, lsl #2
    mov     x4, x22
    mov     x5, x23
    mov     x6, x23
    mul     x13, x24, x27
    add     x13, x13, x23
    MMHOOK  x1, x22, x23, x13
    bl      pmatmul
    PROFTICK x28, 4
    // residual
    ldr     x0, [x28, #G_ACT_X]
    mov     x1, x0
    ldr     x2, [x28, #G_ACT_PROJ]
    mul     x3, x22, x23
    bl      presidual
    PROFTICK x28, 5
    // ln2
    ldr     x0, [x28, #G_ACT_LN]
    ldr     x1, [x28, #G_ACT_X]
    ldr     x2, [x28, #P_LN2W]
    add     x2, x2, x25, lsl #2
    ldr     x3, [x28, #P_LN2B]
    add     x3, x3, x25, lsl #2
    mov     x4, x22
    mov     x5, x23
    bl      playernorm
    PROFTICK x28, 1
    // MLP up: fch = ln2 @ fcw^T + fcb
    lsl     x26, x23, #2                // 4C
    mul     x9, x25, x26
    ldr     x0, [x28, #G_ACT_FCH]
    ldr     x1, [x28, #G_ACT_LN]
    ldr     x2, [x28, #P_FCW]
    add     x2, x2, x9, lsl #2
    mul     x10, x24, x26
    ldr     x3, [x28, #P_FCB]
    add     x3, x3, x10, lsl #2
    mov     x4, x22
    mov     x5, x23
    mov     x6, x26
    mul     x13, x24, x27
    add     x13, x13, x23, lsl #1
    MMHOOK  x1, x22, x23, x13
    bl      pmatmul
    PROFTICK x28, 6
    // gelu, in place
    ldr     x0, [x28, #G_ACT_FCH]
    mov     x1, x0
    mul     x2, x22, x26
    bl      pgelu
    PROFTICK x28, 7
    // MLP down
    mul     x9, x25, x26
    ldr     x0, [x28, #G_ACT_PROJ]
    ldr     x1, [x28, #G_ACT_FCH]
    ldr     x2, [x28, #P_FCPROJW]
    add     x2, x2, x9, lsl #2
    ldr     x3, [x28, #P_FCPROJB]
    add     x3, x3, x25, lsl #2
    mov     x4, x22
    mov     x5, x26
    mov     x6, x23
    mul     x13, x24, x27
    add     x13, x13, x23, lsl #1
    add     x13, x13, x23
    MMHOOK  x1, x22, x26, x13
    bl      pmatmul
    PROFTICK x28, 8
    // residual
    ldr     x0, [x28, #G_ACT_X]
    mov     x1, x0
    ldr     x2, [x28, #G_ACT_PROJ]
    mul     x3, x22, x23
    bl      presidual
    PROFTICK x28, 5
    add     x24, x24, #1
    b       fwd_layer
fwd_final:
    // final layernorm
    ldr     x0, [x28, #G_ACT_LN]
    ldr     x1, [x28, #G_ACT_X]
    ldr     x2, [x28, #P_LNFW]
    ldr     x3, [x28, #P_LNFB]
    mov     x4, x22
    mov     x5, x23
    bl      playernorm
    PROFTICK x28, 1
    // logits = lnf @ wte^T (weights tied, no bias), into the probs buffer
    ldr     x0, [x28, #G_ACT_PROBS]
    ldr     x1, [x28, #G_ACT_LN]
    ldr     x2, [x28, #G_WTE_PACKED]
    cbnz    x2, 8f
    ldr     x2, [x28, #P_WTE]
8:  mov     x3, #0
    mov     x4, x22
    mov     x5, x23
    ldr     x6, [x28, #G_VP]
    ldr     x9, [x28, #G_L]
    mul     x13, x9, x27
    MMHOOK  x1, x22, x23, x13
    bl      pmatmul
    PROFTICK x28, 9
    // softmax, in place
    ldr     x0, [x28, #G_ACT_PROBS]
    mov     x1, x0
    mov     x2, x22
    ldr     x3, [x28, #G_V]
    ldr     x4, [x28, #G_VP]
    bl      psoftmax
    PROFTICK x28, 10
    bl      pool_stop
    ldr     x0, [x28, #G_ACT_PROBS]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// ===========================================================================
// threads

// threads_init: choose the thread count once (GPT2A_THREADS, else all online CPUs)
    .p2align 4
threads_init:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    LOADADDR x9, G
    ldr     x10, [x9, #G_NTHREADS]
    cbnz    x10, 3f
    LOADADDR x0, s_env_threads
    bl      SYM(getenv)
    cbz     x0, 1f
    bl      SYM(atoi)
    sxtw    x0, w0
    cmp     x0, #0
    b.gt    2f
1:
#if defined(__APPLE__)
    mov     x0, #58                     // _SC_NPROCESSORS_ONLN on macOS
#else
    mov     x0, #84                     // _SC_NPROCESSORS_ONLN on Linux
#endif
    bl      SYM(sysconf)
2:  bl      clamp_threads
    LOADADDR x9, G
    str     x0, [x9, #G_NTHREADS]
3:  ldp     x29, x30, [sp], #16
    ret

// x0 -> clamp(x0, 1, MAX_THREADS)
    .p2align 4
clamp_threads:
    cmp     x0, #1
    b.ge    1f
    mov     x0, #1
1:  cmp     x0, #MAX_THREADS
    b.le    2f
    mov     x0, #MAX_THREADS
2:  ret

// void gpt2a_set_threads(long n)
FUNC SYM(gpt2a_set_threads)
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      clamp_threads
    LOADADDR x9, G
    str     x0, [x9, #G_NTHREADS]
    ldp     x29, x30, [sp], #16
    ret

// long gpt2a_get_threads(void)
FUNC SYM(gpt2a_get_threads)
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      threads_init
    LOADADDR x9, G
    ldr     x0, [x9, #G_NTHREADS]
    ldp     x29, x30, [sp], #16
    ret

// pool_start: create workers 1 .. n-1 (thread 0 is the caller)
    .p2align 4
pool_start:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    LOADADDR x19, G
    str     xzr, [x19, #G_POOL_GEN]
    str     xzr, [x19, #G_POOL_DONE]
    mov     x20, #1
1:  ldr     x9, [x19, #G_NTHREADS]
    cmp     x20, x9
    b.ge    3f
    add     x0, x19, #G_TIDS
    add     x0, x0, x20, lsl #3         // &tids[i]
    mov     x1, #0
    LOADADDR x2, worker_main
    mov     x3, x20                     // worker id
    bl      SYM(pthread_create)
    cbz     w0, 2f
    LOADADDR x0, s_err_thread
    b       die
2:  add     x20, x20, #1
    b       1b
3:  ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// pool_stop: tell the workers to exit (fn = 0, new generation) and join them
    .p2align 4
pool_stop:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    LOADADDR x19, G
    ldr     x9, [x19, #G_NTHREADS]
    cmp     x9, #1
    b.le    3f
    str     xzr, [x19, #G_POOL_FN]
    add     x10, x19, #G_POOL_GEN
    ldr     x11, [x10]
    add     x11, x11, #1
    stlr    x11, [x10]                  // release: fn = 0 is visible first
    mov     x20, #1
1:  ldr     x9, [x19, #G_NTHREADS]
    cmp     x20, x9
    b.ge    3f
    add     x9, x19, #G_TIDS
    ldr     x0, [x9, x20, lsl #3]
    mov     x1, #0
    bl      SYM(pthread_join)
    add     x20, x20, #1
    b       1b
3:  ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// worker_main(id): wait for a new generation, run fn(id), count it as done
    .p2align 4
worker_main:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    mov     x19, x0                     // id
    LOADADDR x20, G
    mov     x21, #0                     // last generation seen
1:  add     x10, x20, #G_POOL_GEN
2:  ldar    x9, [x10]                   // acquire: task and fn are visible
    cmp     x9, x21
    b.ne    3f
    yield
    b       2b
3:  mov     x21, x9
    ldr     x9, [x20, #G_POOL_FN]
    cbz     x9, 9f
    mov     x0, x19
    blr     x9
    add     x10, x20, #G_POOL_DONE
4:  ldxr    x11, [x10]
    add     x11, x11, #1
    stlxr   w12, x11, [x10]             // release: our outputs are visible first
    cbnz    w12, 4b
    b       1b
9:  mov     x0, #0
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #48
    ret

// pool_run(fn): run fn(0) here and fn(1..n-1) on the workers, wait for all
    .p2align 4
pool_run:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    LOADADDR x19, G
    mov     x20, x0
    ldr     x9, [x19, #G_NTHREADS]
    cmp     x9, #1
    b.le    2f
    str     x20, [x19, #G_POOL_FN]
    str     xzr, [x19, #G_POOL_DONE]
    add     x10, x19, #G_POOL_GEN
    ldr     x11, [x10]
    add     x11, x11, #1
    stlr    x11, [x10]                  // release: fn, task, done = 0 visible first
2:  mov     x0, #0
    blr     x20
    ldr     x9, [x19, #G_NTHREADS]
    cmp     x9, #1
    b.le    4f
    sub     x9, x9, #1
    add     x10, x19, #G_POOL_DONE
3:  ldar    x11, [x10]                  // acquire: worker outputs are visible
    cmp     x11, x9
    b.eq    4f
    yield
    b       3b
4:  ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret


// store x0..x6 as the task arguments
.macro SET_TASK
    LOADADDR x9, G
    add     x9, x9, #G_TASK
    stp     x0, x1, [x9]
    stp     x2, x3, [x9, #16]
    stp     x4, x5, [x9, #32]
    stp     x6, x7, [x9, #48]
.endm

// parallel wrappers, same arguments as the single-threaded kernels
    .p2align 4
pmatmul:                                // (out, inp, weight, bias, N, C, OC)
    LOADADDR x9, G
    ldr     x10, [x9, #G_PACKED]
    cmp     x10, #3
    b.eq    2f
    cmp     x10, #2
    b.ne    1f
    stp     x29, x30, [sp, #-80]!       // fp16: inp -> fp16 copy, then run on that
    mov     x29, sp
    stp     x0, x1, [sp, #16]
    stp     x2, x3, [sp, #32]
    stp     x4, x5, [sp, #48]
    str     x6, [sp, #64]
    ldr     x0, [x9, #G_ACT_H16]
    mul     x2, x4, x5
    bl      pconv16
    LOADADDR x14, G
    PROFTICK x14, 11
    ldp     x0, x1, [sp, #16]
    ldp     x2, x3, [sp, #32]
    ldp     x4, x5, [sp, #48]
    ldr     x6, [sp, #64]
    LOADADDR x9, G
    ldr     x1, [x9, #G_ACT_H16]
    ldp     x29, x30, [sp], #80
    b       1f
2:  stp     x29, x30, [sp, #-80]!       // int8: quantize the rows of inp, then run on that
    mov     x29, sp
    stp     x0, x1, [sp, #16]
    stp     x2, x3, [sp, #32]
    stp     x4, x5, [sp, #48]
    str     x6, [sp, #64]
    mov     x2, x1
    mov     x3, x4
    mov     x4, x5
    mov     x5, x7                      // per-channel 1/s (or 0)
    ldr     x0, [x9, #G_ACT_Q8]
    ldr     x1, [x9, #G_ACT_XS]
    bl      pquant
    LOADADDR x14, G
    PROFTICK x14, 11
    ldp     x0, x1, [sp, #16]
    ldp     x2, x3, [sp, #32]
    ldp     x4, x5, [sp, #48]
    ldr     x6, [sp, #64]
    LOADADDR x9, G
    ldr     x1, [x9, #G_ACT_Q8]
    ldp     x29, x30, [sp], #80
1:  SET_TASK
    LOADADDR x0, par_matmul
    b       pool_run
    .p2align 4
pattention:                             // (out, qkv, scratch, B, T, C, NH)
    SET_TASK
    LOADADDR x0, par_attention
    b       pool_run
    .p2align 4
pgelu:                                  // (out, inp, N)
    SET_TASK
    LOADADDR x0, par_gelu
    b       pool_run
    .p2align 4
psoftmax:                               // (probs, logits, N, V, Vp)
    SET_TASK
    LOADADDR x0, par_softmax
    b       pool_run

// part functions: x0 = part index. Each computes its slice, then tail-calls the kernel.

// matmul: output channels split in chunks of 16 (64 bytes)
    .p2align 4
par_matmul:
    LOADADDR x9, G
    ldr     x16, [x9, #G_PACKED]
    ldr     x10, [x9, #G_NTHREADS]
    add     x15, x9, #G_TASK
    ldr     x11, [x15, #48]             // OC
    add     x12, x11, x10
    sub     x12, x12, #1
    udiv    x12, x12, x10               // ceil(OC / n)
    add     x12, x12, #31
    lsr     x12, x12, #5
    lsl     x12, x12, #5                // rounded up to 32
    mul     x13, x0, x12                // oA
    cmp     x13, x11
    b.ge    9f
    add     x14, x13, x12
    cmp     x14, x11
    csel    x14, x11, x14, gt           // oB
    ldr     x5, [x15, #40]              // C
    ldr     x0, [x15, #0]
    add     x0, x0, x13, lsl #2         // out + oA
    ldr     x1, [x15, #8]               // inp
    ldr     x2, [x15, #16]
    mul     x9, x13, x5
    add     x2, x2, x9, lsl #2          // weight + oA*C
    ldr     x3, [x15, #24]
    cbz     x3, 1f
    add     x3, x3, x13, lsl #2         // bias + oA
1:  ldr     x4, [x15, #32]              // N
    sub     x6, x14, x13                // channels in this slice
    mov     x7, x11                     // ldo = OC
    cbz     x16, 3f
    cmp     x16, #2
    b.eq    4f
    cmp     x16, #3
    b.eq    5f
    b       SYM(a_matmul_packed)
5:  ldr     x5, [x15, #40]              // int8: panels of 16 C + 64 bytes
    ldr     x2, [x15, #16]
    lsr     x9, x13, #4
    lsl     x10, x5, #4
    add     x10, x10, #64
    mul     x9, x9, x10
    add     x2, x2, x9
    b       SYM(a_matmul_q8)
3:  b       SYM(a_matmul_strided)
4:  ldr     x5, [x15, #40]              // fp16: halves, so re-derive the offsets
    ldr     x2, [x15, #16]
    mul     x9, x13, x5
    add     x2, x2, x9, lsl #1          // weight + oA*C halves
    b       SYM(a_matmul_f16)
9:  ret

// attention: heads split evenly, one scratch row per part
    .p2align 4
par_attention:
    LOADADDR x9, G
    ldr     x10, [x9, #G_NTHREADS]
    ldr     x8, [x9, #G_MAXT]
    add     x8, x8, #8                  // scratch row stride
    add     x15, x9, #G_TASK
    ldr     x6, [x15, #48]              // NH
    add     x12, x6, x10
    sub     x12, x12, #1
    udiv    x12, x12, x10               // heads per part
    mul     x13, x0, x12                // h0
    cmp     x13, x6
    b.ge    9f
    add     x14, x13, x12
    cmp     x14, x6
    csel    x14, x6, x14, gt            // h1
    mul     x9, x0, x8
    ldr     x2, [x15, #16]
    add     x2, x2, x9, lsl #2          // scratch + part*maxT
    ldr     x0, [x15, #0]
    ldr     x1, [x15, #8]
    ldr     x3, [x15, #24]
    ldr     x4, [x15, #32]
    ldr     x5, [x15, #40]
    orr     x7, x13, x14, lsl #32
    b       SYM(a_attention_heads)
9:  ret

// gelu: elements split in chunks of 4
    .p2align 4
par_gelu:
    LOADADDR x9, G
    ldr     x10, [x9, #G_NTHREADS]
    add     x15, x9, #G_TASK
    ldr     x11, [x15, #16]             // N
    add     x12, x11, x10
    sub     x12, x12, #1
    udiv    x12, x12, x10
    add     x12, x12, #3
    lsr     x12, x12, #2
    lsl     x12, x12, #2
    mul     x13, x0, x12                // start
    cmp     x13, x11
    b.ge    9f
    sub     x2, x11, x13
    cmp     x2, x12
    csel    x2, x12, x2, gt             // count
    ldr     x0, [x15, #0]
    add     x0, x0, x13, lsl #2
    ldr     x1, [x15, #8]
    add     x1, x1, x13, lsl #2
    b       SYM(a_gelu_forward)
9:  ret

// softmax: rows split evenly
    .p2align 4
par_softmax:
    LOADADDR x9, G
    ldr     x10, [x9, #G_NTHREADS]
    add     x15, x9, #G_TASK
    ldr     x11, [x15, #16]             // N rows
    add     x12, x11, x10
    sub     x12, x12, #1
    udiv    x12, x12, x10
    mul     x13, x0, x12                // first row
    cmp     x13, x11
    b.ge    9f
    sub     x2, x11, x13
    cmp     x2, x12
    csel    x2, x12, x2, gt             // rows here
    ldr     x3, [x15, #24]              // V
    ldr     x4, [x15, #32]              // Vp
    mul     x9, x13, x4
    ldr     x0, [x15, #0]
    add     x0, x0, x9, lsl #2
    ldr     x1, [x15, #8]
    add     x1, x1, x9, lsl #2
    b       SYM(a_softmax_forward)
9:  ret


// more parallel wrappers: input conversion, quantization, layernorm, residual
    .p2align 4
pconv16:                                // (dst16, src, n)
    SET_TASK
    LOADADDR x0, par_conv16
    b       pool_run
    .p2align 4
pquant:                                 // (q, xs, x, N, C, chanmul)
    SET_TASK
    LOADADDR x0, par_quant
    b       pool_run
    .p2align 4
playernorm:                             // (out, inp, w, b, N, C)
    SET_TASK
    LOADADDR x0, par_layernorm
    b       pool_run
    .p2align 4
presidual:                              // (out, a, b, n)
    SET_TASK
    LOADADDR x0, par_residual
    b       pool_run

// split n elements in chunks of 16: x0 = part -> x13 = start, x2 = count (or ret)
.macro SPLIT16 nslot
    LOADADDR x9, G
    ldr     x10, [x9, #G_NTHREADS]
    add     x15, x9, #G_TASK
    ldr     x11, [x15, #\nslot]
    add     x12, x11, x10
    sub     x12, x12, #1
    udiv    x12, x12, x10
    add     x12, x12, #15
    lsr     x12, x12, #4
    lsl     x12, x12, #4
    mul     x13, x0, x12
    cmp     x13, x11
    b.ge    99f
    sub     x2, x11, x13
    cmp     x2, x12
    csel    x2, x12, x2, gt
.endm
// split N rows evenly: x13 = first row, x14 = rows (or ret)
.macro SPLITROWS nslot
    LOADADDR x9, G
    ldr     x10, [x9, #G_NTHREADS]
    add     x15, x9, #G_TASK
    ldr     x11, [x15, #\nslot]
    add     x12, x11, x10
    sub     x12, x12, #1
    udiv    x12, x12, x10
    mul     x13, x0, x12
    cmp     x13, x11
    b.ge    99f
    sub     x14, x11, x13
    cmp     x14, x12
    csel    x14, x12, x14, gt
.endm

    .p2align 4
par_conv16:
    SPLIT16 16
    ldr     x0, [x15, #0]
    add     x0, x0, x13, lsl #1
    ldr     x1, [x15, #8]
    add     x1, x1, x13, lsl #2
    b       SYM(a_f32_to_f16)
99: ret

    .p2align 4
par_residual:
    SPLIT16 24
    mov     x3, x2
    ldr     x0, [x15, #0]
    add     x0, x0, x13, lsl #2
    ldr     x1, [x15, #8]
    add     x1, x1, x13, lsl #2
    ldr     x2, [x15, #16]
    add     x2, x2, x13, lsl #2
    b       SYM(a_residual_forward)
99: ret

    .p2align 4
par_quant:                              // rows of (q, xs, x, N, C, chanmul)
    SPLITROWS 24
    ldr     x4, [x15, #32]              // C
    mul     x9, x13, x4
    ldr     x0, [x15, #0]
    add     x0, x0, x9                  // q + r0*C
    ldr     x1, [x15, #8]
    add     x1, x1, x13, lsl #2         // xs + r0
    ldr     x2, [x15, #16]
    add     x2, x2, x9, lsl #2          // x + r0*C
    mov     x3, x14
    ldr     x5, [x15, #40]
    b       SYM(a_quant_rows)
99: ret

    .p2align 4
par_layernorm:                          // rows of (out, inp, w, b, N, C)
    SPLITROWS 32
    ldr     x5, [x15, #40]              // C
    mul     x9, x13, x5
    ldr     x0, [x15, #0]
    add     x0, x0, x9, lsl #2
    ldr     x1, [x15, #8]
    add     x1, x1, x9, lsl #2
    ldr     x2, [x15, #16]
    ldr     x3, [x15, #24]
    mov     x4, x14
    b       SYM(a_layernorm_forward)
99: ret

// void gpt2a_profile_print(void): per-stage ms per forward pass (GPT2A_PROFILE=1)
FUNC SYM(gpt2a_profile_print)
    stp     x29, x30, [sp, #-64]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     d8, d9, [sp, #48]
    LOADADDR x19, G
    ldr     x20, [x19, #G_PROF_N]
    cbz     x20, 9f
    LOADADDR x0, s_prof_head
    mov     x1, x20
    bl      pr1
    mrs     x9, cntfrq_el0
    ucvtf   d8, x9
    ucvtf   d0, x20
    fmul    d8, d8, d0                  // ticks per second x passes
    movi    d9, #0
    mov     x21, #0
1:  add     x9, x19, #G_PROF
    ldr     x9, [x9, x21, lsl #3]
    ucvtf   d0, x9
    MOV32   w9, 1000
    scvtf   d1, w9
    fmul    d0, d0, d1
    fdiv    d0, d0, d8                  // ms per pass
    fadd    d9, d9, d0
    LOADADDR x0, s_prof_fmt
    LOADADDR x1, s_prof_names
    add     x1, x1, x21, lsl #4
    bl      pr1f
    add     x21, x21, #1
    cmp     x21, #NPROF
    b.lt    1b
    LOADADDR x0, s_prof_total
    fmov    d0, d9
    bl      prf
9:  ldp     d8, d9, [sp, #48]
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x29, x30, [sp], #64
    ret

// printf with one double: x0 = format, d0
    .p2align 4
prf:
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    sub     sp, sp, #16
    str     d0, [sp]
    bl      SYM(printf)
    mov     sp, x29
    ldp     x29, x30, [sp], #16
    ret

// ===========================================================================
// sampler (llm.c: xorshift64* rng, sample from the cumulative distribution)

// unsigned a_random_u32(uint64_t* state)
FUNC SYM(a_random_u32)
    ldr     x1, [x0]
    eor     x1, x1, x1, lsr #12
    eor     x1, x1, x1, lsl #25
    eor     x1, x1, x1, lsr #27
    str     x1, [x0]
    movz    x2, #0xDD1D
    movk    x2, #0x4F6C, lsl #16
    movk    x2, #0xF491, lsl #32
    movk    x2, #0x2545, lsl #48
    mul     x1, x1, x2
    lsr     x0, x1, #32
    ret

// float a_random_f32(uint64_t* state): uniform in [0, 1)
FUNC SYM(a_random_f32)
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    bl      SYM(a_random_u32)
    lsr     w0, w0, #8
    ucvtf   s0, w0
    FLOADK  s1, w9, F_2POW24
    fdiv    s0, s0, s1
    ldp     x29, x30, [sp], #16
    ret

// int a_sample_mult(const float* probs, int n, float coin)
FUNC SYM(a_sample_mult)
    sxtw    x1, w1
    movi    d1, #0                      // cdf
    mov     x2, #0
1:  cmp     x2, x1
    b.ge    2f
    ldr     s2, [x0, x2, lsl #2]
    fadd    s1, s1, s2
    fcmp    s0, s1
    b.mi    3f                          // coin < cdf
    add     x2, x2, #1
    b       1b
2:  sub     w0, w1, #1
    ret
3:  mov     w0, w2
    ret

// ===========================================================================
// tokenizer (decode only, llm.c format: magic 20240328, version 1 or 2)

// void gpt2a_tokenizer_init(const char* path)
FUNC SYM(gpt2a_tokenizer_init)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    mov     x19, x0
    LOADADDR x20, G
    str     xzr, [x20, #G_TOK_OK]
    LOADADDR x1, s_rb
    bl      SYM(fopen)
    cbnz    x0, 1f
    LOADADDR x0, s_warn_tok
    mov     x1, x19
    bl      pr1
    b       tk_done
1:  mov     x21, x0                     // FILE*
    add     x0, x20, #G_HEADER
    mov     x1, #4
    mov     x2, #256
    mov     x3, x21
    bl      xfread
    ldr     w9, [x20, #G_HEADER]
    MOV32   w10, TOK_MAGIC
    cmp     w9, w10
    b.eq    2f
    LOADADDR x0, s_err_tmagic
    b       die
2:  ldr     w11, [x20, #(G_HEADER + 4)]     // version
    ldr     w12, [x20, #(G_HEADER + 8)]     // vocab size
    str     x12, [x20, #G_TOK_VOCAB]
    cmp     w11, #1
    b.ne    3f
    mov     x9, #50256                  // version 1: GPT-2 end-of-text token
    str     x9, [x20, #G_TOK_EOT]
    b       5f
3:  cmp     w11, #2
    b.ne    4f
    ldr     w9, [x20, #(G_HEADER + 12)]
    str     x9, [x20, #G_TOK_EOT]
    b       5f
4:  LOADADDR x0, s_err_tversion
    b       die
5:  ldr     x0, [x20, #G_TOK_VOCAB]
    lsl     x0, x0, #3
    bl      xmalloc
    str     x0, [x20, #G_TOK_TABLE]
    mov     x22, x0
    mov     x23, #0                     // token index
6:  ldr     x9, [x20, #G_TOK_VOCAB]
    cmp     x23, x9
    b.ge    8f
    add     x0, x20, #G_BYTE
    mov     x1, #1
    mov     x2, #1
    mov     x3, x21
    bl      xfread
    ldrb    w24, [x20, #G_BYTE]         // length
    cbnz    w24, 7f
    LOADADDR x0, s_err_tlen
    b       die
7:  add     x0, x24, #1
    bl      xmalloc
    mov     x25, x0
    mov     x1, #1
    mov     x2, x24
    mov     x3, x21
    bl      xfread
    strb    wzr, [x25, x24]
    str     x25, [x22, x23, lsl #3]
    add     x23, x23, #1
    b       6b
8:  mov     x0, x21
    bl      SYM(fclose)
    mov     x9, #1
    str     x9, [x20, #G_TOK_OK]
tk_done:
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// print_token: x0 = token id. tokenizer_decode + safe_printf from llm.c:
// a lone byte that is neither printable nor whitespace is skipped.
    .p2align 4
print_token:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    str     x19, [sp, #16]
    LOADADDR x9, G
    ldr     x10, [x9, #G_TOK_VOCAB]
    cmp     x0, x10
    b.lo    1f
    mov     x1, x0
    LOADADDR x0, s_invalid_tok
    bl      pr1
    b       9f
1:  ldr     x10, [x9, #G_TOK_TABLE]
    ldr     x19, [x10, x0, lsl #3]
    ldrb    w9, [x19]
    cbz     w9, 9f
    ldrb    w10, [x19, #1]
    cbnz    w10, 2f
    sub     w10, w9, #0x20              // isprint: 0x20 .. 0x7e
    cmp     w10, #0x5e
    b.ls    2f
    sub     w10, w9, #9                 // isspace: 0x09 .. 0x0d (0x20 handled above)
    cmp     w10, #4
    b.ls    2f
    b       9f
2:  ldrb    w0, [x19], #1
    cbz     w0, 9f
    bl      SYM(putchar)
    b       2b
9:  ldr     x19, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret

// ===========================================================================
// main: unconditional generation, like the sampling loop in llm.c.
// Starts from <|endoftext|> and samples num_tokens - 1 tokens (default 64).

#if !defined(GPT2A_NO_MAIN)
FUNC SYM(main)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    sxtw    x19, w0                     // argc
    mov     x20, x1                     // argv
    LOADADDR x21, s_default_ckpt
    cmp     x19, #1
    b.le    1f
    ldr     x21, [x20, #8]
1:  LOADADDR x22, s_default_tok
    cmp     x19, #2
    b.le    2f
    ldr     x22, [x20, #16]
2:  mov     x23, #64                    // genT
    cmp     x19, #3
    b.le    3f
    ldr     x0, [x20, #24]
    bl      SYM(atoi)
    sxtw    x23, w0
3:  mov     x0, x21
    bl      SYM(gpt2a_build_from_checkpoint)
    LOADADDR x28, G
    ldr     x9, [x28, #G_MAXT]          // clamp genT to [2, maxT]
    cmp     x23, x9
    csel    x23, x9, x23, gt
    mov     x9, #2
    cmp     x23, #2
    csel    x23, x9, x23, lt
    bl      SYM(gpt2a_get_threads)
    mov     x1, x0
    LOADADDR x0, s_threads
    bl      pr1
    mov     x0, x22
    bl      SYM(gpt2a_tokenizer_init)
    ldr     x9, [x28, #G_TOK_OK]
    ldr     x24, [x28, #G_TOK_EOT]
    cbnz    x9, 4f
    ldr     x24, [x28, #G_V]            // no tokenizer: use the last token as the start
    sub     x24, x24, #1
4:  lsl     x0, x23, #2
    bl      xmalloc
    mov     x25, x0                     // tokens
    mov     x9, #0
5:  str     w24, [x25, x9, lsl #2]
    add     x9, x9, #1
    cmp     x9, x23
    b.lt    5b
    mov     x0, #1
    mov     x1, x23
    bl      SYM(gpt2a_alloc_acts)
    mov     x9, #1337
    str     x9, [x28, #G_RNG]
    LOADADDR x0, s_generating
    bl      pr0
    bl      SYM(clock)
    mov     x27, x0
    mov     x26, #1                     // t
gen_loop:
    cmp     x26, x23
    b.ge    gen_done
    mov     x0, x25
    mov     x1, #1
    mov     x2, x26
    bl      SYM(gpt2a_forward)          // B=1, T=t: probs for positions 0..t-1
    sub     x9, x26, #1
    ldr     x10, [x28, #G_VP]
    mul     x9, x9, x10
    add     x24, x0, x9, lsl #2         // probs[t-1, :]
    add     x0, x28, #G_RNG
    bl      SYM(a_random_f32)           // s0 = coin
    mov     x0, x24
    ldr     x1, [x28, #G_V]
    bl      SYM(a_sample_mult)
    str     w0, [x25, x26, lsl #2]
    ldr     x9, [x28, #G_TOK_OK]
    cbz     x9, 6f
    bl      print_token
    b       7f
6:  mov     x1, x0
    LOADADDR x0, s_tokid
    bl      pr1
7:  mov     x0, #0
    bl      SYM(fflush)
    add     x26, x26, #1
    b       gen_loop
gen_done:
    bl      SYM(clock)
    sub     x9, x0, x27
    scvtf   d0, x9
    MOV32   w9, 1000000                 // CLOCKS_PER_SEC
    scvtf   d1, w9
    fdiv    d0, d0, d1
    sub     x1, x23, #1
    LOADADDR x0, s_timing
    bl      pr1f
    mov     w0, #0
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret
#endif


// test hook: a_matmul_q8 with an explicit row-scale array (9th argument on the stack is
// avoided: scales passed in G_ACT_XS). void a_matmul_q8_test(out, xq, wq, bias, N, C, OC, ldo, xs)
#if defined(GPT2A_NO_MAIN)
FUNC SYM(a_matmul_q8_test)
    ldr     x9, [sp]                    // 9th argument (stack on both ABIs)
    LOADADDR x10, G
    str     x9, [x10, #G_ACT_XS]
    b       SYM(a_matmul_q8)
#endif

#if !defined(__APPLE__)
    .section .note.GNU-stack,"",%progbits
#endif
