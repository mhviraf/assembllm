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
// Only libc is used: fopen/fread/fclose, malloc, printf/putchar/fflush,
// atoi, clock, exit, and expf/tanhf. Everything else is written by hand.
//
// Calling convention: standard AAPCS64. x18 is never touched (reserved on
// Apple). Varargs for printf go through helpers that put the arguments both
// in registers (Linux) and on the stack (Apple), so one source serves both.

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

// float constants (bit patterns)
.equ F_NEG10000,   0xC61C4000      // -10000.0f, initial max in llm.c
.equ F_EPS,        0x3727C5AC      // 1e-5f, layernorm epsilon
.equ F_GELU_CUBE,  0x3D372713      // 0.044715f
.equ F_GELU_SCALE, 0x3F4C4229      // sqrtf(2.0f / M_PI)
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
.equ G_SIZE,        1472

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
    .p2align 4
G:  .space G_SIZE

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
// Main path: 4 rows x 4 output channels per tile, 16 vector accumulators.
// Leftover rows/channels (or C not a multiple of 4) use a single dot product.
FUNC SYM(a_matmul_forward)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    SAVE_X19_X28
    mov     x19, x0                     // out
    mov     x20, x1                     // inp
    mov     x21, x2                     // weight
    mov     x22, x3                     // bias
    mov     x23, x4                     // N
    mov     x24, x5                     // C
    mov     x25, x6                     // OC
    lsl     x26, x24, #2                // row bytes
    mov     x27, #0                     // r
    tst     x24, #3
    b.ne    mm_rows1
mm_rows4:
    add     x9, x27, #4
    cmp     x9, x23
    b.gt    mm_rows1
    mul     x9, x27, x26
    add     x12, x20, x9                // input rows r .. r+3
    add     x13, x12, x26
    add     x14, x13, x26
    add     x15, x14, x26
    mov     x28, #0                     // o
mm_cols4:
    add     x9, x28, #4
    cmp     x9, x25
    b.gt    mm_cols1
    mov     x0, x12
    mov     x1, x13
    mov     x2, x14
    mov     x3, x15
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
1:  ld1     {v0.4s}, [x0], #16
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
    b.gt    1b
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
    cbz     x22, 2f
    add     x9, x22, x28, lsl #2
    ld1     {v0.4s}, [x9]
    fadd    v16.4s, v16.4s, v0.4s
    fadd    v20.4s, v20.4s, v0.4s
    fadd    v24.4s, v24.4s, v0.4s
    fadd    v28.4s, v28.4s, v0.4s
2:  mul     x9, x27, x25
    add     x9, x9, x28
    add     x9, x19, x9, lsl #2
    lsl     x10, x25, #2
    st1     {v16.4s}, [x9]
    add     x9, x9, x10
    st1     {v20.4s}, [x9]
    add     x9, x9, x10
    st1     {v24.4s}, [x9]
    add     x9, x9, x10
    st1     {v28.4s}, [x9]
    add     x28, x28, #4
    b       mm_cols4
mm_cols1:                               // leftover output channels for rows r..r+3
    cmp     x28, x25
    b.ge    mm_rows4_next
    mov     x0, x12
    mov     x1, x27
    bl      mm_one
    mov     x0, x13
    add     x1, x27, #1
    bl      mm_one
    mov     x0, x14
    add     x1, x27, #2
    bl      mm_one
    mov     x0, x15
    add     x1, x27, #3
    bl      mm_one
    add     x28, x28, #1
    b       mm_cols1
mm_rows4_next:
    add     x27, x27, #4
    b       mm_rows4
mm_rows1:                               // leftover rows, one output at a time
    cmp     x27, x23
    b.ge    mm_done
    mul     x9, x27, x26
    add     x12, x20, x9
    mov     x28, #0
3:  cmp     x28, x25
    b.ge    4f
    mov     x0, x12
    mov     x1, x27
    bl      mm_one
    add     x28, x28, #1
    b       3b
4:  add     x27, x27, #1
    b       mm_rows1
mm_done:
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
    ret

// mm_one: out[x1, o] = bias[o] + dot(x0, weight[o]), o = x28.
// Uses a_matmul_forward's registers x19, x21, x22, x24, x25, x26, x28.
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
1:  mul     x9, x1, x25
    add     x9, x9, x28
    str     s0, [x19, x9, lsl #2]
    ldp     x29, x30, [sp], #16
    ret

// void a_attention_forward(float* out, const float* qkv, float* scratch,
//                          long B, long T, long C, long NH)
// qkv is (B,T,3C) holding q, k, v. out is (B,T,C). Causal softmax attention.
// scratch holds one row of attention weights (at least T floats).
FUNC SYM(a_attention_forward)
    stp     x29, x30, [sp, #-160]!
    mov     x29, sp
    SAVE_X19_X28
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    // locals: [sp, #128] exp-loop pointer, [sp, #136] exp-loop end
    mov     x19, x0                     // out
    mov     x20, x1                     // qkv
    mov     x21, x2                     // scratch
    mov     x22, x4                     // T
    mov     x23, x5                     // C
    mul     x24, x3, x4                 // B*T
    sdiv    x25, x5, x6                 // hs = C / NH
    // scale = 1.0 / sqrtf(hs), float sqrt then double divide, like the C code
    scvtf   s0, x25
    fsqrt   s0, s0
    fcvt    d0, s0
    fmov    d1, #1.0
    fdiv    d0, d1, d0
    fcvt    s8, d0
    mov     x26, #0                     // b*T
att_b:
    cmp     x26, x24
    b.ge    att_done
    mov     x27, #0                     // t
att_t:
    cmp     x27, x22
    b.ge    att_b_next
    mov     x28, #0                     // h*hs
att_h:
    cmp     x28, x23
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
2:  // pass 2: exp(score - max) and its sum
    movi    d10, #0
    str     x21, [sp, #128]
    add     x9, x27, #1
    add     x9, x21, x9, lsl #2
    str     x9, [sp, #136]
3:  ldr     x9, [sp, #128]
    ldr     x10, [sp, #136]
    cmp     x9, x10
    b.ge    4f
    ldr     s0, [x9]
    fsub    s0, s0, s9
    bl      SYM(expf)
    ldr     x9, [sp, #128]
    str     s0, [x9], #4
    str     x9, [sp, #128]
    fadd    s10, s10, s0
    b       3b
4:  movi    d11, #0                     // inv = sum == 0 ? 0 : 1/sum
    fcmp    s10, #0.0
    b.eq    5f
    fmov    s0, #1.0
    fdiv    s11, s0, s10
5:  // pass 3: normalize
    mov     x9, x21
    add     x10, x27, #1
6:  ldr     s0, [x9]
    fmul    s0, s0, s11
    str     s0, [x9], #4
    subs    x10, x10, #1
    b.gt    6b
    // pass 4: out[b,t,h,:] = sum_t2 att[t2] * v_t2
    add     x9, x26, x27
    mul     x9, x9, x23
    add     x9, x9, x28
    add     x10, x19, x9, lsl #2        // output slice
    mov     x11, x10
    mov     x12, x25
7:  cbz     x12, 8f
    str     wzr, [x11], #4
    sub     x12, x12, #1
    b       7b
8:  add     x13, x23, x23, lsl #1       // C3
    lsl     x14, x13, #2                // row bytes (recomputed, expf clobbers x14)
    mul     x9, x26, x13
    add     x9, x9, x23, lsl #1         // + 2C
    add     x9, x9, x28
    add     x13, x20, x9, lsl #2        // value of position 0
    mov     x12, #0                     // t2
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
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #160
    ret

// void a_gelu_forward(float* out, const float* inp, long N)    (tanh GELU)
// out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
FUNC SYM(a_gelu_forward)
    stp     x29, x30, [sp, #-128]!
    mov     x29, sp
    SAVE_X19_X28
    stp     d8, d9, [sp, #96]
    stp     d10, d11, [sp, #112]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    FLOADK  s8, w9, F_GELU_CUBE
    FLOADK  s9, w9, F_GELU_SCALE
1:  cbz     x21, 2f
    ldr     s10, [x20], #4              // x
    fmul    s0, s8, s10
    fmul    s0, s0, s10
    fmul    s0, s0, s10                 // 0.044715 * x * x * x
    fadd    s0, s10, s0
    fmul    s0, s9, s0
    bl      SYM(tanhf)
    fmov    s1, #1.0
    fadd    s0, s1, s0
    fmov    s1, #0.5
    fmul    s1, s1, s10
    fmul    s0, s1, s0
    str     s0, [x19], #4
    sub     x21, x21, #1
    b       1b
2:  ldp     d8, d9, [sp, #96]
    ldp     d10, d11, [sp, #112]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #128
    ret

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
    stp     x29, x30, [sp, #-112]!
    mov     x29, sp
    SAVE_X19_X28
    stp     d8, d9, [sp, #96]
    mov     x19, x0                     // probs
    mov     x20, x1                     // logits
    mov     x21, x2                     // N
    mov     x22, x3                     // V
    mov     x23, x4                     // Vp
    mov     x24, #0                     // row
sm_row:
    cmp     x24, x21
    b.ge    sm_done
    mul     x9, x24, x23
    add     x25, x20, x9, lsl #2        // logits row
    add     x26, x19, x9, lsl #2        // probs row
    // max (starting from -10000, like llm.c)
    FLOADK  s8, w9, F_NEG10000
    dup     v16.4s, v8.s[0]
    mov     x9, x25
    mov     x10, x22
1:  cmp     x10, #4
    b.lt    2f
    ld1     {v0.4s}, [x9], #16
    fmax    v16.4s, v16.4s, v0.4s
    sub     x10, x10, #4
    b       1b
2:  fmaxv   s8, v16.4s
3:  cbz     x10, 4f
    ldr     s0, [x9], #4
    fmax    s8, s8, s0
    sub     x10, x10, #1
    b       3b
4:  // exp and sum, in order
    movi    d9, #0
    mov     x27, #0
5:  cmp     x27, x22
    b.ge    6f
    ldr     s0, [x25, x27, lsl #2]
    fsub    s0, s0, s8
    bl      SYM(expf)
    str     s0, [x26, x27, lsl #2]
    fadd    s9, s9, s0
    add     x27, x27, #1
    b       5b
6:  // divide by the sum
    dup     v1.4s, v9.s[0]
    mov     x9, x26
    mov     x10, x22
7:  cmp     x10, #4
    b.lt    8f
    ld1     {v0.4s}, [x9]
    fdiv    v0.4s, v0.4s, v1.4s
    st1     {v0.4s}, [x9], #16
    sub     x10, x10, #4
    b       7b
8:  cbz     x10, 9f
    ldr     s0, [x9]
    fdiv    s0, s0, s9
    str     s0, [x9], #4
    sub     x10, x10, #1
    b       8b
9:  sub     x10, x23, x22               // zero the padding
10: cbz     x10, 11f
    str     wzr, [x9], #4
    sub     x10, x10, #1
    b       10b
11: add     x24, x24, #1
    b       sm_row
sm_done:
    ldp     d8, d9, [sp, #96]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #112
    ret

// ===========================================================================
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
    RESTORE_X19_X28
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
    lsl     x0, x0, #2
    bl      xmalloc
    str     x0, [x19, #G_ACT_SCRATCH]
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
7:  ldr     x23, [x28, #G_C]            // C
    // token + position embeddings
    ldr     x0, [x28, #G_ACT_X]
    mov     x1, x19
    ldr     x2, [x28, #P_WTE]
    ldr     x3, [x28, #P_WPE]
    mov     x4, x20
    mov     x5, x21
    mov     x6, x23
    bl      SYM(a_encoder_forward)
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
    bl      SYM(a_layernorm_forward)
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
    bl      SYM(a_matmul_forward)
    // attention
    ldr     x0, [x28, #G_ACT_ATTY]
    ldr     x1, [x28, #G_ACT_QKV]
    ldr     x2, [x28, #G_ACT_SCRATCH]
    mov     x3, x20
    mov     x4, x21
    mov     x5, x23
    ldr     x6, [x28, #G_NH]
    bl      SYM(a_attention_forward)
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
    bl      SYM(a_matmul_forward)
    // residual
    ldr     x0, [x28, #G_ACT_X]
    mov     x1, x0
    ldr     x2, [x28, #G_ACT_PROJ]
    mul     x3, x22, x23
    bl      SYM(a_residual_forward)
    // ln2
    ldr     x0, [x28, #G_ACT_LN]
    ldr     x1, [x28, #G_ACT_X]
    ldr     x2, [x28, #P_LN2W]
    add     x2, x2, x25, lsl #2
    ldr     x3, [x28, #P_LN2B]
    add     x3, x3, x25, lsl #2
    mov     x4, x22
    mov     x5, x23
    bl      SYM(a_layernorm_forward)
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
    bl      SYM(a_matmul_forward)
    // gelu, in place
    ldr     x0, [x28, #G_ACT_FCH]
    mov     x1, x0
    mul     x2, x22, x26
    bl      SYM(a_gelu_forward)
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
    bl      SYM(a_matmul_forward)
    // residual
    ldr     x0, [x28, #G_ACT_X]
    mov     x1, x0
    ldr     x2, [x28, #G_ACT_PROJ]
    mul     x3, x22, x23
    bl      SYM(a_residual_forward)
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
    bl      SYM(a_layernorm_forward)
    // logits = lnf @ wte^T (weights tied, no bias), into the probs buffer
    ldr     x0, [x28, #G_ACT_PROBS]
    ldr     x1, [x28, #G_ACT_LN]
    ldr     x2, [x28, #P_WTE]
    mov     x3, #0
    mov     x4, x22
    mov     x5, x23
    ldr     x6, [x28, #G_VP]
    bl      SYM(a_matmul_forward)
    // softmax, in place
    ldr     x0, [x28, #G_ACT_PROBS]
    mov     x1, x0
    mov     x2, x22
    ldr     x3, [x28, #G_V]
    ldr     x4, [x28, #G_VP]
    bl      SYM(a_softmax_forward)
    ldr     x0, [x28, #G_ACT_PROBS]
    RESTORE_X19_X28
    ldp     x29, x30, [sp], #96
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

#if !defined(__APPLE__)
    .section .note.GNU-stack,"",%progbits
#endif
