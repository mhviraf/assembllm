// fp32 matmul on Apple's AMX coprocessor (native macOS on Apple silicon only), plus the
// NEON input transpose it needs. See amx_probe.S for the instruction encoding.
//
// out[r][o] = bias[o] + sum_k x[r][k] * W[o][k], for up to 64 rows r.
// Weights: the 16-channel k-major panels made by a_pack_weights (wp[(o/16*C + k)*16 + j]).
// Input:   transposed into xT[k*64 + r] by a_transpose64.
// AMX fma32 in matrix mode computes a 16 x 16 outer product: Z[4j + z][i] += X[i] * Y[j].
// One panel = 64 rows x 16 channels uses all of Z: row r lives in Z row 4 (r % 16) + r / 16.

#if defined(__APPLE__)
#define SYM(x) _##x
#else
#define SYM(x) x
#endif

.macro AMXOP op, reg
    .word   0x00201000 + ((\op) << 5) + (\reg)
.endm
.macro AMX_SET
    nop
    nop
    nop
    .word   0x00201220
.endm
.macro AMX_CLR
    nop
    nop
    nop
    .word   0x00201221
.endm
.macro FUNC name
    .text
    .p2align 4
    .globl  \name
\name:
.endm

    .data
    .p2align 6
amx_zeros:
    .space  64

// void a_transpose64(float* xT, const float* x, long n, long C)
// xT[k*64 + r] = x[min(r, n-1)*C + k] for r < round_up(n, 4); rows past that are left alone
// (their outputs are never stored). Needs C % 4 == 0, n <= 64.
FUNC SYM(a_transpose64)
    lsl     x9, x3, #2                  // row bytes
    sub     x10, x2, #1                 // last row
    mov     x11, #0                     // r
1:  cmp     x11, x2
    b.ge    9f
    // 4 row pointers, clamped to the last row
    madd    x12, x11, x9, x1
    add     x4, x11, #1
    cmp     x4, x10
    csel    x4, x10, x4, gt
    madd    x13, x4, x9, x1
    add     x4, x11, #2
    cmp     x4, x10
    csel    x4, x10, x4, gt
    madd    x14, x4, x9, x1
    add     x4, x11, #3
    cmp     x4, x10
    csel    x4, x10, x4, gt
    madd    x15, x4, x9, x1
    add     x16, x0, x11, lsl #2        // &xT[0*64 + r]
    mov     x17, x3
2:  ld1     {v0.4s}, [x12], #16
    ld1     {v1.4s}, [x13], #16
    ld1     {v2.4s}, [x14], #16
    ld1     {v3.4s}, [x15], #16
    trn1    v4.4s, v0.4s, v1.4s
    trn2    v5.4s, v0.4s, v1.4s
    trn1    v6.4s, v2.4s, v3.4s
    trn2    v7.4s, v2.4s, v3.4s
    trn1    v0.2d, v4.2d, v6.2d
    trn1    v1.2d, v5.2d, v7.2d
    trn2    v2.2d, v4.2d, v6.2d
    trn2    v3.2d, v5.2d, v7.2d
    str     q0, [x16]
    str     q1, [x16, #256]
    str     q2, [x16, #512]
    str     q3, [x16, #768]
    add     x16, x16, #1024             // 4 k further
    subs    x17, x17, #4
    b.gt    2b
    add     x11, x11, #4
    b       1b
9:  ret

#if defined(__APPLE__)
// void a_matmul_amx(float* out, const float* xT, const float* wp, const float* bias,
//                   long n, long C, long OC, long ldo)
// n <= 64 rows, OC % 16 == 0. Enables AMX for the calling thread and disables it again.
FUNC SYM(a_matmul_amx)
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]
    mov     x19, x0                     // out
    mov     x20, x1                     // xT
    mov     x21, x2                     // packed weights
    mov     x22, x3                     // bias
    mov     x23, x4                     // n
    mov     x24, x5                     // C
    mov     x25, x6                     // OC
    lsl     x26, x7, #2                 // output row bytes
    AMX_SET
    mov     x12, #0                     // fma32 operands: rows 0-15  -> z 0
    movz    x13, #0x11, lsl #16         //   y offset  64 (rows 16-31) -> z 1
    movz    x14, #0x22, lsl #16         //   y offset 128 (rows 32-47) -> z 2
    movz    x15, #0x33, lsl #16         //   y offset 192 (rows 48-63) -> z 3
    mov     x16, #1
    lsl     x16, x16, #62               // ldy: 128-byte pair into Y0, Y1
    movz    x17, #0x4200, lsl #48       // ldy: pair into Y2, Y3 (bit 62 + register 2 << 56)
    adrp    x27, amx_zeros@PAGE
    add     x27, x27, amx_zeros@PAGEOFF
    lsl     x28, x24, #6                // panel bytes (C x 16 floats)
    mov     x7, #0                      // o
1:  cmp     x7, x25
    b.ge    9f
    // Z starts as the bias (or zero) in every row
    mov     x9, x27
    cbz     x22, 2f
    add     x9, x22, x7, lsl #2
2:  mov     x10, #0
3:  orr     x11, x9, x10, lsl #56
    AMXOP   4, 11                       // ldz
    add     x10, x10, #1
    cmp     x10, #64
    b.lt    3b
    // k loop: 16 weights, 64 inputs, 4 outer products per k
    madd    x0, x7, x24, xzr
    add     x0, x21, x0, lsl #2         // this panel
    mov     x1, x20
    mov     x2, x24
4:  AMXOP   0, 0                        // ldx X0 <- W[o .. o+15][k]
    orr     x3, x1, x16
    AMXOP   1, 3                        // ldy Y0, Y1 <- xT[k][0 .. 31]
    add     x4, x1, #128
    orr     x4, x4, x17
    AMXOP   1, 4                        // ldy Y2, Y3 <- xT[k][32 .. 63]
    AMXOP   12, 12
    AMXOP   12, 13
    AMXOP   12, 14
    AMXOP   12, 15
    add     x0, x0, #64
    add     x1, x1, #256
    subs    x2, x2, #1
    b.gt    4b
    // store the n real rows: row r is Z row 4 (r % 16) + r / 16
    mov     x10, #0
5:  cmp     x10, x23
    b.ge    6f
    and     x11, x10, #15
    lsl     x11, x11, #2
    add     x11, x11, x10, lsr #4       // Z row
    madd    x9, x10, x26, x19
    add     x9, x9, x7, lsl #2          // &out[r][o]
    orr     x9, x9, x11, lsl #56
    AMXOP   5, 9                        // stz
    add     x10, x10, #1
    b       5b
6:  add     x7, x7, #16
    b       1b
9:  AMX_CLR
    ldp     x19, x20, [sp, #16]
    ldp     x21, x22, [sp, #32]
    ldp     x23, x24, [sp, #48]
    ldp     x25, x26, [sp, #64]
    ldp     x27, x28, [sp, #80]
    ldp     x29, x30, [sp], #96
    ret
#endif
