// AMX probe, rounds 2 and 3 (native macOS on Apple silicon only): single-instruction wrappers for
// register dumps, plus throughput loops shaped like the matmul kernels.
// Encoding (github.com/corsix/amx): .word 0x00201000 | (op << 5) | Xn.
// ops: 0 ldx, 1 ldy, 2 stx, 3 sty, 4 ldz, 5 stz, 12 fma32, 15 fma16; set/clr = 17 after 3 nops.

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
.macro WRAP name, op
    .globl  _\name
    .p2align 4
_\name:
    AMXOP   \op, 0
    ret
.endm

    .text
    .globl  _amx_set
    .p2align 4
_amx_set:
    AMX_SET
    ret
    .globl  _amx_clr
    .p2align 4
_amx_clr:
    AMX_CLR
    ret
    WRAP    amx_ldx, 0
    WRAP    amx_ldy, 1
    WRAP    amx_stx, 2
    WRAP    amx_sty, 3
    WRAP    amx_ldz, 4
    WRAP    amx_stz, 5
    WRAP    amx_fma32, 12
    WRAP    amx_fma16, 15

// void loop_f16(long n, uint64_t op0, uint64_t op1): 16 fma16 per iteration, no loads
    .globl  _loop_f16
    .p2align 4
_loop_f16:
1:  .rept 8
    AMXOP   15, 1
    AMXOP   15, 2
    .endr
    subs    x0, x0, #1
    b.gt    1b
    ret

// void loop_h32(long n, const void* xb, const void* yb, uint64_t op):
// per k one 64-byte ldx, one 64-byte ldy and one fma16 with operand op
// (the fp16-in / fp32-sum kernel shape); addresses walk 64 bytes per k in a 32 KB window.
    .globl  _loop_h32
    .p2align 4
_loop_h32:
    mov     x9, #0
1:  add     x10, x1, x9
    AMXOP   0, 10
    add     x11, x2, x9
    AMXOP   1, 11
    AMXOP   15, 3
    add     x9, x9, #64
    and     x9, x9, #0x7fc0
    subs    x0, x0, #1
    b.gt    1b
    ret

// void loop_v2(long n, const void* xb, const void* yb): the a_matmul_amx32 k step:
// two 64-byte ldx (panels 16 KB apart), two 64-byte ldy, four fma32.
    .globl  _loop_v2
    .p2align 4
_loop_v2:
    mov     x12, #0
    movz    x13, #0x10, lsl #16
    orr     x13, x13, #64
    movz    x14, #0x21, lsl #16
    movz    x15, #0x31, lsl #16
    orr     x15, x15, #64
    mov     x16, #1
    lsl     x16, x16, #56
    mov     x9, #0
    mov     x10, #0
1:  add     x3, x1, x9
    AMXOP   0, 3
    add     x4, x3, #4, lsl #12
    orr     x4, x4, x16
    AMXOP   0, 4
    add     x5, x2, x10
    AMXOP   1, 5
    add     x6, x5, #64
    orr     x6, x6, x16
    AMXOP   1, 6
    AMXOP   12, 12
    AMXOP   12, 13
    AMXOP   12, 14
    AMXOP   12, 15
    add     x9, x9, #64
    and     x9, x9, #0x3fc0
    add     x10, x10, #256
    and     x10, x10, #0xff00
    subs    x0, x0, #1
    b.gt    1b
    ret

// void loop_pair(long n, const void* xb, const void* yb, uint64_t pairbits): one 128-byte
// ldx and one 128-byte ldy (if pairs work) and four fma32 per k.
    .globl  _loop_pair
    .p2align 4
_loop_pair:
    mov     x12, #0
    movz    x13, #0x10, lsl #16
    orr     x13, x13, #64
    movz    x14, #0x21, lsl #16
    movz    x15, #0x31, lsl #16
    orr     x15, x15, #64
    mov     x9, #0
    mov     x10, #0
1:  add     x4, x1, x9
    orr     x4, x4, x3
    AMXOP   0, 4
    add     x5, x2, x10
    orr     x5, x5, x3
    AMXOP   1, 5
    AMXOP   12, 12
    AMXOP   12, 13
    AMXOP   12, 14
    AMXOP   12, 15
    add     x9, x9, #128
    and     x9, x9, #0x7f80
    add     x10, x10, #256
    and     x10, x10, #0xff00
    subs    x0, x0, #1
    b.gt    1b
    ret
