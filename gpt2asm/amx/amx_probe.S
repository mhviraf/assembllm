// AMX probe (Apple silicon, native macOS only).
// AMX is Apple's undocumented matrix coprocessor. Instruction encoding, from the
// public reverse-engineering notes (github.com/corsix/amx):
//     .word 0x00201000 | (op << 5) | Xn        operand in general register Xn
// set/clr (enable / disable AMX for this thread) are op 17 with imm 0 / 1, after 3 nops.

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

    .text
    .p2align 4
    .globl  _amx_set
_amx_set:
    AMX_SET
    ret
    .globl  _amx_clr
_amx_clr:
    AMX_CLR
    ret
    .globl  _amx_ldx
_amx_ldx:
    AMXOP   0, 0
    ret
    .globl  _amx_ldy
_amx_ldy:
    AMXOP   1, 0
    ret
    .globl  _amx_ldz
_amx_ldz:
    AMXOP   4, 0
    ret
    .globl  _amx_stz
_amx_stz:
    AMXOP   5, 0
    ret
    .globl  _amx_fma32
_amx_fma32:
    AMXOP   12, 0
    ret
    .globl  _amx_fma16
_amx_fma16:
    AMXOP   15, 0
    ret

// void amx_fma32_loop(long n, op0, op1, op2, op3): 16 fma32 per iteration
    .p2align 4
    .globl  _amx_fma32_loop
_amx_fma32_loop:
1:  AMXOP   12, 1
    AMXOP   12, 2
    AMXOP   12, 3
    AMXOP   12, 4
    AMXOP   12, 1
    AMXOP   12, 2
    AMXOP   12, 3
    AMXOP   12, 4
    AMXOP   12, 1
    AMXOP   12, 2
    AMXOP   12, 3
    AMXOP   12, 4
    AMXOP   12, 1
    AMXOP   12, 2
    AMXOP   12, 3
    AMXOP   12, 4
    subs    x0, x0, #1
    b.gt    1b
    ret

// void amx_fma16_loop(long n, op0, op1): 16 fma16 per iteration
    .p2align 4
    .globl  _amx_fma16_loop
_amx_fma16_loop:
1:  AMXOP   15, 1
    AMXOP   15, 2
    AMXOP   15, 1
    AMXOP   15, 2
    AMXOP   15, 1
    AMXOP   15, 2
    AMXOP   15, 1
    AMXOP   15, 2
    AMXOP   15, 1
    AMXOP   15, 2
    AMXOP   15, 1
    AMXOP   15, 2
    AMXOP   15, 1
    AMXOP   15, 2
    AMXOP   15, 1
    AMXOP   15, 2
    subs    x0, x0, #1
    b.gt    1b
    ret

// void amx_ldfma32_loop(long n, const float* x, const float* y): per iteration
// one 128-byte X load, one 128-byte Y load and 4 fma32 (a 32 x 32 fp32 block per k)
    .p2align 4
    .globl  _amx_ldfma32_loop
_amx_ldfma32_loop:
    mov     x9, #1
    lsl     x9, x9, #62                 // pair load
    orr     x1, x1, x9
    orr     x2, x2, x9
    mov     x3, #0                      // fma32: x off 0,  y off 0,  z 0
    mov     x4, #64                     //        x off 64, y off 0,  z 1
    movk    x4, #0x10, lsl #16
    mov     x5, #0                      //        x off 0,  y off 64, z 2
    movk    x5, #0x21, lsl #16          //        (y offset bit 16 and z bits 20+ share this field)
    mov     x6, #64                     //        x off 64, y off 64, z 3
    movk    x6, #0x31, lsl #16
1:  AMXOP   0, 1
    AMXOP   1, 2
    AMXOP   12, 3
    AMXOP   12, 4
    AMXOP   12, 5
    AMXOP   12, 6
    subs    x0, x0, #1
    b.gt    1b
    ret
