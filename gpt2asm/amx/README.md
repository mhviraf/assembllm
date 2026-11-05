# AMX (native macOS on Apple silicon)

AMX is the undocumented matrix coprocessor in Apple silicon. The instruction encoding comes from Peter Cawley's reverse-engineering notes ([corsix/amx](https://github.com/corsix/amx)): `.word 0x00201000 | op << 5 | Xn`, where general register `Xn` carries a 64-bit operand. The operand fields below were measured on an M1 with `amx_probe3`; the matmul kernels in `../gpt2_forward.S` rely on exactly these.

| Instruction | Operand (as measured on M1) |
|---|---|
| `ldx` / `ldy` (op 0 / 1) | Address in bits 0–55, register 0–7 in bits 56+. Bit 62 loads 128 bytes into two consecutive registers, and the address must then be 128-byte aligned (64-byte aligned is a bus error). |
| `ldz` / `stz` (op 4 / 5) | Address in bits 0–55, Z row 0–63 in bits 56+. |
| `fma32` (op 12) | **Y** byte offset in bits 0–8, **X** byte offset in bits 10–18, Z row in bits 20–25, bit 27 ignores the old Z. Matrix mode: `Z[4j + z][i] += X[i] * Y[j]` for i, j < 16. |
| `fma16` (op 15) with bit 62 | fp16 X and Y, fp32 sums: `Z[2j + i % 2][i / 2] += X[i] * Y[j]` for i, j < 32. One 32 × 32 fp32 tile fills Z. |

The first kernels assumed X's offset in bits 0–8 and Y's in bits 10–18. That got rows 0–15 right and everything after wrong, and `amx_probe3` showed the two fields are the other way round.

## Files

- `amx_probe` (round 1) and `amx_probe3` (rounds 2–3) probe the semantics and throughput. `amx_probe3` runs every case in its own process, so a case that faults costs one line. Its M1 output is in `probe_m1.txt`.
- `amx_mm_test` checks every AMX matmul kernel against the NEON fp32 kernel and times it at GPT-2's shapes. The kernels are `a_matmul_amx32`, `a_matmul_amx16`, `a_matmul_amxr0` (fp32) and `a_matmul_amxh` (fp16).
- `run_all.sh` times full forward passes on the real weights and writes `results.txt`. `run_all.sh probe kernels` also runs the probe and the kernel test. `results_m1.txt` holds a full run on a MacBook Air M1. `amx_probe3 ecores` measures the efficiency cores' AMX unit; `amx_probe3 overlap` measures NEON work interleaved with AMX work.
- `a_matmul_amxh_mix` (`../gpt2_forward.S`, `GPT2A_AMX_MIX`) runs AMX and NEON fp16 in one instruction stream. Assembling it with `-DAMX_NOP_TEST` on Linux turns the AMX instructions into nops, so its NEON half can be tested under qemu.
- `build.sh` builds everything on Linux without Xcode (see `../native/README.md`).

## Throughput on M1 (`amx_probe3`)

| Inner loop | 1 thread | 4 threads |
|---|---|---|
| fma32 only, 4 accumulators | 1,194 GFLOP/s | 1,575 GFLOP/s |
| fma16 (fp16 sums) only | 3,049 GFLOP/s | 2,671 GFLOP/s |
| fma16 with fp32 sums only | 1,285 GFLOP/s | 1,451 GFLOP/s |
| fp32 kernel step: 2 ldx + 2 ldy + 4 fma32 | 1,520 GFLOP/s | 1,638 GFLOP/s |
| fp16 kernel step: ldx + ldy + fma16 (fp32 sums) | 1,421 GFLOP/s | 1,640 GFLOP/s |

One AMX unit serves the four performance cores, so more threads add little to raw throughput. For comparison, NEON tops out at about 100 GFLOP/s fp32 per core.
