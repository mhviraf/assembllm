# AMX (experimental)

AMX is the undocumented matrix coprocessor in Apple silicon. The encoding and semantics used here come from Peter Cawley's reverse-engineering notes ([corsix/amx](https://github.com/corsix/amx)): `.word 0x00201000 | op << 5 | Xn`.

- `amx_probe`: checks the instruction semantics this project relies on and measures throughput.
- `amx_mm_test`: checks the fp32 AMX matmul kernel (`amx_matmul.S`, also built into `../gpt2_forward.S` as `GPT2A_PRECISION=amx`) against the NEON kernel, and times it at GPT-2's shapes.

`build.sh` builds both on Linux, without Xcode (see `../native/README.md`).

**Status:** written and assembled, but not yet run on Apple hardware.
