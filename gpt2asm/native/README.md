# Native macOS binaries without Xcode

`build.sh` cross-builds macOS arm64 executables on Linux. `clang -target arm64-apple-macos11` compiles and assembles the Mach-O objects. `ld64.lld` links them against `../amx/libSystem.tbd`, a hand-written list of the libSystem functions these programs use. The linker records a dependency on `/usr/lib/libSystem.B.dylib` and ad-hoc signs the binary, so macOS runs it without the SDK or Xcode.

- `gpt2`: the generator. Run it as `./gpt2 ../../gpt2_124M.bin ../../gpt2_tokenizer.bin 64`.
- `gpt2_profile`: times forward passes of 64 tokens and prints the mean next-token loss. Run it as `./gpt2_profile ../../gpt2_124M.bin ../../tiny_shakespeare_val.bin [threads] [passes]`.

Both accept these settings:
- `GPT2A_PRECISION`: `fp32`, `fp16`, `int8`, `amx` (fp32 on the AMX coprocessor) or `amx16` (fp16 on AMX with fp32 sums).
- `GPT2A_CALIB`: calibration tokens for int8.
- `GPT2A_PROFILE=1`: per-stage timing.
- `GPT2A_AMX_THREADS`: how many threads issue AMX matmuls (default: all).
- `GPT2A_AMX_PF`: how far ahead, in bytes, the CPU prefetches each AMX kernel's weights (default 8192).
- `GPT2A_AMX_E=n`, `GPT2A_AMX_ESHARE=%`, `GPT2A_AMX_EQOS`: experimental. These add n threads on the efficiency cores that take that share of each AMX matmul. It's slower in practice; see step 19 in the top-level readme.

On macOS the default thread count is the number of performance cores, and all threads run at user-interactive QoS so they stay on those cores. If macOS refuses to start a binary that was copied over, run `xattr -c <binary>`.
