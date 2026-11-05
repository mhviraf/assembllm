# Native macOS binaries without Xcode

`build.sh` cross-builds macOS arm64 executables on Linux. `clang -target arm64-apple-macos11` compiles and assembles the Mach-O objects. `ld64.lld` links them against `../amx/libSystem.tbd`, a hand-written list of the libSystem functions these programs use. The linker records a dependency on `/usr/lib/libSystem.B.dylib` and ad-hoc signs the binary, so macOS runs it without the SDK or Xcode.

- `gpt2`: the generator. Run it as `./gpt2 ../../gpt2_124M.bin ../../gpt2_tokenizer.bin 64`.
- `gpt2_profile`: times forward passes of 64 tokens. Run it as `./gpt2_profile ../../gpt2_124M.bin ../../tiny_shakespeare_val.bin [threads] [passes]`.

Both accept `GPT2A_PRECISION=fp32|fp16|int8|amx`, `GPT2A_CALIB` and `GPT2A_PROFILE=1`. If macOS refuses to start a binary that was copied over, run `xattr -c <binary>`.
