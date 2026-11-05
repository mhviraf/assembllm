#!/usr/bin/env bash
# Builds the native macOS AMX probe on Linux, without Xcode: clang for the Mach-O objects,
# ld64.lld for linking against a hand-written libSystem.tbd (the dylib's symbol list).
set -e
cd "$(dirname "$0")"
T="-target arm64-apple-macos11"
clang $T -O2 -fno-stack-protector -fno-builtin -c amx_probe.c -o amx_probe.o
clang $T -c amx_probe.S -o amx_probe_s.o
LD=$(command -v ld64.lld || command -v ld64.lld-18)
"$LD" -arch arm64 -platform_version macos 11.0 11.0 -o amx_probe amx_probe.o amx_probe_s.o libSystem.tbd
clang $T -O2 -fno-stack-protector -fno-builtin -c amx_mm_test.c -o amx_mm_test.o
clang $T -c amx_matmul.S -o amx_matmul.o
clang $T -DGPT2A_NO_MAIN -c ../gpt2_forward.S -o gpt2_nomain_mac.o
"$LD" -arch arm64 -platform_version macos 11.0 11.0 -o amx_mm_test amx_mm_test.o amx_matmul.o gpt2_nomain_mac.o libSystem.tbd
