#!/usr/bin/env bash
# Builds the native macOS AMX probes and matmul test on Linux, without Xcode: clang for the
# Mach-O objects, ld64.lld for linking against a hand-written libSystem.tbd (the dylib's symbol list).
set -e
cd "$(dirname "$0")"
T="-target arm64-apple-macos11"
LD=$(command -v ld64.lld || command -v ld64.lld-18)
LINK="$LD -arch arm64 -platform_version macos 11.0 11.0"
CC="clang $T -O2 -fno-stack-protector -fno-builtin"
$CC -c amx_probe.c -o amx_probe.o
clang $T -c amx_probe.S -o amx_probe_s.o
$LINK -o amx_probe amx_probe.o amx_probe_s.o libSystem.tbd
$CC -c amx_probe3.c -o amx_probe3.o
clang $T -c amx_probe3.S -o amx_probe3_s.o
$LINK -o amx_probe3 amx_probe3.o amx_probe3_s.o libSystem.tbd
$CC -c amx_mm_test.c -o amx_mm_test.o
clang $T -DGPT2A_NO_MAIN -c ../gpt2_forward.S -o gpt2_nomain_mac.o
$LINK -o amx_mm_test amx_mm_test.o gpt2_nomain_mac.o libSystem.tbd
