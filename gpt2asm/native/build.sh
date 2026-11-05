#!/usr/bin/env bash
# Builds native macOS (arm64) binaries on Linux without Xcode: clang makes the Mach-O
# objects, ld64.lld links them against ../amx/libSystem.tbd (libSystem's symbol list).
set -e
cd "$(dirname "$0")"
T="-target arm64-apple-macos11"
LD=$(command -v ld64.lld || command -v ld64.lld-18)
LINK="$LD -arch arm64 -platform_version macos 11.0 11.0"
clang $T -c ../gpt2_forward.S -o gpt2.o
clang $T -DGPT2A_NO_MAIN -c ../gpt2_forward.S -o gpt2_nomain.o
clang $T -O2 -fno-stack-protector -fno-builtin -c profile.c -o profile.o
$LINK -o gpt2 gpt2.o ../amx/libSystem.tbd
$LINK -o gpt2_profile profile.o gpt2_nomain.o ../amx/libSystem.tbd
