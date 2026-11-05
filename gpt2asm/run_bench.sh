#!/usr/bin/env bash
# Benchmark llm.c's C forward pass against the ARM64 assembly one on this machine's CPU,
# using Karpathy's GPT-2 124M weights.
# Usage: ./run_bench.sh [num_samples=1000] [threads=all CPUs]
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-1000}
TH=${2:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}
T=64

# 1. llm.c source + Karpathy's starter pack (GPT-2 124M weights, tokenizer, Tiny Shakespeare)
[ -d llm.c ] || git clone --depth 1 https://github.com/karpathy/llm.c.git
if [ ! -s llm.c/gpt2_124M.bin ] || [ ! -s llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin ]; then
    base="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main"
    mkdir -p llm.c/dev/data/tinyshakespeare
    for f in gpt2_124M.bin gpt2_tokenizer.bin; do
        curl -L --fail -o "llm.c/$f" "$base/$f?download=true"
    done
    for f in tiny_shakespeare_train.bin tiny_shakespeare_val.bin; do
        curl -L --fail -o "llm.c/dev/data/tinyshakespeare/$f" "$base/$f?download=true"
    done
fi

# 2. build: llm.c with its own CPU flags (-Ofast, -march=native if supported) and OpenMP if available
CC=${CC:-cc}
FLAGS="-Ofast"
if echo 'int main(){return 0;}' | $CC -march=native -x c - -o /dev/null 2>/dev/null; then FLAGS="$FLAGS -march=native"; fi
OMP=""
if echo 'int main(){return 0;}' | $CC -fopenmp -x c - -o /dev/null 2>/dev/null; then
    OMP="-fopenmp -DOMP"
elif [ -d /opt/homebrew/opt/libomp ]; then
    OMP="-Xclang -fopenmp -DOMP -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
elif [ "$TH" -gt 1 ]; then
    echo "note: no OpenMP found (on macOS: brew install libomp), so both run single-threaded"
    TH=1
fi
$CC $FLAGS -DGPT2A_NO_MAIN -c gpt2_forward.S -o gpt2_nomain.o
$CC $FLAGS -w -Illm.c test/bench.c gpt2_nomain.o -o bench $OMP -lm -lpthread
$CC $FLAGS gpt2_forward.S -o gpt2 -lm -lpthread

# 3. run on the train split (enough text for 1000 distinct samples).
# int8 calibrates on the train split instead, and is evaluated on val.
DATA=llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin
if [ "${GPT2A_PRECISION:-fp32}" = int8 ]; then
    export GPT2A_CALIB=${GPT2A_CALIB:-$DATA}
    DATA=llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin
fi
echo "compiler: $($CC --version | head -1), flags: $FLAGS, threads: $TH, precision: ${GPT2A_PRECISION:-fp32}"
# passive: llm.c's idle OpenMP threads sleep instead of spinning, so they do not
# steal cores from the assembly run that follows them in the interleaved benchmark
OMP_WAIT_POLICY=passive OMP_NUM_THREADS=$TH ./bench llm.c/gpt2_124M.bin "$DATA" "$N" "$T" 0 "$TH" | tee bench_results.txt
