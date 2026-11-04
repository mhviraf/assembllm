#!/usr/bin/env bash
# Benchmark llm.c's C forward pass against the ARM64 assembly one, on this machine's CPU,
# using Karpathy's GPT-2 124M weights. Usage: ./run_bench.sh [num_samples=1000] [T=64]
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-1000}
T=${2:-64}

# 1. llm.c source + Karpathy's starter pack (GPT-2 124M weights, tokenizer, Tiny Shakespeare)
[ -d llm.c ] || git clone --depth 1 https://github.com/karpathy/llm.c.git
if [ ! -s llm.c/gpt2_124M.bin ] || [ ! -s llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin ]; then
    base="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main"
    mkdir -p llm.c/dev/data/tinyshakespeare
    for f in gpt2_124M.bin gpt2_tokenizer.bin; do
        curl -L --fail -o "llm.c/$f" "$base/$f?download=true"
    done
    for f in tiny_shakespeare_train.bin tiny_shakespeare_val.bin; do
        curl -L --fail -o "llm.c/dev/data/tinyshakespeare/$f" "$base/$f?download=true"
    done
fi

# 2. build: llm.c with its own CPU flags (-Ofast, -march=native if supported), single-threaded
CC=${CC:-cc}
FLAGS="-Ofast"
if echo 'int main(){return 0;}' | $CC -march=native -x c - -o /dev/null 2>/dev/null; then FLAGS="$FLAGS -march=native"; fi
$CC $FLAGS -DGPT2A_NO_MAIN -c gpt2_forward.S -o gpt2_nomain.o
$CC $FLAGS -w -Illm.c test/bench.c gpt2_nomain.o -o bench -lm
$CC $FLAGS gpt2_forward.S -o gpt2 -lm

# 3. run (val split first, then train split for more distinct samples)
DATA=llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin
[ "$N" -gt 500 ] && DATA=llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin
echo "compiler: $($CC --version | head -1), flags: $FLAGS"
./bench llm.c/gpt2_124M.bin "$DATA" "$N" "$T" | tee bench_results.txt
