#!/usr/bin/env bash
# Runs the native AMX checks and timings on an Apple-silicon Mac and saves everything to
# amx/results.txt. Needs the prebuilt binaries (amx/build.sh, native/build.sh) and, for the
# forward-pass timings, the llm.c starter files in the repo root.
# Usage: run_all.sh [probe]   ("probe" also runs the instruction-semantics probe)
cd "$(dirname "$0")/.."
ROOT=$(cd .. && pwd)
OUT=amx/results.txt
B="amx/amx_probe3 amx/amx_mm_test native/gpt2_profile native/gpt2"
chmod +x $B 2>/dev/null
xattr -c $B 2>/dev/null
{
    if [ "$1" = probe ]; then
        echo "### amx_probe3"
        ./amx/amx_probe3
        echo
    fi
    echo "### amx_mm_test"
    ./amx/amx_mm_test
    echo
    CK="$ROOT/gpt2_124M.bin"
    VAL="$ROOT/tiny_shakespeare_val.bin"
    if [ -f "$CK" ] && [ -f "$VAL" ]; then
        P=native/gpt2_profile
        echo "### forward pass, 64 tokens, 4 threads"
        for prec in amx amx16; do
            for at in 2 4; do
                echo "--- $prec, GPT2A_AMX_THREADS=$at"
                GPT2A_PRECISION=$prec GPT2A_AMX_THREADS=$at GPT2A_PROFILE=1 $P "$CK" "$VAL" 4 40 | grep -v "^max_seq\|^vocab\|^padded\|^num_\|^channels"
            done
        done
        echo "--- fp32 NEON"
        GPT2A_PRECISION=fp32 $P "$CK" "$VAL" 4 40 | grep "threads"
        echo "--- int8 NEON (calibrated)"
        GPT2A_PRECISION=int8 GPT2A_CALIB="$ROOT/tiny_shakespeare_train.bin" $P "$CK" "$VAL" 4 40 | grep "threads"
    else
        echo "(no $CK or $VAL: skipping the forward-pass timings)"
    fi
    echo
    echo "ALL DONE"
} 2>&1 | tee "$OUT"
