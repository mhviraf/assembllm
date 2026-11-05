#!/usr/bin/env bash
# Runs the native AMX checks and timings on an Apple-silicon Mac and saves everything to
# amx/results.txt. Needs the prebuilt binaries (amx/build.sh, native/build.sh) and, for the
# forward-pass and generation timings, the llm.c starter files in the repo root.
# Usage: run_all.sh [probe] [kernels]   (also run the instruction probe / the kernel test)
cd "$(dirname "$0")/.."
ROOT=$(cd .. && pwd)
OUT=amx/results.txt
B="amx/amx_probe3 amx/amx_mm_test native/gpt2_profile native/gpt2"
chmod +x $B 2>/dev/null
xattr -c $B 2>/dev/null
QUIET="^max_seq\|^vocab\|^padded\|^num_\|^channels\|^\[GPT-2"
{
    case " $* " in *" probe "*)
        echo "### amx_probe3"; ./amx/amx_probe3; ./amx/amx_probe3 ecores; echo ;; esac
    case " $* " in *" kernels "*)
        echo "### amx_mm_test"; ./amx/amx_mm_test; echo ;; esac
    CK="$ROOT/gpt2_124M.bin"
    VAL="$ROOT/tiny_shakespeare_val.bin"
    TOK="$ROOT/gpt2_tokenizer.bin"
    CAL="$ROOT/tiny_shakespeare_train.bin"
    if [ -f "$CK" ] && [ -f "$VAL" ]; then
        P=native/gpt2_profile
        echo "### forward pass, 64 tokens, 4 performance-core threads"
        echo "--- amx16"
        GPT2A_PRECISION=amx16 $P "$CK" "$VAL" 4 40 | grep "threads"
        for q in 9 17 21 25; do
            echo "--- amx16 + 4 E-core AMX threads at QoS $q, E share 20%"
            GPT2A_PRECISION=amx16 GPT2A_AMX_E=4 GPT2A_AMX_ESHARE=20 GPT2A_AMX_EQOS=$q $P "$CK" "$VAL" 4 20 | grep "threads"
        done
        echo "--- amx16 + 2 E-core AMX threads at QoS 17, E share 15%"
        GPT2A_PRECISION=amx16 GPT2A_AMX_E=2 GPT2A_AMX_ESHARE=15 $P "$CK" "$VAL" 4 20 | grep "threads"
        echo "--- amx16 again"
        GPT2A_PRECISION=amx16 $P "$CK" "$VAL" 4 40 | grep "threads"
        echo "--- int8 NEON (calibrated)"
        GPT2A_PRECISION=int8 GPT2A_CALIB="$CAL" $P "$CK" "$VAL" 4 40 | grep "threads"
    else
        echo "(no $CK or $VAL: skipping the forward-pass timings)"
    fi
    echo
    echo "ALL DONE"
} 2>&1 | tee "$OUT"
