#!/usr/bin/env bash
# Build and run the AMX probe natively on macOS (Apple silicon). Results go to amx_probe.txt.
set -e
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "AMX is only reachable from native macOS on Apple silicon"; exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
    echo "The command line developer tools are missing: run  xcode-select --install"; exit 1
fi
cc -O2 amx_probe.c amx_probe.S -o amx_probe
{ sysctl -n machdep.cpu.brand_string; ./amx_probe; } 2>&1 | tee amx_probe.txt
