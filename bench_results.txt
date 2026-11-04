GPT-2 124M forward pass: llm.c (C) vs ARM64 assembly
Machine: MacBook Air (Apple silicon), Linux arm64 VM, 4 cores, gcc 11.4 -Ofast -march=native
Weights: random, same shape as GPT-2 124M (Karpathy's weights not downloaded yet; speed does not depend on weight values)
Workload: 1000 samples x 64 tokens, one forward pass each, B=1; the two implementations interleaved, alternating which goes first

== single-threaded, both (1000 samples, run in 5 chunks of 200) ==
llm.c      370.5 s total   370.5 ms/sample   172.7 tok/s
assembly   283.2 s total   283.2 ms/sample   226.0 tok/s
assembly speed vs llm.c: 1.31x
agreement: max |prob diff| 9.1e-12, top-token mismatches 0 of 64000

== llm.c with OpenMP on 4 cores vs single-threaded assembly (150 samples) ==
llm.c (C, -Ofast)           22.17       147.77        433.1    10.8255
assembly (ARM64)            45.08       300.50        213.0    10.8255
assembly is 0.49x the speed of llm.c
