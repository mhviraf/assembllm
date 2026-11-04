CC      ?= cc
CFLAGS  ?= -O2
LDFLAGS ?=
LLMC    ?= ../llm.c
RUN     ?=
FWD_T   ?= 64
GEN_N   ?= 32

gpt2: gpt2_forward.S
	$(CC) $(CFLAGS) gpt2_forward.S -o $@ -lm $(LDFLAGS)

gpt2_nomain.o: gpt2_forward.S
	$(CC) $(CFLAGS) -DGPT2A_NO_MAIN -c gpt2_forward.S -o $@

test_vs_llmc: test/test_vs_llmc.c gpt2_nomain.o
	$(CC) $(CFLAGS) -w -I$(LLMC) test/test_vs_llmc.c gpt2_nomain.o -o $@ -lm $(LDFLAGS)

gen_ref: test/gen_ref.c
	$(CC) $(CFLAGS) -w -I$(LLMC) test/gen_ref.c -o $@ -lm $(LDFLAGS)

# Checks the assembly against llm.c's own C code, using the real GPT-2 weights:
# every kernel, the full forward pass, then the generated text byte for byte.
test: gpt2 test_vs_llmc gen_ref
	$(RUN) ./test_vs_llmc
	$(RUN) ./test_vs_llmc $(LLMC)/gpt2_124M.bin 1 $(FWD_T)
	$(RUN) ./gpt2 $(LLMC)/gpt2_124M.bin $(LLMC)/gpt2_tokenizer.bin $(GEN_N) | sed -n '/^---$$/,/^---$$/p' > out_asm.txt
	$(RUN) ./gen_ref $(LLMC)/gpt2_124M.bin $(LLMC)/gpt2_tokenizer.bin $(GEN_N) | grep -v num_activations | sed -n '/^---$$/,/^---$$/p' > out_ref.txt
	cmp out_asm.txt out_ref.txt && echo "generated text identical to llm.c"

clean:
	rm -f gpt2 gpt2_nomain.o test_vs_llmc gen_ref out_asm.txt out_ref.txt

.PHONY: test clean
