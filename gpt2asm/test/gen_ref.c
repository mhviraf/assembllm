// Reference generator: llm.c's own sampling loop (B=1), to diff against the assembly binary.
// Usage: gen_ref checkpoint tokenizer num_tokens
#define TESTING
#include "train_gpt2.c"

unsigned int random_u32(uint64_t *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { return (random_u32(state) >> 8) / 16777216.0f; }
int sample_mult(float* p, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += p[i]; if (coin < cdf) return i; }
    return n - 1;
}

int main(int argc, char** argv) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, argv[1]);
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, argv[2]);
    int genT = atoi(argv[3]);
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(genT * sizeof(int));
    for (int i = 0; i < genT; i++) gen_tokens[i] = tokenizer.eot_token;
    printf("generating:\n---\n");
    for (int t = 1; t < genT; t++) {
        gpt2_forward(&model, gen_tokens, NULL, 1, genT);
        float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
        float coin = random_f32(&rng_state);
        int next_token = sample_mult(probs, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;
        safe_printf(tokenizer_decode(&tokenizer, next_token));
        fflush(stdout);
    }
    printf("\n---\n");
    return 0;
}
