// Compares the assembly kernels and full forward pass against llm.c's C code.
// Build: cc -O2 -I<llm.c> test_vs_llmc.c gpt2_forward.S (assembled with -DGPT2A_NO_MAIN) -lm
// Usage: test_vs_llmc [checkpoint B T]
#define TESTING
#include "train_gpt2.c"

void gpt2a_build_from_checkpoint(const char*);
void gpt2a_alloc_acts(long, long);
float* gpt2a_forward(const int*, long, long);
void a_encoder_forward(float*, const int*, const float*, const float*, long, long, long);
void a_layernorm_forward(float*, const float*, const float*, const float*, long, long);
void a_matmul_forward(float*, const float*, const float*, const float*, long, long, long);
void a_attention_forward(float*, const float*, float*, long, long, long, long);
void a_gelu_forward(float*, const float*, long);
void a_residual_forward(float*, const float*, const float*, long);
void a_softmax_forward(float*, const float*, long, long, long);
unsigned a_random_u32(uint64_t*);
float a_random_f32(uint64_t*);
int a_sample_mult(const float*, int, float);
void a_pack_weights(float*, const float*, long, long);
void a_matmul_packed(float*, const float*, const float*, const float*, long, long, long, long);

// llm.c's sampler (verbatim), which train_gpt2.c compiles out under TESTING
unsigned int random_u32(uint64_t *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { return (random_u32(state) >> 8) / 16777216.0f; }
int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) { return i; }
    }
    return n - 1;
}

static uint64_t seed = 42;
static int failures = 0;
static float* rnd(size_t n, float scale) {
    float* a = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) a[i] = (random_f32(&seed) * 2.0f - 1.0f) * scale;
    return a;
}
static float* zeros(size_t n) { return calloc(n, sizeof(float)); }
// max |a-b| / (1e-6 + max|b|) style check, reported per test
static void check(const char* name, const float* a, const float* b, size_t n, double tol) {
    double maxd = 0, maxb = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > maxd) maxd = d;
        if (fabs(b[i]) > maxb) maxb = fabs(b[i]);
    }
    double rel = maxd / (maxb + 1e-12);
    int ok = rel <= tol;
    if (!ok) failures++;
    printf("%-44s max|diff| %.3e  rel %.3e  %s\n", name, maxd, rel, ok ? "OK" : "FAIL");
}

static void test_kernels(void) {
    // matmul, including shapes that hit every leftover path
    int shapes[][3] = {{1,4,4},{7,12,10},{8,64,36},{5,20,9},{3,7,5},{13,100,33},{4,768,8},{9,3,1}};
    for (int s = 0; s < 8; s++) {
        int N = shapes[s][0], C = shapes[s][1], OC = shapes[s][2];
        for (int wb = 0; wb < 2; wb++) {
            float *inp = rnd(N*C, 1), *w = rnd(OC*C, 1), *b = wb ? rnd(OC, 1) : NULL;
            float *o1 = zeros(N*OC), *o2 = zeros(N*OC);
            matmul_forward(o1, inp, w, b, 1, N, C, OC);
            a_matmul_forward(o2, inp, w, b, N, C, OC);
            char nm[64]; snprintf(nm, 64, "matmul N=%d C=%d OC=%d bias=%d", N, C, OC, wb);
            check(nm, o2, o1, N*OC, 1e-5);
            free(inp); free(w); free(b); free(o1); free(o2);
        }
    }
    // packed 5x16 matmul (weights repacked into panels of 16 channels), incl. partial row blocks
    int pk[][3] = {{64,768,64},{13,64,48},{1,8,16},{9,12,16},{8,3072,16},{7,4,32},{63,768,48},{5,16,16},{6,8,32}};
    for (int s = 0; s < 9; s++) {
        int N = pk[s][0], C = pk[s][1], OC = pk[s][2];
        for (int wb = 0; wb < 2; wb++) {
            float *inp = rnd(N*C, 1), *w = rnd(OC*C, 1), *b = wb ? rnd(OC, 1) : NULL, *wp = zeros(OC*C);
            float *o1 = zeros(N*OC), *o2 = zeros(N*OC);
            a_pack_weights(wp, w, OC, C);
            matmul_forward(o1, inp, w, b, 1, N, C, OC);
            a_matmul_packed(o2, inp, wp, b, N, C, OC, OC);
            char nm[64]; snprintf(nm, 64, "packed matmul N=%d C=%d OC=%d bias=%d", N, C, OC, wb);
            check(nm, o2, o1, N*OC, 1e-5);
        }
    }
    // layernorm
    int ln[][2] = {{5,7},{3,64},{2,768}};
    for (int s = 0; s < 3; s++) {
        int N = ln[s][0], C = ln[s][1];
        float *x = rnd(N*C, 3), *w = rnd(C, 1), *b = rnd(C, 1);
        float *o1 = zeros(N*C), *o2 = zeros(N*C), *m = zeros(N), *r = zeros(N);
        layernorm_forward(o1, m, r, x, w, b, 1, N, C);
        a_layernorm_forward(o2, x, w, b, N, C);
        char nm[64]; snprintf(nm, 64, "layernorm N=%d C=%d", N, C);
        check(nm, o2, o1, N*C, 1e-5);
    }
    // attention (hs=8 vector path, hs=6 scalar tail, hs=64 like GPT-2)
    int at[][4] = {{2,5,24,3},{1,7,12,2},{1,9,128,2}};
    for (int s = 0; s < 3; s++) {
        int B = at[s][0], T = at[s][1], C = at[s][2], NH = at[s][3];
        float *qkv = rnd(B*T*3*C, 2), *o1 = zeros(B*T*C), *o2 = zeros(B*T*C);
        float *pre = zeros(B*NH*T*T), *att = zeros(B*NH*T*T), *scr = zeros(T);
        attention_forward(o1, pre, att, qkv, B, T, C, NH);
        a_attention_forward(o2, qkv, scr, B, T, C, NH);
        char nm[64]; snprintf(nm, 64, "attention B=%d T=%d C=%d NH=%d", B, T, C, NH);
        check(nm, o2, o1, B*T*C, 1e-5);
    }
    // gelu
    { int N = 1001; float *x = rnd(N, 6), *o1 = zeros(N), *o2 = zeros(N);
      gelu_forward(o1, x, N); a_gelu_forward(o2, x, N); check("gelu N=1001 (vector tanh)", o2, o1, N, 1e-6); }
    // residual
    { int N = 37; float *a = rnd(N, 1), *b = rnd(N, 1), *o1 = zeros(N), *o2 = zeros(N);
      residual_forward(o1, a, b, N); a_residual_forward(o2, a, b, N); check("residual N=37", o2, o1, N, 0); }
    // softmax with padding
    { int N = 3, V = 37, Vp = 40; float *l = rnd(N*Vp, 5), *o1 = zeros(N*Vp), *o2 = rnd(N*Vp, 1);
      softmax_forward(o1, l, 1, N, V, Vp); a_softmax_forward(o2, l, N, V, Vp); check("softmax N=3 V=37 Vp=40 (vector exp)", o2, o1, N*Vp, 1e-6); }
    // encoder
    { int B = 2, T = 3, C = 10, V = 5; int inp[6] = {4,0,2,3,3,1};
      float *wte = rnd(V*C, 1), *wpe = rnd(T*C, 1), *o1 = zeros(B*T*C), *o2 = zeros(B*T*C);
      encoder_forward(o1, inp, wte, wpe, B, T, C); a_encoder_forward(o2, inp, wte, wpe, B, T, C);
      check("encoder B=2 T=3 C=10", o2, o1, B*T*C, 0); }
    // rng + sampler must match bit for bit
    { uint64_t s1 = 1337, s2 = 1337; int bad = 0;
      for (int i = 0; i < 10000; i++) {
          if (random_u32(&s1) != a_random_u32(&s2)) bad++;
          if (random_f32(&s1) != a_random_f32(&s2)) bad++;
      }
      float p[5] = {0.1f, 0.2f, 0.3f, 0.25f, 0.15f};
      for (int i = 0; i <= 100; i++) {
          float coin = i / 100.0f * 0.9999f;
          if (sample_mult(p, 5, coin) != a_sample_mult(p, 5, coin)) bad++;
      }
      if (bad) failures++;
      printf("%-44s mismatches %d  %s\n", "rng + sample_mult", bad, bad ? "FAIL" : "OK"); }
}

static void test_forward(const char* ckpt, int B, int T) {
    GPT2 model;
    gpt2_build_from_checkpoint(&model, ckpt);
    gpt2a_build_from_checkpoint(ckpt);
    int V = model.config.vocab_size, Vp = model.config.padded_vocab_size;
    int* tokens = malloc(B*T*sizeof(int));
    for (int i = 0; i < B*T; i++) tokens[i] = random_u32(&seed) % V;
    clock_t c0 = clock();
    gpt2_forward(&model, tokens, NULL, B, T);
    clock_t c1 = clock();
    gpt2a_alloc_acts(B, T);
    float* pa = gpt2a_forward(tokens, B, T);
    clock_t c2 = clock();
    float* pc = model.acts.probs;
    char nm[96]; snprintf(nm, 96, "full forward probs B=%d T=%d", B, T);
    check(nm, pa, pc, (size_t)B*T*Vp, 1e-4);
    // per-row: argmax agreement and KL divergence
    int argmax_bad = 0; double maxkl = 0;
    for (int r = 0; r < B*T; r++) {
        const float *a = pa + (size_t)r*Vp, *c = pc + (size_t)r*Vp;
        int ia = 0, ic = 0; double kl = 0;   // kl holds the total variation distance
        for (int i = 0; i < V; i++) {
            if (a[i] > a[ia]) ia = i;
            if (c[i] > c[ic]) ic = i;
            kl += 0.5 * fabs((double)c[i] - a[i]);
        }
        // a different top token only counts if it is not a near-tie in the C result
        if (ia != ic && c[ia] < c[ic] * (1.0f - 1e-4f)) argmax_bad++;
        if (kl > maxkl) maxkl = kl;
    }
    int bad = argmax_bad || maxkl > 1e-4;
    if (bad) failures++;
    printf("%-44s top-token mismatches %d/%d, max TV distance %.2e  %s\n", "  per-position check", argmax_bad, B*T, maxkl,
           bad ? "FAIL" : "OK");
    printf("  time: C reference %.2fs, assembly %.2fs (emulated)\n",
           (double)(c1-c0)/CLOCKS_PER_SEC, (double)(c2-c1)/CLOCKS_PER_SEC);
}

int main(int argc, char** argv) {
    if (argc < 4) test_kernels();
    else test_forward(argv[1], atoi(argv[2]), atoi(argv[3]));
    printf(failures ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
    return failures != 0;
}
