// Speed benchmark: llm.c's C forward pass vs the ARM64 assembly forward pass, same CPU, same inputs.
//
// Runs num_samples sequences of T tokens from an llm.c token file (e.g. tiny_shakespeare_val.bin)
// through both implementations, one forward pass per sample (B=1, T). The two are interleaved and
// alternate who goes first, so warm-up and thermal throttling (a fanless MacBook Air) hit both equally.
// Only the forward call itself is timed; loss and agreement checks happen outside the timers.
//
// Usage: bench gpt2_124M.bin tiny_shakespeare_val.bin [num_samples=1000] [T=64] [first_sample=0]
// (first_sample lets a long run be split into chunks; each run ends with a RAW line to add up)
#define TESTING
#include "train_gpt2.c"

void gpt2a_build_from_checkpoint(const char*);
void gpt2a_alloc_acts(long, long);
float* gpt2a_forward(const int*, long, long);

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s checkpoint tokens.bin [num_samples] [T]\n", argv[0]); return 1; }
    const char* ckpt = argv[1];
    int N = argc > 3 ? atoi(argv[3]) : 1000;
    int T = argc > 4 ? atoi(argv[4]) : 64;
    int S0 = argc > 5 ? atoi(argv[5]) : 0;

    // llm.c token file: 256 x int32 header (magic 20240520, version 1, count), then uint16 tokens
    FILE* f = fopenCheck(argv[2], "rb");
    int header[256];
    freadCheck(header, sizeof(int), 256, f);
    if (header[0] != 20240520 || header[1] != 1) { printf("bad token file\n"); return 1; }
    long ntok = header[2];
    uint16_t* raw = mallocCheck(ntok * sizeof(uint16_t));
    freadCheck(raw, sizeof(uint16_t), ntok, f);
    fcloseCheck(f);
    if (ntok < T + 1) { printf("token file too small\n"); return 1; }
    int* toks = mallocCheck(ntok * sizeof(int));
    for (long i = 0; i < ntok; i++) toks[i] = raw[i];
    long nwin = (ntok - 1) / T;  // non-overlapping windows; wraps if N > nwin

    GPT2 model;
    gpt2_build_from_checkpoint(&model, ckpt);
    gpt2a_build_from_checkpoint(ckpt);
    gpt2a_alloc_acts(1, T);
    size_t Vp = model.config.padded_vocab_size;
    int V = model.config.vocab_size;
#ifdef OMP
    printf("llm.c built WITH OpenMP (%d threads)\n", omp_get_max_threads());
#else
    printf("llm.c built single-threaded; assembly is single-threaded\n");
#endif
    printf("benchmark: %d samples x %d tokens (%ld distinct windows in file)\n\n", N, T, nwin);

    // warm-up (also triggers llm.c's lazy activation allocation)
    gpt2_forward(&model, toks, NULL, 1, T);
    gpt2a_forward(toks, 1, T);

    double t_c = 0, t_a = 0, loss_c = 0, loss_a = 0, maxdiff = 0;
    int top_mismatch = 0;
    for (int s = 0; s < N; s++) {
        int* in = toks + ((S0 + s) % nwin) * T;
        int* tg = in + 1;
        float* pa = NULL;
        for (int k = 0; k < 2; k++) {
            int which = (S0 + s + k) % 2;  // alternate which implementation runs first
            double t0 = now();
            if (which == 0) gpt2_forward(&model, in, NULL, 1, T);
            else pa = gpt2a_forward(in, 1, T);
            double dt = now() - t0;
            if (which == 0) t_c += dt; else t_a += dt;
        }
        // untimed: loss for each, and how closely they agree
        float* pc = model.acts.probs;
        for (int t = 0; t < T; t++) {
            loss_c -= log(pc[t * Vp + tg[t]]);
            loss_a -= log(pa[t * Vp + tg[t]]);
            int ic = 0, ia = 0;
            for (int i = 0; i < V; i++) {
                double d = fabs(pc[t * Vp + i] - pa[t * Vp + i]);
                if (d > maxdiff) maxdiff = d;
                if (pc[t * Vp + i] > pc[t * Vp + ic]) ic = i;
                if (pa[t * Vp + i] > pa[t * Vp + ia]) ia = i;
            }
            if (ic != ia) top_mismatch++;
        }
        if ((s + 1) % 50 == 0 || s + 1 == N) {
            double toks_done = (double)(s + 1) * T;
            printf("[%4d/%d] llm.c %7.1f tok/s | asm %7.1f tok/s | loss %.4f vs %.4f\n", s + 1, N,
                   toks_done / t_c, toks_done / t_a, loss_c / toks_done, loss_a / toks_done);
            fflush(stdout);
        }
    }
    double ntoks = (double)N * T;
    printf("\n=== results: %d samples x %d tokens, GPT-2 forward pass (%zu params) ===\n", N, T, model.num_parameters);
    printf("%-22s %10s %12s %12s %10s\n", "", "total s", "ms/sample", "tokens/s", "mean loss");
    printf("%-22s %10.2f %12.2f %12.1f %10.4f\n", "llm.c (C, -Ofast)", t_c, 1000 * t_c / N, ntoks / t_c, loss_c / ntoks);
    printf("%-22s %10.2f %12.2f %12.1f %10.4f\n", "assembly (ARM64)", t_a, 1000 * t_a / N, ntoks / t_a, loss_a / ntoks);
    printf("assembly is %.2fx the speed of llm.c\n", t_c / t_a);
    printf("agreement: max |prob diff| %.2e, top-token mismatches %d of %.0f positions\n", maxdiff, top_mismatch, ntoks);
    printf("RAW %d %d %.6f %.6f %.6f %.6f %.3e %d\n", N, T, t_c, t_a, loss_c, loss_a, maxdiff, top_mismatch);
    return 0;
}
