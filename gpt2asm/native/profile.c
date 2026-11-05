// Native macOS timing runner for the assembly forward pass (no llm.c, no system headers).
// Usage: gpt2_profile <checkpoint> <llm.c token file> [threads] [passes]
// Env: GPT2A_PRECISION, GPT2A_CALIB, GPT2A_PROFILE (per-stage breakdown).
typedef unsigned long size_t;
typedef struct FILE FILE;
FILE* fopen(const char*, const char*);
size_t fread(void*, size_t, size_t, FILE*);
int printf(const char*, ...);
void* malloc(size_t);
int atoi(const char*);
struct timespec { long tv_sec; long tv_nsec; };
int clock_gettime(int, struct timespec*);
void gpt2a_build_from_checkpoint(const char*);
void gpt2a_alloc_acts(long, long);
float* gpt2a_forward(const int*, long, long);
void gpt2a_set_threads(long);
long gpt2a_get_threads(void);
void gpt2a_profile_print(void);
static double now(void) { struct timespec t; clock_gettime(6, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s checkpoint tokens.bin [threads] [passes]\n", argv[0]); return 1; }
    FILE* f = fopen(argv[2], "rb");
    if (!f) { printf("cannot open %s\n", argv[2]); return 1; }
    int h[256]; fread(h, 4, 256, f);
    int n = h[2];
    unsigned short* raw = malloc(n * 2); fread(raw, 2, n, f);
    int* tok = malloc(n * 4); for (int i = 0; i < n; i++) tok[i] = raw[i];
    if (argc > 3) gpt2a_set_threads(atoi(argv[3]));
    int passes = argc > 4 ? atoi(argv[4]) : 50;
    gpt2a_build_from_checkpoint(argv[1]);
    gpt2a_alloc_acts(1, 64);
    long windows = (n - 1) / 64;
    gpt2a_forward(tok, 1, 64);
    double t0 = now();
    for (int s = 0; s < passes; s++) gpt2a_forward(tok + (s % windows) * 64, 1, 64);
    double ms = (now() - t0) / passes * 1e3;
    printf("threads %ld: %.2f ms per forward pass of 64 tokens (%.0f tokens/s)\n", gpt2a_get_threads(), ms, 64000.0 / ms);
    gpt2a_profile_print();
    return 0;
}
