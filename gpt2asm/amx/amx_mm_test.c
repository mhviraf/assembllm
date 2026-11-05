// Correctness and speed of the AMX matmul kernels (native macOS). Built by build.sh without the SDK.
// Reference: the NEON fp32 kernel from gpt2_forward.S, already checked against llm.c.
// fp32 kernels: a_matmul_amx32, a_matmul_amx16, a_matmul_amxr0; fp16 kernel: a_matmul_amxh
// (fp16 inputs and weights packed by a_pack_f16i, fp32 sums). All from gpt2_forward.S.
// The fp16 kernel's reference is the fp32 kernel on the same fp16-rounded inputs and weights,
// so only the order of the fp32 sums may differ.
typedef unsigned long size_t;
typedef unsigned short u16;
int printf(const char*, ...);
int fflush(void*);
void* malloc(size_t);
struct timespec { long tv_sec; long tv_nsec; };
int clock_gettime(int, struct timespec*);
typedef void* pthread_t;
int pthread_create(pthread_t*, const void*, void* (*)(void*), void*);
int pthread_join(pthread_t, void**);

void a_matmul_forward(float*, const float*, const float*, const float*, long, long, long);
void a_pack_weights(float*, const float*, long, long);
void a_transpose64(float*, const float*, long, long);
void a_transpose64_h(u16*, const float*, long, long);
void a_pack_f16i(u16*, const float*, long, long);
typedef void (*mmfn)(float*, const void*, const void*, const float*, long, long, long, long);
void a_matmul_amx32(float*, const void*, const void*, const float*, long, long, long, long);
void a_matmul_amx16(float*, const void*, const void*, const float*, long, long, long, long);
void a_matmul_amxr0(float*, const void*, const void*, const float*, long, long, long, long);
void a_matmul_amxh(float*, const void*, const void*, const float*, long, long, long, long);
static mmfn K[] = {a_matmul_amx32, a_matmul_amx16, a_matmul_amxr0, a_matmul_amxh};
static const char* KN[] = {"amx32", "amx16", "amxr0", "amxh"};
#define NK 4
#define FP16 3
static int kok[NK] = {1, 1, 1, 1};

static double now(void) { struct timespec t; clock_gettime(6, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static unsigned long seed = 1;
static float rnd(void) { seed = seed * 6364136223846793005ul + 1442695040888963407ul; return (float)((seed >> 40) & 0xffff) / 32768.0f - 1.0f; }
static void* al(long bytes) { char* a = malloc(bytes + 256); return (void*)(((unsigned long)a + 127) & ~127ul); }
static float* arr(long n, int fill) { float* a = al(n * 4); if (fill) for (long i = 0; i < n; i++) a[i] = rnd(); return a; }
static float fabsf_(float x) { return x < 0 ? -x : x; }
static float r16(float x) { return (float)(_Float16)x; }

// pack W (OC x C) for kernel k: fp32 16-channel panels, or fp16 32-channel panels in lane order
static void* pack(int k, const float* w, long OC, long C) {
    if (k < FP16) { float* wp = arr(OC * C, 0); a_pack_weights(wp, w, OC, C); return wp; }
    u16* wp = al(OC * C * 2);
    a_pack_f16i(wp, w, OC, C);
    return wp;
}
static void* trans(int k, const float* x, long n, long C) {
    if (k < FP16) { float* t = arr(64 * C, 0); a_transpose64(t, x, n, C); return t; }
    u16* t = al(64 * C * 2); a_transpose64_h(t, x, n, C); return t;
}

static void check(int k, long n, long C, long OC, int bias) {
    if (k >= FP16 && OC % 32) return;
    float *x = arr(n * C, 1), *w = arr(OC * C, 1), *b = bias ? arr(OC, 1) : 0;
    float *o1 = arr(n * OC, 0), *o2 = arr(n * OC, 1);
    if (k >= FP16) { for (long i = 0; i < n * C; i++) x[i] = r16(x[i]); for (long i = 0; i < OC * C; i++) w[i] = r16(w[i]); }
    void* wp = pack(k, w, OC, C);
    void* xT = trans(k, x, n, C);
    a_matmul_forward(o1, x, w, b, n, C, OC);
    K[k](o2, xT, wp, b, n, C, OC, OC);
    float md = 0, mb = 0; long at = 0;
    for (long i = 0; i < n * OC; i++) {
        float d = fabsf_(o1[i] - o2[i]);
        if (!(d <= md)) { md = d; at = i; }
        if (fabsf_(o1[i]) > mb) mb = fabsf_(o1[i]);
    }
    int ok = md <= 1e-5f * mb + 1e-6f;
    if (!ok) kok[k] = 0;
    printf("%-6s n=%-3ld C=%-5ld OC=%-5ld bias=%d  max|diff| %.2e (max|out| %.1f)  %s", KN[k], n, C, OC, bias, md, mb, ok ? "OK" : "FAIL");
    if (!ok) printf("  worst at row %ld channel %ld", at / OC, at % OC);
    printf("\n");
    fflush(0);
}

struct job { mmfn f; float* out; void *xT, *wp; float* b; long n, C, OC, o0, o1, reps, half; };
static void* runj(void* a) {
    struct job* j = a;
    long eb = j->half ? 2 : 4;
    for (long r = 0; r < j->reps; r++)
        j->f(j->out + j->o0, j->xT, (char*)j->wp + j->o0 * j->C * eb, j->b + j->o0, j->n, j->C, j->o1 - j->o0, j->OC);
    return 0;
}
static void speed(int k, const char* name, long C, long OC, int threads) {
    long n = 64;
    float *x = arr(n * C, 1), *w = arr(OC * C, 1), *b = arr(OC, 1), *o = arr(n * OC, 0);
    void* wp = pack(k, w, OC, C);
    void* xT = trans(k, x, n, C);
    long reps = OC > 10000 ? 4 : 40;
    struct job jobs[8]; pthread_t t[8];
    long per = (OC / 32 + threads - 1) / threads * 32;
    for (int i = 0; i < threads; i++) {
        long o0 = i * per, o1 = o0 + per > OC ? OC : o0 + per;
        jobs[i] = (struct job){K[k], o, xT, wp, b, n, C, OC, o0, o1, 1, k >= FP16};
    }
    for (int i = 0; i < threads; i++) pthread_create(&t[i], 0, runj, &jobs[i]);   // warm-up
    for (int i = 0; i < threads; i++) pthread_join(t[i], 0);
    double best = 1e9;
    for (int rep = 0; rep < 3; rep++) {
        for (int i = 0; i < threads; i++) jobs[i].reps = reps;
        double t0 = now();
        for (int i = 0; i < threads; i++) pthread_create(&t[i], 0, runj, &jobs[i]);
        for (int i = 0; i < threads; i++) pthread_join(t[i], 0);
        double s = (now() - t0) / reps;
        if (s < best) best = s;
    }
    printf("%-6s %-8s 64 x %-5ld x %-5ld  %d thread%s  %7.3f ms  %6.0f GFLOP/s\n", KN[k], name, C, OC, threads, threads > 1 ? "s" : " ", best * 1e3, 2.0 * n * C * OC / best / 1e9);
    fflush(0);
}

int main(void) {
    printf("== correctness\n");
    for (int k = 0; k < NK; k++) {
        check(k, 64, 768, 64, 1);
        check(k, 64, 768, 64, 0);
        check(k, 37, 768, 48, 1);
        check(k, 1, 16, 16, 1);
        check(k, 64, 3072, 32, 1);
        check(k, 13, 64, 16, 0);
        check(k, 64, 768, 96, 1);
        check(k, 50, 128, 2304, 1);
    }
    printf("== speed (GPT-2 124M shapes, 64 tokens; best of 3)\n");
    const char* nm[] = {"qkv", "attproj", "fc", "fcproj", "logits"};
    long sh[][2] = {{768, 2304}, {768, 768}, {768, 3072}, {3072, 768}, {768, 50304}};
    for (int k = 0; k < NK; k++) {
        if (!kok[k]) { printf("(%s is wrong here, so not timed)\n", KN[k]); continue; }
        for (int t = 1; t <= 4; t *= 2) for (int s = 0; s < 5; s++) speed(k, nm[s], sh[s][0], sh[s][1], t);
    }
    printf("\nPASSED:");
    for (int k = 0; k < NK; k++) if (kok[k]) printf(" %s", KN[k]);
    printf("\nFAILED:");
    for (int k = 0; k < NK; k++) if (!kok[k]) printf(" %s", KN[k]);
    printf("\n");
    return 0;
}
