// Correctness and speed of the AMX fp32 matmul (native macOS). Built by build.sh without the SDK.
// Reference: the NEON fp32 kernel from gpt2_forward.S, already checked against llm.c.
typedef unsigned long size_t;
int printf(const char*, ...);
void* malloc(size_t);
struct timespec { long tv_sec; long tv_nsec; };
int clock_gettime(int, struct timespec*);
typedef void* pthread_t;
int pthread_create(pthread_t*, const void*, void* (*)(void*), void*);
int pthread_join(pthread_t, void**);

void a_matmul_forward(float*, const float*, const float*, const float*, long, long, long);
void a_pack_weights(float*, const float*, long, long);
void a_transpose64(float*, const float*, long, long);
void a_matmul_amx(float*, const float*, const float*, const float*, long, long, long, long);

static double now(void) { struct timespec t; clock_gettime(6, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static unsigned long seed = 1;
static float rnd(void) { seed = seed * 6364136223846793005ul + 1442695040888963407ul; return (float)((seed >> 40) & 0xffff) / 32768.0f - 1.0f; }
static float* arr(long n, int fill) { float* a = malloc(n * 4 + 256); a = (float*)(((unsigned long)a + 127) & ~127ul); if (fill) for (long i = 0; i < n; i++) a[i] = rnd(); return a; }
static float fabsf_(float x) { return x < 0 ? -x : x; }
static int fails = 0;

static void check(long n, long C, long OC, int bias) {
    float *x = arr(n * C, 1), *w = arr(OC * C, 1), *b = bias ? arr(OC, 1) : 0;
    float *wp = arr(OC * C, 0), *xT = arr(64 * C, 0), *o1 = arr(n * OC, 0), *o2 = arr(n * OC, 1);
    a_pack_weights(wp, w, OC, C);
    a_transpose64(xT, x, n, C);
    a_matmul_forward(o1, x, w, b, n, C, OC);
    a_matmul_amx(o2, xT, wp, b, n, C, OC, OC);
    float md = 0, mb = 0;
    for (long i = 0; i < n * OC; i++) { float d = fabsf_(o1[i] - o2[i]); if (d > md) md = d; if (fabsf_(o1[i]) > mb) mb = fabsf_(o1[i]); }
    int ok = md <= 1e-5f * mb + 1e-6f;
    if (!ok) fails++;
    printf("amx matmul n=%-3ld C=%-5ld OC=%-5ld bias=%d  max|diff| %.2e (max|out| %.1f)  %s\n", n, C, OC, bias, md, mb, ok ? "OK" : "FAIL");
    if (!ok) printf("   first outputs: ref %g %g %g, amx %g %g %g\n", o1[0], o1[1], o1[OC], o2[0], o2[1], o2[OC]);
}

struct job { float *out, *xT, *wp, *b; long n, C, OC, o0, o1, reps; };
static void* run(void* a) {
    struct job* j = a;
    for (long r = 0; r < j->reps; r++)
        a_matmul_amx(j->out + j->o0, j->xT, j->wp + j->o0 * j->C, j->b + j->o0, j->n, j->C, j->o1 - j->o0, j->OC);
    return 0;
}
static void speed(const char* name, long C, long OC, int threads) {
    long n = 64;
    float *x = arr(n * C, 1), *w = arr(OC * C, 1), *b = arr(OC, 1), *wp = arr(OC * C, 0), *xT = arr(64 * C, 0), *o = arr(n * OC, 0);
    a_pack_weights(wp, w, OC, C);
    a_transpose64(xT, x, n, C);
    long reps = OC > 10000 ? 4 : 40;
    struct job jobs[8]; pthread_t t[8];
    long per = (OC / 16 + threads - 1) / threads * 16;
    for (int i = 0; i < threads; i++) {
        long o0 = i * per, o1 = o0 + per > OC ? OC : o0 + per;
        jobs[i] = (struct job){o, xT, wp, b, n, C, OC, o0, o1, 1};
    }
    for (int i = 0; i < threads; i++) pthread_create(&t[i], 0, run, &jobs[i]);   // warm-up
    for (int i = 0; i < threads; i++) pthread_join(t[i], 0);
    for (int i = 0; i < threads; i++) jobs[i].reps = reps;
    double t0 = now();
    for (int i = 0; i < threads; i++) pthread_create(&t[i], 0, run, &jobs[i]);
    for (int i = 0; i < threads; i++) pthread_join(t[i], 0);
    double s = (now() - t0) / reps;
    printf("%-8s 64 x %-5ld x %-5ld  %d thread%s  %7.3f ms  %6.0f GFLOP/s\n", name, C, OC, threads, threads > 1 ? "s" : " ", s * 1e3, 2.0 * n * C * OC / s / 1e9);
}

int main(void) {
    printf("== correctness vs the NEON fp32 kernel\n");
    check(64, 768, 64, 1);
    check(64, 768, 64, 0);
    check(37, 768, 48, 1);
    check(1, 16, 16, 1);
    check(64, 3072, 32, 1);
    check(13, 64, 16, 0);
    printf("== speed (GPT-2 124M shapes, 64 tokens)\n");
    const char* nm[] = {"qkv", "attproj", "fc", "fcproj", "logits"};
    long sh[][2] = {{768, 2304}, {768, 768}, {768, 3072}, {3072, 768}, {768, 50304}};
    for (int t = 1; t <= 4; t *= 2) for (int s = 0; s < 5; s++) speed(nm[s], sh[s][0], sh[s][1], t);
    printf(fails ? "\n%d CORRECTNESS CHECKS FAILED\n" : "\nALL CORRECTNESS CHECKS PASSED\n", fails);
    return 0;
}
