// AMX probe: checks the instruction semantics this project relies on and measures throughput.
// Build and run with ./probe.sh (native macOS on Apple silicon).
// No system headers: this is built without the macOS SDK (see build.sh), so the few
// libSystem functions used are declared here.
typedef unsigned long uint64_t;
typedef unsigned long uintptr_t;
int printf(const char*, ...);
struct timespec { long tv_sec; long tv_nsec; };
int clock_gettime(int, struct timespec*);
#define CLOCK_MONOTONIC 6                  // macOS value
typedef void* pthread_t;
int pthread_create(pthread_t*, const void*, void* (*)(void*), void*);
int pthread_join(pthread_t, void**);
int sysctlbyname(const char*, void*, unsigned long*, void*, unsigned long);
static float fabsf(float x) { return x < 0 ? -x : x; }

void amx_set(void), amx_clr(void);
void amx_ldx(uint64_t), amx_ldy(uint64_t), amx_ldz(uint64_t), amx_stz(uint64_t);
void amx_fma32(uint64_t), amx_fma16(uint64_t);
void amx_fma32_loop(long, uint64_t, uint64_t, uint64_t, uint64_t);
void amx_fma16_loop(long, uint64_t, uint64_t);
void amx_ldfma32_loop(long, const float*, const float*);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define PTR(p) ((uint64_t)(uintptr_t)(p))
#define FMA_X(o) ((uint64_t)(o))           // byte offset into X
#define FMA_Y(o) ((uint64_t)(o) << 10)     // byte offset into Y
#define FMA_Z(r) ((uint64_t)(r) << 20)     // Z row offset
#define SKIPZ    (1ull << 27)              // Z = X * Y instead of Z += X * Y
#define REG(r)   ((uint64_t)(r) << 56)     // register index for ldx/ldy/ldz/stz
#define PAIR     (1ull << 62)              // 128-byte load/store

static float X[32] __attribute__((aligned(128))), Y[32] __attribute__((aligned(128)));
static float Z[64][16] __attribute__((aligned(128)));
static int fails = 0;

static void dumpz(void) { for (int r = 0; r < 64; r++) amx_stz(PTR(Z[r]) | REG(r)); }
static void clearz(void) { float zero[16] __attribute__((aligned(64))) = {0}; for (int r = 0; r < 64; r++) amx_ldz(PTR(zero) | REG(r)); }

// expect Z[4*j + zoff][i] == f(i, j) for i, j < 16; report the first mismatch
static void check32(const char* name, int zoff, float (*f)(int, int)) {
    int bad = 0, first_i = -1, first_j = -1; float got = 0, want = 0;
    for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) {
        float w = f(i, j), g = Z[4 * j + zoff][i];
        if (fabsf(g - w) > 1e-3f * fabsf(w) + 1e-6f) { if (!bad++) { first_i = i; first_j = j; got = g; want = w; } }
    }
    if (bad) fails++;
    printf("%-46s %s", name, bad ? "FAIL" : "OK");
    if (bad) printf("  (%d wrong; first i=%d j=%d got %g want %g)", bad, first_i, first_j, got, want);
    printf("\n");
}
static float outer(int i, int j) { return X[i] * Y[j]; }
static float outer2(int i, int j) { return 2 * X[i] * Y[j]; }
static float outer_hi(int i, int j) { return X[16 + i] * Y[16 + j]; }

struct job { int kind; long n; double secs; };
static void* worker(void* a) {
    struct job* j = a;
    amx_set();
    double t0 = now();
    if (j->kind == 0) amx_fma32_loop(j->n, FMA_Z(0), FMA_Z(1), FMA_Z(2), FMA_Z(3));
    else if (j->kind == 1) amx_fma16_loop(j->n, FMA_Z(0), FMA_Z(1));
    else amx_ldfma32_loop(j->n, X, Y);
    j->secs = now() - t0;
    amx_clr();
    return 0;
}
static void throughput(const char* name, int kind, double macs_per_iter, int threads) {
    long n = kind == 2 ? 20000000 : 3000000;
    pthread_t t[8]; struct job jobs[8];
    for (int i = 0; i < threads; i++) { jobs[i] = (struct job){kind, n, 0}; pthread_create(&t[i], 0, worker, &jobs[i]); }
    double worst = 0;
    for (int i = 0; i < threads; i++) { pthread_join(t[i], 0); if (jobs[i].secs > worst) worst = jobs[i].secs; }
    printf("%-34s %d thread%s  %8.0f GFLOP/s\n", name, threads, threads > 1 ? "s" : " ", 2.0 * macs_per_iter * n * threads / worst / 1e9);
}

int main(void) {
    char cpu[128] = {0}; unsigned long len = sizeof cpu;
    if (sysctlbyname("machdep.cpu.brand_string", cpu, &len, 0, 0) == 0) printf("%s\n", cpu);
    for (int i = 0; i < 32; i++) { X[i] = i + 1; Y[i] = 0.5f * (i + 1); }
    printf("== semantics\n");
    amx_set();
    printf("%-46s OK\n", "amx set (AMX is enabled for this thread)");

    clearz();
    amx_ldx(PTR(X) | REG(0));
    amx_ldy(PTR(Y) | REG(0));
    amx_fma32(FMA_X(0) | FMA_Y(0) | FMA_Z(0) | SKIPZ);
    dumpz();
    check32("fma32: Z[4j][i] = X[i] * Y[j] (skip-Z bit 27)", 0, outer);
    amx_fma32(FMA_X(0) | FMA_Y(0) | FMA_Z(0));
    dumpz();
    check32("fma32 accumulate: Z += X * Y", 0, outer2);
    amx_fma32(FMA_X(0) | FMA_Y(0) | FMA_Z(1) | SKIPZ);
    dumpz();
    check32("fma32 with z offset 1 -> rows 4j + 1", 1, outer);

    clearz();
    amx_ldx(PTR(X) | REG(0) | PAIR);            // X0, X1 <- 128 bytes
    amx_ldy(PTR(Y) | REG(2) | PAIR);            // Y2, Y3 <- 128 bytes
    amx_fma32(FMA_X(64) | FMA_Y(3 * 64) | FMA_Z(2) | SKIPZ);
    dumpz();
    check32("pair loads + register offsets (X1, Y3)", 2, outer_hi);

    // fp16: Z[2j + zoff] holds 32 halves, Z[2j][i] = X[i] * Y[j] for i, j < 32
    _Float16 xh[32] __attribute__((aligned(64))), yh[32] __attribute__((aligned(64)));
    for (int i = 0; i < 32; i++) { xh[i] = (_Float16)(0.25f * (i + 1)); yh[i] = (_Float16)(0.125f * ((i % 7) + 1)); }
    clearz();
    amx_ldx(PTR(xh) | REG(0));
    amx_ldy(PTR(yh) | REG(0));
    amx_fma16(FMA_X(0) | FMA_Y(0) | FMA_Z(0) | SKIPZ);
    dumpz();
    int bad16 = 0;
    for (int j = 0; j < 32; j++) for (int i = 0; i < 32; i++) {
        _Float16 g = ((_Float16*)Z[2 * j])[i];
        float w = (float)xh[i] * (float)yh[j];
        if (fabsf((float)g - w) > 2e-3f * fabsf(w)) bad16++;
    }
    if (bad16) fails++;
    printf("%-46s %s\n", "fma16: Z[2j][i] = X[i] * Y[j] (32 x 32 halves)", bad16 ? "FAIL" : "OK");
    if (bad16) {
        printf("   fp16 rows 0-1, first 8 lanes:");
        for (int r = 0; r < 2; r++) for (int i = 0; i < 8; i++) printf(" %g", (float)((_Float16*)Z[r])[i]);
        printf("\n");
    }
    amx_clr();

    printf("== throughput (no loads; one AMX unit is shared by the 4 performance cores)\n");
    for (int t = 1; t <= 4; t *= 2) throughput("fma32 (16x16 fp32 per op)", 0, 16 * 256, t);
    for (int t = 1; t <= 4; t *= 2) throughput("fma16 (32x32 fp16 per op)", 1, 16 * 1024, t);
    printf("== throughput with loads (2 x 128-byte loads + 4 fma32 per k)\n");
    for (int t = 1; t <= 4; t *= 2) throughput("ldx/ldy + fma32 (32x32 block)", 2, 4 * 256, t);
    printf(fails ? "\n%d SEMANTICS CHECKS FAILED\n" : "\nALL SEMANTICS CHECKS PASSED\n", fails);
    return 0;
}
