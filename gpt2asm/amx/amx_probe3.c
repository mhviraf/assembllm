// AMX probe, round 3 (native macOS on Apple silicon). Built by build.sh without the SDK.
// Round 2 died with a bus error before its buffered output reached the file, so now every
// case runs in its own forked process and output is flushed line by line: a crash costs
// one line ("^ crashed"), not the run.
// Reads registers back through stx / sty / stz (store) as well as through fma32 with
// X0 / Y0 (verified in round 1), so each fact is seen two independent ways.
typedef unsigned long uint64_t;
typedef unsigned long uintptr_t;
int printf(const char*, ...);
int fflush(void*);
void exit(int);
int fork(void);
int waitpid(int, int*, int);
struct timespec { long tv_sec; long tv_nsec; };
int clock_gettime(int, struct timespec*);
typedef void* pthread_t;
int pthread_create(pthread_t*, const void*, void* (*)(void*), void*);
int pthread_join(pthread_t, void**);
int sysctlbyname(const char*, void*, unsigned long*, void*, unsigned long);

void amx_set(void), amx_clr(void);
void amx_ldx(uint64_t), amx_ldy(uint64_t), amx_stx(uint64_t), amx_sty(uint64_t);
void amx_ldz(uint64_t), amx_stz(uint64_t), amx_fma32(uint64_t), amx_fma16(uint64_t);
void loop_f16(long, uint64_t, uint64_t);
void loop_h32(long, const void*, const void*, uint64_t);
void loop_v2(long, const void*, const void*);
void loop_pair(long, const void*, const void*, uint64_t);

#define PTR(p)   ((uint64_t)(uintptr_t)(p))
#define REG(r)   ((uint64_t)(r) << 56)
#define FMA_Y(o) ((uint64_t)(o) << 10)
#define FMA_Z(r) ((uint64_t)(r) << 20)
#define SKIPZ    (1ull << 27)
#define B62      (1ull << 62)
#define SENT     (-7.0f)

static double now(void) { struct timespec t; clock_gettime(6, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
typedef void (*amxfn)(uint64_t);

static float src[512] __attribute__((aligned(256)));
static float buf[8][16] __attribute__((aligned(256)));
static float tmp[64] __attribute__((aligned(256)));
static float ones[16] __attribute__((aligned(256)));
static float Zd[64][16] __attribute__((aligned(256)));

static void run(void (*f)(int), int arg) {
    fflush(0);
    int pid = fork();
    if (pid == 0) { amx_set(); f(arg); fflush(0); amx_clr(); exit(0); }
    int st = 0;
    waitpid(pid, &st, 0);
    if (st & 0x7f) printf("   ^ crashed (signal %d)\n", st & 0x7f);
    fflush(0);
}

static void fill_regs(amxfn ld, float v) {
    for (int l = 0; l < 16; l++) tmp[l] = v;
    for (int r = 0; r < 8; r++) ld(PTR(tmp) | REG(r));
}
static void fill_regs_id(amxfn ld) {                     // reg r lane l = 1 + 16 r + l
    for (int r = 0; r < 8; r++) {
        for (int l = 0; l < 16; l++) tmp[l] = 1 + 16 * r + l;
        ld(PTR(tmp) | REG(r));
    }
}
static void fill_z(float v) { for (int l = 0; l < 16; l++) tmp[l] = v; for (int q = 0; q < 64; q++) amx_ldz(PTR(tmp) | REG(q)); }
static void dump_z(void) { for (int q = 0; q < 64; q++) amx_stz(PTR(Zd[q]) | REG(q)); }

// value of the 1 + 16 r + l pattern -> "Xr[l]"
static void pr_src(const char* xy, float v) {
    int k = (int)v - 1;
    if (v == SENT) printf(" -");
    else if (v != (float)(k + 1) || k < 0 || k >= 128) printf(" ?%g", v);
    else printf(" %s%d[%d]", xy, k / 16, k % 16);
}

// --- 1. single loads with a register index, read back two ways
static void c_regs(int isy) {
    amxfn ld = isy ? amx_ldy : amx_ldx, st = isy ? amx_sty : amx_stx;
    const char* n = isy ? "Y" : "X";
    fill_regs_id(ld);                                    // reg r <- pattern r (index in bits 56+)
    printf("%s: ld%s r -> st%s r, lane 0 of each:", n, isy ? "y" : "x", isy ? "y" : "x");
    for (int r = 0; r < 8; r++) { for (int l = 0; l < 16; l++) buf[r][l] = SENT; st(PTR(buf[r]) | REG(r)); pr_src(n, buf[r][0]); }
    printf("\n");
    fflush(0);
    // reg 0 via fma32 (X0 x Y0 verified): with the other operand all ones, Z row 0 = that register
    if (isy) { amx_ldx(PTR(ones)); } else { amx_ldy(PTR(ones)); }
    fill_z(SENT);
    amx_fma32(SKIPZ);
    dump_z();
    printf("%s: %s0 as seen by fma32 (no offsets):", n, n);
    if (isy) { pr_src(n, Zd[0][0]); pr_src(n, Zd[4][0]); pr_src(n, Zd[60][0]); }
    else { pr_src(n, Zd[0][0]); pr_src(n, Zd[0][1]); pr_src(n, Zd[0][15]); }
    printf("\n");
}

// --- 2. fma32 operand offsets (needs the register index of loads to work, see 1)
static const int OFF[] = {0, 64, 128, 192, 448, 4, 32};
static void c_fmax(int k) {
    fill_regs_id(amx_ldx);
    fill_regs(amx_ldy, 1.0f);
    fill_z(SENT);
    amx_fma32((uint64_t)OFF[k] | SKIPZ);
    dump_z();
    printf("fma32 X offset %3d: Z row 0 lanes 0, 1, 15 =", OFF[k]);
    pr_src("X", Zd[0][0]); pr_src("X", Zd[0][1]); pr_src("X", Zd[0][15]);
    printf("\n");
}
static void c_fmay(int k) {
    fill_regs(amx_ldx, 1.0f);
    fill_regs_id(amx_ldy);
    fill_z(SENT);
    amx_fma32(FMA_Y(OFF[k]) | SKIPZ);
    dump_z();
    printf("fma32 Y offset %3d: Z rows 0, 4, 60 lane 0 =", OFF[k]);
    pr_src("Y", Zd[0][0]); pr_src("Y", Zd[4][0]); pr_src("Y", Zd[60][0]);
    printf("\n");
}
// same, but load the register with index 0 from a different source each time
// (independent of the register-index field): ldx X0 <- pattern of reg k, then fma X offset 0
static void c_fmabits(int k) {
    // which fma32 operand bits move the reads? X0 lane l = 1 + l, Y0 lane j = 100 (j + 1),
    // so Z[4j][i] = (1 + i) * 100 (j + 1) when nothing moves; set one bit at a time
    for (int l = 0; l < 16; l++) tmp[l] = 1 + l;
    amx_ldx(PTR(tmp));
    for (int l = 0; l < 16; l++) tmp[l] = 100 * (l + 1);
    amx_ldy(PTR(tmp));
    fill_z(SENT);
    amx_fma32((1ull << k) | SKIPZ);
    dump_z();
    int cnt = 0, first = -1;
    for (int q = 0; q < 64; q++) if (Zd[q][0] != SENT) { if (first < 0) first = q; cnt++; }
    int f = first < 0 ? 0 : first;
    printf("fma32 operand bit %2d: %2d rows written, first %2d; Z[first][0], Z[first][1], Z[first+4][0] = %g %g %g\n",
           k, cnt, first, Zd[f][0], Zd[f][1], f + 4 < 64 ? Zd[f + 4][0] : 0.0f);
}
static void c_fmaz(int zz) {
    fill_regs(amx_ldx, 1.0f);
    fill_regs(amx_ldy, 1.0f);
    fill_z(SENT);
    amx_fma32(FMA_Z(zz) | SKIPZ);
    dump_z();
    int cnt = 0, first[4] = {-1, -1, -1, -1};
    for (int q = 0; q < 64; q++) if (Zd[q][0] != SENT) { if (cnt < 4) first[cnt] = q; cnt++; }
    printf("fma32 Z offset %d: %2d rows written, first %d %d %d %d\n", zz, cnt, first[0], first[1], first[2], first[3]);
}

// --- 3. load variants: where 64 or 128 bytes from src (src[i] = i) land
static const uint64_t BITS[] = {0, B62, B62 | (1ull << 61), B62 | (1ull << 60), 1ull << 61, 1ull << 60};
static const char* BN[] = {"single", "b62", "b62+61", "b62+60", "b61", "b60"};
static void c_load(int code) {                           // code = isy * 1000 + bits * 100 + reg * 10 + (off / 16)
    int isy = code / 1000, b = code / 100 % 10, reg = code / 10 % 10, off = code % 10 * 16;
    amxfn ld = isy ? amx_ldy : amx_ldx, st = isy ? amx_sty : amx_stx;
    fill_regs(ld, SENT);
    ld(PTR(src + off) | REG(reg) | BITS[b]);
    for (int r = 0; r < 8; r++) st(PTR(buf[r]) | REG(r));
    printf("%s %-7s r%d +%3dB:", isy ? "ldy" : "ldx", BN[b], reg, off * 4);
    for (int r = 0; r < 8; r++) {
        float* d = buf[r];
        int untouched = 1, consec = 1;
        for (int l = 0; l < 16; l++) { if (d[l] != SENT) untouched = 0; if (d[l] != d[0] + l) consec = 0; }
        if (untouched) printf("   .");
        else if (consec) printf(" %3d", (int)d[0]);
        else printf(" ?%g", d[0]);
    }
    printf("\n");
}

// --- 4. fma16 with fp32 sums (bit 62)
static int I[64][16], J[64][16];
static void c_f16(int zz) {
    _Float16 h[32] __attribute__((aligned(64)));
    for (int run = 0; run < 2; run++) {
        for (int i = 0; i < 32; i++) h[i] = (_Float16)(i + 1);
        if (run == 0) amx_ldx(PTR(h)); else amx_ldy(PTR(h));
        for (int i = 0; i < 32; i++) h[i] = (_Float16)1.0f;
        if (run == 0) amx_ldy(PTR(h)); else amx_ldx(PTR(h));
        fill_z(SENT);
        amx_fma16(B62 | SKIPZ | FMA_Z(zz));
        dump_z();
        for (int q = 0; q < 64; q++) for (int l = 0; l < 16; l++) {
            float v = Zd[q][l];
            int k = v == SENT ? -1 : (int)v - 1;
            if (run == 0) I[q][l] = k; else J[q][l] = k;
        }
    }
    int written = 0, ha = 0, hb = 0;
    for (int q = 0; q < 64; q++) for (int l = 0; l < 16; l++) {
        if (I[q][l] >= 0) written++;
        if (I[q][l] == 2 * l + (q & 1) && J[q][l] == q / 2) ha++;                // Z[2j + i%2][i/2]
        if (I[q][l] == l + 16 * (q & 1) && J[q][l] == q / 2) hb++;               // Z[2j + i/16][i%16]
    }
    printf("fma16 fp32 sums, Z offset %d: %d of 1024 cells written; Z[2j + i%%2][i/2] matches %d, Z[2j + i/16][i%%16] matches %d\n",
           zz, written, ha, hb);
    if (zz == 0) for (int q = 0; q < 4; q++) {
        printf("  Z row %d holds (i,j):", q);
        for (int l = 0; l < 16; l++) printf(" %d,%d", I[q][l], J[q][l]);
        printf("\n");
    }
}

// --- 5. throughput
struct job { int kind; long n; uint64_t a; double secs; float* xb; float* yb; };
static float bufs[4][3 * 8192 + 64] __attribute__((aligned(256)));
static void* worker(void* p) {
    struct job* j = p;
    amx_set();
    double t0 = now();
    if (j->kind == 0) loop_f16(j->n, j->a, j->a | FMA_Z(1));
    else if (j->kind == 1) loop_f16(j->n, j->a, j->a);
    else if (j->kind == 2) loop_h32(j->n, j->xb, j->yb, j->a);
    else if (j->kind == 3) loop_v2(j->n, j->xb, j->yb);
    else loop_pair(j->n, j->xb, j->yb, j->a);
    j->secs = now() - t0;
    amx_clr();
    return 0;
}
static const char* TN[] = {"fma16 fp32 sums, z 0/1 alternating, no loads", "fma16 fp32 sums, one accumulator, no loads",
                           "ldx + ldy + fma16 fp32 sums (32x32 per k)", "2 ldx + 2 ldy + 4 fma32 (amx32 kernel step)",
                           "ldx b62 + ldy b62 + 4 fma32 (pair step)"};
static void c_tp(int kind) {
    long n = kind < 2 ? 2000000 : 30000000;
    double macs = kind < 2 ? 16 * 1024 : 1024;
    for (int t = 1; t <= 4; t *= 2) {
        pthread_t th[4]; struct job jobs[4];
        for (int i = 0; i < t; i++) {
            jobs[i] = (struct job){kind, n, kind == 3 ? 0 : B62, 0, bufs[i], bufs[i] + 8192};
            pthread_create(&th[i], 0, worker, &jobs[i]);
        }
        double worst = 0;
        for (int i = 0; i < t; i++) { pthread_join(th[i], 0); if (jobs[i].secs > worst) worst = jobs[i].secs; }
        printf("%-46s %d thread%s %6.0f GFLOP/s\n", TN[kind], t, t > 1 ? "s" : " ", 2.0 * macs * n * t / worst / 1e9);
        fflush(0);
    }
}

int main(void) {
    char cpu[128] = {0}; unsigned long len = sizeof cpu;
    if (sysctlbyname("machdep.cpu.brand_string", cpu, &len, 0, 0) == 0) printf("%s\n", cpu);
    for (int i = 0; i < 512; i++) src[i] = i;
    for (int l = 0; l < 16; l++) ones[l] = 1;

    printf("== 1. register index on single loads (pattern: reg r lane l = 1 + 16 r + l)\n");
    run(c_regs, 0);
    run(c_regs, 1);
    printf("== 2. fma32 operands\n");
    for (int k = 0; k < 7; k++) run(c_fmax, k);
    for (int k = 0; k < 7; k++) run(c_fmay, k);
    for (int z = 0; z < 6; z++) run(c_fmaz, z);
    printf("   (no offsets: Z[4j][i] = (1 + i) * 100 (j + 1), e.g. Z[0][0] = 100, Z[0][1] = 200, Z[4][0] = 200)\n");
    for (int k = 0; k < 20; k++) run(c_fmabits, k);
    printf("== 3. loads: which src index (src[i] = i) lands in registers 0..7 (. = untouched)\n");
    for (int b = 0; b < 6; b++)
        for (int reg = 0; reg < 8; reg += 3)
            for (int o = 0; o < 3; o++) run(c_load, b * 100 + reg * 10 + o);
    for (int reg = 0; reg < 8; reg += 3) for (int o = 0; o < 3; o++) run(c_load, 1000 + 100 + reg * 10 + o);
    printf("== 4. fma16 with fp32 sums (bit 62): where X[i] * Y[j] goes (i, j < 32)\n");
    for (int z = 0; z < 3; z++) run(c_f16, z);
    printf("== 5. throughput\n");
    for (int k = 0; k < 5; k++) run(c_tp, k);
    printf("\nPROBE 3 DONE\n");
    return 0;
}
