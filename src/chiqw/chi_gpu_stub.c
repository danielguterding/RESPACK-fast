/* Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
 * GNU General Public License, version 3 (see GPL/GPLv3.txt).
 * Stub of the CUDA path (chi_gpu.cu) for builds without -DRESPACK_CUDA=ON. */
#include <stdio.h>
static int nogpu(void) { fprintf(stderr, "chi_gpu: this calc_chiqw was built without CUDA (cmake -DRESPACK_CUDA=ON)\n"); return -100; }
int chi_gpu_available(void) { return 0; }
double chi_gpu_mem_total_mb(void) { return 0.0; }
int chi_gpu_init(int a, int b, int c, int d, int e, int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return nogpu(); }
int chi_gpu_q_begin(int a, int b, int c, const int *d) { (void)a; (void)b; (void)c; (void)d; return nogpu(); }
int chi_gpu_tile_begin(int a, int b, int c, const int *d, const int *e) { (void)a; (void)b; (void)c; (void)d; (void)e; return nogpu(); }
int chi_gpu_bands(int a, int b, const void *c, int d, const int *e, int f) { (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return nogpu(); }
int chi_gpu_pairs(int a, int b, int c, int d, double e) { (void)a; (void)b; (void)c; (void)d; (void)e; return nogpu(); }
int chi_gpu_gemm(const void *a, const void *b, int c, int d, int e, int f) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; return nogpu();
}
int chi_gpu_q_end(void *a, void *b, void *c) { (void)a; (void)b; (void)c; return nogpu(); }
