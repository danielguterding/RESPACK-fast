// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// One block and frequency of gemm_mp at NG = 1963 (8192 rows): the chunked hipBLAS path against a corrections
// kernel plus the patched kernel for hi*hi in one call, and the parts, timed in alternation (sustained).
//   tcombo <code object corr> <name corr> <code object hh> <name hh>
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <functional>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
struct __attribute__((packed)) KArgs {
  uint32_t gemm_info, kinfo0, kinfo1, numWG, M, N, batch, K;
  void *D, *C, *A, *B;
  uint32_t sD0, sD1, sC0, sC1, sA0, sA1, sB0, sB1;
  float alpha, beta;
  void *scaleAlphaVec, *bias;
  uint32_t biasType, strideBias;
  float actAlpha, actBeta;
  uint32_t actType;
};

__global__ void fill(size_t n, unsigned seed, __half *x) {   // pseudo-random values in [-1.95, 1.95], multiples of 1/512
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)(i * 2654435761u) ^ (seed * 0x9e3779b9u); h ^= h >> 15; h *= 0x85ebca6bu; h ^= h >> 13;
    x[i] = __float2half((float)((int)(h % 2001u) - 1000) / 512.f);
  }
}

static void launchk(hipFunction_t f, int local, __half *X, __half *Y, float *T, int M, int N, int K, int lda, int ldb,
                    float beta) {
  KArgs a; memset(&a, 0, sizeof a);
  a.gemm_info = 1; a.kinfo0 = 0x03200001; a.kinfo1 = 0x04010008;
  const int t0 = (M + 127) / 128, t1 = (N + 127) / 128;
  a.numWG = t0; a.M = M; a.N = N; a.batch = 1; a.K = K;
  a.D = T; a.C = T; a.A = X; a.B = Y;
  a.sD0 = M; a.sD1 = 1; a.sC0 = M; a.sC1 = 1; a.sA0 = lda; a.sA1 = 1; a.sB0 = ldb; a.sB1 = 1;
  a.alpha = 1.f; a.beta = beta;
  size_t sz = sizeof a;
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
  CK(hipExtModuleLaunchKernel(f, t0 * local, t1, 1, local, 1, 1, 0, 0, NULL, extra, NULL, NULL, 0));
}
int main(int argc, char **argv) {
  const int NG = 1963, M = NG, N = 2 * NG, nr = 8192;
  hipModule_t m1, m2; hipFunction_t fc, fh;
  CK(hipModuleLoad(&m1, argv[1])); CK(hipModuleGetFunction(&fc, m1, argv[2]));
  CK(hipModuleLoad(&m2, argv[3])); CK(hipModuleGetFunction(&fh, m2, argv[4]));
  hipblasHandle_t h; hipblasCreate(&h);
  __half *X, *Y; float *T;
  const size_t nx = (size_t)M * 4 * nr, ny = (size_t)N * 6 * nr;
  CK(hipMalloc(&X, nx * 2)); CK(hipMalloc(&Y, ny * 2)); CK(hipMalloc(&T, (size_t)M * N * 4));
  fill<<<4096, 256>>>(nx, 1u, X); fill<<<4096, 256>>>(ny, 2u, Y); CK(hipDeviceSynchronize());
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  const float one = 1.f, zero = 0.f;
  auto corrH = [&]() {
    hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, 4 * nr, &one, X, HIP_R_16F, M, Y + (size_t)N * 2 * nr, HIP_R_16F, N, &zero,
                  T, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  };
  auto hhH = [&]() {
    for (int k0 = 0; k0 < 2 * nr; k0 += 512)
      hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, 512, &one, X + (size_t)M * k0, HIP_R_16F, M, Y + (size_t)N * k0, HIP_R_16F, N,
                    &one, T, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  };
  auto hh1H = [&]() {   // one biased call, reference only
    hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, 2 * nr, &one, X, HIP_R_16F, M, Y, HIP_R_16F, N,
                  &one, T, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  };
  auto corrB = [&]() { launchk(fc, 128, X, Y + (size_t)N * 2 * nr, T, M, N, 4 * nr, M, N, 0.f); };
  auto hhP = [&]() { launchk(fh, 128, X, Y, T, M, N, 2 * nr, M, N, 1.f); };
  auto varA = [&]() { corrH(); hhH(); };
  auto varB = [&]() { corrB(); hhP(); };
  auto varC = [&]() { corrH(); hhP(); };
  std::vector<std::pair<const char *, std::function<void()>>> V = {
      {"A  = corr hipBLAS + hh chunked", varA}, {"B  = corr best + hh patched", varB}, {"C  = corr hipBLAS + hh patched", varC},
      {"corr hipBLAS (K 32768, beta 0)", corrH}, {"corr best", corrB},
      {"hh chunked 32 x 512", hhH}, {"hh patched one call", hhP}, {"hh hipBLAS one call (biased)", hh1H}};
  for (int i = 0; i < 3; i++) for (auto &v : V) v.second();
  CK(hipDeviceSynchronize());
  std::vector<std::vector<float>> t(V.size()); float ms;
  for (int r = 0; r < 11; r++)
    for (size_t v = 0; v < V.size(); v++) {
      CK(hipEventRecord(e0)); for (int i = 0; i < 3; i++) V[v].second(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
      CK(hipEventElapsedTime(&ms, e0, e1)); t[v].push_back(ms / 3);
    }
  for (size_t v = 0; v < V.size(); v++) {
    std::sort(t[v].begin(), t[v].end());
    printf("%-34s median %6.2f ms [%6.2f .. %6.2f]\n", V[v].first, t[v][5], t[v][0], t[v][10]);
  }
  return 0;
}
