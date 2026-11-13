// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// Sustained timing of a list of kernels for one gemm_mp shape at NG = 1963, each against hipblasGemmEx measured right
// before and after it (the RX 9060 XT clocks down under sustained load: burst timings mislead).
//   tsus <code object> <kernel name list> <K> <beta> [Y offset in columns] [reps]
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
static int parse(const std::string &n, const char *key, int idx) {
  size_t p = n.find(key); if (p == std::string::npos) return -1;
  p += strlen(key); int v[3] = {0, 0, 0}; sscanf(n.c_str() + p, "%dx%dx%d", &v[0], &v[1], &v[2]); return v[idx];
}
int main(int argc, char **argv) {
  const int M = 1963, N = 2 * 1963, K = atoi(argv[3]); const float beta = atof(argv[4]), alpha = 1.f;
  const size_t yoff = argc > 5 ? atol(argv[5]) : 0; const int reps = argc > 6 ? atoi(argv[6]) : 4;
  hipModule_t mod; CK(hipModuleLoad(&mod, argv[1]));
  std::vector<std::string> names; { std::ifstream f(argv[2]); std::string l; while (std::getline(f, l)) if (!l.empty()) names.push_back(l); }
  hipblasHandle_t h; hipblasCreate(&h);
  __half *X, *Y; float *T;
  const size_t nx = (size_t)M * K, ny = (size_t)N * (K + yoff);
  CK(hipMalloc(&X, nx * 2)); CK(hipMalloc(&Y, ny * 2)); CK(hipMalloc(&T, (size_t)M * N * 4));
  fill<<<4096, 256>>>(nx, 1u, X); fill<<<4096, 256>>>(ny, 2u, Y); CK(hipDeviceSynchronize());
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  auto ref = [&]() { hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &alpha, X, HIP_R_16F, M, Y + N * yoff, HIP_R_16F, N, &beta, T,
                                   HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT); };
  auto timeit = [&](std::function<void()> fn) { float ms; CK(hipEventRecord(e0)); for (int i = 0; i < reps; i++) fn();
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); return ms / reps; };
  for (int i = 0; i < 20; i++) ref();
  for (auto &kname : names) {
    hipFunction_t f; if (hipModuleGetFunction(&f, mod, kname.c_str()) != hipSuccess) { printf("missing %s\n", kname.c_str()); continue; }
    const int MT0 = parse(kname, "_MT", 0), MT1 = parse(kname, "_MT", 1);
    int wg[3] = {1, 1, 1}; { size_t p = kname.rfind("_WG"); sscanf(kname.c_str() + p + 3, "%d_%d_%d", &wg[0], &wg[1], &wg[2]); }
    const int local = wg[0] * wg[1] * wg[2], t0 = (M + MT0 - 1) / MT0, t1 = (N + MT1 - 1) / MT1;
    KArgs a; memset(&a, 0, sizeof a);
    a.gemm_info = 1; a.kinfo0 = 0x03200001; a.kinfo1 = 0x04010008;
    a.numWG = t0; a.M = M; a.N = N; a.batch = 1; a.K = K;
    a.D = T; a.C = T; a.A = X; a.B = Y + N * yoff;
    a.sD0 = M; a.sD1 = 1; a.sC0 = M; a.sC1 = 1; a.sA0 = M; a.sA1 = 1; a.sB0 = N; a.sB1 = 1;
    a.alpha = alpha; a.beta = beta;
    size_t sz = sizeof a;
    void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    auto kern = [&]() { CK(hipExtModuleLaunchKernel(f, t0 * local, t1, 1, local, 1, 1, 0, 0, NULL, extra, NULL, NULL, 0)); };
    kern(); kern();
    std::vector<float> r, tk;
    for (int round = 0; round < 3; round++) { float a1 = timeit(ref), k1 = timeit(kern), a2 = timeit(ref); r.push_back(k1 / (0.5f * (a1 + a2))); tk.push_back(k1); }
    std::sort(r.begin(), r.end()); std::sort(tk.begin(), tk.end());
    printf("%.4f %.3f %s\n", r[1], tk[1], kname.c_str());
    fflush(stdout);
  }
  return 0;
}
