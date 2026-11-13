// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// Edge cases of a (patched) kernel launched directly: M = NG, N = 2 NG, K from 2 to 16384, beta = 1 on a nonzero
// C, against hipblasGemmEx (the library kernel itself faults for M < 8).
//   tval <code object> <kernel name> [M K]
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
int main(int argc, char **argv) {
  hipModule_t mod; hipFunction_t f;
  CK(hipModuleLoad(&mod, argv[1])); CK(hipModuleGetFunction(&f, mod, argv[2]));
  hipblasHandle_t h; hipblasCreate(&h);
  const int Ms[] = {1, 7, 37, 128, 129, 300, 777, 1963}, Ks[] = {2, 16, 34, 274, 1000, 4098, 16384};
  int bad = 0; setvbuf(stdout, NULL, _IONBF, 0);
  std::vector<int> vM(Ms, Ms + 8), vK(Ks, Ks + 7);
  if (argc > 4) { vM = {atoi(argv[3])}; vK = {atoi(argv[4])}; }
  for (int M : vM) for (int K : vK) {
    const int N = 2 * M;
    __half *X, *Y; float *T0, *T1;
    CK(hipMalloc(&X, (size_t)M * K * 2)); CK(hipMalloc(&Y, (size_t)N * K * 2));
    CK(hipMalloc(&T0, (size_t)M * N * 4)); CK(hipMalloc(&T1, (size_t)M * N * 4));
    fill<<<256, 256>>>((size_t)M * K, 1u + K, X); fill<<<256, 256>>>((size_t)N * K, 2u + M, Y);
    fill<<<256, 256>>>((size_t)M * N, 3u, (__half *)T0);   // garbage bits as C: then copy, so both get the same C
    std::vector<float> c((size_t)M * N); for (size_t i = 0; i < c.size(); i++) c[i] = (float)((i * 7919) % 1001) - 500.f;
    CK(hipMemcpy(T0, c.data(), c.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(T1, c.data(), c.size() * 4, hipMemcpyHostToDevice));
    const float one = 1.f;
    hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &one, X, HIP_R_16F, M, Y, HIP_R_16F, N, &one, T0, HIP_R_32F, M,
                  HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    KArgs a; memset(&a, 0, sizeof a);
    a.gemm_info = 1; a.kinfo0 = 0x03200001; a.kinfo1 = 0x04010008;
    const int t0 = (M + 127) / 128, t1 = (N + 127) / 128;
    a.numWG = t0; a.M = M; a.N = N; a.batch = 1; a.K = K;
    a.D = T1; a.C = T1; a.A = X; a.B = Y;
    a.sD0 = M; a.sD1 = 1; a.sC0 = M; a.sC1 = 1; a.sA0 = M; a.sA1 = 1; a.sB0 = N; a.sB1 = 1;
    a.alpha = 1.f; a.beta = 1.f;
    size_t sz = sizeof a;
    void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(f, t0, t1, 1, 128, 1, 1, 0, 0, NULL, extra));
    CK(hipDeviceSynchronize());
    std::vector<float> r0((size_t)M * N), r1((size_t)M * N);
    CK(hipMemcpy(r0.data(), T0, r0.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r1.data(), T1, r1.size() * 4, hipMemcpyDeviceToHost));
    double mx = 0, md = 0;
    for (size_t i = 0; i < r0.size(); i++) { mx = fmax(mx, fabs(r0[i])); md = fmax(md, fabs((double)r0[i] - r1[i])); }
    const bool ok = md <= 1e-5 * mx;
    if (!ok) bad++;
    printf("M %5d K %6d: max |diff| / max %.2e %s\n", M, K, md / mx, ok ? "ok" : "BAD");
    hipFree(X); hipFree(Y); hipFree(T0); hipFree(T1);
  }
  printf("%d bad\n", bad);
  return bad != 0;
}
