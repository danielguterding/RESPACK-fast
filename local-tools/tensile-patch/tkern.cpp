// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// Launch the hipBLASLt FP16 kernel (Cijk_Ailk_Bjlk_HSS_..., NT, FP16 in, FP32 out) from a given code object with the
// argument layout recorded from hipBLASLt, and compare with hipblasGemmEx.
//   tkern <code object> <kernel name file> [K] [beta]
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
int main(int argc, char **argv) {
  const char *co = argv[1];
  std::ifstream nf(argv[2]); std::string kname; nf >> kname;
  const int M = 1963, N = 2 * 1963, K = argc > 3 ? atoi(argv[3]) : 32768;
  const float beta = argc > 4 ? atof(argv[4]) : 0.f, alpha = 1.f;
  static_assert(sizeof(KArgs) == 140, "kernarg layout");
  std::vector<__half> hx((size_t)M * K), hy((size_t)N * K);
  srand(1);
  for (auto &v : hx) v = __float2half((float)((rand() % 2001) - 1000) / 512.f);
  for (auto &v : hy) v = __float2half((float)((rand() % 2001) - 1000) / 512.f);
  std::vector<float> hc((size_t)M * N);
  for (auto &v : hc) v = (float)((rand() % 2001) - 1000) / 64.f;
  __half *X, *Y; float *T0, *T1;
  CK(hipMalloc(&X, hx.size() * 2)); CK(hipMalloc(&Y, hy.size() * 2));
  CK(hipMalloc(&T0, hc.size() * 4)); CK(hipMalloc(&T1, hc.size() * 4));
  CK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(Y, hy.data(), hy.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(T0, hc.data(), hc.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(T1, hc.data(), hc.size() * 4, hipMemcpyHostToDevice));
  hipblasHandle_t h; hipblasCreate(&h);
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &alpha, X, HIP_R_16F, M, Y, HIP_R_16F, N, &beta, T0, HIP_R_32F, M,
                HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
  CK(hipDeviceSynchronize());
  hipModule_t mod; hipFunction_t f;
  CK(hipModuleLoad(&mod, co));
  CK(hipModuleGetFunction(&f, mod, kname.c_str()));
  KArgs a; memset(&a, 0, sizeof a);
  a.gemm_info = 1; a.kinfo0 = 0x03200001; a.kinfo1 = 0x04010008;
  const int t0 = (M + 127) / 128, t1 = (N + 127) / 128;
  a.numWG = t0; a.M = M; a.N = N; a.batch = 1; a.K = K;
  a.D = T1; a.C = T1; a.A = X; a.B = Y;
  a.sD0 = M; a.sD1 = 1; a.sC0 = M; a.sC1 = 1; a.sA0 = M; a.sA1 = 1; a.sB0 = N; a.sB1 = 1;
  a.alpha = alpha; a.beta = beta;
  size_t sz = sizeof a;
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
  CK(hipExtModuleLaunchKernel(f, t0 * 128, t1, 1, 128, 1, 1, 0, 0, NULL, extra, NULL, NULL, 0));
  CK(hipDeviceSynchronize());
  std::vector<float> r0(hc.size()), r1(hc.size());
  CK(hipMemcpy(r0.data(), T0, r0.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(r1.data(), T1, r1.size() * 4, hipMemcpyDeviceToHost));
  size_t ndiff = 0; double mx = 0;
  for (size_t i = 0; i < r0.size(); i++) if (r0[i] != r1[i]) { ndiff++; mx = fmax(mx, fabs(r0[i] - r1[i])); }
  // timing (beta = 0 so that repeats give the same result)
  a.beta = 0.f;
  // timing: warm-up (GPU clocks), then the best of 5 rounds of 10 launches each
  float ms1 = 1e30f, ms0 = 1e30f, ms;
  for (int i = 0; i < 20; i++) CK(hipExtModuleLaunchKernel(f, t0 * 128, t1, 1, 128, 1, 1, 0, 0, NULL, extra, NULL, NULL, 0));
  for (int r = 0; r < 5; r++) {
    CK(hipEventRecord(e0)); for (int i = 0; i < 10; i++) CK(hipExtModuleLaunchKernel(f, t0 * 128, t1, 1, 128, 1, 1, 0, 0, NULL, extra, NULL, NULL, 0));
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); ms1 = fminf(ms1, ms);
    CK(hipEventRecord(e0)); for (int i = 0; i < 10; i++) hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &alpha, X, HIP_R_16F, M, Y, HIP_R_16F, N, &a.beta, T0, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); ms0 = fminf(ms0, ms);
  }
  double fl = 2.0 * M * N * (double)K;
  printf("%s: K %d beta %g | differing elements vs hipblasGemmEx: %zu of %zu (max |d| %.3g) | %.3f ms (%.1f TFLOP/s) vs hipBLAS %.3f ms\n",
         co, K, beta, ndiff, r0.size(), mx, ms1 / 10, fl / (ms1 / 10 * 1e-3) / 1e12, ms0 / 10);
  return 0;
}
