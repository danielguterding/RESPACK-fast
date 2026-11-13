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

static void launch(hipFunction_t f, __half *X, __half *Y, float *T, int M, int N, int K) {
  KArgs a; memset(&a, 0, sizeof a);
  a.gemm_info = 1; a.kinfo0 = 0x03200001; a.kinfo1 = 0x04010008;
  const int t0 = (M + 127) / 128, t1 = (N + 127) / 128;
  a.numWG = t0; a.M = M; a.N = N; a.batch = 1; a.K = K;
  a.D = T; a.C = T; a.A = X; a.B = Y;
  a.sD0 = M; a.sD1 = 1; a.sC0 = M; a.sC1 = 1; a.sA0 = M; a.sA1 = 1; a.sB0 = N; a.sB1 = 1;
  a.alpha = 1.f; a.beta = 0.f;
  size_t sz = sizeof a;
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
  CK(hipExtModuleLaunchKernel(f, t0 * 128, t1, 1, 128, 1, 1, 0, 0, NULL, extra, NULL, NULL, 0));
  CK(hipDeviceSynchronize());
}
int main(int argc, char **argv) {
  std::ifstream nf(argv[2]); std::string kname; nf >> kname;
  hipModule_t mod; hipFunction_t f;
  CK(hipModuleLoad(&mod, argv[1])); CK(hipModuleGetFunction(&f, mod, kname.c_str()));
  hipblasHandle_t h; hipblasCreate(&h);
  const int M = 256, N = 512;
  const double eps = ldexp(1.0, -24);
  printf("%s\n%6s | %-32s | %-32s\n", argv[1], "K", "this kernel: max  rms  bias (eps)", "hipblasGemmEx: max  rms  bias (eps)");
  for (int K : {512, 2048, 8192, 32768}) {
    std::vector<__half> hx((size_t)M * K), hy((size_t)N * K); std::vector<float> fx(hx.size()), fy(hy.size());
    srand(7);
    for (size_t i = 0; i < hx.size(); i++) { fx[i] = ldexp((float)(512 + rand() % 512), -10); hx[i] = __float2half(fx[i]); }
    for (size_t i = 0; i < hy.size(); i++) { fy[i] = ldexp((float)(512 + rand() % 512), -10); hy[i] = __float2half(fy[i]); }
    std::vector<double> R((size_t)M * N), S((size_t)M * N);
    for (int j = 0; j < N; j++) for (int i = 0; i < M; i++) {
      double r = 0, sa = 0;
      for (int k = 0; k < K; k++) { double p = (double)fx[i + (size_t)M * k] * fy[j + (size_t)N * k]; r += p; sa += fabs(p); }
      R[i + (size_t)M * j] = r; S[i + (size_t)M * j] = sa;
    }
    __half *X, *Y; float *T;
    CK(hipMalloc(&X, hx.size() * 2)); CK(hipMalloc(&Y, hy.size() * 2)); CK(hipMalloc(&T, R.size() * 4));
    CK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(Y, hy.data(), hy.size() * 2, hipMemcpyHostToDevice));
    std::vector<float> res(R.size());
    double st[2][3];
    for (int v = 0; v < 2; v++) {
      CK(hipMemset(T, 0, R.size() * 4));
      if (v == 0) launch(f, X, Y, T, M, N, K);
      else { float one = 1.f, zero = 0.f;
        hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &one, X, HIP_R_16F, M, Y, HIP_R_16F, N, &zero, T, HIP_R_32F, M,
                      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT); CK(hipDeviceSynchronize()); }
      CK(hipMemcpy(res.data(), T, res.size() * 4, hipMemcpyDeviceToHost));
      double mx = 0, ss = 0, bias = 0;
      for (size_t i = 0; i < R.size(); i++) { double e = (res[i] - R[i]) / S[i] / eps; mx = fmax(mx, fabs(e)); ss += e * e; bias += e; }
      st[v][0] = mx; st[v][1] = sqrt(ss / R.size()); st[v][2] = bias / R.size();
    }
    printf("%6d | %9.2f %9.2f %9.2f      | %9.2f %9.2f %9.2f\n", K, st[0][0], st[0][1], st[0][2], st[1][0], st[1][1], st[1][2]);
    CK(hipFree(X)); CK(hipFree(Y)); CK(hipFree(T));
  }
  return 0;
}
