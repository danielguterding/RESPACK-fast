// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// Accumulation error of GPU GEMMs vs K: T = X Y^T (M x N), inputs exactly representable in FP16
// (10-bit mantissa in [0.5,1), optionally random sign) -> all products exact in FP32, reference sum exact in double.
// Reported per method and K: max and rms of |T - Tref| / sum_k |x y| and the mean signed error (bias) in units of eps32.
// Build (ROCm): hipcc -O3 --offload-arch=<gfx...> -std=c++17 accerr.cpp -o accerr -lhipblas -lopenblas
// Build (CUDA): g++ -O3 -DTOOLS_CUDA -I$CUDA/include accerr.cpp -o accerr -L$CUDA/lib64 -lcublas -lcudart -lopenblas
#ifdef TOOLS_CUDA
#include "hip2cuda.h"   // NVIDIA: same source with CUDA/cuBLAS
#else
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CB(x) do { hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("hipBLAS %d line %d\n",(int)s,__LINE__);exit(1);} } while(0)
extern "C" void sgemm_(const char *, const char *, const int *, const int *, const int *, const float *, const float *, const int *,
                       const float *, const int *, const float *, float *, const int *);
int main() {
  const int M = 256, N = 256, Ks[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};
  const double eps = ldexp(1.0, -24);   // unit roundoff of FP32
  hipblasHandle_t h; CB(hipblasCreate(&h));
  std::mt19937_64 rng(12345);
  for (int signed_data = 0; signed_data < 2; signed_data++) {
    printf("\n%s data (inputs in [0.5,1) x 2^-10 grid%s)\n", signed_data ? "SIGN-MIXED" : "SAME-SIGN", signed_data ? ", random signs" : "");
    printf("%6s | %-34s | %-34s | %-34s | %-34s | %-34s | %-34s\n", "K", "CPU OpenBLAS SGEMM", "GPU SGEMM", "GPU CGEMM (real data)", "GPU FP16 GemmEx, FP32 acc.", "FP16, chunks of 256, beta=1", "FP16, chunks of 64, beta=1");
    printf("%6s |", ""); for (int i = 0; i < 6; i++) printf(" %10s %10s %10s |", "max/eps", "rms/eps", "bias/eps"); printf("\n");
    for (int K : Ks) {
      std::vector<float> X((size_t)M * K), Y((size_t)N * K);
      std::uniform_int_distribution<int> mant(512, 1023), sg(0, 1);
      for (auto &v : X) v = ldexp((float)mant(rng), -10) * (signed_data && sg(rng) ? -1.f : 1.f);
      for (auto &v : Y) v = ldexp((float)mant(rng), -10) * (signed_data && sg(rng) ? -1.f : 1.f);
      // exact reference and normalisation sum |x y|
      std::vector<double> R((size_t)M * N), S((size_t)M * N);
      for (int j = 0; j < N; j++) for (int i = 0; i < M; i++) {
        double r = 0, s = 0;
        for (int k = 0; k < K; k++) { double p = (double)X[i + (size_t)M * k] * Y[j + (size_t)N * k]; r += p; s += fabs(p); }
        R[i + (size_t)M * j] = r; S[i + (size_t)M * j] = s;
      }
      std::vector<std::vector<float>> T(6, std::vector<float>((size_t)M * N));
      float one = 1.f, zero = 0.f;
      sgemm_("N", "T", &M, &N, &K, &one, X.data(), &M, Y.data(), &N, &zero, T[0].data(), &M);
      float *dX, *dY, *dT; hipComplex *cX, *cY, *cT; __half *hX, *hY;
      CK(hipMalloc(&dX, X.size() * 4)); CK(hipMalloc(&dY, Y.size() * 4)); CK(hipMalloc(&dT, (size_t)M * N * 4));
      CK(hipMalloc(&cX, X.size() * 8)); CK(hipMalloc(&cY, Y.size() * 8)); CK(hipMalloc(&cT, (size_t)M * N * 8));
      CK(hipMalloc(&hX, X.size() * 2)); CK(hipMalloc(&hY, Y.size() * 2));
      CK(hipMemcpy(dX, X.data(), X.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dY, Y.data(), Y.size() * 4, hipMemcpyHostToDevice));
      std::vector<hipComplex> cx(X.size()), cy(Y.size()); std::vector<__half> hx(X.size()), hy(Y.size());
      for (size_t i = 0; i < X.size(); i++) { cx[i] = make_hipComplex(X[i], 0.f); hx[i] = __float2half(X[i]); }
      for (size_t i = 0; i < Y.size(); i++) { cy[i] = make_hipComplex(Y[i], 0.f); hy[i] = __float2half(Y[i]); }
      CK(hipMemcpy(cX, cx.data(), cx.size() * 8, hipMemcpyHostToDevice)); CK(hipMemcpy(cY, cy.data(), cy.size() * 8, hipMemcpyHostToDevice));
      CK(hipMemcpy(hX, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(hY, hy.data(), hy.size() * 2, hipMemcpyHostToDevice));
      CB(hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &one, dX, M, dY, N, &zero, dT, M));
      CK(hipMemcpy(T[1].data(), dT, (size_t)M * N * 4, hipMemcpyDeviceToHost));
      hipComplex cone = {1.f, 0.f}, czero = {0.f, 0.f};
      CB(hipblasCgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &cone, cX, M, cY, N, &czero, cT, M));
      std::vector<hipComplex> ct((size_t)M * N); CK(hipMemcpy(ct.data(), cT, ct.size() * 8, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < ct.size(); i++) T[2][i] = ct[i].x;
      CB(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &one, hX, HIP_R_16F, M, hY, HIP_R_16F, N, &zero, dT, HIP_R_32F, M,
                       HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
      CK(hipMemcpy(T[3].data(), dT, (size_t)M * N * 4, hipMemcpyDeviceToHost));
      for (int v = 0; v < 2; v++) {   // FP16 GemmEx in K-chunks of 256 / 64, continued with beta = 1
        const int kc = v ? 64 : 256;
        for (int k0 = 0; k0 < K; k0 += kc) {
          int kk = K - k0 < kc ? K - k0 : kc; float b = k0 ? 1.f : 0.f;
          CB(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, kk, &one, hX + (size_t)M * k0, HIP_R_16F, M, hY + (size_t)N * k0, HIP_R_16F, N,
                           &b, dT, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        }
        CK(hipMemcpy(T[4 + v].data(), dT, (size_t)M * N * 4, hipMemcpyDeviceToHost));
      }
      printf("%6d |", K);
      for (int m = 0; m < 6; m++) {
        double mx = 0, ss = 0, bias = 0;
        for (size_t i = 0; i < R.size(); i++) {
          double e = ((double)T[m][i] - R[i]) / S[i] / eps;
          mx = fmax(mx, fabs(e)); ss += e * e; bias += (R[i] >= 0 ? e : -e);   // bias: >0 away from zero, <0 toward zero
        }
        printf(" %10.2f %10.2f %10.2f |", mx, sqrt(ss / R.size()), bias / R.size());
      }
      printf("\n");
      CK(hipFree(dX)); CK(hipFree(dY)); CK(hipFree(dT)); CK(hipFree(cX)); CK(hipFree(cY)); CK(hipFree(cT)); CK(hipFree(hX)); CK(hipFree(hY));
    }
  }
  return 0;
}
