// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// One block (nr rows) and one frequency of the chi0 products at a given NG, timed sustained (all variants in
// alternating rounds, median of 11): the CGEMM path (blocks of kc rows) and the parts of the FP16 path (gemm_mp in
// chi_gpu.cu): corrections (K' = 4 nr, beta = 0), hi*hi (K' = 2 nr) in chunks of 512 .. 4096 or in one call.
//   mpbench [NG] [nr] [kc of the CGEMM path] [leading dimensions rounded up to a multiple of this]
// Build (ROCm): hipcc -O3 --offload-arch=<gfx...> -std=c++17 mpbench.cpp -o mpbench -lhipblas
// Build (CUDA): g++ -O3 -DTOOLS_CUDA -I$CUDA/include mpbench.cpp -o mpbench -L$CUDA/lib64 -lcublas -lcudart
#ifdef TOOLS_CUDA
#include "hip2cuda.h"
#else
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>
#include <string>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("GPU %s line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CB(x) do { hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("BLAS %d line %d\n",(int)s,__LINE__);exit(1);} } while(0)
int main(int argc, char **argv) {
  const int NG = argc > 1 ? atoi(argv[1]) : 1963, nr = argc > 2 ? atoi(argv[2]) : 8192, kc = argc > 3 ? atoi(argv[3]) : 2048;
  const int pad = argc > 4 ? atoi(argv[4]) : 1;
  const int M = NG, N = 2 * NG, ldx = (M + pad - 1) / pad * pad, ldy = (N + pad - 1) / pad * pad, ldt = ldx;
  hipblasHandle_t h; CB(hipblasCreate(&h));
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
  printf("%s: NG %d, %d rows per block, CGEMM path in blocks of %d rows, leading dimensions %d %d %d\n", pr.name, NG, nr, kc, ldx, ldy, ldt);
  // FP16 operands: X (M x 4nr) and Y (N x 6nr) as in gemm_mp; values in [-2, 2) (random bits do not matter for speed,
  // but zeros could: fill with a pattern)
  std::vector<__half> hx((size_t)ldx * 4 * nr), hy((size_t)ldy * 6 * nr);
  for (size_t i = 0; i < hx.size(); i++) hx[i] = __float2half((float)((i * 2654435761u >> 7) % 4001) / 1000.f - 2.f);
  for (size_t i = 0; i < hy.size(); i++) hy[i] = __float2half((float)((i * 40503u >> 3) % 4001) / 1000.f - 2.f);
  __half *X, *Y; float *T; hipComplex *A, *B, *C;
  CK(hipMalloc(&X, hx.size() * 2)); CK(hipMalloc(&Y, hy.size() * 2)); CK(hipMalloc(&T, (size_t)ldt * N * 4));
  CK(hipMemcpy(X, hx.data(), hx.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(Y, hy.data(), hy.size() * 2, hipMemcpyHostToDevice));
  CK(hipMalloc(&A, (size_t)NG * kc * 8)); CK(hipMalloc(&B, (size_t)NG * kc * 8)); CK(hipMalloc(&C, (size_t)NG * NG * 8));
  CK(hipMemset(A, 0x3c, (size_t)NG * kc * 8)); CK(hipMemset(B, 0x3c, (size_t)NG * kc * 8));
  const float one = 1.f, zero = 0.f;
  const hipComplex cone = {1.f, 0.f}, czero = {0.f, 0.f};
  auto gx = [&](int K, const __half *x, const __half *y, float beta) {
    CB(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, &one, x, HIP_R_16F, ldx, y, HIP_R_16F, ldy, &beta, T, HIP_R_32F, ldt,
                     HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  };
  auto hh = [&](int ch) { for (int k0 = 0; k0 < 2 * nr; k0 += ch) gx(std::min(ch, 2 * nr - k0), X + (size_t)ldx * k0, Y + (size_t)ldy * k0, 1.f); };
  std::vector<std::pair<std::string, std::function<void()>>> V;
  V.push_back({"CGEMM path (" + std::to_string((nr + kc - 1) / kc) + " calls)", [&]() {
    for (int r0 = 0; r0 < nr; r0 += kc)
      CB(hipblasCgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, NG, NG, std::min(kc, nr - r0), &cone, A, NG, B, NG, r0 ? &cone : &czero, C, NG)); }});
  V.push_back({"FP16 corrections (one call, beta 0)", [&]() { gx(4 * nr, X, Y + (size_t)ldy * 2 * nr, 0.f); }});
  for (int ch : {512, 1024, 2048, 4096})
    V.push_back({"FP16 hi*hi, chunks of " + std::to_string(ch), [&, ch]() { hh(ch); }});
  V.push_back({"FP16 hi*hi, one call (biased)", [&]() { hh(2 * nr); }});
  V.push_back({"FP16 hi*hi, one call, beta 0", [&]() { gx(2 * nr, X, Y, 0.f); }});
  for (int i = 0; i < 3; i++) for (auto &v : V) v.second();
  CK(hipDeviceSynchronize());
  hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
  std::vector<std::vector<float>> t(V.size());
  for (int r = 0; r < 11; r++)
    for (size_t v = 0; v < V.size(); v++) {
      float ms; CK(hipEventRecord(e0)); for (int i = 0; i < 3; i++) V[v].second(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
      CK(hipEventElapsedTime(&ms, e0, e1)); t[v].push_back(ms / 3);
    }
  const double fl_c = 8.0 * NG * NG * nr, fl_corr = 2.0 * M * N * 4 * nr, fl_hh = 2.0 * M * N * 2 * nr;
  for (size_t v = 0; v < V.size(); v++) {
    std::sort(t[v].begin(), t[v].end());
    const double fl = v == 0 ? fl_c : v == 1 ? fl_corr : fl_hh;
    printf("%-40s median %7.3f ms [%7.3f .. %7.3f]  %6.1f TFLOP/s\n", V[v].first.c_str(), t[v][5], t[v][0], t[v][10], fl / t[v][5] / 1e9);
  }
  printf("FP16 path per block and frequency, chunks of 512: %.3f ms; hi*hi in one call: %.3f ms; CGEMM path: %.3f ms\n",
         t[1][5] + t[2][5], t[1][5] + t[6][5], t[0][5]);
  return 0;
}
