// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// Tests of src/chiqw/hgemm_rn.cuh (FP16 GEMM with round-to-nearest accumulation, NVIDIA):
//   1. accumulation error against an exact reference (same-sign data as in accerr.cpp), units of 2^-24
//   2. edge cases (M, N, K not multiples of the tiles; beta = 1) against cublasGemmEx
//   3. sustained timing at the shapes of gemm_mp in chi_gpu.cu (NG padded to a multiple of 8)
// Build: nvcc -O3 -std=c++17 -arch=sm_89 -I../src/chiqw hgemm_rn_test.cu -o hgemm_rn_test -lcublas
//   hgemm_rn_test [NG] [rows per block]
#include "hgemm_rn.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <string>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLAS %d line %d\n",(int)s,__LINE__);exit(1);} } while(0)
static cublasHandle_t h;
static void cublas_gemm(int M, int N, int K, const __half *A, int lda, const __half *B, int ldb, float *C, int ldc, float beta) {
  const float one = 1.f;
  CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &one, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &beta, C, CUDA_R_32F, ldc,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
__global__ void fill(size_t n, unsigned seed, __half *x) {   // values in [-2, 2), multiples of 1/512
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned v = (unsigned)(i * 2654435761u) ^ (seed * 0x9e3779b9u); v ^= v >> 15; v *= 0x85ebca6bu; v ^= v >> 13;
    x[i] = __float2half((float)((int)(v % 2001u) - 1000) / 512.f);
  }
}
int main(int argc, char **argv) {
  const int NG = argc > 1 ? atoi(argv[1]) : 1963, nr = argc > 2 ? atoi(argv[2]) : 8192;
  CB(cublasCreate(&h));
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("%s\n", pr.name);
  // ---- 1. accumulation error
  {
    const int M = 256, N = 256, Ks[] = {512, 2048, 8192, 16384};
    const double eps = ldexp(1.0, -24);
    std::mt19937_64 rng(12345);
    printf("\nsame-sign data, error / sum|ab| in units of 2^-24: rms / bias\n%6s | %-17s | %-17s | %-17s | %-17s\n", "K",
           "cuBLAS one call", "cuBLAS chunks 512", "kernel, plain", "kernel, RN");
    for (int K : Ks) {
      std::vector<__half> a((size_t)M * K), b((size_t)N * K);
      std::vector<double> af(a.size()), bf(b.size());
      std::uniform_int_distribution<int> mant(512, 1023);
      for (size_t i = 0; i < a.size(); i++) { af[i] = ldexp((double)mant(rng), -10); a[i] = __float2half((float)af[i]); }
      for (size_t i = 0; i < b.size(); i++) { bf[i] = ldexp((double)mant(rng), -10); b[i] = __float2half((float)bf[i]); }
      std::vector<double> R((size_t)M * N, 0.0);
      for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) { const double y = bf[j + (size_t)N * k]; for (int i = 0; i < M; i++) R[i + (size_t)M * j] += af[i + (size_t)M * k] * y; }
      __half *A, *B; float *C;
      CK(cudaMalloc(&A, a.size() * 2)); CK(cudaMalloc(&B, b.size() * 2)); CK(cudaMalloc(&C, (size_t)M * N * 4));
      CK(cudaMemcpy(A, a.data(), a.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(B, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
      printf("%6d |", K);
      for (int v = 0; v < 4; v++) {
        if (v == 0) cublas_gemm(M, N, K, A, M, B, N, C, M, 0.f);
        if (v == 1) for (int k0 = 0; k0 < K; k0 += 512) cublas_gemm(M, N, std::min(512, K - k0), A + (size_t)M * k0, M, B + (size_t)N * k0, N, C, M, k0 ? 1.f : 0.f);
        if (v >= 2) CK(hgemm_rn::gemm(M, N, K, A, M, B, N, C, M, 0.f, v == 3, 128, 0));
        std::vector<float> c((size_t)M * N); CK(cudaMemcpy(c.data(), C, c.size() * 4, cudaMemcpyDeviceToHost));
        double ss = 0, bias = 0;
        for (size_t i = 0; i < c.size(); i++) { const double e = (c[i] - R[i]) / R[i] / eps; ss += e * e; bias += e; }
        printf(" %7.2f / %7.2f |", sqrt(ss / c.size()), bias / c.size());
      }
      printf("\n");
      CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
    }
  }
  // ---- 2. edge cases against cuBLAS (beta = 1, nonzero C)
  {
    int bad = 0, n = 0;
    for (int M : {8, 16, 120, 136, 584, 1968}) for (int K : {1, 2, 16, 34, 274, 1000, 4098, 20000}) for (int bn : {128, 256}) for (int rn : {0, 1}) {
      const int N = 2 * M, lda = M + 8, ldb = N + 16, ldc = M + 4;
      __half *A, *B; float *C0, *C1;
      CK(cudaMalloc(&A, (size_t)lda * K * 2)); CK(cudaMalloc(&B, (size_t)ldb * K * 2));
      CK(cudaMalloc(&C0, (size_t)ldc * N * 4)); CK(cudaMalloc(&C1, (size_t)ldc * N * 4));
      fill<<<256, 256>>>((size_t)lda * K, 1u + K, A); fill<<<256, 256>>>((size_t)ldb * K, 2u + M, B);
      std::vector<float> c((size_t)ldc * N); for (size_t i = 0; i < c.size(); i++) c[i] = (float)((i * 7919) % 1001) - 500.f;
      CK(cudaMemcpy(C0, c.data(), c.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(C1, c.data(), c.size() * 4, cudaMemcpyHostToDevice));
      cublas_gemm(M, N, K, A, lda, B, ldb, C0, ldc, 1.f);
      CK(hgemm_rn::gemm(M, N, K, A, lda, B, ldb, C1, ldc, 1.f, rn, bn, 0));
      std::vector<float> r0(c.size()), r1(c.size());
      CK(cudaMemcpy(r0.data(), C0, c.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r1.data(), C1, c.size() * 4, cudaMemcpyDeviceToHost));
      double mx = 0, md = 0;
      for (int j = 0; j < N; j++) for (int i = 0; i < ldc; i++) {   // padding rows of C (i >= M) must be untouched
        const size_t q = i + (size_t)ldc * j;
        if (i >= M) { if (r1[q] != c[q]) md = 1e30; continue; }
        mx = fmax(mx, fabs(r0[q])); md = fmax(md, fabs((double)r0[q] - r1[q]));
      }
      n++;
      // plain kernel: both it and cuBLAS truncate, in a different order; tolerance 1e-4 at long K
      if (!(md <= (rn ? 1e-5 : 1e-4) * mx)) { bad++; printf("BAD: M %d K %d BN %d rn %d: max |diff| / max %.2e\n", M, K, bn, rn, md / mx); }
      CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C0)); CK(cudaFree(C1));
    }
    printf("\nedge cases: %d of %d agree with cuBLAS\n", n - bad, n);
  }
  // ---- 3. sustained timing, shapes of gemm_mp: P = NG rounded up to 8, X (P x 4nr), Y (2P x 6nr), T (P x 2P)
  {
    const int P = (NG + 7) / 8 * 8, L = 2 * P;
    __half *X, *Y; float *T;
    CK(cudaMalloc(&X, (size_t)P * 4 * nr * 2)); CK(cudaMalloc(&Y, (size_t)L * 6 * nr * 2)); CK(cudaMalloc(&T, (size_t)P * L * 4));
    fill<<<4096, 256>>>((size_t)P * 4 * nr, 1u, X); fill<<<4096, 256>>>((size_t)L * 6 * nr, 2u, Y); CK(cudaDeviceSynchronize());
    std::vector<std::pair<std::string, std::function<void()>>> V;
    V.push_back({"corrections, cuBLAS (beta 0)", [&]() { cublas_gemm(P, L, 4 * nr, X, P, Y + (size_t)L * 2 * nr, L, T, P, 0.f); }});
    V.push_back({"corrections, kernel plain, BN 256", [&]() { CK(hgemm_rn::gemm(P, L, 4 * nr, X, P, Y + (size_t)L * 2 * nr, L, T, P, 0.f, 0, 256, 0)); }});
    V.push_back({"hi*hi, cuBLAS chunks of 512", [&]() { for (int k0 = 0; k0 < 2 * nr; k0 += 512) cublas_gemm(P, L, std::min(512, 2 * nr - k0), X + (size_t)P * k0, P, Y + (size_t)L * k0, L, T, P, 1.f); }});
    V.push_back({"hi*hi, cuBLAS one call (biased)", [&]() { cublas_gemm(P, L, 2 * nr, X, P, Y, L, T, P, 1.f); }});
    for (int bn : {128, 256}) for (int rn : {0, 1})
      V.push_back({"hi*hi, kernel " + std::string(rn == 0 ? "plain" : "RN") + ", BN " + std::to_string(bn),
                   [&, bn, rn]() { CK(hgemm_rn::gemm(P, L, 2 * nr, X, P, Y, L, T, P, 1.f, rn, bn, 0)); }});
    for (int i = 0; i < 3; i++) for (auto &v : V) v.second();
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<std::vector<float>> t(V.size());
    for (int r = 0; r < 9; r++)
      for (size_t v = 0; v < V.size(); v++) {
        float ms; CK(cudaEventRecord(e0)); for (int i = 0; i < 3; i++) V[v].second(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1)); t[v].push_back(ms / 3);
      }
    printf("\nsustained, NG %d (P %d), %d rows per block: median of 9 rounds\n", NG, P, nr);
    for (size_t v = 0; v < V.size(); v++) {
      std::sort(t[v].begin(), t[v].end());
      const double fl = 2.0 * P * L * (v < 2 ? 4.0 : 2.0) * nr;
      printf("%-36s %7.3f ms [%7.3f .. %7.3f]  %6.1f TFLOP/s\n", V[v].first.c_str(), t[v][4], t[v][0], t[v][8], fl / t[v][4] / 1e9);
    }
  }
  return 0;
}
