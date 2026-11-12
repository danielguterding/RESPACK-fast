// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// GEMM microbenchmark for the chi0 products on an AMD GPU (hipBLAS):
//   T(NG,NG) = Bt(NG,K) * A(NG,K)^T, complex single precision, as in chi_gpu_gemm
// variants: CGEMM; CGEMM strided-batched over omega; real 4M as one SGEMM (2NG x NG x 2K);
// 3M (3 SGEMMs NG x NG x K); BF16 / FP16 GemmEx with FP32 accumulation (2NG x NG x 2K, one pass).
// Build (ROCm): hipcc -O3 --offload-arch=<gfx...> -std=c++17 gemmbench.cpp -o gemmbench -lhipblas
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP %s line %d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)
#define CB(x) do { hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("hipBLAS %d line %d\n",(int)s,__LINE__);exit(1);} } while(0)
static hipblasHandle_t h;
template <class F> double timeit(F f, int reps) {
  f(); CK(hipDeviceSynchronize());
  hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
  CK(hipEventRecord(a)); for (int i = 0; i < reps; i++) f(); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
  float ms; CK(hipEventElapsedTime(&ms, a, b)); return ms / 1e3 / reps;
}
static void *dev(size_t bytes) { void *p; CK(hipMalloc(&p, bytes)); CK(hipMemset(p, 0, bytes)); return p; }
int main() {
  CB(hipblasCreate(&h));
  hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0)); printf("%s, %d CUs\n", pr.name, pr.multiProcessorCount);
  {   // warm-up (GPU clocks): about 1 s of CGEMM work
    int n = 2150, k = 2048; hipComplex *a = (hipComplex *)dev((size_t)n * k * 8), *c = (hipComplex *)dev((size_t)n * n * 8);
    hipComplex one = {1.f, 0.f}, zero = {0.f, 0.f};
    for (int i = 0; i < 80; i++) CB(hipblasCgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, n, n, k, &one, a, n, a, n, &zero, c, n));
    CK(hipDeviceSynchronize()); CK(hipFree(a)); CK(hipFree(c));
  }
  int NGs[2] = {583, 2150}, Ks[2] = {512, 2048};
  for (int NG : NGs) for (int K : Ks) {
    double fl = 8.0 * NG * NG * (double)K;   // complex flops of one product
    hipComplex *X = (hipComplex *)dev((size_t)NG * K * 8), *Y = (hipComplex *)dev((size_t)NG * K * 8), *T = (hipComplex *)dev((size_t)NG * NG * 8);
    float *Xr = (float *)dev((size_t)2 * NG * 2 * K * 4), *Yr = (float *)dev((size_t)NG * 2 * K * 4), *Tr = (float *)dev((size_t)2 * NG * NG * 4);
    hipComplex one = {1.f, 0.f}, zero = {0.f, 0.f}; float o = 1.f, z = 0.f;
    int reps = NG > 1000 ? 10 : 50;
    double tc = timeit([&] { CB(hipblasCgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, NG, NG, K, &one, X, NG, Y, NG, &zero, T, NG)); }, reps);
    double ts = timeit([&] { CB(hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, 2 * NG, NG, 2 * K, &o, Xr, 2 * NG, Yr, NG, &z, Tr, 2 * NG)); }, reps);
    double t3 = 3 * timeit([&] { CB(hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_T, NG, NG, K, &o, Xr, NG, Yr, NG, &z, Tr, NG)); }, reps);
    printf("NG %4d K %4d | CGEMM %7.3f ms %5.2f TF | SGEMM-4M %7.3f ms %5.2f TF | 3M %7.3f ms %5.2f TF(eq)", NG, K,
           tc * 1e3, fl / tc / 1e12, ts * 1e3, fl / ts / 1e12, t3 * 1e3, fl / t3 / 1e12);
    for (int hp = 0; hp < 2; hp++) {   // BF16 / FP16 inputs, FP32 accumulate, one pass of the 4M real form
      hipDataType ty = hp ? HIP_R_16F : HIP_R_16BF;
      void *Xh = dev((size_t)2 * NG * 2 * K * 2), *Yh = dev((size_t)NG * 2 * K * 2);
      double th = timeit([&] { CB(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, 2 * NG, NG, 2 * K, &o, Xh, ty, 2 * NG, Yh, ty, NG, &z, Tr,
                                                 HIP_R_32F, 2 * NG, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT)); }, reps);
      printf(" | %s %7.3f ms %6.2f TF", hp ? "FP16" : "BF16", th * 1e3, fl / th / 1e12);
      CK(hipFree(Xh)); CK(hipFree(Yh));
    }
    printf("\n");
    if (K == 512) {   // all 70 frequencies in one strided-batched CGEMM (Y shared: stride 0)
      int nb = NG > 1000 ? 16 : 70;
      hipComplex *XB = (hipComplex *)dev((size_t)NG * K * 8 * nb), *TB = (hipComplex *)dev((size_t)NG * NG * 8 * nb);
      double tb = timeit([&] { CB(hipblasCgemmStridedBatched(h, HIPBLAS_OP_N, HIPBLAS_OP_T, NG, NG, K, &one, XB, NG, (hipblasStride)NG * K,
                                                             Y, NG, 0, &zero, TB, NG, (hipblasStride)NG * NG, nb)); }, NG > 1000 ? 3 : 10);
      printf("            CGEMM strided-batched over %d omega: %7.3f ms per product, %5.2f TF\n", nb, tb / nb * 1e3, fl * nb / tb / 1e12);
      CK(hipFree(XB)); CK(hipFree(TB));
    }
    CK(hipFree(X)); CK(hipFree(Y)); CK(hipFree(T)); CK(hipFree(Xr)); CK(hipFree(Yr)); CK(hipFree(Tr));
  }
  return 0;
}
