// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// which hipBLASLt kernels serve the two FP16 call types of gemm_mp at NG = 1963 (lean layout)
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <cstdio>
int main() {
  hipblasHandle_t h; hipblasCreate(&h);
  const int NG = 1963, M = NG, N = 2 * NG, K = 16384, kc = 512;
  __half *X, *Y; float *T;
  hipMalloc(&X, (size_t)M * 3 * K * 2); hipMalloc(&Y, (size_t)N * 3 * K * 2); hipMalloc(&T, (size_t)M * N * 4);
  hipMemset(X, 0, (size_t)M * 3 * K * 2); hipMemset(Y, 0, (size_t)N * 3 * K * 2);
  float one = 1.f, zero = 0.f;
  for (int it = 0; it < 3; it++) {
    hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, 2 * K, &one, X, HIP_R_16F, M, Y, HIP_R_16F, N, &zero, T, HIP_R_32F, M,
                  HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);                       // corrections (beta = 0)
    hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, kc, &one, X, HIP_R_16F, M, Y, HIP_R_16F, N, &one, T, HIP_R_32F, M,
                  HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);                       // hi*hi chunk (beta = 1)
  }
  hipDeviceSynchronize();
  printf("done\n");
  return 0;
}
