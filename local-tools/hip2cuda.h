// Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
// GNU General Public License, version 3 (see GPL/GPLv3.txt).
// The HIP names used by the GEMM tools in local-tools (accerr.cpp, gemmbench.cpp) mapped onto CUDA/cuBLAS, so the
// same sources build for NVIDIA GPUs: compile with -DTOOLS_CUDA, e.g.
//   g++ -O3 -DTOOLS_CUDA -I$CUDA/include accerr.cpp -o accerr -L$CUDA/lib64 -lcublas -lcudart -lopenblas
#ifndef RESPACK_HIP2CUDA_H
#define RESPACK_HIP2CUDA_H
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define hipError_t cudaError_t
#define hipSuccess cudaSuccess
#define hipGetErrorString cudaGetErrorString
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemset cudaMemset
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipDeviceProp_t cudaDeviceProp
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipEvent_t cudaEvent_t
#define hipEventCreate cudaEventCreate
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipComplex cuComplex
#define make_hipComplex make_cuComplex
#define hipDataType cudaDataType
#define HIP_R_16F CUDA_R_16F
#define HIP_R_16BF CUDA_R_16BF
#define HIP_R_32F CUDA_R_32F
#define hipblasHandle_t cublasHandle_t
#define hipblasStatus_t cublasStatus_t
#define HIPBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define hipblasCreate cublasCreate
#define HIPBLAS_OP_N CUBLAS_OP_N
#define HIPBLAS_OP_T CUBLAS_OP_T
#define hipblasSgemm cublasSgemm
#define hipblasCgemm cublasCgemm
#define hipblasCgemmStridedBatched cublasCgemmStridedBatched
#define hipblasStride long long int
#define hipblasGemmEx cublasGemmEx
#define HIPBLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define HIPBLAS_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT
#endif
