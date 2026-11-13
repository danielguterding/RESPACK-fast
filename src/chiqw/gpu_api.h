// This file is part of the zen-opt fork of RESPACK, a modified version of RESPACK 20240804
// (original: https://sites.google.com/view/kazuma7k6r, Copyright (c) 2017 Kazuma Nakamura et al.).
// Copyright (C) 2026 Daniel Guterding
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, version 3 of the License.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License (GPL/GPLv3.txt) for more details.
//
// gpu_api.h -- one source for the CUDA and the ROCm/HIP path of chi_gpu.cu.
//
// chi_gpu.cu is written against the CUDA runtime, cuFFT, cuBLAS and cuComplex. With CUDA this
// header only includes those. With HIP (RESPACK_HIP=ON in CMake, the file compiled as HIP) it
// maps the names used in chi_gpu.cu one-to-one onto the HIP runtime, hipFFT and hipBLAS. The
// numerics and the precision model are therefore identical on both platforms.

#ifndef RESPACK_GPU_API_H
#define RESPACK_GPU_API_H

#if defined(RESPACK_HIP) || defined(__HIP__)

#ifndef HIPBLAS_V2
#define HIPBLAS_V2   // hipBLAS takes hipComplex/hipDoubleComplex (default from hipBLAS 3 / ROCm 7)
#endif
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hipfft/hipfft.h>
#include <hipblas/hipblas.h>

#define GPU_RT_NAME "HIP"
#define GPU_FFT_NAME "hipFFT"
#define GPU_BLAS_NAME "hipBLAS"

// rows per CGEMM block: rocBLAS accumulates the inner dimension sequentially in FP32, so the error of a
// block grows about linearly with its length (SrVO3, eps^-1 vs CPU double: 3.8e-6 at 2048 rows,
// 8.4e-7 at 512, 2.2e-7 at 128). 512 matches cuBLAS at 2048 rows for about 30 % more GEMM time.
#define GPU_KC_MAX 512

// runtime
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaMemGetInfo hipMemGetInfo
#define cudaSetDevice hipSetDevice
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost(p, n) hipHostMalloc((p), (n), hipHostMallocDefault)
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

// complex numbers
#define cuDoubleComplex hipDoubleComplex
#define cuComplex hipComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define make_cuComplex make_hipComplex
#define cuCreal hipCreal
#define cuCimag hipCimag
#define cuConj hipConj
#define cuCmul hipCmul
#define cuCadd hipCadd
#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuConjf hipConjf
#define cuCmulf hipCmulf

// FFT (hipFFT: same plan interface; CUFFT_INVERSE == HIPFFT_BACKWARD, exp(+i...), unnormalised)
#define cufftHandle hipfftHandle
#define cufftResult hipfftResult
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_Z2Z HIPFFT_Z2Z
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD
#define cufftPlanMany hipfftPlanMany
#define cufftSetStream hipfftSetStream
#define cufftExecZ2Z hipfftExecZ2Z
#define cufftExecC2C hipfftExecC2C
#define cufftDestroy hipfftDestroy

// BLAS
#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define cublasCreate hipblasCreate
#define cublasSetStream hipblasSetStream
#define cublasCgemm hipblasCgemm
#define cublasZgemv hipblasZgemv
#define cublasGemmEx hipblasGemmEx
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_32F HIP_R_32F

#else

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_fp16.h>

#define GPU_RT_NAME "CUDA"
#define GPU_FFT_NAME "cuFFT"
#define GPU_BLAS_NAME "cuBLAS"

#endif

#endif
