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
// hgemm_rn.cuh -- FP16 GEMM on NVIDIA tensor cores (sm_80 and newer) with round-to-nearest accumulation:
//   C(M, N) = A(M, K) B(N, K)^T + beta C,   A, B FP16, C FP32, all column-major (A: lda >= M, M contiguous).
// The tensor cores accumulate with truncation (bias toward zero, growing with K; measured on the L40S:
// -73 eps at K = 512, -371 eps at K = 16384 relative to sum |a b|). Here every mma.sync m16n8k16 starts from
// zero and its 16-term result is added to the FP32 accumulator with an ordinary (round-to-nearest) FADD.
// Requirements: M, N, lda, ldb multiples of 8, A and B 16-byte aligned; any K >= 1. Few output tiles: split K.
// Tiles: BM x BN x 32 per block (BM = 128, BN = 128 or 256), warps of 64 x 64, 4-stage cp.async pipeline,
// XOR-swizzled shared memory read with ldmatrix.trans.
#ifndef RESPACK_HGEMM_RN_CUH
#define RESPACK_HGEMM_RN_CUH
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace hgemm_rn {

constexpr int BK = 32, STAGES = 4;

__device__ __forceinline__ unsigned smem_u32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(unsigned dst, const void *src, bool valid) {
  const int n = valid ? 16 : 0;   // n = 0: zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4_t(unsigned addr, unsigned &r0, unsigned &r1, unsigned &r2, unsigned &r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// d = a b + c, m16n8k16, FP16 inputs, FP32 accumulators
__device__ __forceinline__ void mma16816(float *d, const unsigned *a, const unsigned *b, const float *c) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%10, %11, %12, %13};\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                 "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// shared memory tile: BK rows (k) of W halves (m or n contiguous), 16-byte chunks XOR-swizzled with the row
// (within each 128-byte segment) so that ldmatrix on 8 consecutive rows hits 8 different bank groups
template <int W> __device__ __forceinline__ int swz(int k, int chunk) { return k * (W / 8) + (chunk ^ (k & 7)); }

// RN: 1 = every mma result added with FADD (round to nearest); 0 = plain tensor-core accumulation (reference)
// Main loop: fragments of k16 step s + 1 are loaded from shared memory while the mma of step s run (double
// buffer); a new k tile is waited for, and the next one requested, before the last k16 step of a tile.
// Split K (gridDim.z > 1, for few output tiles): block z takes k in [z kps, (z + 1) kps) and writes its partial
// result to W + z M N (leading dimension M); reduce_kernel adds the partials (round to nearest) and beta C.
template <int BN, int RN>
__global__ void __launch_bounds__(BN == 256 ? 256 : 128)
gemm_kernel(int M, int N, int K, const __half *__restrict__ A, int lda, const __half *__restrict__ B, int ldb,
            float *__restrict__ C, int ldc, float beta, int kps, float *__restrict__ W) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __trap();   // cp.async and mma.sync m16n8k16 need sm_80; the caller checks the compute capability
#else
  if (gridDim.z > 1) {
    const int kz0 = blockIdx.z * kps;
    K = min(K - kz0, kps);
    A += (size_t)kz0 * lda;
    B += (size_t)kz0 * ldb;
    C = W + (size_t)blockIdx.z * M * N;
    ldc = M;
    beta = 0.f;
  }
  constexpr int BM = 128, NWN = BN / 64, NT = 2 * NWN * 32, KS = BK / 16;   // warps: 2 (M) x NWN (N), 64 x 64 each
  constexpr int CA = BK * BM / 8 / NT, CB = BK * BN / 8 / NT;               // 16-byte chunks per thread and tile
  extern __shared__ __align__(128) unsigned char smem[];
  __half *sA = reinterpret_cast<__half *>(smem);                 // STAGES x BK x BM
  __half *sB = sA + STAGES * BK * BM;                            // STAGES x BK x BN
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp % 2, wn = warp / 2;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int nkt = (K + BK - 1) / BK;

  // copy assignment: thread tid copies chunk c = tid % (W / 8) of rows k = tid / (W / 8) + j * RS (RS = NT / (W / 8));
  // since RS is a multiple of 8, the swizzle is the same for all j: one pointer and one offset per matrix
  constexpr int RSA = NT / (BM / 8), RSB = NT / (BN / 8);
  static_assert(RSA % 8 == 0 && RSB % 8 == 0, "copy rows");
  const int kA = tid / (BM / 8), cA = tid % (BM / 8), kB = tid / (BN / 8), cB = tid % (BN / 8);
  const bool vA = m0 + cA * 8 < M, vB = n0 + cB * 8 < N;
  const __half *gA = A + (size_t)kA * lda + (vA ? m0 + cA * 8 : 0), *gB = B + (size_t)kB * ldb + (vB ? n0 + cB * 8 : 0);
  const unsigned sA0 = smem_u32(sA), sB0 = smem_u32(sB);
  const unsigned oA = sA0 + (unsigned)(swz<BM>(kA, cA) * 16), oB = sB0 + (unsigned)(swz<BN>(kB, cB) * 16);
  auto load_tile = [&](int kt) {
    const int k0 = kt * BK, st = kt % STAGES;
#pragma unroll
    for (int j = 0; j < CA; j++) {
      const bool v = vA && k0 + kA + j * RSA < K;
      cp_async16(oA + (unsigned)((st * BK * BM + j * RSA * BM) * 2), v ? gA + (size_t)(k0 + j * RSA) * lda : A, v);
    }
#pragma unroll
    for (int j = 0; j < CB; j++) {
      const bool v = vB && k0 + kB + j * RSB < K;
      cp_async16(oB + (unsigned)((st * BK * BN + j * RSB * BN) * 2), v ? gB + (size_t)(k0 + j * RSB) * ldb : B, v);
    }
  };
  // ldmatrix addresses (byte offsets within a stage) for the k16 step kk: A matrix i = lane / 8: k rows (i / 2) * 8,
  // m offset (i % 2) * 8; B matrix i: k rows (i % 2) * 8, n offset (i / 2) * 8
  const int li = lane >> 3, lr = lane & 7;
  auto frag_load = [&](int st, int kk, unsigned (&af)[4][4], unsigned (&bf)[8][2]) {
    const unsigned bA = sA0 + (unsigned)(st * BK * BM * 2), bB = sB0 + (unsigned)(st * BK * BN * 2);
    {
      const int k = kk * 16 + (li >> 1) * 8 + lr;
#pragma unroll
      for (int mt = 0; mt < 4; mt++) {
        const int m = wm * 64 + mt * 16 + (li & 1) * 8;
        ldsm_x4_t(bA + (unsigned)(swz<BM>(k, m / 8) * 16), af[mt][0], af[mt][1], af[mt][2], af[mt][3]);
      }
    }
    {
      const int k = kk * 16 + (li & 1) * 8 + lr;
#pragma unroll
      for (int np = 0; np < 4; np++) {
        const int n = wn * 64 + np * 16 + (li >> 1) * 8;
        ldsm_x4_t(bB + (unsigned)(swz<BN>(k, n / 8) * 16), bf[2 * np][0], bf[2 * np][1], bf[2 * np + 1][0], bf[2 * np + 1][1]);
      }
    }
  };

  float acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[i][j][r] = 0.f;

#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < nkt) load_tile(s);
    cp_commit();
  }
  cp_wait<STAGES - 2>();
  __syncthreads();
  unsigned af[2][4][4], bf[2][8][2];
  frag_load(0, 0, af[0], bf[0]);
  if (nkt > STAGES - 1) load_tile(STAGES - 1);
  cp_commit();

  const float zero4[4] = {0.f, 0.f, 0.f, 0.f};
  for (int kt = 0; kt < nkt; kt++) {
#pragma unroll
    for (int kk = 0; kk < KS; kk++) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk == KS - 1) {   // the next k tile must be in shared memory before its first fragments are loaded
        cp_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES < nkt) load_tile(kt + STAGES);
        cp_commit();
        frag_load((kt + 1) % STAGES, 0, af[nxt], bf[nxt]);   // harmless past the last tile (stale data)
      } else {
        frag_load(kt % STAGES, kk + 1, af[nxt], bf[nxt]);
      }
#pragma unroll
      for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
          if (!RN) {
            mma16816(acc[mt][nt], af[cur][mt], bf[cur][nt], acc[mt][nt]);
          } else {   // fresh partial tile per k16 step, FADD (round to nearest) into acc
            float t[4];
            mma16816(t, af[cur][mt], bf[cur][nt], zero4);
#pragma unroll
            for (int q = 0; q < 4; q++) acc[mt][nt][q] += t[q];
          }
        }
    }
  }
  cp_wait<0>();

  // epilogue: accumulator (row g, cols 2c, 2c+1) and (row g + 8, ...), g = lane / 4, c = lane % 4
  const int g = lane >> 2, c = lane & 3;
#pragma unroll
  for (int mt = 0; mt < 4; mt++)
#pragma unroll
    for (int nt = 0; nt < 8; nt++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        const int m = m0 + wm * 64 + mt * 16 + g + (r >> 1) * 8, n = n0 + wn * 64 + nt * 8 + 2 * c + (r & 1);
        if (m < M && n < N) {
          float *p = C + (size_t)n * ldc + m;
          *p = beta == 0.f ? acc[mt][nt][r] : acc[mt][nt][r] + beta * *p;
        }
      }
#endif
}

__global__ void reduce_kernel(int M, int N, int S, const float *__restrict__ W, float *__restrict__ C, int ldc, float beta) {
  const size_t mn = (size_t)M * N;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < mn; t += (size_t)gridDim.x * blockDim.x) {
    float sum = 0.f;
    for (int z = 0; z < S; z++) sum += W[(size_t)z * mn + t];
    const size_t n = t / M, m = t - n * M;
    float *p = C + n * ldc + m;
    *p = beta == 0.f ? sum : sum + beta * *p;
  }
}

template <int BN> constexpr size_t smem_bytes() { return (size_t)STAGES * BK * (128 + BN) * sizeof(__half); }

inline int num_sms() {
  static int n = 0;
  if (!n) {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d) != cudaSuccess) n = 1;
  }
  return n;
}
// split-K workspace (one device per process in RESPACK)
inline cudaError_t workspace(size_t n, float **w) {
  static float *buf = nullptr;
  static size_t cap = 0;
  if (n > cap) {
    if (buf) cudaFree(buf);
    buf = nullptr; cap = 0;
    cudaError_t e = cudaMalloc((void **)&buf, n * sizeof(float));
    if (e != cudaSuccess) return e;
    cap = n;
  }
  *w = buf;
  return cudaSuccess;
}

// host launcher; rn = 1: round-to-nearest accumulation, 0: plain tensor-core accumulation (reference)
template <int BN, int RN>
inline cudaError_t gemm_t(int M, int N, int K, const __half *A, int lda, const __half *B, int ldb, float *C, int ldc,
                          float beta, cudaStream_t st) {
  // split K when the output tiles do not fill the GPU (one block per SM): pieces of at least 512
  const long tiles = (long)((M + 127) / 128) * ((N + BN - 1) / BN);
  int S = 1, kps = K;
  if (tiles < num_sms() && K >= 1024) {
    S = (int)((num_sms() + tiles - 1) / tiles);
    if (S > K / 512) S = K / 512;
    kps = ((K + S - 1) / S + BK - 1) / BK * BK;
    S = (K + kps - 1) / kps;
  }
  float *W = nullptr;
  if (S > 1) {
    cudaError_t e = workspace((size_t)S * M * N, &W);
    if (e != cudaSuccess) return e;
  }
  const dim3 grid((M + 127) / 128, (N + BN - 1) / BN, S), block(BN == 256 ? 256 : 128);
  const size_t sm = smem_bytes<BN>();
  static bool attr = false;   // per instantiation; one device per process in RESPACK
  if (!attr) {
    cudaError_t e = cudaFuncSetAttribute(gemm_kernel<BN, RN>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if (e != cudaSuccess) return e;
    attr = true;
  }
  gemm_kernel<BN, RN><<<grid, block, sm, st>>>(M, N, K, A, lda, B, ldb, C, ldc, beta, kps, W);
  if (S > 1) {
    const size_t mn = (size_t)M * N;
    const int nb = (int)(mn / 256 + 1 < 4096 ? mn / 256 + 1 : 4096);
    reduce_kernel<<<nb, 256, 0, st>>>(M, N, S, W, C, ldc, beta);
  }
  return cudaGetLastError();
}
// bn: tile width 128 or 256; 0 = automatic (256 from N = 1024 on)
inline cudaError_t gemm(int M, int N, int K, const __half *A, int lda, const __half *B, int ldb, float *C, int ldc,
                        float beta, int rn, int bn, cudaStream_t st) {
  if (bn == 0) bn = N >= 1024 ? 256 : 128;
  if (bn == 256) return rn ? gemm_t<256, 1>(M, N, K, A, lda, B, ldb, C, ldc, beta, st)
                           : gemm_t<256, 0>(M, N, K, A, lda, B, ldb, C, ldc, beta, st);
  return rn ? gemm_t<128, 1>(M, N, K, A, lda, B, ldb, C, ldc, beta, st) : gemm_t<128, 0>(M, N, K, A, lda, B, ldb, C, ldc, beta, st);
}

}   // namespace hgemm_rn
#endif
