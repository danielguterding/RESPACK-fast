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
// chi0_rdna2.h -- fused chi products for AMD RDNA2 GPUs (gfx103x), included by chi_gpu.cu in HIP builds.
//
//   chi(G, G', w) += sum_r conj(A(G, r)) W(r, w) A(G', r)
//
// as one GEMM without BLAS: M = (G, w) with m = G + NG w, N = G', K = rows. The left operand conj(A) W is formed
// while it is written into LDS. Written for integrated RDNA2 GPUs (Radeon 680M/660M, gfx1035), on which rocBLAS has
// no kernels and its gfx1030 CGEMM kernel runs at a fraction of the FP32 rate; this kernel reaches 75-88 % of the
// FP32 FMA peak per cycle (NG 56-96) on a Radeon 680M.
//
// Kernel family k_chi0_rdna2<TN>: workgroup of 128 threads (4 waves of 32), tile 64 x 8TN, thread tile 4 x TN complex
// (thread (tx, ty) = (tid % 8, tid / 8) owns rows ty + 16 i and columns tx + 8 j). The host picks TN per q point so
// that the N tiles cover NG with fewer than 8 padding columns per tile (one tile up to NG = 88).
//  * Loads: raw buffer loads with the per-thread part of the offset in the VGPR offset and the chunk advance in the
//    scalar offset; reads past the end of a buffer return 0 (no branches). The raw values of the next chunk are kept
//    in registers; conj(a) w is formed when they are stored into LDS, one chunk later, so the loads have a whole
//    chunk of FMAs to arrive.
//  * LDS: double-buffered, KC = 8 rows per chunk, one barrier per chunk. The barrier waits for LDS operations only:
//    __syncthreads() would also wait for the outstanding global loads (workgroup-scope fence in WGP mode).
//  * k loop: software-pipelined (operands of row k+1 read before the FMAs of row k), with scheduling boundaries;
//    without them the compiler hoists the LDS reads of all rows of a chunk (register blow-up, lower occupancy).
//  * The store into LDS and the next prefetch are unconditional (one basic block): at the end of a row range the
//    prefetch reads a chunk inside the buffer into a buffer that is not read any more.
//  * Accumulation: FP32 over blocks of L rows, each block sum added to the double-precision chi (the precision model
//    of the hipBLAS path with blocks of L rows). The row range can be split over gridDim.z slabs; slab z > 0 adds
//    into its own copy part + (z - 1) ncopy (no atomics; the copies are added to chi afterwards, k_addparts).
// Requirements: rows a multiple of 8 (zero weights in the padding rows), L and the slab length multiples of 8,
// NG rows 8 bytes < 2^31 and rows nen 8 bytes < 2^31 (32-bit buffer offsets; the caller splits the rows).

#ifndef RESPACK_CHI0_RDNA2_H
#define RESPACK_CHI0_RDNA2_H

#define CHI0_KC 8

#if defined(__GFX10__)
#define CHI0_RSRC_W3 0x31004000   // buffer resource word 3 (gfx10.x): raw buffer, out-of-range reads return 0
// note: the components are taken with __int_as_float; __builtin_bit_cast on an element of the returned vector
// yields element 0 with ROCm 7.2 (clang 22)
__device__ __forceinline__ float2 chi0_ld2(__amdgpu_buffer_rsrc_t r, int voff, int soff) {
  auto v = __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0);
  return make_float2(__int_as_float(v[0]), __int_as_float(v[1]));
}
#endif

template <int TN>
__global__ __launch_bounds__(128)
void k_chi0_rdna2(int NG, int nen, int rows, int L, int Lz, const float2 *__restrict__ A,
                  const float2 *__restrict__ W, double2 *__restrict__ chi, double2 *__restrict__ part, size_t ncopy) {
#if defined(__GFX10__)
  constexpr int KC = CHI0_KC, TM = 4, GX = 8, GY = 16, NT = 128, BM = TM * GY, BN = TN * GX;
  constexpr int SB = BM * KC / NT;                  // B elements per thread per chunk (rows tid / 64 + 2 s)
  constexpr int SY = (BN * KC + NT - 1) / NT;       // Y elements per thread per chunk
  // flat LDS chunks: B element (k, c) at k BM + c, Y element (k, c) at k BN + c; every store goes to
  // base + (tid + NT s); surplus Y elements of the last round land in the padding behind the chunk
  __shared__ float2 Bs[2][BM * KC];
  __shared__ float2 Ys[2][SY * NT];
  const int M = NG * nen, tid = threadIdx.x;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int rbeg = blockIdx.z * Lz, rend = min(rows, rbeg + Lz);
  if (rbeg >= rend) return;
  double2 *out = blockIdx.z == 0 ? chi : part + (size_t)(blockIdx.z - 1) * ncopy;
  const int tx = tid % GX, ty = tid / GX;
  const __amdgpu_buffer_rsrc_t rA = __builtin_amdgcn_make_buffer_rsrc((void *)A, 0, NG * rows * 8, CHI0_RSRC_W3);
  const __amdgpu_buffer_rsrc_t rW = __builtin_amdgcn_make_buffer_rsrc((void *)W, 0, rows * nen * 8, CHI0_RSRC_W3);
  // constant per-thread byte offsets (first chunk of the slab); the chunk advance goes into the scalar offset
  int vA, vW, vY[SY];
  {
    const int m = m0 + tid % BM, G = m % NG, w = m / NG, k = tid / BM;
    vA = (G + NG * (rbeg + k)) * 8;
    vW = m < M ? (rbeg + k + rows * w) * 8 : 0x7ffffff8;   // m beyond M: out of range, reads return 0
  }
#pragma unroll
  for (int s = 0; s < SY; s++) {
    const int e = tid + s * NT, ok = e < BN * KC;
    const int n = min(n0 + (ok ? e % BN : 0), NG - 1), k = ok ? e / BN : 0;
    vY[s] = (n + NG * (rbeg + k)) * 8;
  }
  const int lastq = (rows - rbeg) / KC - 1;   // last chunk inside A and W: prefetches past it are clamped
  int q = 0;                                  // chunk of the next prefetch (relative to rbeg)
  float2 ra[SB], rw[SB], ry[SY];
  auto gload = [&]() {
    const int qq = __builtin_amdgcn_readfirstlane(min(q, lastq));
    const int sa = qq * NG * KC * 8, sw = qq * KC * 8;
#pragma unroll
    for (int s = 0; s < SB; s++) {
      ra[s] = chi0_ld2(rA, vA, sa + s * (NT / BM) * NG * 8);
      rw[s] = chi0_ld2(rW, vW + s * (NT / BM) * 8, sw);
    }
#pragma unroll
    for (int s = 0; s < SY; s++) ry[s] = chi0_ld2(rA, vY[s], sa);
    q++;
  };
  auto lstore = [&](int buf) {
#pragma unroll
    for (int s = 0; s < SB; s++) {
      const float2 a = ra[s], w = rw[s];
      Bs[buf][tid + s * NT] = make_float2(a.x * w.x + a.y * w.y, a.x * w.y - a.y * w.x);   // conj(a) * w
    }
#pragma unroll
    for (int s = 0; s < SY; s++) Ys[buf][tid + s * NT] = ry[s];
  };
  auto barrier = []() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); };
  float2 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) acc[i][j] = make_float2(0.f, 0.f);
  gload();
  lstore(0);
  gload();
  barrier();
  int cur = 0;
  for (int r0 = rbeg; r0 < rend; r0 += L) {
    const int nch = min(L, rend - r0) / KC;
    for (int c = 0; c < nch; c++) {
      float2 b[2][TM], y[2][TN];
#pragma unroll
      for (int i = 0; i < TM; i++) b[0][i] = Bs[cur][ty + i * GY];
#pragma unroll
      for (int j = 0; j < TN; j++) y[0][j] = Ys[cur][tx + j * GX];
#pragma unroll
      for (int k = 0; k < KC; k++) {
        const int p = k & 1;
        if (k + 1 < KC) {
#pragma unroll
          for (int i = 0; i < TM; i++) b[p ^ 1][i] = Bs[cur][(k + 1) * BM + ty + i * GY];
#pragma unroll
          for (int j = 0; j < TN; j++) y[p ^ 1][j] = Ys[cur][(k + 1) * BN + tx + j * GX];
        }
        asm volatile("" ::: "memory");   // scheduling boundary
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
          for (int j = 0; j < TN; j++) {
            acc[i][j].x = fmaf(b[p][i].x, y[p][j].x, fmaf(-b[p][i].y, y[p][j].y, acc[i][j].x));
            acc[i][j].y = fmaf(b[p][i].x, y[p][j].y, fmaf(b[p][i].y, y[p][j].x, acc[i][j].y));
          }
      }
      lstore(cur ^ 1);   // next chunk into the other buffer (free since the last barrier)
      gload();           // chunk after that
      barrier();
      cur ^= 1;
    }
    // block of L rows done: add the FP32 sums to the double-precision chi (this workgroup owns the tile of its slab).
    // The base indices pass through an asm statement so that the 4 TN addresses are computed here and not kept in
    // registers across the row loop.
    int mb = m0 + ty, nb = n0 + tx;
    asm volatile("" : "+v"(mb), "+v"(nb));
#pragma unroll
    for (int i = 0; i < TM; i++) {
      const int mm = mb + i * GY, G = mm % NG, w = mm / NG;
      double2 *crow = out + G + (size_t)NG * NG * w;
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const int nn = nb + j * GX;
        if (mm < M && nn < NG) {
          double2 *cp = crow + (size_t)NG * nn, v = *cp;
          v.x += (double)acc[i][j].x;
          v.y += (double)acc[i][j].y;
          *cp = v;
        }
        acc[i][j] = make_float2(0.f, 0.f);
      }
    }
  }
#endif
}

// chi(i) += sum_c part(i + c n), c = 0 .. ncp-1
__global__ void k_addparts(size_t n, int ncp, const double2 *__restrict__ part, double2 *__restrict__ chi) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    double2 v = chi[t];
    for (int c = 0; c < ncp; c++) { v.x += part[t + c * n].x; v.y += part[t + c * n].y; }
    chi[t] = v;
  }
}

// A4(G, r) = At(G, r0 + r) for r < nr, 0 for nr <= r < nrp (single precision, padded to nrp rows)
__global__ void k_d2f_pad(int NG, int nr, int nrp, const double2 *__restrict__ At, size_t r0, float2 *__restrict__ A4) {
  const size_t n = (size_t)NG * nrp;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    const size_t r = t / NG;
    if (r < (size_t)nr) {
      const double2 a = At[t + (size_t)NG * r0];
      A4[t] = make_float2((float)a.x, (float)a.y);
    } else {
      A4[t] = make_float2(0.f, 0.f);
    }
  }
}

// W4(r, w) = W(r0 + r, w) (leading dimension ldw) for r < nr, 0 for nr <= r < nrp; W4 leading dimension nrp
__global__ void k_w2f_pad(int nen, int nr, int nrp, const double2 *__restrict__ W, int ldw, size_t r0,
                          float2 *__restrict__ W4) {
  const size_t n = (size_t)nrp * nen;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    const size_t w = t / nrp, r = t - w * nrp;
    if (r < (size_t)nr) {
      const double2 a = W[r0 + r + (size_t)ldw * w];
      W4[t] = make_float2((float)a.x, (float)a.y);
    } else {
      W4[t] = make_float2(0.f, 0.f);
    }
  }
}

// q = 0, all frequencies and directions in one pass over At (replaces 6 nen ZGEMV calls: no BLAS on this path),
// double precision:
//   Yt(G, ix, ie) += sum_r At(G, r) conj(w_r,ie v_r,ix)    (Y += conj(Yt) afterwards)
//   Z(G, ix, ie)  += sum_r At(G, r) w_r,ie conj(v_r,ix)
// W: rows x nen, V: rows x 3, Yt and Z: NG x 3 x nen. Thread (G, ie) = (32 consecutive G per wave, 8 frequencies per
// block); gridDim.z splits the rows, partial sums added atomically.
__global__ __launch_bounds__(256)
void k_yz_at(int NG, int rows, int nen, const double2 *__restrict__ At, const double2 *__restrict__ W,
             const double2 *__restrict__ V, double2 *__restrict__ Yt, double2 *__restrict__ Z) {
  const int G = blockIdx.x * 32 + threadIdx.x % 32, ie = blockIdx.y * 8 + threadIdx.x / 32;
  if (G >= NG || ie >= nen) return;
  const int per = (rows + gridDim.z - 1) / gridDim.z, rb = blockIdx.z * per, re = min(rows, rb + per);
  if (rb >= re) return;
  double2 y[3], z[3];
  for (int ix = 0; ix < 3; ix++) { y[ix] = make_double2(0., 0.); z[ix] = make_double2(0., 0.); }
  for (int r = rb; r < re; r++) {
    const double2 a = At[G + (size_t)NG * r], w = W[r + (size_t)rows * ie];
#pragma unroll
    for (int ix = 0; ix < 3; ix++) {
      const double2 v = V[r + (size_t)rows * ix];
      const double2 wv = make_double2(w.x * v.x - w.y * v.y, w.x * v.y + w.y * v.x);       // w v
      const double2 wc = make_double2(w.x * v.x + w.y * v.y, w.y * v.x - w.x * v.y);       // w conj(v)
      y[ix].x += a.x * wv.x + a.y * wv.y; y[ix].y += a.y * wv.x - a.x * wv.y;                // a conj(w v)
      z[ix].x += a.x * wc.x - a.y * wc.y; z[ix].y += a.x * wc.y + a.y * wc.x;                // a w conj(v)
    }
  }
#pragma unroll
  for (int ix = 0; ix < 3; ix++) {
    const size_t o = G + (size_t)NG * (ie * 3 + ix);
    atomicAdd(&Yt[o].x, y[ix].x); atomicAdd(&Yt[o].y, y[ix].y);
    atomicAdd(&Z[o].x, z[ix].x); atomicAdd(&Z[o].y, z[ix].y);
  }
}

#endif
