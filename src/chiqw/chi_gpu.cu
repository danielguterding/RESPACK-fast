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
// chi_gpu.cu -- GPU path of m_chi_fast (calc_chiqw), selected with flg_gpu=1.
// Written against CUDA (cuFFT, cuBLAS); with RESPACK_HIP=ON the same file is compiled as HIP for AMD GPUs
// (ROCm: hipFFT/rocFFT, hipBLAS/rocBLAS), with the names mapped in gpu_api.h. Below, cuFFT/cuBLAS stand
// for hipFFT/hipBLAS in a HIP build.
//
// Same quantity as the CPU path with fast_gemm_sp=1:
//   chi(G,G',w) += sum_{pairs p, k} conj(M_pk(G)) * W_pk(w) * M_pk(G')
// * pair densities M_pk(G): the CPU rotates the wave functions (make_C0) and passes the plane-wave
//   coefficients; here they are placed on the FFT grid, transformed with cuFFT (double precision),
//   multiplied pairwise, forward transformed and the -G_L components gathered (x 1/N).
//   Storage is transposed, At(G, row) with row = ik + NTK*(p-1), so that gathers are coalesced.
// * chi: single-precision CGEMM blocks (K <= kc rows), each block added to a double-precision chi
//   kept on the device -- identical precision model to the validated CPU fast_gemm_sp=1 path.
// * experimental (RESPACK_GPU_GEMM=fp16): the chi products on the matrix units (tensor cores / WMMA)
//   with FP16 inputs and FP32 accumulation. Each factor is scaled by an exact power of two and split
//   into FP16 hi + lo parts; hi*hi + hi*lo + lo*hi are evaluated in one GEMM of the real block form
//   (see chi_gpu_gemm_mp). Same accumulation into the double-precision chi. hi*hi runs in one call of a GEMM
//   kernel with round-to-nearest accumulation: hgemm_rn.cuh (CUDA, sm_80 and newer) or the patched hipBLASLt
//   kernel in fp16_rn/ (HIP, gfx1200); otherwise hipBLAS/cuBLAS in K chunks.
// * q=0: Y(G) += sum_r conj(A_rG) w_r v_r, Z(G) += sum_r A_rG w_r conj(v_r) in double precision
//   (ZGEMV); the host assembles the head/wing-replaced chi from chi, Y, Z (and D on the CPU).
// Conventions as cuFFT/FFTW: inverse (backward) unnormalised exp(+i...), forward exp(-i...).
// Fortran array u(n1,n2,n3) <-> cuFFT dims {n3,n2,n1}.

#include "gpu_api.h"
#ifndef RESPACK_HIP
#include "hgemm_rn.cuh"   // FP16 GEMM with round-to-nearest accumulation (mma.sync, sm_80 and newer)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "chi_gpu: " GPU_RT_NAME " error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define CF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  fprintf(stderr, "chi_gpu: " GPU_FFT_NAME " error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); return -2; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "chi_gpu: " GPU_BLAS_NAME " error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); return -3; } } while (0)

typedef cuDoubleComplex zc;
typedef cuComplex cc;
typedef __half hf;

static struct {
  int init, dev, verbose;
  int n1, n2, n3, fbatch;
  size_t ngrid;
  int fft_sp;                    // fast_fft_sp=1: single-precision FFTs/products (C2C)
  cc *u4; size_t u4_cap;
  cc *f4;
  cublasHandle_t blas;
  cudaStream_t st;
  zc *u; size_t u_cap;           // band functions on the grid, ngrid each
  zc *f;                         // fbatch pair products, ngrid each
  zc *coef; size_t coef_cap;     // packed coefficients (ncoef, nb)
  cc *coef4; size_t coef4_cap;   // the same in single precision (chi_gpu_bands_sp)
  int *idx; size_t idx_cap;      // scatter indices (0-based)
  // per q
  int NG, nen, lq0;
  int *ext;                      // NG gather indices (0-based)
  zc *chi;                       // NG*NG*nen
  zc *Y, *Z;                     // NG*nen*3 (q=0)
  zc *hbuf; size_t hbuf_cap;     // pinned host staging
  // per tile
  int rows, np;
  int *pj, *pi; size_t pj_cap, pi_cap;
  zc *At; size_t At_cap;         // NG*rows (double)
  cc *A4; size_t A4_cap;         // NG*rows (single)
  zc *W; size_t W_cap;           // rows*nen
  cc *W4; size_t W4_cap;
  zc *V; size_t V_cap;           // rows*3
  zc *xv; size_t xv_cap;         // rows
  zc *tmp; size_t tmp_cap;       // NG
  cc *Bt; size_t Bt_cap;         // NG*kc
  cc *T4; size_t T4_cap;         // NG*NG
  // RESPACK_GPU_GEMM=fp16 (gemm_mp)
  int gemm_mp;
  int fp16_chunk;                // K columns per hi*hi GemmEx call (0: one call)
  int fp16_terms;                // 3: hi*hi + hi*lo + lo*hi; 2: X (1) or Y (2) without its lo part
  int fp16_pad;                  // G dimension of the FP16 operands rounded up to a multiple of this (default 8)
  hf *Xh; size_t Xh_cap;         // NG*4kc
  hf *Yh; size_t Yh_cap;         // 2NG*6kc
  float *Tf; size_t Tf_cap;      // NG*2NG
#ifndef RESPACK_HIP
  int rn_cuda;                   // hi*hi with hgemm_rn (compute capability >= 8.0, RESPACK_GPU_FP16_RN != 0)
#endif
#ifdef RESPACK_FP16_RN
  hipModule_t rn_mod;            // fp16_rn/gemm_fp16_rn_gfx1200.s, loaded on gfx1200 (RESPACK_GPU_FP16_RN != 0)
  hipFunction_t rn_fn;           // NULL: hi*hi in chunks with hipBLAS
#endif
  unsigned int *amax;            // max |component| of At (float bits)
  double *mW; size_t mW_cap;     // host: max |component| of W(:, ie)
  // pruned forward FFTs of the pair products (per q): only the NG components of the eps sphere are needed
  int prune, nsx, npxy;          // distinct x columns, distinct (x, y) pairs of the sphere
  int *sxl;                      // nsx: x values
  int *pxs, *pxy;                // npxy: index into sxl, y value
  int *gp, *gz;                  // NG: (x, y) pair index and z of each G
  char *Bp; size_t Bp_cap;       // after the x pass: fbatch * nsx * n2 * n3 (y contiguous)
  char *Cp; size_t Cp_cap;       // after the y pass: fbatch * npxy * n3 (z contiguous)
} g;

static int grid1(size_t n) {
  size_t b = (n + 255) / 256;
  if (b > 65535u * 16u) b = 65535u * 16u;
  if (b < 1) b = 1;
  return (int)b;
}

template <typename T>
static int ensure(T **p, size_t *cap, size_t n) {
  if (n <= *cap) return 0;
  if (*p) CK(cudaFree(*p));
  *p = NULL; *cap = 0;
  CK(cudaMalloc((void **)p, n * sizeof(T)));
  *cap = n;
  return 0;
}

// batched 3D FFT plans (backward and forward share a plan) for the batch sizes that occur
static struct { int batch, sp; cufftHandle h; } pcache[32];
static int npcache = 0;
static int get_plan(int batch, int sp, cufftHandle *h) {
  for (int i = 0; i < npcache; i++)
    if (pcache[i].batch == batch && pcache[i].sp == sp) { *h = pcache[i].h; return 0; }
  int dims[3] = {g.n3, g.n2, g.n1};
  cufftHandle p;
  CF(cufftPlanMany(&p, 3, dims, NULL, 1, (int)g.ngrid, NULL, 1, (int)g.ngrid, sp ? CUFFT_C2C : CUFFT_Z2Z, batch));
  CF(cufftSetStream(p, g.st));
  if (npcache < 32) {
    pcache[npcache].batch = batch; pcache[npcache].sp = sp; pcache[npcache].h = p; npcache++;
  }
  *h = p;
  return 0;
}

// m in-place 3D FFTs of consecutive grids: batches of fbatch, the remainder as powers of two
// (at most ~12 distinct plans per precision)
static int fft_many(void *data, int m, int sp, int dir) {
  const size_t es = sp ? sizeof(cc) : sizeof(zc);
  char *d = (char *)data;
  while (m > 0) {
    int b = g.fbatch;
    if (m < g.fbatch) { b = 1; while (2 * b <= m) b *= 2; }
    cufftHandle p;
    if (get_plan(b, sp, &p)) return -2;
    if (sp) { CF(cufftExecC2C(p, (cc *)d, (cc *)d, dir)); }
    else { CF(cufftExecZ2Z(p, (zc *)d, (zc *)d, dir)); }
    d += (size_t)b * g.ngrid * es;
    m -= b;
  }
  return 0;
}

// 1D plans for the pruned FFTs, keyed by (length, batch, precision); flushed when full
static struct { int len, batch, sp; cufftHandle h; } p1cache[64];
static int np1cache = 0;
static int get_plan1(int len, int batch, int sp, cufftHandle *h) {
  for (int i = 0; i < np1cache; i++)
    if (p1cache[i].len == len && p1cache[i].batch == batch && p1cache[i].sp == sp) { *h = p1cache[i].h; return 0; }
  if (np1cache == 64) {   // all queued work must be done before the plans are destroyed
    CK(cudaStreamSynchronize(g.st));
    for (int i = 0; i < np1cache; i++) CF(cufftDestroy(p1cache[i].h));
    np1cache = 0;
  }
  cufftHandle p;
  int n[1] = {len};
  CF(cufftPlanMany(&p, 1, n, NULL, 1, len, NULL, 1, len, sp ? CUFFT_C2C : CUFFT_Z2Z, batch));
  CF(cufftSetStream(p, g.st));
  p1cache[np1cache].len = len; p1cache[np1cache].batch = batch; p1cache[np1cache].sp = sp; p1cache[np1cache].h = p;
  np1cache++;
  *h = p;
  return 0;
}

// in-place 1D FFTs of length len for m pairs, each with `lines` contiguous lines (pairs consecutive);
// batches of fbatch pairs, the remainder as powers of two
static int fft1_many(void *data, int m, int lines, int len, int sp, int dir) {
  const size_t es = sp ? sizeof(cc) : sizeof(zc);
  char *d = (char *)data;
  while (m > 0) {
    int b = g.fbatch;
    if (m < g.fbatch) { b = 1; while (2 * b <= m) b *= 2; }
    cufftHandle p;
    if (get_plan1(len, lines * b, sp, &p)) return -2;
    if (sp) { CF(cufftExecC2C(p, (cc *)d, (cc *)d, dir)); }
    else { CF(cufftExecZ2Z(p, (zc *)d, (zc *)d, dir)); }
    d += (size_t)b * lines * len * es;
    m -= b;
  }
  return 0;
}

// pruned FFT: keep the needed x columns after the x pass, with y contiguous: B[b][y + n2*(sx + nsx*z)]
template <typename T>
__global__ void k_prx(size_t n, int n1, int n2, int n3, int nsx, const int *__restrict__ sxl, const T *__restrict__ f,
                      size_t ngrid, T *__restrict__ B) {
  const size_t per = (size_t)nsx * n2 * n3;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / per, j = t - b * per;
    int y = (int)(j % n2);
    size_t r = j / n2;
    int sx = (int)(r % nsx), z = (int)(r / nsx);
    B[t] = f[b * ngrid + sxl[sx] + (size_t)n1 * (y + (size_t)n2 * z)];
  }
}

// keep the needed (x, y) pairs after the y pass, with z contiguous: C[b][z + n3*p]
template <typename T>
__global__ void k_pry(size_t n, int n2, int n3, int nsx, int npxy, const int *__restrict__ pxs, const int *__restrict__ pxy,
                      const T *__restrict__ B, T *__restrict__ C) {
  const size_t per = (size_t)npxy * n3, perB = (size_t)nsx * n2 * n3;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / per, j = t - b * per;
    int z = (int)(j % n3), p = (int)(j / n3);
    C[t] = B[b * perB + pxy[p] + (size_t)n2 * (pxs[p] + (size_t)nsx * z)];
  }
}

__device__ __forceinline__ zc to_zc(zc v) { return v; }
__device__ __forceinline__ zc to_zc(cc v) { return make_cuDoubleComplex((double)cuCrealf(v), (double)cuCimagf(v)); }

// the NG components after the z pass into At(G, row), x scl
template <typename T>
__global__ void k_gatherp(size_t n, int NG, int n3, int npxy, int p0, const int *__restrict__ gp, const int *__restrict__ gz,
                          const T *__restrict__ C, zc *__restrict__ At, int ik0, int NTK, double scl) {
  const size_t per = (size_t)npxy * n3;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / NG, gg = t - b * NG;
    size_t row = (size_t)ik0 + (size_t)NTK * (size_t)(p0 + b);
    zc v = to_zc(C[b * per + gz[gg] + (size_t)n3 * gp[gg]]);
    At[gg + (size_t)NG * row] = make_cuDoubleComplex(cuCreal(v) * scl, cuCimag(v) * scl);
  }
}

//------------------------------------------------------------------------------------------------
__global__ void k_scatter(size_t n, int ncoef, const zc *__restrict__ coef, const int *__restrict__ idx,
                          zc *__restrict__ u, size_t ngrid) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / ncoef, i = t - b * ncoef;
    u[b * ngrid + idx[i]] = coef[t];
  }
}

__global__ void k_products(size_t n, size_t ngrid, int p0, const int *__restrict__ pj, const int *__restrict__ pi,
                           const zc *__restrict__ u, int ubJ, int ubI, zc *__restrict__ f) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / ngrid, r = t - b * ngrid;
    int p = p0 + (int)b;
    zc a = u[(size_t)(ubI + pi[p]) * ngrid + r];
    zc c = u[(size_t)(ubJ + pj[p]) * ngrid + r];
    f[t] = cuCmul(cuConj(a), c);
  }
}

__global__ void k_gather(size_t n, int NG, size_t ngrid, int p0, const int *__restrict__ ext,
                         const zc *__restrict__ f, zc *__restrict__ At, int ik0, int NTK, double scl) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / NG, gg = t - b * NG;
    size_t row = (size_t)ik0 + (size_t)NTK * (size_t)(p0 + b);
    zc v = f[b * ngrid + ext[gg]];
    At[gg + (size_t)NG * row] = make_cuDoubleComplex(cuCreal(v) * scl, cuCimag(v) * scl);
  }
}

// single-precision FFT variants (fast_fft_sp=1)
// single-precision coefficients (chi_gpu_bands_sp)
__global__ void k_scatter4s(size_t n, int ncoef, const cc *__restrict__ coef, const int *__restrict__ idx,
                            cc *__restrict__ u, size_t ngrid) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / ncoef, i = t - b * ncoef;
    u[b * ngrid + idx[i]] = coef[t];
  }
}
__global__ void k_scatter4(size_t n, int ncoef, const zc *__restrict__ coef, const int *__restrict__ idx,
                           cc *__restrict__ u, size_t ngrid) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / ncoef, i = t - b * ncoef;
    u[b * ngrid + idx[i]] = make_cuComplex((float)cuCreal(coef[t]), (float)cuCimag(coef[t]));
  }
}

__global__ void k_products4(size_t n, size_t ngrid, int p0, const int *__restrict__ pj, const int *__restrict__ pi,
                            const cc *__restrict__ u, int ubJ, int ubI, cc *__restrict__ f) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / ngrid, r = t - b * ngrid;
    int p = p0 + (int)b;
    cc a = u[(size_t)(ubI + pi[p]) * ngrid + r];
    cc c = u[(size_t)(ubJ + pj[p]) * ngrid + r];
    f[t] = cuCmulf(cuConjf(a), c);
  }
}

__global__ void k_gather4(size_t n, int NG, size_t ngrid, int p0, const int *__restrict__ ext,
                          const cc *__restrict__ f, zc *__restrict__ At, int ik0, int NTK, double scl) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t b = t / NG, gg = t - b * NG;
    size_t row = (size_t)ik0 + (size_t)NTK * (size_t)(p0 + b);
    cc v = f[b * ngrid + ext[gg]];
    At[gg + (size_t)NG * row] = make_cuDoubleComplex((double)cuCrealf(v) * scl, (double)cuCimagf(v) * scl);
  }
}

__global__ void k_d2f(size_t n, const zc *__restrict__ x, cc *__restrict__ y) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    y[t] = make_cuComplex((float)cuCreal(x[t]), (float)cuCimag(x[t]));
}

// Bt(G, r) = conj(A4t(G, r0+r)) * W4(r0+r, ie)
__global__ void k_bt(size_t n, int NG, const cc *__restrict__ A4, const cc *__restrict__ W4col, int r0,
                     cc *__restrict__ Bt) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t r = t / NG;
    cc a = A4[t + (size_t)NG * r0];
    Bt[t] = cuCmulf(cuConjf(a), W4col[r0 + r]);
  }
}

__global__ void k_addf2d(size_t n, const cc *__restrict__ x, zc *__restrict__ y) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    y[t] = make_cuDoubleComplex(cuCreal(y[t]) + (double)cuCrealf(x[t]), cuCimag(y[t]) + (double)cuCimagf(x[t]));
}

// mode 0: x = conj(w*v) ; mode 1: x = w*conj(v)
__global__ void k_wv(size_t n, const zc *__restrict__ w, const zc *__restrict__ v, int mode, zc *__restrict__ x) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    x[t] = mode == 0 ? cuConj(cuCmul(w[t], v[t])) : cuCmul(w[t], cuConj(v[t]));
}

__global__ void k_addconj(size_t n, const zc *__restrict__ x, zc *__restrict__ y) {
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    y[t] = cuCadd(y[t], cuConj(x[t]));
}

// --- RESPACK_GPU_GEMM=fp16 -----------------------------------------------------------------------
// max over x of max(|Re|, |Im|), as float bits (non-negative floats order like unsigned ints)
__global__ void k_absmax(size_t n, const zc *__restrict__ x, unsigned int *out) {
  __shared__ float s[256];
  float m = 0.f;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x)
    m = fmaxf(m, fmaxf(fabsf((float)cuCreal(x[t])), fabsf((float)cuCimag(x[t]))));
  s[threadIdx.x] = m;
  __syncthreads();
  for (int o = 128; o > 0; o >>= 1) {
    if ((int)threadIdx.x < o) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax(out, __float_as_uint(s[0]));
}

__device__ __forceinline__ void split_h(float x, hf &h, hf &l) {
  h = __float2half_rn(x);
  l = __float2half_rn(x - __half2float(h));   // the difference is exact in float
}

// Real block form of the complex product T = X Y^T on the Y side (packed once per block):
//   Yt (2NG x 2K) = [[Yr, -Yi], [Yi, Yr]],  Xt (NG x 2K) = [Xr | Xi]  ->  Xt Yt^T = [Re T | Im T] (NG x 2NG)
// Yhat (2NG x 6K, ld 2NG) = [Yt_h | Yt_l | Yt_h] of Y = At(:, r0:r0+K-1) * sY
// P >= NG: layout size of the G index in Xhat, Yhat and T (NG rounded up, see gemm_mp); the padding is not written
__global__ void k_packY(size_t n, int NG, int P, int K, const zc *__restrict__ At, int r0, double sY, hf *__restrict__ Yh) {
  const size_t L = 2 * (size_t)P, B = L * 2 * K;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t r = t / NG, gg = t - r * NG;
    zc a = At[gg + (size_t)NG * (r0 + r)];
    hf hr, lr, hi, li;
    split_h((float)(cuCreal(a) * sY), hr, lr);
    split_h((float)(cuCimag(a) * sY), hi, li);
    const size_t p00 = gg + L * r, p01 = gg + L * (K + r), p10 = P + gg + L * r, p11 = P + gg + L * (K + r);
    const hf nhi = __hneg(hi), nli = __hneg(li);
    Yh[p00] = hr; Yh[p01] = nhi; Yh[p10] = hi; Yh[p11] = hr;
    Yh[B + p00] = lr; Yh[B + p01] = nli; Yh[B + p10] = li; Yh[B + p11] = lr;
    Yh[2 * B + p00] = hr; Yh[2 * B + p01] = nhi; Yh[2 * B + p10] = hi; Yh[2 * B + p11] = hr;
  }
}

// Xhat (NG x 4K, ld NG) = [Xt_h | Xt_l] of X(G, r) = conj(A(G, r)) * W(r0+r) * sX, with A taken from the FP16
// hi + lo parts in Yhat (22 bits; 8 bytes read per element instead of 16, 8 bytes written instead of 24)
__global__ void k_packX(size_t n, int NG, int P, int K, const hf *__restrict__ Yh, int r0, const zc *__restrict__ Wcol,
                        float s, int lo, hf *__restrict__ Xh) {   // lo = 0: hi part only (RESPACK_GPU_FP16_TERMS=2x)
  const size_t L = 2 * (size_t)P, B = L * 2 * K, BX = (size_t)P * 2 * K;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    size_t r = t / NG, gg = t - r * NG;
    const size_t p0 = gg + L * r, p1 = gg + L * (K + r);
    const float ar = __half2float(Yh[p0]) + __half2float(Yh[B + p0]);        // A_r * sY
    const float ai = -(__half2float(Yh[p1]) + __half2float(Yh[B + p1]));     // A_i * sY
    const zc w = Wcol[r0 + r];
    const float wr = (float)cuCreal(w), wi = (float)cuCimag(w);
    hf hr, lr, hi, li;                                                       // s = sX / sY (power of two)
    split_h((ar * wr + ai * wi) * s, hr, lr);
    split_h((ar * wi - ai * wr) * s, hi, li);
    const size_t q0 = gg + (size_t)P * r, q1 = gg + (size_t)P * (K + r);
    Xh[q0] = hr; Xh[q1] = hi;
    if (lo) { Xh[BX + q0] = lr; Xh[BX + q1] = li; }
  }
}

// chi(G, G') += (T(G, G') + i T(G, P+G')) * inv, T(P, 2P) = [Re | Im] in FP32 (leading dimension P)
__global__ void k_accmp(size_t n, int NG, int P, const float *__restrict__ T, double inv, zc *__restrict__ chie) {
  const size_t pp = (size_t)P * P;
  for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
    const size_t c = t / NG, q = t - c * NG + (size_t)P * c;
    chie[t] = make_cuDoubleComplex(cuCreal(chie[t]) + (double)T[q] * inv, cuCimag(chie[t]) + (double)T[q + pp] * inv);
  }
}

#ifdef RESPACK_FP16_RN
// FP16 GEMM with round-to-nearest accumulation (gfx1200): T(M, N) = X(M, K) Y(N, K)^T + beta T, FP32 T, all
// column-major with leading dimensions M (X, T) and N (Y). Derived from a hipBLASLt kernel (MIT License, see
// fp16_rn/); argument block and launch as recorded from hipBLASLt (tile 128 x 128, 128 threads per group).
// The library kernel faults for M < 8: the caller requires M >= 16.
extern "C" const unsigned char respack_fp16_rn_hsaco[];
#define RN_KERNEL "respack_hgemm_nt_rn_gfx1200"
struct __attribute__((packed)) rn_args {
  unsigned int gemm_info, kinfo0, kinfo1, numWG, M, N, batch, K;
  const void *D, *C, *A, *B;
  unsigned int sD0, sD1, sC0, sC1, sA0, sA1, sB0, sB1;
  float alpha, beta;
  const void *scaleAlphaVec, *bias;
  unsigned int biasType, strideBias;
  float actAlpha, actBeta;
  unsigned int actType;
};
static_assert(sizeof(rn_args) == 140, "argument block of the Tensile kernel");
static int rn_gemm(const hf *X, const hf *Y, float *T, int M, int N, int K, float beta) {
  rn_args a;
  memset(&a, 0, sizeof a);
  const unsigned int t0 = (M + 127) / 128, t1 = (N + 127) / 128;
  a.gemm_info = 1; a.kinfo0 = 0x03200001; a.kinfo1 = 0x04010008;
  a.numWG = t0; a.M = M; a.N = N; a.batch = 1; a.K = K;
  a.D = T; a.C = T; a.A = X; a.B = Y;
  a.sD0 = M; a.sD1 = 1; a.sC0 = M; a.sC1 = 1; a.sA0 = M; a.sA1 = 1; a.sB0 = N; a.sB1 = 1;
  a.alpha = 1.f; a.beta = beta;
  size_t sz = sizeof a;
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
  CK(hipModuleLaunchKernel(g.rn_fn, t0, t1, 1, 128, 1, 1, 0, g.st, NULL, extra));
  return 0;
}
#endif

//------------------------------------------------------------------------------------------------
static int local_rank(void) {
  const char *v[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "MV2_COMM_WORLD_LOCAL_RANK"};
  for (int i = 0; i < 4; i++) {
    const char *s = getenv(v[i]);
    if (s && *s) return atoi(s);
  }
  return 0;
}

extern "C" int chi_gpu_init(int n1, int n2, int n3, int fbatch, int verbose, int fft_sp) {
  if (g.init) return 0;
  memset(&g, 0, sizeof(g));
  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) { fprintf(stderr, "chi_gpu: no " GPU_RT_NAME " device\n"); return -10; }
  g.dev = local_rank() % ndev;
  CK(cudaSetDevice(g.dev));
  g.n1 = n1; g.n2 = n2; g.n3 = n3;
  g.ngrid = (size_t)n1 * n2 * n3;
  // grids per batched FFT (bands and pair products): fbatch > 0 from the caller, RESPACK_GPU_FBATCH, or
  // automatic from a 256 MB budget (16 .. 1024); small grids need large batches to amortise the launches
  {
    const char *e = getenv("RESPACK_GPU_FBATCH");
    if (e && atoi(e) > 0) fbatch = atoi(e);
    if (fbatch <= 0) {
      size_t per = g.ngrid * (fft_sp ? sizeof(cc) : sizeof(zc));
      size_t fb = ((size_t)256 << 20) / per;
      fbatch = (int)(fb < 16 ? 16 : (fb > 1024 ? 1024 : fb));
    }
    g.fbatch = fbatch;
  }
  g.verbose = verbose;
  CK(cudaStreamCreate(&g.st));
  CB(cublasCreate(&g.blas));
  CB(cublasSetStream(g.blas, g.st));
  g.fft_sp = fft_sp;
  { cufftHandle p; if (get_plan(g.fbatch, fft_sp, &p)) return -2; }   // plan of the full batch
  if (fft_sp) {
    CK(cudaMalloc((void **)&g.f4, g.ngrid * g.fbatch * sizeof(cc)));
  } else {
    CK(cudaMalloc((void **)&g.f, g.ngrid * g.fbatch * sizeof(zc)));
  }
  { const char *e = getenv("RESPACK_GPU_GEMM"); g.gemm_mp = (e && strcmp(e, "fp16") == 0); }
  { const char *e = getenv("RESPACK_GPU_FP16_CHUNK"); g.fp16_chunk = (e && *e) ? atoi(e) : 512; }
  { const char *e = getenv("RESPACK_GPU_FP16_PAD"); g.fp16_pad = (e && atoi(e) > 0) ? atoi(e) : 8; }
  {   // RESPACK_GPU_FP16_TERMS: 3 (default), 2x (X without lo: hi*hi + hi(X)*lo(Y)), 2y (Y without lo)
    const char *e = getenv("RESPACK_GPU_FP16_TERMS");
    g.fp16_terms = 0;
    if (e && strcmp(e, "2x") == 0) g.fp16_terms = 1;
    if (e && strcmp(e, "2y") == 0) g.fp16_terms = 2;
  }
  CK(cudaMalloc((void **)&g.amax, sizeof(unsigned int)));
#ifdef RESPACK_FP16_RN
  g.rn_fn = NULL;
  if (g.gemm_mp) {   // RESPACK_GPU_FP16_RN=0: chunked hipBLAS path instead
    const char *e = getenv("RESPACK_GPU_FP16_RN");
    hipDeviceProp_t pr;
    CK(hipGetDeviceProperties(&pr, g.dev));
    if (!(e && strcmp(e, "0") == 0) && strncmp(pr.gcnArchName, "gfx1200", 7) == 0 &&
        (pr.gcnArchName[7] == 0 || pr.gcnArchName[7] == ':')) {
      if (hipModuleLoadData(&g.rn_mod, respack_fp16_rn_hsaco) != hipSuccess ||
          hipModuleGetFunction(&g.rn_fn, g.rn_mod, RN_KERNEL) != hipSuccess) {
        fprintf(stderr, "chi_gpu: could not load the FP16 round-to-nearest kernel, using hipBLAS in chunks\n");
        g.rn_fn = NULL;
        (void)hipGetLastError();
      }
    }
  }
#endif
#ifndef RESPACK_HIP
  g.rn_cuda = 0;
  if (g.gemm_mp) {
    const char *e = getenv("RESPACK_GPU_FP16_RN");
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, g.dev));
    g.rn_cuda = !(e && strcmp(e, "0") == 0) && pr.major >= 8;
  }
#endif
  g.init = 1;
  if (verbose) {
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, g.dev));
    printf(" chi_gpu: local rank %d -> device %d (%s, %.1f GB), grid %d %d %d, FFT batch %d, %s FFTs\n", local_rank(),
           g.dev, pr.name, pr.totalGlobalMem / 1073741824.0, n1, n2, n3, g.fbatch, fft_sp ? "single-precision" : "double-precision");
    if (g.gemm_mp) {
      const char *terms = g.fp16_terms == 1 ? "2 terms, X without lo" : g.fp16_terms == 2 ? "2 terms, Y without lo" :
                          "3 terms";
      int rn = 0;
#ifdef RESPACK_FP16_RN
      rn = g.rn_fn != NULL;
#endif
#ifndef RESPACK_HIP
      rn = g.rn_cuda;
#endif
#ifdef RESPACK_HIP
      const char *rn_name = "gfx1200 kernel";
#else
      const char *rn_name = "mma.sync kernel";
#endif
      if (rn)
        printf(" chi_gpu: RESPACK_GPU_GEMM=fp16: chi products with FP16 hi/lo split (%s), FP32 accumulation, hi*hi"
               " in one call with round-to-nearest accumulation (%s; experimental)\n", terms, rn_name);
      else
        printf(" chi_gpu: RESPACK_GPU_GEMM=fp16: chi products with FP16 hi/lo split (%s), FP32 accumulation, hi*hi in"
               " K chunks of %d (experimental)\n", terms, g.fp16_chunk);
    }
    fflush(stdout);
  }
  return 0;
}

extern "C" int chi_gpu_q_begin(int NG, int nen, int lq0, const int *ext) {
  g.NG = NG; g.nen = nen; g.lq0 = lq0;
  if (g.ext) CK(cudaFree(g.ext));
  if (g.chi) CK(cudaFree(g.chi));
  if (g.Y) CK(cudaFree(g.Y));
  if (g.Z) CK(cudaFree(g.Z));
  g.ext = NULL; g.chi = g.Y = g.Z = NULL;
  CK(cudaMalloc((void **)&g.ext, NG * sizeof(int)));
  CK(cudaMemcpy(g.ext, ext, NG * sizeof(int), cudaMemcpyHostToDevice));
  // pruning tables from the gather indices ext = x + n1*(y + n2*z)
  {
    const int n1 = g.n1, n2 = g.n2;
    int *xm = (int *)malloc(n1 * sizeof(int)), *sxl = (int *)malloc(n1 * sizeof(int));
    int *gp = (int *)malloc(NG * sizeof(int)), *gz = (int *)malloc(NG * sizeof(int));
    for (int x = 0; x < n1; x++) xm[x] = -1;
    int nsx = 0;
    for (int i = 0; i < NG; i++) { int x = ext[i] % n1; if (xm[x] < 0) xm[x] = 1; }
    for (int x = 0; x < n1; x++) if (xm[x] > 0) { xm[x] = nsx; sxl[nsx++] = x; }
    int *pm = (int *)malloc((size_t)nsx * n2 * sizeof(int)), *pxs = (int *)malloc((size_t)nsx * n2 * sizeof(int)),
        *pxy = (int *)malloc((size_t)nsx * n2 * sizeof(int));
    for (int i = 0; i < nsx * n2; i++) pm[i] = -1;
    int npxy = 0;
    for (int i = 0; i < NG; i++) {
      int x = ext[i] % n1, y = (ext[i] / n1) % n2, k = xm[x] + nsx * y;
      if (pm[k] < 0) { pm[k] = npxy; pxs[npxy] = xm[x]; pxy[npxy] = y; npxy++; }
      gp[i] = pm[k]; gz[i] = ext[i] / (n1 * n2);
    }
    // work of the y and z passes relative to one full pass; pruning pays off below ~1.6 (two passes + copies)
    double rel = ((double)nsx * n2 * g.n3 + (double)npxy * g.n3) / (double)g.ngrid;
    const char *e = getenv("RESPACK_GPU_PRUNE");
    g.prune = e && *e ? atoi(e) : (rel < 1.6);
    g.nsx = nsx; g.npxy = npxy;
    if (g.prune) {
      if (g.sxl) CK(cudaFree(g.sxl));
      if (g.pxs) CK(cudaFree(g.pxs));
      if (g.pxy) CK(cudaFree(g.pxy));
      if (g.gp) CK(cudaFree(g.gp));
      if (g.gz) CK(cudaFree(g.gz));
      CK(cudaMalloc((void **)&g.sxl, nsx * sizeof(int)));
      CK(cudaMalloc((void **)&g.pxs, npxy * sizeof(int)));
      CK(cudaMalloc((void **)&g.pxy, npxy * sizeof(int)));
      CK(cudaMalloc((void **)&g.gp, NG * sizeof(int)));
      CK(cudaMalloc((void **)&g.gz, NG * sizeof(int)));
      CK(cudaMemcpy(g.sxl, sxl, nsx * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(g.pxs, pxs, npxy * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(g.pxy, pxy, npxy * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(g.gp, gp, NG * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(g.gz, gz, NG * sizeof(int), cudaMemcpyHostToDevice));
      const size_t es = g.fft_sp ? sizeof(cc) : sizeof(zc);
      if (ensure(&g.Bp, &g.Bp_cap, (size_t)g.fbatch * nsx * n2 * g.n3 * es)) return -1;
      if (ensure(&g.Cp, &g.Cp_cap, (size_t)g.fbatch * npxy * g.n3 * es)) return -1;
    }
    if (g.verbose)
      printf(" chi_gpu: NG %d, pruned FFTs %s (x columns %d of %d, (x,y) pairs %d of %d; y+z passes %.2f of a full pass)\n",
             NG, g.prune ? "on" : "off", nsx, n1, npxy, n1 * n2, rel);
    free(xm); free(sxl); free(gp); free(gz); free(pm); free(pxs); free(pxy);
  }
  size_t nchi = (size_t)NG * NG * nen;
  CK(cudaMalloc((void **)&g.chi, nchi * sizeof(zc)));
  CK(cudaMemsetAsync(g.chi, 0, nchi * sizeof(zc), g.st));
  if (lq0) {
    CK(cudaMalloc((void **)&g.Y, (size_t)NG * nen * 3 * sizeof(zc)));
    CK(cudaMalloc((void **)&g.Z, (size_t)NG * nen * 3 * sizeof(zc)));
    CK(cudaMemsetAsync(g.Y, 0, (size_t)NG * nen * 3 * sizeof(zc), g.st));
    CK(cudaMemsetAsync(g.Z, 0, (size_t)NG * nen * 3 * sizeof(zc), g.st));
  }
  return 0;
}

// pj, pi: 0-based band slots of the np pairs of this tile; nu: number of band slots
extern "C" int chi_gpu_tile_begin(int rows, int nu, int np, const int *pj, const int *pi) {
  g.rows = rows; g.np = np;
  if (ensure(&g.At, &g.At_cap, (size_t)g.NG * rows)) return -1;
  if (g.fft_sp) {
    if (ensure(&g.u4, &g.u4_cap, g.ngrid * (size_t)nu)) return -1;
  } else {
    if (ensure(&g.u, &g.u_cap, g.ngrid * (size_t)nu)) return -1;
  }
  if (ensure(&g.pj, &g.pj_cap, (size_t)np)) return -1;
  if (ensure(&g.pi, &g.pi_cap, (size_t)np)) return -1;
  CK(cudaMemcpyAsync(g.pj, pj, np * sizeof(int), cudaMemcpyHostToDevice, g.st));
  CK(cudaMemcpyAsync(g.pi, pi, np * sizeof(int), cudaMemcpyHostToDevice, g.st));
  CK(cudaStreamSynchronize(g.st));
  return 0;
}

// nb band functions into slots ubase..ubase+nb-1: coefficients coef(1:ncoef, b) (leading dim ldc),
// scatter indices idx(1:ncoef) (0-based grid positions), then backward FFT
extern "C" int chi_gpu_bands(int nb, int ldc, const zc *coef, int ncoef, const int *idx, int ubase) {
  if (nb <= 0) return 0;
  if (ensure(&g.coef, &g.coef_cap, (size_t)ncoef * nb)) return -1;
  if (ensure(&g.idx, &g.idx_cap, (size_t)ncoef)) return -1;
  CK(cudaMemcpy2DAsync(g.coef, ncoef * sizeof(zc), coef, (size_t)ldc * sizeof(zc), ncoef * sizeof(zc), nb,
                       cudaMemcpyHostToDevice, g.st));
  CK(cudaMemcpyAsync(g.idx, idx, ncoef * sizeof(int), cudaMemcpyHostToDevice, g.st));
  size_t n = (size_t)ncoef * nb;
  if (g.fft_sp) {
    cc *u0 = g.u4 + (size_t)ubase * g.ngrid;
    CK(cudaMemsetAsync(u0, 0, g.ngrid * nb * sizeof(cc), g.st));
    k_scatter4<<<grid1(n), 256, 0, g.st>>>(n, ncoef, g.coef, g.idx, u0, g.ngrid);
    CK(cudaGetLastError());
    if (fft_many(u0, nb, 1, CUFFT_INVERSE)) return -2;   // batched backward FFTs of the bands
  } else {
    zc *u0 = g.u + (size_t)ubase * g.ngrid;
    CK(cudaMemsetAsync(u0, 0, g.ngrid * nb * sizeof(zc), g.st));
    k_scatter<<<grid1(n), 256, 0, g.st>>>(n, ncoef, g.coef, g.idx, u0, g.ngrid);
    CK(cudaGetLastError());
    if (fft_many(u0, nb, 0, CUFFT_INVERSE)) return -2;   // batched backward FFTs of the bands
  }
  // the host buffers (coef, idx) are reused by the caller for the next call
  CK(cudaStreamSynchronize(g.st));
  return 0;
}

// all np pairs of the tile at k-point ik0 (0-based): products, forward FFTs, gather into At
// as chi_gpu_bands with single-precision coefficients (fast_fft_sp=1), uploaded from pinned host memory
// (chi_gpu_host_alloc): half the transfer volume at the full PCIe rate
extern "C" int chi_gpu_bands_sp(int nb, int ldc, const cc *coef, int ncoef, const int *idx, int ubase) {
  if (nb <= 0) return 0;
  if (!g.fft_sp) return -4;
  if (ensure(&g.coef4, &g.coef4_cap, (size_t)ncoef * nb)) return -1;
  if (ensure(&g.idx, &g.idx_cap, (size_t)ncoef)) return -1;
  CK(cudaMemcpy2DAsync(g.coef4, ncoef * sizeof(cc), coef, (size_t)ldc * sizeof(cc), ncoef * sizeof(cc), nb,
                       cudaMemcpyHostToDevice, g.st));
  CK(cudaMemcpyAsync(g.idx, idx, ncoef * sizeof(int), cudaMemcpyHostToDevice, g.st));
  size_t n = (size_t)ncoef * nb;
  cc *u0 = g.u4 + (size_t)ubase * g.ngrid;
  CK(cudaMemsetAsync(u0, 0, g.ngrid * nb * sizeof(cc), g.st));
  k_scatter4s<<<grid1(n), 256, 0, g.st>>>(n, ncoef, g.coef4, g.idx, u0, g.ngrid);
  CK(cudaGetLastError());
  if (fft_many(u0, nb, 1, CUFFT_INVERSE)) return -2;
  // the host buffers (coef, idx) are reused by the caller for the next call
  CK(cudaStreamSynchronize(g.st));
  return 0;
}

// pinned (page-locked) host memory for the coefficient uploads; NULL on failure
extern "C" void *chi_gpu_host_alloc(size_t nbytes) {
  void *p = NULL;
  if (cudaHostAlloc(&p, nbytes, 0) != cudaSuccess) { (void)cudaGetLastError(); return NULL; }
  return p;
}
extern "C" void chi_gpu_host_free(void *p) {
  if (p) (void)cudaFreeHost(p);
}

extern "C" int chi_gpu_pairs(int ik0, int NTK, int ubJ, int ubI, double scl) {
  for (int p0 = 0; p0 < g.np; p0 += g.fbatch) {
    int m = g.np - p0 < g.fbatch ? g.np - p0 : g.fbatch;
    size_t n = (size_t)m * g.ngrid;
    size_t ng = (size_t)m * g.NG;
    if (g.prune) {   // forward FFT as x pass (all lines), y pass (needed x), z pass (needed (x, y))
      const int n1 = g.n1, n2 = g.n2, n3 = g.n3, nsx = g.nsx, npxy = g.npxy, sp = g.fft_sp;
      void *f = sp ? (void *)g.f4 : (void *)g.f;
      if (sp) k_products4<<<grid1(n), 256, 0, g.st>>>(n, g.ngrid, p0, g.pj, g.pi, g.u4, ubJ, ubI, g.f4);
      else k_products<<<grid1(n), 256, 0, g.st>>>(n, g.ngrid, p0, g.pj, g.pi, g.u, ubJ, ubI, g.f);
      CK(cudaGetLastError());
      if (fft1_many(f, m, n2 * n3, n1, sp, CUFFT_FORWARD)) return -2;
      size_t nb = (size_t)m * nsx * n2 * n3, nc = (size_t)m * npxy * n3;
      if (sp) k_prx<cc><<<grid1(nb), 256, 0, g.st>>>(nb, n1, n2, n3, nsx, g.sxl, g.f4, g.ngrid, (cc *)g.Bp);
      else k_prx<zc><<<grid1(nb), 256, 0, g.st>>>(nb, n1, n2, n3, nsx, g.sxl, g.f, g.ngrid, (zc *)g.Bp);
      if (fft1_many(g.Bp, m, nsx * n3, n2, sp, CUFFT_FORWARD)) return -2;
      if (sp) k_pry<cc><<<grid1(nc), 256, 0, g.st>>>(nc, n2, n3, nsx, npxy, g.pxs, g.pxy, (cc *)g.Bp, (cc *)g.Cp);
      else k_pry<zc><<<grid1(nc), 256, 0, g.st>>>(nc, n2, n3, nsx, npxy, g.pxs, g.pxy, (zc *)g.Bp, (zc *)g.Cp);
      if (fft1_many(g.Cp, m, npxy, n3, sp, CUFFT_FORWARD)) return -2;
      if (sp) k_gatherp<cc><<<grid1(ng), 256, 0, g.st>>>(ng, g.NG, n3, npxy, p0, g.gp, g.gz, (cc *)g.Cp, g.At, ik0, NTK, scl);
      else k_gatherp<zc><<<grid1(ng), 256, 0, g.st>>>(ng, g.NG, n3, npxy, p0, g.gp, g.gz, (zc *)g.Cp, g.At, ik0, NTK, scl);
    } else if (g.fft_sp) {
      k_products4<<<grid1(n), 256, 0, g.st>>>(n, g.ngrid, p0, g.pj, g.pi, g.u4, ubJ, ubI, g.f4);
      CK(cudaGetLastError());
      if (fft_many(g.f4, m, 1, CUFFT_FORWARD)) return -2;
      k_gather4<<<grid1(ng), 256, 0, g.st>>>(ng, g.NG, g.ngrid, p0, g.ext, g.f4, g.At, ik0, NTK, scl);
    } else {
      k_products<<<grid1(n), 256, 0, g.st>>>(n, g.ngrid, p0, g.pj, g.pi, g.u, ubJ, ubI, g.f);
      CK(cudaGetLastError());
      if (fft_many(g.f, m, 0, CUFFT_FORWARD)) return -2;
      k_gather<<<grid1(ng), 256, 0, g.st>>>(ng, g.NG, g.ngrid, p0, g.ext, g.f, g.At, ik0, NTK, scl);
    }
    CK(cudaGetLastError());
  }
  return 0;
}

// RESPACK_GPU_GEMM=fp16: chi(:,:,ie) += sum_r conj(A_rG) W_r,ie A_rG' with FP16 inputs on the matrix units.
// Real block form T(NG, 2NG) = [Re | Im] = Xt(NG, 2K) Yt(2NG, 2K)^T with Xt = [Xr | Xi] (packed per frequency)
// and Yt = [[Yr, -Yi], [Yi, Yr]] (packed once per block); hi/lo split: T = Xt_h Yt_h^T + Xt_h Yt_l^T + Xt_l Yt_h^T
// (lo*lo dropped, ~2^-22). k_packX reads A from the FP16 parts of Yhat, so a frequency costs 16 bytes of
// memory traffic per element of the block instead of 40.
// The matrix units accumulate inside one GEMM call with a bias toward zero (truncation; measured on RDNA4,
// reported for NVIDIA tensor cores): the error grows linearly with K for same-sign sums. The output step
// C = AB + beta*C rounds to nearest. Hence: the small corrections (hi*lo + lo*hi, 2^-11 smaller) in one call
// (beta = 0), then hi*hi in K chunks of fp16_chunk columns, each continued with beta = 1.
// fp16_chunk = 0: hi*hi in one call (biased; for comparison).
// gfx1200 (rn_fn): hi*hi in one call of the round-to-nearest kernel (no bias, no chunks; about 15 % faster per block).
// CUDA, sm_80 and newer (rn_cuda): the same with hgemm_rn.cuh (L40S: 1.6 ms instead of 1.9 ms per block at NG 1963,
// and the chunks are not accurate enough there: the tensor cores truncate 16x more than RDNA4 at K = 512).
// Scales: |Y| <= mA, |Xr|, |Xi| <= 2 mA mW(ie); powers of two such that the scaled maxima stay below 2^13.
// W: host weights of the row range (leading dimension ldw), At: device pair densities of the range, rows rows
static int gemm_mp(const zc *W, int ldw, const zc *At, int rows, int kc) {
  const int NG = g.NG, nen = g.nen;
  // G dimension of Xhat, Yhat and T padded to P (multiple of fp16_pad, default 8): with an odd NG, cuBLAS on the
  // L40S picks unaligned FP16 kernels at 135 instead of 190 TFLOP/s (padding the leading dimensions alone does not
  // help). The padding rows of Xhat/Yhat are not written; they only reach the padding of T, which is not read.
  const int P = (NG + g.fp16_pad - 1) / g.fp16_pad * g.fp16_pad;
  const size_t nA = (size_t)NG * rows;
  CK(cudaMemsetAsync(g.amax, 0, sizeof(unsigned int), g.st));
  int nbl = grid1(nA) < 1024 ? grid1(nA) : 1024;
  k_absmax<<<nbl, 256, 0, g.st>>>(nA, At, g.amax);
  CK(cudaGetLastError());
  unsigned int ua = 0;
  CK(cudaMemcpyAsync(&ua, g.amax, sizeof(ua), cudaMemcpyDeviceToHost, g.st));
  CK(cudaStreamSynchronize(g.st));
  float fa;
  memcpy(&fa, &ua, sizeof(fa));
  const double mA = fa;
  if (!(mA > 0.0)) return 0;
  if ((size_t)nen > g.mW_cap) {
    free(g.mW);
    g.mW = (double *)malloc((size_t)nen * sizeof(double));
    if (!g.mW) return -1;
    g.mW_cap = nen;
  }
  for (int ie = 0; ie < nen; ie++) {
    double m = 0.;
    const zc *w = W + (size_t)ldw * ie;
    for (int r = 0; r < rows; r++) m = fmax(m, fmax(fabs(cuCreal(w[r])), fabs(cuCimag(w[r]))));
    g.mW[ie] = m;
  }
  int ex;
  frexp(mA, &ex);
  const double sY = ldexp(1.0, 13 - ex);
  if (ensure(&g.Xh, &g.Xh_cap, (size_t)P * 4 * kc)) return -1;
  if (ensure(&g.Yh, &g.Yh_cap, (size_t)2 * P * 6 * kc)) return -1;
  if (ensure(&g.Tf, &g.Tf_cap, (size_t)2 * P * P)) return -1;
  const float one = 1.f, zero = 0.f;
  for (int r0 = 0; r0 < rows; r0 += kc) {
    const int nr = rows - r0 < kc ? rows - r0 : kc;
    const size_t nb = (size_t)NG * nr;
    k_packY<<<grid1(nb), 256, 0, g.st>>>(nb, NG, P, nr, At, r0, sY, g.Yh);
    for (int ie = 0; ie < nen; ie++) {
      if (!(g.mW[ie] > 0.0)) continue;
      frexp(2.0 * mA * g.mW[ie], &ex);
      const double sX = ldexp(1.0, 13 - ex);
      k_packX<<<grid1(nb), 256, 0, g.st>>>(nb, NG, P, nr, g.Yh, r0, g.W + (size_t)rows * ie, (float)(sX / sY),
                                           g.fp16_terms != 1, g.Xh);
      // T (P x 2P) = Xhat Yhat^T. Corrections: [Xt_h | Xt_l] [Yt_l | Yt_h]^T (columns 0..4nr-1 of Xhat,
      // 2nr..6nr-1 of Yhat), beta = 0; then hi*hi (columns 0..2nr-1) in chunks, each added with beta = 1.
      const int L = 2 * P, ch = (g.fp16_chunk > 0 && g.fp16_chunk < 2 * nr) ? g.fp16_chunk : 2 * nr;
      {   // corrections: 3 terms [Xt_h | Xt_l] [Yt_l | Yt_h]^T; 2x: Xt_h Yt_l^T only; 2y: Xt_l Yt_h^T only
        const hf *xc = g.Xh, *yc = g.Yh + (size_t)L * 2 * nr;
        int kcor = 4 * nr;
        if (g.fp16_terms == 1) kcor = 2 * nr;
        if (g.fp16_terms == 2) { xc = g.Xh + (size_t)P * 2 * nr; yc = g.Yh + (size_t)L * 4 * nr; kcor = 2 * nr; }
        CB(cublasGemmEx(g.blas, CUBLAS_OP_N, CUBLAS_OP_T, P, L, kcor, &one, xc, CUDA_R_16F, P, yc, CUDA_R_16F, L,
                        &zero, g.Tf, CUDA_R_32F, P, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
      int hh_done = 0;   // hi*hi in one call with round-to-nearest accumulation, else in chunks
#ifdef RESPACK_FP16_RN
      if (g.rn_fn && NG >= 16) {
        if (rn_gemm(g.Xh, g.Yh, g.Tf, P, L, 2 * nr, 1.f)) return -1;
        hh_done = 1;
      }
#endif
#ifndef RESPACK_HIP
      if (g.rn_cuda && P % 8 == 0) {
        CK(hgemm_rn::gemm(P, L, 2 * nr, g.Xh, P, g.Yh, L, g.Tf, P, 1.f, 1, 0, g.st));
        hh_done = 1;
      }
#endif
      if (!hh_done)
      for (int k0 = 0; k0 < 2 * nr; k0 += ch) {
        const int kk = 2 * nr - k0 < ch ? 2 * nr - k0 : ch;
        CB(cublasGemmEx(g.blas, CUBLAS_OP_N, CUBLAS_OP_T, P, L, kk, &one, g.Xh + (size_t)P * k0, CUDA_R_16F, P,
                        g.Yh + (size_t)L * k0, CUDA_R_16F, L, &one, g.Tf, CUDA_R_32F, P, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
      }
      k_accmp<<<grid1((size_t)NG * NG), 256, 0, g.st>>>((size_t)NG * NG, NG, P, g.Tf, 1.0 / (sX * sY),
                                                        g.chi + (size_t)NG * NG * ie);
    }
  }
  CK(cudaGetLastError());
  return 0;
}

// phase C for the rows row0 .. row0+nrw-1 of the tile (0-based): W(ldw, nen) and (q=0) V(ldw, 3) on the host,
// the pair densities of these rows on the device; kc = rows per CGEMM block. The tile can be processed in
// several row ranges (chi_fast.f90 overlaps the weights of the next range with this one).
extern "C" int chi_gpu_gemm(const zc *Wh, const zc *Vh, int kc, int ldw, int row0, int nrw) {
  const int NG = g.NG, nen = g.nen, rows = nrw;
  const zc *At = g.At + (size_t)NG * row0, *W = Wh + row0, *V = Vh + row0;
  size_t nA = (size_t)NG * rows, nW = (size_t)rows * nen;
#ifdef GPU_KC_MAX
  if (!g.gemm_mp && kc > GPU_KC_MAX) kc = GPU_KC_MAX;   // HIP CGEMM: shorter blocks for rocBLAS, see gpu_api.h
#endif
  if (g.gemm_mp) kc = 8192;   // FP16 path: accuracy set by the chunk length, longer blocks = fewer chi updates
  { const char *e = getenv("RESPACK_GPU_KC"); if (e && atoi(e) > 0) kc = atoi(e); }   // rows per CGEMM block (testing)
  if (kc < 1 || kc > rows) kc = rows;
  if (ensure(&g.W, &g.W_cap, nW)) return -1;
  CK(cudaMemcpy2DAsync(g.W, (size_t)rows * sizeof(zc), W, (size_t)ldw * sizeof(zc), (size_t)rows * sizeof(zc), nen,
                       cudaMemcpyHostToDevice, g.st));
  if (g.gemm_mp) {
    if (gemm_mp(W, ldw, At, rows, kc)) return -1;
  } else {
    if (ensure(&g.A4, &g.A4_cap, nA)) return -1;
    if (ensure(&g.W4, &g.W4_cap, nW)) return -1;
    if (ensure(&g.Bt, &g.Bt_cap, (size_t)NG * kc)) return -1;
    if (ensure(&g.T4, &g.T4_cap, (size_t)NG * NG)) return -1;
    k_d2f<<<grid1(nA), 256, 0, g.st>>>(nA, At, g.A4);
    k_d2f<<<grid1(nW), 256, 0, g.st>>>(nW, g.W, g.W4);
    CK(cudaGetLastError());
    const cc one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
    for (int ie = 0; ie < nen; ie++) {
      const cc *W4col = g.W4 + (size_t)rows * ie;
      zc *chie = g.chi + (size_t)NG * NG * ie;
      for (int r0 = 0; r0 < rows; r0 += kc) {
        int nr = rows - r0 < kc ? rows - r0 : kc;
        size_t nb = (size_t)NG * nr;
        k_bt<<<grid1(nb), 256, 0, g.st>>>(nb, NG, g.A4, W4col, r0, g.Bt);
        CB(cublasCgemm(g.blas, CUBLAS_OP_N, CUBLAS_OP_T, NG, NG, nr, &one, g.Bt, NG,
                       g.A4 + (size_t)NG * r0, NG, &zero, g.T4, NG));
        k_addf2d<<<grid1((size_t)NG * NG), 256, 0, g.st>>>((size_t)NG * NG, g.T4, chie);
      }
    }
  }
  CK(cudaGetLastError());
  if (g.lq0) {
    if (ensure(&g.V, &g.V_cap, (size_t)rows * 3)) return -1;
    if (ensure(&g.xv, &g.xv_cap, (size_t)rows)) return -1;
    if (ensure(&g.tmp, &g.tmp_cap, (size_t)NG)) return -1;
    CK(cudaMemcpy2DAsync(g.V, (size_t)rows * sizeof(zc), V, (size_t)ldw * sizeof(zc), (size_t)rows * sizeof(zc), 3,
                         cudaMemcpyHostToDevice, g.st));
    const zc zone = make_cuDoubleComplex(1., 0.), zzero = make_cuDoubleComplex(0., 0.);
    for (int ie = 0; ie < nen; ie++) {
      for (int ix = 0; ix < 3; ix++) {
        const zc *w = g.W + (size_t)rows * ie, *v = g.V + (size_t)rows * ix;
        zc *Yc = g.Y + (size_t)NG * (ie * 3 + ix), *Zc = g.Z + (size_t)NG * (ie * 3 + ix);
        // y(G) = sum_r conj(A_rG) w_r v_r = conj( sum_r At(G,r) conj(w_r v_r) )
        k_wv<<<grid1(rows), 256, 0, g.st>>>(rows, w, v, 0, g.xv);
        CB(cublasZgemv(g.blas, CUBLAS_OP_N, NG, rows, &zone, At, NG, g.xv, 1, &zzero, g.tmp, 1));
        k_addconj<<<grid1(NG), 256, 0, g.st>>>(NG, g.tmp, Yc);
        // z(G) = sum_r A_rG w_r conj(v_r)
        k_wv<<<grid1(rows), 256, 0, g.st>>>(rows, w, v, 1, g.xv);
        CB(cublasZgemv(g.blas, CUBLAS_OP_N, NG, rows, &zone, At, NG, g.xv, 1, &zone, Zc, 1));
      }
    }
    CK(cudaGetLastError());
  }
  // W (host) is rewritten by the caller for the next tile
  CK(cudaStreamSynchronize(g.st));
  return 0;
}

// q != 0: chi_host(NG,NG,nen) += chi ; q = 0: chi_host = chi (overwrite), Y_host, Z_host = Y, Z
extern "C" int chi_gpu_q_end(zc *chi_host, zc *Y_host, zc *Z_host) {
  const int NG = g.NG, nen = g.nen;
  size_t ns = (size_t)NG * NG;
  CK(cudaStreamSynchronize(g.st));
  if (ns > g.hbuf_cap) {
    if (g.hbuf) CK(cudaFreeHost(g.hbuf));
    CK(cudaMallocHost((void **)&g.hbuf, ns * sizeof(zc)));
    g.hbuf_cap = ns;
  }
  for (int ie = 0; ie < nen; ie++) {
    CK(cudaMemcpy(g.hbuf, g.chi + ns * ie, ns * sizeof(zc), cudaMemcpyDeviceToHost));
    zc *h = chi_host + ns * ie;
    if (g.lq0) {
      memcpy(h, g.hbuf, ns * sizeof(zc));
    } else {
      for (size_t i = 0; i < ns; i++) h[i] = cuCadd(h[i], g.hbuf[i]);
    }
  }
  if (g.lq0) {
    CK(cudaMemcpy(Y_host, g.Y, (size_t)NG * nen * 3 * sizeof(zc), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Z_host, g.Z, (size_t)NG * nen * 3 * sizeof(zc), cudaMemcpyDeviceToHost));
  }
  return 0;
}

extern "C" int chi_gpu_available(void) { return 1; }
// total memory of the device used by this rank (MB), for the tile size in chi_fast.f90
extern "C" double chi_gpu_mem_total_mb(void) {
  size_t fr = 0, tot = 0;
  if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) return 0.0;
  return (double)tot / 1048576.0;
}
