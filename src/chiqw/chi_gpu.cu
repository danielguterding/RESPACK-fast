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
// * q=0: Y(G) += sum_r conj(A_rG) w_r v_r, Z(G) += sum_r A_rG w_r conj(v_r) in double precision
//   (ZGEMV); the host assembles the head/wing-replaced chi from chi, Y, Z (and D on the CPU).
// Conventions as cuFFT/FFTW: inverse (backward) unnormalised exp(+i...), forward exp(-i...).
// Fortran array u(n1,n2,n3) <-> cuFFT dims {n3,n2,n1}.

#include "gpu_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "chi_gpu: " GPU_RT_NAME " error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define CF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  fprintf(stderr, "chi_gpu: " GPU_FFT_NAME " error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); return -2; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "chi_gpu: " GPU_BLAS_NAME " error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); return -3; } } while (0)

typedef cuDoubleComplex zc;
typedef cuComplex cc;

static struct {
  int init, dev, verbose;
  int n1, n2, n3, fbatch;
  size_t ngrid;
  cufftHandle plan_b, plan_f;
  int fft_sp;                    // fast_fft_sp=1: single-precision FFTs/products (C2C)
  cufftHandle plan_b4, plan_f4;
  cc *u4; size_t u4_cap;
  cc *f4;
  cublasHandle_t blas;
  cudaStream_t st;
  zc *u; size_t u_cap;           // band functions on the grid, ngrid each
  zc *f;                         // fbatch pair products, ngrid each
  zc *coef; size_t coef_cap;     // packed coefficients (ncoef, nb)
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
  g.n1 = n1; g.n2 = n2; g.n3 = n3; g.fbatch = fbatch > 0 ? fbatch : 16;
  g.ngrid = (size_t)n1 * n2 * n3;
  g.verbose = verbose;
  CK(cudaStreamCreate(&g.st));
  CB(cublasCreate(&g.blas));
  CB(cublasSetStream(g.blas, g.st));
  int dims[3] = {n3, n2, n1};
  CF(cufftPlanMany(&g.plan_b, 3, dims, NULL, 1, (int)g.ngrid, NULL, 1, (int)g.ngrid, CUFFT_Z2Z, 1));
  CF(cufftPlanMany(&g.plan_f, 3, dims, NULL, 1, (int)g.ngrid, NULL, 1, (int)g.ngrid, CUFFT_Z2Z, g.fbatch));
  CF(cufftSetStream(g.plan_b, g.st));
  CF(cufftSetStream(g.plan_f, g.st));
  g.fft_sp = fft_sp;
  if (fft_sp) {
    CF(cufftPlanMany(&g.plan_b4, 3, dims, NULL, 1, (int)g.ngrid, NULL, 1, (int)g.ngrid, CUFFT_C2C, 1));
    CF(cufftPlanMany(&g.plan_f4, 3, dims, NULL, 1, (int)g.ngrid, NULL, 1, (int)g.ngrid, CUFFT_C2C, g.fbatch));
    CF(cufftSetStream(g.plan_b4, g.st));
    CF(cufftSetStream(g.plan_f4, g.st));
    CK(cudaMalloc((void **)&g.f4, g.ngrid * g.fbatch * sizeof(cc)));
  } else {
    CK(cudaMalloc((void **)&g.f, g.ngrid * g.fbatch * sizeof(zc)));
  }
  g.init = 1;
  if (verbose) {
    cudaDeviceProp pr;
    CK(cudaGetDeviceProperties(&pr, g.dev));
    printf(" chi_gpu: local rank %d -> device %d (%s, %.1f GB), grid %d %d %d, FFT batch %d, %s FFTs\n", local_rank(),
           g.dev, pr.name, pr.totalGlobalMem / 1073741824.0, n1, n2, n3, g.fbatch, fft_sp ? "single-precision" : "double-precision");
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
    for (int b = 0; b < nb; b++) {
      cc *ub = u0 + (size_t)b * g.ngrid;
      CF(cufftExecC2C(g.plan_b4, ub, ub, CUFFT_INVERSE));
    }
  } else {
    zc *u0 = g.u + (size_t)ubase * g.ngrid;
    CK(cudaMemsetAsync(u0, 0, g.ngrid * nb * sizeof(zc), g.st));
    k_scatter<<<grid1(n), 256, 0, g.st>>>(n, ncoef, g.coef, g.idx, u0, g.ngrid);
    CK(cudaGetLastError());
    for (int b = 0; b < nb; b++) {
      zc *ub = u0 + (size_t)b * g.ngrid;
      CF(cufftExecZ2Z(g.plan_b, ub, ub, CUFFT_INVERSE));
    }
  }
  // the host buffers (coef, idx) are reused by the caller for the next call
  CK(cudaStreamSynchronize(g.st));
  return 0;
}

// all np pairs of the tile at k-point ik0 (0-based): products, forward FFTs, gather into At
extern "C" int chi_gpu_pairs(int ik0, int NTK, int ubJ, int ubI, double scl) {
  for (int p0 = 0; p0 < g.np; p0 += g.fbatch) {
    int m = g.np - p0 < g.fbatch ? g.np - p0 : g.fbatch;
    size_t n = (size_t)m * g.ngrid;
    size_t ng = (size_t)m * g.NG;
    if (g.fft_sp) {
      k_products4<<<grid1(n), 256, 0, g.st>>>(n, g.ngrid, p0, g.pj, g.pi, g.u4, ubJ, ubI, g.f4);
      CK(cudaGetLastError());
      CF(cufftExecC2C(g.plan_f4, g.f4, g.f4, CUFFT_FORWARD));   // full batch; unused slots ignored
      k_gather4<<<grid1(ng), 256, 0, g.st>>>(ng, g.NG, g.ngrid, p0, g.ext, g.f4, g.At, ik0, NTK, scl);
    } else {
      k_products<<<grid1(n), 256, 0, g.st>>>(n, g.ngrid, p0, g.pj, g.pi, g.u, ubJ, ubI, g.f);
      CK(cudaGetLastError());
      CF(cufftExecZ2Z(g.plan_f, g.f, g.f, CUFFT_FORWARD));   // full batch; unused slots ignored
      k_gather<<<grid1(ng), 256, 0, g.st>>>(ng, g.NG, g.ngrid, p0, g.ext, g.f, g.At, ik0, NTK, scl);
    }
    CK(cudaGetLastError());
  }
  return 0;
}

// phase C: W(rows, nen) and (q=0) V(rows, 3) from the host; kc = rows per CGEMM block
extern "C" int chi_gpu_gemm(const zc *W, const zc *V, int kc) {
  const int NG = g.NG, nen = g.nen, rows = g.rows;
  size_t nA = (size_t)NG * rows, nW = (size_t)rows * nen;
#ifdef GPU_KC_MAX
  if (kc > GPU_KC_MAX) kc = GPU_KC_MAX;   // HIP: shorter blocks for rocBLAS, see gpu_api.h
#endif
  { const char *e = getenv("RESPACK_GPU_KC"); if (e && atoi(e) > 0) kc = atoi(e); }   // rows per CGEMM block (testing)
  if (kc < 1 || kc > rows) kc = rows;
  if (ensure(&g.A4, &g.A4_cap, nA)) return -1;
  if (ensure(&g.W, &g.W_cap, nW)) return -1;
  if (ensure(&g.W4, &g.W4_cap, nW)) return -1;
  if (ensure(&g.Bt, &g.Bt_cap, (size_t)NG * kc)) return -1;
  if (ensure(&g.T4, &g.T4_cap, (size_t)NG * NG)) return -1;
  CK(cudaMemcpyAsync(g.W, W, nW * sizeof(zc), cudaMemcpyHostToDevice, g.st));
  k_d2f<<<grid1(nA), 256, 0, g.st>>>(nA, g.At, g.A4);
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
  CK(cudaGetLastError());
  if (g.lq0) {
    if (ensure(&g.V, &g.V_cap, (size_t)rows * 3)) return -1;
    if (ensure(&g.xv, &g.xv_cap, (size_t)rows)) return -1;
    if (ensure(&g.tmp, &g.tmp_cap, (size_t)NG)) return -1;
    CK(cudaMemcpyAsync(g.V, V, (size_t)rows * 3 * sizeof(zc), cudaMemcpyHostToDevice, g.st));
    const zc zone = make_cuDoubleComplex(1., 0.), zzero = make_cuDoubleComplex(0., 0.);
    for (int ie = 0; ie < nen; ie++) {
      for (int ix = 0; ix < 3; ix++) {
        const zc *w = g.W + (size_t)rows * ie, *v = g.V + (size_t)rows * ix;
        zc *Yc = g.Y + (size_t)NG * (ie * 3 + ix), *Zc = g.Z + (size_t)NG * (ie * 3 + ix);
        // y(G) = sum_r conj(A_rG) w_r v_r = conj( sum_r At(G,r) conj(w_r v_r) )
        k_wv<<<grid1(rows), 256, 0, g.st>>>(rows, w, v, 0, g.xv);
        CB(cublasZgemv(g.blas, CUBLAS_OP_N, NG, rows, &zone, g.At, NG, g.xv, 1, &zzero, g.tmp, 1));
        k_addconj<<<grid1(NG), 256, 0, g.st>>>(NG, g.tmp, Yc);
        // z(G) = sum_r A_rG w_r conj(v_r)
        k_wv<<<grid1(rows), 256, 0, g.st>>>(rows, w, v, 1, g.xv);
        CB(cublasZgemv(g.blas, CUBLAS_OP_N, NG, rows, &zone, g.At, NG, g.xv, 1, &zone, Zc, 1));
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
