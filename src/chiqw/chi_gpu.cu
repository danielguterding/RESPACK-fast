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
// chi_gpu.cu -- CUDA path of m_chi_fast (calc_chiqw), selected with flg_gpu=1.
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

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "chi_gpu: CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
#define CF(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  fprintf(stderr, "chi_gpu: cuFFT error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); return -2; } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "chi_gpu: cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); return -3; } } while (0)

typedef cuDoubleComplex zc;
typedef cuComplex cc;

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
  if (ndev < 1) { fprintf(stderr, "chi_gpu: no CUDA device\n"); return -10; }
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

// phase C: W(rows, nen) and (q=0) V(rows, 3) from the host; kc = rows per CGEMM block
extern "C" int chi_gpu_gemm(const zc *W, const zc *V, int kc) {
  const int NG = g.NG, nen = g.nen, rows = g.rows;
  size_t nA = (size_t)NG * rows, nW = (size_t)rows * nen;
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
