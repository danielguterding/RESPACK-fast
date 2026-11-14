# RESPACK 20240804 — `zen-opt` fork

Unofficial fork of [RESPACK](https://sites.google.com/view/kazuma7k6r) 20240804 (GPLv3) with a restructured
polarization and interaction code. Branch `upstream` holds the unmodified release; branch `zen-opt`
holds the changes. Every change can be switched off at run time, which restores the original code paths.

## What changed

| program | change | switch |
|---|---|---|
| `calc_chiqw` | new band-pair loop (`src/chiqw/chi_fast.f90`), see below | `flg_fast` in `&param_chiqw` (default 1) |
| `calc_chiqw` | faster generalized-tetrahedron weights (`src/chiqw/m_tetrahedron_fast.F`) | used by `flg_fast=1` |
| `calc_chiqw` | optional GPU path (`src/chiqw/chi_gpu.cu`) for NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs, see “GPU path” | `flg_gpu` (default 0), build with `-DRESPACK_CUDA=ON` or `-DRESPACK_HIP=ON` |
| `calc_chiqw` | wave-function rotation: index map and phase factors of `make_C0` once per k point instead of once per band (`make_C0_map`); results identical to ≤ 10⁻¹⁵ | used by `flg_fast=1` |
| `calc_chiqw` | ε(ω) built and inverted for all frequencies in parallel (OpenMP); each inversion uses single-threaded LAPACK as before (results identical to ≤ 10⁻¹⁵; SrVO₃, NG = 1963: 88 → 51 s per q) | always on |
| `calc_chiqw` | post-processing of a q point (ε, ε⁻¹, files of `dir-eps/qNNN`) in `m_eps_async`, optionally in a background thread while the next q point is computed; MPI sum of χ in place (one array of NG²·N_ω less) | `flg_async_eps` in `&param_chiqw` (default −1 = auto: off in the CPU build) |
| `calc_w3d`, `calc_j3d` | cached real-space Wannier functions + FFTW, ZGEMM contractions (`src/calc_int/int_fast.f90`) | `flg_fast` in `&param_calc_int` (default 1) |
| `calc_w3d`, `calc_j3d` | `make_LG0` loops limited to the cutoff sphere; `exp(-iq.R)` tables | always on; gives numerically identical results |

### `calc_chiqw` band-pair loop (`m_chi_fast`)

The original loop does the following for every (occupied j, virtual i) pair and every k:
- rotates both wave functions by symmetry;
- transforms both to real space with the built-in FFT;
- applies the umklapp phase with one complex `exp` per grid point;
- forms the product and forward-transforms it;
- accumulates 70 small ZGEMMs into a per-thread copy of χ(G,G′,ω).

The fork computes the same quantity this way:

1. **Tiles of pairs.** Pairs are processed in tiles (a block of occupied × a block of virtual bands). For
   every k, all bands of a tile are rotated and transformed once: (n_occ + n_vir) backward FFTs instead of
   2·n_occ·n_vir.
2. **FFTW.** FFTW, planned once, replaces the built-in FFT, with the identical sign and normalization
   convention.
3. **Exact umklapp shift.** The phase e^{−2πiΔG·r/N} is applied by placing C(G) at grid index G−ΔG
   before the FFT. This is mathematically identical and costs no `exp`.
4. **One ZGEMM per frequency.** The sums over k and over the pairs of a tile are fused into one ZGEMM per
   frequency, with K = NTK·n_pairs. It runs parallel over frequencies into one shared χ, so there are no
   per-thread copies. That saves memory: the original needed n_threads × NG² × n_ω × 16 bytes.
5. **q = 0.** The three head/wing variants differ only in the G = 0 row and column. They are computed as
   one ZGEMM plus ZGEMV corrections instead of three ZGEMMs.
6. **Tetrahedron weights** (`m_tetrahedron_fast`):
   - Every log((g_a+ω)/(g_b+ω)) is computed as ½log(|a|²/|b|²) + i·atan2(a·b̄) in a SIMD loop, so it uses
     libmvec's AVX2 `log`/`atan2` instead of scalar `clog`.
   - Both signs (dsgn = ±1) are done in one pass, so the Fermi-surface cuts are computed once.
   - Non-degenerate tetrahedra use a fused kernel: 6 instead of 12 logarithms, frequency-independent
     denominators inverted once, and powers as explicit products.

`fast_gemm_sp` (default 0): if 1, the χ⁰ accumulation runs as single-precision CGEMM blocks of
about 4096 rows, and each block result is added to a double-precision χ. This halves the GEMM time on
CPUs and is the precision model for a GPU port.

| test | ε⁻¹ vs `fast_gemm_sp=0` | U(ω), J(ω) | static U, U′, J |
|---|---|---|---|
| SrVO₃ cRPA | 4×10⁻⁷ relative | ≤ 1×10⁻⁵ eV | identical |
| Si RPA, Si cRPA | ≤ 4×10⁻⁷ relative | ≤ 1×10⁻⁵ eV | identical |

These deviations are about 200× smaller than the method's own rounding noise.

`fast_fft_sp` (default −1 = auto: 0 on the CPU, 1 with `flg_gpu=1`): if 1, the pair densities are computed with single-precision FFTs (`fftwf`)
and stored in double precision. The q = 0 k·p terms stay in double precision. This needs `libfftw3f`.

| test (with `fast_gemm_sp=1`) | ε⁻¹ vs double precision | U(ω), J(ω) | static U, U′, J |
|---|---|---|---|
| SrVO₃ cRPA | 1.8×10⁻⁶ relative | ≤ 2×10⁻⁵ eV | identical |
| Si cRPA | 2.8×10⁻⁷ relative | ≤ 1×10⁻⁵ eV | identical |

On a CPU this gives about 1.3× for the pair densities on small grids that fit in cache. The large gain
is on GPUs.

`fast_mem_mb` (default 4096) is the memory budget for one tile of pair densities and weights. A larger
budget means larger tiles and fewer backward FFTs.

7. **Pruned forward FFTs.** Of each pair product only the NG components of the ε sphere are needed. The forward
   transform is done as:
   - x pass over all lines;
   - gather of the needed x columns (y contiguous), then the y pass;
   - gather of the needed (x, y) pairs (z contiguous), then the z pass.

   This is used when the y and z passes are below 1.6 full passes (`RESPACK_CPU_PRUNE=0/1` forces it). Results
   are unchanged (Si bitwise, SrVO₃ 1×10⁻¹⁵). SrVO₃ (`Ecut_for_eps` 5 Ry, all q): pair densities 45.7 s
   instead of 58.9 s, χ⁰ total 139.8 s instead of 152.6 s.
8. **Single-precision wave functions (`fast_wfn_sp=1`).** The wave functions are stored as complex(4), which
   halves their memory; rotation and everything after it stay in double precision. Default −1 (auto): single
   precision together with single-precision FFTs (`fast_fft_sp=1`), otherwise double precision. ε⁻¹ deviates by 7×10⁻⁸ (Si) and 2×10⁻⁸ (SrVO₃), U(ω), J(ω) by at most 1×10⁻⁵ eV.
   `RESPACK_WFN_DP=1` forces double precision.
9. **One rank per q group (`MPI_num_qcomm` = number of ranks).**
   - The band ring exchange within a q group is skipped. With one rank it only copied all virtual bands to the
     rank itself, through a full-size buffer; that saves one complete copy of the virtual bands.
   - The broadcast of the wave functions across q groups is skipped, since every rank has read them itself.
   - With `fast_wfn_sp=1`, the ranks of a node share one copy of the wave functions (MPI-3 shared memory,
     `m_wfn_shared`; `RESPACK_WFN_SHARED=0` gives private copies). Only node rank 0 reads them.

### `calc_w3d` / `calc_j3d` (`m_int_fast`)

- **Cached real-space functions.** u_{w,k}(r) of all Wannier-gauge Bloch functions are transformed once and
  cached (NWF × NTK × N_grid × 16 bytes, limited by `fast_mem_mb`, default 8192). Each (q, k, pair) then
  needs one product and one forward FFT; the original did three FFTs per call.
- **`calc_j3d` loop order.** q is the outer loop and the lattice vectors R the inner one: the pair densities
  are computed once per q for all requested R. The original recomputed every FFT for each R.
- **ZGEMM contractions.** ρᵀ ε⁻¹ ρ* is evaluated as ZGEMMs instead of scalar quadruple loops, which matters
  for large NG and many Wannier functions.

## Modified files (GPLv3 §5a; modifications by Daniel Guterding, November 2026)

| file | status | change |
|---|---|---|
| `src/chiqw/chi_fast.f90` | new | restructured band-pair loop (module `m_chi_fast`) |
| `src/chiqw/m_tetrahedron_fast.F` | new (derived from `m_tetrahedron_20170325.F`) | vectorised log ratios, fused ± pass, fused normal tetrahedra |
| `src/chiqw/chiqw_sub.F` | modified | new `make_C0_map` (per-k rotation map; `make_C0` unchanged) |
| `src/chiqw/wfn_shared.f90` | new | module `m_wfn_shared` (wave functions shared by the ranks of a node, MPI-3 shared memory) |
| `src/chiqw/eps_async.f90`, `thread_util.c` | new | module `m_eps_async` (q ≠ 0 post-processing, background thread via pthreads) |
| `src/chiqw/wfn_shared.f90` | new | module `m_wfn_shared` (wave functions shared by the ranks of a node, MPI-3 shared memory) |
| `src/chiqw/chiqw.F`, `config.h` | modified | calls `m_chi_fast` for `flg_fast=1`, original loops kept for `flg_fast=0`; frequency loops of the ε inversion parallel (OpenMP); single-precision wave functions (`fast_wfn_sp`), shared on the node; no band ring exchange with one rank per q group |
| `src/chiqw/m_rdinput.f90` | modified | inputs `flg_fast`, `fast_mem_mb`, `flg_async_eps`, `fast_wfn_sp` |
| `src/chiqw/CMakeLists.txt` | modified | FFTW3, new sources |
| `src/calc_int/int_fast.f90` | new | module `m_int_fast` |
| `src/calc_int/calc_intW.F`, `calc_intJ.F` | modified | call `m_int_fast` for `flg_fast=1`; original loops kept; exp(−iq·R) table in `calc_intW.F` |
| `src/calc_int/calc_int_sub.F` | modified | `make_LG0` loop bounds (identical G lists) |
| `src/calc_int/m_rdinput.f90`, `config_w3d.h`, `config_j3d.h`, `CMakeLists.txt` | modified | inputs, declarations, FFTW3 |
| `CMakeLists.txt` (top level and all subdirectories) | modified | modern target-based CMake (3.18 to 4.x): imported MPI/OpenMP/LAPACK/FFTW3 targets, CPU and LTO options |
| `cmake/FindFFTW3.cmake`, `CMakePresets.json` | new | FFTW3 finder, build presets |
| `src/chiqw/chi_gpu.cu`, `chi_gpu_stub.c` | new | GPU path of the band-pair loop, stub without a GPU build |
| `src/chiqw/gpu_api.h` | new | maps the CUDA names in `chi_gpu.cu` onto HIP/hipFFT/hipBLAS for the ROCm build |
| `src/chiqw/hgemm_rn.cuh` | new | FP16 GEMM with round-to-nearest accumulation for NVIDIA GPUs (sm_80 and newer) |
| `src/chiqw/fp16_rn/` | new | FP16 GEMM kernel with round-to-nearest accumulation for gfx1200 (patched hipBLASLt kernel, MIT License: `LICENSE-hipBLASLt.txt`), embedding (`fp16_rn_blob.c`) |
| `local-tools/*`, `README-zen-opt.md` | new | build script, validation and benchmark tools, this document |

All changed places in modified files are marked with `fork` comments. The full diff is
`git diff upstream zen-opt`.

## Validation (against the unmodified code, same compiler flags)

| test | quantity | deviation fork vs original |
|---|---|---|
| SrVO₃ t₂g cRPA (6³ k, 55 bands, 70 ω) | ε⁻¹(G,G′,ω), all 20 q | ≤ 7×10⁻⁵ rel. (= original −O2 vs −O3 spread; only at ω > 99 eV) |
| | U(ω), J(ω) | ≤ 2×10⁻⁴ eV; static U, U′, J identical to 5 digits |
| Si sp³ RPA and cRPA (nonsymmorphic, time reversal) | ε⁻¹, all 16 q | ≤ 6×10⁻⁶ rel. |
| | U(ω=0) | identical to 5 digits |
| calc_w3d/j3d on identical ε⁻¹ | V, W, X, J | ≤ 7×10⁻⁹ eV |
| tetrahedron weights (model bands) | error vs quad-precision evaluation of the original formulas | identical to the original's own error (1×10⁻⁵ metallic, 4×10⁻⁸ insulating) |

The 10⁻⁵ level is set by the double-precision tetrahedron formulas themselves; the quad-precision test in
`local-tools/acc_tetra_quad.f90` shows the original has the same error.

## Performance (AMD Ryzen 7 PRO 6850U, 8 cores, both built -O3 -march=znver3 -flto)

| | original | fork | speed-up |
|---|---|---|---|
| SrVO₃ `calc_chiqw` (20 q, cRPA) | 1197 s | 169 s | 7.1× |
| SrVO₃ `calc_w3d` / `calc_j3d` | 33 s / 58 s | 9 s / 14 s | 3.7× / 4.3× |
| SrVO₃ whole chain incl. `calc_wannier` | ≈ 1310 s | ≈ 200 s | 6.6× |
| Si `calc_chiqw` (16 q, RPA) | 148 s | 21 s | 7.1× |
| Si `calc_w3d` / `calc_j3d` | 37 s / 119 s | 8 s / 15 s | 4.9× / 8.2× |

Remaining `calc_chiqw` time for SrVO₃ (sum over q): pair densities 64 s, tetrahedron weights 46 s,
ZGEMM 51 s. Use 1 MPI process × 8 OpenMP threads (2 × 4 is ≈ 10 % slower; results identical to 1e-13).

## GPU path (`calc_chiqw`, branch `gpu-cuda`)

`flg_gpu=1` in `&param_chiqw` moves the two expensive parts of the band-pair loop to an NVIDIA GPU
(`src/chiqw/chi_gpu.cu`, CUDA + cuFFT + cuBLAS):

- **Pair densities.** The CPU rotates the wave functions (`make_C0`) and uploads the plane-wave
  coefficients. On the GPU they are placed on the FFT grid and transformed with cuFFT. Then come the pair
  products and the batched forward FFTs, and the needed G-components are gathered into a transposed tile
  `At(G, row)`, which makes the writes contiguous.
- **χ⁰ accumulation.** cuBLAS CGEMM blocks of 2048 rows, each added to a double-precision χ(G,G′,ω) kept on
  the GPU. This is the same precision model as `fast_gemm_sp=1` on the CPU.
- **q = 0 head/wing terms.** Double-precision ZGEMV on the GPU, summed over tiles and assembled on the host.
- **Tetrahedron weights.** These stay on the CPU.

The other switches also apply on the GPU:
- `fast_fft_sp` runs the pair-density FFTs in single precision (cuFFT C2C, 3× faster on an L40S). It defaults to
  −1 (auto), which means on with `flg_gpu=1` and off on the CPU. ε⁻¹ changes by about 3×10⁻⁶. Set
  `fast_fft_sp=0` for double-precision FFTs on the GPU.
- `fast_mem_mb` sets the tile size.

**Build:**

    cmake --preset native -B build/native-cuda -DRESPACK_CUDA=ON \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = L40S/Ada
    cmake --build build/native-cuda -j

Without `RESPACK_CUDA`, a stub is compiled and `flg_gpu=1` stops with a message. The default architecture
list is `all-major`.

**Running:**
- **One MPI rank per GPU.** Rank n uses visible GPU n (local rank modulo device count).
- **Under Slurm, start `mpirun` inside an `srun` step or batch job**, so that `CUDA_VISIBLE_DEVICES` and the
  device cgroup confine the processes to the allocated GPUs:

      srun -c 8 --gres=gpu:2 env OMP_NUM_THREADS=4 mpirun -np 2 --oversubscribe --bind-to none calc_chiqw < input.in

  A `mpirun` started from an `salloc` shell sees all GPUs of the node.
- **BLAS threads:** keep `OPENBLAS_NUM_THREADS=1`.
- **Pruned forward FFTs:** after the forward FFT of a pair product, only the NG components of the ε sphere
  are needed. The FFT runs as 1D passes: x for all lines, y only for the x columns that occur, z only for the
  (x, y) pairs that occur. It is on automatically when it saves work; `RESPACK_GPU_PRUNE=0/1` forces it. Measured
  on the RX 9060 XT for the SrVO₃ chain (NG = 81, 24³ grid; 5 of 24 x columns and 21 of 576 (x, y) pairs
  needed): pair densities 15.1 → 9.8 s with single-precision FFTs and 31.3 → 19.1 s with double-precision FFTs.
  Pruned and full FFTs agree to 1×10⁻¹² (double) and 1×10⁻⁷ (single). With a large ε sphere (NG = 1963 on
  the same grid) there is nothing to prune. For a large cell with a small `Ecut_for_eps`, where the sphere
  covers only about 10 % of each grid direction, the y and z passes shrink to about 15 %.
- **ε inversion in the background:** with `flg_gpu=1`, `flg_async_eps` defaults to on (auto). The inversion of
  ε(ω) and the output of a q point run on the CPU while the GPU computes χ⁰ of the next q point. Only the last
  q point's inversion stays visible. Measured on an RX 9060 XT (SrVO₃, NG = 1963, 3 q points, 8 threads): the
  run takes 116.5 s instead of 137.1 s, and the results are identical (ε⁻¹ bitwise). Set `flg_async_eps=0` to
  switch it off.

### AMD GPUs (ROCm/HIP, branch `gpu-hip`)

The same `chi_gpu.cu` is compiled as HIP. `src/chiqw/gpu_api.h` maps the CUDA runtime, cuFFT, cuBLAS and
cuComplex names one-to-one onto the HIP runtime, hipFFT (rocFFT) and hipBLAS (rocBLAS), so the kernels, the
precision model and all switches (`flg_gpu`, `fast_fft_sp`, `fast_mem_mb`, `RESPACK_GPU_KC`) are the same as
on NVIDIA GPUs. The kernels use no warp-level operations, so the wavefront size (32 or 64) does not matter.

**Build** (ROCm ≥ 6 with hipFFT and hipBLAS, CMake ≥ 3.21):

    cmake --preset hip -B build/hip -DCMAKE_HIP_ARCHITECTURES=gfx1200   # e.g. gfx1200 = RX 9060/9070, gfx942 = MI300
    cmake --build build/hip -j

Without `CMAKE_HIP_ARCHITECTURES`, CMake builds for the GPU(s) of the build machine. ROCm is looked up in
`ROCM_PATH` (CMake or environment variable), default `/opt/rocm`. `RESPACK_CUDA` and `RESPACK_HIP` are
mutually exclusive.

For gfx1200, the build also assembles the FP16 round-to-nearest GEMM kernel (`src/chiqw/fp16_rn/`, used with
`RESPACK_GPU_GEMM=fp16`) with ROCm's `llvm-mc` and `ld.lld`. `-DRESPACK_HIP_FP16_RN=OFF` leaves it out.

**Running:** as on NVIDIA GPUs, one MPI rank per GPU (local rank modulo device count); restrict the visible
GPUs with `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES`. On consumer Radeon GPUs, which are weak in
FP64 like the L40S, use `fast_fft_sp=1`.

**CGEMM block length:** the HIP build uses blocks of at most 512 rows (`GPU_KC_MAX` in `gpu_api.h`) instead
of 2048. rocBLAS accumulates the inner dimension sequentially in FP32, so the error grows about linearly with
the block length. For SrVO₃ with double-precision FFTs, ε⁻¹ deviates from the CPU double-precision run by
3.8×10⁻⁶ at 2048 rows, 8.4×10⁻⁷ at 512 and 2.2×10⁻⁷ at 128. The CGEMM time is 13.0 s, 16.6 s and 33.1 s. At
512 rows the accuracy matches cuBLAS at 2048 rows.

**Validation** (Radeon RX 9060 XT, gfx1200, 16 GB; ROCm 7.2.1; Ryzen 9 9950X3D, 1 rank × 4 threads), against the
CPU double-precision run of the same build:

| test | ε⁻¹ (all q) | U(ω), J(ω) | static U, U′, J |
|---|---|---|---|
| SrVO₃ cRPA | 8.4×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| SrVO₃ cRPA, `fast_fft_sp=1` | 3.1×10⁻⁶ | ≤ 3×10⁻⁵ eV | identical |
| Si cRPA | 9.8×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| Si cRPA, `fast_fft_sp=1` | 8.9×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |

Compare with a CPU run of the same build. Builds with and without LTO differed by 3.4×10⁻⁵ in ε⁻¹ for SrVO₃
(Si: 3.9×10⁻⁶), even on the same machine. The cause was the complex library calls in LTO builds (see "gfortran and
LTO" under Build). With `-fcx-fortran-rules` at the link step the difference is 1.3×10⁻⁵: the tetrahedron weights
amplify ordinary code-generation differences. It is below the method's rounding level of 7×10⁻⁵ (see above). The static
U, U′, J agree to all printed digits.

The test machine was also running another job on 16 cores, so the timings above are not a benchmark.

### χ⁰ products on the matrix units: FP16 split (experimental, `RESPACK_GPU_GEMM=fp16`)

**Motivation.** On the RX 9060 XT the CGEMM reaches only 5–6 TFLOP/s. RDNA4's matrix units (WMMA) do not
support FP32, but they run FP16 or BF16 inputs with FP32 accumulation about 10× faster. A GEMM microbenchmark
for the χ⁰ product T = Bt·Aᵀ (block length K; TFLOP/s of complex-equivalent work) gives:

| RX 9060 XT, ROCm 7.2.1 | NG 583, K 512 | NG 2150, K 512 | NG 2150, K 2048 |
|---|---|---|---|
| CGEMM (default path) | 5.3 | 5.6 | 6.1 |
| CGEMM, all ω in one strided-batched call | 5.5 | 6.0 | – |
| real form as one SGEMM (`[Xr −Xi; Xi Xr]`) | 2.0 | 3.0 | 2.5 |
| 3M (3 SGEMMs) | 2.0 | 3.9 | 4.1 |
| FP16 inputs, FP32 accumulation, one pass | 15.4 | 49.6 | 68.3 |
| BF16 inputs, FP32 accumulation, one pass | 14.3 | 46.1 | 57.3 |

**Method.** One pass in FP16 is too inaccurate (11-bit mantissa), so each factor is split, x = x_hi + x_lo
with x_hi = fp16(x) and x_lo = fp16(x − x_hi), about 22 bits together. The product is evaluated as
hi·hi + hi·lo + lo·hi (lo·lo, of order 2⁻²², is dropped). All three terms of the real block form
`[[Xr, −Xi], [Xi, Xr]]·[Yr, Yi]ᵀ` are computed in one `GemmEx` call with K′ = 6K, FP16 inputs and FP32
accumulation. That is three times the work of one pass, so at large NG (≈ 2000) the expected gain over CGEMM is
3–4×; at NG ≈ 600 there is little gain.

FP16 covers only 6×10⁻⁵ to 65504. The pair densities and the weights are therefore scaled by exact powers of
two before the split, so that their bounds lie below 2¹³. The pair densities use one factor per tile, the
weights one per frequency. The factors are removed exactly in double precision when T is added to χ.
Elements far below the largest lose relative precision in the FP16 lo part, but their contribution to χ is
correspondingly small.

**Accumulation inside the matrix units is biased.** A first version computed all three products in one call
(K′ = 6K). Its ε⁻¹ deviation was 1.1×10⁻⁵ (SrVO₃) and 3.4×10⁻⁵ (Si), and it grew strictly in proportion to the
block length. A microbenchmark (`accerr.cpp`) shows why. Its inputs are exactly representable in FP16, so all
products are exact, and the reference is an exact double-precision sum. Errors are in units of the FP32
rounding unit ε, relative to Σ|x·y|, for same-sign data:

| method | rms, K = 512 | K = 2048 | K = 16384 | mean (bias), K = 16384 |
|---|---|---|---|---|
| CPU OpenBLAS SGEMM | 2.9 | 1.9 | 2.0 | 0.0 |
| rocBLAS SGEMM / CGEMM (FP32) | 5.6 | 11.3 | 32 | ≈ 0 |
| FP16 matrix units, one call | 5.2 | 37 | 360 | −360 |
| FP16, K in chunks of 256, continued with β = 1 | 1.7 | 1.2 | 2.1 | −0.5 |
| FP16, K in chunks of 64, continued with β = 1 | 0.9 | 1.5 | 4.0 | 0.0 |

- **rocBLAS FP32 rounds to nearest.** There is no bias, and the error grows as √K.
- **The matrix units truncate.** Inside one call they accumulate with a bias toward zero: the error equals
  the bias and grows linearly with K, about one biased rounding per 16-term matrix operation.
- **Effect on χ.** In χ, sums dominated by one sign, such as the diagonal χ(G,G), add this bias up over all
  blocks.
- **Sign-mixed data** shows small, unbiased errors for all methods.
- **The output step `C = A·B + β·C` rounds to nearest.** K chunks continued with β = 1 therefore keep the bias
  bounded.

**Chunked accumulation (current implementation):**
- **Corrections.** The small corrections hi·lo + lo·hi (2¹¹ smaller) are computed in one call per block
  (β = 0). Their bias is negligible relative to χ.
- **hi·hi in chunks.** hi·hi follows in K chunks of `RESPACK_GPU_FP16_CHUNK` columns (default 512; 0 = all
  three products in one call), each continued with β = 1, so the chunk results are combined with rounding to
  nearest.
- **Blocks of 8192 rows.** The FP16 path uses blocks of 8192 rows (`RESPACK_GPU_KC` overrides). The accuracy
  depends on the chunk length only: at NG = 1963, 2048-row blocks give 1.14×10⁻⁶ and 8192-row blocks
  1.07×10⁻⁶. Longer blocks mean fewer updates of the double-precision χ.

**Profile** (`rocprofv3`, NG = 1963, one q, chunk 512, 2048-row blocks; GEMM phase 34 s, no idle time on the
GPU):

| kernel | calls | time | rate |
|---|---|---|---|
| corrections, K′ = 8192, β = 0 | 6090 | 14.1 s | 55 TFLOP/s |
| hi·hi chunks, K′ = 512, β = 1 | 47460 | 13.1 s | 29 TFLOP/s |
| `k_packX` | 6090 | 3.1 s | memory bound |
| `k_accmp` (FP32 → double χ) | 6090 | 3.1 s | memory bound |

**Why the chunks are slow.** Each β = 1 call reads and writes the whole FP32 output (62 MB at NG = 1963). In
rocBLAS this traffic does not overlap with the compute: a call costs about compute time + 0.19 ms. The cost
of chunking is therefore set by the number of β = 1 calls (total inner length ÷ chunk), not by their
arithmetic intensity.

**Tried and rejected: interleaving.** The corrections were put in the same calls ahead of each hi·hi chunk
(K = 3 × chunk per call, no separate correction call). It was not faster: at NG = 1963 and chunk 512 the GEMM
phase took 33.4 s, and the calls ran at 39 TFLOP/s, which is compute plus the same 0.19 ms each. It was also
2–4× less accurate (4.6×10⁻⁶ instead of 1.1×10⁻⁶), for reasons not understood (the internal accumulation
order of the kernel). It was reverted.

**Usage:** `RESPACK_GPU_GEMM=fp16` in the environment of `calc_chiqw` with `flg_gpu=1`. The input file is
unchanged, and the CGEMM path stays the default. The code is shared by the CUDA and HIP builds
(`cublasGemmEx` / `hipblasGemmEx`).

**Results** (RX 9060 XT; ε⁻¹ against CPU double precision of the same build):

| | SrVO₃ chain | Si chain | NG = 1963, one q: ε⁻¹ | χ⁰ products | total |
|---|---|---|---|---|---|
| CGEMM (default path) | 8.4×10⁻⁷ | 9.8×10⁻⁷ | 4.4×10⁻⁷ | 82.7 s | 139.4 s |
| FP16, one call, 512-row blocks | 1.1×10⁻⁵ | 3.4×10⁻⁵ | 3.5×10⁻⁶ | 37.0 s | 93.8 s |
| FP16, chunks of 1024 | 2.3×10⁻⁶ | – | 2.1×10⁻⁶ | 32.2 s | 90.1 s |
| FP16, chunks of 512 | 1.1×10⁻⁶ | 3.5×10⁻⁶ | 1.1×10⁻⁶ | 33.9 s | 92.0 s |
| FP16, chunks of 512, 8192-row blocks (default) | – | – | 1.07×10⁻⁶ | 31.8 s | 88.2 s |
| FP16, chunks of 256 | 5.5×10⁻⁷ | 1.8×10⁻⁶ | 7.2×10⁻⁷ | 36.7 s | 93.5 s |
| CPU double precision, 9950X3D, 16 threads | – | – | – | 265.8 s | 326.8 s |

The NG = 1963 case is SrVO₃ with `Ecut_for_eps` = 45 Ry, q #2, 805 pairs, 216 k, 70 ω and `fast_fft_sp=1`.
Unless noted, the FP16 runs use 2048-row blocks. U(ω), J(ω) deviate by at most 2–4×10⁻⁵ eV with chunks
(7×10⁻⁴ eV without chunks at 2048 rows). The static U agrees to all digits for SrVO₃; for Si it differs by 1 in
the last printed digit.

The remaining error is proportional to the chunk length (the truncation inside each chunk). With chunking it
no longer depends on the block length: SrVO₃ gives 1.147×10⁻⁶ with 512-row and 1.148×10⁻⁶ with 2048-row
blocks. At NG = 1963 the FP16 path is 2.4–2.6× faster than CGEMM in the χ⁰ products and 1.5× faster in
total. The CPU part (ε inversion, writing ε⁻¹, setup) is about 55 s of the total.

**Leaner packing.** The 2×2 real block form `[[Yr, −Yi], [Yi, Yr]]` is stored on the Y side, which is packed once
per block. Per frequency only `[Xr | Xi]` (hi and lo) is written, with A read from the FP16 parts of Y. That is 16
instead of 40 bytes of memory traffic per element and frequency. At NG = 1963, `k_packX` takes 1.37 s instead of
3.07 s per q point and the GEMM phase 30.6 s instead of 31.8 s; ε⁻¹ is unchanged (1.085×10⁻⁶).

**Fast mode with two products (`RESPACK_GPU_FP16_TERMS=2x`).** X is used with its hi part only, which drops the
lo(X)·hi(Y) product (`2y` drops hi(X)·lo(Y) instead). That is a third less GEMM work, but 10–30× less accurate. Use
it for exploratory runs (convergence parameters, Wannier windows); production runs use the default of three products.
Against CPU double precision (double-precision FFTs for the chains):

| | 3 products (default) | `2x` | `2y` |
|---|---|---|---|
| ε⁻¹, SrVO₃ chain | 1.2×10⁻⁶ | 2.1×10⁻⁵ | 2.5×10⁻⁵ |
| ε⁻¹, Si chain | 3.2×10⁻⁶ | 6.9×10⁻⁵ | 9.2×10⁻⁵ |
| U(ω), J(ω), Si: max. deviation | 4×10⁻⁵ eV | 3.3×10⁻⁴ eV | 3.2×10⁻⁴ eV |
| static U₁₁ | identical | identical | Si: last digit |
| NG = 1963, one q: ε⁻¹ / χ⁰ products / total | 1.1×10⁻⁶ / 30.6 s / 48.1 s | 9.0×10⁻⁶ / 22.5 s / 40 s | 1.1×10⁻⁵ / 23.0 s / 40 s |

With `2x`, `k_packX` writes only the hi part of X.

**Round-to-nearest kernel for hi·hi (gfx1200, HIP build).** On the RX 9060 XT, hi·hi is no longer chunked. It
runs as one call of an FP16 GEMM kernel whose matrix-unit results are accumulated with round-to-nearest vector
adds (`src/chiqw/fp16_rn/`). The kernel is one of hipBLASLt's own FP16 kernels (MIT License, © AMD), taken from
the installed library and patched in its assembly (`local-tools/tensile-patch/`):
- **Patch.** Every accumulating `v_wmma_f32_16x16x16_f16` writes a fresh 16-term partial tile into a temporary
  tile, and the accumulator is updated with 4 `v_dual_add_f32`.
- **Accuracy.** No bias at any K; the rms error grows as √K (11 ε at K = 32768, against 731 ε bias without the
  patch).
- **Kernel choice.** The kernel was chosen from all 237 FP16 NT kernels of the library by sustained timing. The
  card clocks down under load, so burst timings rank the kernels differently. For one β = 1 call at K = 16384
  it takes 4.8 ms unpatched, against 5.2 ms for hipBLAS's choice, and 5.5 ms patched.
- **Build.** The source is assembled at build time with ROCm's `llvm-mc` and `ld.lld` and embedded in
  `calc_chiqw`. This happens when `CMAKE_HIP_ARCHITECTURES` contains gfx1200 (option `RESPACK_HIP_FP16_RN`,
  default ON).
- **Runtime.** The kernel is used on gfx1200 devices with NG ≥ 16 (the library kernel faults for M < 8).
  `RESPACK_GPU_FP16_RN=0` selects the chunked hipBLAS path, which other AMD GPUs use. The CUDA build has its own
  kernel (see “FP16 split on the L40S”).
- **Corrections.** These stay with hipBLAS. The best library kernel would be 3 % faster for them, not worth a
  second embedded kernel.

Per block and frequency at NG = 1963 (8192 rows; sustained, `tcombo.cpp`):

| | corrections (K′ = 32768, β = 0) | hi·hi (K′ = 16384) | total |
|---|---|---|---|
| chunked hipBLAS path | 10.3 ms | 8.7 ms (32 calls, β = 1) | 18.9 ms |
| hi·hi with the round-to-nearest kernel | 10.3 ms | 5.5 ms (one call) | 16.1 ms (−14 %) |
| (hipBLAS, hi·hi in one biased call) | – | 5.2 ms | – |

In `calc_chiqw` (NG = 1963, one q, two runs each):

| | ε⁻¹ vs CPU double | χ⁰ products | total |
|---|---|---|---|
| chunked hipBLAS path (`RESPACK_GPU_FP16_RN=0`) | 1.11×10⁻⁶ | 30.7 s | 48.3 s |
| round-to-nearest kernel (default on gfx1200) | 6.1×10⁻⁷ | 27.4 s | 45.0 s |

The full chains agree as well:
- **SrVO₃.** ε⁻¹ 2.5×10⁻⁶ (chunked 3.1×10⁻⁶); U(ω), J(ω) within 3×10⁻⁵ eV (chunked 6×10⁻⁵ eV).
- **Si.** ε⁻¹ 1.3×10⁻⁶ (chunked 2.9×10⁻⁶); within 1×10⁻⁵ eV (chunked 4×10⁻⁵ eV).

The static U₁₁ is identical in both. Both chains use single-precision FFTs, the default with the GPU.

The remaining error comes from the corrections call (one β = 0 call per block, truncating) and from FP32
rounding. Longer blocks make it worse and are not faster: 16384 rows give 1.7×10⁻⁶ and 28.3 s, 4096 rows give
7.9×10⁻⁷ and 27.2 s. The default stays at 8192 rows.

**Status:** experimental, validated on the RX 9060 XT. At NG ≈ 2000 the χ⁰ products are 3× faster than with
CGEMM (27.4 s against 82.7 s), at CGEMM-level accuracy. The round-to-nearest kernel removed the chunk overhead.
What remains besides the GEMMs is packing and accumulation (about 3 s at NG = 1963).

**Padding of the G dimension (both GPUs).** The FP16 operands and T use a G dimension P = NG rounded up to a
multiple of 8 (`RESPACK_GPU_FP16_PAD`, default 8; padding rows are not written and not read back):
- **L40S.** With an odd NG, cuBLAS picks unaligned FP16 kernels: 135 instead of 190–200 TFLOP/s. Padding only
  the leading dimensions does not help. At NG = 1963 the χ⁰ products of the chunked path take 7.8 s instead of
  10.9 s.
- **RX 9060 XT.** The corrections call takes 8.5 instead of 9.9 ms per block and frequency, and the χ⁰ products
  7 % less time; ε⁻¹ is unchanged.

### FP16 split on the L40S (CUDA build)

The same switch (`RESPACK_GPU_GEMM=fp16`) on an L40S. CUDA 13.0, one GPU, 8 cores of an EPYC 9454; the CUDA
versions of `accerr.cpp` and `gemmbench.cpp` build with `-DTOOLS_CUDA` (`local-tools/hip2cuda.h`).

**The tensor cores truncate, and much more than RDNA4 at short K.** Same-sign data, `accerr.cpp`, bias in units
of 2⁻²⁴:

| | K = 128 | 512 | 2048 | 16384 |
|---|---|---|---|---|
| L40S, FP16 GemmEx, one call | −8.6 | −73 | −28 (other kernel) | −371 |
| L40S, chunks of 256 / 64, β = 1 | −8.6 / −1.9 | −28 / −1.9 | −28 / −1.9 | −28 / −1.9 |
| RX 9060 XT, one call | – | −4.4 | −37 | −360 |

With chunks of 512, as on the RX 9060 XT, ε⁻¹ deviates by 2.9×10⁻⁵ (SrVO₃ chain) and 3.8×10⁻⁶ (Si), 7–9× more
than with CGEMM. Shorter chunks would cost too much output traffic.

**Kernel with round-to-nearest accumulation (`src/chiqw/hgemm_rn.cuh`).** Own CUDA code, used for hi·hi on
compute capability ≥ 8.0; `RESPACK_GPU_FP16_RN=0` selects the chunked cuBLAS path.
- **Accumulation.** Every `mma.sync` m16n8k16 starts from zero, and its 16-term result is added to the FP32
  accumulator with an ordinary FADD (round to nearest).
- **Tiles.** 128 × 256 (or 128 × 128 for N < 1024) by 32, 8 warps of 64 × 64, a 4-stage `cp.async` pipeline,
  XOR-swizzled shared memory read with `ldmatrix.trans`, and warp fragments double-buffered.
- **Split K.** When the output tiles do not fill the GPU, K is split and the partial results are added in a
  second kernel (round to nearest).
- **Tests.** `local-tools/hgemm_rn_test.cu`:
  - error against an exact reference: bias 0.0, rms 1–8 units of 2⁻²⁴ up to K = 16384;
  - 192 edge cases (M from 8 to 1968, K from 1 to 20000, β = 1, padded leading dimensions) agree with cuBLAS;
  - sustained timing at the `gemm_mp` shapes.

Per block and frequency (8192 rows, sustained):

| | NG = 1963 | NG = 583 | NG = 61 |
|---|---|---|---|
| corrections, cuBLAS (K′ = 32768, β = 0) | 2.53 ms (200 TFLOP/s) | 0.25 ms | 0.015 ms |
| hi·hi, cuBLAS in chunks of 512 | 1.88 ms | 0.28 ms | 0.149 ms |
| hi·hi, cuBLAS in one call (biased) | 1.32 ms | 0.13 ms | 0.011 ms |
| hi·hi, round-to-nearest kernel | 1.65 ms (154 TFLOP/s) | 0.22 ms | 0.018 ms |
| (kernel with plain tensor-core accumulation) | 1.38 ms | 0.19 ms | 0.016 ms |
| CGEMM path (4 CGEMMs of 2048 rows) | ≈ 6 ms (41 TFLOP/s) | 0.68 ms | – |

In `calc_chiqw` on the L40S:

| | CGEMM (default) | FP16, chunks of 512, not padded | FP16, chunks of 512 | FP16, round-to-nearest kernel | FP16 `2x` |
|---|---|---|---|---|---|
| NG = 1963, one q: ε⁻¹ vs CPU double | 1.8×10⁻⁶ | 5.1×10⁻⁶ | 5.1×10⁻⁶ | 7.8×10⁻⁷ | 1.1×10⁻⁵ |
| NG = 1963, one q: χ⁰ products | 12.06 s | 10.89 s | 7.80 s | 7.30 s | 5.79 s |
| NG = 1963, one q: total | 61.3 s | 59.9 s | 56.1 s | 55.8 s | 55.2 s |
| SrVO₃ chain (`Ecut_for_eps` 5 Ry): ε⁻¹ / χ⁰ products | 4.0×10⁻⁶ / 2.5 s | 2.1×10⁻⁵ / 8.4 s | 2.9×10⁻⁵ / 6.6 s | 4.5×10⁻⁶ / 3.2 s | – |
| Si chain: ε⁻¹ / χ⁰ products | 4.4×10⁻⁷ / 0.5 s | 3.8×10⁻⁶ / 1.6 s | 3.8×10⁻⁶ / 1.3 s | 4.0×10⁻⁷ / 0.6 s | – |
| U(ω), J(ω): max. deviation (SrVO₃ / Si) | 6×10⁻⁵ / 1×10⁻⁵ eV | 8×10⁻⁵ / 1×10⁻⁵ eV | 1.9×10⁻⁴ / 2×10⁻⁵ eV | 6×10⁻⁵ / 1×10⁻⁵ eV | – |

With the round-to-nearest kernel, the FP16 path on the L40S is as accurate as CGEMM, and at NG = 1963 more
accurate. There its χ⁰ products are 1.65× faster. (CPU double precision on the same 8 cores: 2338 s.) At the small NG of the chains (56–70) it is slightly slower than CGEMM: CGEMM is the
better choice there. The total of one q is dominated by the CPU (ε inversion and output on 8 cores, about
45 s). With several q, the asynchronous ε post-processing (`flg_async_eps`) overlaps that with the GPU work of
the next q.

### Validation (GPU vs CPU double precision; L40S, CUDA 13.0)

| test | ε⁻¹ (all q) | U(ω), J(ω) | static U, U′, J |
|---|---|---|---|
| SrVO₃ cRPA, 1 GPU | 4.1×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| SrVO₃ cRPA, 2 GPUs (2 ranks) | 1.5×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| Si cRPA, 1 GPU | 2.0×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| SrVO₃ cRPA, 1 GPU, `fast_fft_sp=1` | 4.0×10⁻⁶ | ≤ 6×10⁻⁵ eV | identical |
| Si cRPA, 1 GPU, `fast_fft_sp=1` | 4.4×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |

The method's own rounding level (tetrahedron formulas, original code −O2 vs −O3) is 7×10⁻⁵.

The CGEMM block length matters. For SrVO₃ at NG = 583 (q = 2), ε⁻¹ deviates by 6.9×10⁻⁶ with 7992 rows,
1.9×10⁻⁶ with 3888 and 4.7×10⁻⁷ with 2048 (the default), at 8 % extra CGEMM time. `RESPACK_GPU_KC`
overrides it, for testing only.

### Performance (SrVO₃, `Ecut_for_eps` = 20 Ry, NG = 583, one q, 805 pairs, 216 k, 70 ω; 8 cores per run)

| | χ⁰ products | total `calc_chiqw` |
|---|---|---|
| CPU double precision (EPYC 9454, 8 cores) | 182 s | 205 s |
| CPU `fast_gemm_sp=1` | 91 s | 112 s |
| 1 × L40S | 1.2 s | 18 s |

Nsight Systems kernel times on the L40S:

| | CGEMM | FFT kernels | other kernels | GPU busy, share of wall time |
|---|---|---|---|---|
| double-precision FFTs | 1.05 s | 0.48 s | 0.14 s | 1.67 s of 17.6 s |
| `fast_fft_sp=1` | 1.09 s | 0.16 s | 0.15 s | 1.39 s |

CGEMM runs at about 31 TFlop/s. At this size the run is dominated by CPU work: setup, wave-function rotation,
tetrahedron weights, ε inversion and output. For large systems the GPU share grows strongly, because
CGEMM ∝ NG²·N_k·N_pair·N_ω and the FFTs ∝ N_pair·N_k·N_grid.

### Remaining GPU candidates

Measured on a large molecular crystal: about 120 atoms per cell, 75 k points, FFT grid 90 × 48 × 72, SCF wave
functions with 289 bands, 11,858 band pairs; one q with FP16 and the round-to-nearest kernel on one L40S and 4
physical CPU cores (8 hardware threads). Production runs with about 650 bands have about 8× as many pairs.

| one q | `Ecut_for_eps` 3.5 Ry (NG 1231) | 5 Ry (NG 2100) | 5 Ry, CGEMM |
|---|---|---|---|
| pair densities (rotation, uploads, FFTs) | 18.9 s | 21.5 s | 21.6 s |
| tetrahedron weights | 4.2 s (before the sub-blocks: 1 tile, not overlapped) | 4.3 s | 4.4 s |
| χ⁰ products | 16.6 s | 49.6 s | 73.3 s |
| ε inversion / writing ε⁻¹ (in the background with several q) | 7.6 / 1.6 s | 36.3 / 4.9 s | 36.2 / 5.4 s |
| total, including reading the wave functions | 90 s | 160 s | 184 s |

With 8 physical cores and the sub-blocks below, the totals are 69 s and 128 s; the ε inversion takes 4.9 and
23.8 s.

**Tile size on the GPU path.** Each band's wave function is rotated on the CPU, uploaded and transformed once per
tile that contains it, so small tiles repeat that work many times. The tile size used to follow the CPU path's
rules: a 4 GB host budget that included the pair densities, and per-thread real-space buffers for 8 threads. On
the GPU path the pair densities and the real-space work are on the GPU, but these rules still cut the tiles to
10 × 11 band pairs, 125 tiles for this q. Now the GPU path takes 60 % of the device memory left after χ and 2 GB
for the FFT and GEMM buffers (`RESPACK_GPU_TILE_MB` overrides). The case above then has 1 tile at 3.5 Ry and 2 at
5 Ry on the L40S:

| one q, L40S | pair densities, old → new tiles | total |
|---|---|---|
| 3.5 Ry | 280.2 → 18.9 s | 345 → 90 s |
| 5 Ry | 184.0 → 21.5 s | 316 → 160 s |

The results are unchanged:
- SrVO₃ and Si chains: identical, with one tile per q before and after.
- NG = 1963: ε⁻¹ deviates from CPU double precision by 8.4×10⁻⁷ (7.8×10⁻⁷ before); only the summation
  order changed.

On a 16 GB card the same case gets about 3 tiles at 3.5 Ry and 6 at 5 Ry. On the RX 9060 XT, with the old tiles
and other jobs on the CPU, the pair densities took 173 s (3.5 Ry) and 198 s (5 Ry), and the χ⁰ products 56 s and
158 s (CGEMM: 484 s).

**Where the pair-density time goes** (Nsight Systems, 5 Ry case above, L40S, 8 physical cores). The phase is
limited by the GPU; the CPU rotation (5 s) runs while the GPU transforms the previous k point. GPU time:

| | time |
|---|---|
| pair products (`k_products4`) | 5.7 s |
| forward FFT, x pass (all 90-point lines) | 5.8 s |
| pruning gather after the x pass (`k_prx`) | 1.5 s |
| y and z passes, band FFTs, scatter | ≈ 1.3 s |
| host → GPU copies of the rotated wave functions (double precision, pageable) | 3.7 s (55.8 GB, 15 GB/s) |

**Single-precision uploads.** With single-precision FFTs (`fast_fft_sp=1`, the default with the GPU) the rotated
coefficients are now written in single precision into a pinned host buffer (`c0_apply_sp`,
`chi_gpu_bands_sp`). The buffer is kept between tiles, grown as needed and is a few hundred MB; there is a
pageable fallback. The FFTs already ran in single precision, so the results are bitwise identical. At q = 0 the
k·p terms need the double-precision coefficients, so that q keeps the old path. Effect on the case above:
- copies: 28.5 GB in 1.1 s (26 GB/s);
- pair densities: 15.6 s instead of 18.2 s (5 Ry), 15.0 s instead of 17.1 s (3.5 Ry).

**Fused pair products and x pass (`k_prodx4`).** With single-precision pruned FFTs, the pair products, the x pass
and the gather of the needed x frequencies (19–21 of 90) are one kernel (`RESPACK_GPU_FUSEX=0` switches it off):
- **Per block:** one pair and 32 lines (y, z). It loads the lines of both bands with coalesced reads and forms
  their product in shared memory (about 24 KB, so it also fits AMD GPUs).
- **Per thread:** the needed frequencies of one line, as a direct DFT with an exact twiddle table, written in the
  layout of the y pass.
- **Order:** blocks run pair-fastest, so concurrently running blocks share the band lines in L2.

The full product grid is no longer written and re-read; a direct DFT costs more arithmetic than an FFT, but
memory traffic was the limit. Effect on the case above:

| one q, L40S | products + x pass + gather (GPU) | pair densities | total |
|---|---|---|---|
| 3.5 Ry, before | 13.0 s | 15.0 s | 67.8 s |
| 3.5 Ry, fused | 4.0 s | 6.2 s | 57.5 s |
| 5 Ry, fused | 4.0 s | 7.8 s | 125 s (under Nsight Systems) |

ε⁻¹ changes at the rounding level (fused against not fused: 1.1×10⁻⁷). Against CPU double precision: SrVO₃
chain 4.3×10⁻⁶ (4.6×10⁻⁶ before), Si chain 4.1×10⁻⁷; U(ω), J(ω) unchanged. The pair densities are now split
about evenly between the CPU rotation (about 3–5 s, overlapped) and the GPU. The kernel reaches about 12
TFLOP/s.

**Two-stage x DFT.** The needed x frequencies form one block around 0 (19–21 of 90), so `k_prodx4` splits
n1 = P·Q (x = p + P q):
- stage 1, in place in shared memory: h[p][r] = Σ_q f[p + P q] e^(−2πi rq/Q);
- stage 2: out(k) = Σ_p e^(−2πi kp/n1) h[p][k mod Q].

This needs n1·Q + nsx·P instead of nsx·n1 operations per line (Q = 5: 828 instead of 1890). Q is the divisor of
n1 up to 16 with the least work, chosen per q; `RESPACK_GPU_PRODX_Q=1` gives the direct DFT. L40S, the case
above:

| one q | kernel (Nsight) | pair densities | total |
|---|---|---|---|
| 3.5 Ry, direct | – | 7.0 s | 52.2 s |
| 3.5 Ry, two stages (Q = 5) | – | 4.4 s | 47.6 s |
| 5 Ry, direct | 4.0 s | 7.3 s | 110.0 s |
| 5 Ry, two stages | 2.9 s | 5.3 s | 108.6 s |

The kernel gains 1.36×, less than the 2.3× fewer operations, so part of it is memory-bound. ε⁻¹ two stages
against direct: 7–9×10⁻⁸. Chains against CPU double precision: SrVO₃ 4.4×10⁻⁶, Si 4.1×10⁻⁷; U(ω), J(ω) unchanged
(small grids choose Q = 2 and 3).

**Several GPUs (q-parallel, `MPI_num_qcomm`, one rank per GPU).** Tested with the FP16 path and the
round-to-nearest kernel: 4 q points of the case above (3.5 Ry), 8 physical cores per rank.
- 1 L40S: 217.9 s; 2 L40S: 145.7 s; ε⁻¹ bitwise identical.
- Both ranks first read all wave functions (about 30 s here), so the speedup approaches 2× only with many q.
- Every rank holds all wave functions in double precision: 21.7 GB peak per rank here. With about 650 bands
  that is an estimated 45–50 GB per rank (wave functions alone about 36 GB). This matters on machines with
  less RAM.

**Single-precision wave functions (GPU path).** With single-precision FFTs, `calc_chiqw` stores the wave
functions as complex(4) (`OCC4`, `VIR4`; `RESPACK_WFN_DP=1` keeps double precision; the CPU path is unchanged).
- **Memory.** The case above at 3.5 Ry needs 10.9 GB instead of 18.2 GB.
- **Accuracy.** ε⁻¹ deviates by 1.9×10⁻⁹ from the double-precision wave functions (RX 9060 XT: 2.4×10⁻⁹); the
  chains against CPU double precision are unchanged.
- **q = 0.** The k·p and intra-band terms use double-precision copies of the coefficients they need.
- **One rank per q group.** The band ring exchange is skipped there. It copied all virtual bands to the rank
  itself through a full-size `VIR_new`, about 22 GB for 650 bands in double precision. The broadcast across q
  groups is skipped as well, since every rank has read all bands itself.

**Shared on the node (`m_wfn_shared`).** With one rank per q group, all ranks of a node need the same
read-only wave functions. Node rank 0 reads them into an MPI-3 shared-memory window and the others map it
(`RESPACK_WFN_SHARED=0` gives private copies; if the window cannot be created, private copies are used).
4 q points on 2 L40S, results bitwise identical:

| | wall time | memory of both ranks (PSS) |
|---|---|---|
| private copies | 111.5 s | 23.3 GB |
| shared | 79.3 s | 15.0 GB |

**Rotation on the GPU.** Per tile, the irreducible coefficients of the tile's bands (all k_irr, single precision)
are copied once into a GPU cache: about 7 GB here on the L40S, 1.8–2.7 GB on a 16 GB card. Per k point the CPU
only builds the rotation map (`make_C0_map`); `k_rotscatter` gathers, multiplies by the phase (in double
precision, as on the CPU) and writes the band to the FFT grid.
- **Tile size.** The cache counts in the tile budget: square tiles t × t with t² bytes per pair + 2t bytes per
  band (cache and real-space grid) within the device budget. If a tile would have fewer than 256 pairs, the CPU
  rotation is used (`RESPACK_GPU_ROT=0` forces it).
- **Testing smaller cards.** `RESPACK_GPU_MEM_MB` caps the device memory assumed for the tile size.
- **Results.** ε⁻¹ is bitwise identical to the CPU rotation for the same tiles; with other tiles it changes by
  2–3×10⁻⁷ through the summation order.
- **Effect.** The CPU rotation shrinks from 2.5–3.5 s to 0.4–0.9 s per q, and the copies to the GPU drop about
  3-fold (once per tile instead of per k point).

| one q, case above | L40S, rotation on CPU → GPU | RX 9060 XT, rotation on CPU → GPU |
|---|---|---|
| 3.5 Ry: pair densities | 6.3 → 5.6 s | 27.9 → 27.5 s |
| 3.5 Ry: total | 48.4 → 47.3 s | 90.3 → 95.1 s (3 → 4 tiles) |
| 5 Ry: pair densities | 7.4 → 6.2 s | 35.3 → 32.2 s |
| 5 Ry: total | 107.5 → 105.7 s | 208.8 → 206.0 s |
| 5 Ry, L40S capped at 16 GB (14 tiles) | 110.4 s | – |

The main gain is in runs with several q, where the CPU rotation competed with the background ε inversion.

**Overlap of the tetrahedron weights.** With the GPU, phases B and C run in sub-blocks of about 32,768 rows.
While the main thread drives the χ⁰ products of sub-block s on the GPU, a second OpenMP thread computes the
weights of sub-block s + 1, or of the first sub-block of the next tile; all GPU calls stay on the main thread.
`chi_gpu_gemm` takes the row range and copies only that part of W. The weights start only after the pair
densities of a tile: run concurrently with those, they slowed the CPU part of phase A 5-fold. Only the first
sub-block of the first tile runs alone, so this also works with one or two large tiles.
- **Accuracy.** The FP16 scales are now chosen per sub-block. ε⁻¹ changes at the rounding level (NG = 1963:
  1.7×10⁻⁷ against one block per tile, 8.3×10⁻⁷ against CPU double precision); the chains are unchanged.
- **Gain on the molecular crystal.** About 4 s per q (3.6–3.7 s of weights hidden), about 30 s per q with
  650 bands.
- **Small NG.** When the weights take longer than the χ⁰ products (e.g. the SrVO₃ and Si chains), the products
  are hidden instead.

In the timing line, the tetrahedron time runs concurrently, so the three phases no longer add up to the wall
time.

**Remaining candidates.** With large tiles, the pair densities are 20–40 % of a q in this case. The χ⁰ products
dominate at 5 Ry. GPU-resident irreducible wave functions (rotation on the GPU) would now save only the
remaining rotation and upload time within that share. The larger levers are:
- **More GPUs, parallel over q:** `MPI_num_qcomm`, one rank per GPU.
- **Fewer frequencies in the χ⁰ products:** a spectral (Hilbert-transform) representation. Each transition
  contributes to a few energy bins instead of all 70 frequencies, with real weights. This changes the method
  and would need validation against the original code.

## Build

**Requirements:**
- a Fortran and C compiler (tested: GNU 13);
- CMake ≥ 3.18 (presets need ≥ 3.21; tested with 3.28 and 4.4);
- MPI with Fortran bindings;
- BLAS/LAPACK;
- FFTW3 (double precision, with the Fortran interface file `fftw3.f03`).

For example:

    # Debian/Ubuntu
    sudo apt install gfortran cmake libopenmpi-dev libopenblas-dev libfftw3-dev
    # Fedora
    sudo dnf install gcc-gfortran cmake openmpi-devel openblas-devel fftw-devel

**With presets** (`CMakePresets.json`):

    cmake --preset default             # portable Release build with the system libraries
    cmake --build --preset default
    cmake --install build/default --prefix $HOME/.local/respack-zen-opt
    # or in one go:  local-tools/build.sh [preset] [prefix]

| preset | what |
|---|---|
| `default` | Release (-O3), compiler-default target CPU, system BLAS/LAPACK, FFTW3, MPI |
| `native` | adds `-march=native` and link-time optimization |
| `zen3` | AMD Zen 3: `-march=znver3`, LTO, static OpenBLAS. Reads `OPENBLAS_LIB_DIR` (directory containing `libopenblas.a`) and `FFTW3_ROOT` from the environment. |
| `debug` | -O0 -g |

**Without presets:**

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
          [-DRESPACK_NATIVE=ON | -DRESPACK_ARCH=znver3] [-DRESPACK_LTO=ON] \
          [-DFFTW3_ROOT=/path/to/fftw] [-DBLA_VENDOR=OpenBLAS|Intel10_64lp_seq|...] [-DBLA_STATIC=ON]
    cmake --build build -j
    cmake --install build --prefix <dir>

**Options:**
- `RESPACK_NATIVE`, `RESPACK_ARCH`: CPU tuning.
- `RESPACK_LTO`: link-time optimization. It brings no measurable gain for RESPACK (SrVO₃ `calc_chiqw`: 58.7 s
  with, 55.4 s without, on a Zen 5 CPU). With gfortran the build passes `-fcx-fortran-rules` to the LTO link
  step, see the note below.
- `RESPACK_BUILD_XTAPP`: also build the xTAPP converters (default ON).

Libraries are found through CMake's standard `FindMPI`, `FindOpenMP` and `FindLAPACK` (use `BLA_VENDOR`
to choose) and `cmake/FindFFTW3.cmake` (`FFTW3_ROOT`, pkg-config, system paths; `FFTW3_USE_STATIC=ON`
prefers `libfftw3.a`). The upstream route `-DCONFIG=gcc` (`config/*.cmake`) still works. The original
per-directory Makefiles are unchanged but don't know about the fork's new files; use CMake.

**Notes:**
- **Single-threaded BLAS:** BLAS is called inside OpenMP regions. Use a single-threaded BLAS, or set
  `OPENBLAS_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`) at run time.
- **FFTW built with LTO:** if your FFTW was itself built with LTO-only objects, enable `RESPACK_LTO` too.
- **gfortran and LTO:** the LTO code generator does not inherit gfortran's rules for complex multiplication and
  division (`-fcx-fortran-rules`, set by the Fortran front end). Without the flag at the link step, every complex
  product or quotient becomes a call to `__muldc3`/`__divdc3` (434 + 147 call sites in `calc_chiqw`, 308 + 87 in
  `calc_wannier`), and the loops containing them are not vectorized. The tetrahedron weights then ran 2–3×
  slower (SrVO₃: 17.2 s instead of 8.7 s), and ε⁻¹ differed from a build without LTO by 3.4×10⁻⁵ (1.3×10⁻⁵ with
  the flag). The CMake build sets the flag automatically. Builds made before this fix with LTO (e.g. the `zen3`
  preset) should be rebuilt.
- **Other compilers:** the GNU toolchain is tested. Intel/ifx get `-xHost` with `RESPACK_NATIVE`;
  other compilers use their defaults.
- **Validated builds:** the `zen3` and `default` builds were both checked against the reference results
  with `local-tools/check_build.sh` (ε⁻¹ within 4×10⁻¹⁶ and 1×10⁻⁵ respectively; U, J identical).

## Tools

| file | purpose |
|---|---|
| `local-tools/build.sh` | configure, build and install with a preset |
| `local-tools/run_chain_cmp.sh` | full chain with extra `&param_chiqw` entries vs a reference run (also under Slurm) |
| `local-tools/validate_sp_gemm.sh` | accuracy of `fast_gemm_sp=1` vs double precision (full chains) |
| `local-tools/check_build.sh` | quick numerical check of a build against a reference run (one q of `calc_chiqw`, `calc_w3d`/`calc_j3d`) |
| `local-tools/cmp_epsqw.py` | compare `dir-eps/qNNN/dat.epsqw.600` of two runs |
| `local-tools/bench_tetra.f90`, `acc_tetra_quad.f90` | tetrahedron speed, and accuracy vs quad precision |
| `local-tools/eps_freq_dev.py` | deviation of ε⁻¹ per frequency |
| `local-tools/validate_si.sh`, `validate_full.sh` | end-to-end validation against the original |
| `local-tools/accerr.cpp`, `gemmbench.cpp`, `mpbench.cpp` | accumulation error of GEMM paths against an exact reference; GEMM rates for the χ⁰ shapes; one block and frequency of the FP16 path (HIP, or CUDA with `-DTOOLS_CUDA` and `hip2cuda.h`) |
| `local-tools/hgemm_rn_test.cu` | accuracy, edge cases and sustained timing of `hgemm_rn.cuh` (CUDA) |
| `local-tools/tensile-patch/` | extract, reassemble, patch, check and time hipBLASLt's FP16 kernels (see its README) |
