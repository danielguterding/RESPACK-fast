# RESPACK 20240804 — `zen-opt` fork

Unofficial fork of [RESPACK](https://sites.google.com/view/kazuma7k6r) 20240804 (GPLv3) with a restructured
polarization and interaction code. Branch `upstream` holds the unmodified release; branch `zen-opt`
holds the changes. Every change can be switched off at run time, which restores the original code paths.

## What changed

| program | change | switch |
|---|---|---|
| `calc_chiqw` | new band-pair loop (`src/chiqw/chi_fast.f90`), see below | `flg_fast` in `&param_chiqw` (default 1) |
| `calc_chiqw` | faster generalized-tetrahedron weights (`src/chiqw/m_tetrahedron_fast.F`) | used by `flg_fast=1` |
| `calc_chiqw` | optional CUDA path (`src/chiqw/chi_gpu.cu`), see “GPU path” | `flg_gpu` (default 0), build with `-DRESPACK_CUDA=ON` |
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
| `src/chiqw/chi_gpu.cu`, `chi_gpu_stub.c` | new | CUDA path of the band-pair loop, stub without CUDA |
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

The figures below are projected for a large molecular crystal (about 120 atoms per cell; NG ≈ 2150,
N_pair ≈ 10⁵, N_k = 64, N_grid ≈ 3×10⁵, 70 ω) from the measured rates, not measured on such a system.

| part | where | projected cost per q | assessment |
|---|---|---|---|
| χ⁰ products (CGEMM) | GPU | ≈ 9 min | done |
| pair-density FFTs (`fast_fft_sp=1`) | GPU | ≈ 3–5 min | done |
| **wave-function rotation (`make_C0`) + uploads** | **CPU, serial with the GPU** | **≈ 2.5–4 min** | **next candidate** |
| tetrahedron weights | CPU | ≈ 1 min (8 cores) | overlap with GPU work rather than port |
| ε inversion (70 ω) | CPU | ≈ 0.5 min | cuSOLVER: a few % |
| writing ε⁻¹ | CPU / disk | ≈ 5 GB per q | no |
| `calc_w3d` / `calc_j3d` | CPU | minutes per system | no |
| `calc_wannier`, DFT (QE) | CPU | – | no (QE needs FP64; the L40S is FP64-weak) |

**Why the rotation:** for every tile and k, the CPU rotates the tile's wave functions and uploads them while
the GPU waits. That is a complex `exp` per coefficient, and the uploads repeat in every tile containing the
band, about 480 GB per q in the projection. It also ties up CPU cores, which limit how many GPUs a shared node
can drive.

**Planned steps, by effort:**
1. **`exp` lookup table in `make_C0`.** The phase exp(−2πi(G·τ)/nnp) only takes nnp discrete values, so a
   table of roots of unity replaces it exactly. This helps the CPU and GPU paths.
2. **GPU-resident irreducible wave functions.** For the projected case this is about 10 GB, uploaded once
   per q, with the rotation as a gather kernel on the GPU. That removes the uploads and the CPU work in phase A.
3. **Overlap of the tetrahedron weights with the GPU work.** The weights are only needed in phase C.
4. **cuSOLVER for the ε inversion.**

Measure a q point of a real large system with Nsight Systems before implementing steps 2–4.

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
