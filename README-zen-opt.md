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
| `calc_chiqw` | ε(ω) built and inverted for all frequencies in parallel (OpenMP); each inversion uses single-threaded LAPACK as before (results identical to ≤ 10⁻¹⁵; SrVO₃, NG = 1963: 88 → 51 s per q) | always on |
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

`fast_fft_sp` (default 0): if 1, the pair densities are computed with single-precision FFTs (`fftwf`)
and stored in double precision. The q = 0 k·p terms stay in double precision. This needs `libfftw3f`.

| test (with `fast_gemm_sp=1`) | ε⁻¹ vs double precision | U(ω), J(ω) | static U, U′, J |
|---|---|---|---|
| SrVO₃ cRPA | 1.8×10⁻⁶ relative | ≤ 2×10⁻⁵ eV | identical |
| Si cRPA | 2.8×10⁻⁷ relative | ≤ 1×10⁻⁵ eV | identical |

On a CPU this gives about 1.3× for the pair densities on small grids that fit in cache. The large gain
is on GPUs.

`fast_mem_mb` (default 4096) is the memory budget for one tile of pair densities and weights. A larger
budget means larger tiles and fewer backward FFTs.

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
| `src/chiqw/chiqw.F` | modified | calls `m_chi_fast` for `flg_fast=1`, original loops kept for `flg_fast=0`; frequency loops of the ε inversion parallel (OpenMP) |
| `src/chiqw/m_rdinput.f90` | modified | inputs `flg_fast`, `fast_mem_mb` |
| `src/chiqw/CMakeLists.txt` | modified | FFTW3, new sources |
| `src/calc_int/int_fast.f90` | new | module `m_int_fast` |
| `src/calc_int/calc_intW.F`, `calc_intJ.F` | modified | call `m_int_fast` for `flg_fast=1`; original loops kept; exp(−iq·R) table in `calc_intW.F` |
| `src/calc_int/calc_int_sub.F` | modified | `make_LG0` loop bounds (identical G lists) |
| `src/calc_int/m_rdinput.f90`, `config_w3d.h`, `config_j3d.h`, `CMakeLists.txt` | modified | inputs, declarations, FFTW3 |
| `CMakeLists.txt` (top level and all subdirectories) | modified | modern target-based CMake (3.18 to 4.x): imported MPI/OpenMP/LAPACK/FFTW3 targets, CPU and LTO options |
| `cmake/FindFFTW3.cmake`, `CMakePresets.json` | new | FFTW3 finder, build presets |
| `src/chiqw/chi_gpu.cu`, `chi_gpu_stub.c` | new | GPU path of the band-pair loop, stub without a GPU build |
| `src/chiqw/gpu_api.h` | new | maps the CUDA names in `chi_gpu.cu` onto HIP/hipFFT/hipBLAS for the ROCm build |
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
- `fast_fft_sp=1` runs the pair-density FFTs in single precision (cuFFT C2C, 3× faster on an L40S).
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

Compare with a CPU run of the same build. Builds with and without LTO differ by 3.4×10⁻⁵ in ε⁻¹ for SrVO₃
(Si: 3.9×10⁻⁶), even on the same machine. `zen-opt` without LTO and `gpu-hip` (no LTO) agree to 1×10⁻¹⁵,
while `zen-opt` with LTO differs from both by 3.4×10⁻⁵. The tetrahedron weights are this sensitive to compiler
optimisation (inlining, FMA contraction). It is below the method's rounding level of 7×10⁻⁵ (see above). The static
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

**Status:** experimental, validated on the RX 9060 XT. At NG ≈ 2000 the χ⁰ products are 2.6× faster than
with CGEMM, at CGEMM-level accuracy. The remaining overhead is the output traffic of the β = 1 chunk calls
(about 9 s at NG = 1963) plus packing and accumulation (about 4 s). A custom kernel with two-level
accumulation in registers would remove most of it; the upper bound is about 35 % of the GEMM phase.

**To do: evaluate the FP16 tensor cores on the L40S** (CUDA build, same switch). Measure accuracy against CPU
double precision and speed against the CGEMM path (≈ 31 TFLOP/s) at SrVO₃ size and at large NG. If accurate,
the FP16 tensor-core rate should give at least a 2× gain there as well. NVIDIA tensor cores are reported to
truncate in their internal accumulation too (Fasi, Higham, Mikaitis, Pranesh, PeerJ Comput. Sci. 7, e330
(2021)). So first run `accerr.cpp` (CUDA port) on the L40S and choose the chunk length from it. Not yet
compiled or run with CUDA.

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
| χ⁰ products (CGEMM) | GPU | ≈ 9 min | done; FP16 split on the matrix units under evaluation (RX 9060 XT), L40S to do |
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
- `RESPACK_LTO`: link-time optimization.
- `RESPACK_BUILD_XTAPP`: also build the xTAPP converters (default ON).

Libraries are found through CMake's standard `FindMPI`, `FindOpenMP` and `FindLAPACK` (use `BLA_VENDOR`
to choose) and `cmake/FindFFTW3.cmake` (`FFTW3_ROOT`, pkg-config, system paths; `FFTW3_USE_STATIC=ON`
prefers `libfftw3.a`). The upstream route `-DCONFIG=gcc` (`config/*.cmake`) still works. The original
per-directory Makefiles are unchanged but don't know about the fork's new files; use CMake.

**Notes:**
- **Single-threaded BLAS:** BLAS is called inside OpenMP regions. Use a single-threaded BLAS, or set
  `OPENBLAS_NUM_THREADS=1` (or `MKL_NUM_THREADS=1`) at run time.
- **FFTW built with LTO:** if your FFTW was itself built with LTO-only objects, enable `RESPACK_LTO` too.
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
