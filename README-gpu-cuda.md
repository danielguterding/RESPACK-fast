# RESPACK 20240804 — `gpu-cuda` branch: GPU path for `calc_chiqw`

This branch is [`zen-opt`](README-zen-opt.md) plus a GPU path for the band-pair loop of `calc_chiqw` on NVIDIA
GPUs (CUDA, cuFFT, cuBLAS). Everything in [README-zen-opt.md](README-zen-opt.md) applies here as well: the CPU
changes, inputs, validation and build. This file describes only what the GPU path adds. The branch `gpu-hip`
builds on this one and adds AMD GPUs (ROCm/HIP) and further GPU optimizations
([README-gpu-hip.md](https://github.com/danielguterding/RESPACK-fast/blob/gpu-hip/README-gpu-hip.md)).

| program | change | switch |
|---|---|---|
| `calc_chiqw` | optional CUDA path of the band-pair loop (`src/chiqw/chi_gpu.cu`) | `flg_gpu` in `&param_chiqw` (default 0), build with `-DRESPACK_CUDA=ON` |

## GPU path

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
  −1 (auto), which means on, as on the CPU. ε⁻¹ changes by about 3×10⁻⁶. Set
  `fast_fft_sp=0` for double-precision FFTs on the GPU.
- `fast_mem_mb` sets the tile size.

## Build

The CPU build and its options are described in [README-zen-opt.md](README-zen-opt.md#build); the GPU path adds
one option, `RESPACK_CUDA` (CUDA toolkit with cuFFT and cuBLAS; CMake ≥ 3.18):

    cmake --preset native -B build/native-cuda -DRESPACK_CUDA=ON \
          -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89   # 89 = L40S/Ada
    cmake --build build/native-cuda -j

Without `RESPACK_CUDA`, a stub is compiled and `flg_gpu=1` stops with a message. The default architecture
list is `all-major`.

## Running

- **One MPI rank per GPU.** Rank n uses visible GPU n (local rank modulo device count).
- **Under Slurm, start `mpirun` inside an `srun` step or batch job**, so that `CUDA_VISIBLE_DEVICES` and the
  device cgroup confine the processes to the allocated GPUs:

      srun -c 8 --gres=gpu:2 env OMP_NUM_THREADS=4 mpirun -np 2 --oversubscribe --bind-to none calc_chiqw < input.in

  A `mpirun` started from an `salloc` shell sees all GPUs of the node.
- **BLAS threads:** keep `OPENBLAS_NUM_THREADS=1`.
- **Pruned forward FFTs:** after the forward FFT of a pair product, only the NG components of the ε sphere
  are needed. The FFT runs as 1D passes: x for all lines, y only for the x columns that occur, z only for the
  (x, y) pairs that occur. It is on automatically when it saves work; `RESPACK_GPU_PRUNE=0/1` forces it. Measured
  on the RX 9060 XT (HIP build of the branch `gpu-hip`) for the SrVO₃ chain (NG = 81, 24³ grid; 5 of 24 x columns and 21 of 576 (x, y) pairs
  needed): pair densities 15.1 → 9.8 s with single-precision FFTs and 31.3 → 19.1 s with double-precision FFTs.
  Pruned and full FFTs agree to 1×10⁻¹² (double) and 1×10⁻⁷ (single). With a large ε sphere (NG = 1963 on
  the same grid) there is nothing to prune. For a large cell with a small `Ecut_for_eps`, where the sphere
  covers only about 10 % of each grid direction, the y and z passes shrink to about 15 %.
- **ε inversion in the background:** with `flg_gpu=1`, `flg_async_eps` defaults to on (auto). The inversion of
  ε(ω) and the output of a q point run on the CPU while the GPU computes χ⁰ of the next q point. Only the last
  q point's inversion stays visible. Measured on an RX 9060 XT (HIP build, `gpu-hip`; SrVO₃, NG = 1963, 3 q points, 8 threads): the
  run takes 116.5 s instead of 137.1 s, and the results are identical (ε⁻¹ bitwise). Set `flg_async_eps=0` to
  switch it off.

## Validation (GPU vs CPU double precision; L40S, CUDA 13.0)

| test | ε⁻¹ (all q) | U(ω), J(ω) | static U, U′, J |
|---|---|---|---|
| SrVO₃ cRPA, 1 GPU, `fast_fft_sp=0` | 4.1×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| SrVO₃ cRPA, 2 GPUs (2 ranks), `fast_fft_sp=0` | 1.5×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| Si cRPA, 1 GPU, `fast_fft_sp=0` | 2.0×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |
| SrVO₃ cRPA, 1 GPU, single-precision FFTs (default) | 4.0×10⁻⁶ | ≤ 6×10⁻⁵ eV | identical |
| Si cRPA, 1 GPU, single-precision FFTs (default) | 4.4×10⁻⁷ | ≤ 1×10⁻⁵ eV | identical |

The method's own rounding level (tetrahedron formulas, original code −O2 vs −O3) is 7×10⁻⁵.

The CGEMM block length matters. For SrVO₃ at NG = 583 (q = 2), ε⁻¹ deviates by 6.9×10⁻⁶ with 7992 rows,
1.9×10⁻⁶ with 3888 and 4.7×10⁻⁷ with 2048 (the default), at 8 % extra CGEMM time. `RESPACK_GPU_KC`
overrides it, for testing only.

## Performance (SrVO₃, `Ecut_for_eps` = 20 Ry, NG = 583, one q, 805 pairs, 216 k, 70 ω; 8 cores per run)

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

## Remaining GPU candidates

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

Steps 2 and 3 (rotation on the GPU with a per-tile cache of the irreducible wave functions, and the overlap of
the tetrahedron weights with the χ⁰ products), the measurement on a large system and further GPU optimizations
are on the branch `gpu-hip`, see its
[README-gpu-hip.md](https://github.com/danielguterding/RESPACK-fast/blob/gpu-hip/README-gpu-hip.md).

## Switches

| switch | default | effect |
|---|---|---|
| `flg_gpu` (`&param_chiqw`) | 0 | 1: pair densities and χ⁰ products on the GPU |
| `fast_fft_sp` (`&param_chiqw`) | −1 (auto = 1) | single-precision FFTs, CPU and GPU; 0: double precision |
| `flg_async_eps` (`&param_chiqw`) | −1 (auto) | ε inversion and output of a q point in a background thread; auto = on with `flg_gpu=1` |
| `fast_mem_mb` (`&param_chiqw`) | 4096 | tile size |
| `RESPACK_GPU_PRUNE` (environment) | auto (when the y and z passes are below 1.6 full passes) | 0/1: pruned forward FFTs on the GPU off/on |
| `RESPACK_GPU_FBATCH` (environment) | auto (about 256 MB of grids) | number of FFT grids per batched cuFFT call |
| `RESPACK_GPU_KC` (environment) | 2048 | CGEMM block length in rows; for testing only |

## Modified files (GPLv3 §5a; modifications by Daniel Guterding, November 2026)

In addition to those listed in [README-zen-opt.md](README-zen-opt.md):

| file | status | change |
|---|---|---|
| `src/chiqw/chi_gpu.cu` | new | GPU part of the band-pair loop: wave functions on the FFT grid, pair products, batched and pruned cuFFT transforms, gather of the ε sphere, CGEMM blocks into a double-precision χ on the GPU, q = 0 head/wing terms |
| `src/chiqw/chi_gpu_stub.c` | new | stubs for builds without CUDA (`flg_gpu=1` then stops with a message) |
| `src/chiqw/chi_fast.f90` | modified | GPU branch of the pair densities and the χ⁰ products (`use_gpu`) |
| `src/chiqw/chiqw.F`, `m_rdinput.f90` | modified | input `flg_gpu`; `flg_async_eps` auto = on with the GPU |
| `CMakeLists.txt`, `src/chiqw/CMakeLists.txt` | modified | option `RESPACK_CUDA` (CUDA language, cuFFT, cuBLAS; default architectures `all-major`) |

## Tools

In addition to those listed in [README-zen-opt.md](README-zen-opt.md#tools):

| file | purpose |
|---|---|
| `local-tools/run_chain_cmp.sh` | full chain with extra `&param_chiqw` entries (e.g. `flg_gpu=1`) vs a reference run (also under Slurm) |
