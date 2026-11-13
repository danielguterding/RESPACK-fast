# Patching the hipBLASLt FP16 GEMM kernel (gfx1200)

Tools to take the FP16 GEMM kernel that hipBLASLt uses for the χ⁰ products (`RESPACK_GPU_GEMM=fp16`) out of the
installed ROCm library, reassemble it, launch it directly and patch its accumulation. Tested with ROCm 7.2.1 on
an RX 9060 XT (gfx1200). The kernel source is not stored here: it is derived from the hipBLASLt binary (MIT
license, © Advanced Micro Devices) and is regenerated from the installed library with the steps below.

## Why

Inside one call, the matrix units accumulate with a bias toward zero (truncation): for same-sign sums the error
grows linearly with K (see `../accerr.cpp`). The chunked accumulation in `chi_gpu.cu` limits this with β = 1 calls,
whose output traffic costs about a third of the hi·hi time. The patched kernel accumulates every WMMA result
with round-to-nearest vector adds, which removes the bias without chunking.

## Steps

| step | tool / command |
|---|---|
| Which kernels hipBLASLt uses for our shapes (NT, FP16 → FP32, M = NG, N = 2NG) | `whichk.cpp` under `rocprofv3 --kernel-trace` |
| Launch parameters and the 140-byte argument block | `LD_PRELOAD=launchspy.so whichk` (wraps `hipExtModuleLaunchKernel`) |
| Unpack the compressed bundle (`CCOB`) | `clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1200 --input=<hipblaslt .co> --output=hs_nt.hsaco` |
| Disassemble the kernel's address range (the symbol has size 0) | `llvm-objdump -d --mcpu=gfx1200 --start-address=… --stop-address=…` |
| Kernel descriptor as `.amdhsa` directives | `llvm-objdump -D --disassemble-symbols=<kernel>.kd` |
| Metadata (arguments, register counts) | `llvm-readelf --notes` (the kernel's entry only) |
| Clean source: all labels from the symbol table, `s_getpc_b64` offsets as label differences | `gen_clean.py k.s allsyms.txt clean.s --symbolic` |
| Patch: `acc = WMMA(A, B, acc)` → `tmp = WMMA(A, B, 0)`, `acc += tmp` (4 × `v_dual_add_f32`) | `patch_rn.py clean.s patched.s` |
| Assemble | `mk_asm.py patched.s out.s [_suffix]`, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj`, `ld.lld -shared` |
| Check results bitwise against hipBLAS and time | `tkern <code object> <name file> [K] [beta]` |
| Accumulation error against an exact reference | `tacc <code object> <name file>` |

## Results (RX 9060 XT, M = 1963, N = 3926)

**Round trip.** The kernel is `Cijk_Ailk_Bjlk_HSS_BH_…_MT128x128x16_MI16x16x1_…_DTVA1_DTVB1_…_WG32_4_1`: 18,288
instructions, 256 VGPRs, 80 WMMA. Both the plain and the symbolic reassembly reproduce all 18,288 encodings.
Launched directly, it gives 0 differing elements of 7.7 million against `hipblasGemmEx` (β = 0 and β = 1), at the
same speed (10.0 ms, 50 TFLOP/s at K = 32768).

**Patch.**
- **Main loop.** Per wave, 4×4 output tiles of 16×16 and 2 K steps per iteration (32 WMMA). It is
  double-buffered, and the operands are loaded directly into registers.
- **Temporary tiles.** v232–v247 are unused from the first to the last WMMA and are written before their next
  read afterwards, so they serve as two alternating temporary tiles.

| same-sign data, error in units of 2⁻²⁴ | K = 512 | 2048 | 8192 | 32768 |
|---|---|---|---|---|
| original: rms / bias | 5.2 / −4.4 | 37 / −37 | 175 / −175 | 731 / −731 |
| patched: rms / bias | 1.5 / 0.0 | 2.9 / 0.0 | 5.7 / 0.0 | 11.3 / 0.0 |

| time | K = 512 | 4096 | 16384 (β = 1) | 32768 |
|---|---|---|---|---|
| original | 0.27 ms | 1.65 ms | 5.72 ms | 10.04 ms |
| patched | 0.30 ms | 1.84 ms | 6.54 ms | 13.12 ms (+31 %) |

The 4 dual adds per WMMA do not overlap with the matrix units. Per block and frequency at NG = 1963:
- chunked hipBLAS path: 19.1 ms;
- everything with the patched kernel: about 19.7 ms;
- hybrid (corrections with the original kernel, hi·hi in one patched call): 16.6 ms (−13 %).
