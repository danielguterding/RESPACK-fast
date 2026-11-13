# Patching the hipBLASLt FP16 GEMM kernel (gfx1200)

Tools to take the FP16 GEMM kernels of hipBLASLt for the χ⁰ products (`RESPACK_GPU_GEMM=fp16`) out of the
installed ROCm library, reassemble them, launch them directly, time them and patch their accumulation. Tested
with ROCm 7.2.1 on an RX 9060 XT (gfx1200). The kernels are derived from the hipBLASLt binary (MIT License,
© Advanced Micro Devices). The one used by `calc_chiqw` is in `src/chiqw/fp16_rn/`, with the license; all can
be regenerated from the installed library with the steps below.

## Why

Inside one call, the matrix units accumulate with a bias toward zero (truncation): for same-sign sums the error
grows linearly with K (see `../accerr.cpp`). The chunked accumulation in `chi_gpu.cu` limits this with β = 1 calls,
whose output traffic costs about a third of the hi·hi time. The patched kernel accumulates every WMMA result
with round-to-nearest vector adds, which removes the bias without chunking.

## Steps

| step | tool / command |
|---|---|
| Which kernels hipBLASLt uses for our shapes (NT, FP16 → FP32, M = NG, N = 2NG) | `whichk.cpp` under `rocprofv3 --kernel-trace` |
| All NT FP16 → FP32 kernels of the code object | `llvm-readelf -s` (`allsyms.txt`), `llvm-readelf --notes` (`notes.txt`) |
| Sustained timing of all of them for both `gemm_mp` shapes | `tsus <code object> <name list> <K> <beta> [Y offset] [reps]` |
| Launch parameters and the 140-byte argument block | `LD_PRELOAD=launchspy.so whichk` (wraps `hipExtModuleLaunchKernel`) |
| Unpack the compressed bundle (`CCOB`) | `clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1200 --input=<hipblaslt .co> --output=hs_nt.hsaco` |
| Disassemble the kernel's address range (the symbol has size 0) | `llvm-objdump -d --mcpu=gfx1200 --start-address=… --stop-address=…` |
| Kernel descriptor as `.amdhsa` directives | `llvm-objdump -D --disassemble-symbols=<kernel>.kd` |
| Metadata (arguments, register counts) | `llvm-readelf --notes` (the kernel's entry only) |
| Clean source: all labels from the symbol table, `s_getpc_b64` offsets as label differences | `gen_clean.py k.s allsyms.txt clean.s --symbolic` |
| All of the above for one kernel, plus the registers free around the WMMAs | `extract.py <code object> allsyms.txt notes.txt <name> <workdir>` |
| Patch: `acc = WMMA(A, B, acc)` → `tmp = WMMA(A, B, 0)`, `acc += tmp` (4 × `v_dual_add_f32`) | `patch_rn.py clean.s patched.s [--temps a,b] [--depth d]` |
| Assemble | `mk_asm.py patched.s out.s [_suffix]`, `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj`, `ld.lld -shared` |
| Check results bitwise against hipBLAS and time | `tkern <code object> <name file> [K] [beta]` |
| Accumulation error against an exact reference | `tacc <code object> <name file>` |
| Edge cases (M from 1 to 1963, K from 2 to 16384, β = 1) against hipBLAS | `tval <code object> <name> [M K]` |
| One block and frequency of `gemm_mp`, all variants in alternation | `tcombo <co corr> <name corr> <co hh> <name hh>` |

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

Timing (`tkern`: 20 warm-up launches, then the best of 5 × 10). The additions cost issue slots, not latency:

| K = 32768 | time |
|---|---|
| original | 10.12 ms (49.9 TFLOP/s) |
| patched, adds 1 / 2 / 4 WMMA later (`--depth`) | 12.7–12.9 / 12.7–12.9 / 12.6–12.8 ms (+26 %) |
| patched, single `v_add_f32` (`--single`) | 14.0–14.2 ms (+40 %) |

Fewer adds per WMMA would need longer chains in the temporary tiles, but the 4×4-tile kernel has no registers left
for a second accumulator. Earlier single-shot timings:

| time | K = 512 | 4096 | 16384 (β = 1) | 32768 |
|---|---|---|---|---|
| original | 0.27 ms | 1.65 ms | 5.72 ms | 10.04 ms |
| patched | 0.30 ms | 1.84 ms | 6.54 ms | 13.12 ms (+31 %) |

The 4 dual adds per WMMA do not overlap with the matrix units. Per block and frequency at NG = 1963, from these
burst timings:
- chunked hipBLAS path: 19.1 ms;
- everything with the patched kernel: about 19.7 ms;
- hybrid (corrections with the original kernel, hi·hi in one patched call): 16.6 ms (−13 %).

Timed sustained (`tcombo`), this hybrid takes 18.1 ms against 18.9 ms; see the next section for the kernel that is
used.

## Kernel choice and the kernel in `calc_chiqw`

Burst timings (a few launches) mislead on this card: under sustained load it clocks down, by different amounts
for different kernels. `tsus` times each kernel in rounds between two hipBLAS calls of the same shape. Of the 237
NT kernels in the code object, all give correct results for our argument layout (M = 1963). Sustained:

| kernel (all MT128x128x16, MIWT4_4) | corrections K = 32768, β = 0 | hi·hi K = 16384, β = 1 |
|---|---|---|
| hipBLAS's choice (DTVA1/DTVB1, the kernel above) | 10.2–10.3 ms | 5.2 ms |
| best for both shapes: LDSB1, PGR1, PLR1, GRVWA8, SIA2, CLR1 | 0.97 × hipBLAS | 0.92 × hipBLAS (4.8 ms) |
| best in burst timing for the corrections (PGR2, LDSB1, DTVA0/DTVB0) | 1.24 × hipBLAS | – |

The best kernel (256 VGPRs, 64 WMMA, accumulators v0–v127) leaves v224–v255 free around its WMMAs.
`patch_rn.py --temps 224,232 --depth 1` gives the kernel in `src/chiqw/fp16_rn/` (renamed
`respack_hgemm_nt_rn_gfx1200`). Its round trip reproduces all 16,760 instruction encodings.

Patched:
- **Accuracy.** No bias; rms 1.5 / 2.9 / 5.7 / 11.3 ε at K = 512 / 2048 / 8192 / 32768.
- **Edge cases.** All 42 `tval` cases agree with hipBLAS. M < 8 faults with the unpatched library kernel as
  well; `calc_chiqw` requires NG ≥ 16.
- **Cost.** Unlike the DTV kernel, which costs +26 % with the patch, this one costs about +15 % over unpatched.
  That is 5 % over hipBLAS's biased one-call choice.

Per block and frequency at NG = 1963 (`tcombo`, sustained):

| corrections + hi·hi | time |
|---|---|
| hipBLAS + hipBLAS in 32 chunks of 512 (β = 1) | 18.9 ms |
| hipBLAS + patched kernel, one call (used) | 16.1 ms (−14 %) |
| best library kernel + patched kernel, one call | 15.9 ms |

In `calc_chiqw` at NG = 1963 (one q) the χ⁰ products take 27.4 s instead of 30.7 s. The ε⁻¹ deviation from CPU
double precision drops from 1.1×10⁻⁶ to 6.1×10⁻⁷.
