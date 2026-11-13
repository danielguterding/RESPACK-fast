/* Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
   GNU General Public License, version 3 (see GPL/GPLv3.txt).
   Embeds the code object assembled from gemm_fp16_rn_gfx1200.s (path RESPACK_FP16_RN_HSACO, set by CMake) as
   respack_fp16_rn_hsaco[]; chi_gpu.cu loads it with hipModuleLoadData. */
#ifndef RESPACK_FP16_RN_HSACO
#error "RESPACK_FP16_RN_HSACO (path of the assembled kernel) is set by src/chiqw/CMakeLists.txt"
#endif
__asm__(".section .rodata\n"
        ".global respack_fp16_rn_hsaco\n"
        ".type respack_fp16_rn_hsaco, @object\n"
        ".balign 64\n"
        "respack_fp16_rn_hsaco:\n"
        ".incbin \"" RESPACK_FP16_RN_HSACO "\"\n"
        ".size respack_fp16_rn_hsaco, . - respack_fp16_rn_hsaco\n"
        ".previous\n");
