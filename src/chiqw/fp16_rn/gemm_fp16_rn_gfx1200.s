// FP16 GEMM D = alpha A B^T + beta C (A: M x K, B: N x K, column-major, FP32 C/D) for gfx1200 with
// round-to-nearest accumulation of the matrix-unit results. Used for the hi*hi product of RESPACK_GPU_GEMM=fp16
// (chi_gpu.cu, gemm_mp); assembled at build time (src/chiqw/CMakeLists.txt) and embedded in calc_chiqw.
//
// Derived from the hipBLASLt kernel (ROCm 7.2.1, code object of the HSS NT kernels for gfx1200)
//   Cijk_Ailk_Bjlk_HSS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x16_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1200_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB2048_LBSPPM0_LPA0_LPB0_LPM0_LRVW8_LWPMn1_MIAV1_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL0_PGR1_PLR1_PKA0_SIA2_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_SKXCCM0_TLDS0_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_WSGRB0_WS32_WG32_4_1
// Copyright (C) Advanced Micro Devices, Inc., MIT License (see LICENSE-hipBLASLt.txt in this directory).
// Changes (zen-opt fork of RESPACK, 2026): every accumulating v_wmma_f32_16x16x16_f16 acc, A, B, acc is replaced by
// v_wmma ... tmp, A, B, 0 into one of two alternating temporary tiles (v[224:231], v[232:239], unused by the
// original kernel between its first and last WMMA) followed by acc += tmp as 4 v_dual_add_f32, issued after the
// next WMMA; kernel renamed. Regenerate with local-tools/tensile-patch (extract.py, gen_clean.py --symbolic,
// patch_rn.py --temps 224,232 --depth 1, mk_asm.py).
.amdhsa_code_object_version 4
.amdgcn_target "amdgcn-amd-amdhsa--gfx1200"
.text
.protected respack_hgemm_nt_rn_gfx1200
.globl respack_hgemm_nt_rn_gfx1200
.p2align 8
.type respack_hgemm_nt_rn_gfx1200,@function
respack_hgemm_nt_rn_gfx1200:
label_ASM_Start:
	s_load_b32 s20, s[0:1], 0x0
	s_load_b32 s22, s[0:1], 0x4
	s_load_b32 s11, s[0:1], 0x8
	s_load_b32 s23, s[0:1], 0xc
	s_wait_kmcnt 0x0
	s_lshr_b32 s21, s20, 30
	s_and_b32 s20, 0x3fffffff, s20
	s_cmp_eq_u32 s21, 0
	s_cbranch_scc0 label_HBMArgs
	s_add_co_u32 s0, s0, 16
	s_add_co_ci_u32 s1, s1, 0
	s_load_b512 s[24:39], s[0:1], 0x0
	s_load_b128 s[40:43], s[0:1], 0x40
	s_load_b64 s[44:45], s[0:1], 0x50
	s_branch label_LoadArgsEnd
label_HBMArgs:
	s_load_b64 s[0:1], s[0:1], 0x10
	s_wait_kmcnt 0x0
label_LoadArgsEnd:
	s_and_b32 s10, s22, 0xffff0000
	s_lshr_b32 s10, s10, 16
	s_and_b32 s46, s22, 0xffff
	s_mov_b32 s5, s21
	s_mov_b32 m0, 0x2000
	v_mov_b32_e32 v218, v0
	s_mov_b32 vcc_hi, 0
	s_lshr_b32 s52, s11, 16
	s_ctz_i32_b32 s52, s52
	s_lshr_b32 s53, s11, 22
	s_cmp_gt_i32 s52, 0
	s_cbranch_scc0 label_skip_WGMXCC
	s_lshr_b32 s49, s23, s52
	s_lshl_b32 s49, s49, s52
	s_cmp_ge_u32 s2, s49
	s_cbranch_scc1 label_skip_WGMXCC
	s_cmp_eq_u32 s53, 0
	s_cbranch_scc0 label_XCCG_nonzero
	s_lshr_b32 s49, s2, s52
	s_bfm_b32 s50, s52, 0
	s_and_b32 s50, s2, s50
	s_lshr_b32 s51, s23, s52
	s_mul_i32 s50, s50, s51
	s_add_co_u32 s2, s49, s50
	s_branch label_skip_WGMXCC
label_XCCG_nonzero:
	v_cvt_f64_u32_e32 v[6:7], s53
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s2
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s53
	v_sub_nc_u32_e32 v8, s2, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s53
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_mul_lo_u32 v7, v6, s53
	v_sub_nc_u32_e32 v8, s2, v7
	v_readfirstlane_b32 s49, v6
	v_readfirstlane_b32 s50, v8
	s_mul_i32 s49, s49, s53
	s_lshr_b32 s50, s50, s52
	s_add_co_u32 s49, s49, s50
	v_cvt_f64_u32_e32 v[6:7], s53
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s23
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s53
	v_sub_nc_u32_e32 v8, s23, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s53
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s50, v6
	s_mul_i32 s50, s50, s53
	s_sub_co_u32 s51, s23, s50
	s_cmp_gt_u32 s2, s50
	s_cselect_b32 s50, s51, s53
	s_lshr_b32 s50, s50, s52
	s_bfm_b32 s51, s52, 0
	s_and_b32 s51, s2, s51
	s_mul_i32 s50, s50, s51
	s_add_co_u32 s2, s49, s50
label_skip_WGMXCC:
	s_cmp_eq_u32 s21, 0
	s_cbranch_scc0 label_MultiGemm
	v_and_b32_e32 v1, 31, v218
	v_and_b32_e32 v0, 15, v1
	v_lshlrev_b32_e32 v0, 2, v0
	v_lshrrev_b32_e32 v1, 4, v1
	v_lshl_add_u32 v0, v1, 10, v0
	v_lshrrev_b32_e32 v4, 5, v218
	v_and_b32_e32 v4, 1, v4
	v_lshl_add_u32 v0, v4, 6, v0
	v_and_b32_e32 v2, 31, v218
	v_and_b32_e32 v1, 15, v2
	v_lshlrev_b32_e32 v1, 2, v1
	v_lshrrev_b32_e32 v2, 4, v2
	v_lshl_add_u32 v1, v2, 10, v1
	v_lshrrev_b32_e32 v3, 6, v218
	v_and_b32_e32 v3, 1, v3
	v_lshl_add_u32 v1, v3, 6, v1
	v_lshrrev_b32_e32 v2, 5, v218
	v_lshrrev_b32_e32 v2, 2, v2
	s_mov_b32 s16, 0x800
	v_mul_lo_u32 v2, s16, v2
	v_add_lshl_u32 v134, v2, v0, 1
	v_lshrrev_b32_e32 v0, 5, v218
	v_lshrrev_b32_e32 v0, 2, v0
	v_mul_lo_u32 v0, s16, v0
	v_add_lshl_u32 v135, v0, v1, 1
	v_add_co_u32 v135, vcc_lo, 0x1000, v135
	v_lshrrev_b32_e32 v1, 4, v218
	v_and_b32_e32 v0, 15, v218
	v_lshlrev_b32_e32 v0, 3, v0
	v_mov_b32_e32 v4, v1
	v_lshrrev_b32_e32 v3, 4, v218
	v_and_b32_e32 v2, 15, v218
	v_lshlrev_b32_e32 v2, 3, v2
	v_mov_b32_e32 v5, v3
	v_mul_u32_u24_e32 v132, 0x80, v4
	v_add_lshl_u32 v132, v0, v132, 1
	v_mul_u32_u24_e32 v133, 0x80, v5
	v_add_lshl_u32 v133, v2, v133, 1
	v_add_co_u32 v133, vcc_lo, 0x1000, v133
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v8, 0x80
	v_mov_b32_e32 v7, s24
	v_cvt_f32_u32_e32 v6, v8
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v9, v7
	v_mul_f32_e32 v6, v6, v9
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e32 v9, v6, v8
	v_sub_nc_u32_e32 v9, v7, v9
	v_cmp_ne_u32_e64 vcc_lo, v9, 0
	v_add_co_ci_u32_e64 v6, vcc_lo, v6, 0, vcc_lo
	v_mov_b32_e32 v8, 0x80
	v_mov_b32_e32 v7, s25
	v_readfirstlane_b32 s14, v6
	v_cvt_f32_u32_e32 v6, v8
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v9, v7
	v_mul_f32_e32 v6, v6, v9
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e32 v9, v6, v8
	v_sub_nc_u32_e32 v9, v7, v9
	v_cmp_ne_u32_e64 vcc_lo, v9, 0
	v_add_co_ci_u32_e64 v6, vcc_lo, v6, 0, vcc_lo
	v_readfirstlane_b32 s15, v6
	s_mul_i32 s16, s14, s15
	s_and_b32 s17, s46, 0x3fff
	s_mul_i32 s16, s16, s17
	v_cvt_f32_u32_e32 v6, s16
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v7, s2
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s2, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s16, v6
	s_mov_b32 s4, s16
	s_mul_i32 s16, s15, s14
	s_mul_i32 s16, s16, s4
	s_mul_i32 s16, s16, s17
	s_sub_co_u32 s2, s2, s16
	v_cvt_f32_u32_e32 v6, s14
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v7, s2
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e64 v7, v6, s14
	v_sub_nc_u32_e32 v7, s2, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s14
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v7, s14
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s16, v6
	s_mov_b32 s3, s16
	s_mul_i32 s16, s3, s14
	s_sub_co_u32 s2, s2, s16
	s_branch label_MultiGemmEnd
label_MultiGemm:
	s_cmp_eq_u32 s5, 2
	s_cbranch_scc1 label_IsExternalValid
	s_mov_b32 s15, 0x7c
	s_mul_i32 s54, s20, 4
	s_mov_b64 s[48:49], s[0:1]
	s_branch label_IsExternalValidEnd
label_IsExternalValid:
	s_mov_b32 s15, 0xc4
	s_mov_b32 s54, 0
	s_mov_b64 s[48:49], s[0:1]
label_IsExternalValidEnd:
	s_mov_b32 s14, 1
	s_mov_b32 s55, 0
	s_load_b128 s[24:27], s[48:49], s54 offset:0x0
	s_mov_b32 s16, 1
	s_cmp_eq_u32 s20, s16
	s_cbranch_scc1 label_wgTable_noLoadLoop
label_Loop_GemmCount:
	s_wait_kmcnt 0x0
	s_lshr_b32 s52, s24, 7
	s_and_b32 s50, 0x7f, s24
	s_add_co_ci_u32 s52, s52, 0
	s_lshr_b32 s53, s25, 7
	s_and_b32 s50, 0x7f, s25
	s_add_co_ci_u32 s53, s53, 0
	s_mul_i32 s52, s52, s53
	s_mul_i32 s52, s52, s26
	s_and_b32 s53, s46, 0x3fff
	s_mul_i32 s52, s52, s53
	s_add_co_u32 s55, s55, s52
	s_cmp_lt_u32 s2, s55
	s_cbranch_scc1 label_FOUND
	s_add_co_u32 s54, s54, s15
	s_load_b128 s[24:27], s[48:49], s54 offset:0x0
	s_add_co_u32 s14, s14, 1
	s_cmp_lt_u32 s14, s20
	s_cbranch_scc1 label_Loop_GemmCount
label_wgTable_noLoadLoop:
	s_wait_kmcnt 0x0
	s_lshr_b32 s52, s24, 7
	s_and_b32 s50, 0x7f, s24
	s_add_co_ci_u32 s52, s52, 0
	s_lshr_b32 s53, s25, 7
	s_and_b32 s50, 0x7f, s25
	s_add_co_ci_u32 s53, s53, 0
	s_mul_i32 s52, s52, s53
	s_mul_i32 s52, s52, s26
	s_and_b32 s48, s46, 0x3fff
	s_mul_i32 s52, s52, s48
	s_add_co_u32 s55, s55, s52
label_FOUND:
	s_sub_co_u32 s49, s14, 1
	s_sub_co_u32 s48, s55, s52
	s_sub_co_u32 s2, s2, s48
	s_cmp_eq_u32 s5, 2
	s_cbranch_scc1 label_LoadExternalStruct
	s_lshl2_add_u32 s0, s20, s0
	s_add_co_ci_u32 s1, s1, 0
	s_mul_i32 s49, s49, 0x7c
	s_add_co_u32 s0, s0, s49
	s_add_co_ci_u32 s1, s1, 0
	s_load_b512 s[28:43], s[0:1], 0x10
	s_load_b64 s[44:45], s[0:1], 0x50
	s_branch label_LoadExternalStructEnd
label_LoadExternalStruct:
	s_mul_i32 s49, s49, 0xc4
	s_add_co_u32 s0, s0, s49
	s_add_co_ci_u32 s1, s1, 0
	s_load_b512 s[28:43], s[0:1], 0x10
	s_load_b32 s44, s[0:1], 0x50
	s_load_b32 s45, s[0:1], 0x60
label_LoadExternalStructEnd:
	v_and_b32_e32 v1, 31, v218
	v_and_b32_e32 v0, 15, v1
	v_lshlrev_b32_e32 v0, 2, v0
	v_lshrrev_b32_e32 v1, 4, v1
	v_lshl_add_u32 v0, v1, 10, v0
	v_lshrrev_b32_e32 v4, 5, v218
	v_and_b32_e32 v4, 1, v4
	v_lshl_add_u32 v0, v4, 6, v0
	v_and_b32_e32 v2, 31, v218
	v_and_b32_e32 v1, 15, v2
	v_lshlrev_b32_e32 v1, 2, v1
	v_lshrrev_b32_e32 v2, 4, v2
	v_lshl_add_u32 v1, v2, 10, v1
	v_lshrrev_b32_e32 v3, 6, v218
	v_and_b32_e32 v3, 1, v3
	v_lshl_add_u32 v1, v3, 6, v1
	v_lshrrev_b32_e32 v2, 5, v218
	v_lshrrev_b32_e32 v2, 2, v2
	s_mov_b32 s16, 0x800
	v_mul_lo_u32 v2, s16, v2
	v_add_lshl_u32 v134, v2, v0, 1
	v_lshrrev_b32_e32 v0, 5, v218
	v_lshrrev_b32_e32 v0, 2, v0
	v_mul_lo_u32 v0, s16, v0
	v_add_lshl_u32 v135, v0, v1, 1
	v_add_co_u32 v135, vcc_lo, 0x1000, v135
	v_lshrrev_b32_e32 v1, 4, v218
	v_and_b32_e32 v0, 15, v218
	v_lshlrev_b32_e32 v0, 3, v0
	v_mov_b32_e32 v4, v1
	v_lshrrev_b32_e32 v3, 4, v218
	v_and_b32_e32 v2, 15, v218
	v_lshlrev_b32_e32 v2, 3, v2
	v_mov_b32_e32 v5, v3
	v_mul_u32_u24_e32 v132, 0x80, v4
	v_add_lshl_u32 v132, v0, v132, 1
	v_mul_u32_u24_e32 v133, 0x80, v5
	v_add_lshl_u32 v133, v2, v133, 1
	v_add_co_u32 v133, vcc_lo, 0x1000, v133
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v8, 0x80
	v_mov_b32_e32 v7, s24
	v_cvt_f32_u32_e32 v6, v8
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v9, v7
	v_mul_f32_e32 v6, v6, v9
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e32 v9, v6, v8
	v_sub_nc_u32_e32 v9, v7, v9
	v_cmp_ne_u32_e64 vcc_lo, v9, 0
	v_add_co_ci_u32_e64 v6, vcc_lo, v6, 0, vcc_lo
	v_mov_b32_e32 v8, 0x80
	v_mov_b32_e32 v7, s25
	v_readfirstlane_b32 s14, v6
	v_cvt_f32_u32_e32 v6, v8
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v9, v7
	v_mul_f32_e32 v6, v6, v9
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e32 v9, v6, v8
	v_sub_nc_u32_e32 v9, v7, v9
	v_cmp_ne_u32_e64 vcc_lo, v9, 0
	v_add_co_ci_u32_e64 v6, vcc_lo, v6, 0, vcc_lo
	v_readfirstlane_b32 s15, v6
	s_cmp_eq_u32 s25, 0
	s_cbranch_scc0 label_NoEarlyStop_N0
label_EarlyStop_if_N_is_0:
	s_endpgm
label_NoEarlyStop_N0:
	s_mul_i32 s16, s14, s15
	s_and_b32 s17, s46, 0x3fff
	s_mul_i32 s16, s16, s17
	v_cvt_f32_u32_e32 v6, s16
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v7, s2
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s2, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s16, v6
	s_mov_b32 s4, s16
	s_mul_i32 s16, s15, s14
	s_mul_i32 s16, s16, s4
	s_mul_i32 s16, s16, s17
	s_sub_co_u32 s2, s2, s16
	v_cvt_f32_u32_e32 v6, s14
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v7, s2
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e64 v7, v6, s14
	v_sub_nc_u32_e32 v7, s2, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s14
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v6, 1, v6
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v7, s14
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s16, v6
	s_mov_b32 s3, s16
	s_mul_i32 s16, s3, s14
	s_sub_co_u32 s2, s2, s16
	s_cmp_ge_u32 s4, s26
	s_cbranch_scc0 label_MultiGemmEnd
label_EarlyStop_if_wg_exceed:
	s_endpgm
label_MultiGemmEnd:
label_NoEarlyStop_wgExceed:
	s_mov_b32 s66, 0x5040100
	s_mov_b32 s67, 0x7060302
	s_sub_co_u32 s32, s32, 16
	s_sub_co_ci_u32 s33, s33, 0
	s_sub_co_u32 s34, s34, 16
	s_sub_co_ci_u32 s35, s35, 0
	v_cmp_eq_f32_e64 vcc_lo, s44, 0
	s_cbranch_vccz label_AlphaNonZero
	s_mov_b32 s27, 0
label_AlphaNonZero:
	s_mov_b32 s2, ttmp9
	s_and_b32 s3, 0xffff, ttmp7
	s_lshr_b32 s4, ttmp7, 16
	s_and_b32 s16, s46, 0x3fff
	s_cmp_eq_u32 s16, 1
	s_cbranch_scc1 label_GSU
	s_and_b32 s16, s46, 0x4000
	s_cbranch_scc1 label_GSUWGMRR
	s_and_b32 s16, s46, 0x3fff
	v_cvt_f32_u32_e32 v6, s16
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v7, s3
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s3, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v6, 1, v6
	v_mov_b32_e32 v7, 0
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v7, s16
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v6, v6, 1
	v_mul_u32_u24_e64 v7, v6, s16
	v_sub_nc_u32_e32 v7, s3, v7
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s3, v6
	v_readfirstlane_b32 s6, v7
	s_branch label_GSUWGMRR_End
label_GSUWGMRR:
	v_cvt_f32_u32_e32 v6, s15
	v_rcp_iflag_f32_e32 v6, v6
	v_cvt_f32_u32_e32 v7, s3
	v_mul_f32_e32 v6, v6, v7
	v_cvt_u32_f32_e32 v6, v6
	v_mul_u32_u24_e64 v7, v6, s15
	v_sub_nc_u32_e32 v7, s3, v7
	v_cmp_eq_u32_e64 vcc_lo, v7, s15
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v6, 1, v6
	v_mov_b32_e32 v7, 0
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v7, s15
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v6, v6, 1
	v_mul_u32_u24_e64 v7, v6, s15
	v_sub_nc_u32_e32 v7, s3, v7
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s3, v7
label_GSUWGMRR_End:
	s_mov_b32 s8, 2
	s_mov_b32 s9, 2
	s_branch label_GSU_End
label_GSU:
	s_mov_b64 s[6:7], 0
	s_mov_b32 s8, 2
	s_mov_b32 s9, 2
label_GSU_End:
	s_mov_b32 s16, s11
	s_sext_i32_i16 s16, s16
	s_cmp_gt_i32 s16, 1
	s_cbranch_scc1 label_WGMPositive
	s_cmp_ge_i32 s16, 0
	s_cbranch_scc1 label_WGM
	s_abs_i32 s16, s16
	v_cvt_f64_u32_e32 v[6:7], s16
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s2
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s16
	v_sub_nc_u32_e32 v8, s2, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s16
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s17, v6
	s_mul_i32 s20, s17, s16
	s_sub_co_u32 s20, s2, s20
	s_mul_i32 s20, s20, s15
	s_add_co_u32 s20, s20, s3
	v_cvt_f64_u32_e32 v[6:7], s16
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s14
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s16
	v_sub_nc_u32_e32 v8, s14, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s16
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s18, v6
	s_mul_i32 s19, s16, s18
	s_sub_co_u32 s19, s14, s19
	s_cmp_eq_u32 s19, 0
	s_cmov_b32 s19, s16
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s18, s19, s16
	v_cvt_f64_u32_e32 v[6:7], s18
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s20
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s18
	v_sub_nc_u32_e32 v8, s20, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s18
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_mul_lo_u32 v7, v6, s18
	v_sub_nc_u32_e32 v8, s20, v7
	v_readfirstlane_b32 s3, v6
	v_readfirstlane_b32 s2, v8
	s_mul_i32 s2, s3, s18
	s_sub_co_u32 s2, s20, s2
	s_mul_i32 s17, s17, s16
	s_add_co_u32 s2, s2, s17
	s_branch label_WGM
label_WGMPositive:
	s_mov_b32 s16, s16
	v_cvt_f64_u32_e32 v[6:7], s16
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s3
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s16
	v_sub_nc_u32_e32 v8, s3, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s16
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s17, v6
	s_mul_i32 s20, s17, s16
	s_sub_co_u32 s20, s3, s20
	s_mul_i32 s20, s20, s14
	s_add_co_u32 s20, s20, s2
	v_cvt_f64_u32_e32 v[6:7], s16
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s15
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s16
	v_sub_nc_u32_e32 v8, s15, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s16
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s18, v6
	s_mul_i32 s19, s16, s18
	s_sub_co_u32 s19, s15, s19
	s_cmp_eq_u32 s19, 0
	s_cmov_b32 s19, s16
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s18, s19, s16
	v_cvt_f64_u32_e32 v[6:7], s18
	v_rcp_f64_e32 v[6:7], v[6:7]
	v_cvt_f64_u32_e32 v[8:9], s20
	v_mul_f64_e32 v[6:7], v[6:7], v[8:9]
	v_cvt_u32_f64_e32 v6, v[6:7]
	v_mul_lo_u32 v7, v6, s18
	v_sub_nc_u32_e32 v8, s20, v7
	v_cmp_ge_u32_e64 vcc_lo, v8, s18
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e64 v6, v6, 1
	s_mov_b32 exec_lo, -1
	v_mul_lo_u32 v7, v6, s18
	v_sub_nc_u32_e32 v8, s20, v7
	v_readfirstlane_b32 s2, v6
	v_readfirstlane_b32 s3, v8
	s_mul_i32 s3, s2, s18
	s_sub_co_u32 s3, s20, s3
	s_mul_i32 s17, s17, s16
	s_add_co_u32 s3, s3, s17
label_WGM:
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v1
	v_add_co_u32 v9, vcc_lo, 8, v8
	v_mov_b32_e32 v10, v3
	v_add_co_u32 v11, vcc_lo, 8, v10
	s_mul_i32 s16, s2, 0x80
	s_sub_co_u32 s16, s24, s16
	s_sub_co_u32 s16, s16, 8
	v_mov_b32_e32 v12, s16
	v_min_i32_e32 v6, v12, v6
	s_mul_i32 s16, s3, 0x80
	s_sub_co_u32 s16, s25, s16
	s_sub_co_u32 s16, s16, 8
	v_mov_b32_e32 v12, s16
	v_min_i32_e32 v7, v12, v7
	s_mul_hi_u32 s19, s2, 0x80
	s_mul_i32 s18, s2, 0x80
	s_and_b32 s16, s46, 0x8000
	s_cbranch_scc1 label_GSUC_A
	s_mul_hi_u32 s17, 16, s6
	s_mul_i32 s16, 16, s6
	s_branch label_GSUC_A_End
label_GSUC_A:
	s_lshr_b32 s12, s27, 4
	s_and_b32 s7, s46, 0x3fff
	v_cvt_f32_u32_e32 v12, s7
	v_rcp_iflag_f32_e32 v12, v12
	v_cvt_f32_u32_e32 v13, s12
	v_mul_f32_e32 v12, v12, v13
	v_cvt_u32_f32_e32 v12, v12
	v_mul_u32_u24_e64 v13, v12, s7
	v_sub_nc_u32_e32 v13, s12, v13
	v_cmp_eq_u32_e64 vcc_lo, v13, s7
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v12, 1, v12
	v_mov_b32_e32 v13, 0
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v13, s7
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v12, v12, 1
	v_mul_u32_u24_e64 v13, v12, s7
	v_sub_nc_u32_e32 v13, s12, v13
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s12, v12
	v_readfirstlane_b32 s7, v13
	s_mul_i32 s17, s12, s6
	s_add_co_u32 s16, 1, s12
	s_add_co_u32 s17, s17, s7
	s_mul_i32 s16, s16, s6
	s_cmp_lt_u32 s6, s7
	s_cselect_b32 s16, s16, s17
	s_mul_hi_u32 s17, s16, 16
	s_mul_i32 s16, s16, 16
label_GSUC_A_End:
	s_mul_hi_u32 s17, s16, s40
	s_mul_i32 s16, s16, s40
	s_add_co_u32 s18, s18, s16
	s_add_co_ci_u32 s19, s19, s17
	s_mov_b64 s[56:57], 1
	s_sub_co_u32 s16, s24, 1
	s_mul_hi_u32 s17, 1, s16
	s_mul_i32 s16, 1, s16
	s_add_co_u32 s56, s56, s16
	s_add_co_ci_u32 s57, s57, s17
	s_sub_co_u32 s16, s27, 1
	s_mul_hi_u32 s17, s40, s16
	s_mul_i32 s16, s40, s16
	s_add_co_u32 s56, s56, s16
	s_add_co_ci_u32 s57, s57, s17
	s_sub_co_u32 s56, s56, s18
	s_sub_co_ci_u32 s57, s57, s19
	s_lshl_b64 s[56:57], s[56:57], 1
	s_add_co_u32 s56, s56, 16
	s_add_co_ci_u32 s57, s57, 0
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1
	s_mul_hi_u32 s17, s41, s4
	s_mul_i32 s16, s41, s4
	s_add_co_u32 s18, s18, s16
	s_add_co_ci_u32 s19, s19, s17
	s_lshl_b64 s[18:19], s[18:19], 1
	s_add_co_u32 s48, s32, s18
	s_add_co_ci_u32 s49, s33, s19
	s_mov_b32 s51, 0x30020000
	s_mul_hi_u32 s19, s3, 0x80
	s_mul_i32 s18, s3, 0x80
	s_and_b32 s16, s46, 0x8000
	s_cbranch_scc1 label_GSUC_B
	s_mul_hi_u32 s17, 16, s6
	s_mul_i32 s16, 16, s6
	s_branch label_GSUC_B_End
label_GSUC_B:
	s_lshr_b32 s12, s27, 4
	s_and_b32 s7, s46, 0x3fff
	v_cvt_f32_u32_e32 v12, s7
	v_rcp_iflag_f32_e32 v12, v12
	v_cvt_f32_u32_e32 v13, s12
	v_mul_f32_e32 v12, v12, v13
	v_cvt_u32_f32_e32 v12, v12
	v_mul_u32_u24_e64 v13, v12, s7
	v_sub_nc_u32_e32 v13, s12, v13
	v_cmp_eq_u32_e64 vcc_lo, v13, s7
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v12, 1, v12
	v_mov_b32_e32 v13, 0
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v13, s7
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v12, v12, 1
	v_mul_u32_u24_e64 v13, v12, s7
	v_sub_nc_u32_e32 v13, s12, v13
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s12, v12
	v_readfirstlane_b32 s7, v13
	s_mul_i32 s17, s12, s6
	s_add_co_u32 s16, 1, s12
	s_add_co_u32 s17, s17, s7
	s_mul_i32 s16, s16, s6
	s_cmp_lt_u32 s6, s7
	s_cselect_b32 s16, s16, s17
	s_mul_hi_u32 s17, s16, 16
	s_mul_i32 s16, s16, 16
label_GSUC_B_End:
	s_mul_hi_u32 s17, s16, s42
	s_mul_i32 s16, s16, s42
	s_add_co_u32 s18, s18, s16
	s_add_co_ci_u32 s19, s19, s17
	s_mov_b64 s[58:59], 1
	s_sub_co_u32 s16, s25, 1
	s_mul_hi_u32 s17, 1, s16
	s_mul_i32 s16, 1, s16
	s_add_co_u32 s58, s58, s16
	s_add_co_ci_u32 s59, s59, s17
	s_sub_co_u32 s16, s27, 1
	s_mul_hi_u32 s17, s42, s16
	s_mul_i32 s16, s42, s16
	s_add_co_u32 s58, s58, s16
	s_add_co_ci_u32 s59, s59, s17
	s_sub_co_u32 s58, s58, s18
	s_sub_co_ci_u32 s59, s59, s19
	s_lshl_b64 s[58:59], s[58:59], 1
	s_add_co_u32 s58, s58, 16
	s_add_co_ci_u32 s59, s59, 0
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1
	s_mul_hi_u32 s17, s43, s4
	s_mul_i32 s16, s43, s4
	s_add_co_u32 s18, s18, s16
	s_add_co_ci_u32 s19, s19, s17
	s_lshl_b64 s[18:19], s[18:19], 1
	s_add_co_u32 s52, s34, s18
	s_add_co_ci_u32 s53, s35, s19
	s_mov_b32 s55, 0x30020000
	v_mul_lo_u32 v12, s40, v8
	v_add_co_u32 v128, vcc_lo, v6, v12
	v_add_nc_u32_e32 v128, 8, v128
	v_lshlrev_b32_e32 v128, 1, v128
	v_mul_lo_u32 v12, s40, v9
	v_add_co_u32 v129, vcc_lo, v6, v12
	v_add_nc_u32_e32 v129, 8, v129
	v_lshlrev_b32_e32 v129, 1, v129
	v_mul_lo_u32 v12, s42, v10
	v_add_co_u32 v130, vcc_lo, v7, v12
	v_add_nc_u32_e32 v130, 8, v130
	v_lshlrev_b32_e32 v130, 1, v130
	v_mul_lo_u32 v12, s42, v11
	v_add_co_u32 v131, vcc_lo, v7, v12
	v_add_nc_u32_e32 v131, 8, v131
	v_lshlrev_b32_e32 v131, 1, v131
	s_and_b32 s17, s46, 0x3fff
	s_mul_i32 s17, s17, 32
	s_and_b32 s16, s46, 0x8000
	s_cmov_b32 s17, 32
	s_mul_i32 s64, s17, s40
	s_and_b32 s17, s46, 0x3fff
	s_mul_i32 s17, s17, 32
	s_and_b32 s16, s46, 0x8000
	s_cmov_b32 s17, 32
	s_mul_i32 s65, s17, s42
	s_lshr_b32 s12, s27, 4
	s_and_b32 s16, s46, 0x3fff
	s_cmp_eq_u32 s16, 1
	s_cbranch_scc1 label_GSU_1
	s_and_b32 s7, s46, 0x3fff
	v_cvt_f32_u32_e32 v0, s7
	v_rcp_iflag_f32_e32 v0, v0
	v_cvt_f32_u32_e32 v1, s12
	v_mul_f32_e32 v0, v0, v1
	v_cvt_u32_f32_e32 v0, v0
	v_mul_u32_u24_e64 v1, v0, s7
	v_sub_nc_u32_e32 v1, s12, v1
	v_cmp_eq_u32_e64 vcc_lo, v1, s7
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v1, s7
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v0, v0, 1
	v_mul_u32_u24_e64 v1, v0, s7
	v_sub_nc_u32_e32 v1, s12, v1
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s7, v1
	s_add_co_u32 s16, 1, s12
	s_cmp_lt_u32 s6, s7
	s_cmov_b32 s12, s16
label_GSU_1:
	s_mov_b32 s13, s12
	s_and_b32 s18, s10, 0x1f00
	s_lshr_b32 s18, s18, 8
	s_and_b32 s19, s10, 0xe000
	s_and_b32 s10, s10, 0xff
	s_mov_b32 s16, s10
label_beginStaggerUIter:
	s_lshl_b32 s17, s16, s18
	s_cmp_ge_u32 s13, s17
	s_cbranch_scc1 label_endStaggerUIter
	s_lshr_b32 s16, s16, 1
	s_branch label_beginStaggerUIter
label_endStaggerUIter:
	s_sub_co_u32 s17, s16, 1
	s_cmp_ge_u32 s16, 1
	s_cselect_b32 s47, s17, 0
	s_cmp_eq_u32 s19, 0
	s_cbranch_scc1 label_StaggerUMapping_1
	s_mov_b32 s16, s2
	s_branch label_staggerInputEnd
label_StaggerUMapping_1:
	s_cmp_eq_u32 s19, 0x2000
	s_cbranch_scc1 label_StaggerUMapping_2
	s_mov_b32 s16, s3
	s_branch label_staggerInputEnd
label_StaggerUMapping_2:
	s_cmp_eq_u32 s19, 0x4000
	s_cbranch_scc1 label_StaggerUMapping_3
	s_mov_b32 s16, -1
	s_branch label_staggerInputEnd
label_StaggerUMapping_3:
	s_cmp_eq_u32 s19, 0x6000
	s_cbranch_scc1 label_StaggerUMapping_4
	s_mul_i32 s17, s14, s3
	s_add_co_u32 s16, s16, s17
	s_add_co_u32 s16, s16, s2
	s_branch label_staggerInputEnd
label_StaggerUMapping_4:
	s_cmp_eq_u32 s19, 0x8000
	s_cbranch_scc1 label_staggerInputEnd
	s_mov_b32 s16, -1
	s_branch label_staggerInputEnd
label_staggerInputEnd:
	s_and_b32 s47, s47, s16
	s_lshl_b32 s47, s47, s18
	s_mul_hi_i32 s17, s47, s64
	s_mul_i32 s16, s47, s64
	s_mul_hi_i32 s61, s12, s64
	s_mul_i32 s60, s12, s64
	s_sub_co_u32 s60, s64, s60
	s_sub_co_ci_u32 s61, 0, s61
	s_add_co_u32 s48, s48, s16
	s_add_co_ci_u32 s49, s49, s17
	s_sub_co_u32 s56, s56, s16
	s_sub_co_ci_u32 s57, s57, s17
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1
	s_mul_hi_i32 s17, s47, s65
	s_mul_i32 s16, s47, s65
	s_mul_hi_i32 s63, s12, s65
	s_mul_i32 s62, s12, s65
	s_sub_co_u32 s62, s65, s62
	s_sub_co_ci_u32 s63, 0, s63
	s_add_co_u32 s52, s52, s16
	s_add_co_ci_u32 s53, s53, s17
	s_sub_co_u32 s58, s58, s16
	s_sub_co_ci_u32 s59, s59, s17
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1
	s_add_co_u32 s47, s47, 2
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc1 label_ShadowInitStart
	buffer_load_b128 v[202:205], v128, s[48:51], null offen
	buffer_load_b128 v[206:209], v129, s[48:51], null offen
	buffer_load_b128 v[210:213], v130, s[52:55], null offen
	buffer_load_b128 v[214:217], v131, s[52:55], null offen
	s_add_co_u32 s18, s12, 1
	s_cmp_eq_u32 s47, s18
	s_cselect_b32 s16, s60, s64
	s_cselect_b32 s17, s61, 0
	s_add_co_u32 s48, s48, s16
	s_add_co_ci_u32 s49, s49, s17
	s_sub_co_u32 s56, s56, s16
	s_sub_co_ci_u32 s57, s57, s17
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1
	s_add_co_u32 s18, s12, 1
	s_cmp_eq_u32 s47, s18
	s_cselect_b32 s16, s62, s65
	s_cselect_b32 s17, s63, 0
	s_add_co_u32 s52, s52, s16
	s_add_co_ci_u32 s53, s53, s17
	s_sub_co_u32 s58, s58, s16
	s_sub_co_ci_u32 s59, s59, s17
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1
label_ShadowInitStart:
	s_mov_b64 s[16:17], s[28:29]
	s_mov_b32 s18, 0x80000000
	s_mov_b32 s19, 0x30020000
	s_mov_b64 s[20:21], s[30:31]
	s_mov_b32 s22, 0x80000000
	s_mov_b32 s23, 0x30020000
	s_mul_i32 s70, 0x80, s3
	s_mul_hi_u32 s69, s70, s38
	s_mul_i32 s68, s70, s38
	s_lshl_b64 s[68:69], s[68:69], s8
	s_add_co_u32 s20, s30, s68
	s_add_co_ci_u32 s21, s31, s69
	s_mul_hi_u32 s69, s70, s36
	s_mul_i32 s68, s70, s36
	s_lshl_b64 s[68:69], s[68:69], s9
	s_add_co_u32 s16, s28, s68
	s_add_co_ci_u32 s17, s29, s69
	s_mul_hi_u32 s69, s4, s39
	s_mul_i32 s68, s4, s39
	s_lshl_b64 s[68:69], s[68:69], s8
	s_add_co_u32 s20, s20, s68
	s_add_co_ci_u32 s21, s21, s69
	s_mul_hi_u32 s69, s4, s37
	s_mul_i32 s68, s4, s37
	s_lshl_b64 s[68:69], s[68:69], s9
	s_add_co_u32 s16, s16, s68
	s_add_co_ci_u32 s17, s17, s69
	s_and_b32 s68, s46, 0x3fff
	s_cmp_eq_u32 s68, 1
	s_cbranch_scc1 label_GSU_2
	s_mul_hi_u32 s69, s24, s6
	s_mul_i32 s68, s24, s6
	s_sub_co_u32 s70, s25, 1
	s_mul_i32 s70, s70, s6
	s_mul_hi_u32 s71, s70, s38
	s_mul_i32 s70, s70, s38
	s_add_co_u32 s68, s68, s70
	s_add_co_ci_u32 s69, s69, s71
	s_sub_co_u32 s70, s26, 1
	s_mul_i32 s70, s70, s6
	s_mul_hi_u32 s71, s70, s39
	s_mul_i32 s70, s70, s39
	s_add_co_u32 s68, s68, s70
	s_add_co_ci_u32 s69, s69, s71
	s_lshl_b64 s[68:69], s[68:69], 2
	s_add_co_u32 s16, s16, s68
	s_add_co_ci_u32 s17, s17, s69
label_GSU_2:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v14, 0
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v20, 0
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v24, 0
	v_mov_b32_e32 v25, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v127, 0
	s_cmp_eq_u32 s12, 0
	s_cbranch_scc0 label_NoBranch_SQN84M7GV65IV2UO
	s_getpc_b64 s[68:69]
Lgpc_after_0:
	s_add_co_i32 s70, label_PrefetchGlobalLastIterEnd-Lgpc_after_0-4, 4
	s_add_co_u32 s68, s68, s70
	s_add_co_ci_u32 s69, s69, 0
	s_setpc_b64 s[68:69]
label_NoBranch_SQN84M7GV65IV2UO:
	s_wait_loadcnt 0x0
	ds_store_b128 v132, v[202:205]
	ds_store_b128 v132, v[206:209] offset:2048
	ds_store_b128 v133, v[210:213]
	ds_store_b128 v133, v[214:217] offset:2048
label_openLoopL:
	s_cmp_le_u32 s12, 1
	s_cbranch_scc1 label_LoopEndL
	s_nop 0
	s_nop 0
	s_nop 0
label_LoopBeginL:
	buffer_load_b128 v[202:205], v128, s[48:51], null offen
	buffer_load_b128 v[206:209], v129, s[48:51], null offen
	buffer_load_b128 v[210:213], v130, s[52:55], null offen
	buffer_load_b128 v[214:217], v131, s[52:55], null offen
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s68, s60, s64
	s_cselect_b32 s69, s61, 0
	s_add_co_u32 s48, s48, s68
	s_add_co_ci_u32 s49, s49, s69
	s_sub_co_u32 s56, s56, s68
	s_sub_co_ci_u32 s57, s57, s69
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s68, s62, s65
	s_cselect_b32 s69, s63, 0
	s_add_co_u32 s52, s52, s68
	s_add_co_ci_u32 s53, s53, s69
	s_sub_co_u32 s58, s58, s68
	s_sub_co_ci_u32 s59, s59, s69
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b64 v[152:153], v134
	ds_load_b64 v[154:155], v134 offset:256
	ds_load_b64 v[156:157], v134 offset:512
	ds_load_b64 v[158:159], v134 offset:768
	ds_load_b64 v[160:161], v134 offset:1024
	ds_load_b64 v[162:163], v134 offset:1280
	ds_load_b64 v[164:165], v134 offset:1536
	ds_load_b64 v[166:167], v134 offset:1792
	ds_load_b64 v[185:186], v135
	ds_load_b64 v[187:188], v135 offset:256
	ds_load_b64 v[189:190], v135 offset:512
	ds_load_b64 v[191:192], v135 offset:768
	ds_load_b64 v[193:194], v135 offset:1024
	ds_load_b64 v[195:196], v135 offset:1280
	ds_load_b64 v[197:198], v135 offset:1536
	ds_load_b64 v[199:200], v135 offset:1792
	s_wait_dscnt 0x0
	v_perm_b32 v136, v154, v152, s66
	v_perm_b32 v137, v158, v156, s66
	v_perm_b32 v138, v162, v160, s66
	v_perm_b32 v139, v166, v164, s66
	v_perm_b32 v169, v187, v185, s66
	v_perm_b32 v170, v191, v189, s66
	v_perm_b32 v171, v195, v193, s66
	v_perm_b32 v172, v199, v197, s66
	v_perm_b32 v140, v154, v152, s67
	v_perm_b32 v141, v158, v156, s67
	v_perm_b32 v142, v162, v160, s67
	v_perm_b32 v143, v166, v164, s67
	s_setprio 3
	v_wmma_f32_16x16x16_f16 v[224:231], v[169:172], v[136:139], 0
	v_perm_b32 v144, v155, v153, s66
	v_perm_b32 v145, v159, v157, s66
	v_perm_b32 v146, v163, v161, s66
	v_perm_b32 v147, v167, v165, s66
	v_perm_b32 v148, v155, v153, s67
	v_perm_b32 v149, v159, v157, s67
	v_perm_b32 v150, v163, v161, s67
	v_perm_b32 v151, v167, v165, s67
	v_wmma_f32_16x16x16_f16 v[232:239], v[169:172], v[140:143], 0
	v_dual_add_f32 v0, v0, v224 :: v_dual_add_f32 v1, v1, v225
	v_dual_add_f32 v2, v2, v226 :: v_dual_add_f32 v3, v3, v227
	v_dual_add_f32 v4, v4, v228 :: v_dual_add_f32 v5, v5, v229
	v_dual_add_f32 v6, v6, v230 :: v_dual_add_f32 v7, v7, v231
	v_perm_b32 v173, v187, v185, s67
	v_perm_b32 v174, v191, v189, s67
	v_perm_b32 v175, v195, v193, s67
	v_perm_b32 v176, v199, v197, s67
	v_perm_b32 v177, v188, v186, s66
	v_perm_b32 v178, v192, v190, s66
	v_perm_b32 v179, v196, v194, s66
	v_perm_b32 v180, v200, v198, s66
	v_wmma_f32_16x16x16_f16 v[224:231], v[169:172], v[144:147], 0
	v_dual_add_f32 v8, v8, v232 :: v_dual_add_f32 v9, v9, v233
	v_dual_add_f32 v10, v10, v234 :: v_dual_add_f32 v11, v11, v235
	v_dual_add_f32 v12, v12, v236 :: v_dual_add_f32 v13, v13, v237
	v_dual_add_f32 v14, v14, v238 :: v_dual_add_f32 v15, v15, v239
	v_perm_b32 v181, v188, v186, s67
	v_perm_b32 v182, v192, v190, s67
	v_perm_b32 v183, v196, v194, s67
	v_perm_b32 v184, v200, v198, s67
	v_wmma_f32_16x16x16_f16 v[232:239], v[169:172], v[148:151], 0
	v_dual_add_f32 v16, v16, v224 :: v_dual_add_f32 v17, v17, v225
	v_dual_add_f32 v18, v18, v226 :: v_dual_add_f32 v19, v19, v227
	v_dual_add_f32 v20, v20, v228 :: v_dual_add_f32 v21, v21, v229
	v_dual_add_f32 v22, v22, v230 :: v_dual_add_f32 v23, v23, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[173:176], v[136:139], 0
	v_dual_add_f32 v24, v24, v232 :: v_dual_add_f32 v25, v25, v233
	v_dual_add_f32 v26, v26, v234 :: v_dual_add_f32 v27, v27, v235
	v_dual_add_f32 v28, v28, v236 :: v_dual_add_f32 v29, v29, v237
	v_dual_add_f32 v30, v30, v238 :: v_dual_add_f32 v31, v31, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[173:176], v[140:143], 0
	v_dual_add_f32 v32, v32, v224 :: v_dual_add_f32 v33, v33, v225
	v_dual_add_f32 v34, v34, v226 :: v_dual_add_f32 v35, v35, v227
	v_dual_add_f32 v36, v36, v228 :: v_dual_add_f32 v37, v37, v229
	v_dual_add_f32 v38, v38, v230 :: v_dual_add_f32 v39, v39, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[173:176], v[144:147], 0
	v_dual_add_f32 v40, v40, v232 :: v_dual_add_f32 v41, v41, v233
	v_dual_add_f32 v42, v42, v234 :: v_dual_add_f32 v43, v43, v235
	v_dual_add_f32 v44, v44, v236 :: v_dual_add_f32 v45, v45, v237
	v_dual_add_f32 v46, v46, v238 :: v_dual_add_f32 v47, v47, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[173:176], v[148:151], 0
	v_dual_add_f32 v48, v48, v224 :: v_dual_add_f32 v49, v49, v225
	v_dual_add_f32 v50, v50, v226 :: v_dual_add_f32 v51, v51, v227
	v_dual_add_f32 v52, v52, v228 :: v_dual_add_f32 v53, v53, v229
	v_dual_add_f32 v54, v54, v230 :: v_dual_add_f32 v55, v55, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[177:180], v[136:139], 0
	v_dual_add_f32 v56, v56, v232 :: v_dual_add_f32 v57, v57, v233
	v_dual_add_f32 v58, v58, v234 :: v_dual_add_f32 v59, v59, v235
	v_dual_add_f32 v60, v60, v236 :: v_dual_add_f32 v61, v61, v237
	v_dual_add_f32 v62, v62, v238 :: v_dual_add_f32 v63, v63, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[177:180], v[140:143], 0
	v_dual_add_f32 v64, v64, v224 :: v_dual_add_f32 v65, v65, v225
	v_dual_add_f32 v66, v66, v226 :: v_dual_add_f32 v67, v67, v227
	v_dual_add_f32 v68, v68, v228 :: v_dual_add_f32 v69, v69, v229
	v_dual_add_f32 v70, v70, v230 :: v_dual_add_f32 v71, v71, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[177:180], v[144:147], 0
	v_dual_add_f32 v72, v72, v232 :: v_dual_add_f32 v73, v73, v233
	v_dual_add_f32 v74, v74, v234 :: v_dual_add_f32 v75, v75, v235
	v_dual_add_f32 v76, v76, v236 :: v_dual_add_f32 v77, v77, v237
	v_dual_add_f32 v78, v78, v238 :: v_dual_add_f32 v79, v79, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[177:180], v[148:151], 0
	v_dual_add_f32 v80, v80, v224 :: v_dual_add_f32 v81, v81, v225
	v_dual_add_f32 v82, v82, v226 :: v_dual_add_f32 v83, v83, v227
	v_dual_add_f32 v84, v84, v228 :: v_dual_add_f32 v85, v85, v229
	v_dual_add_f32 v86, v86, v230 :: v_dual_add_f32 v87, v87, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[181:184], v[136:139], 0
	v_dual_add_f32 v88, v88, v232 :: v_dual_add_f32 v89, v89, v233
	v_dual_add_f32 v90, v90, v234 :: v_dual_add_f32 v91, v91, v235
	v_dual_add_f32 v92, v92, v236 :: v_dual_add_f32 v93, v93, v237
	v_dual_add_f32 v94, v94, v238 :: v_dual_add_f32 v95, v95, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[181:184], v[140:143], 0
	v_dual_add_f32 v96, v96, v224 :: v_dual_add_f32 v97, v97, v225
	v_dual_add_f32 v98, v98, v226 :: v_dual_add_f32 v99, v99, v227
	v_dual_add_f32 v100, v100, v228 :: v_dual_add_f32 v101, v101, v229
	v_dual_add_f32 v102, v102, v230 :: v_dual_add_f32 v103, v103, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[181:184], v[144:147], 0
	v_dual_add_f32 v104, v104, v232 :: v_dual_add_f32 v105, v105, v233
	v_dual_add_f32 v106, v106, v234 :: v_dual_add_f32 v107, v107, v235
	v_dual_add_f32 v108, v108, v236 :: v_dual_add_f32 v109, v109, v237
	v_dual_add_f32 v110, v110, v238 :: v_dual_add_f32 v111, v111, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[181:184], v[148:151], 0
	v_dual_add_f32 v112, v112, v224 :: v_dual_add_f32 v113, v113, v225
	v_dual_add_f32 v114, v114, v226 :: v_dual_add_f32 v115, v115, v227
	v_dual_add_f32 v116, v116, v228 :: v_dual_add_f32 v117, v117, v229
	v_dual_add_f32 v118, v118, v230 :: v_dual_add_f32 v119, v119, v231
	s_setprio 1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_wait_loadcnt 0x3
	ds_store_b128 v132, v[202:205]
	s_wait_loadcnt 0x2
	ds_store_b128 v132, v[206:209] offset:2048
	s_wait_loadcnt 0x1
	ds_store_b128 v133, v[210:213]
	s_wait_loadcnt 0x0
	ds_store_b128 v133, v[214:217] offset:2048
	s_setprio 2
	s_sub_co_u32 s12, s12, 1
	s_cmp_eq_i32 s12, 1
	v_dual_add_f32 v120, v120, v232 :: v_dual_add_f32 v121, v121, v233
	v_dual_add_f32 v122, v122, v234 :: v_dual_add_f32 v123, v123, v235
	v_dual_add_f32 v124, v124, v236 :: v_dual_add_f32 v125, v125, v237
	v_dual_add_f32 v126, v126, v238 :: v_dual_add_f32 v127, v127, v239
	s_cbranch_scc1 label_LoopEndL
	buffer_load_b128 v[202:205], v128, s[48:51], null offen
	buffer_load_b128 v[206:209], v129, s[48:51], null offen
	buffer_load_b128 v[210:213], v130, s[52:55], null offen
	buffer_load_b128 v[214:217], v131, s[52:55], null offen
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s68, s60, s64
	s_cselect_b32 s69, s61, 0
	s_add_co_u32 s48, s48, s68
	s_add_co_ci_u32 s49, s49, s69
	s_sub_co_u32 s56, s56, s68
	s_sub_co_ci_u32 s57, s57, s69
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1
	s_cmp_eq_u32 s12, s47
	s_cselect_b32 s68, s62, s65
	s_cselect_b32 s69, s63, 0
	s_add_co_u32 s52, s52, s68
	s_add_co_ci_u32 s53, s53, s69
	s_sub_co_u32 s58, s58, s68
	s_sub_co_ci_u32 s59, s59, s69
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b64 v[152:153], v134
	ds_load_b64 v[154:155], v134 offset:256
	ds_load_b64 v[156:157], v134 offset:512
	ds_load_b64 v[158:159], v134 offset:768
	ds_load_b64 v[160:161], v134 offset:1024
	ds_load_b64 v[162:163], v134 offset:1280
	ds_load_b64 v[164:165], v134 offset:1536
	ds_load_b64 v[166:167], v134 offset:1792
	ds_load_b64 v[185:186], v135
	ds_load_b64 v[187:188], v135 offset:256
	ds_load_b64 v[189:190], v135 offset:512
	ds_load_b64 v[191:192], v135 offset:768
	ds_load_b64 v[193:194], v135 offset:1024
	ds_load_b64 v[195:196], v135 offset:1280
	ds_load_b64 v[197:198], v135 offset:1536
	ds_load_b64 v[199:200], v135 offset:1792
	s_wait_dscnt 0x0
	v_perm_b32 v136, v154, v152, s66
	v_perm_b32 v137, v158, v156, s66
	v_perm_b32 v138, v162, v160, s66
	v_perm_b32 v139, v166, v164, s66
	v_perm_b32 v169, v187, v185, s66
	v_perm_b32 v170, v191, v189, s66
	v_perm_b32 v171, v195, v193, s66
	v_perm_b32 v172, v199, v197, s66
	v_perm_b32 v140, v154, v152, s67
	v_perm_b32 v141, v158, v156, s67
	v_perm_b32 v142, v162, v160, s67
	v_perm_b32 v143, v166, v164, s67
	s_setprio 3
	v_wmma_f32_16x16x16_f16 v[224:231], v[169:172], v[136:139], 0
	v_perm_b32 v144, v155, v153, s66
	v_perm_b32 v145, v159, v157, s66
	v_perm_b32 v146, v163, v161, s66
	v_perm_b32 v147, v167, v165, s66
	v_perm_b32 v148, v155, v153, s67
	v_perm_b32 v149, v159, v157, s67
	v_perm_b32 v150, v163, v161, s67
	v_perm_b32 v151, v167, v165, s67
	v_wmma_f32_16x16x16_f16 v[232:239], v[169:172], v[140:143], 0
	v_dual_add_f32 v0, v0, v224 :: v_dual_add_f32 v1, v1, v225
	v_dual_add_f32 v2, v2, v226 :: v_dual_add_f32 v3, v3, v227
	v_dual_add_f32 v4, v4, v228 :: v_dual_add_f32 v5, v5, v229
	v_dual_add_f32 v6, v6, v230 :: v_dual_add_f32 v7, v7, v231
	v_perm_b32 v173, v187, v185, s67
	v_perm_b32 v174, v191, v189, s67
	v_perm_b32 v175, v195, v193, s67
	v_perm_b32 v176, v199, v197, s67
	v_perm_b32 v177, v188, v186, s66
	v_perm_b32 v178, v192, v190, s66
	v_perm_b32 v179, v196, v194, s66
	v_perm_b32 v180, v200, v198, s66
	v_wmma_f32_16x16x16_f16 v[224:231], v[169:172], v[144:147], 0
	v_dual_add_f32 v8, v8, v232 :: v_dual_add_f32 v9, v9, v233
	v_dual_add_f32 v10, v10, v234 :: v_dual_add_f32 v11, v11, v235
	v_dual_add_f32 v12, v12, v236 :: v_dual_add_f32 v13, v13, v237
	v_dual_add_f32 v14, v14, v238 :: v_dual_add_f32 v15, v15, v239
	v_perm_b32 v181, v188, v186, s67
	v_perm_b32 v182, v192, v190, s67
	v_perm_b32 v183, v196, v194, s67
	v_perm_b32 v184, v200, v198, s67
	v_wmma_f32_16x16x16_f16 v[232:239], v[169:172], v[148:151], 0
	v_dual_add_f32 v16, v16, v224 :: v_dual_add_f32 v17, v17, v225
	v_dual_add_f32 v18, v18, v226 :: v_dual_add_f32 v19, v19, v227
	v_dual_add_f32 v20, v20, v228 :: v_dual_add_f32 v21, v21, v229
	v_dual_add_f32 v22, v22, v230 :: v_dual_add_f32 v23, v23, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[173:176], v[136:139], 0
	v_dual_add_f32 v24, v24, v232 :: v_dual_add_f32 v25, v25, v233
	v_dual_add_f32 v26, v26, v234 :: v_dual_add_f32 v27, v27, v235
	v_dual_add_f32 v28, v28, v236 :: v_dual_add_f32 v29, v29, v237
	v_dual_add_f32 v30, v30, v238 :: v_dual_add_f32 v31, v31, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[173:176], v[140:143], 0
	v_dual_add_f32 v32, v32, v224 :: v_dual_add_f32 v33, v33, v225
	v_dual_add_f32 v34, v34, v226 :: v_dual_add_f32 v35, v35, v227
	v_dual_add_f32 v36, v36, v228 :: v_dual_add_f32 v37, v37, v229
	v_dual_add_f32 v38, v38, v230 :: v_dual_add_f32 v39, v39, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[173:176], v[144:147], 0
	v_dual_add_f32 v40, v40, v232 :: v_dual_add_f32 v41, v41, v233
	v_dual_add_f32 v42, v42, v234 :: v_dual_add_f32 v43, v43, v235
	v_dual_add_f32 v44, v44, v236 :: v_dual_add_f32 v45, v45, v237
	v_dual_add_f32 v46, v46, v238 :: v_dual_add_f32 v47, v47, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[173:176], v[148:151], 0
	v_dual_add_f32 v48, v48, v224 :: v_dual_add_f32 v49, v49, v225
	v_dual_add_f32 v50, v50, v226 :: v_dual_add_f32 v51, v51, v227
	v_dual_add_f32 v52, v52, v228 :: v_dual_add_f32 v53, v53, v229
	v_dual_add_f32 v54, v54, v230 :: v_dual_add_f32 v55, v55, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[177:180], v[136:139], 0
	v_dual_add_f32 v56, v56, v232 :: v_dual_add_f32 v57, v57, v233
	v_dual_add_f32 v58, v58, v234 :: v_dual_add_f32 v59, v59, v235
	v_dual_add_f32 v60, v60, v236 :: v_dual_add_f32 v61, v61, v237
	v_dual_add_f32 v62, v62, v238 :: v_dual_add_f32 v63, v63, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[177:180], v[140:143], 0
	v_dual_add_f32 v64, v64, v224 :: v_dual_add_f32 v65, v65, v225
	v_dual_add_f32 v66, v66, v226 :: v_dual_add_f32 v67, v67, v227
	v_dual_add_f32 v68, v68, v228 :: v_dual_add_f32 v69, v69, v229
	v_dual_add_f32 v70, v70, v230 :: v_dual_add_f32 v71, v71, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[177:180], v[144:147], 0
	v_dual_add_f32 v72, v72, v232 :: v_dual_add_f32 v73, v73, v233
	v_dual_add_f32 v74, v74, v234 :: v_dual_add_f32 v75, v75, v235
	v_dual_add_f32 v76, v76, v236 :: v_dual_add_f32 v77, v77, v237
	v_dual_add_f32 v78, v78, v238 :: v_dual_add_f32 v79, v79, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[177:180], v[148:151], 0
	v_dual_add_f32 v80, v80, v224 :: v_dual_add_f32 v81, v81, v225
	v_dual_add_f32 v82, v82, v226 :: v_dual_add_f32 v83, v83, v227
	v_dual_add_f32 v84, v84, v228 :: v_dual_add_f32 v85, v85, v229
	v_dual_add_f32 v86, v86, v230 :: v_dual_add_f32 v87, v87, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[181:184], v[136:139], 0
	v_dual_add_f32 v88, v88, v232 :: v_dual_add_f32 v89, v89, v233
	v_dual_add_f32 v90, v90, v234 :: v_dual_add_f32 v91, v91, v235
	v_dual_add_f32 v92, v92, v236 :: v_dual_add_f32 v93, v93, v237
	v_dual_add_f32 v94, v94, v238 :: v_dual_add_f32 v95, v95, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[181:184], v[140:143], 0
	v_dual_add_f32 v96, v96, v224 :: v_dual_add_f32 v97, v97, v225
	v_dual_add_f32 v98, v98, v226 :: v_dual_add_f32 v99, v99, v227
	v_dual_add_f32 v100, v100, v228 :: v_dual_add_f32 v101, v101, v229
	v_dual_add_f32 v102, v102, v230 :: v_dual_add_f32 v103, v103, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[181:184], v[144:147], 0
	v_dual_add_f32 v104, v104, v232 :: v_dual_add_f32 v105, v105, v233
	v_dual_add_f32 v106, v106, v234 :: v_dual_add_f32 v107, v107, v235
	v_dual_add_f32 v108, v108, v236 :: v_dual_add_f32 v109, v109, v237
	v_dual_add_f32 v110, v110, v238 :: v_dual_add_f32 v111, v111, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[181:184], v[148:151], 0
	v_dual_add_f32 v112, v112, v224 :: v_dual_add_f32 v113, v113, v225
	v_dual_add_f32 v114, v114, v226 :: v_dual_add_f32 v115, v115, v227
	v_dual_add_f32 v116, v116, v228 :: v_dual_add_f32 v117, v117, v229
	v_dual_add_f32 v118, v118, v230 :: v_dual_add_f32 v119, v119, v231
	s_setprio 1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_wait_loadcnt 0x3
	ds_store_b128 v132, v[202:205]
	s_wait_loadcnt 0x2
	ds_store_b128 v132, v[206:209] offset:2048
	s_wait_loadcnt 0x1
	ds_store_b128 v133, v[210:213]
	s_wait_loadcnt 0x0
	ds_store_b128 v133, v[214:217] offset:2048
	s_setprio 2
	s_sub_co_u32 s12, s12, 1
	s_cmp_eq_i32 s12, 1
	v_dual_add_f32 v120, v120, v232 :: v_dual_add_f32 v121, v121, v233
	v_dual_add_f32 v122, v122, v234 :: v_dual_add_f32 v123, v123, v235
	v_dual_add_f32 v124, v124, v236 :: v_dual_add_f32 v125, v125, v237
	v_dual_add_f32 v126, v126, v238 :: v_dual_add_f32 v127, v127, v239
	s_cbranch_scc0 label_LoopBeginL
label_LoopEndL:
label_LoopEndL_oddexit:
label_LoopEndL_evenexit:
	s_and_b32 s8, s46, 0x3fff
	s_cmp_eq_u32 s8, 1
	s_cbranch_scc0 label_GSU_3
label_GSU_3:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b64 v[152:153], v134
	ds_load_b64 v[154:155], v134 offset:256
	ds_load_b64 v[156:157], v134 offset:512
	ds_load_b64 v[158:159], v134 offset:768
	ds_load_b64 v[160:161], v134 offset:1024
	ds_load_b64 v[162:163], v134 offset:1280
	ds_load_b64 v[164:165], v134 offset:1536
	ds_load_b64 v[166:167], v134 offset:1792
	ds_load_b64 v[185:186], v135
	ds_load_b64 v[187:188], v135 offset:256
	ds_load_b64 v[189:190], v135 offset:512
	ds_load_b64 v[191:192], v135 offset:768
	ds_load_b64 v[193:194], v135 offset:1024
	ds_load_b64 v[195:196], v135 offset:1280
	ds_load_b64 v[197:198], v135 offset:1536
	ds_load_b64 v[199:200], v135 offset:1792
	s_wait_dscnt 0x0
	v_perm_b32 v136, v154, v152, s66
	v_perm_b32 v137, v158, v156, s66
	v_perm_b32 v138, v162, v160, s66
	v_perm_b32 v139, v166, v164, s66
	v_perm_b32 v169, v187, v185, s66
	v_perm_b32 v170, v191, v189, s66
	v_perm_b32 v171, v195, v193, s66
	v_perm_b32 v172, v199, v197, s66
	v_perm_b32 v140, v154, v152, s67
	v_perm_b32 v141, v158, v156, s67
	v_perm_b32 v142, v162, v160, s67
	v_perm_b32 v143, v166, v164, s67
	s_setprio 3
	v_wmma_f32_16x16x16_f16 v[224:231], v[169:172], v[136:139], 0
	v_perm_b32 v144, v155, v153, s66
	v_perm_b32 v145, v159, v157, s66
	v_perm_b32 v146, v163, v161, s66
	v_perm_b32 v147, v167, v165, s66
	v_perm_b32 v148, v155, v153, s67
	v_perm_b32 v149, v159, v157, s67
	v_perm_b32 v150, v163, v161, s67
	v_perm_b32 v151, v167, v165, s67
	v_wmma_f32_16x16x16_f16 v[232:239], v[169:172], v[140:143], 0
	v_dual_add_f32 v0, v0, v224 :: v_dual_add_f32 v1, v1, v225
	v_dual_add_f32 v2, v2, v226 :: v_dual_add_f32 v3, v3, v227
	v_dual_add_f32 v4, v4, v228 :: v_dual_add_f32 v5, v5, v229
	v_dual_add_f32 v6, v6, v230 :: v_dual_add_f32 v7, v7, v231
	v_perm_b32 v173, v187, v185, s67
	v_perm_b32 v174, v191, v189, s67
	v_perm_b32 v175, v195, v193, s67
	v_perm_b32 v176, v199, v197, s67
	v_perm_b32 v177, v188, v186, s66
	v_perm_b32 v178, v192, v190, s66
	v_perm_b32 v179, v196, v194, s66
	v_perm_b32 v180, v200, v198, s66
	v_wmma_f32_16x16x16_f16 v[224:231], v[169:172], v[144:147], 0
	v_dual_add_f32 v8, v8, v232 :: v_dual_add_f32 v9, v9, v233
	v_dual_add_f32 v10, v10, v234 :: v_dual_add_f32 v11, v11, v235
	v_dual_add_f32 v12, v12, v236 :: v_dual_add_f32 v13, v13, v237
	v_dual_add_f32 v14, v14, v238 :: v_dual_add_f32 v15, v15, v239
	v_perm_b32 v181, v188, v186, s67
	v_perm_b32 v182, v192, v190, s67
	v_perm_b32 v183, v196, v194, s67
	v_perm_b32 v184, v200, v198, s67
	v_wmma_f32_16x16x16_f16 v[232:239], v[169:172], v[148:151], 0
	v_dual_add_f32 v16, v16, v224 :: v_dual_add_f32 v17, v17, v225
	v_dual_add_f32 v18, v18, v226 :: v_dual_add_f32 v19, v19, v227
	v_dual_add_f32 v20, v20, v228 :: v_dual_add_f32 v21, v21, v229
	v_dual_add_f32 v22, v22, v230 :: v_dual_add_f32 v23, v23, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[173:176], v[136:139], 0
	v_dual_add_f32 v24, v24, v232 :: v_dual_add_f32 v25, v25, v233
	v_dual_add_f32 v26, v26, v234 :: v_dual_add_f32 v27, v27, v235
	v_dual_add_f32 v28, v28, v236 :: v_dual_add_f32 v29, v29, v237
	v_dual_add_f32 v30, v30, v238 :: v_dual_add_f32 v31, v31, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[173:176], v[140:143], 0
	v_dual_add_f32 v32, v32, v224 :: v_dual_add_f32 v33, v33, v225
	v_dual_add_f32 v34, v34, v226 :: v_dual_add_f32 v35, v35, v227
	v_dual_add_f32 v36, v36, v228 :: v_dual_add_f32 v37, v37, v229
	v_dual_add_f32 v38, v38, v230 :: v_dual_add_f32 v39, v39, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[173:176], v[144:147], 0
	v_dual_add_f32 v40, v40, v232 :: v_dual_add_f32 v41, v41, v233
	v_dual_add_f32 v42, v42, v234 :: v_dual_add_f32 v43, v43, v235
	v_dual_add_f32 v44, v44, v236 :: v_dual_add_f32 v45, v45, v237
	v_dual_add_f32 v46, v46, v238 :: v_dual_add_f32 v47, v47, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[173:176], v[148:151], 0
	v_dual_add_f32 v48, v48, v224 :: v_dual_add_f32 v49, v49, v225
	v_dual_add_f32 v50, v50, v226 :: v_dual_add_f32 v51, v51, v227
	v_dual_add_f32 v52, v52, v228 :: v_dual_add_f32 v53, v53, v229
	v_dual_add_f32 v54, v54, v230 :: v_dual_add_f32 v55, v55, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[177:180], v[136:139], 0
	v_dual_add_f32 v56, v56, v232 :: v_dual_add_f32 v57, v57, v233
	v_dual_add_f32 v58, v58, v234 :: v_dual_add_f32 v59, v59, v235
	v_dual_add_f32 v60, v60, v236 :: v_dual_add_f32 v61, v61, v237
	v_dual_add_f32 v62, v62, v238 :: v_dual_add_f32 v63, v63, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[177:180], v[140:143], 0
	v_dual_add_f32 v64, v64, v224 :: v_dual_add_f32 v65, v65, v225
	v_dual_add_f32 v66, v66, v226 :: v_dual_add_f32 v67, v67, v227
	v_dual_add_f32 v68, v68, v228 :: v_dual_add_f32 v69, v69, v229
	v_dual_add_f32 v70, v70, v230 :: v_dual_add_f32 v71, v71, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[177:180], v[144:147], 0
	v_dual_add_f32 v72, v72, v232 :: v_dual_add_f32 v73, v73, v233
	v_dual_add_f32 v74, v74, v234 :: v_dual_add_f32 v75, v75, v235
	v_dual_add_f32 v76, v76, v236 :: v_dual_add_f32 v77, v77, v237
	v_dual_add_f32 v78, v78, v238 :: v_dual_add_f32 v79, v79, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[177:180], v[148:151], 0
	v_dual_add_f32 v80, v80, v224 :: v_dual_add_f32 v81, v81, v225
	v_dual_add_f32 v82, v82, v226 :: v_dual_add_f32 v83, v83, v227
	v_dual_add_f32 v84, v84, v228 :: v_dual_add_f32 v85, v85, v229
	v_dual_add_f32 v86, v86, v230 :: v_dual_add_f32 v87, v87, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[181:184], v[136:139], 0
	v_dual_add_f32 v88, v88, v232 :: v_dual_add_f32 v89, v89, v233
	v_dual_add_f32 v90, v90, v234 :: v_dual_add_f32 v91, v91, v235
	v_dual_add_f32 v92, v92, v236 :: v_dual_add_f32 v93, v93, v237
	v_dual_add_f32 v94, v94, v238 :: v_dual_add_f32 v95, v95, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[181:184], v[140:143], 0
	v_dual_add_f32 v96, v96, v224 :: v_dual_add_f32 v97, v97, v225
	v_dual_add_f32 v98, v98, v226 :: v_dual_add_f32 v99, v99, v227
	v_dual_add_f32 v100, v100, v228 :: v_dual_add_f32 v101, v101, v229
	v_dual_add_f32 v102, v102, v230 :: v_dual_add_f32 v103, v103, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[181:184], v[144:147], 0
	v_dual_add_f32 v104, v104, v232 :: v_dual_add_f32 v105, v105, v233
	v_dual_add_f32 v106, v106, v234 :: v_dual_add_f32 v107, v107, v235
	v_dual_add_f32 v108, v108, v236 :: v_dual_add_f32 v109, v109, v237
	v_dual_add_f32 v110, v110, v238 :: v_dual_add_f32 v111, v111, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[181:184], v[148:151], 0
	v_dual_add_f32 v112, v112, v224 :: v_dual_add_f32 v113, v113, v225
	v_dual_add_f32 v114, v114, v226 :: v_dual_add_f32 v115, v115, v227
	v_dual_add_f32 v116, v116, v228 :: v_dual_add_f32 v117, v117, v229
	v_dual_add_f32 v118, v118, v230 :: v_dual_add_f32 v119, v119, v231
	s_setprio 1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_setprio 2
	v_dual_add_f32 v120, v120, v232 :: v_dual_add_f32 v121, v121, v233
	v_dual_add_f32 v122, v122, v234 :: v_dual_add_f32 v123, v123, v235
	v_dual_add_f32 v124, v124, v236 :: v_dual_add_f32 v125, v125, v237
	v_dual_add_f32 v126, v126, v238 :: v_dual_add_f32 v127, v127, v239
label_PrefetchGlobalLastIterEnd:
label_toPGR1end_OrdNLL:
	s_and_b32 s12, 15, s27
	s_and_b32 s68, s46, 0x8000
	s_cbranch_scc1 label_GSUC_TL
	s_cmp_lg_u32 s6, s7
	s_cmov_b32 s12, 0
	s_branch label_GSUC_TL_End
label_GSUC_TL:
	s_lshr_b32 s69, s27, 4
	s_and_b32 s70, s46, 0x3fff
	v_cvt_f32_u32_e32 v152, s70
	v_rcp_iflag_f32_e32 v152, v152
	v_cvt_f32_u32_e32 v153, s69
	v_mul_f32_e32 v152, v152, v153
	v_cvt_u32_f32_e32 v152, v152
	v_mul_u32_u24_e64 v153, v152, s70
	v_sub_nc_u32_e32 v153, s69, v153
	v_cmp_eq_u32_e64 vcc_lo, v153, s70
	s_mov_b32 exec_lo, vcc_lo
	v_add_nc_u32_e32 v152, 1, v152
	v_mov_b32_e32 v153, 0
	s_mov_b32 exec_lo, -1
	v_cmp_gt_u32_e64 vcc_lo, v153, s70
	s_mov_b32 exec_lo, vcc_lo
	v_sub_nc_u32_e64 v152, v152, 1
	v_mul_u32_u24_e64 v153, v152, s70
	v_sub_nc_u32_e32 v153, s69, v153
	s_mov_b32 exec_lo, -1
	v_readfirstlane_b32 s68, v152
	v_readfirstlane_b32 s7, v153
	s_sub_co_u32 s69, s70, 1
	s_cmp_eq_u32 s68, 0
	s_cselect_b32 s68, s7, s69
	s_cmp_lg_u32 s6, s68
	s_cmov_b32 s12, 0
label_GSUC_TL_End:
	s_cmp_eq_u32 s12, 0
	s_mov_b32 s13, 0
	s_cbranch_scc1 label_SkipTailLoopL
	s_sub_co_i32 s68, 3, s47
	s_cmp_ge_i32 s68, 0
	s_cbranch_scc0 label_Negative_BVRL75R8H2W2AGXN
	s_mul_hi_u32 s69, s68, s64
	s_mul_i32 s68, s68, s64
	s_branch label_MultiplyDone_CY56LFEMU5DWQLV7
label_Negative_BVRL75R8H2W2AGXN:
	s_abs_i32 s68, s68
	s_mul_hi_u32 s69, s68, s64
	s_mul_i32 s68, s68, s64
	s_xor_b32 s68, s68, -1
	s_xor_b32 s69, s69, -1
	s_add_co_u32 s68, s68, 1
	s_add_co_ci_u32 s69, s69, 0
label_MultiplyDone_CY56LFEMU5DWQLV7:
	s_sub_co_u32 s68, s68, s60
	s_sub_co_ci_u32 s69, s69, s61
	s_add_co_u32 s48, s48, s68
	s_add_co_ci_u32 s49, s49, s69
	s_sub_co_u32 s56, s56, s68
	s_sub_co_ci_u32 s57, s57, s69
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s50, s56, -1
	s_sub_co_i32 s68, 3, s47
	s_cmp_ge_i32 s68, 0
	s_cbranch_scc0 label_Negative_6TIJ4FHB7JKNP707
	s_mul_hi_u32 s69, s68, s65
	s_mul_i32 s68, s68, s65
	s_branch label_MultiplyDone_BW33HOF1JZ3GAPD7
label_Negative_6TIJ4FHB7JKNP707:
	s_abs_i32 s68, s68
	s_mul_hi_u32 s69, s68, s65
	s_mul_i32 s68, s68, s65
	s_xor_b32 s68, s68, -1
	s_xor_b32 s69, s69, -1
	s_add_co_u32 s68, s68, 1
	s_add_co_ci_u32 s69, s69, 0
label_MultiplyDone_BW33HOF1JZ3GAPD7:
	s_sub_co_u32 s68, s68, s62
	s_sub_co_ci_u32 s69, s69, s63
	s_add_co_u32 s52, s52, s68
	s_add_co_ci_u32 s53, s53, s69
	s_sub_co_u32 s58, s58, s68
	s_sub_co_ci_u32 s59, s59, s69
	s_cmp_eq_u32 s59, 0
	s_cselect_b32 s54, s58, -1
	buffer_load_d16_b16 v136, v128, s[48:51], null offen
	buffer_load_d16_hi_b16 v136, v128, s[48:51], null offen offset:2
	buffer_load_d16_b16 v137, v128, s[48:51], null offen offset:4
	buffer_load_d16_hi_b16 v137, v128, s[48:51], null offen offset:6
	buffer_load_d16_b16 v138, v128, s[48:51], null offen offset:8
	buffer_load_d16_hi_b16 v138, v128, s[48:51], null offen offset:10
	buffer_load_d16_b16 v139, v128, s[48:51], null offen offset:12
	buffer_load_d16_hi_b16 v139, v128, s[48:51], null offen offset:14
	buffer_load_d16_b16 v140, v129, s[48:51], null offen
	buffer_load_d16_hi_b16 v140, v129, s[48:51], null offen offset:2
	buffer_load_d16_b16 v141, v129, s[48:51], null offen offset:4
	buffer_load_d16_hi_b16 v141, v129, s[48:51], null offen offset:6
	buffer_load_d16_b16 v142, v129, s[48:51], null offen offset:8
	buffer_load_d16_hi_b16 v142, v129, s[48:51], null offen offset:10
	buffer_load_d16_b16 v143, v129, s[48:51], null offen offset:12
	buffer_load_d16_hi_b16 v143, v129, s[48:51], null offen offset:14
	buffer_load_d16_b16 v144, v130, s[52:55], null offen
	buffer_load_d16_hi_b16 v144, v130, s[52:55], null offen offset:2
	buffer_load_d16_b16 v145, v130, s[52:55], null offen offset:4
	buffer_load_d16_hi_b16 v145, v130, s[52:55], null offen offset:6
	buffer_load_d16_b16 v146, v130, s[52:55], null offen offset:8
	buffer_load_d16_hi_b16 v146, v130, s[52:55], null offen offset:10
	buffer_load_d16_b16 v147, v130, s[52:55], null offen offset:12
	buffer_load_d16_hi_b16 v147, v130, s[52:55], null offen offset:14
	buffer_load_d16_b16 v148, v131, s[52:55], null offen
	buffer_load_d16_hi_b16 v148, v131, s[52:55], null offen offset:2
	buffer_load_d16_b16 v149, v131, s[52:55], null offen offset:4
	buffer_load_d16_hi_b16 v149, v131, s[52:55], null offen offset:6
	buffer_load_d16_b16 v150, v131, s[52:55], null offen offset:8
	buffer_load_d16_hi_b16 v150, v131, s[52:55], null offen offset:10
	buffer_load_d16_b16 v151, v131, s[52:55], null offen offset:12
	buffer_load_d16_hi_b16 v151, v131, s[52:55], null offen offset:14
	s_wait_loadcnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_store_b128 v132, v[136:139]
	ds_store_b128 v132, v[140:143] offset:2048
	ds_store_b128 v133, v[144:147]
	ds_store_b128 v133, v[148:151] offset:2048
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_nop 0
label_TailLoopBeginL:
	ds_load_b64 v[152:153], v134
	ds_load_b64 v[154:155], v134 offset:256
	ds_load_b64 v[156:157], v134 offset:512
	ds_load_b64 v[158:159], v134 offset:768
	ds_load_b64 v[160:161], v134 offset:1024
	ds_load_b64 v[162:163], v134 offset:1280
	ds_load_b64 v[164:165], v134 offset:1536
	ds_load_b64 v[166:167], v134 offset:1792
	ds_load_b64 v[184:185], v135
	ds_load_b64 v[186:187], v135 offset:256
	ds_load_b64 v[188:189], v135 offset:512
	ds_load_b64 v[190:191], v135 offset:768
	ds_load_b64 v[192:193], v135 offset:1024
	ds_load_b64 v[194:195], v135 offset:1280
	ds_load_b64 v[196:197], v135 offset:1536
	ds_load_b64 v[198:199], v135 offset:1792
	s_mov_b32 s8, 0x1000
	v_add_co_u32 v134, vcc_lo, s8, v134
	v_add_co_u32 v135, vcc_lo, s8, v135
	s_wait_dscnt 0x0
	v_perm_b32 v136, v154, v152, s66
	v_perm_b32 v137, v158, v156, s66
	v_perm_b32 v138, v162, v160, s66
	v_perm_b32 v139, v166, v164, s66
	v_perm_b32 v140, v154, v152, s67
	v_perm_b32 v141, v158, v156, s67
	v_perm_b32 v142, v162, v160, s67
	v_perm_b32 v143, v166, v164, s67
	v_perm_b32 v144, v155, v153, s66
	v_perm_b32 v145, v159, v157, s66
	v_perm_b32 v146, v163, v161, s66
	v_perm_b32 v147, v167, v165, s66
	v_perm_b32 v148, v155, v153, s67
	v_perm_b32 v149, v159, v157, s67
	v_perm_b32 v150, v163, v161, s67
	v_perm_b32 v151, v167, v165, s67
	v_perm_b32 v168, v186, v184, s66
	v_perm_b32 v169, v190, v188, s66
	v_perm_b32 v170, v194, v192, s66
	v_perm_b32 v171, v198, v196, s66
	v_perm_b32 v172, v186, v184, s67
	v_perm_b32 v173, v190, v188, s67
	v_perm_b32 v174, v194, v192, s67
	v_perm_b32 v175, v198, v196, s67
	v_perm_b32 v176, v187, v185, s66
	v_perm_b32 v177, v191, v189, s66
	v_perm_b32 v178, v195, v193, s66
	v_perm_b32 v179, v199, v197, s66
	v_perm_b32 v180, v187, v185, s67
	v_perm_b32 v181, v191, v189, s67
	v_perm_b32 v182, v195, v193, s67
	v_perm_b32 v183, v199, v197, s67
	v_and_b32_e32 v200, 31, v218
	v_lshrrev_b32_e32 v200, 4, v200
	v_lshlrev_b32_e32 v200, 3, v200
	v_add_nc_u32_e64 v201, v200, 0
	v_cmp_ge_i32_e64 s68, v201, s12
	v_cndmask_b32_e64 v136, v136, 0, s68
	v_cndmask_b32_e64 v140, v140, 0, s68
	v_cndmask_b32_e64 v144, v144, 0, s68
	v_cndmask_b32_e64 v148, v148, 0, s68
	v_cndmask_b32_e64 v137, v137, 0, s68
	v_cndmask_b32_e64 v141, v141, 0, s68
	v_cndmask_b32_e64 v145, v145, 0, s68
	v_cndmask_b32_e64 v149, v149, 0, s68
	v_cndmask_b32_e64 v138, v138, 0, s68
	v_cndmask_b32_e64 v142, v142, 0, s68
	v_cndmask_b32_e64 v146, v146, 0, s68
	v_cndmask_b32_e64 v150, v150, 0, s68
	v_cndmask_b32_e64 v139, v139, 0, s68
	v_cndmask_b32_e64 v143, v143, 0, s68
	v_cndmask_b32_e64 v147, v147, 0, s68
	v_cndmask_b32_e64 v151, v151, 0, s68
	v_and_b32_e32 v200, 31, v218
	v_lshrrev_b32_e32 v200, 4, v200
	v_lshlrev_b32_e32 v200, 3, v200
	v_add_nc_u32_e64 v201, v200, 0
	v_cmp_ge_i32_e64 s68, v201, s12
	v_cndmask_b32_e64 v168, v168, 0, s68
	v_cndmask_b32_e64 v172, v172, 0, s68
	v_cndmask_b32_e64 v176, v176, 0, s68
	v_cndmask_b32_e64 v180, v180, 0, s68
	v_cndmask_b32_e64 v169, v169, 0, s68
	v_cndmask_b32_e64 v173, v173, 0, s68
	v_cndmask_b32_e64 v177, v177, 0, s68
	v_cndmask_b32_e64 v181, v181, 0, s68
	v_cndmask_b32_e64 v170, v170, 0, s68
	v_cndmask_b32_e64 v174, v174, 0, s68
	v_cndmask_b32_e64 v178, v178, 0, s68
	v_cndmask_b32_e64 v182, v182, 0, s68
	v_cndmask_b32_e64 v171, v171, 0, s68
	v_cndmask_b32_e64 v175, v175, 0, s68
	v_cndmask_b32_e64 v179, v179, 0, s68
	v_cndmask_b32_e64 v183, v183, 0, s68
	v_sub_nc_u32_e32 v201, s12, v200
	v_cmp_lt_i32_e64 s68, v201, 8
	s_and_b32 s70, s27, 7
	s_cmp_eq_u32 s70, 0
	s_cbranch_scc1 label_TailLoop_SkipZeroOutMask_O2GI74JLDTYT65QO
	s_and_b32 s70, s12, 7
	s_sub_co_u32 s70, 8, s70
	s_lshl_b32 s70, s70, 4
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_a_64_Shift
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_a_32_Shift
	v_lshlrev_b64_e32 v[202:203], s71, v[136:137]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[137:138]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[138:139]
	v_mov_b32_e32 v207, v203
	s_branch label_a_shift_end
label_a_32_Shift:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[136:137]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[137:138]
	v_mov_b32_e32 v207, v203
	s_add_co_u32 s71, s71, 32
	s_branch label_a_shift_end
label_a_64_Shift:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[136:137]
	s_add_co_u32 s71, s71, 64
label_a_shift_end:
	v_cndmask_b32_e64 v136, v136, v204, s68
	v_cndmask_b32_e64 v137, v137, v205, s68
	v_cndmask_b32_e64 v138, v138, v206, s68
	v_cndmask_b32_e64 v139, v139, v207, s68
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_a_64_Shift_1
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_a_32_Shift_1
	v_lshlrev_b64_e32 v[202:203], s71, v[140:141]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[141:142]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[142:143]
	v_mov_b32_e32 v207, v203
	s_branch label_a_shift_end_1
label_a_32_Shift_1:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[140:141]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[141:142]
	v_mov_b32_e32 v207, v203
	s_add_co_u32 s71, s71, 32
	s_branch label_a_shift_end_1
label_a_64_Shift_1:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[140:141]
	s_add_co_u32 s71, s71, 64
label_a_shift_end_1:
	v_cndmask_b32_e64 v140, v140, v204, s68
	v_cndmask_b32_e64 v141, v141, v205, s68
	v_cndmask_b32_e64 v142, v142, v206, s68
	v_cndmask_b32_e64 v143, v143, v207, s68
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_a_64_Shift_2
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_a_32_Shift_2
	v_lshlrev_b64_e32 v[202:203], s71, v[144:145]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[145:146]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[146:147]
	v_mov_b32_e32 v207, v203
	s_branch label_a_shift_end_2
label_a_32_Shift_2:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[144:145]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[145:146]
	v_mov_b32_e32 v207, v203
	s_add_co_u32 s71, s71, 32
	s_branch label_a_shift_end_2
label_a_64_Shift_2:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[144:145]
	s_add_co_u32 s71, s71, 64
label_a_shift_end_2:
	v_cndmask_b32_e64 v144, v144, v204, s68
	v_cndmask_b32_e64 v145, v145, v205, s68
	v_cndmask_b32_e64 v146, v146, v206, s68
	v_cndmask_b32_e64 v147, v147, v207, s68
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_a_64_Shift_3
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_a_32_Shift_3
	v_lshlrev_b64_e32 v[202:203], s71, v[148:149]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[149:150]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[150:151]
	v_mov_b32_e32 v207, v203
	s_branch label_a_shift_end_3
label_a_32_Shift_3:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[148:149]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[149:150]
	v_mov_b32_e32 v207, v203
	s_add_co_u32 s71, s71, 32
	s_branch label_a_shift_end_3
label_a_64_Shift_3:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[148:149]
	s_add_co_u32 s71, s71, 64
label_a_shift_end_3:
	v_cndmask_b32_e64 v148, v148, v204, s68
	v_cndmask_b32_e64 v149, v149, v205, s68
	v_cndmask_b32_e64 v150, v150, v206, s68
	v_cndmask_b32_e64 v151, v151, v207, s68
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_b_64_Shift
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_b_32_Shift
	v_lshlrev_b64_e32 v[202:203], s71, v[168:169]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[169:170]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[170:171]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end
label_b_32_Shift:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[168:169]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[169:170]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end
label_b_64_Shift:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[168:169]
label_b_shift_end:
	v_cndmask_b32_e64 v168, v168, v204, s68
	v_cndmask_b32_e64 v169, v169, v205, s68
	v_cndmask_b32_e64 v170, v170, v206, s68
	v_cndmask_b32_e64 v171, v171, v207, s68
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_b_64_Shift_1
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_b_32_Shift_1
	v_lshlrev_b64_e32 v[202:203], s71, v[172:173]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[173:174]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[174:175]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end_1
label_b_32_Shift_1:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[172:173]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[173:174]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end_1
label_b_64_Shift_1:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[172:173]
label_b_shift_end_1:
	v_cndmask_b32_e64 v172, v172, v204, s68
	v_cndmask_b32_e64 v173, v173, v205, s68
	v_cndmask_b32_e64 v174, v174, v206, s68
	v_cndmask_b32_e64 v175, v175, v207, s68
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_b_64_Shift_2
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_b_32_Shift_2
	v_lshlrev_b64_e32 v[202:203], s71, v[176:177]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[177:178]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[178:179]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end_2
label_b_32_Shift_2:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[176:177]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[177:178]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end_2
label_b_64_Shift_2:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[176:177]
label_b_shift_end_2:
	v_cndmask_b32_e64 v176, v176, v204, s68
	v_cndmask_b32_e64 v177, v177, v205, s68
	v_cndmask_b32_e64 v178, v178, v206, s68
	v_cndmask_b32_e64 v179, v179, v207, s68
	s_mov_b32 s71, s70
	s_cmp_ge_i32 s71, 64
	s_cbranch_scc1 label_b_64_Shift_3
	s_cmp_ge_i32 s71, 32
	s_cbranch_scc1 label_b_32_Shift_3
	v_lshlrev_b64_e32 v[202:203], s71, v[180:181]
	v_mov_b32_e32 v204, v202
	v_mov_b32_e32 v205, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[181:182]
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[182:183]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end_3
label_b_32_Shift_3:
	s_sub_co_u32 s71, s71, 32
	v_mov_b32_e32 v204, 0
	v_lshlrev_b64_e32 v[202:203], s71, v[180:181]
	v_mov_b32_e32 v205, v202
	v_mov_b32_e32 v206, v203
	v_lshlrev_b64_e32 v[202:203], s71, v[181:182]
	v_mov_b32_e32 v207, v203
	s_branch label_b_shift_end_3
label_b_64_Shift_3:
	s_sub_co_u32 s71, s71, 64
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_lshlrev_b64_e32 v[206:207], s71, v[180:181]
label_b_shift_end_3:
	v_cndmask_b32_e64 v180, v180, v204, s68
	v_cndmask_b32_e64 v181, v181, v205, s68
	v_cndmask_b32_e64 v182, v182, v206, s68
	v_cndmask_b32_e64 v183, v183, v207, s68
label_TailLoop_SkipZeroOutMask_O2GI74JLDTYT65QO:
	s_nop 1
	v_wmma_f32_16x16x16_f16 v[224:231], v[168:171], v[136:139], 0
	v_wmma_f32_16x16x16_f16 v[232:239], v[168:171], v[140:143], 0
	v_dual_add_f32 v0, v0, v224 :: v_dual_add_f32 v1, v1, v225
	v_dual_add_f32 v2, v2, v226 :: v_dual_add_f32 v3, v3, v227
	v_dual_add_f32 v4, v4, v228 :: v_dual_add_f32 v5, v5, v229
	v_dual_add_f32 v6, v6, v230 :: v_dual_add_f32 v7, v7, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[168:171], v[144:147], 0
	v_dual_add_f32 v8, v8, v232 :: v_dual_add_f32 v9, v9, v233
	v_dual_add_f32 v10, v10, v234 :: v_dual_add_f32 v11, v11, v235
	v_dual_add_f32 v12, v12, v236 :: v_dual_add_f32 v13, v13, v237
	v_dual_add_f32 v14, v14, v238 :: v_dual_add_f32 v15, v15, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[168:171], v[148:151], 0
	v_dual_add_f32 v16, v16, v224 :: v_dual_add_f32 v17, v17, v225
	v_dual_add_f32 v18, v18, v226 :: v_dual_add_f32 v19, v19, v227
	v_dual_add_f32 v20, v20, v228 :: v_dual_add_f32 v21, v21, v229
	v_dual_add_f32 v22, v22, v230 :: v_dual_add_f32 v23, v23, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[172:175], v[136:139], 0
	v_dual_add_f32 v24, v24, v232 :: v_dual_add_f32 v25, v25, v233
	v_dual_add_f32 v26, v26, v234 :: v_dual_add_f32 v27, v27, v235
	v_dual_add_f32 v28, v28, v236 :: v_dual_add_f32 v29, v29, v237
	v_dual_add_f32 v30, v30, v238 :: v_dual_add_f32 v31, v31, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[172:175], v[140:143], 0
	v_dual_add_f32 v32, v32, v224 :: v_dual_add_f32 v33, v33, v225
	v_dual_add_f32 v34, v34, v226 :: v_dual_add_f32 v35, v35, v227
	v_dual_add_f32 v36, v36, v228 :: v_dual_add_f32 v37, v37, v229
	v_dual_add_f32 v38, v38, v230 :: v_dual_add_f32 v39, v39, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[172:175], v[144:147], 0
	v_dual_add_f32 v40, v40, v232 :: v_dual_add_f32 v41, v41, v233
	v_dual_add_f32 v42, v42, v234 :: v_dual_add_f32 v43, v43, v235
	v_dual_add_f32 v44, v44, v236 :: v_dual_add_f32 v45, v45, v237
	v_dual_add_f32 v46, v46, v238 :: v_dual_add_f32 v47, v47, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[172:175], v[148:151], 0
	v_dual_add_f32 v48, v48, v224 :: v_dual_add_f32 v49, v49, v225
	v_dual_add_f32 v50, v50, v226 :: v_dual_add_f32 v51, v51, v227
	v_dual_add_f32 v52, v52, v228 :: v_dual_add_f32 v53, v53, v229
	v_dual_add_f32 v54, v54, v230 :: v_dual_add_f32 v55, v55, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[176:179], v[136:139], 0
	v_dual_add_f32 v56, v56, v232 :: v_dual_add_f32 v57, v57, v233
	v_dual_add_f32 v58, v58, v234 :: v_dual_add_f32 v59, v59, v235
	v_dual_add_f32 v60, v60, v236 :: v_dual_add_f32 v61, v61, v237
	v_dual_add_f32 v62, v62, v238 :: v_dual_add_f32 v63, v63, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[176:179], v[140:143], 0
	v_dual_add_f32 v64, v64, v224 :: v_dual_add_f32 v65, v65, v225
	v_dual_add_f32 v66, v66, v226 :: v_dual_add_f32 v67, v67, v227
	v_dual_add_f32 v68, v68, v228 :: v_dual_add_f32 v69, v69, v229
	v_dual_add_f32 v70, v70, v230 :: v_dual_add_f32 v71, v71, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[176:179], v[144:147], 0
	v_dual_add_f32 v72, v72, v232 :: v_dual_add_f32 v73, v73, v233
	v_dual_add_f32 v74, v74, v234 :: v_dual_add_f32 v75, v75, v235
	v_dual_add_f32 v76, v76, v236 :: v_dual_add_f32 v77, v77, v237
	v_dual_add_f32 v78, v78, v238 :: v_dual_add_f32 v79, v79, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[176:179], v[148:151], 0
	v_dual_add_f32 v80, v80, v224 :: v_dual_add_f32 v81, v81, v225
	v_dual_add_f32 v82, v82, v226 :: v_dual_add_f32 v83, v83, v227
	v_dual_add_f32 v84, v84, v228 :: v_dual_add_f32 v85, v85, v229
	v_dual_add_f32 v86, v86, v230 :: v_dual_add_f32 v87, v87, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[180:183], v[136:139], 0
	v_dual_add_f32 v88, v88, v232 :: v_dual_add_f32 v89, v89, v233
	v_dual_add_f32 v90, v90, v234 :: v_dual_add_f32 v91, v91, v235
	v_dual_add_f32 v92, v92, v236 :: v_dual_add_f32 v93, v93, v237
	v_dual_add_f32 v94, v94, v238 :: v_dual_add_f32 v95, v95, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[180:183], v[140:143], 0
	v_dual_add_f32 v96, v96, v224 :: v_dual_add_f32 v97, v97, v225
	v_dual_add_f32 v98, v98, v226 :: v_dual_add_f32 v99, v99, v227
	v_dual_add_f32 v100, v100, v228 :: v_dual_add_f32 v101, v101, v229
	v_dual_add_f32 v102, v102, v230 :: v_dual_add_f32 v103, v103, v231
	v_wmma_f32_16x16x16_f16 v[224:231], v[180:183], v[144:147], 0
	v_dual_add_f32 v104, v104, v232 :: v_dual_add_f32 v105, v105, v233
	v_dual_add_f32 v106, v106, v234 :: v_dual_add_f32 v107, v107, v235
	v_dual_add_f32 v108, v108, v236 :: v_dual_add_f32 v109, v109, v237
	v_dual_add_f32 v110, v110, v238 :: v_dual_add_f32 v111, v111, v239
	v_wmma_f32_16x16x16_f16 v[232:239], v[180:183], v[148:151], 0
	v_dual_add_f32 v112, v112, v224 :: v_dual_add_f32 v113, v113, v225
	v_dual_add_f32 v114, v114, v226 :: v_dual_add_f32 v115, v115, v227
	v_dual_add_f32 v116, v116, v228 :: v_dual_add_f32 v117, v117, v229
	v_dual_add_f32 v118, v118, v230 :: v_dual_add_f32 v119, v119, v231
	s_sub_co_i32 s12, s12, 16
	s_add_co_u32 s13, s13, 16
	s_cmp_le_i32 s12, 0
	v_dual_add_f32 v120, v120, v232 :: v_dual_add_f32 v121, v121, v233
	v_dual_add_f32 v122, v122, v234 :: v_dual_add_f32 v123, v123, v235
	v_dual_add_f32 v124, v124, v236 :: v_dual_add_f32 v125, v125, v237
	v_dual_add_f32 v126, v126, v238 :: v_dual_add_f32 v127, v127, v239
	s_cbranch_scc0 label_TailLoopBeginL
label_SkipTailLoopL:
label_TailLoopEndL:
label_Summation_End_70HEE253RZP8U9L8:
	s_and_b32 s8, s46, 0x3fff
	s_cmp_eq_u32 s8, 1
	s_cbranch_scc0 label_GSU_4
	s_cmp_eq_u32 s5, 2
	s_cbranch_scc1 label_LoadExternalEpilogueStruct
	s_load_b256 s[48:55], s[0:1], 0x58
	s_load_b32 s56, s[0:1], 0x78
	s_branch label_GSU_4
label_LoadExternalEpilogueStruct:
	s_load_b128 s[48:51], s[0:1], 0x90
	s_load_b64 s[52:53], s[0:1], 0xa0
	s_load_b64 s[54:55], s[0:1], 0xb8
	s_load_b32 s56, s[0:1], 0xc0
label_GSU_4:
label_LoadExternalEpilogueStructEnd:
	v_mov_b32_e32 v131, s2
	v_mul_i32_i24_e32 v131, 0xffffff80, v131
	v_add_co_u32 v131, vcc_lo, s24, v131
	v_mov_b32_e32 v132, 0x80
	v_cmp_lt_u32_e64 s8, v131, v132
	v_cndmask_b32_e64 v131, v132, v131, s8
	v_lshrrev_b32_e32 v133, 5, v218
	v_and_b32_e32 v133, 1, v133
	v_lshrrev_b32_e32 v134, 6, v131
	v_and_b32_e32 v134, 1, v134
	v_cmp_eq_u32_e64 s8, v134, v133
	v_cndmask_b32_e64 v131, v132, v131, s8
	v_lshrrev_b32_e32 v132, 6, v131
	v_lshlrev_b32_e32 v134, 0, v133
	v_sub_nc_u32_e32 v132, v132, v134
	v_lshrrev_b32_e32 v134, 3, v131
	v_lshrrev_b32_e32 v135, 0, v218
	v_and_b32_e32 v135, 15, v135
	v_lshlrev_b32_e32 v135, 2, v135
	v_lshrrev_b32_e32 v135, 3, v135
	v_lshlrev_b32_e32 v133, 3, v133
	v_add_co_u32 v135, vcc_lo, v133, v135
	v_sub_nc_u32_e32 v134, v134, v135
	v_and_b32_e32 v133, 3, v131
	v_lshrrev_b32_e32 v133, 3, v133
	v_and_b32_e32 v135, 7, v131
	v_cmp_eq_u32_e64 vcc_lo, v135, 1
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1
	v_cmp_eq_u32_e64 vcc_lo, v135, 2
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2
	v_cmp_eq_u32_e64 vcc_lo, v135, 3
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3
	v_cmp_eq_u32_e64 vcc_lo, v135, 4
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4
	v_cmp_eq_u32_e64 vcc_lo, v135, 5
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5
	v_cmp_eq_u32_e64 vcc_lo, v135, 6
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6
	v_cmp_eq_u32_e64 vcc_lo, v135, 7
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW1:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0
label_ShiftVectorComponents0_GLVW2:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0
label_ShiftVectorComponents0_GLVW3:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0
label_ShiftVectorComponents0_GLVW4:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0
label_ShiftVectorComponents0_GLVW5:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0
label_ShiftVectorComponents0_GLVW6:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0
label_ShiftVectorComponents0_GLVW7:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0
label_ShiftVectorComponents0_GLVW1_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW1_BM0_VW0
label_ShiftVectorComponents0_GLVW2_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW2_BM0_VW0
label_ShiftVectorComponents0_GLVW3_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW3_BM0_VW0
label_ShiftVectorComponents0_GLVW4_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW4_BM0_VW0
label_ShiftVectorComponents0_GLVW5_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW5_BM0_VW0
label_ShiftVectorComponents0_GLVW6_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW6_BM0_VW0
label_ShiftVectorComponents0_GLVW7_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents0_GLVW7_BM0_VW0
label_ShiftVectorComponents0_GLVW1_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v24
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v135, v56
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v32, v135
	v_mov_b32_e32 v135, v88
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v64, v135
	v_mov_b32_e32 v135, v120
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v96, v135
	v_mov_b32_e32 v135, v25
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v1, v135
	v_mov_b32_e32 v135, v57
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v33, v135
	v_mov_b32_e32 v135, v89
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v65, v135
	v_mov_b32_e32 v135, v121
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v97, v135
	v_mov_b32_e32 v135, v26
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v135, v58
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v34, v135
	v_mov_b32_e32 v135, v90
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v66, v135
	v_mov_b32_e32 v135, v122
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v98, v135
	v_mov_b32_e32 v135, v27
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v3, v135
	v_mov_b32_e32 v135, v59
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v35, v135
	v_mov_b32_e32 v135, v91
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v67, v135
	v_mov_b32_e32 v135, v123
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v99, v135
	v_mov_b32_e32 v135, v28
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v135, v60
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v36, v135
	v_mov_b32_e32 v135, v92
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v68, v135
	v_mov_b32_e32 v135, v124
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v100, v135
	v_mov_b32_e32 v135, v29
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v5, v135
	v_mov_b32_e32 v135, v61
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v37, v135
	v_mov_b32_e32 v135, v93
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v69, v135
	v_mov_b32_e32 v135, v125
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v101, v135
	v_mov_b32_e32 v135, v30
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v135, v62
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v38, v135
	v_mov_b32_e32 v135, v94
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v70, v135
	v_mov_b32_e32 v135, v126
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v102, v135
	v_mov_b32_e32 v135, v31
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v7, v135
	v_mov_b32_e32 v135, v63
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v135, v95
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v71, v135
	v_mov_b32_e32 v135, v127
	ds_bpermute_b32 v135, v128, v135 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v103, v135
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW2_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v16
	v_mov_b32_e32 v136, v24
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v8, v136
	v_mov_b32_e32 v135, v48
	v_mov_b32_e32 v136, v56
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v32, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v135, v80
	v_mov_b32_e32 v136, v88
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v64, v135
	v_mov_b32_e32 v72, v136
	v_mov_b32_e32 v135, v112
	v_mov_b32_e32 v136, v120
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v96, v135
	v_mov_b32_e32 v104, v136
	v_mov_b32_e32 v135, v17
	v_mov_b32_e32 v136, v25
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v1, v135
	v_mov_b32_e32 v9, v136
	v_mov_b32_e32 v135, v49
	v_mov_b32_e32 v136, v57
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v33, v135
	v_mov_b32_e32 v41, v136
	v_mov_b32_e32 v135, v81
	v_mov_b32_e32 v136, v89
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v65, v135
	v_mov_b32_e32 v73, v136
	v_mov_b32_e32 v135, v113
	v_mov_b32_e32 v136, v121
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v97, v135
	v_mov_b32_e32 v105, v136
	v_mov_b32_e32 v135, v18
	v_mov_b32_e32 v136, v26
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v10, v136
	v_mov_b32_e32 v135, v50
	v_mov_b32_e32 v136, v58
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v34, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v135, v82
	v_mov_b32_e32 v136, v90
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v66, v135
	v_mov_b32_e32 v74, v136
	v_mov_b32_e32 v135, v114
	v_mov_b32_e32 v136, v122
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v98, v135
	v_mov_b32_e32 v106, v136
	v_mov_b32_e32 v135, v19
	v_mov_b32_e32 v136, v27
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v3, v135
	v_mov_b32_e32 v11, v136
	v_mov_b32_e32 v135, v51
	v_mov_b32_e32 v136, v59
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v35, v135
	v_mov_b32_e32 v43, v136
	v_mov_b32_e32 v135, v83
	v_mov_b32_e32 v136, v91
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v67, v135
	v_mov_b32_e32 v75, v136
	v_mov_b32_e32 v135, v115
	v_mov_b32_e32 v136, v123
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v99, v135
	v_mov_b32_e32 v107, v136
	v_mov_b32_e32 v135, v20
	v_mov_b32_e32 v136, v28
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v12, v136
	v_mov_b32_e32 v135, v52
	v_mov_b32_e32 v136, v60
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v36, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v135, v84
	v_mov_b32_e32 v136, v92
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v68, v135
	v_mov_b32_e32 v76, v136
	v_mov_b32_e32 v135, v116
	v_mov_b32_e32 v136, v124
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v100, v135
	v_mov_b32_e32 v108, v136
	v_mov_b32_e32 v135, v21
	v_mov_b32_e32 v136, v29
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v5, v135
	v_mov_b32_e32 v13, v136
	v_mov_b32_e32 v135, v53
	v_mov_b32_e32 v136, v61
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v37, v135
	v_mov_b32_e32 v45, v136
	v_mov_b32_e32 v135, v85
	v_mov_b32_e32 v136, v93
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v69, v135
	v_mov_b32_e32 v77, v136
	v_mov_b32_e32 v135, v117
	v_mov_b32_e32 v136, v125
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v101, v135
	v_mov_b32_e32 v109, v136
	v_mov_b32_e32 v135, v22
	v_mov_b32_e32 v136, v30
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v14, v136
	v_mov_b32_e32 v135, v54
	v_mov_b32_e32 v136, v62
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v38, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v135, v86
	v_mov_b32_e32 v136, v94
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v70, v135
	v_mov_b32_e32 v78, v136
	v_mov_b32_e32 v135, v118
	v_mov_b32_e32 v136, v126
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v102, v135
	v_mov_b32_e32 v110, v136
	v_mov_b32_e32 v135, v23
	v_mov_b32_e32 v136, v31
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v7, v135
	v_mov_b32_e32 v15, v136
	v_mov_b32_e32 v135, v55
	v_mov_b32_e32 v136, v63
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v47, v136
	v_mov_b32_e32 v135, v87
	v_mov_b32_e32 v136, v95
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v71, v135
	v_mov_b32_e32 v79, v136
	v_mov_b32_e32 v135, v119
	v_mov_b32_e32 v136, v127
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v103, v135
	v_mov_b32_e32 v111, v136
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW3_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v8
	v_mov_b32_e32 v136, v16
	v_mov_b32_e32 v137, v24
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v8, v136
	v_mov_b32_e32 v16, v137
	v_mov_b32_e32 v135, v40
	v_mov_b32_e32 v136, v48
	v_mov_b32_e32 v137, v56
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v32, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v48, v137
	v_mov_b32_e32 v135, v72
	v_mov_b32_e32 v136, v80
	v_mov_b32_e32 v137, v88
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v64, v135
	v_mov_b32_e32 v72, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v135, v104
	v_mov_b32_e32 v136, v112
	v_mov_b32_e32 v137, v120
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v96, v135
	v_mov_b32_e32 v104, v136
	v_mov_b32_e32 v112, v137
	v_mov_b32_e32 v135, v9
	v_mov_b32_e32 v136, v17
	v_mov_b32_e32 v137, v25
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v1, v135
	v_mov_b32_e32 v9, v136
	v_mov_b32_e32 v17, v137
	v_mov_b32_e32 v135, v41
	v_mov_b32_e32 v136, v49
	v_mov_b32_e32 v137, v57
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v33, v135
	v_mov_b32_e32 v41, v136
	v_mov_b32_e32 v49, v137
	v_mov_b32_e32 v135, v73
	v_mov_b32_e32 v136, v81
	v_mov_b32_e32 v137, v89
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v65, v135
	v_mov_b32_e32 v73, v136
	v_mov_b32_e32 v81, v137
	v_mov_b32_e32 v135, v105
	v_mov_b32_e32 v136, v113
	v_mov_b32_e32 v137, v121
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v97, v135
	v_mov_b32_e32 v105, v136
	v_mov_b32_e32 v113, v137
	v_mov_b32_e32 v135, v10
	v_mov_b32_e32 v136, v18
	v_mov_b32_e32 v137, v26
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v10, v136
	v_mov_b32_e32 v18, v137
	v_mov_b32_e32 v135, v42
	v_mov_b32_e32 v136, v50
	v_mov_b32_e32 v137, v58
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v34, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v50, v137
	v_mov_b32_e32 v135, v74
	v_mov_b32_e32 v136, v82
	v_mov_b32_e32 v137, v90
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v66, v135
	v_mov_b32_e32 v74, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v135, v106
	v_mov_b32_e32 v136, v114
	v_mov_b32_e32 v137, v122
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v98, v135
	v_mov_b32_e32 v106, v136
	v_mov_b32_e32 v114, v137
	v_mov_b32_e32 v135, v11
	v_mov_b32_e32 v136, v19
	v_mov_b32_e32 v137, v27
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v3, v135
	v_mov_b32_e32 v11, v136
	v_mov_b32_e32 v19, v137
	v_mov_b32_e32 v135, v43
	v_mov_b32_e32 v136, v51
	v_mov_b32_e32 v137, v59
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v35, v135
	v_mov_b32_e32 v43, v136
	v_mov_b32_e32 v51, v137
	v_mov_b32_e32 v135, v75
	v_mov_b32_e32 v136, v83
	v_mov_b32_e32 v137, v91
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v67, v135
	v_mov_b32_e32 v75, v136
	v_mov_b32_e32 v83, v137
	v_mov_b32_e32 v135, v107
	v_mov_b32_e32 v136, v115
	v_mov_b32_e32 v137, v123
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v99, v135
	v_mov_b32_e32 v107, v136
	v_mov_b32_e32 v115, v137
	v_mov_b32_e32 v135, v12
	v_mov_b32_e32 v136, v20
	v_mov_b32_e32 v137, v28
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v12, v136
	v_mov_b32_e32 v20, v137
	v_mov_b32_e32 v135, v44
	v_mov_b32_e32 v136, v52
	v_mov_b32_e32 v137, v60
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v36, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v52, v137
	v_mov_b32_e32 v135, v76
	v_mov_b32_e32 v136, v84
	v_mov_b32_e32 v137, v92
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v68, v135
	v_mov_b32_e32 v76, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v135, v108
	v_mov_b32_e32 v136, v116
	v_mov_b32_e32 v137, v124
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v100, v135
	v_mov_b32_e32 v108, v136
	v_mov_b32_e32 v116, v137
	v_mov_b32_e32 v135, v13
	v_mov_b32_e32 v136, v21
	v_mov_b32_e32 v137, v29
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v5, v135
	v_mov_b32_e32 v13, v136
	v_mov_b32_e32 v21, v137
	v_mov_b32_e32 v135, v45
	v_mov_b32_e32 v136, v53
	v_mov_b32_e32 v137, v61
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v37, v135
	v_mov_b32_e32 v45, v136
	v_mov_b32_e32 v53, v137
	v_mov_b32_e32 v135, v77
	v_mov_b32_e32 v136, v85
	v_mov_b32_e32 v137, v93
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v69, v135
	v_mov_b32_e32 v77, v136
	v_mov_b32_e32 v85, v137
	v_mov_b32_e32 v135, v109
	v_mov_b32_e32 v136, v117
	v_mov_b32_e32 v137, v125
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v101, v135
	v_mov_b32_e32 v109, v136
	v_mov_b32_e32 v117, v137
	v_mov_b32_e32 v135, v14
	v_mov_b32_e32 v136, v22
	v_mov_b32_e32 v137, v30
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v14, v136
	v_mov_b32_e32 v22, v137
	v_mov_b32_e32 v135, v46
	v_mov_b32_e32 v136, v54
	v_mov_b32_e32 v137, v62
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v38, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v54, v137
	v_mov_b32_e32 v135, v78
	v_mov_b32_e32 v136, v86
	v_mov_b32_e32 v137, v94
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v70, v135
	v_mov_b32_e32 v78, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v135, v110
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v126
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v102, v135
	v_mov_b32_e32 v110, v136
	v_mov_b32_e32 v118, v137
	v_mov_b32_e32 v135, v15
	v_mov_b32_e32 v136, v23
	v_mov_b32_e32 v137, v31
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v7, v135
	v_mov_b32_e32 v15, v136
	v_mov_b32_e32 v23, v137
	v_mov_b32_e32 v135, v47
	v_mov_b32_e32 v136, v55
	v_mov_b32_e32 v137, v63
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v47, v136
	v_mov_b32_e32 v55, v137
	v_mov_b32_e32 v135, v79
	v_mov_b32_e32 v136, v87
	v_mov_b32_e32 v137, v95
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v71, v135
	v_mov_b32_e32 v79, v136
	v_mov_b32_e32 v87, v137
	v_mov_b32_e32 v135, v111
	v_mov_b32_e32 v136, v119
	v_mov_b32_e32 v137, v127
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v103, v135
	v_mov_b32_e32 v111, v136
	v_mov_b32_e32 v119, v137
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW4_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v0
	v_mov_b32_e32 v136, v8
	v_mov_b32_e32 v137, v16
	v_mov_b32_e32 v138, v24
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v8, v136
	v_mov_b32_e32 v16, v137
	v_mov_b32_e32 v24, v138
	v_mov_b32_e32 v135, v32
	v_mov_b32_e32 v136, v40
	v_mov_b32_e32 v137, v48
	v_mov_b32_e32 v138, v56
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v32, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v48, v137
	v_mov_b32_e32 v56, v138
	v_mov_b32_e32 v135, v64
	v_mov_b32_e32 v136, v72
	v_mov_b32_e32 v137, v80
	v_mov_b32_e32 v138, v88
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v64, v135
	v_mov_b32_e32 v72, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v88, v138
	v_mov_b32_e32 v135, v96
	v_mov_b32_e32 v136, v104
	v_mov_b32_e32 v137, v112
	v_mov_b32_e32 v138, v120
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v96, v135
	v_mov_b32_e32 v104, v136
	v_mov_b32_e32 v112, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v135, v1
	v_mov_b32_e32 v136, v9
	v_mov_b32_e32 v137, v17
	v_mov_b32_e32 v138, v25
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v1, v135
	v_mov_b32_e32 v9, v136
	v_mov_b32_e32 v17, v137
	v_mov_b32_e32 v25, v138
	v_mov_b32_e32 v135, v33
	v_mov_b32_e32 v136, v41
	v_mov_b32_e32 v137, v49
	v_mov_b32_e32 v138, v57
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v33, v135
	v_mov_b32_e32 v41, v136
	v_mov_b32_e32 v49, v137
	v_mov_b32_e32 v57, v138
	v_mov_b32_e32 v135, v65
	v_mov_b32_e32 v136, v73
	v_mov_b32_e32 v137, v81
	v_mov_b32_e32 v138, v89
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v65, v135
	v_mov_b32_e32 v73, v136
	v_mov_b32_e32 v81, v137
	v_mov_b32_e32 v89, v138
	v_mov_b32_e32 v135, v97
	v_mov_b32_e32 v136, v105
	v_mov_b32_e32 v137, v113
	v_mov_b32_e32 v138, v121
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v97, v135
	v_mov_b32_e32 v105, v136
	v_mov_b32_e32 v113, v137
	v_mov_b32_e32 v121, v138
	v_mov_b32_e32 v135, v2
	v_mov_b32_e32 v136, v10
	v_mov_b32_e32 v137, v18
	v_mov_b32_e32 v138, v26
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v10, v136
	v_mov_b32_e32 v18, v137
	v_mov_b32_e32 v26, v138
	v_mov_b32_e32 v135, v34
	v_mov_b32_e32 v136, v42
	v_mov_b32_e32 v137, v50
	v_mov_b32_e32 v138, v58
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v34, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v50, v137
	v_mov_b32_e32 v58, v138
	v_mov_b32_e32 v135, v66
	v_mov_b32_e32 v136, v74
	v_mov_b32_e32 v137, v82
	v_mov_b32_e32 v138, v90
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v66, v135
	v_mov_b32_e32 v74, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v90, v138
	v_mov_b32_e32 v135, v98
	v_mov_b32_e32 v136, v106
	v_mov_b32_e32 v137, v114
	v_mov_b32_e32 v138, v122
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v98, v135
	v_mov_b32_e32 v106, v136
	v_mov_b32_e32 v114, v137
	v_mov_b32_e32 v122, v138
	v_mov_b32_e32 v135, v3
	v_mov_b32_e32 v136, v11
	v_mov_b32_e32 v137, v19
	v_mov_b32_e32 v138, v27
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v3, v135
	v_mov_b32_e32 v11, v136
	v_mov_b32_e32 v19, v137
	v_mov_b32_e32 v27, v138
	v_mov_b32_e32 v135, v35
	v_mov_b32_e32 v136, v43
	v_mov_b32_e32 v137, v51
	v_mov_b32_e32 v138, v59
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v35, v135
	v_mov_b32_e32 v43, v136
	v_mov_b32_e32 v51, v137
	v_mov_b32_e32 v59, v138
	v_mov_b32_e32 v135, v67
	v_mov_b32_e32 v136, v75
	v_mov_b32_e32 v137, v83
	v_mov_b32_e32 v138, v91
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v67, v135
	v_mov_b32_e32 v75, v136
	v_mov_b32_e32 v83, v137
	v_mov_b32_e32 v91, v138
	v_mov_b32_e32 v135, v99
	v_mov_b32_e32 v136, v107
	v_mov_b32_e32 v137, v115
	v_mov_b32_e32 v138, v123
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v99, v135
	v_mov_b32_e32 v107, v136
	v_mov_b32_e32 v115, v137
	v_mov_b32_e32 v123, v138
	v_mov_b32_e32 v135, v4
	v_mov_b32_e32 v136, v12
	v_mov_b32_e32 v137, v20
	v_mov_b32_e32 v138, v28
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v12, v136
	v_mov_b32_e32 v20, v137
	v_mov_b32_e32 v28, v138
	v_mov_b32_e32 v135, v36
	v_mov_b32_e32 v136, v44
	v_mov_b32_e32 v137, v52
	v_mov_b32_e32 v138, v60
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v36, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v52, v137
	v_mov_b32_e32 v60, v138
	v_mov_b32_e32 v135, v68
	v_mov_b32_e32 v136, v76
	v_mov_b32_e32 v137, v84
	v_mov_b32_e32 v138, v92
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v68, v135
	v_mov_b32_e32 v76, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v92, v138
	v_mov_b32_e32 v135, v100
	v_mov_b32_e32 v136, v108
	v_mov_b32_e32 v137, v116
	v_mov_b32_e32 v138, v124
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v100, v135
	v_mov_b32_e32 v108, v136
	v_mov_b32_e32 v116, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v135, v5
	v_mov_b32_e32 v136, v13
	v_mov_b32_e32 v137, v21
	v_mov_b32_e32 v138, v29
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v5, v135
	v_mov_b32_e32 v13, v136
	v_mov_b32_e32 v21, v137
	v_mov_b32_e32 v29, v138
	v_mov_b32_e32 v135, v37
	v_mov_b32_e32 v136, v45
	v_mov_b32_e32 v137, v53
	v_mov_b32_e32 v138, v61
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v37, v135
	v_mov_b32_e32 v45, v136
	v_mov_b32_e32 v53, v137
	v_mov_b32_e32 v61, v138
	v_mov_b32_e32 v135, v69
	v_mov_b32_e32 v136, v77
	v_mov_b32_e32 v137, v85
	v_mov_b32_e32 v138, v93
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v69, v135
	v_mov_b32_e32 v77, v136
	v_mov_b32_e32 v85, v137
	v_mov_b32_e32 v93, v138
	v_mov_b32_e32 v135, v101
	v_mov_b32_e32 v136, v109
	v_mov_b32_e32 v137, v117
	v_mov_b32_e32 v138, v125
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v101, v135
	v_mov_b32_e32 v109, v136
	v_mov_b32_e32 v117, v137
	v_mov_b32_e32 v125, v138
	v_mov_b32_e32 v135, v6
	v_mov_b32_e32 v136, v14
	v_mov_b32_e32 v137, v22
	v_mov_b32_e32 v138, v30
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v14, v136
	v_mov_b32_e32 v22, v137
	v_mov_b32_e32 v30, v138
	v_mov_b32_e32 v135, v38
	v_mov_b32_e32 v136, v46
	v_mov_b32_e32 v137, v54
	v_mov_b32_e32 v138, v62
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v38, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v54, v137
	v_mov_b32_e32 v62, v138
	v_mov_b32_e32 v135, v70
	v_mov_b32_e32 v136, v78
	v_mov_b32_e32 v137, v86
	v_mov_b32_e32 v138, v94
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v70, v135
	v_mov_b32_e32 v78, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v94, v138
	v_mov_b32_e32 v135, v102
	v_mov_b32_e32 v136, v110
	v_mov_b32_e32 v137, v118
	v_mov_b32_e32 v138, v126
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v102, v135
	v_mov_b32_e32 v110, v136
	v_mov_b32_e32 v118, v137
	v_mov_b32_e32 v126, v138
	v_mov_b32_e32 v135, v7
	v_mov_b32_e32 v136, v15
	v_mov_b32_e32 v137, v23
	v_mov_b32_e32 v138, v31
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v7, v135
	v_mov_b32_e32 v15, v136
	v_mov_b32_e32 v23, v137
	v_mov_b32_e32 v31, v138
	v_mov_b32_e32 v135, v39
	v_mov_b32_e32 v136, v47
	v_mov_b32_e32 v137, v55
	v_mov_b32_e32 v138, v63
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v47, v136
	v_mov_b32_e32 v55, v137
	v_mov_b32_e32 v63, v138
	v_mov_b32_e32 v135, v71
	v_mov_b32_e32 v136, v79
	v_mov_b32_e32 v137, v87
	v_mov_b32_e32 v138, v95
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v71, v135
	v_mov_b32_e32 v79, v136
	v_mov_b32_e32 v87, v137
	v_mov_b32_e32 v95, v138
	v_mov_b32_e32 v135, v103
	v_mov_b32_e32 v136, v111
	v_mov_b32_e32 v137, v119
	v_mov_b32_e32 v138, v127
	ds_bpermute_b32 v135, v128, v135 offset:4
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v103, v135
	v_mov_b32_e32 v111, v136
	v_mov_b32_e32 v119, v137
	v_mov_b32_e32 v127, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW5_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v24
	v_mov_b32_e32 v136, v0
	v_mov_b32_e32 v137, v8
	v_mov_b32_e32 v138, v16
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v8, v136
	v_mov_b32_e32 v16, v137
	v_mov_b32_e32 v24, v138
	v_mov_b32_e32 v135, v56
	v_mov_b32_e32 v136, v32
	v_mov_b32_e32 v137, v40
	v_mov_b32_e32 v138, v48
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v32, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v48, v137
	v_mov_b32_e32 v56, v138
	v_mov_b32_e32 v135, v88
	v_mov_b32_e32 v136, v64
	v_mov_b32_e32 v137, v72
	v_mov_b32_e32 v138, v80
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v64, v135
	v_mov_b32_e32 v72, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v88, v138
	v_mov_b32_e32 v135, v120
	v_mov_b32_e32 v136, v96
	v_mov_b32_e32 v137, v104
	v_mov_b32_e32 v138, v112
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v96, v135
	v_mov_b32_e32 v104, v136
	v_mov_b32_e32 v112, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v135, v25
	v_mov_b32_e32 v136, v1
	v_mov_b32_e32 v137, v9
	v_mov_b32_e32 v138, v17
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v1, v135
	v_mov_b32_e32 v9, v136
	v_mov_b32_e32 v17, v137
	v_mov_b32_e32 v25, v138
	v_mov_b32_e32 v135, v57
	v_mov_b32_e32 v136, v33
	v_mov_b32_e32 v137, v41
	v_mov_b32_e32 v138, v49
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v33, v135
	v_mov_b32_e32 v41, v136
	v_mov_b32_e32 v49, v137
	v_mov_b32_e32 v57, v138
	v_mov_b32_e32 v135, v89
	v_mov_b32_e32 v136, v65
	v_mov_b32_e32 v137, v73
	v_mov_b32_e32 v138, v81
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v65, v135
	v_mov_b32_e32 v73, v136
	v_mov_b32_e32 v81, v137
	v_mov_b32_e32 v89, v138
	v_mov_b32_e32 v135, v121
	v_mov_b32_e32 v136, v97
	v_mov_b32_e32 v137, v105
	v_mov_b32_e32 v138, v113
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v97, v135
	v_mov_b32_e32 v105, v136
	v_mov_b32_e32 v113, v137
	v_mov_b32_e32 v121, v138
	v_mov_b32_e32 v135, v26
	v_mov_b32_e32 v136, v2
	v_mov_b32_e32 v137, v10
	v_mov_b32_e32 v138, v18
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v10, v136
	v_mov_b32_e32 v18, v137
	v_mov_b32_e32 v26, v138
	v_mov_b32_e32 v135, v58
	v_mov_b32_e32 v136, v34
	v_mov_b32_e32 v137, v42
	v_mov_b32_e32 v138, v50
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v34, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v50, v137
	v_mov_b32_e32 v58, v138
	v_mov_b32_e32 v135, v90
	v_mov_b32_e32 v136, v66
	v_mov_b32_e32 v137, v74
	v_mov_b32_e32 v138, v82
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v66, v135
	v_mov_b32_e32 v74, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v90, v138
	v_mov_b32_e32 v135, v122
	v_mov_b32_e32 v136, v98
	v_mov_b32_e32 v137, v106
	v_mov_b32_e32 v138, v114
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v98, v135
	v_mov_b32_e32 v106, v136
	v_mov_b32_e32 v114, v137
	v_mov_b32_e32 v122, v138
	v_mov_b32_e32 v135, v27
	v_mov_b32_e32 v136, v3
	v_mov_b32_e32 v137, v11
	v_mov_b32_e32 v138, v19
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v3, v135
	v_mov_b32_e32 v11, v136
	v_mov_b32_e32 v19, v137
	v_mov_b32_e32 v27, v138
	v_mov_b32_e32 v135, v59
	v_mov_b32_e32 v136, v35
	v_mov_b32_e32 v137, v43
	v_mov_b32_e32 v138, v51
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v35, v135
	v_mov_b32_e32 v43, v136
	v_mov_b32_e32 v51, v137
	v_mov_b32_e32 v59, v138
	v_mov_b32_e32 v135, v91
	v_mov_b32_e32 v136, v67
	v_mov_b32_e32 v137, v75
	v_mov_b32_e32 v138, v83
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v67, v135
	v_mov_b32_e32 v75, v136
	v_mov_b32_e32 v83, v137
	v_mov_b32_e32 v91, v138
	v_mov_b32_e32 v135, v123
	v_mov_b32_e32 v136, v99
	v_mov_b32_e32 v137, v107
	v_mov_b32_e32 v138, v115
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v99, v135
	v_mov_b32_e32 v107, v136
	v_mov_b32_e32 v115, v137
	v_mov_b32_e32 v123, v138
	v_mov_b32_e32 v135, v28
	v_mov_b32_e32 v136, v4
	v_mov_b32_e32 v137, v12
	v_mov_b32_e32 v138, v20
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v12, v136
	v_mov_b32_e32 v20, v137
	v_mov_b32_e32 v28, v138
	v_mov_b32_e32 v135, v60
	v_mov_b32_e32 v136, v36
	v_mov_b32_e32 v137, v44
	v_mov_b32_e32 v138, v52
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v36, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v52, v137
	v_mov_b32_e32 v60, v138
	v_mov_b32_e32 v135, v92
	v_mov_b32_e32 v136, v68
	v_mov_b32_e32 v137, v76
	v_mov_b32_e32 v138, v84
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v68, v135
	v_mov_b32_e32 v76, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v92, v138
	v_mov_b32_e32 v135, v124
	v_mov_b32_e32 v136, v100
	v_mov_b32_e32 v137, v108
	v_mov_b32_e32 v138, v116
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v100, v135
	v_mov_b32_e32 v108, v136
	v_mov_b32_e32 v116, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v135, v29
	v_mov_b32_e32 v136, v5
	v_mov_b32_e32 v137, v13
	v_mov_b32_e32 v138, v21
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v5, v135
	v_mov_b32_e32 v13, v136
	v_mov_b32_e32 v21, v137
	v_mov_b32_e32 v29, v138
	v_mov_b32_e32 v135, v61
	v_mov_b32_e32 v136, v37
	v_mov_b32_e32 v137, v45
	v_mov_b32_e32 v138, v53
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v37, v135
	v_mov_b32_e32 v45, v136
	v_mov_b32_e32 v53, v137
	v_mov_b32_e32 v61, v138
	v_mov_b32_e32 v135, v93
	v_mov_b32_e32 v136, v69
	v_mov_b32_e32 v137, v77
	v_mov_b32_e32 v138, v85
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v69, v135
	v_mov_b32_e32 v77, v136
	v_mov_b32_e32 v85, v137
	v_mov_b32_e32 v93, v138
	v_mov_b32_e32 v135, v125
	v_mov_b32_e32 v136, v101
	v_mov_b32_e32 v137, v109
	v_mov_b32_e32 v138, v117
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v101, v135
	v_mov_b32_e32 v109, v136
	v_mov_b32_e32 v117, v137
	v_mov_b32_e32 v125, v138
	v_mov_b32_e32 v135, v30
	v_mov_b32_e32 v136, v6
	v_mov_b32_e32 v137, v14
	v_mov_b32_e32 v138, v22
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v14, v136
	v_mov_b32_e32 v22, v137
	v_mov_b32_e32 v30, v138
	v_mov_b32_e32 v135, v62
	v_mov_b32_e32 v136, v38
	v_mov_b32_e32 v137, v46
	v_mov_b32_e32 v138, v54
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v38, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v54, v137
	v_mov_b32_e32 v62, v138
	v_mov_b32_e32 v135, v94
	v_mov_b32_e32 v136, v70
	v_mov_b32_e32 v137, v78
	v_mov_b32_e32 v138, v86
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v70, v135
	v_mov_b32_e32 v78, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v94, v138
	v_mov_b32_e32 v135, v126
	v_mov_b32_e32 v136, v102
	v_mov_b32_e32 v137, v110
	v_mov_b32_e32 v138, v118
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v102, v135
	v_mov_b32_e32 v110, v136
	v_mov_b32_e32 v118, v137
	v_mov_b32_e32 v126, v138
	v_mov_b32_e32 v135, v31
	v_mov_b32_e32 v136, v7
	v_mov_b32_e32 v137, v15
	v_mov_b32_e32 v138, v23
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v7, v135
	v_mov_b32_e32 v15, v136
	v_mov_b32_e32 v23, v137
	v_mov_b32_e32 v31, v138
	v_mov_b32_e32 v135, v63
	v_mov_b32_e32 v136, v39
	v_mov_b32_e32 v137, v47
	v_mov_b32_e32 v138, v55
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v47, v136
	v_mov_b32_e32 v55, v137
	v_mov_b32_e32 v63, v138
	v_mov_b32_e32 v135, v95
	v_mov_b32_e32 v136, v71
	v_mov_b32_e32 v137, v79
	v_mov_b32_e32 v138, v87
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v71, v135
	v_mov_b32_e32 v79, v136
	v_mov_b32_e32 v87, v137
	v_mov_b32_e32 v95, v138
	v_mov_b32_e32 v135, v127
	v_mov_b32_e32 v136, v103
	v_mov_b32_e32 v137, v111
	v_mov_b32_e32 v138, v119
	ds_bpermute_b32 v136, v128, v136 offset:4
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v103, v135
	v_mov_b32_e32 v111, v136
	v_mov_b32_e32 v119, v137
	v_mov_b32_e32 v127, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW6_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v16
	v_mov_b32_e32 v136, v24
	v_mov_b32_e32 v137, v0
	v_mov_b32_e32 v138, v8
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v8, v136
	v_mov_b32_e32 v16, v137
	v_mov_b32_e32 v24, v138
	v_mov_b32_e32 v135, v48
	v_mov_b32_e32 v136, v56
	v_mov_b32_e32 v137, v32
	v_mov_b32_e32 v138, v40
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v32, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v48, v137
	v_mov_b32_e32 v56, v138
	v_mov_b32_e32 v135, v80
	v_mov_b32_e32 v136, v88
	v_mov_b32_e32 v137, v64
	v_mov_b32_e32 v138, v72
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v64, v135
	v_mov_b32_e32 v72, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v88, v138
	v_mov_b32_e32 v135, v112
	v_mov_b32_e32 v136, v120
	v_mov_b32_e32 v137, v96
	v_mov_b32_e32 v138, v104
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v96, v135
	v_mov_b32_e32 v104, v136
	v_mov_b32_e32 v112, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v135, v17
	v_mov_b32_e32 v136, v25
	v_mov_b32_e32 v137, v1
	v_mov_b32_e32 v138, v9
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v1, v135
	v_mov_b32_e32 v9, v136
	v_mov_b32_e32 v17, v137
	v_mov_b32_e32 v25, v138
	v_mov_b32_e32 v135, v49
	v_mov_b32_e32 v136, v57
	v_mov_b32_e32 v137, v33
	v_mov_b32_e32 v138, v41
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v33, v135
	v_mov_b32_e32 v41, v136
	v_mov_b32_e32 v49, v137
	v_mov_b32_e32 v57, v138
	v_mov_b32_e32 v135, v81
	v_mov_b32_e32 v136, v89
	v_mov_b32_e32 v137, v65
	v_mov_b32_e32 v138, v73
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v65, v135
	v_mov_b32_e32 v73, v136
	v_mov_b32_e32 v81, v137
	v_mov_b32_e32 v89, v138
	v_mov_b32_e32 v135, v113
	v_mov_b32_e32 v136, v121
	v_mov_b32_e32 v137, v97
	v_mov_b32_e32 v138, v105
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v97, v135
	v_mov_b32_e32 v105, v136
	v_mov_b32_e32 v113, v137
	v_mov_b32_e32 v121, v138
	v_mov_b32_e32 v135, v18
	v_mov_b32_e32 v136, v26
	v_mov_b32_e32 v137, v2
	v_mov_b32_e32 v138, v10
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v10, v136
	v_mov_b32_e32 v18, v137
	v_mov_b32_e32 v26, v138
	v_mov_b32_e32 v135, v50
	v_mov_b32_e32 v136, v58
	v_mov_b32_e32 v137, v34
	v_mov_b32_e32 v138, v42
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v34, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v50, v137
	v_mov_b32_e32 v58, v138
	v_mov_b32_e32 v135, v82
	v_mov_b32_e32 v136, v90
	v_mov_b32_e32 v137, v66
	v_mov_b32_e32 v138, v74
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v66, v135
	v_mov_b32_e32 v74, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v90, v138
	v_mov_b32_e32 v135, v114
	v_mov_b32_e32 v136, v122
	v_mov_b32_e32 v137, v98
	v_mov_b32_e32 v138, v106
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v98, v135
	v_mov_b32_e32 v106, v136
	v_mov_b32_e32 v114, v137
	v_mov_b32_e32 v122, v138
	v_mov_b32_e32 v135, v19
	v_mov_b32_e32 v136, v27
	v_mov_b32_e32 v137, v3
	v_mov_b32_e32 v138, v11
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v3, v135
	v_mov_b32_e32 v11, v136
	v_mov_b32_e32 v19, v137
	v_mov_b32_e32 v27, v138
	v_mov_b32_e32 v135, v51
	v_mov_b32_e32 v136, v59
	v_mov_b32_e32 v137, v35
	v_mov_b32_e32 v138, v43
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v35, v135
	v_mov_b32_e32 v43, v136
	v_mov_b32_e32 v51, v137
	v_mov_b32_e32 v59, v138
	v_mov_b32_e32 v135, v83
	v_mov_b32_e32 v136, v91
	v_mov_b32_e32 v137, v67
	v_mov_b32_e32 v138, v75
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v67, v135
	v_mov_b32_e32 v75, v136
	v_mov_b32_e32 v83, v137
	v_mov_b32_e32 v91, v138
	v_mov_b32_e32 v135, v115
	v_mov_b32_e32 v136, v123
	v_mov_b32_e32 v137, v99
	v_mov_b32_e32 v138, v107
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v99, v135
	v_mov_b32_e32 v107, v136
	v_mov_b32_e32 v115, v137
	v_mov_b32_e32 v123, v138
	v_mov_b32_e32 v135, v20
	v_mov_b32_e32 v136, v28
	v_mov_b32_e32 v137, v4
	v_mov_b32_e32 v138, v12
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v12, v136
	v_mov_b32_e32 v20, v137
	v_mov_b32_e32 v28, v138
	v_mov_b32_e32 v135, v52
	v_mov_b32_e32 v136, v60
	v_mov_b32_e32 v137, v36
	v_mov_b32_e32 v138, v44
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v36, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v52, v137
	v_mov_b32_e32 v60, v138
	v_mov_b32_e32 v135, v84
	v_mov_b32_e32 v136, v92
	v_mov_b32_e32 v137, v68
	v_mov_b32_e32 v138, v76
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v68, v135
	v_mov_b32_e32 v76, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v92, v138
	v_mov_b32_e32 v135, v116
	v_mov_b32_e32 v136, v124
	v_mov_b32_e32 v137, v100
	v_mov_b32_e32 v138, v108
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v100, v135
	v_mov_b32_e32 v108, v136
	v_mov_b32_e32 v116, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v135, v21
	v_mov_b32_e32 v136, v29
	v_mov_b32_e32 v137, v5
	v_mov_b32_e32 v138, v13
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v5, v135
	v_mov_b32_e32 v13, v136
	v_mov_b32_e32 v21, v137
	v_mov_b32_e32 v29, v138
	v_mov_b32_e32 v135, v53
	v_mov_b32_e32 v136, v61
	v_mov_b32_e32 v137, v37
	v_mov_b32_e32 v138, v45
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v37, v135
	v_mov_b32_e32 v45, v136
	v_mov_b32_e32 v53, v137
	v_mov_b32_e32 v61, v138
	v_mov_b32_e32 v135, v85
	v_mov_b32_e32 v136, v93
	v_mov_b32_e32 v137, v69
	v_mov_b32_e32 v138, v77
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v69, v135
	v_mov_b32_e32 v77, v136
	v_mov_b32_e32 v85, v137
	v_mov_b32_e32 v93, v138
	v_mov_b32_e32 v135, v117
	v_mov_b32_e32 v136, v125
	v_mov_b32_e32 v137, v101
	v_mov_b32_e32 v138, v109
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v101, v135
	v_mov_b32_e32 v109, v136
	v_mov_b32_e32 v117, v137
	v_mov_b32_e32 v125, v138
	v_mov_b32_e32 v135, v22
	v_mov_b32_e32 v136, v30
	v_mov_b32_e32 v137, v6
	v_mov_b32_e32 v138, v14
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v14, v136
	v_mov_b32_e32 v22, v137
	v_mov_b32_e32 v30, v138
	v_mov_b32_e32 v135, v54
	v_mov_b32_e32 v136, v62
	v_mov_b32_e32 v137, v38
	v_mov_b32_e32 v138, v46
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v38, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v54, v137
	v_mov_b32_e32 v62, v138
	v_mov_b32_e32 v135, v86
	v_mov_b32_e32 v136, v94
	v_mov_b32_e32 v137, v70
	v_mov_b32_e32 v138, v78
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v70, v135
	v_mov_b32_e32 v78, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v94, v138
	v_mov_b32_e32 v135, v118
	v_mov_b32_e32 v136, v126
	v_mov_b32_e32 v137, v102
	v_mov_b32_e32 v138, v110
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v102, v135
	v_mov_b32_e32 v110, v136
	v_mov_b32_e32 v118, v137
	v_mov_b32_e32 v126, v138
	v_mov_b32_e32 v135, v23
	v_mov_b32_e32 v136, v31
	v_mov_b32_e32 v137, v7
	v_mov_b32_e32 v138, v15
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v7, v135
	v_mov_b32_e32 v15, v136
	v_mov_b32_e32 v23, v137
	v_mov_b32_e32 v31, v138
	v_mov_b32_e32 v135, v55
	v_mov_b32_e32 v136, v63
	v_mov_b32_e32 v137, v39
	v_mov_b32_e32 v138, v47
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v47, v136
	v_mov_b32_e32 v55, v137
	v_mov_b32_e32 v63, v138
	v_mov_b32_e32 v135, v87
	v_mov_b32_e32 v136, v95
	v_mov_b32_e32 v137, v71
	v_mov_b32_e32 v138, v79
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v71, v135
	v_mov_b32_e32 v79, v136
	v_mov_b32_e32 v87, v137
	v_mov_b32_e32 v95, v138
	v_mov_b32_e32 v135, v119
	v_mov_b32_e32 v136, v127
	v_mov_b32_e32 v137, v103
	v_mov_b32_e32 v138, v111
	ds_bpermute_b32 v137, v128, v137 offset:4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v103, v135
	v_mov_b32_e32 v111, v136
	v_mov_b32_e32 v119, v137
	v_mov_b32_e32 v127, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW7_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v8
	v_mov_b32_e32 v136, v16
	v_mov_b32_e32 v137, v24
	v_mov_b32_e32 v138, v0
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v8, v136
	v_mov_b32_e32 v16, v137
	v_mov_b32_e32 v24, v138
	v_mov_b32_e32 v135, v40
	v_mov_b32_e32 v136, v48
	v_mov_b32_e32 v137, v56
	v_mov_b32_e32 v138, v32
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v32, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v48, v137
	v_mov_b32_e32 v56, v138
	v_mov_b32_e32 v135, v72
	v_mov_b32_e32 v136, v80
	v_mov_b32_e32 v137, v88
	v_mov_b32_e32 v138, v64
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v64, v135
	v_mov_b32_e32 v72, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v88, v138
	v_mov_b32_e32 v135, v104
	v_mov_b32_e32 v136, v112
	v_mov_b32_e32 v137, v120
	v_mov_b32_e32 v138, v96
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v96, v135
	v_mov_b32_e32 v104, v136
	v_mov_b32_e32 v112, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v135, v9
	v_mov_b32_e32 v136, v17
	v_mov_b32_e32 v137, v25
	v_mov_b32_e32 v138, v1
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v1, v135
	v_mov_b32_e32 v9, v136
	v_mov_b32_e32 v17, v137
	v_mov_b32_e32 v25, v138
	v_mov_b32_e32 v135, v41
	v_mov_b32_e32 v136, v49
	v_mov_b32_e32 v137, v57
	v_mov_b32_e32 v138, v33
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v33, v135
	v_mov_b32_e32 v41, v136
	v_mov_b32_e32 v49, v137
	v_mov_b32_e32 v57, v138
	v_mov_b32_e32 v135, v73
	v_mov_b32_e32 v136, v81
	v_mov_b32_e32 v137, v89
	v_mov_b32_e32 v138, v65
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v65, v135
	v_mov_b32_e32 v73, v136
	v_mov_b32_e32 v81, v137
	v_mov_b32_e32 v89, v138
	v_mov_b32_e32 v135, v105
	v_mov_b32_e32 v136, v113
	v_mov_b32_e32 v137, v121
	v_mov_b32_e32 v138, v97
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v97, v135
	v_mov_b32_e32 v105, v136
	v_mov_b32_e32 v113, v137
	v_mov_b32_e32 v121, v138
	v_mov_b32_e32 v135, v10
	v_mov_b32_e32 v136, v18
	v_mov_b32_e32 v137, v26
	v_mov_b32_e32 v138, v2
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v10, v136
	v_mov_b32_e32 v18, v137
	v_mov_b32_e32 v26, v138
	v_mov_b32_e32 v135, v42
	v_mov_b32_e32 v136, v50
	v_mov_b32_e32 v137, v58
	v_mov_b32_e32 v138, v34
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v34, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v50, v137
	v_mov_b32_e32 v58, v138
	v_mov_b32_e32 v135, v74
	v_mov_b32_e32 v136, v82
	v_mov_b32_e32 v137, v90
	v_mov_b32_e32 v138, v66
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v66, v135
	v_mov_b32_e32 v74, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v90, v138
	v_mov_b32_e32 v135, v106
	v_mov_b32_e32 v136, v114
	v_mov_b32_e32 v137, v122
	v_mov_b32_e32 v138, v98
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v98, v135
	v_mov_b32_e32 v106, v136
	v_mov_b32_e32 v114, v137
	v_mov_b32_e32 v122, v138
	v_mov_b32_e32 v135, v11
	v_mov_b32_e32 v136, v19
	v_mov_b32_e32 v137, v27
	v_mov_b32_e32 v138, v3
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v3, v135
	v_mov_b32_e32 v11, v136
	v_mov_b32_e32 v19, v137
	v_mov_b32_e32 v27, v138
	v_mov_b32_e32 v135, v43
	v_mov_b32_e32 v136, v51
	v_mov_b32_e32 v137, v59
	v_mov_b32_e32 v138, v35
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v35, v135
	v_mov_b32_e32 v43, v136
	v_mov_b32_e32 v51, v137
	v_mov_b32_e32 v59, v138
	v_mov_b32_e32 v135, v75
	v_mov_b32_e32 v136, v83
	v_mov_b32_e32 v137, v91
	v_mov_b32_e32 v138, v67
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v67, v135
	v_mov_b32_e32 v75, v136
	v_mov_b32_e32 v83, v137
	v_mov_b32_e32 v91, v138
	v_mov_b32_e32 v135, v107
	v_mov_b32_e32 v136, v115
	v_mov_b32_e32 v137, v123
	v_mov_b32_e32 v138, v99
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v99, v135
	v_mov_b32_e32 v107, v136
	v_mov_b32_e32 v115, v137
	v_mov_b32_e32 v123, v138
	v_mov_b32_e32 v135, v12
	v_mov_b32_e32 v136, v20
	v_mov_b32_e32 v137, v28
	v_mov_b32_e32 v138, v4
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v12, v136
	v_mov_b32_e32 v20, v137
	v_mov_b32_e32 v28, v138
	v_mov_b32_e32 v135, v44
	v_mov_b32_e32 v136, v52
	v_mov_b32_e32 v137, v60
	v_mov_b32_e32 v138, v36
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v36, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v52, v137
	v_mov_b32_e32 v60, v138
	v_mov_b32_e32 v135, v76
	v_mov_b32_e32 v136, v84
	v_mov_b32_e32 v137, v92
	v_mov_b32_e32 v138, v68
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v68, v135
	v_mov_b32_e32 v76, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v92, v138
	v_mov_b32_e32 v135, v108
	v_mov_b32_e32 v136, v116
	v_mov_b32_e32 v137, v124
	v_mov_b32_e32 v138, v100
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v100, v135
	v_mov_b32_e32 v108, v136
	v_mov_b32_e32 v116, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v135, v13
	v_mov_b32_e32 v136, v21
	v_mov_b32_e32 v137, v29
	v_mov_b32_e32 v138, v5
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v5, v135
	v_mov_b32_e32 v13, v136
	v_mov_b32_e32 v21, v137
	v_mov_b32_e32 v29, v138
	v_mov_b32_e32 v135, v45
	v_mov_b32_e32 v136, v53
	v_mov_b32_e32 v137, v61
	v_mov_b32_e32 v138, v37
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v37, v135
	v_mov_b32_e32 v45, v136
	v_mov_b32_e32 v53, v137
	v_mov_b32_e32 v61, v138
	v_mov_b32_e32 v135, v77
	v_mov_b32_e32 v136, v85
	v_mov_b32_e32 v137, v93
	v_mov_b32_e32 v138, v69
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v69, v135
	v_mov_b32_e32 v77, v136
	v_mov_b32_e32 v85, v137
	v_mov_b32_e32 v93, v138
	v_mov_b32_e32 v135, v109
	v_mov_b32_e32 v136, v117
	v_mov_b32_e32 v137, v125
	v_mov_b32_e32 v138, v101
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v101, v135
	v_mov_b32_e32 v109, v136
	v_mov_b32_e32 v117, v137
	v_mov_b32_e32 v125, v138
	v_mov_b32_e32 v135, v14
	v_mov_b32_e32 v136, v22
	v_mov_b32_e32 v137, v30
	v_mov_b32_e32 v138, v6
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v14, v136
	v_mov_b32_e32 v22, v137
	v_mov_b32_e32 v30, v138
	v_mov_b32_e32 v135, v46
	v_mov_b32_e32 v136, v54
	v_mov_b32_e32 v137, v62
	v_mov_b32_e32 v138, v38
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v38, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v54, v137
	v_mov_b32_e32 v62, v138
	v_mov_b32_e32 v135, v78
	v_mov_b32_e32 v136, v86
	v_mov_b32_e32 v137, v94
	v_mov_b32_e32 v138, v70
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v70, v135
	v_mov_b32_e32 v78, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v94, v138
	v_mov_b32_e32 v135, v110
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v126
	v_mov_b32_e32 v138, v102
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v102, v135
	v_mov_b32_e32 v110, v136
	v_mov_b32_e32 v118, v137
	v_mov_b32_e32 v126, v138
	v_mov_b32_e32 v135, v15
	v_mov_b32_e32 v136, v23
	v_mov_b32_e32 v137, v31
	v_mov_b32_e32 v138, v7
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v7, v135
	v_mov_b32_e32 v15, v136
	v_mov_b32_e32 v23, v137
	v_mov_b32_e32 v31, v138
	v_mov_b32_e32 v135, v47
	v_mov_b32_e32 v136, v55
	v_mov_b32_e32 v137, v63
	v_mov_b32_e32 v138, v39
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v39, v135
	v_mov_b32_e32 v47, v136
	v_mov_b32_e32 v55, v137
	v_mov_b32_e32 v63, v138
	v_mov_b32_e32 v135, v79
	v_mov_b32_e32 v136, v87
	v_mov_b32_e32 v137, v95
	v_mov_b32_e32 v138, v71
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v71, v135
	v_mov_b32_e32 v79, v136
	v_mov_b32_e32 v87, v137
	v_mov_b32_e32 v95, v138
	v_mov_b32_e32 v135, v111
	v_mov_b32_e32 v136, v119
	v_mov_b32_e32 v137, v127
	v_mov_b32_e32 v138, v103
	ds_bpermute_b32 v138, v128, v138 offset:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	s_wait_loadcnt 0x0
	s_wait_storecnt 0x0
	v_mov_b32_e32 v103, v135
	v_mov_b32_e32 v111, v136
	v_mov_b32_e32 v119, v137
	v_mov_b32_e32 v127, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents0_GLVW0
label_ShiftVectorComponents0_GLVW0:
	v_mov_b32_e32 v131, s3
	v_mul_i32_i24_e32 v131, 0xffffff80, v131
	v_add_co_u32 v131, vcc_lo, s25, v131
	v_mov_b32_e32 v132, 0x80
	v_cmp_lt_u32_e64 s8, v131, v132
	v_cndmask_b32_e64 v131, v132, v131, s8
	v_lshrrev_b32_e32 v133, 6, v218
	v_and_b32_e32 v133, 1, v133
	v_lshrrev_b32_e32 v134, 6, v131
	v_and_b32_e32 v134, 1, v134
	v_cmp_eq_u32_e64 s8, v134, v133
	v_cndmask_b32_e64 v131, v132, v131, s8
	v_lshrrev_b32_e32 v132, 6, v131
	v_lshlrev_b32_e32 v134, 0, v133
	v_sub_nc_u32_e32 v132, v132, v134
	v_lshrrev_b32_e32 v134, 3, v131
	v_lshrrev_b32_e32 v135, 4, v218
	v_and_b32_e32 v135, 1, v135
	v_lshlrev_b32_e32 v135, 5, v135
	v_lshrrev_b32_e32 v135, 3, v135
	v_lshlrev_b32_e32 v133, 3, v133
	v_add_co_u32 v135, vcc_lo, v133, v135
	v_sub_nc_u32_e32 v134, v134, v135
	v_and_b32_e32 v133, 31, v131
	v_lshrrev_b32_e32 v133, 3, v133
	v_and_b32_e32 v135, 7, v131
	v_cmp_eq_u32_e64 vcc_lo, v135, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW1
	v_cmp_eq_u32_e64 vcc_lo, v135, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW2
	v_cmp_eq_u32_e64 vcc_lo, v135, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW3
	v_cmp_eq_u32_e64 vcc_lo, v135, 4
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW4
	v_cmp_eq_u32_e64 vcc_lo, v135, 5
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW5
	v_cmp_eq_u32_e64 vcc_lo, v135, 6
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW6
	v_cmp_eq_u32_e64 vcc_lo, v135, 7
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW7
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW1:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW1_BM0
label_ShiftVectorComponents1_GLVW2:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW2_BM0
label_ShiftVectorComponents1_GLVW3:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW3_BM0
label_ShiftVectorComponents1_GLVW4:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW4_BM0
label_ShiftVectorComponents1_GLVW5:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW5_BM0
label_ShiftVectorComponents1_GLVW6:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW6_BM0
label_ShiftVectorComponents1_GLVW7:
	v_cmp_eq_u32_e64 vcc_lo, v132, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW7_BM0
label_ShiftVectorComponents1_GLVW1_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW1_BM0_VW0
	v_cmp_eq_u32_e64 vcc_lo, v133, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW1_BM0_VW1
	v_cmp_eq_u32_e64 vcc_lo, v133, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW1_BM0_VW2
	v_cmp_eq_u32_e64 vcc_lo, v133, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW1_BM0_VW3
label_ShiftVectorComponents1_GLVW2_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW2_BM0_VW0
	v_cmp_eq_u32_e64 vcc_lo, v133, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW2_BM0_VW1
	v_cmp_eq_u32_e64 vcc_lo, v133, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW2_BM0_VW2
	v_cmp_eq_u32_e64 vcc_lo, v133, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW2_BM0_VW3
label_ShiftVectorComponents1_GLVW3_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW3_BM0_VW0
	v_cmp_eq_u32_e64 vcc_lo, v133, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW3_BM0_VW1
	v_cmp_eq_u32_e64 vcc_lo, v133, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW3_BM0_VW2
	v_cmp_eq_u32_e64 vcc_lo, v133, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW3_BM0_VW3
label_ShiftVectorComponents1_GLVW4_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW4_BM0_VW0
	v_cmp_eq_u32_e64 vcc_lo, v133, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW4_BM0_VW1
	v_cmp_eq_u32_e64 vcc_lo, v133, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW4_BM0_VW2
	v_cmp_eq_u32_e64 vcc_lo, v133, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW4_BM0_VW3
label_ShiftVectorComponents1_GLVW5_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW5_BM0_VW0
	v_cmp_eq_u32_e64 vcc_lo, v133, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW5_BM0_VW1
	v_cmp_eq_u32_e64 vcc_lo, v133, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW5_BM0_VW2
	v_cmp_eq_u32_e64 vcc_lo, v133, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW5_BM0_VW3
label_ShiftVectorComponents1_GLVW6_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW6_BM0_VW0
	v_cmp_eq_u32_e64 vcc_lo, v133, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW6_BM0_VW1
	v_cmp_eq_u32_e64 vcc_lo, v133, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW6_BM0_VW2
	v_cmp_eq_u32_e64 vcc_lo, v133, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW6_BM0_VW3
label_ShiftVectorComponents1_GLVW7_BM0:
	v_cmp_eq_u32_e64 vcc_lo, v133, 0
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW7_BM0_VW0
	v_cmp_eq_u32_e64 vcc_lo, v133, 1
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW7_BM0_VW1
	v_cmp_eq_u32_e64 vcc_lo, v133, 2
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW7_BM0_VW2
	v_cmp_eq_u32_e64 vcc_lo, v133, 3
	s_cbranch_vccnz label_ShiftVectorComponents1_GLVW7_BM0_VW3
label_ShiftVectorComponents1_GLVW1_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v97
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v135, v105
	v_mov_b32_e32 v8, v135
	v_mov_b32_e32 v135, v113
	v_mov_b32_e32 v16, v135
	v_mov_b32_e32 v135, v121
	v_mov_b32_e32 v24, v135
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW1_BM0_VW1:
	s_mov_b32 s8, 1
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v99
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v135, v107
	v_mov_b32_e32 v10, v135
	v_mov_b32_e32 v135, v115
	v_mov_b32_e32 v18, v135
	v_mov_b32_e32 v135, v123
	v_mov_b32_e32 v26, v135
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW1_BM0_VW2:
	s_mov_b32 s8, 2
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v101
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v135, v109
	v_mov_b32_e32 v12, v135
	v_mov_b32_e32 v135, v117
	v_mov_b32_e32 v20, v135
	v_mov_b32_e32 v135, v125
	v_mov_b32_e32 v28, v135
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW1_BM0_VW3:
	s_mov_b32 s8, 3
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v103
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v135, v111
	v_mov_b32_e32 v14, v135
	v_mov_b32_e32 v135, v119
	v_mov_b32_e32 v22, v135
	v_mov_b32_e32 v135, v127
	v_mov_b32_e32 v30, v135
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW2_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v65
	v_mov_b32_e32 v136, v97
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v32, v136
	v_mov_b32_e32 v135, v73
	v_mov_b32_e32 v136, v105
	v_mov_b32_e32 v8, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v135, v81
	v_mov_b32_e32 v136, v113
	v_mov_b32_e32 v16, v135
	v_mov_b32_e32 v48, v136
	v_mov_b32_e32 v135, v89
	v_mov_b32_e32 v136, v121
	v_mov_b32_e32 v24, v135
	v_mov_b32_e32 v56, v136
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW2_BM0_VW1:
	s_mov_b32 s8, 1
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v67
	v_mov_b32_e32 v136, v99
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v34, v136
	v_mov_b32_e32 v135, v75
	v_mov_b32_e32 v136, v107
	v_mov_b32_e32 v10, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v135, v83
	v_mov_b32_e32 v136, v115
	v_mov_b32_e32 v18, v135
	v_mov_b32_e32 v50, v136
	v_mov_b32_e32 v135, v91
	v_mov_b32_e32 v136, v123
	v_mov_b32_e32 v26, v135
	v_mov_b32_e32 v58, v136
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW2_BM0_VW2:
	s_mov_b32 s8, 2
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v69
	v_mov_b32_e32 v136, v101
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v36, v136
	v_mov_b32_e32 v135, v77
	v_mov_b32_e32 v136, v109
	v_mov_b32_e32 v12, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v135, v85
	v_mov_b32_e32 v136, v117
	v_mov_b32_e32 v20, v135
	v_mov_b32_e32 v52, v136
	v_mov_b32_e32 v135, v93
	v_mov_b32_e32 v136, v125
	v_mov_b32_e32 v28, v135
	v_mov_b32_e32 v60, v136
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW2_BM0_VW3:
	s_mov_b32 s8, 3
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v71
	v_mov_b32_e32 v136, v103
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v38, v136
	v_mov_b32_e32 v135, v79
	v_mov_b32_e32 v136, v111
	v_mov_b32_e32 v14, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v135, v87
	v_mov_b32_e32 v136, v119
	v_mov_b32_e32 v22, v135
	v_mov_b32_e32 v54, v136
	v_mov_b32_e32 v135, v95
	v_mov_b32_e32 v136, v127
	v_mov_b32_e32 v30, v135
	v_mov_b32_e32 v62, v136
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW3_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v33
	v_mov_b32_e32 v136, v65
	v_mov_b32_e32 v137, v97
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v32, v136
	v_mov_b32_e32 v64, v137
	v_mov_b32_e32 v135, v41
	v_mov_b32_e32 v136, v73
	v_mov_b32_e32 v137, v105
	v_mov_b32_e32 v8, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v72, v137
	v_mov_b32_e32 v135, v49
	v_mov_b32_e32 v136, v81
	v_mov_b32_e32 v137, v113
	v_mov_b32_e32 v16, v135
	v_mov_b32_e32 v48, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v135, v57
	v_mov_b32_e32 v136, v89
	v_mov_b32_e32 v137, v121
	v_mov_b32_e32 v24, v135
	v_mov_b32_e32 v56, v136
	v_mov_b32_e32 v88, v137
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW3_BM0_VW1:
	s_mov_b32 s8, 1
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v35
	v_mov_b32_e32 v136, v67
	v_mov_b32_e32 v137, v99
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v34, v136
	v_mov_b32_e32 v66, v137
	v_mov_b32_e32 v135, v43
	v_mov_b32_e32 v136, v75
	v_mov_b32_e32 v137, v107
	v_mov_b32_e32 v10, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v74, v137
	v_mov_b32_e32 v135, v51
	v_mov_b32_e32 v136, v83
	v_mov_b32_e32 v137, v115
	v_mov_b32_e32 v18, v135
	v_mov_b32_e32 v50, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v135, v59
	v_mov_b32_e32 v136, v91
	v_mov_b32_e32 v137, v123
	v_mov_b32_e32 v26, v135
	v_mov_b32_e32 v58, v136
	v_mov_b32_e32 v90, v137
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW3_BM0_VW2:
	s_mov_b32 s8, 2
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v37
	v_mov_b32_e32 v136, v69
	v_mov_b32_e32 v137, v101
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v36, v136
	v_mov_b32_e32 v68, v137
	v_mov_b32_e32 v135, v45
	v_mov_b32_e32 v136, v77
	v_mov_b32_e32 v137, v109
	v_mov_b32_e32 v12, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v76, v137
	v_mov_b32_e32 v135, v53
	v_mov_b32_e32 v136, v85
	v_mov_b32_e32 v137, v117
	v_mov_b32_e32 v20, v135
	v_mov_b32_e32 v52, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v135, v61
	v_mov_b32_e32 v136, v93
	v_mov_b32_e32 v137, v125
	v_mov_b32_e32 v28, v135
	v_mov_b32_e32 v60, v136
	v_mov_b32_e32 v92, v137
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW3_BM0_VW3:
	s_mov_b32 s8, 3
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v39
	v_mov_b32_e32 v136, v71
	v_mov_b32_e32 v137, v103
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v38, v136
	v_mov_b32_e32 v70, v137
	v_mov_b32_e32 v135, v47
	v_mov_b32_e32 v136, v79
	v_mov_b32_e32 v137, v111
	v_mov_b32_e32 v14, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v78, v137
	v_mov_b32_e32 v135, v55
	v_mov_b32_e32 v136, v87
	v_mov_b32_e32 v137, v119
	v_mov_b32_e32 v22, v135
	v_mov_b32_e32 v54, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v135, v63
	v_mov_b32_e32 v136, v95
	v_mov_b32_e32 v137, v127
	v_mov_b32_e32 v30, v135
	v_mov_b32_e32 v62, v136
	v_mov_b32_e32 v94, v137
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW4_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v1
	v_mov_b32_e32 v136, v33
	v_mov_b32_e32 v137, v65
	v_mov_b32_e32 v138, v97
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v32, v136
	v_mov_b32_e32 v64, v137
	v_mov_b32_e32 v96, v138
	v_mov_b32_e32 v135, v9
	v_mov_b32_e32 v136, v41
	v_mov_b32_e32 v137, v73
	v_mov_b32_e32 v138, v105
	v_mov_b32_e32 v8, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v72, v137
	v_mov_b32_e32 v104, v138
	v_mov_b32_e32 v135, v17
	v_mov_b32_e32 v136, v49
	v_mov_b32_e32 v137, v81
	v_mov_b32_e32 v138, v113
	v_mov_b32_e32 v16, v135
	v_mov_b32_e32 v48, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v112, v138
	v_mov_b32_e32 v135, v25
	v_mov_b32_e32 v136, v57
	v_mov_b32_e32 v137, v89
	v_mov_b32_e32 v138, v121
	v_mov_b32_e32 v24, v135
	v_mov_b32_e32 v56, v136
	v_mov_b32_e32 v88, v137
	v_mov_b32_e32 v120, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW4_BM0_VW1:
	s_mov_b32 s8, 1
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v3
	v_mov_b32_e32 v136, v35
	v_mov_b32_e32 v137, v67
	v_mov_b32_e32 v138, v99
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v34, v136
	v_mov_b32_e32 v66, v137
	v_mov_b32_e32 v98, v138
	v_mov_b32_e32 v135, v11
	v_mov_b32_e32 v136, v43
	v_mov_b32_e32 v137, v75
	v_mov_b32_e32 v138, v107
	v_mov_b32_e32 v10, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v74, v137
	v_mov_b32_e32 v106, v138
	v_mov_b32_e32 v135, v19
	v_mov_b32_e32 v136, v51
	v_mov_b32_e32 v137, v83
	v_mov_b32_e32 v138, v115
	v_mov_b32_e32 v18, v135
	v_mov_b32_e32 v50, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v114, v138
	v_mov_b32_e32 v135, v27
	v_mov_b32_e32 v136, v59
	v_mov_b32_e32 v137, v91
	v_mov_b32_e32 v138, v123
	v_mov_b32_e32 v26, v135
	v_mov_b32_e32 v58, v136
	v_mov_b32_e32 v90, v137
	v_mov_b32_e32 v122, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW4_BM0_VW2:
	s_mov_b32 s8, 2
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v5
	v_mov_b32_e32 v136, v37
	v_mov_b32_e32 v137, v69
	v_mov_b32_e32 v138, v101
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v36, v136
	v_mov_b32_e32 v68, v137
	v_mov_b32_e32 v100, v138
	v_mov_b32_e32 v135, v13
	v_mov_b32_e32 v136, v45
	v_mov_b32_e32 v137, v77
	v_mov_b32_e32 v138, v109
	v_mov_b32_e32 v12, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v76, v137
	v_mov_b32_e32 v108, v138
	v_mov_b32_e32 v135, v21
	v_mov_b32_e32 v136, v53
	v_mov_b32_e32 v137, v85
	v_mov_b32_e32 v138, v117
	v_mov_b32_e32 v20, v135
	v_mov_b32_e32 v52, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v116, v138
	v_mov_b32_e32 v135, v29
	v_mov_b32_e32 v136, v61
	v_mov_b32_e32 v137, v93
	v_mov_b32_e32 v138, v125
	v_mov_b32_e32 v28, v135
	v_mov_b32_e32 v60, v136
	v_mov_b32_e32 v92, v137
	v_mov_b32_e32 v124, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW4_BM0_VW3:
	s_mov_b32 s8, 3
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v7
	v_mov_b32_e32 v136, v39
	v_mov_b32_e32 v137, v71
	v_mov_b32_e32 v138, v103
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v38, v136
	v_mov_b32_e32 v70, v137
	v_mov_b32_e32 v102, v138
	v_mov_b32_e32 v135, v15
	v_mov_b32_e32 v136, v47
	v_mov_b32_e32 v137, v79
	v_mov_b32_e32 v138, v111
	v_mov_b32_e32 v14, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v78, v137
	v_mov_b32_e32 v110, v138
	v_mov_b32_e32 v135, v23
	v_mov_b32_e32 v136, v55
	v_mov_b32_e32 v137, v87
	v_mov_b32_e32 v138, v119
	v_mov_b32_e32 v22, v135
	v_mov_b32_e32 v54, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v118, v138
	v_mov_b32_e32 v135, v31
	v_mov_b32_e32 v136, v63
	v_mov_b32_e32 v137, v95
	v_mov_b32_e32 v138, v127
	v_mov_b32_e32 v30, v135
	v_mov_b32_e32 v62, v136
	v_mov_b32_e32 v94, v137
	v_mov_b32_e32 v126, v138
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW5_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v96
	v_mov_b32_e32 v136, v1
	v_mov_b32_e32 v137, v33
	v_mov_b32_e32 v138, v65
	v_mov_b32_e32 v139, v97
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v32, v136
	v_mov_b32_e32 v64, v137
	v_mov_b32_e32 v96, v138
	v_mov_b32_e32 v1, v139
	v_mov_b32_e32 v135, v104
	v_mov_b32_e32 v136, v9
	v_mov_b32_e32 v137, v41
	v_mov_b32_e32 v138, v73
	v_mov_b32_e32 v139, v105
	v_mov_b32_e32 v8, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v72, v137
	v_mov_b32_e32 v104, v138
	v_mov_b32_e32 v9, v139
	v_mov_b32_e32 v135, v112
	v_mov_b32_e32 v136, v17
	v_mov_b32_e32 v137, v49
	v_mov_b32_e32 v138, v81
	v_mov_b32_e32 v139, v113
	v_mov_b32_e32 v16, v135
	v_mov_b32_e32 v48, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v112, v138
	v_mov_b32_e32 v17, v139
	v_mov_b32_e32 v135, v120
	v_mov_b32_e32 v136, v25
	v_mov_b32_e32 v137, v57
	v_mov_b32_e32 v138, v89
	v_mov_b32_e32 v139, v121
	v_mov_b32_e32 v24, v135
	v_mov_b32_e32 v56, v136
	v_mov_b32_e32 v88, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v25, v139
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW5_BM0_VW1:
	s_mov_b32 s8, 1
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v98
	v_mov_b32_e32 v136, v3
	v_mov_b32_e32 v137, v35
	v_mov_b32_e32 v138, v67
	v_mov_b32_e32 v139, v99
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v34, v136
	v_mov_b32_e32 v66, v137
	v_mov_b32_e32 v98, v138
	v_mov_b32_e32 v3, v139
	v_mov_b32_e32 v135, v106
	v_mov_b32_e32 v136, v11
	v_mov_b32_e32 v137, v43
	v_mov_b32_e32 v138, v75
	v_mov_b32_e32 v139, v107
	v_mov_b32_e32 v10, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v74, v137
	v_mov_b32_e32 v106, v138
	v_mov_b32_e32 v11, v139
	v_mov_b32_e32 v135, v114
	v_mov_b32_e32 v136, v19
	v_mov_b32_e32 v137, v51
	v_mov_b32_e32 v138, v83
	v_mov_b32_e32 v139, v115
	v_mov_b32_e32 v18, v135
	v_mov_b32_e32 v50, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v114, v138
	v_mov_b32_e32 v19, v139
	v_mov_b32_e32 v135, v122
	v_mov_b32_e32 v136, v27
	v_mov_b32_e32 v137, v59
	v_mov_b32_e32 v138, v91
	v_mov_b32_e32 v139, v123
	v_mov_b32_e32 v26, v135
	v_mov_b32_e32 v58, v136
	v_mov_b32_e32 v90, v137
	v_mov_b32_e32 v122, v138
	v_mov_b32_e32 v27, v139
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW5_BM0_VW2:
	s_mov_b32 s8, 2
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v100
	v_mov_b32_e32 v136, v5
	v_mov_b32_e32 v137, v37
	v_mov_b32_e32 v138, v69
	v_mov_b32_e32 v139, v101
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v36, v136
	v_mov_b32_e32 v68, v137
	v_mov_b32_e32 v100, v138
	v_mov_b32_e32 v5, v139
	v_mov_b32_e32 v135, v108
	v_mov_b32_e32 v136, v13
	v_mov_b32_e32 v137, v45
	v_mov_b32_e32 v138, v77
	v_mov_b32_e32 v139, v109
	v_mov_b32_e32 v12, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v76, v137
	v_mov_b32_e32 v108, v138
	v_mov_b32_e32 v13, v139
	v_mov_b32_e32 v135, v116
	v_mov_b32_e32 v136, v21
	v_mov_b32_e32 v137, v53
	v_mov_b32_e32 v138, v85
	v_mov_b32_e32 v139, v117
	v_mov_b32_e32 v20, v135
	v_mov_b32_e32 v52, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v116, v138
	v_mov_b32_e32 v21, v139
	v_mov_b32_e32 v135, v124
	v_mov_b32_e32 v136, v29
	v_mov_b32_e32 v137, v61
	v_mov_b32_e32 v138, v93
	v_mov_b32_e32 v139, v125
	v_mov_b32_e32 v28, v135
	v_mov_b32_e32 v60, v136
	v_mov_b32_e32 v92, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v29, v139
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW5_BM0_VW3:
	s_mov_b32 s8, 3
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v102
	v_mov_b32_e32 v136, v7
	v_mov_b32_e32 v137, v39
	v_mov_b32_e32 v138, v71
	v_mov_b32_e32 v139, v103
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v38, v136
	v_mov_b32_e32 v70, v137
	v_mov_b32_e32 v102, v138
	v_mov_b32_e32 v7, v139
	v_mov_b32_e32 v135, v110
	v_mov_b32_e32 v136, v15
	v_mov_b32_e32 v137, v47
	v_mov_b32_e32 v138, v79
	v_mov_b32_e32 v139, v111
	v_mov_b32_e32 v14, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v78, v137
	v_mov_b32_e32 v110, v138
	v_mov_b32_e32 v15, v139
	v_mov_b32_e32 v135, v118
	v_mov_b32_e32 v136, v23
	v_mov_b32_e32 v137, v55
	v_mov_b32_e32 v138, v87
	v_mov_b32_e32 v139, v119
	v_mov_b32_e32 v22, v135
	v_mov_b32_e32 v54, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v118, v138
	v_mov_b32_e32 v23, v139
	v_mov_b32_e32 v135, v126
	v_mov_b32_e32 v136, v31
	v_mov_b32_e32 v137, v63
	v_mov_b32_e32 v138, v95
	v_mov_b32_e32 v139, v127
	v_mov_b32_e32 v30, v135
	v_mov_b32_e32 v62, v136
	v_mov_b32_e32 v94, v137
	v_mov_b32_e32 v126, v138
	v_mov_b32_e32 v31, v139
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW6_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v64
	v_mov_b32_e32 v136, v96
	v_mov_b32_e32 v137, v1
	v_mov_b32_e32 v138, v33
	v_mov_b32_e32 v139, v65
	v_mov_b32_e32 v140, v97
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v32, v136
	v_mov_b32_e32 v64, v137
	v_mov_b32_e32 v96, v138
	v_mov_b32_e32 v1, v139
	v_mov_b32_e32 v33, v140
	v_mov_b32_e32 v135, v72
	v_mov_b32_e32 v136, v104
	v_mov_b32_e32 v137, v9
	v_mov_b32_e32 v138, v41
	v_mov_b32_e32 v139, v73
	v_mov_b32_e32 v140, v105
	v_mov_b32_e32 v8, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v72, v137
	v_mov_b32_e32 v104, v138
	v_mov_b32_e32 v9, v139
	v_mov_b32_e32 v41, v140
	v_mov_b32_e32 v135, v80
	v_mov_b32_e32 v136, v112
	v_mov_b32_e32 v137, v17
	v_mov_b32_e32 v138, v49
	v_mov_b32_e32 v139, v81
	v_mov_b32_e32 v140, v113
	v_mov_b32_e32 v16, v135
	v_mov_b32_e32 v48, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v112, v138
	v_mov_b32_e32 v17, v139
	v_mov_b32_e32 v49, v140
	v_mov_b32_e32 v135, v88
	v_mov_b32_e32 v136, v120
	v_mov_b32_e32 v137, v25
	v_mov_b32_e32 v138, v57
	v_mov_b32_e32 v139, v89
	v_mov_b32_e32 v140, v121
	v_mov_b32_e32 v24, v135
	v_mov_b32_e32 v56, v136
	v_mov_b32_e32 v88, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v25, v139
	v_mov_b32_e32 v57, v140
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW6_BM0_VW1:
	s_mov_b32 s8, 1
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v66
	v_mov_b32_e32 v136, v98
	v_mov_b32_e32 v137, v3
	v_mov_b32_e32 v138, v35
	v_mov_b32_e32 v139, v67
	v_mov_b32_e32 v140, v99
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v34, v136
	v_mov_b32_e32 v66, v137
	v_mov_b32_e32 v98, v138
	v_mov_b32_e32 v3, v139
	v_mov_b32_e32 v35, v140
	v_mov_b32_e32 v135, v74
	v_mov_b32_e32 v136, v106
	v_mov_b32_e32 v137, v11
	v_mov_b32_e32 v138, v43
	v_mov_b32_e32 v139, v75
	v_mov_b32_e32 v140, v107
	v_mov_b32_e32 v10, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v74, v137
	v_mov_b32_e32 v106, v138
	v_mov_b32_e32 v11, v139
	v_mov_b32_e32 v43, v140
	v_mov_b32_e32 v135, v82
	v_mov_b32_e32 v136, v114
	v_mov_b32_e32 v137, v19
	v_mov_b32_e32 v138, v51
	v_mov_b32_e32 v139, v83
	v_mov_b32_e32 v140, v115
	v_mov_b32_e32 v18, v135
	v_mov_b32_e32 v50, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v114, v138
	v_mov_b32_e32 v19, v139
	v_mov_b32_e32 v51, v140
	v_mov_b32_e32 v135, v90
	v_mov_b32_e32 v136, v122
	v_mov_b32_e32 v137, v27
	v_mov_b32_e32 v138, v59
	v_mov_b32_e32 v139, v91
	v_mov_b32_e32 v140, v123
	v_mov_b32_e32 v26, v135
	v_mov_b32_e32 v58, v136
	v_mov_b32_e32 v90, v137
	v_mov_b32_e32 v122, v138
	v_mov_b32_e32 v27, v139
	v_mov_b32_e32 v59, v140
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW6_BM0_VW2:
	s_mov_b32 s8, 2
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v68
	v_mov_b32_e32 v136, v100
	v_mov_b32_e32 v137, v5
	v_mov_b32_e32 v138, v37
	v_mov_b32_e32 v139, v69
	v_mov_b32_e32 v140, v101
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v36, v136
	v_mov_b32_e32 v68, v137
	v_mov_b32_e32 v100, v138
	v_mov_b32_e32 v5, v139
	v_mov_b32_e32 v37, v140
	v_mov_b32_e32 v135, v76
	v_mov_b32_e32 v136, v108
	v_mov_b32_e32 v137, v13
	v_mov_b32_e32 v138, v45
	v_mov_b32_e32 v139, v77
	v_mov_b32_e32 v140, v109
	v_mov_b32_e32 v12, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v76, v137
	v_mov_b32_e32 v108, v138
	v_mov_b32_e32 v13, v139
	v_mov_b32_e32 v45, v140
	v_mov_b32_e32 v135, v84
	v_mov_b32_e32 v136, v116
	v_mov_b32_e32 v137, v21
	v_mov_b32_e32 v138, v53
	v_mov_b32_e32 v139, v85
	v_mov_b32_e32 v140, v117
	v_mov_b32_e32 v20, v135
	v_mov_b32_e32 v52, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v116, v138
	v_mov_b32_e32 v21, v139
	v_mov_b32_e32 v53, v140
	v_mov_b32_e32 v135, v92
	v_mov_b32_e32 v136, v124
	v_mov_b32_e32 v137, v29
	v_mov_b32_e32 v138, v61
	v_mov_b32_e32 v139, v93
	v_mov_b32_e32 v140, v125
	v_mov_b32_e32 v28, v135
	v_mov_b32_e32 v60, v136
	v_mov_b32_e32 v92, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v29, v139
	v_mov_b32_e32 v61, v140
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW6_BM0_VW3:
	s_mov_b32 s8, 3
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v70
	v_mov_b32_e32 v136, v102
	v_mov_b32_e32 v137, v7
	v_mov_b32_e32 v138, v39
	v_mov_b32_e32 v139, v71
	v_mov_b32_e32 v140, v103
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v38, v136
	v_mov_b32_e32 v70, v137
	v_mov_b32_e32 v102, v138
	v_mov_b32_e32 v7, v139
	v_mov_b32_e32 v39, v140
	v_mov_b32_e32 v135, v78
	v_mov_b32_e32 v136, v110
	v_mov_b32_e32 v137, v15
	v_mov_b32_e32 v138, v47
	v_mov_b32_e32 v139, v79
	v_mov_b32_e32 v140, v111
	v_mov_b32_e32 v14, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v78, v137
	v_mov_b32_e32 v110, v138
	v_mov_b32_e32 v15, v139
	v_mov_b32_e32 v47, v140
	v_mov_b32_e32 v135, v86
	v_mov_b32_e32 v136, v118
	v_mov_b32_e32 v137, v23
	v_mov_b32_e32 v138, v55
	v_mov_b32_e32 v139, v87
	v_mov_b32_e32 v140, v119
	v_mov_b32_e32 v22, v135
	v_mov_b32_e32 v54, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v118, v138
	v_mov_b32_e32 v23, v139
	v_mov_b32_e32 v55, v140
	v_mov_b32_e32 v135, v94
	v_mov_b32_e32 v136, v126
	v_mov_b32_e32 v137, v31
	v_mov_b32_e32 v138, v63
	v_mov_b32_e32 v139, v95
	v_mov_b32_e32 v140, v127
	v_mov_b32_e32 v30, v135
	v_mov_b32_e32 v62, v136
	v_mov_b32_e32 v94, v137
	v_mov_b32_e32 v126, v138
	v_mov_b32_e32 v31, v139
	v_mov_b32_e32 v63, v140
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW7_BM0_VW0:
	s_mov_b32 s8, 0
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v32
	v_mov_b32_e32 v136, v64
	v_mov_b32_e32 v137, v96
	v_mov_b32_e32 v138, v1
	v_mov_b32_e32 v139, v33
	v_mov_b32_e32 v140, v65
	v_mov_b32_e32 v141, v97
	v_mov_b32_e32 v0, v135
	v_mov_b32_e32 v32, v136
	v_mov_b32_e32 v64, v137
	v_mov_b32_e32 v96, v138
	v_mov_b32_e32 v1, v139
	v_mov_b32_e32 v33, v140
	v_mov_b32_e32 v65, v141
	v_mov_b32_e32 v135, v40
	v_mov_b32_e32 v136, v72
	v_mov_b32_e32 v137, v104
	v_mov_b32_e32 v138, v9
	v_mov_b32_e32 v139, v41
	v_mov_b32_e32 v140, v73
	v_mov_b32_e32 v141, v105
	v_mov_b32_e32 v8, v135
	v_mov_b32_e32 v40, v136
	v_mov_b32_e32 v72, v137
	v_mov_b32_e32 v104, v138
	v_mov_b32_e32 v9, v139
	v_mov_b32_e32 v41, v140
	v_mov_b32_e32 v73, v141
	v_mov_b32_e32 v135, v48
	v_mov_b32_e32 v136, v80
	v_mov_b32_e32 v137, v112
	v_mov_b32_e32 v138, v17
	v_mov_b32_e32 v139, v49
	v_mov_b32_e32 v140, v81
	v_mov_b32_e32 v141, v113
	v_mov_b32_e32 v16, v135
	v_mov_b32_e32 v48, v136
	v_mov_b32_e32 v80, v137
	v_mov_b32_e32 v112, v138
	v_mov_b32_e32 v17, v139
	v_mov_b32_e32 v49, v140
	v_mov_b32_e32 v81, v141
	v_mov_b32_e32 v135, v56
	v_mov_b32_e32 v136, v88
	v_mov_b32_e32 v137, v120
	v_mov_b32_e32 v138, v25
	v_mov_b32_e32 v139, v57
	v_mov_b32_e32 v140, v89
	v_mov_b32_e32 v141, v121
	v_mov_b32_e32 v24, v135
	v_mov_b32_e32 v56, v136
	v_mov_b32_e32 v88, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v25, v139
	v_mov_b32_e32 v57, v140
	v_mov_b32_e32 v89, v141
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW7_BM0_VW1:
	s_mov_b32 s8, 1
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v34
	v_mov_b32_e32 v136, v66
	v_mov_b32_e32 v137, v98
	v_mov_b32_e32 v138, v3
	v_mov_b32_e32 v139, v35
	v_mov_b32_e32 v140, v67
	v_mov_b32_e32 v141, v99
	v_mov_b32_e32 v2, v135
	v_mov_b32_e32 v34, v136
	v_mov_b32_e32 v66, v137
	v_mov_b32_e32 v98, v138
	v_mov_b32_e32 v3, v139
	v_mov_b32_e32 v35, v140
	v_mov_b32_e32 v67, v141
	v_mov_b32_e32 v135, v42
	v_mov_b32_e32 v136, v74
	v_mov_b32_e32 v137, v106
	v_mov_b32_e32 v138, v11
	v_mov_b32_e32 v139, v43
	v_mov_b32_e32 v140, v75
	v_mov_b32_e32 v141, v107
	v_mov_b32_e32 v10, v135
	v_mov_b32_e32 v42, v136
	v_mov_b32_e32 v74, v137
	v_mov_b32_e32 v106, v138
	v_mov_b32_e32 v11, v139
	v_mov_b32_e32 v43, v140
	v_mov_b32_e32 v75, v141
	v_mov_b32_e32 v135, v50
	v_mov_b32_e32 v136, v82
	v_mov_b32_e32 v137, v114
	v_mov_b32_e32 v138, v19
	v_mov_b32_e32 v139, v51
	v_mov_b32_e32 v140, v83
	v_mov_b32_e32 v141, v115
	v_mov_b32_e32 v18, v135
	v_mov_b32_e32 v50, v136
	v_mov_b32_e32 v82, v137
	v_mov_b32_e32 v114, v138
	v_mov_b32_e32 v19, v139
	v_mov_b32_e32 v51, v140
	v_mov_b32_e32 v83, v141
	v_mov_b32_e32 v135, v58
	v_mov_b32_e32 v136, v90
	v_mov_b32_e32 v137, v122
	v_mov_b32_e32 v138, v27
	v_mov_b32_e32 v139, v59
	v_mov_b32_e32 v140, v91
	v_mov_b32_e32 v141, v123
	v_mov_b32_e32 v26, v135
	v_mov_b32_e32 v58, v136
	v_mov_b32_e32 v90, v137
	v_mov_b32_e32 v122, v138
	v_mov_b32_e32 v27, v139
	v_mov_b32_e32 v59, v140
	v_mov_b32_e32 v91, v141
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW7_BM0_VW2:
	s_mov_b32 s8, 2
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v36
	v_mov_b32_e32 v136, v68
	v_mov_b32_e32 v137, v100
	v_mov_b32_e32 v138, v5
	v_mov_b32_e32 v139, v37
	v_mov_b32_e32 v140, v69
	v_mov_b32_e32 v141, v101
	v_mov_b32_e32 v4, v135
	v_mov_b32_e32 v36, v136
	v_mov_b32_e32 v68, v137
	v_mov_b32_e32 v100, v138
	v_mov_b32_e32 v5, v139
	v_mov_b32_e32 v37, v140
	v_mov_b32_e32 v69, v141
	v_mov_b32_e32 v135, v44
	v_mov_b32_e32 v136, v76
	v_mov_b32_e32 v137, v108
	v_mov_b32_e32 v138, v13
	v_mov_b32_e32 v139, v45
	v_mov_b32_e32 v140, v77
	v_mov_b32_e32 v141, v109
	v_mov_b32_e32 v12, v135
	v_mov_b32_e32 v44, v136
	v_mov_b32_e32 v76, v137
	v_mov_b32_e32 v108, v138
	v_mov_b32_e32 v13, v139
	v_mov_b32_e32 v45, v140
	v_mov_b32_e32 v77, v141
	v_mov_b32_e32 v135, v52
	v_mov_b32_e32 v136, v84
	v_mov_b32_e32 v137, v116
	v_mov_b32_e32 v138, v21
	v_mov_b32_e32 v139, v53
	v_mov_b32_e32 v140, v85
	v_mov_b32_e32 v141, v117
	v_mov_b32_e32 v20, v135
	v_mov_b32_e32 v52, v136
	v_mov_b32_e32 v84, v137
	v_mov_b32_e32 v116, v138
	v_mov_b32_e32 v21, v139
	v_mov_b32_e32 v53, v140
	v_mov_b32_e32 v85, v141
	v_mov_b32_e32 v135, v60
	v_mov_b32_e32 v136, v92
	v_mov_b32_e32 v137, v124
	v_mov_b32_e32 v138, v29
	v_mov_b32_e32 v139, v61
	v_mov_b32_e32 v140, v93
	v_mov_b32_e32 v141, v125
	v_mov_b32_e32 v28, v135
	v_mov_b32_e32 v60, v136
	v_mov_b32_e32 v92, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v29, v139
	v_mov_b32_e32 v61, v140
	v_mov_b32_e32 v93, v141
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW7_BM0_VW3:
	s_mov_b32 s8, 3
	v_cmp_eq_u32_e64 s8, v134, s8
	s_mov_b32 exec_lo, s8
	v_and_b32_e32 v128, 31, v218
	v_lshlrev_b32_e32 v128, 2, v128
	v_mov_b32_e32 v135, v38
	v_mov_b32_e32 v136, v70
	v_mov_b32_e32 v137, v102
	v_mov_b32_e32 v138, v7
	v_mov_b32_e32 v139, v39
	v_mov_b32_e32 v140, v71
	v_mov_b32_e32 v141, v103
	v_mov_b32_e32 v6, v135
	v_mov_b32_e32 v38, v136
	v_mov_b32_e32 v70, v137
	v_mov_b32_e32 v102, v138
	v_mov_b32_e32 v7, v139
	v_mov_b32_e32 v39, v140
	v_mov_b32_e32 v71, v141
	v_mov_b32_e32 v135, v46
	v_mov_b32_e32 v136, v78
	v_mov_b32_e32 v137, v110
	v_mov_b32_e32 v138, v15
	v_mov_b32_e32 v139, v47
	v_mov_b32_e32 v140, v79
	v_mov_b32_e32 v141, v111
	v_mov_b32_e32 v14, v135
	v_mov_b32_e32 v46, v136
	v_mov_b32_e32 v78, v137
	v_mov_b32_e32 v110, v138
	v_mov_b32_e32 v15, v139
	v_mov_b32_e32 v47, v140
	v_mov_b32_e32 v79, v141
	v_mov_b32_e32 v135, v54
	v_mov_b32_e32 v136, v86
	v_mov_b32_e32 v137, v118
	v_mov_b32_e32 v138, v23
	v_mov_b32_e32 v139, v55
	v_mov_b32_e32 v140, v87
	v_mov_b32_e32 v141, v119
	v_mov_b32_e32 v22, v135
	v_mov_b32_e32 v54, v136
	v_mov_b32_e32 v86, v137
	v_mov_b32_e32 v118, v138
	v_mov_b32_e32 v23, v139
	v_mov_b32_e32 v55, v140
	v_mov_b32_e32 v87, v141
	v_mov_b32_e32 v135, v62
	v_mov_b32_e32 v136, v94
	v_mov_b32_e32 v137, v126
	v_mov_b32_e32 v138, v31
	v_mov_b32_e32 v139, v63
	v_mov_b32_e32 v140, v95
	v_mov_b32_e32 v141, v127
	v_mov_b32_e32 v30, v135
	v_mov_b32_e32 v62, v136
	v_mov_b32_e32 v94, v137
	v_mov_b32_e32 v126, v138
	v_mov_b32_e32 v31, v139
	v_mov_b32_e32 v63, v140
	v_mov_b32_e32 v95, v141
	s_mov_b32 s8, -1
	s_or_saveexec_b32 vcc_lo, s8
	s_branch label_ShiftVectorComponents1_GLVW0
label_ShiftVectorComponents1_GLVW0:
	v_lshrrev_b32_e32 v132, 5, v218
	v_lshrrev_b32_e32 v133, 1, v132
	v_mul_lo_u32 v133, 16, v133
	v_and_b32_e32 v129, 31, v218
	v_lshrrev_b32_e32 v129, 4, v129
	v_lshlrev_b32_e32 v129, 3, v129
	v_add_lshl_u32 v129, v133, v129, 2
	v_mul_lo_u32 v130, v129, s38
	v_mul_lo_u32 v131, v129, s36
	v_and_b32_e32 v128, 1, v132
	v_mul_lo_u32 v128, 16, v128
	v_and_b32_e32 v133, 15, v218
	v_add_lshl_u32 v128, v133, v128, 2
	s_mul_i32 s8, 0x80, s2
	v_add_nc_u32_e32 v128, s8, v128
	s_mul_i32 s8, 0x80, s3
	v_add_nc_u32_e32 v129, s8, v129
	s_wait_kmcnt 0x0
	s_and_b32 s8, s46, 0x3fff
	s_cmp_eq_u32 s8, 1
	s_cbranch_scc1 label_GSU_5
	s_and_b32 s30, 0x7f, s24
	s_add_co_u32 s31, -1, s14
	s_cmp_ge_u32 s2, s31
	s_cselect_b32 s30, s30, 0
	s_mov_b32 s8, 0
	s_cmp_gt_u32 s30, s8
	s_cbranch_scc1 label_GW_B0_E1_M
	s_and_b32 s30, 0x7f, s25
	s_add_co_u32 s31, -1, s15
	s_cmp_ge_u32 s3, s31
	s_cselect_b32 s30, s30, 0
	s_mov_b32 s8, 0
	s_cmp_gt_u32 s30, s8
	s_cbranch_scc1 label_GW_B0_E1_N
label_GW_B0_E0:
	v_add_lshl_u32 v135, v131, v128, 2
	v_mov_b32_e32 v140, v0
	v_mov_b32_e32 v141, v8
	v_mov_b32_e32 v142, v16
	v_mov_b32_e32 v143, v24
	v_mov_b32_e32 v144, v32
	v_mov_b32_e32 v145, v40
	v_mov_b32_e32 v146, v48
	v_mov_b32_e32 v147, v56
	v_mov_b32_e32 v148, v64
	v_mov_b32_e32 v149, v72
	v_mov_b32_e32 v150, v80
	v_mov_b32_e32 v151, v88
	v_mov_b32_e32 v152, v96
	v_mov_b32_e32 v153, v104
	v_mov_b32_e32 v154, v112
	v_mov_b32_e32 v155, v120
	v_mov_b32_e32 v156, v1
	v_mov_b32_e32 v157, v9
	v_mov_b32_e32 v158, v17
	v_mov_b32_e32 v159, v25
	v_mov_b32_e32 v160, v33
	v_mov_b32_e32 v161, v41
	v_mov_b32_e32 v162, v49
	v_mov_b32_e32 v163, v57
	v_mov_b32_e32 v164, v65
	v_mov_b32_e32 v165, v73
	v_mov_b32_e32 v166, v81
	v_mov_b32_e32 v167, v89
	v_mov_b32_e32 v168, v97
	v_mov_b32_e32 v169, v105
	v_mov_b32_e32 v170, v113
	v_mov_b32_e32 v171, v121
	v_mov_b32_e32 v172, v2
	v_mov_b32_e32 v173, v10
	v_mov_b32_e32 v174, v18
	v_mov_b32_e32 v175, v26
	v_mov_b32_e32 v176, v34
	v_mov_b32_e32 v177, v42
	v_mov_b32_e32 v178, v50
	v_mov_b32_e32 v179, v58
	v_mov_b32_e32 v180, v66
	v_mov_b32_e32 v181, v74
	v_mov_b32_e32 v182, v82
	v_mov_b32_e32 v183, v90
	v_mov_b32_e32 v184, v98
	v_mov_b32_e32 v185, v106
	v_mov_b32_e32 v186, v114
	v_mov_b32_e32 v187, v122
	v_mov_b32_e32 v188, v3
	v_mov_b32_e32 v189, v11
	v_mov_b32_e32 v190, v19
	v_mov_b32_e32 v191, v27
	v_mov_b32_e32 v192, v35
	v_mov_b32_e32 v193, v43
	v_mov_b32_e32 v194, v51
	v_mov_b32_e32 v195, v59
	v_mov_b32_e32 v196, v67
	v_mov_b32_e32 v197, v75
	v_mov_b32_e32 v198, v83
	v_mov_b32_e32 v199, v91
	v_mov_b32_e32 v200, v99
	v_mov_b32_e32 v201, v107
	v_mov_b32_e32 v202, v115
	v_mov_b32_e32 v203, v123
	v_mov_b32_e32 v204, v4
	v_mov_b32_e32 v205, v12
	v_mov_b32_e32 v206, v20
	v_mov_b32_e32 v207, v28
	v_mov_b32_e32 v208, v36
	v_mov_b32_e32 v209, v44
	v_mov_b32_e32 v210, v52
	v_mov_b32_e32 v211, v60
	v_mov_b32_e32 v212, v68
	v_mov_b32_e32 v213, v76
	v_mov_b32_e32 v214, v84
	v_mov_b32_e32 v215, v92
	v_mov_b32_e32 v220, v100
	v_mov_b32_e32 v221, v108
	v_mov_b32_e32 v222, v116
	v_mov_b32_e32 v223, v124
	v_mov_b32_e32 v224, v5
	v_mov_b32_e32 v225, v13
	v_mov_b32_e32 v226, v21
	v_mov_b32_e32 v227, v29
	v_mov_b32_e32 v228, v37
	v_mov_b32_e32 v229, v45
	v_mov_b32_e32 v230, v53
	v_mov_b32_e32 v231, v61
	v_mov_b32_e32 v232, v69
	v_mov_b32_e32 v233, v77
	v_mov_b32_e32 v234, v85
	v_mov_b32_e32 v235, v93
	v_mov_b32_e32 v236, v101
	v_mov_b32_e32 v237, v109
	v_mov_b32_e32 v238, v117
	v_mov_b32_e32 v239, v125
	v_mov_b32_e32 v240, v6
	v_mov_b32_e32 v241, v14
	v_mov_b32_e32 v242, v22
	v_mov_b32_e32 v243, v30
	v_mov_b32_e32 v244, v38
	v_mov_b32_e32 v245, v46
	v_mov_b32_e32 v246, v54
	v_mov_b32_e32 v247, v62
	v_mov_b32_e32 v248, v70
	v_mov_b32_e32 v249, v78
	v_mov_b32_e32 v250, v86
	v_mov_b32_e32 v251, v94
	v_mov_b32_e32 v252, v102
	v_mov_b32_e32 v253, v110
	v_mov_b32_e32 v254, v118
	v_mov_b32_e32 v255, v126
	buffer_store_b128 v[140:143], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[164:167], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[168:171], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[172:175], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[176:179], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[180:183], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[184:187], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[188:191], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[192:195], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[196:199], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[200:203], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[204:207], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[208:211], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[212:215], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[220:223], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[224:227], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[228:231], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[232:235], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[236:239], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[240:243], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[244:247], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[248:251], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[252:255], v135, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v140, v7
	v_mov_b32_e32 v141, v15
	v_mov_b32_e32 v142, v23
	v_mov_b32_e32 v143, v31
	v_mov_b32_e32 v144, v39
	v_mov_b32_e32 v145, v47
	v_mov_b32_e32 v146, v55
	v_mov_b32_e32 v147, v63
	v_mov_b32_e32 v148, v71
	v_mov_b32_e32 v149, v79
	v_mov_b32_e32 v150, v87
	v_mov_b32_e32 v151, v95
	v_mov_b32_e32 v152, v103
	v_mov_b32_e32 v153, v111
	v_mov_b32_e32 v154, v119
	v_mov_b32_e32 v155, v127
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v135, s[16:19], null offen
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v135, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End
label_GW_B0_E1_N:
	v_mov_b32_e32 v134, 0x80000000
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v135, v131, v128, 2
	v_cndmask_b32_e64 v135, v134, v135, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v216, v131, v128, 2
	v_cndmask_b32_e64 v216, v134, v216, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v217, v131, v128, 2
	v_cndmask_b32_e64 v217, v134, v217, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v219, v131, v128, 2
	v_cndmask_b32_e64 v219, v134, v219, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v228, v131, v128, 2
	v_cndmask_b32_e64 v228, v134, v228, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v229, v131, v128, 2
	v_cndmask_b32_e64 v229, v134, v229, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v230, v131, v128, 2
	v_cndmask_b32_e64 v230, v134, v230, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v231, v131, v128, 2
	v_cndmask_b32_e64 v231, v134, v231, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v232, v131, v128, 2
	v_cndmask_b32_e64 v232, v134, v232, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v233, v131, v128, 2
	v_cndmask_b32_e64 v233, v134, v233, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v234, v131, v128, 2
	v_cndmask_b32_e64 v234, v134, v234, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v235, v131, v128, 2
	v_cndmask_b32_e64 v235, v134, v235, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v236, v131, v128, 2
	v_cndmask_b32_e64 v236, v134, v236, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v237, v131, v128, 2
	v_cndmask_b32_e64 v237, v134, v237, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v238, v131, v128, 2
	v_cndmask_b32_e64 v238, v134, v238, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v239, v131, v128, 2
	v_cndmask_b32_e64 v239, v134, v239, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v240, v131, v128, 2
	v_cndmask_b32_e64 v240, v134, v240, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v241, v131, v128, 2
	v_cndmask_b32_e64 v241, v134, v241, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v242, v131, v128, 2
	v_cndmask_b32_e64 v242, v134, v242, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v243, v131, v128, 2
	v_cndmask_b32_e64 v243, v134, v243, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v244, v131, v128, 2
	v_cndmask_b32_e64 v244, v134, v244, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v245, v131, v128, 2
	v_cndmask_b32_e64 v245, v134, v245, s32
	v_mov_b32_e32 v136, v0
	v_mov_b32_e32 v137, v8
	v_mov_b32_e32 v138, v16
	v_mov_b32_e32 v139, v24
	v_mov_b32_e32 v140, v32
	v_mov_b32_e32 v141, v40
	v_mov_b32_e32 v142, v48
	v_mov_b32_e32 v143, v56
	v_mov_b32_e32 v144, v64
	v_mov_b32_e32 v145, v72
	v_mov_b32_e32 v146, v80
	v_mov_b32_e32 v147, v88
	v_mov_b32_e32 v148, v96
	v_mov_b32_e32 v149, v104
	v_mov_b32_e32 v150, v112
	v_mov_b32_e32 v151, v120
	v_mov_b32_e32 v152, v1
	v_mov_b32_e32 v153, v9
	v_mov_b32_e32 v154, v17
	v_mov_b32_e32 v155, v25
	v_mov_b32_e32 v156, v33
	v_mov_b32_e32 v157, v41
	v_mov_b32_e32 v158, v49
	v_mov_b32_e32 v159, v57
	v_mov_b32_e32 v160, v65
	v_mov_b32_e32 v161, v73
	v_mov_b32_e32 v162, v81
	v_mov_b32_e32 v163, v89
	v_mov_b32_e32 v164, v97
	v_mov_b32_e32 v165, v105
	v_mov_b32_e32 v166, v113
	v_mov_b32_e32 v167, v121
	v_mov_b32_e32 v168, v2
	v_mov_b32_e32 v169, v10
	v_mov_b32_e32 v170, v18
	v_mov_b32_e32 v171, v26
	v_mov_b32_e32 v172, v34
	v_mov_b32_e32 v173, v42
	v_mov_b32_e32 v174, v50
	v_mov_b32_e32 v175, v58
	v_mov_b32_e32 v176, v66
	v_mov_b32_e32 v177, v74
	v_mov_b32_e32 v178, v82
	v_mov_b32_e32 v179, v90
	v_mov_b32_e32 v180, v98
	v_mov_b32_e32 v181, v106
	v_mov_b32_e32 v182, v114
	v_mov_b32_e32 v183, v122
	v_mov_b32_e32 v184, v3
	v_mov_b32_e32 v185, v11
	v_mov_b32_e32 v186, v19
	v_mov_b32_e32 v187, v27
	v_mov_b32_e32 v188, v35
	v_mov_b32_e32 v189, v43
	v_mov_b32_e32 v190, v51
	v_mov_b32_e32 v191, v59
	v_mov_b32_e32 v192, v67
	v_mov_b32_e32 v193, v75
	v_mov_b32_e32 v194, v83
	v_mov_b32_e32 v195, v91
	v_mov_b32_e32 v196, v99
	v_mov_b32_e32 v197, v107
	v_mov_b32_e32 v198, v115
	v_mov_b32_e32 v199, v123
	v_mov_b32_e32 v200, v4
	v_mov_b32_e32 v201, v12
	v_mov_b32_e32 v202, v20
	v_mov_b32_e32 v203, v28
	v_mov_b32_e32 v204, v36
	v_mov_b32_e32 v205, v44
	v_mov_b32_e32 v206, v52
	v_mov_b32_e32 v207, v60
	v_mov_b32_e32 v208, v68
	v_mov_b32_e32 v209, v76
	v_mov_b32_e32 v210, v84
	v_mov_b32_e32 v211, v92
	v_mov_b32_e32 v212, v100
	v_mov_b32_e32 v213, v108
	v_mov_b32_e32 v214, v116
	v_mov_b32_e32 v215, v124
	v_mov_b32_e32 v220, v5
	v_mov_b32_e32 v221, v13
	v_mov_b32_e32 v222, v21
	v_mov_b32_e32 v223, v29
	v_mov_b32_e32 v224, v37
	v_mov_b32_e32 v225, v45
	v_mov_b32_e32 v226, v53
	v_mov_b32_e32 v227, v61
	buffer_store_b128 v[136:139], v135, s[16:19], null offen
	buffer_store_b128 v[140:143], v216, s[16:19], null offen
	buffer_store_b128 v[144:147], v217, s[16:19], null offen
	buffer_store_b128 v[148:151], v219, s[16:19], null offen
	buffer_store_b128 v[152:155], v228, s[16:19], null offen
	buffer_store_b128 v[156:159], v229, s[16:19], null offen
	buffer_store_b128 v[160:163], v230, s[16:19], null offen
	buffer_store_b128 v[164:167], v231, s[16:19], null offen
	buffer_store_b128 v[168:171], v232, s[16:19], null offen
	buffer_store_b128 v[172:175], v233, s[16:19], null offen
	buffer_store_b128 v[176:179], v234, s[16:19], null offen
	buffer_store_b128 v[180:183], v235, s[16:19], null offen
	buffer_store_b128 v[184:187], v236, s[16:19], null offen
	buffer_store_b128 v[188:191], v237, s[16:19], null offen
	buffer_store_b128 v[192:195], v238, s[16:19], null offen
	buffer_store_b128 v[196:199], v239, s[16:19], null offen
	buffer_store_b128 v[200:203], v240, s[16:19], null offen
	buffer_store_b128 v[204:207], v241, s[16:19], null offen
	buffer_store_b128 v[208:211], v242, s[16:19], null offen
	buffer_store_b128 v[212:215], v243, s[16:19], null offen
	buffer_store_b128 v[220:223], v244, s[16:19], null offen
	buffer_store_b128 v[224:227], v245, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v134, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v135, v131, v128, 2
	v_cndmask_b32_e64 v135, v134, v135, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v176, v131, v128, 2
	v_cndmask_b32_e64 v176, v134, v176, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v177, v131, v128, 2
	v_cndmask_b32_e64 v177, v134, v177, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v178, v131, v128, 2
	v_cndmask_b32_e64 v178, v134, v178, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v134, v179, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v180, v131, v128, 2
	v_cndmask_b32_e64 v180, v134, v180, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v181, v131, v128, 2
	v_cndmask_b32_e64 v181, v134, v181, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v182, v131, v128, 2
	v_cndmask_b32_e64 v182, v134, v182, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v134, v183, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v184, v131, v128, 2
	v_cndmask_b32_e64 v184, v134, v184, s32
	v_mov_b32_e32 v136, v69
	v_mov_b32_e32 v137, v77
	v_mov_b32_e32 v138, v85
	v_mov_b32_e32 v139, v93
	v_mov_b32_e32 v140, v101
	v_mov_b32_e32 v141, v109
	v_mov_b32_e32 v142, v117
	v_mov_b32_e32 v143, v125
	v_mov_b32_e32 v144, v6
	v_mov_b32_e32 v145, v14
	v_mov_b32_e32 v146, v22
	v_mov_b32_e32 v147, v30
	v_mov_b32_e32 v148, v38
	v_mov_b32_e32 v149, v46
	v_mov_b32_e32 v150, v54
	v_mov_b32_e32 v151, v62
	v_mov_b32_e32 v152, v70
	v_mov_b32_e32 v153, v78
	v_mov_b32_e32 v154, v86
	v_mov_b32_e32 v155, v94
	v_mov_b32_e32 v156, v102
	v_mov_b32_e32 v157, v110
	v_mov_b32_e32 v158, v118
	v_mov_b32_e32 v159, v126
	v_mov_b32_e32 v160, v7
	v_mov_b32_e32 v161, v15
	v_mov_b32_e32 v162, v23
	v_mov_b32_e32 v163, v31
	v_mov_b32_e32 v164, v39
	v_mov_b32_e32 v165, v47
	v_mov_b32_e32 v166, v55
	v_mov_b32_e32 v167, v63
	v_mov_b32_e32 v168, v71
	v_mov_b32_e32 v169, v79
	v_mov_b32_e32 v170, v87
	v_mov_b32_e32 v171, v95
	v_mov_b32_e32 v172, v103
	v_mov_b32_e32 v173, v111
	v_mov_b32_e32 v174, v119
	v_mov_b32_e32 v175, v127
	buffer_store_b128 v[136:139], v135, s[16:19], null offen
	buffer_store_b128 v[140:143], v176, s[16:19], null offen
	buffer_store_b128 v[144:147], v177, s[16:19], null offen
	buffer_store_b128 v[148:151], v178, s[16:19], null offen
	buffer_store_b128 v[152:155], v179, s[16:19], null offen
	buffer_store_b128 v[156:159], v180, s[16:19], null offen
	buffer_store_b128 v[160:163], v181, s[16:19], null offen
	buffer_store_b128 v[164:167], v182, s[16:19], null offen
	buffer_store_b128 v[168:171], v183, s[16:19], null offen
	buffer_store_b128 v[172:175], v184, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End
label_GW_B0_E1_M:
	v_mov_b32_e32 v134, 0x80000000
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v193, v131, v128, 2
	v_cndmask_b32_e64 v193, v134, v193, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v194, v131, v132, 2
	v_cndmask_b32_e64 v194, v134, v194, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v195, v131, v132, 2
	v_cndmask_b32_e64 v195, v134, v195, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v196, v131, v132, 2
	v_cndmask_b32_e64 v196, v134, v196, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v197, v131, v128, 2
	v_cndmask_b32_e64 v197, v134, v197, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v198, v131, v132, 2
	v_cndmask_b32_e64 v198, v134, v198, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v199, v131, v132, 2
	v_cndmask_b32_e64 v199, v134, v199, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v200, v131, v132, 2
	v_cndmask_b32_e64 v200, v134, v200, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v201, v131, v128, 2
	v_cndmask_b32_e64 v201, v134, v201, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v202, v131, v132, 2
	v_cndmask_b32_e64 v202, v134, v202, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v134, v203, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v204, v131, v132, 2
	v_cndmask_b32_e64 v204, v134, v204, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v205, v131, v128, 2
	v_cndmask_b32_e64 v205, v134, v205, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v206, v131, v132, 2
	v_cndmask_b32_e64 v206, v134, v206, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v207, v131, v132, 2
	v_cndmask_b32_e64 v207, v134, v207, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v208, v131, v132, 2
	v_cndmask_b32_e64 v208, v134, v208, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v209, v131, v128, 2
	v_cndmask_b32_e64 v209, v134, v209, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v210, v131, v132, 2
	v_cndmask_b32_e64 v210, v134, v210, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v211, v131, v132, 2
	v_cndmask_b32_e64 v211, v134, v211, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v212, v131, v132, 2
	v_cndmask_b32_e64 v212, v134, v212, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v213, v131, v128, 2
	v_cndmask_b32_e64 v213, v134, v213, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v214, v131, v132, 2
	v_cndmask_b32_e64 v214, v134, v214, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v134, v215, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v216, v131, v132, 2
	v_cndmask_b32_e64 v216, v134, v216, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v217, v131, v128, 2
	v_cndmask_b32_e64 v217, v134, v217, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v219, v131, v132, 2
	v_cndmask_b32_e64 v219, v134, v219, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v220, v131, v132, 2
	v_cndmask_b32_e64 v220, v134, v220, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v221, v131, v132, 2
	v_cndmask_b32_e64 v221, v134, v221, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v222, v131, v128, 2
	v_cndmask_b32_e64 v222, v134, v222, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v223, v131, v132, 2
	v_cndmask_b32_e64 v223, v134, v223, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v224, v131, v132, 2
	v_cndmask_b32_e64 v224, v134, v224, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v225, v131, v132, 2
	v_cndmask_b32_e64 v225, v134, v225, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v226, v131, v128, 2
	v_cndmask_b32_e64 v226, v134, v226, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v227, v131, v132, 2
	v_cndmask_b32_e64 v227, v134, v227, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v228, v131, v132, 2
	v_cndmask_b32_e64 v228, v134, v228, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v229, v131, v132, 2
	v_cndmask_b32_e64 v229, v134, v229, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v230, v131, v128, 2
	v_cndmask_b32_e64 v230, v134, v230, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v231, v131, v132, 2
	v_cndmask_b32_e64 v231, v134, v231, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v232, v131, v132, 2
	v_cndmask_b32_e64 v232, v134, v232, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v233, v131, v132, 2
	v_cndmask_b32_e64 v233, v134, v233, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v234, v131, v128, 2
	v_cndmask_b32_e64 v234, v134, v234, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v235, v131, v132, 2
	v_cndmask_b32_e64 v235, v134, v235, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v236, v131, v132, 2
	v_cndmask_b32_e64 v236, v134, v236, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v237, v131, v132, 2
	v_cndmask_b32_e64 v237, v134, v237, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v238, v131, v128, 2
	v_cndmask_b32_e64 v238, v134, v238, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v239, v131, v132, 2
	v_cndmask_b32_e64 v239, v134, v239, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v240, v131, v132, 2
	v_cndmask_b32_e64 v240, v134, v240, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v241, v131, v132, 2
	v_cndmask_b32_e64 v241, v134, v241, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v242, v131, v128, 2
	v_cndmask_b32_e64 v242, v134, v242, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v243, v131, v132, 2
	v_cndmask_b32_e64 v243, v134, v243, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v244, v131, v132, 2
	v_cndmask_b32_e64 v244, v134, v244, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v245, v131, v132, 2
	v_cndmask_b32_e64 v245, v134, v245, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v246, v131, v128, 2
	v_cndmask_b32_e64 v246, v134, v246, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v247, v131, v132, 2
	v_cndmask_b32_e64 v247, v134, v247, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v248, v131, v132, 2
	v_cndmask_b32_e64 v248, v134, v248, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v249, v131, v132, 2
	v_cndmask_b32_e64 v249, v134, v249, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v250, v131, v128, 2
	v_cndmask_b32_e64 v250, v134, v250, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v251, v131, v132, 2
	v_cndmask_b32_e64 v251, v134, v251, s32
	v_mov_b32_e32 v135, v0
	v_mov_b32_e32 v136, v8
	v_mov_b32_e32 v137, v16
	v_mov_b32_e32 v138, v24
	v_mov_b32_e32 v139, v32
	v_mov_b32_e32 v140, v40
	v_mov_b32_e32 v141, v48
	v_mov_b32_e32 v142, v56
	v_mov_b32_e32 v143, v64
	v_mov_b32_e32 v144, v72
	v_mov_b32_e32 v145, v80
	v_mov_b32_e32 v146, v88
	v_mov_b32_e32 v147, v96
	v_mov_b32_e32 v148, v104
	v_mov_b32_e32 v149, v112
	v_mov_b32_e32 v150, v120
	v_mov_b32_e32 v151, v1
	v_mov_b32_e32 v152, v9
	v_mov_b32_e32 v153, v17
	v_mov_b32_e32 v154, v25
	v_mov_b32_e32 v155, v33
	v_mov_b32_e32 v156, v41
	v_mov_b32_e32 v157, v49
	v_mov_b32_e32 v158, v57
	v_mov_b32_e32 v159, v65
	v_mov_b32_e32 v160, v73
	v_mov_b32_e32 v161, v81
	v_mov_b32_e32 v162, v89
	v_mov_b32_e32 v163, v97
	v_mov_b32_e32 v164, v105
	v_mov_b32_e32 v165, v113
	v_mov_b32_e32 v166, v121
	v_mov_b32_e32 v167, v2
	v_mov_b32_e32 v168, v10
	v_mov_b32_e32 v169, v18
	v_mov_b32_e32 v170, v26
	v_mov_b32_e32 v171, v34
	v_mov_b32_e32 v172, v42
	v_mov_b32_e32 v173, v50
	v_mov_b32_e32 v174, v58
	v_mov_b32_e32 v175, v66
	v_mov_b32_e32 v176, v74
	v_mov_b32_e32 v177, v82
	v_mov_b32_e32 v178, v90
	v_mov_b32_e32 v179, v98
	v_mov_b32_e32 v180, v106
	v_mov_b32_e32 v181, v114
	v_mov_b32_e32 v182, v122
	v_mov_b32_e32 v183, v3
	v_mov_b32_e32 v184, v11
	v_mov_b32_e32 v185, v19
	v_mov_b32_e32 v186, v27
	v_mov_b32_e32 v187, v35
	v_mov_b32_e32 v188, v43
	v_mov_b32_e32 v189, v51
	v_mov_b32_e32 v190, v59
	v_mov_b32_e32 v191, v67
	v_mov_b32_e32 v192, v75
	buffer_store_b32 v135, v193, s[16:19], null offen
	buffer_store_b32 v136, v194, s[16:19], null offen
	buffer_store_b32 v137, v195, s[16:19], null offen
	buffer_store_b32 v138, v196, s[16:19], null offen
	buffer_store_b32 v139, v197, s[16:19], null offen
	buffer_store_b32 v140, v198, s[16:19], null offen
	buffer_store_b32 v141, v199, s[16:19], null offen
	buffer_store_b32 v142, v200, s[16:19], null offen
	buffer_store_b32 v143, v201, s[16:19], null offen
	buffer_store_b32 v144, v202, s[16:19], null offen
	buffer_store_b32 v145, v203, s[16:19], null offen
	buffer_store_b32 v146, v204, s[16:19], null offen
	buffer_store_b32 v147, v205, s[16:19], null offen
	buffer_store_b32 v148, v206, s[16:19], null offen
	buffer_store_b32 v149, v207, s[16:19], null offen
	buffer_store_b32 v150, v208, s[16:19], null offen
	buffer_store_b32 v151, v209, s[16:19], null offen
	buffer_store_b32 v152, v210, s[16:19], null offen
	buffer_store_b32 v153, v211, s[16:19], null offen
	buffer_store_b32 v154, v212, s[16:19], null offen
	buffer_store_b32 v155, v213, s[16:19], null offen
	buffer_store_b32 v156, v214, s[16:19], null offen
	buffer_store_b32 v157, v215, s[16:19], null offen
	buffer_store_b32 v158, v216, s[16:19], null offen
	buffer_store_b32 v159, v217, s[16:19], null offen
	buffer_store_b32 v160, v219, s[16:19], null offen
	buffer_store_b32 v161, v220, s[16:19], null offen
	buffer_store_b32 v162, v221, s[16:19], null offen
	buffer_store_b32 v163, v222, s[16:19], null offen
	buffer_store_b32 v164, v223, s[16:19], null offen
	buffer_store_b32 v165, v224, s[16:19], null offen
	buffer_store_b32 v166, v225, s[16:19], null offen
	buffer_store_b32 v167, v226, s[16:19], null offen
	buffer_store_b32 v168, v227, s[16:19], null offen
	buffer_store_b32 v169, v228, s[16:19], null offen
	buffer_store_b32 v170, v229, s[16:19], null offen
	buffer_store_b32 v171, v230, s[16:19], null offen
	buffer_store_b32 v172, v231, s[16:19], null offen
	buffer_store_b32 v173, v232, s[16:19], null offen
	buffer_store_b32 v174, v233, s[16:19], null offen
	buffer_store_b32 v175, v234, s[16:19], null offen
	buffer_store_b32 v176, v235, s[16:19], null offen
	buffer_store_b32 v177, v236, s[16:19], null offen
	buffer_store_b32 v178, v237, s[16:19], null offen
	buffer_store_b32 v179, v238, s[16:19], null offen
	buffer_store_b32 v180, v239, s[16:19], null offen
	buffer_store_b32 v181, v240, s[16:19], null offen
	buffer_store_b32 v182, v241, s[16:19], null offen
	buffer_store_b32 v183, v242, s[16:19], null offen
	buffer_store_b32 v184, v243, s[16:19], null offen
	buffer_store_b32 v185, v244, s[16:19], null offen
	buffer_store_b32 v186, v245, s[16:19], null offen
	buffer_store_b32 v187, v246, s[16:19], null offen
	buffer_store_b32 v188, v247, s[16:19], null offen
	buffer_store_b32 v189, v248, s[16:19], null offen
	buffer_store_b32 v190, v249, s[16:19], null offen
	buffer_store_b32 v191, v250, s[16:19], null offen
	buffer_store_b32 v192, v251, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v134, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v193, v131, v132, 2
	v_cndmask_b32_e64 v193, v134, v193, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v194, v131, v132, 2
	v_cndmask_b32_e64 v194, v134, v194, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v134, v195, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v196, v131, v132, 2
	v_cndmask_b32_e64 v196, v134, v196, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v134, v197, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v198, v131, v132, 2
	v_cndmask_b32_e64 v198, v134, v198, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v199, v131, v128, 2
	v_cndmask_b32_e64 v199, v134, v199, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v200, v131, v132, 2
	v_cndmask_b32_e64 v200, v134, v200, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v201, v131, v132, 2
	v_cndmask_b32_e64 v201, v134, v201, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v202, v131, v132, 2
	v_cndmask_b32_e64 v202, v134, v202, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v203, v131, v128, 2
	v_cndmask_b32_e64 v203, v134, v203, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v204, v131, v132, 2
	v_cndmask_b32_e64 v204, v134, v204, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v205, v131, v132, 2
	v_cndmask_b32_e64 v205, v134, v205, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v206, v131, v132, 2
	v_cndmask_b32_e64 v206, v134, v206, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v207, v131, v128, 2
	v_cndmask_b32_e64 v207, v134, v207, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v208, v131, v132, 2
	v_cndmask_b32_e64 v208, v134, v208, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v134, v209, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v210, v131, v132, 2
	v_cndmask_b32_e64 v210, v134, v210, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v211, v131, v128, 2
	v_cndmask_b32_e64 v211, v134, v211, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v212, v131, v132, 2
	v_cndmask_b32_e64 v212, v134, v212, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v213, v131, v132, 2
	v_cndmask_b32_e64 v213, v134, v213, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v214, v131, v132, 2
	v_cndmask_b32_e64 v214, v134, v214, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v215, v131, v128, 2
	v_cndmask_b32_e64 v215, v134, v215, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v216, v131, v132, 2
	v_cndmask_b32_e64 v216, v134, v216, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v217, v131, v132, 2
	v_cndmask_b32_e64 v217, v134, v217, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v219, v131, v132, 2
	v_cndmask_b32_e64 v219, v134, v219, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v220, v131, v128, 2
	v_cndmask_b32_e64 v220, v134, v220, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v221, v131, v132, 2
	v_cndmask_b32_e64 v221, v134, v221, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v222, v131, v132, 2
	v_cndmask_b32_e64 v222, v134, v222, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v223, v131, v132, 2
	v_cndmask_b32_e64 v223, v134, v223, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v224, v131, v128, 2
	v_cndmask_b32_e64 v224, v134, v224, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v225, v131, v132, 2
	v_cndmask_b32_e64 v225, v134, v225, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v226, v131, v132, 2
	v_cndmask_b32_e64 v226, v134, v226, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v227, v131, v132, 2
	v_cndmask_b32_e64 v227, v134, v227, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v228, v131, v128, 2
	v_cndmask_b32_e64 v228, v134, v228, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v229, v131, v132, 2
	v_cndmask_b32_e64 v229, v134, v229, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v230, v131, v132, 2
	v_cndmask_b32_e64 v230, v134, v230, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v231, v131, v132, 2
	v_cndmask_b32_e64 v231, v134, v231, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v232, v131, v128, 2
	v_cndmask_b32_e64 v232, v134, v232, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v233, v131, v132, 2
	v_cndmask_b32_e64 v233, v134, v233, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v234, v131, v132, 2
	v_cndmask_b32_e64 v234, v134, v234, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v235, v131, v132, 2
	v_cndmask_b32_e64 v235, v134, v235, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v236, v131, v128, 2
	v_cndmask_b32_e64 v236, v134, v236, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v237, v131, v132, 2
	v_cndmask_b32_e64 v237, v134, v237, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v238, v131, v132, 2
	v_cndmask_b32_e64 v238, v134, v238, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v239, v131, v132, 2
	v_cndmask_b32_e64 v239, v134, v239, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v240, v131, v128, 2
	v_cndmask_b32_e64 v240, v134, v240, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v241, v131, v132, 2
	v_cndmask_b32_e64 v241, v134, v241, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v242, v131, v132, 2
	v_cndmask_b32_e64 v242, v134, v242, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v243, v131, v132, 2
	v_cndmask_b32_e64 v243, v134, v243, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v244, v131, v128, 2
	v_cndmask_b32_e64 v244, v134, v244, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v245, v131, v132, 2
	v_cndmask_b32_e64 v245, v134, v245, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v246, v131, v132, 2
	v_cndmask_b32_e64 v246, v134, v246, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v247, v131, v132, 2
	v_cndmask_b32_e64 v247, v134, v247, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v248, v131, v128, 2
	v_cndmask_b32_e64 v248, v134, v248, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v249, v131, v132, 2
	v_cndmask_b32_e64 v249, v134, v249, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v250, v131, v132, 2
	v_cndmask_b32_e64 v250, v134, v250, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v251, v131, v132, 2
	v_cndmask_b32_e64 v251, v134, v251, s32
	v_mov_b32_e32 v135, v83
	v_mov_b32_e32 v136, v91
	v_mov_b32_e32 v137, v99
	v_mov_b32_e32 v138, v107
	v_mov_b32_e32 v139, v115
	v_mov_b32_e32 v140, v123
	v_mov_b32_e32 v141, v4
	v_mov_b32_e32 v142, v12
	v_mov_b32_e32 v143, v20
	v_mov_b32_e32 v144, v28
	v_mov_b32_e32 v145, v36
	v_mov_b32_e32 v146, v44
	v_mov_b32_e32 v147, v52
	v_mov_b32_e32 v148, v60
	v_mov_b32_e32 v149, v68
	v_mov_b32_e32 v150, v76
	v_mov_b32_e32 v151, v84
	v_mov_b32_e32 v152, v92
	v_mov_b32_e32 v153, v100
	v_mov_b32_e32 v154, v108
	v_mov_b32_e32 v155, v116
	v_mov_b32_e32 v156, v124
	v_mov_b32_e32 v157, v5
	v_mov_b32_e32 v158, v13
	v_mov_b32_e32 v159, v21
	v_mov_b32_e32 v160, v29
	v_mov_b32_e32 v161, v37
	v_mov_b32_e32 v162, v45
	v_mov_b32_e32 v163, v53
	v_mov_b32_e32 v164, v61
	v_mov_b32_e32 v165, v69
	v_mov_b32_e32 v166, v77
	v_mov_b32_e32 v167, v85
	v_mov_b32_e32 v168, v93
	v_mov_b32_e32 v169, v101
	v_mov_b32_e32 v170, v109
	v_mov_b32_e32 v171, v117
	v_mov_b32_e32 v172, v125
	v_mov_b32_e32 v173, v6
	v_mov_b32_e32 v174, v14
	v_mov_b32_e32 v175, v22
	v_mov_b32_e32 v176, v30
	v_mov_b32_e32 v177, v38
	v_mov_b32_e32 v178, v46
	v_mov_b32_e32 v179, v54
	v_mov_b32_e32 v180, v62
	v_mov_b32_e32 v181, v70
	v_mov_b32_e32 v182, v78
	v_mov_b32_e32 v183, v86
	v_mov_b32_e32 v184, v94
	v_mov_b32_e32 v185, v102
	v_mov_b32_e32 v186, v110
	v_mov_b32_e32 v187, v118
	v_mov_b32_e32 v188, v126
	v_mov_b32_e32 v189, v7
	v_mov_b32_e32 v190, v15
	v_mov_b32_e32 v191, v23
	v_mov_b32_e32 v192, v31
	buffer_store_b32 v135, v193, s[16:19], null offen
	buffer_store_b32 v136, v194, s[16:19], null offen
	buffer_store_b32 v137, v195, s[16:19], null offen
	buffer_store_b32 v138, v196, s[16:19], null offen
	buffer_store_b32 v139, v197, s[16:19], null offen
	buffer_store_b32 v140, v198, s[16:19], null offen
	buffer_store_b32 v141, v199, s[16:19], null offen
	buffer_store_b32 v142, v200, s[16:19], null offen
	buffer_store_b32 v143, v201, s[16:19], null offen
	buffer_store_b32 v144, v202, s[16:19], null offen
	buffer_store_b32 v145, v203, s[16:19], null offen
	buffer_store_b32 v146, v204, s[16:19], null offen
	buffer_store_b32 v147, v205, s[16:19], null offen
	buffer_store_b32 v148, v206, s[16:19], null offen
	buffer_store_b32 v149, v207, s[16:19], null offen
	buffer_store_b32 v150, v208, s[16:19], null offen
	buffer_store_b32 v151, v209, s[16:19], null offen
	buffer_store_b32 v152, v210, s[16:19], null offen
	buffer_store_b32 v153, v211, s[16:19], null offen
	buffer_store_b32 v154, v212, s[16:19], null offen
	buffer_store_b32 v155, v213, s[16:19], null offen
	buffer_store_b32 v156, v214, s[16:19], null offen
	buffer_store_b32 v157, v215, s[16:19], null offen
	buffer_store_b32 v158, v216, s[16:19], null offen
	buffer_store_b32 v159, v217, s[16:19], null offen
	buffer_store_b32 v160, v219, s[16:19], null offen
	buffer_store_b32 v161, v220, s[16:19], null offen
	buffer_store_b32 v162, v221, s[16:19], null offen
	buffer_store_b32 v163, v222, s[16:19], null offen
	buffer_store_b32 v164, v223, s[16:19], null offen
	buffer_store_b32 v165, v224, s[16:19], null offen
	buffer_store_b32 v166, v225, s[16:19], null offen
	buffer_store_b32 v167, v226, s[16:19], null offen
	buffer_store_b32 v168, v227, s[16:19], null offen
	buffer_store_b32 v169, v228, s[16:19], null offen
	buffer_store_b32 v170, v229, s[16:19], null offen
	buffer_store_b32 v171, v230, s[16:19], null offen
	buffer_store_b32 v172, v231, s[16:19], null offen
	buffer_store_b32 v173, v232, s[16:19], null offen
	buffer_store_b32 v174, v233, s[16:19], null offen
	buffer_store_b32 v175, v234, s[16:19], null offen
	buffer_store_b32 v176, v235, s[16:19], null offen
	buffer_store_b32 v177, v236, s[16:19], null offen
	buffer_store_b32 v178, v237, s[16:19], null offen
	buffer_store_b32 v179, v238, s[16:19], null offen
	buffer_store_b32 v180, v239, s[16:19], null offen
	buffer_store_b32 v181, v240, s[16:19], null offen
	buffer_store_b32 v182, v241, s[16:19], null offen
	buffer_store_b32 v183, v242, s[16:19], null offen
	buffer_store_b32 v184, v243, s[16:19], null offen
	buffer_store_b32 v185, v244, s[16:19], null offen
	buffer_store_b32 v186, v245, s[16:19], null offen
	buffer_store_b32 v187, v246, s[16:19], null offen
	buffer_store_b32 v188, v247, s[16:19], null offen
	buffer_store_b32 v189, v248, s[16:19], null offen
	buffer_store_b32 v190, v249, s[16:19], null offen
	buffer_store_b32 v191, v250, s[16:19], null offen
	buffer_store_b32 v192, v251, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v134, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v147, v131, v128, 2
	v_cndmask_b32_e64 v147, v134, v147, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v148, v131, v132, 2
	v_cndmask_b32_e64 v148, v134, v148, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v149, v131, v132, 2
	v_cndmask_b32_e64 v149, v134, v149, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v150, v131, v132, 2
	v_cndmask_b32_e64 v150, v134, v150, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v151, v131, v128, 2
	v_cndmask_b32_e64 v151, v134, v151, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v152, v131, v132, 2
	v_cndmask_b32_e64 v152, v134, v152, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v153, v131, v132, 2
	v_cndmask_b32_e64 v153, v134, v153, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v154, v131, v132, 2
	v_cndmask_b32_e64 v154, v134, v154, s32
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s30, v128, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v155, v131, v128, 2
	v_cndmask_b32_e64 v155, v134, v155, s32
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v156, v131, v132, 2
	v_cndmask_b32_e64 v156, v134, v156, s32
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v157, v131, v132, 2
	v_cndmask_b32_e64 v157, v134, v157, s32
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s30, v132, s24
	v_cmp_lt_u32_e64 s32, v129, s25
	s_and_b32 s32, s30, s32
	v_add_lshl_u32 v158, v131, v132, 2
	v_cndmask_b32_e64 v158, v134, v158, s32
	v_mov_b32_e32 v135, v39
	v_mov_b32_e32 v136, v47
	v_mov_b32_e32 v137, v55
	v_mov_b32_e32 v138, v63
	v_mov_b32_e32 v139, v71
	v_mov_b32_e32 v140, v79
	v_mov_b32_e32 v141, v87
	v_mov_b32_e32 v142, v95
	v_mov_b32_e32 v143, v103
	v_mov_b32_e32 v144, v111
	v_mov_b32_e32 v145, v119
	v_mov_b32_e32 v146, v127
	buffer_store_b32 v135, v147, s[16:19], null offen
	buffer_store_b32 v136, v148, s[16:19], null offen
	buffer_store_b32 v137, v149, s[16:19], null offen
	buffer_store_b32 v138, v150, s[16:19], null offen
	buffer_store_b32 v139, v151, s[16:19], null offen
	buffer_store_b32 v140, v152, s[16:19], null offen
	buffer_store_b32 v141, v153, s[16:19], null offen
	buffer_store_b32 v142, v154, s[16:19], null offen
	buffer_store_b32 v143, v155, s[16:19], null offen
	buffer_store_b32 v144, v156, s[16:19], null offen
	buffer_store_b32 v145, v157, s[16:19], null offen
	buffer_store_b32 v146, v158, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End
label_GW_End:
	s_getpc_b64 s[30:31]
Lgpc_after_1:
	s_add_co_i32 s32, label_KernelEnd-Lgpc_after_1-4, 4
	s_add_co_u32 s30, s30, s32
	s_add_co_ci_u32 s31, s31, 0
	s_setpc_b64 s[30:31]
label_GSU_5:
	s_mov_b64 s[32:33], s[48:49]
	s_mov_b32 s35, 0x30020000
	s_cmp_eq_u64 s[48:49], 0
	s_cbranch_scc0 label_ScaleAlphaVecAddrValid
	s_mov_b32 s34, 0
	s_branch label_ScaleAlphaVecAddrValid_End
label_ScaleAlphaVecAddrValid:
	s_mov_b32 s34, s24
label_ScaleAlphaVecAddrValid_End:
	s_mul_i32 s34, 4, s34
	s_add_co_u32 s8, s4, 1
	s_mul_i32 s8, s53, s8
	s_cmp_eq_u32 s8, 0
	s_cselect_b32 s8, s24, s8
	s_mov_b64 s[40:41], s[50:51]
	s_mov_b32 s43, 0x30020000
	s_cmp_eq_u64 s[50:51], 0
	s_cbranch_scc0 label_BiasAddrValid
	s_mov_b32 s42, 0
	s_branch label_BiasAddrValid_End
label_BiasAddrValid:
	s_mov_b32 s42, s8
label_BiasAddrValid_End:
label_Load_Biasf32_0:
	s_mov_b32 s8, 0
	s_cmp_lg_u32 s52, s8
	s_cbranch_scc1 label_Load_Biasf16_0
	s_mul_i32 s8, 0x80, s2
	v_add_nc_u32_e32 v136, s8, v218
	s_mul_i32 s42, 4, s42
	s_mul_i32 s8, s53, s4
	v_add_nc_u32_e32 v134, s8, v136
	v_lshlrev_b32_e32 v134, 2, v134
	v_lshlrev_b32_e32 v135, 2, v136
	s_mul_i32 s8, 0x80, s3
	v_add_nc_u32_e32 v136, s8, v218
	buffer_load_b32 v132, v134, s[40:43], null offen
	buffer_load_b32 v133, v135, s[32:35], null offen
	v_lshlrev_b32_e32 v136, 2, v218
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_wait_loadcnt 0x1
	ds_store_b32 v136, v132
	v_cmp_gt_u32_e64 s48, s34, 0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v133, 1.0, v133, s48
	ds_store_b32 v136, v133 offset:512
	s_branch label_Load_Bias_End
label_Load_Biasf16_0:
	s_mov_b32 s8, 4
	s_cmp_lg_u32 s52, s8
	s_cbranch_scc1 label_Load_Bias_End
	s_mul_i32 s8, 0x80, s2
	v_add_nc_u32_e32 v136, s8, v218
	s_mul_i32 s42, 2, s42
	s_mul_i32 s8, s53, s4
	v_add_nc_u32_e32 v134, s8, v136
	v_lshlrev_b32_e32 v134, 1, v134
	v_lshlrev_b32_e32 v135, 2, v136
	s_mul_i32 s8, 0x80, s3
	v_add_nc_u32_e32 v136, s8, v218
	buffer_load_d16_b16 v132, v134, s[40:43], null offen
	buffer_load_b32 v133, v135, s[32:35], null offen
	v_lshlrev_b32_e32 v136, 2, v218
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_wait_loadcnt 0x1
	v_cvt_f32_f16_e64 v132, v132
	ds_store_b32 v136, v132
	v_cmp_gt_u32_e64 s48, s34, 0
	s_wait_loadcnt 0x0
	v_cndmask_b32_e64 v133, 1.0, v133, s48
	ds_store_b32 v136, v133 offset:512
	s_branch label_Load_Bias_End
label_Load_Bias_End:
	s_mov_b32 s11, 0
	s_cmp_eq_u32 s45, s11
	s_cbranch_scc0 label_GW_Beta_1
	s_and_b32 s32, 0x7f, s24
	s_add_co_u32 s33, -1, s14
	s_cmp_ge_u32 s2, s33
	s_cselect_b32 s32, s32, 0
	s_mov_b32 s8, 0
	s_cmp_gt_u32 s32, s8
	s_cbranch_scc1 label_GW_B0_E1_M_1
	s_and_b32 s32, 0x7f, s25
	s_add_co_u32 s33, -1, s15
	s_cmp_ge_u32 s3, s33
	s_cselect_b32 s32, s32, 0
	s_mov_b32 s8, 0
	s_cmp_gt_u32 s32, s8
	s_cbranch_scc1 label_GW_B0_E1_N_1
label_GW_B0_E0_1:
	s_mov_b32 s8, 3
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Gelu_VW4_beta_0_edge_0
	s_mov_b32 s8, 5
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Relu_VW4_beta_0_edge_0
	s_mov_b32 s8, 6
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Sigmoid_VW4_beta_0_edge_0
	s_mov_b32 s8, 10
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Silu_VW4_beta_0_edge_0
	s_mov_b32 s8, 12
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Clamp_VW4_beta_0_edge_0
label_To_Activation_None_VW4_beta_0_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_2:
	s_add_co_i32 s8, label_Activation_None_VW4-Lgpc_after_2-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_5
label_To_Activation_Gelu_VW4_beta_0_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_3:
	s_add_co_i32 s8, label_Activation_Gelu_VW4-Lgpc_after_3-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_5
label_To_Activation_Relu_VW4_beta_0_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_4:
	s_add_co_i32 s8, label_Activation_Relu_VW4-Lgpc_after_4-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_5
label_To_Activation_Sigmoid_VW4_beta_0_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_5:
	s_add_co_i32 s8, label_Activation_Sigmoid_VW4-Lgpc_after_5-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_5
label_To_Activation_Silu_VW4_beta_0_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_6:
	s_add_co_i32 s8, label_Activation_Silu_VW4-Lgpc_after_6-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_5
label_To_Activation_Clamp_VW4_beta_0_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_7:
	s_add_co_i32 s8, label_Activation_Clamp_VW4-Lgpc_after_7-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_5
label_ActivationSetPCAddrEnd_5:
	s_mul_i32 s8, 0x80, s2
	v_sub_nc_u32_e64 v139, v128, s8
	v_lshlrev_b32_e32 v139, 2, v139
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[172:175], v139
	ds_load_b128 v[176:179], v139 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_mul_f32_e32 v140, s44, v0
	v_mul_f32_e32 v141, s44, v8
	v_mul_f32_e32 v142, s44, v16
	v_mul_f32_e32 v143, s44, v24
	v_mul_f32_e32 v144, s44, v32
	v_mul_f32_e32 v145, s44, v40
	v_mul_f32_e32 v146, s44, v48
	v_mul_f32_e32 v147, s44, v56
	v_mul_f32_e32 v148, s44, v64
	v_mul_f32_e32 v149, s44, v72
	v_mul_f32_e32 v150, s44, v80
	v_mul_f32_e32 v151, s44, v88
	v_mul_f32_e32 v152, s44, v96
	v_mul_f32_e32 v153, s44, v104
	v_mul_f32_e32 v154, s44, v112
	v_mul_f32_e32 v155, s44, v120
	v_mul_f32_e32 v156, s44, v1
	v_mul_f32_e32 v157, s44, v9
	v_mul_f32_e32 v158, s44, v17
	v_mul_f32_e32 v159, s44, v25
	v_mul_f32_e32 v160, s44, v33
	v_mul_f32_e32 v161, s44, v41
	v_mul_f32_e32 v162, s44, v49
	v_mul_f32_e32 v163, s44, v57
	v_mul_f32_e32 v164, s44, v65
	v_mul_f32_e32 v165, s44, v73
	v_mul_f32_e32 v166, s44, v81
	v_mul_f32_e32 v167, s44, v89
	v_mul_f32_e32 v168, s44, v97
	v_mul_f32_e32 v169, s44, v105
	v_mul_f32_e32 v170, s44, v113
	v_mul_f32_e32 v171, s44, v121
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v176, v140
	v_mul_f32_e32 v141, v177, v141
	v_mul_f32_e32 v142, v178, v142
	v_mul_f32_e32 v143, v179, v143
	v_add_f32_e32 v132, v172, v140
	v_add_f32_e32 v133, v173, v141
	v_add_f32_e32 v134, v174, v142
	v_add_f32_e32 v135, v175, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v176, v144
	v_mul_f32_e32 v145, v177, v145
	v_mul_f32_e32 v146, v178, v146
	v_mul_f32_e32 v147, v179, v147
	v_add_f32_e32 v132, v172, v144
	v_add_f32_e32 v133, v173, v145
	v_add_f32_e32 v134, v174, v146
	v_add_f32_e32 v135, v175, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	v_mul_f32_e32 v148, v176, v148
	v_mul_f32_e32 v149, v177, v149
	v_mul_f32_e32 v150, v178, v150
	v_mul_f32_e32 v151, v179, v151
	v_add_f32_e32 v132, v172, v148
	v_add_f32_e32 v133, v173, v149
	v_add_f32_e32 v134, v174, v150
	v_add_f32_e32 v135, v175, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	v_mul_f32_e32 v152, v176, v152
	v_mul_f32_e32 v153, v177, v153
	v_mul_f32_e32 v154, v178, v154
	v_mul_f32_e32 v155, v179, v155
	v_add_f32_e32 v132, v172, v152
	v_add_f32_e32 v133, v173, v153
	v_add_f32_e32 v134, v174, v154
	v_add_f32_e32 v135, v175, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	v_mul_f32_e32 v156, v176, v156
	v_mul_f32_e32 v157, v177, v157
	v_mul_f32_e32 v158, v178, v158
	v_mul_f32_e32 v159, v179, v159
	v_add_f32_e32 v132, v172, v156
	v_add_f32_e32 v133, v173, v157
	v_add_f32_e32 v134, v174, v158
	v_add_f32_e32 v135, v175, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	v_mul_f32_e32 v160, v176, v160
	v_mul_f32_e32 v161, v177, v161
	v_mul_f32_e32 v162, v178, v162
	v_mul_f32_e32 v163, v179, v163
	v_add_f32_e32 v132, v172, v160
	v_add_f32_e32 v133, v173, v161
	v_add_f32_e32 v134, v174, v162
	v_add_f32_e32 v135, v175, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	v_mul_f32_e32 v164, v176, v164
	v_mul_f32_e32 v165, v177, v165
	v_mul_f32_e32 v166, v178, v166
	v_mul_f32_e32 v167, v179, v167
	v_add_f32_e32 v132, v172, v164
	v_add_f32_e32 v133, v173, v165
	v_add_f32_e32 v134, v174, v166
	v_add_f32_e32 v135, v175, v167
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v164, v132
	v_mov_b32_e32 v165, v133
	v_mov_b32_e32 v166, v134
	v_mov_b32_e32 v167, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[164:167], v137, s[16:19], null offen
	v_mul_f32_e32 v168, v176, v168
	v_mul_f32_e32 v169, v177, v169
	v_mul_f32_e32 v170, v178, v170
	v_mul_f32_e32 v171, v179, v171
	v_add_f32_e32 v132, v172, v168
	v_add_f32_e32 v133, v173, v169
	v_add_f32_e32 v134, v174, v170
	v_add_f32_e32 v135, v175, v171
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v168, v132
	v_mov_b32_e32 v169, v133
	v_mov_b32_e32 v170, v134
	v_mov_b32_e32 v171, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[168:171], v137, s[16:19], null offen
	s_nop 0
	ds_load_b128 v[172:175], v139
	ds_load_b128 v[176:179], v139 offset:512
	v_mul_f32_e32 v140, s44, v2
	v_mul_f32_e32 v141, s44, v10
	v_mul_f32_e32 v142, s44, v18
	v_mul_f32_e32 v143, s44, v26
	v_mul_f32_e32 v144, s44, v34
	v_mul_f32_e32 v145, s44, v42
	v_mul_f32_e32 v146, s44, v50
	v_mul_f32_e32 v147, s44, v58
	v_mul_f32_e32 v148, s44, v66
	v_mul_f32_e32 v149, s44, v74
	v_mul_f32_e32 v150, s44, v82
	v_mul_f32_e32 v151, s44, v90
	v_mul_f32_e32 v152, s44, v98
	v_mul_f32_e32 v153, s44, v106
	v_mul_f32_e32 v154, s44, v114
	v_mul_f32_e32 v155, s44, v122
	v_mul_f32_e32 v156, s44, v3
	v_mul_f32_e32 v157, s44, v11
	v_mul_f32_e32 v158, s44, v19
	v_mul_f32_e32 v159, s44, v27
	v_mul_f32_e32 v160, s44, v35
	v_mul_f32_e32 v161, s44, v43
	v_mul_f32_e32 v162, s44, v51
	v_mul_f32_e32 v163, s44, v59
	v_mul_f32_e32 v164, s44, v67
	v_mul_f32_e32 v165, s44, v75
	v_mul_f32_e32 v166, s44, v83
	v_mul_f32_e32 v167, s44, v91
	v_mul_f32_e32 v168, s44, v99
	v_mul_f32_e32 v169, s44, v107
	v_mul_f32_e32 v170, s44, v115
	v_mul_f32_e32 v171, s44, v123
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v176, v140
	v_mul_f32_e32 v141, v177, v141
	v_mul_f32_e32 v142, v178, v142
	v_mul_f32_e32 v143, v179, v143
	v_add_f32_e32 v132, v172, v140
	v_add_f32_e32 v133, v173, v141
	v_add_f32_e32 v134, v174, v142
	v_add_f32_e32 v135, v175, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v176, v144
	v_mul_f32_e32 v145, v177, v145
	v_mul_f32_e32 v146, v178, v146
	v_mul_f32_e32 v147, v179, v147
	v_add_f32_e32 v132, v172, v144
	v_add_f32_e32 v133, v173, v145
	v_add_f32_e32 v134, v174, v146
	v_add_f32_e32 v135, v175, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	v_mul_f32_e32 v148, v176, v148
	v_mul_f32_e32 v149, v177, v149
	v_mul_f32_e32 v150, v178, v150
	v_mul_f32_e32 v151, v179, v151
	v_add_f32_e32 v132, v172, v148
	v_add_f32_e32 v133, v173, v149
	v_add_f32_e32 v134, v174, v150
	v_add_f32_e32 v135, v175, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	v_mul_f32_e32 v152, v176, v152
	v_mul_f32_e32 v153, v177, v153
	v_mul_f32_e32 v154, v178, v154
	v_mul_f32_e32 v155, v179, v155
	v_add_f32_e32 v132, v172, v152
	v_add_f32_e32 v133, v173, v153
	v_add_f32_e32 v134, v174, v154
	v_add_f32_e32 v135, v175, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	v_mul_f32_e32 v156, v176, v156
	v_mul_f32_e32 v157, v177, v157
	v_mul_f32_e32 v158, v178, v158
	v_mul_f32_e32 v159, v179, v159
	v_add_f32_e32 v132, v172, v156
	v_add_f32_e32 v133, v173, v157
	v_add_f32_e32 v134, v174, v158
	v_add_f32_e32 v135, v175, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	v_mul_f32_e32 v160, v176, v160
	v_mul_f32_e32 v161, v177, v161
	v_mul_f32_e32 v162, v178, v162
	v_mul_f32_e32 v163, v179, v163
	v_add_f32_e32 v132, v172, v160
	v_add_f32_e32 v133, v173, v161
	v_add_f32_e32 v134, v174, v162
	v_add_f32_e32 v135, v175, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	v_mul_f32_e32 v164, v176, v164
	v_mul_f32_e32 v165, v177, v165
	v_mul_f32_e32 v166, v178, v166
	v_mul_f32_e32 v167, v179, v167
	v_add_f32_e32 v132, v172, v164
	v_add_f32_e32 v133, v173, v165
	v_add_f32_e32 v134, v174, v166
	v_add_f32_e32 v135, v175, v167
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v164, v132
	v_mov_b32_e32 v165, v133
	v_mov_b32_e32 v166, v134
	v_mov_b32_e32 v167, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[164:167], v137, s[16:19], null offen
	v_mul_f32_e32 v168, v176, v168
	v_mul_f32_e32 v169, v177, v169
	v_mul_f32_e32 v170, v178, v170
	v_mul_f32_e32 v171, v179, v171
	v_add_f32_e32 v132, v172, v168
	v_add_f32_e32 v133, v173, v169
	v_add_f32_e32 v134, v174, v170
	v_add_f32_e32 v135, v175, v171
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v168, v132
	v_mov_b32_e32 v169, v133
	v_mov_b32_e32 v170, v134
	v_mov_b32_e32 v171, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[168:171], v137, s[16:19], null offen
	s_nop 0
	ds_load_b128 v[172:175], v139
	ds_load_b128 v[176:179], v139 offset:512
	v_mul_f32_e32 v140, s44, v4
	v_mul_f32_e32 v141, s44, v12
	v_mul_f32_e32 v142, s44, v20
	v_mul_f32_e32 v143, s44, v28
	v_mul_f32_e32 v144, s44, v36
	v_mul_f32_e32 v145, s44, v44
	v_mul_f32_e32 v146, s44, v52
	v_mul_f32_e32 v147, s44, v60
	v_mul_f32_e32 v148, s44, v68
	v_mul_f32_e32 v149, s44, v76
	v_mul_f32_e32 v150, s44, v84
	v_mul_f32_e32 v151, s44, v92
	v_mul_f32_e32 v152, s44, v100
	v_mul_f32_e32 v153, s44, v108
	v_mul_f32_e32 v154, s44, v116
	v_mul_f32_e32 v155, s44, v124
	v_mul_f32_e32 v156, s44, v5
	v_mul_f32_e32 v157, s44, v13
	v_mul_f32_e32 v158, s44, v21
	v_mul_f32_e32 v159, s44, v29
	v_mul_f32_e32 v160, s44, v37
	v_mul_f32_e32 v161, s44, v45
	v_mul_f32_e32 v162, s44, v53
	v_mul_f32_e32 v163, s44, v61
	v_mul_f32_e32 v164, s44, v69
	v_mul_f32_e32 v165, s44, v77
	v_mul_f32_e32 v166, s44, v85
	v_mul_f32_e32 v167, s44, v93
	v_mul_f32_e32 v168, s44, v101
	v_mul_f32_e32 v169, s44, v109
	v_mul_f32_e32 v170, s44, v117
	v_mul_f32_e32 v171, s44, v125
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v176, v140
	v_mul_f32_e32 v141, v177, v141
	v_mul_f32_e32 v142, v178, v142
	v_mul_f32_e32 v143, v179, v143
	v_add_f32_e32 v132, v172, v140
	v_add_f32_e32 v133, v173, v141
	v_add_f32_e32 v134, v174, v142
	v_add_f32_e32 v135, v175, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v176, v144
	v_mul_f32_e32 v145, v177, v145
	v_mul_f32_e32 v146, v178, v146
	v_mul_f32_e32 v147, v179, v147
	v_add_f32_e32 v132, v172, v144
	v_add_f32_e32 v133, v173, v145
	v_add_f32_e32 v134, v174, v146
	v_add_f32_e32 v135, v175, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	v_mul_f32_e32 v148, v176, v148
	v_mul_f32_e32 v149, v177, v149
	v_mul_f32_e32 v150, v178, v150
	v_mul_f32_e32 v151, v179, v151
	v_add_f32_e32 v132, v172, v148
	v_add_f32_e32 v133, v173, v149
	v_add_f32_e32 v134, v174, v150
	v_add_f32_e32 v135, v175, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	v_mul_f32_e32 v152, v176, v152
	v_mul_f32_e32 v153, v177, v153
	v_mul_f32_e32 v154, v178, v154
	v_mul_f32_e32 v155, v179, v155
	v_add_f32_e32 v132, v172, v152
	v_add_f32_e32 v133, v173, v153
	v_add_f32_e32 v134, v174, v154
	v_add_f32_e32 v135, v175, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	v_mul_f32_e32 v156, v176, v156
	v_mul_f32_e32 v157, v177, v157
	v_mul_f32_e32 v158, v178, v158
	v_mul_f32_e32 v159, v179, v159
	v_add_f32_e32 v132, v172, v156
	v_add_f32_e32 v133, v173, v157
	v_add_f32_e32 v134, v174, v158
	v_add_f32_e32 v135, v175, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	v_mul_f32_e32 v160, v176, v160
	v_mul_f32_e32 v161, v177, v161
	v_mul_f32_e32 v162, v178, v162
	v_mul_f32_e32 v163, v179, v163
	v_add_f32_e32 v132, v172, v160
	v_add_f32_e32 v133, v173, v161
	v_add_f32_e32 v134, v174, v162
	v_add_f32_e32 v135, v175, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	v_mul_f32_e32 v164, v176, v164
	v_mul_f32_e32 v165, v177, v165
	v_mul_f32_e32 v166, v178, v166
	v_mul_f32_e32 v167, v179, v167
	v_add_f32_e32 v132, v172, v164
	v_add_f32_e32 v133, v173, v165
	v_add_f32_e32 v134, v174, v166
	v_add_f32_e32 v135, v175, v167
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v164, v132
	v_mov_b32_e32 v165, v133
	v_mov_b32_e32 v166, v134
	v_mov_b32_e32 v167, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[164:167], v137, s[16:19], null offen
	v_mul_f32_e32 v168, v176, v168
	v_mul_f32_e32 v169, v177, v169
	v_mul_f32_e32 v170, v178, v170
	v_mul_f32_e32 v171, v179, v171
	v_add_f32_e32 v132, v172, v168
	v_add_f32_e32 v133, v173, v169
	v_add_f32_e32 v134, v174, v170
	v_add_f32_e32 v135, v175, v171
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v168, v132
	v_mov_b32_e32 v169, v133
	v_mov_b32_e32 v170, v134
	v_mov_b32_e32 v171, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[168:171], v137, s[16:19], null offen
	s_nop 0
	ds_load_b128 v[172:175], v139
	ds_load_b128 v[176:179], v139 offset:512
	v_mul_f32_e32 v140, s44, v6
	v_mul_f32_e32 v141, s44, v14
	v_mul_f32_e32 v142, s44, v22
	v_mul_f32_e32 v143, s44, v30
	v_mul_f32_e32 v144, s44, v38
	v_mul_f32_e32 v145, s44, v46
	v_mul_f32_e32 v146, s44, v54
	v_mul_f32_e32 v147, s44, v62
	v_mul_f32_e32 v148, s44, v70
	v_mul_f32_e32 v149, s44, v78
	v_mul_f32_e32 v150, s44, v86
	v_mul_f32_e32 v151, s44, v94
	v_mul_f32_e32 v152, s44, v102
	v_mul_f32_e32 v153, s44, v110
	v_mul_f32_e32 v154, s44, v118
	v_mul_f32_e32 v155, s44, v126
	v_mul_f32_e32 v156, s44, v7
	v_mul_f32_e32 v157, s44, v15
	v_mul_f32_e32 v158, s44, v23
	v_mul_f32_e32 v159, s44, v31
	v_mul_f32_e32 v160, s44, v39
	v_mul_f32_e32 v161, s44, v47
	v_mul_f32_e32 v162, s44, v55
	v_mul_f32_e32 v163, s44, v63
	v_mul_f32_e32 v164, s44, v71
	v_mul_f32_e32 v165, s44, v79
	v_mul_f32_e32 v166, s44, v87
	v_mul_f32_e32 v167, s44, v95
	v_mul_f32_e32 v168, s44, v103
	v_mul_f32_e32 v169, s44, v111
	v_mul_f32_e32 v170, s44, v119
	v_mul_f32_e32 v171, s44, v127
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v176, v140
	v_mul_f32_e32 v141, v177, v141
	v_mul_f32_e32 v142, v178, v142
	v_mul_f32_e32 v143, v179, v143
	v_add_f32_e32 v132, v172, v140
	v_add_f32_e32 v133, v173, v141
	v_add_f32_e32 v134, v174, v142
	v_add_f32_e32 v135, v175, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v176, v144
	v_mul_f32_e32 v145, v177, v145
	v_mul_f32_e32 v146, v178, v146
	v_mul_f32_e32 v147, v179, v147
	v_add_f32_e32 v132, v172, v144
	v_add_f32_e32 v133, v173, v145
	v_add_f32_e32 v134, v174, v146
	v_add_f32_e32 v135, v175, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	v_mul_f32_e32 v148, v176, v148
	v_mul_f32_e32 v149, v177, v149
	v_mul_f32_e32 v150, v178, v150
	v_mul_f32_e32 v151, v179, v151
	v_add_f32_e32 v132, v172, v148
	v_add_f32_e32 v133, v173, v149
	v_add_f32_e32 v134, v174, v150
	v_add_f32_e32 v135, v175, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	v_mul_f32_e32 v152, v176, v152
	v_mul_f32_e32 v153, v177, v153
	v_mul_f32_e32 v154, v178, v154
	v_mul_f32_e32 v155, v179, v155
	v_add_f32_e32 v132, v172, v152
	v_add_f32_e32 v133, v173, v153
	v_add_f32_e32 v134, v174, v154
	v_add_f32_e32 v135, v175, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	v_mul_f32_e32 v156, v176, v156
	v_mul_f32_e32 v157, v177, v157
	v_mul_f32_e32 v158, v178, v158
	v_mul_f32_e32 v159, v179, v159
	v_add_f32_e32 v132, v172, v156
	v_add_f32_e32 v133, v173, v157
	v_add_f32_e32 v134, v174, v158
	v_add_f32_e32 v135, v175, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	v_mul_f32_e32 v160, v176, v160
	v_mul_f32_e32 v161, v177, v161
	v_mul_f32_e32 v162, v178, v162
	v_mul_f32_e32 v163, v179, v163
	v_add_f32_e32 v132, v172, v160
	v_add_f32_e32 v133, v173, v161
	v_add_f32_e32 v134, v174, v162
	v_add_f32_e32 v135, v175, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	v_mul_f32_e32 v164, v176, v164
	v_mul_f32_e32 v165, v177, v165
	v_mul_f32_e32 v166, v178, v166
	v_mul_f32_e32 v167, v179, v167
	v_add_f32_e32 v132, v172, v164
	v_add_f32_e32 v133, v173, v165
	v_add_f32_e32 v134, v174, v166
	v_add_f32_e32 v135, v175, v167
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v164, v132
	v_mov_b32_e32 v165, v133
	v_mov_b32_e32 v166, v134
	v_mov_b32_e32 v167, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[164:167], v137, s[16:19], null offen
	v_mul_f32_e32 v168, v176, v168
	v_mul_f32_e32 v169, v177, v169
	v_mul_f32_e32 v170, v178, v170
	v_mul_f32_e32 v171, v179, v171
	v_add_f32_e32 v132, v172, v168
	v_add_f32_e32 v133, v173, v169
	v_add_f32_e32 v134, v174, v170
	v_add_f32_e32 v135, v175, v171
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v168, v132
	v_mov_b32_e32 v169, v133
	v_mov_b32_e32 v170, v134
	v_mov_b32_e32 v171, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[168:171], v137, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End_1
label_GW_B0_E1_N_1:
	s_mov_b32 s8, 3
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Gelu_VW4_beta_0_edge_1
	s_mov_b32 s8, 5
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Relu_VW4_beta_0_edge_1
	s_mov_b32 s8, 6
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Sigmoid_VW4_beta_0_edge_1
	s_mov_b32 s8, 10
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Silu_VW4_beta_0_edge_1
	s_mov_b32 s8, 12
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Clamp_VW4_beta_0_edge_1
label_To_Activation_None_VW4_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_8:
	s_add_co_i32 s8, label_Activation_None_VW4-Lgpc_after_8-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_4
label_To_Activation_Gelu_VW4_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_9:
	s_add_co_i32 s8, label_Activation_Gelu_VW4-Lgpc_after_9-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_4
label_To_Activation_Relu_VW4_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_10:
	s_add_co_i32 s8, label_Activation_Relu_VW4-Lgpc_after_10-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_4
label_To_Activation_Sigmoid_VW4_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_11:
	s_add_co_i32 s8, label_Activation_Sigmoid_VW4-Lgpc_after_11-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_4
label_To_Activation_Silu_VW4_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_12:
	s_add_co_i32 s8, label_Activation_Silu_VW4-Lgpc_after_12-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_4
label_To_Activation_Clamp_VW4_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_13:
	s_add_co_i32 s8, label_Activation_Clamp_VW4-Lgpc_after_13-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_4
label_ActivationSetPCAddrEnd_4:
	v_mov_b32_e32 v136, 0x80000000
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[164:167], v138
	ds_load_b128 v[168:171], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v172, v128, s32
	v_lshlrev_b32_e32 v172, 2, v172
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v128, s32
	v_lshlrev_b32_e32 v174, 2, v174
	v_add_lshl_u32 v173, v131, v128, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v128, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v128, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_mul_f32_e32 v140, s44, v0
	v_mul_f32_e32 v141, s44, v8
	v_mul_f32_e32 v142, s44, v16
	v_mul_f32_e32 v143, s44, v24
	v_mul_f32_e32 v144, s44, v32
	v_mul_f32_e32 v145, s44, v40
	v_mul_f32_e32 v146, s44, v48
	v_mul_f32_e32 v147, s44, v56
	v_mul_f32_e32 v148, s44, v64
	v_mul_f32_e32 v149, s44, v72
	v_mul_f32_e32 v150, s44, v80
	v_mul_f32_e32 v151, s44, v88
	v_mul_f32_e32 v152, s44, v96
	v_mul_f32_e32 v153, s44, v104
	v_mul_f32_e32 v154, s44, v112
	v_mul_f32_e32 v155, s44, v120
	v_mul_f32_e32 v156, s44, v1
	v_mul_f32_e32 v157, s44, v9
	v_mul_f32_e32 v158, s44, v17
	v_mul_f32_e32 v159, s44, v25
	v_mul_f32_e32 v160, s44, v33
	v_mul_f32_e32 v161, s44, v41
	v_mul_f32_e32 v162, s44, v49
	v_mul_f32_e32 v163, s44, v57
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v168, v140
	v_mul_f32_e32 v141, v169, v141
	v_mul_f32_e32 v142, v170, v142
	v_mul_f32_e32 v143, v171, v143
	v_add_f32_e32 v132, v164, v140
	v_add_f32_e32 v133, v165, v141
	v_add_f32_e32 v134, v166, v142
	v_add_f32_e32 v135, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v168, v144
	v_mul_f32_e32 v145, v169, v145
	v_mul_f32_e32 v146, v170, v146
	v_mul_f32_e32 v147, v171, v147
	v_add_f32_e32 v132, v164, v144
	v_add_f32_e32 v133, v165, v145
	v_add_f32_e32 v134, v166, v146
	v_add_f32_e32 v135, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v168, v148
	v_mul_f32_e32 v149, v169, v149
	v_mul_f32_e32 v150, v170, v150
	v_mul_f32_e32 v151, v171, v151
	v_add_f32_e32 v132, v164, v148
	v_add_f32_e32 v133, v165, v149
	v_add_f32_e32 v134, v166, v150
	v_add_f32_e32 v135, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v173, s[16:19], null offen
	v_mul_f32_e32 v152, v168, v152
	v_mul_f32_e32 v153, v169, v153
	v_mul_f32_e32 v154, v170, v154
	v_mul_f32_e32 v155, v171, v155
	v_add_f32_e32 v132, v164, v152
	v_add_f32_e32 v133, v165, v153
	v_add_f32_e32 v134, v166, v154
	v_add_f32_e32 v135, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v175, s[16:19], null offen
	v_mul_f32_e32 v156, v168, v156
	v_mul_f32_e32 v157, v169, v157
	v_mul_f32_e32 v158, v170, v158
	v_mul_f32_e32 v159, v171, v159
	v_add_f32_e32 v132, v164, v156
	v_add_f32_e32 v133, v165, v157
	v_add_f32_e32 v134, v166, v158
	v_add_f32_e32 v135, v167, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v177, s[16:19], null offen
	v_mul_f32_e32 v160, v168, v160
	v_mul_f32_e32 v161, v169, v161
	v_mul_f32_e32 v162, v170, v162
	v_mul_f32_e32 v163, v171, v163
	v_add_f32_e32 v132, v164, v160
	v_add_f32_e32 v133, v165, v161
	v_add_f32_e32 v134, v166, v162
	v_add_f32_e32 v135, v167, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v179, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[164:167], v138
	ds_load_b128 v[168:171], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v172, v128, s32
	v_lshlrev_b32_e32 v172, 2, v172
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v128, s32
	v_lshlrev_b32_e32 v174, 2, v174
	v_add_lshl_u32 v173, v131, v128, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v128, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v128, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_mul_f32_e32 v140, s44, v65
	v_mul_f32_e32 v141, s44, v73
	v_mul_f32_e32 v142, s44, v81
	v_mul_f32_e32 v143, s44, v89
	v_mul_f32_e32 v144, s44, v97
	v_mul_f32_e32 v145, s44, v105
	v_mul_f32_e32 v146, s44, v113
	v_mul_f32_e32 v147, s44, v121
	v_mul_f32_e32 v148, s44, v2
	v_mul_f32_e32 v149, s44, v10
	v_mul_f32_e32 v150, s44, v18
	v_mul_f32_e32 v151, s44, v26
	v_mul_f32_e32 v152, s44, v34
	v_mul_f32_e32 v153, s44, v42
	v_mul_f32_e32 v154, s44, v50
	v_mul_f32_e32 v155, s44, v58
	v_mul_f32_e32 v156, s44, v66
	v_mul_f32_e32 v157, s44, v74
	v_mul_f32_e32 v158, s44, v82
	v_mul_f32_e32 v159, s44, v90
	v_mul_f32_e32 v160, s44, v98
	v_mul_f32_e32 v161, s44, v106
	v_mul_f32_e32 v162, s44, v114
	v_mul_f32_e32 v163, s44, v122
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v168, v140
	v_mul_f32_e32 v141, v169, v141
	v_mul_f32_e32 v142, v170, v142
	v_mul_f32_e32 v143, v171, v143
	v_add_f32_e32 v132, v164, v140
	v_add_f32_e32 v133, v165, v141
	v_add_f32_e32 v134, v166, v142
	v_add_f32_e32 v135, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v168, v144
	v_mul_f32_e32 v145, v169, v145
	v_mul_f32_e32 v146, v170, v146
	v_mul_f32_e32 v147, v171, v147
	v_add_f32_e32 v132, v164, v144
	v_add_f32_e32 v133, v165, v145
	v_add_f32_e32 v134, v166, v146
	v_add_f32_e32 v135, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v168, v148
	v_mul_f32_e32 v149, v169, v149
	v_mul_f32_e32 v150, v170, v150
	v_mul_f32_e32 v151, v171, v151
	v_add_f32_e32 v132, v164, v148
	v_add_f32_e32 v133, v165, v149
	v_add_f32_e32 v134, v166, v150
	v_add_f32_e32 v135, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v173, s[16:19], null offen
	v_mul_f32_e32 v152, v168, v152
	v_mul_f32_e32 v153, v169, v153
	v_mul_f32_e32 v154, v170, v154
	v_mul_f32_e32 v155, v171, v155
	v_add_f32_e32 v132, v164, v152
	v_add_f32_e32 v133, v165, v153
	v_add_f32_e32 v134, v166, v154
	v_add_f32_e32 v135, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v175, s[16:19], null offen
	v_mul_f32_e32 v156, v168, v156
	v_mul_f32_e32 v157, v169, v157
	v_mul_f32_e32 v158, v170, v158
	v_mul_f32_e32 v159, v171, v159
	v_add_f32_e32 v132, v164, v156
	v_add_f32_e32 v133, v165, v157
	v_add_f32_e32 v134, v166, v158
	v_add_f32_e32 v135, v167, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v177, s[16:19], null offen
	v_mul_f32_e32 v160, v168, v160
	v_mul_f32_e32 v161, v169, v161
	v_mul_f32_e32 v162, v170, v162
	v_mul_f32_e32 v163, v171, v163
	v_add_f32_e32 v132, v164, v160
	v_add_f32_e32 v133, v165, v161
	v_add_f32_e32 v134, v166, v162
	v_add_f32_e32 v135, v167, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v179, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[164:167], v138
	ds_load_b128 v[168:171], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v172, v128, s32
	v_lshlrev_b32_e32 v172, 2, v172
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v128, s32
	v_lshlrev_b32_e32 v174, 2, v174
	v_add_lshl_u32 v173, v131, v128, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v128, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v128, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_mul_f32_e32 v140, s44, v3
	v_mul_f32_e32 v141, s44, v11
	v_mul_f32_e32 v142, s44, v19
	v_mul_f32_e32 v143, s44, v27
	v_mul_f32_e32 v144, s44, v35
	v_mul_f32_e32 v145, s44, v43
	v_mul_f32_e32 v146, s44, v51
	v_mul_f32_e32 v147, s44, v59
	v_mul_f32_e32 v148, s44, v67
	v_mul_f32_e32 v149, s44, v75
	v_mul_f32_e32 v150, s44, v83
	v_mul_f32_e32 v151, s44, v91
	v_mul_f32_e32 v152, s44, v99
	v_mul_f32_e32 v153, s44, v107
	v_mul_f32_e32 v154, s44, v115
	v_mul_f32_e32 v155, s44, v123
	v_mul_f32_e32 v156, s44, v4
	v_mul_f32_e32 v157, s44, v12
	v_mul_f32_e32 v158, s44, v20
	v_mul_f32_e32 v159, s44, v28
	v_mul_f32_e32 v160, s44, v36
	v_mul_f32_e32 v161, s44, v44
	v_mul_f32_e32 v162, s44, v52
	v_mul_f32_e32 v163, s44, v60
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v168, v140
	v_mul_f32_e32 v141, v169, v141
	v_mul_f32_e32 v142, v170, v142
	v_mul_f32_e32 v143, v171, v143
	v_add_f32_e32 v132, v164, v140
	v_add_f32_e32 v133, v165, v141
	v_add_f32_e32 v134, v166, v142
	v_add_f32_e32 v135, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v168, v144
	v_mul_f32_e32 v145, v169, v145
	v_mul_f32_e32 v146, v170, v146
	v_mul_f32_e32 v147, v171, v147
	v_add_f32_e32 v132, v164, v144
	v_add_f32_e32 v133, v165, v145
	v_add_f32_e32 v134, v166, v146
	v_add_f32_e32 v135, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v168, v148
	v_mul_f32_e32 v149, v169, v149
	v_mul_f32_e32 v150, v170, v150
	v_mul_f32_e32 v151, v171, v151
	v_add_f32_e32 v132, v164, v148
	v_add_f32_e32 v133, v165, v149
	v_add_f32_e32 v134, v166, v150
	v_add_f32_e32 v135, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v173, s[16:19], null offen
	v_mul_f32_e32 v152, v168, v152
	v_mul_f32_e32 v153, v169, v153
	v_mul_f32_e32 v154, v170, v154
	v_mul_f32_e32 v155, v171, v155
	v_add_f32_e32 v132, v164, v152
	v_add_f32_e32 v133, v165, v153
	v_add_f32_e32 v134, v166, v154
	v_add_f32_e32 v135, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v175, s[16:19], null offen
	v_mul_f32_e32 v156, v168, v156
	v_mul_f32_e32 v157, v169, v157
	v_mul_f32_e32 v158, v170, v158
	v_mul_f32_e32 v159, v171, v159
	v_add_f32_e32 v132, v164, v156
	v_add_f32_e32 v133, v165, v157
	v_add_f32_e32 v134, v166, v158
	v_add_f32_e32 v135, v167, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v177, s[16:19], null offen
	v_mul_f32_e32 v160, v168, v160
	v_mul_f32_e32 v161, v169, v161
	v_mul_f32_e32 v162, v170, v162
	v_mul_f32_e32 v163, v171, v163
	v_add_f32_e32 v132, v164, v160
	v_add_f32_e32 v133, v165, v161
	v_add_f32_e32 v134, v166, v162
	v_add_f32_e32 v135, v167, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v179, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[164:167], v138
	ds_load_b128 v[168:171], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v172, v128, s32
	v_lshlrev_b32_e32 v172, 2, v172
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v128, s32
	v_lshlrev_b32_e32 v174, 2, v174
	v_add_lshl_u32 v173, v131, v128, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v128, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v128, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_mul_f32_e32 v140, s44, v68
	v_mul_f32_e32 v141, s44, v76
	v_mul_f32_e32 v142, s44, v84
	v_mul_f32_e32 v143, s44, v92
	v_mul_f32_e32 v144, s44, v100
	v_mul_f32_e32 v145, s44, v108
	v_mul_f32_e32 v146, s44, v116
	v_mul_f32_e32 v147, s44, v124
	v_mul_f32_e32 v148, s44, v5
	v_mul_f32_e32 v149, s44, v13
	v_mul_f32_e32 v150, s44, v21
	v_mul_f32_e32 v151, s44, v29
	v_mul_f32_e32 v152, s44, v37
	v_mul_f32_e32 v153, s44, v45
	v_mul_f32_e32 v154, s44, v53
	v_mul_f32_e32 v155, s44, v61
	v_mul_f32_e32 v156, s44, v69
	v_mul_f32_e32 v157, s44, v77
	v_mul_f32_e32 v158, s44, v85
	v_mul_f32_e32 v159, s44, v93
	v_mul_f32_e32 v160, s44, v101
	v_mul_f32_e32 v161, s44, v109
	v_mul_f32_e32 v162, s44, v117
	v_mul_f32_e32 v163, s44, v125
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v168, v140
	v_mul_f32_e32 v141, v169, v141
	v_mul_f32_e32 v142, v170, v142
	v_mul_f32_e32 v143, v171, v143
	v_add_f32_e32 v132, v164, v140
	v_add_f32_e32 v133, v165, v141
	v_add_f32_e32 v134, v166, v142
	v_add_f32_e32 v135, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v168, v144
	v_mul_f32_e32 v145, v169, v145
	v_mul_f32_e32 v146, v170, v146
	v_mul_f32_e32 v147, v171, v147
	v_add_f32_e32 v132, v164, v144
	v_add_f32_e32 v133, v165, v145
	v_add_f32_e32 v134, v166, v146
	v_add_f32_e32 v135, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v168, v148
	v_mul_f32_e32 v149, v169, v149
	v_mul_f32_e32 v150, v170, v150
	v_mul_f32_e32 v151, v171, v151
	v_add_f32_e32 v132, v164, v148
	v_add_f32_e32 v133, v165, v149
	v_add_f32_e32 v134, v166, v150
	v_add_f32_e32 v135, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v173, s[16:19], null offen
	v_mul_f32_e32 v152, v168, v152
	v_mul_f32_e32 v153, v169, v153
	v_mul_f32_e32 v154, v170, v154
	v_mul_f32_e32 v155, v171, v155
	v_add_f32_e32 v132, v164, v152
	v_add_f32_e32 v133, v165, v153
	v_add_f32_e32 v134, v166, v154
	v_add_f32_e32 v135, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v175, s[16:19], null offen
	v_mul_f32_e32 v156, v168, v156
	v_mul_f32_e32 v157, v169, v157
	v_mul_f32_e32 v158, v170, v158
	v_mul_f32_e32 v159, v171, v159
	v_add_f32_e32 v132, v164, v156
	v_add_f32_e32 v133, v165, v157
	v_add_f32_e32 v134, v166, v158
	v_add_f32_e32 v135, v167, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v177, s[16:19], null offen
	v_mul_f32_e32 v160, v168, v160
	v_mul_f32_e32 v161, v169, v161
	v_mul_f32_e32 v162, v170, v162
	v_mul_f32_e32 v163, v171, v163
	v_add_f32_e32 v132, v164, v160
	v_add_f32_e32 v133, v165, v161
	v_add_f32_e32 v134, v166, v162
	v_add_f32_e32 v135, v167, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v179, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[164:167], v138
	ds_load_b128 v[168:171], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v172, v128, s32
	v_lshlrev_b32_e32 v172, 2, v172
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v128, s32
	v_lshlrev_b32_e32 v174, 2, v174
	v_add_lshl_u32 v173, v131, v128, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v128, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v128, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_mul_f32_e32 v140, s44, v6
	v_mul_f32_e32 v141, s44, v14
	v_mul_f32_e32 v142, s44, v22
	v_mul_f32_e32 v143, s44, v30
	v_mul_f32_e32 v144, s44, v38
	v_mul_f32_e32 v145, s44, v46
	v_mul_f32_e32 v146, s44, v54
	v_mul_f32_e32 v147, s44, v62
	v_mul_f32_e32 v148, s44, v70
	v_mul_f32_e32 v149, s44, v78
	v_mul_f32_e32 v150, s44, v86
	v_mul_f32_e32 v151, s44, v94
	v_mul_f32_e32 v152, s44, v102
	v_mul_f32_e32 v153, s44, v110
	v_mul_f32_e32 v154, s44, v118
	v_mul_f32_e32 v155, s44, v126
	v_mul_f32_e32 v156, s44, v7
	v_mul_f32_e32 v157, s44, v15
	v_mul_f32_e32 v158, s44, v23
	v_mul_f32_e32 v159, s44, v31
	v_mul_f32_e32 v160, s44, v39
	v_mul_f32_e32 v161, s44, v47
	v_mul_f32_e32 v162, s44, v55
	v_mul_f32_e32 v163, s44, v63
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v168, v140
	v_mul_f32_e32 v141, v169, v141
	v_mul_f32_e32 v142, v170, v142
	v_mul_f32_e32 v143, v171, v143
	v_add_f32_e32 v132, v164, v140
	v_add_f32_e32 v133, v165, v141
	v_add_f32_e32 v134, v166, v142
	v_add_f32_e32 v135, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v168, v144
	v_mul_f32_e32 v145, v169, v145
	v_mul_f32_e32 v146, v170, v146
	v_mul_f32_e32 v147, v171, v147
	v_add_f32_e32 v132, v164, v144
	v_add_f32_e32 v133, v165, v145
	v_add_f32_e32 v134, v166, v146
	v_add_f32_e32 v135, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v168, v148
	v_mul_f32_e32 v149, v169, v149
	v_mul_f32_e32 v150, v170, v150
	v_mul_f32_e32 v151, v171, v151
	v_add_f32_e32 v132, v164, v148
	v_add_f32_e32 v133, v165, v149
	v_add_f32_e32 v134, v166, v150
	v_add_f32_e32 v135, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v173, s[16:19], null offen
	v_mul_f32_e32 v152, v168, v152
	v_mul_f32_e32 v153, v169, v153
	v_mul_f32_e32 v154, v170, v154
	v_mul_f32_e32 v155, v171, v155
	v_add_f32_e32 v132, v164, v152
	v_add_f32_e32 v133, v165, v153
	v_add_f32_e32 v134, v166, v154
	v_add_f32_e32 v135, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v175, s[16:19], null offen
	v_mul_f32_e32 v156, v168, v156
	v_mul_f32_e32 v157, v169, v157
	v_mul_f32_e32 v158, v170, v158
	v_mul_f32_e32 v159, v171, v159
	v_add_f32_e32 v132, v164, v156
	v_add_f32_e32 v133, v165, v157
	v_add_f32_e32 v134, v166, v158
	v_add_f32_e32 v135, v167, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v177, s[16:19], null offen
	v_mul_f32_e32 v160, v168, v160
	v_mul_f32_e32 v161, v169, v161
	v_mul_f32_e32 v162, v170, v162
	v_mul_f32_e32 v163, v171, v163
	v_add_f32_e32 v132, v164, v160
	v_add_f32_e32 v133, v165, v161
	v_add_f32_e32 v134, v166, v162
	v_add_f32_e32 v135, v167, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v179, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[148:151], v138
	ds_load_b128 v[152:155], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v156, v128, s32
	v_lshlrev_b32_e32 v156, 2, v156
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_mul_f32_e32 v140, s44, v71
	v_mul_f32_e32 v141, s44, v79
	v_mul_f32_e32 v142, s44, v87
	v_mul_f32_e32 v143, s44, v95
	v_mul_f32_e32 v144, s44, v103
	v_mul_f32_e32 v145, s44, v111
	v_mul_f32_e32 v146, s44, v119
	v_mul_f32_e32 v147, s44, v127
	s_wait_dscnt 0x0
	v_mul_f32_e32 v140, v152, v140
	v_mul_f32_e32 v141, v153, v141
	v_mul_f32_e32 v142, v154, v142
	v_mul_f32_e32 v143, v155, v143
	v_add_f32_e32 v132, v148, v140
	v_add_f32_e32 v133, v149, v141
	v_add_f32_e32 v134, v150, v142
	v_add_f32_e32 v135, v151, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v152, v144
	v_mul_f32_e32 v145, v153, v145
	v_mul_f32_e32 v146, v154, v146
	v_mul_f32_e32 v147, v155, v147
	v_add_f32_e32 v132, v148, v144
	v_add_f32_e32 v133, v149, v145
	v_add_f32_e32 v134, v150, v146
	v_add_f32_e32 v135, v151, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End_1
label_GW_B0_E1_M_1:
	s_mov_b32 s8, 3
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Gelu_VW1_beta_0_edge_1
	s_mov_b32 s8, 5
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Relu_VW1_beta_0_edge_1
	s_mov_b32 s8, 6
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Sigmoid_VW1_beta_0_edge_1
	s_mov_b32 s8, 10
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Silu_VW1_beta_0_edge_1
	s_mov_b32 s8, 12
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Clamp_VW1_beta_0_edge_1
label_To_Activation_None_VW1_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_14:
	s_add_co_i32 s8, label_Activation_None_VW1-Lgpc_after_14-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_3
label_To_Activation_Gelu_VW1_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_15:
	s_add_co_i32 s8, label_Activation_Gelu_VW1-Lgpc_after_15-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_3
label_To_Activation_Relu_VW1_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_16:
	s_add_co_i32 s8, label_Activation_Relu_VW1-Lgpc_after_16-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_3
label_To_Activation_Sigmoid_VW1_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_17:
	s_add_co_i32 s8, label_Activation_Sigmoid_VW1-Lgpc_after_17-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_3
label_To_Activation_Silu_VW1_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_18:
	s_add_co_i32 s8, label_Activation_Silu_VW1-Lgpc_after_18-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_3
label_To_Activation_Clamp_VW1_beta_0_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_19:
	s_add_co_i32 s8, label_Activation_Clamp_VW1-Lgpc_after_19-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_3
label_ActivationSetPCAddrEnd_3:
	v_mov_b32_e32 v136, 0x80000000
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v162, v128, s32
	v_lshlrev_b32_e32 v162, 2, v162
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b32 v159, v162
	ds_load_b32 v160, v162 offset:512
	v_add_lshl_u32 v161, v131, v128, 2
	v_cndmask_b32_e64 v161, v136, v161, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v166, v132, s32
	v_lshlrev_b32_e32 v166, 2, v166
	ds_load_b32 v163, v166
	ds_load_b32 v164, v166 offset:512
	v_add_lshl_u32 v165, v131, v132, 2
	v_cndmask_b32_e64 v165, v136, v165, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v170, v132, s32
	v_lshlrev_b32_e32 v170, 2, v170
	ds_load_b32 v167, v170
	ds_load_b32 v168, v170 offset:512
	v_add_lshl_u32 v169, v131, v132, 2
	v_cndmask_b32_e64 v169, v136, v169, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v132, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v132, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v132, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v132, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v184, v128, s32
	v_lshlrev_b32_e32 v184, 2, v184
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v188, v132, s32
	v_lshlrev_b32_e32 v188, 2, v188
	v_add_lshl_u32 v187, v131, v132, 2
	v_cndmask_b32_e64 v187, v136, v187, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v190, v132, s32
	v_lshlrev_b32_e32 v190, 2, v190
	v_add_lshl_u32 v189, v131, v132, 2
	v_cndmask_b32_e64 v189, v136, v189, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v128, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v132, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v132, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v196, v132, s32
	v_lshlrev_b32_e32 v196, 2, v196
	v_add_lshl_u32 v195, v131, v132, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v200, v128, s32
	v_lshlrev_b32_e32 v200, 2, v200
	v_add_lshl_u32 v199, v131, v128, 2
	v_cndmask_b32_e64 v199, v136, v199, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v202, v132, s32
	v_lshlrev_b32_e32 v202, 2, v202
	v_add_lshl_u32 v201, v131, v132, 2
	v_cndmask_b32_e64 v201, v136, v201, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v206, v132, s32
	v_lshlrev_b32_e32 v206, 2, v206
	v_add_lshl_u32 v205, v131, v132, 2
	v_cndmask_b32_e64 v205, v136, v205, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v208, v128, s32
	v_lshlrev_b32_e32 v208, 2, v208
	v_add_lshl_u32 v207, v131, v128, 2
	v_cndmask_b32_e64 v207, v136, v207, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_mul_f32_e32 v137, s44, v0
	v_mul_f32_e32 v138, s44, v8
	v_mul_f32_e32 v139, s44, v16
	v_mul_f32_e32 v140, s44, v24
	v_mul_f32_e32 v141, s44, v32
	v_mul_f32_e32 v142, s44, v40
	v_mul_f32_e32 v143, s44, v48
	v_mul_f32_e32 v144, s44, v56
	v_mul_f32_e32 v145, s44, v64
	v_mul_f32_e32 v146, s44, v72
	v_mul_f32_e32 v147, s44, v80
	v_mul_f32_e32 v148, s44, v88
	v_mul_f32_e32 v149, s44, v96
	v_mul_f32_e32 v150, s44, v104
	v_mul_f32_e32 v151, s44, v112
	v_mul_f32_e32 v152, s44, v120
	v_mul_f32_e32 v153, s44, v1
	v_mul_f32_e32 v154, s44, v9
	v_mul_f32_e32 v155, s44, v17
	v_mul_f32_e32 v156, s44, v25
	v_mul_f32_e32 v157, s44, v33
	v_mul_f32_e32 v158, s44, v41
	s_wait_dscnt 0x0
	v_mul_f32_e32 v137, v160, v137
	v_add_f32_e32 v132, v159, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v161, s[16:19], null offen
	v_mul_f32_e32 v138, v164, v138
	v_add_f32_e32 v132, v163, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v165, s[16:19], null offen
	v_mul_f32_e32 v139, v168, v139
	v_add_f32_e32 v132, v167, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v169, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v160, v141
	v_add_f32_e32 v132, v159, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v175, s[16:19], null offen
	v_mul_f32_e32 v142, v164, v142
	v_add_f32_e32 v132, v163, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v177, s[16:19], null offen
	v_mul_f32_e32 v143, v168, v143
	v_add_f32_e32 v132, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v179, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v181, s[16:19], null offen
	v_mul_f32_e32 v145, v160, v145
	v_add_f32_e32 v132, v159, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v183, s[16:19], null offen
	v_mul_f32_e32 v146, v164, v146
	v_add_f32_e32 v132, v163, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v185, s[16:19], null offen
	v_mul_f32_e32 v147, v168, v147
	v_add_f32_e32 v132, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v187, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v189, s[16:19], null offen
	v_mul_f32_e32 v149, v160, v149
	v_add_f32_e32 v132, v159, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v191, s[16:19], null offen
	v_mul_f32_e32 v150, v164, v150
	v_add_f32_e32 v132, v163, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v193, s[16:19], null offen
	v_mul_f32_e32 v151, v168, v151
	v_add_f32_e32 v132, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v195, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v197, s[16:19], null offen
	v_mul_f32_e32 v153, v160, v153
	v_add_f32_e32 v132, v159, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v199, s[16:19], null offen
	v_mul_f32_e32 v154, v164, v154
	v_add_f32_e32 v132, v163, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v201, s[16:19], null offen
	v_mul_f32_e32 v155, v168, v155
	v_add_f32_e32 v132, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v155, v132
	buffer_store_b32 v155, v203, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_add_f32_e32 v132, v171, v156
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	buffer_store_b32 v156, v205, s[16:19], null offen
	v_mul_f32_e32 v157, v160, v157
	v_add_f32_e32 v132, v159, v157
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v157, v132
	buffer_store_b32 v157, v207, s[16:19], null offen
	v_mul_f32_e32 v158, v164, v158
	v_add_f32_e32 v132, v163, v158
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v158, v132
	buffer_store_b32 v158, v209, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v162, v132, s32
	v_lshlrev_b32_e32 v162, 2, v162
	ds_load_b32 v159, v162
	ds_load_b32 v160, v162 offset:512
	v_add_lshl_u32 v161, v131, v132, 2
	v_cndmask_b32_e64 v161, v136, v161, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v166, v132, s32
	v_lshlrev_b32_e32 v166, 2, v166
	ds_load_b32 v163, v166
	ds_load_b32 v164, v166 offset:512
	v_add_lshl_u32 v165, v131, v132, 2
	v_cndmask_b32_e64 v165, v136, v165, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v170, v128, s32
	v_lshlrev_b32_e32 v170, 2, v170
	ds_load_b32 v167, v170
	ds_load_b32 v168, v170 offset:512
	v_add_lshl_u32 v169, v131, v128, 2
	v_cndmask_b32_e64 v169, v136, v169, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v132, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v132, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v132, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v132, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v132, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v132, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v184, v132, s32
	v_lshlrev_b32_e32 v184, 2, v184
	v_add_lshl_u32 v183, v131, v132, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v188, v128, s32
	v_lshlrev_b32_e32 v188, 2, v188
	v_add_lshl_u32 v187, v131, v128, 2
	v_cndmask_b32_e64 v187, v136, v187, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v190, v132, s32
	v_lshlrev_b32_e32 v190, 2, v190
	v_add_lshl_u32 v189, v131, v132, 2
	v_cndmask_b32_e64 v189, v136, v189, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v132, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v132, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v196, v128, s32
	v_lshlrev_b32_e32 v196, 2, v196
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v200, v132, s32
	v_lshlrev_b32_e32 v200, 2, v200
	v_add_lshl_u32 v199, v131, v132, 2
	v_cndmask_b32_e64 v199, v136, v199, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v202, v132, s32
	v_lshlrev_b32_e32 v202, 2, v202
	v_add_lshl_u32 v201, v131, v132, 2
	v_cndmask_b32_e64 v201, v136, v201, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v128, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v128, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v206, v132, s32
	v_lshlrev_b32_e32 v206, 2, v206
	v_add_lshl_u32 v205, v131, v132, 2
	v_cndmask_b32_e64 v205, v136, v205, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v208, v132, s32
	v_lshlrev_b32_e32 v208, 2, v208
	v_add_lshl_u32 v207, v131, v132, 2
	v_cndmask_b32_e64 v207, v136, v207, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_mul_f32_e32 v137, s44, v49
	v_mul_f32_e32 v138, s44, v57
	v_mul_f32_e32 v139, s44, v65
	v_mul_f32_e32 v140, s44, v73
	v_mul_f32_e32 v141, s44, v81
	v_mul_f32_e32 v142, s44, v89
	v_mul_f32_e32 v143, s44, v97
	v_mul_f32_e32 v144, s44, v105
	v_mul_f32_e32 v145, s44, v113
	v_mul_f32_e32 v146, s44, v121
	v_mul_f32_e32 v147, s44, v2
	v_mul_f32_e32 v148, s44, v10
	v_mul_f32_e32 v149, s44, v18
	v_mul_f32_e32 v150, s44, v26
	v_mul_f32_e32 v151, s44, v34
	v_mul_f32_e32 v152, s44, v42
	v_mul_f32_e32 v153, s44, v50
	v_mul_f32_e32 v154, s44, v58
	v_mul_f32_e32 v155, s44, v66
	v_mul_f32_e32 v156, s44, v74
	v_mul_f32_e32 v157, s44, v82
	v_mul_f32_e32 v158, s44, v90
	s_wait_dscnt 0x0
	v_mul_f32_e32 v137, v160, v137
	v_add_f32_e32 v132, v159, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v161, s[16:19], null offen
	v_mul_f32_e32 v138, v164, v138
	v_add_f32_e32 v132, v163, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v165, s[16:19], null offen
	v_mul_f32_e32 v139, v168, v139
	v_add_f32_e32 v132, v167, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v169, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v160, v141
	v_add_f32_e32 v132, v159, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v175, s[16:19], null offen
	v_mul_f32_e32 v142, v164, v142
	v_add_f32_e32 v132, v163, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v177, s[16:19], null offen
	v_mul_f32_e32 v143, v168, v143
	v_add_f32_e32 v132, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v179, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v181, s[16:19], null offen
	v_mul_f32_e32 v145, v160, v145
	v_add_f32_e32 v132, v159, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v183, s[16:19], null offen
	v_mul_f32_e32 v146, v164, v146
	v_add_f32_e32 v132, v163, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v185, s[16:19], null offen
	v_mul_f32_e32 v147, v168, v147
	v_add_f32_e32 v132, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v187, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v189, s[16:19], null offen
	v_mul_f32_e32 v149, v160, v149
	v_add_f32_e32 v132, v159, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v191, s[16:19], null offen
	v_mul_f32_e32 v150, v164, v150
	v_add_f32_e32 v132, v163, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v193, s[16:19], null offen
	v_mul_f32_e32 v151, v168, v151
	v_add_f32_e32 v132, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v195, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v197, s[16:19], null offen
	v_mul_f32_e32 v153, v160, v153
	v_add_f32_e32 v132, v159, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v199, s[16:19], null offen
	v_mul_f32_e32 v154, v164, v154
	v_add_f32_e32 v132, v163, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v201, s[16:19], null offen
	v_mul_f32_e32 v155, v168, v155
	v_add_f32_e32 v132, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v155, v132
	buffer_store_b32 v155, v203, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_add_f32_e32 v132, v171, v156
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	buffer_store_b32 v156, v205, s[16:19], null offen
	v_mul_f32_e32 v157, v160, v157
	v_add_f32_e32 v132, v159, v157
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v157, v132
	buffer_store_b32 v157, v207, s[16:19], null offen
	v_mul_f32_e32 v158, v164, v158
	v_add_f32_e32 v132, v163, v158
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v158, v132
	buffer_store_b32 v158, v209, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v162, v128, s32
	v_lshlrev_b32_e32 v162, 2, v162
	ds_load_b32 v159, v162
	ds_load_b32 v160, v162 offset:512
	v_add_lshl_u32 v161, v131, v128, 2
	v_cndmask_b32_e64 v161, v136, v161, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v166, v132, s32
	v_lshlrev_b32_e32 v166, 2, v166
	ds_load_b32 v163, v166
	ds_load_b32 v164, v166 offset:512
	v_add_lshl_u32 v165, v131, v132, 2
	v_cndmask_b32_e64 v165, v136, v165, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v170, v132, s32
	v_lshlrev_b32_e32 v170, 2, v170
	ds_load_b32 v167, v170
	ds_load_b32 v168, v170 offset:512
	v_add_lshl_u32 v169, v131, v132, 2
	v_cndmask_b32_e64 v169, v136, v169, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v132, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v132, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v132, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v132, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v184, v128, s32
	v_lshlrev_b32_e32 v184, 2, v184
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v188, v132, s32
	v_lshlrev_b32_e32 v188, 2, v188
	v_add_lshl_u32 v187, v131, v132, 2
	v_cndmask_b32_e64 v187, v136, v187, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v190, v132, s32
	v_lshlrev_b32_e32 v190, 2, v190
	v_add_lshl_u32 v189, v131, v132, 2
	v_cndmask_b32_e64 v189, v136, v189, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v128, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v132, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v132, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v196, v132, s32
	v_lshlrev_b32_e32 v196, 2, v196
	v_add_lshl_u32 v195, v131, v132, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v200, v128, s32
	v_lshlrev_b32_e32 v200, 2, v200
	v_add_lshl_u32 v199, v131, v128, 2
	v_cndmask_b32_e64 v199, v136, v199, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v202, v132, s32
	v_lshlrev_b32_e32 v202, 2, v202
	v_add_lshl_u32 v201, v131, v132, 2
	v_cndmask_b32_e64 v201, v136, v201, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v206, v132, s32
	v_lshlrev_b32_e32 v206, 2, v206
	v_add_lshl_u32 v205, v131, v132, 2
	v_cndmask_b32_e64 v205, v136, v205, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v208, v128, s32
	v_lshlrev_b32_e32 v208, 2, v208
	v_add_lshl_u32 v207, v131, v128, 2
	v_cndmask_b32_e64 v207, v136, v207, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_mul_f32_e32 v137, s44, v98
	v_mul_f32_e32 v138, s44, v106
	v_mul_f32_e32 v139, s44, v114
	v_mul_f32_e32 v140, s44, v122
	v_mul_f32_e32 v141, s44, v3
	v_mul_f32_e32 v142, s44, v11
	v_mul_f32_e32 v143, s44, v19
	v_mul_f32_e32 v144, s44, v27
	v_mul_f32_e32 v145, s44, v35
	v_mul_f32_e32 v146, s44, v43
	v_mul_f32_e32 v147, s44, v51
	v_mul_f32_e32 v148, s44, v59
	v_mul_f32_e32 v149, s44, v67
	v_mul_f32_e32 v150, s44, v75
	v_mul_f32_e32 v151, s44, v83
	v_mul_f32_e32 v152, s44, v91
	v_mul_f32_e32 v153, s44, v99
	v_mul_f32_e32 v154, s44, v107
	v_mul_f32_e32 v155, s44, v115
	v_mul_f32_e32 v156, s44, v123
	v_mul_f32_e32 v157, s44, v4
	v_mul_f32_e32 v158, s44, v12
	s_wait_dscnt 0x0
	v_mul_f32_e32 v137, v160, v137
	v_add_f32_e32 v132, v159, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v161, s[16:19], null offen
	v_mul_f32_e32 v138, v164, v138
	v_add_f32_e32 v132, v163, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v165, s[16:19], null offen
	v_mul_f32_e32 v139, v168, v139
	v_add_f32_e32 v132, v167, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v169, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v160, v141
	v_add_f32_e32 v132, v159, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v175, s[16:19], null offen
	v_mul_f32_e32 v142, v164, v142
	v_add_f32_e32 v132, v163, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v177, s[16:19], null offen
	v_mul_f32_e32 v143, v168, v143
	v_add_f32_e32 v132, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v179, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v181, s[16:19], null offen
	v_mul_f32_e32 v145, v160, v145
	v_add_f32_e32 v132, v159, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v183, s[16:19], null offen
	v_mul_f32_e32 v146, v164, v146
	v_add_f32_e32 v132, v163, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v185, s[16:19], null offen
	v_mul_f32_e32 v147, v168, v147
	v_add_f32_e32 v132, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v187, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v189, s[16:19], null offen
	v_mul_f32_e32 v149, v160, v149
	v_add_f32_e32 v132, v159, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v191, s[16:19], null offen
	v_mul_f32_e32 v150, v164, v150
	v_add_f32_e32 v132, v163, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v193, s[16:19], null offen
	v_mul_f32_e32 v151, v168, v151
	v_add_f32_e32 v132, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v195, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v197, s[16:19], null offen
	v_mul_f32_e32 v153, v160, v153
	v_add_f32_e32 v132, v159, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v199, s[16:19], null offen
	v_mul_f32_e32 v154, v164, v154
	v_add_f32_e32 v132, v163, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v201, s[16:19], null offen
	v_mul_f32_e32 v155, v168, v155
	v_add_f32_e32 v132, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v155, v132
	buffer_store_b32 v155, v203, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_add_f32_e32 v132, v171, v156
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	buffer_store_b32 v156, v205, s[16:19], null offen
	v_mul_f32_e32 v157, v160, v157
	v_add_f32_e32 v132, v159, v157
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v157, v132
	buffer_store_b32 v157, v207, s[16:19], null offen
	v_mul_f32_e32 v158, v164, v158
	v_add_f32_e32 v132, v163, v158
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v158, v132
	buffer_store_b32 v158, v209, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v162, v132, s32
	v_lshlrev_b32_e32 v162, 2, v162
	ds_load_b32 v159, v162
	ds_load_b32 v160, v162 offset:512
	v_add_lshl_u32 v161, v131, v132, 2
	v_cndmask_b32_e64 v161, v136, v161, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v166, v132, s32
	v_lshlrev_b32_e32 v166, 2, v166
	ds_load_b32 v163, v166
	ds_load_b32 v164, v166 offset:512
	v_add_lshl_u32 v165, v131, v132, 2
	v_cndmask_b32_e64 v165, v136, v165, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v170, v128, s32
	v_lshlrev_b32_e32 v170, 2, v170
	ds_load_b32 v167, v170
	ds_load_b32 v168, v170 offset:512
	v_add_lshl_u32 v169, v131, v128, 2
	v_cndmask_b32_e64 v169, v136, v169, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v132, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v132, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v132, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v132, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v128, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v132, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v132, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v184, v132, s32
	v_lshlrev_b32_e32 v184, 2, v184
	v_add_lshl_u32 v183, v131, v132, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v188, v128, s32
	v_lshlrev_b32_e32 v188, 2, v188
	v_add_lshl_u32 v187, v131, v128, 2
	v_cndmask_b32_e64 v187, v136, v187, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v190, v132, s32
	v_lshlrev_b32_e32 v190, 2, v190
	v_add_lshl_u32 v189, v131, v132, 2
	v_cndmask_b32_e64 v189, v136, v189, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v132, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v132, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v196, v128, s32
	v_lshlrev_b32_e32 v196, 2, v196
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v200, v132, s32
	v_lshlrev_b32_e32 v200, 2, v200
	v_add_lshl_u32 v199, v131, v132, 2
	v_cndmask_b32_e64 v199, v136, v199, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v202, v132, s32
	v_lshlrev_b32_e32 v202, 2, v202
	v_add_lshl_u32 v201, v131, v132, 2
	v_cndmask_b32_e64 v201, v136, v201, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v128, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v128, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v206, v132, s32
	v_lshlrev_b32_e32 v206, 2, v206
	v_add_lshl_u32 v205, v131, v132, 2
	v_cndmask_b32_e64 v205, v136, v205, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v208, v132, s32
	v_lshlrev_b32_e32 v208, 2, v208
	v_add_lshl_u32 v207, v131, v132, 2
	v_cndmask_b32_e64 v207, v136, v207, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_mul_f32_e32 v137, s44, v20
	v_mul_f32_e32 v138, s44, v28
	v_mul_f32_e32 v139, s44, v36
	v_mul_f32_e32 v140, s44, v44
	v_mul_f32_e32 v141, s44, v52
	v_mul_f32_e32 v142, s44, v60
	v_mul_f32_e32 v143, s44, v68
	v_mul_f32_e32 v144, s44, v76
	v_mul_f32_e32 v145, s44, v84
	v_mul_f32_e32 v146, s44, v92
	v_mul_f32_e32 v147, s44, v100
	v_mul_f32_e32 v148, s44, v108
	v_mul_f32_e32 v149, s44, v116
	v_mul_f32_e32 v150, s44, v124
	v_mul_f32_e32 v151, s44, v5
	v_mul_f32_e32 v152, s44, v13
	v_mul_f32_e32 v153, s44, v21
	v_mul_f32_e32 v154, s44, v29
	v_mul_f32_e32 v155, s44, v37
	v_mul_f32_e32 v156, s44, v45
	v_mul_f32_e32 v157, s44, v53
	v_mul_f32_e32 v158, s44, v61
	s_wait_dscnt 0x0
	v_mul_f32_e32 v137, v160, v137
	v_add_f32_e32 v132, v159, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v161, s[16:19], null offen
	v_mul_f32_e32 v138, v164, v138
	v_add_f32_e32 v132, v163, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v165, s[16:19], null offen
	v_mul_f32_e32 v139, v168, v139
	v_add_f32_e32 v132, v167, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v169, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v160, v141
	v_add_f32_e32 v132, v159, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v175, s[16:19], null offen
	v_mul_f32_e32 v142, v164, v142
	v_add_f32_e32 v132, v163, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v177, s[16:19], null offen
	v_mul_f32_e32 v143, v168, v143
	v_add_f32_e32 v132, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v179, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v181, s[16:19], null offen
	v_mul_f32_e32 v145, v160, v145
	v_add_f32_e32 v132, v159, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v183, s[16:19], null offen
	v_mul_f32_e32 v146, v164, v146
	v_add_f32_e32 v132, v163, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v185, s[16:19], null offen
	v_mul_f32_e32 v147, v168, v147
	v_add_f32_e32 v132, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v187, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v189, s[16:19], null offen
	v_mul_f32_e32 v149, v160, v149
	v_add_f32_e32 v132, v159, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v191, s[16:19], null offen
	v_mul_f32_e32 v150, v164, v150
	v_add_f32_e32 v132, v163, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v193, s[16:19], null offen
	v_mul_f32_e32 v151, v168, v151
	v_add_f32_e32 v132, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v195, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v197, s[16:19], null offen
	v_mul_f32_e32 v153, v160, v153
	v_add_f32_e32 v132, v159, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v199, s[16:19], null offen
	v_mul_f32_e32 v154, v164, v154
	v_add_f32_e32 v132, v163, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v201, s[16:19], null offen
	v_mul_f32_e32 v155, v168, v155
	v_add_f32_e32 v132, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v155, v132
	buffer_store_b32 v155, v203, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_add_f32_e32 v132, v171, v156
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	buffer_store_b32 v156, v205, s[16:19], null offen
	v_mul_f32_e32 v157, v160, v157
	v_add_f32_e32 v132, v159, v157
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v157, v132
	buffer_store_b32 v157, v207, s[16:19], null offen
	v_mul_f32_e32 v158, v164, v158
	v_add_f32_e32 v132, v163, v158
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v158, v132
	buffer_store_b32 v158, v209, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v162, v128, s32
	v_lshlrev_b32_e32 v162, 2, v162
	ds_load_b32 v159, v162
	ds_load_b32 v160, v162 offset:512
	v_add_lshl_u32 v161, v131, v128, 2
	v_cndmask_b32_e64 v161, v136, v161, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v166, v132, s32
	v_lshlrev_b32_e32 v166, 2, v166
	ds_load_b32 v163, v166
	ds_load_b32 v164, v166 offset:512
	v_add_lshl_u32 v165, v131, v132, 2
	v_cndmask_b32_e64 v165, v136, v165, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v170, v132, s32
	v_lshlrev_b32_e32 v170, 2, v170
	ds_load_b32 v167, v170
	ds_load_b32 v168, v170 offset:512
	v_add_lshl_u32 v169, v131, v132, 2
	v_cndmask_b32_e64 v169, v136, v169, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v132, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v132, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v132, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v132, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v184, v128, s32
	v_lshlrev_b32_e32 v184, 2, v184
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v188, v132, s32
	v_lshlrev_b32_e32 v188, 2, v188
	v_add_lshl_u32 v187, v131, v132, 2
	v_cndmask_b32_e64 v187, v136, v187, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v190, v132, s32
	v_lshlrev_b32_e32 v190, 2, v190
	v_add_lshl_u32 v189, v131, v132, 2
	v_cndmask_b32_e64 v189, v136, v189, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v128, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v132, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v132, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v196, v132, s32
	v_lshlrev_b32_e32 v196, 2, v196
	v_add_lshl_u32 v195, v131, v132, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v200, v128, s32
	v_lshlrev_b32_e32 v200, 2, v200
	v_add_lshl_u32 v199, v131, v128, 2
	v_cndmask_b32_e64 v199, v136, v199, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v202, v132, s32
	v_lshlrev_b32_e32 v202, 2, v202
	v_add_lshl_u32 v201, v131, v132, 2
	v_cndmask_b32_e64 v201, v136, v201, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v206, v132, s32
	v_lshlrev_b32_e32 v206, 2, v206
	v_add_lshl_u32 v205, v131, v132, 2
	v_cndmask_b32_e64 v205, v136, v205, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v208, v128, s32
	v_lshlrev_b32_e32 v208, 2, v208
	v_add_lshl_u32 v207, v131, v128, 2
	v_cndmask_b32_e64 v207, v136, v207, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_mul_f32_e32 v137, s44, v69
	v_mul_f32_e32 v138, s44, v77
	v_mul_f32_e32 v139, s44, v85
	v_mul_f32_e32 v140, s44, v93
	v_mul_f32_e32 v141, s44, v101
	v_mul_f32_e32 v142, s44, v109
	v_mul_f32_e32 v143, s44, v117
	v_mul_f32_e32 v144, s44, v125
	v_mul_f32_e32 v145, s44, v6
	v_mul_f32_e32 v146, s44, v14
	v_mul_f32_e32 v147, s44, v22
	v_mul_f32_e32 v148, s44, v30
	v_mul_f32_e32 v149, s44, v38
	v_mul_f32_e32 v150, s44, v46
	v_mul_f32_e32 v151, s44, v54
	v_mul_f32_e32 v152, s44, v62
	v_mul_f32_e32 v153, s44, v70
	v_mul_f32_e32 v154, s44, v78
	v_mul_f32_e32 v155, s44, v86
	v_mul_f32_e32 v156, s44, v94
	v_mul_f32_e32 v157, s44, v102
	v_mul_f32_e32 v158, s44, v110
	s_wait_dscnt 0x0
	v_mul_f32_e32 v137, v160, v137
	v_add_f32_e32 v132, v159, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v161, s[16:19], null offen
	v_mul_f32_e32 v138, v164, v138
	v_add_f32_e32 v132, v163, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v165, s[16:19], null offen
	v_mul_f32_e32 v139, v168, v139
	v_add_f32_e32 v132, v167, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v169, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v160, v141
	v_add_f32_e32 v132, v159, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v175, s[16:19], null offen
	v_mul_f32_e32 v142, v164, v142
	v_add_f32_e32 v132, v163, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v177, s[16:19], null offen
	v_mul_f32_e32 v143, v168, v143
	v_add_f32_e32 v132, v167, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v179, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v181, s[16:19], null offen
	v_mul_f32_e32 v145, v160, v145
	v_add_f32_e32 v132, v159, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v183, s[16:19], null offen
	v_mul_f32_e32 v146, v164, v146
	v_add_f32_e32 v132, v163, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v185, s[16:19], null offen
	v_mul_f32_e32 v147, v168, v147
	v_add_f32_e32 v132, v167, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v187, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v189, s[16:19], null offen
	v_mul_f32_e32 v149, v160, v149
	v_add_f32_e32 v132, v159, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v191, s[16:19], null offen
	v_mul_f32_e32 v150, v164, v150
	v_add_f32_e32 v132, v163, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v193, s[16:19], null offen
	v_mul_f32_e32 v151, v168, v151
	v_add_f32_e32 v132, v167, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v195, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v197, s[16:19], null offen
	v_mul_f32_e32 v153, v160, v153
	v_add_f32_e32 v132, v159, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v199, s[16:19], null offen
	v_mul_f32_e32 v154, v164, v154
	v_add_f32_e32 v132, v163, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v201, s[16:19], null offen
	v_mul_f32_e32 v155, v168, v155
	v_add_f32_e32 v132, v167, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v155, v132
	buffer_store_b32 v155, v203, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_add_f32_e32 v132, v171, v156
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	buffer_store_b32 v156, v205, s[16:19], null offen
	v_mul_f32_e32 v157, v160, v157
	v_add_f32_e32 v132, v159, v157
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v157, v132
	buffer_store_b32 v157, v207, s[16:19], null offen
	v_mul_f32_e32 v158, v164, v158
	v_add_f32_e32 v132, v163, v158
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v158, v132
	buffer_store_b32 v158, v209, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v158, v132, s32
	v_lshlrev_b32_e32 v158, 2, v158
	ds_load_b32 v155, v158
	ds_load_b32 v156, v158 offset:512
	v_add_lshl_u32 v157, v131, v132, 2
	v_cndmask_b32_e64 v157, v136, v157, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v162, v132, s32
	v_lshlrev_b32_e32 v162, 2, v162
	ds_load_b32 v159, v162
	ds_load_b32 v160, v162 offset:512
	v_add_lshl_u32 v161, v131, v132, 2
	v_cndmask_b32_e64 v161, v136, v161, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v166, v128, s32
	v_lshlrev_b32_e32 v166, 2, v166
	ds_load_b32 v163, v166
	ds_load_b32 v164, v166 offset:512
	v_add_lshl_u32 v165, v131, v128, 2
	v_cndmask_b32_e64 v165, v136, v165, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v170, v132, s32
	v_lshlrev_b32_e32 v170, 2, v170
	ds_load_b32 v167, v170
	ds_load_b32 v168, v170 offset:512
	v_add_lshl_u32 v169, v131, v132, 2
	v_cndmask_b32_e64 v169, v136, v169, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v172, v132, s32
	v_lshlrev_b32_e32 v172, 2, v172
	v_add_lshl_u32 v171, v131, v132, 2
	v_cndmask_b32_e64 v171, v136, v171, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v176, v128, s32
	v_lshlrev_b32_e32 v176, 2, v176
	v_add_lshl_u32 v175, v131, v128, 2
	v_cndmask_b32_e64 v175, v136, v175, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v178, v132, s32
	v_lshlrev_b32_e32 v178, 2, v178
	v_add_lshl_u32 v177, v131, v132, 2
	v_cndmask_b32_e64 v177, v136, v177, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v132, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v132, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v184, v128, s32
	v_lshlrev_b32_e32 v184, 2, v184
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v188, v132, s32
	v_lshlrev_b32_e32 v188, 2, v188
	v_add_lshl_u32 v187, v131, v132, 2
	v_cndmask_b32_e64 v187, v136, v187, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v190, v132, s32
	v_lshlrev_b32_e32 v190, 2, v190
	v_add_lshl_u32 v189, v131, v132, 2
	v_cndmask_b32_e64 v189, v136, v189, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v128, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v132, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v132, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v196, v132, s32
	v_lshlrev_b32_e32 v196, 2, v196
	v_add_lshl_u32 v195, v131, v132, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_mul_f32_e32 v137, s44, v118
	v_mul_f32_e32 v138, s44, v126
	v_mul_f32_e32 v139, s44, v7
	v_mul_f32_e32 v140, s44, v15
	v_mul_f32_e32 v141, s44, v23
	v_mul_f32_e32 v142, s44, v31
	v_mul_f32_e32 v143, s44, v39
	v_mul_f32_e32 v144, s44, v47
	v_mul_f32_e32 v145, s44, v55
	v_mul_f32_e32 v146, s44, v63
	v_mul_f32_e32 v147, s44, v71
	v_mul_f32_e32 v148, s44, v79
	v_mul_f32_e32 v149, s44, v87
	v_mul_f32_e32 v150, s44, v95
	v_mul_f32_e32 v151, s44, v103
	v_mul_f32_e32 v152, s44, v111
	v_mul_f32_e32 v153, s44, v119
	v_mul_f32_e32 v154, s44, v127
	s_wait_dscnt 0x0
	v_mul_f32_e32 v137, v156, v137
	v_add_f32_e32 v132, v155, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v157, s[16:19], null offen
	v_mul_f32_e32 v138, v160, v138
	v_add_f32_e32 v132, v159, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v161, s[16:19], null offen
	v_mul_f32_e32 v139, v164, v139
	v_add_f32_e32 v132, v163, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v165, s[16:19], null offen
	v_mul_f32_e32 v140, v168, v140
	v_add_f32_e32 v132, v167, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v169, s[16:19], null offen
	v_mul_f32_e32 v141, v156, v141
	v_add_f32_e32 v132, v155, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v171, s[16:19], null offen
	v_mul_f32_e32 v142, v160, v142
	v_add_f32_e32 v132, v159, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v173, s[16:19], null offen
	v_mul_f32_e32 v143, v164, v143
	v_add_f32_e32 v132, v163, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v175, s[16:19], null offen
	v_mul_f32_e32 v144, v168, v144
	v_add_f32_e32 v132, v167, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v177, s[16:19], null offen
	v_mul_f32_e32 v145, v156, v145
	v_add_f32_e32 v132, v155, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v179, s[16:19], null offen
	v_mul_f32_e32 v146, v160, v146
	v_add_f32_e32 v132, v159, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v181, s[16:19], null offen
	v_mul_f32_e32 v147, v164, v147
	v_add_f32_e32 v132, v163, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v183, s[16:19], null offen
	v_mul_f32_e32 v148, v168, v148
	v_add_f32_e32 v132, v167, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v185, s[16:19], null offen
	v_mul_f32_e32 v149, v156, v149
	v_add_f32_e32 v132, v155, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v187, s[16:19], null offen
	v_mul_f32_e32 v150, v160, v150
	v_add_f32_e32 v132, v159, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v189, s[16:19], null offen
	v_mul_f32_e32 v151, v164, v151
	v_add_f32_e32 v132, v163, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v191, s[16:19], null offen
	v_mul_f32_e32 v152, v168, v152
	v_add_f32_e32 v132, v167, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v193, s[16:19], null offen
	v_mul_f32_e32 v153, v156, v153
	v_add_f32_e32 v132, v155, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v195, s[16:19], null offen
	v_mul_f32_e32 v154, v160, v154
	v_add_f32_e32 v132, v159, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v197, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End_1
label_GW_Beta_1:
	s_and_b32 s32, 0x7f, s24
	s_add_co_u32 s33, -1, s14
	s_cmp_ge_u32 s2, s33
	s_cselect_b32 s32, s32, 0
	s_mov_b32 s8, 0
	s_cmp_gt_u32 s32, s8
	s_cbranch_scc1 label_GW_B1_E1_M
	s_and_b32 s32, 0x7f, s25
	s_add_co_u32 s33, -1, s15
	s_cmp_ge_u32 s3, s33
	s_cselect_b32 s32, s32, 0
	s_mov_b32 s8, 0
	s_cmp_gt_u32 s32, s8
	s_cbranch_scc1 label_GW_B1_E1_N
label_GW_B1_E0:
	s_mov_b32 s8, 3
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Gelu_VW4_beta_1_edge_0
	s_mov_b32 s8, 5
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Relu_VW4_beta_1_edge_0
	s_mov_b32 s8, 6
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Sigmoid_VW4_beta_1_edge_0
	s_mov_b32 s8, 10
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Silu_VW4_beta_1_edge_0
	s_mov_b32 s8, 12
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Clamp_VW4_beta_1_edge_0
label_To_Activation_None_VW4_beta_1_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_20:
	s_add_co_i32 s8, label_Activation_None_VW4-Lgpc_after_20-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_2
label_To_Activation_Gelu_VW4_beta_1_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_21:
	s_add_co_i32 s8, label_Activation_Gelu_VW4-Lgpc_after_21-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_2
label_To_Activation_Relu_VW4_beta_1_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_22:
	s_add_co_i32 s8, label_Activation_Relu_VW4-Lgpc_after_22-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_2
label_To_Activation_Sigmoid_VW4_beta_1_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_23:
	s_add_co_i32 s8, label_Activation_Sigmoid_VW4-Lgpc_after_23-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_2
label_To_Activation_Silu_VW4_beta_1_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_24:
	s_add_co_i32 s8, label_Activation_Silu_VW4-Lgpc_after_24-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_2
label_To_Activation_Clamp_VW4_beta_1_edge_0:
	s_getpc_b64 s[12:13]
Lgpc_after_25:
	s_add_co_i32 s8, label_Activation_Clamp_VW4-Lgpc_after_25-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_2
label_ActivationSetPCAddrEnd_2:
	v_add_lshl_u32 v138, v130, v128, 2
	buffer_load_b128 v[164:167], v138, s[20:23], null offen
	s_mul_i32 s8, 0x80, s2
	v_sub_nc_u32_e64 v139, v128, s8
	v_lshlrev_b32_e32 v139, 2, v139
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[168:171], v139
	ds_load_b128 v[172:175], v139 offset:512
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[176:179], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[180:183], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[184:187], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[188:191], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[192:195], v138, s[20:23], null offen
	v_add_lshl_u32 v137, v131, v128, 2
	v_mul_f32_e32 v140, s44, v0
	v_mul_f32_e32 v141, s44, v8
	v_mul_f32_e32 v142, s44, v16
	v_mul_f32_e32 v143, s44, v24
	v_mul_f32_e32 v144, s44, v32
	v_mul_f32_e32 v145, s44, v40
	v_mul_f32_e32 v146, s44, v48
	v_mul_f32_e32 v147, s44, v56
	v_mul_f32_e32 v148, s44, v64
	v_mul_f32_e32 v149, s44, v72
	v_mul_f32_e32 v150, s44, v80
	v_mul_f32_e32 v151, s44, v88
	v_mul_f32_e32 v152, s44, v96
	v_mul_f32_e32 v153, s44, v104
	v_mul_f32_e32 v154, s44, v112
	v_mul_f32_e32 v155, s44, v120
	v_mul_f32_e32 v156, s44, v1
	v_mul_f32_e32 v157, s44, v9
	v_mul_f32_e32 v158, s44, v17
	v_mul_f32_e32 v159, s44, v25
	v_mul_f32_e32 v160, s44, v33
	v_mul_f32_e32 v161, s44, v41
	v_mul_f32_e32 v162, s44, v49
	v_mul_f32_e32 v163, s44, v57
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v180, s45
	v_fmac_f32_e64 v149, v181, s45
	v_fmac_f32_e64 v150, v182, s45
	v_fmac_f32_e64 v151, v183, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v184, s45
	v_fmac_f32_e64 v153, v185, s45
	v_fmac_f32_e64 v154, v186, s45
	v_fmac_f32_e64 v155, v187, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v188, s45
	v_fmac_f32_e64 v157, v189, s45
	v_fmac_f32_e64 v158, v190, s45
	v_fmac_f32_e64 v159, v191, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v192, s45
	v_fmac_f32_e64 v161, v193, s45
	v_fmac_f32_e64 v162, v194, s45
	v_fmac_f32_e64 v163, v195, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	s_nop 0
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[164:167], v138, s[20:23], null offen
	ds_load_b128 v[168:171], v139
	ds_load_b128 v[172:175], v139 offset:512
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[176:179], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[180:183], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[184:187], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[188:191], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[192:195], v138, s[20:23], null offen
	v_mul_f32_e32 v140, s44, v65
	v_mul_f32_e32 v141, s44, v73
	v_mul_f32_e32 v142, s44, v81
	v_mul_f32_e32 v143, s44, v89
	v_mul_f32_e32 v144, s44, v97
	v_mul_f32_e32 v145, s44, v105
	v_mul_f32_e32 v146, s44, v113
	v_mul_f32_e32 v147, s44, v121
	v_mul_f32_e32 v148, s44, v2
	v_mul_f32_e32 v149, s44, v10
	v_mul_f32_e32 v150, s44, v18
	v_mul_f32_e32 v151, s44, v26
	v_mul_f32_e32 v152, s44, v34
	v_mul_f32_e32 v153, s44, v42
	v_mul_f32_e32 v154, s44, v50
	v_mul_f32_e32 v155, s44, v58
	v_mul_f32_e32 v156, s44, v66
	v_mul_f32_e32 v157, s44, v74
	v_mul_f32_e32 v158, s44, v82
	v_mul_f32_e32 v159, s44, v90
	v_mul_f32_e32 v160, s44, v98
	v_mul_f32_e32 v161, s44, v106
	v_mul_f32_e32 v162, s44, v114
	v_mul_f32_e32 v163, s44, v122
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v180, s45
	v_fmac_f32_e64 v149, v181, s45
	v_fmac_f32_e64 v150, v182, s45
	v_fmac_f32_e64 v151, v183, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v184, s45
	v_fmac_f32_e64 v153, v185, s45
	v_fmac_f32_e64 v154, v186, s45
	v_fmac_f32_e64 v155, v187, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v188, s45
	v_fmac_f32_e64 v157, v189, s45
	v_fmac_f32_e64 v158, v190, s45
	v_fmac_f32_e64 v159, v191, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v192, s45
	v_fmac_f32_e64 v161, v193, s45
	v_fmac_f32_e64 v162, v194, s45
	v_fmac_f32_e64 v163, v195, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	s_nop 0
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[164:167], v138, s[20:23], null offen
	ds_load_b128 v[168:171], v139
	ds_load_b128 v[172:175], v139 offset:512
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[176:179], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[180:183], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[184:187], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[188:191], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[192:195], v138, s[20:23], null offen
	v_mul_f32_e32 v140, s44, v3
	v_mul_f32_e32 v141, s44, v11
	v_mul_f32_e32 v142, s44, v19
	v_mul_f32_e32 v143, s44, v27
	v_mul_f32_e32 v144, s44, v35
	v_mul_f32_e32 v145, s44, v43
	v_mul_f32_e32 v146, s44, v51
	v_mul_f32_e32 v147, s44, v59
	v_mul_f32_e32 v148, s44, v67
	v_mul_f32_e32 v149, s44, v75
	v_mul_f32_e32 v150, s44, v83
	v_mul_f32_e32 v151, s44, v91
	v_mul_f32_e32 v152, s44, v99
	v_mul_f32_e32 v153, s44, v107
	v_mul_f32_e32 v154, s44, v115
	v_mul_f32_e32 v155, s44, v123
	v_mul_f32_e32 v156, s44, v4
	v_mul_f32_e32 v157, s44, v12
	v_mul_f32_e32 v158, s44, v20
	v_mul_f32_e32 v159, s44, v28
	v_mul_f32_e32 v160, s44, v36
	v_mul_f32_e32 v161, s44, v44
	v_mul_f32_e32 v162, s44, v52
	v_mul_f32_e32 v163, s44, v60
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v180, s45
	v_fmac_f32_e64 v149, v181, s45
	v_fmac_f32_e64 v150, v182, s45
	v_fmac_f32_e64 v151, v183, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v184, s45
	v_fmac_f32_e64 v153, v185, s45
	v_fmac_f32_e64 v154, v186, s45
	v_fmac_f32_e64 v155, v187, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v188, s45
	v_fmac_f32_e64 v157, v189, s45
	v_fmac_f32_e64 v158, v190, s45
	v_fmac_f32_e64 v159, v191, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v192, s45
	v_fmac_f32_e64 v161, v193, s45
	v_fmac_f32_e64 v162, v194, s45
	v_fmac_f32_e64 v163, v195, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	s_nop 0
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[164:167], v138, s[20:23], null offen
	ds_load_b128 v[168:171], v139
	ds_load_b128 v[172:175], v139 offset:512
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[176:179], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[180:183], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[184:187], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[188:191], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[192:195], v138, s[20:23], null offen
	v_mul_f32_e32 v140, s44, v68
	v_mul_f32_e32 v141, s44, v76
	v_mul_f32_e32 v142, s44, v84
	v_mul_f32_e32 v143, s44, v92
	v_mul_f32_e32 v144, s44, v100
	v_mul_f32_e32 v145, s44, v108
	v_mul_f32_e32 v146, s44, v116
	v_mul_f32_e32 v147, s44, v124
	v_mul_f32_e32 v148, s44, v5
	v_mul_f32_e32 v149, s44, v13
	v_mul_f32_e32 v150, s44, v21
	v_mul_f32_e32 v151, s44, v29
	v_mul_f32_e32 v152, s44, v37
	v_mul_f32_e32 v153, s44, v45
	v_mul_f32_e32 v154, s44, v53
	v_mul_f32_e32 v155, s44, v61
	v_mul_f32_e32 v156, s44, v69
	v_mul_f32_e32 v157, s44, v77
	v_mul_f32_e32 v158, s44, v85
	v_mul_f32_e32 v159, s44, v93
	v_mul_f32_e32 v160, s44, v101
	v_mul_f32_e32 v161, s44, v109
	v_mul_f32_e32 v162, s44, v117
	v_mul_f32_e32 v163, s44, v125
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v180, s45
	v_fmac_f32_e64 v149, v181, s45
	v_fmac_f32_e64 v150, v182, s45
	v_fmac_f32_e64 v151, v183, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v184, s45
	v_fmac_f32_e64 v153, v185, s45
	v_fmac_f32_e64 v154, v186, s45
	v_fmac_f32_e64 v155, v187, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v188, s45
	v_fmac_f32_e64 v157, v189, s45
	v_fmac_f32_e64 v158, v190, s45
	v_fmac_f32_e64 v159, v191, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v192, s45
	v_fmac_f32_e64 v161, v193, s45
	v_fmac_f32_e64 v162, v194, s45
	v_fmac_f32_e64 v163, v195, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	s_nop 0
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[164:167], v138, s[20:23], null offen
	ds_load_b128 v[168:171], v139
	ds_load_b128 v[172:175], v139 offset:512
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[176:179], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[180:183], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[184:187], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[188:191], v138, s[20:23], null offen
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[192:195], v138, s[20:23], null offen
	v_mul_f32_e32 v140, s44, v6
	v_mul_f32_e32 v141, s44, v14
	v_mul_f32_e32 v142, s44, v22
	v_mul_f32_e32 v143, s44, v30
	v_mul_f32_e32 v144, s44, v38
	v_mul_f32_e32 v145, s44, v46
	v_mul_f32_e32 v146, s44, v54
	v_mul_f32_e32 v147, s44, v62
	v_mul_f32_e32 v148, s44, v70
	v_mul_f32_e32 v149, s44, v78
	v_mul_f32_e32 v150, s44, v86
	v_mul_f32_e32 v151, s44, v94
	v_mul_f32_e32 v152, s44, v102
	v_mul_f32_e32 v153, s44, v110
	v_mul_f32_e32 v154, s44, v118
	v_mul_f32_e32 v155, s44, v126
	v_mul_f32_e32 v156, s44, v7
	v_mul_f32_e32 v157, s44, v15
	v_mul_f32_e32 v158, s44, v23
	v_mul_f32_e32 v159, s44, v31
	v_mul_f32_e32 v160, s44, v39
	v_mul_f32_e32 v161, s44, v47
	v_mul_f32_e32 v162, s44, v55
	v_mul_f32_e32 v163, s44, v63
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x5
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v180, s45
	v_fmac_f32_e64 v149, v181, s45
	v_fmac_f32_e64 v150, v182, s45
	v_fmac_f32_e64 v151, v183, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[148:151], v137, s[16:19], null offen
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v184, s45
	v_fmac_f32_e64 v153, v185, s45
	v_fmac_f32_e64 v154, v186, s45
	v_fmac_f32_e64 v155, v187, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[152:155], v137, s[16:19], null offen
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v188, s45
	v_fmac_f32_e64 v157, v189, s45
	v_fmac_f32_e64 v158, v190, s45
	v_fmac_f32_e64 v159, v191, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[156:159], v137, s[16:19], null offen
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v192, s45
	v_fmac_f32_e64 v161, v193, s45
	v_fmac_f32_e64 v162, v194, s45
	v_fmac_f32_e64 v163, v195, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[160:163], v137, s[16:19], null offen
	s_nop 0
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[148:151], v138, s[20:23], null offen
	ds_load_b128 v[152:155], v139
	ds_load_b128 v[156:159], v139 offset:512
	s_lshl_b32 s8, s38, 2
	s_add_co_u32 s20, s20, s8
	s_add_co_ci_u32 s21, s21, 0
	buffer_load_b128 v[160:163], v138, s[20:23], null offen
	v_mul_f32_e32 v140, s44, v71
	v_mul_f32_e32 v141, s44, v79
	v_mul_f32_e32 v142, s44, v87
	v_mul_f32_e32 v143, s44, v95
	v_mul_f32_e32 v144, s44, v103
	v_mul_f32_e32 v145, s44, v111
	v_mul_f32_e32 v146, s44, v119
	v_mul_f32_e32 v147, s44, v127
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v140, v156, v140
	v_mul_f32_e32 v141, v157, v141
	v_mul_f32_e32 v142, v158, v142
	v_mul_f32_e32 v143, v159, v143
	v_fmac_f32_e64 v140, v148, s45
	v_fmac_f32_e64 v141, v149, s45
	v_fmac_f32_e64 v142, v150, s45
	v_fmac_f32_e64 v143, v151, s45
	v_add_f32_e32 v132, v152, v140
	v_add_f32_e32 v133, v153, v141
	v_add_f32_e32 v134, v154, v142
	v_add_f32_e32 v135, v155, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v144, v156, v144
	v_mul_f32_e32 v145, v157, v145
	v_mul_f32_e32 v146, v158, v146
	v_mul_f32_e32 v147, v159, v147
	v_fmac_f32_e64 v144, v160, s45
	v_fmac_f32_e64 v145, v161, s45
	v_fmac_f32_e64 v146, v162, s45
	v_fmac_f32_e64 v147, v163, s45
	v_add_f32_e32 v132, v152, v144
	v_add_f32_e32 v133, v153, v145
	v_add_f32_e32 v134, v154, v146
	v_add_f32_e32 v135, v155, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	s_lshl_b32 s8, s36, 2
	s_add_co_u32 s16, s16, s8
	s_add_co_ci_u32 s17, s17, 0
	buffer_store_b128 v[144:147], v137, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End_1
label_GW_B1_E1_N:
	s_mov_b32 s8, 3
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Gelu_VW4_beta_1_edge_1
	s_mov_b32 s8, 5
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Relu_VW4_beta_1_edge_1
	s_mov_b32 s8, 6
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Sigmoid_VW4_beta_1_edge_1
	s_mov_b32 s8, 10
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Silu_VW4_beta_1_edge_1
	s_mov_b32 s8, 12
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Clamp_VW4_beta_1_edge_1
label_To_Activation_None_VW4_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_26:
	s_add_co_i32 s8, label_Activation_None_VW4-Lgpc_after_26-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Gelu_VW4_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_27:
	s_add_co_i32 s8, label_Activation_Gelu_VW4-Lgpc_after_27-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Relu_VW4_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_28:
	s_add_co_i32 s8, label_Activation_Relu_VW4-Lgpc_after_28-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Sigmoid_VW4_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_29:
	s_add_co_i32 s8, label_Activation_Sigmoid_VW4-Lgpc_after_29-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Silu_VW4_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_30:
	s_add_co_i32 s8, label_Activation_Silu_VW4-Lgpc_after_30-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_1
label_To_Activation_Clamp_VW4_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_31:
	s_add_co_i32 s8, label_Activation_Clamp_VW4-Lgpc_after_31-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd_1
label_ActivationSetPCAddrEnd_1:
	v_mov_b32_e32 v136, 0x80000000
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v137, v130, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	buffer_load_b128 v[164:167], v137, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b128 v[168:171], v138
	ds_load_b128 v[172:175], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v139, v130, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	buffer_load_b128 v[176:179], v139, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v181, v130, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	buffer_load_b128 v[184:187], v181, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v128, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v183, v130, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	buffer_load_b128 v[188:191], v183, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v193, v130, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	buffer_load_b128 v[196:199], v193, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v128, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v195, v130, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	buffer_load_b128 v[200:203], v195, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v128, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_mul_f32_e32 v140, s44, v0
	v_mul_f32_e32 v141, s44, v8
	v_mul_f32_e32 v142, s44, v16
	v_mul_f32_e32 v143, s44, v24
	v_mul_f32_e32 v144, s44, v32
	v_mul_f32_e32 v145, s44, v40
	v_mul_f32_e32 v146, s44, v48
	v_mul_f32_e32 v147, s44, v56
	v_mul_f32_e32 v148, s44, v64
	v_mul_f32_e32 v149, s44, v72
	v_mul_f32_e32 v150, s44, v80
	v_mul_f32_e32 v151, s44, v88
	v_mul_f32_e32 v152, s44, v96
	v_mul_f32_e32 v153, s44, v104
	v_mul_f32_e32 v154, s44, v112
	v_mul_f32_e32 v155, s44, v120
	v_mul_f32_e32 v156, s44, v1
	v_mul_f32_e32 v157, s44, v9
	v_mul_f32_e32 v158, s44, v17
	v_mul_f32_e32 v159, s44, v25
	v_mul_f32_e32 v160, s44, v33
	v_mul_f32_e32 v161, s44, v41
	v_mul_f32_e32 v162, s44, v49
	v_mul_f32_e32 v163, s44, v57
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v184, s45
	v_fmac_f32_e64 v149, v185, s45
	v_fmac_f32_e64 v150, v186, s45
	v_fmac_f32_e64 v151, v187, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v181, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v188, s45
	v_fmac_f32_e64 v153, v189, s45
	v_fmac_f32_e64 v154, v190, s45
	v_fmac_f32_e64 v155, v191, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v183, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v196, s45
	v_fmac_f32_e64 v157, v197, s45
	v_fmac_f32_e64 v158, v198, s45
	v_fmac_f32_e64 v159, v199, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v193, s[16:19], null offen
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v200, s45
	v_fmac_f32_e64 v161, v201, s45
	v_fmac_f32_e64 v162, v202, s45
	v_fmac_f32_e64 v163, v203, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v195, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v137, v130, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	buffer_load_b128 v[164:167], v137, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[168:171], v138
	ds_load_b128 v[172:175], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v139, v130, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	buffer_load_b128 v[176:179], v139, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v181, v130, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	buffer_load_b128 v[184:187], v181, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v128, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v183, v130, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	buffer_load_b128 v[188:191], v183, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v193, v130, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	buffer_load_b128 v[196:199], v193, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v128, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v195, v130, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	buffer_load_b128 v[200:203], v195, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v128, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_mul_f32_e32 v140, s44, v65
	v_mul_f32_e32 v141, s44, v73
	v_mul_f32_e32 v142, s44, v81
	v_mul_f32_e32 v143, s44, v89
	v_mul_f32_e32 v144, s44, v97
	v_mul_f32_e32 v145, s44, v105
	v_mul_f32_e32 v146, s44, v113
	v_mul_f32_e32 v147, s44, v121
	v_mul_f32_e32 v148, s44, v2
	v_mul_f32_e32 v149, s44, v10
	v_mul_f32_e32 v150, s44, v18
	v_mul_f32_e32 v151, s44, v26
	v_mul_f32_e32 v152, s44, v34
	v_mul_f32_e32 v153, s44, v42
	v_mul_f32_e32 v154, s44, v50
	v_mul_f32_e32 v155, s44, v58
	v_mul_f32_e32 v156, s44, v66
	v_mul_f32_e32 v157, s44, v74
	v_mul_f32_e32 v158, s44, v82
	v_mul_f32_e32 v159, s44, v90
	v_mul_f32_e32 v160, s44, v98
	v_mul_f32_e32 v161, s44, v106
	v_mul_f32_e32 v162, s44, v114
	v_mul_f32_e32 v163, s44, v122
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v184, s45
	v_fmac_f32_e64 v149, v185, s45
	v_fmac_f32_e64 v150, v186, s45
	v_fmac_f32_e64 v151, v187, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v181, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v188, s45
	v_fmac_f32_e64 v153, v189, s45
	v_fmac_f32_e64 v154, v190, s45
	v_fmac_f32_e64 v155, v191, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v183, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v196, s45
	v_fmac_f32_e64 v157, v197, s45
	v_fmac_f32_e64 v158, v198, s45
	v_fmac_f32_e64 v159, v199, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v193, s[16:19], null offen
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v200, s45
	v_fmac_f32_e64 v161, v201, s45
	v_fmac_f32_e64 v162, v202, s45
	v_fmac_f32_e64 v163, v203, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v195, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v137, v130, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	buffer_load_b128 v[164:167], v137, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[168:171], v138
	ds_load_b128 v[172:175], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v139, v130, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	buffer_load_b128 v[176:179], v139, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v181, v130, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	buffer_load_b128 v[184:187], v181, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v128, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v183, v130, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	buffer_load_b128 v[188:191], v183, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v193, v130, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	buffer_load_b128 v[196:199], v193, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v128, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v195, v130, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	buffer_load_b128 v[200:203], v195, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v128, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_mul_f32_e32 v140, s44, v3
	v_mul_f32_e32 v141, s44, v11
	v_mul_f32_e32 v142, s44, v19
	v_mul_f32_e32 v143, s44, v27
	v_mul_f32_e32 v144, s44, v35
	v_mul_f32_e32 v145, s44, v43
	v_mul_f32_e32 v146, s44, v51
	v_mul_f32_e32 v147, s44, v59
	v_mul_f32_e32 v148, s44, v67
	v_mul_f32_e32 v149, s44, v75
	v_mul_f32_e32 v150, s44, v83
	v_mul_f32_e32 v151, s44, v91
	v_mul_f32_e32 v152, s44, v99
	v_mul_f32_e32 v153, s44, v107
	v_mul_f32_e32 v154, s44, v115
	v_mul_f32_e32 v155, s44, v123
	v_mul_f32_e32 v156, s44, v4
	v_mul_f32_e32 v157, s44, v12
	v_mul_f32_e32 v158, s44, v20
	v_mul_f32_e32 v159, s44, v28
	v_mul_f32_e32 v160, s44, v36
	v_mul_f32_e32 v161, s44, v44
	v_mul_f32_e32 v162, s44, v52
	v_mul_f32_e32 v163, s44, v60
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v184, s45
	v_fmac_f32_e64 v149, v185, s45
	v_fmac_f32_e64 v150, v186, s45
	v_fmac_f32_e64 v151, v187, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v181, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v188, s45
	v_fmac_f32_e64 v153, v189, s45
	v_fmac_f32_e64 v154, v190, s45
	v_fmac_f32_e64 v155, v191, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v183, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v196, s45
	v_fmac_f32_e64 v157, v197, s45
	v_fmac_f32_e64 v158, v198, s45
	v_fmac_f32_e64 v159, v199, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v193, s[16:19], null offen
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v200, s45
	v_fmac_f32_e64 v161, v201, s45
	v_fmac_f32_e64 v162, v202, s45
	v_fmac_f32_e64 v163, v203, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v195, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v137, v130, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	buffer_load_b128 v[164:167], v137, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[168:171], v138
	ds_load_b128 v[172:175], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v139, v130, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	buffer_load_b128 v[176:179], v139, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v181, v130, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	buffer_load_b128 v[184:187], v181, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v128, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v183, v130, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	buffer_load_b128 v[188:191], v183, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v193, v130, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	buffer_load_b128 v[196:199], v193, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v128, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v195, v130, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	buffer_load_b128 v[200:203], v195, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v128, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_mul_f32_e32 v140, s44, v68
	v_mul_f32_e32 v141, s44, v76
	v_mul_f32_e32 v142, s44, v84
	v_mul_f32_e32 v143, s44, v92
	v_mul_f32_e32 v144, s44, v100
	v_mul_f32_e32 v145, s44, v108
	v_mul_f32_e32 v146, s44, v116
	v_mul_f32_e32 v147, s44, v124
	v_mul_f32_e32 v148, s44, v5
	v_mul_f32_e32 v149, s44, v13
	v_mul_f32_e32 v150, s44, v21
	v_mul_f32_e32 v151, s44, v29
	v_mul_f32_e32 v152, s44, v37
	v_mul_f32_e32 v153, s44, v45
	v_mul_f32_e32 v154, s44, v53
	v_mul_f32_e32 v155, s44, v61
	v_mul_f32_e32 v156, s44, v69
	v_mul_f32_e32 v157, s44, v77
	v_mul_f32_e32 v158, s44, v85
	v_mul_f32_e32 v159, s44, v93
	v_mul_f32_e32 v160, s44, v101
	v_mul_f32_e32 v161, s44, v109
	v_mul_f32_e32 v162, s44, v117
	v_mul_f32_e32 v163, s44, v125
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v184, s45
	v_fmac_f32_e64 v149, v185, s45
	v_fmac_f32_e64 v150, v186, s45
	v_fmac_f32_e64 v151, v187, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v181, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v188, s45
	v_fmac_f32_e64 v153, v189, s45
	v_fmac_f32_e64 v154, v190, s45
	v_fmac_f32_e64 v155, v191, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v183, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v196, s45
	v_fmac_f32_e64 v157, v197, s45
	v_fmac_f32_e64 v158, v198, s45
	v_fmac_f32_e64 v159, v199, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v193, s[16:19], null offen
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v200, s45
	v_fmac_f32_e64 v161, v201, s45
	v_fmac_f32_e64 v162, v202, s45
	v_fmac_f32_e64 v163, v203, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v195, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v137, v130, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	buffer_load_b128 v[164:167], v137, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[168:171], v138
	ds_load_b128 v[172:175], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v139, v130, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	buffer_load_b128 v[176:179], v139, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v128, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v181, v130, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	buffer_load_b128 v[184:187], v181, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v182, v128, s32
	v_lshlrev_b32_e32 v182, 2, v182
	v_add_lshl_u32 v181, v131, v128, 2
	v_cndmask_b32_e64 v181, v136, v181, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v183, v130, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	buffer_load_b128 v[188:191], v183, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v128, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v183, v131, v128, 2
	v_cndmask_b32_e64 v183, v136, v183, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v193, v130, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	buffer_load_b128 v[196:199], v193, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v194, v128, s32
	v_lshlrev_b32_e32 v194, 2, v194
	v_add_lshl_u32 v193, v131, v128, 2
	v_cndmask_b32_e64 v193, v136, v193, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v195, v130, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	buffer_load_b128 v[200:203], v195, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v128, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v195, v131, v128, 2
	v_cndmask_b32_e64 v195, v136, v195, s34
	v_mul_f32_e32 v140, s44, v6
	v_mul_f32_e32 v141, s44, v14
	v_mul_f32_e32 v142, s44, v22
	v_mul_f32_e32 v143, s44, v30
	v_mul_f32_e32 v144, s44, v38
	v_mul_f32_e32 v145, s44, v46
	v_mul_f32_e32 v146, s44, v54
	v_mul_f32_e32 v147, s44, v62
	v_mul_f32_e32 v148, s44, v70
	v_mul_f32_e32 v149, s44, v78
	v_mul_f32_e32 v150, s44, v86
	v_mul_f32_e32 v151, s44, v94
	v_mul_f32_e32 v152, s44, v102
	v_mul_f32_e32 v153, s44, v110
	v_mul_f32_e32 v154, s44, v118
	v_mul_f32_e32 v155, s44, v126
	v_mul_f32_e32 v156, s44, v7
	v_mul_f32_e32 v157, s44, v15
	v_mul_f32_e32 v158, s44, v23
	v_mul_f32_e32 v159, s44, v31
	v_mul_f32_e32 v160, s44, v39
	v_mul_f32_e32 v161, s44, v47
	v_mul_f32_e32 v162, s44, v55
	v_mul_f32_e32 v163, s44, v63
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v140, v172, v140
	v_mul_f32_e32 v141, v173, v141
	v_mul_f32_e32 v142, v174, v142
	v_mul_f32_e32 v143, v175, v143
	v_fmac_f32_e64 v140, v164, s45
	v_fmac_f32_e64 v141, v165, s45
	v_fmac_f32_e64 v142, v166, s45
	v_fmac_f32_e64 v143, v167, s45
	v_add_f32_e32 v132, v168, v140
	v_add_f32_e32 v133, v169, v141
	v_add_f32_e32 v134, v170, v142
	v_add_f32_e32 v135, v171, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_mul_f32_e32 v145, v173, v145
	v_mul_f32_e32 v146, v174, v146
	v_mul_f32_e32 v147, v175, v147
	v_fmac_f32_e64 v144, v176, s45
	v_fmac_f32_e64 v145, v177, s45
	v_fmac_f32_e64 v146, v178, s45
	v_fmac_f32_e64 v147, v179, s45
	v_add_f32_e32 v132, v168, v144
	v_add_f32_e32 v133, v169, v145
	v_add_f32_e32 v134, v170, v146
	v_add_f32_e32 v135, v171, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_mul_f32_e32 v149, v173, v149
	v_mul_f32_e32 v150, v174, v150
	v_mul_f32_e32 v151, v175, v151
	v_fmac_f32_e64 v148, v184, s45
	v_fmac_f32_e64 v149, v185, s45
	v_fmac_f32_e64 v150, v186, s45
	v_fmac_f32_e64 v151, v187, s45
	v_add_f32_e32 v132, v168, v148
	v_add_f32_e32 v133, v169, v149
	v_add_f32_e32 v134, v170, v150
	v_add_f32_e32 v135, v171, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	v_mov_b32_e32 v149, v133
	v_mov_b32_e32 v150, v134
	v_mov_b32_e32 v151, v135
	buffer_store_b128 v[148:151], v181, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_mul_f32_e32 v153, v173, v153
	v_mul_f32_e32 v154, v174, v154
	v_mul_f32_e32 v155, v175, v155
	v_fmac_f32_e64 v152, v188, s45
	v_fmac_f32_e64 v153, v189, s45
	v_fmac_f32_e64 v154, v190, s45
	v_fmac_f32_e64 v155, v191, s45
	v_add_f32_e32 v132, v168, v152
	v_add_f32_e32 v133, v169, v153
	v_add_f32_e32 v134, v170, v154
	v_add_f32_e32 v135, v171, v155
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	v_mov_b32_e32 v153, v133
	v_mov_b32_e32 v154, v134
	v_mov_b32_e32 v155, v135
	buffer_store_b128 v[152:155], v183, s[16:19], null offen
	v_mul_f32_e32 v156, v172, v156
	v_mul_f32_e32 v157, v173, v157
	v_mul_f32_e32 v158, v174, v158
	v_mul_f32_e32 v159, v175, v159
	v_fmac_f32_e64 v156, v196, s45
	v_fmac_f32_e64 v157, v197, s45
	v_fmac_f32_e64 v158, v198, s45
	v_fmac_f32_e64 v159, v199, s45
	v_add_f32_e32 v132, v168, v156
	v_add_f32_e32 v133, v169, v157
	v_add_f32_e32 v134, v170, v158
	v_add_f32_e32 v135, v171, v159
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v156, v132
	v_mov_b32_e32 v157, v133
	v_mov_b32_e32 v158, v134
	v_mov_b32_e32 v159, v135
	buffer_store_b128 v[156:159], v193, s[16:19], null offen
	v_mul_f32_e32 v160, v172, v160
	v_mul_f32_e32 v161, v173, v161
	v_mul_f32_e32 v162, v174, v162
	v_mul_f32_e32 v163, v175, v163
	v_fmac_f32_e64 v160, v200, s45
	v_fmac_f32_e64 v161, v201, s45
	v_fmac_f32_e64 v162, v202, s45
	v_fmac_f32_e64 v163, v203, s45
	v_add_f32_e32 v132, v168, v160
	v_add_f32_e32 v133, v169, v161
	v_add_f32_e32 v134, v170, v162
	v_add_f32_e32 v135, v171, v163
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v160, v132
	v_mov_b32_e32 v161, v133
	v_mov_b32_e32 v162, v134
	v_mov_b32_e32 v163, v135
	buffer_store_b128 v[160:163], v195, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v137, v130, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	buffer_load_b128 v[148:151], v137, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v138, v128, s32
	v_lshlrev_b32_e32 v138, 2, v138
	ds_load_b128 v[152:155], v138
	ds_load_b128 v[156:159], v138 offset:512
	v_add_lshl_u32 v137, v131, v128, 2
	v_cndmask_b32_e64 v137, v136, v137, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v139, v130, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	buffer_load_b128 v[160:163], v139, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v128, s32
	v_lshlrev_b32_e32 v164, 2, v164
	v_add_lshl_u32 v139, v131, v128, 2
	v_cndmask_b32_e64 v139, v136, v139, s34
	v_mul_f32_e32 v140, s44, v71
	v_mul_f32_e32 v141, s44, v79
	v_mul_f32_e32 v142, s44, v87
	v_mul_f32_e32 v143, s44, v95
	v_mul_f32_e32 v144, s44, v103
	v_mul_f32_e32 v145, s44, v111
	v_mul_f32_e32 v146, s44, v119
	v_mul_f32_e32 v147, s44, v127
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v140, v156, v140
	v_mul_f32_e32 v141, v157, v141
	v_mul_f32_e32 v142, v158, v142
	v_mul_f32_e32 v143, v159, v143
	v_fmac_f32_e64 v140, v148, s45
	v_fmac_f32_e64 v141, v149, s45
	v_fmac_f32_e64 v142, v150, s45
	v_fmac_f32_e64 v143, v151, s45
	v_add_f32_e32 v132, v152, v140
	v_add_f32_e32 v133, v153, v141
	v_add_f32_e32 v134, v154, v142
	v_add_f32_e32 v135, v155, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	v_mov_b32_e32 v141, v133
	v_mov_b32_e32 v142, v134
	v_mov_b32_e32 v143, v135
	buffer_store_b128 v[140:143], v137, s[16:19], null offen
	v_mul_f32_e32 v144, v156, v144
	v_mul_f32_e32 v145, v157, v145
	v_mul_f32_e32 v146, v158, v146
	v_mul_f32_e32 v147, v159, v147
	v_fmac_f32_e64 v144, v160, s45
	v_fmac_f32_e64 v145, v161, s45
	v_fmac_f32_e64 v146, v162, s45
	v_fmac_f32_e64 v147, v163, s45
	v_add_f32_e32 v132, v152, v144
	v_add_f32_e32 v133, v153, v145
	v_add_f32_e32 v134, v154, v146
	v_add_f32_e32 v135, v155, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	v_mov_b32_e32 v145, v133
	v_mov_b32_e32 v146, v134
	v_mov_b32_e32 v147, v135
	buffer_store_b128 v[144:147], v139, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End_1
label_GW_B1_E1_M:
	s_mov_b32 s8, 3
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Gelu_VW1_beta_1_edge_1
	s_mov_b32 s8, 5
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Relu_VW1_beta_1_edge_1
	s_mov_b32 s8, 6
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Sigmoid_VW1_beta_1_edge_1
	s_mov_b32 s8, 10
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Silu_VW1_beta_1_edge_1
	s_mov_b32 s8, 12
	s_cmp_eq_u32 s56, s8
	s_cbranch_scc1 label_To_Activation_Clamp_VW1_beta_1_edge_1
label_To_Activation_None_VW1_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_32:
	s_add_co_i32 s8, label_Activation_None_VW1-Lgpc_after_32-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Gelu_VW1_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_33:
	s_add_co_i32 s8, label_Activation_Gelu_VW1-Lgpc_after_33-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Relu_VW1_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_34:
	s_add_co_i32 s8, label_Activation_Relu_VW1-Lgpc_after_34-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Sigmoid_VW1_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_35:
	s_add_co_i32 s8, label_Activation_Sigmoid_VW1-Lgpc_after_35-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Silu_VW1_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_36:
	s_add_co_i32 s8, label_Activation_Silu_VW1-Lgpc_after_36-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd
label_To_Activation_Clamp_VW1_beta_1_edge_1:
	s_getpc_b64 s[12:13]
Lgpc_after_37:
	s_add_co_i32 s8, label_Activation_Clamp_VW1-Lgpc_after_37-4, 4
	s_add_co_u32 s12, s12, s8
	s_add_co_ci_u32 s13, s13, 0
	s_branch label_ActivationSetPCAddrEnd
label_ActivationSetPCAddrEnd:
	v_mov_b32_e32 v136, 0x80000000
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v158, v130, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	buffer_load_b32 v155, v158, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v159, v128, s32
	v_lshlrev_b32_e32 v159, 2, v159
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b32 v156, v159
	ds_load_b32 v157, v159 offset:512
	v_add_lshl_u32 v158, v131, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v163, v130, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	buffer_load_b32 v160, v163, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v132, s32
	v_lshlrev_b32_e32 v164, 2, v164
	ds_load_b32 v161, v164
	ds_load_b32 v162, v164 offset:512
	v_add_lshl_u32 v163, v131, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v168, v130, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	buffer_load_b32 v165, v168, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v169, v132, s32
	v_lshlrev_b32_e32 v169, 2, v169
	ds_load_b32 v166, v169
	ds_load_b32 v167, v169 offset:512
	v_add_lshl_u32 v168, v131, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v173, v130, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	buffer_load_b32 v170, v173, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v176, v130, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	buffer_load_b32 v175, v176, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v177, v128, s32
	v_lshlrev_b32_e32 v177, 2, v177
	v_add_lshl_u32 v176, v131, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v179, v130, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	buffer_load_b32 v178, v179, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v182, v130, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	buffer_load_b32 v181, v182, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v183, v132, s32
	v_lshlrev_b32_e32 v183, 2, v183
	v_add_lshl_u32 v182, v131, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v185, v130, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	buffer_load_b32 v184, v185, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v188, v130, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	buffer_load_b32 v187, v188, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v189, v128, s32
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v188, v131, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v191, v130, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	buffer_load_b32 v190, v191, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v194, v130, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	buffer_load_b32 v193, v194, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v195, v132, s32
	v_lshlrev_b32_e32 v195, 2, v195
	v_add_lshl_u32 v194, v131, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v197, v130, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	buffer_load_b32 v196, v197, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v200, v130, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	buffer_load_b32 v199, v200, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v201, v128, s32
	v_lshlrev_b32_e32 v201, 2, v201
	v_add_lshl_u32 v200, v131, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v203, v130, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	buffer_load_b32 v202, v203, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v206, v130, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	buffer_load_b32 v205, v206, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v207, v132, s32
	v_lshlrev_b32_e32 v207, 2, v207
	v_add_lshl_u32 v206, v131, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v209, v130, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	buffer_load_b32 v208, v209, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v212, v130, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	buffer_load_b32 v211, v212, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v213, v128, s32
	v_lshlrev_b32_e32 v213, 2, v213
	v_add_lshl_u32 v212, v131, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v215, v130, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	buffer_load_b32 v214, v215, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v216, v132, s32
	v_lshlrev_b32_e32 v216, 2, v216
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	v_mul_f32_e32 v137, s44, v0
	v_mul_f32_e32 v138, s44, v8
	v_mul_f32_e32 v139, s44, v16
	v_mul_f32_e32 v140, s44, v24
	v_mul_f32_e32 v141, s44, v32
	v_mul_f32_e32 v142, s44, v40
	v_mul_f32_e32 v143, s44, v48
	v_mul_f32_e32 v144, s44, v56
	v_mul_f32_e32 v145, s44, v64
	v_mul_f32_e32 v146, s44, v72
	v_mul_f32_e32 v147, s44, v80
	v_mul_f32_e32 v148, s44, v88
	v_mul_f32_e32 v149, s44, v96
	v_mul_f32_e32 v150, s44, v104
	v_mul_f32_e32 v151, s44, v112
	v_mul_f32_e32 v152, s44, v120
	v_mul_f32_e32 v153, s44, v1
	v_mul_f32_e32 v154, s44, v9
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v157, v137
	v_fmac_f32_e64 v137, v155, s45
	v_add_f32_e32 v132, v156, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v158, s[16:19], null offen
	v_mul_f32_e32 v138, v162, v138
	v_fmac_f32_e64 v138, v160, s45
	v_add_f32_e32 v132, v161, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v163, s[16:19], null offen
	v_mul_f32_e32 v139, v167, v139
	v_fmac_f32_e64 v139, v165, s45
	v_add_f32_e32 v132, v166, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v168, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_fmac_f32_e64 v140, v170, s45
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v157, v141
	v_fmac_f32_e64 v141, v175, s45
	v_add_f32_e32 v132, v156, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v176, s[16:19], null offen
	v_mul_f32_e32 v142, v162, v142
	v_fmac_f32_e64 v142, v178, s45
	v_add_f32_e32 v132, v161, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v179, s[16:19], null offen
	v_mul_f32_e32 v143, v167, v143
	v_fmac_f32_e64 v143, v181, s45
	v_add_f32_e32 v132, v166, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v182, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_fmac_f32_e64 v144, v184, s45
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v185, s[16:19], null offen
	v_mul_f32_e32 v145, v157, v145
	v_fmac_f32_e64 v145, v187, s45
	v_add_f32_e32 v132, v156, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v188, s[16:19], null offen
	v_mul_f32_e32 v146, v162, v146
	v_fmac_f32_e64 v146, v190, s45
	v_add_f32_e32 v132, v161, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v191, s[16:19], null offen
	v_mul_f32_e32 v147, v167, v147
	v_fmac_f32_e64 v147, v193, s45
	v_add_f32_e32 v132, v166, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v194, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_fmac_f32_e64 v148, v196, s45
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v197, s[16:19], null offen
	v_mul_f32_e32 v149, v157, v149
	v_fmac_f32_e64 v149, v199, s45
	v_add_f32_e32 v132, v156, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v200, s[16:19], null offen
	v_mul_f32_e32 v150, v162, v150
	v_fmac_f32_e64 v150, v202, s45
	v_add_f32_e32 v132, v161, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v203, s[16:19], null offen
	v_mul_f32_e32 v151, v167, v151
	v_fmac_f32_e64 v151, v205, s45
	v_add_f32_e32 v132, v166, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v206, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_fmac_f32_e64 v152, v208, s45
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v209, s[16:19], null offen
	v_mul_f32_e32 v153, v157, v153
	v_fmac_f32_e64 v153, v211, s45
	v_add_f32_e32 v132, v156, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v212, s[16:19], null offen
	v_mul_f32_e32 v154, v162, v154
	v_fmac_f32_e64 v154, v214, s45
	v_add_f32_e32 v132, v161, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v215, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v158, v130, v132, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	buffer_load_b32 v155, v158, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v159, v132, s32
	v_lshlrev_b32_e32 v159, 2, v159
	ds_load_b32 v156, v159
	ds_load_b32 v157, v159 offset:512
	v_add_lshl_u32 v158, v131, v132, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v163, v130, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	buffer_load_b32 v160, v163, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v132, s32
	v_lshlrev_b32_e32 v164, 2, v164
	ds_load_b32 v161, v164
	ds_load_b32 v162, v164 offset:512
	v_add_lshl_u32 v163, v131, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v168, v130, v128, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	buffer_load_b32 v165, v168, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v169, v128, s32
	v_lshlrev_b32_e32 v169, 2, v169
	ds_load_b32 v166, v169
	ds_load_b32 v167, v169 offset:512
	v_add_lshl_u32 v168, v131, v128, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v173, v130, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	buffer_load_b32 v170, v173, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v176, v130, v132, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	buffer_load_b32 v175, v176, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v177, v132, s32
	v_lshlrev_b32_e32 v177, 2, v177
	v_add_lshl_u32 v176, v131, v132, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v179, v130, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	buffer_load_b32 v178, v179, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v182, v130, v128, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	buffer_load_b32 v181, v182, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v183, v128, s32
	v_lshlrev_b32_e32 v183, 2, v183
	v_add_lshl_u32 v182, v131, v128, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v185, v130, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	buffer_load_b32 v184, v185, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v188, v130, v132, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	buffer_load_b32 v187, v188, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v189, v132, s32
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v188, v131, v132, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v191, v130, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	buffer_load_b32 v190, v191, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v194, v130, v128, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	buffer_load_b32 v193, v194, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v195, v128, s32
	v_lshlrev_b32_e32 v195, 2, v195
	v_add_lshl_u32 v194, v131, v128, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v197, v130, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	buffer_load_b32 v196, v197, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v200, v130, v132, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	buffer_load_b32 v199, v200, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v201, v132, s32
	v_lshlrev_b32_e32 v201, 2, v201
	v_add_lshl_u32 v200, v131, v132, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v203, v130, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	buffer_load_b32 v202, v203, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v206, v130, v128, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	buffer_load_b32 v205, v206, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v207, v128, s32
	v_lshlrev_b32_e32 v207, 2, v207
	v_add_lshl_u32 v206, v131, v128, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v209, v130, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	buffer_load_b32 v208, v209, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v212, v130, v132, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	buffer_load_b32 v211, v212, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v213, v132, s32
	v_lshlrev_b32_e32 v213, 2, v213
	v_add_lshl_u32 v212, v131, v132, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v215, v130, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	buffer_load_b32 v214, v215, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v216, v132, s32
	v_lshlrev_b32_e32 v216, 2, v216
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	v_mul_f32_e32 v137, s44, v17
	v_mul_f32_e32 v138, s44, v25
	v_mul_f32_e32 v139, s44, v33
	v_mul_f32_e32 v140, s44, v41
	v_mul_f32_e32 v141, s44, v49
	v_mul_f32_e32 v142, s44, v57
	v_mul_f32_e32 v143, s44, v65
	v_mul_f32_e32 v144, s44, v73
	v_mul_f32_e32 v145, s44, v81
	v_mul_f32_e32 v146, s44, v89
	v_mul_f32_e32 v147, s44, v97
	v_mul_f32_e32 v148, s44, v105
	v_mul_f32_e32 v149, s44, v113
	v_mul_f32_e32 v150, s44, v121
	v_mul_f32_e32 v151, s44, v2
	v_mul_f32_e32 v152, s44, v10
	v_mul_f32_e32 v153, s44, v18
	v_mul_f32_e32 v154, s44, v26
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v157, v137
	v_fmac_f32_e64 v137, v155, s45
	v_add_f32_e32 v132, v156, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v158, s[16:19], null offen
	v_mul_f32_e32 v138, v162, v138
	v_fmac_f32_e64 v138, v160, s45
	v_add_f32_e32 v132, v161, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v163, s[16:19], null offen
	v_mul_f32_e32 v139, v167, v139
	v_fmac_f32_e64 v139, v165, s45
	v_add_f32_e32 v132, v166, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v168, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_fmac_f32_e64 v140, v170, s45
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v157, v141
	v_fmac_f32_e64 v141, v175, s45
	v_add_f32_e32 v132, v156, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v176, s[16:19], null offen
	v_mul_f32_e32 v142, v162, v142
	v_fmac_f32_e64 v142, v178, s45
	v_add_f32_e32 v132, v161, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v179, s[16:19], null offen
	v_mul_f32_e32 v143, v167, v143
	v_fmac_f32_e64 v143, v181, s45
	v_add_f32_e32 v132, v166, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v182, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_fmac_f32_e64 v144, v184, s45
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v185, s[16:19], null offen
	v_mul_f32_e32 v145, v157, v145
	v_fmac_f32_e64 v145, v187, s45
	v_add_f32_e32 v132, v156, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v188, s[16:19], null offen
	v_mul_f32_e32 v146, v162, v146
	v_fmac_f32_e64 v146, v190, s45
	v_add_f32_e32 v132, v161, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v191, s[16:19], null offen
	v_mul_f32_e32 v147, v167, v147
	v_fmac_f32_e64 v147, v193, s45
	v_add_f32_e32 v132, v166, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v194, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_fmac_f32_e64 v148, v196, s45
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v197, s[16:19], null offen
	v_mul_f32_e32 v149, v157, v149
	v_fmac_f32_e64 v149, v199, s45
	v_add_f32_e32 v132, v156, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v200, s[16:19], null offen
	v_mul_f32_e32 v150, v162, v150
	v_fmac_f32_e64 v150, v202, s45
	v_add_f32_e32 v132, v161, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v203, s[16:19], null offen
	v_mul_f32_e32 v151, v167, v151
	v_fmac_f32_e64 v151, v205, s45
	v_add_f32_e32 v132, v166, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v206, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_fmac_f32_e64 v152, v208, s45
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v209, s[16:19], null offen
	v_mul_f32_e32 v153, v157, v153
	v_fmac_f32_e64 v153, v211, s45
	v_add_f32_e32 v132, v156, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v212, s[16:19], null offen
	v_mul_f32_e32 v154, v162, v154
	v_fmac_f32_e64 v154, v214, s45
	v_add_f32_e32 v132, v161, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v215, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v158, v130, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	buffer_load_b32 v155, v158, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v159, v128, s32
	v_lshlrev_b32_e32 v159, 2, v159
	ds_load_b32 v156, v159
	ds_load_b32 v157, v159 offset:512
	v_add_lshl_u32 v158, v131, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v163, v130, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	buffer_load_b32 v160, v163, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v132, s32
	v_lshlrev_b32_e32 v164, 2, v164
	ds_load_b32 v161, v164
	ds_load_b32 v162, v164 offset:512
	v_add_lshl_u32 v163, v131, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v168, v130, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	buffer_load_b32 v165, v168, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v169, v132, s32
	v_lshlrev_b32_e32 v169, 2, v169
	ds_load_b32 v166, v169
	ds_load_b32 v167, v169 offset:512
	v_add_lshl_u32 v168, v131, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v173, v130, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	buffer_load_b32 v170, v173, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v176, v130, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	buffer_load_b32 v175, v176, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v177, v128, s32
	v_lshlrev_b32_e32 v177, 2, v177
	v_add_lshl_u32 v176, v131, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v179, v130, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	buffer_load_b32 v178, v179, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v182, v130, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	buffer_load_b32 v181, v182, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v183, v132, s32
	v_lshlrev_b32_e32 v183, 2, v183
	v_add_lshl_u32 v182, v131, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v185, v130, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	buffer_load_b32 v184, v185, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v188, v130, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	buffer_load_b32 v187, v188, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v189, v128, s32
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v188, v131, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v191, v130, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	buffer_load_b32 v190, v191, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v194, v130, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	buffer_load_b32 v193, v194, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v195, v132, s32
	v_lshlrev_b32_e32 v195, 2, v195
	v_add_lshl_u32 v194, v131, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v197, v130, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	buffer_load_b32 v196, v197, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v200, v130, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	buffer_load_b32 v199, v200, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v201, v128, s32
	v_lshlrev_b32_e32 v201, 2, v201
	v_add_lshl_u32 v200, v131, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v203, v130, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	buffer_load_b32 v202, v203, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v206, v130, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	buffer_load_b32 v205, v206, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v207, v132, s32
	v_lshlrev_b32_e32 v207, 2, v207
	v_add_lshl_u32 v206, v131, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v209, v130, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	buffer_load_b32 v208, v209, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v212, v130, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	buffer_load_b32 v211, v212, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v213, v128, s32
	v_lshlrev_b32_e32 v213, 2, v213
	v_add_lshl_u32 v212, v131, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v215, v130, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	buffer_load_b32 v214, v215, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v216, v132, s32
	v_lshlrev_b32_e32 v216, 2, v216
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	v_mul_f32_e32 v137, s44, v34
	v_mul_f32_e32 v138, s44, v42
	v_mul_f32_e32 v139, s44, v50
	v_mul_f32_e32 v140, s44, v58
	v_mul_f32_e32 v141, s44, v66
	v_mul_f32_e32 v142, s44, v74
	v_mul_f32_e32 v143, s44, v82
	v_mul_f32_e32 v144, s44, v90
	v_mul_f32_e32 v145, s44, v98
	v_mul_f32_e32 v146, s44, v106
	v_mul_f32_e32 v147, s44, v114
	v_mul_f32_e32 v148, s44, v122
	v_mul_f32_e32 v149, s44, v3
	v_mul_f32_e32 v150, s44, v11
	v_mul_f32_e32 v151, s44, v19
	v_mul_f32_e32 v152, s44, v27
	v_mul_f32_e32 v153, s44, v35
	v_mul_f32_e32 v154, s44, v43
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v157, v137
	v_fmac_f32_e64 v137, v155, s45
	v_add_f32_e32 v132, v156, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v158, s[16:19], null offen
	v_mul_f32_e32 v138, v162, v138
	v_fmac_f32_e64 v138, v160, s45
	v_add_f32_e32 v132, v161, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v163, s[16:19], null offen
	v_mul_f32_e32 v139, v167, v139
	v_fmac_f32_e64 v139, v165, s45
	v_add_f32_e32 v132, v166, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v168, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_fmac_f32_e64 v140, v170, s45
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v157, v141
	v_fmac_f32_e64 v141, v175, s45
	v_add_f32_e32 v132, v156, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v176, s[16:19], null offen
	v_mul_f32_e32 v142, v162, v142
	v_fmac_f32_e64 v142, v178, s45
	v_add_f32_e32 v132, v161, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v179, s[16:19], null offen
	v_mul_f32_e32 v143, v167, v143
	v_fmac_f32_e64 v143, v181, s45
	v_add_f32_e32 v132, v166, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v182, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_fmac_f32_e64 v144, v184, s45
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v185, s[16:19], null offen
	v_mul_f32_e32 v145, v157, v145
	v_fmac_f32_e64 v145, v187, s45
	v_add_f32_e32 v132, v156, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v188, s[16:19], null offen
	v_mul_f32_e32 v146, v162, v146
	v_fmac_f32_e64 v146, v190, s45
	v_add_f32_e32 v132, v161, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v191, s[16:19], null offen
	v_mul_f32_e32 v147, v167, v147
	v_fmac_f32_e64 v147, v193, s45
	v_add_f32_e32 v132, v166, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v194, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_fmac_f32_e64 v148, v196, s45
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v197, s[16:19], null offen
	v_mul_f32_e32 v149, v157, v149
	v_fmac_f32_e64 v149, v199, s45
	v_add_f32_e32 v132, v156, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v200, s[16:19], null offen
	v_mul_f32_e32 v150, v162, v150
	v_fmac_f32_e64 v150, v202, s45
	v_add_f32_e32 v132, v161, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v203, s[16:19], null offen
	v_mul_f32_e32 v151, v167, v151
	v_fmac_f32_e64 v151, v205, s45
	v_add_f32_e32 v132, v166, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v206, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_fmac_f32_e64 v152, v208, s45
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v209, s[16:19], null offen
	v_mul_f32_e32 v153, v157, v153
	v_fmac_f32_e64 v153, v211, s45
	v_add_f32_e32 v132, v156, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v212, s[16:19], null offen
	v_mul_f32_e32 v154, v162, v154
	v_fmac_f32_e64 v154, v214, s45
	v_add_f32_e32 v132, v161, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v215, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v158, v130, v132, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	buffer_load_b32 v155, v158, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v159, v132, s32
	v_lshlrev_b32_e32 v159, 2, v159
	ds_load_b32 v156, v159
	ds_load_b32 v157, v159 offset:512
	v_add_lshl_u32 v158, v131, v132, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v163, v130, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	buffer_load_b32 v160, v163, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v132, s32
	v_lshlrev_b32_e32 v164, 2, v164
	ds_load_b32 v161, v164
	ds_load_b32 v162, v164 offset:512
	v_add_lshl_u32 v163, v131, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v168, v130, v128, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	buffer_load_b32 v165, v168, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v169, v128, s32
	v_lshlrev_b32_e32 v169, 2, v169
	ds_load_b32 v166, v169
	ds_load_b32 v167, v169 offset:512
	v_add_lshl_u32 v168, v131, v128, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v173, v130, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	buffer_load_b32 v170, v173, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v176, v130, v132, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	buffer_load_b32 v175, v176, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v177, v132, s32
	v_lshlrev_b32_e32 v177, 2, v177
	v_add_lshl_u32 v176, v131, v132, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v179, v130, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	buffer_load_b32 v178, v179, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v182, v130, v128, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	buffer_load_b32 v181, v182, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v183, v128, s32
	v_lshlrev_b32_e32 v183, 2, v183
	v_add_lshl_u32 v182, v131, v128, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v185, v130, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	buffer_load_b32 v184, v185, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v188, v130, v132, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	buffer_load_b32 v187, v188, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v189, v132, s32
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v188, v131, v132, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v191, v130, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	buffer_load_b32 v190, v191, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v194, v130, v128, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	buffer_load_b32 v193, v194, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v195, v128, s32
	v_lshlrev_b32_e32 v195, 2, v195
	v_add_lshl_u32 v194, v131, v128, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v197, v130, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	buffer_load_b32 v196, v197, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v200, v130, v132, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	buffer_load_b32 v199, v200, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v201, v132, s32
	v_lshlrev_b32_e32 v201, 2, v201
	v_add_lshl_u32 v200, v131, v132, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v203, v130, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	buffer_load_b32 v202, v203, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v206, v130, v128, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	buffer_load_b32 v205, v206, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v207, v128, s32
	v_lshlrev_b32_e32 v207, 2, v207
	v_add_lshl_u32 v206, v131, v128, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v209, v130, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	buffer_load_b32 v208, v209, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v212, v130, v132, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	buffer_load_b32 v211, v212, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v213, v132, s32
	v_lshlrev_b32_e32 v213, 2, v213
	v_add_lshl_u32 v212, v131, v132, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v215, v130, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	buffer_load_b32 v214, v215, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v216, v132, s32
	v_lshlrev_b32_e32 v216, 2, v216
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	v_mul_f32_e32 v137, s44, v51
	v_mul_f32_e32 v138, s44, v59
	v_mul_f32_e32 v139, s44, v67
	v_mul_f32_e32 v140, s44, v75
	v_mul_f32_e32 v141, s44, v83
	v_mul_f32_e32 v142, s44, v91
	v_mul_f32_e32 v143, s44, v99
	v_mul_f32_e32 v144, s44, v107
	v_mul_f32_e32 v145, s44, v115
	v_mul_f32_e32 v146, s44, v123
	v_mul_f32_e32 v147, s44, v4
	v_mul_f32_e32 v148, s44, v12
	v_mul_f32_e32 v149, s44, v20
	v_mul_f32_e32 v150, s44, v28
	v_mul_f32_e32 v151, s44, v36
	v_mul_f32_e32 v152, s44, v44
	v_mul_f32_e32 v153, s44, v52
	v_mul_f32_e32 v154, s44, v60
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v157, v137
	v_fmac_f32_e64 v137, v155, s45
	v_add_f32_e32 v132, v156, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v158, s[16:19], null offen
	v_mul_f32_e32 v138, v162, v138
	v_fmac_f32_e64 v138, v160, s45
	v_add_f32_e32 v132, v161, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v163, s[16:19], null offen
	v_mul_f32_e32 v139, v167, v139
	v_fmac_f32_e64 v139, v165, s45
	v_add_f32_e32 v132, v166, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v168, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_fmac_f32_e64 v140, v170, s45
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v157, v141
	v_fmac_f32_e64 v141, v175, s45
	v_add_f32_e32 v132, v156, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v176, s[16:19], null offen
	v_mul_f32_e32 v142, v162, v142
	v_fmac_f32_e64 v142, v178, s45
	v_add_f32_e32 v132, v161, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v179, s[16:19], null offen
	v_mul_f32_e32 v143, v167, v143
	v_fmac_f32_e64 v143, v181, s45
	v_add_f32_e32 v132, v166, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v182, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_fmac_f32_e64 v144, v184, s45
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v185, s[16:19], null offen
	v_mul_f32_e32 v145, v157, v145
	v_fmac_f32_e64 v145, v187, s45
	v_add_f32_e32 v132, v156, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v188, s[16:19], null offen
	v_mul_f32_e32 v146, v162, v146
	v_fmac_f32_e64 v146, v190, s45
	v_add_f32_e32 v132, v161, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v191, s[16:19], null offen
	v_mul_f32_e32 v147, v167, v147
	v_fmac_f32_e64 v147, v193, s45
	v_add_f32_e32 v132, v166, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v194, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_fmac_f32_e64 v148, v196, s45
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v197, s[16:19], null offen
	v_mul_f32_e32 v149, v157, v149
	v_fmac_f32_e64 v149, v199, s45
	v_add_f32_e32 v132, v156, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v200, s[16:19], null offen
	v_mul_f32_e32 v150, v162, v150
	v_fmac_f32_e64 v150, v202, s45
	v_add_f32_e32 v132, v161, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v203, s[16:19], null offen
	v_mul_f32_e32 v151, v167, v151
	v_fmac_f32_e64 v151, v205, s45
	v_add_f32_e32 v132, v166, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v206, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_fmac_f32_e64 v152, v208, s45
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v209, s[16:19], null offen
	v_mul_f32_e32 v153, v157, v153
	v_fmac_f32_e64 v153, v211, s45
	v_add_f32_e32 v132, v156, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v212, s[16:19], null offen
	v_mul_f32_e32 v154, v162, v154
	v_fmac_f32_e64 v154, v214, s45
	v_add_f32_e32 v132, v161, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v215, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v158, v130, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	buffer_load_b32 v155, v158, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v159, v128, s32
	v_lshlrev_b32_e32 v159, 2, v159
	ds_load_b32 v156, v159
	ds_load_b32 v157, v159 offset:512
	v_add_lshl_u32 v158, v131, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v163, v130, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	buffer_load_b32 v160, v163, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v132, s32
	v_lshlrev_b32_e32 v164, 2, v164
	ds_load_b32 v161, v164
	ds_load_b32 v162, v164 offset:512
	v_add_lshl_u32 v163, v131, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v168, v130, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	buffer_load_b32 v165, v168, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v169, v132, s32
	v_lshlrev_b32_e32 v169, 2, v169
	ds_load_b32 v166, v169
	ds_load_b32 v167, v169 offset:512
	v_add_lshl_u32 v168, v131, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v173, v130, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	buffer_load_b32 v170, v173, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v176, v130, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	buffer_load_b32 v175, v176, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v177, v128, s32
	v_lshlrev_b32_e32 v177, 2, v177
	v_add_lshl_u32 v176, v131, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v179, v130, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	buffer_load_b32 v178, v179, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v182, v130, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	buffer_load_b32 v181, v182, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v183, v132, s32
	v_lshlrev_b32_e32 v183, 2, v183
	v_add_lshl_u32 v182, v131, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v185, v130, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	buffer_load_b32 v184, v185, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v188, v130, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	buffer_load_b32 v187, v188, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v189, v128, s32
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v188, v131, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v191, v130, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	buffer_load_b32 v190, v191, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v194, v130, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	buffer_load_b32 v193, v194, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v195, v132, s32
	v_lshlrev_b32_e32 v195, 2, v195
	v_add_lshl_u32 v194, v131, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v197, v130, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	buffer_load_b32 v196, v197, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v200, v130, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	buffer_load_b32 v199, v200, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v201, v128, s32
	v_lshlrev_b32_e32 v201, 2, v201
	v_add_lshl_u32 v200, v131, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v203, v130, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	buffer_load_b32 v202, v203, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v206, v130, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	buffer_load_b32 v205, v206, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v207, v132, s32
	v_lshlrev_b32_e32 v207, 2, v207
	v_add_lshl_u32 v206, v131, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v209, v130, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	buffer_load_b32 v208, v209, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v212, v130, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	buffer_load_b32 v211, v212, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v213, v128, s32
	v_lshlrev_b32_e32 v213, 2, v213
	v_add_lshl_u32 v212, v131, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v215, v130, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	buffer_load_b32 v214, v215, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v216, v132, s32
	v_lshlrev_b32_e32 v216, 2, v216
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	v_mul_f32_e32 v137, s44, v68
	v_mul_f32_e32 v138, s44, v76
	v_mul_f32_e32 v139, s44, v84
	v_mul_f32_e32 v140, s44, v92
	v_mul_f32_e32 v141, s44, v100
	v_mul_f32_e32 v142, s44, v108
	v_mul_f32_e32 v143, s44, v116
	v_mul_f32_e32 v144, s44, v124
	v_mul_f32_e32 v145, s44, v5
	v_mul_f32_e32 v146, s44, v13
	v_mul_f32_e32 v147, s44, v21
	v_mul_f32_e32 v148, s44, v29
	v_mul_f32_e32 v149, s44, v37
	v_mul_f32_e32 v150, s44, v45
	v_mul_f32_e32 v151, s44, v53
	v_mul_f32_e32 v152, s44, v61
	v_mul_f32_e32 v153, s44, v69
	v_mul_f32_e32 v154, s44, v77
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v157, v137
	v_fmac_f32_e64 v137, v155, s45
	v_add_f32_e32 v132, v156, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v158, s[16:19], null offen
	v_mul_f32_e32 v138, v162, v138
	v_fmac_f32_e64 v138, v160, s45
	v_add_f32_e32 v132, v161, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v163, s[16:19], null offen
	v_mul_f32_e32 v139, v167, v139
	v_fmac_f32_e64 v139, v165, s45
	v_add_f32_e32 v132, v166, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v168, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_fmac_f32_e64 v140, v170, s45
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v157, v141
	v_fmac_f32_e64 v141, v175, s45
	v_add_f32_e32 v132, v156, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v176, s[16:19], null offen
	v_mul_f32_e32 v142, v162, v142
	v_fmac_f32_e64 v142, v178, s45
	v_add_f32_e32 v132, v161, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v179, s[16:19], null offen
	v_mul_f32_e32 v143, v167, v143
	v_fmac_f32_e64 v143, v181, s45
	v_add_f32_e32 v132, v166, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v182, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_fmac_f32_e64 v144, v184, s45
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v185, s[16:19], null offen
	v_mul_f32_e32 v145, v157, v145
	v_fmac_f32_e64 v145, v187, s45
	v_add_f32_e32 v132, v156, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v188, s[16:19], null offen
	v_mul_f32_e32 v146, v162, v146
	v_fmac_f32_e64 v146, v190, s45
	v_add_f32_e32 v132, v161, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v191, s[16:19], null offen
	v_mul_f32_e32 v147, v167, v147
	v_fmac_f32_e64 v147, v193, s45
	v_add_f32_e32 v132, v166, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v194, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_fmac_f32_e64 v148, v196, s45
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v197, s[16:19], null offen
	v_mul_f32_e32 v149, v157, v149
	v_fmac_f32_e64 v149, v199, s45
	v_add_f32_e32 v132, v156, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v200, s[16:19], null offen
	v_mul_f32_e32 v150, v162, v150
	v_fmac_f32_e64 v150, v202, s45
	v_add_f32_e32 v132, v161, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v203, s[16:19], null offen
	v_mul_f32_e32 v151, v167, v151
	v_fmac_f32_e64 v151, v205, s45
	v_add_f32_e32 v132, v166, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v206, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_fmac_f32_e64 v152, v208, s45
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v209, s[16:19], null offen
	v_mul_f32_e32 v153, v157, v153
	v_fmac_f32_e64 v153, v211, s45
	v_add_f32_e32 v132, v156, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v212, s[16:19], null offen
	v_mul_f32_e32 v154, v162, v154
	v_fmac_f32_e64 v154, v214, s45
	v_add_f32_e32 v132, v161, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v215, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v158, v130, v132, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	buffer_load_b32 v155, v158, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v159, v132, s32
	v_lshlrev_b32_e32 v159, 2, v159
	ds_load_b32 v156, v159
	ds_load_b32 v157, v159 offset:512
	v_add_lshl_u32 v158, v131, v132, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v163, v130, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	buffer_load_b32 v160, v163, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v132, s32
	v_lshlrev_b32_e32 v164, 2, v164
	ds_load_b32 v161, v164
	ds_load_b32 v162, v164 offset:512
	v_add_lshl_u32 v163, v131, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v168, v130, v128, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	buffer_load_b32 v165, v168, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v169, v128, s32
	v_lshlrev_b32_e32 v169, 2, v169
	ds_load_b32 v166, v169
	ds_load_b32 v167, v169 offset:512
	v_add_lshl_u32 v168, v131, v128, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v173, v130, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	buffer_load_b32 v170, v173, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v176, v130, v132, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	buffer_load_b32 v175, v176, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v177, v132, s32
	v_lshlrev_b32_e32 v177, 2, v177
	v_add_lshl_u32 v176, v131, v132, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v179, v130, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	buffer_load_b32 v178, v179, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v182, v130, v128, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	buffer_load_b32 v181, v182, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v183, v128, s32
	v_lshlrev_b32_e32 v183, 2, v183
	v_add_lshl_u32 v182, v131, v128, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v185, v130, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	buffer_load_b32 v184, v185, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v188, v130, v132, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	buffer_load_b32 v187, v188, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v189, v132, s32
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v188, v131, v132, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v191, v130, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	buffer_load_b32 v190, v191, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v194, v130, v128, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	buffer_load_b32 v193, v194, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v195, v128, s32
	v_lshlrev_b32_e32 v195, 2, v195
	v_add_lshl_u32 v194, v131, v128, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v197, v130, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	buffer_load_b32 v196, v197, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v200, v130, v132, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	buffer_load_b32 v199, v200, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v201, v132, s32
	v_lshlrev_b32_e32 v201, 2, v201
	v_add_lshl_u32 v200, v131, v132, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v203, v130, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	buffer_load_b32 v202, v203, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v206, v130, v128, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	buffer_load_b32 v205, v206, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v207, v128, s32
	v_lshlrev_b32_e32 v207, 2, v207
	v_add_lshl_u32 v206, v131, v128, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v209, v130, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	buffer_load_b32 v208, v209, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v212, v130, v132, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	buffer_load_b32 v211, v212, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v213, v132, s32
	v_lshlrev_b32_e32 v213, 2, v213
	v_add_lshl_u32 v212, v131, v132, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v215, v130, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	buffer_load_b32 v214, v215, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v216, v132, s32
	v_lshlrev_b32_e32 v216, 2, v216
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	v_mul_f32_e32 v137, s44, v85
	v_mul_f32_e32 v138, s44, v93
	v_mul_f32_e32 v139, s44, v101
	v_mul_f32_e32 v140, s44, v109
	v_mul_f32_e32 v141, s44, v117
	v_mul_f32_e32 v142, s44, v125
	v_mul_f32_e32 v143, s44, v6
	v_mul_f32_e32 v144, s44, v14
	v_mul_f32_e32 v145, s44, v22
	v_mul_f32_e32 v146, s44, v30
	v_mul_f32_e32 v147, s44, v38
	v_mul_f32_e32 v148, s44, v46
	v_mul_f32_e32 v149, s44, v54
	v_mul_f32_e32 v150, s44, v62
	v_mul_f32_e32 v151, s44, v70
	v_mul_f32_e32 v152, s44, v78
	v_mul_f32_e32 v153, s44, v86
	v_mul_f32_e32 v154, s44, v94
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v157, v137
	v_fmac_f32_e64 v137, v155, s45
	v_add_f32_e32 v132, v156, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v158, s[16:19], null offen
	v_mul_f32_e32 v138, v162, v138
	v_fmac_f32_e64 v138, v160, s45
	v_add_f32_e32 v132, v161, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v163, s[16:19], null offen
	v_mul_f32_e32 v139, v167, v139
	v_fmac_f32_e64 v139, v165, s45
	v_add_f32_e32 v132, v166, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v168, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_fmac_f32_e64 v140, v170, s45
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v157, v141
	v_fmac_f32_e64 v141, v175, s45
	v_add_f32_e32 v132, v156, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v176, s[16:19], null offen
	v_mul_f32_e32 v142, v162, v142
	v_fmac_f32_e64 v142, v178, s45
	v_add_f32_e32 v132, v161, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v179, s[16:19], null offen
	v_mul_f32_e32 v143, v167, v143
	v_fmac_f32_e64 v143, v181, s45
	v_add_f32_e32 v132, v166, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v182, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_fmac_f32_e64 v144, v184, s45
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v185, s[16:19], null offen
	v_mul_f32_e32 v145, v157, v145
	v_fmac_f32_e64 v145, v187, s45
	v_add_f32_e32 v132, v156, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v188, s[16:19], null offen
	v_mul_f32_e32 v146, v162, v146
	v_fmac_f32_e64 v146, v190, s45
	v_add_f32_e32 v132, v161, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v191, s[16:19], null offen
	v_mul_f32_e32 v147, v167, v147
	v_fmac_f32_e64 v147, v193, s45
	v_add_f32_e32 v132, v166, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v194, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_fmac_f32_e64 v148, v196, s45
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v197, s[16:19], null offen
	v_mul_f32_e32 v149, v157, v149
	v_fmac_f32_e64 v149, v199, s45
	v_add_f32_e32 v132, v156, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v200, s[16:19], null offen
	v_mul_f32_e32 v150, v162, v150
	v_fmac_f32_e64 v150, v202, s45
	v_add_f32_e32 v132, v161, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v203, s[16:19], null offen
	v_mul_f32_e32 v151, v167, v151
	v_fmac_f32_e64 v151, v205, s45
	v_add_f32_e32 v132, v166, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v206, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_fmac_f32_e64 v152, v208, s45
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v209, s[16:19], null offen
	v_mul_f32_e32 v153, v157, v153
	v_fmac_f32_e64 v153, v211, s45
	v_add_f32_e32 v132, v156, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v212, s[16:19], null offen
	v_mul_f32_e32 v154, v162, v154
	v_fmac_f32_e64 v154, v214, s45
	v_add_f32_e32 v132, v161, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v215, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v158, v130, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	buffer_load_b32 v155, v158, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v159, v128, s32
	v_lshlrev_b32_e32 v159, 2, v159
	ds_load_b32 v156, v159
	ds_load_b32 v157, v159 offset:512
	v_add_lshl_u32 v158, v131, v128, 2
	v_cndmask_b32_e64 v158, v136, v158, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v163, v130, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	buffer_load_b32 v160, v163, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v164, v132, s32
	v_lshlrev_b32_e32 v164, 2, v164
	ds_load_b32 v161, v164
	ds_load_b32 v162, v164 offset:512
	v_add_lshl_u32 v163, v131, v132, 2
	v_cndmask_b32_e64 v163, v136, v163, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v168, v130, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	buffer_load_b32 v165, v168, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v169, v132, s32
	v_lshlrev_b32_e32 v169, 2, v169
	ds_load_b32 v166, v169
	ds_load_b32 v167, v169 offset:512
	v_add_lshl_u32 v168, v131, v132, 2
	v_cndmask_b32_e64 v168, v136, v168, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v173, v130, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	buffer_load_b32 v170, v173, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v174, v132, s32
	v_lshlrev_b32_e32 v174, 2, v174
	ds_load_b32 v171, v174
	ds_load_b32 v172, v174 offset:512
	v_add_lshl_u32 v173, v131, v132, 2
	v_cndmask_b32_e64 v173, v136, v173, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v176, v130, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	buffer_load_b32 v175, v176, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v177, v128, s32
	v_lshlrev_b32_e32 v177, 2, v177
	v_add_lshl_u32 v176, v131, v128, 2
	v_cndmask_b32_e64 v176, v136, v176, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v179, v130, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	buffer_load_b32 v178, v179, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v180, v132, s32
	v_lshlrev_b32_e32 v180, 2, v180
	v_add_lshl_u32 v179, v131, v132, 2
	v_cndmask_b32_e64 v179, v136, v179, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v182, v130, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	buffer_load_b32 v181, v182, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v183, v132, s32
	v_lshlrev_b32_e32 v183, 2, v183
	v_add_lshl_u32 v182, v131, v132, 2
	v_cndmask_b32_e64 v182, v136, v182, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v185, v130, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	buffer_load_b32 v184, v185, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v186, v132, s32
	v_lshlrev_b32_e32 v186, 2, v186
	v_add_lshl_u32 v185, v131, v132, 2
	v_cndmask_b32_e64 v185, v136, v185, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v188, v130, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	buffer_load_b32 v187, v188, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v189, v128, s32
	v_lshlrev_b32_e32 v189, 2, v189
	v_add_lshl_u32 v188, v131, v128, 2
	v_cndmask_b32_e64 v188, v136, v188, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v191, v130, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	buffer_load_b32 v190, v191, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v192, v132, s32
	v_lshlrev_b32_e32 v192, 2, v192
	v_add_lshl_u32 v191, v131, v132, 2
	v_cndmask_b32_e64 v191, v136, v191, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v194, v130, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	buffer_load_b32 v193, v194, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v195, v132, s32
	v_lshlrev_b32_e32 v195, 2, v195
	v_add_lshl_u32 v194, v131, v132, 2
	v_cndmask_b32_e64 v194, v136, v194, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v197, v130, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	buffer_load_b32 v196, v197, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v198, v132, s32
	v_lshlrev_b32_e32 v198, 2, v198
	v_add_lshl_u32 v197, v131, v132, 2
	v_cndmask_b32_e64 v197, v136, v197, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v200, v130, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	buffer_load_b32 v199, v200, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v201, v128, s32
	v_lshlrev_b32_e32 v201, 2, v201
	v_add_lshl_u32 v200, v131, v128, 2
	v_cndmask_b32_e64 v200, v136, v200, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v203, v130, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	buffer_load_b32 v202, v203, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v204, v132, s32
	v_lshlrev_b32_e32 v204, 2, v204
	v_add_lshl_u32 v203, v131, v132, 2
	v_cndmask_b32_e64 v203, v136, v203, s34
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v206, v130, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	buffer_load_b32 v205, v206, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v207, v132, s32
	v_lshlrev_b32_e32 v207, 2, v207
	v_add_lshl_u32 v206, v131, v132, 2
	v_cndmask_b32_e64 v206, v136, v206, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v209, v130, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	buffer_load_b32 v208, v209, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v210, v132, s32
	v_lshlrev_b32_e32 v210, 2, v210
	v_add_lshl_u32 v209, v131, v132, 2
	v_cndmask_b32_e64 v209, v136, v209, s34
	v_add_co_u32 v129, vcc_lo, v129, 1
	v_add_nc_u32_e64 v130, v130, s38
	v_add_nc_u32_e64 v131, v131, s36
	v_cmp_lt_u32_e64 s32, v128, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v212, v130, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	buffer_load_b32 v211, v212, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v213, v128, s32
	v_lshlrev_b32_e32 v213, 2, v213
	v_add_lshl_u32 v212, v131, v128, 2
	v_cndmask_b32_e64 v212, v136, v212, s34
	v_add_co_u32 v132, vcc_lo, v128, 1
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v215, v130, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	buffer_load_b32 v214, v215, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v216, v132, s32
	v_lshlrev_b32_e32 v216, 2, v216
	v_add_lshl_u32 v215, v131, v132, 2
	v_cndmask_b32_e64 v215, v136, v215, s34
	v_mul_f32_e32 v137, s44, v102
	v_mul_f32_e32 v138, s44, v110
	v_mul_f32_e32 v139, s44, v118
	v_mul_f32_e32 v140, s44, v126
	v_mul_f32_e32 v141, s44, v7
	v_mul_f32_e32 v142, s44, v15
	v_mul_f32_e32 v143, s44, v23
	v_mul_f32_e32 v144, s44, v31
	v_mul_f32_e32 v145, s44, v39
	v_mul_f32_e32 v146, s44, v47
	v_mul_f32_e32 v147, s44, v55
	v_mul_f32_e32 v148, s44, v63
	v_mul_f32_e32 v149, s44, v71
	v_mul_f32_e32 v150, s44, v79
	v_mul_f32_e32 v151, s44, v87
	v_mul_f32_e32 v152, s44, v95
	v_mul_f32_e32 v153, s44, v103
	v_mul_f32_e32 v154, s44, v111
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v157, v137
	v_fmac_f32_e64 v137, v155, s45
	v_add_f32_e32 v132, v156, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v158, s[16:19], null offen
	v_mul_f32_e32 v138, v162, v138
	v_fmac_f32_e64 v138, v160, s45
	v_add_f32_e32 v132, v161, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v163, s[16:19], null offen
	v_mul_f32_e32 v139, v167, v139
	v_fmac_f32_e64 v139, v165, s45
	v_add_f32_e32 v132, v166, v139
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v139, v132
	buffer_store_b32 v139, v168, s[16:19], null offen
	v_mul_f32_e32 v140, v172, v140
	v_fmac_f32_e64 v140, v170, s45
	v_add_f32_e32 v132, v171, v140
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v140, v132
	buffer_store_b32 v140, v173, s[16:19], null offen
	v_mul_f32_e32 v141, v157, v141
	v_fmac_f32_e64 v141, v175, s45
	v_add_f32_e32 v132, v156, v141
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v141, v132
	buffer_store_b32 v141, v176, s[16:19], null offen
	v_mul_f32_e32 v142, v162, v142
	v_fmac_f32_e64 v142, v178, s45
	v_add_f32_e32 v132, v161, v142
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v142, v132
	buffer_store_b32 v142, v179, s[16:19], null offen
	v_mul_f32_e32 v143, v167, v143
	v_fmac_f32_e64 v143, v181, s45
	v_add_f32_e32 v132, v166, v143
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v143, v132
	buffer_store_b32 v143, v182, s[16:19], null offen
	v_mul_f32_e32 v144, v172, v144
	v_fmac_f32_e64 v144, v184, s45
	v_add_f32_e32 v132, v171, v144
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v144, v132
	buffer_store_b32 v144, v185, s[16:19], null offen
	v_mul_f32_e32 v145, v157, v145
	v_fmac_f32_e64 v145, v187, s45
	v_add_f32_e32 v132, v156, v145
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v145, v132
	buffer_store_b32 v145, v188, s[16:19], null offen
	v_mul_f32_e32 v146, v162, v146
	v_fmac_f32_e64 v146, v190, s45
	v_add_f32_e32 v132, v161, v146
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v146, v132
	buffer_store_b32 v146, v191, s[16:19], null offen
	v_mul_f32_e32 v147, v167, v147
	v_fmac_f32_e64 v147, v193, s45
	v_add_f32_e32 v132, v166, v147
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v147, v132
	buffer_store_b32 v147, v194, s[16:19], null offen
	v_mul_f32_e32 v148, v172, v148
	v_fmac_f32_e64 v148, v196, s45
	v_add_f32_e32 v132, v171, v148
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v148, v132
	buffer_store_b32 v148, v197, s[16:19], null offen
	v_mul_f32_e32 v149, v157, v149
	v_fmac_f32_e64 v149, v199, s45
	v_add_f32_e32 v132, v156, v149
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v149, v132
	buffer_store_b32 v149, v200, s[16:19], null offen
	v_mul_f32_e32 v150, v162, v150
	v_fmac_f32_e64 v150, v202, s45
	v_add_f32_e32 v132, v161, v150
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v150, v132
	buffer_store_b32 v150, v203, s[16:19], null offen
	v_mul_f32_e32 v151, v167, v151
	v_fmac_f32_e64 v151, v205, s45
	v_add_f32_e32 v132, v166, v151
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v151, v132
	buffer_store_b32 v151, v206, s[16:19], null offen
	v_mul_f32_e32 v152, v172, v152
	v_fmac_f32_e64 v152, v208, s45
	v_add_f32_e32 v132, v171, v152
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v152, v132
	buffer_store_b32 v152, v209, s[16:19], null offen
	v_mul_f32_e32 v153, v157, v153
	v_fmac_f32_e64 v153, v211, s45
	v_add_f32_e32 v132, v156, v153
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v153, v132
	buffer_store_b32 v153, v212, s[16:19], null offen
	v_mul_f32_e32 v154, v162, v154
	v_fmac_f32_e64 v154, v214, s45
	v_add_f32_e32 v132, v161, v154
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v154, v132
	buffer_store_b32 v154, v215, s[16:19], null offen
	s_nop 0
	v_mov_b32_e32 v136, 0x80000000
	v_add_co_u32 v132, vcc_lo, v128, 2
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v142, v130, v132, 2
	v_cndmask_b32_e64 v142, v136, v142, s34
	buffer_load_b32 v139, v142, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v143, v132, s32
	v_lshlrev_b32_e32 v143, 2, v143
	ds_load_b32 v140, v143
	ds_load_b32 v141, v143 offset:512
	v_add_lshl_u32 v142, v131, v132, 2
	v_cndmask_b32_e64 v142, v136, v142, s34
	v_add_co_u32 v132, vcc_lo, v128, 3
	v_cmp_lt_u32_e64 s32, v132, s24
	v_cmp_lt_u32_e64 s34, v129, s25
	s_and_b32 s34, s32, s34
	v_add_lshl_u32 v147, v130, v132, 2
	v_cndmask_b32_e64 v147, v136, v147, s34
	buffer_load_b32 v144, v147, s[20:23], null offen
	s_mul_i32 s32, 0x80, s2
	v_sub_nc_u32_e64 v148, v132, s32
	v_lshlrev_b32_e32 v148, 2, v148
	ds_load_b32 v145, v148
	ds_load_b32 v146, v148 offset:512
	v_add_lshl_u32 v147, v131, v132, 2
	v_cndmask_b32_e64 v147, v136, v147, s34
	v_mul_f32_e32 v137, s44, v119
	v_mul_f32_e32 v138, s44, v127
	s_wait_dscnt 0x0
	s_wait_loadcnt 0x0
	v_mul_f32_e32 v137, v141, v137
	v_fmac_f32_e64 v137, v139, s45
	v_add_f32_e32 v132, v140, v137
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v137, v132
	buffer_store_b32 v137, v142, s[16:19], null offen
	v_mul_f32_e32 v138, v146, v138
	v_fmac_f32_e64 v138, v144, s45
	v_add_f32_e32 v132, v145, v138
	s_swappc_b64 s[30:31], s[12:13]
	v_mov_b32_e32 v138, v132
	buffer_store_b32 v138, v147, s[16:19], null offen
	s_nop 0
	s_branch label_GW_End_1
label_Activation_None_VW4:
	s_setpc_b64 s[30:31]
label_Activation_Gelu_VW4:
	v_mul_f32_e32 v136, 0x3d372713, v132
	v_fma_f32 v136, v132, v136, 1.0
	v_mul_f32_e32 v136, v132, v136
	v_mul_f32_e32 v136, 0x40135761, v136
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_fma_f32 v136, -2.0, v136, 2.0
	v_mul_f32_e32 v136, v132, v136
	v_mul_f32_e32 v132, 0.5, v136
	v_mul_f32_e32 v136, 0x3d372713, v133
	v_fma_f32 v136, v133, v136, 1.0
	v_mul_f32_e32 v136, v133, v136
	v_mul_f32_e32 v136, 0x40135761, v136
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_fma_f32 v136, -2.0, v136, 2.0
	v_mul_f32_e32 v136, v133, v136
	v_mul_f32_e32 v133, 0.5, v136
	v_mul_f32_e32 v136, 0x3d372713, v134
	v_fma_f32 v136, v134, v136, 1.0
	v_mul_f32_e32 v136, v134, v136
	v_mul_f32_e32 v136, 0x40135761, v136
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_fma_f32 v136, -2.0, v136, 2.0
	v_mul_f32_e32 v136, v134, v136
	v_mul_f32_e32 v134, 0.5, v136
	v_mul_f32_e32 v136, 0x3d372713, v135
	v_fma_f32 v136, v135, v136, 1.0
	v_mul_f32_e32 v136, v135, v136
	v_mul_f32_e32 v136, 0x40135761, v136
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_fma_f32 v136, -2.0, v136, 2.0
	v_mul_f32_e32 v136, v135, v136
	v_mul_f32_e32 v135, 0.5, v136
	s_setpc_b64 s[30:31]
label_Activation_Relu_VW4:
	v_max_num_f32_e64 v132, v132, 0
	v_max_num_f32_e64 v133, v133, 0
	v_max_num_f32_e64 v134, v134, 0
	v_max_num_f32_e64 v135, v135, 0
	s_setpc_b64 s[30:31]
label_Activation_Sigmoid_VW4:
	v_mul_f32_e32 v132, 0xbfb8aa3b, v132
	v_exp_f32_e32 v132, v132
	v_add_f32_e32 v132, 1.0, v132
	v_rcp_f32_e32 v132, v132
	v_mul_f32_e32 v133, 0xbfb8aa3b, v133
	v_exp_f32_e32 v133, v133
	v_add_f32_e32 v133, 1.0, v133
	v_rcp_f32_e32 v133, v133
	v_mul_f32_e32 v134, 0xbfb8aa3b, v134
	v_exp_f32_e32 v134, v134
	v_add_f32_e32 v134, 1.0, v134
	v_rcp_f32_e32 v134, v134
	v_mul_f32_e32 v135, 0xbfb8aa3b, v135
	v_exp_f32_e32 v135, v135
	v_add_f32_e32 v135, 1.0, v135
	v_rcp_f32_e32 v135, v135
	s_setpc_b64 s[30:31]
label_Activation_Silu_VW4:
	v_mul_f32_e32 v136, 0xbfb8aa3b, v132
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_mul_f32_e32 v132, v132, v136
	v_mul_f32_e32 v136, 0xbfb8aa3b, v133
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_mul_f32_e32 v133, v133, v136
	v_mul_f32_e32 v136, 0xbfb8aa3b, v134
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_mul_f32_e32 v134, v134, v136
	v_mul_f32_e32 v136, 0xbfb8aa3b, v135
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_mul_f32_e32 v135, v135, v136
	s_setpc_b64 s[30:31]
label_Activation_Clamp_VW4:
	v_min_num_f32_e32 v132, s55, v132
	v_max_num_f32_e32 v132, s54, v132
	v_min_num_f32_e32 v133, s55, v133
	v_max_num_f32_e32 v133, s54, v133
	v_min_num_f32_e32 v134, s55, v134
	v_max_num_f32_e32 v134, s54, v134
	v_min_num_f32_e32 v135, s55, v135
	v_max_num_f32_e32 v135, s54, v135
	s_setpc_b64 s[30:31]
label_Activation_None_VW1:
	s_setpc_b64 s[30:31]
label_Activation_Gelu_VW1:
	v_mul_f32_e32 v136, 0x3d372713, v132
	v_fma_f32 v136, v132, v136, 1.0
	v_mul_f32_e32 v136, v132, v136
	v_mul_f32_e32 v136, 0x40135761, v136
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_fma_f32 v136, -2.0, v136, 2.0
	v_mul_f32_e32 v136, v132, v136
	v_mul_f32_e32 v132, 0.5, v136
	s_setpc_b64 s[30:31]
label_Activation_Relu_VW1:
	v_max_num_f32_e64 v132, v132, 0
	s_setpc_b64 s[30:31]
label_Activation_Sigmoid_VW1:
	v_mul_f32_e32 v132, 0xbfb8aa3b, v132
	v_exp_f32_e32 v132, v132
	v_add_f32_e32 v132, 1.0, v132
	v_rcp_f32_e32 v132, v132
	s_setpc_b64 s[30:31]
label_Activation_Silu_VW1:
	v_mul_f32_e32 v136, 0xbfb8aa3b, v132
	v_exp_f32_e32 v136, v136
	v_add_f32_e32 v136, 1.0, v136
	v_rcp_f32_e32 v136, v136
	v_mul_f32_e32 v132, v132, v136
	s_setpc_b64 s[30:31]
label_Activation_Clamp_VW1:
	v_min_num_f32_e32 v132, s55, v132
	v_max_num_f32_e32 v132, s54, v132
	s_setpc_b64 s[30:31]
label_KernelEnd:
label_GW_End_1:
	s_endpgm
label_ASM_End:
.Lfunc_end0:
.size respack_hgemm_nt_rn_gfx1200, .Lfunc_end0-respack_hgemm_nt_rn_gfx1200
.rodata
.p2align 6
.amdhsa_kernel respack_hgemm_nt_rn_gfx1200
  .amdhsa_group_segment_fixed_size 8192
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_kernarg_size 0
  .amdhsa_inst_pref_size 0
  .amdhsa_next_free_vgpr 256
  .amdhsa_reserve_vcc 0
  .amdhsa_reserve_xnack_mask 0
  .amdhsa_next_free_sgpr 8
  .amdhsa_float_round_mode_32 0
  .amdhsa_float_round_mode_16_64 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
  .amdhsa_fp16_overflow 0
  .amdhsa_workgroup_processor_mode 1
  .amdhsa_memory_ordered 1
  .amdhsa_forward_progress 1
  .amdhsa_round_robin_scheduling 0
  .amdhsa_enable_private_segment 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_sgpr_workgroup_info 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_exception_fp_ieee_invalid_op 0
  .amdhsa_exception_fp_denorm_src 0
  .amdhsa_exception_fp_ieee_div_zero 0
  .amdhsa_exception_fp_ieee_overflow 0
  .amdhsa_exception_fp_ieee_underflow 0
  .amdhsa_exception_fp_ieee_inexact 0
  .amdhsa_exception_int_div_zero 0
  .amdhsa_user_sgpr_dispatch_ptr 0
  .amdhsa_user_sgpr_queue_ptr 0
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_user_sgpr_dispatch_id 0
  .amdhsa_user_sgpr_private_segment_size 0
  .amdhsa_wavefront_size32 1
.end_amdhsa_kernel
.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .name:           Gemm info
        .offset:         0
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           kernel info0
        .offset:         4
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           kernel info1
        .offset:         8
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           numWG
        .offset:         12
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesFree0
        .offset:         16
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesFree1
        .offset:         20
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesFree2
        .offset:         24
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           SizesSum0
        .offset:         28
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .address_space:  generic
        .name:           D
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  generic
        .name:           C
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  generic
        .name:           A
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16
      - .address_space:  generic
        .name:           B
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f16
      - .name:           strideD0
        .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideD1
        .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideC0
        .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideC1
        .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideA0
        .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideA1
        .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideB0
        .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           strideB1
        .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           alpha
        .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .name:           beta
        .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .address_space:  generic
        .name:           AddressScaleAlphaVec
        .offset:         104
        .size:           8
        .value_kind:     global_buffer
        .value_type:     f32
      - .address_space:  generic
        .name:           bias
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
        .value_type:     void
      - .name:           biasType
        .offset:         120
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           StrideBias
        .offset:         124
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .name:           activationAlpha
        .offset:         128
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .name:           activationBeta
        .offset:         132
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .name:           activationType
        .offset:         136
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           respack_hgemm_nt_rn_gfx1200
    .private_segment_fixed_size: 0
    .sgpr_count:     72
    .sgpr_spill_count: 0
    .symbol:         respack_hgemm_nt_rn_gfx1200.kd
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1200
amdhsa.version:
  - 1
  - 1
...
.end_amdgpu_metadata
