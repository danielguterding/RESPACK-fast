# zen-opt fork: target-based CMake (modernised from the upstream CMake 2.8 file)
add_library(chiqw_com STATIC
  est_NTK.f90 est_nwx2.f90 gen_grid.f90 inv.F90 invZGE.F90 m_fft3d_20150826.f90 m_rdinput.f90
  m_tetrahedron_20170325.F chiqw_sub.F wrt_optical_property.f90
  m_tetrahedron_fast.F chi_fast.f90 eps_async.f90 thread_util.c wfn_shared.f90)
respack_no_lto(chi_fast.f90)
if(RESPACK_CUDA)
  target_sources(chiqw_com PRIVATE chi_gpu.cu)
  target_link_libraries(chiqw_com PUBLIC CUDA::cudart CUDA::cublas CUDA::cufft)
elseif(RESPACK_HIP)
  # the CUDA source compiled as HIP; gpu_api.h maps the CUDA names onto HIP/hipFFT/hipBLAS
  set_source_files_properties(chi_gpu.cu PROPERTIES LANGUAGE HIP)
  target_sources(chiqw_com PRIVATE chi_gpu.cu)
  target_compile_definitions(chiqw_com PRIVATE $<$<COMPILE_LANGUAGE:HIP>:RESPACK_HIP>)
  target_link_libraries(chiqw_com PRIVATE hip::host hip::hipfft roc::hipblas)
  # ROCm clang does not default to PIE, while e.g. Ubuntu's gfortran links PIE executables
  set_target_properties(chiqw_com PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # gfx1200 (RX 9060/9070): FP16 GEMM kernel with round-to-nearest accumulation for the hi*hi product of
  # RESPACK_GPU_GEMM=fp16 (fp16_rn/: a patched hipBLASLt kernel, MIT License), assembled with ROCm's LLVM tools and
  # embedded; without it, or on other GPUs, chi_gpu.cu uses the chunked hipBLAS path
  option(RESPACK_HIP_FP16_RN "Embed the round-to-nearest FP16 GEMM kernel for gfx1200" ON)
  if(RESPACK_HIP_FP16_RN AND "${CMAKE_HIP_ARCHITECTURES}" MATCHES "gfx1200")
    find_program(RESPACK_LLVM_MC llvm-mc HINTS "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/lib/llvm/bin" NO_DEFAULT_PATH)
    find_program(RESPACK_LD_LLD ld.lld HINTS "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/lib/llvm/bin" NO_DEFAULT_PATH)
    if(RESPACK_LLVM_MC AND RESPACK_LD_LLD)
      set(_rn_src "${CMAKE_CURRENT_SOURCE_DIR}/fp16_rn/gemm_fp16_rn_gfx1200.s")
      set(_rn_obj "${CMAKE_CURRENT_BINARY_DIR}/gemm_fp16_rn_gfx1200.o")
      set(_rn_co "${CMAKE_CURRENT_BINARY_DIR}/gemm_fp16_rn_gfx1200.hsaco")
      add_custom_command(OUTPUT "${_rn_co}"
        COMMAND "${RESPACK_LLVM_MC}" -triple=amdgcn-amd-amdhsa -mcpu=gfx1200 -filetype=obj "${_rn_src}" -o "${_rn_obj}"
        COMMAND "${RESPACK_LD_LLD}" -shared "${_rn_obj}" -o "${_rn_co}"
        DEPENDS "${_rn_src}" COMMENT "Assembling the FP16 round-to-nearest GEMM kernel (gfx1200)" VERBATIM)
      target_sources(chiqw_com PRIVATE fp16_rn/fp16_rn_blob.c "${_rn_co}")
      set_source_files_properties(fp16_rn/fp16_rn_blob.c PROPERTIES OBJECT_DEPENDS "${_rn_co}"
        COMPILE_DEFINITIONS "RESPACK_FP16_RN_HSACO=\"${_rn_co}\"")
      respack_no_lto(fp16_rn/fp16_rn_blob.c)
      target_compile_definitions(chiqw_com PRIVATE $<$<COMPILE_LANGUAGE:HIP>:RESPACK_FP16_RN>)
      message(STATUS "HIP: FP16 round-to-nearest GEMM kernel for gfx1200 embedded")
    else()
      message(STATUS "HIP: llvm-mc/ld.lld not found under ${ROCM_PATH}: FP16 round-to-nearest kernel not built")
    endif()
  endif()
else()
  target_sources(chiqw_com PRIVATE chi_gpu_stub.c)
endif()
find_package(Threads REQUIRED)
target_link_libraries(chiqw_com PUBLIC respack_options FFTW3::fftw3 FFTW3::fftw3f LAPACK::LAPACK MPI::MPI_Fortran Threads::Threads m)
add_executable(calc_chiqw chiqw.F)
target_link_libraries(calc_chiqw PRIVATE chiqw_com)
set_target_properties(calc_chiqw PROPERTIES LINKER_LANGUAGE Fortran)
if((RESPACK_CUDA OR RESPACK_HIP) AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(calc_chiqw PRIVATE stdc++)
endif()
install(TARGETS calc_chiqw RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
