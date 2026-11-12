# zen-opt fork: target-based CMake (modernised from the upstream CMake 2.8 file)
add_library(chiqw_com STATIC
  est_NTK.f90 est_nwx2.f90 gen_grid.f90 inv.F90 invZGE.F90 m_fft3d_20150826.f90 m_rdinput.f90
  m_tetrahedron_20170325.F chiqw_sub.F wrt_optical_property.f90
  m_tetrahedron_fast.F chi_fast.f90)
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
else()
  target_sources(chiqw_com PRIVATE chi_gpu_stub.c)
endif()
target_link_libraries(chiqw_com PUBLIC respack_options FFTW3::fftw3 FFTW3::fftw3f LAPACK::LAPACK MPI::MPI_Fortran m)
add_executable(calc_chiqw chiqw.F)
target_link_libraries(calc_chiqw PRIVATE chiqw_com)
set_target_properties(calc_chiqw PROPERTIES LINKER_LANGUAGE Fortran)
if((RESPACK_CUDA OR RESPACK_HIP) AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_link_libraries(calc_chiqw PRIVATE stdc++)
endif()
install(TARGETS calc_chiqw RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
