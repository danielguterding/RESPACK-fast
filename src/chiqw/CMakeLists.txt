# zen-opt fork: target-based CMake (modernised from the upstream CMake 2.8 file)
add_library(chiqw_com STATIC
  est_NTK.f90 est_nwx2.f90 gen_grid.f90 inv.F90 invZGE.F90 m_fft3d_20150826.f90 m_rdinput.f90
  m_tetrahedron_20170325.F chiqw_sub.F wrt_optical_property.f90
  m_tetrahedron_fast.F m_tetrahedron_soa.f90 chi_fast.f90 eps_async.f90 thread_util.c wfn_shared.f90)
respack_no_lto(chi_fast.f90)
# work arrays of the tetrahedron weights (sized by the number of frequencies) on the stack, not on the heap:
# gfortran otherwise allocates the automatic arrays of every call with malloc
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(m_tetrahedron_fast.F PROPERTIES COMPILE_OPTIONS "-fstack-arrays")
endif()
find_package(Threads REQUIRED)
target_link_libraries(chiqw_com PUBLIC respack_options FFTW3::fftw3 FFTW3::fftw3f LAPACK::LAPACK MPI::MPI_Fortran Threads::Threads m)
add_executable(calc_chiqw chiqw.F)
target_link_libraries(calc_chiqw PRIVATE chiqw_com)
install(TARGETS calc_chiqw RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
