# zen-opt fork: target-based CMake (modernised from the upstream CMake 2.8 file)
add_library(chiqw_com STATIC
  est_NTK.f90 est_nwx2.f90 gen_grid.f90 inv.F90 invZGE.F90 m_fft3d_20150826.f90 m_rdinput.f90
  m_tetrahedron_20170325.F chiqw_sub.F wrt_optical_property.f90
  m_tetrahedron_fast.F chi_fast.f90 eps_async.f90 thread_util.c)
respack_no_lto(chi_fast.f90)
find_package(Threads REQUIRED)
target_link_libraries(chiqw_com PUBLIC respack_options FFTW3::fftw3 LAPACK::LAPACK MPI::MPI_Fortran Threads::Threads m)
add_executable(calc_chiqw chiqw.F)
target_link_libraries(calc_chiqw PRIVATE chiqw_com)
install(TARGETS calc_chiqw RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
