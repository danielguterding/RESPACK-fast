# zen-opt fork: target-based CMake (modernised from the upstream CMake 2.8 file)
add_executable(calc_w4
  calc_intW4.F inv.F90 est_NTK.f90 est_nwx2.f90 calc_intW4_sub.F m_rdinput.f90 m_fft3d_20150826.f90)
target_link_libraries(calc_w4 PRIVATE respack_options LAPACK::LAPACK m)
install(TARGETS calc_w4 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
