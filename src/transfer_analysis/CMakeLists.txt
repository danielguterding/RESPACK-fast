# zen-opt fork: target-based CMake (modernised from the upstream CMake 2.8 file)
add_executable(calc_tr
  m_band.f90 m_dos.f90 m_eigenstate.f90 m_frmsf.f90 m_hist.f90 m_rd_dat_zvo.f90 m_rd_transdef.f90
  m_tetrahedron_20170325.F m_truncation.f90 transfer_analysis.f90 libtetrabz_dos.F90)
target_link_libraries(calc_tr PRIVATE respack_options LAPACK::LAPACK)
install(TARGETS calc_tr RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
