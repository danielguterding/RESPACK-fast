# zen-opt fork: target-based CMake (modernised from the upstream CMake 2.8 file)
add_executable(calc_wannier
  wannier.F wannier_sub.F
  det.F90 diagV.F90 est_NTK.f90 est_latparam.f90 est_nwx2.f90 gencif.f90 inv.F90
  inv_ge_LAPACK.F90 search_Rmin.f90 m_bvector_20171208.f90 m_dos.f90 m_eigenstate.f90
  m_fft3d_20150826.f90 m_gencif_sub.f90 m_rdinput.f90 m_tetrahedron_20170325.F makekpts.f90
  m_frmsf_wan.F90 m_wrt_frmsf.f90 m_wrt_model.f90 m_dmx.f90 m_fat_band.f90 m_tetrainteg.F)
target_link_libraries(calc_wannier PRIVATE respack_options LAPACK::LAPACK m)
install(TARGETS calc_wannier RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
