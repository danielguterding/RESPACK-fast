cmake_minimum_required(VERSION 2.8)
if(${CMAKE_PROJECT_NAME} STREQUAL "Project")
  message(FATAL_ERROR "cmake should be executed not for 'src' subdirectory, but for the top directory of RESPACK.")
endif(${CMAKE_PROJECT_NAME} STREQUAL "Project")

set(SOURCES_COMMON 
inv.F90 
est_NTK.f90 
est_nwx2.f90 
calc_intW4_sub.F 
m_rdinput.f90 
m_fft3d_20150826.f90
)
add_custom_target(w4 ALL)
add_executable(calc_w4 calc_intW4.F ${SOURCES_COMMON})
target_link_libraries(calc_w4 ${LAPACK_LIBRARIES} m)
add_dependencies(w4 calc_w4)

install(TARGETS calc_w4 RUNTIME DESTINATION bin)
