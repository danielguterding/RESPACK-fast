# RESPACK -- zen-opt fork. Build system modernised for CMake 3.18 ... 4.x (the upstream file required
# CMake 2.8 compatibility, which CMake >= 4.0 no longer provides).
# Modifications: Copyright (C) 2026 Daniel Guterding. GNU General Public License, version 3.
#
# Quick start (see README-zen-opt.md, section "Build"):
#   cmake --preset default            # or: native, zen3, debug   (CMakePresets.json)
#   cmake --build --preset default
#   cmake --install build/default --prefix <dir>
# or classic:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release [-DRESPACK_NATIVE=ON] [-DRESPACK_LTO=ON] [-DFFTW3_ROOT=<dir>]
cmake_minimum_required(VERSION 3.18...4.1)

# Languages are enabled after the optional legacy configuration file, which may set the compilers.
project(RESPACK VERSION 2024.08.04 DESCRIPTION "RESPACK (zen-opt fork)" LANGUAGES NONE)

# Legacy upstream configurations (config/gcc.cmake, config/intel.cmake, ...): -DCONFIG=gcc
if(CONFIG)
  message(STATUS "Loading legacy configuration: ${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake")
  include(${PROJECT_SOURCE_DIR}/config/${CONFIG}.cmake)
endif()
enable_language(Fortran)
enable_language(C)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GNUInstallDirs)

get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release, RelWithDebInfo, Debug)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#-----------------------------------------------------------------------------------------------------
# Options
option(RESPACK_NATIVE "Optimise for the build machine (-march=native / -xHost)" OFF)
set(RESPACK_ARCH "" CACHE STRING "Target CPU for -march/-mtune with GNU compilers (e.g. znver3); empty: compiler default")
option(RESPACK_LTO "Link-time (interprocedural) optimisation" OFF)
option(RESPACK_BUILD_XTAPP "Build the xTAPP converters (strconv, wfn2respack)" ON)
option(RESPACK_CUDA "Build the CUDA path of calc_chiqw (flg_gpu=1; needs CUDA toolkit with cuBLAS/cuFFT)" OFF)

#-----------------------------------------------------------------------------------------------------
# Dependencies
#   BLAS/LAPACK: standard FindLAPACK; select e.g. -DBLA_VENDOR=OpenBLAS -DBLA_STATIC=ON.
#   A single-threaded BLAS is recommended: BLAS is called from inside OpenMP regions
#   (otherwise set OPENBLAS_NUM_THREADS=1 at run time).
find_package(OpenMP REQUIRED COMPONENTS Fortran)
find_package(MPI REQUIRED COMPONENTS Fortran)
find_package(LAPACK REQUIRED)
find_package(FFTW3 REQUIRED)
if(RESPACK_CUDA)
  # default: all major GPU architectures (portable); e.g. -DCMAKE_CUDA_ARCHITECTURES=89 for L40S only
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES all-major)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA: ${CMAKE_CUDA_COMPILER_VERSION}, architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

#-----------------------------------------------------------------------------------------------------
# Common compile settings for all RESPACK targets
add_library(respack_options INTERFACE)
target_link_libraries(respack_options INTERFACE OpenMP::OpenMP_Fortran)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(respack_options INTERFACE
    $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none -fbacktrace>)
  if(RESPACK_NATIVE)
    target_compile_options(respack_options INTERFACE $<$<COMPILE_LANGUAGE:Fortran>:-march=native -mtune=native>)
  elseif(RESPACK_ARCH)
    target_compile_options(respack_options INTERFACE
      $<$<COMPILE_LANGUAGE:Fortran>:-march=${RESPACK_ARCH} -mtune=${RESPACK_ARCH}>)
  endif()
  # Release: -O3 (validated); the fork never uses -ffast-math
  if(NOT CMAKE_Fortran_FLAGS_RELEASE MATCHES "-O")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3" CACHE STRING "Fortran flags for Release" FORCE)
  endif()
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "^Intel")
  if(RESPACK_NATIVE)
    target_compile_options(respack_options INTERFACE $<$<COMPILE_LANGUAGE:Fortran>:-xHost>)
  endif()
endif()

if(RESPACK_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _respack_ipo OUTPUT _respack_ipo_msg LANGUAGES Fortran C)
  if(_respack_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS "Link-time optimisation: ON")
  else()
    message(WARNING "RESPACK_LTO requested but not supported: ${_respack_ipo_msg}")
  endif()
endif()

# The Fortran 2003 FFTW interface declares some arguments with types that differ from the C library
# (e.g. int vs unsigned).  Keep the Fortran->FFTW boundary out of LTO for the files calling FFTW.
function(respack_no_lto)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(${ARGN} PROPERTIES COMPILE_OPTIONS "-fno-lto")
  endif()
endfunction()

# installed executables find non-system shared libraries (e.g. a local OpenBLAS)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(src)
add_subdirectory(util)

install(FILES README.md README-zen-opt.md LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})

message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
message(STATUS "Fortran flags (${CMAKE_BUILD_TYPE}): ${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_RELEASE}")
message(STATUS "RESPACK_NATIVE=${RESPACK_NATIVE} RESPACK_ARCH=${RESPACK_ARCH} RESPACK_LTO=${RESPACK_LTO}")
message(STATUS "LAPACK: ${LAPACK_LIBRARIES}")
message(STATUS "FFTW3:  ${FFTW3_LIBRARY}")
