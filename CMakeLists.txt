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
option(RESPACK_HIP "Build the ROCm/HIP path of calc_chiqw for AMD GPUs (flg_gpu=1; needs ROCm with hipBLAS/hipFFT and CMake >= 3.21)" OFF)

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
if(RESPACK_CUDA AND RESPACK_HIP)
  message(FATAL_ERROR "RESPACK_CUDA and RESPACK_HIP are mutually exclusive")
endif()
if(RESPACK_HIP)
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(FATAL_ERROR "RESPACK_HIP needs CMake >= 3.21 (HIP language support)")
  endif()
  # ROCm location: -DROCM_PATH=... or environment ROCM_PATH, default /opt/rocm
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}")
    else()
      set(ROCM_PATH /opt/rocm)
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
  # default: the GPU architecture(s) of the build machine, detected by CMake;
  # e.g. -DCMAKE_HIP_ARCHITECTURES=gfx1200 (RX 9060/9070), gfx90a (MI210/MI250X), gfx942 (MI300)
  enable_language(HIP)
  find_package(hip REQUIRED CONFIG)
  find_package(hipfft REQUIRED CONFIG)
  find_package(hipblas REQUIRED CONFIG)
  message(STATUS "HIP: ${CMAKE_HIP_COMPILER_VERSION}, architectures ${CMAKE_HIP_ARCHITECTURES}, ROCm ${ROCM_PATH}")
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

# gfortran + LTO (RESPACK_LTO, or -flto given by hand in the Fortran flags): the LTO code generator does
# not inherit gfortran's rules for complex multiplication and division (-fcx-fortran-rules, set by the
# Fortran front end). It applies the C rules and calls __muldc3/__divdc3 for every complex product and
# quotient, which also prevents vectorisation (tetrahedron weights 2-3x slower, eps^-1 changed by ~3e-5).
# Pass the flag to the Fortran link steps.
string(TOUPPER "${CMAKE_BUILD_TYPE}" _respack_bt)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND
   (CMAKE_INTERPROCEDURAL_OPTIMIZATION OR "${CMAKE_Fortran_FLAGS} ${CMAKE_Fortran_FLAGS_${_respack_bt}}" MATCHES "-flto"))
  add_link_options("$<$<LINK_LANGUAGE:Fortran>:-fcx-fortran-rules>")
  message(STATUS "gfortran with LTO: -fcx-fortran-rules added to the link step")
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
message(STATUS "FFTW3:  ${FFTW3_LIBRARY}  (single precision: ${FFTW3F_LIBRARY})")
