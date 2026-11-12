# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
#
# FindFFTW3
# ---------
# Finds the double-precision FFTW3 library and the Fortran 2003 interface file fftw3.f03.
#
# Hints (CMake or environment variables):
#   FFTW3_ROOT        installation prefix (include/fftw3.f03, lib/libfftw3.*)
#   FFTW3_USE_STATIC  prefer libfftw3.a over the shared library
# pkg-config (module fftw3) is used as a further hint when available.
#
# Also finds the single-precision library libfftw3f (used by fast_fft_sp=1); hint FFTW3F_ROOT.
#
# Result: FFTW3_FOUND, FFTW3_INCLUDE_DIR, FFTW3_LIBRARY, FFTW3F_LIBRARY and the imported targets
# FFTW3::fftw3 and FFTW3::fftw3f.

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_FFTW3 QUIET fftw3)
endif()

find_path(FFTW3_INCLUDE_DIR
  NAMES fftw3.f03
  HINTS ${FFTW3_ROOT} ENV FFTW3_ROOT ${PC_FFTW3_INCLUDEDIR} ${PC_FFTW3_INCLUDE_DIRS}
  PATH_SUFFIXES include)

set(_fftw3_names fftw3)
if(FFTW3_USE_STATIC OR "$ENV{FFTW3_USE_STATIC}")
  set(_fftw3_names libfftw3.a fftw3)
endif()
find_library(FFTW3_LIBRARY
  NAMES ${_fftw3_names}
  HINTS ${FFTW3_ROOT} ENV FFTW3_ROOT ${PC_FFTW3_LIBDIR} ${PC_FFTW3_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)
unset(_fftw3_names)
set(_fftw3f_names fftw3f)
if(FFTW3_USE_STATIC OR "$ENV{FFTW3_USE_STATIC}")
  set(_fftw3f_names libfftw3f.a fftw3f)
endif()
find_library(FFTW3F_LIBRARY
  NAMES ${_fftw3f_names}
  HINTS ${FFTW3F_ROOT} ENV FFTW3F_ROOT ${FFTW3_ROOT} ENV FFTW3_ROOT ${PC_FFTW3_LIBDIR} ${PC_FFTW3_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)
unset(_fftw3f_names)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
  REQUIRED_VARS FFTW3_LIBRARY FFTW3F_LIBRARY FFTW3_INCLUDE_DIR
  FAIL_MESSAGE "FFTW3 (double and single precision, with fftw3.f03) not found: set FFTW3_ROOT (and FFTW3F_ROOT)")

if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_LOCATION "${FFTW3_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
  if(FFTW3_LIBRARY MATCHES "\\.a$")
    # a static libfftw3 needs libm
    set_property(TARGET FFTW3::fftw3 APPEND PROPERTY INTERFACE_LINK_LIBRARIES m)
  endif()
endif()
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3f)
  add_library(FFTW3::fftw3f UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3f PROPERTIES
    IMPORTED_LOCATION "${FFTW3F_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
  if(FFTW3F_LIBRARY MATCHES "\\.a$")
    set_property(TARGET FFTW3::fftw3f APPEND PROPERTY INTERFACE_LINK_LIBRARIES m)
  endif()
endif()
mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY FFTW3F_LIBRARY)
