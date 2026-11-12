#!/usr/bin/env bash
# Build + install the zen-opt fork of RESPACK on AMD Zen 3 (gfortran 13, OpenMPI 4.1.6).
#   usage: local-tools/build-zen3.sh [install-prefix]
# Dependencies (static, linked into the executables):
#   OpenBLAS 0.3.34 TARGET=ZEN single-threaded  (~/local/siesta-5.4.2/lib/libopenblas.a)
#   FFTW 3.3.11 AVX2/FMA, -march=znver3, LTO       (~/local/qe-7.6-zen3-lto/lib/libfftw3.a)
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${1:-$HOME/local/respack-20240804-zenopt}
BUILD=$HOME/local/build/respack-fork-zen3
FLAGS="-O3 -march=znver3 -mtune=znver3 -flto=auto -fopenmp -g -fbacktrace -ffree-line-length-none"
rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran \
  -DCMAKE_AR=/usr/bin/gcc-ar -DCMAKE_RANLIB=/usr/bin/gcc-ranlib \
  -DCMAKE_Fortran_FLAGS_RELEASE="$FLAGS" -DCMAKE_C_FLAGS_RELEASE="-O2" \
  -DMPI_Fortran_COMPILER=/usr/bin/mpif90 \
  -DBLA_VENDOR=OpenBLAS -DBLA_STATIC=ON -DCMAKE_LIBRARY_PATH="$HOME/local/siesta-5.4.2/lib" \
  -DFFTW3_ROOT="$HOME/local/qe-7.6-zen3-lto" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -Wno-dev
cmake --build "$BUILD" -j8
cmake --install "$BUILD"
echo "installed to $PREFIX (git $(git -C "$SRC" describe --always --dirty))"
