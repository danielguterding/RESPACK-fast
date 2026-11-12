#!/usr/bin/env bash
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
#
# Configure, build and install with one of the CMake presets (CMakePresets.json):
#   local-tools/build.sh [preset] [install-prefix]
#   presets: default (portable), native (-march=native + LTO), zen3 (znver3 + LTO + static OpenBLAS), debug
# The zen3 preset reads OPENBLAS_LIB_DIR (directory with libopenblas.a) and FFTW3_ROOT from the environment;
# FFTW3_ROOT is honoured by all presets.  Requires CMake >= 3.21 for presets.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
PRESET=${1:-default}
PREFIX=${2:-$HOME/.local/respack-zen-opt}
cmake -S "$SRC" --preset "$PRESET" -DCMAKE_INSTALL_PREFIX="$PREFIX"
cmake --build "$SRC/build/$PRESET" -j "${JOBS:-$(nproc)}"
cmake --install "$SRC/build/$PRESET"
echo "installed to $PREFIX (preset $PRESET, git $(git -C "$SRC" describe --always --dirty 2>/dev/null || echo n/a))"
