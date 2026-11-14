#!/usr/bin/env bash
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
#
# Quick numerical check of a build against a reference run of the same system:
#   check_build.sh <bin-dir> <reference-run-dir> <work-dir> [q-index]
# The reference run directory must contain dir-wfn (or a link), dir-wan, dir-model, dir-eps (all q),
# input.in and the dir-intW/dir-intJ results of calc_w3d/calc_j3d.
# Runs calc_chiqw for one q point and calc_w3d/calc_j3d on a copy of the reference dir-eps.
set -euo pipefail
BIN=$(readlink -f "$1"); REF=$(readlink -f "$2"); W=$(realpath -m "$3"); Q=${4:-2}
TOOLS=$(cd "$(dirname "$0")" && pwd)
ulimit -s unlimited
export OMP_STACKSIZE=512M OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
rm -rf "$W"; mkdir -p "$W/chiqw" "$W/int"
# (1) calc_chiqw, one q point
cd "$W/chiqw"
ln -s "$(readlink -f "$REF/dir-wfn")" dir-wfn
cp -r "$REF/dir-wan" .
python3 -I - "$REF/input.in" "$Q" > input.in <<'PY'
import sys
lines = open(sys.argv[1]).read().split('\n'); q = sys.argv[2]; out = []; inq = False; done = False
txt = open(sys.argv[1]).read().lower()
dp = [k + '=0,' for k in ('fast_gemm_sp', 'fast_fft_sp') if k not in txt]   # double precision unless set
for l in lines:
    if l.strip().lower().startswith('&param_chiqw'):
        out += [l, 'flg_calc_type=2,', 'n_calc_q=1,'] + dp; inq = True; continue
    out.append(l)
    if inq and not done and '/' in l.split('!')[0]:
        out.append(q); done = True
print('\n'.join(out))
PY
mpirun -np 1 --bind-to none "$BIN/calc_chiqw" < input.in > log.chiqw 2>&1
python3 -I "$TOOLS/cmp_epsqw.py" "$REF/dir-eps" dir-eps | tail -1
# (2) calc_w3d / calc_j3d on a copy (never a symlink) of the reference dir-eps
cd "$W/int"
ln -s "$(readlink -f "$REF/dir-wfn")" dir-wfn
cp -r "$REF/dir-wan" "$REF/dir-model" "$REF/dir-eps" .
cp "$REF/input.in" .
"$BIN/calc_w3d" < input.in > log.calc_w3d 2>&1
"$BIN/calc_j3d" < input.in > log.calc_j3d 2>&1
python3 -I "$TOOLS/compare_runs.py" "$REF" "$W/int" | grep -v -E 'MISSING|dir-eps|dir-wan' | head -4
