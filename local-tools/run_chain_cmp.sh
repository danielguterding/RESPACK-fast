#!/usr/bin/env bash
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
#
# Full chain (calc_chiqw, all q; calc_w3d; calc_j3d) with extra &param_chiqw entries, compared with a
# reference run of the same system:
#   run_chain_cmp.sh <bin-dir> <reference-run-dir> <work-dir> [entry ...]      e.g.  flg_gpu=1
# <bin-dir> contains calc_chiqw, calc_w3d, calc_j3d (or is a build tree: build/<preset>/src).
# The reference directory must contain dir-wfn, dir-wan, dir-model, input.in and its results.
# Environment: NP (MPI ranks, default 1), OMP_NUM_THREADS (default 8).
set -euo pipefail
BIN=$(readlink -f "$1"); REF=$(readlink -f "$2"); W=$(realpath -m "$3"); shift 3
TOOLS=$(cd "$(dirname "$0")" && pwd)
CQ=$( [[ -x $BIN/calc_chiqw ]] && echo $BIN/calc_chiqw || echo $BIN/chiqw/calc_chiqw )
CW=$( [[ -x $BIN/calc_w3d ]] && echo $BIN/calc_w3d || echo $BIN/calc_int/calc_w3d )
CJ=$( [[ -x $BIN/calc_j3d ]] && echo $BIN/calc_j3d || echo $BIN/calc_int/calc_j3d )
ulimit -s unlimited
export OMP_STACKSIZE=512M OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} LC_NUMERIC=C
rm -rf "$W"; mkdir -p "$W"; cd "$W"
ln -s "$(readlink -f "$REF/dir-wfn")" dir-wfn
cp -r "$REF/dir-wan" "$REF/dir-model" .
python3 -I - "$REF/input.in" "$@" > input.in <<'PY'
import sys
extra = sys.argv[2:]; keys = {e.split('=')[0].strip().lower() for e in extra}; out = []
for l in open(sys.argv[1]).read().split('\n'):
    if l.strip().lower().startswith('&param_chiqw'):
        out.append(l); out += [e.rstrip(',') + ',' for e in extra]; continue
    if l.split('=')[0].strip().lower() in keys: continue
    out.append(l)
print('\n'.join(out))
PY
t0=$(date +%s.%N); mpirun -np ${NP:-1} --bind-to none "$CQ" < input.in > log.chiqw 2>&1; t1=$(date +%s.%N)
"$CW" < input.in > log.calc_w3d 2>&1; t2=$(date +%s.%N)
"$CJ" < input.in > log.calc_j3d 2>&1; t3=$(date +%s.%N)
printf "chiqw %.1f s  w3d %.1f s  j3d %.1f s\n" "$(echo "$t1-$t0" | bc)" "$(echo "$t2-$t1" | bc)" "$(echo "$t3-$t2" | bc)"
grep 'chi_fast: time' log.chiqw | awk '{a+=$10; b+=$11; c+=$12} END {printf "chiqw phases (sum over q): pair densities %.1f s, tetrahedron %.1f s, gemm %.1f s\n", a, b, c}'
python3 -I "$TOOLS/cmp_epsqw.py" "$REF/dir-eps" dir-eps | tail -1
python3 -I "$TOOLS/compare_runs.py" "$REF" "$W" | grep -E 'dir-intW|dir-intJ' | head -2
for f in dir-intW/dat.UvsE.001-001 dir-intW/dat.UvsE.001-002 dir-intJ/dat.JvsE.001-002; do
  printf "  %-26s ref %s | this %s\n" $f "$(sed -n 3p "$REF/$f" | awk '{print $3}')" "$(sed -n 3p $f | awk '{print $3}')"
done
