#!/usr/bin/env bash
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
#
# Accuracy of fast_gemm_sp=1 (chi0 accumulation as single-precision CGEMM blocks, double-precision sum)
# against the double-precision fork, full chains (calc_chiqw + calc_w3d + calc_j3d).
# Example of the test reported in README-zen-opt.md; adapt B/BIN to your machine.
set -uo pipefail
B=${B:-$HOME/local/build/respack-20240804-zen3-20261112}
BIN=${BIN:-$HOME/local/src/respack-fork/build/zen3/src}
TOOLS=$(cd "$(dirname "$0")" && pwd)
ulimit -s unlimited
export OMP_STACKSIZE=512M OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 LC_NUMERIC=C

mkinput() {  # src-input sp crpa(empty=unchanged) -> stdout
  python3 -I - "$1" "$2" "$3" <<'PY'
import sys
lines = open(sys.argv[1]).read().split('\n'); sp, crpa = sys.argv[2], sys.argv[3]; out = []
for l in lines:
    out.append(l)
    if l.strip().lower().startswith('&param_chiqw'):
        out.append(f'fast_gemm_sp={sp},')
        if crpa: out.append(f'flg_cRPA={crpa},')
print('\n'.join(out))
PY
}
chain() {  # run-dir qe-dir wan-src sp crpa
  local R=$1 QE=$2 WS=$3 SP=$4 CR=$5
  rm -rf $R; mkdir -p $R; cd $R
  ln -s $QE/dir-wfn dir-wfn; cp -r $WS/dir-wan $WS/dir-model .
  mkinput $QE/input.in $SP "$CR" > input.in
  s=$(date +%s.%N); mpirun -np 1 --bind-to none $BIN/chiqw/calc_chiqw < input.in > log.chiqw 2>&1; e=$(date +%s.%N)
  $BIN/calc_int/calc_w3d < input.in > log.calc_w3d 2>&1
  $BIN/calc_int/calc_j3d < input.in > log.calc_j3d 2>&1
  printf "  %-34s chiqw %6.1f s   zgemm/cgemm phase %6.1f s\n" "$(basename $(dirname $R))/$(basename $R)" \
    "$(python3 -c "print($e-$s)")" "$(grep 'chi_fast: time' log.chiqw | awk '{c+=$12} END {print c}')"
}
report() {  # dp-dir sp-dir
  python3 -I $TOOLS/cmp_epsqw.py $1/dir-eps $2/dir-eps | tail -1
  python3 -I $TOOLS/compare_runs.py $1 $2 | grep -E 'dir-intW|dir-intJ' | head -3
  for f in dir-intW/dat.UvsE.001-001 dir-intW/dat.UvsE.001-002 dir-intJ/dat.JvsE.001-002; do
    printf "    %-26s dp %s | sp %s\n" $f "$(sed -n 3p $1/$f | awk '{print $3}')" "$(sed -n 3p $2/$f | awk '{print $3}')"
  done
}
T=$B/test-SrVO3; S=$B/test-Si
echo "=== SrVO3 cRPA"
chain $T/sp-check/dp $T/qe $T/run-opt 0 ""
chain $T/sp-check/sp $T/qe $T/run-opt 1 ""
report $T/sp-check/dp $T/sp-check/sp
echo "=== Si RPA"
chain $S/sp-check/rpa-dp $S/qe $S/qe 0 ""
chain $S/sp-check/rpa-sp $S/qe $S/qe 1 ""
report $S/sp-check/rpa-dp $S/sp-check/rpa-sp
echo "=== Si cRPA"
chain $S/sp-check/crpa-dp $S/qe $S/qe 0 1
chain $S/sp-check/crpa-sp $S/qe $S/qe 1 1
report $S/sp-check/crpa-dp $S/sp-check/crpa-sp
