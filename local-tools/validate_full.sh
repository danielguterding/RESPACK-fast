#!/usr/bin/env bash
# Example of the validation reported in README-zen-opt.md. Directory layout (B, test runs of the
# unmodified build) is specific to the machine it was run on -- adapt the variables below.
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
# Final validation of the zen-opt fork against the original RESPACK 20240804 (opt build).
#  A: SrVO3 cRPA full chain (chiqw+w3d+j3d) fork vs original
#  B: SrVO3 fork chiqw with 2 MPI x 4 OMP vs 1 MPI x 8 OMP
#  C: Si RPA (sample default) full chain fork vs original (incl. j3d)
set -uo pipefail
B=$HOME/local/build/respack-20240804-zen3-20261112
T=$B/test-SrVO3; S=$B/test-Si
ORIG=$HOME/local/respack-20240804/bin   # the original RESPACK (~/local/respack is the fork)
FORK=$HOME/local/build/respack-fork-zen3/src
CMPE=$HOME/local/src/respack-fork/local-tools/cmp_epsqw.py
CMPR=$(cd "$(dirname "$0")" && pwd)/compare_runs.py
ulimit -s unlimited
export OMP_STACKSIZE=512M OPENBLAS_NUM_THREADS=1

run_chain() {  # dir, chiqw-bin, int-bin-dir, np, omp, qe-dir, wan-src
  local R=$1 CQ=$2 IB=$3 NP=$4 NT=$5 QE=$6 WS=$7
  rm -rf $R; mkdir -p $R; cd $R
  ln -s $QE/dir-wfn dir-wfn; cp -r $WS/dir-wan $WS/dir-model .; cp $QE/input.in .
  # the fork in double precision (its default is single precision; the original would reject the entries)
  [[ $CQ != $ORIG/* ]] && sed -i '0,/&param_chiqw/I{s/&param_chiqw/&\nfast_gemm_sp=0,\nfast_fft_sp=0,/I}' input.in
  s=$(date +%s.%N); OMP_NUM_THREADS=$NT /usr/bin/mpirun -np $NP --bind-to none $CQ < input.in > log.chiqw 2>&1; e=$(date +%s.%N)
  echo "  chiqw ($NP x $NT): $(python3 -c "print(round($e-$s,1))") s"
  if [[ -n $IB ]]; then
    s=$(date +%s.%N); OMP_NUM_THREADS=8 $IB/calc_w3d < input.in > log.calc_w3d 2>&1; e=$(date +%s.%N)
    echo "  w3d: $(python3 -c "print(round($e-$s,1))") s"
    s=$(date +%s.%N); OMP_NUM_THREADS=8 $IB/calc_j3d < input.in > log.calc_j3d 2>&1; e=$(date +%s.%N)
    echo "  j3d: $(python3 -c "print(round($e-$s,1))") s"
  fi
}
statics() {  # dir-a dir-b
  for f in dir-intW/dat.UvsE.001-001 dir-intW/dat.UvsE.001-002 dir-intJ/dat.JvsE.001-002; do
    printf "  %-26s %s | %s\n" $f "$(sed -n 3p $1/$f | awk '{print $3,$4}')" "$(sed -n 3p $2/$f | awk '{print $3,$4}')"
  done
}

echo "=== A: SrVO3 full chain, fork vs original (reference: run-opt eps + int-orig w3d/j3d)"
run_chain $T/final-fork $FORK/chiqw/calc_chiqw $FORK/calc_int 1 8 $T/qe $T/run-opt
python3 -I $CMPE $T/run-opt/dir-eps $T/final-fork/dir-eps | tail -1
python3 -I $CMPR $T/int-orig $T/final-fork | grep -v -E 'MISSING|dir-eps|dir-wan' | head -6
echo "  static values orig | fork"; statics $T/int-orig $T/final-fork

echo "=== B: SrVO3 fork chiqw 2 MPI x 4 OMP vs 1 x 8"
run_chain $T/final-fork-mpi2 $FORK/chiqw/calc_chiqw "" 2 4 $T/qe $T/run-opt
python3 -I $CMPE $T/final-fork/dir-eps $T/final-fork-mpi2/dir-eps | tail -1

echo "=== C: Si RPA full chain, fork vs original"
for v in orig fork; do
  if [[ $v == orig ]]; then CQ=$ORIG/calc_chiqw; IB=$ORIG; else CQ=$FORK/chiqw/calc_chiqw; IB=$FORK/calc_int; fi
  echo " $v:"; run_chain $S/final-$v $CQ $IB 1 8 $S/qe $S/qe
done
python3 -I $CMPE $S/final-orig/dir-eps $S/final-fork/dir-eps | tail -1
python3 -I $CMPR $S/final-orig $S/final-fork | grep -v -E 'MISSING|dir-eps|dir-wan' | head -6
echo "  static values orig | fork"; statics $S/final-orig $S/final-fork
