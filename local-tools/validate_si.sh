#!/usr/bin/env bash
# Example of the validation reported in README-zen-opt.md. Directory layout (B, test runs of the
# unmodified build) is specific to the machine it was run on -- adapt the variables below.
# Part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
# GNU General Public License, version 3 (see GPL/GPLv3.txt).
# Si (diamond, nonsymmorphic Fd-3m: fractional translations + time reversal) validation of the fork:
# original calc_chiqw vs fork, RPA and cRPA (sp3 Wannier, entangled), all 16 irreducible q
set -euo pipefail
cd "$(dirname "$0")"
B=$HOME/local/build/respack-20240804-zen3-20261112
SAMPLE=$HOME/local/src/respack-fork/sample/quantum-espresso/Si.fcc.6x6x6
ORIG=$HOME/local/respack/bin
FORK=$HOME/local/build/respack-fork-zen3/src/chiqw/calc_chiqw
ulimit -s unlimited
export OMP_STACKSIZE=512M OPENBLAS_NUM_THREADS=1

if [[ ! -d qe/dir-wfn ]]; then
  rm -rf qe; mkdir -p qe/tmp/scf; cp -r $SAMPLE/* qe/
  sed -i "s/diagonalization = cg/diagonalization = 'cg'/" qe/Si.scf.in   # sample bug: unquoted string
  ( cd qe; source ~/local/qe-7.6-zen3-lto/env.sh
    mpirun -np 8 pw.x -nk 4 -in Si.scf.in > Si.scf.out
    python3 ~/local/respack/bin/qe2respack.py tmp/scf/Si.save > log.qe2respack
    OMP_NUM_THREADS=8 $ORIG/calc_wannier < input.in > log.wannier )
fi

for crpa in 0 1; do
  for code in orig fork; do
    R=run-crpa$crpa-$code; rm -rf $R; mkdir -p $R; cd $R
    ln -s ../qe/dir-wfn dir-wfn; cp -r ../qe/dir-wan ../qe/dir-model .
    { echo "&param_chiqw"; echo "flg_cRPA=$crpa/"
      sed -n '/&param_wannier/,$p' ../qe/input.in; } > input.in
    bin=$ORIG/calc_chiqw; [[ $code == fork ]] && bin=$FORK
    s=$(date +%s.%N)
    OMP_NUM_THREADS=8 /usr/bin/mpirun -np 1 --bind-to none $bin < input.in > log.chiqw 2>&1
    e=$(date +%s.%N)
    OMP_NUM_THREADS=8 $ORIG/calc_w3d < input.in > log.calc_w3d 2>&1
    echo "crpa=$crpa $code chiqw wall $(python3 -c "print(round($e-$s,1))") s"
    cd ..
  done
  echo "== crpa=$crpa: eps^-1 orig vs fork"
  python3 -I ~/local/src/respack-fork/local-tools/cmp_epsqw.py run-crpa$crpa-orig/dir-eps run-crpa$crpa-fork/dir-eps
  echo "== crpa=$crpa: U(w=0) (1-1, 1-2, 1-5) orig | fork"
  for f in 001-001 001-002 001-005; do
    echo "  $f  $(sed -n 3p run-crpa$crpa-orig/dir-intW/dat.UvsE.$f | awk '{print $3,$4}') | $(sed -n 3p run-crpa$crpa-fork/dir-intW/dat.UvsE.$f | awk '{print $3,$4}')"
  done
  python3 -I $(dirname "$(readlink -f "$0")")/compare_runs.py run-crpa$crpa-orig run-crpa$crpa-fork | grep -v MISSING | head -5
done
