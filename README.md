# RESPACK — zen-opt fork (unofficial)

> **This is an unofficial, modified version of RESPACK 20240804.** It is not endorsed by or affiliated
> with the RESPACK developers. The original code and documentation are at
> <https://sites.google.com/view/kazuma7k6r>. Please cite the RESPACK paper:
> K. Nakamura *et al.*, Comput. Phys. Commun. **258**, 107781 (2021).
>
> **Modified by Daniel Guterding, November 2026.**
> - **`calc_chiqw`:** the band-pair loop is restructured: tiles, FFTW, one ZGEMM per frequency into a
>   shared χ. The generalized-tetrahedron weights are faster.
> - **`calc_w3d` / `calc_j3d`:** a real-space Wannier-function cache, FFTW, and ZGEMM contractions.
> - **Result:** about 5–8× faster on the test systems, and χ(G,G′,ω) is no longer duplicated per thread.
> - **Turning it off:** all changes can be switched off with `flg_fast=0` in `&param_chiqw` and
>   `&param_calc_int`.
> - **Details:** see [README-zen-opt.md](README-zen-opt.md): the modified files, validation against the
>   unmodified code, and performance.
>
> - **Validation scope:** checked on SrVO₃ (cRPA) and Si (RPA/cRPA) only. Please compare against
>   `flg_fast=0` for your own system before production use.
> - **Branches:** `upstream` contains the unmodified release tarball (RESPACK-20240804.tar.gz) and
>   `zen-opt` the modifications.
> - **Incomplete download:** the official tarball download ended prematurely in its last directory
>   (`src/calc_w4`). All files listed in the build system are present, but trailing files of the
>   original archive may be missing.
> - **License:** RESPACK is distributed under GPLv3 (see `GPL/`), and so is this fork. The
>   generalized-tetrahedron routines carry their own citation condition, or alternatively the MIT
>   license; see the header of `src/chiqw/m_tetrahedron_20170325.F`.

---

## RESPACK

## License

The distribution of the program package and the source codes for RESPACK follow GNU General Public License version 3 ([GPL v3](http://www.gnu.org/licenses/gpl-3.0.en.html)). 

## Requirement
  - Fortran compiler (intel, GNU)  
  - Python 2.7 required (Python 3 not supported at the moment)
  - BLAS library 
  - LAPACK library (intel MKL)  
  - MPI library (intelMPI, openMPI)  

  Note that RESPACK supports band-calculation softwares xTAPP and Quantum ESPRESSO at the moment, and before RESPACK calculations, these band calculations should be performed.  

## How to build

  - wannier (executable file: calc_wannier)  
    ```  
    $ cd RESPACK/src/wannier/  
    $ make   
  
  - chiqw (executable file: calc_chiqw)   
    ```  
    $ cd RESPACK/src/chiqw/  
    $ make  

  - calc_int (executable file: calc_w3d and calc_j3d)    
    ```  
    $ cd RESPACK/src/calc_int/  
    $ make  
  
## How to use

When the calculation directory is $(CalcDir), procedures to execute the job are as follows:

  - From xTAPP to RESPACK   
    ```
    $ cd RESPACK/util/xtapp2respack/  
    $ make  
    $ cp wfn2respack $(CalcDir)/.  
    $ cp strconv $(CalcDir)/.  
    $ cp xtapp2respack.sh $(CalcDir)/.  
    $ cd $(CalcDir)/  
    $ ./xtapp2respack.sh -b ./wfn2respack -s ./strconv MATERIAL-NAME 

    +++ MATERIAL-NAME is the name of the material performed in the xTAPP band calculation.  

  - From Quantum ESPRESSO to RESPACK   
    ```  
    $ cp RESPACK/util/qe2respack/qe2respack.py $(CalcDir)/.  
    $ python ./qe2respack.py outdir/prefix.save/ 

    +++ outdir/prefix.save/ is the directory containing the output of the Quantum-ESPRESSO band calculation.

  - RUN RESPACK at $(CalcDir)     
    ```
    $ export OMP_STACKSIZE=16 
    $ export OMP_NUM_THREADS=16
    $ mpirun -np 1 ./calc_wannier < input.in > log.wannier  
    $ mpirun -np 1 ./calc_chiqw < input.in > log.chiqw  
    $ mpirun -np 1 ./calc_w3d < input.in > log.calc_w3d  
    $ mpirun -np 1 ./calc_j3d < input.in > log.calc_j3d  

  - Transfer_analysis after RESPACK   
    ```  
    $ cp RESPACK/util/transfer_analysis/tr.py $(CalcDir)/   
    $ cd RESPACK/src/transfer_analysis/   
    $ make   
    $ cp calc_tr $(CalcDir)/.   
    $ cd $(CalcDir)/  
    $ python ./tr.py  

## Folder structure  

    ```
    RESPACK
    |-- GPL
    |-- config
    |-- man
    |   `-- en
    |-- sample
    |   |-- quantum-espresso
    |   |   |-- Al.fcc.6x6x6
    |   |   |   `-- PP
    |   |   |-- La2CuO4.bct.6x6x6
    |   |   |   `-- PP
    |   |   |-- Si.fcc.6x6x6
    |   |   |   `-- PP
    |   |   `-- SrVO3.sc.6x6x6
    |   |       `-- PP
    |   `-- xtapp
    |       |-- Al.fcc.6x6x6
    |       |-- La2CuO4.bct.6x6x6
    |       |-- Si.fcc.6x6x6
    |       |-- SrVO3.sc.6x6x6
    |       `-- TiO2.ort.6x6x6
    |           `-- mklocal
    |-- src
    |   |-- calc_int
    |   |-- chiqw
    |   |-- gw
    |   |-- transfer_analysis
    |   `-- wannier
    `-- util
        |-- qe2respack
        |-- transfer_analysis
        `-- xtapp2respack
