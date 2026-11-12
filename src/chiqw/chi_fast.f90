! This file is part of the zen-opt fork of RESPACK, a modified version of RESPACK 20240804
! (original: https://sites.google.com/view/kazuma7k6r, Copyright (c) 2017 Kazuma Nakamura et al.).
! Copyright (C) 2026 Daniel Guterding
!
! This program is free software: you can redistribute it and/or modify it under the terms of the
! GNU General Public License as published by the Free Software Foundation, version 3 of the License.
! This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
! even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
! General Public License (GPL/GPLv3.txt) for more details.
!
!
! m_chi_fast -- restructured band-pair loop of calc_chiqw (RESPACK fork, branch zen-opt)
!
! Computes exactly the same quantity as the original OpenMP band-pair loops in chiqw.F
! (q/=0: pc0d -> pchiqw ; q=0: pc0dgm -> pchiqwgm), i.e. for every frequency w
!
!   chi(G,G',w) += sum_{pairs p} sum_k conj(M_pk(G)) * x_pk(w)/NTK * (1-P_occ*P_vir) * M_pk(G')
!
! with M_pk(G) the pair density of (occupied band j at k, virtual band i at k+q) and x_pk(w) the
! generalized-tetrahedron weights.  Only the organisation of the work differs:
!
!  1. Pairs are processed in tiles (block of occupied x block of virtual bands).  For every k the
!     symmetry-rotated wave functions of all bands of a tile are FFT'ed to real space once (the
!     original code did this twice per band pair and k).
!  2. FFTW (planned once) replaces the built-in FFT.  Conventions are identical: backward FFT
!     unnormalised with exp(+i2pi G.r/N), forward FFT with exp(-i2pi G.r/N) and 1/N.
!  3. The umklapp phase exp(-i2pi dG.r/N) applied to u_{k+q}(r) in the original is applied
!     exactly by placing the coefficient C(G) at grid index G-dG before the FFT.
!  4. Sum over k and pairs is one ZGEMM per frequency with K = NTK*npairs rows, parallelised over
!     frequencies into a single shared accumulator (no per-thread copies of chi(G,G',w)).
!  5. q=0: the three head/wing variants differ only in the G=0 row/column of the pair density,
!     so one ZGEMM plus three ZGEMV-based row/column corrections replace three ZGEMMs.
!
! Rounding-level differences to the original arise from the different summation order and FFT.
!
module m_chi_fast
  use, intrinsic :: iso_c_binding
  use omp_lib
  use m_tetrahedron_fast, only: ttrhdrn_chi0_pm
  implicit none
  include 'fftw3.f03'
  private
  public :: chi_fast_init, chi_fast_pairs

  integer, save :: n1 = 0, n2 = 0, n3 = 0, ngrid = 0, ldg = 0
  type(c_ptr), save :: plan_bw = c_null_ptr, plan_fw = c_null_ptr
  type(c_ptr), save :: plan_bw4 = c_null_ptr, plan_fw4 = c_null_ptr   ! fast_fft_sp=1 (fftwf)
  logical, save :: fft_sp = .false.
  real(8), save :: mem_budget_mb = 4096d0
  logical, save :: verbose = .false.
  logical, save :: gemm_sp = .false.   ! fast_gemm_sp=1: single-precision CGEMM blocks, double-precision accumulation
  logical, save :: use_gpu = .false.   ! flg_gpu=1: CUDA path (chi_gpu.cu)

  interface
    integer(c_int) function chi_gpu_available() bind(C, name='chi_gpu_available')
      import :: c_int
    end function chi_gpu_available
    integer(c_int) function chi_gpu_init(n1, n2, n3, fbatch, verbose, fft_sp) bind(C, name='chi_gpu_init')
      import :: c_int
      integer(c_int), value :: n1, n2, n3, fbatch, verbose, fft_sp
    end function chi_gpu_init
    integer(c_int) function chi_gpu_q_begin(NG, nen, lq0, ext) bind(C, name='chi_gpu_q_begin')
      import :: c_int
      integer(c_int), value :: NG, nen, lq0
      integer(c_int), intent(in) :: ext(*)
    end function chi_gpu_q_begin
    integer(c_int) function chi_gpu_tile_begin(rows, nu, np, pj, pi) bind(C, name='chi_gpu_tile_begin')
      import :: c_int
      integer(c_int), value :: rows, nu, np
      integer(c_int), intent(in) :: pj(*), pi(*)
    end function chi_gpu_tile_begin
    integer(c_int) function chi_gpu_bands(nb, ldc, coef, ncoef, idx, ubase) bind(C, name='chi_gpu_bands')
      import :: c_int, c_double_complex
      integer(c_int), value :: nb, ldc, ncoef, ubase
      complex(c_double_complex), intent(in) :: coef(*)
      integer(c_int), intent(in) :: idx(*)
    end function chi_gpu_bands
    integer(c_int) function chi_gpu_pairs(ik0, NTK, ubJ, ubI, scl) bind(C, name='chi_gpu_pairs')
      import :: c_int, c_double
      integer(c_int), value :: ik0, NTK, ubJ, ubI
      real(c_double), value :: scl
    end function chi_gpu_pairs
    integer(c_int) function chi_gpu_gemm(W, V, kc) bind(C, name='chi_gpu_gemm')
      import :: c_int, c_double_complex
      complex(c_double_complex), intent(in) :: W(*), V(*)
      integer(c_int), value :: kc
    end function chi_gpu_gemm
    integer(c_int) function chi_gpu_q_end(chi, Y, Z) bind(C, name='chi_gpu_q_end')
      import :: c_int, c_double_complex
      complex(c_double_complex) :: chi(*), Y(*), Z(*)
    end function chi_gpu_q_end
  end interface

  complex(8), parameter :: zone = (1d0, 0d0), zzero = (0d0, 0d0), zimag = (0d0, 1d0)

contains

  !---------------------------------------------------------------------------------------------
  subroutine chi_fast_init(nx, ny, nz, mem_mb, lverbose, lgemm_sp, lgpu, lfft_sp)
    integer, intent(in) :: nx, ny, nz
    real(8), intent(in) :: mem_mb
    logical, intent(in) :: lverbose, lgemm_sp, lgpu, lfft_sp
    integer(c_int) :: rc
    complex(c_float_complex), pointer :: buf4(:)
    type(c_ptr) :: p
    complex(c_double_complex), pointer :: buf(:)

    n1 = nx; n2 = ny; n3 = nz
    ngrid = nx*ny*nz
    ldg = ((ngrid + 7)/8)*8        ! column stride: multiple of 128 bytes -> all columns keep the planning alignment
    if (mem_mb > 0d0) mem_budget_mb = mem_mb
    verbose = lverbose
    gemm_sp = lgemm_sp
    p = fftw_alloc_complex(int(ldg, c_size_t))
    call c_f_pointer(p, buf, [ldg])
    ! Fortran array u(n1,n2,n3) <-> FFTW row-major (n3,n2,n1)
    plan_bw = fftw_plan_dft_3d(n3, n2, n1, buf, buf, FFTW_BACKWARD, FFTW_MEASURE)
    plan_fw = fftw_plan_dft_3d(n3, n2, n1, buf, buf, FFTW_FORWARD, FFTW_MEASURE)
    call fftw_free(p)
    fft_sp = lfft_sp
    if (fft_sp .and. .not. lgpu) then
      p = fftwf_alloc_complex(int(ldg, c_size_t))
      call c_f_pointer(p, buf4, [ldg])
      plan_bw4 = fftwf_plan_dft_3d(n3, n2, n1, buf4, buf4, FFTW_BACKWARD, FFTW_MEASURE)
      plan_fw4 = fftwf_plan_dft_3d(n3, n2, n1, buf4, buf4, FFTW_FORWARD, FFTW_MEASURE)
      call fftwf_free(p)
      if (.not. c_associated(plan_bw4) .or. .not. c_associated(plan_fw4)) then
        write(6,*) 'chi_fast_init: FFTW (single precision) planning failed'; stop
      end if
    end if
    if (.not. c_associated(plan_bw) .or. .not. c_associated(plan_fw)) then
      write(6,*) 'chi_fast_init: FFTW planning failed'; stop
    end if
    if (verbose) write(6,'(a,3i5,a,f10.1,a,l2,a,l2)') ' chi_fast: FFTW grid ', n1, n2, n3, ' ; memory budget (MB) ', &
         mem_budget_mb, ' ; single-precision GEMM:', gemm_sp, ' ; single-precision FFT:', fft_sp
    use_gpu = lgpu
    if (use_gpu) then
      if (chi_gpu_available() /= 1) then
        write(6,*) 'flg_gpu=1, but calc_chiqw was built without CUDA (cmake -DRESPACK_CUDA=ON)'; stop
      end if
      rc = chi_gpu_init(nx, ny, nz, 16, merge(1, 0, lverbose), merge(1, 0, lfft_sp))
      if (rc /= 0) then
        write(6,*) 'chi_gpu_init failed, code ', rc; stop
      end if
      if (verbose) write(6,'(a)') ' chi_fast: GPU path (pair densities and chi0 on the GPU, single-precision GEMM blocks)'
    end if
  end subroutine chi_fast_init

  !---------------------------------------------------------------------------------------------
  pure integer function gidx(i, j, k)
    integer, intent(in) :: i, j, k
    gidx = modulo(i, n1) + 1 + modulo(j, n2)*n1 + modulo(k, n3)*n1*n2
  end function gidx

  !---------------------------------------------------------------------------------------------
  subroutine chi_fast_pairs(lq0, NTK, NTG, Nk_irr, nbufo, nbufv, NTB, N_PAIR, ipair, jpair, &
       OCC, VIR, E_OCC, E_VIR, P_OCC, P_VIR, &
       SK0, q, numirr, numrot, trs, NG0, KG0, RW, rginv, pg, nsymq, nnp, L1, L2, L3, packing, &
       LG0, NGeps, No_G_0, b1, b2, b3, MAT_NL, off_v, off_o, &
       dmna, dmnr, nkb1, nkb2, nkb3, imt1, index_kpt, Ef, delt, nen, em, &
       pchi, pchigm)
    logical, intent(in) :: lq0
    integer, intent(in) :: NTK, NTG, Nk_irr, nbufo, nbufv, NTB, N_PAIR
    integer, intent(in) :: ipair(N_PAIR), jpair(N_PAIR)
    complex(8), intent(in) :: OCC(NTG, nbufo, Nk_irr), VIR(NTG, nbufv, Nk_irr)
    real(8), intent(in) :: E_OCC(nbufo, Nk_irr), E_VIR(nbufv, Nk_irr)
    complex(8), intent(in) :: P_OCC(nbufo, NTK), P_VIR(nbufv, NTK)
    real(8), intent(in) :: SK0(3, NTK), q(3)
    integer, intent(in) :: numirr(NTK), numrot(NTK), trs(NTK), NG0(NTK), KG0(3, NTG, NTK), RW(3, NTK)
    integer, intent(in) :: nsymq, nnp, L1, L2, L3
    real(8), intent(in) :: rginv(3, 3, nsymq)
    integer, intent(in) :: pg(3, nsymq)
    integer, intent(in) :: packing(-L1:L1, -L2:L2, -L3:L3, Nk_irr)
    integer, intent(in) :: NGeps, No_G_0
    integer, intent(in) :: LG0(3, NTG)
    real(8), intent(in) :: b1(3), b2(3), b3(3)
    complex(8), intent(in) :: MAT_NL(3, NTB, NTB, Nk_irr)
    integer, intent(in) :: off_v, off_o
    real(8), intent(in) :: dmna, dmnr, Ef, delt
    integer, intent(in) :: nkb1, nkb2, nkb3, nen
    integer, intent(in) :: imt1(*), index_kpt(nkb1, nkb2, nkb3)
    complex(8), intent(in) :: em(nen)
    complex(8), intent(inout), optional, contiguous :: pchi(:, :, :)        ! (NGeps,NGeps,nen)   q/=0
    complex(8), intent(inout), optional, contiguous :: pchigm(:, :, :, :)   ! (NGeps,NGeps,nen,3) q=0

    external :: search_kq, make_C0

    integer, allocatable :: kpq(:), shG(:, :), ext(:), k3d(:)
    integer, allocatable :: jlist(:), ilist(:), posj(:), posi(:)
    integer, allocatable :: tcount(:), tstart(:), tpairs(:), fill(:)
    integer :: nJ, nI, bj, bi, nbj, nbi, ntile, it, ip, ik, ikb1, ikb2, ikb3, t
    integer :: np, rows, nthr
    real(8) :: bytes_pair, bytes_thrA, budget, scl
    integer(8) :: npmax
    complex(8), allocatable :: A(:, :), W(:, :), V(:, :)
    complex(4), allocatable :: A4(:, :)
    complex(8), allocatable :: chiT(:, :, :), Yh(:, :, :), Zh(:, :, :), Dacc(:, :)
    integer, allocatable :: ext0(:)
    integer(c_int) :: rcg
    integer, allocatable :: tj(:), ti(:), pjl(:), pil(:)
    integer :: ntj, nti
    real(8) :: t0, tA, tB, tC

    if (ngrid == 0) then
      write(6,*) 'chi_fast_pairs: chi_fast_init not called'; stop
    end if
    if (N_PAIR <= 0) return
    nthr = omp_get_max_threads()
    scl = 1d0/dble(ngrid)

    !--- k+q mapping and umklapp shifts
    allocate(kpq(NTK), shG(3, NTK))
    do ik = 1, NTK
      shG(:, ik) = 0
      if (lq0) then
        kpq(ik) = ik
      else
        call search_kq(NTK, SK0, q(1), q(2), q(3), ik, kpq(ik), shG(1, ik))
      end if
    end do
    !--- extraction indices (-G_L on the compact grid), k -> position in the 3D tetrahedron mesh
    allocate(ext(NGeps))
    do t = 1, NGeps
      ext(t) = gidx(-LG0(1, t), -LG0(2, t), -LG0(3, t))
    end do
    allocate(k3d(NTK))
    do ikb3 = 1, nkb3
      do ikb2 = 1, nkb2
        do ikb1 = 1, nkb1
          k3d(index_kpt(ikb1, ikb2, ikb3)) = ikb1 + (ikb2 - 1)*nkb1 + (ikb3 - 1)*nkb1*nkb2
        end do
      end do
    end do

    !--- distinct occupied / virtual bands occurring in the pair list
    allocate(posj(nbufo), posi(nbufv))
    posj = 0; posi = 0
    do ip = 1, N_PAIR
      posj(jpair(ip)) = 1; posi(ipair(ip)) = 1
    end do
    nJ = count(posj > 0); nI = count(posi > 0)
    allocate(jlist(nJ), ilist(nI))
    nJ = 0
    do t = 1, nbufo
      if (posj(t) > 0) then; nJ = nJ + 1; jlist(nJ) = t; posj(t) = nJ; end if
    end do
    nI = 0
    do t = 1, nbufv
      if (posi(t) > 0) then; nI = nI + 1; ilist(nI) = t; posi(t) = nI; end if
    end do

    !--- tile size from the memory budget
    bytes_pair = 16d0*dble(NTK)*dble(NGeps + nen + merge(3, 0, lq0))     ! A, W, V rows of one pair
    budget = 0.7d0*mem_budget_mb*1048576d0
    npmax = max(1_8, int(budget/bytes_pair, 8))
    bj = int(min(dble(nJ), max(1d0, sqrt(dble(npmax)))))
    bi = int(min(dble(nI), max(1d0, dble(npmax)/dble(bj))))
    if (bi == nI) bj = int(min(dble(nJ), max(1d0, dble(npmax)/dble(nI))))
    ! per-thread real-space buffers of phase A must also fit (into the remaining 30 %)
    do
      bytes_thrA = 16d0*dble(ldg)*dble(bj + bi + 1)*dble(nthr)
      if (bytes_thrA <= 0.3d0*mem_budget_mb*1048576d0 .or. (bj == 1 .and. bi == 1)) exit
      if (bj >= bi) then; bj = max(1, bj/2); else; bi = max(1, bi/2); end if
    end do
    nbj = (nJ + bj - 1)/bj
    nbi = (nI + bi - 1)/bi
    ntile = nbj*nbi

    !--- bucket the pairs into tiles (original order kept inside a tile)
    allocate(tcount(ntile), tstart(ntile + 1), tpairs(N_PAIR), fill(ntile))
    tcount = 0
    do ip = 1, N_PAIR
      t = ((posj(jpair(ip)) - 1)/bj)*nbi + (posi(ipair(ip)) - 1)/bi + 1
      tcount(t) = tcount(t) + 1
    end do
    tstart(1) = 1
    do t = 1, ntile
      tstart(t + 1) = tstart(t) + tcount(t)
    end do
    fill = 0
    do ip = 1, N_PAIR
      t = ((posj(jpair(ip)) - 1)/bj)*nbi + (posi(ipair(ip)) - 1)/bi + 1
      tpairs(tstart(t) + fill(t)) = ip
      fill(t) = fill(t) + 1
    end do
    if (verbose) write(6,'(a,l2,a,i8,a,2i6,a,2i6,a,i6)') ' chi_fast: q0=', lq0, ' pairs=', N_PAIR, &
         ' bands(occ,vir)=', nJ, nI, ' tile=', bj, bi, ' ntile=', ntile

    tA = 0d0; tB = 0d0; tC = 0d0
    allocate(tj(bj), ti(bi))
    if (use_gpu) then
      allocate(ext0(NGeps))
      ext0 = ext - 1
      rcg = chi_gpu_q_begin(NGeps, nen, merge(1, 0, lq0), ext0)
      call gpu_check(rcg, 'chi_gpu_q_begin')
      if (lq0) then
        allocate(Dacc(nen, 3))
        Dacc = zzero
      end if
    end if
    do it = 1, ntile
      np = tcount(it)
      if (np == 0) cycle
      rows = NTK*np
      ! bands of this tile and local pair -> (jl, il)
      ntj = 0; nti = 0
      allocate(pjl(np), pil(np))
      block
        integer :: ljpos(nbufo), lipos(nbufv), jb, ib
        ljpos = 0; lipos = 0
        do t = 1, np
          ip = tpairs(tstart(it) + t - 1)
          jb = jpair(ip); ib = ipair(ip)
          if (ljpos(jb) == 0) then; ntj = ntj + 1; tj(ntj) = jb; ljpos(jb) = ntj; end if
          if (lipos(ib) == 0) then; nti = nti + 1; ti(nti) = ib; lipos(ib) = nti; end if
          pjl(t) = ljpos(jb); pil(t) = lipos(ib)
        end do
      end block

      if (use_gpu) then
        allocate(A(1, 1), W(rows, nen))         ! pair densities live on the GPU
      else
        allocate(A(rows, NGeps), W(rows, nen))
      end if
      if (lq0) then
        allocate(V(rows, 3))
      else
        allocate(V(1, 1))
      end if

      t0 = omp_get_wtime()
      if (use_gpu) then
        call phase_a_gpu()
      else if (fft_sp) then
        call phase_a_sp()
      else
        call phase_a()
      end if
      tA = tA + omp_get_wtime() - t0
      t0 = omp_get_wtime()
      call phase_b()
      tB = tB + omp_get_wtime() - t0
      t0 = omp_get_wtime()
      if (use_gpu) then
        call phase_c_gpu()
      else
        call phase_c()
      end if
      tC = tC + omp_get_wtime() - t0

      deallocate(A, W, V, pjl, pil)
    end do
    if (use_gpu) then
      if (lq0) then
        allocate(chiT(NGeps, NGeps, nen), Yh(NGeps, 3, nen), Zh(NGeps, 3, nen))
        rcg = chi_gpu_q_end(chiT, Yh, Zh)
        call gpu_check(rcg, 'chi_gpu_q_end')
        ! same assembly as the CPU path, with chi, y, z, d summed over all tiles
        block
          integer :: ie, ix, G0
          G0 = No_G_0
          do ie = 1, nen
            do ix = 1, 3
              pchigm(:, :, ie, ix) = pchigm(:, :, ie, ix) + chiT(:, :, ie)
              pchigm(:, G0, ie, ix) = pchigm(:, G0, ie, ix) + Yh(:, ix, ie) - chiT(:, G0, ie)
              pchigm(G0, :, ie, ix) = pchigm(G0, :, ie, ix) + Zh(:, ix, ie) - chiT(G0, :, ie)
              pchigm(G0, G0, ie, ix) = pchigm(G0, G0, ie, ix) + Dacc(ie, ix) - Yh(G0, ix, ie) - Zh(G0, ix, ie) &
                   + chiT(G0, G0, ie)
            end do
          end do
        end block
        deallocate(chiT, Yh, Zh, Dacc)
      else
        block
          complex(8) :: dmy(1)
          rcg = chi_gpu_q_end(pchi, dmy, dmy)
        end block
        call gpu_check(rcg, 'chi_gpu_q_end')
      end if
      deallocate(ext0)
    end if
    if (verbose) write(6,'(a,3f10.2)') ' chi_fast: time (s) pair densities / tetrahedron / zgemm ', tA, tB, tC

  contains

    !------------------------------------------------------------------------------------------
    ! phase A: pair densities M_pk(G) -> A(ik+NTK*(p-1), G); q=0 also the k.p G=0 elements -> V
    subroutine phase_a()
      !$omp parallel default(shared)
      block
        type(c_ptr) :: pbuf
        complex(c_double_complex), pointer :: ubuf(:, :)
        complex(8), allocatable :: cbuf(:), cJ(:, :), cI(:, :), pmap(:)
        integer, allocatable :: jmap(:), gpos(:)
        integer :: ik, ikq, ikir, iop, ikirq, iopq, jl, il, ig, lp, row, jb, ib, ix, g
        complex(8) :: sumv(3), vm(3), nlvm(3)
        real(8) :: gv(3), del

        pbuf = fftw_alloc_complex(int(ldg, c_size_t)*int(bj + bi + 1, c_size_t))
        call c_f_pointer(pbuf, ubuf, [ldg, bj + bi + 1])     ! columns 1..ntj: occ, bj+1..bj+nti: vir, last: product
        allocate(cbuf(NTG), pmap(NTG), jmap(NTG), gpos(NTG))
        if (lq0) then
          allocate(cJ(NTG, bj), cI(NTG, bi))
        end if

        !$omp do schedule(dynamic,1)
        do ik = 1, NTK
          ikq = kpq(ik)
          ikir = numirr(ik); iop = numrot(ik)
          ! rotation map, phase factors and grid positions once per k point (make_C0 per band before)
          call make_C0_map(NTG, trs(ik), NG0(ik), KG0(1, 1, ik), RW(1, ik), rginv(1, 1, iop), pg(1, iop), &
               nnp, L1, L2, L3, packing(-L1, -L2, -L3, ikir), jmap, pmap)
          do ig = 1, NG0(ik)
            gpos(ig) = gidx(KG0(1, ig, ik), KG0(2, ig, ik), KG0(3, ig, ik))
          end do
          do jl = 1, ntj
            jb = tj(jl)
            call c0_apply(NG0(ik), trs(ik), jmap, pmap, OCC(1, jb, ikir), cbuf)
            ubuf(:, jl) = zzero
            do ig = 1, NG0(ik)
              ubuf(gpos(ig), jl) = cbuf(ig)
            end do
            call fftw_execute_dft(plan_bw, ubuf(:, jl), ubuf(:, jl))
            if (lq0) cJ(1:NG0(ik), jl) = cbuf(1:NG0(ik))
          end do
          ikirq = numirr(ikq); iopq = numrot(ikq)
          call make_C0_map(NTG, trs(ikq), NG0(ikq), KG0(1, 1, ikq), RW(1, ikq), rginv(1, 1, iopq), pg(1, iopq), &
               nnp, L1, L2, L3, packing(-L1, -L2, -L3, ikirq), jmap, pmap)
          do ig = 1, NG0(ikq)
            gpos(ig) = gidx(KG0(1, ig, ikq) - shG(1, ik), KG0(2, ig, ikq) - shG(2, ik), KG0(3, ig, ikq) - shG(3, ik))
          end do
          do il = 1, nti
            ib = ti(il)
            call c0_apply(NG0(ikq), trs(ikq), jmap, pmap, VIR(1, ib, ikirq), cbuf)
            ubuf(:, bj + il) = zzero
            do ig = 1, NG0(ikq)
              ubuf(gpos(ig), bj + il) = cbuf(ig)
            end do
            call fftw_execute_dft(plan_bw, ubuf(:, bj + il), ubuf(:, bj + il))
            if (lq0) cI(1:NG0(ikq), il) = cbuf(1:NG0(ikq))
          end do
          do lp = 1, np
            jl = pjl(lp); il = pil(lp)
            row = ik + NTK*(lp - 1)
            ubuf(1:ngrid, bj + bi + 1) = conjg(ubuf(1:ngrid, bj + il))*ubuf(1:ngrid, jl)
            call fftw_execute_dft(plan_fw, ubuf(:, bj + bi + 1), ubuf(:, bj + bi + 1))
            do g = 1, NGeps
              A(row, g) = ubuf(ext(g), bj + bi + 1)*scl
            end do
            if (lq0) then
              ! calc_VMab (ikq=ik at q=0) + non-local correction + energy denominator, as in chiqw.F
              jb = tj(jl); ib = ti(il)
              sumv = zzero
              do ig = 1, NG0(ik)
                gv(:) = dble(KG0(1, ig, ik))*b1(:) + dble(KG0(2, ig, ik))*b2(:) + dble(KG0(3, ig, ik))*b3(:)
                sumv(:) = sumv(:) + gv(:)*conjg(cI(ig, il))*cJ(ig, jl)
              end do
              vm(:) = sumv(:)*zimag
              del = E_VIR(ib, numirr(ikq)) - E_OCC(jb, numirr(ik))
              if (abs(del) < delt) del = delt
              nlvm(:) = MAT_NL(:, ib + off_v, jb + off_o, numirr(ik))
              vm(:) = -zimag*(vm(:) - nlvm(:))/del
              do ix = 1, 3
                V(row, ix) = vm(ix)
                if (abs(del) <= delt) V(row, ix) = zzero
              end do
            end if
          end do
        end do
        !$omp end do

        deallocate(cbuf, pmap, jmap, gpos)
        if (lq0) deallocate(cJ, cI)
        call fftw_free(pbuf)
      end block
      !$omp end parallel
    end subroutine phase_a

    subroutine phase_a_sp()   ! as phase_a, grid functions/FFTs in single precision (fast_fft_sp=1)
      !$omp parallel default(shared)
      block
        type(c_ptr) :: pbuf
        complex(c_float_complex), pointer :: ubuf(:, :)
        complex(8), allocatable :: cbuf(:), cJ(:, :), cI(:, :), pmap(:)
        integer, allocatable :: jmap(:), gpos(:)
        integer :: ik, ikq, ikir, iop, ikirq, iopq, jl, il, ig, lp, row, jb, ib, ix, g
        complex(8) :: sumv(3), vm(3), nlvm(3)
        real(8) :: gv(3), del

        pbuf = fftwf_alloc_complex(int(ldg, c_size_t)*int(bj + bi + 1, c_size_t))
        call c_f_pointer(pbuf, ubuf, [ldg, bj + bi + 1])     ! columns 1..ntj: occ, bj+1..bj+nti: vir, last: product
        allocate(cbuf(NTG), pmap(NTG), jmap(NTG), gpos(NTG))
        if (lq0) then
          allocate(cJ(NTG, bj), cI(NTG, bi))
        end if

        !$omp do schedule(dynamic,1)
        do ik = 1, NTK
          ikq = kpq(ik)
          ikir = numirr(ik); iop = numrot(ik)
          call make_C0_map(NTG, trs(ik), NG0(ik), KG0(1, 1, ik), RW(1, ik), rginv(1, 1, iop), pg(1, iop), &
               nnp, L1, L2, L3, packing(-L1, -L2, -L3, ikir), jmap, pmap)
          do ig = 1, NG0(ik)
            gpos(ig) = gidx(KG0(1, ig, ik), KG0(2, ig, ik), KG0(3, ig, ik))
          end do
          do jl = 1, ntj
            jb = tj(jl)
            call c0_apply(NG0(ik), trs(ik), jmap, pmap, OCC(1, jb, ikir), cbuf)
            ubuf(:, jl) = (0.0, 0.0)
            do ig = 1, NG0(ik)
              ubuf(gpos(ig), jl) = cmplx(cbuf(ig), kind=4)
            end do
            call fftwf_execute_dft(plan_bw4, ubuf(:, jl), ubuf(:, jl))
            if (lq0) cJ(1:NG0(ik), jl) = cbuf(1:NG0(ik))
          end do
          ikirq = numirr(ikq); iopq = numrot(ikq)
          call make_C0_map(NTG, trs(ikq), NG0(ikq), KG0(1, 1, ikq), RW(1, ikq), rginv(1, 1, iopq), pg(1, iopq), &
               nnp, L1, L2, L3, packing(-L1, -L2, -L3, ikirq), jmap, pmap)
          do ig = 1, NG0(ikq)
            gpos(ig) = gidx(KG0(1, ig, ikq) - shG(1, ik), KG0(2, ig, ikq) - shG(2, ik), KG0(3, ig, ikq) - shG(3, ik))
          end do
          do il = 1, nti
            ib = ti(il)
            call c0_apply(NG0(ikq), trs(ikq), jmap, pmap, VIR(1, ib, ikirq), cbuf)
            ubuf(:, bj + il) = (0.0, 0.0)
            do ig = 1, NG0(ikq)
              ubuf(gpos(ig), bj + il) = cmplx(cbuf(ig), kind=4)
            end do
            call fftwf_execute_dft(plan_bw4, ubuf(:, bj + il), ubuf(:, bj + il))
            if (lq0) cI(1:NG0(ikq), il) = cbuf(1:NG0(ikq))
          end do
          do lp = 1, np
            jl = pjl(lp); il = pil(lp)
            row = ik + NTK*(lp - 1)
            ubuf(1:ngrid, bj + bi + 1) = conjg(ubuf(1:ngrid, bj + il))*ubuf(1:ngrid, jl)
            call fftwf_execute_dft(plan_fw4, ubuf(:, bj + bi + 1), ubuf(:, bj + bi + 1))
            do g = 1, NGeps
              A(row, g) = cmplx(ubuf(ext(g), bj + bi + 1), kind=8)*scl
            end do
            if (lq0) then
              ! calc_VMab (ikq=ik at q=0) + non-local correction + energy denominator, as in chiqw.F
              jb = tj(jl); ib = ti(il)
              sumv = zzero
              do ig = 1, NG0(ik)
                gv(:) = dble(KG0(1, ig, ik))*b1(:) + dble(KG0(2, ig, ik))*b2(:) + dble(KG0(3, ig, ik))*b3(:)
                sumv(:) = sumv(:) + gv(:)*conjg(cI(ig, il))*cJ(ig, jl)
              end do
              vm(:) = sumv(:)*zimag
              del = E_VIR(ib, numirr(ikq)) - E_OCC(jb, numirr(ik))
              if (abs(del) < delt) del = delt
              nlvm(:) = MAT_NL(:, ib + off_v, jb + off_o, numirr(ik))
              vm(:) = -zimag*(vm(:) - nlvm(:))/del
              do ix = 1, 3
                V(row, ix) = vm(ix)
                if (abs(del) <= delt) V(row, ix) = zzero
              end do
            end if
          end do
        end do
        !$omp end do

        deallocate(cbuf, pmap, jmap, gpos)
        if (lq0) deallocate(cJ, cI)
        call fftwf_free(pbuf)
      end block
      !$omp end parallel
    end subroutine phase_a_sp

    !------------------------------------------------------------------------------------------
    ! GPU phase A: wave-function rotation on the CPU, grid/FFT/pair-density work on the GPU
    subroutine phase_a_gpu()
      complex(8), allocatable :: cJ(:, :), cI(:, :)
      integer(c_int), allocatable :: idxJ(:), idxI(:), pj0(:), pi0(:)
      integer :: ik, ikq, ikir, iop, ikirq, iopq, jl, il, ig, lp
      allocate(cJ(NTG, ntj), cI(NTG, nti), idxJ(NTG), idxI(NTG), pj0(np), pi0(np))
      pj0 = pjl(1:np) - 1
      pi0 = pil(1:np) - 1
      rcg = chi_gpu_tile_begin(rows, ntj + nti, np, pj0, pi0)
      call gpu_check(rcg, 'chi_gpu_tile_begin')
      do ik = 1, NTK
        ikq = kpq(ik)
        ikir = numirr(ik); iop = numrot(ik)
        ikirq = numirr(ikq); iopq = numrot(ikq)
        !$omp parallel do schedule(dynamic,1)
        do jl = 1, ntj
          call make_C0(NTG, trs(ik), NG0(ik), KG0(1, 1, ik), RW(1, ik), rginv(1, 1, iop), pg(1, iop), &
               nnp, L1, L2, L3, packing(-L1, -L2, -L3, ikir), OCC(1, tj(jl), ikir), cJ(1, jl))
        end do
        !$omp end parallel do
        !$omp parallel do schedule(dynamic,1)
        do il = 1, nti
          call make_C0(NTG, trs(ikq), NG0(ikq), KG0(1, 1, ikq), RW(1, ikq), rginv(1, 1, iopq), pg(1, iopq), &
               nnp, L1, L2, L3, packing(-L1, -L2, -L3, ikirq), VIR(1, ti(il), ikirq), cI(1, il))
        end do
        !$omp end parallel do
        do ig = 1, NG0(ik)
          idxJ(ig) = gidx(KG0(1, ig, ik), KG0(2, ig, ik), KG0(3, ig, ik)) - 1
        end do
        do ig = 1, NG0(ikq)
          idxI(ig) = gidx(KG0(1, ig, ikq) - shG(1, ik), KG0(2, ig, ikq) - shG(2, ik), KG0(3, ig, ikq) - shG(3, ik)) - 1
        end do
        rcg = chi_gpu_bands(ntj, NTG, cJ, NG0(ik), idxJ, 0)
        call gpu_check(rcg, 'chi_gpu_bands (occ)')
        rcg = chi_gpu_bands(nti, NTG, cI, NG0(ikq), idxI, ntj)
        call gpu_check(rcg, 'chi_gpu_bands (vir)')
        if (lq0) then
          ! k.p G=0 elements, exactly as in the CPU phase A
          !$omp parallel do schedule(dynamic,16)
          do lp = 1, np
            block
              integer :: jb, ib, ix, row, igg
              complex(8) :: sumv(3), vm(3), nlvm(3)
              real(8) :: gv(3), del
              jb = tj(pjl(lp)); ib = ti(pil(lp))
              row = ik + NTK*(lp - 1)
              sumv = zzero
              do igg = 1, NG0(ik)
                gv(:) = dble(KG0(1, igg, ik))*b1(:) + dble(KG0(2, igg, ik))*b2(:) + dble(KG0(3, igg, ik))*b3(:)
                sumv(:) = sumv(:) + gv(:)*conjg(cI(igg, pil(lp)))*cJ(igg, pjl(lp))
              end do
              vm(:) = sumv(:)*zimag
              del = E_VIR(ib, numirr(ikq)) - E_OCC(jb, numirr(ik))
              if (abs(del) < delt) del = delt
              nlvm(:) = MAT_NL(:, ib + off_v, jb + off_o, numirr(ik))
              vm(:) = -zimag*(vm(:) - nlvm(:))/del
              do ix = 1, 3
                V(row, ix) = vm(ix)
                if (abs(del) <= delt) V(row, ix) = zzero
              end do
            end block
          end do
          !$omp end parallel do
        end if
        rcg = chi_gpu_pairs(ik - 1, NTK, 0, ntj, scl)
        call gpu_check(rcg, 'chi_gpu_pairs')
      end do
      deallocate(cJ, cI, idxJ, idxI, pj0, pi0)
    end subroutine phase_a_gpu

    !------------------------------------------------------------------------------------------
    ! GPU phase C: single-precision CGEMM blocks into the double-precision chi on the device
    subroutine phase_c_gpu()
      integer :: kc, ie, ix
      kc = 2048                ! rows per single-precision CGEMM block (accuracy: SrVO3 eps^-1 4.7e-7 vs 6.9e-6 at 8192)
      if (lq0) then
        rcg = chi_gpu_gemm(W, V, kc)
        call gpu_check(rcg, 'chi_gpu_gemm')
        do ie = 1, nen
          do ix = 1, 3
            Dacc(ie, ix) = Dacc(ie, ix) + sum(conjg(V(:, ix))*W(:, ie)*V(:, ix))
          end do
        end do
      else
        rcg = chi_gpu_gemm(W, W, kc)
        call gpu_check(rcg, 'chi_gpu_gemm')
      end if
    end subroutine phase_c_gpu

    subroutine gpu_check(rc, where)
      integer(c_int), intent(in) :: rc
      character(*), intent(in) :: where
      if (rc /= 0) then
        write(6,*) 'chi_fast: GPU error in ', where, ', code ', rc
        stop
      end if
    end subroutine gpu_check

    !------------------------------------------------------------------------------------------
    ! phase B: tetrahedron weights x_pk(w) (same formulas, log ratios via vectorised log/atan2), times cRPA factor and 1/NTK -> W(ik+NTK*(p-1), w)
    subroutine phase_b()
      !$omp parallel default(shared)
      block
        complex(8), allocatable :: e1(:), e2(:), xow(:, :), ca1(:)
        complex(8) :: fac
        integer :: lp, ip, jb, ib, ik, ikq, row
        allocate(e1(NTK), e2(NTK), xow(nen, NTK), ca1(4*nen))
        !$omp do schedule(dynamic,1)
        do lp = 1, np
          ip = tpairs(tstart(it) + lp - 1)
          jb = jpair(ip); ib = ipair(ip)
          do ik = 1, NTK
            e1(k3d(ik)) = cmplx(E_OCC(jb, numirr(ik)), 0d0, 8)
            e2(k3d(ik)) = cmplx(E_VIR(ib, numirr(kpq(ik))), 0d0, 8)
          end do
          ! both signs (dsgn=+1,-1) in one pass, vectorised log ratios (m_tetrahedron_fast)
          xow = zzero
          ca1 = zzero
          call ttrhdrn_chi0_pm(dmna, dmnr, nkb1, nkb2, nkb3, imt1, e1, e2, Ef, Ef, delt, nen, em, ca1, xow)
          do ik = 1, NTK
            ikq = kpq(ik)
            row = ik + NTK*(lp - 1)
            fac = (1d0 - P_OCC(jb, ik)*P_VIR(ib, ikq))/dble(NTK)
            W(row, :) = xow(:, k3d(ik))*fac
          end do
        end do
        !$omp end do
        deallocate(e1, e2, xow, ca1)
      end block
      !$omp end parallel
    end subroutine phase_b

    !------------------------------------------------------------------------------------------
    ! phase C: chi(:,:,w) += A^H diag(W(:,w)) A, parallel over w (each thread owns whole w slices)
    subroutine phase_c()
      integer :: chunk
      chunk = max(1, min(np, 4096/NTK))          ! pairs per GEMM block (K = NTK*chunk rows)
      if (gemm_sp) then
        allocate(A4(rows, NGeps))
        A4 = cmplx(A, kind=4)
      end if
      !$omp parallel default(shared)
      block
        complex(8), allocatable :: Bc(:, :), T(:, :), wv(:), wcv(:), y(:), z(:)
        complex(4), allocatable :: Bc4(:, :), T4(:, :)
        complex(8) :: d
        integer :: ie, p0, p1, r0, nr, g, ix, G0
        if (gemm_sp) then
          allocate(Bc4(NTK*chunk, NGeps), T4(NGeps, NGeps))
        else
          allocate(Bc(NTK*chunk, NGeps))
        end if
        if (lq0) allocate(T(NGeps, NGeps), wv(rows), wcv(rows), y(NGeps), z(NGeps))
        G0 = No_G_0
        !$omp do schedule(dynamic,1)
        do ie = 1, nen
          if (lq0) T = zzero
          do p0 = 1, np, chunk
            p1 = min(np, p0 + chunk - 1)
            r0 = NTK*(p0 - 1) + 1
            nr = NTK*(p1 - p0 + 1)
            if (gemm_sp) then
              ! single-precision block product, accumulated in double precision
              do g = 1, NGeps
                Bc4(1:nr, g) = A4(r0:r0 + nr - 1, g)*cmplx(W(r0:r0 + nr - 1, ie), kind=4)
              end do
              call cgemm('C', 'N', NGeps, NGeps, nr, (1.0, 0.0), A4(r0, 1), rows, Bc4, NTK*chunk, (0.0, 0.0), &
                   T4, NGeps)
              if (lq0) then
                T = T + cmplx(T4, kind=8)
              else
                pchi(:, :, ie) = pchi(:, :, ie) + cmplx(T4, kind=8)
              end if
            else
              do g = 1, NGeps
                Bc(1:nr, g) = A(r0:r0 + nr - 1, g)*W(r0:r0 + nr - 1, ie)
              end do
              if (lq0) then
                call zgemm('C', 'N', NGeps, NGeps, nr, zone, A(r0, 1), rows, Bc, NTK*chunk, zone, T, NGeps)
              else
                call zgemm('C', 'N', NGeps, NGeps, nr, zone, A(r0, 1), rows, Bc, NTK*chunk, zone, &
                     pchi(:, :, ie), NGeps)
              end if
            end if
          end do
          if (lq0) then
            do ix = 1, 3
              wv(:) = W(:, ie)*V(:, ix)
              wcv(:) = W(:, ie)*conjg(V(:, ix))
              call zgemv('C', rows, NGeps, zone, A, rows, wv, 1, zzero, y, 1)    ! sum conj(A) w v
              call zgemv('T', rows, NGeps, zone, A, rows, wcv, 1, zzero, z, 1)   ! sum conj(v) w A
              d = sum(conjg(V(:, ix))*W(:, ie)*V(:, ix))
              pchigm(:, :, ie, ix) = pchigm(:, :, ie, ix) + T
              pchigm(:, G0, ie, ix) = pchigm(:, G0, ie, ix) + y - T(:, G0)
              pchigm(G0, :, ie, ix) = pchigm(G0, :, ie, ix) + z - T(G0, :)
              ! (G0,G0) now holds old + y0 + z0 - T00; it must be old + d
              pchigm(G0, G0, ie, ix) = pchigm(G0, G0, ie, ix) + d - y(G0) - z(G0) + T(G0, G0)
            end do
          end if
        end do
        !$omp end do
        if (gemm_sp) then
          deallocate(Bc4, T4)
        else
          deallocate(Bc)
        end if
        if (lq0) deallocate(T, wv, wcv, y, z)
      end block
      !$omp end parallel
      if (gemm_sp) deallocate(A4)
    end subroutine phase_c

  end subroutine chi_fast_pairs

  ! one band of make_C0 from the per-k map of make_C0_map: dst = src(jmap)*pmap, conjugated for itrs = -1
  subroutine c0_apply(n, itrs, jmap, pmap, src, dst)
    integer, intent(in) :: n, itrs, jmap(n)
    complex(8), intent(in) :: pmap(n), src(*)
    complex(8), intent(inout) :: dst(*)
    integer :: ig
    if (itrs == 1) then
      do ig = 1, n
        dst(ig) = src(jmap(ig))*pmap(ig)
      end do
    else if (itrs == -1) then
      do ig = 1, n
        dst(ig) = conjg(src(jmap(ig))*pmap(ig))
      end do
    else
      dst(1:n) = (0.0d0, 0.0d0)
    end if
  end subroutine c0_apply

end module m_chi_fast
