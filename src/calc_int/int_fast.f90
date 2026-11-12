!
! m_int_fast -- faster pair densities and W/J contractions for calc_w3d / calc_j3d
! (RESPACK fork, branch zen-opt)
!
! Same quantities as the original loops in calc_intW.F / calc_intJ.F; only the organisation differs:
!  * The Wannier-gauge Bloch functions u_{w,k}(r) = FFT^-1[C0(:,w,k)] are transformed to real space
!    once (FFTW) and cached, instead of two backward FFTs per (q,k,pair) call of
!    calc_InterStateMatrix.  Per (q,k,pair) only the product and one forward FFT remain.
!  * The umklapp phase exp(-i2pi dG.r/N) of u_{k+q} is applied with the same separable 1D tables
!    (phx*phy*phz) as calc_InterStateMatrix in calc_int_sub.F.
!  * Pair densities are summed over k with weights kph(k,R) for all requested lattice vectors R in
!    one pass (calc_j3d: the original recomputed all FFTs for every R).
!  * rho^T eps^-1 conj(rho) contractions as ZGEMMs (eps promoted exactly from complex(4)).
! Conventions: backward FFT unnormalised exp(+i2pi G.r/N), forward exp(-i2pi G.r/N) with 1/N.
!
module m_int_fast
  use, intrinsic :: iso_c_binding
  use omp_lib
  implicit none
  include 'fftw3.f03'
  private
  public :: int_fast_init, int_fast_pairdens, int_fast_funcw_mat, int_fast_funcw_diag

  logical, save, public :: int_fast_ok = .false.
  integer, save :: n1 = 0, n2 = 0, n3 = 0, ngrid = 0, ldg = 0, nwf_c = 0, ntk_c = 0
  type(c_ptr), save :: plan_bw = c_null_ptr, plan_fw = c_null_ptr, pcache = c_null_ptr
  complex(c_double_complex), pointer, contiguous, save :: uc(:, :, :) => null()   ! (ldg, NWF, NTK)

  complex(8), parameter :: zzero = (0d0, 0d0), zone = (1d0, 0d0), zimag = (0d0, 1d0)
  real(8), parameter :: tpi = 2d0*3.14159265358979323846d0

contains

  pure integer function gidx(i, j, k)
    integer, intent(in) :: i, j, k
    gidx = modulo(i, n1) + 1 + modulo(j, n2)*n1 + modulo(k, n3)*n1*n2
  end function gidx

  !---------------------------------------------------------------------------------------------
  subroutine int_fast_init(nx, ny, nz, NTK, NTG, NWF, NG0, KG0, C0, mem_mb)
    integer, intent(in) :: nx, ny, nz, NTK, NTG, NWF
    integer, intent(in) :: NG0(NTK), KG0(3, NTG, NTK)
    complex(8), intent(in) :: C0(NTG, NWF, NTK)
    real(8), intent(in) :: mem_mb
    real(8) :: need_mb
    type(c_ptr) :: p
    complex(c_double_complex), pointer :: buf(:)
    integer :: ik, iw, ig

    n1 = nx; n2 = ny; n3 = nz; ngrid = nx*ny*nz
    ldg = ((ngrid + 7)/8)*8
    nwf_c = NWF; ntk_c = NTK
    need_mb = 16d0*dble(ldg)*dble(NWF)*dble(NTK)/1048576d0
    if (need_mb > mem_mb) then
      write(6,'(a,f12.1,a,f12.1,a)') ' int_fast: real-space cache would need ', need_mb, &
           ' MB > fast_mem_mb =', mem_mb, ' MB -> original code path'
      int_fast_ok = .false.
      return
    end if
    p = fftw_alloc_complex(int(ldg, c_size_t))
    call c_f_pointer(p, buf, [ldg])
    plan_bw = fftw_plan_dft_3d(n3, n2, n1, buf, buf, FFTW_BACKWARD, FFTW_MEASURE)
    plan_fw = fftw_plan_dft_3d(n3, n2, n1, buf, buf, FFTW_FORWARD, FFTW_MEASURE)
    call fftw_free(p)
    pcache = fftw_alloc_complex(int(ldg, c_size_t)*int(NWF, c_size_t)*int(NTK, c_size_t))
    call c_f_pointer(pcache, uc, [ldg, NWF, NTK])
    !$omp parallel do collapse(2) schedule(dynamic,1) private(ig)
    do ik = 1, NTK
      do iw = 1, NWF
        uc(:, iw, ik) = zzero
        do ig = 1, NG0(ik)
          uc(gidx(KG0(1, ig, ik), KG0(2, ig, ik), KG0(3, ig, ik)), iw, ik) = C0(ig, iw, ik)
        end do
        call fftw_execute_dft(plan_bw, uc(:, iw, ik), uc(:, iw, ik))
      end do
    end do
    !$omp end parallel do
    int_fast_ok = .true.
    write(6,'(a,3i5,a,f10.1,a)') ' int_fast: FFTW grid ', n1, n2, n3, ' ; real-space cache ', need_mb, ' MB'
  end subroutine int_fast_init

  !---------------------------------------------------------------------------------------------
  ! S(G,p,iR) = sum_k kph(k,iR) * M_p(k,G)   (G = 1..NGpsi, list LG)
  ! M_p(k,G) = rho_tmp of calc_InterStateMatrix with C0_K = C0(:,jwp(p),k), C0_KQ = C0(:,iwp(p),k+q)
  subroutine int_fast_pairdens(NTK, SK0, q, lq0, NGpsi, LG, npair, iwp, jwp, nR, kph, S)
    integer, intent(in) :: NTK, NGpsi, npair, nR
    real(8), intent(in) :: SK0(3, NTK), q(3)
    logical, intent(in) :: lq0
    integer, intent(in) :: LG(3, *), iwp(npair), jwp(npair)
    complex(8), intent(in) :: kph(NTK, nR)
    complex(8), intent(out) :: S(NGpsi, npair, nR)
    external :: search_kq
    integer, allocatable :: kpq(:), shG(:, :), ext(:)
    complex(8), allocatable :: phx(:, :), phy(:, :), phz(:, :)
    integer :: ik, g, nthr, ir1, ir2, ir3
    logical :: by_pair
    real(8) :: scl, phase

    if (.not. int_fast_ok) stop 'int_fast_pairdens: not initialised'
    scl = 1d0/dble(ngrid)
    nthr = omp_get_max_threads()
    allocate(kpq(NTK), shG(3, NTK), ext(NGpsi))
    allocate(phx(n1, NTK), phy(n2, NTK), phz(n3, NTK))
    do ik = 1, NTK
      shG(:, ik) = 0
      if (lq0) then
        kpq(ik) = ik
      else
        call search_kq(NTK, SK0, q(1), q(2), q(3), ik, kpq(ik), shG(1, ik))
      end if
      ! same 1D phase tables as calc_InterStateMatrix (calc_int_sub.F)
      do ir1 = 1, n1
        phase = tpi*(dble(ir1 - 1)*dble(shG(1, ik))/dble(n1))
        phx(ir1, ik) = exp(-zimag*phase)
      end do
      do ir2 = 1, n2
        phase = tpi*(dble(ir2 - 1)*dble(shG(2, ik))/dble(n2))
        phy(ir2, ik) = exp(-zimag*phase)
      end do
      do ir3 = 1, n3
        phase = tpi*(dble(ir3 - 1)*dble(shG(3, ik))/dble(n3))
        phz(ir3, ik) = exp(-zimag*phase)
      end do
    end do
    do g = 1, NGpsi
      ext(g) = gidx(-LG(1, g), -LG(2, g), -LG(3, g))
    end do
    S = zzero
    by_pair = (npair >= 2*nthr)

    if (by_pair) then
      ! each thread owns whole pairs -> no reduction needed
      !$omp parallel default(shared)
      block
        type(c_ptr) :: pf
        complex(c_double_complex), pointer, contiguous :: f(:)
        complex(8), allocatable :: m(:)
        integer :: p, ik2, iR
        pf = fftw_alloc_complex(int(ldg, c_size_t))
        call c_f_pointer(pf, f, [ldg])
        allocate(m(NGpsi))
        !$omp do schedule(dynamic,1)
        do p = 1, npair
          do ik2 = 1, NTK
            call one_density(ik2, p, f, m)
            do iR = 1, nR
              S(:, p, iR) = S(:, p, iR) + kph(ik2, iR)*m(:)
            end do
          end do
        end do
        !$omp end do
        deallocate(m)
        call fftw_free(pf)
      end block
      !$omp end parallel
    else
      ! few pairs: parallel over k with private accumulators
      !$omp parallel default(shared)
      block
        type(c_ptr) :: pf
        complex(c_double_complex), pointer, contiguous :: f(:)
        complex(8), allocatable :: m(:), Sl(:, :, :)
        integer :: p, ik2, iR
        pf = fftw_alloc_complex(int(ldg, c_size_t))
        call c_f_pointer(pf, f, [ldg])
        allocate(m(NGpsi), Sl(NGpsi, npair, nR))
        Sl = zzero
        !$omp do schedule(dynamic,1)
        do ik2 = 1, NTK
          do p = 1, npair
            call one_density(ik2, p, f, m)
            do iR = 1, nR
              Sl(:, p, iR) = Sl(:, p, iR) + kph(ik2, iR)*m(:)
            end do
          end do
        end do
        !$omp end do
        !$omp critical
        S = S + Sl
        !$omp end critical
        deallocate(m, Sl)
        call fftw_free(pf)
      end block
      !$omp end parallel
    end if

  contains

    subroutine one_density(ik, p, f, m)
      integer, intent(in) :: ik, p
      complex(c_double_complex), intent(inout) :: f(ldg)
      complex(8), intent(out) :: m(NGpsi)
      integer :: ikq, iw, jw, i1, i2, i3, ir, g
      complex(8) :: pf23
      ikq = kpq(ik); iw = iwp(p); jw = jwp(p)
      if (all(shG(:, ik) == 0)) then
        f(1:ngrid) = conjg(uc(1:ngrid, iw, ikq))*uc(1:ngrid, jw, ik)
      else
        ir = 0
        do i3 = 1, n3
          do i2 = 1, n2
            do i1 = 1, n1
              ir = ir + 1
              ! as calc_InterStateMatrix: pf = phx*phy*phz ; u_kq*pf ; conjg(u_kq*pf)*u_k
              pf23 = phx(i1, ik)*phy(i2, ik)*phz(i3, ik)
              f(ir) = conjg(uc(ir, iw, ikq)*pf23)*uc(ir, jw, ik)
            end do
          end do
        end do
      end if
      call fftw_execute_dft(plan_fw, f, f)
      do g = 1, NGpsi
        m(g) = f(ext(g))*scl
      end do
    end subroutine one_density

  end subroutine int_fast_pairdens

  !---------------------------------------------------------------------------------------------
  ! calc_w3d: funcw(iw,jw,ie) = sum_{G,G'<=NGe} rho(G,iw) eps(G,G',ie) conj(rho(G',jw))
  !                           + sum_{NGe<G<=NGpsi} rho(G,iw) conj(rho(G,jw))
  subroutine int_fast_funcw_mat(NGe, NGpsi, NWF, rho, ldr, ne, eps, lde, funcw)
    integer, intent(in) :: NGe, NGpsi, NWF, ldr, ne, lde
    complex(8), intent(in) :: rho(ldr, NWF)
    complex(4), intent(in) :: eps(lde, lde, ne)
    complex(8), intent(out) :: funcw(NWF, NWF, ne)
    complex(8), allocatable :: crho(:, :), tail(:, :)
    integer :: nt
    allocate(crho(NGpsi, NWF), tail(NWF, NWF))
    crho = conjg(rho(1:NGpsi, :))
    tail = zzero
    nt = NGpsi - NGe
    if (nt > 0) call zgemm('T', 'N', NWF, NWF, nt, zone, rho(NGe + 1, 1), ldr, crho(NGe + 1, 1), NGpsi, &
         zzero, tail, NWF)
    !$omp parallel default(shared)
    block
      complex(8), allocatable :: E8(:, :), T(:, :)
      integer :: ie
      allocate(E8(NGe, NGe), T(NGe, NWF))
      !$omp do schedule(dynamic,1)
      do ie = 1, ne
        E8 = cmplx(eps(1:NGe, 1:NGe, ie), kind=8)
        call zgemm('N', 'N', NGe, NWF, NGe, zone, E8, NGe, crho, NGpsi, zzero, T, NGe)
        call zgemm('T', 'N', NWF, NWF, NGe, zone, rho, ldr, T, NGe, zzero, funcw(1, 1, ie), NWF)
        funcw(:, :, ie) = funcw(:, :, ie) + tail
      end do
      !$omp end do
      deallocate(E8, T)
    end block
    !$omp end parallel
    deallocate(crho, tail)
  end subroutine int_fast_funcw_mat

  !---------------------------------------------------------------------------------------------
  ! calc_j3d: funcw(p,ie) = sum_{G,G'<=NGe} rho(G,p) eps(G,G',ie) conj(rho(G',p))
  !                       + sum_{NGe<G<=NGpsi} rho(G,p) conj(rho(G,p))
  subroutine int_fast_funcw_diag(NGe, NGpsi, np, rho, ldr, ne, eps, lde, funcw)
    integer, intent(in) :: NGe, NGpsi, np, ldr, ne, lde
    complex(8), intent(in) :: rho(ldr, np)
    complex(4), intent(in) :: eps(lde, lde, ne)
    complex(8), intent(out) :: funcw(np, ne)
    complex(8), allocatable :: crho(:, :), tail(:)
    integer :: p
    allocate(crho(NGe, np), tail(np))
    crho = conjg(rho(1:NGe, :))
    do p = 1, np
      tail(p) = sum(rho(NGe + 1:NGpsi, p)*conjg(rho(NGe + 1:NGpsi, p)))
    end do
    !$omp parallel default(shared)
    block
      complex(8), allocatable :: E8(:, :), T(:, :)
      integer :: ie, pp
      allocate(E8(NGe, NGe), T(NGe, np))
      !$omp do schedule(dynamic,1)
      do ie = 1, ne
        E8 = cmplx(eps(1:NGe, 1:NGe, ie), kind=8)
        call zgemm('N', 'N', NGe, np, NGe, zone, E8, NGe, crho, NGe, zzero, T, NGe)
        do pp = 1, np
          funcw(pp, ie) = sum(rho(1:NGe, pp)*T(:, pp)) + tail(pp)
        end do
      end do
      !$omp end do
      deallocate(E8, T)
    end block
    !$omp end parallel
    deallocate(crho, tail)
  end subroutine int_fast_funcw_diag

end module m_int_fast
