! Compare/benchmark original ttrhdrn_chi0 (m_tetrahedron) vs m_tetrahedron_fast
! (ttrhdrn_chi0 with vectorised log ratios, and the fused +/- ttrhdrn_chi0_pm).
program bench_tetra
  use m_tetrahedron, only: ttrhdrn_mkidx, chi0_orig => ttrhdrn_chi0
  use m_tetrahedron_fast, only: chi0_fast => ttrhdrn_chi0, chi0_pm => ttrhdrn_chi0_pm
  implicit none
  integer, parameter :: n = 6, nk = n*n*n, ne = 70, npair = 5, nrep = 10
  real(8), parameter :: au = 27.21151d0, pi = acos(-1d0)
  real(8) :: delt, dmna, dmnr, ef, b(3,3), kx, ky, kz, c, t0, t1, t2, t3, tf, tp
  complex(8) :: em(ne), e1(nk), e2(nk), ca1(4*ne)
  complex(8), allocatable :: xo(:,:), xf(:,:), xp(:,:)
  integer, allocatable :: imt1(:,:)
  integer :: i1, i2, i3, ik, ip, ir
  real(8) :: worst_f, worst_p
  allocate(imt1(4, nk*6), xo(ne,nk), xf(ne,nk), xp(ne,nk))
  b = 0d0; b(1,1) = 1d0; b(2,2) = 1d0; b(3,3) = 1d0
  call ttrhdrn_mkidx(n, n, n, imt1, b(:,1), b(:,2), b(:,3))
  call gen_grid(ne, 1.0d0, em)
  delt = 0.1d0/au; dmna = 0.001d0; dmnr = 0.001d0; ef = 0d0
  worst_f = 0d0; worst_p = 0d0; tf = 0d0; tp = 0d0; t0 = 0d0
  do ip = 1, npair
    do i3 = 1, n; do i2 = 1, n; do i1 = 1, n
      ik = i1 + (i2-1)*n + (i3-1)*n*n
      kx = 2*pi*(i1-1)/n; ky = 2*pi*(i2-1)/n; kz = 2*pi*(i3-1)/n
      c = cos(kx) + cos(ky) + cos(kz)
      select case (ip)
      case (1)   ! metallic -> metallic (both cross Ef)
        e1(ik) = 0.05d0*c; e2(ik) = 0.06d0*(cos(kx)+cos(ky)) + 0.02d0*cos(kz)
      case (2)   ! insulating
        e1(ik) = -0.3d0 + 0.02d0*c; e2(ik) = 0.25d0 + 0.04d0*c
      case (3)   ! flat bands: degenerate tetrahedra
        e1(ik) = -0.2d0; e2(ik) = 0.1d0
      case (4)   ! dispersion along one direction only: partially degenerate
        e1(ik) = -0.1d0 + 0.03d0*cos(kx); e2(ik) = 0.2d0 + 0.03d0*cos(kx)
      case (5)   ! occupied band crossing Ef, empty band above
        e1(ik) = 0.08d0*c - 0.02d0; e2(ik) = 0.5d0 + 0.1d0*c
      end select
    end do; end do; end do
    call cpu_time(t0)
    do ir = 1, nrep
      xo = 0; ca1 = 0
      call chi0_orig(dmna, dmnr, n, n, n, imt1, e1, e2, ef, ef, delt, 1d0, ne, em, ca1, xo)
      ca1 = 0
      call chi0_orig(dmna, dmnr, n, n, n, imt1, e1, e2, ef, ef, delt, -1d0, ne, em, ca1, xo)
    end do
    call cpu_time(t1)
    do ir = 1, nrep
      xf = 0; ca1 = 0
      call chi0_fast(dmna, dmnr, n, n, n, imt1, e1, e2, ef, ef, delt, 1d0, ne, em, ca1, xf)
      ca1 = 0
      call chi0_fast(dmna, dmnr, n, n, n, imt1, e1, e2, ef, ef, delt, -1d0, ne, em, ca1, xf)
    end do
    call cpu_time(t2)
    do ir = 1, nrep
      xp = 0; ca1 = 0
      call chi0_pm(dmna, dmnr, n, n, n, imt1, e1, e2, ef, ef, delt, ne, em, ca1, xp)
    end do
    call cpu_time(t3)
    write(6,'(a,i2,a,3f9.2,a,2es10.2)') ' pair', ip, '  ms/pair orig fast pm:', &
         (t1-t0)/nrep*1d3, (t2-t1)/nrep*1d3, (t3-t2)/nrep*1d3, &
         '  maxdev/max fast,pm:', maxval(abs(xf-xo))/maxval(abs(xo)), maxval(abs(xp-xo))/maxval(abs(xo))
    worst_f = max(worst_f, maxval(abs(xf-xo))/maxval(abs(xo)))
    worst_p = max(worst_p, maxval(abs(xp-xo))/maxval(abs(xo)))
    tf = tf + (t1-t0); tp = tp + (t3-t2)
  end do
  write(6,'(a,2es10.2,a,f6.2)') ' worst rel. deviation fast, pm:', worst_f, worst_p, '   speed-up pm vs orig:', tf/tp
end program
