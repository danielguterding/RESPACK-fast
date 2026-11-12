! accuracy vs quad-precision reference of the original formulas
program acc
  use m_tetrahedron, only: ttrhdrn_mkidx, chi0_orig => ttrhdrn_chi0
  use m_tetrahedron_fast, only: chi0_pm => ttrhdrn_chi0_pm
  use m_tetrahedron_q, only: chi0_q => ttrhdrn_chi0
  implicit none
  integer, parameter :: n = 6, nk = n*n*n, ne = 70, qp = 16
  real(8), parameter :: au = 27.21151d0, pi = acos(-1d0)
  real(8) :: delt, ef, b(3,3), kx, ky, kz, c
  complex(8) :: em(ne), e1(nk), e2(nk), ca1(4*ne)
  complex(8), allocatable :: xo(:,:), xp(:,:)
  complex(qp), allocatable :: xq(:,:), caq(:), emq(:), e1q(:), e2q(:)
  integer, allocatable :: imt1(:,:)
  integer :: i1, i2, i3, ik, ip
  real(8) :: eo, epm, scale
  allocate(imt1(4, nk*6), xo(ne,nk), xp(ne,nk), xq(ne,nk), caq(4*ne), emq(ne), e1q(nk), e2q(nk))
  b = 0d0; b(1,1) = 1d0; b(2,2) = 1d0; b(3,3) = 1d0
  call ttrhdrn_mkidx(n, n, n, imt1, b(:,1), b(:,2), b(:,3))
  call gen_grid(ne, 1.0d0, em)
  delt = 0.1d0/au; ef = 0d0
  do ip = 1, 5
    do i3 = 1, n; do i2 = 1, n; do i1 = 1, n
      ik = i1 + (i2-1)*n + (i3-1)*n*n
      kx = 2*pi*(i1-1)/n; ky = 2*pi*(i2-1)/n; kz = 2*pi*(i3-1)/n
      c = cos(kx) + cos(ky) + cos(kz)
      select case (ip)
      case (1); e1(ik) = 0.05d0*c; e2(ik) = 0.06d0*(cos(kx)+cos(ky)) + 0.02d0*cos(kz)
      case (2); e1(ik) = -0.3d0 + 0.02d0*c; e2(ik) = 0.25d0 + 0.04d0*c
      case (3); e1(ik) = -0.2d0; e2(ik) = 0.1d0
      case (4); e1(ik) = -0.1d0 + 0.03d0*cos(kx); e2(ik) = 0.2d0 + 0.03d0*cos(kx)
      case (5); e1(ik) = 0.08d0*c - 0.02d0; e2(ik) = 0.5d0 + 0.1d0*c
      end select
    end do; end do; end do
    ! identical double inputs for all three; the reference just evaluates the formulas in quad
    e1q = e1; e2q = e2; emq = em
    xq = 0; caq = 0
    call chi0_q(real(0.001d0,qp), real(0.001d0,qp), n, n, n, imt1, e1q, e2q, real(ef,qp), real(ef,qp), real(delt,qp), 1.0_qp, ne, emq, caq, xq)
    caq = 0
    call chi0_q(real(0.001d0,qp), real(0.001d0,qp), n, n, n, imt1, e1q, e2q, real(ef,qp), real(ef,qp), real(delt,qp), -1.0_qp, ne, emq, caq, xq)
    xo = 0; ca1 = 0
    call chi0_orig(0.001d0, 0.001d0, n, n, n, imt1, e1, e2, ef, ef, delt, 1d0, ne, em, ca1, xo)
    ca1 = 0
    call chi0_orig(0.001d0, 0.001d0, n, n, n, imt1, e1, e2, ef, ef, delt, -1d0, ne, em, ca1, xo)
    xp = 0; ca1 = 0
    call chi0_pm(0.001d0, 0.001d0, n, n, n, imt1, e1, e2, ef, ef, delt, ne, em, ca1, xp)
    scale = real(maxval(abs(xq)), 8)
    eo = real(maxval(abs(cmplx(xo, kind=qp) - xq)), 8)/scale
    epm = real(maxval(abs(cmplx(xp, kind=qp) - xq)), 8)/scale
    write(6,'(a,i2,a,es10.2,a,es10.2)') ' pair', ip, '  error vs quad: original', eo, '   fork', epm
  end do
end program
