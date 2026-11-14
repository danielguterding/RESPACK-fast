! This file is part of the zen-opt fork of RESPACK. Copyright (C) 2026 Daniel Guterding.
! Generalized tetrahedron method (T. Fujiwara, S. Yamamoto, Y. Ishii, J. Phys. Soc. Jpn. 72, 777 (2003);
! Y. Nohara, S. Yamamoto, T. Fujiwara, Phys. Rev. B 79, 195110 (2009)); MIT License as m_tetrahedron_fast.F.
!
! Normal-tetrahedron kernel of ttrhdrn_chi0_pm (m_tetrahedron_fast.F) in real arithmetic on separate real and
! imaginary arrays ("structure of arrays"), so that the loop over the frequencies is vectorised (AVX2: 4
! frequencies per instruction). Same formulas, the same order of operations within each complex product as
! calc0_fused; the complex version runs as scalar code, because the compiler does not vectorise the interleaved
! complex arithmetic of that loop.
module m_tetrahedron_soa
  implicit none
  private
  public :: lratio_soa, calc0_soa

contains

  ! lr + i li = log((ga + em)/(gb + em)), principal value, as lratio in m_tetrahedron_fast.F (all arguments in one
  ! open half plane); em = emr + i emi
  subroutine lratio_soa(ne, ga, gb, emr, emi, lr, li)
    integer, intent(in) :: ne
    complex(8), intent(in) :: ga, gb
    real(8), intent(in) :: emr(ne), emi(ne)
    real(8), intent(out) :: lr(ne), li(ne)
    real(8) :: gar, gai, gbr, gbi, ar, ai, br, bi
    integer :: i
    gar = dble(ga); gai = dimag(ga); gbr = dble(gb); gbi = dimag(gb)
    !$omp simd private(ar, ai, br, bi)
    do i = 1, ne
      ar = gar + emr(i)
      ai = gai + emi(i)
      br = gbr + emr(i)
      bi = gbi + emi(i)
      lr(i) = .5d0*log((ar*ar + ai*ai)/(br*br + bi*bi))
      li(i) = atan2(ai*br - ar*bi, ar*br + ai*bi)
    end do
  end subroutine lratio_soa

  ! calc0_fused of m_tetrahedron_fast.F for one tetrahedron with vertex energies g(1:4) (all differences large:
  ! case 0 of calc) and real vertex factors fr(1:4): xr + i xi (columns = vertices) += contributions of the four
  ! terms sub2normal(B,A,C,D), sub2normal(A,B,C,D), sub2normal(A,C,B,D), sub1normal(A,D,B,C) with A..D = g(ia4).
  ! w(ne, 12): workspace for the six log ratios (real parts in columns 1-6, imaginary parts in 7-12).
  subroutine calc0_soa(ne, emr, emi, g, fr, ia4, w, xr, xi)
    integer, intent(in) :: ne, ia4(4)
    real(8), intent(in) :: emr(ne), emi(ne), fr(4)
    complex(8), intent(in) :: g(4)
    real(8), intent(out) :: w(ne, 12)
    real(8), intent(inout) :: xr(ne, 4), xi(ne, 4)
    complex(8) :: a, b, c, d, dab, dac, dad, dbc, dbd, dcd
    complex(8) :: k11, k12, k13, k14, k21, k22, k23, k24, k31, k32, k33, k34, m1, m2, m3
    real(8) :: k11r, k11i, k12r, k12i, k13r, k13i, k14r, k14i, k21r, k21i, k22r, k22i, k23r, k23i, k24r, k24i
    real(8) :: k31r, k31i, k32r, k32i, k33r, k33i, k34r, k34i, m1r, m1i, m2r, m2i, m3r, m3i
    real(8) :: ar, ai, br, bi, cr, ci, dr, di, f1, f2, f3, f4
    real(8) :: er, ei, zar, zai, zbr, zbi, zcr, zci, zdr, zdi
    real(8) :: a2r, a2i, b2r, b2i, c2r, c2i, d2r, d2i, a3r, a3i, b3r, b3i, c3r, c3i, d3r, d3i
    real(8) :: labr, labi, lacr, laci, ladr, ladi, lbcr, lbci, lbdr, lbdi, lcdr, lcdi
    real(8) :: pr, pi, qr, qi, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, t1r, t1i, ur, ui, vr, vi
    integer :: i, j1, j2, j3, j4
    j1 = ia4(1); j2 = ia4(2); j3 = ia4(3); j4 = ia4(4)
    a = g(j1); b = g(j2); c = g(j3); d = g(j4)
    call lratio_soa(ne, a, b, emr, emi, w(:, 1), w(:, 7))
    call lratio_soa(ne, a, c, emr, emi, w(:, 2), w(:, 8))
    call lratio_soa(ne, a, d, emr, emi, w(:, 3), w(:, 9))
    call lratio_soa(ne, b, c, emr, emi, w(:, 4), w(:, 10))
    call lratio_soa(ne, b, d, emr, emi, w(:, 5), w(:, 11))
    call lratio_soa(ne, c, d, emr, emi, w(:, 6), w(:, 12))
    ! frequency-independent factors as in calc0_fused (squares as explicit products)
    dab = a - b; dac = a - c; dad = a - d; dbc = b - c; dbd = b - d; dcd = c - d
    k11 = 1.d0/(6.d0*(b - a)*(a - c)*(a - d))
    k12 = 1.d0/(6.d0*(b - c)*(dac*dac)*(c - d))
    k13 = 1.d0/(6.d0*(b - d)*(dad*dad)*(c - d))
    k14 = 1.d0/(6.d0*((b - a)*(b - a))*(b - c)*(b - d))
    k21 = 1.d0/(6.d0*(a - b)*(b - c)*(b - d))
    k22 = 1.d0/(6.d0*(a - c)*(dbc*dbc)*(c - d))
    k23 = 1.d0/(6.d0*(a - d)*(dbd*dbd)*(c - d))
    k24 = 1.d0/(6.d0*(dab*dab)*(a - c)*(a - d))
    k31 = 1.d0/(6.d0*(a - c)*(c - b)*(c - d))
    k32 = 1.d0/(6.d0*(a - b)*((c - b)*(c - b))*(b - d))
    k33 = 1.d0/(6.d0*(a - d)*(dcd*dcd)*(b - d))
    k34 = 1.d0/(6.d0*(dac*dac)*(a - b)*(a - d))
    m1 = 1.d0/((a - b)*(a - c)*(a - d))
    m2 = 1.d0/((a - b)*(d - b)*(b - c))
    m3 = 1.d0/((d - c)*(a - c)*(b - c))
    k11r = dble(k11); k11i = dimag(k11); k12r = dble(k12); k12i = dimag(k12)
    k13r = dble(k13); k13i = dimag(k13); k14r = dble(k14); k14i = dimag(k14)
    k21r = dble(k21); k21i = dimag(k21); k22r = dble(k22); k22i = dimag(k22)
    k23r = dble(k23); k23i = dimag(k23); k24r = dble(k24); k24i = dimag(k24)
    k31r = dble(k31); k31i = dimag(k31); k32r = dble(k32); k32i = dimag(k32)
    k33r = dble(k33); k33i = dimag(k33); k34r = dble(k34); k34i = dimag(k34)
    m1r = dble(m1); m1i = dimag(m1); m2r = dble(m2); m2i = dimag(m2); m3r = dble(m3); m3i = dimag(m3)
    ar = dble(a); ai = dimag(a); br = dble(b); bi = dimag(b)
    cr = dble(c); ci = dimag(c); dr = dble(d); di = dimag(d)
    f1 = fr(j1); f2 = fr(j2); f3 = fr(j3); f4 = fr(j4)
    !$omp simd private(er, ei, zar, zai, zbr, zbi, zcr, zci, zdr, zdi, a2r, a2i, b2r, b2i, c2r, c2i, d2r, d2i, &
    !$omp&             a3r, a3i, b3r, b3i, c3r, c3i, d3r, d3i, labr, labi, lacr, laci, ladr, ladi, lbcr, lbci, &
    !$omp&             lbdr, lbdi, lcdr, lcdi, pr, pi, qr, qi, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, t1r, t1i, &
    !$omp&             ur, ui, vr, vi)
    do i = 1, ne
      er = emr(i); ei = emi(i)
      zar = ar + er; zai = ai + ei
      zbr = br + er; zbi = bi + ei
      zcr = cr + er; zci = ci + ei
      zdr = dr + er; zdi = di + ei
      ! z^2 = z*z and z^3 = z^2*z as complex products
      a2r = zar*zar - zai*zai; a2i = zar*zai + zai*zar
      b2r = zbr*zbr - zbi*zbi; b2i = zbr*zbi + zbi*zbr
      c2r = zcr*zcr - zci*zci; c2i = zcr*zci + zci*zcr
      d2r = zdr*zdr - zdi*zdi; d2i = zdr*zdi + zdi*zdr
      a3r = a2r*zar - a2i*zai; a3i = a2r*zai + a2i*zar
      b3r = b2r*zbr - b2i*zbi; b3i = b2r*zbi + b2i*zbr
      c3r = c2r*zcr - c2i*zci; c3i = c2r*zci + c2i*zcr
      d3r = d2r*zdr - d2i*zdi; d3i = d2r*zdi + d2i*zdr
      labr = w(i, 1); labi = w(i, 7); lacr = w(i, 2); laci = w(i, 8); ladr = w(i, 3); ladi = w(i, 9)
      lbcr = w(i, 4); lbci = w(i, 10); lbdr = w(i, 5); lbdi = w(i, 11); lcdr = w(i, 6); lcdi = w(i, 12)
      ! term 1 (b,a,c,d): x1 = -za2*k11 + zc3*lac*k12 - zd3*lad*k13 - zb3*lab*k14
      x1r = -(a2r*k11r - a2i*k11i); x1i = -(a2r*k11i + a2i*k11r)
      qr = c3r*lacr - c3i*laci; qi = c3r*laci + c3i*lacr
      x1r = x1r + (qr*k12r - qi*k12i); x1i = x1i + (qr*k12i + qi*k12r)
      qr = d3r*ladr - d3i*ladi; qi = d3r*ladi + d3i*ladr
      x1r = x1r - (qr*k13r - qi*k13i); x1i = x1i - (qr*k13i + qi*k13r)
      qr = b3r*labr - b3i*labi; qi = b3r*labi + b3i*labr
      x1r = x1r - (qr*k14r - qi*k14i); x1i = x1i - (qr*k14i + qi*k14r)
      ! term 2 (a,b,c,d): x2 = -zb2*k21 + zc3*lbc*k22 - zd3*lbd*k23 + za3*lab*k24
      x2r = -(b2r*k21r - b2i*k21i); x2i = -(b2r*k21i + b2i*k21r)
      qr = c3r*lbcr - c3i*lbci; qi = c3r*lbci + c3i*lbcr
      x2r = x2r + (qr*k22r - qi*k22i); x2i = x2i + (qr*k22i + qi*k22r)
      qr = d3r*lbdr - d3i*lbdi; qi = d3r*lbdi + d3i*lbdr
      x2r = x2r - (qr*k23r - qi*k23i); x2i = x2i - (qr*k23i + qi*k23r)
      qr = a3r*labr - a3i*labi; qi = a3r*labi + a3i*labr
      x2r = x2r + (qr*k24r - qi*k24i); x2i = x2i + (qr*k24i + qi*k24r)
      ! term 3 (a,c,b,d): x3 = -zc2*k31 - zb3*lbc*k32 - zd3*lcd*k33 + za3*lac*k34
      x3r = -(c2r*k31r - c2i*k31i); x3i = -(c2r*k31i + c2i*k31r)
      qr = b3r*lbcr - b3i*lbci; qi = b3r*lbci + b3i*lbcr
      x3r = x3r - (qr*k32r - qi*k32i); x3i = x3i - (qr*k32i + qi*k32r)
      qr = d3r*lcdr - d3i*lcdi; qi = d3r*lcdi + d3i*lcdr
      x3r = x3r - (qr*k33r - qi*k33i); x3i = x3i - (qr*k33i + qi*k33r)
      qr = a3r*lacr - a3i*laci; qi = a3r*laci + a3i*lacr
      x3r = x3r + (qr*k34r - qi*k34i); x3i = x3i + (qr*k34i + qi*k34r)
      ! term 4 (a,d,b,c): t1 = -zd2*lad, x4 = (-t1*m1 - (t1 + zb2*lab)*m2 + (t1 + zc2*lac)*m3)*.5
      t1r = -(d2r*ladr - d2i*ladi); t1i = -(d2r*ladi + d2i*ladr)
      pr = t1r*m1r - t1i*m1i; pi = t1r*m1i + t1i*m1r
      x4r = -pr; x4i = -pi
      ur = t1r + (b2r*labr - b2i*labi); ui = t1i + (b2r*labi + b2i*labr)
      x4r = x4r - (ur*m2r - ui*m2i); x4i = x4i - (ur*m2i + ui*m2r)
      vr = t1r + (c2r*lacr - c2i*laci); vi = t1i + (c2r*laci + c2i*lacr)
      x4r = (x4r + (vr*m3r - vi*m3i))*.5d0; x4i = (x4i + (vr*m3i + vi*m3r))*.5d0
      xr(i, j1) = xr(i, j1) + f1*x1r; xi(i, j1) = xi(i, j1) + f1*x1i
      xr(i, j2) = xr(i, j2) + f2*x2r; xi(i, j2) = xi(i, j2) + f2*x2i
      xr(i, j3) = xr(i, j3) + f3*x3r; xi(i, j3) = xi(i, j3) + f3*x3i
      xr(i, j4) = xr(i, j4) + f4*(x4r - x3r - x2r - x1r); xi(i, j4) = xi(i, j4) + f4*(x4i - x3i - x2i - x1i)
    end do
  end subroutine calc0_soa

end module m_tetrahedron_soa
