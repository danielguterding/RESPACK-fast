!
! Copyright (c) 2014 Mitsuaki Kawamura
!
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the
! "Software"), to deal in the Software without restriction, including
! without limitation the rights to use, copy, modify, merge, publish,
! distribute, sublicense, and/or sell copies of the Software, and to
! permit persons to whom the Software is furnished to do so, subject to
! the following conditions:
! 
! The above copyright notice and this permission notice shall be included
! in all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
! OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
! MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
! IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
! CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
! TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
! SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
!
MODULE libtetrabz_dos_mod
  !
  IMPLICIT NONE
  !
  PRIVATE
  PUBLIC libtetrabz_dos
  !
CONTAINS
!
! Initialize grid
!
SUBROUTINE libtetrabz_kgrid(ivvec,ng,k_indx)
  !
  IMPLICIT NONE
  !
  INTEGER,INTENT(IN) :: ivvec(3,20,6), ng(3)
  INTEGER,INTENT(OUT),ALLOCATABLE :: k_indx(:,:)
  !
  INTEGER :: it, i1, i2, i3, ii, ikv(3), nt
  !
  ALLOCATE(k_indx(20, 6*PRODUCT(ng(1:3))))
  !
  ! k-index for energy (Global index)
  !
  nt = 0
  DO i1 = 1, ng(1)
     DO i2 = 1, ng(2)
        DO i3 = 1, ng(3)
           !
           DO it = 1, 6
              !
              nt = nt + 1
              !
              DO ii = 1, 20
                 !
                 ikv(1:3) = (/i1, i2, i3/) + ivvec(1:3,ii,it) - 1
                 ikv(1:3) = MODULO(ikv(1:3), ng(1:3))
                 !
                 k_indx(ii,nt) = 1 + ikv(3) + ng(3) * ikv(2) + ng(3) * ng(2) * ikv(1)
                 !
              END DO
              !
           END DO
           !
        END DO
     END DO
  END DO
  !
END SUBROUTINE libtetrabz_kgrid
!
! define shortest diagonal line & define type of tetragonal
!
SUBROUTINE libtetrabz_initialize(ng,bvec,wlsm,k_indx)
  !
  IMPLICIT NONE
  !
  INTEGER,INTENT(IN) :: ng(3)
  REAL(8),INTENT(IN) :: bvec(3,3)
  REAL(8),INTENT(OUT) :: wlsm(4,20)
  INTEGER,INTENT(OUT),ALLOCATABLE :: k_indx(:,:)
  !
  INTEGER :: itype, i1, i2, i3, it, divvec(4,4), ivvec0(4), ivvec(3,20,6)
  REAL(8) :: l(4), bvec2(3,3), bvec3(3,4)
  !
  DO i1 = 1, 3
     bvec2(1:3,i1) = bvec(1:3,i1) / DBLE(ng(i1))
  END DO
  !
  bvec3(1:3,1) = -bvec2(1:3,1) + bvec2(1:3,2) + bvec2(1:3,3)
  bvec3(1:3,2) =  bvec2(1:3,1) - bvec2(1:3,2) + bvec2(1:3,3)
  bvec3(1:3,3) =  bvec2(1:3,1) + bvec2(1:3,2) - bvec2(1:3,3)
  bvec3(1:3,4) =  bvec2(1:3,1) + bvec2(1:3,2) + bvec2(1:3,3)
  !
  ! length of delta bvec
  !
  DO i1 = 1, 4
     l(i1) = DOT_PRODUCT(bvec3(1:3,i1),bvec3(1:3,i1))
  END DO
  !
  itype = MINLOC(l(1:4),1)
  !
  ! start & last
  !
  ivvec0(1:4) = (/ 0, 0, 0, 0 /)
  !
  divvec(1:4,1) = (/ 1, 0, 0, 0 /)
  divvec(1:4,2) = (/ 0, 1, 0, 0 /)
  divvec(1:4,3) = (/ 0, 0, 1, 0 /)
  divvec(1:4,4) = (/ 0, 0, 0, 1 /)
  !
  ivvec0(itype) = 1
  divvec(itype, itype) = - 1
  !
  ! Corners of tetrahedra
  !
  it = 0
  DO i1 = 1, 3
     DO i2 = 1, 3
        IF(i2 == i1) CYCLE
        DO i3 = 1, 3
           IF(i3 == i1 .OR. i3 == i2) CYCLE
           !
           it = it + 1
           !
           ivvec(1:3,1,it) = ivvec0(1:3)
           ivvec(1:3,2,it) = ivvec(1:3,1,it) + divvec(1:3,i1)
           ivvec(1:3,3,it) = ivvec(1:3,2,it) + divvec(1:3,i2)
           ivvec(1:3,4,it) = ivvec(1:3,3,it) + divvec(1:3,i3)
           !
        END DO
     END DO
  END DO
  !
  ! Additional points
  !
  ivvec(1:3, 5,1:6) = 2 * ivvec(1:3,1,1:6) - ivvec(1:3,2,1:6)
  ivvec(1:3, 6,1:6) = 2 * ivvec(1:3,2,1:6) - ivvec(1:3,3,1:6)
  ivvec(1:3, 7,1:6) = 2 * ivvec(1:3,3,1:6) - ivvec(1:3,4,1:6)
  ivvec(1:3, 8,1:6) = 2 * ivvec(1:3,4,1:6) - ivvec(1:3,1,1:6)
  !
  ivvec(1:3, 9,1:6) = 2 * ivvec(1:3,1,1:6) - ivvec(1:3,3,1:6)
  ivvec(1:3,10,1:6) = 2 * ivvec(1:3,2,1:6) - ivvec(1:3,4,1:6)
  ivvec(1:3,11,1:6) = 2 * ivvec(1:3,3,1:6) - ivvec(1:3,1,1:6)
  ivvec(1:3,12,1:6) = 2 * ivvec(1:3,4,1:6) - ivvec(1:3,2,1:6)
  !
  ivvec(1:3,13,1:6) = 2 * ivvec(1:3,1,1:6) - ivvec(1:3,4,1:6)
  ivvec(1:3,14,1:6) = 2 * ivvec(1:3,2,1:6) - ivvec(1:3,1,1:6)
  ivvec(1:3,15,1:6) = 2 * ivvec(1:3,3,1:6) - ivvec(1:3,2,1:6)
  ivvec(1:3,16,1:6) = 2 * ivvec(1:3,4,1:6) - ivvec(1:3,3,1:6)
  !
  ivvec(1:3,17,1:6) =  ivvec(1:3,4,1:6) - ivvec(1:3,1,1:6) + ivvec(1:3,2,1:6)
  ivvec(1:3,18,1:6) =  ivvec(1:3,1,1:6) - ivvec(1:3,2,1:6) + ivvec(1:3,3,1:6)
  ivvec(1:3,19,1:6) =  ivvec(1:3,2,1:6) - ivvec(1:3,3,1:6) + ivvec(1:3,4,1:6)
  ivvec(1:3,20,1:6) =  ivvec(1:3,3,1:6) - ivvec(1:3,4,1:6) + ivvec(1:3,1,1:6)
  !
  !WRITE(*,*) "[libtetrabz] Improved tetrahedron method is used."
  !
  wlsm(1, 1: 4) = DBLE((/1440,    0,   30,    0/))
  wlsm(2, 1: 4) = DBLE((/   0, 1440,    0,   30/))
  wlsm(3, 1: 4) = DBLE((/  30,    0, 1440,    0/))
  wlsm(4, 1: 4) = DBLE((/   0,   30,    0, 1440/))
  !
  wlsm(1, 5: 8) = DBLE((/ -38,    7,   17,  -28/))
  wlsm(2, 5: 8) = DBLE((/ -28,  -38,    7,   17/))
  wlsm(3, 5: 8) = DBLE((/  17,  -28,  -38,    7/))
  wlsm(4, 5: 8) = DBLE((/   7,   17,  -28,  -38/))
  !
  wlsm(1, 9:12) = DBLE((/ -56,    9,  -46,    9/))
  wlsm(2, 9:12) = DBLE((/   9,  -56,    9,  -46/))
  wlsm(3, 9:12) = DBLE((/ -46,    9,  -56,    9/))
  wlsm(4, 9:12) = DBLE((/   9,  -46,    9,  -56/))
  !
  wlsm(1,13:16) = DBLE((/ -38,  -28,   17,    7/))
  wlsm(2,13:16) = DBLE((/   7,  -38,  -28,   17/))
  wlsm(3,13:16) = DBLE((/  17,    7,  -38,  -28/))
  wlsm(4,13:16) = DBLE((/ -28,   17,    7,  -38/))
  !
  wlsm(1,17:20) = DBLE((/ -18,  -18,   12,  -18/))
  wlsm(2,17:20) = DBLE((/ -18,  -18,  -18,   12/))
  wlsm(3,17:20) = DBLE((/  12,  -18,  -18,  -18/))
  wlsm(4,17:20) = DBLE((/ -18,   12,  -18,  -18/))
  !
  wlsm(1:4,1:20) = wlsm(1:4,1:20) / 1260d0
  !
  CALL libtetrabz_kgrid(ivvec,ng,k_indx)
  !
END SUBROUTINE libtetrabz_initialize
!
! Compute DOS
!
SUBROUTINE libtetrabz_dos(bvec,nb,ng,eig,wght,ne,e0)
  !
  IMPLICIT NONE
  !
  INTEGER,INTENT(IN) :: nb, ng(3), ne
  REAL(8),INTENT(IN) :: bvec(9), eig(nb,PRODUCT(ng(1:3))), e0(ne)
  REAL(8),INTENT(OUT) :: wght(ne,nb,PRODUCT(ng(1:3)))
  !
  INTEGER,ALLOCATABLE :: k_indx(:,:)
  REAL(8) :: wlsm(4,20)
  !
  CALL libtetrabz_initialize(ng,bvec,wlsm,k_indx)
  !
  CALL libtetrabz_dos_main(wlsm,k_indx,nb,PRODUCT(ng(1:3)),eig,ne,e0,wght)
  !
  DEALLOCATE(k_indx)
  !
END SUBROUTINE libtetrabz_dos
!
! Main SUBROUTINE for Dos : Delta(E - E1)
!
SUBROUTINE libtetrabz_dos_main(wlsm,k_indx,nb,nkBZ,eig,ne,e0,dos)
  !
  IMPLICIT NONE
  !
  INTEGER,INTENT(IN) :: nb, nkBZ, ne, &
  &                     k_indx(20,6*nkBZ)
  REAL(8),INTENT(IN) :: wlsm(4,20), eig(nb,nkBZ), e0(ne)
  REAL(8),INTENT(OUT) :: dos(ne,nb,nkBZ)
  !
  INTEGER :: ib, it, ie, indx(4)
  REAL(8) :: e(4), ei1(4,nb), tsmall(3,4), V, w1(ne,4)
  !
  dos(1:ne, 1:nb, 1:nkBZ) = 0d0
  !
  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP & SHARED(dos,eig,e0,k_indx,nb,ne,wlsm,nkBZ) &
  !$OMP & PRIVATE(e,ei1,ib,ie,indx,it,tsmall,V,w1)
  !
  DO it = 1, 6*nkBZ
     !
     DO ib = 1, nb
        ei1(1:4,ib) = MATMUL(wlsm(1:4,1:20), eig(ib,k_indx(1:20,it)))
     END DO
     !
     !$OMP DO
     DO ib = 1, nb
        !
        w1(1:ne,1:4) = 0d0
        e(1:4) = ei1(1:4, ib)
        CALL libtetrabz_sort(4,e,indx)
        !
        DO ie = 1, ne
           !
           IF((e(1) <  e0(ie) .AND. e0(ie) <= e(2)) .OR. &
           &  (e(1) <= e0(ie) .AND. e0(ie) <  e(2))) THEN
              !
              CALL libtetrabz_triangle_a1(e(1:4) - e0(ie),V,tsmall)
              w1(ie,indx(1:4)) = w1(ie,indx(1:4)) + V * SUM(tsmall(1:3,1:4), 1) / 3d0
              !
           ELSE IF((e(2) <  e0(ie) .AND. e0(ie) <= e(3)) .OR. &
           &       (e(2) <= e0(ie) .AND. e0(ie) <  e(3))) THEN
              !
              CALL libtetrabz_triangle_b1(e(1:4) - e0(ie),V,tsmall)
              w1(ie,indx(1:4)) = w1(ie,indx(1:4)) + V * SUM(tsmall(1:3,1:4), 1) / 3d0
              !
              CALL libtetrabz_triangle_b2(e(1:4) - e0(ie),V,tsmall)
              w1(ie,indx(1:4)) = w1(ie,indx(1:4)) + V * SUM(tsmall(1:3,1:4), 1) / 3d0
              !
           ELSE IF((e(3) < e0(ie) .AND. e0(ie) <= e(4)) .OR. &
           &       (e(3) <= e0(ie) .AND. e0(ie) < e(4))) THEN
              !
              CALL libtetrabz_triangle_c1(e(1:4) - e0(ie),V,tsmall)
              w1(ie,indx(1:4)) = w1(ie,indx(1:4)) + V * SUM(tsmall(1:3,1:4), 1) / 3d0
              !
           END IF
           !
        END DO ! ie
        !
        dos(1:ne,ib,k_indx(1:20,it)) = dos(1:ne,ib,      k_indx(1:20,it)) &
        &                      + MATMUL(w1(1:ne, 1:4), wlsm(1:4,1:20))
        !
     END DO ! ib
     !$OMP END DO NOWAIT
     !
  END DO ! it
  !
  !$OMP END PARALLEL
  !
  dos(1:ne,1:nb,1:nkBZ) = dos(1:ne,1:nb,1:nkBZ) / DBLE(6 * nkBZ)
  !
END SUBROUTINE libtetrabz_dos_main
!
! Simple sort
!
SUBROUTINE libtetrabz_sort(n,key,indx)
  !
  IMPLICIT NONE
  !
  integer,INTENT(IN) :: n
  REAL(8),INTENT(inout) :: key(n)
  INTEGER,INTENT(OUT) :: indx(n)
  !
  INTEGER :: i, i0, indx0
  REAL(8) :: key0
  !
  DO i = 1, n
     indx(i) = i
  END DO
  !
  DO i = 1, n - 1
     key0 = MINVAL(key(i+1:n))
     i0   = MINLOC(key(i+1:n),1) + i
     IF(key(i) > key0) THEN
        key(i0) = key(i)
        key(i) = key0
        !
        indx0 = indx(i0)
        indx(i0) = indx(i)
        indx(i) = indx0
     END IF
  END DO
  !
END SUBROUTINE libtetrabz_sort
!
! Cut triangle A1
!
SUBROUTINE libtetrabz_triangle_a1(e,V,tsmall)
  !
  IMPLICIT NONE
  !
  REAL(8),INTENT(IN) :: e(4)
  REAL(8),INTENT(OUT) :: V
  REAL(8),INTENT(OUT) :: tsmall(3,4)
  !
  INTEGER :: ii
  REAL(8) :: a(4,4)
  !
  DO ii = 1, 4
     a(1:4,ii) = (0d0 - e(ii)) / (e(1:4) - e(ii))
  END DO
  !
  !V = 3d0 * a(2,1) * a(3,1) * a(4,1) / (0d0 - e(1))
  V = 3d0 * a(2,1) * a(3,1)           / (e(4) - e(1))
  !
  tsmall(1,1:4) = (/a(1,2), a(2,1),    0d0,    0d0/)
  tsmall(2,1:4) = (/a(1,3),    0d0, a(3,1),    0d0/) 
  tsmall(3,1:4) = (/a(1,4),    0d0,    0d0, a(4,1)/)
  !
END SUBROUTINE libtetrabz_triangle_a1
!
! Cut triangle B1
!
SUBROUTINE libtetrabz_triangle_b1(e,V,tsmall)
  !
  IMPLICIT NONE
  !
  REAL(8),INTENT(IN) :: e(4)
  REAL(8),INTENT(OUT) :: V
  REAL(8),INTENT(OUT) :: tsmall(3,4)
  !
  INTEGER :: ii
  REAL(8) :: a(4,4)
  !
  DO ii = 1, 4
     a(1:4,ii) = (0d0 - e(ii)) / (e(1:4) - e(ii))
  END DO
  !
  !V = 3d0 * a(3,1) * a(4,1) * a(2,4) / (0d0 - e(1))
  V = 3d0           * a(4,1) * a(2,4) / (e(3) - e(1))
  !
  tsmall(1,1:4) = (/a(1,3),    0d0, a(3,1),    0d0/)
  tsmall(2,1:4) = (/a(1,4),    0d0,    0d0, a(4,1)/)
  tsmall(3,1:4) = (/   0d0, a(2,4),    0d0, a(4,2)/)
  !
END SUBROUTINE libtetrabz_triangle_b1
!
! Cut triangle B2
!
SUBROUTINE libtetrabz_triangle_b2(e,V,tsmall)
  !
  IMPLICIT NONE
  !
  REAL(8),INTENT(IN) :: e(4)
  REAL(8),INTENT(OUT) :: V
  REAL(8),INTENT(OUT) :: tsmall(3,4)
  !
  INTEGER :: ii
  REAL(8) :: a(4,4)
  !
  DO ii = 1, 4
     a(1:4,ii) = (0d0 - e(ii)) / (e(1:4) - e(ii))
  END DO
  !
  !V = 3d0 * a(2,3) * a(3,1) * a(4,2) / (0d0 - e(1))
  V = 3d0 * a(2,3)           * a(4,2) / (e(3) - e(1))
  !
  tsmall(1,1:4) = (/a(1,3),    0d0, a(3,1),    0d0/)
  tsmall(2,1:4) = (/   0d0, a(2,3), a(3,2),    0d0/)
  tsmall(3,1:4) = (/   0d0, a(2,4),    0d0, a(4,2)/)
  !
END SUBROUTINE libtetrabz_triangle_b2
!
! Cut triangle C1
!
SUBROUTINE libtetrabz_triangle_c1(e,V,tsmall)
  !
  IMPLICIT NONE
  !
  REAL(8),INTENT(IN) :: e(4)
  REAL(8),INTENT(OUT) :: V
  REAL(8),INTENT(OUT) :: tsmall(3,4)
  !
  INTEGER :: ii
  REAL(8) :: a(4,4)
  !
  DO ii = 1, 4
     a(1:4,ii) = (0d0 - e(ii)) / (e(1:4) - e(ii))
  END DO
  !
  !V = 3d0 * a(1,4) * a(2,4) * a(3,4) / (e(4) - 0d0)
  V = 3d0 * a(1,4) * a(2,4)           / (e(4) - e(3))
  !
  tsmall(1,1:4) = (/a(1,4),    0d0,    0d0, a(4,1)/)
  tsmall(2,1:4) = (/   0d0, a(2,4),    0d0, a(4,2)/)
  tsmall(3,1:4) = (/   0d0,    0d0, a(3,4), a(4,3)/)
  !
END SUBROUTINE libtetrabz_triangle_c1
!
END MODULE libtetrabz_dos_mod
