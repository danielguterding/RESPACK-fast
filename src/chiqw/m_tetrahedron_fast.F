!
!     Generalized Tetrahedron Method
!
!     (Copyright) 2003 Takeo Fujiwara, Susumu Yamamoto, Yasushi Ishii, and
!     Yoshiro Nohara
!
!     This program uses the generalized tetrahedron method and implements
!     an efficient tool in the GW-LMTO package developed by Fujiwara group
!     in University of Tokyo. The algorithm of the generalized tetrahedron
!     method is written in the following published papers:
!
!     [Generalized tetrahedron method] Takeo Fujiwara, Susumu Yamamoto, and
!     Yasushi Ishii, J. Phys. Soc. Jpn. 72, No.4, 777-780 (2003).
!     (cond.mat/0211159)
!
!     [GW-LMTO package] Yoshiro Nohara, Susumu Yamamoto, and Takeo Fujiwara,
!     Phys. Rev. B 79, 195110 (2009)
!
!     One can use this part of the program package in one's developed code
!     on the condition of citing these two papers and mentioning the name of
!     `the generalized tetrahedron method'  in the published paper.
!
!     For those not following the above, this part of the program package is also
!     released under the MIT License. http://opensource.org/licenses/mit-license.php
!
!     Modified 2026 by Daniel Guterding for the zen-opt fork of RESPACK
!     (copy of m_tetrahedron_20170325.F): log ratios via vectorisable real
!     log/atan2 (lratio), both signs in one pass (ttrhdrn_chi0_pm), fused
!     normal-tetrahedron kernel (calc0_fused). Same formulas and conditions.
!
      module m_tetrahedron_fast
      implicit none
      private

      public :: ttrhdrn_mkidx
      public :: ttrhdrn_chi0
      public :: ttrhdrn_chi0_pm
      public :: ttrhdrn_sum
      public :: ttrhdrn_simple
      public :: ttrhdrn_sub1normal
      public :: ttrhdrn_sub1simple
      public :: ttrhdrn_sub1g30g20
      public :: ttrhdrn_sub1g30g21
      public :: ttrhdrn_sub1g30
      public :: ttrhdrn_sub2normal
      public :: ttrhdrn_sub2simple
      public :: ttrhdrn_sub2g30g20
      public :: ttrhdrn_sub2g30g10
      public :: ttrhdrn_sub2g30g21
      public :: ttrhdrn_sub2g30
      public :: ttrhdrn_sub2g10


      integer, parameter, public :: ttrhdrn_vtx=4
      integer, parameter, public :: ttrhdrn_div=6
      integer, parameter :: nvtx=ttrhdrn_vtx
      integer, parameter :: ndiv=ttrhdrn_div
      integer, parameter, public :: ttrhdrn_tbl1(nvtx,ndiv)
     $                       =reshape((/1,2,3,6
     $                                 ,1,3,5,6
     $                                 ,3,5,6,7
     $                                 ,2,3,4,6
     $                                 ,3,4,6,8
     $                                 ,3,6,7,8/)
     $                               ,(/nvtx,ndiv/))

      contains

      subroutine ttrhdrn_simple(dmna,dmnr,n1,n2,n3,imt1,fk,gk,xo)
      real(8), intent(in) :: dmna,dmnr
      integer, intent(in) :: n1
      integer, intent(in) :: n2
      integer, intent(in) :: n3
      integer, intent(in) :: imt1(nvtx,n1*n2*n3*ndiv)
      complex(8), intent(in) :: fk(n1*n2*n3)
      complex(8), intent(in) :: gk(n1*n2*n3)
      complex(8), intent(inout) :: xo(n1*n2*n3)
      complex(8) :: ca1(nvtx),ca2(1)
      ca2=(0.d0,0.d0)
      call ttrhdrn_sum(dmna,dmnr,n1,n2,n3,imt1,fk,gk
     $                ,1,ca2,ca1,xo)
      end subroutine

      subroutine ttrhdrn_sum(dmna,dmnr,n1,n2,n3,imt1,fk,gk,ne,em,ca1,xo)
      real(8), intent(in) :: dmna,dmnr
      integer, intent(in) :: n1
      integer, intent(in) :: n2
      integer, intent(in) :: n3
      integer, intent(in) :: imt1(nvtx,n1*n2*n3*ndiv)
      complex(8), intent(in) :: fk(n1*n2*n3)
      complex(8), intent(in) :: gk(n1*n2*n3)
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(  out) :: ca1(ne*nvtx)
      complex(8), intent(inout) :: xo(ne,n1*n2*n3)
      integer :: i1,i2,i3,i4,i5,i6,i7
      do i1=1,n1*n2*n3*ndiv
        call calc(dmna,dmnr,imt1(:,i1),n1*n2*n3
     $           ,fk
     $           ,gk,ne,em,ca1
     $           ,xo)
      end do
      end subroutine

      subroutine ttrhdrn_chi0(dmna,dmnr,n1,n2,n3,imt1
     $                       ,e1,e2,f1,f2,delt,dsgn,ne,em,ca1,xo)
      real(8), intent(in) :: dmna,dmnr
      integer, intent(in) :: n1,n2,n3,imt1(nvtx,n1*n2*n3*ndiv)
      complex(8), intent(in) :: e1(n1*n2*n3),e2(n1*n2*n3)
      real(8), intent(in) :: f1,f2,delt,dsgn
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(  out) :: ca1(ne*nvtx)
      complex(8), intent(inout) :: xo(ne,n1*n2*n3)
      integer :: i1,i2,i3,i4,i5,i6,i7
      real(8) :: ee1(4),ee2(4)
      complex(8) :: ffk(4),ggk(4),xxo(ne,4)
      integer, parameter :: ia1(4)=(/1,2,3,4/),nmax=99
!--- 2013.4.23 
!     real(8), allocatable :: dwgt(:,:,:),dvol(:)
!     allocate(dwgt(4,4,nmax),dvol(nmax))
!---
      real(8) :: dwgt(4,4,nmax),dvol(nmax)
      do i1=1,n1*n2*n3*ndiv
        dwgt(:,:,1)=0.d0; do i2=1,4; dwgt(i2,i2,1)=1.d0; end do
        dvol(1)=1.d0; i2=1
        ee2=dble(e2(imt1(:,i1))); call ttct(f2,ee2,+1,nmax,i2,dwgt,dvol)
        ee1=dble(e1(imt1(:,i1))); call ttct(f1,ee1,-1,nmax,i2,dwgt,dvol)
        do i3=1,i2
          xxo=0.d0; ffk=dvol(i3)
          do i4=1,4
            ggk(i4)=-dcmplx(sum((ee2-ee1)*dwgt(:,i4,i3)),-delt)
          end do
          ffk=ffk*dsgn; ggk=ggk*dsgn
          call calc(dmna,dmnr,ia1,4
     $             ,ffk
     $             ,ggk,ne,em,ca1
     $             ,xxo)
          do i4=1,4; do i5=1,4
            xo(:,imt1(i5,i1))=xo(:,imt1(i5,i1))+xxo(:,i4)*dwgt(i5,i4,i3)
          end do; end do
        end do
      end do
!--- 2013.4.23 
!     deallocate(dwgt,dvol)
!---
      end subroutine

      subroutine ttct(dval,dvtx,imod,nmax,ntet,dwgt,dvol)
      real(8), intent(in) :: dval,dvtx(4)
      integer, intent(in) :: imod,nmax
      integer, intent(inout) :: ntet
      real(8), intent(inout) :: dwgt(4,4,nmax),dvol(nmax)
      logical :: bvtx0(4)
      logical :: b1
      integer :: i1,i2
      real(8) :: d1,d2
      real(8) :: dvtx0(4),dvol0
      integer :: itet
!--- 2013.4.23 
!      logical, allocatable :: bvtx(:,:),btsk(:)
!      allocate(bvtx(4,nmax),btsk(nmax)); bvtx=.false.; btsk=.false.
!---
      logical :: bvtx(4,nmax),btsk(nmax)
      bvtx=.false.
      btsk=.false.
!---
      itet=0
      do while(itet.lt.ntet); itet=itet+1
        do i1=1,4; dvtx0(i1)=sum(dwgt(:,i1,itet)*dvtx(:)); end do
        dvol0=dvol(itet); bvtx0=bvtx(:,itet); b1=.false.
        do i1=   1,3; if(bvtx0(i1))cycle
        do i2=i1+1,4; if(bvtx0(i2))cycle
          b1=(dvtx0(i1)-dval)*(dval-dvtx0(i2)).gt.0.d0; if(b1)exit
        end do; if(b1)exit
        end do
        if(b1)then
!         write(*,*)"divide",dvtx0(i2),dvtx0(i1)
          d1=(dvtx0(i2)-dval)/(dvtx0(i2)-dvtx0(i1))
          d2=(dval-dvtx0(i1))/(dvtx0(i2)-dvtx0(i1))
          ntet=ntet+1; if(ntet.gt.nmax)stop "ntet.gt.nmax"
          bvtx(:,ntet)=bvtx(:,itet); dwgt(:,:,ntet)=dwgt(:,:,itet)
          bvtx(i1,ntet)=.true.
          dwgt(:,i1,ntet)=dwgt(:,i1,itet)*d1+dwgt(:,i2,itet)*d2
          dvol(ntet)=dvol(itet)*d1
          bvtx(i2,itet)=.true.
          dwgt(:,i2,itet)=dwgt(:,i1,itet)*d1+dwgt(:,i2,itet)*d2
          dvol(itet)=dvol(itet)*d2
          itet=itet-1
        else
          do i1=1,4; if(bvtx0(i1))cycle
            btsk(itet)=btsk(itet)
     $        .or.dble(imod)*(dvtx0(i1)-dval).gt.0.d0
!           if(btsk(itet))write(*,"(4g15.7)")dvtx0
!           btsk(itet)=.true.!!!This makes last checking values become 1!!!
          end do
        end if
      end do
!     write(*,*)"ntet=",ntet
!     write(*,*)"btet=",btsk(1:ntet)
      itet=0! dvtx0=0.d0
      do i1=1,ntet; if(.not.btsk(i1))cycle
        itet=itet+1; dwgt(:,:,itet)=dwgt(:,:,i1);dvol(itet)=dvol(i1)
!       write(*,"(4g15.7)")itet,dvol(itet)
!       write(*,"(4g15.7)")dwgt(:,:,itet)
!       dvtx0=dvtx0+dvol(itet)
!    $  *(dwgt(:,1,itet)+dwgt(:,2,itet)+dwgt(:,3,itet)+dwgt(:,4,itet))
      end do; ntet=itet
!     write(*,*)"check",dvtx0
!--- 2013.4.23 
!      deallocate(bvtx,btsk)
!---
      end subroutine

      subroutine calc(dmna,dmnr,ia1,nnn,fk,gk,ne,em,ca1,xo)
      real(8), intent(in) :: dmna,dmnr
      integer, intent(in) :: ia1(nvtx)
      integer, intent(in) :: nnn
      complex(8), intent(in) :: fk(nnn)
      complex(8), intent(in) :: gk(nnn)
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(  out) :: ca1(ne,nvtx)
      complex(8), intent(inout) :: xo(ne,nnn)
      integer :: i1,i2,i3,i4,i5,i6
      integer :: ia2(4),ia3(4),ia4(4)
      complex(8) :: geq1,geq2
      call chk1(dmna,dmnr,nnn,gk,ia1,ia2,ia3,ia4,i3)
      select case(sum(ia2))
      case(0)
!fork (zen-opt): fused evaluation of the four normal-tetrahedron terms
        call calc0_fused(nnn,fk,gk,ia4,ne,em,xo)
      case(2)
          geq1=sum(gk(ia3(1:2)))/2.d0
          call ttrhdrn_sub2g10   (
     $             geq1      ,geq1      ,gk(ia4(1)),gk(ia4(2))
     $            ,ne,em,ca1(:,1))
          call ttrhdrn_sub2g30   (
     $             geq1      ,gk(ia4(1)),gk(ia4(2)),geq1      
     $            ,ne,em,ca1(:,2))
          call ttrhdrn_sub1g30   (
     $             geq1      ,gk(ia4(2)),gk(ia4(1)),geq1      
     $            ,ne,em,ca1(:,3))
          xo(:,ia3(1))=xo(:,ia3(1))+fk(ia3(1))*ca1(:,1)
          xo(:,ia3(2))=xo(:,ia3(2))+fk(ia3(2))*ca1(:,1)
          xo(:,ia4(1))=xo(:,ia4(1))+fk(ia4(1))*ca1(:,2)
          xo(:,ia4(2))=xo(:,ia4(2))+fk(ia4(2))
     $            *(ca1(:,3)-ca1(:,2)-ca1(:,1)-ca1(:,1))
      case(3)
          geq1=sum(gk(ia3(1:3)))/3.d0
          call ttrhdrn_sub2g30g10(
     $             geq1      ,geq1      ,gk(ia4(1)),geq1      
     $            ,ne,em,ca1(:,1))
          call ttrhdrn_sub1g30g20(
     $             geq1      ,gk(ia4(1)),geq1      ,geq1      
     $            ,ne,em,ca1(:,2))
          xo(:,ia3(1))=xo(:,ia3(1))+fk(ia3(1))*ca1(:,1)
          xo(:,ia3(2))=xo(:,ia3(2))+fk(ia3(2))*ca1(:,1)
          xo(:,ia3(3))=xo(:,ia3(3))+fk(ia3(3))*ca1(:,1)
          xo(:,ia4(1))=xo(:,ia4(1))+fk(ia4(1))
     $            *(ca1(:,2)-ca1(:,1)-ca1(:,1)-ca1(:,1))
      case(4)
        select case(i3)
        case(2)
          geq1=sum(gk(ia3(1:2)))/2.d0; geq2=sum(gk(ia3(3:4)))/2.d0
          call ttrhdrn_sub2g30g21(
     $             geq2      ,geq1      ,geq1      ,geq2      
     $            ,ne,em,ca1(:,1))
          call ttrhdrn_sub1g30g21(
     $             geq1      ,geq2      ,geq2      ,geq1      
     $            ,ne,em,ca1(:,2))
          xo(:,ia3(1))=xo(:,ia3(1))+fk(ia3(1))*ca1(:,1)
          xo(:,ia3(2))=xo(:,ia3(2))+fk(ia3(2))*ca1(:,1)
          xo(:,ia3(3))=xo(:,ia3(3))+fk(ia3(3))
     $                     *(ca1(:,2)-ca1(:,1)-ca1(:,1))*.5d0
          xo(:,ia3(4))=xo(:,ia3(4))+fk(ia3(4))
     $                     *(ca1(:,2)-ca1(:,1)-ca1(:,1))*.5d0
        case(3:6)
          geq1=sum(gk(ia3(1:4)))/4.d0
          call ttrhdrn_sub2simple(
     $             geq1      ,geq1      ,geq1      ,geq1      
     $            ,ne,em,ca1(:,1))
          xo(:,ia3(1))=xo(:,ia3(1))+ca1(:,1)*fk(ia3(1))
          xo(:,ia3(2))=xo(:,ia3(2))+ca1(:,1)*fk(ia3(2))
          xo(:,ia3(3))=xo(:,ia3(3))+ca1(:,1)*fk(ia3(3))
          xo(:,ia3(4))=xo(:,ia3(4))+ca1(:,1)*fk(ia3(4))
        case default
          call errexec("bad i3")
        end select
      case default
        call errexec("bad sum ia2")
      end select
      end subroutine

      subroutine ttrhdrn_sub1simple(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      x0=1.d0/((g0+em)*6.d0)
      end subroutine

      subroutine ttrhdrn_sub1g30g21(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_01(ne)
      call lratio(ne,g0,g1,em,lr_01)
      x0=   (.5d0*(g0+g1)+em)/(g0-g1)**2
      x0=x0-   lr_01*(g0+em)*(g1+em)/(g0-g1)**3
      end subroutine

      subroutine ttrhdrn_sub1g30g20(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_01(ne)
      call lratio(ne,g0,g1,em,lr_01)
      x0=   (g0-3.d0*g1-2.d0*em)*.25d0/(g0-g1)**2
      x0=x0+   (g1+em)**2*lr_01*.5d0/(g0-g1)**3
      end subroutine

      subroutine ttrhdrn_sub1g30   (g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_01(ne),lr_12(ne)
      call lratio(ne,g0,g1,em,lr_01)
      call lratio(ne,g1,g2,em,lr_12)
      x0=   .5d0/(g0-g2)
      x0=x0-   .5d0/((g0-g2)*(g0-g1))**2
     $        *(g0+em)
     $        *(-2.d0*(g1+em)*(g2+em)+(g0+em)*(g1+g2+2.d0*em))
     $        *lr_01
      x0=x0+   .5d0/(g0-g2)**2
     $        *(g2+em)**2*lr_12/(g1-g2)
      x0=x0+   .5d0/((g0-g2)*(g0-g1))
     $        *(g1+em)
      end subroutine

      subroutine ttrhdrn_sub1normal(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_10(ne),lr_20(ne),lr_30(ne)
      call lratio(ne,g1,g0,em,lr_10)
      call lratio(ne,g2,g0,em,lr_20)
      call lratio(ne,g3,g0,em,lr_30)
!     x0=   (-g0-g1-2.d0*em)/((g0-g2)*(g0-g3))*.25d0
      x0=  +   (
     $         -(g1+em)**2*lr_10)
     $        /((g0-g2)*(g0-g3)*(g0-g1))*.5d0
!     x0=x0-   (-g1-g2-2.d0*em)/((g0-g2)*(g2-g3))*.25d0
      x0=x0-   ((g1+em)**2*lr_10
     $         -(g2+em)**2*lr_20)
     $        /((g0-g2)*(g1-g2)*(g2-g3))*.5d0
!     x0=x0+   (-g1-g3-2.d0*em)/((g0-g3)*(g2-g3))*.25d0
      x0=x0+   ((g1+em)**2*lr_10
     $         -(g3+em)**2*lr_30)
     $        /((g1-g3)*(g0-g3)*(g2-g3))*.5d0
      end subroutine

      subroutine ttrhdrn_sub2simple(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      x0=1.d0/((g0+em)*24.d0)
      end subroutine

      subroutine ttrhdrn_sub2g30g20(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_01(ne)
      call lratio(ne,g0,g1,em,lr_01)
      x0=   ((g0-g1)*(g0+2.d0*g1+3.d0*em)
     $         -6.d0*(g0+em)*(g1+em))
     $        /(12.d0*(g0-g1)**3)
      x0=x0+   (g0+em)*(g1+em)**2*lr_01
     $        /(g0-g1)**4*.5d0
      end subroutine

      subroutine ttrhdrn_sub2g30g21(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      call ttrhdrn_sub2g30g20(g1,g0,g1,g1,ne,em,x0)
      end subroutine

      subroutine ttrhdrn_sub2g30g10(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_02(ne)
      call lratio(ne,g0,g2,em,lr_02)
      x0=   (2.d0*(g0+em)**2-7.d0*(g0+em)*(g2+em)+11.d0*(g2+em)**2)
     $        /(36.d0*(g0-g2)**3)
      x0=x0-   (g2+em)**3*lr_02
     $        /(6.d0*(g0-g2)**4)
      end subroutine

      subroutine ttrhdrn_sub2g10   (g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_20(ne),lr_30(ne)
      call lratio(ne,g2,g0,em,lr_20)
      call lratio(ne,g3,g0,em,lr_30)
      x0=   (g0+em)*((g0+em)**2+5.d0*(g2+em)*(g3+em)
     $                 -3.d0*(g0+em)*(g2+g3+2.d0*em))
     $        /(12.d0*(g0-g2)**2*(g0-g3)**2)
!     x0=x0+   (g0+em)*(3.d0*(g2+em)*(g3+em)*(g0-g2)*(g0-g3)
!    $                 +((g0+em)*(g2-g3))**2)
!    $        *log(g0+em)/(6.d0*(g0-g2)**3*(g0-g3)**3)
      x0=x0-   (g2+em)**3*lr_20/(6.d0*(g0-g2)**3*(g2-g3))
      x0=x0+   (g3+em)**3*lr_30/(6.d0*(g0-g3)**3*(g2-g3))
      end subroutine

      subroutine ttrhdrn_sub2g30   (g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_02(ne),lr_12(ne)
      call lratio(ne,g0,g2,em,lr_02)
      call lratio(ne,g1,g2,em,lr_12)
      x0=   ((g0+em)**2*(g1-g2)+(g1+em)**2*(g0-g2))
     $        /(6.d0*(g0-g2)*(g0-g1)**2*(g1-g2))
      x0=x0-   (g0+em)**2*(2.d0*(g1+em)*(g0-g2)+(g2+em)*(g0-g1))
     $                   *lr_02
     $        /(6.d0*(g0-g2)**2*(g0-g1)**3)
      x0=x0+   (g1+em)**2*(2.d0*(g0+em)*(g1-g2)-(g2+em)*(g0-g1))
     $                   *lr_12
     $        /(6.d0*(g0-g1)**3*(g1-g2)**2)
!     x0=x0+   (g2+em)**3*log(g2+em)/(6.d0*(g0-g2)**2*(g1-g2)**2)
      end subroutine

      subroutine ttrhdrn_sub2normal(g0,g1,g2,g3,ne,em,x0)
      complex(8), intent(in) :: g0,g1,g2,g3
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(out) :: x0(ne)
      complex(8) :: lr_21(ne),lr_31(ne),lr_01(ne)
      call lratio(ne,g2,g1,em,lr_21)
      call lratio(ne,g3,g1,em,lr_31)
      call lratio(ne,g0,g1,em,lr_01)
      x0=-   (g1+em)**2/(6.d0*(g0-g1)*(g1-g2)*(g1-g3))
!     x0=x0+   ((g1+em)*(g2+em)**2*(g1-g3)*(g0-g3)
!    $         +(g1+em)*(g2+em)*(g3+em)*(g1-g2)*(g0-g3)
!    $         +(g0+em)*(g3+em)**2*(g1-g2)**2)*log(g1+em)
!    $        /(6.d0*(g0-g2)*(g0-g3)*(g1-g2)**2*(g1-g3)**2)
      x0=x0-   (g2+em)**3*lr_21
     $        /(6.d0*(g0-g2)*(g1-g2)**2*(g2-g3))
      x0=x0+   (g3+em)**3*lr_31
     $        /(6.d0*(g0-g3)*(g1-g3)**2*(g2-g3))
      x0=x0+   (g0+em)**3*lr_01
     $        /(6.d0*(g0-g1)**2*(g0-g2)*(g0-g3))
      end subroutine





      subroutine chk1(dmna,dmnr,nnn,gk,ia1,ia2,ia3,ia4,i3)
      real(8), intent(in) :: dmna,dmnr
      integer, intent(in) :: nnn
      complex(8), intent(in) :: gk(nnn)
      integer, intent(in) :: ia1(4)
      integer, intent(out):: ia2(4)
      integer, intent(out):: ia3(4)
      integer, intent(out):: ia4(4)
      integer, intent(out):: i3
      integer :: i1,i2,i4,i5
      real(8) :: d1,d2
      i3=0
      i4=0
      ia2=0
      ia3=0
      do i1=1,3
        do i2=i1+1,4
          d1=abs(gk(ia1(mod(i1,4)+1))-gk(ia1(mod(i2,4)+1)))
          d2=abs(gk(ia1(mod(i1,4)+1))+gk(ia1(mod(i2,4)+1)))*.5d0
           if(d1.lt.dmna.or.d1/d2.lt.dmnr)then
!          if(d1.lt.dmna.and.d1/d2.lt.dmnr)then
            i3=i3+1
            if(ia2(mod(i1,4)+1).eq.0)then
              i4=i4+1
              ia3(i4)=ia1(mod(i1,4)+1)
              ia2(mod(i1,4)+1)=1
            end if
            if(ia2(mod(i2,4)+1).eq.0)then
              i4=i4+1
              ia3(i4)=ia1(mod(i2,4)+1)
              ia2(mod(i2,4)+1)=1
            end if
          else
          end if
        end do
      end do
      i5=0
      ia4=0
      do i1=1,4
        if(ia2(mod(i1,4)+1).eq.0) then
          i5=i5+1
          ia4(i5)=ia1(mod(i1,4)+1)
        end if
      end do
      if(sum(ia2).ne.i4) call errexec("bad ia2 and i4")
      if(i4+i5.ne.4)     call errexec("bad i4 and i5")
      end subroutine

      subroutine ttrhdrn_mkidx(n1,n2,n3,idx1,bvec1,bvec2,bvec3)
      integer, intent(in) :: n1
      integer, intent(in) :: n2
      integer, intent(in) :: n3
      integer, intent(out):: idx1(nvtx,n1*n2*n3*ndiv)
      real(8), intent(in), optional :: bvec1(3),bvec2(3),bvec3(3)
      real(8) :: bvec(3,3)
      integer :: i1,i2,i3,i4,i5,i6,i7
      integer :: imt1(8,8)
      integer :: ia1(8)
      !
      integer :: nvec(3)!20180413 Y.Nohara 
      !
      bvec=reshape((/1d0,0d0,0d0,0d0,1d0,0d0,0d0,0d0,1d0/),(/3,3/))
      if(present(bvec1))bvec(:,1)=bvec1
      if(present(bvec2))bvec(:,2)=bvec2
      if(present(bvec3))bvec(:,3)=bvec3
      !
      !20180413 Y.Nohara    
      !
      !(/n1,n2,n3/) gives a warning at run time
      !
      !call libtetrabzinitialize(bvec,(/n1,n2,n3/),idx1)
      !
      nvec(1)=n1; nvec(2)=n2; nvec(3)=n3
      call libtetrabzinitialize(bvec,nvec(1),idx1)
      !
      end subroutine

!!!FROM LIBTETRABZ, MIT License, Copyright (c) 2014 Mitsuaki Kawamura!!!
      subroutine libtetrabzinitialize(bvec,ng,indx1)
      real(8), intent(in   ) :: bvec(3,3)
      integer, intent(in   ) :: ng(3)
      integer, intent(  out) :: indx1(4,*)
      integer :: nk,nt, ivvec(3,4,6), ikv(3)
      integer :: itype, i1, i2, i3, it, ii, divvec(4,4), ivvec0(4)
      real(8) :: l(4), bvec2(3,3), bvec3(3,4)
      nk = product(ng(1:3))
      nt = nk * 6
      do i1 = 1, 3
        bvec2(1:3,i1) = bvec(1:3,i1) / dble(ng(i1))
      end do
      bvec3(1:3,1) = -bvec2(1:3,1) + bvec2(1:3,2) + bvec2(1:3,3)
      bvec3(1:3,2) =  bvec2(1:3,1) - bvec2(1:3,2) + bvec2(1:3,3)
      bvec3(1:3,3) =  bvec2(1:3,1) + bvec2(1:3,2) - bvec2(1:3,3)
      bvec3(1:3,4) =  bvec2(1:3,1) + bvec2(1:3,2) + bvec2(1:3,3)
      ! length of delta bvec
      do i1 = 1, 4
        l(i1) = dot_product(bvec3(1:3,i1),bvec3(1:3,i1))
      end do
      itype = minloc(l(1:4),1)
      ! start & last
      ivvec0(1:4) = (/ 0, 0, 0, 0 /)
      divvec(1:4,1) = (/ 1, 0, 0, 0 /)
      divvec(1:4,2) = (/ 0, 1, 0, 0 /)
      divvec(1:4,3) = (/ 0, 0, 1, 0 /)
      divvec(1:4,4) = (/ 0, 0, 0, 1 /)
      ivvec0(itype) = 1
      divvec(itype, itype) = - 1
      ! Corners of tetrahedra
      it = 0
      do i1 = 1, 3
        do i2 = 1, 3
          if(i2 == i1) cycle
          do i3 = 1, 3
            if(i3 == i1 .or. i3 == i2) cycle
            it = it + 1
            ivvec(1:3,1,it) = ivvec0(1:3)
            ivvec(1:3,2,it) = ivvec(1:3,1,it) + divvec(1:3,i1)
            ivvec(1:3,3,it) = ivvec(1:3,2,it) + divvec(1:3,i2)
            ivvec(1:3,4,it) = ivvec(1:3,3,it) + divvec(1:3,i3)
          end do
        end do
      end do
!     allocate(indx1(20, 6 * nk), indx2(20, 6 * nk), indx3(20 * 6 * nk))
      nt = 0
      do i3 = 1, ng(3)
        do i2  = 1, ng(2)
          do i1 = 1, ng(1)
            do it = 1, 6
              nt = nt + 1
              do ii = 1, 4
                ikv(1:3) = (/i1, i2, i3/) + ivvec(1:3,ii,it) - 1
                ikv(1:3) = modulo(ikv(1:3), ng(1:3))
                indx1(ii,nt) = 1 + ikv(1) + ng(1) * ikv(2) 
     $                                    + ng(1) * ng(2) * ikv(3)
              end do
            end do
          end do
        end do
      end do
      end subroutine
!!!FROM LIBTETRABZ, MIT License, Copyright (c) 2014 Mitsuaki Kawamura!!!
        
      subroutine errexec(smsg)
      character(*), intent(in) :: smsg
      integer :: i(1:1)
      integer :: j
      j=0
      write(*,"(a)")"#ERROR in ttrhdrn_ : "//smsg
      i(j)=0
      write(*,"('#errexec cannot stop')")
      write(*,"('#please use debug option')")
      stop
      end subroutine


!--- fork (zen-opt): both signs of ttrhdrn_chi0 in one pass; the Fermi
!    surface cuts (ttct) do not depend on dsgn and are done only once.
!    xo accumulates the dsgn=+1 and dsgn=-1 contributions, i.e. the same
!    as ttrhdrn_chi0(..,+1,..) followed by ttrhdrn_chi0(..,-1,..).
      subroutine ttrhdrn_chi0_pm(dmna,dmnr,n1,n2,n3,imt1
     $                          ,e1,e2,f1,f2,delt,ne,em,ca1,xo)
      real(8), intent(in) :: dmna,dmnr
      integer, intent(in) :: n1,n2,n3,imt1(nvtx,n1*n2*n3*ndiv)
      complex(8), intent(in) :: e1(n1*n2*n3),e2(n1*n2*n3)
      real(8), intent(in) :: f1,f2,delt
      integer, intent(in) :: ne
      complex(8), intent(in) :: em(ne)
      complex(8), intent(  out) :: ca1(ne*nvtx)
      complex(8), intent(inout) :: xo(ne,n1*n2*n3)
      integer :: i1,i2,i3,i4,i5,isg
      real(8) :: ee1(4),ee2(4),dsg
      complex(8) :: ffk(4),ggk(4),ggs(4),xxo(ne,4)
      integer, parameter :: ia1(4)=(/1,2,3,4/),nmax=99
      real(8) :: dwgt(4,4,nmax),dvol(nmax)
      do i1=1,n1*n2*n3*ndiv
        dwgt(:,:,1)=0.d0; do i2=1,4; dwgt(i2,i2,1)=1.d0; end do
        dvol(1)=1.d0; i2=1
        ee2=dble(e2(imt1(:,i1))); call ttct(f2,ee2,+1,nmax,i2,dwgt,dvol)
        ee1=dble(e1(imt1(:,i1))); call ttct(f1,ee1,-1,nmax,i2,dwgt,dvol)
        do i3=1,i2
          do i4=1,4
            ggk(i4)=-dcmplx(sum((ee2-ee1)*dwgt(:,i4,i3)),-delt)
          end do
          do isg=1,2
            dsg=dble(3-2*isg)
            xxo=0.d0; ffk=dvol(i3)
            ffk=ffk*dsg; ggs=ggk*dsg
            call calc(dmna,dmnr,ia1,4
     $               ,ffk
     $               ,ggs,ne,em,ca1
     $               ,xxo)
            do i4=1,4; do i5=1,4
              xo(:,imt1(i5,i1))=xo(:,imt1(i5,i1))
     $                          +xxo(:,i4)*dwgt(i5,i4,i3)
            end do; end do
          end do
        end do
      end do
      end subroutine

!--- fork (zen-opt): lr = log((ga+em)/(gb+em)), principal value, from the
!    real functions log and atan2 so that the loop is vectorised with
!    libmvec (AVX2).  log|a/b| = log(|a|^2/|b|^2)/2, arg(a/b) =
!    atan2 of a*conjg(b): same error behaviour as clog of the ratio.
!    All arguments lie in one open half plane (Im = dsgn*delt), so the
!    principal value equals the original clog((ga+em)/(gb+em)).
      subroutine lratio(ne,ga,gb,em,lr)
      integer, intent(in) :: ne
      complex(8), intent(in) :: ga,gb,em(ne)
      complex(8), intent(out) :: lr(ne)
      real(8) :: ar,ai,br,bi,xr(ne),xi(ne)
      integer :: i
!$omp simd private(ar,ai,br,bi)
      do i=1,ne
        ar=dble(ga)+dble(em(i))
        ai=dimag(ga)+dimag(em(i))
        br=dble(gb)+dble(em(i))
        bi=dimag(gb)+dimag(em(i))
        xr(i)=.5d0*log((ar*ar+ai*ai)/(br*br+bi*bi))
        xi(i)=atan2(ai*br-ar*bi,ar*br+ai*bi)
      end do
      lr=dcmplx(xr,xi)
      end subroutine

!--- fork (zen-opt): case(0) of calc, i.e. the four calls
!      ca1(:,1)=sub2normal(B,A,C,D)  ca1(:,2)=sub2normal(A,B,C,D)
!      ca1(:,3)=sub2normal(A,C,B,D)  ca1(:,4)=sub1normal(A,D,B,C)
!    with A..D = ia4(1..4), fused into one pass over the frequencies:
!    the 6 distinct log ratios are computed once (log(b/a)=-log(a/b), all
!    arguments in one half plane), the frequency-independent denominators
!    are inverted once, and powers are explicit products.
      subroutine calc0_fused(nnn,fk,gk,ia4,ne,em,xo)
      integer, intent(in) :: nnn,ia4(4),ne
      complex(8), intent(in) :: fk(nnn),gk(nnn),em(ne)
      complex(8), intent(inout) :: xo(ne,nnn)
      complex(8) :: a,b,c,d
      complex(8) :: lab(ne),lac(ne),lad(ne),lbc(ne),lbd(ne),lcd(ne)
      complex(8) :: k11,k12,k13,k14,k21,k22,k23,k24
      complex(8) :: k31,k32,k33,k34,m1,m2,m3
      complex(8) :: za,zb,zc,zd,za2,zb2,zc2,zd2,x1,x2,x3,x4,t1
      integer :: i
      a=gk(ia4(1)); b=gk(ia4(2)); c=gk(ia4(3)); d=gk(ia4(4))
      call lratio(ne,a,b,em,lab)
      call lratio(ne,a,c,em,lac)
      call lratio(ne,a,d,em,lad)
      call lratio(ne,b,c,em,lbc)
      call lratio(ne,b,d,em,lbd)
      call lratio(ne,c,d,em,lcd)
!     sub2normal(g0,g1,g2,g3): -z1^2/(6(g0-g1)(g1-g2)(g1-g3))
!       -z2^3 L21/(6(g0-g2)(g1-g2)^2(g2-g3)) +z3^3 L31/(6(g0-g3)(g1-g3)^2(g2-g3))
!       +z0^3 L01/(6(g0-g1)^2(g0-g2)(g0-g3))
!     term 1: (g0,g1,g2,g3)=(b,a,c,d)
      k11=1.d0/(6.d0*(b-a)*(a-c)*(a-d))
      k12=1.d0/(6.d0*(b-c)*(a-c)**2*(c-d))
      k13=1.d0/(6.d0*(b-d)*(a-d)**2*(c-d))
      k14=1.d0/(6.d0*(b-a)**2*(b-c)*(b-d))
!     term 2: (a,b,c,d)
      k21=1.d0/(6.d0*(a-b)*(b-c)*(b-d))
      k22=1.d0/(6.d0*(a-c)*(b-c)**2*(c-d))
      k23=1.d0/(6.d0*(a-d)*(b-d)**2*(c-d))
      k24=1.d0/(6.d0*(a-b)**2*(a-c)*(a-d))
!     term 3: (a,c,b,d)
      k31=1.d0/(6.d0*(a-c)*(c-b)*(c-d))
      k32=1.d0/(6.d0*(a-b)*(c-b)**2*(b-d))
      k33=1.d0/(6.d0*(a-d)*(c-d)**2*(b-d))
      k34=1.d0/(6.d0*(a-c)**2*(a-b)*(a-d))
!     sub1normal(g0,g1,g2,g3)=(a,d,b,c):
!       0.5*[-z1^2 L10/((g0-g2)(g0-g3)(g0-g1))
!            -(z1^2 L10-z2^2 L20)/((g0-g2)(g1-g2)(g2-g3))
!            +(z1^2 L10-z3^2 L30)/((g1-g3)(g0-g3)(g2-g3))]
      m1=1.d0/((a-b)*(a-c)*(a-d))
      m2=1.d0/((a-b)*(d-b)*(b-c))
      m3=1.d0/((d-c)*(a-c)*(b-c))
!$omp simd private(za,zb,zc,zd,za2,zb2,zc2,zd2,x1,x2,x3,x4,t1)
      do i=1,ne
        za=a+em(i); zb=b+em(i); zc=c+em(i); zd=d+em(i)
        za2=za*za; zb2=zb*zb; zc2=zc*zc; zd2=zd*zd
!       term 1 (b,a,c,d): L(c,a)=-lac, L(d,a)=-lad, L(b,a)=-lab
        x1=-za2*k11+zc2*zc*lac(i)*k12-zd2*zd*lad(i)*k13
     $     -zb2*zb*lab(i)*k14
!       term 2 (a,b,c,d): L(c,b)=-lbc, L(d,b)=-lbd, L(a,b)=lab
        x2=-zb2*k21+zc2*zc*lbc(i)*k22-zd2*zd*lbd(i)*k23
     $     +za2*za*lab(i)*k24
!       term 3 (a,c,b,d): L(b,c)=lbc, L(d,c)=-lcd, L(a,c)=lac
        x3=-zc2*k31-zb2*zb*lbc(i)*k32-zd2*zd*lcd(i)*k33
     $     +za2*za*lac(i)*k34
!       term 4 (a,d,b,c): L(d,a)=-lad, L(b,a)=-lab, L(c,a)=-lac
        t1=-zd2*lad(i)
        x4=(-t1*m1-(t1+zb2*lab(i))*m2+(t1+zc2*lac(i))*m3)*.5d0
        xo(i,ia4(1))=xo(i,ia4(1))+fk(ia4(1))*x1
        xo(i,ia4(2))=xo(i,ia4(2))+fk(ia4(2))*x2
        xo(i,ia4(3))=xo(i,ia4(3))+fk(ia4(3))*x3
        xo(i,ia4(4))=xo(i,ia4(4))+fk(ia4(4))*(x4-x3-x2-x1)
      end do
      end subroutine

      end module
