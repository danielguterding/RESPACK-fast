module m_wrt_model  
  implicit none
contains
subroutine wrt_model_hr(Na1,Na2,Na3,n_occ,HR,WR,spin) 
  implicit none 
  integer,intent(in)::spin!20240801 KN  
  character(99)::filename!20240801 KN  
  integer::Na1,Na2,Na3,n_occ 
  integer::N_element 
  integer::ia1,ia2,ia3,ib,jb,i 
  integer,allocatable::unit_vec(:)!unit_vec(N_element) 
  complex(8)::HR(n_occ,n_occ,-Na1:Na1,-Na2:Na2,-Na3:Na3)
  real(8)::WR(-Na1:Na1,-Na2:Na2,-Na3:Na3)
  real(8),parameter::au=27.21151d0
  !
  N_element=(2*Na1+1)*(2*Na2+1)*(2*Na3+1)  
  !
  !OPEN(300,W,FILE='zvo_hr.dat') 
  !
  !20240801 KN
  !
  if(spin==0) filename='./dir-model/zvo_hr.dat'
  if(spin==1) filename='./dir-model.up/zvo_hr.dat'
  if(spin==2) filename='./dir-model.dn/zvo_hr.dat'
  ! 
  OPEN(300,FILE=filename) 
  write(300,'(a)')'wannier90 format for vmcdry.out or HPhi -sdry'
  write(300,'(i10)') n_occ
  write(300,'(i10)') N_element
  !
  allocate(unit_vec(N_element)); unit_vec=1
  write(300,'(15i5)')(unit_vec(i),i=1,N_element) 
  deallocate(unit_vec) 
  !
  do ia1=-Na1,Na1
   do ia2=-Na2,Na2
    do ia3=-Na3,Na3 
     do ib=1,n_occ
      do jb=1,n_occ
       write(300,'(i5,i5,i5,i5,i5,2f15.10)') ia1,ia2,ia3,ib,jb,HR(ib,jb,ia1,ia2,ia3)*WR(ia1,ia2,ia3)*au   
      enddo!jb
     enddo!ib
    enddo!ia3
   enddo!ia2
  enddo!ia1 
  return
end subroutine wrt_model_hr 
!
subroutine wrt_model_wcenter(n_occ,a1,a2,a3,wcenter,spin)
  implicit none 
  integer,intent(in)::spin!20240801 KN 
  character(99)::filename!20240801 KN  
  integer::n_occ 
  real(8)::a1(3),a2(3),a3(3),wcenter(3,n_occ) 
  real(8)::ainv(3,3)
  integer::ib,i,j
  real(8)::wcenter_lat(3) 
  real(8)::SUM_REAL
  real(8),parameter::bohr=0.529177249d0 
  integer,parameter::unity=1 
  !
  ainv(:,1)=a1(:)
  ainv(:,2)=a2(:)
  ainv(:,3)=a3(:)
  !
  call inv_amat(3,ainv(1,1)) 
  !
  !OPEN(303,W,FILE='zvo_geom.dat') 
  !
  !20240801 KN
  !
  if(spin==0) filename='./dir-model/zvo_geom.dat'
  if(spin==1) filename='./dir-model.up/zvo_geom.dat'
  if(spin==2) filename='./dir-model.dn/zvo_geom.dat'
  !
  OPEN(303,FILE=filename) 
  write(303,'(3f15.10)')(a1(i)*bohr,i=1,3) 
  write(303,'(3f15.10)')(a2(i)*bohr,i=1,3) 
  write(303,'(3f15.10)')(a3(i)*bohr,i=1,3) 
  write(303,'(i10)') n_occ
  do ib=1,n_occ 
   wcenter_lat=0.0d0 
   do i=1,3
    SUM_REAL=0.0d0 
    do j=1,3
     SUM_REAL=SUM_REAL+ainv(i,j)*wcenter(j,ib) 
    enddo!j 
    wcenter_lat(i)=SUM_REAL
   enddo!i
   write(303,*)(wcenter_lat(i),i=1,3) 
  enddo 
  !
  !OPEN(307,W,FILE='zvo_geom.xsf') 
  !
  !20240801 KN
  !
  if(spin==0) filename='./dir-model/zvo_geom.xsf'
  if(spin==1) filename='./dir-model.up/zvo_geom.xsf'
  if(spin==2) filename='./dir-model.dn/zvo_geom.xsf'
  !
  OPEN(307,FILE=filename) 
  write(307,'(a10)')'CRYSTAL'
  write(307,'(a10)')'PRIMVEC'
  write(307,'(3f15.10)')(a1(i)*bohr,i=1,3)  
  write(307,'(3f15.10)')(a2(i)*bohr,i=1,3)  
  write(307,'(3f15.10)')(a3(i)*bohr,i=1,3)  
  write(307,'(a10)')'PRIMCOORD'
  write(307,'(2i10)') n_occ,unity 
  do ib=1,n_occ
   write(307,'(a5,3f15.10)')' H ',(wcenter(j,ib)*bohr,j=1,3)
  enddo 
  !
  return
end subroutine wrt_model_wcenter 
!
subroutine wrt_model_SK_BAND_DISP(Ndiv,NN_sym_points,N_sym_points,NSK_BAND_DISP,SK_BAND_DISP,spin)
  implicit none 
  integer,intent(in)::Ndiv,NN_sym_points,N_sym_points(NN_sym_points),NSK_BAND_DISP
  real(8),intent(in)::SK_BAND_DISP(3,NSK_BAND_DISP)
  integer,intent(in)::spin!20240801 KN 
  character(99)::filename!20240801 KN  
  integer::ik,i
  !
  !OPEN(304,W,FILE='zvo_bandkpts.dat') 
  !
  !20240801 KN
  !
  if(spin==0) filename='./dir-model/zvo_bandkpts.dat'
  if(spin==1) filename='./dir-model.up/zvo_bandkpts.dat'
  if(spin==2) filename='./dir-model.dn/zvo_bandkpts.dat'
  !
  OPEN(304,FILE=filename) 
  write(304,'(50i10)') NSK_BAND_DISP,Ndiv,N_sym_points(1:NN_sym_points)
  do ik=1,NSK_BAND_DISP
   write(304,*)(SK_BAND_DISP(i,ik),i=1,3) 
  enddo 
  return
end subroutine wrt_model_SK_BAND_DISP 
!
subroutine wrt_model_SK0(NTK,SK0,spin)
  implicit none 
  integer,intent(in)::NTK 
  real(8),intent(in)::SK0(3,NTK) 
  integer,intent(in)::spin!20240801 KN 
  character(99)::filename!20240801 KN  
  integer::ik,i 
  !
  !OPEN(305,W,FILE='zvo_mkkpts.dat') 
  !
  !20240801 KN
  !
  if(spin==0) filename='./dir-model/zvo_mkkpts.dat'
  if(spin==1) filename='./dir-model.up/zvo_mkkpts.dat'
  if(spin==2) filename='./dir-model.dn/zvo_mkkpts.dat'
  !
  OPEN(305,FILE=filename) 
  write(305,'(i10)') NTK 
  do ik=1,NTK 
   write(305,*)(SK0(i,ik),i=1,3) 
  enddo 
  return
end subroutine wrt_model_SK0 
!
subroutine wrt_model_ef(FermiEnergy,spin) 
  implicit none 
  real(8),intent(in)::FermiEnergy 
  integer,intent(in)::spin!20240801 KN 
  character(99)::filename!20240801 KN  
  !
  !OPEN(306,W,FILE='zvo_ef.dat') 
  !
  !20240801 KN
  !
  if(spin==0) filename='./dir-model/zvo_ef.dat'
  if(spin==1) filename='./dir-model.up/zvo_ef.dat'
  if(spin==2) filename='./dir-model.dn/zvo_ef.dat'
  !
  OPEN(306,FILE=filename) 
  write(306,'(f15.10)') FermiEnergy 
  return 
end subroutine wrt_model_ef 
!
subroutine inv_amat(nm,mat)
  implicit none 
  integer,intent(in)::nm
  real(8),intent(inout)::mat(nm,nm)
  integer::ipiv(nm)
  integer::Lwork 
  real(8),allocatable::work(:)
  integer::info 
  Lwork=10*nm
  allocate(work(Lwork))
  info=0
  call dgetrf(nm,nm,mat,nm,ipiv,info)
  call dgetri(nm,mat,nm,ipiv,work,Lwork,info)
  if(info/=0) then
  write(6,*) 'info (subrouitine inv):',info
  stop
  endif 
  deallocate(work)
  return 
end subroutine inv_amat 
!
end module m_wrt_model 
