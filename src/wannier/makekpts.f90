subroutine makekpts(Ndiv,Nblk_max,NNblk,Nblk,NSK_BAND_DISP,SKI,SK)
implicit none 
integer::NNblk,Nblk_max,Nblk(NNblk),Ndiv,NSK_BAND_DISP
real(8)::SKI(3,Nblk_max,NNblk)
real(8)::SK(3,NSK_BAND_DISP)
integer::i,k,ipath,ik
ik = 0
do ipath = 1, NNblk
 do k=1,Nblk(ipath)-1
  do i=1,Ndiv
   ik = ik + 1
   SK(:,ik)=SKI(:,k,ipath)+(SKI(:,k+1,ipath)-SKI(:,k,ipath))/dble(Ndiv)*dble(i-1)
  enddo
 enddo
 ik = ik + 1
 SK(:,ik)=SKI(:,Nblk(ipath),ipath)
enddo
!write(6,*) NSK_BAND_DISP
!do i=1,Ndiv*(Nblk-1)+1
!write(6,*) SK(:,i)
!enddo 
return 
end 
