implicit none        
!
!parameters 
!
real(8),parameter::au=27.21151d0
real(8),parameter::bohr=0.529177d0 
real(8),parameter::pi=dacos(-1.0d0)
real(8),parameter::tpi=2.0d0*pi 
complex(8),parameter::ci=(0.0D0,1.0D0) 
complex(8),parameter::tci=(0.0D0,2.0D0) 
!
integer::ik,ix 
integer::file_num 
character(99)::filename 
!
