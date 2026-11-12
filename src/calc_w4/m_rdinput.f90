module m_rdinput 
implicit none
private
public::read_input 
!--
integer,public::calc_ifreq!    
integer,public::ix_intJ_min!
integer,public::ix_intJ_max!
integer,public::iy_intJ_min!
integer,public::iy_intJ_max!
integer,public::iz_intJ_min!
integer,public::iz_intJ_max!
!real(8),public::wcut_mvmc!Cutoff of U' for mvmc 
!real(8),public::jcut_mvmc!Cutoff of J  for mvmc 
namelist/param_calc_int/calc_ifreq,ix_intJ_min,ix_intJ_max,iy_intJ_min,iy_intJ_max,iz_intJ_min,iz_intJ_max !,wcut_mvmc,jcut_mvmc   
contains
subroutine read_input 
!open(999,file='input.in')
!read(999,nml=param_calc_int)
read(5,nml=param_calc_int)
!default
!if(wcut_mvmc==0.0d0) wcut_mvmc=2.0d0!unit:eV
!if(jcut_mvmc==0.0d0) jcut_mvmc=0.3d0!unit:eV
if(calc_ifreq==0) calc_ifreq=1
if(ix_intJ_min==0) ix_intJ_min=0
if(ix_intJ_max==0) ix_intJ_max=0
if(iy_intJ_min==0) iy_intJ_min=0
if(iy_intJ_max==0) iy_intJ_max=0
if(iz_intJ_min==0) iz_intJ_min=0
if(iz_intJ_max==0) iz_intJ_max=0
write(6,param_calc_int) 
end subroutine
end module
