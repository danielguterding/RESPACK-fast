PROGRAM generate_sample_mk 
  use m_rd_dat_wfn
  include "config.h" 
  !read dir-wfn 
  call rd_dat_symmetry 
  call rd_dat_sample_k 
  call rd_dat_lattice 
  call rd_dat_nkm 
  call rd_dat_kg 
  call rd_dat_eigenvalue  
  call rd_dat_wavefunction 
  !OPEN(100,W,FILE='dat.sample_mk')
  file_num=100  
  write(filename,"('dat.sample_mk')") 
  OPEN(file_num,FILE=filename) 
  REWIND(file_num)
  write(file_num,'(i10)')NTK 
  do ik=1,NTK 
    write(file_num,'(3f15.10)')(SK0(ix,ik),ix=1,3) 
  enddo
  stop
end
