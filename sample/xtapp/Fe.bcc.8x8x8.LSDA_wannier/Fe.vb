# file map data
 &filemap
 basename = 'Fe',
 number_PP_file = 1,
 /
 ps-Fe ps-Fe.ichr  

# symmetry data
 &SYMMETRY
 SYMMETRY_FORMAT = 'reciprocal',
 NUMBER_SYM_OP = 48, 
 denom_trans = 1,
 /
  1  0  0  0  1  0  0  0  1  0  0  0
 -1  0  0  0 -1  0  0  0 -1  0  0  0
  1  0 -1  1  0  0  1 -1  0  0  0  0
 -1  0  1 -1  0  0 -1  1  0  0  0  0
  0  1 -1  1  0 -1  0  0 -1  0  0  0
  0 -1  1 -1  0  1  0  0  1  0  0  0
  0  1  0  0  1 -1 -1  1  0  0  0  0
  0 -1  0  0 -1  1  1 -1  0  0  0  0
 -1  0  0 -1  0  1 -1  1  0  0  0  0
  1  0  0  1  0 -1  1 -1  0  0  0  0
  0 -1  0 -1  0  0  0  0 -1  0  0  0
  0  1  0  1  0  0  0  0  1  0  0  0
  0 -1  1  0 -1  0  1 -1  0  0  0  0
  0  1 -1  0  1  0 -1  1  0  0  0  0
 -1  0  1  0 -1  1  0  0  1  0  0  0
  1  0 -1  0  1 -1  0  0 -1  0  0  0
  0  1  0  0  0  1  1  0  0  0  0  0
  0 -1  0  0  0 -1 -1  0  0  0  0  0
 -1  1  0  0  1  0  0  1 -1  0  0  0
  1 -1  0  0 -1  0  0 -1  1  0  0  0
 -1  0  1 -1  1  0 -1  0  0  0  0  0
  1  0 -1  1 -1  0  1  0  0  0  0  0
  0  0  1 -1  0  1  0 -1  1  0  0  0
  0  0 -1  1  0 -1  0  1 -1  0  0  0
  0 -1  0  1 -1  0  0 -1  1  0  0  0
  0  1  0 -1  1  0  0  1 -1  0  0  0
  0  0 -1  0 -1  0 -1  0  0  0  0  0
  0  0  1  0  1  0  1  0  0  0  0  0
  1  0 -1  0  0 -1  0  1 -1  0  0  0
 -1  0  1  0  0  1  0 -1  1  0  0  0
  1 -1  0  1  0 -1  1  0  0  0  0  0
 -1  1  0 -1  0  1 -1  0  0  0  0  0
  0  0  1  1  0  0  0  1  0  0  0  0
  0  0 -1 -1  0  0  0 -1  0  0  0  0
  0 -1  1  0  0  1 -1  0  1  0  0  0
  0  1 -1  0  0 -1  1  0 -1  0  0  0
  1 -1  0  0 -1  1  0 -1  0  0  0  0
 -1  1  0  0  1 -1  0  1  0  0  0  0
  1  0  0  1 -1  0  1  0 -1  0  0  0
 -1  0  0 -1  1  0 -1  0  1  0  0  0
  0  0 -1  0  1 -1  1  0 -1  0  0  0
  0  0  1  0 -1  1 -1  0  1  0  0  0
 -1  0  0  0  0 -1  0 -1  0  0  0  0
  1  0  0  0  0  1  0  1  0  0  0  0
 -1  1  0 -1  0  0 -1  0  1  0  0  0
  1 -1  0  1  0  0  1  0 -1  0  0  0
  0  1 -1 -1  1  0  0  1  0  0  0  0
  0 -1  1  1 -1  0  0 -1  0  0  0  0

# atom data
 8 26 
 1  0.000  0.000  0.000  

# k-points data
&smpl_kpt
dos_mode = 'METHFESSEL_PAXTON',
bz_mesh = 16, 
bz_number_tile = 1,
/ 
  8  8  8  
  2  2  2  

# main data
&tappinput
lattice_factor = 5.4169 
LATTICE_LIST = 0.5, 0.5,-0.5,-0.5, 0.5, 0.5, 0.5,-0.5, 0.5,
cutoff_wave_function = 10.0,
number_element = 1,
number_atom = 1,
number_band = 50,
number_spin = 2, 
store_wfn = 1,
!STORE_VXC = 1, 
control_uptime = 3600,
SCF_CONVERGE = 1.0E-015,
xc_type = 'PBE',
elec_kbt=0.01,
spin_mode = 2, ! 1: LSDA mode, 2: Free relax
!diff_spin = 1.0,!0.0 
initial_lpt=1, 
initial_wfn=1, 
/

# struct_opt data
&struct_opt
number_cycle = 0,
/

# str_opt_constr data
 1
 0
 
# trace band data
 &trace_band
 number_band = 50,
 number_trace_block = 5,   
 /
   G       H       N       P       G       N    # nkpt(nbk+1)
 0.00D0  0.50D0  0.00D0  0.25D0  0.00D0  0.00d0 # ak(1,nbk+1)
 0.00D0 -0.50D0  0.00D0  0.25D0  0.00D0  0.00d0 # ak(2,nbk+1)
 0.00D0  0.50D0  0.50D0  0.25D0  0.00D0  0.50d0 # ak(3,nbk+1)
       20      20      20      20      20  
