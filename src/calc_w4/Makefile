######################################
# Copyright (c) 2017 Kazuma Nakamura #
###################################### 
#--------------------------------------------------------------------
# for Intel Linux and Intel Fortran Compiler 11.0 and above 
#--------------------------------------------------------------------
FC = ifort  
FFLAGS = -O2 -qopenmp -xHost 

#######
#DEBUG#
#######
#FFLAGS = -qopenmp -debug full -check all -warn all -traceback -g 

LIBBLAS = -lmkl_intel_lp64 -Wl,--start-group -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread 
#--------------------------------------------------------------------
.SUFFIXES:
.SUFFIXES: .o .F90 .F .f90 .c
# default suffix rules. Do not edit this area.
.F.o:
	${FC} ${F90FLAGS} ${FFLAGS} -c $<
.f90.o:
	${FC} ${F90FLAGS} ${FFLAGS} -c $<
.F90.o:
	${FC} ${F90FLAGS} ${FFLAGS} -c $<
#--------------------------------------------------------------------
TARGET = calc_w4 
OBJECTS = m_rdinput.o m_fft3d_20150826.o inv.o est_NTK.o est_nwx2.o calc_intW4_sub.o calc_intW4.o 
MOD_FILES = m_rdinput.mod fft_3d.mod 

${TARGET} : ${OBJECTS} 
	${FC} ${LDFLAGS} -o $@ ${OBJECTS} ${LIBBLAS} 

calc_intW4.o: calc_intW4.F m_rdinput.o m_fft3d_20150826.o inv.o est_NTK.o est_nwx2.o calc_intW4_sub.o 

calc_intW4_sub.o: calc_intW4_sub.F m_fft3d_20150826.o 

clean: 
	${RM} ${TARGET} ${OBJECTS} ${MOD_FILES} *.mod *__genmod.f90* 
#--------------------------------------------------------------------
