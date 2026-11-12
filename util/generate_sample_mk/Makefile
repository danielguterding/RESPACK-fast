######################################
# Copyright (c) 2017 Kazuma Nakamura #
###################################### 
#--------------------------------------------------------------------
# for Intel Linux and Intel Fortran Compiler 11.0 and above 
#--------------------------------------------------------------------
FC = ifort  
FFLAGS = -O2 -qopenmp -xHost -g -traceback
#FFLAGS = -O2 -openmp -xHost -g -traceback

#######
#DEBUG#
#######
#FFLAGS = -qopenmp -debug full -check all -warn all -traceback -g 

LIBBLAS = -lmkl_intel_lp64 -Wl,--start-group -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread -lm 

#GNU
#FC = gfortran 
##FFLAGS = -O2 -fopenmp -g -fbacktrace
#FFLAGS = -fopenmp -debug -check -warn -fbacktrace -g 
#LIBBLAS = -llapack -lblas -fopenmp 
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
TARGET = gen_mk.x   

OBJECTS = generate_sample_mk.o m_rd_dat_wfn.o util.o \

MOD_FILES = m_rd_dat_wfn.mod  

${TARGET} : ${OBJECTS} 
	${FC} ${LDFLAGS} -o $@ ${OBJECTS} ${LIBBLAS} 

generate_sample_mk.o: generate_sample_mk.f90 m_rd_dat_wfn.o util.o 

m_rd_dat_wfn.o: m_rd_dat_wfn.f90 util.o 

clean: 
	${RM} ${TARGET} ${OBJECTS} ${MOD_FILES} *.mod *__genmod.f90* 
#--------------------------------------------------------------------
