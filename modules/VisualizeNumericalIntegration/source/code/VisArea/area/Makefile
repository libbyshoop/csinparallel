PROGRAM = area
CSRCS = area.c
INCS = area.h Xarea.h structs.h colors.h
CLEAN = $(PROGRAM).c-serial $(PROGRAM).c-mpi $(PROGRAM).c-openmp $(PROGRAM).c-mpi-openmp

ifdef MPI_CHUNKS_OF_ONE
all: c-mpi
else
ifneq ($(H_CHUNKS_OF_ONE)$(H_CHUNKS_OF_ONE_STATIC)$(H_CHUNKS_OF_ONE_DYNAMIC),)
all: c-mpi c-mpi-openmp
else
all: c-serial c-mpi c-openmp c-mpi-openmp
endif
endif

c-serial: $(CSRCS) $(INCS) $(PROGRAM).c-serial
$(PROGRAM).c-serial: $(CSRCS) $(INCS) 

c-mpi: $(CSRCS) $(INCS) $(PROGRAM).c-mpi
$(PROGRAM).c-mpi: $(CSRCS) $(INCS) 

c-openmp: $(CSRCS) $(INCS) $(PROGRAM).c-openmp
$(PROGRAM).c-openmp: $(CSRCS) $(INCS) 

c-mpi-openmp: $(CSRCS) $(INCS) $(PROGRAM).c-mpi-openmp
$(PROGRAM).c-mpi-openmp: $(CSRCS) $(INCS) 

include ../Makefile