###############################################################################
# Makefile for the modules included in ~bccd on the BCCD (http://bccd.net)
# 
# Add NO_X11=1 to the make command line to omit X libraries.
# Add STRIPING=1 to include the mpe libraries
###############################################################################

# Test for Intel compilers, act accordingly
ifeq ($(findstring intel,$(LOADEDMODULES)),intel)
	CC = icc
	OMPFLAGS = -openmp -DOMP
else
	CC = gcc
	OMPFLAGS = -fopenmp -DOMP
endif

MPICC = mpicc
MPICXX = mpicxx
NVCC = nvcc

CFLAGS = -O2
NVCCFLAGS = --compiler-bindir=gcc
NVMPICCFLAGS = --compiler-bindir=mpicc
MPIFLAGS = -D_MPI


ifdef MPI_CHUNKS_OF_ONE
	NO_X11 = 1
	CFLAGS += -DMPI_CHUNKS_OF_ONE
	LIBS += -lmpe -lX11
	LDFLAGS += -L/usr/X11R6/lib
	INCFLAGS += -I/bccd/software/openmpi/1.4.5/Linux/BCCD/x86_64/include
endif

ifdef H_CHUNKS_OF_ONE
	NO_X11 = 1
	CFLAGS += -DH_CHUNKS_OF_ONE
	LIBS += -lmpe -lX11
	LDFLAGS += -L/usr/X11R6/lib
	INCFLAGS += -I/bccd/software/openmpi/1.4.5/Linux/BCCD/x86_64/include
endif

ifdef H_CHUNKS_OF_ONE_STATIC
	NO_X11 = 1
	CFLAGS += -DH_CHUNKS_OF_ONE
	CFLAGS += -DH_CHUNKS_OF_ONE_STATIC
	LIBS += -lmpe -lX11
	LDFLAGS += -L/usr/X11R6/lib
	INCFLAGS += -I/bccd/software/openmpi/1.4.5/Linux/BCCD/x86_64/include
endif
ifdef H_CHUNKS_OF_ONE_DYNAMIC
	NO_X11 = 1
	CFLAGS += -DH_CHUNKS_OF_ONE
	CFLAGS += -DH_CHUNKS_OF_ONE_DYNAMIC
	LIBS += -lmpe -lX11
	LDFLAGS += -L/usr/X11R6/lib
	INCFLAGS += -I/bccd/software/openmpi/1.4.5/Linux/BCCD/x86_64/include
endif

ifdef OMP_CHUNKS_OF_ONE_STATIC
	CFLAGS += -DOMP_CHUNKS_OF_ONE_STATIC
endif

ifdef OMP_CHUNKS_OF_ONE_DYNAMIC
	CFLAGS += -DOMP_CHUNKS_OF_ONE_DYNAMIC
endif

ifdef NO_X11
	CFLAGS += -DNO_X11
else
	LIBS += -lX11
	LDFLAGS += -L/usr/X11R6/lib
endif

LIBS += -lm
LDFLAGS += $(LIBS)
	
$(PROGRAM).c-serial:
	$(CC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -o $@


$(PROGRAM).cxx-mpi:
	$(MPICXX) $(CXXSRCS) $(EXTRA_CXXOBJS) $(CFLAGS) $(INCFLAGS) $(MPIFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).c-openmp:
	$(CC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(OMPFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).c-mpi-openmp:
	$(MPICC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(OMPFLAGS) $(MPIFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).c-mpi:
	$(MPICC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(MPIFLAGS) $(LDFLAGS) -o $@

#$(PROGRAM).c-cuda: 
	#$(NVCC) $(CUDASRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(NVCCFLAGS) $(LDFLAGS) -o $@

help:

	@echo "The following options are supported by most modules:"
	@echo "  NO_X11=1 (do not include an X11 GUI component)"
	@echo "  STAT_KIT=1 (enables the StatKit performance collector tool)"
	@echo "  STRIPING=1 (enables striping in Area-under-curve)"
	@echo "You can also cd to a module's directory and type make (options optional)."


clean-all:
	#@echo "-------------------------------------"
	#cd CUDA; $(MAKE) clean
	#@echo "-------------------------------------"
	#cd Life; $(MAKE) clean
	@echo "-------------------------------------"
	cd Area-under-curve; $(MAKE) clean
	#@echo "-------------------------------------"
	#cd Pandemic; $(MAKE) clean
	#@echo "-------------------------------------"
	#cd StatKit/petakit; $(MAKE) clean
	@echo "-------------------------------------"
	
clean:
	/bin/rm -f $(CLEAN)
