# $\Id$ 
#
# This file is part of BCCD, an open-source live CD for computational science
# education.
# 
# Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave Joiner, 
#   Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

###############################################################################
# Makefile for the modules included in ~bccd on the BCCD (http://bccd.net)
# 
# Add NO_X11=1 to the make command line to omit X libraries.
# Add DEBUG=1 to add -ggdb to CFLAGS
# Add PROFILE=1 to add -pg to CFLAGS and LDFLAGS
# Add STAT_KIT=1 to enable StatKit
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

ifdef STAT_KIT
	EXTRA_COBJS += ../StatKit/petakit/pkit-c.o
	EXTRA_CXXOBJS += ../StatKit/petakit/pkit-cxx.o
	CFLAGS += -DSTAT_KIT
endif

########for Area-under-curve visualization###############
ifdef STRIPING
	NO_X11 = 1
	CFLAGS += -DSTRIPING
	LIBS += -lmpe -lX11
	LDFLAGS += -L/usr/X11R6/lib
	INCFLAGS += -I/bccd/software/openmpi/1.4.5/Linux/BCCD/x86_64/include
endif
#########################################################

ifdef NO_X11
	CFLAGS += -DNO_X11
else
	LIBS += -lX11
	LDFLAGS += -L/usr/X11R6/lib
endif

ifdef DEBUG
	CFLAGS += -ggdb
endif

ifdef PROFILE
	CFLAGS += -pg
	LDFLAGS += -pg
endif

# For Modules using Atlas 
ifeq ($(ATLAS), 1)
	ifeq ($(wildcard /etc/bccd-revision),)   # not the BCCD
		CFLAGS += -I/mounts/al-salam/software/atlas-3.9.24/include
		LDFLAGS += -L/mounts/al-salam/software/atlas-3.9.24/lib 
		LIBS += -latlas -lcblas 
	else                                     # the BCCD
		# this is a placeholder until we build ATLAS into a module
		# CFLAGS += 
		# LDFLAGS += 
		# LIBS += 
	endif
endif

#For Modules using FFTW (version 3.3)
ifeq ($(FFTW), 1)
	ifeq ($(wildcard /etc/bccd-revision),)   # not the BCCD
		CFLAGS += -I/mounts/al-salam/software/fftw-3.3/include
		LDFLAGS += -L/mounts/al-salam/software/fftw-3.3/lib 
		LIBS += -lfftw3f 
	else                                     # the BCCD
		# this is a placeholder until we build FFTW into a module
		# FLAGS += 
		# LDFLAGS += 
		# LIBS += 
	endif
endif

# For Modules using NetCDF
ifeq ($(NETCDF), 1)
	ifeq ($(wildcard /etc/bccd-revision),) # Not the BCCD
		CFLAGS += -I$(NETCDF_INCLUDE)/ -DHAS_NETCDF
		LDFLAGS += -L$(NETCDF_PATH)/ 
		LIBS += -lnetcdf
	else
		# this is a placeholder until we build NETCDF into a module
		# CFLAGS +=
		# LDFLAGS +=
		# LIBS +=
	endif
endif

# For Modules using Gadget2
ifeq ($(GADGET2), 1)
	ifeq ($(wildcard /etc/bccd-revision),) # Not the BCCD
		CFLAGS += -I/share/apps/hdf5/gnu/include/ -DHAS_HDF5
		LDFLAGS += -L/share/apps/hdf5/gnu/lib/ 
		LIBS += -lhdf5 -lhdf5_hl
	else 
		# This is a placeholder until we build Gadget2 into a module
		# CFLAGS +=
		# LDFLAGS +=
		# LIBS +=
	endif
endif

# For Modules using LIBGD
ifeq	($(LIBGD), 1)
	ifeq	($(wildcard /etc/bccd-revision),) # Not the BCCD
		CFLAGS += -DHAS_LIBGD
		LIBS += -ljpeg -lpng -lz -lgd
	else
		# This is a placeholder until we build LibGD into a module
		# CFLAGS +=
		# LDFLAGS +=
		# LIBS +=
	endif
endif

# For modules using CUBLAS
ifeq ($(CUBLAS), 1)
	LIBS += -lcublas
endif

LIBS += -lm
LDFLAGS += $(LIBS)
	
$(PROGRAM).c-serial:
	$(CC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).c-mpi:
	$(MPICC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(MPIFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).cxx-mpi:
	$(MPICXX) $(CXXSRCS) $(EXTRA_CXXOBJS) $(CFLAGS) $(INCFLAGS) $(MPIFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).c-openmp:
	$(CC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(OMPFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).c-mpi-openmp:
	$(MPICC) $(CSRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(OMPFLAGS) $(MPIFLAGS) $(LDFLAGS) -o $@

$(PROGRAM).c-cuda: 
	$(NVCC) $(CUDASRCS) $(EXTRA_COBJS) $(CFLAGS) $(INCFLAGS) $(NVCCFLAGS) $(LDFLAGS) -o $@

help:
	@echo "This is the driver Makefile for all the modules shipped in ~bccd."
	@echo "The following commands are supported:"
	@echo "  $ make build-all (to build all modules)"
	@echo "  $ make clean-all (to clean all modules)"
	@echo "The following options are supported by most modules:"
	@echo "  NO_X11=1 (do not include an X11 GUI component)"
	@echo "  DEBUG=1 (adds -ggdb to CFLAGS)"
	@echo "  PROFILE=1 (adds -pg to CFLAGS and LDFLAGS)"
	@echo "  STAT_KIT=1 (enables the StatKit performance collector tool)"
	@echo "You can also cd to a module's directory and type make (options optional)."

build-all:
	@echo "-------------------------------------"
	cd CUDA; $(MAKE) 
	@echo "-------------------------------------"
	cd Life; $(MAKE)
	@echo "-------------------------------------"
	cd GalaxSee; $(MAKE)
	@echo "-------------------------------------"
	cd Hello-world; $(MAKE)
	@echo "-------------------------------------"
	cd Parameter-space; $(MAKE)
	@echo "-------------------------------------"
	cd GalaxSee-v2;	$(MAKE)
	@echo "-------------------------------------"
	cd Area-under-curve; $(MAKE)
	@echo "-------------------------------------"
	cd Pandemic; $(MAKE)
	@echo "-------------------------------------"
	cd Sieve; $(MAKE)
	@echo "-------------------------------------"
	cd Tree-sort; $(MAKE)
	@echo "-------------------------------------"

clean-all:
	@echo "-------------------------------------"
	cd CUDA; $(MAKE) clean
	@echo "-------------------------------------"
	cd Life; $(MAKE) clean
	@echo "-------------------------------------"
	cd GalaxSee; $(MAKE) clean
	@echo "-------------------------------------"
	cd Hello-world; $(MAKE) clean
	@echo "-------------------------------------"
	cd Parameter-space; $(MAKE) clean
	@echo "-------------------------------------"
	cd GalaxSee-v2;	$(MAKE)	clean
	@echo "-------------------------------------"
	cd Area-under-curve; $(MAKE) clean
	@echo "-------------------------------------"
	cd Pandemic; $(MAKE) clean
	@echo "-------------------------------------"
	cd Sieve; $(MAKE) clean
	@echo "-------------------------------------"
	cd Tree-sort; $(MAKE) clean
	@echo "-------------------------------------"
	cd StatKit/petakit; $(MAKE) clean
	@echo "-------------------------------------"
	
clean:
	/bin/rm -f $(CLEAN)
