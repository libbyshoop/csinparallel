# DESCRIPTION:  Makefile for serial codes in C
# AUTHOR:       Yu Zhao, Macalester College
# DATE:         Original for Area Under A Curve module, September, 2011.
#				Modified for Infectious Disease module, November, 2011. by Aaron Weedon
#				Modified for Infectious Disease module, July, 2013, by Yu Zhao

# Code prefix
PROGRAM_PREFIX=Pandemic

# Compilers and flags
MPICC=mpicc

#CFLAGS+=-DTEXT_DISPLAY # Uncomment to show text display
#CFLAGS+=-DX_DISPLAY -L/usr/X11R6/lib -lX11 # Uncomment to show X display
CFLAGS+=-DSHOW_RESULTS # Uncomment to make the program print its results

# Source files
SRCS=$(PROGRAM_PREFIX).c

# Make targets
all: $(PROGRAM_PREFIX)-mpi

clean: 
	rm -f $(PROGRAM_PREFIX)-mpi

run:
	mpirun -machinefile machines -np 6 ./$(PROGRAM_PREFIX)-mpi

# Make rules
$(PROGRAM_PREFIX)-mpi: $(SRCS)
	$(MPICC) -o $(PROGRAM_PREFIX)-mpi $(SRCS) $(CFLAGS)