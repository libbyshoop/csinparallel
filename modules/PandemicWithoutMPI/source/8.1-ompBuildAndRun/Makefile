# DESCRIPTION:  Makefile for OpenMP code in C
# AUTHOR:       Yu Zhao, Macalester College
# DATE:         Original for Area Under A Curve module, September, 2011.
#		Modified for Infectious Disease module, November, 2011. by Aaron Weedon
#		Modified for Infectious Disease module, July, 2013, by Yu Zhao

# Code prefix
PROGRAM_PREFIX=Pandemic

# Compilers and flags
CC=gcc
XLIB_LOC=/opt/X11/lib    #Mac OS X XQuartz installed here
#XLIB_LOC=/usr/X11R6/lib   #some unix systems may have this
XLIB_INC=/opt/X11/include    #Mac OS X XQuartz installed here


# OpenMP
OPENMP_FLAGS=-fopenmp

#CFLAGS+=-DTEXT_DISPLAY # Uncomment to show text display

#CFLAGS+=-DX_DISPLAY -I $(XLIB_INC) -L$(XLIB_LOC) -lX11 # Uncomment to show X display

CFLAGS+=-DSHOW_RESULTS # Uncomment to make the program print its results

# Source files
SRCS=$(PROGRAM_PREFIX).c 

# Make targets
all: $(PROGRAM_PREFIX)-openmp

clean: 
	rm -f $(PROGRAM_PREFIX)-openmp

run:
	./$(PROGRAM_PREFIX).c-openmp

# Make rules
$(PROGRAM_PREFIX)-openmp: $(SRCS)
	$(CC) -o $(PROGRAM_PREFIX)-openmp $(SRCS) $(OPENMP_FLAGS) $(CFLAGS)

$(SRCS): Core.h Defaults.h Display.h Finalize.h Infection.h Initialize.h
