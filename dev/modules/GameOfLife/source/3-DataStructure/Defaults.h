/*
  * $Id: Defaults.h,v 1.2 2012/06/27 15:41:34 ibabic09 Exp $
  * This file is part of BCCD, an open-source live CD for computational science
  * education.
  * 
  * Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave 
  *   Joiner, Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden 

  * 
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  * 
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  * 
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*******************************************
MPI Life 1.0
Copyright 2002, David Joiner and
  The Shodor Education Foundation, Inc.
Updated 2010, Andrew Fitz Gibbon and
  The Shodor Education Foundation, Inc.
*******************************************/

#ifndef BCCD_LIFE_DEFAULTS_H
#define BCCD_LIFE_DEFAULTS_H

#include <stdbool.h>
#include <getopt.h>

static const char * opts = "c:r:g:i:o:s:t::xh?";
static const struct option long_opts[] = {
    { "columns", required_argument, NULL, 'c' },
    { "rows", required_argument, NULL, 'r' },
    { "gens", required_argument, NULL, 'g' },
    { "no-display", no_argument, NULL, 0 },
    { "display", no_argument, NULL, 'x' },
    { "output", required_argument, NULL, 'o' },
    { "input", required_argument, NULL, 'i' },
    { "output-stats", required_argument, NULL, 's' },
    { "throttle", optional_argument, NULL, 't' },
    { "help", no_argument, NULL, 'h' },
    { NULL, no_argument, NULL, 0 }
};

// Default parameters for the simulation
const int     DEFAULT_THROTTLE = 20000;
const int     DEFAULT_SIZE = 105;
const int     DEFAULT_GENS = 200;
const double  INIT_PROB = 0.25;

#ifndef NO_X11
    const bool    DEFAULT_DISP = true;
#else
    const bool    DEFAULT_DISP = false;
#endif

// Size, in pixels, of the X window(s)
//const int  DEFAULT_WIDTH = 500;
//const int DEFAULT_HEIGHT = 500;
const int DEFAULT_WIDTH = 1020;
const int DEFAULT_HEIGHT = 720;

// Number of possible shades of gray
#define NUM_GRAYSCALE 10

// All the data needed for an X display
struct display_t 
{
    #ifndef NO_X11
    Window    w;
    GC        gc;
    Display * dpy;
    Pixmap    buffer;
    Colormap  theColormap;
    XColor    Xgrayscale[NUM_GRAYSCALE];

    int deadColor;
    int liveColor;
    int width;
    int height;
    #endif
};

// All the data needed by an instance of Life
struct life_t 
{
    int  rank;
    int  size;
    int  throttle;
    int  ncols;
    int  nrows;
    int  * grid;
    int  * next_grid;
    bool do_display;
    int  generations;
    char * infile;
    char * outfile;
    
    char * statsfile; //new; used in write_stats
    int offset;       //new; used in moveWindow
    int remainder;    //new; used in setupWindow

    struct display_t disp;
};

enum CELL_STATES 
{
    DEAD = 0,
    ALIVE
};

// All the data needed for CUDA operation: CUDA needs memory 
// pointers and other information on CPU side. As more than
// one function (mainly used by CUDA.cu) need to use these 
// data, we decided to use a struct to hold all these data.
struct cuda_t 
{
    #if defined(__CUDACC__) || defined(MPICUDA)
        int *grid_dev;
        int *next_grid_dev;
        #ifdef CUDA_STAT
            float elapsedTime;  // timing
        #endif
    #endif
};

// Cells become DEAD with more than UPPER_THRESH 
// or fewer than LOWER_THRESH neighbors
const int UPPER_THRESH = 3;
const int LOWER_THRESH = 2;

// Cells with exactly SPAWN_THRESH neighbors become ALIVE
const int SPAWN_THRESH = 3;

#endif
