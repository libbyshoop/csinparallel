/* structs.h
*	holds all structs and the mathematical functions as well as some constants
* AUTHOR:	Ivana Marincic, Macalester College '15
* DATE:		July 2013
*/

#ifndef STRUCTS_H
#define STRUCTS_H

#include <stdio.h>

#if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
#ifdef _MPI
#include <mpe.h>
#include <mpe_graphics.h>
#endif
#endif

#ifndef NO_X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#endif

#include <math.h>

#define DEFAULT_WIDTH 300
#define EXTRA_PIXELS 5 //extra width to the right edge of the X window

#define UNIT 1 	//change this to use a different radius for the ACTUAL circle
#define FREQ 1 	//change this to use a different frequency for ACTUAL sin(x)
#define AMP 1	//change this to use a different amplitude for ACTUAL sin(x)

#define RADIUS 600 //radius of DRAWN circle
#define AMPLITUDE pi*200 //amplitude of the DRAWN sin function
#define FREQUENCY 0.005 //frequency of the DRAWN sin function

// All the data needed for an X display
struct display_t {
  int height;
  int width;
  #ifndef NO_X11
  Window    w;
  GC        gc; //graphics context
  Display * dpy; 
  Pixmap    buffer;
  Colormap  theColormap;
  XColor green_col, red_col, blue_col, yellow_col, bisque_col,
  violet_col, cyan_col, olive_col, gray_col, pink_col, lavender_col,
  orange_col, brown_col;
  /*extra colors*/
  XColor aquam_col, coral_col,dgreen_col,crimson_col,seagreen_col,plum_col,
  navajo_col,teal_col,steelblue_col,saddle_col,lightpink_col,skyblue_col,medblue_col;
  XEvent e;
  int posx; //position x
  int posy; //position y
  #endif
  #if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
  #if _MPI
  char * display;
  MPE_XGraph canvas;
  MPE_Color * colors;
  MPE_Color myColor;
  int w_width;
  #endif
  #endif
};

// Properties of rectangles
struct rec_t {
  int private_num_rect; //number of rectangles private to each process
  int num_rect; 	//total number of rectangles
  double width; 	//rectangle width
  double circle_width; 	//this is for the width of a rectangle for the DRAWN circle
  double sin_width; 	//this is for the width of a rectangle for the DRAWN sin function
};

// Properties of a particular rectangle
typedef struct current_rec_t {
  double x;
  double h;
  int color_index;
} current_rec_t;

// Properties of the curve to be drawn
struct curve_t {
  double my_xleft;	//left x coordinate private to each process
  double xleft;		//overall left x coordinate of the domain 
  double xright;	//overall right x coordinate of the domain
  double width;		//width of the domain
  double circle_width;	 //rescaled witdth under the curve for the DRAWN circle
  double circle_my_xleft;//x left coordinate for each process rescaled for the DRAWN circle
  double sin_width; 	 //rescaled witdth under the curve for the DRAWN sin function
  double sin_my_xleft;	 //x left coordinate for each process rescaled for the DRAWN sin function
};

// Main struct that each process uses
struct area_t {
  int rank;		//process rank
  int num_threads;	//number of threads
  int numProcs;		//number of processes
  double my_sum;	//sum private to each process
  double total_sum;
  int linear;		//if 0, don't use linear function
  int circle;		//if 0, don't use unit circle
  int sin;		//if 0, don't use sin(x)
  int do_display;	//if 0, don't use display
  int gap;		//if 0, do not separate the X windows
  double time;		//throttle time
  struct current_rec_t* recs_t;
  #ifdef H_CHUNKS_OF_ONE
  int group_size;
  #endif
  
  struct curve_t curve;
  struct rec_t rect;
  struct display_t d;
};

const double pi = 3.141592653589793238462643383079;

/***************************************************************************
 * Mathematical functions of curves whose areas we wish to approximate *
 ***************************************************************************/
double FUNC(struct area_t * area, double x)
{
  if(area->linear) return x;
  else if(area->circle) return sqrt(UNIT*UNIT-x*x);
  else if(area->sin) return AMP*sin(FREQ*x);
  else return area->curve.xright+x*sin(0.05*x);
}

/***************************************************************************
 * Scaling function of a bigger equivalent circle to be drawn*
 ***************************************************************************/
double scale_circle(double x){
  return sqrt(RADIUS * RADIUS - x * x);
}

/***************************************************************************
 * Scaling function of a bigger equivalent sin function to be drawn*
 ***************************************************************************/
double scale_sin(double x){
  return AMPLITUDE*sin(FREQUENCY*x);
}

#endif