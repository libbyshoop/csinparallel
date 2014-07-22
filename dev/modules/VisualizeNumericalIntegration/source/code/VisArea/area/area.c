/* DESCRIPTION: Parallel code for approximating the area under a curve using a 
*                  left Riemann sum. This version of this code is designed 
*		   specifically for visualizing various methods of data decomposition.
*		   It is not intended to be used for timing. 
* AUTHOR:      Aaron Weeden, Shodor Education Foundation, Inc.
* MODIFICATIONS AUTHOR:	Ivana Marincic, Macalester College '15
* DATE:        September 2011
*              Revised December 2011
* MODIFICATION DATE:	July 2013
*
* EXAMPLE USAGE:
*        To run a program (e.g. serial) with the default domain width, default function
*	  and default number of rectangles, use:    ./area.serial
*        To run a program (e.g. openmp) with the default domain width and 
*          100000 rectangles, use:    ./area.openmp -n 100000
*        To run a program (e.g. mpi) with a domain from 100.0 to 200.0 and the
*          default number of rectangles, use:   ./area.mpi -l 100.0 -r 200.0
*        To run a program (e.g. serial) with the default left x-boundary of the
*          domain and 500.0 as the right x-boundary of the domain with the
*          default number of rectangles, use:  ./area.serial -r 500.0
*	To run a program with other functions with the default domain width and default
*	number of rectangles, use: ./area.serial -f 1 (or 2 or 3)
*/

/***********************
* Libraries to import *
***********************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "area.h"
#include "structs.h"

/******************************************
* MAIN function -- execution starts here *
******************************************/
int main(int argc, char** argv)
{
  
  struct area_t area;
  struct current_rec_t current_rect;
  init_data(&area, &current_rect);  
  area_under_curve(&area, &current_rect, argc,argv);
  finish(&area);
  /* The code has finished executing successfully. */
  return 0;
}