#ifndef AREA_H
#define AREA_H

#include <stdio.h>
#include <stdlib.h>

#include "colors.h"
#include "structs.h"
#include "Xarea.h"

#ifdef _MPI
#include <mpi.h>
#endif

#ifdef OMP
#include <omp.h>
#endif

/***************************************************************************
 * Declare all functions *
 ***************************************************************************/
void	init_data(struct area_t * area);
void 	init(struct area_t * area, int argc, char ** argv);
void 	allocate_array(struct area_t * area);
void	setup_right_boundary(struct area_t * area);
void 	setup_rectangles(struct area_t * area);
void	setup_window_height(struct area_t * area);
void	setup_window_width(struct area_t * area);
void	get_rectangle_area(struct area_t * area);
void 	sum_rectangles(struct area_t * area);
void 	cleanup(struct area_t * area);
void 	parse_args(struct area_t * area, int argc, char ** argv);

/***************************************************************************
 * Initialze all defualt data *
 ***************************************************************************/
void init_data(struct area_t * area){
  area->rank 			= 0;
  area->num_threads 		= 1;
  area->numProcs 		= 1;
  area->id 			= 0;
  area->my_sum 			= 0.0;
  area->cardinality 		= 0;
  area->total_sum 		= 0.0;
  area->linear 			= 0;
  area->circle 			= 0;
  area->sin 			= 0;
  area->do_display 		= 1;
  area->gap 			= 1; //separate the X windows by default
  
  area->curve.my_xleft 		= 0.0;
  area->curve.circle_my_xleft 	= 0.0;
  area->curve.sin_my_xleft 	= 0.0;
  area->curve.xleft 		= 0.0;
  area->curve.xright 		= 300;
  area->curve.width 		= 0.0;
  area->curve.circle_width 	= RADIUS;
  area->curve.sin_width 	= (1/FREQUENCY)*pi;
  
  area->rect.private_num_rect 	= 20; //default for serial 
  area->rect.num_rect 		= 20;
  area->rect.current_rectangle 	= 0;
  area->rect.width 		= 0.0;
  area->rect.circle_width 	= 0.0;
  area->rect.sin_width 		= 0.0;
}

/***************************************************************************
 * Initialze the environment *
 ***************************************************************************/
void init(struct area_t * area, int argc, char ** argv){  
  /*initialize the MPI environment*/
  #ifdef _MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &area->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &area->numProcs);
  area->cardinality += area->numProcs;
  #endif
  
  #ifdef OMP
  #pragma omp parallel
  {
    area->num_threads = omp_get_num_threads();
  }
  #endif
  
  parse_args(area,argc,argv); //get optional arguments 
  
  setup_right_boundary(area); //define correct xright value for sin(x) and circle
  setup_rectangles(area); //set all data related to rectangles
  
  /*set up X windows*/
  if(area->do_display) {
    setup_window_height(area); //height same for X11 and MPE
    setup_window_width(area);
    #ifndef NO_X11
    setupWindow(area);
    init_colors(area);
    moveWindow(area);
    #endif
  }
  
  get_rectangle_area(area); //calculates area of each rectangle and draws the rectangles
  
  /*curve is drawn after rectangles for visibility*/
  if(area->do_display){
    draw_curve(area); //draw the function
    if(!area->gap) draw_process_barrier(area);
  }
  
  sum_rectangles(area);//final summation and printing of result 
}


/***************************************************************************
 * Set up all the information processes/threads need to know about rectangles *
 ***************************************************************************/
void setup_rectangles(struct area_t * area) {
    struct curve_t * curve= &(area->curve);
    struct rec_t * rect 	= &(area->rect);
    struct display_t * d	= &(area->d);
    
    /* Calculate the overall width of the domain of the
    *  function and the width of a rectangle.*/
    curve->width = curve->xright - curve->xleft;
    rect->width = curve->width / rect->num_rect;
    
    /* Calculate the number of rectangles for 
    *  which the process is responsible. */
    rect->private_num_rect = rect->num_rect / area->numProcs; 
    
    /* Calculate the left x-boundary of the process. */
    curve->my_xleft = curve->xleft + area->rank*rect->private_num_rect*rect->width;
    
    /*Add extra rectangles to the last process*/
    if(area->rank == area->numProcs - 1){
      rect->private_num_rect += rect->num_rect % area->numProcs;
    }
    
    /***ACTIVITY 0***/
    /* instead of giving the rest of rectangles to the last process, write an
    improved version of this function with a solution in which the number of rectangles
    will be distributed equally*/
}

/***************************************************************************
 * Find boundaries of each rectangle and calculate the area for each*
 ***************************************************************************/
void get_rectangle_area(struct area_t * area) {
  struct display_t * d = &(area->d);
  struct rec_t * rect = &(area->rect);
  struct curve_t * curve = &(area->curve);
  int current_rectangle;
  double current_left,current_height;

  #ifdef OMP
  #pragma omp parallel for private(current_rectangle, current_left, current_height)
  #endif
  for(current_rectangle = 0; current_rectangle < rect->private_num_rect; current_rectangle++)
  {
    /* Calculate the x-value of the left side of the rectangle */
    current_left = curve->my_xleft + current_rectangle *rect->width;
    
    /* Calculate the height of the rectangle */
    current_height = FUNC(area, current_left);
    
    /* Calculate the area of the rectangle and add it to the sum private to each process*/
    area->my_sum += rect->width * current_height;
    
    /*Draw the current rectangle*/
    if(area->do_display){
      if(area->circle){
	current_left = curve->circle_my_xleft + current_rectangle*rect->circle_width; 
	current_height = scale_circle(current_left);
      }
      if(area->sin){
	current_left = curve->sin_my_xleft + current_rectangle*rect->sin_width;
	current_height = scale_sin(current_left);
      }
      draw_rectangle(area,current_left,current_height);
    }
  }
}

/***************************************************************************
 * Sum the areas of all rectangles*
 ***************************************************************************/
void sum_rectangles(struct area_t * area) {
  struct rec_t * rect = &(area->rect);
 
  /* Calculate the overall sum */
  #ifdef _MPI
  MPI_Reduce(&area->my_sum, &area->total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  #else
  area->total_sum = area->my_sum;
  #endif
  
  /* Print the total sum */
  if (area->rank == 0){
    if(area->linear){
      printf("%f\n", area->total_sum);
      printf("Expected: %f\n",0.5 * area->curve.width*FUNC(area,area->curve.xright));
    }
    else if(area->circle){
      printf("1/4 * %f * %d\n",area->total_sum*4,UNIT*UNIT);
      if(area->curve.xright == UNIT) printf("Expected: 1/4 * pi * radius^2\n");
    }
    else if(area->sin){
      printf("%f\n",area->total_sum);
      if(area->curve.xright == pi*FREQ) printf("Expected: 2.0\n");
    }
    else printf("%f\n", area->total_sum);
  }
}

/***************************************************************************
 * Define the right boundary depending on function specified*
 ***************************************************************************/
void setup_right_boundary(struct area_t * area){
  if(area->circle) { 
    if(area->curve.xright > UNIT) 
      area->curve.xright = UNIT; //default right boundary, unless smaller 
  }
  if(area->sin) {
    if(area->curve.xright > pi) 
      area->curve.xright = pi; //default right boundary, unless smaller
  }
}

/***************************************************************************
 * Cleaning and finishing up  *
 ***************************************************************************/
void finish(struct area_t * area)
{  
  if(area->do_display){
    displayForever(area);
  }
  #ifdef _MPI
  MPI_Finalize();
  #endif
  
}

/***************************************************************************
 * Parse command line arguments *
 ***************************************************************************/
void parse_args(struct area_t * area, int argc, char ** argv){
  int c,num;
  while((c = getopt(argc, argv, "n:l:r:f:x:w:")) != -1)
  {
    switch(c)
    {
      case 'n':
	area->rect.num_rect = atoi(optarg);
	break;
      case 'l':
	area->curve.xleft = atof(optarg);
	break;
      case 'r':
	area->curve.xright = atof(optarg);
	break;
      case 'f':
	num = atoi(optarg);
	if(num==1) {area->linear = 1;}
	else if(num==2) { 
	  area->circle = 1;
	  area->linear = 0; //reset the other functions
	  area->sin = 0;
	}
	else if(num==3) {
	  area->sin = 1;
	  area->circle = 0; //reset the other functions
	  area->linear = 0;
	}
	else {
	  fprintf(stderr, "Usage: [-f 1 for linear|2 for circle|3 for sin]\n");
	  exit(-1);
	}
	break;
      case 'x':
	if(strcmp("no-display",optarg)==0) area->do_display = 0;
	else if(strcmp("display",optarg)==0)area->do_display = 1;
	else{
	  fprintf(stderr, "Usage: [-x display | no-display]\n");
	  exit(-1);
	}
	break;
      case 'w':
	if(strcmp("no-gap",optarg)==0) area->gap = 0;
	else if(strcmp("gap",optarg)==0) area->gap = 1;
	else{
	  fprintf(stderr, "Usage: [-w gap | no-gap] \n");
	  exit(-1);
	}
	break;
      case '?':
      default:
	#ifdef _MPI
	fprintf(stderr, "Usage: mpirun -np NUMBER_OF_PROCESSES %s [-n NUMBER_OF_RECTANGLES] [-l X_LEFT] [-r X_RIGHT] [-f 1 for linear|2 for circle|3 for sin] [-x display | no-display] [-w gap | no-gap].\n", argv[0]);
	#else
	fprintf(stderr, "Usage: %s [-n NUMBER_OF_RECTANGLES] [-l X_LEFT] [-r X_RIGHT] [-f 1 for linear|2 for circle|3 for sin] [-x display | no-display] [-w gap | no-gap]\n", argv[0]);
	#endif
	exit(-1);
    }
  }
  argc -= optind;
  argv += optind;
  
}
#endif