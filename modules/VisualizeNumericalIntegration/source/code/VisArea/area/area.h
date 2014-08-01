/*area.h
*	Performs the main computations.
* AUTHOR: Ivana Marincic, Macalester College '15
* DATE:   July, 2013
*/
#ifndef AREA_H
#define AREA_H

#include <stdio.h>
#include <stdlib.h>

#include "colors.h"
#include "structs.h"
#include "Xarea.h"

#if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
#if _MPI
#include "MPEarea.h"
#endif
#endif


#ifdef _MPI
#include <mpi.h>
#endif

#ifdef OMP
#include <omp.h>
#endif

/***************************************************************************
 * Declare all functions *
 ***************************************************************************/
 void init_data(struct area_t * area, struct current_rec_t * current_rect);
 void area_under_curve(struct area_t * area, struct current_rec_t * current_rect, int argc, char ** argv);
 void setup_right_boundary(struct area_t * area);
 void setup_rectangles(struct area_t * area);
 void setup_window_height(struct area_t * area);
 void setup_window_width(struct area_t * area);
 void handle_rectangles(struct area_t * area, struct current_rec_t * current_rect);
 void sum_rectangles(struct area_t * area);
 void cleanup(struct area_t * area);
 void parse_args(struct area_t * area, int argc, char ** argv);


/***************************************************************************
 * Initialze all defualt data *
 ***************************************************************************/
void init_data(struct area_t * area, struct current_rec_t * current_rect){
  /*struct area*/
  area->rank 			= 0;
  area->num_threads 		= 1;
  area->numProcs 		= 1;
  area->my_sum 			= 0.0;
  area->total_sum 		= 0.0;
  area->linear 			= 0;
  area->circle 			= 0;
  area->sin 			= 0;
  area->do_display 		= 1;
  area->gap 			= 1; //separate the X windows by default
  area->time			= 0.1; //0.1 second
  #ifdef H_CHUNKS_OF_ONE
  area->group_size = 0;
  #endif
  /*struct curve*/
  area->curve.my_xleft 		= 0.0;
  area->curve.circle_my_xleft 	= 0.0;
  area->curve.sin_my_xleft 	= 0.0;
  area->curve.xleft 		= 0.0;
  area->curve.xright 		= DEFAULT_WIDTH;
  area->curve.width 		= 0.0;
  area->curve.circle_width 	= RADIUS;
  area->curve.sin_width 	= (1/FREQUENCY)*pi;
  /*struct rect*/
  area->rect.private_num_rect 	= 40; //default for serial 
  area->rect.num_rect 		= 20;
  area->rect.width 		= 0.0;
  area->rect.circle_width 	= 0.0;
  area->rect.sin_width 		= 0.0;
  /*struct current_rect*/
  current_rect->x = 0.0;
  current_rect->h = 0.0;
  current_rect->color_index = 0;
}

/***************************************************************************
 * Initialze the environment and call all other functions*
 ***************************************************************************/
 void area_under_curve(struct area_t * area, struct current_rec_t * current_rect, int argc, char ** argv){  
  /*initialize the MPI environment*/
  #ifdef _MPI
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &area->rank);
  MPI_Comm_size(MPI_COMM_WORLD, &area->numProcs);
  #endif
  
  #ifdef OMP
  #pragma omp parallel
  {
    area->num_threads = omp_get_num_threads();
  }
  #endif
  parse_args(area,argc,argv); //get optional arguments 
  
  #ifdef H_CHUNKS_OF_ONE
  if(area->group_size == 0)
  {
    printf("ERROR: For this hybrid version you must specify group size by setting the option '-g'.\n");
    exit(-1);
  }
  #endif
  
  setup_right_boundary(area); //define correct xright value for sin(x) and circle
  setup_rectangles(area); //set all data related to rectangles
  /*set up X windows*/
  if(area->do_display) 
  {
    setup_window_height(area);
    setup_window_width(area);
    #ifndef NO_X11
    setupWindow(area);
    moveWindow(area);
    #ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD); //make sure all windows are set up before we go further
    #endif
    init_colors(area);
    #endif

    #if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
    #ifdef _MPI
    setupWindow_mpe(area);
    #endif
    #endif
  }

  handle_rectangles(area, current_rect); //calculates area of each rectangle and draws the rectangles
  
  /*curve is drawn after rectangles for visibility*/
  if(area->do_display)
  {
    #if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
    #ifdef _MPI
    draw_curve_mpe(area);
    if(area->num_threads < 11) draw_ranks_mpe(area); //XLib has problems with this function if num_threads > 10
    #endif
    #else
    draw_curve(area); //draw the function
    if(!area->gap) draw_process_barrier(area);
    #endif
  }
  
  sum_rectangles(area);//final summation and printing of result 
}
/*************************END OF area_under_curve()*****************************/

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
    
  int remainder = rect->num_rect % area->numProcs;
  
  if(remainder > 0){
    if(remainder > area->rank){
      rect->private_num_rect++;
      curve->my_xleft = curve->xleft + area->rank*rect->private_num_rect*rect->width;
    } else {
      curve->my_xleft = curve->xleft + remainder*rect->width + area->rank*rect->private_num_rect*rect->width;
    }
  } else {
    /* Calculate the left x-boundary of the process. */
    curve->my_xleft = curve->xleft + area->rank*rect->private_num_rect*rect->width;
  }
  /*Allocate memory for array of rectangles*/
  area->recs_t = malloc(rect->num_rect * sizeof (current_rec_t));
}

/***************************************************************************
 * Find boundaries of each rectangle and calculate the area for each*
 ***************************************************************************/
void handle_rectangles(struct area_t * area, struct current_rec_t * current_rect) {
  struct rec_t * rect = &(area->rect);
  struct curve_t * curve = &(area->curve);
  int current_rectangle=0;
  double current_left=0.0,current_height=0.0;
  double my_sum = 0.0;
  int start = 0, end = rect->private_num_rect, incrementFactor = 1;

  #ifdef MPI_CHUNKS_OF_ONE //only MPI chunks of one

    start = area->rank;
    incrementFactor = area->numProcs;
    end = rect->num_rect;
    for(current_rectangle = start; current_rectangle< end; current_rectangle += incrementFactor)
    {
      /* Calculate the x-value of the left side of the rectangle */
      current_left = curve->xleft + current_rectangle *rect->width;
      /* Calculate the height of the rectangle */
      current_height = FUNC(area, current_left);  
      /* Calculate the area of the rectangle and add it to the sum private to each process*/
      my_sum += rect->width * current_height; 
      /*Draw the current rectangle*/
      if(area->do_display)
      {
        /*For circle and sine we will use the scaled width and height*/
        if(area->circle)
        {
          current_left = curve->xleft + current_rectangle *rect->circle_width;
          current_height = scale_circle(current_left);
        }
        if(area->sin)
        {
          current_left = curve->xleft + current_rectangle *rect->sin_width;
          current_height = scale_sin(current_left);
        }
        area->recs_t[current_rectangle].x = current_left;
        area->recs_t[current_rectangle].h = current_height;            
        area->recs_t[current_rectangle].color_index = area->rank;
        draw_rectangle_mpe(area, current_rectangle);
      }//END IF(area->do_display)      
    } //END FOR

  #elif H_CHUNKS_OF_ONE // MPI chunks of g and OpenMP chunks of one or equal chunks

    start = area->rank;
    if(area->rect.num_rect%area->group_size == 0)
    {
      end = area->rect.num_rect/area->group_size;
    }
    else 
    {
      end = area->rect.num_rect/area->group_size +1;
    }
    incrementFactor = area->numProcs;
    int current_group = 0;
    int group_size = area->group_size;

    for(current_group = start; current_group < end; current_group += incrementFactor)
    {
      #ifdef OMP
      
      #ifdef H_CHUNKS_OF_ONE_STATIC
      #pragma omp parallel for shared(area, rect, curve, incrementFactor, end, start, current_group, group_size) private(current_rectangle, current_left, current_height, current_rect) reduction(+: my_sum) schedule(static,1)
      #elif H_CHUNKS_OF_ONE_DYNAMIC
      #pragma omp parallel for shared(area, rect, curve, incrementFactor, end, start, current_group, group_size) private(current_rectangle, current_left, current_height, current_rect) reduction(+: my_sum) schedule(dynamic)
      #else
      #pragma omp parallel for shared(area, rect, curve, incrementFactor, end, start, current_group, group_size) private(current_rectangle, current_left, current_height, current_rect) reduction(+: my_sum) schedule(static,group_size/area->num_threads)
      #endif
      #endif

      for(current_rectangle = current_group*group_size; current_rectangle < group_size*(current_group +1); current_rectangle++)
      {
        if(current_rectangle < area->rect.num_rect) //check that we are not going over the boundaries
        {
        /* Calculate the x-value of the left side of the rectangle */
        current_left = curve->xleft + current_rectangle*rect->width;
        /* Calculate the height of the rectangle */
        current_height = FUNC(area, current_left);
        /* Calculate the area of the rectangle and add it to the sum private to each process*/
        my_sum += rect->width * current_height;    
        /*Draw the current rectangle*/
        if(area->do_display)
        {
          /*For circle and sine we will use the scaled width and height*/
          if(area->circle)
          {
            current_left = curve->xleft + current_rectangle*rect->circle_width;
            current_height = scale_circle(current_left);
          }
          if(area->sin)
          {
            current_left = curve->xleft + current_rectangle*rect->sin_width;
            current_height = scale_sin(current_left);
          }
          area->recs_t[current_rectangle].x = current_left;
          area->recs_t[current_rectangle].h = current_height;
          #ifdef OMP
          area->recs_t[current_rectangle].color_index = area->num_threads * area->rank + omp_get_thread_num();
          #pragma omp critical //XLib is not thread safe!
          #else
          area->recs_t[current_rectangle].color_index = area->rank;
          #endif
          draw_rectangle_mpe(area,current_rectangle);          
        }//END IF(area->do_display)
        } //END IF        
      } //END Inner For
    } //END OUTER FOR

  #else //serial or equal chunks/equal chunks
    
    start = 0;
    incrementFactor = 1;
    end = rect->private_num_rect;
    #ifdef OMP
    #ifdef OMP_CHUNKS_OF_ONE_STATIC
    #pragma omp parallel for shared(area,rect,curve, incrementFactor, end) private(current_rectangle, current_left, current_height) reduction(+: my_sum) schedule(static,1)
    #elif OMP_CHUNKS_OF_ONE_DYNAMIC
    #pragma omp parallel for shared(area,rect,curve, incrementFactor, end) private(current_rectangle, current_left, current_height) reduction(+: my_sum) schedule(dynamic)
    #else
    #pragma omp parallel for shared(area,rect,curve, incrementFactor, end) private(current_rectangle, current_left, current_height) reduction(+: my_sum) schedule(static,area->rect.private_num_rect/area->num_threads)
    #endif
    #endif
    for(current_rectangle = start; current_rectangle< end; current_rectangle += incrementFactor)
    {
      /* Calculate the x-value of the left side of the rectangle */
      current_left = curve->my_xleft + current_rectangle *rect->width;
      /* Calculate the height of the rectangle */
      current_height = FUNC(area, current_left);
      /* Calculate the area of the rectangle and add it to the sum private to each process*/
      my_sum += rect->width * current_height;

      /*Draw the current rectangle*/
      if(area->do_display)
      {
        /*For circle and sine we will use the scaled width and height*/
        if(area->circle)
        {
	        current_left = curve->circle_my_xleft + current_rectangle*rect->circle_width; 
	        current_height = scale_circle(current_left);
        }
        if(area->sin)
        {
          current_left = curve->sin_my_xleft + current_rectangle*rect->sin_width;
	        current_height = scale_sin(current_left);
        }
        
          area->recs_t[current_rectangle].x = current_left;
          area->recs_t[current_rectangle].h = current_height;
          #ifdef OMP
          area->recs_t[current_rectangle].color_index = area->num_threads * area->rank + omp_get_thread_num();
          #pragma omp critical
          #else
          area->recs_t[current_rectangle].color_index = area->rank;
          #endif
          draw_rectangle(area,current_rectangle);        
      }//END IF(area->do_display)   
    } //END FOR

  #endif

  area->my_sum = my_sum;
}
/*************************END OF handle_rectangles()*****************************/

/***************************************************************************
 * Sum the areas of all rectangles*
 ***************************************************************************/
void sum_rectangles(struct area_t * area) {
  struct rec_t * rect = &(area->rect);
 
  /* Calculate the overall sum */
  #ifdef _MPI
  MPI_Reduce(&area->my_sum, &area->total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  #else
  area->total_sum = area->my_sum; //if not MPI then we only have one node, so my_sum becomes total_sum
  #endif
  
  /* Print the total sum */
  if (area->rank == 0){
    if(area->linear){
      printf("%f\n", area->total_sum);
      printf("Expected: %f\n",0.5 * area->curve.width*FUNC(area,area->curve.xright));
    }
    else if(area->circle){
      printf("1/4 * %1.48f * %d\n",area->total_sum*4,UNIT*UNIT);
      if(area->curve.xright == UNIT) printf("Expected: \n1/4 * %1.48f * radius^2\n",pi);
    }
    else if(area->sin){
      printf("%f\n",area->total_sum);
      if(area->curve.xright == pi*FREQ) printf("Expected: 2.0\n");
    }
    else printf("%f\n", area->total_sum);
  }
}


/***************************************************************************
 * Cleaning and finishing up  *
 ***************************************************************************/
void finish(struct area_t * area)
{ 
  #ifndef NO_X11
  if(area->do_display){
    displayForever(area);
  }
  #endif
  #if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
  #ifdef _MPI
  free_graphics(area);
  if(area->do_display) free(area->d.colors);
  #endif
  #endif
  #ifdef _MPI
  MPI_Finalize();
  #endif
  
}

/***************************************************************************
 * Parse command line arguments *
 ***************************************************************************/
void parse_args(struct area_t * area, int argc, char ** argv){
  int c,num;
  while((c = getopt(argc, argv, "n:l:r:f:x:w:t:T:g:")) != -1)
  {
    switch(c)
    {
      case 'n':
	     area->rect.num_rect = atoi(optarg);
	     break;
      case 'l':
	     if(atof(optarg) >=0) area->curve.xleft = atof(optarg);
	     else printf("Left boundary cannot be negative. Switching to default value...\n");
	     break;
      case 'r':
	     if(atof(optarg) >=0) area->curve.xright = atof(optarg);
	     else printf("Right boundary cannot be negative. Switching to default value...\n");
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
	       area->sin = 0;
	       area->circle = 0; //reset the other functions
	       area->linear = 0;
	       //if(area->rank == 0) printf("Usage: [-f 1 | 2 | 3 ]. Switching to default value(s)...\n");
	     }
	     break;
      case 'x':
	     if(strcmp("no-display",optarg)==0) area->do_display = 0;
	     else if(strcmp("display",optarg)==0)area->do_display = 1;
	     else{
	     if(area->rank == 0) printf("Usage: [-x display | no-display]. Switching to default value(s)...\n");
	     }
	     break;
      case 'w':
	     if(strcmp("no-gap",optarg)==0) area->gap = 0;
	     else if(strcmp("gap",optarg)==0) area->gap = 1;
	     else{
	       if(area->rank == 0) printf("Usage: [-w gap | no-gap]. Switching to default value(s).\n");
	     }
	     break;
      case 't':
	     area->time = atof(optarg);
	     break;
      case 'T':
        #ifdef OMP
        area->num_threads = atoi(optarg);
        omp_set_num_threads(area->num_threads);
        #else
        printf("Warning: OpenMP environment not specified, ignoring the '-T' option.\n");
        #endif
        break;
      case 'g':
        #ifdef H_CHUNKS_OF_ONE
        area->group_size = atoi(optarg);
        #else
        printf("Warning: Hybrid environment not specified, ignoring the '-g' option.\n");
        #endif
        break;
      case '?':
      default:
	     #ifdef _MPI
       fprintf(stderr, "Usage: mpirun -np NUMBER_OF_PROCESSES %s [-n NUMBER_OF_RECTANGLES] [-l X_LEFT] [-r X_RIGHT] [-f 1 for linear|2 for circle|3 for sin] [-x display | no-display] [-w gap | no-gap] [-t THROTTLE(seconds)] [-T number of threads if using OpenMP].\n", argv[0]);
	     #elif H_CHUNKS_OF_ONE
       fprintf(stderr, "Usage: mpirun -np NUMBER_OF_PROCESSES %s [-n NUMBER_OF_RECTANGLES] [-l X_LEFT] [-r X_RIGHT] [-f 1 for linear|2 for circle|3 for sin] [-x display | no-display] [-w gap | no-gap] [-t THROTTLE(seconds)] [-T number of threads if using OpenMP] [-g group size per process].\n", argv[0]);
       #else
       fprintf(stderr, "Usage: %s [-n NUMBER_OF_RECTANGLES] [-l X_LEFT] [-r X_RIGHT] [-f 1 for linear|2 for circle|3 for sin] [-x display | no-display] [-w gap | no-gap] [-t THROTTLE(seconds)] [-T number of threads if using OpenMP]\n", argv[0]);
	     #endif
  	   exit(-1);
    } //END switch
  } //END while
  if(area->numProcs * area->num_threads > NUM_COLORS) printf("Warning: processes*threads is larger than the amount of defined colors. Decrease the number of processing units or define more colors.\n");
  argc -= optind;
  argv += optind;
  
}
#endif