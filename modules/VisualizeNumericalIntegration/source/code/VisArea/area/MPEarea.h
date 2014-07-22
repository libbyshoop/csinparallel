/*MPEarea.h
*	Hold all drawing functions using the MPE graphics library.
* AUTHOR: Ivana Marincic, Macalester College '15
* DATE: July, 2013
*/
#ifndef MPEAREA_H
#define MPEAREA_H

#ifdef _MPI
#include <mpi.h>
#include <mpe.h>
#include <mpe_graphics.h>
#endif

#include <stdio.h> //for printf
#include <unistd.h> //for usleep
#include <stdlib.h> //for getenv()
#include <string.h> // for strncmp()

#include "structs.h"
#include "MPEcolors.h"

/***************************************************************************
 * Retrieves the display *
 ***************************************************************************/
char* getDisplay(struct area_t * area) {
  struct display_t * d = &(area->d);
  #if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
  d->display = getenv("DISPLAY");
  if(strncmp(d->display, "(null)", 7) == 0) {
    fprintf(stderr, "\n*** Fatal: DISPLAY variable not set.\n");
    exit(1);
  }
  return d->display;
  #else
  return NULL;
  #endif
  
  
}

/***************************************************************************
 * Opens the display, allocates colors, and sets black background *
 ***************************************************************************/
void setupWindow_mpe(struct area_t * area) {
  struct display_t * d = &(area->d);
  #if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
  /*Determine window width*/
  d->w_width = area->curve.width;
  if(area->circle) d->w_width = area->curve.circle_width;
  if(area->sin) d->w_width = area->curve.sin_width;

  #ifdef OMP
  //Status xit = XInitThreads();
  if(XInitThreads() == 0) {
    fprintf(stderr,"no threads;\n");
    exit(EXIT_FAILURE);
  }
  #endif
  
  /*Allocate color array and open the display*/
  d->colors = NULL;
  MPE_Open_graphics(&d->canvas, MPI_COMM_WORLD, getDisplay(area), 5,5,d->w_width,d->height,0);
  d->colors = malloc(NUM_COLORS * sizeof(MPE_Color));
  //d->colors = malloc(area->rect.num_rect * sizeof(MPE_Color));
  
  /*Color the background of the window black*/
  if(area->rank == 0) MPE_Fill_rectangle(d->canvas,0,0,d->w_width,d->height,MPE_BLACK);
  MPE_Update(d->canvas);
  #endif
}

/***************************************************************************
 * Draws a rectangle*
 ***************************************************************************/
void draw_rectangle_mpe(struct area_t * area, int i) {
  struct display_t * d = &(area->d);
  struct rec_t * rect = &(area->rect);
  double rect_width = area->rect.width;
  if(area->circle) {
    rect_width = area->rect.circle_width;
  }
  else if(area->sin) {
    rect_width = area->rect.sin_width;
  }
  get_color_mpe(area, area->recs_t[i].color_index);
  MPE_Fill_rectangle(d->canvas,area->recs_t[i].x,d->height-area->recs_t[i].h,rect_width,area->recs_t[i].h,d->myColor);
  MPE_Update(d->canvas);
  usleep(area->time * 1000);

}


/***************************************************************************
 * Draws the curve *
 ***************************************************************************/
void draw_curve_mpe(struct area_t * area){
  struct display_t * d = &(area->d);
  struct curve_t * curve = &(area->curve);
  int x1;
  int x2 = d->w_width;
  double y;
  if(area->rank == 0){
    for(x1=0;x1<=x2;x1++){
      y = FUNC(area, x1+curve->xleft);
      if(area->circle) y = scale_circle(x1+curve->circle_my_xleft);
      if(area->sin) y = scale_sin(x1 + curve->sin_my_xleft);
      MPE_Draw_point(d->canvas,x1,d->height-y,MPE_GREEN); 
      MPE_Update(d->canvas);
      usleep(5000);
    }
  }
}

/***************************************************************************
 * Draws vertical white lines that visually separate the processes*
 ***************************************************************************/
void draw_process_barrier_mpe(struct area_t * area){
  struct display_t * d = &(area->d);
  if(area->circle) MPE_Draw_line(d->canvas,area->curve.circle_my_xleft,0,area->curve.circle_my_xleft,d->height,MPE_GRAY);
  else if(area->sin) MPE_Draw_line(d->canvas,area->curve.sin_my_xleft,0,area->curve.sin_my_xleft,d->height,MPE_GRAY);
  else MPE_Draw_line(d->canvas,area->curve.my_xleft,0,area->curve.my_xleft,d->height,MPE_GRAY);
  MPE_Update(d->canvas);
}

/***************************************************************************
 * Draws a string specifying what process or thread is what color*
 ***************************************************************************/
void draw_ranks_mpe(struct area_t * area){
  struct display_t * d = &(area->d);
  char string[2];
  int x,y,id,color_index,proc;
  if(area->rank == 0)
  {
    
    for(proc=0; proc<area->numProcs; proc++)
    {
      y = 10 + 20*proc;
      #ifndef OMP
      x = 5;
      sprintf(string,"%d",proc);
      get_color_mpe(area,proc);
      MPE_Draw_string(d->canvas,x,y,d->myColor,string);
      MPE_Update(d->canvas);
      #else
      for(id=0;id<area->num_threads;id++)
      {
        x = 10 + 25*id;
        sprintf(string,"%d",id);
        color_index = area->num_threads*proc+id;
        get_color_mpe(area,color_index);
        MPE_Draw_string(d->canvas,x,y,d->myColor,string);
        MPE_Update(d->canvas);
      }
      #endif
    } 
  }
}

/***************************************************************************
 * Closes the display upon a mouse click*
 ***************************************************************************/
void free_graphics(struct area_t * area) {
  struct display_t * d = &(area->d);
  if(area->rank==0) printf("\n\nClick on the window or hit Ctrl+C to exit the demo.\n\n");
  int x = 0, y = 0, button = -1;
  if(area->rank == 0) MPE_Get_mouse_press(d->canvas,&x,&y,&button);
  MPE_Close_graphics(&d->canvas);
}


#endif