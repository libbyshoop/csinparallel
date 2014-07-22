/*MPEcolors.h
*	creates a range of specific colors (25 colors in total)
* AUTHOR: Ivana Marincic, Macalester College '15
* DATE: July, 2013
*/
#ifndef MPE_COLORS_H
#define MPE_COLORS_H

#ifdef _MPI
#include <mpe.h>
#include "colors.h" //for color constants
#include "structs.h"
#endif

#define MAX_RGB 65535
#define NUM_COLORS 25

void get_color_mpe(struct area_t * area, int index) {
  struct display_t * d = &(area->d);
  #if defined(MPI_CHUNKS_OF_ONE) || defined(H_CHUNKS_OF_ONE)
  int color_index, red=0, blue=0, green=0;
  double p_red, p_blue, p_green;
  switch(index){
    case RED:
      p_red = 1.0;
      p_blue = 0.0;
      p_green = 0.0;
      break;
    case BLUE:
      p_red = 0.0;
      p_blue = 1.0;
      p_green = 0.0;
      break;
    case YELLOW:
      p_red = 1.0;
      p_blue = 0.0;
      p_green = 1.0;
      break;
    case BISQUE:
      p_red = 1.00;
      p_green = 0.89;
      p_blue = 0.77;
      break;
    case VIOLET:
      p_red = 0.54;
      p_green = 0.17;
      p_blue = 0.89;
      break;
    case CYAN:
      p_red = 0.0;
      p_green = 1.0;
      p_blue = 1.0;
      break;
    case OLIVE:
      p_red = 0.50;
      p_green = 0.50;
      p_blue = 0.0;
      break;
    case GRAY:
      p_red = 0.75;
      p_green = 0.75;
      p_blue = 0.75;
      break;

    case PINK:
      p_red = 1.0;
      p_green = 0.41;
      p_blue = 0.71;
      break;
    case LAVENDER:
      p_red = 0.9;
      p_green = 0.9;
      p_blue = 0.98;
      break;
    case ORANGE:
      p_red =1.0;
      p_green = 0.65;
      p_blue = 0.0;
      break;
    case BROWN:
      p_red =0.63;
      p_green = 0.32;
      p_blue = 0.18;
      break;
    case AQUAMARINE:
      p_red = 0.5;
      p_green = 1.0;
      p_blue = 0.83;
      break;
    case CORAL:
      p_red = 1.0;
      p_green = 0.5;
      p_blue = 0.31;
      break;
    case DGREEN:
      p_red = 0.0;
      p_green = 0.39;
      p_blue = 0.0;
      break;
    case CRIMSON:
      p_red = 1.00;
      p_green = 0.89;
      p_blue = 0.77;
      break;
    case SEAGREEN:
      p_red = 0.86;
      p_green = 0.8;
      p_blue = 0.24;
      break;
    case PLUM:
      p_red = 0.87;
      p_green = 0.63;
      p_blue = 0.87;
      break;
    case NAVAJO:
      p_red = 1.0;
      p_green = 0.87;
      p_blue = 0.68;
      break;
    case TEAL:
      p_red = 0.0;
      p_green = 0.5;
      p_blue = 0.5;
      break;
    case STEELBLUE:
      p_red = 0.27;
      p_green = 0.51;
      p_blue = 0.71;
      break;
    case SADDLE:
      p_red = 0.55;
      p_green = 0.27;
      p_blue = 0.07;
      break;
    case LIGHTPINK:
      p_red =1.0;
      p_green = 0.71;
      p_blue = 0.76;
      break;
    case SKYBLUE:
      p_red =0.53;
      p_green = 0.81;
      p_blue = 0.98;
      break;
    case MEDBLUE:
      p_red =0.0;
      p_green = 0.0;
      p_blue = 0.8;
      break;
    default:
      break;
  }
  red = p_red * MAX_RGB;
  green = p_green * MAX_RGB;
  blue = p_blue * MAX_RGB;
  color_index = MPE_Add_RGB_color(d->canvas,red,green,blue,d->colors);
  d->myColor = d->colors[color_index];
  #endif
}

#endif