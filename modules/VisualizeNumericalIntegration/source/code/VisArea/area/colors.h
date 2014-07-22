/*colors.h
*	Holds all functions and constants responsible for setting the desired colors.
* AUTHOR: Ivana Marincic, Macalester College '15
* DATE: July, 2013
*/
#ifndef COLORS_H
#define COLORS_H

#include "structs.h"

#ifndef NO_X11
#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#endif

#define RED		0
#define BLUE		1
#define YELLOW		2
#define BISQUE		3
#define VIOLET		4
#define CYAN		5 
#define OLIVE		6
#define GRAY		7
#define PINK		8
#define LAVENDER	9
#define ORANGE		10
#define BROWN		11

/*some extra colors*/
#define AQUAMARINE	12
#define CORAL		13
#define DGREEN		14
#define CRIMSON		15
#define SEAGREEN	16
#define PLUM		17
#define NAVAJO		18
#define TEAL		19
#define STEELBLUE	20
#define SADDLE		21
#define LIGHTPINK	22
#define SKYBLUE		23
#define MEDBLUE		24

#define NUM_COLORS 25

/*colors of the rectangles*/
char red[] 	= "#FF0000";
char blue[] 	= "#0000FF";
char yellow[] 	= "#FFFF00";
char bisque[]	= "#FFE4C4";
char violet[] 	= "#8A2BE2";
char cyan[] 	= "#00FFFF";
char olive[]	= "#808000";
char gray[] 	= "#BEBEBE";
char pink[] 	= "#FF69B4";
char lavender[] = "#E6E6FA";
char orange[] 	= "#FFA500";
char brown[] 	= "#A0522D";
/*extra colors*/
char aqmarine[] = "#7FFFD4";
char coral[] 	= "#FF7F50";
char dgreen[]	= "#006400";
char crimson[]	= "#DC143C";
char seagreen[]	= "#8FBC8F";
char plum[]	= "#DDA0DD";
char navajo[]	= "#FFDEAD";
char teal[]	= "#008080";
char steelblue[]= "#4682B4";
char saddle[]	= "#8B4513";
char lightpink[]= "#FFB6C1";
char skyblue[]	= "#87CEFA";
char medblue[]	= "#0000CD";

char green[] = "#00FF00"; //color of the curve

/*Parsing and allocation of colors*/
void init_colors(struct area_t * area){
  struct display_t * d = &(area->d);
  #ifndef NO_X11
  //green
  XParseColor(d->dpy, d->theColormap, green, &(d->green_col));
  XAllocColor(d->dpy, d->theColormap, &(d->green_col));
  //red
  XParseColor(d->dpy, d->theColormap, red, &(d->red_col));
  XAllocColor(d->dpy, d->theColormap, &(d->red_col));
  //blue  
  XParseColor(d->dpy, d->theColormap, blue, &(d->blue_col));
  XAllocColor(d->dpy, d->theColormap, &(d->blue_col)); 
  //yellow 
  XParseColor(d->dpy, d->theColormap, yellow, &(d->yellow_col));
  XAllocColor(d->dpy, d->theColormap, &(d->yellow_col));
  //bisque
  XParseColor(d->dpy, d->theColormap, bisque, &(d->bisque_col));
  XAllocColor(d->dpy, d->theColormap, &(d->bisque_col));
  //violet
  XParseColor(d->dpy, d->theColormap, violet, &(d->violet_col));
  XAllocColor(d->dpy, d->theColormap, &(d->violet_col)); 
  //cyan
  XParseColor(d->dpy, d->theColormap, cyan, &(d->cyan_col));
  XAllocColor(d->dpy, d->theColormap, &(d->cyan_col));
  //olive
  XParseColor(d->dpy, d->theColormap, olive, &(d->olive_col));
  XAllocColor(d->dpy, d->theColormap, &(d->olive_col));
  //gray
  XParseColor(d->dpy, d->theColormap, gray, &(d->gray_col));
  XAllocColor(d->dpy, d->theColormap, &(d->gray_col)); 
  //pink
  XParseColor(d->dpy, d->theColormap, pink, &(d->pink_col));
  XAllocColor(d->dpy, d->theColormap, &(d->pink_col));
  //lavender
  XParseColor(d->dpy, d->theColormap, lavender, &(d->lavender_col));
  XAllocColor(d->dpy, d->theColormap, &(d->lavender_col));
  //orange
  XParseColor(d->dpy, d->theColormap, orange, &(d->orange_col));
  XAllocColor(d->dpy, d->theColormap, &(d->orange_col));
  //brown
  XParseColor(d->dpy, d->theColormap, brown, &(d->brown_col));
  XAllocColor(d->dpy, d->theColormap, &(d->brown_col));
  
  /*extra colors*/
  //aquamarine
  XParseColor(d->dpy, d->theColormap, aqmarine, &(d->aquam_col));
  XAllocColor(d->dpy, d->theColormap, &(d->aquam_col));
  //coral
  XParseColor(d->dpy, d->theColormap, coral, &(d->coral_col));
  XAllocColor(d->dpy, d->theColormap, &(d->coral_col));
  //dark green
  XParseColor(d->dpy, d->theColormap, dgreen, &(d->dgreen_col));
  XAllocColor(d->dpy, d->theColormap, &(d->dgreen_col));
  //crimson
  XParseColor(d->dpy, d->theColormap, crimson, &(d->crimson_col));
  XAllocColor(d->dpy, d->theColormap, &(d->crimson_col));
  //seagreen
  XParseColor(d->dpy, d->theColormap, seagreen, &(d->seagreen_col));
  XAllocColor(d->dpy, d->theColormap, &(d->seagreen_col));
  //plum
  XParseColor(d->dpy, d->theColormap, plum, &(d->plum_col));
  XAllocColor(d->dpy, d->theColormap, &(d->plum_col));
  //navajo white
  XParseColor(d->dpy, d->theColormap, navajo, &(d->navajo_col));
  XAllocColor(d->dpy, d->theColormap, &(d->navajo_col));
  //teal
  XParseColor(d->dpy, d->theColormap, teal, &(d->teal_col));
  XAllocColor(d->dpy, d->theColormap, &(d->teal_col));
  //steel blue
  XParseColor(d->dpy, d->theColormap, steelblue, &(d->steelblue_col));
  XAllocColor(d->dpy, d->theColormap, &(d->steelblue_col));
  //saddle brown
  XParseColor(d->dpy, d->theColormap, saddle, &(d->saddle_col));
  XAllocColor(d->dpy, d->theColormap, &(d->saddle_col));
  //light pink
  XParseColor(d->dpy, d->theColormap, lightpink, &(d->lightpink_col));
  XAllocColor(d->dpy, d->theColormap, &(d->lightpink_col));
  //sky blue
  XParseColor(d->dpy, d->theColormap, skyblue, &(d->skyblue_col));
  XAllocColor(d->dpy, d->theColormap, &(d->skyblue_col));
  //medium blue
  XParseColor(d->dpy, d->theColormap, medblue, &(d->medblue_col));
  XAllocColor(d->dpy, d->theColormap, &(d->medblue_col));
  #endif
}

#ifndef NO_X11
/*Setting the colors to foreground*/
void cred(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->red_col.pixel);
}

void cblue(struct display_t * d){ 
  XSetForeground(d->dpy, d->gc, d->blue_col.pixel);
}

void cyellow(struct display_t * d){ 
  XSetForeground(d->dpy, d->gc, d->yellow_col.pixel);
}

void cbisque(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->bisque_col.pixel);
}

void cviolet(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->violet_col.pixel);
}
void ccyan(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->cyan_col.pixel);
}

void colive(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->olive_col.pixel);
}

void cgray(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->gray_col.pixel);
}

void cpink(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->pink_col.pixel);
}

void clavender(struct display_t * d){ 
  XSetForeground(d->dpy, d->gc, d->lavender_col.pixel);
}

void corange(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->orange_col.pixel);
}

void cbrown(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->brown_col.pixel);
}

/*extra colors*/
void caqua(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->aquam_col.pixel);
}

void ccoral(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->coral_col.pixel);
}

void cdgreen(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->dgreen_col.pixel);
}

void ccrimson(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->crimson_col.pixel);
}

void cseagreen(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->seagreen_col.pixel);
}

void cplum(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->plum_col.pixel);
}

void cnavajo(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->navajo_col.pixel);
}

void cteal(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->teal_col.pixel);
}

void csteelblue(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->steelblue_col.pixel);
}

void csaddle(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->saddle_col.pixel);
}

void clightpink(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->lightpink_col.pixel);
}

void cskyblue(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->skyblue_col.pixel);
}

void cmedblue(struct display_t * d){
  XSetForeground(d->dpy, d->gc, d->medblue_col.pixel);
}


void get_extra_colors(int order, struct display_t * d){
  switch(order){
    case AQUAMARINE:
      caqua(d);
      break;
    case CORAL:
      ccoral(d);
      break;
    case DGREEN:
      cdgreen(d);
      break;
    case CRIMSON:
      ccrimson(d);
      break;
    case SEAGREEN:
      cseagreen(d);
      break;
    case PLUM:
      cplum(d);
      break;
    case NAVAJO:
      cnavajo(d);
      break;
    case TEAL:
      cteal(d);
      break;
    case STEELBLUE:
      csteelblue(d);
      break;
    case SADDLE:
      csaddle(d);
      break;
    case LIGHTPINK:
      clightpink(d);
      break;
    case SKYBLUE:
      cskyblue(d);
      break;
    case MEDBLUE:
      cmedblue(d);
      break;
    default: break;
  }
}
#endif
#endif