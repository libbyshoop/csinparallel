/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized and restructured the original code) */

#ifndef PANDEMIC_DEFAULTS_H
#define PANDEMIC_DEFAULTS_H

// States of people -- all people are one of these 4 states 
// These are const char because they are displayed as ASCII
// if TEXT_DISPLAY is enabled 
const char INFECTED = 'X';
const char IMMUNE = 'I';
const char SUSCEPTIBLE = 'o';
const char DEAD = ' ';

// Size, in pixels, of the X window(s) for each person
#ifdef X_DISPLAY
const int PIXEL_WIDTH_PER_PERSON = 10;
const int PIXEL_HEIGHT_PER_PERSON = 10;
#endif

// Default parameters for the simulation
const int DEFAULT_ENVIRO_SIZE = 30;
const int DEFAULT_RADIUS = 3;
const int DEFAULT_DURATION = 50;
const int DEFAULT_CONT_FACTOR = 30;
const int DEFAULT_DEAD_FACTOR = 30;
const int DEFAULT_DAYS = 250;
const int DEFAULT_MICROSECS = 100000;
const int DEFAULT_SIZE = 50;
const int DEFAULT_INIT_INFECTED = 1;

// Data being used for the X display
struct display_t {

	#ifdef TEXT_DISPLAY
    // Array of character arrays for text display 
	char **environment;
	#endif

	#ifdef X_DISPLAY
	// Declare X-related variables 
	Display   		*display;
	Window    		window;
	int 			screen;
	Atom 			delete_window;
	GC        		gc;
	XColor 			infected_color;
	XColor 			immune_color;
	XColor 			susceptible_color;
	XColor 			dead_color;
	Colormap 		colormap;
	char 			*red;
	char 			*green;
	char 			*black;
	char 			*white;
	#endif
};

// All the data needed globally. Holds EVERYONE's location, 
// states and other necessary counters.
struct global_t 
{
	// current day
	int current_day;
	// people counters
	int number_of_people;
	int num_initially_infected;
	// states counters
	int num_infected;
	int num_susceptible;
	int num_immune;
	int num_dead;  
	// locations
	int *x_locations;
	int *y_locations;
	// infected people's locations
	int *infected_x_locations;
	int *infected_y_locations;
	// state
	char *states;
	// infected time
	int *num_days_infected;
};

// Data being used as constant
struct const_t 
{
    // environment
    int environment_width;
    int environment_height;
    // disease
    int infection_radius;
    int duration_of_disease;
    int contagiousness_factor;
    int deadliness_factor;
    // time
    int total_number_of_days;
    int microseconds_per_day;
};

// Data being used for SHOW_RESULTS
struct stats_t 
{
    double num_infections;
    double num_infection_attempts;
    double num_deaths;
    double num_recovery_attempts; 
};

#endif