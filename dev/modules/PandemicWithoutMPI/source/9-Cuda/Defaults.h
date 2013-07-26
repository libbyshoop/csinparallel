/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized and restructured the original code, added CUDA) */

#ifndef PANDEMIC_DEFAULTS_H
#define PANDEMIC_DEFAULTS_H

#include <curand.h>     // cuda random number gen lib
#include <time.h>       // seed the random number generator 

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

// All the data needed for an X display
struct display_t 
{
    #ifdef TEXT_DISPLAY
    // Array of character arrays for text display
    char **environment;
    #endif

    #ifdef X_DISPLAY
    // Declare X-related variables 
    Display         *display;
    Window          window;
    int             screen;
    Atom            delete_window;
    GC              gc;
    XColor          infected_color;
    XColor          immune_color;
    XColor          susceptible_color;
    XColor          dead_color;
    Colormap        colormap;
    char            *red;
    char            *green;
    char            *black;
    char            *white;
    #endif
};

// All the data needed for CUDA operation: CUDA needs memory 
// pointers and other information on CPU side. As more than
// one function (mainly used by CUDA.cu) need to use these 
// data, we decided to use a struct to hold all these data.
struct cuda_t 
{
    // correspond with infected_locations in global struct
    int *infected_x_locations_dev; 
    int *infected_y_locations_dev;

    // correspond with locations in global struct
    int *x_locations_dev;
    int *y_locations_dev; 
    // correspond with states and num_days_infected in global struct 
    int *num_days_infected_dev;
    char *states_dev;

    // some counter variables require atomic operations 
    // correspond with states counters in global struct
    int *num_susceptible_dev;
    int *num_immune_dev;
    int *num_dead_dev;
    int *num_infected_dev;

    // correspond with variables in stats struct
    int *num_infections_dev;
    int *num_infection_attempts_dev;
    int *num_deaths_dev;
    int *num_recovery_attempts_dev;

    // the following four variables serve as the intermediate 
    // variables. we initialized variables in stats struct as 
    // doubles, but cuda atomic operations works better for 
    // int. So we cast doubles to int and then cast them back 
    int num_infections_int;
    int num_infection_attempts_int;
    int num_deaths_int;
    int num_recovery_attempts_int;

    // size used by cudaMalloc
    int people_size;
    int states_size;

    // size used by cuda kernel calls
    int numThread;
    int numBlock;

    // cuRAND random number generator
    float *rand_nums;
    time_t current_time;
    curandGenerator_t gen;

    // cuda timing required
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime;
};

#endif