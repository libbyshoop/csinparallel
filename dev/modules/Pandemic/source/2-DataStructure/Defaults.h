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

// All the data needed for an X display
struct display_t 
{
    #ifdef TEXT_DISPLAY
    // Array of character arrays, a.k.a. array of character pointers,
    // for text display 
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

// All the data needed globally. Holds EVERYONE's location, 
// states and other necessary counters.
struct global_t 
{
    // people counters
    int total_number_of_people;
    int total_num_initially_infected; 
    int total_num_infected;
    // locations
    int *x_locations;
    int *y_locations;
    // infected people's locations
    int *their_infected_x_locations;
    int *their_infected_y_locations;
    // state
    char *states;
    // MPI related
    int total_number_of_processes;
};

// All the data needed locally. Holds people's location, 
// states and other necessary counters on each node.
struct our_t 
{
    // current day
    int current_day;
    // MPI related 
    int our_rank;
    // people counters
    int our_number_of_people;
    int our_num_initially_infected;
    // states counters
    int our_num_infected;
    int our_num_susceptible;
    int our_num_immune;
    int our_num_dead; 
    // locations
    int *our_x_locations;
    int *our_y_locations;
    // our infected people's locations
    int *our_infected_x_locations;
    int *our_infected_y_locations;
    // states
    char *our_states;
    // infected time
    int *our_num_days_infected;
};

// All the data needed as constant
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

// All the data needed to show stats in results
struct stats_t 
{
    double our_num_infections;
    double our_num_infection_attempts;
    double our_num_deaths;
    double our_num_recovery_attempts; 
};

// All the data needed for CUDA operation
struct cuda_t 
{
    // only CUDA-only version and MPICUDA version can see what 
    // is inside the cuda_t struct, and all other versions 
    // will initialize a empty cuda_t struct 
    #if defined(__CUDACC__) || defined(MPICUDA)

    // correspond with their_infected_locations in global struct
    int *their_infected_x_locations_dev; 
    int *their_infected_y_locations_dev; 
    // correspond with our_infected_locations in our struct
    int *our_x_locations_dev;
    int *our_y_locations_dev;
    // correspond with our_states and our_num_days_infected in our struct 
    int *our_num_days_infected_dev;
    char *our_states_dev;

    // some counter variables require atomic operations 
    // correspond with states counters in our struct
    int *our_num_susceptible_dev;
    int *our_num_immune_dev;
    int *our_num_dead_dev;
    int *our_num_infected_dev;

    // correspond with variables in stats struct
    int *our_num_infections_dev;
    int *our_num_infection_attempts_dev;
    int *our_num_deaths_dev;
    int *our_num_recovery_attempts_dev;

    // the following four variables serve as the intermediate 
    // variables. we initialized variables in stats struct as 
    // doubles, but cuda atomic operations works better for 
    // int. So we cast doubles to int and then cast them back 
    int our_num_infections_int;
    int our_num_infection_attempts_int;
    int our_num_deaths_int;
    int our_num_recovery_attempts_int;

    // size used by cudaMalloc
    int our_size;
    int their_size;
    int our_states_size;

    // size used by cuda kernel calls
    int numThread;
    int numBlock;
    #endif
};

#endif