/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized and restructured the original code) */

#ifndef PANDEMIC_INITIALIZE_H
#define PANDEMIC_INITIALIZE_H

#include <stdlib.h>             // for malloc, free, and various others
#include <unistd.h>             // for random, getopt, some others
#include <time.h>               // for time is used to seed the random number generator

#ifdef X_DISPLAY
#include "Display.h"  // for init_display()
#endif

int         init (struct global_t *global, struct const_t *constant,
                struct stats_t *stats, struct display_t *dpy, int *c, char ***v);
void        parse_args (struct global_t *global, struct const_t *constant, 
                int argc, char ** argv);
void        init_check(struct global_t *global);
void        allocate_array(struct global_t *global, 
                struct const_t *constant, struct display_t *dpy);
void        init_array(struct global_t *global, struct const_t *constant);

/*
    init()
        Initialize runtime environment.
*/
int init (struct global_t *global, struct const_t *constant,
    struct stats_t *stats, struct display_t *dpy, int *c, char ***v) 
{
    // command line arguments
    int argc                        = *c;
    char ** argv                    = *v;

    // initialize constant values using DEFAULT values
    constant->environment_width     = DEFAULT_ENVIRO_SIZE;
    constant->environment_height    = DEFAULT_ENVIRO_SIZE;
    constant->infection_radius      = DEFAULT_RADIUS;
    constant->duration_of_disease   = DEFAULT_DURATION;
    constant->contagiousness_factor = DEFAULT_CONT_FACTOR;
    constant->deadliness_factor     = DEFAULT_DEAD_FACTOR;
    constant->total_number_of_days  = DEFAULT_DAYS;
    constant->microseconds_per_day  = DEFAULT_MICROSECS;

    // initialize global people counters using DEFAULT values
    global->number_of_people        = DEFAULT_SIZE;
    global->num_initially_infected  = DEFAULT_INIT_INFECTED;

    // initialize stats data in stats struct
    stats->num_infections = 0.0;
    stats->num_infection_attempts = 0.0;
    stats->num_deaths = 0.0;
    stats->num_recovery_attempts = 0.0;

    // initialize states counters in global struct
    global->num_infected = 0;
    global->num_susceptible = 0;
    global->num_immune = 0;
    global->num_dead = 0;

    // assign different colors for different states
    #ifdef X_DISPLAY
    dpy->red = "#FF0000";
    dpy->green = "#00FF00";
    dpy->black = "#000000";
    dpy->white = "#FFFFFF";
    #endif

    parse_args(global, constant, argc, argv);

    init_check(global);

    allocate_array(global, constant, dpy);

    // Seeds the random number generator based on the current time
    srandom(time(NULL));

    init_array(global, constant);

    // if use X_DISPLAY, do init_display()
    #ifdef X_DISPLAY
        init_display(constant, dpy);
    #endif

    return(0);
}

/*
    parse_args()
        Each process is given the parameters of the simulation
*/
void parse_args(struct global_t *global, struct const_t *constant, int argc, char ** argv) 
{
    int c = 0;

    // Get command line options -- this follows the idiom presented in the
    // getopt man page (enter 'man 3 getopt' on the shell for more)
    while((c = getopt(argc, argv, "n:i:w:h:t:T:c:d:D:m:")) != -1)
    {
        switch(c)
        {
            case 'n':
            global->number_of_people = atoi(optarg);
            break;
            case 'i':
            global->num_initially_infected = atoi(optarg);
            break;
            case 'w':
            constant->environment_width = atoi(optarg);
            break;
            case 'h':
            constant->environment_height = atoi(optarg);
            break;
            case 't':
            constant->total_number_of_days = atoi(optarg);
            break;
            case 'T':
            constant->duration_of_disease = atoi(optarg);
            break;
            case 'c':
            constant->contagiousness_factor = atoi(optarg);
            break;
            case 'd':
            constant->infection_radius = atoi(optarg);
            break;
            case 'D':
            constant->deadliness_factor = atoi(optarg);
            break;
            case 'm':
            constant->microseconds_per_day = atoi(optarg);
            break;
            // If the user entered "-?" or an unrecognized option, we need 
            // to print a usage message before exiting.
            case '?':
            default:
            fprintf(stderr, "Usage: ");
            fprintf(stderr, "%s [-n number_of_people][-i num_initially_infected][-w environment_width][-h environment_height][-t total_number_of_days][-T duration_of_disease][-c contagiousness_factor][-d infection_radius][-D deadliness_factor][-m microseconds_per_day]\n", argv[0]);
            exit(-1);
        }
    }
    argc -= optind;
    argv += optind;
}

/*
    init_check()
        Each process makes sure that the total number of initially 
        infected people is less than the total number of people
*/
void init_check(struct global_t *global)
{
    int num_initially_infected = global->num_initially_infected;
    int number_of_people = global->number_of_people;

    if(num_initially_infected > number_of_people)
    {
        fprintf(stderr, "ERROR: initial number of infected (%d) must be less than total number of people (%d)\n", 
            num_initially_infected, number_of_people);
        exit(-1);
    }
}

/*
    allocate_array()
        Allocate the arrays
*/
void allocate_array(struct global_t *global, struct const_t *constant, 
    struct display_t *dpy)
{
    int number_of_people = global->number_of_people;

    // Allocate the arrays in global struct
    global->x_locations = (int*)malloc(number_of_people * sizeof(int));
    global->y_locations = (int*)malloc(number_of_people * sizeof(int));
    global->infected_x_locations = (int*)malloc(number_of_people * sizeof(int));
    global->infected_y_locations = (int*)malloc(number_of_people * sizeof(int));
    global->states = (char*)malloc(number_of_people * sizeof(char));
    global->num_days_infected = (int*)malloc(number_of_people * sizeof(int));

    // Allocate the arrays for text display
    #ifdef TEXT_DISPLAY
    dpy->environment = (char**)malloc(constant->environment_width * 
        constant->environment_height * sizeof(char*));
    int current_location_x;
    for(current_location_x = 0; 
        current_location_x <= constant->environment_width - 1;
            current_location_x++)
    {
        dpy->environment[current_location_x] = (char*)malloc(
            constant->environment_height * sizeof(char));
    }
    #endif  
}

/*
    init_array()
        initialize arrays allocated with data in global 
        struct
*/
void init_array(struct global_t *global, struct const_t *constant)
{
    // counter to keep track of current person
    int current_person_id;

    int number_of_people = global->number_of_people;
    int num_initially_infected = global->num_initially_infected;

    // Each process spawns threads to set the states of the initially 
    // infected people and set the count of its infected people
    for(current_person_id = 0; current_person_id 
        <= num_initially_infected - 1; current_person_id++)
    {
        global->states[current_person_id] = INFECTED;
        global->num_infected++;
    }

    // Each process spawns threads to set the states of the rest of
    // its people and set the count of its susceptible people
    for(current_person_id = num_initially_infected; 
        current_person_id <= number_of_people - 1; 
        current_person_id++)
    {
        global->states[current_person_id] = SUSCEPTIBLE;
        global->num_susceptible++;
    }

    // Each process spawns threads to set random x and y locations for 
    // each of its people
    for(current_person_id = 0;
        current_person_id <= number_of_people - 1; 
        current_person_id++)
    {
        global->x_locations[current_person_id] = random() % constant->environment_width;
        global->y_locations[current_person_id] = random() % constant->environment_height;
    }

    // Each process spawns threads to initialize the number of days 
    // infected of each of its people to 0
    for(current_person_id = 0;
        current_person_id <= number_of_people - 1;
        current_person_id++)
    {
        global->num_days_infected[current_person_id] = 0;
    }
}

#endif