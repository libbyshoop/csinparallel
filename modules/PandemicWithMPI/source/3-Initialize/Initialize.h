/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized the original code) */

#ifndef PANDEMIC_INITIALIZE_H
#define PANDEMIC_INITIALIZE_H

#include <stdlib.h>     // for malloc, free, and various others
#include <unistd.h>     // for random, getopt, some others
#include <time.h>       // for time is used to seed the random number generator

#ifdef _MPI
#include <mpi.h>        // for MPI_Init, MPI_Comm_rank, MPI_Comm_size
#endif

#ifdef X_DISPLAY
#include "Display.h"    // for init_display()
#endif

/* For Line Up Purpose */
/* For Line Up Purpose */
/* For Line Up Purpose */

int         init (struct global_t *global, struct our_t *our, struct const_t *constant,
                struct stats_t *stats, struct display_t *dpy, int *c, char ***v);
/* For Line Up Purpose */
void        parse_args (struct global_t *global, struct const_t *constant, 
                int argc, char ** argv);
void        init_check(struct global_t *global);
void        allocate_array(struct global_t *global, struct our_t *our, 
                struct const_t *constant, struct display_t *dpy);
void        find_size(struct global_t *global, struct our_t *our);
void        init_array(struct our_t *our, struct const_t *constant);

/*
    init()
        Initialize runtime environment.
*/
int init (struct global_t *global, struct our_t *our, struct const_t *constant,
    struct stats_t *stats, struct display_t *dpy, int *c, char ***v) 
{
    // command line arguments
    int argc                       = *c;
    char ** argv                   = *v;

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
    global->total_number_of_people         = DEFAULT_SIZE;
    global->total_num_initially_infected   = DEFAULT_INIT_INFECTED;
    global->total_num_infected             = DEFAULT_INIT_INFECTED;

    // initialize stats data in stats struct
    stats->our_num_infections = 0.0;
    stats->our_num_infection_attempts = 0.0;
    stats->our_num_deaths = 0.0;
    stats->our_num_recovery_attempts = 0.0;

    // initialize states counters in our struct
    our->our_num_infected = 0;
    our->our_num_susceptible = 0;
    our->our_num_immune = 0;
    our->our_num_dead = 0;

    // assign different colors for different states
    #ifdef X_DISPLAY
    dpy->red = "#FF0000";
    dpy->green = "#00FF00";
    dpy->black = "#000000";
    dpy->white = "#FFFFFF";
    #endif

    #ifdef _MPI
    // Each process initializes the distributed memory environment
    MPI_Init(&argc, &argv);
    #endif

    // Each process determines its rank and the total number of processes
    #ifdef _MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &our->our_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &global->total_number_of_processes);
    #else
    our->our_rank = 0;
    global->total_number_of_processes = 1;
    #endif

    init_check(global);

    parse_args(global, constant, argc, argv);

    allocate_array(global, our, constant, dpy);

    // Each process seeds the random number generator based on the
    // current time
    srandom(time(NULL));

    init_array(our, constant);

    // if use X_DISPLAY, do init_display()
    #ifdef X_DISPLAY
        init_display(our, constant, dpy);
    #endif

    /* For Line Up Purpose */
    /* For Line Up Purpose */
    /* For Line Up Purpose */
    /* For Line Up Purpose */

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
            global->total_number_of_people = atoi(optarg);
            break;
            case 'i':
            global->total_num_initially_infected = atoi(optarg);
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
            #ifdef MPI
            fprintf(stderr, "mpirun -np total_number_of_processes ");
            #endif
            fprintf(stderr, "%s [-n total_number_of_people][-i total_num_initially_infected][-w environment_width][-h environment_height][-t total_number_of_days][-T duration_of_disease][-c contagiousness_factor][-d infection_radius][-D deadliness_factor][-m microseconds_per_day]\n", argv[0]);
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
    int total_num_initially_infected = global->total_num_initially_infected;
    int total_number_of_people = global->total_number_of_people;

    if(total_num_initially_infected > total_number_of_people)
    {
        fprintf(stderr, "ERROR: initial number of infected (%d) must be less than total number of people (%d)\n", 
            total_num_initially_infected, total_number_of_people);
        exit(-1);
    }
}

/*
    allocate_array()
        Allocate the arrays
*/
void allocate_array(struct global_t *global, struct our_t *our, 
    struct const_t *constant, struct display_t *dpy)
{
    find_size(global, our);

    int total_number_of_people = global->total_number_of_people;
    int our_number_of_people = our->our_number_of_people;
    
    // Allocate the arrays in global struct
    global->x_locations = (int*)malloc(total_number_of_people * sizeof(int));
    global->y_locations = (int*)malloc(total_number_of_people * sizeof(int));
    global->their_infected_x_locations = (int*)malloc(total_number_of_people * sizeof(int));
    global->their_infected_y_locations = (int*)malloc(total_number_of_people * sizeof(int));
    global->states = (char*)malloc(total_number_of_people * sizeof(char));

    // Allocate the arrays in our struct
    our->our_x_locations = (int*)malloc(our_number_of_people * sizeof(int));
    our->our_y_locations = (int*)malloc(our_number_of_people * sizeof(int));
    our->our_infected_x_locations = (int*)malloc(our_number_of_people * sizeof(int));
    our->our_infected_y_locations = (int*)malloc(our_number_of_people * sizeof(int));
    our->our_states = (char*)malloc(our_number_of_people * sizeof(char));
    our->our_num_days_infected = (int*)malloc(our_number_of_people * sizeof(int));

    // Allocate the arrays for text display
    #ifdef TEXT_DISPLAY
    dpy->environment = (char**)malloc(constant->environment_width * 
        constant->environment_height * sizeof(char*));
    int our_current_location_x;
    for(our_current_location_x = 0; 
        our_current_location_x <= constant->environment_width - 1;
        our_current_location_x++)
    {
        dpy->environment[our_current_location_x] = (char*)malloc(
            constant->environment_height * sizeof(char));
    }
    #endif  
}

/*
    find_size()
        determine sizes for global struct and our struct 
        if using MPI
*/
void find_size(struct global_t *global, struct our_t *our)
{
    int our_rank = our->our_rank;
    int total_number_of_people          = global->total_number_of_people;
    int total_number_of_processes       = global->total_number_of_processes;
    int total_num_initially_infected    = global->total_num_initially_infected;

    // Each process determines the number of people for which 
    // it is responsible
    our->our_number_of_people = total_number_of_people / total_number_of_processes;

    // The last process is responsible for the remainder
    if(our_rank == total_number_of_processes - 1)
    {
        our->our_number_of_people += total_number_of_people % total_number_of_processes;
    }

    // Each process determines the number of initially infected people 
    // for which it is responsible
    our->our_num_initially_infected = total_num_initially_infected 
    / total_number_of_processes;

    // The last process is responsible for the remainder
    if(our_rank == total_number_of_processes - 1)
    {
        our->our_num_initially_infected += total_num_initially_infected 
        % total_number_of_processes;
    }
}

/*
    init_array()
        initialize arrays allocated with data in global 
        struct and our struct
*/
void init_array(struct our_t *our, struct const_t *constant)
{
    // counter to keep track of current person
    int my_current_person_id;

    int our_num_initially_infected = our->our_num_initially_infected;
    int our_number_of_people = our->our_number_of_people;

    // Each process spawns threads to set the states of the initially 
    // infected people and set the count of its infected people
    for(my_current_person_id = 0; my_current_person_id 
        <= our_num_initially_infected - 1; my_current_person_id++)
    {
        our->our_states[my_current_person_id] = INFECTED;
        our->our_num_infected++;
    }

    // Each process spawns threads to set the states of the rest of
    // its people and set the count of its susceptible people
    for(my_current_person_id = our_num_initially_infected; 
        my_current_person_id <= our_number_of_people - 1; 
        my_current_person_id++)
    {
        our->our_states[my_current_person_id] = SUSCEPTIBLE;
        our->our_num_susceptible++;
    }

    // Each process spawns threads to set random x and y locations for 
    // each of its people
    for(my_current_person_id = 0;
        my_current_person_id <= our_number_of_people - 1; 
        my_current_person_id++)
    {
        our->our_x_locations[my_current_person_id] = random() % constant->environment_width;
        our->our_y_locations[my_current_person_id] = random() % constant->environment_height;
    }

    // Each process spawns threads to initialize the number of days 
    // infected of each of its people to 0
    for(my_current_person_id = 0;
        my_current_person_id <= our_number_of_people - 1;
        my_current_person_id++)
    {
        our->our_num_days_infected[my_current_person_id] = 0;
    }
}

#endif