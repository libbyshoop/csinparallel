#include <stdio.h>              // for printf
#include <stdlib.h>             // for malloc, free, and various others
#include <time.h>               // for time is used to seed the random number generator
#include <unistd.h>             // for random, getopt, some others
#include <X11/Xlib.h>           // for X display

#include "XPandemic.h"          // For display routines

#ifdef _MPI
#include <mpi.h>                // MPI_Allgather, MPI_Init, MPI_Comm_rank, MPI_Comm_size
#endif

#ifdef __CUDACC__
#include "kernel_functions.cu"  // cuda_init() and cuda_finish()
#endif

int         init (struct global_t *global, struct our_t *our, struct const_t *constant,
                struct stats_t *stats, struct display_t *dpy, struct cuda_t *cuda, 
                int *c, char ***v);
void        init_check(struct global_t *global);
void        allocate_array(struct global_t *global, struct our_t *our, 
                struct const_t *constant, struct display_t *dpy);
void        find_size(struct global_t *global, struct our_t *our);
void        init_array(struct our_t *our, struct const_t *constant);
void        find_infected(struct global_t *global, struct our_t *our);
void        share_infected(struct global_t *global, struct our_t *our);
void        share_location(struct global_t *global, struct our_t *our);
/* if not using cuda, include the following functions */
#if !defined(__CUDACC__) && !defined(MPICUDA)
void        move(struct our_t *our, struct const_t *constant);
void        susceptible(struct global_t *global, struct our_t *our, struct const_t *constant, 
                struct stats_t *stats);
void        infected(struct our_t *our, struct const_t *constant, struct stats_t *stats);
void        updateDays(struct our_t *our, struct const_t *constant);
#endif
void        parse_args (struct global_t *global, struct const_t *constant, 
                int argc, char ** argv);
void        show_results(struct our_t *our, struct stats_t *stats);
void        cleanup(struct global_t *global, struct our_t *our, 
                struct const_t *constant, struct display_t *dpy, struct cuda_t *cuda);

/*
    init()
        Initialize runtime environment.
*/
int init (struct global_t *global, struct our_t *our, struct const_t *constant,
    struct stats_t *stats, struct display_t *dpy, struct cuda_t *cuda, int *c, char ***v) 
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

    parse_args(global, constant, argc, argv);

    init_check(global);

    allocate_array(global, our, constant, dpy);

    // Each process seeds the random number generator based on the
    // current time
    srandom(time(NULL));

    init_array(our, constant);

    init_display(our, constant, dpy);

    // if use CUDA, do cuda_init()
    #if defined(__CUDACC__) || defined(MPICUDA)
        cuda_init(global, our, cuda);
    #endif

    return(0);
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

    // OMP does not support reduction to struct, create local instance
    // and then put local instance back to struct
    int our_num_infected_local = our->our_num_infected;
    int our_num_susceptible_local = our->our_num_susceptible;

    // Each process spawns threads to set the states of the initially 
    // infected people and set the count of its infected people
    #ifdef _OPENMP 
    #pragma omp parallel for private(my_current_person_id) \
        reduction(+:our_num_infected_local)
    #endif
    for(my_current_person_id = 0; my_current_person_id 
        <= our_num_initially_infected - 1; my_current_person_id++)
    {
        our->our_states[my_current_person_id] = INFECTED;
        our_num_infected_local++;
    }
    our->our_num_infected = our_num_infected_local;

    // Each process spawns threads to set the states of the rest of
    // its people and set the count of its susceptible people
    #ifdef _OPENMP 
    #pragma omp parallel for private(my_current_person_id) \
        reduction(+:our_num_susceptible_local)
    #endif
    for(my_current_person_id = our_num_initially_infected; 
        my_current_person_id <= our_number_of_people - 1; 
        my_current_person_id++)
    {
        our->our_states[my_current_person_id] = SUSCEPTIBLE;
        our_num_susceptible_local++;
    }
    our->our_num_susceptible = our_num_susceptible_local;

    // Each process spawns threads to set random x and y locations for 
    // each of its people
    #ifdef _OPENMP 
    #pragma omp parallel for private(my_current_person_id)
    #endif
    for(my_current_person_id = 0;
        my_current_person_id <= our_number_of_people - 1; 
        my_current_person_id++)
    {
        our->our_x_locations[my_current_person_id] = random() % constant->environment_width;
        our->our_y_locations[my_current_person_id] = random() % constant->environment_height;
    }

    // Each process spawns threads to initialize the number of days 
    // infected of each of its people to 0
    #ifdef _OPENMP 
    #pragma omp parallel for private(my_current_person_id)
    #endif
    for(my_current_person_id = 0;
        my_current_person_id <= our_number_of_people - 1;
        my_current_person_id++)
    {
        our->our_num_days_infected[my_current_person_id] = 0;
    }
}

/*
    find_infected()
        Each process determines its infected x locations 
        and infected y locations
*/
void find_infected(struct global_t *global, struct our_t *our)
{
    // counter to keep track of person in our struct
    int our_person1;

    int our_current_infected_person = 0;
    for(our_person1 = 0; our_person1 <= our->our_number_of_people - 1; our_person1++)
    {
        if(our->our_states[our_person1] == INFECTED)
        {
            our->our_infected_x_locations[our_current_infected_person] = 
            our->our_x_locations[our_person1];
            our->our_infected_y_locations[our_current_infected_person] =
            our->our_y_locations[our_person1];
            our_current_infected_person++;
        }
    }
}

/*
    share_infected()
        Each process sends its infectection information to all
        the other processes and receive information as well
*/
void share_infected(struct global_t *global, struct our_t *our)
{
    #ifdef _MPI
    int total_number_of_processes = global->total_number_of_processes;

    // Distributed Memory Information
    int *recvcounts;
    int *displs;
    recvcounts = (int*)malloc(total_number_of_processes * sizeof(int));  
    displs = (int*)malloc(total_number_of_processes * sizeof(int));

    // Each process sends its count of infected people to all the
    // other processes and receives their counts
    MPI_Allgather(&our->our_num_infected, 1, MPI_INT, recvcounts, 1, 
        MPI_INT, MPI_COMM_WORLD);

    global->total_num_infected = 0;
    int current_rank;
    for(current_rank = 0; current_rank <= total_number_of_processes - 1;
        current_rank++)
    {
        global->total_num_infected += recvcounts[current_rank];
    }

    // Set up the displacements in the receive buffer (see the man page for 
    // MPI_Allgatherv)
    int current_displ = 0;
    for(current_rank = 0; current_rank <= total_number_of_processes - 1;
        current_rank++)
    {
        displs[current_rank] = current_displ;
        current_displ += recvcounts[current_rank];
    }

    // Each process sends the x locations of its infected people to 
    // all the other processes and receives the x locations of their 
    // infected people
    MPI_Allgatherv(our->our_infected_x_locations, our->our_num_infected, MPI_INT, 
        global->their_infected_x_locations, recvcounts, displs, 
        MPI_INT, MPI_COMM_WORLD);

    // Each process sends the y locations of its infected people 
    // to all the other processes and receives the y locations of their 
    // infected people
    MPI_Allgatherv(our->our_infected_y_locations, our->our_num_infected, MPI_INT, 
        global->their_infected_y_locations, recvcounts, displs, 
        MPI_INT, MPI_COMM_WORLD);

    free(displs);
    free(recvcounts);
    #else
    global->total_num_infected = our->our_num_infected;
    int my_current_person_id;
    for(my_current_person_id = 0;
        my_current_person_id <= global->total_num_infected - 1;
        my_current_person_id++)
    {
        global->their_infected_x_locations[my_current_person_id] = 
        our->our_infected_x_locations[my_current_person_id];
        global->their_infected_y_locations[my_current_person_id] =
        our->our_infected_y_locations[my_current_person_id];
    }
    #endif
}

/*
    share_location()
        Each process sends its location information to all
        the other processes and receive information as well
*/
void share_location(struct global_t *global, struct our_t *our)
{
    #if defined(X_DISPLAY) || defined(TEXT_DISPLAY)  
        #ifdef _MPI
        int total_number_of_processes = global->total_number_of_processes;
        int total_number_of_people = global->total_number_of_people;

        // Distributed Memory Information
        int *recvcounts;
        int *displs;
        recvcounts = (int*)malloc(total_number_of_processes * sizeof(int));  
        displs = (int*)malloc(total_number_of_processes * sizeof(int));

        // Set up the receive counts and displacements in the 
        // receive buffer (see the man page for MPI_Gatherv)
        int current_displ = 0;
        int current_rank;
        for(current_rank = 0; current_rank <= total_number_of_processes - 1;
           current_rank++)
        {
            displs[current_rank] = current_displ;
            recvcounts[current_rank] = total_number_of_people / total_number_of_processes;
            if(current_rank == global->total_number_of_processes - 1)
            {
                recvcounts[current_rank] += total_number_of_people
                % total_number_of_processes;
            }
            current_displ += recvcounts[current_rank];
        }

        MPI_Gatherv(our->our_states, our->our_number_of_people, MPI_CHAR, 
            global->states, recvcounts, displs, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Gatherv(our->our_x_locations, our->our_number_of_people, MPI_INT, 
            global->x_locations, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(our->our_y_locations, our->our_number_of_people, MPI_INT, 
            global->y_locations, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        free(displs);
        free(recvcounts);
        #else
        int my_current_person_id;
        #ifdef _OPENMP 
            #pragma omp parallel for private(my_current_person_id)
        #endif
        for(my_current_person_id = 0; my_current_person_id 
           <= global->total_number_of_people - 1; my_current_person_id++)
        {
           global->states[my_current_person_id] 
           = our->our_states[my_current_person_id];
           global->x_locations[my_current_person_id] 
           = our->our_x_locations[my_current_person_id];
           global->y_locations[my_current_person_id] 
           = our->our_y_locations[my_current_person_id];
        }
        #endif
    #endif
}

/* if not using cuda, include the following functions */
#if !defined(__CUDACC__) && !defined(MPICUDA)

/*
    move()
        For each of the process’s people, each process spawns 
        threads to move everyone randomly
*/
void move(struct our_t *our, struct const_t *constant)
{
    // counter
    int my_current_person_id;
    int my_x_move_direction;
    int my_y_move_direction;

    int environment_width = constant->environment_width;
    int environment_height = constant->environment_height;
    char *our_states = our->our_states;
    int *our_x_locations = our->our_x_locations;
    int *our_y_locations = our->our_y_locations;

    #ifdef _OPENMP 
    #pragma omp parallel for private(my_current_person_id, my_x_move_direction, \
        my_y_move_direction)
    #endif
    for(my_current_person_id = 0; my_current_person_id 
        <= our->our_number_of_people - 1; my_current_person_id++)
    {
        // If the person is not dead, then
        if(our_states[my_current_person_id] != DEAD)
        {
            // The thread randomly picks whether the person moves left 
            // or right or does not move in the x dimension
            my_x_move_direction = (random() % 3) - 1;

            // The thread randomly picks whether the person moves up
            // or down or does not move in the y dimension
            my_y_move_direction = (random() % 3) - 1;

            // If the person will remain in the bounds of the
            // environment after moving, then
            if( (our_x_locations[my_current_person_id] 
                    + my_x_move_direction >= 0) &&
                (our_x_locations[my_current_person_id] 
                    + my_x_move_direction < environment_width) &&
                (our_y_locations[my_current_person_id] 
                    + my_y_move_direction >= 0) &&
                (our_y_locations[my_current_person_id] 
                    + my_y_move_direction < environment_height) )
            {
                // The thread moves the person
                our_x_locations[my_current_person_id] += my_x_move_direction;
                our_y_locations[my_current_person_id] += my_y_move_direction;
            }
        }
    }   
}

/*
    susceptible()
        For each of the process’s people, each process spawns 
        threads to handle susceptible personales
*/
void susceptible(struct global_t *global, struct our_t *our, struct const_t *constant, 
    struct stats_t *stats) 
{
    int infection_radius = constant->infection_radius;
    int contagiousness_factor = constant->contagiousness_factor;
    int total_num_infected = global->total_num_infected;

    // counters
    int my_current_person_id;
    int my_num_infected_nearby;
    int my_person2;

    // pointers in global struct
    int *their_infected_x_locations = global->their_infected_x_locations;
    int *their_infected_y_locations = global->their_infected_y_locations;

    // pointers in our struct
    int *our_x_locations = our->our_x_locations;
    int *our_y_locations = our->our_y_locations;
    char *our_states = our->our_states;

    // OMP does not support reduction to struct, create local instance
    // and then put local instance back to struct
    int our_num_infection_attempts_local = stats->our_num_infection_attempts;
    int our_num_infections_local = stats->our_num_infections;
    int our_num_infected_local = our->our_num_infected;
    int our_num_susceptible_local = our->our_num_susceptible;

    #ifdef _OPENMP 
    #pragma omp parallel for private(my_current_person_id, my_num_infected_nearby, \
        my_person2) reduction(+:our_num_infection_attempts_local) \
        reduction(+:our_num_infected_local) reduction(+:our_num_susceptible_local) \
        reduction(+:our_num_infections_local)
    #endif

    for(my_current_person_id = 0; my_current_person_id 
          <= our->our_number_of_people - 1; my_current_person_id++)
    {
        // If the person is susceptible, then
       if(our_states[my_current_person_id] == SUSCEPTIBLE)
       {
            // For each of the infected people (received earlier 
            // from all processes) or until the number of infected 
            // people nearby is 1, the thread does the following
            my_num_infected_nearby = 0;
            for(my_person2 = 0; my_person2 <= total_num_infected - 1
                && my_num_infected_nearby < 1; my_person2++)
            {
                // If person 1 is within the infection radius, then
                if((our_x_locations[my_current_person_id] 
                    > their_infected_x_locations[my_person2] - infection_radius) &&
                   (our_x_locations[my_current_person_id] 
                    < their_infected_x_locations[my_person2] + infection_radius) &&
                   (our_y_locations[my_current_person_id]
                    > their_infected_y_locations[my_person2] - infection_radius) &&
                   (our_y_locations[my_current_person_id]
                    < their_infected_y_locations[my_person2] + infection_radius))
                {
                    // The thread increments the number of infected people nearby
                    my_num_infected_nearby++;
                }
            }

            #ifdef SHOW_RESULTS
            if(my_num_infected_nearby >= 1)
              our_num_infection_attempts_local++;
            #endif

            // If there is at least one infected person nearby, and 
            // a random number less than 100 is less than or equal 
            // to the contagiousness factor, then
            if(my_num_infected_nearby >= 1 && (random() % 100) 
                <= contagiousness_factor)
            {
                // The thread changes person1’s state to infected
                our_states[my_current_person_id] = INFECTED;

                // The thread updates the counters
                our_num_infected_local++;
                our_num_susceptible_local--;

                #ifdef SHOW_RESULTS
                our_num_infections_local++;
                #endif
            }
        }
    }
    // update struct data with local instances
    stats->our_num_infection_attempts = our_num_infection_attempts_local;
    stats->our_num_infections = our_num_infections_local;
    our->our_num_infected = our_num_infected_local;
    our->our_num_susceptible = our_num_susceptible_local;
}

/*
    infected()
        For each of the process’s people, each process spawns 
        threads to handle infected personales
*/
void infected(struct our_t *our, struct const_t *constant, 
    struct stats_t *stats)
{
    int duration_of_disease = constant->duration_of_disease;
    int deadliness_factor = constant->deadliness_factor;

    // counter
    int my_current_person_id;

    // pointers in our struct
    char *our_states = our->our_states;
    int *our_x_locations = our->our_x_locations;
    int *our_y_locations = our->our_y_locations;
    int *our_num_days_infected = our->our_num_days_infected;

    // OMP does not support reduction to struct, create local instance
    // and then put local instance back to struct
    int our_num_recovery_attempts_local = stats->our_num_recovery_attempts;
    int our_num_deaths_local = stats->our_num_deaths;
    int our_num_dead_local = our->our_num_dead;
    int our_num_infected_local = our->our_num_infected;
    int our_num_immune_local = our->our_num_immune;

    #ifdef _OPENMP 
    #pragma omp parallel for private(my_current_person_id) \
        reduction(+:our_num_recovery_attempts_local) reduction(+:our_num_dead_local) \
        reduction(+:our_num_infected_local) reduction(+:our_num_deaths_local) \
        reduction(+:our_num_immune_local)
    #endif

    for(my_current_person_id = 0; my_current_person_id 
        <= our->our_number_of_people - 1; my_current_person_id++)
    {
        // If the person is infected and has been for the full 
        // duration of the disease, then 
        if(our_states[my_current_person_id] == INFECTED
            && our_num_days_infected[my_current_person_id] == duration_of_disease)
        {
            #ifdef SHOW_RESULTS
                our_num_recovery_attempts_local++;
            #endif
            // If a random number less than 100 is less than 
            // the deadliness factor, then 
            if((random() % 100) < deadliness_factor)
            {
                // The thread changes the person’s state to dead
                our_states[my_current_person_id] = DEAD;
                // The thread updates the counters
                our_num_dead_local++;
                our_num_infected_local--;
                // The thread updates stats counter
                #ifdef SHOW_RESULTS
                    our_num_deaths_local++;
                #endif
            }
            // Otherwise,
            else
            {
                // The thread changes the person’s state to immune
                our_states[my_current_person_id] = IMMUNE;
                // The thread updates the counters
                our_num_immune_local++;
                our_num_infected_local--;
            }
        }
    }

    // update struct data with local instances
    stats->our_num_recovery_attempts = our_num_recovery_attempts_local;
    stats->our_num_deaths = our_num_deaths_local;
    our->our_num_dead = our_num_dead_local;
    our->our_num_infected = our_num_infected_local;
    our->our_num_immune = our_num_immune_local;
}

/*
    updateDays()
        For each of the process’s people, each process spawns 
        threads to increase infected days
*/
void updateDays(struct our_t *our, struct const_t *constant)
{
    int my_current_person_id;

    // pointers in our struct
    char *our_states = our->our_states;
    int *our_num_days_infected = our->our_num_days_infected;

    #ifdef _OPENMP 
        #pragma omp parallel for private(my_current_person_id)
    #endif
    for(my_current_person_id = 0; my_current_person_id 
        <= our->our_number_of_people - 1; my_current_person_id++)
    {
        // If the person is infected, then
        if(our_states[my_current_person_id] == INFECTED)
        {
            // Increment the number of days the person has been infected
            our_num_days_infected[my_current_person_id]++;
        }
    }
}

#endif

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
    show_results()
        print out results
*/
void show_results(struct our_t *our, struct stats_t *stats)
{
    #ifdef SHOW_RESULTS
    printf("Rank %d final counts: %d susceptible, %d infected, %d immune, %d dead \nRank %d actual contagiousness: %f \nRank %d actual deadliness: %f \n", 
        our->our_rank, our->our_num_susceptible, our->our_num_infected, 
        our->our_num_immune, our->our_num_dead, our->our_rank, 
        100.0 * (stats->our_num_infections / (stats->our_num_infection_attempts 
            == 0 ? 1 : stats->our_num_infection_attempts)),our->our_rank, 
        100.0 * (stats->our_num_deaths / (stats->our_num_recovery_attempts 
            == 0 ? 1 : stats->our_num_recovery_attempts)));
    #endif

    #ifdef _OPENMP
    printf("I am using OpenMP!! \n");
    #endif

    #ifdef __CUDACC__
    printf("I am using CUDA!! \n");
    #endif

    #ifdef MPICUDA
    printf("I am using CUDA and MPI!! \n");
    #endif
}

/*
    cleanups()
        Deallocate the arrays -- we have finished using 
        the memory, so now we "free" it back to the heap 
*/
void cleanup(struct global_t *global, struct our_t *our, 
    struct const_t *constant, struct display_t *dpy, struct cuda_t *cuda)
{
    // if use CUDA, do cuda_finish()
    #if defined(__CUDACC__) || defined(MPICUDA)
    cuda_finish(cuda);
    #endif

    close_display(our, dpy);

    #ifdef TEXT_DISPLAY 
    int our_current_location_x;
    for(our_current_location_x = constant->environment_width - 1; 
        our_current_location_x >= 0; our_current_location_x--)
    {
        free(dpy->environment[our_current_location_x]);
    }
    free(dpy->environment);
    #endif

    // free arrays allocated in global struct
    free(global->x_locations);
    free(global->y_locations);
    free(global->their_infected_y_locations);
    free(global->their_infected_x_locations);
    free(global->states);

    // free arrays allocated in local struct
    free(our->our_x_locations);
    free(our->our_y_locations);
    free(our->our_infected_x_locations);
    free(our->our_infected_y_locations);
    free(our->our_states);
    free(our->our_num_days_infected);

    #ifdef _MPI
        // MPI execution is finished; no MPI calls are allowed after this
        MPI_Finalize();
    #endif
}