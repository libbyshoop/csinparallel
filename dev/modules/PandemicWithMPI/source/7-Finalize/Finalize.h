/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized the original code) */

#ifndef PANDEMIC_FINALIZE_H
#define PANDEMIC_FINALIZE_H

#include <stdlib.h>     // for malloc, free, and various others

#ifdef _MPI
#include <mpi.h>        // for MPI_Finalize()
#endif

#ifdef X_DISPLAY
#include "Display.h"    // for close_display()
#endif

void        show_results(struct our_t *our, struct stats_t *stats);
void        cleanup(struct global_t *global, struct our_t *our, 
                struct const_t *constant, struct display_t *dpy);

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
}

/*
    cleanups()
        Deallocate the arrays -- we have finished using 
        the memory, so now we "free" it back to the heap 
*/
void cleanup(struct global_t *global, struct our_t *our, 
    struct const_t *constant, struct display_t *dpy)
{
    // if use X_DISPLAY, do close_display()
    #ifdef X_DISPLAY
    close_display(our, dpy);
    #endif

    // free text display environment
    #ifdef TEXT_DISPLAY 
    int our_current_location_x;
    for(our_current_location_x = constant->environment_width - 1; 
        our_current_location_x >= 0; our_current_location_x--)
    {
        free(dpy->environment[our_current_location_x]);
    }
    free(dpy->environment);
    #endif

    // free arrays allocated in our struct
    free(our->our_num_days_infected);
    free(our->our_states);
    free(our->our_infected_y_locations);
    free(our->our_infected_x_locations);
    free(our->our_y_locations);
    free(our->our_x_locations);

    // free arrays allocated in global struct
    free(global->states);
    free(global->their_infected_x_locations);
    free(global->their_infected_y_locations);
    free(global->y_locations);
    free(global->x_locations);
    
    #ifdef _MPI
        // MPI execution is finished; no MPI calls are allowed after this
        MPI_Finalize();
    #endif
}

#endif