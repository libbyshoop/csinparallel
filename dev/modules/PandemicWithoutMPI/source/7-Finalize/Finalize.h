/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized and restructured the original code) */

#ifndef PANDEMIC_FINALIZE_H
#define PANDEMIC_FINALIZE_H

#include <stdlib.h>             // for free, and various others

#ifdef X_DISPLAY
#include "Display.h"  // for close_display()
#endif

void 		show_results(struct global_t *global, struct stats_t *stats);
void 		cleanup(struct global_t *global,
				struct const_t *constant, struct display_t *dpy);

/*
    show_results()
        print out results
*/
void show_results(struct global_t *global, struct stats_t *stats)
{
	#ifdef SHOW_RESULTS
	printf("final counts: %d susceptible, %d infected, %d immune, %d dead \nActual contagiousness: %f \nActual deadliness: %f \n", 
		global->num_susceptible, global->num_infected, 
		global->num_immune, global->num_dead, 
		100.0 * (stats->num_infections / (stats->num_infection_attempts 
			== 0 ? 1 : stats->num_infection_attempts)), 
		100.0 * (stats->num_deaths / (stats->num_recovery_attempts 
			== 0 ? 1 : stats->num_recovery_attempts)));
	#endif
}

/*
    cleanups()
        Deallocate the arrays -- we have finished using 
        the memory, so now we "free" it back to the heap 
*/
void cleanup(struct global_t *global, struct const_t *constant, 
	struct display_t *dpy)
{
	// if use X_DISPLAY, do close_display()
    #ifdef X_DISPLAY
    close_display(dpy);
    #endif

    // free text display environment
	#ifdef TEXT_DISPLAY 
	int current_location_x;
	for(current_location_x = constant->environment_width - 1; 
		current_location_x >= 0; current_location_x--)
	{
		free(dpy->environment[current_location_x]);
	}
	free(dpy->environment);
	#endif

	// free arrays allocated in global struct
	free(global->x_locations);
	free(global->y_locations);
	free(global->infected_y_locations);
	free(global->infected_x_locations);
	free(global->states);
	free(global->num_days_infected);
}

#endif