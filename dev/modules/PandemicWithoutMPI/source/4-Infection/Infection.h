/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized and restructured the original code) */

#ifndef PANDEMIC_INFECTION_H
#define PANDEMIC_INFECTION_H

void        find_infected(struct global_t *global);

/*
    find_infected()
        Each process determines its infected x locations 
        and infected y locations
*/
void find_infected(struct global_t *global)
{
    // counter to keep track of person in global struct
    int current_person_id;

    int current_infected_person = 0;
    for(current_person_id = 0; current_person_id <= global->number_of_people - 1; 
        current_person_id++)
    {
        if(global->states[current_person_id] == INFECTED)
        {
            global->infected_x_locations[current_infected_person] = 
            global->x_locations[current_person_id];
            global->infected_y_locations[current_infected_person] =
            global->y_locations[current_person_id];
            current_infected_person++;
        }
    }
}

#endif