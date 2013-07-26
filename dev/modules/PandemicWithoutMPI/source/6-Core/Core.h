/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized and restructured the original code) */

#ifndef PANDEMIC_CORE_H
#define PANDEMIC_CORE_H

#include <unistd.h>             // for random

void        move(struct global_t *global, struct const_t *constant);
void        susceptible(struct global_t *global, struct const_t *constant, 
                struct stats_t *stats);
void        infected(struct global_t *global, struct const_t *constant, 
                struct stats_t *stats);
void        update_days_infected(struct global_t *global, struct const_t *constant);

/*
    move()
        For each of the process’s people, each process spawns 
        threads to move everyone randomly
*/
void move(struct global_t *global, struct const_t *constant)
{
    // counter
    int current_person_id;

    // movement
    int x_move_direction;
    int y_move_direction;

    // display envrionment variables
    int environment_width = constant->environment_width;
    int environment_height = constant->environment_height;

    // arrays in global struct
    char *states = global->states;
    int *x_locations = global->x_locations;
    int *y_locations = global->y_locations;

    for(current_person_id = 0; current_person_id 
        <= global->number_of_people - 1; current_person_id++)
    {
        // If the person is not dead, then
        if(states[current_person_id] != DEAD)
        {
            // The thread randomly picks whether the person moves left 
            // or right or does not move in the x dimension
            x_move_direction = (random() % 3) - 1;

            // The thread randomly picks whether the person moves up
            // or down or does not move in the y dimension
            y_move_direction = (random() % 3) - 1;

            // If the person will remain in the bounds of the
            // environment after moving, then
            if((x_locations[current_person_id] + x_move_direction >= 0)
                && (x_locations[current_person_id] 
                    + x_move_direction < environment_width)
                && (y_locations[current_person_id] 
                    + y_move_direction >= 0)
                && (y_locations[current_person_id] 
                    + y_move_direction < environment_height))
            {
                // The thread moves the person
                x_locations[current_person_id] 
                += x_move_direction;
                y_locations[current_person_id] 
                += y_move_direction;
            }
        }
    }   
}

/*
    susceptible()
        For each of the process’s people, each process spawns threads 
        to handle those that are ssusceptible by deciding whether or
        not they should be marked infected.
*/
void susceptible(struct global_t *global, struct const_t *constant, 
    struct stats_t *stats) 
{
    // disease
    int infection_radius = constant->infection_radius;
    int contagiousness_factor = constant->contagiousness_factor;

    // counters
    int current_person_id;
    int num_infected_nearby;
    int my_person;

    // pointers to arrays in global struct
    char *states = global->states;
    int *x_locations = global->x_locations;
    int *y_locations = global->y_locations;
    int *infected_x_locations = global->infected_x_locations;
    int *infected_y_locations = global->infected_y_locations;

    for(current_person_id = 0; current_person_id 
        <= global->number_of_people - 1; current_person_id++)
    {
        // If the person is susceptible, then
        if(states[current_person_id] == SUSCEPTIBLE)
        {
            // For each of the infected people (received earlier 
            // from all processes) or until the number of infected 
            // people nearby is 1, the thread does the following
            num_infected_nearby = 0;
            for(my_person = 0; my_person <= global->num_infected - 1
                && num_infected_nearby < 1; my_person++)
            {
                // If person 1 is within the infection radius, then
                if((x_locations[current_person_id] 
                    > infected_x_locations[my_person] - infection_radius)
                    && (x_locations[current_person_id] 
                        < infected_x_locations[my_person] + infection_radius)
                    && (y_locations[current_person_id]
                        > infected_y_locations[my_person] - infection_radius)
                    && (y_locations[current_person_id]
                        < infected_y_locations[my_person] + infection_radius))
                {
                    // The thread increments the number of infected people nearby
                    num_infected_nearby++;
                }
            }

            // The thread updates stats counter
            #ifdef SHOW_RESULTS
            if(num_infected_nearby >= 1)
                stats->num_infection_attempts++;
            #endif

            // If there is at least one infected person nearby, and 
            // a random number less than 100 is less than or equal 
            // to the contagiousness factor, then
            if(num_infected_nearby >= 1 && (random() % 100) 
                <= contagiousness_factor)
            {
                // The thread changes person1’s state to infected
                states[current_person_id] = INFECTED;
                // The thread updates the counters
                global->num_infected++;
                global->num_susceptible--;
                // The thread updates stats counter
                #ifdef SHOW_RESULTS
                stats->num_infections++;
                #endif
            }
        }
    }
}

/*
    infected()
        For each of the process’s people, each process spawns 
        threads to handle those that are infected by deciding 
        whether they should be marked immune or dead.
*/
void infected(struct global_t *global, struct const_t *constant, 
    struct stats_t *stats)
{
    // disease
    int duration_of_disease = constant->duration_of_disease;
    int deadliness_factor = constant->deadliness_factor;

    // counter
    int current_person_id;

    // pointers to arrays in global struct
    char *states = global->states;
    int *x_locations = global->x_locations;
    int *y_locations =global->y_locations;
    int *num_days_infected = global->num_days_infected;

    for(current_person_id = 0; current_person_id 
        <= global->number_of_people - 1; current_person_id++)
    {
        // If the person is infected and has been for the full 
        // duration of the disease, then 
        if(states[current_person_id] == INFECTED
            && num_days_infected[current_person_id] 
            == duration_of_disease)
        {
            // The thread updates stats counter
            #ifdef SHOW_RESULTS
            stats->num_recovery_attempts++;
            #endif
            // If a random number less than 100 is less than 
            // the deadliness factor, then 
            if((random() % 100) < deadliness_factor)
            {
                // The thread changes the person’s state to dead
                states[current_person_id] = DEAD;
                // The thread updates the counters
                global->num_dead++;
                global->num_infected--;
                // The thread updates stats counter
                #ifdef SHOW_RESULTS
                    stats->num_deaths++;
                #endif
            }
            // Otherwise, 
            else
            {
                // The thread changes the person’s state to immune 
                states[current_person_id] = IMMUNE;
                // The thread updates the counters
                global->num_immune++;
                global->num_infected--;
            }
        }
    }
}

/*
    update_days_infected()
        For each of the process’s people, each process spawns 
        threads to handle those that are infected by increasing
        the number of days infected.
*/
void update_days_infected(struct global_t *global, struct const_t *constant)
{
    // counter
    int current_person_id;

    // pointers to arrays in global struct
    char *states = global->states;
    int *num_days_infected = global->num_days_infected;

    for(current_person_id = 0; current_person_id 
        <= global->number_of_people - 1; current_person_id++)
    {
        // If the person is infected, then
        if(states[current_person_id] == INFECTED)
        {
            // Increment the number of days the person has been infected
            num_days_infected[current_person_id]++;
        }
    }
}
#endif