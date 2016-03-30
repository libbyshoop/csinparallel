/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized the original code) */

#ifndef PANDEMIC_CORE_H
#define PANDEMIC_CORE_H

#include <unistd.h>             // for random

void        move(struct our_t *our, struct const_t *constant);
void        susceptible(struct global_t *global, struct our_t *our, struct const_t *constant, 
                struct stats_t *stats);
void        infected(struct our_t *our, struct const_t *constant, struct stats_t *stats);
void        update_days_infected(struct our_t *our, struct const_t *constant);

/*
    move()
        For each of the process’s people, each process spawns 
        threads to move everyone randomly
*/
void move(struct our_t *our, struct const_t *constant)
{
    // counter
    int my_current_person_id;

    // movement
    int my_x_move_direction;
    int my_y_move_direction;

    // display envrionment variables
    int environment_width = constant->environment_width;
    int environment_height = constant->environment_height;

    // arrays in our struct
    char *our_states = our->our_states;
    int *our_x_locations = our->our_x_locations;
    int *our_y_locations = our->our_y_locations;

    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
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
        For each of the process’s people, each process spawns threads 
        to handle those that are ssusceptible by deciding whether or
        not they should be marked infected.
*/
void susceptible(struct global_t *global, struct our_t *our, struct const_t *constant, 
    struct stats_t *stats) 
{
    // disease
    int infection_radius = constant->infection_radius;
    int contagiousness_factor = constant->contagiousness_factor;
    int total_num_infected = global->total_num_infected;

    // counters
    int my_current_person_id;
    int my_num_infected_nearby;
    int my_person2;

    // pointers to arrays in global struct
    int *their_infected_x_locations = global->their_infected_x_locations;
    int *their_infected_y_locations = global->their_infected_y_locations;

    // pointers to arrays in our struct
    int *our_x_locations = our->our_x_locations;
    int *our_y_locations = our->our_y_locations;
    char *our_states = our->our_states;

    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */

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

            // The thread updates stats counter
            #ifdef SHOW_RESULTS
            if(my_num_infected_nearby >= 1)
                stats->our_num_infection_attempts++;
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
                our->our_num_infected++;
                our->our_num_susceptible--;
                // The thread updates stats counter
                #ifdef SHOW_RESULTS
                stats->our_num_infections++;
                #endif
            }
        }
    }
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
}

/*
    infected()
        For each of the process’s people, each process spawns 
        threads to handle those that are infected by deciding 
        whether they should be marked immune or dead.
*/
void infected(struct our_t *our, struct const_t *constant, 
    struct stats_t *stats)
{
    // disease
    int duration_of_disease = constant->duration_of_disease;
    int deadliness_factor = constant->deadliness_factor;

    // counter
    int my_current_person_id;

    // pointers to arrays in our struct
    char *our_states = our->our_states;
    int *our_x_locations = our->our_x_locations;
    int *our_y_locations = our->our_y_locations;
    int *our_num_days_infected = our->our_num_days_infected;

    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */

    for(my_current_person_id = 0; my_current_person_id 
        <= our->our_number_of_people - 1; my_current_person_id++)
    {
        // If the person is infected and has been for the full 
        // duration of the disease, then 
        if(our_states[my_current_person_id] == INFECTED
            && our_num_days_infected[my_current_person_id] == duration_of_disease)
        {
            // The thread updates stats counter
            #ifdef SHOW_RESULTS
                stats->our_num_recovery_attempts++;
            #endif
            // If a random number less than 100 is less than 
            // the deadliness factor, then 
            if((random() % 100) < deadliness_factor)
            {
                // The thread changes the person’s state to dead
                our_states[my_current_person_id] = DEAD;
                // The thread updates the counters
                our->our_num_dead++;
                our->our_num_infected--;
                // The thread updates stats counter
                #ifdef SHOW_RESULTS
                    stats->our_num_deaths++;
                #endif
            }
            // Otherwise,
            else
            {
                // The thread changes the person’s state to immune
                our_states[my_current_person_id] = IMMUNE;
                // The thread updates the counters
                our->our_num_immune++;
                our->our_num_infected--;
            }
        }
    }

    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
}

/*
   update_days_infected()
        For each of the process’s people, each process spawns 
        threads to handle those that are infected by increasing
        the number of days infected.
*/
void update_days_infected(struct our_t *our, struct const_t *constant)
{
    // counter
    int my_current_person_id;

    // pointers to arrays in our struct
    char *our_states = our->our_states;
    int *our_num_days_infected = our->our_num_days_infected;

    /* For Line Up Purpos */
    /* For Line Up Purpos */
    /* For Line Up Purpos */
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