/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized the original code and added CUDA Programming) */

#ifndef PANDEMIC_INFECTION_H
#define PANDEMIC_INFECTION_H

#ifdef _MPI
#include <mpi.h>        // for MPI_Finalize()
#endif

void        find_infected(struct our_t *our);
void        share_infected(struct global_t *global, struct our_t *our);

/*
    find_infected()
        Each process determines its infected x locations 
        and infected y locations
*/
void find_infected(struct our_t *our)
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
        Each process sends its infecion information to all
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
    // if not using MPI
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



#endif