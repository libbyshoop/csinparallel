**************************
Parallelizing the C++ Code
**************************

Once you've got a functional sequential epidemic model, take a look at some of these strategies for making it run in parallel. This page describes parallelizing your sequential code by adding OpenMP and MPI code to it.

OpenMP
######

- parallelize the ``for`` loop which loops through ``Population``

- when the final conditions are evaluated, perform a reduction to count the number of Infected, Susceptible, and Recovered ``Person``\ s. 


MPI
###
- Divide ``Population`` equally among the number of threads being used. Similarly, infect ``Person``\ s equally across the threads. 

- Have each thread determine the position of each infected ``Person`` on that thread. Send the position information of the infected ``Person``\ s to all other threads. (Use an ``Allgather`` MPI function call to collect position arrays from each thread to be stored on a larger position array. This larger position array is present on each thread and holds the positions of each infected person.)

- Run through the simulation on each thread as normal, but using the larger position array to check the position of each susceptible ``Person`` against the position of each infected person.
