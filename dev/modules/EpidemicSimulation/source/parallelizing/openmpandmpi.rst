**************************
Parallelizing the C++ Code
**************************

Once you've got a functional sequential epidemic model, take a look at some of these strategies for making it run in parallel. This page describes parallelizing your sequential code by adding OpenMP and MPI code to it. Try running versions of the code that just implement one or the other, then try a hybrid that includes both!

OpenMP
######

OpenMP gives us a set of preprocessor directives that can be used to run multiple threads. If you've never used it before, there's a great `tutorial`_ available through UIUC called Introduction to OpenMP. Otherwise, the links below can provide more specific guidance or a syntax refresher for those already familiar with OpenMP.

- Use ``#pragma omp parallel for`` to parallelize the ``for`` loop that loops through ``Population``. Experiment with setting different numbers of threads! `More information`_.

- When the final conditions are evaluated, perform a `reduction`_ to count the number of Infected, Susceptible, and Recovered ``Person``\ s. 

.. _tutorial: http://www.citutor.org/

.. _More information: http://msdn.microsoft.com/en-us/library/6z19s8e0.aspx

.. _reduction: http://msdn.microsoft.com/en-us/library/2etkydkz.aspx

MPI
###

MPI specifies a set of directives for doing shared-memory concurrency using message passing. The link above also contains a very thorough tutorial on MPI.

- Divide ``Population`` equally among the number of threads being used. Similarly, infect the first ``initialInfected`` of them equally across the threads. 

- Have each thread determine the position of each infected ``Person`` on that thread. Send the position information of the infected ``Person``\ s to all other threads. Use an ``Allgather`` MPI function call to collect position arrays from each thread to be stored on a larger position array. This larger position array is present on each thread and holds the positions of each infected person (see more `details here`_).

- Run through the simulation on each thread as normal, but using the larger position array to check the position of each susceptible ``Person`` against the position of each infected person.

.. _details here: http://www.mpitutorial.com/mpi-scatter-gather-and-allgather/

