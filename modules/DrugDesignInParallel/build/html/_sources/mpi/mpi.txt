*******************************************
A Message Passing Interface (MPI) Solution
*******************************************

In the complete archive, :download:`dd.tar.gz <../code/dd.tar.gz>`, this example is under the dd/MPI directory.

Alternatively, for this chapter, these are the individual files to download:

:download:`dd_mpi.cpp <../code/dd/MPI/dd_mpi.cpp>`

:download:`Makefile <../code/dd/MPI/Makefile>`

The Makefile is for use on linux systems.

A cluster system
################

Now we turn to a solution for use on clusters of computer systems. Because each computer in the cluster is a standalone machine, the work will need to be coordinated by distributing it across the machines in separate processes and ccommunicating between those processes using message passing, which is provided by the MPI library.

Single Program, Multiple Data
#############################

This program uses the master-worker strategy within a single program. This strategy is implemented within the ``run`` method of the ``MR`` class. One process, called the root, or master, has the responsibility for:

- generating all of the ligand scoring tasks
- sending the next available ligand scoring task to a worker when asked
- cordinating when all scoring tasks have been completed

All of the other processes, called workers, will be responsible for:

- asking the master for some work by sending that process a message
- receiving the work and computing the score for that ligand

In the code, the separation of these tasks is done by keeping track of the *rank* of each process (a unique number given to each process that was initialized) and using it to determine what it will do. In this code, this line indicates the section of code for the master (by tradition number 0):

	.. code-block:: cpp

		if (rank == root) {

and the else block corresponding to this if statement holds the code that each worker process on other machines will execute. This way of indicating code for different types of processes (master and worker) within the same program is commonly refered to as the single-program, multiple data software pattern in parallel and distributed computing. The MPI library was designed to use this pattern in this manner.

In a cluster, memory is not shared between all processes, so not every worker process running on a different machine will have a copy of the vector containing the pairs of processed ligands and their scores. This will be maintained by the master, or root process. Because of this, the Map function found in previous examples that looked like this:

	.. code-block:: cpp

		void MR::Map(const string &ligand, tbb::concurrent_vector<Pair> &pairs) {
		  Pair p(Help::score(ligand.c_str(), protein.c_str()), ligand);
		  pairs.push_back(p);
		}

now must be split up between the workers, who will do the scoring, and the master, who will take on the task of pushing the result received from each worker back onto the vector. Take note of where the score method is called in the worker portion of the ``run`` function of the MR class, and the result is sent to the master process. Then note where that score is received in the master process section of the code and pushed onto the pairs vector.

Questions for Explortion
########################

- Compile and run the code on a cluster (using mpirun). Generally speaking, does it seem faster for a given set of problem sizes (number of ligands, size of input protein string). As you add processes, does it seem to get faster?

- Investigate how to time how long the code takes to run, using a function called *MPI_get_wtime()*. Improve this code by adding the capability to determine its running time and report it with the results.

- What issues arise with timing code like this when the ligands are randomly generated?




