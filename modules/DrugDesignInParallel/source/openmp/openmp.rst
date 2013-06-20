***************
OpenMP Solution
***************

Here, we implement our drug design simulation in parallel using OpenMP, an API that providing compiler directives, library routines, and environment variables that allow shared-memory multithreading in C/C++: a master thread forks and assigns parts of a task to a specified number of slave threads (read `more`_).

Implementation
##############

The implementation :download:`dd_omp.cpp <code/dd_omp.cpp>` parallelizes the ``Map()`` loop using OpenMP, and uses a thread-safe container from `TBB`_, a C++ template library designed to help avoid some of the difficulties associated with multithreading.

Since we expect the docking algorithm (here represented by computing a match score for comparing a ligand string to a protein string) to require the bulk of compute time, we will parallelize the ``Map()`` stage in our sequential algorithm :download:`dd_serial.cpp <code/dd_serial.cpp>`\ . 

.. code-block:: c++
	:emphasize-lines: 2
	:linenos:

	while (!tasks.empty()) {
   		Map(tasks.front(), pairs);
   	tasks.pop();
 	}

We will now parallelize that loop by converting it to a ``for`` loop, then applying OpenMP’s ``parallel for`` feature.  Hence, we will replace the ``tasks`` queue with a vector (of the same name) and iterate on index values for that vector.  

However, this causes a potential concurrency problem, because multiple OpenMP threads will now each be calling ``Map()``\ , and those multiple calls by parallel threads may overlap.  There is no potential for error from the first argument ``ligand`` of ``Map()``, since ``Map()`` requires mere read-only access for that argument...but multiple calls of ``Map()`` in different threads might interfere with each other when changing the writable second argument ``pairs`` of ``Map()``, leading to a data race condition. The STL containers are *not* thread safe, meaning that they provide no protection against such interference, which may lead to errors.  

Therefore, we will use TBB’s thread-safe ``concurrent_vector`` container for ``pairs``, leading to the following code segments in our OpenMP implementation.

.. code-block:: cpp
	:emphasize-lines: 2,8,10
	:linenos:

	vector<string> tasks;
	tbb::concurrent_vector<Pair> pairs;
	vector<Pair> results;

	Generate_tasks(tasks);
	// assert -- tasks is non-empty

	#pragma omp parallel for num_threads(nthreads) 
		for (int t = 0;  t < tasks.size();  t++) {
			Map(tasks[t], pairs);
  		}

Since the main thread (i.e., the thread that executes ``run()``\ ) is the only thread that performs the stages that call ``Generate_tasks()``, ``to_sort()``, and ``Reduce()``, it is safe for the vectors ``tasks`` or ``results`` to remain implemented as (non-thread safe) STL containers.  See the implementation :download:`dd_omp.cpp <code/dd_omp.cpp>` for complete details. 

.. _more: http://en.wikipedia.org/wiki/OpenMP 

.. _TBB: http://en.wikipedia.org/wiki/Intel_Threading_Building_Blocks

Further Notes
#############

- Most of the changes between the sequential version and this OpenMP version arise from the change in type for the data member ``MR::pairs``, to a thread-safe data type. A few changes have to do with managing the number of threads to use ``nthreads``. In fact, the one-line ``#pragma`` directive shown above is entirely responsible for specifying parallel computation. Without it, the computation would proceed sequentially.

- This OpenMP implementation has four (optional) command-line arguments.  The third argument specifies the number of OpenMP threads to use (note that this differs from the third argument in the sequential version). In dd_omp.cpp, the command-line arguments have these effects:

	#. maximum length of a (randomly generated) ligand string

	#. number of ligands generated

	#. number of OpenMP threads to request

	#. protein string to which ligands will be compared

Questions for Exploration
#########################

- Compare the performance of ``dd_serial.cpp`` with ``dd_omp.cpp`` on a multicore computer using the same values for ``max_ligand`` and ``nligands``.  Do you observe speedup for the parallel version?  

- Our development system has four cores, and ``nthreads=4`` was used for one of our test runs.  We found that ``dd_omp.cpp`` performed about three times as fast as ``dd_serial.cpp`` for the same values of ``max_ligand`` and ``nligands``.  Can you explain why it didn’t perform four times as fast?

- Use the command-line arguments to experiment with varying the number of OpenMP threads in an invocation of ``dd_omp.cpp``, while holding ``max_ligand`` and ``nligands`` unchanged. On a multi-core system, we hope for better performance when more threads are used.  Do you observe such performance improvement when you time the execution?  What happens when the number of threads exceeds the number of cores (or hyperthreads) on your system?  Explain as much as you can about the timing results you observe when you vary the number of threads.

- You may notice that ``dd_omp.cpp`` computes the same maximal score and identifies the same ligands as ``dd_serial.cpp`` that produce that score, but if more than one ligand yields the maximal score, the *order* of those maximal-scoring ligands may differ between the two versions. Can you explain why? 

- Our sequential program dd_serial.cpp always produces the same results for given values of the ``max_ligand``, ``nligands``,  and ``protein`` command-line arguments.  This is because we use the default random-number seed in our code.  Because of this consistency, we can describe the sequential version as being a *deterministic* computation. Is ``dd_omp.cpp`` a deterministic computation?  Explain your answer, and/or state what more you need to know in order to answer this question.

- If you have *more realistic algorithms for docking* and/or *more realistic data for ligands and proteins*, modify the program ``dd_omp.cpp`` to incorporate those elements, and compare the results from your modified program to results obtained by other means (other software, wet-lab results, etc.).  How does the performance of your modified OpenMP version compare to what you observed from your modified sequential version (if you implemented one)?  

- Whereas our serial implementation used a queue data structure for ``tasks``, this implementation uses a vector data structure, and parallelizes the “map” stage using OpenMP’s ``omp parallel for`` pragma. This suffices for our simplified example, because we generate all ligands before processing any of them. However, some computations require a task queue, since processing some tasks may generate others. (This is not out of the question for drug design, since high-scoring ligands might lead one to consider similar ligands in search of even higher scores.) **Challenge problem:** Modify :download:`dd_omp.cpp <code/dd_omp.cpp>` to use a task queue instead of a task vector.  
	
	.. note:: 
		- Use a thread-safe queue data structure for  tasks , such as ``tbb::concurrent_queue`` or ``tbb::concurrent_bounded_queue``, because multiple threads may attempt to modify that queue at the same time.
		
		- Instead of ``omp parallel for``, use OpenMP 3.0 tasks.  You can parallelize a ``while`` loop that moves through the task queue using ``omp parallel`` enclosing that loop.
		
		- Depending on your algorithm, it may help to use “sentinel” values, as described in Chapter 8 of [Clay’s book], **WHAT IS THIS A REFERENCE TO??** or as used by the Boost threads implementation in the next page.  
