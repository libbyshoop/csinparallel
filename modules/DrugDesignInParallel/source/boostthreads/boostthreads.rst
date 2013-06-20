**********************
Boost Threads Solution
**********************

Boost threads
The implementation dd_boost.cpp uses Boost threads, with TBB thread-safe containers.

In the OpenMP implementation above, the OpenMP runtime system implicitly creates and manages threads for us. The implementation :download:`dd_boost.cpp <code/dd_boost.cpp>` parallelizes the computationally expensive ``Map()`` stage by using Boost threads (precursors of the new C++11 standard threads) instead of OpenMP.  This requires us to explicitly create and manage our own threads, using a master-worker parallel programming pattern driven by a task queue  tasks  produced by ``Generate_tasks()``.

We will examine the Boost threads implementation dd_boost.cpp by comparing it to the sequential implementation :download:`dd_serial.cpp <code/dd_serial.cpp>`. 
The main routine for map-reduce computing is ``MR::run()``, and this routine is identical in the two implementations except for the “map” stage and for ``dd_boost.cpp`` handling an extra argument ``nthreads``. In the serial implementation, the “map” stage simply removes elements from the task queue and calls ``Map()`` for each such element, via the following code.

.. code-block:: cpp
	:emphasize-lines: 2
	:linenos:

	while (!tasks.empty()) {
		Map(tasks.front(), pairs);
		tasks.pop();
	}

However, the Boost threads implementation of the “map” stage creates an array ``pool`` of Boost threads to perform the calls to ``Map()``, then waits for those threads to complete their work by calling the ``join()`` method for each thread.

.. code-block:: cpp
	:emphasize-lines: 1,3,7
	:linenos:
	
	boost::thread *pool = new boost::thread[nthreads];
	for (int i = 0;  i < nthreads;  i++) {
		pool[i] = boost::thread(boost::bind(&MR::do_Maps, this));
	}

	for (int i = 0;  i < nthreads;  i++) {
		pool[i].join();
	}

Here, we define the function ``MR::do_Maps()`` for performing calls to ``Map()`` as follows:


.. code-block:: cpp
	:emphasize-lines: 3,5
	:linenos:

	void MR::do_Maps(void) {
		string lig;
		tasks.pop(lig);
		while (lig != SENTINEL) {
			Map(lig, pairs);
			tasks.pop(lig);
		}
	tasks.push(SENTINEL);  // restore end marker for another thread
	}

This method ``do_Maps()`` serves as the “main program” for each thread, and that method repeatedly pops a new ligand string ``lig`` from the task queue, and calls ``Map()`` with ``lig`` until it encounters the end marker ``SENTINEL``\ .  

Since multiple threads may access the shared task queue ``tasks`` at the same time, that task queue must be thread-safe, so we defined it using a TBB container:

``tbb::concurrent_bounded_queue<string> tasks;``

We chose ``tbb::concurrent_bounded_queue`` instead of ``tbb::concurrent_queue`` because the bounded type offers a blocking ``pop()`` method, which will cause a thread to wait until work becomes available in the queue; also, we do not anticipate a need for a task queue of unbounded size. Blocking on the task queue isn’t actually necessary for our simplified application, because all the tasks are generated before any of the threads begin operating on those tasks. However, this blocking strategy supports a dynamic task queue, in which new tasks can be added to the queue while other tasks are being processed, a requirement that often arises in other applications. 

Further Notes
#############

- The ``SENTINEL`` task value indicates that no more tasks remain. Each thread consumes one ``SENTINEL`` from the task queue so it can know when to exit, and adds one ``SENTINEL`` to the task queue just before that thread exits, which then enables another thread to finish.

- As with :download:`dd_omp.cpp <code/dd_omp.cpp>`, the Boost threads implementation uses a thread-safe vector for storing the key-value pairs produced by calls to ``Map()``, 

	``tbb::concurrent_vector<Pair> pairs;``

since multiple threads might access that shared vector at the same time.  


Questions for exploration
#########################

- Compile and run :download:`dd_boost.cpp <code/dd_boost.cpp>`, and compare its performance to :download:`dd_serial.cpp <code/dd_serial.cpp>` and to other parallel implementations.

- *Concurrent task queue:* consider the “map” stage in our sequential implementation ``dd_serial.cpp``\ , which uses an STL container instead of a TBB container for the task queue ``tasks``\ :

	.. code-block:: cpp
		:linenos:

		while (!tasks.empty()) {
			Map(tasks.front(), pairs);
			tasks.pop();
		}

	#. TBB container classes ``tbb::concurrent_queue`` and ``tbb::concurrent_bounded_queue`` do not provide a method ``front()``.  Instead, they provide a method ``try_pop()`` with one argument, that retrieves one value from the front of the queue and removes it from that queue if there is such an element, and that returns immediately (non-blocking) without making any changes if there is no element in that queue.  This accomplishes the work of an STL queue’s ``front()`` and ``pop()`` methods in a single operation.

	#. Describe a parallel computing scenario in which a single (atomic) operation ``try_pop()`` is preferable to separate operations ``front()`` and ``pop()``, and explain why we should prefer it.

	#. Given that we choose a TBB queue container for the type of ``tasks``, would it be safe to have multiple threads execute the following code (which more closely mirrors our sequential operation)?

		.. code-block:: cpp
			:linenos:

			string lig;  
			while (!tasks.empty()) {
				tasks.try_pop(lig);
				Map(lig, pairs);
			}
	
		If it’s safe, explain how you know it is safe.  If something can go wrong with this code, describe a scenario in which it fails to behave correctly.

	#. The purpose of ``SENTINEL`` in our Boost threads implementation is to insure that every (non-\ ``SENTINEL``\ ) element in the task queue ``tasks`` is processed by some thread, and that all threads terminate (return from ``do_Maps()``\ ) when no more (non-\ ``SENTINEL``\ ) elements are available. Verify that this purpose is achieved in ``dd_boost.cpp``\ , or describe a scenario in which that purpose fails.  

	#. Revise ``dd_boost.cpp`` to use a ``tbb::concurrent_queue`` container instead of a ``tbb::concurrent_bounded_queue`` container for the task queue ``tasks``.  

	:Notes: 
		- ``tbb::concurrent_queue`` does not provide the blocking method ``pop()`` used in ``dd_boost.cpp``\ , so some other synchronization strategy will be required. 

		- However, in our simplified problem, the task queue ``tasks`` doesn’t change during the “map” stage, so threads may finish once ``tasks`` becomes empty.

		- Be sure to understand the concurrent task queue exercise above (italicized) before attempting this exercise.

		- Is a ``SENTINEL`` value needed for your solution? See also exercises for other implementations. **WHAT DOES THIS MEAN??**


**ALSO WHY DO THESE SWITCH FROM - TO #? DO THEY NEED TO? ARE THEY MEANT TO BE NESTED?**
