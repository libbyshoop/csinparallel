***********
Go Solution
***********

Google’s Go language makes it possible to program with implicitly launched threads, and its channel feature enables simplified thread-safe processing of shared data.
 
We will compare the “map” stage in the Go implementation :download:`dd_go.go <code/dd_go.go>` to the “map” stage in the Boost thread code :download:`dd_boost.cpp <code/dd_boost.cpp>`. The segment of ``main()`` in ``dd_go.go`` that implements the “map” stage appears below.
 
	.. code-block:: go
		:emphasize-lines: 1,3,6,8
		:linenos:

		pairs := make(chan Pair, 1024)
		for i := 0; i < *nCPU; i++ {
    		go func() {
				p := []byte(*protein)
				for l := range ligands {
					pairs <- Pair{score(l, p), l}
				}
			}()
		}
 
Instead of a vector of ``Pair`` as in ``dd_boost.cpp``\ , the Go implementation creates a channel object called ``pairs`` for communicating ``Pair`` objects through message passing. The “map” stage will send ``Pair``\ s into the channel ``pairs``, and the sorting stage will receive those ``Pair``\ s from that same channel. In effect, that channel ``pairs`` behaves like a queue, in which the ``send`` operation functions like ``push_back`` and the receive operation acts like ``pop``. 

The Boost implementation allocates an array ``pool`` of threads, then has each thread call ``do_Map()`` in order to carry out that thread’s work in the “map” stage. The following code from ``dd_boost.cpp`` accomplishes these operations.

	.. code-block:: cpp
		:emphasize-lines: 1-3
		:linenos:

 		boost::thread *pool = new boost::thread[nthreads];
		for (int i = 0;  i < nthreads;  i++)
			pool[i] = boost::thread(boost::bind(&MR::do_Maps, this));
 
Instead of explicitly constructing and storing threads, the Go implementation uses the construct
 
	.. code-block:: go
		:emphasize-lines: 1,3
		:linenos:
	
		go func() {
					…
			}()
 
This ``go`` statement launches threads that each execute an (anonymous) function to do their work, i.e., carry out the (omitted) instructions indicated by the ellipses … (in essence, these instructions carry out the work corresponding to ``do_Maps``). 

In the Boost threads implementation, the threads must retrieve ligand values repeatedly from a queue  ligands  then append that retrieved ligand and its score to the vector ``pairs``\ . The methods ``do_Maps()`` and ``Map()`` accomplish these steps; their code is equivalent to the following.

	.. code-block:: cpp
		:emphasize-lines: 4-5
		:linenos:

		string lig;
		tasks.pop(lig);
		while (lig != SENTINEL) {
			Pair p(Help::score(ligand.c_str(), protein.c_str()), ligand);
			pairs.push_back(p);
			tasks.pop(lig);
		}
		tasks.push(SENTINEL);  // restore end marker for another thread
    	
In comparison, the goroutines (threads) in the Go implementation carry out the following code.

	.. code-block:: go
		:emphasize-lines: 3
		:linenos:

		p := []byte(*protein)
			for l := range ligands {
				pairs <- Pair{score(l, p), l}
			}
 
Here, a goroutine obtains its ligand work tasks from a channel ``ligands`` (created and filled during the “task generation” stage), similarly to the work queue ``tasks`` in the Boost threads implementation. Also, that ligand and its score are sent to the channel ``pairs`` discussed above. 
 
Further Notes
#############

- The use of Go’s channel feature made some key parts of the Go code more concise, as seen above.  For example, highlighted sections above show that we needed fewer lines of (arguably) less complex code to process a ligand and produce a ``Pair`` in the Go code than in the Boost threads code.  Also, the Go runtime manages thread creation implicitly, somewhat like OpenMP, whereas we must allocate and manage Boost threads explicitly. 

- Using channels also simplified the synchronization logic in our Go implementation. 

	- We used (thread-safe) Go channels in place of the task queue ``tasks`` and the vector of ``Pair`` pairs  to manage the flow of our data.  Reasoning with the ``send`` and ``receive`` operations on channels is at least as easy as reasoning about queue and vector operations.
	
	- The Boost implementation used TBB ``concurrent_bounded_queue`` instead of ``concurrent_queue`` because of the availability of a blocking ``pop()`` operation, so that one could modify ``dd_boost.cpp`` to include dynamic ligand generation in a straightforward and correct way, and used a value ``SENTINEL`` to detect when ligands were actually exhausted.   Go channels provide these features in a simpler and readily understood way. 

- Just after the “map” stage, the Go implementation stores all ``Pair``\ s in the channel ``pairs`` into an array for sorting. We cannot store into that array directly during the parallel “map” stage, since that array is not thread-safe.

