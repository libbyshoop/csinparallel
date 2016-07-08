***********
Go Solution
***********

In the complete archive, :download:`dd.tar.gz <../code/dd.tar.gz>`, this example is under the dd/go directory.

Alternatively, for this chapter, this is the individual file to download:

:download:`dd_go.go <../code/dd/go/dd_go.go>`

You will also need to refer to the C++11 threads solution, found in the dd/threads directory in the full archive or available individually:

:download:`dd_threads.cpp <../code/dd/threads/dd_threads.cpp>`

Google’s Go language makes it possible to program with implicitly launched threads, and its channel feature enables simplified thread-safe processing of shared data.
 
We will compare the “map” stage in the Go implementation to the “map” stage in the C++11 thread code. The segment of ``main()`` in dd_go.go that implements the “map” stage appears below.
 
	.. code-block:: go
		:emphasize-lines: 1,3,6,8

		pairs := make(chan Pair, 1024)
		for i := 0; i < *nCPU; i++ {
    		go func() {
				p := []byte(*protein)
				for l := range ligands {
					pairs <- Pair{score(l, p), l}
				}
			}()
		}
 
Instead of a vector of ``Pair`` as in dd_threads.cpp, the Go implementation creates a *channel* object called ``pairs`` for communicating ``Pair`` objects through message passing. The “map” stage will send ``Pair``\ s into the channel ``pairs``, and the sorting stage will receive those ``Pair``\ s from that same channel. In effect, channel ``pairs`` behaves like a queue, in which the send operation (\ ``<-``\ ) functions like ``push_back`` and the receive operation (also ``<-``, but with the channel on the right side; not shown in the snippet above) acts like ``pop``. 

The C++11 threads implementation allocated an array ``pool`` of threads, then had each thread call ``do_Map()`` in order to carry out that thread’s work in the “map” stage. The following code from dd_threads.cpp  accomplished these operations.

	.. code-block:: cpp
		:emphasize-lines: 1-3

		  thread *pool = new thread[nthreads];
		  for (int i = 0;  i < nthreads;  i++)
		    pool[i] = thread(&MR::do_Maps, this);

 
Instead of explicitly constructing and storing threads, the Go implementation uses the construct
 
	.. code-block:: go
		:emphasize-lines: 1,3
	
		go func() {
					…
			}()
 
This ``go`` statement launches threads that each execute an (anonymous) function to do their work, i.e., carry out the (omitted) instructions indicated by the ellipses … (in essence, these instructions carry out the work corresponding to ``do_Maps``). Note that we could also have defined that as a function ``foo()`` elsewhere and called it this way (i.e., ``go foo()``\ ), but Go is able to employ anonymous functions because it is garbage-collected.

In the C++11 threads implementation, the threads must retrieve ligand values repeatedly from a queue ``ligands`` and then append the retrieved ligand and its score to the vector ``pairs``\ . The methods ``do_Maps()`` and ``Map()`` in our C++11 threads implementation accomplish these steps; their code could be combined into something like this:

	.. code-block:: cpp
		:emphasize-lines: 4-5

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

		p := []byte(*protein)
			for l := range ligands {
				pairs <- Pair{score(l, p), l}
			}
 
Here, a goroutine obtains its ligand work tasks from a channel ``ligands`` (created and filled during the “task generation” stage), similarly to the work queue ``tasks`` in the C++11 threads implementation. Also, that ligand and its score are sent to the channel ``pairs`` discussed above. 
 
Further Notes
#############

- The use of Go’s channel feature made some key parts of the Go code more concise, as seen above. For example, highlighted sections above show that we needed fewer lines of (arguably) less complex code to process a ligand and produce a ``Pair`` in the Go code than in the C++11 threads code. Also, the Go runtime manages thread creation implicitly, somewhat like OpenMP, whereas we must allocate and manage C++11 threads explicitly.

- Using channels also simplified the synchronization logic in our Go implementation. 

	- We used (thread-safe) Go channels in place of the task queue ``tasks`` and the vector of Pair ``pairs`` to manage the flow of our data. Reasoning with the send and receive operations on channels is at least as easy as reasoning about queue and vector operations.
	
	- The C++11 threads implementation used TBB ``concurrent_bounded_queue`` instead of ``concurrent_queue`` because of the availability of a blocking ``pop()`` operation, so that one could modify dd_threads.cpp to include dynamic ligand generation in a straightforward and correct way, and used a value ``SENTINEL`` to detect when ligands were actually exhausted. Go channels provide these features in a simpler and readily understood way. 

- Just after the “map” stage, the Go implementation stores all Pairs in the channel ``pairs`` into an array for sorting. We cannot store into that array directly during the parallel “map” stage, since that array is not thread-safe.



Questions for exploration
#########################

- Compile and run dd_go.go, and compare its performance to dd_serial.cpp and to other parallel implementations.

- For further ideas, see exercises for other parallel implementations.