******************************
Evaluating the Implementations
******************************

Strategic simplifications of the problem
########################################

We consider the effects of some of the simplifying choices we have made.

- Our string-comparison algorithm for the “map” stage only vaguely suggests the chemistry computations of an actual docking algorithm.  However, the computational complexity properties of our representative algorithm allow us to generate lengthy computation time by increasing the length of ligands (and having a long protein).  

- Our implementations generate all of the candidate ligands before proceeding to process any of them.  As mentioned in exercises, it might be reasonable to generate new ligands as a result of processing.  The implementations :download:`dd_serial.cpp <code/dd_serial.cpp>` and :download:`dd_boost.cpp <code/dd_boost.cpp>` use a queue of ligands to generate the “map” stage work, and could be adapted to enable new ligands to be generated while others are being processed. We could also modify the Go implementation :download:`dd_go.go <code/dd_go.go>` similarly, since we could dynamically add new ligands to the channel ``ligands``.  

- The amount of time it takes to process a ligand depends greatly on its length. This sometimes shows up in tests of performance: testing a *few* more ligands might require a *great deal* more time to compute. This may or may not fit with the computational pattern of a realistic docking algorithm. If one wants to model more consistent running time per ligand, the minimum length of ligands could be raised or lengths of ligands could be held constant.

The impact of scheduling threads
################################

The way we schedule work for threads in our various parallel implmenentations may have a sizable impact on running time, since different ligands may vary greatly in computational time in our simplified model. 

- By default, OpenMP’s ``omp parallel for``, as used by :download:`dd_omp.cpp <code/dd_omp.cpp>`, presumably divides the vector of ligands into roughly equal segments, one per thread. With small ``nligands``\ , if one segment contains more lengthy ligands than another, it may disproportionately extend the running time of the entire program, with one thread taking considerably longer than the others. With large ``nligands``, we expect less variability in the computational load for the threads.

- In our Boost thread implementation :download:`dd_boost.cpp <code/dd_boost.cpp>`, each thread draws a new ligand to process as soon as it finishes its current ligand. Likewise, the Go code :download:`dd_go.go <code/dd_go.go>` draws ligands from the channel named  ligands . This scheduling strategy should have a load-balancing effect, unless a thread draws a long ligand late in the “map” stage. One might try reordering the generated ligands in order to achieve better load balancing. For example, if ligands were sorted from longest to shortest before the “map” stage in the Boost thread implementation, the amount of imbalance of loads is limited by the shortness of the final ligands.  

Barriers to performance improvement
###################################

The degree of parallelism in these implementations is theoretically limited by the implicit barrier after each stage of processing.  

- In all of our implementations, the task generation stage produces all ligands before proceeding to any the “map” stage.  In a different algorithm, parallel processing of ligands might begin as soon as those ligands appear in the task queue. We wouldn’t expect much speedup from this optimization in our example, since generating a ligand requires little time, but generation of tasks might take much longer in other problems, and allowing threads to process those tasks sooner might increase performance in those cases.

- The “map” stage produces all key-value pairs before those pairs are sorted and reduced. This barrier occurs implicitly when finishing the ``omp parallel for`` pragma in our OpenMP implementation :download:`dd_omp.cpp <code/dd_omp.cpp>`, and as part of the map-reduce framework Hadoop used by :download:`dd_hadoop.java <code/dd_hadoop.java>`. That barrier appears explicitly in the loop of ``join()`` calls in our Boost threads code :download:`dd_boost.cpp <code/dd_boost.cpp>`. At the point of the barrier, some threads (or processes) have nothing to do while other threads complete their work.  

- Perhaps threads that finish early could help carry out a parallel sort of the pairs, for better thread utilization, but identifying and implementing such as sort takes us beyond the scope of this example.

- The other stages are executed sequentially, and the “barrier” after each of those stages has no effect on computation time.

The convenience of a framework
##############################

Using a map-reduce framework such as Hadoop enables a programmer to reuse an effective infrastructure of parallel code, for greater productivity and reliability. A Hadoop implementation hides parallel algorithm complexities such as managing the granularity, load balancing, collective communication, synchronization to maintain the thread-safe task queue, are common to map-reduce problems and easily represented in a general map-reduce framework. Also, the fault-tolerance properties of Hadoop makes that tool scalable for computing with extremely large data on very large clusters.  
