The Three Musketeers
=====================

Another Heterogeneous programming model that we can look at is the parallel programming technique that combines MPI, CUDA, and OpenMP. Each of the parallel programming models uses different resource for their computation. OpenMP works on Multicore CPUs, CUDA works on GPUs, and MPI works on distributed memory cluster. Therefore, if you have a cluster, whose nodes contain more than one core, and at least a GPU, you can try this parallel programming method. However, in this last section of the module we just want introduce the idea of having this hardcore parallelism. 

.. topic:: Recommended Reading

	* Please read `Scalability of Incompressible Flow Computations on Multi-GPU Clusters Using Dual-Level and TriLevel Parallelism <http://scholarworks.boisestate.edu/cgi/viewcontent.cgi?article=1010&context=mecheng_facpubs>`_.
