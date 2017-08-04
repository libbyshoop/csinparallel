



First, two parallel programming models
---------------------------------------

To prepare for what you will be working on, you need a basic understanding of parallel computer architectures. In particular, it is useful to know the difference between these two parallel computer architectures:

	- Shared Memory Model
	- Distributed Memory Model

General Characteristics of Shared Memory Model:

"
	- Shared memory parallel computers vary widely, but generally have in common the ability for all processors to access all memory as global address space.

	- Multiple processors can operate independently but share the same memory resources.

	- Changes in a memory location effected by one processor are visible to all other processors." [2]_

.. image:: images/SharedMemoryUMA.png
	:width: 350px
	:align: center
	:height: 250px
	:alt: Shared Memory architecture

.. centered:: Figure 1: Shared Memory: Uniform Memory Access Obtained from www.computing.llnl.gov [3]_


The rest of this course module is primarily focused on the distributed memory model of computing, which is different from the shared memory model.

According to [4]_, the general characteristics of Distributed Memory Model are:


	- Distributed memory systems require a communication network to connect inter-processor memory.

	- Processors have their own local memory. Memory addresses in one processor do not map to another processor, so there is no concept of global address space across all processors.

	- Because each processor has its own local memory, it operates independently. Changes it makes to its local memory have no effect on the memory of other processors. Hence, the concept of cache coherency does not apply.

	- When a processor needs access to data in another processor, it is usually the task of the programmer to explicitly define how and when data is communicated. Synchronization between tasks is likewise the programmer's responsibility.


.. image:: images/DistributedMemory.png
	:width: 450px
	:align: center
	:height: 200px
	:alt: MPI Structure

.. centered:: Figure 2: Distributed Memory System Obtained from www.computing.llnl.gov [5]_

Clusters of Computers
---------------------

Distributed Memory systems often manifest themseleves in the form of clusters of computers networked together over a high-speed network. Clusters of workstations connected through a highspeed switch are often called beowulf clusters.

**Definition**: A cluster is a type of parallel or distributed processing system, which consists of a collection of interconnected stand-alone computers cooperatively working together as a single, integrated computing resource. [1]_

A cluster is usually a linux-based operating system. Basically, a cluster has four major components:

	- Network is to provide communications between nodes and server.
	- Each node has its own processor, memory, and storage.
	- Server is to provide network services to the cluster.
	- Gateway acts as a firewall between the outside world and the cluster.


Some benefits of using clusters are:

	- Inexpensive: Hardware and software of a cluster cost significantly much less than those of a supercomputer.
	- Scalability: extra nodes can be added to a cluster when work exceeds the capacities of the current system in the cluster.
	- Maintenance: A cluster is relatively easy to set up and maintain.
	- High Performance: Operations should be optimized and efficient.
	- Great capacity: Ability to solve a larger problem size.

There are many applications of clustering such as:

	- Scientific computation
	- Parametric Simulations
	- Database Applications
	- Internet Applications
	- E-commerce Applications

.. topic:: Recommended Reading:

	* Please read `Cluster Computing: High-Performance, High-Availability, and High-Throughput Processing on a Network of Computers <http://www.cloudbus.org/papers/ic_cluster.pdf>`_ [6]_.

	* Case Studies on Cluster Applications: read from page 16 - 22.

In order to use a cluster effectively, we need to have some programming environments such as Message Passing Interface (MPI), and OpenMP, etc. In this module, we will be learning about MPI on distributed memory cluster.


.. rubric:: References

.. [1] Rajkumar Buyya, "High Performance Cluster Computing: Systems and Architectures", Vol. 1, 1/e, Prentice Hall PTR, NJ, 1999.
.. [2] https://computing.llnl.gov/tutorials/parallel_comp/#SharedMemory
.. [3] https://computing.llnl.gov/tutorials/parallel_comp/#SharedMemory
.. [4] https://computing.llnl.gov/tutorials/parallel_comp/#DistributedMemory
.. [5] https://computing.llnl.gov/tutorials/parallel_comp/#DistributedMemory
.. [6] Chee Shin Yeo, Rajkumar Buyya, Hossein Pourreza, Rasit Eskicioglu, Peter Graham, and Frank Sommers, "Cluster Computing: High-Performance, High-Availability, and High-Throughput Processing on a Network of Computers", in Handbook of Nature-Inspired and Innovative Computing: Integrating Classical Models with Emerging Technologies, chapter 16, page 521 - 551, 2006
