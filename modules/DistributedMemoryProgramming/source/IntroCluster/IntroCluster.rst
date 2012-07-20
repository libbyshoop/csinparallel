Introduction to Cluster
=======================

**Definition**: "A cluster is a type of parallel or distributed processing system, which consists of a collection of interconnected stand-alone computers cooperatively working together as a single, integrated computing resource." - by Rajkumar Buyya

A cluster is usually a linux-based operating system. Basically, a cluster has four major parts:
	
	- Network: Provides communications between nodes, server, and gateway.
	- Nodes: Each node has its own processor, memory, and storage.
	- Server: Provides network services to the cluster.
	- Gateway: Acts as a firewall between the cluster and outside world.

In order to prepare for what you will be working on, you should have a good understanding of parallel computer architectures. We are going to look at two parallel computer architectures: 

	- Shared Memory Model
	- Distributed Memory Model

General Characteristics of Shared Memory Model:
	
	"
	
	- Shared memory parallel computers vary widely, but generally have in common the ability for all processors to access all memory as global address space.
	
	- Multiple processors can operate independently but share the same memory resources.
	
	- Changes in a memory location effected by one processor are visible to all other processors." [1]

.. image:: images/SharedMemoryUMA.png
	:width: 350px
	:align: center
	:height: 250px
	:alt: MPI Structure

.. centered:: Figure 1: Shared Memory: Uniform Memory Access Obtained from www.computing.llnl.gov [2]

General Characteristics of Distributed Memory Model:

	"

	- Distributed memory systems require a communication network to connect inter-processor memory.
	
	- Processors have their own local memory. Memory addresses in one processor do not map to another processor, so there is no concept of global address space across all processors.
	
	- Because each processor has its own local memory, it operates independently. Changes it makes to its local memory have no effect on the memory of other processors. Hence, the concept of cache coherency does not apply.
	
	- When a processor needs access to data in another processor, it is usually the task of the programmer to explicitly define how and when data is communicated. Synchronization between tasks is likewise the programmer's responsibility." [3]

.. image:: images/DistributedMemory.png
	:width: 450px
	:align: center
	:height: 200px
	:alt: MPI Structure

.. centered:: Figure 2: Distributed Memory System Obtained from www.computing.llnl.gov [4]

Some benefits of using clusters are:

	- Inexpensive: Hardware and software of a cluster cost significantly much less than those of a supercomputer.
	- Scalability: Able to scale without impact on performance.
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
	
	* Please read `Cluster Computing: High-Performance, High-Availability, and High-Throughput Processing on a Network of Computers <http://www.cloudbus.org/papers/ic_cluster.pdf>`_ [5]. 

	* Case Studies on Cluster Applications: read from page 16 - 22.


In order to use a cluster effectively, we need to have some programming environments such as Message Passing Interface (MPI), and OpenMP, etc. In this module, we will be learning about MPI on distributed memory cluster.


.. rubric:: References

.. [1] https://computing.llnl.gov/tutorials/parallel_comp/#SharedMemory
.. [2] https://computing.llnl.gov/tutorials/parallel_comp/#SharedMemory
.. [3] https://computing.llnl.gov/tutorials/parallel_comp/#DistributedMemory
.. [4] https://computing.llnl.gov/tutorials/parallel_comp/#DistributedMemory
.. [5] Chee Shin Yeo, Rajkumar Buyya, Hossein Pourreza, Rasit Eskicioglu, Peter Graham, and Frank Sommers, "Cluster Computing: High-Performance, High-Availability, and High-Throughput Processing on a Network of Computers", in Handbook of Nature-Inspired and Innovative Computing: Integrating Classical Models with Emerging Technologies, chapter 16, page 521 - 551, 2006
