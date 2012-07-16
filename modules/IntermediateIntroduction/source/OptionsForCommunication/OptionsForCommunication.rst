******************************
Some Options For Communication
******************************

In simple data parallelism, it may not be necessary for parallel computations to share data with each other during the executions of their programs. However, most other forms of concurrency require communication between parallel computations.  Here are three options for communicating between various processes/threads running in parallel.

1. **message passing** - communicating with basic operations **send** and **receive** to transmit information from one computation to another.

2. **shared memory** - communicating by reading and writing from local memory locations that are accessible by multiple computations

3. **distributed memory** - some parallel computing systems provide a service for sharing memory locations on a remote computer system, enabling non-local reads and writes to a memory location for communication.

:Comments:

    For distributed systems, message passing and distributed memory (if available) may be used. All three approaches may be used in concurrent programming in multi-core parallelism. However, shared (local) memory access typically offers an attractive speed advantage over message passing and remote distributed memory access.

When multiple processes or threads have both read and write access to a memory location, there is potential for a **race condition**, in which the correct behavior of the system depends on timing. (Example: filling a shared array, with algorithms for accessing and updating a variable *nextindex*.)

* **resource** - a hardware or software entity that can be allocated to a process by an operating system

:Comments: 

    A memory location is an example of a (OS) resource; other examples are: files; open files; network connections; disks; GPUs; print queue entries. Race conditions may occur around the handling of any OS resource, not just memory locations.

* In the study of Operating Systems, **inter-process communication (IPC)** strategies are used to avoid problems such as race conditions. Message passing is an example of an IPC strategy. (Example: solving the array access problem with message passing.) 

* Message passing can be used to solve IPC problems in a distributed system. Other common IPC strategies (semaphores, monitors, etc.) are designed for a single (possibly multi-core) computer.





