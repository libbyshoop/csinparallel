**************************
Some issues in concurrency
**************************

We will use the Hadoop implementation of map-reduce for clusters as a running example.

**Fault tolerance** is the capacity of a computing system to continue to satisfy its spec in the presence of faults (causes of error) 

:Comments: 
    
    With more parallel computing components and more interactions between them, more faults become possible. Also, in large computations, the cost of restarting a computation may become greater. Thus, fault tolerance becomes more important and more challenging as one increases the use of parallelism. Systems (such as map-reduce) that automatically provide for fault tolerance help programmers of parallel systems become more productive.

**Mutually exclusive access to shared resources** means that at most one computation (process or thread) can access a resource (such as a shared memory location) at a time. This is one of the requirements for correct IPC.  One approach to mutually exclusive access is locking, in which a mechanism is provided for one computation to acquire a "lock" that only one computation may hold at any given time. A computation possessing the lock may then use that lock's resource without fear of interference by another process, then release the lock when done. 

:Comments: 

    Designing computationally correct locking systems and using them correctly for IPC can often be quite tricky.

**Scheduling** means assigning computations (processes or threads) to processors (cores, distributed computers, etc.) according to time. For example, in map-reduce computing, we mapper processes are scheduled to particular cluster nodes having the necessary local data; they are rescheduled in the case of faults; reducers are scheduled at a later time.
