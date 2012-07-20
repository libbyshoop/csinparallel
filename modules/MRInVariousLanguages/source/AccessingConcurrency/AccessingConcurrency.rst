*******************************************************************
Accessing concurrency and parallelism within a programming language
*******************************************************************

Features that are part of a programming language
################################################

* **Java** ``synchronized``: a method or a segment of code within a method can be marked as ``synchronized``, meaning that no two threads of execution may execute in synchronized sections for the same object at the same time.

* **Ada** (developed for DOD applications in 1980s): threads ("concurrent tasks"), which may be created dynamically; "rendezvous" (cf. remote procedure call) with synchronized communication.

* **Erlang**: The language consists of (functional programming) *sequential* constructs plus additional *concurrent* constructs for carrying out sequential code in parallel. No threads, just processes – a design decision to allow for easier fault tolerance, because shared resources such as memory are very difficult to manage correctly in the presence of faults.

Libraries
#########

* **MPI (Message Passing Interface)**: Library allowing for send, receive, and other communication calls for both *point-to-point* and *collective* communication in a distributed system. Supports a notion of "communicator groups" of processes.

* **Java** ``Thread`` **class**; ``java.util.concurrent``: These are standard packages in the Java language. The ``Thread`` class may be subclassed to provide a (sequential) ``run()`` method for carrying out specified code when that thread object is started. The package ``java.util.concurrent`` provides programming interfaces and classes for concurrency-safe data structures, thread management/reuse and scheduling, synchronization primitives such as semaphores, etc.

Other approaches
################

* **Operating system calls**: For example, Linux provides ``fork()`` for creating new processes, ``socket()`` and related system calls for creating communication lines between processes that may be on separate machines, ``read()`` and ``write()`` for sending and receiving along socket connections, ``select()`` for synchronizing communication, and various thread packages exist for Linux.

* **OpenMP C/C++ pragmas**: A convenient approach for incrementally adding synchronization in a shared-memory multiprocessor (such as a multicore system), in which one adds preprocessor ``pragmas`` to request parallelization of a ``for`` loop, creation of threads, etc.

* **CUDA programming of a GPGPU**: Modern video controllers are highly parallel devices designed for highly parallel, very fast linear algebra computations that feed a pipeline for adding further graphics features (such as texturing). NVIDIA and other manufacturers now provide a programming interface enabling a programmer to make general-purpose computations with that specialized hardware---"General Purpose Graphics Processing Unit (GPGPU)". NVIDIA's CUDA language provides a library for C and C++ programs for the CPU that interact with separate "kernel" programs written for the GPU in order to perform such GPGPU computations.

