************
Introduction
************

Improvement and new inventions in technology over the past few decades have greatly improved performance in computing. Up until around 2005, however, the vast majority of computer systems contained only single-core CPU’s. With the introduction of **multi-core CPUs** the industry was able to develop even faster performing systems. Most systems for personal use nowadays have a CPU with 2 or 4 cores, while server machines used in various institutions and businesses can have triple that amount of cores or more. Benefits of using multi-core systems include better performance, ability to process large and complex data sets but also power consumption. Since multi-core processors have become the mainstream in computer systems, so did parallel computing become standard in computer architectures.

Parallel computing takes advantage of multi-core CPUs by performing many calculations simultaneously on the cores of a CPU using sequences of execution of a program called **threads**. Writing a parallel program is more complicated than writing serial programs, because there are many software bugs and obstacles that come with this kind of computing. 

There are various kinds of computing that support this notion of executing many different instructions at the same time, and how they perform depends on the hardware and software available to us. As already mentioned, multi-core systems are needed for parallel computing. Several threads can be initiated that each perform a task on each core of the processor. 

There are other types of hardware that support parallelism. Some computers have many processors, known as multi-processor computers. Systems with tens of thousands processors are called supercomputers. The world’s fastest supercomputer was announced June 2013 and it is the Tianhe-2 located in China. 

On the other hand it is possible to connect many computers via an internal network, known as **clusters**. Clusters are composed of individual machines, each being able to independently perform a task. They are connected with each other through a TCP/IP Ethernet local area network, which enables them to communicate with each other. A form of computation that requires data to split among the processing units of a cluster is referred to as **distributed processing**.

A recent trend in computer engineering is the use of **GPUs** which are multi-core co-processors optimized for computer graphics processing. GPUs have a much better performance than CPUs for solving large sized problems.

To take advantage of hardware that supports parallel and/or distributed computing the software that we use to execute instruction on the hardware must also support this type of computing. There are several programming languages that support parallel computing (such as Ada, Java, etc) but there are also libraries and various APIs (Application Programming Interface) and parallel programming models that aid in developing parallel programs also for languages that are not created to support parallelism. Some widely used APIs include **MPI (Message Passing Interface)** for distributed processing, as well as **POSIX Threads** and **OpenMP (Open Multi Processing)** for parallel computing. 

It is possible to combine these various software (and hardware) paradigms to perform so called **heterogenous computing**. Heterogeneous computing typically involves splitting the computation among several processing units (distributed processing) and further splitting it amongst the cores within each processing unit’s CPU. 

