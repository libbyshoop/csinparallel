************
Introduction
************

Motivation
##########

`Moore's "Law"`_: an empirical observation by Intel co-founder Gordon Moore in 1965. The number of components in computer circuits had doubled each year since 1958, and Moore predicted that this doubling trend would continue for another decade.  Incredibly, over four decades later, that number has continued to double each two years or less.

However, since about 2005, it has been impossible to achieve such performance improvements by making larger and faster single CPU circuits. Instead, the industry has created *multi-core* CPUs – single chips that contain multiple circuits for carrying out instructions (cores) per chip. 

The number of cores per CPU chip is growing exponentially, in order to maintain the exponential growth curve of Moore's Law. But most **software** has been designed for single cores.

.. figure:: MooresLaw.png
    :width: 1000px
    :align: center
    :height: 899px
    :alt: alternate text
    :figclass: align-center

    Plot of CPU transistor counts against dates of introduction. Note the logarithmic vertical scale; the line corresponds to exponential growth with transistor count doubling every two years. This figure is from `Wikimedia Commons`_.

Therefore, CS students must learn principles of parallel computing to be prepared for careers that will require increasing understanding of how to take advantage of multi-core systems.

.. _Wikimedia Commons: http://en.wikipedia.org/wiki/File:Transistor_Count_and_Moore%27s_Law_-_2011.svg

.. _`Moore's "Law"`: http://en.wikipedia.org/wiki/Moore%27s_law

Some pairs of terms
###################

.. glossary::
     parallelism
        multiple (computer) actions physically taking place at the same time

.. glossary::
     concurrency
	programming in order to take advantage of parallelism (or virtual parallelism)

:Comments:

	Thus, parallelism takes place in hardware, whereas concurrency takes place in software. Operating systems must use concurrency, since they must manage multiple processes that are abstractly executing at the same time--and can physically execute at the same time, given parallel hardware (and a capable OS).

.. glossary::
     process
	the execution of a program

.. glossary::
     thread
	a sequence of execution within a program

:Comments:

	Every process has at least one thread of execution, defined by that process's program counter. If there are multiple threads within a process, they share resources such as the process's memory allocation. This reduces the computational overhead for switching among threads (also called *lightweight processes*), and enables efficient sharing of resources (e.g., communication through shared memory locations).

.. glossary::
     sequential programming
       programming for a single core

.. glossary::
     concurrent programming
	programming for multiple cores or multiple computers

:Comments:

	CS students have primarily learned sequential programming in the past. These skills are still relevant, because concurrent programs ordinarily consist of sets of sequential programs intended for various cores or computers.

.. glossary::
     multi-core computing
       computing with systems that provide multiple computational circuits per CPU package

.. glossary::
     distributed computing
	computing with systems consisting of multiple computers connected by computer network(s)

:Comments:

	Both of these types of computing may be present in the same system (as in our MistRider and Helios clusters). 

.. glossary::
     data parallelism
       the same processing is applied to multiple subsets of a large data set in parallel

.. glossary::
     task parallelism
	different tasks or stages of a computation are performed in parallel

:Comments:

	A telephone call center illustrates data parallelism: each incoming customer call (or outgoing telemarketer call) represents the services processing on different data. An assembly line (or computational pipeline) illustrates task parallelism: each stage is carried out by a different person (or processor), and all persons are working in parallel (but on different stages of different entities.)

.. glossary::
     shared memory multiprocessing
	e.g., multi-core system, and/or multiple CPU packages in a single computer, all sharing the same main memory

.. glossary::
     cluster
	multiple networked computers managed as a single resource and designed for working as a unit on large computational problems 

.. glossary::
     grid computing
	distributed systems at multiple locations, typically with separate management, coordinated for working on large-scale problems

.. glossary::
     cloud computing
	computing services are accessed via networking on large, centrally managed clusters at data centers, typically at unknown remote locations

.. glossary::
     SETI@home
	another example of distributed computing

:Comments:

	Although multi-core processors are driving the movement to introduce more parallelism in CS courses, distributed computing concepts also merit study. For example, Intel's recently announced 48-core chip for research behaves like a distributed system with regards to interactions between its cache memories. 


































