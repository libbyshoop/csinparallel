*****************************************************************
Shared Memory Parallel Patternlets with Pthreads vath library
*****************************************************************

Pthreads, which is a lower-level thread package, can be used by programmers when writing
programs for shared-memory hardware with multiple cores. Pthreads uses an **explicit**
multithreading model in which the programmer must explicitly create and manage threads.
To make the programmer's task simpler, we have opted to incorporate the vath library written
by Victor Alessandrini. Alessandrini's book, "Shared Memory Application Programming" utilizes
his library throughout. The vath library includes utilities build upon C++ classes that are
easy to use and high level. For our purposes, we will make use of the Pthreads implentation of
the vath library, vath_pth. We include only the static vath_pth library and include files in
our source code. For the complete version of the vath library (Pthreads and C++11
implementation) along with code examples from the book, refer to the book's software site listed below.

* **Book:** *Shared Memory Application Programming*, Victor Alessandrini, Morgan Kaufmann Publishers, 2016
* **Site:** http://booksite.elsevier.com/9780128037614/software.php


The following are examples of C++ code with Pthreads and various classes (SPool,
CpuTimer, Rand, and Reduction) from the vath_pth library. There is one example that
is used to illustrate a point about the difference between C and C++ languages.
The first three are basic illustrations so you can get used to the SPool
utility and conceptualize the two primary patterns used as **program structure
implementation strategies** that almost all shared-memory parallel programs have:

* **fork/join**:  forking threads and joining them back, and
* **single program, multiple data**:  writing one program in which separate threads maybe performing different computations simultaneously on different data, some of which might be shared in memory.

The other examples illustrate how to implement other patterns
along with the above two and what can go wrong when mutual exclusion
is not properly ensured.

Note: the SPool utility uses the **Thread Pool** pattern of concurrent execution control.
The utility allows for initialization of a group of threads to be used by a given program
(often called a pool of threads). These threads will execute concurrently
the thread function present in the code specified by the programmer. The SPool utility
is different from OpenMP in that the master thread performs an idle wait while waiting
to join other threads and goes into a blocked state where it does not use CPU resources
which adds a level of flexibility.


Source Code
************

Please download all examples from this tarball:
:download:`Vath_pth.tar.gz <../patternlets/Vath_pth.tar.gz>`

A C++ code file and a Makefile for each example below can be found in
subdirectories of the Vath_pth directory created by extracting the above tarball.
The number for each example below corresponds to one used in subdirectory
names containing each one.

To compile and run these examples, you will need a C++ compiler with Pthreads.
The GNU C++ compiler is Pthreads compliant. We assume you are building and
executing these on a Unix command line.


Patternlets Grouped By Type
***************************

:doc:`ProgStructure_Barrier`

:doc:`DataDecomp_Reduction`

:doc:`MutualExclusion`

.. toctree::
	:hidden:

	ProgStructure_Barrier
	DataDecomp_Reduction
	MutualExclusion
