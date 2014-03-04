********************************************
Shared Memory Parallel Patternlets in OpenMP
********************************************


When writing programs for shared-memory hardware with multiple cores, 
a programmer could use a
low-level thread package, such as pthreads. An alternative is to use
a compiler that processes OpenMP *pragmas*, which are compiler directives that
enable the compiler to generate threaded code.  Whereas pthreads uses an **explicit**
multithreading model in which the programmer must explicitly create and manage threads,
OpenMP uses an **implicit** multithreading model in which the library handles
thread creation and management, thus making the programmer's task much simpler and
less error-prone.  OpenMP is a standard that compilers who implement it must adhere to. 

The following are examples of C code with OpenMP pragmas.  There is one C++
example that is used to illustrate a point about that language. The first
three are basic illustrations so you can get used to the OpenMP pragmas and
conceptualize the two primary patterns used as
**program structure implementation strategies** that almost all shared-memory
parallel programs have:

	* **fork/join**:  forking threads and joining them back, and 
	* **single program, multiple data**:  writing one program in which separate threads maybe performing different computations simultaneously on different data, some of which might be shared in memory.

The rest of the examples illustrate how to implement other patterns
along with the above two and what can go wrong when mutual exclusion
is not properly ensured.

Note: by default OpenMP uses the **Thread Pool** pattern of concurrent execution control.
OpenMP programs initialze a group of threads to be used by a given program 
(often called a pool of threads).  These threads will execute concurrently
during portions of the code specified by the programmer.  In addition, the **multiple instructtion, multiple data** pattern is used in OpenMP programs because multiple threads can be exituting different instructions on different data in memory at the same point in time.

Source Code
************

Please download all examples from this tarball: 
:download:`openMP.tgz <../patternlets/openMP.tgz>`

A C code file and a Makefile or each example below can be found in 
subdirectories of the openMP directory created by extracting the above tarball.  
The number for each example below corresponds to one used in subdirectory 
names containing each one.

To compile and run these examples, you will need a C compiler with OpenMP.  The gnu C compiler is OpenMP compliant.  We assume you are building and executing these on a unix command line.


Patternlets Grouped By Type
***************************

If you are working on these for the first time, you may want to visit them in order.  If you are returning to review a particular patternlet or the pattern categorization diagram, you can refer to them individually.

:doc:`ProgStructure_Barrier`

:doc:`DataDecomp_Reduction`

:doc:`MutualExclusion`

:doc:`TaskDecomp`

:doc:`patterns_diagram`

.. toctree::
	:hidden: 

	ProgStructure_Barrier
	DataDecomp_Reduction
	MutualExclusion
	TaskDecomp
	patterns_diagram
	









