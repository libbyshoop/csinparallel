**************************************
Merge Sort
**************************************
The idea of merge sort is to divide an unsorted listed into sublists
until each sublist contains only one element. These one element sublists
are then merged together to produce new sorted sublists. When we have one
sublist remaining, we are done and the list has been sorted. Conceptually,
the algorithm works like this:

.. image:: mergeSort.png
	:width: 600

Parallel Algorithm
=====================

To parallelize this algorithm, we will use a mixed strategy in which the
sublists are sorted by a sequential sorting algorithm and the merging
of sublists is done in parallel between processes. We chose to stick
with cases in which the number of processes is a power of two so that
all processes are doing roughly the same amount of work.

**Part I: Divide list into unsorted sublists**

For this portion of the problem, we begin with a single unsorted list. This
list is scattered to all of the processes such that each process has an
equal chunk of the list. Suppose we have 4 processes and a list containing
8 integers. The code is executing as follows:

.. image:: scatter.png
	:width: 800

**Part II: Sort sublists**

We can sort these sublists by applying a serial sorting algorithm. We use
the C library function qsort on each process to sort the local sublist.
After sorting the processes have the following sorted sublists:

.. image:: sort.png
	:width: 800

**Part III: Merge sublists**

The merging of the sublists to form a single list is done by sending
and receiving sublists between processes and merging them together. Each
initial sorted sublist (with a size of 2) provides the sorted result to the
parent process. That process combines the two sublists to generate a list
of size 4, and then sends that result to its parent process. Lastly, the
root process merges the two lists to obtain a list of size 8 that is fully
sorted.

.. image:: merge.png
	:width: 800

Code
=====================

*file: MPI_examples/mergeSort/mergeSortMPI/mergeSortMPI.c*

*Build insidemergeSortMPI directory:*
::

  make mergeSort

*Execute on the command line inside mergeSortMPI directory:*
::

  mpirun -np <number of processes> ./mergeSort < list size >

Main Function
--------------------

.. literalinclude:: ../MPI_examples/mergeSort/mergeSortMPI/mergeSortMPI.c
    :language: c
    :lines: 244 - 312


:Comments:
		* **Get Processor Name:** When running this code on a cluster, obtaining the
		  processor name allows us to check how the processes are being distributed.
		* **Height:** Height represents the number of levels needed to ensure we obtain a single sorted list. In the example above with 4 processes and a list of 8 integers, we need 3 levels (0, 1, 2).
		* **Timing:** The parallel time as well as the individual time of each process
		  is determined. This should demonstrate that the root process (process 0) accounts for most of the parallel time.

.. topic:: To do:

  **Test code:** Uncomment lines A, B and C. Run and compile the code using a small integer
	for the size of the list to insure that the list is being sorted correctly.
	Once you are confident that the final list is sorted the right way, recomment
	the lines.

  **Analysis:** Compile and run the sequential version of merge sort located in the
	mergeSort/mergeSortSeq directory using 4, 8, 16, 32, 64 million for the list size.
	Record sequential time. Next, try running the parallel program with 2, 4, 8 processes
	and 4, 8, 16, 32, 64 million for the list size. Record parallel time for each.
	Can you answer whether merge sort displays good scalability, speedup and efficiency?
	Explain your answer.



Merging Sublists
=====================
There are several functions that play an integral role in merging including
mergeSort, Compare and merge. We will walk through the mergeSort function and
explain how it works.

mergeSort Function
--------------------

**Setup**

Each process calls and sends the appropriate arguments to mergeSort from main.
In the mergeSort function, we begin by setting the process' individual height
to 0 and sorting its portion of the list using qsort with the simple Compare
function. We assign a pointer, *half1*, to the process' sublist that was just
sorted.

**Loop**

Next, we enter a loop that will continue until we have reached the
total number of levels needed to have a single sorted list. Within the loop, we
find the parent of the current process. This will determine whether or not
the current process is a left child or right child of the parent. Note that a
left child and its parent will be same process. The diagram below shows how
children and parents are related. For example, process 0 is the left
child and process 1 is the right child (as indicated by its orange outline)
of parent process 0.

.. image:: parentChildren.png
	:width: 700

Depending on the child status of the process, the process will do the following:

**Left Child:**

* Find right child
* Allocate memory needed for storing right child's list in *half2*
* Receive right child's list in *half2*
* Allocate memory needed for result of merging lists in *mergeResult*
* Merge *half1* and *half2* into *mergeResult* with merge function
* Reassign *half1* to point to *mergeResult*
* Free memory for *half2*
* Set *mergeResult* to NULL
* Increase process height by 1

**Right Child:**

* Send current process' portion of list to parent
* Set process height to overall height - DONE!

The left child is the process that is actually taking part in the merging.
In comparison, the right child simply sends its list portion to its parent (left child).
Once it has done this, the process is now finished - it has nothing else to do!
This loop continues until each process' individual height reaches the overall
height required to guarantee that we have a single sorted list.

Visual Example
--------------------
This simple example follows the mergeSort algorithm.
Assume each colored box represents a single process' current sorted sublist, *localArray*.

Setup
^^^^^^^^^^^
With four processes, the setup looks like this:


.. image:: mergeSort1.png
	:width: 800

Each process has *half1* initially pointing to its sorted portion of the list.

Loop at Height 0
^^^^^^^^^^^^^^^^^^^

.. image:: mergeSort2.png
	:width: 800

Notice that *half1* is reassigned to point to the result of the merge.
Let's take a closer look at part of the communication process between Process 0 and
Process 1 at height 0. These two processes are executing at the same time. In the
diagram below time is moving from top to bottom.

.. image:: communication.png
	:width: 600

For the next height, height 1, Process 0 has access to the merged array
through *half1*.

Loop at Height 1
^^^^^^^^^^^^^^^^^^^

.. image:: mergeSort3.png
	:width: 800

End at Height 2
^^^^^^^^^^^^^^^^^^^

.. image:: mergeSort4.png
	:width: 700

Code
------------------

.. literalinclude:: ../MPI_examples/mergeSort/mergeSortMPI/mergeSortMPI.c
    :language: c
    :lines: 196 - 240
