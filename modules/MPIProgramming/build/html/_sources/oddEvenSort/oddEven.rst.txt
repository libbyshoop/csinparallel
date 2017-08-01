**************************************
Odd Even Transposition Sort
**************************************


Algorithm
=====================
The odd even transposition sort is a variation of bubble sort. Like in bubble sort,
elements of a list are compared pairwise and swapped when necessary. However, these
compare-swaps are done in two phases: odd and even. Suppose that *a* is a list of
integers. The compare-swaps for the phases are as follows

**Odd phase:** (a[1], a[2]),(a[3], a[4]), (a[5],a[6]), ...

**Even phase:** (a[0], a[1]),(a[2], a[3]), (a[4],a[5]), ...

The algorithm guarantees that for a list with n elements, after n phases the
list will be sorted. the list may be sorted with fewer phases, but it will
always be sorted after n phases. Below is a simple example for reference.

List: 6, 2, 7, 4

============== ==================================== ===============
Phase          Compare-Swap                         Resulting List
============== ==================================== ===============
Odd            | Compare-swap (2,7)                 | 6, 2, 7, 4
Even           | Compare-swap (6,2) and (7,4)       | 2, 6, 4, 7
Odd            | Compare-swap (6,4)                 | 2, 4, 6, 7
Even           | Compare-swap (2,4) and (6,7)       | 2, 4, 6, 7
============== ==================================== ===============

Parallelization
==================
We begin by deciding how to split up the work of sorting list *a*. If we have
*n* elements in the list and *p* processes, then naturally each process should
receive *n/p* elements. To sort the local elements in each process, we can use a fast
serial sorting algorithm like quicksort (qsort). Now we are left with independent
processes that each contain a local portion of sorted elements. If each process
had only one element, we could go about the odd even sort easily. Processes 1
and 2 would exchange their elements for an odd phase and so on. We will apply this
logic to our parallel version. We will have process 1 and 2 exchange all of
their elements with process 1 keeping the smallest half of the elements. Continuing
this for all n phases will result in sorted elements stored in processes
of increasing rank.

The parallel odd even transposition sort depends on the number of process to
guarantee a sorted list. If a sort is run on p processes, then after p phases
the list will be sorted. Example below.

List: 5, 3, 7, 8, 2, 1, 6, 9, 4

================= =========== =========== ===========
Phase             Process 0   Process 1   Process 2
================= =========== =========== ===========
Begin             5, 3, 7     8, 2, 1     6, 9, 4
After Local Sort  3, 5, 7     1, 2, 8     4, 6, 9
Odd               3, 5, 7     1, 2, 4     6, 8, 9
Even              1, 2, 3     4, 5, 7     6, 8, 9
Odd               1, 2, 3     4, 5, 6     7, 8, 9
================= =========== =========== ===========


Code
================
*file: MPI_examples/oddEvenSort/mpi_odd_even.c*

The code for this example is from Peter Pacheco's book, An Introduction to Parallel Programming.
For further implementations and reading corresponding to this example, refer
to his book which is listed below.

* **Book:** *An Introduction to Parallel Programming*, Peter Pacheco, Morgan Kaufmann Publishers, 2011

*Build inside mpi_odd_even directory:*
::

  make oddEvenSort

*Execute on the command line inside oddEvenSort directory:*
::

  mpirun -np <number of processes> ./mpi_odd_even

**Main Function**

.. literalinclude:: ../MPI_examples/oddEvenSort/mpi_odd_even.c
    :language: c
    :lines: 75 - 131

:Comments:
  * **Debug statements:** In C, we can debug code by adding the directives
    **#ifdef DEBUG** and **#endif** around code. There are multiple instances of
    this in the main function. The print statements located between these directives
    are only printed if a problem arises.
  * **Free function:** The free function in C deallocates the memory that has
    been distributed using calls to malloc. Note that at the end of the
    main function, variable *local_a* is deallocated as we no longer need to store
    the entire list.

User Input
=======================================
The following command is entered on the command line for running the executable.

**usage:**  mpirun -np <p> mpi_odd_even <g|i> <global_n>

For this example, users must enter the number of processes (*p*) as well as the
number of elements for the list (*global_n*) as command line arguments. Users also
have a choice of whether to have the program sort a randomly generated list (*g*)
or an input list (*i*). For our purposes, we will stick to using randomly generated
lists.

Odd Even Sorting
=======================================
There are several functions that play an integral role in sorting including Sort,
Compare, Odd_even_iter, Merge_low and Merge_high. We will walk through each
function and describe what is being done. The following diagrams follow from a
starting list of 5, 3, 7, 8, 2, 1, 6, 9, 4.

**Sort**
-----------
From main, each process calls and sends the appropriate arguments to
Sort. In the Sort function, we begin by allocating some memory needed later on for
merging the local lists. Then, we determine the odd and even phase partner of the
current process which has rank *my_rank*. We will need to know this in order to
do the odd-even phase swaps. If we have a process whose rank is odd,
it has two possible partners. Its even phase partner will have a rank of
*my_rank - 1* where as its odd phase partner will have a rank of *my_rank + 1*.
Similarly, a process whose rank is even will have an even phase partner with
rank *my_rank + 1* and an odd phase partner of rank *my_rank + 1*.

Even phase: An even rank will have a partner of rank *my_rank + 1*.
An odd rank will a partner of rank *my_rank - 1*.

.. image:: EvenPhase.png
  :width: 500

Odd phase: An even rank will a partner of rank *my_rank - 1*.
An odd rank will have a partner of rank *my_rank + 1*.

.. image:: OddPhase.png
	:width: 500

Next, we sort the local list using qsort with the basic Compare function.
Lastly, we loop through all possible phases (*p* phases). For each phase,
the function Odd_even_iter is called which performs one iteration of the
odd-even transposition between local lists. Lastly, all temporary memory is deallocated.

.. image:: LocalSorting.png
	:width: 500

.. literalinclude:: ../MPI_examples/oddEvenSort/mpi_odd_even.c
    :language: c
    :lines: 302 - 339

Compare
----------
The compare function is very simple. It compares two integers
and is used solely by qsort to sort each local list.

Odd_even_iter
---------------
We will discuss this function in two parts: even phase and odd phase. In both
cases, we check to make sure that the current process has the necessary partner
needed for the swap. If there are an odd number of processes, there is a chance
that one process will not have a partner to swap with. This check ensures that
we will not run into an error in this situation.

.. literalinclude:: ../MPI_examples/oddEvenSort/mpi_odd_even.c
    :language: c
    :lines: 349 - 377


Even phase: As long as the process has an even partner, we can proceed.
The *even_partner* is sent and received by previously allocated
memory, *temp_B*. *temp_B* is needed for merging. This is done using the
MPI function MPI_Sendrecv. MPI_Sendrecv  is a thread-safe function to send and
receive a message in a single call. If we have an odd rank process, we want the
local list for this process to merge with its *even_partner* so that
it will contain the largest elements between the two. Otherwise, the process has
an even rank and we will merge the smallest elements between the current
process and its *even_partner* to the current process'local list.
Below is a diagram following from the above list.

.. image:: EvenPhaseMerge.png
	:width: 500

Odd Phase: Once again, if we have an odd phase and the process has an odd partner,
the *odd_partner* is sent and received by previously allocated memory, *temp_B*.
An odd rank process will merge with its *odd_partner* so that its local list will
have the smallest elements between the two. Similarly, the local list of an
even rank process will contain the largest elements.

.. image:: OddPhaseMerge.png
	:width: 500

Merge_low and Merge_high
-----------------------------
The Merge_low and Merge_high functions take part in the comparison swap between
the local lists of processes. Temporary allocated memory is necessary for a
merge to take place. We fill temporary storage variable *temp_C*
with the smallest or highest elements respectively from both the local list and
*temp_B* (partner) of a process. The elements from *temp_C* are then copied into the local list.

Print Functions
========================================
There are several print functions that serve to make printing various lists
easier. The two main printing functions include Print_global_list and
Print_local_list.

* Print_global_list function: Prints a global list of all elements by gathering
  together elements from each local list. The master process uses the MPI function
  MPI_Gather to collect each *local_A* list from all processes. It then stores
  the components in global list *A*.
* Print_local_lists: Prints a local list for each process using helper function
  Print_list. The master process begins by printing its local list of elements.
  Next, the master process alternates between receiving a local list from another
  process and printing it until all processes have had their lists printed.
