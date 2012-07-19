*********************************
Shared Memory Dining Philosophers
*********************************

In this section, we consider the case where each philosopher is a
thread. They all share the same memory space.

Observing deadlock
##################

Take a look at the file ``deadlock.c`` in the code distributed with
this module. This is a simple shared memory simulation (but not a
solution!) of the dining philosophers problem. The process is
divided into five threads using an OpenMP parallel section.

.. literalinclude:: deadlock.c
    :language: c
    :lines: 52-67

OpenMP (which stands for Open Multi-Processing) is an API that makes it
easier to write multithreaded programs in C, C++, and Fortran. It
works mainly by using **pragmas**, directives that give information
to the compiler. It also offers a number of functions such as
``omp_get_thread_num()``. To use OpenMP in a C or C++
program, you must pass the flag ``-fopenmp`` to GCC. This flag must
be specified both for the compilation phase, so that the OpenMP
pragmas are interpreted, and for the linking phase, so that the
programs are linked with ``libgomp``, the GNU implementation of the
OpenMP API (which is part of gcc). Also, if any OpenMP functions
are called explicitly, ``omp.h`` must be included.

The forks are represented by **mutexes**. These are program objects
that can only be modified by one thread at a time -- that is,
modifying them is an atomic operation. 

.. literalinclude:: deadlock.c
    :language: c
    :lines: 42-43

More specifically, a mutex is a special case of a **semaphore**, 
an object that allows atomic increment and decrement operations. A mutex is a
**binary semaphore** that only allows two values. These two states
can be referred to as "locked" and "unlocked" A mutex can be used
to ensure exclusive access to objects or sections of code because
only one thread can have locked the mutex at a time.

Each thread (philosopher) enters a loop in which it thinks and
eats. 

.. literalinclude:: deadlock.c
    :language: c
    :lines: 71-91

A **reentrant** function-- that is, a function that returns
consistent results when called by multiple threads concurrently--
that sleeps the calling thread for a random number of milliseconds
is defined to implement the "thinking" and "eating".

.. figure:: Figure1.png
    :width: 672px
    :align: center
    :height: 703px
    :alt: alternate text
    :figclass: align-center

    Figure 1. In this figure, red circles represent philosophers, or threads. Orange rectangulars represent forks, or common resource that threads are sharing. 

Compile ``deadlock.c`` using ``make deadlock`` (or just type ``make``
to compile all the examples at once), then run it. Observe the
simulation for a while before interrupting it.

Did you notice any problems? Most likely not, if you ran the
simulation for only a short period of time. There is a major
problem with the code though. In the function *philosopher_cycle*,
try adding a very short delay between the time the philosophers
pick up their left fork and the time the philosophers pick up their
right fork, like this:

::

    pthread_mutex_lock(left_fork);
    printf("Philosopher %d picked up his left fork.\n", thread_num);
    millisleep(5);
    pthread_mutex_lock(right_fork);

Now try running the code again. What happens?

Most likely all the philosophers picked up their left fork, at
which point they all became permanently **deadlocked** because each
philosopher was waiting on each other in turn. All the threads were
waiting for a resource that will never be released.

This problem can only occur if 4 of the 5 threads are preempted
after picking up their left fork but before picking up their right
fork. This is unlikely to occur without a delay between the locks.
But even so, the possibility of deadlock is a major flaw in this
code. After all, you wouldn't want to use an operating system or
program that could lock up at any time!

A "Solution" that avoids deadlock
#################################

Is there a simple way to prevent deadlock from occurring?

Suppose that not all of the philosophers were to pick up their
forks in the same order. That is, some of the philosophers would
pick up their forks left-right, and others would pick up their
forks right-left. This would therefore be an asymmetrical solution,
since not all of the philosophers would act in the same way.

It turns out that deadlock will be avoided simply if the fifth
philosopher picks up his forks in the opposite order from everyone
else. If the first four philosophers have all picked up their left
fork, then the fifth philosopher will be unable to pick up the last
fork, since it would be his second fork to pick up. Similarly, if
the fifth philosopher is holding one fork, then it is impossible
for the first philosopher to pick up any forks. It is therefore
guaranteed that deadlock will not occur.

Compile and run ``partial_order.c``. 

.. literalinclude:: partial_order.c
    :language: c
    :lines: 68-82

It is very similar to
``deadlock.c``, but it adds in a simple if-else statement around
the code where the philosophers pick up their forks, so that the
last philosopher picks up his forks in the opposite order from the
others. A delay between picking up the two forks has already been
added. You will see that this code will not deadlock.

This solution can be generalized to the idea of assigning a partial
order to the resources. In the classic dining philosophers problem,
the forks would be numbered from 0 to 4. In this situation,
processes must acquire their resources in order from lowest to
highest. This will work for any number of processes acquiring any
number of resources. However, if the needed resources are not known
in advance, this can be an inconvenient solution; this is because
if a process has decided it needs a resource after it has already
acquired higher numbered resources, it must release the higher
numbered resources first, acquire the needed resource, then
reacquire the released resources in order.

Starvation
##########

There is still a problem with the solution in ``partial_order.c,
however. Take a look at the code in ``partial_order2.c``. This is
similar to the original ``partial_order.c``, but an interrupt
handler has been added. This interrupt handler prints out the
number of times each philosopher has eaten when the program is
interrupted (when Control-C is pressed). The philosophers also
think and eat much faster and have no delay between acquiring the
forks.

Compile ``partial_order_2.c``, then let it run for a little
while before interrupting it. Look at how many times each
philosopher has eaten and thought. Are they about the same? Or does
it look like some philosophers had an advantage over others?

Your results will vary, but here are our results after running the
simulation for a few minutes:

========================= ===========================
Philosopher		  Times Eaten					
========================= ===========================
Philsopher 0:             4341 times eaten
Philsopher 1:             4529 times eaten
Philsopher 2:             4612 times eaten
Philsopher 3:             4673 times eaten
Philsopher 4:             4340 times eaten
========================= ===========================

Philosopher 4 and 0 appear to eat and think less often than the
other philosophers; thus, they spend more time in the "hungry"
state, waiting to acquire forks.

This imperfection is a result of the asymmetry of the solution:
since not all the philosophers are acting in the same way, it is
possible that some philosophers do not have the same chance to eat
as others do.

If you think about it, you may realize there is a more fundamental
problem as well. Suppose that philosopher 2 wants to eat, but
philosophers 1 and 3 are currently eating. Philosopher 2's thread
is put to sleep as it waits for the forks. Meanwhile, philosophers
1 and 3 finish eating, but philosopher 2's thread is not scheduled
to run right away. Rather, before philosopher 2's thread is run
again, philosophers 1 and 3 start to eat again. Philosopher 2 never
had a chance to eat! What if this keeps happening over and over?

The problem here is that one of the philosophers could potentially
"starve" because of a timing problem. This is independent from the
possibility of deadlock, which has already been eliminated in the
partial order solution.

We will return to the starvation problem later when we discuss the
Chandry-Misra solution.



