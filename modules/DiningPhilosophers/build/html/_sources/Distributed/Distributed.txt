*******************************
Distributed Dining Philosophers
*******************************

OpenMPI
#######

In this section we discuss the Dining Philosophers problem where
each philosopher is a separate process. Memory is not shared. We
make use of **OpenMPI**, an implementation of **MPI**, which stands
for **Message Passing Interface**. OpenMPI provides an API for
sending data from one process to another. Programs using OpenMPI
must include ``mpi.h`` and link with a number of libraries. You can
use the wrapper compiler ``mpicc`` to avoid having to specify all
the MPI libraries when linking your program.

MPI programs are run by using the ``mpirun`` command. This command
allows you to specify the number of processes to start and which
hosts to start them on. Although the processes may or may not run
on different hosts, in the program code itself there is no
difference between sending data to a process running on the same
computer and sending data to a process running on a different
computer. This is because OpenMPI provides the concept of a
"communicator" containing a number of processes, where each process
has a number. To send a message to a process, you simply need to
specify its number and communicator.

OpenMPI supports both blocking and unblocking sends and receives.
Blocking sends and receives, such as ``MPI_Send()`` and
``MPI_Recv()``, block until the message is actually send or
received, while nonblocking sends and receives, such as
``MPI_Isend()`` and ``MPI_Irecv()``, return immediately. If using
nonblocking message passing, you can later test to see if a message
has completed using the ``MPI_Test()`` class of functions or wait
for a message to complete using the ``MPI_Wait()`` class of
functions. If you have installed the OpenMPI documentation, there
should be a man page on each MPI function.

A "Waiter" solution to the distributed Dining Philosophers problem
##################################################################

One of the simplest solutions to resource management problems like
the Dining Philosophers problem is to centralize the problem by
having a master thread or master process that determines which
threads or processes are able to access resources. For the Dining
Philosophers problem, this can be called the **waiter** solution.

In order to solve the dining philosophers problem in a distributed
manner using the waiter solution, the philosopher processes must
communicate with the waiter using message passing. That is, they
must ask for permission to eat, and the waiter, who has control of
all the forks, decides who gets to eat and sends them a message
telling them to go ahead.

The waiter may only be concerned with preventing deadlock, or he
may also keep track of which philosophers have eaten and give an
advantage to philosophers who are starving.

In the file ``distributed_waiter.c``, there is an implementation
of the waiter solution to the distributed dining philosophers
problem. It is not a full solution, since it only solves the
deadlock problem and not the starvation problem. Take a look at the
code and read the comments to try to understand it. Support for MPI
threads is not needed to run this program as it only uses
processes, not threads.

Since the waiter is its own process, ``distributed_waiter`` must be
run using 6 processes in order to simulate 5 philosophers.

The Chandry-Misra solution to the distributed Dining Philosophers problem
#########################################################################

Although having a waiter process can fully solve the dining
philosophers problem, the disadvantage is that all philosophers
have to wait for the one waiter, who could be overloaded with work
if there are too many philosophers. Is it possible to solve the
dining philosophers problem in a distributed, decentralized
manner?

In 1984, K. M. Chandry and J. Misra published a paper titled
*The drinking philosophers problem* [1]_. In it, they
provide a completely distributed solution to the Dining
Philosophers problem that avoids both deadlock and starvation. They
also generalize their solution to what they call the
"Drinking Philosophers problem", where an arbitrary number of
agents can share any number of resources ("bottles") with other
agents and require any number of these resources for each
"drinking" session.

For the full details of their solution, you should see their
original paper, which as of this writing is freely available
online. We will give you a summary of their solution and show you
some code that implements it.

In Chandry and Misra's solution to the Dining Philosophers problem,
idea of the forks being *clean* and *dirty* is introduced. The
solution is completely distributed, and philosophers must send
"request tokens" to other philosophers to request their forks.
"Dirty" forks must be given up if they are requested, while "clean"
forks may be kept, unless they are not needed. Whenever a fork is
used to eat, it becomes dirty, and whenever a fork is sent to
another philosopher, it is cleaned (if it is not already clean).

The deadlock problem is solved by ensuring that no cycles can
develop in the precedence graph that represents which philosophers
have priorities over others. Additionally, the starvation problem
is solved because the cleanliness of the forks will ensure that any
one philosopher will not wait forever to eat.

In the file ``chandry_misra.c``, there is an implementation of
this solution. Messages are passed between processes using OpenMPI.
Read some of the comments in the code to better understand the
solution.

In order to run this program, the MPI library must support multiple
threads executing in the MPI library concurrently. This is because
each philosopher process is divided into 2 threads: a main thread
that does the thinking and eating, and a helper thread that listens
for requests for forks from the philosophers' neighbors at all
times, including when the main thread is thinking. This design is
necessary if we do not allow for the possibility that forks can be
owned by no philosophers. If at any point in time, the fork must be
held by one of the two philosophers or be in transit between the
two, it cannot be guaranteed that the other philosopher can get the
fork if one philosopher decides to think for an arbitrarily long
period of time, during which he cannot release the fork-- but the
Chandry-Misra solution requires that he does in fact release the
fork somehow. The helper thread solves this problem by allowing the
philosopher's forks to be given up while he is thinking.

Run the program with ``mpirun -n 5 chandry_misra [SECONDS]``. The
default number of seconds to run the simulation for is 5.
Statistics are shown when the simulation finishes. The default
eating and thinking times are set to be very fast, but you can
adjust this in the arrays near the top of the code. Try running the
code for a hundred seconds or so and look at the results. Does it
look like all the philosophers had an equal chance to eat? You will
be able to tell from the results, although they won't be able to
tell you if starvation is theoretically possible. But the solution
does, in fact, guarantee that each any every philosopher will be
able to eat within a finite amount of time once he becomes hungry.

.. [1] K. M. Chandry and J. Misra. The drinking philosophers problem. ACM Transactions on Programming Languages and Systems, 6(4):632–646, October 1984.

          


