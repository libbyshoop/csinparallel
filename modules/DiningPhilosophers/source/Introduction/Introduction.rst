************
Introduction
************

In this module, you will learn about the Dining Philosophers
Problem and some of the solutions that have been developed for it.
This is a classic problem in concurrency that describes a situation
where multiple processes contend for shared resources. The problem
is originally based on an examination question given by Edsger
Dijkstra in 1965.

Dining Philosophers source file: 
:download:`dining_philosophers_code.tar.gz <dining_philosophers_code.tar.gz>`

Learning Objectives
###################

-  Understand the Dining Philosophers Problem and the situations
   that it provides a useful model for.

-  Understand several solutions for the Dining Philosophers problem
   and the differences between them.

-  Be able to implement a simulation in C or C++ that solves the
   Dining Philosophers Problem (or at least solves the deadlock
   portion of the problem).

Requirements
############

This module will make use of a number of parallel programming
techniques such as threads (through OpenMP) and message passing
(through OpenMPI). The code provided with this module has only been
tested being compiled with GCC on Linux, although it may work on
other Unix systems. GCC has had support for OpenMP since version
4.2, which was released in 2007. Using OpenMPI requires the OpenMPI
libraries and headers, which can be found in the software the
repositories for many Linux distributions.

In order to run one of the example programs, the OpenMPI
installation needs to have support for multiple threads executing
in the MPI library concurrently. This requires it to be configured
with the ``--enable-mpi-thread-multiple`` option (or
``--enable-mpi-threads`` for older versions of OpenMPI). To see if
your compilation of OpenMPI supports this feature, the output of
``ompi_info | grep Thread`` should list **yes** for
``MPI_THREAD_MULTIPLE`` (or yes for "mpi" threads for older
versions of OpenMPI). Note that this only applies to 1 of the 5
example programs.

There are a number of MPI libraries available other than OpenMPI;
it should be possible to use a different one.

Introduction
############

Suppose that 5 philosophers are sitting around a circular table. A
plate of spaghetti is given to each philosopher, and 5 forks are
distributed around the table so that each philosopher has a left
fork and a right fork that are shared with his neighbors. Each
philosopher independently alternates between thinking and eating.
In order to eat, a philosopher must acquire both his left and right
forks. When done eating, he releases his forks and resumes
thinking. Philosophers are allowed to think for an arbitrary amount
of time, even an infinite amount of time. Philosophers are allowed
to eat for an arbitrary but finite amount of time.

(The problem can also be explained using rice and chopsticks, which
perhaps makes more sense, as it is much harder to eat with one
chopstick than with one fork! We will stick with forks and
spaghetti since it seems to be the more common explanation.)

Requirements for a Solution
###########################

The philosophers, of course, represent processes or threads running
in a computer, and the forks represent some kind of shared
resource. These shared resources could be a number of things, such
as records in a database or files that can be accessed by only one
process at a time.

We would like to examine the ways that a solution to the Dining
Philosophers problem could be implemented on a real computer or
cluster of computers.

A successful solution to the problem must meet the following
criteria:


-  Each fork is in use by at most one philosopher at any instant in
   time.

-  All philosophers, upon reaching an hungry state, will be able to
   eat within a finite amount of time.


