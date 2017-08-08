Instructor/Student Notes
========================

Before getting started with this module, please keep in mind that we assume you have **some grasp of C programming**.

In the section called 'Local Cluster Configuration' we provide some information about two types of clusters that are available at Macalester College.  If you are using this course module at your own institution, you will need to change this information to refer to your own cluster.

This module contains both background information and several activities for students to try using Message Passing Interface (MPI) programs on a cluster
of machines.  The examples and activities follow from simple introductory examples to activities, each of which increases in difficulty as you proceed.

**For the four activities, we recommend having the example code open in one window while you go through the explanation of the code.**  You can download them all here first, or there will be ways to dowload or look at each one in your browser as you go along.

Files to try on your cluster
-----------------------------

The following examples are used in the next few pages of this module.  You can download each one (right-click and save) and eventually use a client such as scp to copy them up to a cluster machine.  *However, they may already be available on your cluster. Check with your instructor.*

The first one is a complete tar and gzipped archive file of all of them. The rest are each individual files.

Zip File:
:download:`download dist_example_code.tgz <dist_example_code.tgz>`

Simplest Hello World Example:
:download:`download hellompi.c <dist_example_code/hellompi.c>`

Hello World by Sending messages:
:download:`download example2.c <dist_example_code/example2.c>`

**Activity 1, Computing pi as area under the curve:**


:download:`download seq_pi_done.c <dist_example_code/seq_pi_done.c>`

:download:`download mpi_pi_done.c <dist_example_code/mpi_pi_done.c>`

**Activity 2, Vector-Matrix Multiplication:**

:download:`download vector_matrix_buggy_todo.c <dist_example_code/vector_matrix_buggy_todo.c>`

:download:`download vector_matrix_buggy_done.c <dist_example_code/vector_matrix_buggy_done.c>`


**Activity 3, Improved Vector-Matrix Multiplication**

:download:`download vector_matrix_mpi_todo.c <dist_example_code/vector_matrix_mpi_todo.c>`

:download:`download vector_matrix_mpi.c <dist_example_code/vector_matrix_mpi.c>`

**Activity 4, Matrix-Matrix Multiplication**

:download:`download matrix_multiplication_todo.c <dist_example_code/matrix_multiplication_todo.c>`

:download:`download matrix_multiplication.c <dist_example_code/matrix_multiplication.c>`
