Using the Intel Phi
===================

Setup on the MTL
----------------

First, you need to ssh into the Manycore Testing Lab where the Phi is located. Use the class password::

    ssh rab-s01@192.55.51.81

.. tip:: Optionally, you can use the ``-Y`` argument for X-forwarding, if you wish to use a GUI text editor like emacs.

Then you need to SSH into xeon1 (the machine with the Phi installed)::

    ssh xeon1

.. tip:: Again, optionally use the ``-Y`` argument for X-forwarding (this requires that it was used for the initial SSH connection into the MTL)

Make a subdirectory for yourself using your username and go to that
directory::

    mkdir <username>
    cd username

Setup on the Phi coprocessor
----------------------------

Then, ssh into the Phi coprocessor:: 

    ssh mic0

If this does not work, use ``ssh 172.31.1.1``.  The terminal prompt should look like this: ``[rab-s01@xeon1-mic0 rab-s01]$``.

On the Phi, create a subdirectory for yourself using your username::

     mkdir <username>

Then logout of the Phi::

     exit

Using the Trapezoid Demo on the Xeon
------------------------------------

Download the following code and copy it into yourCopy the trapezoid code from the phiDemo directory to your subdirectory
and switch to that directory and list the contents::

    cp -r ~/phiDemo/trap .
    cd trap
    ls

This contains a correct version of trap.C, which calculates Pi using trapezoids,and a Make file.  Look at both files and note that we are using the intel C++ compiler instead of GCC

Build trap on xeon1 and perform some trial runs, keeping track of the
runtimes, with varying numbers of threads::

    make
    time ./trap 1
    time ./trap 2
    time ./trap 32

.. tip:: The xeon1 machine has 2 CPUs with 8 cores (16 threads) each

Using the Trapezoid Demo on the Phi
-----------------------------------

Now, let's build trap to run natively on the Phi and move the executable there.

First, compile it on the Xeon for the Phi with ``make phi``.  Then run ``scp trap mic0:<username>/`` to copy it to the Phi, where ``<username>`` is your user name.

Then SSH into the Phi and run trial runs there, keeping track of the
runtimes, with varying numbers of threads::

    ssh mic0
    cd username
    time ./trap 1
    time ./trap 61
    time ./trap 244

.. tip::  The Phi coprocessor has 61 cores and can run a total of 244 threads

.. note:: Typically, the Phi does not have the OpenMP runtime library. This needs to be copied to the Phi and added to the ``LD_LIBRARY_PATH`` environment variable. This has already been done for the ``rab-s01`` account and you can see the Phi version of the runtime library in the ``~/libs`` directory on the Phi.*

Compare the speedups achieved on xeon1 (a traditional multicore
computer) and the Phi coprocessor

.. tip::  xeon1 has a 2.60 GHz clock speed and the Phi has a 1.053 GHz
   clock speed

Optional: Test on the MTL which has 4 CPUS with 10 cores (20 threads)
each

Optional: copy other OpenMP programs, such as the sieve of eratosthenes,
from previous labs to the Phi and try them

-  Look at the trap Makefile for how to compile for the Phi using icc

Offloading Sections of Code to the Phi
--------------------------------------

Copy the example offload code to your directory with following command::

   cp -r ~/phiDemo/offload .

Examine the file offload.C.  Note the offload pragma.

Now run the makefile and try the code

-  It prints the available threads first on the host machine and then on
   the Phi coprocessor

Modifying trap.C
----------------

Modify your trap.C to run on the host machine but offload the parallel portion to the Phi. To do so, add an offload pragma
.. hint:: See `this page <http://software.intel.com/sites/products/documentation/doclib/stdxe/2013/composerxe/compiler/cpp-lin/index.htm>`_ and search for offload for documentation on the offload pragma

