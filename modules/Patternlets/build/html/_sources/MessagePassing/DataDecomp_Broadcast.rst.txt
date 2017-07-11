*********************************************************
Broadcast
*********************************************************

06. Broadcast: a special form of message passing
**************************************************

*file: patternlets/MPI/06.broadcast/broadcast.c*

*Build inside 06.broadcast directory:*
::

  make broadcast

*Execute on the command line inside 06.broadcast directory:*
::

  mpirun -np <number of processes> ./broadcast

This example shows how a data item read from a file can be sent to all the processes.
Lines 29 through 34 demonstrate reading data from a file. After opening the file and
asserting that the file is not empty, the file is read by the *fscanf* function.
This function then stores the data from the file as an integer in the answer
variable. Note that only process 0 has the data from the file stored in answer.

In order to send the data from process 0 to all of the processes in the
communicator, it is necessary to *broadcast*. During a broadcast, one process
sends the same data to all of the processes. A common use of broadcasting is
to send user input to all of the processes in a parallel program. In our example,
the broadcast is sent from process 0 and looks like this:

.. image:: Broadcast.png
	:width: 700

.. literalinclude:: ../patternlets/MPI/06.broadcast/broadcast.c
    :language: c
    :linenos:

07. Broadcast: send receive equivalent
**************************************************

file: patternlets/MPI/07.broadcastSendReceive/broadcastSendReceive.c*

*Build inside 07.broadcastSendReceive directory:*
::

  make broadcastSendReceive

*Execute on the command line inside 06.broadcastSendReceive directory:*
::

  mpirun -np <number of processes> ./broadcastSendReceive

This example shows how to ensure that all processes have a copy of an array
created by a single *master* process. Master process 0 sends the array to each process,
all of which receive the modified array.

.. literalinclude:: ../patternlets/MPI/07.broadcastSendReceive/broadcastSendReceive.c
  :language: c
  :linenos:

08. Broadcast: send data to all processes
**************************************************

*file: patternlets/MPI/08.broadcast2/broadcast2.c*

*Build inside 08.broadcast2 directory:*
::

  make broadcast2

*Execute on the command line inside 08.broadcast2 directory:*
::

  mpirun -np <number of processes> ./broadcast2

The send and receive pattern where one process sends the same data to all
processes is used frequently. Broadcast was created for this purpose. This example
is the same as the previous example except that we send the modified array
using broadcast.

.. literalinclude:: ../patternlets/MPI/08.broadcast2/broadcast2.c
    :language: c
    :linenos:
