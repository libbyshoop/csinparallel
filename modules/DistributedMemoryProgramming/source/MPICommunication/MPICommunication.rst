==================
MPI Communications
==================

Point-to-point Communication
----------------------------

Point-to-point communication is a way that pair of processors transmits the data between one another, one processor sending, and the other receiving. MPI provides SEND(MPI_Send) and RECEIVE(MPI_Recv) functions that allow point-to-point communication taking place in the MPI world. ::

	MPI_Send(void* buffer, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm comm)

		- buffer:	initial address of send buffer
		- count:	number of entries to send
		- datatype:	type of each entry
		- destination:	rank of the receiving processor
		- tag:		message tag is a way to identify types of messages
		- comm:		communicator (MPI_COMM_WORLD)

	MPI_Recv(void* buffer, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)

		- source:	rank of the sending processor	
		- status:	return status

.. note:: To read more on MPI_Status, please read `MPI_Status <http://www.netlib.org/utk/papers/mpi-book/node31.html>`_.


MPI Datatype
^^^^^^^^^^^^

In most MPI functions, which you will be using, require you specify the datatype of your message. Below is the table showing the corresponding datatype between MPI Datatype and C Datatype.

=================== =================
    MPI Datatype         C Datatype   
=================== =================
MPI_CHAR                signed char
MPI_SHORT               signed short int 
MPI_INT                 signed int 
MPI_LONG                signed long int
MPI_UNSIGNED_CHAR       unsigned char 
MPI_UNSIGNED_SHORT      unsigned short int
MPI_UNSIGNED            unsigned int
MPI_UNSIGNED_LONG       unsigned long int
MPI_FLOAT               float
MPI_DOUBLE              double 
MPI_LONG_DOUBLE         long double 
=================== =================

**Table 1: Corresponding datatype between MPI and C**

Example 2: Send and Receive Hello World
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. highlight:: c

.. literalinclude:: example2.c
	:linenos:


This MPI program illustrates the use of MPI_Send and MPI_Recv functions. Basically, the master sends a message, “Hello, world”, to the process whose rank is 1, and then after having received the message, the process prints out the message along with its rank.

Collective Communication
------------------------

Collective communication is a communication that must have all processes involved in the scope of a communicator. We will be using MPI_COMM_WORLD as our communicator; therefore, the collective communication will include all processes.

.. image:: images/collective.png
	:width: 450px
	:align: center
	:height: 350px
	:alt: MPI_COMM_WORLD

.. centered:: Figure 3: Collective Communications[1] 

::

	MPI_Barrier(comm)

This function creates a barrier synchronization in a commmunicator(MPI_COMM_WORLD). Each task waits at MPI_Barrier call until all tasks in the communicator reach the same MPI_Barrier call. ::

	MPI_Bcast(&buffer, int count, MPI_Datatype datatype, int root, comm)

This function displays the message to all other processes in MPI_COMM_WORLD from the process whose rank is root. ::

	MPI_Reduce(&buffer, &receivebuffer, int count, MPI_Datatype datatype, MPI_Op op, int root, comm)

This function applies a reduction operation on all tasks in MPI_COMM_WORLD and reduces results from each process into one value. MPI_Op includes for example, MPI_MAX, MPI_MIN, MPI_PROD, and MPI_SUM .etc. 

	.. note:: To read more on MPI_Op, please read `MPI_Op <http://www.mpi-forum.org/docs/mpi-11-html/node78.html#Node78>`_. 

::

	MPI_Scatter(&buffer, int count, MPI_Datatype, &receivebuffer, int count, MPI_Datatype, int root, comm)

This function divides a message into equal contiguous parts and sends each part to each node. The master gets the first part, and the process whose rank is 1 gets the second part, and so on. The number of elements get sent to each worker is specified by count. ::

	MPI_Gather(&buffer, int count, MPI_Datatype, &receivebuffer, int count, MPI_Datatype, int root, comm)

This function collects distinct messages from each task in the communicator to a single task. This function is the reverse operation of MPI_Scatter. 

.. rubric:: Footnotes
.. [1] https://computing.llnl.gov/tutorials/mpi/
