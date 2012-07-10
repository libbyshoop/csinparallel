==================
MPI Communications
==================

Point-to-point Communication
----------------------------

Point-to-point communication is a way that pair of processors transmit the data between one another, one processor sending, and the other receiving. MPI provides SEND(MPI_Send) and RECEIVE(MPI_Recv) functions that allow point-to-point communication taking place in the MPI world.

- MPI_Send(void* buffer, int count, MPI_Datatype datatype, int destination, int tag, MPI_Comm comm)
	
	- buffer:		initial address of send buffer
	- count:		number of entries to send
	- datatype:		type of each entry
	- destination:		rank of the receiving processor
	- tag:		message tag is a way to identify types of messages
	- comm:		communicator (MPI_COMM_WORLD)

- MPI_Recv(void* buffer, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)

	- source:		rank of the sending processor	
	- status:		return status


MPI Datatype
^^^^^^^^^^^^



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
Table 1: Corresponding datatype between MPI and C

Example 2: Send and Receive Hello World
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. highlight:: c

.. literalinclude:: example2.c
	:linenos:


This MPI program is just to illustrate MPI_Send and MPI_Recv functions. Basically, the master just send a message, “Hello, world”, to the process whose rank is 1, and then after having received the message, the process just prints out the message along with its rank.

Collective Communication
------------------------

Collective communication is a communication that must have all processes involved in the scope of a communicator. We will be using MPI_COMM_WORLD as our communicator; therefore, the collective communication will include all processes.

Figure 3: 


- MPI_Barrier(communicator)

This function creates a barrier synchronization in the MPI_COMM_WORLD. Each task waits at MPI_Barrier call until all tasks in the group reach the same MPI_Barrier call.

- MPI_Bcast(&buffer, int count, MPI_Datatype datatype, int root, comm)

This function displays the data to all other processes in MPI_COMM_WORLD from the process whose rank is root.

- MPI_Reduce(&buffer, &receivebuffer, int count, MPI_Datatype datatype, MPI_Op op, int root, comm)

This function applies a reduction operation on all tasks in MPI_COMM_WORLD and reduces the result into one value. MPI_Op# includes for example, MPI_MAX, MPI_MIN, MPI_PROD, and MPI_SUM .etc.

- MPI_Scatter(&buffer, count, MPI_Datatype, &receivebuffer, count, MPI_Datatype, Master, comm)

This function divides a message into equal contiguous parts and sends each part to each node. The head node(Master) gets the first part, and the next node whose rank is 1 gets the second part, and so on. The number of elements get sent to each node is specified by count.

- MPI_Gather(&buffer, count, MPI_Datatype, &receivebuffer, count, MPI_Datatype, Master, comm)

This function gathers distinct messages from each task in the group to a single destination task. This routine is the reverse operation of MPI_Scatter. 