*************
MPI Functions
*************


find_infected
*************

Each process determines its infected x locations and infected y
locations

.. figure:: img-16.png
   :align: center
   :alt: image

We have already set the states of the infected people and the positions
of all the people, but we need to specifically set the positions of the
infected people and store them in the **our\_infected\_x\_locations** and
**our\_infected\_y\_locations** arrays. We do this by marching through the
**our\_states** array and checking whether the state at each cell is
INFECTED. If it is, we add the locations of the current infected person
from the **our\_x\_locations** and **our\_y\_locations** arrays to the
**our\_infected\_x\_locations** and **our\_infected\_y\_locations** arrays. We
determine the ID of the current infected person using the
**our\_current\_infected\_person** variable:

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 302-313

share_infected
**************

First, each process sends its count of infected people to all the other
processes and receives their counts

.. figure:: img-17.png
   :align: center
   :alt: image

This step is handled by the MPI command **MPI\_Allgather** whose arguments
are as follows:

* **&our\_num\_infected** – the address of the sending buffer (the thing being sent).

* **1** – the count of things being sent.

* **MPI\_INT** – the datatype of things being sent.

* **recvcounts** – the receive buffer (an array of things being received).

* **1** – the count of things being received.

* **MPI\_INT** – the datatype of things being received.

* **MPI\_COMM\_WORLD** – the communicator of processes that send and receive data.

Once the data has been sent and received, we count the total number of
infected people by adding up the values in the **recvcounts** array and
storing the result in the **total\_num\_infected** variable:

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 337-343

Next, each process sends the x locations of its infected people to all
the other processes and receives the x locations of their infected
people

For this send and receive, we need to use **MPI\_Allgatherv** instead of
**MPI\_Allgather**. This is because each process has a varying
number of infected people, so it needs to be able to send a variable
number of x locations. To do this, we first need to set up the
displacements in the receive buffer; that is, we need to indicate how
many elements each process will send and at what points in the receive
array they will appear. We can do this with a **displs** array, which will
contain a list of the displacements in the receive buffer:

.. literalinclude:: Pandemic.h	
    :language: c
    :lines: 347-353

We are now ready to call the **MPI\_Allgatherv**. Here are its arguments:

* **our\_infected\_x\_locations** – the send buffer (array of things to send).

* **our\_num\_infected** – the count of elements in the send buffer.

* **MPI\_INT** – the datatype of the elements in the send buffer.

* **their\_infected\_x\_locations** – the receive buffer (array of things to receive).

* **recvcounts** – an array of counts of elements in the receive buffer

* **displs** – the list of displacements in the receive buffer, as determined above.

* **MPI\_INT** – the data type of the elements in the receive buffer.

* **MPI\_COMM\_WORLD** – the communicator of processes that send and receive data.

Once the command is complete, each process will have the full array of
the x locations of the infected people from each process, stored in the
**their\_infected\_x\_locations** array.

Finally, each process sends the y locations of its infected people to all
the other processes and receives the y locations of their infected
people

The y locations are sent and received just as the x locations are sent
and received. In fact, the function calls have exactly 2 letters
difference; the x’s in the **Allgatherv** from last step. are replaced by
y’s in the **Allgatherv** in this step.

Note that the code will only execute previous two steps if MPI is
enabled. If it is not enabled, the code simply copies the
**our\_infected\_x\_locations** and **our\_infected\_y\_locations** arrays into
the **their\_infected\_x\_locations** and **their\_infected\_y\_locations**
arrays and the **our\_num\_infected** variable into the **total\_num\_infected**
variable.

share_location
**************

If display is enabled, Rank 0 gathers the states, x locations,
and y locations of the people for which each process is responsible

.. figure:: img-18.png
   :align: center
   :alt: image

We set up the displs here just as we did in function share_infected. Three calls to
Gatherv take place for each process to send each of their **our\_states**,
**our\_x\_locations**, and **our\_y\_locations arrays**. Rank 0 copies these
into its **states**, **x\_locations**, and **y\_locations** arrays, respectively.
Note that if MPI is not enabled, Rank 0 just does a direct copy of the
arrays without using Gatherv.