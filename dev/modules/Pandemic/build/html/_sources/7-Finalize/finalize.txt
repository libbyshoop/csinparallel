****************
Finish Functions
****************

show_results()
**************

At the end of the code, if we are choosing to show results, we print out
the final counts of susceptible, infected, immune, and dead people. We
also print the actual contagiousness and actual deadliness of the
disease. To perform these two calculations, we use the following code
(using the contagiousness as the example):

.. literalinclude:: Finalize.h	
    :language: c
    :lines: 35-43

In this code, the ternary operators (? and :) are used to return one
value if something is true and another value if it isn’t. The thing we
are checking for truth is **our\_num\_infection\_attempts == 0**. If this is
true, i.e. if we didn’t attempt any infection attempts at all, then we
say there was actually 1 infection attempt (this is to avoid a divide by
zero error). Otherwise, we return the actual number of infection
attempts. This value becomes the dividend for **our\_num\_infections**; in
other words, we divide the number of actual infections by the number of
total infections. This will give us a value less than 1, so we multiply
it by 100 to obtain the actual contagiousness factor of the disease. A
similar procedure is performed to calculate the actual deadliness
factor.

cleanup
*******

If we are using CUDA, we have to destroy the CUDA environment. We do this by calling **cuda_finish()** function.

.. literalinclude:: Finalize.h	
    :language: c
    :lines: 67-69


If X display is enabled, then Rank 0 destroys the X Window and closes the display

.. literalinclude:: Finalize.h	
    :language: c
    :lines: 72-74

Since we allocated our arrays dynamically, we need to release them back
to the heap using the **free** function. We do this in the reverse order
than we used **malloc**, so **environment** will come first and **x\_locations**
will come last.

.. literalinclude:: Finalize.h	
    :language: c
    :lines: 87-100

Just as we initialized the MPI environment with **MPI\_Init**, we must
finalize it with **MPI\_Finalize()**. No MPI functions are allowed to occur
after **MPI\_Finalize**.

.. literalinclude:: Finalize.h	
    :language: c
    :lines: 102-105