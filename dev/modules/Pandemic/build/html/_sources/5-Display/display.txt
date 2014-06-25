*****************
Display Functions
*****************

init_display
************

Rank 0 initializes the graphics display. The code uses X to handle graphics display.

share_location
**************

If display is enabled, Rank 0 gathers the states, x locations, and y locations of the people for which each process is responsible

.. figure:: img-18.png
   :align: center
   :alt: image

We set up the displs here just as we did in function **share_infected()**. 

.. literalinclude:: Display.h	
    :language: c
    :lines: 83-104

Three calls to Gatherv take place for each process to send each of their **our\_states**, **our\_x\_locations**, and **our\_y\_locations arrays**. Rank 0 copies these into its **states**, **x\_locations**, and **y\_locations** arrays, respectively. 

.. literalinclude:: Display.h	
    :language: c
    :lines: 106-111

Note that if MPI is not enabled, Rank 0 just does a direct copy of the arrays without using Gatherv.

.. literalinclude:: Display.h	
    :language: c
    :lines: 117-130

do_display
**********

If display is enabled, Rank 0 displays a graphic of the current day

.. figure:: img-19.png
   :align: center
   :alt: image

close_display
*************

If X display is enabled, then Rank 0 destroys the X Window and closes the display

throttle
********

In order for better display, we wait between frames of animation.