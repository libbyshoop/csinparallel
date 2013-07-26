*****************
Display Functions
*****************

init_display
************

Rank 0 initializes the graphics display.
The code uses X to handle graphics display. 

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