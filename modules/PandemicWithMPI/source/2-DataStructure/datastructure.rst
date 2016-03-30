***************
Data Structures
***************

Here is the list of variables and arrays used by the program. Note the
naming scheme; variables whose names begin with “my” are private to the
threads that use them. Variables whose names begin with “our” are
private to the processes that use them, but public to the threads within
that process. Variables are thus named from a thread’s perspective; “my”
variables are ones that I use, “our” variables are ones that I and the
other threads in my process use.

global_t struct
***************

.. literalinclude:: Defaults.h	
    :language: c
    :lines: 39-57

**total\_number\_of\_people**

the total number of all people in the simulation; the sum of people assigned to each process. The value of this variable can be specified on the command line with the –n option.

**total\_num\_initially\_infected**

the total number of people who are initially infected; the sum of initially infected people assigned to each process. The value of this variable can be specified on the command line with the –i option. This is a subset of the total number of people, so the value of this variable must be smaller or equal to the value for **total\_number\_of\_people**.

**total\_num\_infected**

the total number of infected people; the sum of the number of infected people assigned to each process. This value changes throughout the course of the simulation.

**x\_locations**

array, holds the x locations of all of the people; only used if the environment needs to be displayed; otherwise, **our\_x\_locations** is used.

**y\_locations**

array, holds the y locations of all of the people; only used if the environment needs to be displayed; otherwise, **our\_y\_locations** is used.

**their\_infected\_x\_locations**

array, used in **susceptible()** function to keep track of the x locations of the infected people for which each process is responsible.

**their\_infected\_y\_locations**

array, used in step **susceptible()** function to keep track of the y locations of the infected people for which each process is responsible.

**states**

array, holds the states of all of the people; only used if the environment needs to be displayed; otherwise, **our\_states** is used.

**total\_number\_of\_processes**

used to keep track of how many processes are being used. If MPI is disabled, the value of this variable will be 1. If it is enabled, the value is set in **init()** function.

our_t struct
************

.. literalinclude:: Defaults.h	
    :language: c
    :lines: 59-87

**our\_current\_day**

a loop iterator representing the ID of the current day being simulated by the current process.

**our\_rank**

used to keep track of the rank of the current process. If MPI is disabled, the value of this variable will be 0. If it is enabled, the value is set in **init()** function.

**our\_number\_of\_people**

the number of people for which the current process is responsible. This will be a number less than or equal to the total number of people. The value is determined in **find_size()** function.

**our\_num\_initially\_infected**

the count of initially infected people for which the current process is responsible.

**our\_num\_infected**

a count of the number of infected people for which the current process is responsible.

**our\_num\_susceptible**

a count of the number of susceptible people for which the current process is responsible.

**our\_num\_immune**

a count of the number of immune people for which the current process is responsible.

**our\_num\_dead**

a count of the number of dead people for which the current process is responsible.

**our\_x\_locations**

array, holds the x locations of all the people for which the current process is responsible.

**our\_y\_locations**

array, holds the y locations of all the people for which the current process is responsible.

**our\_infected\_x\_locations**

array, holds the x locations of all the infected people for which the current process is responsible.

**our\_infected\_y\_locations**

array, holds the y locations of all the infected people for which the current process is responsible.

**our\_states**

array, holds the states of all the people for which the current process is responsible.

**our\_num\_days\_infected**

array, used to keep track of the number of days each person has been infected for which the current process is responsible.

const_t struct
**************

.. literalinclude:: Defaults.h	
    :language: c
    :lines: 89-103

**environment\_width**

indicates how wide the environment is; used to draw the environment and to make sure people stay within the bounds of the environment.

**environment\_height**

indicates how high the environment is; used to draw the environment and to make sure people stay within the bounds of the environment.

**infection\_radius**

see the Introduction Chapter above. The value of this variable can be specified on the command line with the –d option.

**duration\_of\_disease**

see the Introduction Chapter above. The value of this variable can be specified on the command line with the –T option.

**contagiousness\_factor**

see the Introduction Chapter above. The value of this variable can be specified on the command line with the –c option.

**deadliness\_factor**

see the Introduction Chapter above. The value of this variable can be specified on the command line with the –D option.

**total\_number\_of\_days**

the total number of days over which to run the simulation.

**microseconds\_per\_day**

used to tell how many microseconds to freeze in between frames of animation. The value of this variable can be specified on the command line with the –m option.

stats_t struct
**************

.. literalinclude:: Defaults.h	
    :language: c
    :lines: 105-114

**our\_num\_infections**

used to count the number of actual infections that take place (in which an infected person transmits the disease to a susceptible person). Only used if the showing of results is enabled (i.e., if the program is to print out final results from the simulation). Used to determine the actual contagiousness of the disease, which can be compared to the contagiousness factor by the user.

**our\_num\_infection\_attempts**

used to count the number of times a susceptible person is within an infection radius of an infected person, even if the infection fails. Only used if the showing of results is enabled (i.e., if the program is to print out final results from the simulation). Used to determine the actual contagiousness of the disease, which can be compared to the contagiousness factor by the user.

**our\_num\_deaths**

used to count the number of times a person dies. Only used if the showing of results is enabled (i.e., if the program is to print out final results from the simulation). Used to determine the actual deadliness of the disease, which can be compared to the deadliness factor by the user.

**our\_num\_recovery\_attempts**

used to count the number of times an infected person is able to become immune. Only used if the showing of results is enabled (i.e., if the program is to print out final results from the simulation). Used to determine the actual deadliness of the disease, which can be compared to the deadliness factor by the user.

display_t struct
****************

.. literalinclude:: Defaults.h  
    :language: c
    :lines: 116-141

**environment**

2D array, holds an ASCII representation of the environment (see “state” under “Person” in the Introduction Chapter). This variable is used only when we are using Text Display.

**display**

Display, display pointer for the connection to the X server

**window**

Window, variable to holds the window id.

**screen**

Screen, variable to holds default screen

**delete_window**

**gc**

**infected_color**

**immune_color**

**susceptible_color**

**dead_color**

**red**

array of char, holds value \#FF0000, which is the hex code for color red.

**green**

array of char, holds value \#00FF00, which is the hex code for color green.

**black**

array of char, holds value \#000000, which is the hex code for color black.

**white**

array of char, holds value \#FFFFFF, which is the hex code for color white.

**colormap**