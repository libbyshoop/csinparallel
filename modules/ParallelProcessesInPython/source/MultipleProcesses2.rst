*****************************************
Creating multiple child processes, part 2
*****************************************


Here is a possible solution for creating a variable number of child processes:

::

    def manyGreetings2():
        name = raw_input("Enter your name: ")
        numProc = input("How many processes? ")

        for i in range(numProc):
            (Process(target=sayHi2, args=(name,))).start()

Anonymous Processes
-------------------

After obtaining the
user’s name and desired number of processes, we create and start that
many Process objects with a loop. Note in this case that the single line
of the loop body could also be written as two lines as follows:

::

        p = Process(target=sayHi2, args=(name,))
        p.start()

We can say that the one-line version includes the use of *anonymous*
Process objects. They are anonymous since the individual objects are
never stored in variables for later use. They are simply created and
started immediately. The one-line version might look confusing at first,
but note that (Process(target=sayHi2, args=(name,))) creates a Process
object. We’re then just calling the start method on that Process object,
instead of storing it in a variable and calling start on that variable.
For our purposes, the end result is the same.

Now, consider the following:

::

        for i in range(numProc):
            pi = Process(target=sayHi2, args=(name,))
            pi.start()

This would work as well, as it merely substitutes variable p in
the previous example for pi. However, it is important to point
out that this code does **not** actually create several variables, p0,
p1, p2, etc. Sometimes this kind of mistake happens because we’re
working in a different context now – parallel programming – but it’s
important to remember that the same programming principles you’ve
already learned continue to apply here. For example, consider the
following example:

::

        for a in range(10):
            grade = 97

Clearly this code does not create the variables gr0de, gr1de, gr2de,
etc. Similarly, then, pi does not become p0, p1, p2, etc.

Another important question can be considered in reviewing
the ``manyGreetings2`` code above again. Which approach is better, the
explicit use of p, or the anonymous version given in the original? It depends. In this
example, we don’t need access to the ``Process`` objects later, so there’s
no need to store them. So the anonymous version is acceptable in that
regard. But we might also think about which version we find to be more
readable. To an extent this may be a matter of personal opinion, but it
is something that should be considered in programming.


