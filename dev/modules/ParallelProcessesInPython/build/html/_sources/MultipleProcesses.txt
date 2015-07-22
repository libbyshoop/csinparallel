*************************************
Creating multiple child processes
*************************************

A solution to the previous exercise is the following:

::

    def procEx2():
        print "Hi from process", current_process().pid, "(parent process)"

        p1 = Process(target=sayHi, args=())
        p2 = Process(target=sayHi, args=())
        p3 = Process(target=sayHi, args=())

        p1.start()
        p2.start()
        p3.start()

Here we make three different ``Process`` objects. It is important to note that
each process uses the same ``sayHi`` function defined before, but each
process executes that function independent of the others. Each child
process will print its own unique pid.

Let’s push this a little further now, using a sayHi2 function that takes
an argument. Observe the following code: 

::

    def sayHi2(n):
        print "Hi", n, "from process", current_process().pid

    def manyGreetings():
        print "Hi from process", current_process().pid, "(main process)"
        
        name = "Jimmy"
        p1 = Process(target=sayHi2, args=(name,))
        p2 = Process(target=sayHi2, args=(name,))
        p3 = Process(target=sayHi2, args=(name,))

        p1.start()
        p2.start()
        p3.start()

Note in the
``manyGreetings`` function that we create three ``Process`` objects, but this
time the ``args`` argument is not an empty tuple, but rather a tuple with a
single value in it. (Recall that the comma after name is used in
single-element tuples to distinguish them from the other use of
parentheses: syntactic grouping.) With the ``args`` tuple set up in this
way, ``name`` is passed in for ``n`` in the ``sayHi2`` function. So the result here
is that each of the three child processes has the same name, “Jimmy”,
which is included in the child process’s output. Of course, we could
trivially pass distinct names to the children by adjusting the ``args``
tuple accordingly.

.. topic:: Try the code

	:download:`Download manyGreetings.py <code/manyGreetings.py>` and try the above example on your system.


Variable Number of Processes
----------------------------

Let’s try another exercise now. Write a function that first asks for
your name, and then asks how many processes to spawn. That many
processes are created, and each greets you by name and gives its pid.
Try this on your own before moving on. *Hint*: use a loop to create the number of desired child processes.

