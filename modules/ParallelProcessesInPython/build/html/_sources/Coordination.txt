**************************
Coordination of Processes
**************************

The Join Method
---------------

In parallel programming, a *join* operation instructs the executing
process to block until the process on which the join is called
completes. For example, if a parent process creates a child process in
variable ``p1`` and then calls ``p1.join()``, then the parent process will block
on that join call until p1 completes. One important point to emphasize
again in this example is that the *parent* process blocks, not the
process on which join is called (``p1``). Hence the careful language at the
start of this paragraph: the executing process blocks until the process
on which the join is called completes.

The word “join” can be confusing sometimes. The following example
provides an analogy of the parent process waiting
(using join) for a “slowpoke” child process to catch up. 

::

    def slowpoke(lock):
        time.sleep(10)
        lock.acquire()
        print "Slowpoke: Ok, I'm coming"
        lock.release()

    def haveToWait():
        lock = Lock()
        p1 = Process(target=slowpoke, args=(lock,))
        p1.start()
        print "Waiter: Any day now..."

        p1.join()
        print "Waiter: Finally! Geez."


The child
process is slow due to the ``time.sleep(10)`` call. Note also the use of a
lock to manage the shared use of stdout.

It should be pointed out, however, that join is not always necessary for
process coordination. Often a similar result can be obtained by blocking
on a queue get, as described in the previous section and later in this section.

.. topic:: Try the code

    :download:`Download haveToWait.py <code/haveToWait.py>` and try the above example on your system.



Obtaining a Result from a Single Child
--------------------------------------

While earlier examples demonstrated a parent process sending data to a
child via a queue, this example allows us to practice the other way
around: a child that performs a computation which is then obtained by
the parent. Consider two functions: ``addTwo-Numbers``, and ``addTwoPar``.
``addTwoNumbers`` takes two numbers as arguments, adds them, and places the
result on a queue (which was also passed as an argument). ``addTwoPar`` asks
the user to enter two numbers, passes them and a queue to addTwo-Numbers
in a new process, waits for the result, and then prints it.

Consider the following starting code:

::

    def addTwoNumbers(a, b, q):
        # time.sleep(5) # In case you want to slow things down to see what is happening.
        q.put(a+b)

    def addTwoPar():
        x = input("Enter first number: ")
        y = input("Enter second number: ")

        q = Queue()
        p1 = Process(target=addTwoNumbers, args=(x, y, q))
        p1.start()



The parent passes two numbers inputted by the user, and a shared queue,
to a child process, ``p1``, which will execute ``addTwoNumbers``. The child process puts the sum of the numbers onto
the queue, with an optional sleep call before, to slow the computation
down for illustrative purposes. 

Here is an exercise for you to consider
now: starting with the above code, which you can :download:`download as addTwoPar.py <code/addTwoPar.py>`, write
code to make the parent obtain the result from the child and print it.
Do not move on until you have done this.


