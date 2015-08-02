****************************
Process Coordination, Part 2
****************************

Solution to Exercise
---------------------

The completed exercise is as follows: 

::

    def addTwoNumbers(a, b, q):
        # sleep(5) # In case you want to slow things down to see what is happening.
        q.put(a+b)

    def addTwoPar():
        x = input("Enter first number: ")
        y = input("Enter second number: ")

        q = Queue()
        p1 = Process(target=addTwoNumbers, args=(x, y, q))
        p1.start()

        # p1.join()
        result = q.get()
        print "The sum is:", result


As you
can see, it requires only a small addition. The parent must call the get
method on the queue. Once the child has put something on the queue, the
parent’s get will succeed, and the variable result will get a value and
be printed.

Did you attempt to use join in your solution, as in the commented-out
line in the above solution? In this example the join is
not harmful, but is not required. This is because the get will already
cause the parent process to block until data is on the queue. So there’s
no need for the parent process to wait for the child to be finished with
a join as well. The get already causes the required wait.





Using a Queue to Merge Multiple Child Process Results
-----------------------------------------------------

The following example is a nice extension of the one
above. 

::

    from multiprocessing import *
    from random import randint
    import time
    def addManyNumbers(numNumbers, q):
        s = 0
        for i in range(numNumbers):
            s = s + randint(1, 100)
        q.put(s)

    def addManyPar():
        totalNumNumbers = 1000000

        q = Queue()
        p1 = Process(target=addManyNumbers, args=(totalNumNumbers/2, q))
        p2 = Process(target=addManyNumbers, args=(totalNumNumbers/2, q))
        p1.start()
        p2.start()

        answerA = q.get()
        answerB = q.get()
        print "Sum:", answerA + answerB

Here, the task is to add up a
large number of random numbers. This is accomplished by creating two
child processes, each of which is responsible for half of the work. Note
that a shared queue, plus half the total number of numbers, is passed to
each child. Each child creates many random numbers and adds them up,
putting the final sum on the queue. The parent makes two get calls, to
obtain the result from each child. Note that the parent will likely
block on at least the first get call, since it will need to wait until
one of the children finishes and places its result on the queue.

Here’s an interesting question to consider: which child’s result will be
in answerA and which in answerB? The answer is that this is
indeterminate. Whichever child process finishes first will have its
answer in answerA, and the other will be in answerB. This is not a
problem for commutative merging operations, like the addition of this
example, but of course could be a complication for non-commutative
merging.

.. topic:: Try the code

    :download:`Download addManyPar.py <code/addManyPar.py>` and try the above example on your system.

Conclusion
----------

Of course there are many places you could go next, but here we have seen
an introduction to parallel programming in Python using the
multiprocessing module. We’ve explored the parent-child model of
parallel programming, in which the parent creates many child processes
to perform some task. We’ve seen how shared resources lead to a need for
locks to ensure uninterrupted access. Finally, we’ve seen how to pass
data between processes, both via the Process constructor’s ``args``
argument, and also through the use of a shared queue.
