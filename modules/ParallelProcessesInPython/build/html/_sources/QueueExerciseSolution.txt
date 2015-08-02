***********************
Queue Exercise Solution
***********************

Recall the English pseudocode for our original simple examle problem
of sending 5 pieces of data from a parent to a child process:

::

    '''
    def greet2():
        for 5 times
            get name from queue
            say hello

    def sendName2():
        queue
        make a child process, give it the queue
        start it

        for 5 times
            sleep for a bit
            put another name in the queue
    '''

Here is a Python solution that follows the structure of
the pseudocode very closely. It’s just a matter of filling in the syntax
we’re learning for queues, along with a review of working with
processes and how we can sleep for a randomly defined amount of time.

::

    from random import randint

    def greet2(q):
        for i in range(5):
            print
            print "(child process) Waiting for name", i
            name = q.get()
            print "(child process) Well, hi", name

    def sendName2():
        q = Queue()
       
        p1 = Process(target=greet2, args=(q,))
        p1.start()

        for i in range(5):
            time.sleep(randint(1,4))
            print "(main process) Ok, I'll send the name"
            q.put("George"+str(i))

.. topic:: Try the code

    :download:`Download sendName2.py <code/sendName2.py>` and try the above example on your system.

Once you are comfortable with this example of using queues to communicate data and coordinate the handling of that data using ``put`` and ``get``, you will be ready to look at some other coordination mechanisms in the next section.
