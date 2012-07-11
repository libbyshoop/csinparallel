Concurrent Data Structures in C++: Web crawler lab
===================================================

In this lab you will complete some code provided to you that will 'crawl' the web from a beginning page to a given depth by following every linked page and scraping it for more links to follow.  The links found on each page are kept in a data structure until they are processed.
 
Your goals
----------

The goals for this lab are:

* complete and test a web crawler application, which fetches web pages then visits the links contained in those web pages, using STL containers;

* experiment with an example of threads programming, a type of multicore parallel programming;

* to complete a correct multi-threaded web crawler application that uses threaded building block (TBB) containers.


Source Code
-----------

.. topic:: Still need

   The work on this lab requires a "tarball" named ``cds.tar.gz``.

   Instructors, please contact us for the complete code.

Packages needed
----------------

-  C++ compiler

-  Standard Template Library STL

-  CURL library for web access

-  Boost library, for threads and mutexes

-  Intel's Threading Building Blocks (TBB)

-  Make program

Preparation
------------

Copy the tarball into your directory on a multicore linux machine.  Then 'unxip' and 'untar' it like this:

   
   ::

       % tar xfz cds.tar.gz 

   This will create a directory ``cds`` that contains the code. Change
   to that directory.

   ::

       % cd cds

To Do
-----

#. The directory ``serial`` contains several subdirectories, and is
   organized in a structure suitable for a software project that is
   capable of growing very large.

   Examine the code for this program. Observe the following:

   -  The source files (``.cpp`` implementation and driver modules) are
      contained in a subdirectory named ``src``, and the header files
      (interface modules) are named with ``.hcc`` and are stored in a
      subdirectory named ``include``.

   -  Some of the state variables in classes within ``serial`` are STL
      containers, as described in `a class
      handout <http://serc.carleton.edu/files/csinparallel/sigcse_demos/introduction_stl_containers.doc>`_.

   -  Three classes are defined:

      -  ``spider`` is the main class, with methods for crawling from
         page to page and for processing each page, and state variables
         for recording the work to be done (i.e., web addresses or
         *URLs* of pages to visit), the finished work (URLs already
         processes), and a ``vector`` of ``page`` objects.

      -  ``page`` contains state variables for the URL of a particular
         web page (as a ``string``) and a ``vector`` of URLs found in
         that web page (which are candidates for future processing).
         ``page`` also contains a method for scanning a web page and
         filling that ``vector`` with URLs that are contained in that
         page.

      -  ``raw_page`` has helper methods for fetching pages from the web
         and for delivering the HTML code from a fetched web page.

   -  The main program is in the file ``spider_driver.cpp``. It obtains
      two values from the command line, namely the starting URL to crawl
      from, and the maximum number of URLs to visit (which must be
      *parsed* from a string to an integer). Then, the program performs
      the crawl and prints results.

   -  The ``Makefile`` uses ``make`` variables to make it easier to
      change items such as compilation flags. Source files in the
      ``src`` subdirectory are compiled to produce object (``.o``) files
      in the ``build`` subdirectory, and those object files are linked
      to creat an executable named ``bin/spider``.

   **DO THIS:** Write down other observations or comments about the
   ``serial`` code. Feel free to include opinions you may have about the
   code or its organization.

#. Comments in ``spider.cpp`` indicate that two parts of the code need
   to be filled in for the crawler program to work. Before actually
   fillin that code in, we will see if we can compile and run the code
   successfully.

   First, insert output statements in those two locations, to indicate
   whether those sections of the code are reached. One message might
   state "processing the page x" where x is the URL of the page being
   processed, and the other might state "crawling the page x".

   Then ``cd`` to the ``serial`` directory, and issue a ``make`` command
   in that directory ``serial``. This should compile the code and create
   an executable ``bin/spider``

#. Now fill in the missing code for the ``spider::crawl()`` method.
   *Notes:*

   -  You will have to use appropriate methods for fetching a web page,
      processing that page, and adding that page to the finished
      ``queue``, then adding 1 to the variable named ``processed``.

      *Note:* The method for fetching requires a C-style string
      (null-terminated array of characters) for its argument, but the
      URL you are crawling is stored in a ``string`` object. Use the
      online documentation for ``string`` to look for a method of
      ``string`` that returns a C-style string with the same characters
      as that ``string`` object.

   -  Do you get the expected output, given that ``spider::process()``
      is still only printing a message?

#. Finally, complete the implementation of the ``spider::process()``
   method, compile, and test. *Note:*

   -  The method ``spider::is_image()`` currently always returns
      ``false``. In a more sophisticated crawler, this method could
      examine file extensions in order to determine whether a URL is an
      image (no need to crawl) or not.

#. Change directory to the ``cds/threads`` directory within your lab1
   directory.

   ::

          % cd ~/lab1/cds/threads

#. First, look at the sequential program ``primes.cpp`` in that
   directory. It computes a table of primes up to a maximum number, then
   counts the number of primes produced.

   ``primes.cpp`` does *not* use threads, but is an ordinary program
   with multiple loops.

   Then, compile and run the program, using the Makefile provided
   (``make primes``). Feel free to experiment. *Note:* Be prepared to
   wait...

#. Now, examine the multi-threaded program ``threads/primes2.cpp``. This
   program uses the same algorithm for determining primes, but it
   divides the range 2..\ ``MAX`` up into ``threadct`` non-overlapping
   subranges or "chunks", and has a separate ``boost`` thread determine
   the primes within each chunk. Observe the following:

   -  The loop for determining primes from 2 to ``MAX``, which was
      located in ``main()`` in the program ``primes.cpp``, has been
      relocated to a separate function ``work()``. That function has two
      arguments ``min`` and ``max``, and the loop has been modified to
      check only the subrange ``min``..\ ``max`` instead of the entire
      range 2..\ ``MAX``.

      Each thread will execute ``work()`` as if it were a "main program"
      for that thread.

   -  ``pool`` is an array of pointers to ``boost`` threads.

   -  Each ``boost`` thread is initialized with a unique subrange of
      2..\ ``MAX``. Threads other than the first and the last receive
      subranges of length ``segsize``. The first and last subranges are
      treated specially, to insure that the complete computation covers
      exactly the range 2..\ ``MAX``.

   -  To construct a ``boost`` thread running ``work()`` with arguments
      ``a`` and ``b``, the following constructor call would be used:

      ::

              thread(boost::bind(work, a, b))

   -  The call above constructs a thread, but that thread doesn't begin
      executing until its ``join()`` method is called. Thus, there is a
      separate loop that calls ``join()`` for all the threads in the
      array ``pool``, which starts up all the threads.

#. Now, compile and run the program ``threads/primes2.cpp``, using the
   ``Makefile`` provided (``make primes2``). This program takes an
   optional positive integer argument, for the thread count (default is
   1 thread).

#. You can time how long a program runs at the Linux or Macintosh
   command line by preceding the command with the word ``time``. For
   example, the two commands

   ::

          % time primes
          % time primes2 4

   will report the timing for running both the sequential ``primes``
   program, and the multi-threaded ``primes2`` program with four
   threads.

   Perform time tests of these two programs, for at least one run of
   ``primes``, one run of ``primes2`` with one thread, and one run of
   ``primes2`` with 4 threads. Feel free to experiment further.

#. Examine the code for the program ``parallel``. Observe the following:

   -  The same four-directory structure is used as for the ``serial``
      directory in the previous lab.

   -  The header files ``serial/include/page.hpp`` and
      ``parallel/include/page.hpp`` are identical. You can use the
      following ``diff`` command to verify this:

      ::

             $ cd ~/SD/lab1 
             $ diff serial/include/page.hpp parallel/include/page.hpp

      If the ``diff`` command finds any differences between its two file
      arguments, it will report those differences; if there are no
      differences, ``diff`` will print nothing.

   **DO THIS:** Use the ``diff`` command to compare ``raw_page.hpp`` and
   ``spider.hpp`` for these two versions ``serial`` and ``parallel``.
   *Note:* The ``diff`` program will report differences by printing
   lines that appear differently in those files. Lines that appear only
   in the first file argument to ``diff`` will be prefixed by ``<``, and
   lines that appear only in the second file will be prefixed by ``>``.

   Here are differences between ``serial/include/spider.hpp`` and
   ``parallel/include/spider.hpp``.

   -  A different selection of ``#include`` directives appears in the
      two files. In particular,

      -  ``serial/include/spider.hpp`` includes ``<queue>``, for the STL
         queue container.

      -  ``parallel/include/spider.hpp`` includes three other files
         instead of ``<queue>``. One refers to a new file
         ``atomic_counter.hpp`` that is part of this program (in the
         directory ``parallel/include``). The others provide two *TBB
         containers*, named ``tbb::concurrent_queue`` and
         ``tbb::concurrent_vector``

         TBB stands for *Intel Threading Building Blocks*, which
         provides an alternative implementation of some common
         containers. TBB also provides various parallel algorithms, but
         we will not use those algorithm features in this lab.

   -  The state variables ``m_work``, ``m_finished``, and ``m_pages``
      use the TBB container types ``tbb::concurrent_queue`` or
      ``tbb::concurrent_vector`` instead of the STL containers
      ``std::queue`` and ``std::vector``.

   -  ``parallel/include/spider.hpp`` has two new state variables:

      -  ``m_processed``, which has type ``atomic_counter`` (defined in
         ``atomic_counter.hpp``)

      -  ``m_threadCount`` of type ``size_t``, which is an integer type

   -  In ``parallel``, one of the constructors for ``spider`` has a
      second argument of type ``size_t``.

   -  There are two new methods, named ``work()`` and ``work_once()``.

   **Note:** TBB containers are used instead of STL containers because
   TBB containers are *thread safe*. This means that multiple threads
   can safely interact with a TBB container at the same time without
   causing errors. STL containers are *not* thread-safe: with STL
   containers, it is possible for two threads to interact with a
   container in a way that produces incorrect results.

   When the correct behavior of a program depends on timing, we say that
   program has a *race condition*. The parallel version of the program
   uses TBB containers in order to avoid race conditions. (Race
   conditions are discussed in other
   `CSinParallel <http://csinparallel.org/>`_ modules.)

   Likewise, the state variable ``m_processed`` has the type
   ``atomic_counter`` instead of ``int`` or ``long`` because the
   ``atomic_counter`` type is thread-safe, enabling us to avoid a race
   condition that may arise if multiple threads interact with an integer
   variable at about the same time.

#. The files ``serial/src/spider.cpp`` and ``parallel/src/spider.cpp``
   contain the main differences between these programs -- the
   ``parallel`` version uses multiple threads. Running ``diff`` shows
   these differences:

   -  In ``parallel``, one of the constructors for ``spider`` has a
      second argument, to specify the number of threads to use.

   -  The counter ``processed`` is a local variable in ``spider::crawl``
      in ``serial``. This local variable is replaced by a *state
      variable* ``m_processed`` in ``parallel``.

   -  The main work of ``crawl`` is moved into a method ``work()`` for
      the multithreaded version ``parallel``, and that version creates
      ``threadCount`` threads to carry that work out. Note that
      ``work()`` has one argument, an integer index of a thread in the
      array ``threads[]``.

   -  The method ``spider::process()`` and the rest of the code in
      ``spider.cpp`` are identical (except for missing code). The
      comment within ``process()`` indicates that the same algorithm can
      be used for ``parallel`` as for ``serial``. Why can the same
      algorithm be used in the multi-threaded version as in the
      sequential version?

#. Fill in the code indicated in two locations for the ``parallel``
   version of ``spider.cpp``, working from the code you wrote for the
   ``serial`` version, as indicated by comments in the ``parallel``
   code. Then compile and run your program.

   *Note:* This version of the program requires three command-line
   arguments: the maximum number of URLs; the number of threads to use
   (this arg is new); and the starting URL to crawl.

   Run the program with multiple threads (say, 4 threads). What do you
   observe about the run?

   -  You can examine the beginning of the output using the ``more``
      program, e.g.,

      ::

              % bin/spider 100 4 www.stolaf.edu | more

      Each thread is programmed to print a message such as

      ::

              Thread 2 finished after processing 29 URLs

      when it completes.

   -  You will probably find that a small number of threads processed
      all of the URLs, and that the other threads finished early without
      doing any work. How many threads processed URLs in your run? Can
      you think of a reason why the others finished early without
      processing any URLs? (*Hint:* Think about the work queue near the
      beginning of the program, just as the threads are starting their
      work.)

#. To spread the computational work out better among the threads,
   observe that the method ``spider::crawl()`` includes a call to a
   method ``work_once()`` that has been commented out.

   ::

          /* work_once(); */

   Remove those comments, in order to enable that call to
   ``work_once();`` . This will cause the program to process one web
   page before beginning multi-thread processing. If that first
   processed page includes several links, they will be added to the
   queue ``m_work``, so that several threads can retrieve web pages to
   process when they first begin.


