Using WebMapReduce (WMR)
========================

Introduction
------------

For this activity, you need to have read the accompanying
background reading in the previous section entitled
*Using Parallelism to Analyze Very Large Files: Google's MapReduce Paradigm*.

In this lab you will use a web-based program called `WebMapReduce`
(**WMR**) that enables you to formulate map-reduce computations and
run them on an underlying *Hadoop* map-reduce system running on a
cluster of computers.

You will use WebMapReduce on a cluster of computers at Macalester
College. You should access WebMapReduce now and register for a
login by going to this URL on your web browser:

http://selkie.macalester.edu/wmr

Choose the link at the very upper right of this page that says
'Register'. Use your Macalester email address as your login name,
and provide the other information asked for. You choose your own
password, so that you can remember it and have control of your
account.


For later reference, you may want to check this `documentation for
WMR <http://webmapreduce.sourceforge.net/docs/using/index.html>`_.

For this activity, you should be able to follow along with the
instructions below and determine how to use WMR.

An example of map-reduce computing with WMR: counting words
-----------------------------------------------------------

To program with map-reduce, you must first decide how to use
mappers and reducers in order to accomplish your computational
goal. The mapper function should take a line of input and decompose
it somehow into key-value pairs; then the reducer should somehow
condense or analyze all the key-value pairs having a *common key*,
and produce a desired result.

*The following example is small to illustrate how the process works.*
In realistic applications, which you will try later in this
activity and in homework, the input data is much larger (several to
hundreds of Gigabytes) and is stored in the Hadoop system. You will
go through the following exercise first to ensure that the code is
working and that you understand how it works. Then you can move on
to bigger files. This is the process you should go through when
doing this kind of coding: work on small amounts of data first to
ensure correctness of your code, then try larger amounts of data.

As an example, consider the problem of counting how frequently each
word appears in a collection of text data. For example, if the input
data in a file is:

::

    One fish, Two fish,
    Red Fish, Blue fish.
    Blue Fish, Two Fish.

then the output should be:

::

    Blue 2
    One 1
    Red 1
    Two 2
    fish, 2
    Fish, 2
    fish. 1
    Fish. 1

As this output indicates, we did not make any attempt to trim
punctuation characters in this first example. Nor did we consider that
some words in our text might be capatalized and some may not.
We will also not do so as we
practice using WebMapReduce with the initial functions described
below. However, you can consider adding punctuation removal and 
lowercase conversion to your mapper code as you work through the example.

.. Note::

  The WebMapReduce system will sort the words
  according to the ASCII codes of the characters within words.

What follows is a plan for the mapper and reducer functions. 

Map-reduce plan
^^^^^^^^^^^^^^^

In WMR, mapper functions work simultaneously on lines of input from
files, where a line ends with a newline charater. The mapper will
produce one key-value pair (*w*, *count*) for each and every word encountered
in the input line that it is working on.

Thus, on the above input, three mappers working together, one on each line,
could emit the following combined output:

::

    One 1
    fish, 1
    fish, 1
    Two 1

    Red 1
    Fish, 1
    Blue 1
    fish. 1

    Blue 1
    Fish, 1
    Two 1
    Fish. 1

The reducers will compute the sum of all the *count* values for a
given word *w*, then produce the key-value pair (*w*, *sum*).

In this example, we can envision a reducer for each distinct word found
by the three mappers, where the reducer gets a list of single counts 
per occurance of the word that a mapper found, loking like this:

::

    One    [1]
    fish,  [1, 1]
    Two    [1,1]
    Red    [1]
    Fish,  [1,1]
    Blue   [1,1]
    fish.  [1]
    Fish.  [1]

Each reducer works on one of the pairs of data shown above, and the system
handles creating words with the lists of counts as shown above.

One System, Many Languages
--------------------------

In map-reduce framework systems in general and in WMR specifically, you can
use one of several programming languages to code your mapper and reducer functions.  
The following table contains links to solutions in several languages for 
the word-count solution we describe below.

========    ====================================================    =====================================================
Lnaguage    Mapper function code                                    Reducer function code
========    ====================================================    =====================================================
Python      :download:`wcmapper.py <code/python3/wcmapper.py>`      :download:`wcreducer.py <code/python3/wcreducer.py>`
C++         :download:`wcmapper.cpp <code/cpp/wcmapper.cpp>`        :download:`wcreducer.cpp <code/cpp/wcreducer.cpp>`
C           :download:`wcmapper.c <code/c/wcmapper.c>`              :download:`wcreducer.c <code/c/wcreducer.c>`
Java        :download:`wcmapper.java <code/java/wcmapper.java>`     :download:`wcreducer.java <code/java/wcreducer.java>`
========    ====================================================    =====================================================



The mapper function
-------------------

Each mapper process is receiving a line from a file as its key initially
when the process starts (the value is empty, or null).  You write one mapper 
function to be executed by those prcesses on any given line from any particular file.  
Our goal is to have the mapper output a new (key, value) containing a word found
and the number one, as shown for the three-mapper example above.

Here is psedocode for what a typical mapper might accomplish::

  # key is a single line from a file.
  #
  # value is empty in this case, since this is the first mapper function
  # we are applying.
  #
  function mapper(key, value)
    1) Take the key argument to this function, which is the line of text, 
       and split it on whitespace
    
    2) For every word resulting from the split key line:
        
        'emit' a pair (word, "1") to the system for the reducers to handle


Here is a Python3 mapper function for accomplishing this task using
the WMR system. We include the feature of stripping away
puncuation characters from the input and converting each word found to
lowercase.


.. literalinclude::  code/python3/wcmapper.py
    :linenos:
    :language: python


This code is available for download in the table above, as are versions 
in other languages.  Note that in each language you will need to know
how to specify the (key, value) pair to emit to the system for the reducers 
to process. We see this for Python in line 4 above.



The reducer function
--------------------

In the system, there will be reducer processes to handle each word 'key' 
that was emitted by various mappers.  You write reducer code as if your reducer 
function is getting one word key and a container of counts, where each count 
came from a different mapper that was working on a different line of a file or files.
In this simplest example, each count is a '1', each of which will be summed together 
by the reducer handling the particular word as a key.

Pseudocode for a reducer for this problem looks like this::

  # key is a single word, values is a 'container' of counts that were
  # gathered by the system from every mapper
  #
  function reducer(key, values)
    
    set a sum accumulator to zero

    for each count in values
      accumulate the sum by adding count to it

    'emit' the (key, sum) pair

A reducer function for solving the word-count problem in Python is


.. literalinclude::  code/python3/wcreducer.py
    :linenos:
    :language: python


This code is also available in the table above containing versions
in several languages.

The function ``reducer()`` is called once for each distinct key
that appears among the key-value pairs emitted by the mapper, and
that call processes all of the key-value pairs that use that key.
On line 1, the two parameters that are arguments of ``reducer()``
are that one distinct ``key`` and a Python3 *iterator* (similar to a
list, but not quite) called ``values``, which provides access to
all the values for that key. Iterators in Python3 are designed for
``for`` loops- note in line 3 that we can simply ask for each value
one at a time from the set of values held by the iterator.

*Rationale:* WMR ``reducer()`` functions use iterators instead of
lists because the number of values may be very large in the case of
large data. For example, there would be billions of occurrences of
the word "the" if our data consisted of all pages on the web. When
there are a lot of key-value pairs, it is more efficient to
dispense pairs one at a time through an iterator than to create a
gigantic complete list and hold that list in main memory; also, an
enormous list may overfill main memory.

The ``reducer()`` function adds up all the counts that appear in
key-value pairs for the ``key`` that appears as ``reducer()``'s
first argument (recall these come from separate mappers). Each
count provided by the iterator ``values`` is a string, so in line 4
we must first convert it to an integer before adding it to the
running total ``sum``.

The method ``Wmr.emit()`` is used to produce key-value pairs as
output from the mapper. This time, only one pair is emitted,
consisting of the word being counted and ``sum``, which holds the
number of times that word appeared in *all* of the original data.

Running the example code on WebMapReduce
----------------------------------------

To run WMR with this combination of data, mapper, and reducer,
carry out the following steps.

- In a browser, visit the WMR site at (if you don't already have it
  open from registering):

    http://selkie.macalester.edu/wmr

- After you have registered, you can use your email address and
  password to login. After successfully logging in, you are taken to
  the WMR page where you can complete your work.

- Enter a job name (perhaps involving your username, for uniqueness;
  avoid spaces in the job name and make sure that it is more than 4
  characters long).

- Choose the language that you wish to try.

- For now, you can leave the number of map tasks and reduce tasks
  blank. This will let the system decide this for itself. You can
  also leave the default choice of sorting alphabetically.

- Enter the input data, e.g., the fish lines above. You can use the
  `Direct Input` option and enter that data in the text box
  provided.

- Enter the mapper. It's probably best to use se the \`\`Upload"
  option and navigate to a file that contains the mapper, which you
  have entered using an editor (this is more convenient for repeated
  runs). Or you can use the file we provided in a table of links above.
    
    **Beware:** cutting and pasting your code from a pdf file or
    a web page or typing it into the \`'direct' entry box for Python
    code is a bit problematic, because the needed tabs in the code
    might not be preserved (although using spaces should work). Check
    that the appropriate radio button is clicked to indicate the source
    option you're actually using.

- Also enter the reducer.  Again, it's easier to use the file provided 
  with a link in the table above.

- Click on the submit button.

A page should appear indicating that the job started successfully.
This page will refresh itself as it is working on the job to show
you progress.

Once the job runs to completion, you should see a Job Complete page.
This page will include your output. If you used the fish input,
your output should match the illustration above, except that the
punctuation should also be taken care of.


If something doesn't work as described here, the following section
may help with troubleshooting. *Read it next in any case so that you
know what you can do when you work on your own new examples.*

Using WMR and its test mode
---------------------------

Here is some information about developing WMR map-reduce
programs,and what to do if something goes wrong with your WMR job.

-  First, a reminder:

   -  At present, the WMR interface does not automatically reset radio
      buttons for you when you upload a file or use `Distributed
      FileSystem` data generated from a prior map-reduce run.
      *Always check to see that the radio buttons select the data, mapper, and reducer resources you intend.*


-  You can test your mapper alone without using your reducer by
   using the *identity reducer*, which simply emits the same key-value
   pairs that it receives. Here is an implementation of the identity
   reducer for Python:

  .. literalinclude:: code/id-identity/idreducer.py
    :linenos:
    :language: python




  As an example, if you use the word-count mapper with the identity reducer, then the "fish" data 
  above should produce the following output:

   ::

      Blue 1
      Blue  1
      fish, 1
      fish, 1
      fish. 1
      Fish, 1
      Fish, 1
      Fish. 1
      One 1
      Red 1
      Two 1
      Two 1

  Observe that the output is sorted, due to the shuffling step.
  However, this does show all the key-value pairs that result from
  the word-count mapper.

-  Likewise, you can test your reducer alone without using your
   mapper by substituting the ``identity mapper``, which simply copies
   key-value pairs from lines of input data. Here is an implementation
   of the identity mapper in Python:

   
   .. literalinclude:: code/id-identity/idmapper.py
    :linenos:
    :language: python

Here are identity mappers and reducers for some languages.

========    ===========================================================    =============================================================
Lnaguage    Mapper function code                                           Reducer function code
========    ===========================================================    =============================================================
Python      :download:`idmapper.py <code/id-identity/idmapper.py>`         :download:`idreducer.py <code/id-identity/idreducer.py>`
C++         :download:`idmapper.cpp <code/id-identity/idmapper.cpp>`       :download:`idreducer.cpp <code/id-identity/idreducer.cpp>`
Java        :download:`idmapper.java <code/id-identity/idmapper.java>`     :download:`idreducer.java <code/id-identity/idreducer.java>`
========    ===========================================================    =============================================================


   For example, you could enter a small amount of input data that you
   expect your mapper to produce, such as the ``TAB``-separated
   key-value pairs listed above from using the identity reducer. If
   you then use the identity mapper ``idmapper.py`` with the
   word-count reducer ``wcreducer.py`` you should get the following
   output, which we would expect from each stage working:

   ::

      Blue 2
      fish, 2
      fish. 1
      Fish, 2
      Fish. 1
      One 1
      Red 1
      Two 2

   *Note:* Use a ``TAB`` character to separate the key and value in
   the input lines above. To keep a test case around, it is easier to
   enter your data in an editor, then cut and paste to enter that data
   in the text box. Alternatively, you can"Upload" a file that
   contains the data.

-  Unfortunately, the current WMR system does *not* provide very
   useful error messages in many cases. For example, if there's an
   error in your Python code, no clue about that error can be passed
   along in the current system.

-  In order to test or debug a mapper and reducer, you can use the
   ``Test`` Button at the bottom of the WMR web page. The job output
   from this choice shows you what both the mapper and reducer
   emitted, which can be helpful for debugging your code.

   .. note:: Do not use ``Test`` for large data, but only to debug
               your mappers and reducers. This option does *not* use cluster
               computing, so it cannot handle large data.


Next Steps
----------


#. In WMR, you can choose to use your own input data files. Try
   choosing to 'browse' for a file, and using this
   :download:`mobydick.txt <mobydick.txt>` as file
   input.

#. You have likely noticed that capitalized words are treated
   separately from lowercase words. Change your mapper to also convert
   each word to lowercase before checking whether it is in the
   dictionary.

#. Also remove punctuation from each word after splitting the line 
   (so all the 'fish' variations get counted together).

#. There are a large number of files of books from Project
   Gutenberg available on the Hadoop system underlying WebMapReduce.
   First start by trying this book as an input file by choosing
   'Cluster Path' as Input in WMR and typing one of these into the
   text box:

   | /shared/gutenberg/WarAndPeace.txt
   | /shared/gutenberg/CompleteShakespeare.txt
   | /shared/gutenberg/AnnaKarenina.txt

   These books have many lines of text with 'newline' chacters at the
   end of each line. Each of you mapper functions works on one line.
   Try one of these.

#. Next, you should try a collection of many books, each of which
   has no newline characters in them. In this case, each mapper 'task'
   in Hadoop will work on one whole book (your dictionary of words per
   mapper will be for the whole book, and the mappers will be running
   on many books at one time). In the Hadoop file system inside WMR we
   have these datasets available for this:

       =====================================    ================
       'Cluster path' to enter in WMR           Number of books
       =====================================    ================
       /shared/gutenberg/all\_nonl/group10      2018
       /shared/gutenberg/all\_nonl/group11      294
       /shared/gutenberg/all\_nonl/group6       830
       /shared/gutenberg/all\_nonl/group8       541
       =====================================    ================

   While using many books, it will be useful for you to experiment
   with the different datasets so that you can get a sense for how
   much a system like Hadoop can process.

   To do this, it will also be useful for you to save your
   configuration so that you can use it again with a different number
   of reducer tasks. Once you have entered your mapper and reducer
   code, picked the Python3 language, and given your job a descriptive
   name, choose the `'Save'` button at the bottom of the WMR panel.
   This will now be a `'Saved Configuration'` that you can retrieve
   using the link on the left in the WMR page.

   Try using the smallest set first (group11). Do not enter anything
   in the map tasks box- notice that the system will choose the same
   number of mappers as the number of books (this will show up once
   you submit the job). Also do not enter anything for the number of
   reduce tasks. With that many books, when the job completes you will
   see there are many pages of output, and some interesting 'words'.
   For the 294 books in group11, note how you obtain several pages of
   results. You will also notice that the stripping of punctuation
   isn't perfect. If you wish to try improving this you could, but it
   is not necessary.


Additional Notes
----------------

It is possible that input data files to mappers may be treated
differently than as described in the above example. For example, a
mapper function might be used as a second pass by operating on the
reducer results from a previous map-reduce cycle. Or the data may
be formatted differently than a text file from a novel or poem.
These notes pertain to those cases.

In WMR, each line of data is treated as a key-value pair of
strings, where the key consists of all characters before the first
``TAB`` character in that line, and the value consists of all
characters after that first ``TAB`` character. Each call of
``mapper()`` operates on one line and that function's two arguments
are the key and value from that line.

If there are multiple ``TAB`` characters in a line, all ``TAB``
characters after the first remain as part of the ``value`` argument
of ``mapper()``.

If there are *no* ``TAB`` characters in a data line (as is the case
with all of our fish lines above), an empty string is created for
the value and the entire line's content is treated as the key. This
is why the key is split in the mapper shown above.


