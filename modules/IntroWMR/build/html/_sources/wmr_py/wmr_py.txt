Using WebMapReduce (WMR)
========================

Introduction
------------

For this activity, you need to have read the accompanying
background reading entitled
*Using Parallelism to Analyze Very Large Files: Google's MapReduce Paradigm*.

In this lab you will use a web-based program called `WebMapReduce`
(**WMR**) that enables you to formulate map-reduce computations and
run them on an underlying *Hadoop* map-reduce system running on a
cluster of computers.

You will use WebMapReduce on a cluster of computers at St. Olaf
College. You should access WebMapReduce now and register for a
login by going to this URL on your web browser:

http://selkie.macalester.edu/wmr

Choose the link at the very upper right of this page that says
'Register'. Use your Macalester email address as your login name,
and provide the other information asked for. You choose your own
password, so that you can remember it and have control of your
account.

WMR has several languages as options, and Python is one of them,
although it expects Python3. Though we have been using Python 2.7
in our class, the difference for the following example is really
close to what you have already used. We will highlight the
difference below.

For later reference, you may want to check this documentation for
WMR:

http://webmapreduce.sourceforge.net/docs/User_Guide/index.html

This WMR user guide has images from the first version of WMR, but a
great deal of the information about how to use it remains the same.
For this activity, you should be able to follow along with the
instructions and determine how to use WMR.

An example of map-reduce computing with WMR: counting words
-----------------------------------------------------------

To program with map-reduce, you must first decide how to use
mappers and reducers in order to accomplish your computational
goal. The mapper function should take a line of input and decompose
it somehow into key-value pairs; then the reducer should somehow
condense or analyze all the key-value pairs having a common key,
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
word appears in a collection of data. For example, if the input
data in a file is:

::

    One fish, Two fish,
    Red fish, Blue fish.

then the output should be:

::

    Blue 1
    One 1
    Red 1
    Two 1
    fish, 3
    fish. 1

As this output indicates, we did not make any attempt to trim
punctuation characters in this first example. We will do so as we
practice using WebMapReduce with the initial functions described
below. (Note that the WebMapReduce system will sort the words
according to the ASCII codes of the characters within words.)

What follows is a plan for the mapper and reducer functions. You
should compare and note the similarity between these and your
sequential function for completing this same task on a single input
file.

Map-reduce plan
^^^^^^^^^^^^^^^

In WMR, mapper functions work simultaneously on lines of input from
files, where a line ends with a newline charater. The mapper will
produce one key-value pair (*w*, *count*) foreach word encountered
in the input line that it is working on.

Thus, on the above input, two mappers working together on each line
would emit the following combined output:

::

    One 1
    fish, 2
    Two 1
    Red 1
    fish, 1
    Blue 1
    fish. 1

The reducers will compute the sum of all the *count* values for a
given word *w*, then produce the key-value pair (*w*, *sum*).


The mapper function
-------------------

Here is a Python3 mapper function for accomplishing this task using
the WMR system. We also add the feature of stripping away
puncuation characters from the input.

.. sourcecode:: python

    import string

    def mapper(key, value):
        counts = dict()
        words=key.split()
        for word in words:
            word = word.strip(string.punctuation)
            if word not in counts:
                counts[word] = 1
            else:
                counts[word] += 1

        for foundword in counts:
            Wmr.emit(foundword, counts[foundword])



This code is available on moodle as `wc\\\_comb\\\_mapper.py`.

Let's examine this code carefully. In line 1 we import the Python
``string`` package so that we can use its method for returning
punctuation characters, found in line 7. Line 3 shows how all
mapper functions in WMR should be defined, with two parameters
called `key` and `value`. Each of these parameters is a *String*
data type. In the case of our first mapper functions reading each
line of the file, the whole line is passed into the key in the
map-reduce system underlying WMR, and the value is empty. (See
additional notes section below for more details you will need when
trying other examples.)

In line 4, we create a Python dictionary called `counts` to hold
each distinct word and the number of time it appears. In the small
input example we describe here, this will not have many entries.
When we next read files where a whole book may be contained in one
line of data, the dictionary called counts will contain many
words.

Line 5 is where we take the input line, which was in the `key`, and
break it into words. Then the loop in lines 6-11 goes word by word
and strips punctuation and increments the count of that word.

The loop in lines 13 and 14 is how we send the data off to the
reducers. The WMR system for Python3 defines a class ``Wmr``that
includes a class method ``emit()`` for producing key-value pairs to
be forwarded (via shuffling) to a reducer. ``Wmr.emit()`` requires
two string arguments, so both `foundword` and `counts[foundword]`
are being interpreted as strings in line 14.


The reducer function
--------------------

A reducer function for solving the word-count problem is

        def reducer(key, values): sum = 0 for count in values: sum +=
        int(count) Wmr.emit(key, sum)



This code is available on moodle as ``wcreducer.py``.

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

    In a browser, visit the WMR site at (if you don't already have it
    open from registering):

    http://selkie.macalester.edu/wmr

    After you have registered, you can use your email address and
    password to login. After successfully logging in, you are taken to
    the WMR page where you can complete your work.

    Enter a job name (perhaps involving your username, for uniqueness;
    avoid spaces in the job name and make sure that it is more than 4
    characters long).

    Choose the Python3 language.

    For now, you can leave the number of map tasks and reduce tasks
    blank. This will let the system decide this for itself. You canhttp://selkie.macalester.edu/csinparallel/modules/DiningPhilosophers/source/Introduction/dining_philosophers_code.tar.gz
    also leave the default choice of sorting alphabetically.

    Enter the input data, e.g., the fish lines above. You can use the
    \`\`Direct Input" option and enter that data in the text box
    provided.

    Enter the mapper. It's probably best to use se the \`\`Upload"
    option and navigate to a file that contains the mapper, which you
    have entered using an editor (this is more convenient for repeated
    runs). **Beware:** cutting and pasting your code from a pdf file or
    a web page or typing it into the \`'direct' entry box for python
    code is a bit problematic, because the needed tabs in the code
    might not be preserved (although using spaces should work). Check
    that the appropriate radio button is clicked to indicate the source
    option you're actually using.

    Also enter the reducer.

    Click on the submit button.

    A page should appear indicating that the job started successfully.
    This page will refresh itself as it is working on the job to show
    you progress.

    Once the job runs to completion, you should see a Job Complete page.
    This page will include your output. If you used the fish input,
    your output should match the illustration above, except that the
    punctuation should also be taken care of.


If something doesn't work as described here, the following section
may help with troubleshooting. Read it next in any case so that you
know what you can do when you work on your own new examples.

Using WMR and its test mode
---------------------------

Here is some information about developing WMR map-reduce
programs,and what to do if something goes wrong with your WMR job.

-  First, some reminders:

   -  At present, only the Python3 language is supported for providing
      only mapper and reducer functions in WMR programming. For us, this
      should not be any different than the Python 2 programming that
      we've been doing for this course, except for the use of the
      iterator in the reducer, as described above.

   -  At present, the WMR interface does not automatically reset radio
      buttons for you when you upload a file or use \`Distributed
      FileSystem" data generated from a prior map-reduce run.
      *Always check to see that the radio buttons select the data, mapper, and reduce resources you intend.*


-  You can test your mapper alone without using your reducer by
   using the *identity reducer*, which simply emits the same key-value
   pairs that it receives. Here is an implementation of the identity
   reducer for Python:

           def reducer(key, iter): for s in iter: Wmr.emit(key, s)



   (Available as ``idreducer.py``.)

   For example, if you use the word-count mapper
   `wc\\\_comb\\\_mapper.py` with the identity reducer
   ``idreducer.py``, then the "fish" data above should produce the
   following output:

   ::

       Blue    1
       fish    2
       fish    2
       One 1
       Red 1
       Two 1

   Observe that the output is sorted, due to the shuffling step.
   However, this does show all the key-value pairs that result from
   the word-count mapper.

-  Likewise, you can test your reducer alone without using your
   mapper by substituting the ``identity mapper``, which simply copies
   key-value pairs from lines of input data. Here is an implementation
   of the identity mapper in Python:

   .. sourcecode:: python

           def mapper(key, value):
               Wmr.emit(key, value)



   (Available as ``idmapper.py``.)

   For example, you could enter a small amount of input data that you
   expect your mapper to produce, such as the ``TAB``-separated
   key-value pairs listed above from using the identity reducer. If
   you then use the identity mapper ``idmapper.py`` with the
   word-count reducer ``wcreducer.py`` you should get the following
   output, which we would expect from each stage working:

   ::

       Blue    1
       fish    4
       One     1
       Red     1
       Two     1

   *Note:* Use a ``TAB`` character to separate the key and value in
   the input lines above. To kep a test case around, it is easier to
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

   .. note:: `Do not use ``Test`` for large data`, but only to debug
               your mappers and reducers. This option does *not* use cluster
               computing, so it cannot handle large data.


Next Steps
----------


#. In WMR, you can choose to use your own input data files. Try
   choosing to 'browse' for a file, and using ``mobydick.txt`` as file
   input.

#. You have likely noticed that capitalized words are treated
   separately from lowercase words. Change your mapper to also convert
   each word to lowercase before checking whether it is in the
   dictionary.

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

       ===================================    ================
       'Cluster path' to enter in WMR         Number of books
       ===================================    ================
       /shared/gutenberg/all\_nonl/group10    2018
       /shared/gutenberg/all\_nonl/group11    294
       /shared/gutenberg/all\_nonl/group6     830
       /shared/gutenberg/all\_nonl/group8     541
       ===================================    ================

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


