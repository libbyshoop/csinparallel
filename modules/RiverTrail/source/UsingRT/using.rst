Using River Trail to Parallelize Javascript
=============================================

Installing River Trail and Firebug
--------------------------------------------

The River Trail  project provides prebuilt binaries for Windows and Mac OS X. The project does not provide binaries for Linux, although with some difficulty one can build a version for Linux. 

On a Windows 7 or OS X machine, open Firefox, and note the version. For OS X, this can be a public machine that you do not have administrative access to.

For Windows, you need to first download and install `the intel OpenCL
sdk <http://software.intel.com/en-us/vcsource/tools/opencl-sdk>`_

Then, on either operating system, install `the River Trail
Plugin <https://github.com/RiverTrail/RiverTrail/wiki/downloads>`_
appropriate for your version of Firefox.

Now, install firebug: `https://getfirebug.com/ <https://getfirebug.com/>`_. This is a web development debugger and general purpose console tool. It should install the proper version on its own. It is needed to do code timings the nice way.

First Experimenting
----------------------

Now open the `interpreter <http://rivertrail.github.io/interactive/>`_. Make sure that the right hand window has the text “Enabling Parallel Mode” to be sure that the add-on and OpenCL library are installed correctly.

Now, try each of the example statements from the left-hand window to get a
feel for each call.  Note constructor function for ParallelArray.

The example uses a literal for the array it changes to ParallelArray but you can use something like::

    var a = [0,2,4,5,6]; //if you want to
    var b = new ParallelArray(a);

Note the reduce call at the bottom of the example statements which calls a function 'sum' to sum all elements.  Check out the `documentation <https://github.com/RiverTrail/RiverTrail/wiki/ParallelArray>`_ for an explanation of each argument (scroll down for methods).

Download for the Lab Exercise
-----------------------------

You will need to download two files. One, :download:`ts.html <ts.html>`, is an HTML file which has a super simple interface. The other, :download:`MapRedux.js <MapRedux.js>`, is a Javascript file to modify.  Download these into the same directory.

Also, download :download:`data.zip <data.zip>`, a zip file which contains three data files.  Each contains a different number of lines from the movData Movielens dataset.


Timing Word Count
-----------------------

Open the html file in Firefox, and try pasting a paragraph or more of text into the input box. Then click the 'Run Word Count' button and see the results.

Now open the MapRedux.js file in an editor, and look at the function
``p``. Note the calls to create a new ParallelArray, to map, and then the
reduce call.

In a README.txt file:

Copy the call to 'scatter' and, using `the documentation <https://github.com/RiverTrail/RiverTrail/blob/master/tutorial/RiverTrail-tutorial.pdf?raw=true>`_, explain what this function is doing.

In your README, also explain how this is different than reduce

Now lets add some timing code.

When you have firebug installed you can add a timer with a name like so::

    console.time(“timerName”)

and close the timer like this::

    console.timeEnd(“timerName”)

Add the timer code around the calls to ``map`` and ``scatter`` and assign the return of console.timeEnd(“timerName”) to a variable like this::

    var timed = console.timeEnd(“timerName”) to a variable

Then add that variable to the output string (see comments for hints)::

    rstr += '<p>'+timed+'</p>';

Now, with the timing code in place, test inputs of various sizes. Copy the word count, and time in ms to your README.  Do 3 runs: one with a paragraph, one with about a page, and one with several pages.

Average Rating by User
-----------------------

Open one of the movie lens text files, copy everything, and paste it
into the input box.

Hit the 'Run Average' button, and view the results. This gives us how long it took total, but it does not tell us how many ratings there were.

Now look in the function ``q`` and read the comments to get a feel for what is
happening. Then go to the for-in loop and find the line where we add the data for each id to an output string.

Add another table-data element within the row (after the </td> but before </tr>) containing the number of ratings for that id::

    <td>+aValueLikeRatings.length+</td>

Do not forget to balance the rows (see comment).

This gives you the number of ratings per id, lets you add the total number of
ratings

If you like, you can print the number of lines.  Declare a var outside the for-in, ``var collecta = 0;`` and within the for-in, insert ``collecta += ratings.length;``. Add this in a <p> before the table (like the timer).

Now time each of the different sized files. Copy the time, and the number of ratings into your README.

Average Rating per Movie
------------------------

In the for loop that copies bigArr into smArr, we add keys to an object when they don't already exist (bucketing). When the key does exist, we add the rating for that key to a string. This gives unique keys, with their accompanying values

Note the ordering of the values

Change the operation to add the movie-id to the object instead of
user-id. Think smArr[j+1] instead of smArr[j].  Rerun your datasets, and record times.

Are the times different? If so, why do you think it was different?

