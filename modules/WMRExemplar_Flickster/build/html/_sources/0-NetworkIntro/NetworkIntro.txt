############################
Network Analysis with Hadoop
############################

Network analysis is an important tool that has wide-ranging
application from biology to marketing. This chapter will
teach some basic techniques and show you how to chain 
together hadoop jobs using WMR to answer complicated questions.

The Dataset
***********

    
The dataset we are using is a list of friendships on Flixster,
a social movie recommendation website. The keys and values are
numbers representing the two parties involved in a friendship. 
There is no significance to whether a friend is a key or a value.

.. topic:: System-dependent Alert

    The path of the dataset shown below may not be the same on your WMR system.
    It is correct for this WMR server:
    
    selkie.macalester.edu/wmr

The location of the dataset to use is called `/shared/Flixster/edges.tsv`.
Enter this in the **Cluster Path** field on the WMR interface.

Getting a List of Friends
*************************

One of the basic network operations is retrieving a list of
neighbors per node from a list of edges. In our case this
means getting a list of friends from a list of friendships.
The algorithm is quite simple, for each friend f in a friendship
we must add f's friend to the list of f's friends.

Here's our :download:`mapper <friendListMapper.py>`:

.. code-block:: python
    :linenos:

    def mapper(key, value):
      #make sure our input is good
      if not(key in ('', None) or value in ('', None)):
        Wmr.emit(key, value)
        Wmr.emit(value, key)
 
We want our :download:`reducer <friendListReducer.py>` to output a comma
seperated list:

.. code-block:: python
    :linenos:

    def reducer(key, values):
      neighbors = set()           #using a set ensures uniqueness
      for value in values:
        neighbors.add(value)
      output = ','.join(neighbors)
      Wmr.emit(key, output)

Average Friend Count
********************

The output of the last job was interesting but doesn't tell us
much about the dataset as a whole. What if we wanted to know
the average number of friends per Flixster account? This answer
would be extremely difficult to answer in a single job. Luckily
we can use the output of the last job as input for a new job.
All you need to do is click the Use Output button at the top or
bottom of the WMR results page.

To get the average, our :download:`mapper <friendCountMapper.py>` will
output the number of friends each account has to one :download:`reducer <averageReducer.py>` 
that then calculates the average.

.. code-block:: python
    :linenos:

    def mapper(key, value):
      friends = value.split(',')
      Wmr.emit('Avg:', len(friends))

    def reducer(key, values):
      count = 0
      total = 0
      for value in values:
         count += 1
         total += int(value)
      Wmr.emit(key, total / count)

.. note::
    It's always a good idea to save the code you write for
    hadoop jobs as it is easily reusable.

Submit the job. If you did everything correctly, you should get
Avg: 7.289679 as the output. That's it, you now know how to
chain Hadoop jobs. In the next chapter we'll cover some more
advanced network analysis operations.
