#########################
Advanced Network Analysis
#########################

For this next excersise we will try to find the clustering
coefficient for each node. The clustering coefficient is a number
from 0-1 that represents how closely connected a node's
neighborhood (the set of all the points connected to the node)
is. It is calculated by counting all of the edges within a 
neighborhood and then dividing that number by the number of 
edges that could exist within the neighborhood. So if all of
an account's friends are friends with each other, that account's
clustering coefficient is 1 and if none of the account's
friends are friends with each other, the account's clustering
coefficient is 0.

First we will need to have a list of the friends and friends of
friends for every account. We can do this by sending each
account's list of friends to each of it's friends. 

If you remember from the last chapter we already created friend
lists for each account. However some accounts have thousands of
friends. This is a problem because hadoop limits the size of 
the values that a mapper can emit. We can get around this
limitation by breaking the friend lists into chunks.

Use the same mapper from last chapter, but use this modified 
reducer instead:

.. code-block:: python
    :linenos:
    
    def reducer(key, values):
      neighbors = set()
      for value in values:
        if len(neighbors) > 50:
          Wmr.emit(key, ','.join(neighbors))
          neighbors = set()
        neighbors.add(value)
      if len(neighbors) > 0:
        Wmr.emit(key, ','.join(neighbors))

Take the output from that job and use it as input for this
job.

Now that the input is broken up into useable chunks we can write
a mapper note that the mapper emits both the friends list and 
the friend from which the list came. This is so that the 
accounts can reconstruct their friends list in the reducer..

.. code-block:: python
    :linenos:

    def mapper(key, value):
      friends = value.split(',')
      for friend in friends:
        Wmr.emit(friend, (key, value))

The reducer takes these lists and constructs a set of it's 
neighbors that are one hop away and a dictionary that counts
how many times a point appears in the two hop lists. Now we
have to do a bit of mathematical reasoning.

Alice has N friends. The sum of the twoHop dictionary entries for
all of her friends represents how many times each of her friends
appear on each of her other friends' friend lists. We'll refer
to this number as T.

If all of Alice's friends are friends with each other, 
T = N * (N-1) because each of her N friend's are friends with the
other N-1. 

However each friendship adds 2 to the value of T because each
member of the friendship increments the value of the other in the
twoHop dict.

Therefore T/2 is the total number of friendships within Alice's 
friend group and (N * (N-1)) / 2 is the most possible friendships

Therefore Alice's clustering coefficient is 
(T/2) / (N * (N-1)/2) or simply T / (N * (N-1)) after algebraic
manipulation.

After putting that all together we can finally write our reducer:

.. code-block:: python
    :linenos:

    def reducer(key, values):
      oneHops = set()             #Alice's friend list
      twoHops = {}                #the two hop dict
      for value in values:        
        node, hops = eval(value)  #unpack the values
        oneHops.add(node)         #reconstruct the friend list
        hops = hops.split(',')
        for hop in hops:          #build the two hop dict
          if hop in twoHops:
            twoHops[hop] += 1
          else:
            twoHops[hop] = 1
            n = len(oneHops)
      if n < 2:                   #if a point has less than 2 
        Wmr.emit(key, 0)          #neighbors it's cc is 0
      else:
        total = 0    
        for hop in oneHops:       #calculate the value of T
          if hop in twoHops:
            total += twoHops[hop]
        cc = total / (n * (n-1))  #calculate the cc
        Wmr.emit(key, cc)
