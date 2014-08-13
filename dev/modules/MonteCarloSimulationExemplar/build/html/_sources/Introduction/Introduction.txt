What are Monte Carlo Methods?
#############################

**By Devin Bjelland, Macalester College**

Overview
--------

'Monte Carlo Methods'_ are a class of numerical methods which use repeated 
simulations to obtain a random sampling from an underlying unkown probablity 
distribution. Becuase Monte Carlo methods rely on repeated independant 
simulations, they are very well suited to distributed methods. Monte Carlo 
methods are often employed when there is no closed form solution or 
deterministic solution algorithm to the underlying problem. As this sort of 
problem is quite common, Monte Carlo methods are used in a wide variety of 
fields--from computational chemistry to finance.

The Simplest Example
--------------------

To make this concrete, imagine you have a circular target inside a larger
target. You want to find the probabibilty that if you throw a dart it will
hit the inner target. To run a 'Monte Carlo Simulation' to solve this problem,
you would simply throw a bunch of darts at the target and record the percentage
that land in the inner target. 

We can extend this idea to approximate Pi quite easily. Suppose the inner target
is a circle that is inscribed inside a sqare outer target. Since the sides of the 
square are twice the radius of the circle, we have that the ratio, :math:'\Chi' 
of the area of the circle to the area of the sqaure is :math:`\frac{\pi {r}^2}{{\left( 2r \right)^2}`. We can rearrange this formula to solve for Pi as follows:

Now, we can empirically calculate a value for the ratio of the area of the circle
to the area of the square with a Monte Carlo simulation. We pick lots of random
points in the squre and the ratio is the number of points inside the circle 
divided by the total number of points. 

Simulating Card Games for Fun and Profit
---------------------------------------- 

The original motivating example that Monte Carlo methods draw their name 
from is gambling and game playing. In this module, we develop parallel 
algorithms that approximate the probabilities of various outcomes in card 
games. 

.. _Monte Carlo Methods: http://en.wikipedia.org/wiki/Monte_Carlo_method

About Randomness
################

How Computers Generate Random Numbers
-------------------------------------

Algorithms that implement Monte Carlo methods require a source of of 
randomness. If we are writing a serial algorithm, we can simply use the 
standard random number generator, for example, rand() in C. Computers (unless 
specifically fitted with a chip which gathers randomness from some other source) 
cannot produce true random numbers. Rather, the standard library random number 
generator takes some relatively random input--in the case of unix systems 
the number of seconds since January 1, 1970--and transforms it sequentially 
to produce a stream of psuedo-random numbers. 

What this means for distributed programming
-------------------------------------------

If your algorithm creates multiple threads to perform a Monte Carlo simulation 
and you invoke the standard random number generator with the same seed, all the 
threads will get the same random numbers. If this happens, all the extra threads 
aren't doing any good since they are just running the same simulation over again. 
We get around this by seeding the random number generate with more information--
some combination of the time and the thread ID. 

