##############
More Exercises
##############

.. note:: If your device is incompatable with CUDA 6, don't edit or build the files ending with a 6

Exercise II: Vector Multiplication
##################################

Let's try the same research questions, but using a more expensive operation like multiplication.

In your vectorAdd directory, run

    ``make clean``

to remove the binary. Then run

    ``cd ..``

    ``cp -r vectorAdd vectorMult``

to create a copy of your vectorAdd folder named vectorMult. Inside there, 
rename vectorAdd.cu and vectorAdd6.cu vectorMult.cu and vectorMult6.cu
and modify the Makefile to build vectorMult and vectorMult 6 instead of vectorAdd and vectorAdd6.
Then edit vectorMult.cu and vectorMult6.cu so that they store the 
product of A[i] times B[i] in C[i] instead of the sum. 
Note that for both programs you will need to change:

- The CUDA version.
- The verification test for the CUDA version.
- The sequential version.
- The verification test for the sequential version.

Then build vectorMult and vectorMult6 and run them using 50,000
500,000 5,000,000 and 50,000,000 element arrays. 
Record the timings for each of these in your spreadsheet, 
and create charts to help us visualize the results 
like you did with the vectorAdd programs.

What are the answers to our research questions? 
How do your results compare to those of Exercise I?

Exercise III: Vector Square Root
################################

Let's try the same research questions, but this time using an even 
more expensive operation AND reducing the amount of data we're transferring.
Calculating a square root is a more expensive operation than multiplication, so let's try that.

As in Exercise II, clean and make a copy of your vectorMult folder named vectorRoot. 
Inside it, rename vectorMult.cu and vectorMult6.cu vectorRoot.cu and vectorRoot6.cu respectively. 
Modify the Makefile to build vectorRoot and vectorRoot6.

Then edit vectorRoot.cu and vectorRoot6.cu so they compute the square root of A[i] in C[i].

Then build vectorRoot and vectorRoot6 and run them using 50,000
500,000 5,000,000 and 50,000,000 element arrays. 
As before, record the timings for each of these in your 
spreadsheet, and create charts to help us visualize the results.

What are the answers to our research questions? 
How do these results compare to those of Exercises I and II?

Exercise IV: Vector Square
##########################

Let's try the same research questions one more time.
This time, we will use a less expensive operation than square root,
but keep the amount of data we're transferring the same.

As in Part III, clean and make a copy of your vectorRoot folder named vectorSquare.
Inside it, rename vectorRoot.cu and vectorRoot6.cu vectorSquare.cu and vectorSquare6.cu respectively.
Modify the Makefile to build vectorSquare and vectorSquare6.

Then edit vectorSquare.cu and vectorSquare6.cu so they compute the square of A[i] in C[i].

Then build vectorSquare and vectorSquare6 and run them using 50,000
500,000 5,000,000 and 50,000,000 array elements.
As before, record the timings for each of these in your spreadsheet, 
and create charts to help us visualize the results.

What are the answers to our research questions?
How do your results compare to those of the previous parts?
