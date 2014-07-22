#include <stdlib.h>   //malloc and free
#include <stdio.h>    //printf
#include <omp.h>      //OpenMP

// Very small values for this simple illustrative example
#define ARRAY_SIZE 8     //Size of arrays whose elements will be added together.
#define NUM_THREADS 4    //Number of threads to use for vector addition.

/*
 *  Classic vector addition using openMP default data decomposition.
 *
 *  Compile using gcc like this:
 *  	gcc -o va-omp-simple VA-OMP-simple.c -fopenmp
 *
 *  Execute:
 *  	./va-omp-simple
 */
int main (int argc, char *argv[]) 
{
	// elements of arrays a and b will be added
	// and placed in array c
	int * a;
	int * b; 
	int * c;
        
        int n = ARRAY_SIZE;                 // number of array elements
	int n_per_thread;                   // elements per thread
	int total_threads = NUM_THREADS;    // number of threads to use  
	int i;       // loop index
        
        // allocate spce for the arrays
        a = (int *) malloc(sizeof(int)*n);
	b = (int *) malloc(sizeof(int)*n);
	c = (int *) malloc(sizeof(int)*n);

        // initialize arrays a and b with consecutive integer values
	// as a simple example
        for(i=0; i<n; i++) {
            a[i] = i;
        }
        for(i=0; i<n; i++) {
            b[i] = i;
        }   
        
	// Additional work to set the number of threads.
	// We hard-code to 4 for illustration purposes only.
	omp_set_num_threads(total_threads);
	
	// determine how many elements each process will work on
	n_per_thread = n/total_threads;
	
        // Compute the vector addition
	// Here is where the 4 threads are specifically 'forked' to
	// execute in parallel. This is directed by the pragma and
	// thread forking is compiled into the resulting exacutable.
	// Here we use a 'static schedule' so each thread works on  
	// a 2-element chunk of the original 8-element arrays.
	#pragma omp parallel for shared(a, b, c) private(i) schedule(static, n_per_thread)
        for(i=0; i<n; i++) {
		c[i] = a[i]+b[i];
		// Which thread am I? Show who works on what for this samll example
		printf("Thread %d works on element%d\n", omp_get_thread_num(), i);
        }
	
	// Check for correctness (only plausible for small vector size)
	// A test we would eventually leave out
	printf("i\ta[i]\t+\tb[i]\t=\tc[i]\n");
        for(i=0; i<n; i++) {
		printf("%d\t%d\t\t%d\t\t%d\n", i, a[i], b[i], c[i]);
        }
	
        // clean up memory
        free(a);  free(b); free(c);
	
	return 0;
}