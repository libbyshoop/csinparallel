#include <stdlib.h>   //malloc and free
#include <stdio.h>    //printf

#define ARRAY_SIZE 8     //Size of arrays whose elements will be added together.

/*
 *  Classic vector addition.
 */
int main (int argc, char *argv[]) 
{
	// elements of arrays a and b will be added
	// and placed in array c
	int * a;
	int * b; 
	int * c;
        
        int n = ARRAY_SIZE;   // number of array elements
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
        
        // Compute the vector addition
        for(i=0; i<n; i++) {
		c[i] = a[i]+b[i];
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