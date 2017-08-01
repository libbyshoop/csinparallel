/*
 * Peter S. Pacheco, An Introduction to Parallel Programming,
 * Morgan Kaufmann Publishers, 2011
 * IPP:   Section 3.4.2 (pp. 104 and ff.)
 * Sequential version by Hannah Sonsalla, Macalester College, 2017
 *
 * seq_trap.c
 *
 * ... sequential verison of the trapezoidal rule.  
 *
 * Input:    Number of trapezoids
 * Output:   Estimate of the integral from a to b of f(x)
 *           using the trapezoidal rule and n trapezoids.
 *
 * Usage:    ./mpi_trap.c
 *
 * Note:  f(x) is all hardwired to x*x.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const double a = 0;
const double b = 2000000000;

/* Function declarations */
void Get_input(int argc, char* argv[], double* n_p);
double Trap(double left_endpt, double right_endpt, int trap_count,
   double base_len);
double f(double x);

int main(int argc, char** argv) {
   double n, h, total_int;
   double start, finish, elapsed;

   Get_input(argc, argv, &n); /*Read user input */

   h = (b-a)/n;          /* length of each trapezoid */

   start = clock();
   /* Calculate integral using endpoints*/
   total_int = Trap(a, b, n, h);
   finish = clock();
   elapsed = (double)(finish-start)/CLOCKS_PER_SEC;
 
   printf("With n = %.0f trapezoids, our estimate\n", n);
   printf("of the integral from %.0f to %.0f = %.0f\n", a, b, total_int);
   printf("Elapsed time = %f milliseconds \n", elapsed * 1000);

   return 0;
} /*  main  */

/*------------------------------------------------------------------
 * Function:     Get_input
 * Purpose:      Get the user input: the number of trapezoids
 * Input args:   my_rank:  process rank in MPI_COMM_WORLD
 *               comm_sz:  number of processes in MPI_COMM_WORLD
 * Output args:  n_p:  pointer to number of trapezoids
 */
void Get_input(int argc, char* argv[], double* n_p){
	if (argc!= 2){
		fprintf(stderr, "usage: %s <number of trapezoids> \n", argv[0]);
        fflush(stderr);
        *n_p = -1;
    } else {
		*n_p = atoi(argv[1]);
	}

	// negative n ends the program
    if (*n_p <= 0) {
		exit(-1);
    }
}  /* Get_input */

/*------------------------------------------------------------------
 * Function:     Trap
 * Purpose:      Serial function for estimating a definite integral
 *               using the trapezoidal rule
 * Input args:   left_endpt
 *               right_endpt
 *               trap_count
 *               base_len
 * Return val:   Trapezoidal rule estimate of integral from
 *               left_endpt to right_endpt using trap_count
 *               trapezoids
 */
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len) {
   double estimate, x;
   int i;

   estimate = (f(left_endpt) + f(right_endpt))/2.0;
   for (i = 1; i <= trap_count-1; i++) {
      x = left_endpt + i*base_len;
      estimate += f(x);
   }
   estimate = estimate*base_len;

   return estimate;
} /*  Trap  */


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Compute value of function to be integrated
 * Input args:  x
 */
double f(double x) {
   return x*x;
} /* f */
