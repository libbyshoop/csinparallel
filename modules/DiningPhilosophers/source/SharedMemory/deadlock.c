/* 
 * This simulation illustrates the dining philosophers problem. 
 *
 * Each fork is associated with a mutex. The philosophers pick up their left
 * fork, then their right fork; when they finish eating, they put down their
 * left shopstick, then their right fork.
 *
 * The simulation will reach deadlock after a while, because it some point all
 * the philosophers will pick up their left forks at the same time and all be
 * waiting for their right fork, which will never be released. However, it may
 * take a while before this problem is seen because it requires that the threads
 * just happen to get preempted after locking the left fork mutex but before
 * locking the right fork mutex.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

#define NUM_PHILOSOPHERS 5

/* These arrays define the minimum and maximum amounts of time that each
 * philosopher spends eating or thinking. */
static const int min_eat_ms[NUM_PHILOSOPHERS]   = {300, 300, 300, 300, 300};
static const int max_eat_ms[NUM_PHILOSOPHERS]   = {600, 600, 600, 600, 600};
static const int min_think_ms[NUM_PHILOSOPHERS] = {300, 300, 300, 300, 300};
static const int max_think_ms[NUM_PHILOSOPHERS] = {600, 600, 600, 600, 600};

/* Functions. */
static void philosopher_cycle(int thread_num, pthread_mutex_t *left_fork, 
		pthread_mutex_t *right_fork, unsigned *seed);
static void millisleep(int ms);
static void sleep_rand_r(int min_ms, int max_ms, unsigned *seed);


int main(int argc, char **argv)
{
	/* The forks are represented by an array of mutexes. */
	pthread_mutex_t forks[NUM_PHILOSOPHERS];

	/* Get the current time, in seconds since the Epoch. */
	time_t t = time(NULL);

	/* Initialize the forks' mutexes. */
	for (int i = 0; i < NUM_PHILOSOPHERS; i++)
		pthread_mutex_init(&forks[i], NULL);

	/* Each philosopher is a thread */
	#pragma omp parallel num_threads(NUM_PHILOSOPHERS)
	{
		/* Get the number of this thread and figure out which fork is on
		 * the right and which is on the left. */
		int thread_num = omp_get_thread_num();
		pthread_mutex_t *left_fork = &forks[thread_num];
		pthread_mutex_t *right_fork = &forks[(thread_num + 1) % NUM_PHILOSOPHERS];

		/* Make this thread have a random seed different from the other
		 * threads. */
		unsigned seed = t + thread_num;
		while (1) {
			philosopher_cycle(thread_num, left_fork, right_fork, &seed);
		}
	}
	return 0;
}

/* Called each time a philosopher wants to eat. */
static void philosopher_cycle(int thread_num, pthread_mutex_t *left_fork, 
		pthread_mutex_t *right_fork, unsigned *seed)
{
	printf("Philosopher %d wants to eat!\n", thread_num);

	pthread_mutex_lock(left_fork);
	printf("Philosopher %d picked up his left fork.\n", thread_num);
	pthread_mutex_lock(right_fork);
	printf("Philosopher %d picked up his right fork and "
			"started eating.\n", thread_num);

	sleep_rand_r(min_eat_ms[thread_num], max_eat_ms[thread_num], seed);

	pthread_mutex_unlock(left_fork);
	pthread_mutex_unlock(right_fork);
	printf("Philosopher %d is done eating and has released his "
			"forks.\n", thread_num);

	sleep_rand_r(min_think_ms[thread_num], max_think_ms[thread_num], seed);
}

/* Suspends the execution of the calling thread for the specified number of
 * milliseconds. */
static void millisleep(int ms)
{
	usleep(ms * 1000);
}

/* Suspends the execution of the calling thread for a random time between
 * min_ms milliseconds and max_ms milliseconds.  */
static void sleep_rand_r(int min_ms, int max_ms, unsigned *seed)
{
	int range = max_ms - min_ms + 1;
	int ms = rand_r(seed) % range + min_ms;
	millisleep(ms);
}
