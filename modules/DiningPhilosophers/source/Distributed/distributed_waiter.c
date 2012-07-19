/*
 * This is an example of a solution to the dining philosophers problem in the
 * context of processes running on separate computers that communicate using
 * message passing (OpenMPI).  Here, one process acts as the waiter, or master
 * process, and the other processes act as the philosophers.  Each process can
 * run on a different computer in a cluster, or they can all run on the same
 * computer, or some combination of the two can be used.  Since they communicate
 * with message passing, it doesn't matter.
 *
 * The waiter process keeps track of which philosophers want to eat and which
 * forks are available.  Philosophers send a message to the waiter process when
 * they want to eat and when they are done eating.  Whenever the waiter receives
 * either of these messages from any philosopher, it will try to assign as many
 * philosophers to forks as it can.  When the waiter assigns a philosopher to
 * forks, the waiter will mark the philosopher as no longer hungry, the forks as
 * in use, and the waiter will message philosopher telling him to begin eating.
 * More specifically, since the forks are representing resources that are being
 * shared, the waiter will physically send a number of bytes of data to the
 * philosophers that represent the fork "resources."
 *
 * Run this program as 
 *
 * mpirun -n <num_processes> <program_name> [seconds to run simulation]
 *
 * num_processes is 1 more than the number of philosophers, since there is 1
 * waiter. There must be a minimum of 2 philosophers for the simulation to run.
 */
#include <mpi.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define NUM_PHILOSOPHERS 5
#define WAITER_RANK NUM_PHILOSOPHERS

/* Messages sent from philosophers to waiter. */
enum message_type {
	MSG_WANT_TO_EAT,
	MSG_DONE_EATING
};

/* A tag for each fork. */
enum tag_type {
	TAG_LEFT_FORK,
	TAG_RIGHT_FORK,
};

/* The "forks" are resources that the philosopher processes are requesting.  We
 * arbitrarily make each fork 10 bytes of data.  Pretend that these are
 * resources that can only be used by 1 process at a time.  */
typedef struct {
	char data[10];
} ForkResource;

/* These numbers define the time that each philosopher may spend eating or
 * thinking. */
static const int min_eat_ms[NUM_PHILOSOPHERS]   = {300, 300, 300, 300, 300};
static const int max_eat_ms[NUM_PHILOSOPHERS]   = {600, 600, 600, 600, 600};
static const int min_think_ms[NUM_PHILOSOPHERS] = {300, 300, 300, 300, 300};
static const int max_think_ms[NUM_PHILOSOPHERS] = {600, 600, 600, 600, 600};

/* Number of seconds the simulation runs for. Override with command line
 * argument */
static int simulation_time = 5;

/* MPI rank of the current process */
static int my_rank;

/* Process IDs of all the processes (actual pids, not MPI ranks). */
static int pids[NUM_PHILOSOPHERS];

/* Statistics about the philosopher */
static double eating_time   = 0.0;
static double thinking_time = 0.0;
static double hungry_time   = 0.0;
static int times_eaten      = 0;

/* Random seed */
static unsigned seed;

/* Functions */
static void waiter_proc(void);
static void philosopher_proc(void);
static void think(void);
static void eat(ForkResource *resource1, ForkResource *resource2);
static void sleep_rand_r(int min_ms, int max_ms, unsigned *seed);
static void millisleep(int ms);
static void sigalrm_handler(int sig);
static void sigusr1_handler(int sig);
static void usage(const char *program_name);


int main(int argc, char **argv)
{
	int num_processes;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	if (num_processes != NUM_PHILOSOPHERS + 1) {
		if (my_rank == 0)
			usage(argv[0]);
		MPI_Finalize();
		exit(1);
	}
	

	/* Specifying the simulation time overrides the default value given in
	 * the declaration of simulation_time. */
	if (argc > 1) {
		simulation_time = atoi(argv[1]);
		if (simulation_time < 1) {
			if (my_rank == 0) {
				fprintf(stderr, "Error: Simulation time of %d "
					"seconds is not long enough\n", 
					simulation_time);
			}
			MPI_Finalize();
			exit(1);
		}
	}

	/* The waiter will set up SIGALRM handler for when the simulation ends.  */
	struct sigaction act;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	if (my_rank == WAITER_RANK) {
		act.sa_handler = sigalrm_handler;
		sigaddset(&act.sa_mask, SIGALRM);
		sigaction(SIGALRM, &act, NULL);

		alarm(simulation_time);
	} else {
		/* Set up SIGUSR handler for when waiter ends the simulation. */
		act.sa_handler = sigusr1_handler;
		sigaddset(&act.sa_mask, SIGUSR1);
		sigaction(SIGUSR1, &act, NULL);
	}

	/* Give the waiter the pids of all the processes so it can
	 * send a SIGUSR1 to each of them when the alarm goes off. */
	int pid = getpid();
	MPI_Gather(&pid, 1, MPI_INT, pids, 1, MPI_INT, WAITER_RANK, 
							MPI_COMM_WORLD);

	/* Set a "random" seed for the process. */
	time_t t = time(NULL);
	seed = t + my_rank;
	
	/* Get the name of this processor. */
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int processor_name_len;
	MPI_Get_processor_name(processor_name, &processor_name_len);

	if (my_rank == WAITER_RANK) {
		printf("Waiter running on processor %s: beginning %d "
			"second simulation.\n", 
			processor_name, simulation_time);
		waiter_proc();
	} else {
		printf("Philosopher %d running on processor %s.\n", 
					my_rank, processor_name);
		philosopher_proc();
	}
	return 0;
}


/* The procedure executed by the waiter process. */
static void waiter_proc()
{
	ForkResource fork_resources[NUM_PHILOSOPHERS];

	bool forks_in_use[NUM_PHILOSOPHERS];
	bool hungry[NUM_PHILOSOPHERS];

	unsigned long times_eaten[NUM_PHILOSOPHERS];

	int left, right, msg, philosopher_num;
	MPI_Status status;

	while (1) {

		/* Wait for any one of the receives to complete. */
		MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, 
						MPI_COMM_WORLD, &status);
		philosopher_num = status.MPI_SOURCE;

		/* Interpret the message that was received. */
		switch (msg) {
		case MSG_DONE_EATING:
			/* Done eating: mark philosopher as not hungry and mark
			 * forks as unused. */
			left  = philosopher_num;
			right = (philosopher_num + 1) % NUM_PHILOSOPHERS;
			printf("Philosopher %d is done eating and has released "
					"forks %d and %d.\n", philosopher_num, 
					left, right);
			hungry[philosopher_num] = false;
			forks_in_use[left]      = false;
			forks_in_use[right]     = false;
			break;
		/* want to eat: mark philosopher as hungry. */
		case MSG_WANT_TO_EAT:
			printf("Philosopher %d wants to eat!\n",  
					philosopher_num);
			hungry[philosopher_num] = true;
			break;
		}

		/* No matter what the message received was, try to assign as
		 * many philosophers to forks as possible.  A philosopher will
		 * be granted permission to eat only if both of his forks are
		 * free; that also means that neither of his two neighbors are
		 * eating.  
		 *
		 * It would be possible to use a different algorithm for this
		 * that wouldn't have to iterate through all the philosophers
		 * every time a message is received.  For smaller numbers of
		 * philosophers this should hardly take any time compared to
		 * actually sending the messages, though. */
		for (int phil_num = 0; phil_num < NUM_PHILOSOPHERS; phil_num++) 
		{
			left  = phil_num;
			right = (phil_num + 1) % NUM_PHILOSOPHERS;
			if (hungry[phil_num] && 
				!forks_in_use[left] && !forks_in_use[right])
			{
				printf("The waiter is letting philosopher %d "
					"eat with forks %d and %d\n", 
					phil_num, left, right);

				forks_in_use[left]  = true;
				forks_in_use[right] = true;
				hungry[phil_num]    = false;

				times_eaten[phil_num]++;

				/* Send the fork resources over to the
				 * philosopher */
				MPI_Send(&fork_resources[left], 
						sizeof(ForkResource), 
						MPI_CHAR, 
						phil_num, 
						TAG_LEFT_FORK, 
						MPI_COMM_WORLD);

				MPI_Send(&fork_resources[right], 
						sizeof(ForkResource), 
						MPI_CHAR, 
						phil_num, 
						TAG_RIGHT_FORK, 
						MPI_COMM_WORLD);
			}
		}
	}
}

/* The procedure executed by each of the philosopher processes. */
static void philosopher_proc()
{
	ForkResource my_fork_resources[2];
	int msg;
	double t1, t2;

	t1 = MPI_Wtime();
	while (1) {
		/* Think */
		t2 = MPI_Wtime();
		eating_time += (t2 - t1);
		t1 = t2;

		think();

		t2 = MPI_Wtime();
		thinking_time += (t2 - t1);
		t1 = t2;

		/* Tell waiter we want to eat. */

		/* Note:  This MPI_Send(), after the first time through this
		 * loop, happens after the other MPI_Send() a number of lines
		 * below, which tells the waiter this philosopher is done
		 * eating, and there are no intervening calls to MPI_Recv().  It
		 * would be erroneous if the messages the philosopher is sending
		 * were to arrive at the waiter in the wrong order.  However,
		 * the MPI standard guarantees the correct receive order of
		 * messages sent from the same process, so the code should work
		 * correctly.
		 */
		msg = MSG_WANT_TO_EAT;
		MPI_Send(&msg, 1, MPI_INT, WAITER_RANK, 0, MPI_COMM_WORLD);

		/* Wait for waiter to give us our forks. */
		MPI_Recv(&my_fork_resources[0], sizeof(ForkResource), MPI_CHAR, 
				WAITER_RANK, TAG_LEFT_FORK, MPI_COMM_WORLD, 
				MPI_STATUS_IGNORE);

		MPI_Recv(&my_fork_resources[1], sizeof(ForkResource), MPI_CHAR, 
				WAITER_RANK, TAG_RIGHT_FORK, MPI_COMM_WORLD, 
				MPI_STATUS_IGNORE);
		
		t2 = MPI_Wtime();
		hungry_time += (t2 - t1);
		t1 = t2;

		/* Eat */
		eat(&my_fork_resources[0], &my_fork_resources[1]);
		
		times_eaten++;

		/* Tell waiter we are done. */
		msg = MSG_DONE_EATING;
		MPI_Send(&msg, 1, MPI_INT, WAITER_RANK, 0, MPI_COMM_WORLD);
	}
}



/* The think() and eat() functions are placeholders for things that a real
 * application could be doing while not holding resources and while holding
 * resources, respectively.
 */
static void think()
{
	sleep_rand_r(min_think_ms[my_rank], max_think_ms[my_rank], &seed);
}

static void eat(ForkResource* resource1, ForkResource* resource2)
{
	sleep_rand_r(min_eat_ms[my_rank], max_eat_ms[my_rank], &seed);
}

/* Suspends the execution of the calling thread for the specified number of
 * milliseconds. */
static void millisleep(int ms)
{
	usleep(ms * 1000);
}

/* Suspends the execution of the calling thread for a random time between
 * min_ms mseconds and max_ms mseconds.  */
static void sleep_rand_r(int min_ms, int max_ms, unsigned *seed)
{
	int range = max_ms - min_ms + 1;
	int ms = rand_r(seed) % range + min_ms;
	millisleep(ms);
}

/* This function was registered with SIGALRM and will be invoked by the waiter
 * when the simulation terminates. */
static void sigalrm_handler(int sig)
{
	printf( "\nSimulation ending after %d seconds as planned.\n\n" , 
						simulation_time);

	/* Make all the other philosophers call sigusr1_handler() */
	for (int i = 0; i < NUM_PHILOSOPHERS; i++)
		kill((pid_t) pids[i], SIGUSR1);

	sigusr1_handler(0);
}

static void sigusr1_handler(int sig)
{
	/* The waiter will retrieve results from all the philosophers and print
	 * them out. */

	double data[4] = {eating_time, thinking_time, 
			hungry_time, (double)times_eaten};

	double results[NUM_PHILOSOPHERS * 4 * sizeof(double) + 1];

	MPI_Gather(data, 4, MPI_DOUBLE, results, 4, MPI_DOUBLE, 
					WAITER_RANK, MPI_COMM_WORLD);

	if (my_rank == WAITER_RANK) {
		for (int i = 0; i < NUM_PHILOSOPHERS; i++) {

			printf("Results for Philosopher %d:\n" , i);

			printf("\t%.2f sec. spent eating\n", 
					results[i * 4]);

			printf("\t%.2f sec. spent thinking\n", 
					results[i * 4 + 1]);

			printf("\t%.2f sec. spent hungry\n", 
					results[i * 4 + 2]);

			printf("\t%d times eaten\n\n", 
					(int)results[i * 4 + 3]);
			fflush(stdout);
		}
	}
	MPI_Finalize();
	exit(0);
}

static void usage(const char* program_name)
{
	const char* txt = 
	"\n"
	"ERROR: This program must be run using one more process than the number\n"
	"of philosophers.  NUM_PHILOSOPHERS is currently defined as %d, so the\n"
	"program should be run with %d processes.\n"
	"\n"
	"Run with `mpirun -n NUM_PROCESSES %s [SECONDS]'\n"
	"\n"
	;
	fprintf(stderr, txt, NUM_PHILOSOPHERS, NUM_PHILOSOPHERS + 1, program_name);
}
