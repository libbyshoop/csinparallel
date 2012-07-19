/*
 * This is an implementation of the solution to the Dining Philosophers Problem
 * described in the paper "The Drinking Philosophers Problem" by K.M. Chandy and
 * J. Misra (1984). The problem is described in terms of a graph where each
 * process is a node on a graph and each edge represents a resource shared
 * between two processes. 
 *
 * Any any instant in time, for each 2 processes connected by an edge, one of
 * the processes is required to have precedence over the other.  This is
 * represented in the graph making the edge directed, pointed towards the
 * process with lower precedence.
 *
 * Deadlock can only occur if there is a cycle in the precedence graph, because
 * then it would be possible for a set of processes to reach a state where they
 * are each waiting for the next one in the set. The Chandry-Misra solution
 * requires that the graph starts out acyclic, and then the solution guarantees
 * that no cycle can appear in the graph at any later point in time.
 *
 * The Chandry-Misra solution also guarantees that every philosopher will have a
 * chance to eat. This is done by having the forks be "clean" or "dirty", and
 * using this in such a way that disadvantages philosophers that have just
 * eaten.
 *
 * Unlike some solutions to the Dining Philosophers' problem, the Chandry-Misra
 * solution is completely distributed and requires no central process to keep
 * track of the overall state and/or dictate who gets to eat.
 *
 * Chandry and Misra also generalize the Dining Philosophers problem to the
 * Drinking Philosophers problem, which allows the philosophers to require any
 * nonempty set of the resources available to them, and possibly different sets
 * for each drinking (eating) session. This implementation does not implement a
 * solution to this; however, this implementation does support arbitrarily
 * changing the number of philosophers and the resource sharing relationships
 * among them, and arbitrarily changing how long each individual philosopher
 * spends thinking and eating.
 *
 * See the original paper for more information about the solution.
 *
 * Run as mpirun -n NUM_PHILOSOPHERSOSOPHERS chandy_misra [SECONDS TO RUN SIMULATION]
 */
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define NUM_PHILOSOPHERS 5

/* The "forks" are resources that the philosopher processes are requesting.  We
 * arbitrarily make each fork 10 bytes of data.  Pretend that these are
 * resources that can only be used by 1 process at a time.  */
typedef struct {
	char data[10];
} ForkResource;


/* Messages to be sent, although the actual message content does not necessarily
 * have to be actually read. */
enum message_type {
	MSG_REQ,
	MSG_FORK,
};

/* Possible statuses of the philosopher */
enum philosopher_status {
	EATING,
	HUNGRY,
	THINKING,
};

/* 
 * This array of 2-integer arrays specifies the resources that the philosophers
 * are contending for.  Each resource is shared between exactly 2 philosophers,
 * but a philosopher may be sharing any number of resources with others. Each
 * element of the resources array represents one resource, and the 2 numbers of
 * each element give the 2 philosophers that are contending for that resource.
 * The philosophers are numbered starting at 0.
 *
 * The first philosopher each each pair is the one that starts off with the
 * resource. This solution requires that the resources are initially distributed
 * acyclically. 
 */
static const int resources[][2] = { {0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 4} };

/* The total number of resources distributed among the philosophers;
 * alternatively, the total number of edges in the precedence graph associated
 * with the processes. */
#define NUM_RESOURCES (sizeof(resources) / sizeof(resources[0]))

/* These numbers define the time that each philosopher may spend eating or
 * thinking. */
static const int min_eat_ms[NUM_PHILOSOPHERS]   = {180, 180, 180, 180, 180};
static const int max_eat_ms[NUM_PHILOSOPHERS]   = {360, 360, 360, 360, 360};
static const int min_think_ms[NUM_PHILOSOPHERS] = {180, 180, 180, 180, 180};
static const int max_think_ms[NUM_PHILOSOPHERS] = {360, 360, 360, 360, 360};

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
static void philosopher_proc(void);
static void* philosopher_main_thread_proc(void* ignore);
static void* philosopher_helper_thread_proc(void* ignore);
static void philosopher_main_thread_cycle(void);
static void philosopher_helper_thread_cycle(void);
static void think(void);
static void eat(ForkResource resources[], int num_resources);
static void sleep_rand_r(int min_ms, int max_ms, unsigned *seed);
static void millisleep(int ms);
static void sigalrm_handler(int sig);
static void sigusr1_handler(int sig);
static void usage(const char *program_name);
static void print_no_mpi_threads_msg(void);

int main(int argc, char **argv)
{
	int provided;
	int num_processes;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (provided != MPI_THREAD_MULTIPLE) {
		if (my_rank == 0)
			print_no_mpi_threads_msg();
		MPI_Finalize();
		exit(1);
	}

	if (num_processes != NUM_PHILOSOPHERS) {
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

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int processor_name_len;
	MPI_Get_processor_name(processor_name, &processor_name_len);
	printf("Philosopher %d running on processor %s: beginning %d second "
		"simulation\n", my_rank, processor_name, simulation_time);

	/* Process 0 will receive an alarm signal when the time specified for
	 * the simulation terminates.  It will then signal the other processes
	 * with SIGUSR1 and collect statistics from them.  However, 0 is not
	 * special with regards to the actual solution of the dining
	 * philosophers problem. */
	struct sigaction act;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	if (my_rank == 0) {
		act.sa_handler = sigalrm_handler;
		sigaddset(&act.sa_mask, SIGALRM);
		sigaction(SIGALRM, &act, NULL);

		alarm(simulation_time);
	} else {
		/* Set up SIGUSR handler for when the simulation ends.*/
		act.sa_handler = sigusr1_handler;
		sigaddset(&act.sa_mask, SIGUSR1);
		sigaction(SIGUSR1, &act, NULL);
	}

	/* Give process 0 the pids of all the processes so it can send a signal
	 * to each of them when the alarm goes off.  Note that process 0 is only
	 * made special for information reporting reasons; the actual solution
	 * of the dining philosophers problem itself does not distinguish any of
	 * the processes as special. */
	int pid = getpid();
	MPI_Gather(&pid, 1, MPI_INT, pids, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* Set a "random" seed for the process. */
	time_t t = time(NULL);
	seed = t + my_rank;

	philosopher_proc();
	return 0;
}


/* The following variables need to be shared between the two threads. 
 * They alternatively could be stuffed into a structure, and a pointer to that
 * structure passed to the threads. */

/* The philosopher's status: THINKING, EATING, or HUNGRY */
static enum philosopher_status status;

/* Number of other philosophers this philosopher is sharing resources with. */
static int num_neighbors;

/* MPI process ranks of this processes' neighbors. */
static int* neighbors;

/* Whether this philosopher possesses each fork that is shared with its
 * neighbors. */
static bool* has_fork;

/* Whether this philosopher possesses each fork that is shared with its
 * neighbors, and it is dirty */
static bool* dirty;

/* Whether this philosopher possesses the request token associated with
 * each fork that is shared with its neighbors. */
static bool *req;

/* Per-neighbor information for sending and receiving MPI messages. */
static int *in_reqs_data;
static MPI_Request *in_reqs;
static MPI_Request *out_reqs;
static ForkResource *forks_data;
static MPI_Request *in_forks;
static MPI_Request *out_forks;

/* Number of forks this philosopher is holding. */
static int num_forks;

/* Mutex that coordinates the two threads. */
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

#define ALLOC_PER_NEIGHBOR(ptr) \
	ptr = malloc(num_neighbors * sizeof(*ptr))

static pthread_t helper_thread;

void philosopher_proc()
{
	/* Find the number of neighbors of this philosopher. */
	for (int i = 0; i < NUM_RESOURCES; i++) {
		if (resources[i][0] == my_rank || resources[i][1] == my_rank) {
			num_neighbors++;
		}
	}

	ALLOC_PER_NEIGHBOR(neighbors);
	ALLOC_PER_NEIGHBOR(has_fork);
	ALLOC_PER_NEIGHBOR(dirty);
	ALLOC_PER_NEIGHBOR(req);
	ALLOC_PER_NEIGHBOR(in_reqs);
	ALLOC_PER_NEIGHBOR(out_reqs);
	ALLOC_PER_NEIGHBOR(in_reqs_data);
	ALLOC_PER_NEIGHBOR(forks_data);
	ALLOC_PER_NEIGHBOR(in_forks);
	ALLOC_PER_NEIGHBOR(out_forks);

	/* Initialize neighbors, fork, dirty, and req; also count up the number
	 * of forks this philosopher has. */
	int j = 0;
	num_forks = 0;
	for (int i = 0; i < NUM_RESOURCES; i++) {
		/* The philosopher listed first in the pair starts with the
		 * fork. The other one starts with the request token. All forks
		 * start out dirty. */
		if (resources[i][0] == my_rank) {
			neighbors[j] = resources[i][1];
			has_fork[j]  = true;
			dirty[j]     = true;
			req[j]       = false;
			num_forks++;
			j++;
		} else if (resources[i][1] == my_rank) {
			neighbors[j] = resources[i][0];
			has_fork[j]  = false;
			dirty[j]     = false;
			req[j]       = true;
			j++;
		}
	}

	/* Post all the receives: A receive for a request from each neighbor,
	 * and a receive for a fork from each neighbor.  All the tags need to be
	 * unique, so fork messages are tagged with the rank of the originating
	 * process, and request messages are tagged with the rank of the
	 * originating process plus NUM_PHILOSOPHERS. */
	for (int i = 0; i < num_neighbors; i++) {
		MPI_Irecv(&forks_data[i], sizeof(ForkResource), MPI_CHAR, 
				neighbors[i], neighbors[i], 
				MPI_COMM_WORLD, &in_forks[i]);

		MPI_Irecv(&in_reqs_data[i], 1, MPI_INT, neighbors[i], 
				neighbors[i] + NUM_PHILOSOPHERS, 
				MPI_COMM_WORLD, &in_reqs[i]);
	}


	/* Split the process into 2 threads.  This is done so that if a
	 * philosopher is thinking and has a dirty fork that another philosopher
	 * wants, then the fork can be given to that philosopher without waiting
	 * for the current holder of the fork to finish thinking.  This is
	 * essential since the specification given for the dining philosophers
	 * problem indicates that a philosopher may think for an arbitrary
	 * amount of time, even forever.
	 */
	pthread_create(&helper_thread, NULL, philosopher_helper_thread_proc, NULL);
	philosopher_main_thread_proc(NULL);
}

static void* philosopher_main_thread_proc(void* ignore)
{
	while (1) {
		philosopher_main_thread_cycle();
	}
	return NULL;
}

static void* philosopher_helper_thread_proc(void* ignore)
{
	while (1) {
		philosopher_helper_thread_cycle();
	}
	return NULL;
}


static void philosopher_main_thread_cycle()
{
	/* Messages to be sent */
	static int msg_req  = MSG_REQ;
	static bool first_time = true;

	static double t1, t2;

	if (first_time) {
		first_time = false;
		t1 = MPI_Wtime();
	}

	/**************************
	 *   PREPARING TO THINK   *
	 **************************/
	printf("Philosopher %d is done eating and is preparing to think "
			"(Eaten %d times so far).\n", my_rank, times_eaten);
	t2 = MPI_Wtime();
	eating_time += (t2 - t1);
	t1 = t2;

	/* We must fulfill any outstanding requests that have been made while we
	 * were eating. */
	pthread_mutex_lock(&mutex);
	status = THINKING;
	for (int i = 0; i < num_neighbors; i++) {
		/* If req[i] is true and we have fork[i] and it is dirty, then
		 * it has been requested and we are required to send it. */
		if (req[i] && dirty[i]) {
			printf("Before beginning to think, Philosopher %d "
				"gave a fork to Philosopher %d.\n", 
				my_rank, neighbors[i]);
			has_fork[i] = false;
			dirty[i] = false;
			num_forks--;
			MPI_Isend(&forks_data[i], 1, MPI_INT, neighbors[i], 
				my_rank, MPI_COMM_WORLD, &out_forks[i]);
		}
	}
	pthread_mutex_unlock(&mutex);

	/*********************
	 *       THINKING    *
	 *********************/
	printf("Philosopher %d is thinking.\n", my_rank);
	think();
	t2 = MPI_Wtime();
	thinking_time += (t2 - t1);
	t1 = t2;

	/**********************
	 *       HUNGRY       *
	 **********************/

	/* Hungry now. If we have the forks we need, we can begin eating.
	 * Otherwise, we must send requests for forks we don't have. */
	pthread_mutex_lock(&mutex);
	status = HUNGRY;
	if (num_forks != num_neighbors) {
		printf("Philosopher %d is hungry and only has %d forks!\n", 
							my_rank, num_forks);
		for (int i = 0; i < num_neighbors; i++) {
			if (req[i] && !has_fork[i]) {
				MPI_Isend(&msg_req, 1, MPI_INT, neighbors[i], 
						my_rank + NUM_PHILOSOPHERS,
						MPI_COMM_WORLD, &out_reqs[i]);

				printf("Philosopher %d requested a fork from "
					"Philosopher %d.\n", my_rank, neighbors[i]);
				req[i] = false;
			}
		}
		do {
			/* Wait for all the forks to arrive. */
			pthread_mutex_unlock(&mutex);
			printf("Philosopher %d is waiting with %d forks.\n", 
							my_rank, num_forks);

			int idx;
			MPI_Waitany(num_neighbors, in_forks, &idx, 
							MPI_STATUS_IGNORE);

			int neighbor_idx;
			for (int i = 0; i < num_neighbors; i++) {
				if (neighbors[i] == idx) {
					neighbor_idx = i;
					break;
				}
			}
			MPI_Irecv(&forks_data[neighbor_idx], sizeof(ForkResource), 
					MPI_CHAR, neighbors[idx], neighbors[idx],
					MPI_COMM_WORLD, &in_forks[idx]);

			pthread_mutex_lock(&mutex);

			printf("Philosopher %d received a fork from "
				"Philosopher %d.\n", my_rank, neighbors[idx]);

			has_fork[idx] = true;
			num_forks++;
		} while (num_forks < num_neighbors);
		printf("Philosopher %d now has %d forks and started to eat.\n", 
						my_rank, num_forks);
	} else {
		printf("Philosopher %d is hungry and already has %d forks, "
				"so he started eating.\n", my_rank, num_forks);
	}
	/**********************
	 *       EATING       *
	 **********************/

	/* All forks become dirty when the philosopher begins eating. */
	status = EATING;
	for (int i = 0; i < num_neighbors; i++)
		dirty[i] = true;

	pthread_mutex_unlock(&mutex);
	t2 = MPI_Wtime();
	hungry_time += (t2 - t1);
	t1 = t2;

	eat(forks_data, num_forks);
	times_eaten++;
}



static void philosopher_helper_thread_cycle()
{
	bool send_fork;
	bool request_fork;
	int msg_req  = MSG_REQ;
	int idx;

	/* Wait for any request to come in. */
	MPI_Waitany(num_neighbors, in_reqs, &idx, MPI_STATUS_IGNORE);
	/* Re-post the request. */
	MPI_Irecv(&in_reqs_data[idx], 1, MPI_INT, neighbors[idx], 
			neighbors[idx] + NUM_PHILOSOPHERS, 
			MPI_COMM_WORLD, &in_reqs[idx]);

	send_fork    = false;
	request_fork = false;

	pthread_mutex_lock(&mutex);
	req[idx] = true;
	switch (status) {
	case THINKING:
		/* If the philosopher is thinking at the moment and we receive a
		* request for a dirty fork, it must be sent. The actual MPI_Isend
		* can be moved out of the mutex-locked region, but the fork must be
		* marked as gone before unlocking the mutex. */
		if (dirty[idx]) {
			printf(
			   "A dirty fork was taken from Philosopher %d, who is "
			   "currently thinking, and given to Philosopher %d "
			   "cleaned.\n" , my_rank, neighbors[idx]);
			fflush(stdout);
			has_fork[idx]  = false;
			dirty[idx] = false;
			send_fork  = true;
			num_forks--;
		}
		break;
	case HUNGRY:
		/* If the philosopher is hungry, he is still required to give up
		 * a fork if it is requested and it is dirty. However, he also
		 * must request it back right away. */
		if (dirty[idx]) {
			printf("A dirty fork was taken from Philosopher %d, who "
				"is currently hungry, and given to Philosopher "
				"%d cleaned; then it was requested back.\n",
					my_rank, neighbors[idx]);
			fflush(stdout);
			has_fork[idx]    = false;
			dirty[idx]   = false;
			req[idx]     = false;
			send_fork    = true;
			request_fork = true;
			num_forks--;
		}
		break;
	case EATING:
		/* If the philosopher is eating, then req[i] is marked as true
		 * (already done before the switch statement), but the fork
		 * cannot be sent. */
		break;
	}
	pthread_mutex_unlock(&mutex);

	if (send_fork) {
		MPI_Isend(&forks_data[idx], 1, MPI_INT, neighbors[idx], my_rank, 
				MPI_COMM_WORLD, &out_forks[idx]);
	}

	if (request_fork) {
		MPI_Isend(&msg_req, 1, MPI_INT, neighbors[idx], 
				my_rank + NUM_PHILOSOPHERS, 
				MPI_COMM_WORLD, &out_reqs[idx]);
	}
}

/* The think() and eat() functions are placeholders for things that a real application
 * could be doing while not holding resources and while holding resources, respectively.
 */
static void think()
{
	sleep_rand_r(min_think_ms[my_rank], max_think_ms[my_rank], &seed);
}

static void eat(ForkResource resources[], int num_resources)
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

/* This function was registered with SIGALRM and will be invoked by process 0
 * when the simulation terminates. */
static void sigalrm_handler(int sig)
{
	printf( "\nSimulation ending after %d seconds as planned.\n\n" , 
						simulation_time);

	/* Make all the other philosophers call sigusr1_handler() */
	for (int i = 1; i < NUM_PHILOSOPHERS; i++)
		kill((pid_t) pids[i], SIGUSR1);

	sigusr1_handler(0);
}

static void sigusr1_handler(int sig)
{
	/* Process 0 will retrieve results from all the philosophers and print
	 * them out. */

	double data[4] = {eating_time, thinking_time, hungry_time, (double)times_eaten};

	double results[NUM_PHILOSOPHERS * 4 * sizeof(double) + 1];

	/* Probably not actually safe to call MPI_Gather() from an interrupt
	 * handler. */
	MPI_Gather(data, 4, MPI_DOUBLE, results, 4, MPI_DOUBLE, 
					0, MPI_COMM_WORLD);
	if (my_rank == 0) {
		for (int i = 0; i < NUM_PHILOSOPHERS; i++) {
			printf("Results for Philosopher %d:\n" , i);
			printf("\t%.2f sec. spent eating\n", results[i * 4]);
			printf("\t%.2f sec. spent thinking\n", results[i * 4 + 1]);
			printf("\t%.2f sec. spent hungry\n", results[i * 4 + 2]);
			printf("\t%d times eaten\n\n", (int)results[i * 4 + 3]);
			fflush(stdout);
		}
	}

	/* It is extremely difficult to cleanly terminate this program for a
	 * couple reasons:
	 *
	 * - MPI_Finalize() cannot be called while there are pending
	 *   communications, and there can be such communications going on in
	 *   both the helper thread and the main thread.  (Note this is a signal
	 *   handling context right now.)
	 * - We would have to wait an unknown amount of time for any pending
	 *   communications to complete and it would require adding extra checks
	 *   to both thread procedures.  But really we want to exit right now,
	 *   as the results have been (or are just about to be) printed.
	 *
	 * So just sleep a while to give the results enough time to get printed
	 * on process 0, then kill the threads. */
	millisleep(300);
	pthread_kill(helper_thread, SIGKILL);
	pthread_kill(pthread_self(), SIGKILL);
}

static void usage(const char* program_name)
{
	const char* txt = 
	"\n"
	"ERROR: This program must be run using a number of processes equal to\n"
	"the number of philosophers.  NUM_PHILOSOPHERS is currently defined as %d,\n"
	"so the program should be run with %d processes.\n"
	"\n"
	"Run with `mpirun -n NUM_PROCESSES %s [SECONDS]'\n"
	"\n"
	;
	fprintf(stderr, txt, NUM_PHILOSOPHERS, NUM_PHILOSOPHERS, program_name);
}

static void print_no_mpi_threads_msg(void)
{
	const char* txt = 
	"\n"
	"ERROR: MPI_Init_thread() failed to request support for MPI_THREAD_MULTIPLE.\n"
	"This OpenMPI compilation does not support multiple threads executing in the\n"
	"MPI library concurrently.  OpenMPI must be recompiled with the\n"
	"--enable-mpi-threads configure flag to run this program.\n"
	"\n"
	;
	fputs(txt, stderr);
}
