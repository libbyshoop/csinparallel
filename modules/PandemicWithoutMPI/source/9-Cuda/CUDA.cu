/* Parallelization: Infectious Disease
 * By Yu Zhao, Macalester College.
 * July 2013 */

#ifndef PANDEMIC_CUDA_CU
#define PANDEMIC_CUDA_CU

#include "Defaults.h" 

/* CUDA shared memory allocation */
extern __shared__ int array[];

/*
    move()
        Spawns threads to move everyone randomly
*/
__global__ void cuda_move(char *states_dev, int *x_locations_dev, int *y_locations_dev, 
    char DEAD, int environment_width, int environment_height, float *rand_nums, int SIZE)
{
    // set up thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // If the person is not dead, then
    if(states_dev[id] != DEAD){

        // The thread randomly picks whether the person moves left
        // or right or does not move in the x dimension
        int x_move_direction = (int)(rand_nums[id]*3) - 1;

        // The thread randomly picks whether the person moves up 
        // or down or does not move in the y dimension
        int y_move_direction = (int)(rand_nums[id+SIZE]*3) - 1;


        // If the person will remain in the bounds of the
        // environment after moving, then
        if( (x_locations_dev[id] + x_move_direction >= 0) && 
            (x_locations_dev[id] + x_move_direction < environment_width) && 
            (y_locations_dev[id] + y_move_direction >= 0) &&
            (y_locations_dev[id] + y_move_direction < environment_height) )
        {
            // The thread moves the person
            x_locations_dev[id] = x_locations_dev[id] + x_move_direction;
            y_locations_dev[id] = y_locations_dev[id] + y_move_direction;
        }
    }
}

/*
    cuda_susceptible()
        Spawns threads to handle those that are ssusceptible by 
        deciding whether or not they should be marked infected.
*/
__global__ void cuda_susceptible(char *states_dev, int *x_locations_dev, 
    int *y_locations_dev, int *infected_x_locations_dev, 
    int *infected_y_locations_dev, int *num_infected_dev, 
    int *num_susceptible_dev, int *num_infection_attempts_dev,
    int *num_infections_dev, float *rand_nums, int global_num_infected, 
    int infection_radius, int contagiousness_factor, char SUSCEPTIBLE, char INFECTED)
{
    // set up thread id, block id and block dimension
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int blockId = threadIdx.x;
    int numThread = blockDim.x;

    // counters
    int i, num_infected_nearby;

    // set up shared memory
    int *num_infected = (int*)array; 
    int *num_susceptible = (int*)&num_infected[numThread];
    #ifdef SHOW_RESULTS
    int *num_infection_attempts = (int*)&num_susceptible[numThread];
    int *num_infections = (int*)&num_infection_attempts[numThread];
    #endif

    // reset the shared memory
    num_infected[blockId] = 0;
    num_susceptible[blockId] = 0;
    #ifdef SHOW_RESULTS
    num_infection_attempts[blockId] = 0;
    num_infections[blockId] = 0;
    #endif

    // If the person is susceptible, then
    if(states_dev[id] == SUSCEPTIBLE)
    {
        // For each of the infected people (received earlier from 
        // all processes) or until the number of infected people 
        // nearby is 1, the thread does the following
        num_infected_nearby = 0;
        for(i=0; i<=global_num_infected-1 && num_infected_nearby<1; i++)
        {
            // If this person is within the infection radius, then
            if( (x_locations_dev[id] > infected_x_locations_dev[i] - infection_radius) && 
                (x_locations_dev[id] < infected_x_locations_dev[i] + infection_radius) && 
                (y_locations_dev[id] > infected_y_locations_dev[i] - infection_radius) &&
                (y_locations_dev[id] < infected_y_locations_dev[i] + infection_radius) )
            {
                // The thread increments the number of infected people nearby
                num_infected_nearby++;
            }
        }

        if(num_infected_nearby >= 1){
            #ifdef SHOW_RESULTS
            num_infection_attempts[blockId]++;
            #endif
        }
        
        // generate a random number between 0 and 100
        int rand_num = (int)(rand_nums[id]*100);

        // If there is at least one infected person nearby, and 
        // a random number less than 100 is less than or equal 
        // to the contagiousness factor, then
        if(num_infected_nearby >= 1 && rand_num <= contagiousness_factor)
        {
            // The thread changes person1’s state to infected
            states_dev[id] = INFECTED;
            // The thread updates the counters
            num_infected[blockId]++;
            num_susceptible[blockId]--;
            #ifdef SHOW_RESULTS
            num_infections[blockId]++;
            #endif
        }
    }

    __syncthreads();
    // if we have numThread to the power of 2, we can use binary
    // tree reduction to increase performance
    if(((numThread!=0) && !(numThread & (numThread-1)))){
        i = numThread/2;
        while (i != 0) {
            if (blockId < i){
                num_infected[blockId] += num_infected[blockId + i];
                num_susceptible[blockId] += num_susceptible[blockId + i];
                #ifdef SHOW_RESULTS
                num_infection_attempts[blockId] += num_infection_attempts[blockId + i];
                num_infections[blockId] += num_infections[blockId + i];
                #endif
            }
            __syncthreads();
            i /= 2; 
        }
    }
    // Else, we can only add-up results in shared memory using
    // the first thread in each block
    else{
        if(blockId == 0) {
            for(i=1; i<numThread; i++){
                num_infected[0] += num_infected[i];
                num_susceptible[0] += num_susceptible[i];
                #ifdef SHOW_RESULTS
                num_infection_attempts[0] += num_infection_attempts[i];
                num_infections[0] += num_infections[i];
                #endif
            }
        }
    }

    // use atomicAdd function to add the results to device pointers
    if(blockId == 0) {
        atomicAdd(num_infected_dev, num_infected[0]);
        atomicAdd(num_susceptible_dev, num_susceptible[0]);
        #ifdef SHOW_RESULTS
        atomicAdd(num_infection_attempts_dev, num_infection_attempts[0]);
        atomicAdd(num_infections_dev, num_infections[0]);
        #endif
    }
}

/*
    cuda_infected()
        Spawns threads to handle those that are infected by  
        deciding whether they should be marked immune or dead.
*/
__global__ void cuda_infected(char *states_dev, int *num_days_infected_dev, 
    int *num_recovery_attempts_dev, int *num_deaths_dev, 
    int *num_infected_dev, int *num_immune_dev, int *num_dead_dev,
    int duration_of_disease, int deadliness_factor, char IMMUNE, char DEAD, 
    char INFECTED, float *rand_nums)
{
    // set up thread id, block id and block dimension
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int blockId = threadIdx.x;
    int numThread = blockDim.x;

    // counter
    int i;

    // set up shared memory
    int* num_infected = (int*)array; 
    int* num_immune = (int*)&num_infected[numThread];
    int* num_dead = (int*)&num_immune[numThread];
    #ifdef SHOW_RESULTS
    int* num_recovery_attempts = (int*)&num_dead[numThread];
    int* num_deaths = (int*)&num_recovery_attempts[numThread];
    #endif

    // reset the shared memory
    num_infected[blockId] = 0;
    num_immune[blockId] = 0;
    num_dead[blockId] = 0;
    #ifdef SHOW_RESULTS
    num_recovery_attempts[blockId] = 0;
    num_deaths[blockId] = 0;
    #endif

    // If the person is infected and has been for the full 
    // duration of the disease, then
    if(states_dev[id] == INFECTED && num_days_infected_dev[id] == duration_of_disease)
    {
        #ifdef SHOW_RESULTS
        num_recovery_attempts[blockId]++;
        #endif

        // generate a random number between 0 and 100
        int rand_num = (int)(rand_nums[id]*100);

        // If a random number less than 100 is less than 
        // the deadliness factor, then
        if(rand_num <= deadliness_factor)
        {
            // The thread changes the person’s state to dead 
            states_dev[id] = DEAD;
            // The thread updates the counters
            num_dead[blockId]++;
            num_infected[blockId]--;
            #ifdef SHOW_RESULTS
            num_deaths[blockId]++;
            #endif
        }
        else
        {
            // The thread changes the person’s state to immune
            states_dev[id] = IMMUNE;
            // The thread updates the counters
            num_immune[blockId]++;
            num_infected[blockId]--;
        }
    }

    __syncthreads();

    // if we have numThread to the power of 2, we can use binary
    // tree reduction to increase performance
    if(((numThread!=0) && !(numThread & (numThread-1)))){
        i = numThread/2;
        while (i != 0) {
            if (blockId < i){
                num_infected[blockId] += num_infected[blockId + i];
                num_immune[blockId] += num_immune[blockId + i];
                num_dead[blockId] += num_dead[blockId + i];
                #ifdef SHOW_RESULTS
                num_recovery_attempts[blockId] += num_recovery_attempts[blockId + i];
                num_deaths[blockId] += num_deaths[blockId + i];
                #endif
            }
            __syncthreads();
            i /= 2; 
        }
    }
    // Else, we can only add-up results in shared memory using
    // the first thread in each block
    else{
        if(blockId == 0) {
            for(i=1; i<numThread; i++){
                num_infected[0] += num_infected[i];
                num_immune[0] += num_immune[i];
                num_dead[0] += num_dead[i];
                #ifdef SHOW_RESULTS
                num_recovery_attempts[0] += num_recovery_attempts[i];
                num_deaths[0] += num_deaths[i];
                #endif
            }
        }
    }

    // use atomicAdd function to add the results to device pointers
    if(blockId == 0) {
        atomicAdd(num_infected_dev, num_infected[0]);
        atomicAdd(num_immune_dev, num_immune[0]);
        atomicAdd(num_dead_dev, num_dead[0]);
        #ifdef SHOW_RESULTS
        atomicAdd(num_recovery_attempts_dev, num_recovery_attempts[0]);
        atomicAdd(num_deaths_dev, num_deaths[0]);
        #endif
    }
}

/*
    cuda_update_days_infected()
        Spawns threads to handle those that are infected by
        increasing the number of days infected.
*/
__global__ void cuda_update_days_infected(char *states_dev, int *num_days_infected_dev,
    char INFECTED)
{
    // set up thread id
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // If the person is infected, then
    if(states_dev[id] == INFECTED)
    {
        // Increment the number of days the person has been infected
        num_days_infected_dev[id]++;
    }
}

/*
    cuda_init()
        initialize cuda environment
*/
void cuda_init(struct global_t *global, struct cuda_t *cuda)
{
    // for timing purpose
    cudaEventCreate(&cuda->start);
    cudaEventCreate(&cuda->stop);
    cudaEventRecord(cuda->start,0);

    // initialize size needed for cudamalloc operations
    cuda->people_size = sizeof(int) * global->number_of_people;
    cuda->states_size = sizeof(char) * global->number_of_people;

    // allocate the memory on the GPU
    // arrays in global struct
    cudaMalloc((void**)&cuda->infected_x_locations_dev, cuda->people_size);
    cudaMalloc((void**)&cuda->infected_y_locations_dev, cuda->people_size);
    cudaMalloc((void**)&cuda->x_locations_dev, cuda->people_size);
    cudaMalloc((void**)&cuda->y_locations_dev, cuda->people_size);
    cudaMalloc((void**)&cuda->states_dev, cuda->states_size);
    cudaMalloc((void**)&cuda->num_days_infected_dev, cuda->people_size);
    // states counters in global struct
    cudaMalloc((void**)&cuda->num_susceptible_dev, sizeof(int));
    cudaMalloc((void**)&cuda->num_immune_dev, sizeof(int));
    cudaMalloc((void**)&cuda->num_dead_dev, sizeof(int));
    cudaMalloc((void**)&cuda->num_infected_dev, sizeof(int));
    #ifdef SHOW_RESULTS
    // stats variables in stats struct
    cudaMalloc((void**)&cuda->num_infections_dev, sizeof(int));
    cudaMalloc((void**)&cuda->num_infection_attempts_dev, sizeof(int));
    cudaMalloc((void**)&cuda->num_deaths_dev, sizeof(int));
    cudaMalloc((void**)&cuda->num_recovery_attempts_dev, sizeof(int));
    #endif

    // create cuda random number generator
    curandCreateGenerator(&cuda->gen, CURAND_RNG_PSEUDO_DEFAULT);
    // get time 
    time(&cuda->current_time);
    // generate seed for the rand number generator
    curandSetPseudoRandomGeneratorSeed(cuda->gen, (unsigned long)cuda->current_time);
    // array to hold random number
    cudaMalloc((void**)&cuda->rand_nums, 2 * global->number_of_people * sizeof(float));
    
    // set up 1D array for cuda
    // if we have less than 256 people, initialize only people size of threads
    cuda->numThread = (global->number_of_people < 512 ? global->number_of_people : 512);
    cuda->numBlock = (global->number_of_people+cuda->numThread-1)/cuda->numThread;
}

/*
    cuda_run()
        run cuda environment
*/
void cuda_run(struct global_t *global, struct stats_t *stats, 
    struct const_t *constant, struct cuda_t *cuda)
{
    // copy infected locations to device in EVERY ITERATION
    cudaMemcpy(cuda->infected_x_locations_dev, global->infected_x_locations, cuda->people_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda->infected_y_locations_dev, global->infected_y_locations, cuda->people_size, cudaMemcpyHostToDevice);
    
    // copy other information to device only in FIRST ITERATION
    // we don't need to copy these information every iteration 
    // becuase they can be reused in each iteration without any
    // process at the host end.
    if(global->current_day == 0){
        // variables in stats data are initialized as doubles, yet CUDA
        // atomic operations prefer integer than doubles. Therefore, we
        // cast doubles to integer before the cudaMemcpy operations.
        cuda->num_infections_int = (int)stats->num_infections;
        cuda->num_infection_attempts_int = (int)stats->num_infection_attempts;
        cuda->num_deaths_int = (int)stats->num_deaths;
        cuda->num_recovery_attempts_int = (int)stats->num_recovery_attempts;

        #ifdef SHOW_RESULTS
        // copy arrays in global struct
        cudaMemcpy(cuda->num_infections_dev, &cuda->num_infections_int, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->num_infection_attempts_dev, &cuda->num_infection_attempts_int, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->num_deaths_dev, &cuda->num_deaths_int, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->num_recovery_attempts_dev, &cuda->num_recovery_attempts_int, sizeof(int), cudaMemcpyHostToDevice);
        // copy states counters in global struct
        cudaMemcpy(cuda->x_locations_dev, global->x_locations, cuda->people_size, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->y_locations_dev, global->y_locations, cuda->people_size, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->states_dev, global->states, cuda->states_size, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->num_days_infected_dev, global->num_days_infected, cuda->people_size, cudaMemcpyHostToDevice);
        // copy stats variables in stats struct
        cudaMemcpy(cuda->num_susceptible_dev, &global->num_susceptible, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->num_immune_dev, &global->num_immune, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->num_dead_dev, &global->num_dead, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda->num_infected_dev, &global->num_infected, sizeof(int), cudaMemcpyHostToDevice);
        #endif
    }

    // generate 2 * number_of_people many of randome numbers.
    // we need twice many of people number because movement are in
    // both X direction and Y direction
    curandGenerateUniform(cuda->gen, cuda->rand_nums, 2*global->number_of_people);

    // execute device code on updating people's movement
    int environment_width = constant->environment_width;
    int environment_height = constant->environment_height;
    cuda_move<<<cuda->numBlock, cuda->numThread>>>(cuda->states_dev, 
        cuda->x_locations_dev, cuda->y_locations_dev, DEAD, 
        environment_width, environment_height, cuda->rand_nums, global->number_of_people);
    // Sync Threads
    cudaThreadSynchronize();

    // generate number_of_people many of randome numbers.
    curandGenerateUniform(cuda->gen, cuda->rand_nums, global->number_of_people);

    // execute device code on susceptible people
    int infection_radius = constant->infection_radius;
    int contagiousness_factor = constant->contagiousness_factor;
    int num_infected = global->num_infected;
    cuda_susceptible<<<cuda->numBlock, cuda->numThread, 4*cuda->numThread*sizeof(int)>>>(
        cuda->states_dev, cuda->x_locations_dev, cuda->y_locations_dev, 
        cuda->infected_x_locations_dev, cuda->infected_y_locations_dev, 
        cuda->num_infected_dev, cuda->num_susceptible_dev, 
        cuda->num_infection_attempts_dev, cuda->num_infections_dev, 
        cuda->rand_nums, num_infected, infection_radius, 
        contagiousness_factor, SUSCEPTIBLE, INFECTED);
    // Sync Threads
    cudaThreadSynchronize();

    // generate number_of_people many of randome numbers.
    curandGenerateUniform(cuda->gen, cuda->rand_nums, global->number_of_people);

    // execute device code on infected people
    int duration_of_disease = constant->duration_of_disease;
    int deadliness_factor = constant->deadliness_factor;    
    cuda_infected<<<cuda->numBlock, cuda->numThread, 5*cuda->numThread*sizeof(int)>>>(
        cuda->states_dev, cuda->num_days_infected_dev, 
        cuda->num_recovery_attempts_dev, cuda->num_deaths_dev, 
        cuda->num_infected_dev, cuda->num_immune_dev, 
        cuda->num_dead_dev, duration_of_disease, deadliness_factor, 
        IMMUNE, DEAD, INFECTED, cuda->rand_nums);
    // Sync Threads
    cudaThreadSynchronize();
    
    // execute device code to update infected days
    cuda_update_days_infected<<<cuda->numBlock, cuda->numThread>>>(cuda->states_dev, 
        cuda->num_days_infected_dev, INFECTED);
    // Sync Threads
    cudaThreadSynchronize();

    // copy locations, states and num_infected back to host
    // in EVERY ITERATION
    cudaMemcpy(global->x_locations, cuda->x_locations_dev, cuda->people_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(global->y_locations, cuda->y_locations_dev, cuda->people_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(global->states, cuda->states_dev, cuda->states_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&global->num_infected, cuda->num_infected_dev, sizeof(int), cudaMemcpyDeviceToHost);

    // copy other information back to host only in LAST ITERATION
    // we only copy the counters back for results calculation.
    // we don't need to copy num_days_infected back.
    if(global->current_day == constant->total_number_of_days){
        // copy states counters in our struct
        cudaMemcpy(&global->num_susceptible, cuda->num_susceptible_dev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&global->num_immune, cuda->num_immune_dev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&global->num_dead, cuda->num_dead_dev, sizeof(int), cudaMemcpyDeviceToHost);
        
        #ifdef SHOW_RESULTS
        // copy stats variables in stats struct
        cudaMemcpy(&cuda->num_infections_int, cuda->num_infections_dev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cuda->num_infection_attempts_int, cuda->num_infection_attempts_dev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cuda->num_deaths_int, cuda->num_deaths_dev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cuda->num_recovery_attempts_int, cuda->num_recovery_attempts_dev, sizeof(int), cudaMemcpyDeviceToHost);
        // cast interger back to double after the cudaMemcpy operations.
        stats->num_infections = (double)cuda->num_infections_int;
        stats->num_infection_attempts = (double)cuda->num_infection_attempts_int;
        stats->num_deaths = (double)cuda->num_deaths_int;
        stats->num_recovery_attempts = (double)cuda->num_recovery_attempts_int;
        #endif
    }
}

/*
    cuda_finish()
        clean up cuda environment
*/
void cuda_finish(struct cuda_t *cuda)
{
    // free the memory on the GPU
    // arrays in global and our struct
    cudaFree(cuda->infected_x_locations_dev);
    cudaFree(cuda->infected_y_locations_dev);
    cudaFree(cuda->x_locations_dev);
    cudaFree(cuda->y_locations_dev);
    cudaFree(cuda->states_dev);
    cudaFree(cuda->num_days_infected_dev);
    // states counters in our struct
    cudaFree(cuda->num_susceptible_dev);
    cudaFree(cuda->num_immune_dev);
    cudaFree(cuda->num_dead_dev);
    cudaFree(cuda->num_infected_dev);

    #ifdef SHOW_RESULTS
    // stats variables in stats struct
    cudaFree(cuda->num_infections_dev);
    cudaFree(cuda->num_infection_attempts_dev);
    cudaFree(cuda->num_deaths_dev);
    cudaFree(cuda->num_recovery_attempts_dev);
    #endif

    // array to hold random number
    cudaFree(cuda->rand_nums);
    // destroy cuda random number generator
    curandDestroyGenerator(cuda->gen);

    // get stop time, and display the timing results
    cudaEventRecord(cuda->stop, 0);
    cudaEventSynchronize(cuda->stop);
    cudaEventElapsedTime(&cuda->elapsedTime, cuda->start, cuda->stop);
}

#endif