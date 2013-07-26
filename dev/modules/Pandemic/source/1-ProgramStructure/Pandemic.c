/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized the original code and added CUDA Programming)
 *
 * Parallel code -- MPI for distributed memory (processes), OpenMP for shared
 * memory (threads), CUDA for GPU programming. Two versions of "Hybrid" uses 
 * either both MPI and OpenMP (in which case each MPI process can spawn OpenMP 
 * threads) or both MPI and CUDA (in which case each MPI process can call GPU
 * card for computation). */

#include <stdio.h>      // for printf
#include <stdlib.h>     // for malloc, free, and various others
#include <time.h>       // for time is used to seed the random number generator
#include <unistd.h>     // for random, getopt, some others
#include <X11/Xlib.h>   // for X display

#include "Defaults.h"
#include "Initialize.h"
#include "Infection.h"
#include "Finalize.h"

#if defined(X_DISPLAY) || defined(TEXT_DISPLAY)
#include "Display.h"
#endif

#if !defined(__CUDACC__) && !defined(MPICUDA)
#include "Core.h"
#endif

#ifdef __CUDACC__
#include "CUDA.cu"
#endif

int main(int argc, char ** argv) {

    /**** In Defaults.h ****/
    struct global_t global;
    struct our_t our;
    struct const_t constant;
    struct stats_t stats;
    struct display_t dpy;
    struct cuda_t cuda;
    /***********************/

    /************************ In Initialize.h ************************/
    init(&global, &our, &constant, &stats, &dpy, &cuda, &argc, &argv);
    /*****************************************************************/

    // Each process starts a loop to run the simulation 
    // for the specified number of days
    for(our.current_day = 0; our.current_day <= constant.total_number_of_days; 
        our.current_day++)
    {
        /****** In Infection.h ******/
        find_infected(&our);

        share_infected(&global, &our);
        /****************************/

        /**************** In Display.h *****************/
        #if defined(X_DISPLAY) || defined(TEXT_DISPLAY)

        share_display_info(&global, &our);

        do_display(&global, &our, &constant, &dpy);

        throttle(&constant);
        
        #endif
        /***********************************************/

        // If MPICUDA or __CUDACC__ is defined, then we 
        // are running CUDA only or MPI-CUDA hybrid
        #if defined(__CUDACC__) || defined(MPICUDA)

        /****************** In CUDA.cu ******************/
        cuda_run(&global, &our, &constant, &stats, &cuda);
        /************************************************/

        // Else, we are running serial, OMP only, MPI only or MPI-OMP hybrid
        #else

        /******************* In Core.h ******************/
        move(&our, &constant);      

        susceptible(&global, &our, &constant, &stats);

        infected(&our, &constant, &stats);

        update_days_infected(&our, &constant);
        /************************************************/

        #endif
    }

    /**************** In Finialize.h **************/
    show_results(&our, &stats);

    cleanup(&global, &our, &constant, &dpy, &cuda);
    /**********************************************/

    exit(EXIT_SUCCESS);
}