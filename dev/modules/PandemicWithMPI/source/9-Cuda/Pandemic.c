/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized and restructured the original code, added CUDA) */

#include <stdio.h>      // for printf
#include <stdlib.h>     // for malloc, free, and various others
#include <time.h>       // for time is used to seed the random number generator
#include <unistd.h>     // for random, getopt, some others
#include <X11/Xlib.h>   // for X display

#include "Defaults.h"
#include "Initialize.h"
#include "Infection.h"
#include "CUDA.cu"
#include "Finalize.h"

#if defined(X_DISPLAY) || defined(TEXT_DISPLAY)
#include "Display.h"
#endif

int main(int argc, char ** argv) 
{
    /**** In Defaults.h ****/
    struct global_t global;
    struct const_t constant;
    struct stats_t stats;
    struct display_t dpy;
    struct cuda_t cuda;
    /***********************/

    /******************** In Initialize.h *********************/
    init(&global, &constant, &stats, &dpy, &cuda, &argc, &argv);
    /**********************************************************/ 

    // Process starts a loop to run the simulation for the
    // specified number of days
    for(global.current_day = 0; global.current_day <= constant.total_number_of_days; 
        global.current_day++)
    {
        /* for debug purpose */
        printf("Iteration %d \n", global.current_day);

        /****** In Infection.h ******/
        find_infected(&global);
        /****************************/

        /**************** In Display.h *****************/
        #if defined(X_DISPLAY) || defined(TEXT_DISPLAY)

        do_display(&global, &constant, &dpy);

        throttle(&constant);

        #endif
        /***********************************************/

        /**************** In CUDA.CU ***************/
        cuda_run(&global, &stats, &constant, &cuda);
        /*******************************************/
    }

    /************ In Finialize.h ************/
    show_results(&global, &stats);

    cleanup(&global, &constant, &dpy, &cuda);
    /****************************************/

    /* for timing purpose */
    printf("Total time %3.1f ms with size %d and generation %d !\n", cuda.elapsedTime, 
        global.number_of_people, constant.total_number_of_days);

    /* for debug purpose */
    printf("num_infections is %f \n", stats.num_infections);
    printf("num_infection_attempts is %f \n", stats.num_infection_attempts);
    printf("num_deaths is %f \n", stats.num_deaths);
    printf("num_recovery_attempts is %f \n", stats.num_recovery_attempts);

    exit(EXIT_SUCCESS);
}