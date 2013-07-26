/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 * Modified by Yu Zhao, Macalester College.
 * July 2013
 * (Modularized the original code and added CUDA Programming) */

#ifndef PANDEMIC_DISPLAY_H
#define PANDEMIC_DISPLAY_H

#include <stdio.h>      // printf
#include <stdlib.h>     // malloc, free, and various others
#include <X11/Xlib.h>   // X display

#ifdef _MPI
#include <mpi.h>        // for MPI_Finalize()
#endif

void        init_display(struct our_t *our, struct const_t *constant, 
                struct display_t *dpy);
void        share_display_info(struct global_t *global, 
                struct our_t *our);
void        do_display(struct global_t *global, struct our_t *our, 
                struct const_t *constant, struct display_t *dpy);
void        close_display(struct our_t *our, struct display_t *dpy);
void        throttle(struct const_t *constant);

/*
    init_display()
        Rank 0 initializes the graphics display
*/
void init_display(struct our_t *our, struct const_t *constant, struct display_t *dpy)
{
    if(our->our_rank == 0)
    {
        /* Initialize the X Windows Environment
         * This all comes from 
         *   http://en.wikibooks.org/wiki/X_Window_Programming/XLib
         *   http://tronche.com/gui/x/xlib-tutorial
         *   http://user.xmission.com/~georgeps/documentation/tutorials/
         *      Xlib_Beginner.html
         */
        /* Open a connection to the X server */
        dpy->display = XOpenDisplay(NULL);
        if(dpy->display == NULL)
        {
            fprintf(stderr, "Error: could not open X dpy->display\n");
        }
        dpy->screen = DefaultScreen(dpy->display);
        dpy->window = XCreateSimpleWindow(dpy->display, RootWindow(dpy->display, dpy->screen),
                0, 0, constant->environment_width * PIXEL_WIDTH_PER_PERSON, 
                constant->environment_height * PIXEL_HEIGHT_PER_PERSON, 1,
                BlackPixel(dpy->display, dpy->screen), WhitePixel(dpy->display, dpy->screen));
        dpy->delete_window = XInternAtom(dpy->display, "WM_DELETE_WINDOW", 0);
        XSetWMProtocols(dpy->display, dpy->window, &dpy->delete_window, 1);
        XSelectInput(dpy->display, dpy->window, ExposureMask | KeyPressMask);
        XMapWindow(dpy->display, dpy->window);
        dpy->colormap = DefaultColormap(dpy->display, 0);
        dpy->gc = XCreateGC(dpy->display, dpy->window, 0, 0);
        XParseColor(dpy->display, dpy->colormap, dpy->red, &dpy->infected_color);
        XParseColor(dpy->display, dpy->colormap, dpy->green, &dpy->immune_color);
        XParseColor(dpy->display, dpy->colormap, dpy->white, &dpy->dead_color);
        XParseColor(dpy->display, dpy->colormap, dpy->black, &dpy->susceptible_color);
        XAllocColor(dpy->display, dpy->colormap, &dpy->infected_color);
        XAllocColor(dpy->display, dpy->colormap, &dpy->immune_color);
        XAllocColor(dpy->display, dpy->colormap, &dpy->susceptible_color);
        XAllocColor(dpy->display, dpy->colormap, &dpy->dead_color);
    }
}

/*
    share_display_info()
        Each process sends its location information and state information
        to all the other processes and receive those information from other
        processes as well
*/
void share_display_info(struct global_t *global, struct our_t *our)
{
    #ifdef _MPI
    int total_number_of_processes = global->total_number_of_processes;
    int total_number_of_people = global->total_number_of_people;

    // Distributed Memory Information
    int *recvcounts;
    int *displs;
    recvcounts = (int*)malloc(total_number_of_processes * sizeof(int));  
    displs = (int*)malloc(total_number_of_processes * sizeof(int));

    // Set up the receive counts and displacements in the 
    // receive buffer (see the man page for MPI_Gatherv)
    int current_displ = 0;
    int current_rank;
    for(current_rank = 0; current_rank <= total_number_of_processes - 1;
           current_rank++)
    {
        displs[current_rank] = current_displ;
        recvcounts[current_rank] = total_number_of_people / total_number_of_processes;
        if(current_rank == global->total_number_of_processes - 1)
        {
            recvcounts[current_rank] += total_number_of_people
            % total_number_of_processes;
        }
        current_displ += recvcounts[current_rank];
    }

    MPI_Gatherv(our->our_states, our->our_number_of_people, MPI_CHAR, 
        global->states, recvcounts, displs, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gatherv(our->our_x_locations, our->our_number_of_people, MPI_INT, 
        global->x_locations, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(our->our_y_locations, our->our_number_of_people, MPI_INT, 
        global->y_locations, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    free(displs);
    free(recvcounts);
    // if not using MPI
    #else
    int my_current_person_id;
    #ifdef _OPENMP 
        #pragma omp parallel for private(my_current_person_id)
    #endif
    for(my_current_person_id = 0; my_current_person_id 
       <= global->total_number_of_people - 1; my_current_person_id++)
    {
        global->states[my_current_person_id] 
        = our->our_states[my_current_person_id];
        global->x_locations[my_current_person_id] 
        = our->our_x_locations[my_current_person_id];
        global->y_locations[my_current_person_id] 
        = our->our_y_locations[my_current_person_id];
    }
    #endif
}

/*
    do_display()
        If display is enabled, Rank 0 displays a graphic of the current day
*/
void do_display(struct global_t *global, struct our_t *our, 
    struct const_t *constant, struct display_t *dpy)
{
    #ifdef X_DISPLAY
    int my_current_person_id;

    char *states = global->states;
    int *x_locations = global->x_locations;
    int *y_locations = global->y_locations;

    if(our->our_rank == 0)
    {
        XClearWindow(dpy->display, dpy->window);
        for(my_current_person_id = 0; my_current_person_id 
            <= global->total_number_of_people - 1; my_current_person_id++)
        {
            if(states[my_current_person_id] == INFECTED)
            {
                XSetForeground(dpy->display, dpy->gc, dpy->infected_color.pixel);
            }
            else if(states[my_current_person_id] == IMMUNE)
            {
                XSetForeground(dpy->display, dpy->gc, dpy->immune_color.pixel);
            }
            else if(states[my_current_person_id] == SUSCEPTIBLE)
            {
                XSetForeground(dpy->display, dpy->gc, dpy->susceptible_color.pixel);
            }
            else if(states[my_current_person_id] == DEAD)
            {
                XSetForeground(dpy->display, dpy->gc, dpy->dead_color.pixel);
            }
            else
            {
                fprintf(stderr, "ERROR: person %d has state '%c'\n",
                    my_current_person_id, states[my_current_person_id]);
                exit(-1);
            }
            XFillRectangle(dpy->display, dpy->window, dpy->gc,
                x_locations[my_current_person_id] 
                * PIXEL_WIDTH_PER_PERSON, 
                y_locations[my_current_person_id]
                * PIXEL_HEIGHT_PER_PERSON, 
                PIXEL_WIDTH_PER_PERSON, 
                PIXEL_HEIGHT_PER_PERSON);
        }
        XFlush(dpy->display);
    }
    #endif

    #ifdef TEXT_DISPLAY
    int our_current_location_x;
    int our_current_location_y;
    int environment_height = constant->environment_height
    int environment_width = constant->environment_width;

    char *states = global->states;
    int *x_locations = global->x_locations;
    int *y_locations = global->y_locations;

    if(our->our_rank == 0)
    {
        for(our_current_location_y = 0; 
            our_current_location_y <= environment_height - 1;
            our_current_location_y++)
        {
            for(our_current_location_x = 0; our_current_location_x 
                <= environment_width - 1; our_current_location_x++)
            {
                dpy->environment[our_current_location_x][our_current_location_y] 
                = ' ';
            }
        }

        for(my_current_person_id = 0; 
            my_current_person_id <= global->total_number_of_people - 1;
            my_current_person_id++)
        {
            dpy->environment[x_locations[my_current_person_id]]
            [y_locations[my_current_person_id]] = 
            states[my_current_person_id];
        }

        printf("----------------------\n");
        for(our_current_location_y = 0;
            our_current_location_y <= environment_height - 1;
            our_current_location_y++)
        {
            for(our_current_location_x = 0; our_current_location_x 
                <= environment_width - 1; our_current_location_x++)
            {
                printf("%c", dpy->environment[our_current_location_x]
                    [our_current_location_y]);
            }
            printf("\n");
        }
    }
    #endif
}

/*
    close_display()
        If X display is enabled, then Rank 0 destroys the 
        X Window and closes the display
*/
void close_display(struct our_t *our, struct display_t *dpy)
{
    if(our->our_rank == 0)
    {
        XDestroyWindow(dpy->display, dpy->window);
        XCloseDisplay(dpy->display);
    }
}

/*
    throttle()
        Slows down the simulation to make X display easier to watch.
*/
void throttle(struct const_t *constant)
{
    // Wait between frames of animation
    usleep(constant->microseconds_per_day);
}

#endif