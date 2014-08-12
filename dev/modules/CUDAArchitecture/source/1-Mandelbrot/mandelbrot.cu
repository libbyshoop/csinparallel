#include <stdio.h>
#include <unistd.h>
#include <err.h>
#include <stdint.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <omp.h>


static int dim = 512;
static int n = 512;
static int m = 512;
static int max_iter = 100;
static uint32_t *colors;
uint32_t *dev_colors;
// X11 data 
#ifdef SHOW_X
static Display *dpy;
static XImage *bitmap;
static Window win;
static Atom wmDeleteMessage;
static GC gc;

//destroy window and x variables 
static void exit_x11(void){
	XDestroyWindow(dpy, win);
	XCloseDisplay(dpy);
}

// create Xwindow 
static void init_x11(){
	// Attempt to open the display 
	dpy = XOpenDisplay(NULL);
	
	// Failure
	if (!dpy) exit(0);
	
	uint32_t long white = WhitePixel(dpy,DefaultScreen(dpy));
	uint32_t long black = BlackPixel(dpy,DefaultScreen(dpy));
	

	win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy),
            0, 0, dim, dim, 0, black, white);
	
	// We want to be notified when the window appears 
	XSelectInput(dpy, win, StructureNotifyMask);
	
	// Make it appear 
	XMapWindow(dpy, win);
	
	while (1){
        XEvent e;
		XNextEvent(dpy, &e);
		if (e.type == MapNotify) break;
	}
	
	XTextProperty tp;
	char name[128] = "Mandelbrot";
	char *n = name;
	Status st = XStringListToTextProperty(&n, 1, &tp);
	if (st) XSetWMName(dpy, win, &tp);

	// Wait for the MapNotify event 
	XFlush(dpy);
    int depth = DefaultDepth(dpy, DefaultScreen(dpy));    
    Visual *visual = DefaultVisual(dpy, DefaultScreen(dpy));

    bitmap = XCreateImage(dpy, visual, depth, ZPixmap, 0,
            (char*) malloc(dim * dim * 32), dim, dim, 32, 0);

	// Init GC 
	gc = XCreateGC(dpy, win, 0, NULL);
	XSetForeground(dpy, gc, black);
	
	XSelectInput(dpy, win, ExposureMask | KeyPressMask | StructureNotifyMask);
	
	wmDeleteMessage = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
	XSetWMProtocols(dpy, win, &wmDeleteMessage, 1);
}
#endif

//create colors used to draw the mandelbrot set 
void init_colours(void) {
    float freq = 6.3 / max_iter;
	for (int i = 0; i < max_iter; i++){
        char r = sin(freq * i + 3) * 127 + 128;
        char g = sin(freq * i + 5) * 127 + 128;
        char b = sin(freq * i + 1) * 127 + 128;
		
		colors[i] = b + 256 * g + 256 * 256 * r;
	}
	
	colors[max_iter] = 0;
}

void checkErr(cudaError_t err, char* msg){
    if (err != cudaSuccess){
        fprintf(stderr, "%s (error code %d: '%s'", msg, err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* the mandelbrot set is defined as all complex numbers c such that the 
   equation z = z^2 + c remains bounded. In practice, we calculate max_iter
   iterations of this formula and if the magnitude of z is < 2 we assume it
   is in the set. The greater max_iters the more accurate our representation */
__device__ uint32_t mandel_double(double cr, double ci, int max_iter) {
    double zr = 0;
    double zi = 0;
    double zrsqr = 0;
    double zisqr = 0;

    uint32_t i;

    for (i = 0; i < max_iter; i++){
		zi = zr * zi;
		zi += zi;
		zi += ci;
		zr = zrsqr - zisqr + cr;
		zrsqr = zr * zr;
		zisqr = zi * zi;
		
    //the fewer iterations it takes to diverge, the farther from the set
		if (zrsqr + zisqr > 4.0) break;
    }
	
    return i;
}

/* turn each x y coordinate into a complex number and run the mandelbrot formula on it */
__global__ void mandel_kernel(uint32_t *counts, double xmin, double ymin,
            double step, int max_iter, int dim, uint32_t *colors) {
    int pix_per_thread = dim * dim / (gridDim.x * blockDim.x);
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = pix_per_thread * tId;
    for (int i = offset; i < offset + pix_per_thread; i++){
        int x = i % dim;
        int y = i / dim;
        double cr = xmin + x * step;
        double ci = ymin + y * step;
        counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)];
    }
    if (gridDim.x * blockDim.x * pix_per_thread < dim * dim
            && tId < (dim * dim) - (blockDim.x * gridDim.x)){
        int i = blockDim.x * gridDim.x * pix_per_thread + tId;
        int x = i % dim;
        int y = i / dim;
        double cr = xmin + x * step;
        double ci = ymin + y * step;
        counts[y * dim + x]  = colors[mandel_double(cr, ci, max_iter)];
    }
    
}

/* For each point, evaluate its colour */
static void display_double(double xcen, double ycen, double scale,
        uint32_t *dev_counts, uint32_t *colors){
    dim3 numBlocks(dim,dim);
    double xmin = xcen - (scale/2);
    double ymin = ycen - (scale/2);
    double step = scale / dim;
    cudaError_t err = cudaSuccess;

#ifdef BENCHMARK
    double start = omp_get_wtime();
#endif 

    mandel_kernel<<<n, m>>>(dev_counts, xmin , ymin, step, max_iter, dim, colors);
    checkErr(err, "Failed to run Kernel");
#ifdef SHOW_X
    err = cudaMemcpy(bitmap->data, dev_counts, dim * dim * sizeof(uint32_t), cudaMemcpyDeviceToHost);
#else
    void *data = malloc(dim * dim * sizeof(uint32_t));
    err = cudaMemcpy(data, dev_counts, dim * dim * sizeof(uint32_t), cudaMemcpyDeviceToHost);
#endif
    checkErr(err, "Failed to copy dev_counts back");

#ifdef BENCHMARK
    double stop = omp_get_wtime();
    printf("Blocks: %d\tThreads per Block: %d\tSize:%dx%d\tDepth: %d\tTime: %f\n",
            n, m, dim, dim, max_iter, stop - start);
#endif
#ifdef SHOW_X
    XPutImage(dpy, win, gc, bitmap,
        0, 0, 0, 0,
        dim, dim);
    XFlush(dpy); 
#endif
}


int main(int argc, char** argv){
    cudaError_t err = cudaSuccess;
    if (argc >= 2) 
        n = atoi(argv[1]);
    if (argc >= 3) 
        m = atoi(argv[2]);
    if (argc >= 4) 
        dim = atoi(argv[3]);
    if (argc >= 5) 
        max_iter = atoi(argv[4]);
    size_t color_size = (max_iter +1) * sizeof(uint32_t);
    colors = (uint32_t *) malloc(color_size);
    cudaMalloc((void**)&dev_colors, color_size);
    double xcen = -0.5;
    double ycen = 0;
    double scale = 3;

#ifdef SHOW_X
	init_x11();
#endif
	init_colours();
    cudaMemcpy(dev_colors, colors, color_size, cudaMemcpyHostToDevice);
    free(colors);

    uint32_t *dev_counts = NULL;
    size_t img_size = dim * dim * sizeof(uint32_t);
    err = cudaMalloc(&dev_counts, img_size);
    checkErr(err, "Failed to allocate dev_counts");

	display_double(xcen, ycen, scale, dev_counts, dev_colors);

#ifdef SHOW_X
	while(1) {
		XEvent event;
		KeySym key;
		char text[255];
		
		XNextEvent(dpy, &event);
        while (XPending(dpy) > 0)
            XNextEvent(dpy, &event);
		/* Just redraw everything on expose */
		if ((event.type == Expose) && !event.xexpose.count){
			XPutImage(dpy, win, gc, bitmap,
				0, 0, 0, 0,
				dim, dim);
		}
		
		/* Press 'x' to exit */
		if ((event.type == KeyPress) &&
			XLookupString(&event.xkey, text, 255, &key, 0) == 1)
			if (text[0] == 'x') break;
		
		/* Press 'a' to go left */
		if ((event.type == KeyPress) &&
			XLookupString(&event.xkey, text, 255, &key, 0) == 1)
			if (text[0] == 'a'){
                xcen -= 20 * scale / dim;
                display_double(xcen, ycen, scale, dev_counts, dev_colors);
            }

		/* Press 'w' to go up */
		if ((event.type == KeyPress) &&
			XLookupString(&event.xkey, text, 255, &key, 0) == 1)
			if (text[0] == 'w'){
                ycen -= 20 * scale / dim;
                display_double(xcen, ycen, scale, dev_counts, dev_colors);
            }

		/* Press 's' to go down */
		if ((event.type == KeyPress) &&
			XLookupString(&event.xkey, text, 255, &key, 0) == 1)
			if (text[0] == 's'){
                ycen += 20 * scale / dim;
                display_double(xcen, ycen, scale, dev_counts, dev_colors);
            }

		/* Press 'd' to go right */
		if ((event.type == KeyPress) &&
			XLookupString(&event.xkey, text, 255, &key, 0) == 1)
			if (text[0] == 'd'){
                xcen += 20 * scale / dim;
                display_double(xcen, ycen, scale, dev_counts, dev_colors);
            }

		/* Press 'q' to zoom out */
		if ((event.type == KeyPress) &&
			XLookupString(&event.xkey, text, 255, &key, 0) == 1)
			if (text[0] == 'q'){
                scale *= 1.25;
                display_double(xcen, ycen, scale, dev_counts, dev_colors);
            }

		/* Press 'e' to zoom in */
		if ((event.type == KeyPress) &&
			XLookupString(&event.xkey, text, 255, &key, 0) == 1)
			if (text[0] == 'e'){
                scale *= .80;
                display_double(xcen, ycen, scale, dev_counts, dev_colors);
            }

		/* Or simply close the window */
		if ((event.type == ClientMessage) &&
			((Atom) event.xclient.data.l[0] == wmDeleteMessage))
			break;
	}

    exit_x11();
#endif
    cudaFree(dev_counts);
    cudaFree(dev_colors);
	
	return 0;
}
