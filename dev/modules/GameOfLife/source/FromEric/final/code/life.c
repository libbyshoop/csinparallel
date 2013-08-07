#include <string.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <getopt.h>
#include <sys/time.h>
#include "callbacks.h"
#include "life.h"
#include "fileio.h"
#include "compute.h"

struct GridInfo grid;
struct View view;
struct ComputeParams compute_params;

static bool create_window(unsigned window_width, unsigned window_height);

static struct option longopts[] = {
	{"grid-size", required_argument, NULL, 's'},
	{"window-size", required_argument, NULL, 'w'},
	{"fullscreen", no_argument, NULL, 'f'},
	{"time", required_argument, NULL, 't'},
	{"help", no_argument, NULL, 'h'},
};

static void usage(const char* program_name)
{
	const char* a = 
	"Usage: %s [OPTION...] [FILE]\n"
	"Run a Conway's Game of Life simulation.\n"
	"\n"	
	"  -s, --grid-size WIDTHxHEIGHT     Set the size of the grid.  Overrides the size\n"
	"                                     that would normally be used for any input \n"
	"                                     files.\n"
	"  -w, --window-size WIDTHxHEIGHT   Set the size of the window.  \n"
	"  -t, --time NUM_GENERATIONS       Print how long it takes to simulate this many generations.\n"
	"                                      The GUI is not created.\n"
	"  -f, --fullscreen                 Start fullscreen.  \n"
	"  -h, --help                       Show this help.\n"
	;

	fprintf(stderr, a, program_name);
	exit(1);
}

int main(int argc, char** argv)
{
	char c;
	unsigned width_override = 0;
	unsigned height_override = 0;
	unsigned window_width = 512;
	unsigned window_height = 512;
	unsigned long time_generations = 0;
	view.fullscreen = true;

	DEBUG("Initializing GLUT\n");
	glutInit (&argc, argv);
	glutInitDisplayMode (GLUT_RGBA);

	while ((c = getopt_long(argc, argv, "w:s:hft:", longopts, NULL)) != -1) {
		switch (c) {
		case 's':
			sscanf(optarg, "%ux%u\n", &width_override, &height_override);
			break;
		case 'w':
			sscanf(optarg, "%ux%u\n", &window_width, &window_height);
			break;
		case 'f':
			view.fullscreen = true;
			break;
		case 't':
			time_generations = strtoul(optarg, NULL, 10);
			break;
		case 'h':
		default:
			usage(argv[0]);
		}
	}
	argc -= optind;
	argv += optind;


	atexit(cleanup);
	init_parallel_component();

	compute_params.generations_per_redraw = 0.05;

	create_window(window_width, window_height);

	uint8_t* data = NULL;
	unsigned width = width_override ? width_override : 128;
	unsigned height = height_override ? height_override : 128;
	if (argc >= 1) {
		const char* filename = argv[0];
		DEBUG("Filename argument \"%s\" given\n", filename);
		unsigned __width, __height;
		data = read_file(filename, &__width, &__height, width_override, height_override);
		if (data != NULL) {
			width = __width;
			height = __height;
		}
	}
	create_grid(width, height, data);

	if (time_generations) {
		struct timeval t1, t2;
		gettimeofday(&t1, NULL);
		DEBUG("Timing how long it takes to advance %lu generations\n", time_generations);
		advance_generations(time_generations);
		gettimeofday(&t2, NULL);
		unsigned long us_elapsed =  (t2.tv_sec * 1000000 + t2.tv_usec) - 
							(t1.tv_sec * 1000000 + t1.tv_usec);
		printf("%lu milliseconds elapsed\n", us_elapsed / 1000);
	} else {
		reset_view();
		glutMainLoop();

	}

	return 0;
}


static bool create_window(unsigned window_width, unsigned window_height)
{
	view.window_width = window_width;
	view.window_height = window_height;

	glutInitWindowSize (view.window_width, view.window_height);

	DEBUG("Creating window\n");
	glutCreateWindow ("Conway's Game of Life");

	DEBUG("Registering callbacks\n");
	glutDisplayFunc (display_cb);
	glutKeyboardFunc (keyboard_cb);
	glutSpecialFunc (special_keyboard_cb);
	glutMotionFunc (motion_cb);
	glutMouseFunc (mouse_cb);
	glutReshapeFunc (reshape_cb);
	glutMouseWheelFunc (mousewheel_cb);

	DEBUG("Querying for OpenGL 2.1\n");
	GLenum glew_err = glewInit();
	if (glew_err != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_err));
		return false;
	}

	if (!GLEW_VERSION_2_1) {
		fprintf(stderr, "Error: Support for OpenGL 2.1 not found.");
		return false;
	}

	DEBUG("Setting up initial OpenGL state\n");
	glClearColor(0.0f, 0.0f, 0.2f, 1.0f);
	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, view.window_width, view.window_height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, (GLdouble)view.window_width, 0.0, (GLdouble)view.window_height, 0.0, 1.0);

	check_gl_error();

	if (view.fullscreen) {
		view.fullscreen = false;
		toggle_fullscreen();
	}

	return true;
}

void* xmalloc(size_t size)
{
	void* p = malloc(size);
	if (p == NULL) {
		fprintf(stderr, "Error: Out of memory (tried to allocate %lu bytes)\n", 
				(unsigned long)size);
		exit(1);
	}
	return p;
}

void* zalloc(size_t size)
{
	void *p = xmalloc(size);
	memset(p, 0, size);
	return p;
}

void add_R_pentonimo(uint8_t* buf, unsigned width, unsigned height)
{
	/*
	 *  R Pentonimo
	 *      11
	 *     11
	 *      1
	 */
	unsigned half_height = height / 2;
	unsigned half_width = width / 2;
	buf[half_width + half_height * width] = 1;
	buf[half_width + half_height * width - 1] = 1;
	buf[half_width + (half_height - 1) * width] = 1;
	buf[half_width + (half_height + 1) * width] = 1;
	buf[half_width + (half_height + 1) * width + 1] = 1;
}


void check_gl_error()
{
	GLenum gl_err = glGetError();
	if (gl_err != GL_NO_ERROR) {
		fprintf(stderr, "GL error: %s\n", gluErrorString(gl_err));
		exit(1);
	}
}


