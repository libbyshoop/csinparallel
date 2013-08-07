#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdbool.h>
#include "callbacks.h"
#include "life.h"
#include "compute.h"

static void zoom(float x_factor, float y_factor);
static void get_rect_to_draw(int* x_ret, int* y_ret, int* width_ret, int* height_ret);

void display_cb()
{
	DEBUG2("Displaying contents of buffer %lu\n", grid.which_buf);
	glClear(GL_COLOR_BUFFER_BIT);

	int x, y, width, height;

	get_rect_to_draw(&x, &y, &width, &height);
	if (width > 0 && height > 0) {
		draw(x, y, width, height);
	}

	glutSwapBuffers();
	glutPostRedisplay();

	check_gl_error();

	
	compute_params.real_gen += compute_params.generations_per_redraw;

	unsigned long g = (unsigned long)compute_params.real_gen;
	if (g > compute_params.generation) {
		unsigned long generations_to_advance = g - compute_params.generation;
		printf("Advancing %lu generations to generation %lu\n", generations_to_advance, g);
		advance_generations(generations_to_advance);
	}
	compute_params.generation = g;
}

static void get_rect_to_draw(int* x_ret, int* y_ret, int* width_ret, int* height_ret)
{
	/* Use the zoom and the grid dimensions to find the number of screen pixels that
	 * it would take to show the full grid. */
	int grid_screen_width = (int)((float)grid.width * view.zoom_x);
	int grid_screen_height = (int)((float)grid.height * view.zoom_y);

	/* Find the screen pixel that the lower left corner of the grid would occupy. 
	 * It may be off the window. (0,0) is considered to be the lower left corner of the window. */
	int default_screen_x = (view.window_width - grid_screen_width) / 2;
	int default_screen_y = (view.window_height - grid_screen_height) / 2;
	int screen_x = default_screen_x - (int)(view.translate_x * view.zoom_x);
	int screen_y = default_screen_y - (int)(view.translate_y * view.zoom_y);

	/* find how many screen pixels need to be clipped on each border, and
	 * adjust screen_x/y and grid_screen_width/height to the clipped values. */
	int screen_left_cutoff = screen_x < 0 ? -screen_x : 0;
	int screen_bottom_cutoff = screen_y < 0 ? -screen_y : 0;
	grid_screen_width -= screen_left_cutoff;
	grid_screen_height -= screen_bottom_cutoff;
	screen_x += screen_left_cutoff;
	screen_y += screen_bottom_cutoff;
	int screen_right_cutoff = (grid_screen_width > view.window_width - screen_x) ?
	                           grid_screen_width - (view.window_width - screen_x) : 0;
	int screen_top_cutoff = (grid_screen_height > view.window_height - screen_y) ?
	                           grid_screen_height - (view.window_height - screen_y) : 0;
	grid_screen_width -= screen_right_cutoff;
	grid_screen_height -= screen_top_cutoff;

	/* Find how many grid pixels each of the screen pixel cutoffs correspond to */
	float invzoom_x = 1.0f / view.zoom_x;
	float invzoom_y = 1.0f / view.zoom_y;
	int grid_left_cutoff = screen_left_cutoff * invzoom_x;
	int grid_right_cutoff = screen_right_cutoff * invzoom_x;
	int grid_bottom_cutoff = screen_bottom_cutoff * invzoom_y;
	int grid_top_cutoff = screen_top_cutoff * invzoom_y;

	int grid_cropped_width = grid.width - grid_left_cutoff - grid_right_cutoff;
	int grid_cropped_height = grid.height - grid_bottom_cutoff - grid_top_cutoff;

	/* Draw the grid on the screen. */
	DEBUG2("Drawing a %d by %d rectangle of pixels on the screen, zoom %f in the x\n"
		 "direction and %f in the y direction, offset (%d, %d) from grid cell (0, 0)\n", 
		 grid_cropped_width, grid_cropped_height, view.zoom_x, view.zoom_y, 
		 grid_bottom_cutoff, grid_left_cutoff);

	/* set the raster position to the lower left corner of the clipped grid */
	glRasterPos2i(screen_x, screen_y);

	/* Give OpenGL the actual width of the grid */
	glPixelStorei(GL_UNPACK_SKIP_ROWS, grid_bottom_cutoff);
	glPixelStorei(GL_UNPACK_SKIP_PIXELS, grid_left_cutoff);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, grid.width);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	/* Set the zoom to draw the grid at */
	glPixelZoom(view.zoom_x, view.zoom_y);

	check_gl_error();

	*width_ret = grid_cropped_width;
	*height_ret = grid_cropped_height;
	*y_ret = grid_bottom_cutoff;
	*x_ret = grid_left_cutoff;
}

void special_keyboard_cb(int key, int x, int y)
{
	DEBUG2("Pressed key 0x%x at (%d, %d)\n", key, x, y);

	int modifiers = glutGetModifiers();
	bool ctrl = (modifiers & GLUT_ACTIVE_CTRL) != 0;

	switch (key) {
	case GLUT_KEY_F11:
		toggle_fullscreen();
		break;
	case GLUT_KEY_LEFT:
		view.translate_x -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		break;
	case GLUT_KEY_RIGHT:
		view.translate_x += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		break;
	case GLUT_KEY_UP:
		view.translate_y += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case GLUT_KEY_DOWN:
		view.translate_y -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case GLUT_KEY_HOME:
		view.translate_x -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case GLUT_KEY_END:
		view.translate_x -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case GLUT_KEY_PAGE_UP:
		view.translate_x += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case GLUT_KEY_PAGE_DOWN:
		view.translate_x += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case GLUT_KEY_INSERT:
		reset_view();
		break;
	}
}

void keyboard_cb(unsigned char key, int x, int y)
{
	DEBUG2("Pressed key 0x%x at (%d, %d)\n", key, x, y);

	int modifiers = glutGetModifiers();
	bool ctrl = (modifiers & GLUT_ACTIVE_CTRL) != 0;

	switch(key) {

	case 'f':
		toggle_fullscreen();
		break;
	case '0':
		reset_view();
		break;
	case ' ':
		compute_params.generations_per_redraw *= 1.50;
		if (compute_params.generations_per_redraw > MAX_GENERATIONS_PER_REDRAW)
			compute_params.generations_per_redraw = MAX_GENERATIONS_PER_REDRAW;
		printf("%f generations per redraw\n", compute_params.generations_per_redraw);
		break;
	case 'b':
		compute_params.generations_per_redraw *= 0.66;
		printf("%f generations per redraw\n", compute_params.generations_per_redraw);
		break;
	case '4':
	case 'h':
		view.translate_x -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		break;
	case '6':
	case 'l':
		view.translate_x += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		break;
	case '8':
	case 'k':
		view.translate_y += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case '2':
	case 'j':
		view.translate_y -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case '7':
		view.translate_x -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case '1':
		view.translate_x -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case '9':
		view.translate_x += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case '3':
		view.translate_x += (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_x;
		view.translate_y -= (ctrl ? SCROLL_SPEED_CTRL : SCROLL_SPEED) / view.zoom_y;
		break;
	case '+':
	case 'z':
		zoom(1.1f, 1.1f);
		break;
	case '-':
	case 'Z':
		zoom(0.9f, 0.9f);
		break;
	case 27:
		exit(0);
		break;
	}
}

void mouse_cb(int button, int state, int x, int y)
{
	DEBUG2("mouse_cb(button=%d, state=%d, x=%d, y=%d)\n", button, state, x, y);
	y = view.window_height - y;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float tx = (float)x / (float)view.window_width - 0.5f;
		float ty = (float)y / (float)view.window_height - 0.5f;
		view.translate_x += tx * view.window_width / view.zoom_x;
		view.translate_y += ty * view.window_height / view.zoom_y;
	}
}

void motion_cb(int x, int y)
{
}

void mousewheel_cb(int button, int dir, int x, int y)
{
	DEBUG2("mousewheel_cb(button=%d, dir=%d, x=%d, y=%d\n", button, dir, x, y);

	int modifiers = glutGetModifiers();
	bool ctrl = (modifiers & GLUT_ACTIVE_CTRL) != 0;
	float factor;
	if (dir > 0) {
		factor = ctrl ? 1.5f : 1.2f;
	} else {
		factor = ctrl ? 0.6f : 0.8f;
	}
	zoom(factor, factor);
}

void reshape_cb(int width, int height)
{
	DEBUG("Changed window size to %d by %d pixels\n", width, height);
	view.old_window_width = view.window_width;
	view.old_window_height = view.window_height;
	view.window_width = width;
	view.window_height = height;

	float bigger_factor_x = (float)view.window_width / (float)view.old_window_width;
	float bigger_factor_y = (float)view.window_width / (float)view.old_window_width;
	float min_bigger_factor = min(bigger_factor_x, bigger_factor_y);

	zoom(min_bigger_factor, min_bigger_factor);

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, (GLdouble)view.window_width, 0.0, (GLdouble)view.window_height, 0.0, 1.0);
}


void toggle_fullscreen()
{
	DEBUG("Toggling fullscreen\n");
	if (!view.fullscreen) {
		glutFullScreen();
		view.fullscreen = true;
	} else {
		glutReshapeWindow(view.old_window_width, view.old_window_height);
		view.fullscreen = false;
	}
}

void reset_view()
{
	DEBUG("Resetting view\n");
	view.translate_x = 0.0f;
	view.translate_y = 0.0f;

	float zoom_x_full_window_width = (float)view.window_width / (float)grid.width;
	float zoom_y_full_window_height = (float)view.window_height / (float)grid.height;

	float min_zoom = min(zoom_x_full_window_width, zoom_y_full_window_height);
	view.zoom_x = min_zoom;
	view.zoom_y = min_zoom;
}

static void zoom(float x_factor, float y_factor)
{
	view.zoom_x *= x_factor;
	view.zoom_y *= y_factor;
}
