#include "life.h"
#include "compute.h"
#include <string.h>
#include <omp.h>
#include <GL/glew.h>

static void compute_next_generation(uint8_t* grid_in, uint8_t* grid_out, int width, int height);

static uint8_t *bufs[2] = {NULL, NULL};


void init_parallel_component()
{
	DEBUG("Beginning OpenMP Life simulation\n");
}

void create_grid(unsigned width, unsigned height, uint8_t* data)
{
	grid.which_buf = 0;
	grid.width = width;
	grid.height = height;
	size_t size = (size_t)width * (size_t)height;

	DEBUG("Creating %u by %u grid \n", width, height);
	DEBUG("Allocating %d buffer%s of size %lu\n", data ? 1 : 2, data ? "" : "s", (unsigned long)size);

	if (data) {
		bufs[0] = data;
	} else {
		bufs[0] = zalloc(size);
		add_R_pentonimo(bufs[0], width, height);
	}
	bufs[1] = zalloc(size);
}

void cleanup()
{
	if (bufs[0])
		free(bufs[0]);
	
	if (bufs[1])
		free(bufs[1]);
}

void draw(int x, int y, int width, int height)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glPixelTransferf(GL_GREEN_SCALE, 255.0f);
	glDrawPixels(width, height, GL_GREEN, GL_UNSIGNED_BYTE, (GLvoid*)bufs[grid.which_buf]);

	/*float red[2]   = {0.0f, 1.0f};*/
	/*float green[2] = {0.0f, 1.0f};*/
	/*float blue[2]  = {0.0f, 0.0f};*/
	/*float alpha[2] = {0.0f, 0.0f};*/
	/*glPixelMapfv(GL_PIXEL_MAP_I_TO_R, 2, red);*/
	/*glPixelMapfv(GL_PIXEL_MAP_I_TO_G, 2, green);*/
	/*glPixelMapfv(GL_PIXEL_MAP_I_TO_B, 2, blue);*/
	/*glPixelMapfv(GL_PIXEL_MAP_I_TO_A, 2, alpha);*/
	/*glDrawPixels(width, height, GL_COLOR_INDEX, GL_UNSIGNED_BYTE, (GLvoid*)bufs[grid.which_buf]);*/


	check_gl_error();
}

/* 
 * Advance the simulation by <n> generations.
 */
void advance_generations(unsigned long n)
{
#pragma omp parallel
	{
		int which_buf = grid.which_buf;
		int my_rank = omp_get_thread_num();
		int num_threads = omp_get_num_threads();

		int width = grid.width;
		int height = grid.height - 2;

		int remainder = height % num_threads;
		int min_height_per_thread = height / num_threads;


		int my_height, my_start;
		if (my_rank < remainder) {
			my_height = min_height_per_thread + 1;
			my_start = min_height_per_thread * my_rank + my_rank;
		} else {
			my_height = min_height_per_thread;
			my_start = min_height_per_thread * my_rank + remainder;
		}

		unsigned char* buf1 = bufs[which_buf] + width * my_start + width;
		unsigned char* buf2 = bufs[!which_buf] + width * my_start + width;

		unsigned long i;
		for (i = 0; i < n; i++) {

			compute_next_generation(buf1, buf2, width, my_height);

			swap(buf1, buf2);

			#pragma omp barrier
		}
	}
	grid.which_buf ^= (n & 1);
}


#define cell(x, y) grid_in[(x) + (y) * (width)]
#define cell_out(x, y) grid_out[(x) + (y) * (width)]
static void compute_next_generation(uint8_t* grid_in, uint8_t* grid_out, 
									int width, int height)
{
	int y, x;
	uint8_t num_neighbors;
	for (y = 0; y < height; y++) {
		for (x = 1; x < width - 1; x++) {
			num_neighbors =
			    cell(x - 1, y + 1) + cell(x, y + 1) + cell(x + 1, y + 1)
			  + cell(x - 1, y    )                  + cell(x + 1, y    )
			  + cell(x - 1, y - 1) + cell(x, y - 1) + cell(x + 1, y - 1);

			if ((num_neighbors == 3) || (cell(x, y) && num_neighbors == 2)) {
				cell_out(x, y) = 1;
			} else {
				cell_out(x, y) = 0;
			}
		}
	}
}

/*static void compute_next_generation(uint8_t* grid_in, uint8_t* grid_out, unsigned width, unsigned height)*/
/*{*/
	/*unsigned x, y, num_neighbors, neighboring_total;*/

	/*unsigned w_neigh, c_neigh, e_neigh;*/

	/*memset(grid_out, 0, width * height);*/

	/*for (y = 0; y < height; y++) {*/


		/*w_neigh = 0;*/
		/*c_neigh = cell(1, y + 1) + cell(1, y) + cell(1, y - 1);*/
		/*e_neigh = cell(2, y + 1) + cell(2, y) + cell(2, y - 1);*/

		/*neighboring_total = w_neigh + c_neigh + e_neigh;*/

		/*x = 1;*/
		/*while (x < width - 1) {*/

			/*num_neighbors = (neighboring_total + 8) >> 8;*/

			/*if ((num_neighbors == 3) || (cell(x, y) && num_neighbors == 4)) {*/
				/*cell_out(x, y) = 0xff;*/
			/*} */
			/*[>else {<]*/
				/*[>cell_out(x, y) = 0;<]*/
			/*[>}<]*/

			/*x++;*/

			/*neighboring_total -= w_neigh;*/

			/*w_neigh = c_neigh;*/
			/*c_neigh = e_neigh;*/
			/*e_neigh = cell(x + 1, y + 1) + cell(x + 1, y) + cell(x + 1, y - 1);*/

			/*neighboring_total += e_neigh;*/

		/*}*/
	/*}*/
/*}*/
