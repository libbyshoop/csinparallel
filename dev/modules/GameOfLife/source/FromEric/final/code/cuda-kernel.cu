
#define cell(x, y) (grid_in[(x) + (y) * (width)])
#define cell_out(x, y) (grid_out[(x) + (y) * (width)])



static __global__ void kernel(const unsigned char* grid_in, 
	unsigned char* grid_out, unsigned width, unsigned height)
{
	unsigned block_start_x = blockIdx.x * blockDim.x;
	unsigned block_start_y = blockIdx.y * blockDim.y;

	unsigned x = block_start_x + threadIdx.x;
	unsigned y = block_start_y + threadIdx.y;

	if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {

		unsigned char num_neighbors = 
		    cell(x - 1, y + 1) + cell(x, y + 1) + cell(x + 1, y + 1)
		  + cell(x - 1, y    )                  + cell(x + 1, y    )
		  + cell(x - 1, y - 1) + cell(x, y - 1) + cell(x + 1, y - 1);


		cell_out(x, y)= ((num_neighbors == 3) || cell(x, y) && num_neighbors == 2);
	}
}

extern "C"
{

void launch_kernel(const unsigned char* grid_in, unsigned char* grid_out, 
		unsigned width, unsigned height)
{
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	kernel<<< grid, block>>>(grid_in, grid_out, width, height);
}

}



/* Junk */


#if 0
#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 16

#define DATA_BLOCK_DIM_X (BLOCK_DIM_X + 2 * sizeof(unsigned))
#define DATA_BLOCK_DIM_Y (BLOCK_DIM_Y + 2 * sizeof(unsigned))

#define DATA_BLOCK_WORDS_X (DATA_BLOCK_DIM_X / sizeof(unsigned))
#define DATA_BLOCK_WORDS_Y (DATA_BLOCK_DIM_Y / sizeof(unsigned))

#define cell(x, y) (data_block[y][x])

#define cell_out(x, y) (grid_out[(x) + (y) * (width)])

static __global__ void kernel(const unsigned char* grid_in, 
	unsigned char* grid_out, unsigned width, unsigned height)
{
	const unsigned block_start_x = blockIdx.x * (BLOCK_DIM_X / 2);
	const unsigned block_start_y = blockIdx.y * (BLOCK_DIM_Y / 2);

	unsigned grid_x = block_start_x + threadIdx.x;
	unsigned grid_y = block_start_y + threadIdx.y;


	__shared__ unsigned char data_block[BLOCK_DIM_Y][BLOCK_DIM_X];

	data_block[threadIdx.y][threadIdx.x] = grid_in[grid_x + grid_y * width];

	__syncthreads();

	unsigned x = threadIdx.x;
	unsigned y = threadIdx.y;

	if (x > 0 && x < BLOCK_DIM_X - 1 && y > 0 && y < BLOCK_DIM_Y) {

		unsigned char num_neighbors = 
		    cell(x - 1, y + 1) + cell(x, y + 1) + cell(x + 1, y + 1)
		  + cell(x - 1, y    )                  + cell(x + 1, y    )
		  + cell(x - 1, y - 1) + cell(x, y - 1) + cell(x + 1, y - 1);


		cell_out(grid_x, grid_y)= ((num_neighbors == 3) || cell(x, y) && num_neighbors == 2);
	}

	/*data_block[threadIdx.x + 1][threadIdx.y + 1] = cell(x, y);*/

	/*if (threadIdx.x == 0) { [> Left side w/ corners <]*/

		/*data_block[0][threadIdx.y + 1] = cell(x - 1, y);*/

		/*if (threadIdx.y == 0) { [> Bottom-left corner <]*/
			/*data_block[0][0] = cell(x - 1, y - 1);*/
		/*} else if (threadIdx.y == BLOCK_DIM_Y - 1) { [> Top-left corner <]*/
			/*data_block[0][BLOCK_DIM_Y + 1] = cell(x - 1, y + 1);*/
		/*}*/
	/*} else if (threadIdx.x == BLOCK_DIM_X - 1) { [> Right side w /corners <]*/
	/*}*/

	/*if (threadIdx.y == 0) [> Bottom <]*/
		/*data_block[threadIdx.x + 1][0] = cell(x, y - 1);*/
	/*else if (threadIdx.y == 0) [> Top <]*/
		/*data_block[threadIdx.x + 1][BLOCK_DIM_Y + 1] = cell(x, y - 1);*/

	/*__syncthreads();*/


	/*if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {*/
		/*unsigned char num_neighbors = */
		    /*cell(x - 1, y + 1) + cell(x, y + 1) + cell(x + 1, y + 1)*/
		  /*+ cell(x - 1, y    )                  + cell(x + 1, y    )*/
		  /*+ cell(x - 1, y - 1) + cell(x, y - 1) + cell(x + 1, y - 1);*/


		/*cell_out(x, y) = ((num_neighbors == 3) || cell(x, y) &&*/
		/*num_neighbors == 2);*/
	/*}*/
}

/*static __global__ void zero_memory_kernel(unsigned long* grid_out, unsigned long size)*/
/*{*/
	/*unsigned long x = blockIdx.x * blockDim.x + threadIdx.x;*/
	/*if (x < size) {*/
		/*grid_out[x] = 0;*/
	/*}*/
/*}*/
#endif
