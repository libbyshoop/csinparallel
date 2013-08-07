#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string.h>
#include "life.h"

extern void launch_kernel(const unsigned char* grid_in, unsigned char* grid_out, 
		unsigned width, unsigned height);

static void check_cuda_error();
/*static void device_query();*/

static GLuint pixel_buffer_objects[2];
static struct cudaGraphicsResource* cuda_graphics_resources[2];


void init_parallel_component()
{
	/*DEBUG("Looking for CUDA devices\n");*/
	/*device_query();*/
	DEBUG("Setting CUDA GL device\n");
	cudaError_t cuda_err = cudaGLSetGLDevice(0);
	check_cuda_error();
}

void create_grid(unsigned width, unsigned height, uint8_t* data)
{
	grid.which_buf = 0;
	grid.width = width;
	grid.height = height;
	size_t size = (size_t)width * (size_t)height;

	DEBUG("Creating %u by %u grid \n", width, height);

	uint8_t* buf;
	if (data) {
		buf = data;
	} else {
		DEBUG("Allocating host buffer of size %lu\n", (unsigned long)size);
		buf = (uint8_t*)zalloc(size);
		add_R_pentonimo(buf, width, height);
	}


	DEBUG("Generating OpenGL buffer object names\n");

	glGenBuffers(2, &pixel_buffer_objects[0]);

	DEBUG("Binding and initializing OpenGL buffer objects\n");
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_objects[0]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, buf, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_objects[1]);
	memset(buf, 0, size);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size, buf, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


	DEBUG("Registering OpenGL buffer objects with CUDA\n");
	cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources[0], 
						pixel_buffer_objects[0], 
						cudaGraphicsMapFlagsNone);

	cudaGraphicsGLRegisterBuffer(&cuda_graphics_resources[1], 
						pixel_buffer_objects[1], 
						cudaGraphicsMapFlagsNone);

	check_cuda_error();

	DEBUG("Freeing host buffer\n");
	free(buf);
}

void cleanup()
{
	cudaGraphicsUnregisterResource(cuda_graphics_resources[0]);
	cudaGraphicsUnregisterResource(cuda_graphics_resources[1]);
	glDeleteBuffers(2, pixel_buffer_objects);
}

void draw(int x, int y, int width, int height)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_objects[grid.which_buf]);
	glPixelTransferf(GL_GREEN_SCALE, 255.0f);
	glDrawPixels(width, height, GL_GREEN, GL_UNSIGNED_BYTE, (GLvoid*)0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	check_gl_error();
}

/* 
 * Advance the simulation by <n> generations by mapping the OpenGL pixel buffer
 * objects for writing from CUDA, executing the kernel <n> times, and unmapping
 * the pixel buffer object.
 */
void advance_generations(unsigned long n)
{
	uint8_t* device_bufs[2];
	size_t size;

	DEBUG2("Mapping CUDA resources and retrieving device buffer pointers\n");
	cudaGraphicsMapResources(2, cuda_graphics_resources, (cudaStream_t)0);

	cudaGraphicsResourceGetMappedPointer((void**)&device_bufs[0], &size, 
								cuda_graphics_resources[0]);

	cudaGraphicsResourceGetMappedPointer((void**)&device_bufs[1], &size, 
								cuda_graphics_resources[1]);

	check_cuda_error();

	while (n--) {

		DEBUG2("Launching kernel (grid.width = %u, grid.height = %u)\n",
				grid.width, grid.height);

		launch_kernel(device_bufs[grid.which_buf], device_bufs[!grid.which_buf], 
									grid.width, grid.height);

		grid.which_buf ^= 1;
	}

	DEBUG2("Unmapping CUDA resources\n");

	cudaGraphicsUnmapResources(2, cuda_graphics_resources, (cudaStream_t)0);
	cudaStreamSynchronize(0);
}





static void check_cuda_error()
{
	cudaError_t cuda_err = cudaGetLastError();
	if (cuda_err != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_err));		
		exit(1);
	}
}


/* From BCCD example */
#if 0
static void device_query()
{
	int deviceCount = 0, device;
	cudaError_t status = (cudaError_t)0; 
	struct cudaDeviceProp deviceProperties;
	int driverVersion = 0, runtimeVersion = 0;	 

	if ((status = cudaGetDeviceCount(&deviceCount)) != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount() FAILED, status = %d (%s)\n", 
							status, cudaGetErrorString(status));
		exit(1); 
	}
	
	if (deviceCount == 0) { 
		printf("There are no hardware devices which support CUDA\n");
	} else {
		printf("There %s %d CUDA capable hardware device%s\n", 
			deviceCount == 1 ? "is" : "are", deviceCount, deviceCount > 1 ? "s" : ""); 
	} 
	
	if ((status = cudaDriverGetVersion(&driverVersion)) != cudaSuccess) {
		fprintf(stderr, "cudaDriverGetVersion() FAILED, status = %d (%s)\n", 
							status, cudaGetErrorString(status));
		exit(1); 
	} else {
		printf("CUDA driver version: %d.%d\n", driverVersion / 1000, driverVersion % 100);
	}

	if ((status = cudaRuntimeGetVersion(&runtimeVersion)) != cudaSuccess) {
		fprintf(stderr, "cudaRuntimeGetVersion() FAILED, status = %d (%s)\n", 
								status, cudaGetErrorString(status));
		exit(1); 
	} else {
		printf("CUDA runtime version: %d.%d\n", runtimeVersion / 1000, runtimeVersion % 100);
	}

	for (device = 0; device < deviceCount; ++device) {
	
		if ((status = cudaGetDeviceProperties(&deviceProperties, device)) != cudaSuccess) {
			fprintf(stderr, "cudaGetDeviceProperties() FAILED, status = %d (%s)\n", 
									status, cudaGetErrorString(status));
			exit(1); 
		}

		printf("Device %d:\n", device); 
		printf("\tname = %s\n", deviceProperties.name);
		printf("\tCUDA capability major.minor version = %d.%d\n", 
					deviceProperties.major, deviceProperties.minor);
		printf("\tmultiProcessorCount = %d\n", deviceProperties.multiProcessorCount);

		printf("\ttotalGlobalMem = %ld bytes\n", (long)deviceProperties.totalGlobalMem); 
		printf("\tsharedMemPerBlock = %d bytes\n", (int)deviceProperties.sharedMemPerBlock);
		printf("\tregsPerBlock = %d\n", deviceProperties.regsPerBlock);
		printf("\twarpSize = %d\n", deviceProperties.warpSize);
		printf("\tmemPitch = %d bytes\n", (int)deviceProperties.memPitch);
		printf("\tmaxThreadsPerBlock = %d\n", deviceProperties.maxThreadsPerBlock);
		printf("\tmaxThreadsDim = %d x %d x %d\n", deviceProperties.maxThreadsDim[0], 
		  deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
		printf("\tmaxGridSize = %d x %d x %d\n", deviceProperties.maxGridSize[0], 
		  deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
		printf("\n");	
		printf("\tmemPitch = %ld bytes\n", (long)deviceProperties.memPitch);
		printf("\ttextureAlignment = %ld bytes\n", (long)deviceProperties.textureAlignment);
		printf("\tclockRate = %.2f GHz\n", deviceProperties.clockRate * 1e-6f);
	
	#if CUDART_VERSION >= 2000
		printf("\tdeviceOverlap = %s\n", deviceProperties.deviceOverlap ? "Yes" : "No");
	#endif

	#if CUDART_VERSION >= 2020
		printf("\tkernelExecTimeoutEnabled = %s\n", deviceProperties.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("\tintegrated = %s\n", deviceProperties.integrated ? "Yes" : "No");
		printf("\tcanMapHostMemory = %s\n", deviceProperties.canMapHostMemory ? "Yes" : "No");
		printf("\tcomputeMode = %s\n", deviceProperties.computeMode == cudaComputeModeDefault ?
		  "Default (multiple host threads can use this device simultaneously)" :
		  deviceProperties.computeMode == cudaComputeModeExclusive ?
		  "Exclusive (only one host thread at a time can use this device)" :
		  deviceProperties.computeMode == cudaComputeModeProhibited ?
		  "Prohibited (no host thread can use this device)" :
		  "Unknown");
	#endif
	
	#if CUDART_VERSION >= 3000
		printf("\tconcurrentKernels = %s\n", deviceProperties.concurrentKernels ? "Yes" : "No");
	#endif
	
	#if CUDART_VERSION >= 3010
		printf("\tECCEnabled = %s\n", deviceProperties.ECCEnabled ? "Yes" : "No");
	#endif
	
	#if CUDART_VERSION >= 3020
		printf("\ttccDriver = %s\n", deviceProperties.tccDriver ? "Yes" : "No");
	#endif	
	
		printf("\n"); 
	}
}
#endif
