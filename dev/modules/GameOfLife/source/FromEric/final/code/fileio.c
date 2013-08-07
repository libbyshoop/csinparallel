#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "fileio.h"
#include "life.h"




static unsigned char* read_lif_file(const char* filename, unsigned* width_ret, unsigned* height_ret,
	unsigned width_override, unsigned height_override);


static void lif_file_get_grid_size(char* file_buf, size_t size, 
						unsigned* width_ret, unsigned* height_ret);

static unsigned char* read_rle_file(const char* filename, unsigned* width_ret, unsigned* height_ret,
	unsigned width_override, unsigned height_override);

static void rle_file_fill_grid(FILE* file, unsigned char* grid, unsigned width, unsigned height);

static uint8_t* file_get_contents(const char* filename, size_t* size_ret);

static size_t file_get_size(const char* filename);

static FILE* xfopen(const char* filename, const char* mode);

static void round_width_and_height(int* width, int* height);

unsigned char* read_file(const char* filename, unsigned* width_ret, unsigned* height_ret,
	unsigned width_override, unsigned height_override)
{
	char *p, *file_suffix;
	
	p = strrchr(filename, '.');
	
	if (p == NULL)
		goto invalid_suffix;

	file_suffix = p + 1;

	if (strcmp(file_suffix, "lif") == 0) {
		return read_lif_file(filename, width_ret, height_ret, width_override, height_override);
	}

	if (strcmp(file_suffix, "rle") == 0) {
		return read_rle_file(filename, width_ret, height_ret, width_override, height_override);
	}

invalid_suffix:
	fprintf(stderr, "Not sure how to interpret file \"%s\": invalid or nonexistent suffix.\n", filename);
	exit(1);
	return NULL;
}


/* Reads the file <filename> in the .lif format.  Returns a pointer to the grid
 * of cells, allocated on the heap, with the number of rows and number of
 * columns returned.
 *
 * If reading the file fails, an error message is printed and NULL is returned.
 */
uint8_t* read_lif_file(const char* filename, unsigned* width_ret, unsigned* height_ret,
	unsigned width_override, unsigned height_override)
{
	size_t size;
	char* file_buf = (char*)file_get_contents(filename, &size);
	if (file_buf == NULL) {
		return NULL;
	}

	char* p = file_buf - 1;

	DEBUG("Determining size of grid in file \"%s\"\n", filename);

	unsigned width = 0, height = 0;
	if (width_override == 0 && height_override == 0) {
		lif_file_get_grid_size(file_buf, size, &width, &height);
	} else {
		width = width_override;
		height = height_override;
	}


	int center_col = width / 2;
	int center_row = height / 2;

	unsigned char* grid = zalloc((size_t)width * (size_t)height);


	DEBUG("Grid size %d by %d\n", width, height);
	DEBUG("Setting grid cells from data in file \"%s\"\n", filename);
	p = file_buf - 1;
	while ((p = strchr(p + 1, '#')) != NULL) {
		if (*(p + 1) == 'P') {
			int x, y;
			sscanf(p + 3, "%d %d", &x, &y);
			
			char* pc = p + 1;
			int start_col = center_col + x;
			int col = start_col;
			int row = center_row + y;
			while (*pc != '#' && *pc != '\0') {
				switch (*pc) {
				case '\n':
					if (*(pc + 1) == '\r' || *(pc + 1) == '\n') {
						break;
					}
					row++;
					col = start_col;
					break;
				case '.':
					col++;
					break;
				case '*':
					grid[row * width + col] = 1;
					col++;
					break;
				}
				pc++;
			}
		}
	}

	free(file_buf);
	*height_ret = height;
	*width_ret = width;
	return grid;
}


static void lif_file_get_grid_size(char* file_buf, size_t size, 
						unsigned* width_ret, unsigned* height_ret)
{
	int min_x = 1000, max_x = 0, min_y = 1000, max_y = 0;
	char* p = file_buf;
	while ((p = strchr(p + 1, '#')) != NULL) {
		switch (*(p + 1)) {
		case 'R':
			fprintf(stderr, "Error: Cannot interpret file, which requests alternate rules\n");
			free(file_buf);
			return;
		case 'P':
		{
			int x, y;
			sscanf(p + 3, "%d %d", &x, &y);
			min_x = min(x, min_x);
			max_x = max(x + 80, max_x);
			min_y = min(y, min_y);
			int num_lines = 0;
			char* pc = p + 1;
			while (*pc != '#' && *pc != '\0') {
				if (*pc == '\n') {
					num_lines++;
				}
				pc++;
			}
			max_y = max(y + num_lines, max_y);
		}
		}
	}

	*width_ret = max_x - min_x + 30;
	*height_ret = max_y - min_y + 30;

	round_width_and_height((int*)width_ret, (int*)height_ret);
}

unsigned char* read_rle_file(const char* filename, unsigned* width_ret, unsigned* height_ret,
	unsigned width_override, unsigned height_override)
{
	FILE* file = xfopen(filename, "r");
	char line_buf[100];

	do {
		fgets(line_buf, 100, file);
	} while (line_buf[0] == '#');

	int width = 0, height = 0;

	if (width_override == 0 && height_override == 0) {
		sscanf(line_buf, "x = %d, y = %d", &width, &height);
		width = max(width, 256);
		height = max(height, 256);
	} else {
		width = width_override;
		height = height_override;
	}
	round_width_and_height(&width, &height);

	unsigned char* grid = zalloc((size_t)width * (size_t)height);

	rle_file_fill_grid(file, grid, width, height);

	*height_ret = height;
	*width_ret = width;
	return grid;
}

static void rle_file_fill_grid(FILE* file, unsigned char* grid, unsigned width, unsigned height)
{
	char c;
	unsigned run_count = 0;
	unsigned x = 0;
	unsigned y = 0;

	while ((c = fgetc(file)) != EOF) {
		switch (c) {
		case '!':
			return;
		case '0' ... '9':
			run_count = run_count * 10 + (c - '0');
			break;
		case 'b':
			if (run_count == 0) {
				x++;
			} else {
				x += run_count;
				run_count = 0;
			}
			break;
		case 'o':
			memset(grid + x + (y * width), 1, run_count);
			if (run_count == 0) {
				x++;
			} else {
				x += run_count;
				run_count = 0;
			}
			break;
		case '$':
			x = 0;
			if (run_count == 0) {
				y++;
			} else {
				y += run_count;
				run_count = 0;
			}
			break;
		default:
			break;
		}
	}
}

static uint8_t* file_get_contents(const char* filename, size_t* size_ret)
{
	size_t size = file_get_size(filename);
	FILE* file = xfopen(filename, "r");

	DEBUG("Reading file \"%s\" into memory\n", filename);
	uint8_t* file_buf = xmalloc(size);
	size_t bytes_read = fread(file_buf, 1, size, file);

	if (bytes_read != size) {
		fprintf(stderr, "Error: Failed to fully read the file \"%s\"\n", filename);
		fclose(file);
		free(file_buf);
		exit(1);
	}

	fclose(file);
	*size_ret = size;
	return file_buf;
}

static size_t file_get_size(const char* filename)
{
	struct stat stat_buf;
	DEBUG("Reading metadata of file \"%s\"\n", filename);
	if (stat(filename, &stat_buf) != 0) {
		fprintf(stderr, "Error: Failed to read the metadata of file \"%s\": %s\n", 
				filename, strerror(errno));
		exit(1);
	}
	return stat_buf.st_size;
}

static FILE* xfopen(const char* filename, const char* mode)
{
	DEBUG("Opening file \"%s\"\n", filename);
	FILE* file = fopen(filename, mode);
	if (file == NULL) {
		fprintf(stderr, "Error: Failed to open the file \"%s\": %s\n", 
				filename, strerror(errno));
		exit(1);
	}
	return file;
}


static void round_width_and_height(int* width, int* height)
{
	*width += 50;
	*height += 50;
	*width -= (*width % 32);
	*height -= (*height % 32);

	*width = max(*width, 128);
	*height = max(*height, 128);
}
