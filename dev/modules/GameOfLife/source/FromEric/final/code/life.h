#ifndef _LIFE_H
#define _LIFE_H

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <inttypes.h>

struct GridInfo {
	int which_buf;
	unsigned height;
	unsigned width;
};


struct View {
	float zoom_x;
	float zoom_y;
	float translate_x;
	float translate_y;
	int window_width;
	int window_height;
	int old_window_width;
	int old_window_height;
	bool fullscreen;
};

struct ComputeParams {
	unsigned long generation;
	double generations_per_redraw;
	double real_gen;
};

extern struct GridInfo grid;
extern struct View view;
extern struct ComputeParams compute_params;


#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern 
#endif

EXTERN void* xmalloc(size_t size);
EXTERN void* zalloc(size_t size);
EXTERN void check_gl_error();
EXTERN void add_R_pentonimo(uint8_t* buf, unsigned width, unsigned height);

#define __DEBUG(format, ...) do { \
	printf(format, ## __VA_ARGS__); \
	/*fflush(stdout);*/ \
} while (0)

#ifdef NO_DEBUG
#define DEBUG(...)
#else
#define DEBUG __DEBUG
#endif

#ifdef MORE_DEBUG
#define DEBUG2 __DEBUG
#else
#define DEBUG2(...)
#endif

#define min(a, b) ({typeof(a) _a = (a); typeof(b) _b = (b); (_a < _b) ? _a : _b; })
#define max(a, b) ({typeof(a) _a = (a); typeof(b) _b = (b); (_a > _b) ? _a : _b; })

#define swap(a, b) ({ 	typeof(a) tmp = a;  \
				a = b; \
				b = tmp;   \
			})

#endif /* _LIFE_H */
