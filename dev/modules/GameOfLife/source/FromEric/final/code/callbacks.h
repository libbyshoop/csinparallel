#ifndef _CALLBACKS_H
#define _CALLBACKS_H

void display_cb();
void keyboard_cb(unsigned char key, int x, int y);
void special_keyboard_cb(int key, int x, int y);
void mouse_cb(int button, int state, int x, int y);
void motion_cb(int x, int y);
void mousewheel_cb(int button, int dir, int x, int y);
void reshape_cb(int width, int height);
void reset_view();
void toggle_fullscreen();

#define SCROLL_SPEED 15
#define SCROLL_SPEED_CTRL (3.0 * SCROLL_SPEED)
#define MAX_GENERATIONS_PER_REDRAW 500

#endif
