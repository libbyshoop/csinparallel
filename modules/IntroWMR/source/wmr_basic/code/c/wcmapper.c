#include <string.h>
#include "wmr.h"
void mapper(char* key, char* val) {
  while (key != NULL) {
    wmr_emit(strsep(&key, " "), "1");
  }
}
