#include "wmr.h"
#include <stdlib.h>
void reducer(char* key, wmr_handle handle) {
  long sum = 0;  /* sum of values encountered so far */
  char *str;  /* to hold string representation of an integer 
		 value */
  char sumstr[20];  /* to hold string representation of sum */
  while (str = wmr_get_val(handle))
    sum += strtol(str, NULL, 10);
  sprintf(sumstr, "%d", sum);
  wmr_emit(key, sumstr);
}
