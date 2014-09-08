/*
 * Program to demonstrate pthreads (doesn't work yet)
 *
 * Compile with: gcc -Wall -std=c99 -o helloThread helloThread.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void* hello(void* arg) {
  //function to run as thread body

  printf("Hello from a thread\n");
  return NULL;
}

void failUnless0(int cond) {   //fail if argument is true
  if (cond) {
    perror("");
    exit(1);
  }
}

int main() {
  pthread_t t1;    //data structures to store threads
  pthread_t t2;
  failUnless0(pthread_create(&t1, NULL, hello, NULL));  //create threads
  failUnless0(pthread_create(&t2, NULL, hello, NULL));

  printf("Program done\n");
}
