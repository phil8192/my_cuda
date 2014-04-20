/*
 * use global to indicate function runs on the device. 
 */

#include <stdio.h>

/* cuda c keyword __global__ indicates that a function
 * is run on the device and called from the host.
 *
 * nvcc will split the source into host and device 
 * components. the nvidia compiler will handle functions
 * like kernel(), whilst gcc will handle rest.
 */
__global__ void kernel(void)
{

}

int main(void)
{

  /* "kernel launch"
   * a call from host code to device code
   */
  kernel<<<1,1>>>();

  printf("hi again.\n");
  return 0;
}
