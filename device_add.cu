#include <stdio.h>

/* add a and b on the device.
 * uses pointers because function will run on the device.
 */
__global__ void device_add(int *a, int *b, int *res) 
{
  *res = *a + *b;
}

int main(void)
{
  int a, b, res; /* host copies */
  int *dev_a, *dev_b, *dev_res; /* device copies */
  int size = sizeof(int); /* space needed for an integer */

  /* allocate device copies of a, b, c */
  cudaMalloc((void**)&dev_a, size);
  cudaMalloc((void**)&dev_b, size);
  cudaMalloc((void**)&dev_res, size);

  a = 2;
  b = 7;

  /* copy the inputs to the device */
  cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

  /* launch device_add() on the gpu. */
  device_add<<<1, 1>>>(dev_a, dev_b, dev_res);

  /* copy the device result dev_res, back to the host copy, res. */
  cudaMemcpy(&res, dev_res, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_res);

  printf("result: %i\n", res);

  return 0;
}
