/* threads and blocks.
 * for blocks = 2, threads = 4:
 * a = [0,1,2,3 | 0,1,2,3]
 * index = threadIdx.x + blockIdx.x * threads 
 * = [0,1,2,3 | 4,5,6,7]
 */

#include <stdio.h>

/* this kernel uses threads and blocks. the width of a block
 * (number of threads per block) can be accessed with the 
 * built in variable blockDim.x so that the index can be obtained.
 */
__global__ void device_add(int *a, int *b, int *res) 
{
  int idx  = threadIdx.x + blockIdx.x * blockDim.x;
  res[idx] = a[idx] + b[idx];   
}

#define N 8
#define THREADS_PER_BLOCK 4
#define BLOCKS (N / THREADS_PER_BLOCK)

void random_ints(int *arr, int n) 
{
  int i;
  for(i = 0; i < n; i++)
    arr[i] = rand();
}

void print_arr(int *arr, int n) 
{
  int i, last;
  for(i = 0, last = n -1; i < last; i++)
    printf("%i,", arr[i]);
  printf("%i\n", arr[last]);
}

int main(void) 
{
  int *a, *b, *res;
  int *dev_a, *dev_b, *dev_res;
  int size = N * sizeof(int);
  
  cudaMalloc((void**) &dev_a, size);
  cudaMalloc((void**) &dev_b, size);
  cudaMalloc((void**) &dev_res, size);

  a = (int*) malloc(size);
  b = (int*) malloc(size);
  res = (int*) malloc(size);
  
  random_ints(a, N);
  random_ints(b, N);
  
  /* copy dev_a, dev_b to the device */
  cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

  /* launch device_add kernel with M blocks of N threads. */
  device_add<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_res);
  
  /* copy the device result (dev_res) back to res (on host) */
  cudaMemcpy(res, dev_res, size, cudaMemcpyDeviceToHost);

  print_arr(res, N);

  free(a);
  free(b);
  free(res);
  
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_res);

  return 0;
}


