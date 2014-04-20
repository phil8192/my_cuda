/* threads and blocks.
 * for blocks = 2, threads = 4:
 * a = [0,1,2,3 | 0,1,2,3]
 * index = threadIdx.x + blockIdx.x * threads 
 * = [0,1,2,3 | 4,5,6,7]
 */

#include <stdio.h>

#define N 8
#define THREADS_PER_BLOCK 8
#define BLOCKS (N / THREADS_PER_BLOCK)

__global__ void dot_product(int *a, int *b, int *res)
{
  /* shared memory for threads in a block.
   * only visible to threads running in a block.
   */
  __shared__ int temp[N];
  temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
  
  /* no threads can advance in this block until all have
   * reached this point. otherwise, thread 0 could finish
   * before other threads have finished writting and read
   * garbage from the shared memory.
   */
  __syncthreads();

  if(0 == threadIdx.x) 
  {
    int sum = 0;
    for(int i = 0; i < N; i++)
      sum += temp[i];
    *res = sum;
  }
}

void random_ints(int *arr, int n) 
{
  int i;
  for(i = 0; i < n; i++)
    arr[i] = i; /*rand();*/
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
  cudaMalloc((void**) &dev_res, sizeof(int));

  a = (int*) malloc(size);
  b = (int*) malloc(size);
  res = (int*) malloc(sizeof(int));
 
  random_ints(a, N);
  random_ints(b, N);
  
  /* copy dev_a, dev_b to the device */
  cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

  /* launch device_add kernel with M blocks of N threads. */
  dot_product<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_res);
  
  /* copy the device result (dev_res) back to res (on host) */
  cudaMemcpy(res, dev_res, sizeof(int), cudaMemcpyDeviceToHost);

  printf("result = %i\n", *res);
  
  free(a);
  free(b);
  free(res);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_res);

  return 0;
}


