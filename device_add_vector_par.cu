#include <stdio.h>

__global__ void device_add(int *a, int *b, int *res) 
{
  res[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 8

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
  
  cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

  device_add<<<N, 1>>>(dev_a, dev_b, dev_res);
  
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


