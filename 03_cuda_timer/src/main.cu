#include "cuda_timer.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  const int N = 1024 * 1024;
  const size_t bytes = N * sizeof(int);

  int *h_a = (int *)malloc(bytes);

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_a[i] = (int)rand() / RAND_MAX * 100;
  }

  int *d_a;

  {
    CudaTimer timer("Allocating device mermory");
    cudaMalloc(&d_a, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  }

  cudaFree(d_a);
  free(h_a);
  return 0;
}
