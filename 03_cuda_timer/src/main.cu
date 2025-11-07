#include "cuda_timer.cuh"
#include <ctime>

void cuda_error(const char *prefix) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", prefix, cudaGetErrorString(err));
    exit(1);
  }
}

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
    cuda_error("cudaMalloc d_a");
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cuda_error("cudaMemcpy d_a to h_a");
  }

  cudaFree(d_a);
  free(h_a);
  return 0;
}
