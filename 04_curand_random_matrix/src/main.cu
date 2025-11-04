#include "cuda_timer.cuh"
#include "random_matrix.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>

int main() {
  const size_t rows = 8;
  const size_t cols = 8;
  printf("Generating random matrix of size %zux%zu\n\n", rows, cols);

  unsigned long seed = time(NULL);
  const int total_size = rows * cols;
  const size_t bytes = total_size * sizeof(int);
  int *h_matrix = (int *)malloc(bytes);
  int *d_matrix;
  curandState *d_states;

  cudaMalloc(&d_matrix, bytes);
  cudaMalloc(&d_states, total_size * sizeof(curandState));

  int threads_per_blocks = 256;
  int blocks_per_grid =
      (total_size + threads_per_blocks - 1) / threads_per_blocks;

  {
    CudaTimer timer("Initializing cuRAND states");
    init_curand_states<<<blocks_per_grid, threads_per_blocks>>>(d_states, seed,
                                                                rows, cols);
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Generating random matrix on GPU");
    generate_random_matrix_int<<<blocks_per_grid, threads_per_blocks>>>(
        d_states, d_matrix, rows, cols, 100);
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Copying matrix from device to host");
    cudaMemcpy(h_matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);
  }

  printf("\n");
  print_matrix(h_matrix, rows, cols, 8);

  cudaFree(d_matrix);
  cudaFree(d_states);
  free(h_matrix);
  return 0;
}
