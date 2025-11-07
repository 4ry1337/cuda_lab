#include "cuda_timer.cuh"
#include "random_matrix.cuh"
#include "utils.cuh"
#include <ctime>

int main() {
  const size_t rows = 8;
  const size_t cols = 8;
  const int total_size = rows * cols;
  printf("Generating random matrix of size %zux%zu\n\n", rows, cols);

  // pinned host malloc
  int *h_a;
  cudaMallocHost(&h_a, total_size * sizeof(int));
  cuda_error("cudaMallocHost h_a");

  float *h_b;
  cudaMallocHost(&h_b, total_size * sizeof(float));
  cuda_error("cudaMallocHost h_b");

  // device malloc
  int *d_a;
  cudaMalloc(&d_a, total_size * sizeof(int));
  cuda_error("cudaMallocHost d_a");

  float *d_b;
  cudaMalloc(&d_b, total_size * sizeof(float));
  cuda_error("cudaMallocHost d_b");

  // random matrix
  const int num_states = 1024;
  unsigned long seed = time(NULL);
  curandState *d_states;
  cudaMalloc(&d_states, total_size * sizeof(curandState));
  cuda_error("cudaMallocHost d_states");

  {
    CudaTimer timer("Initializing cuRAND states");
    init_curand_states<<<CEIL_DIV(total_size, 256), 256>>>(d_states, seed,
                                                           num_states);
    cuda_error("FAILED: Initializing cuRAND states: kernel launch");
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Generating random int matrix d_a");
    random_matrix<int>
        <<<dim3{CEIL_DIV(rows, 32), CEIL_DIV(cols, 32)}, dim3{32, 32}>>>(
            num_states, d_states, d_a, rows, cols, 100);
    cuda_error("FAILED: Generating random int matrix d_a: kernel launch");
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Generating random float matrix d_b");
    random_matrix<float>
        <<<dim3{CEIL_DIV(rows, 32), CEIL_DIV(cols, 32)}, dim3{32, 32}>>>(
            num_states, d_states, d_b, rows, cols, 10.0);
    cuda_error("FAILED: Generating random float matrix d_b: kernel launch");
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Copying int matrix (d_a to h_a)");
    cudaMemcpy(h_a, d_a, total_size * sizeof(int), cudaMemcpyDeviceToHost);
    cuda_error("FAILED: Copying int matrix (d_a to h_a): cudaMemcpy");
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Copying float matrix (d_b to h_b)");
    cudaMemcpy(h_b, d_b, total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cuda_error("FAILED: Copying float matrix (d_b to h_b): cudaMemcpy");
    cudaDeviceSynchronize();
  }

  printf("\n");
  print_matrix<int>(h_a, rows, cols, 8);
  printf("\n");
  print_matrix<float>(h_b, rows, cols, 8);

  cudaFree(d_a);
  cudaFree(d_states);
  cudaFreeHost(h_a);
  return 0;
}
