#include "cuda_timer.cuh"
#include "multiplication.cuh"
#include "random_matrix.cuh"
#include "utils.cuh"

void device_properties(cudaDeviceProp &device, int device_index = 0) {
  cudaGetDeviceProperties(&device, device_index);

  printf("  --- General information for device ---\n");
  printf("Name: %s;\n", device.name);
  printf("Compute capability: %d.%d\n", device.major, device.minor);
  printf("Total global memory: %zu\n", device.totalGlobalMem);
  printf("Total constant memory: %zu\n", device.totalConstMem);
  printf("Multiprocessor count: %d\n", device.multiProcessorCount);
  printf("Shared memory per block: %zu\n", device.sharedMemPerBlock);
  printf("Registers per block: %d\n", device.regsPerBlock);
  printf("Threads in warp: %d\n", device.warpSize);
  printf("Max threads Per Block: %d\n", device.maxThreadsPerBlock);
  printf("Max thread dimensions: (%d, %d, %d)\n", device.maxThreadsDim[0],
         device.maxThreadsDim[1], device.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", device.maxGridSize[0],
         device.maxGridSize[1], device.maxGridSize[2]);
  printf("  --- General information for device ---\n\n");
}

__device__ void matrix_multplication_naive(int *d_a, int *d_b, int *d_out,
                                           size_t M, size_t N, size_t K) {
  size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  size_t y = threadIdx.y + blockIdx.y * blockDim.y;

  if ((y < M) && (x < K)) {
    int val = 0;
    for (size_t i = 0; i < N; i++) {
      val += d_a[y * N + i] * d_b[i * K + x];
    }
    d_out[y * K + x] = val;
  }
}

// sources:
// [1] https://khushi-411.github.io/multidim_grids_and_data/#link2
// [2] https://siboehm.com/articles/22/CUDA-MMM
__global__ void matrix_multplication(int *d_a, int *d_b, int *d_out, size_t M,
                                     size_t N, size_t K) {
  matrix_multplication_naive(d_a, d_b, d_out, M, N, K);
}

void matrix_multiplication_cpu(int *h_a, int *h_b, int *h_out, size_t M,
                               size_t N, size_t K) {
  for (size_t x = 0; x < M; x++) {
    for (size_t y = 0; y < K; y++) {
      int val = 0;
      for (size_t z = 0; z < N; z++) {
        val += h_a[x * N + z] * h_b[z * K + y];
      }
      h_out[x * K + y] = val;
    }
  }
}

bool compare_results(int *gpu_result, int *cpu_result, size_t total_size,
                     int max_errors_to_show = 10) {
  bool results_match = true;
  int errors_shown = 0;

  for (size_t i = 0; i < total_size; i++) {
    if (gpu_result[i] != cpu_result[i]) {
      if (errors_shown < max_errors_to_show) {
        printf("VERIFICATION: Mismatch at index %zu: GPU=%d, CPU=%d\n", i,
               gpu_result[i], cpu_result[i]);
        errors_shown++;
      }
      results_match = false;
    }
  }

  if (results_match) {
    printf("\nVERIFICATION: Results match!\n");
  } else {
    printf("\nVERIFICATION: Results DO NOT match! (%d+ errors found)\n",
           errors_shown);
  }

  return results_match;
}

void wrapper() {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device;
  device_properties(device, device_id);

  // sizes
  // M x N * N x K = M * K
  size_t M = 1024, N = 512 + 256, K = 512;

  // pinned host matrices (for async copy)
  int *h_out;
  cudaMallocHost(&h_out, M * K * sizeof(int));
  cuda_error("cudaMallocHost h_out");

  // device matrices
  int *d_a;
  cudaMalloc(&d_a, M * N * sizeof(int));
  cuda_error("cudaMalloc d_a");

  int *d_b;
  cudaMalloc(&d_b, N * K * sizeof(int));
  cuda_error("cudaMalloc d_b");

  int *d_out;
  cudaMalloc(&d_out, M * K * sizeof(int));
  cuda_error("cudaMalloc d_out");

  // random matrix
  unsigned long seed = 42; // time(NULL);
  const int num_states = 1024;
  curandState *d_states;
  cudaMalloc(&d_states, num_states * sizeof(curandState));
  cuda_error("cudaMalloc d_states");

  // streams
  cudaStream_t streams[2];
  cudaStreamCreate(&streams[0]);
  cuda_error("cudaStreamCreate streams[0]");
  cudaStreamCreate(&streams[1]);
  cuda_error("cudaStreamCreate streams[1]");

  dim3 threads_per_block(32, 32, 1); // 32x32 = 1024 threads per block
  dim3 blocks_per_grid((M + threads_per_block.x - 1) / threads_per_block.x,
                       (N + threads_per_block.y - 1) / threads_per_block.y);

  {
    CudaTimer timer("Initializing cuRAND states");
    int threads_init = 256;
    int blocks_init = (num_states + threads_init - 1) / threads_init;
    init_curand_states<<<blocks_init, threads_init, 0, streams[0]>>>(
        d_states, seed, num_states);
    cudaStreamSynchronize(streams[0]);
  }

  {
    CudaTimer timer("Generating random matrices on GPU");
    generate_random_matrix_int<<<blocks_per_grid, threads_per_block, 0,
                                 streams[0]>>>(d_states, d_a, M, N, 100,
                                               num_states);
    cuda_error("generate_random_matrix_int A");
    generate_random_matrix_int<<<blocks_per_grid, threads_per_block, 0,
                                 streams[1]>>>(d_states, d_b, N, K, 100,
                                               num_states);
    cuda_error("generate_random_matrix_int B");
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  {
    CudaTimer timer("Matrix multiplication on GPU");
    matrix_multplication<<<blocks_per_grid, threads_per_block, 0, streams[0]>>>(
        d_a, d_b, d_out, M, N, K);
    cuda_error("matrix_multplication");
    cudaStreamSynchronize(streams[0]);
  }

  {
    CudaTimer timer("Copying results from device to host");
    cudaMemcpyAsync(h_out, d_out, M * K * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[0]);
    cuda_error("cudaMemcpyAsync d_out to h_out");
    cudaStreamSynchronize(streams[0]);
  }

#ifdef VERIFY_RESULTS
  int *h_a;
  cudaMallocHost(&h_a, M * N * sizeof(int));
  cuda_error("cudaMallocHost h_a");

  int *h_b;
  cudaMallocHost(&h_b, N * K * sizeof(int));
  cuda_error("cudaMallocHost h_b");

  {
    CudaTimer timer("VERIFICATION: Copying matrices from device to host");
    cudaMemcpyAsync(h_a, d_a, M * N * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[0]);
    cuda_error("cudaMemcpyAsync d_a to h_a");
    cudaMemcpyAsync(h_b, d_b, N * K * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[1]);
    cuda_error("cudaMemcpyAsync d_b to h_b");
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  int *result;
  cudaMallocHost(&result, M * K * sizeof(int));
  cuda_error("VERIFICATION: cudaMallocHost result");

  {
    CudaTimer timer("VERIFICATION: Matrix multiplication on CPU");
    matrix_multiplication_cpu(h_a, h_b, result, M, N, K);
  }

  // Compare results
  compare_results(h_out, result, M * K);

  // Cleanup
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(result);
#endif

  // Cleanup
  cudaFreeHost(h_out);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  cudaFree(d_states);
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
}
