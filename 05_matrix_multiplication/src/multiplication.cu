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
                                           size_t width) {
  size_t col = threadIdx.x + blockIdx.x * blockDim.x;
  size_t row = threadIdx.y + blockIdx.y * blockDim.y;

  if ((row < width) && (col < width)) {
    int val = 0;
    for (size_t k = 0; k < width; k++) {
      val += d_a[row * width + k] * d_b[k * width + col];
    }
    d_out[row * width + col] = val;
  }
}

__global__ void matrix_multplication(int *d_a, int *d_b, int *d_out,
                                     size_t width, size_t height) {
  matrix_multplication_naive(d_a, d_b, d_out, width);
}

// CPU matrix multiplication for verification
void matrix_multiplication_cpu(int *h_a, int *h_b, int *h_out, size_t width) {
  for (size_t row = 0; row < width; row++) {
    for (size_t col = 0; col < width; col++) {
      int val = 0;
      for (size_t k = 0; k < width; k++) {
        val += h_a[row * width + k] * h_b[k * width + col];
      }
      h_out[row * width + col] = val;
    }
  }
}

// Compare GPU and CPU results
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
    printf("\nnVERIFICATION: Results DO NOT match! (%d+ errors found)\n",
           errors_shown);
  }

  return results_match;
}

void wrapper() {
  cudaDeviceProp device;
  device_properties(device, 0);

  // sizes
  size_t rows = device.maxThreadsDim[0];
  size_t cols = device.maxThreadsDim[1];
  printf("Matrix of size %zux%zu\n\n", rows, cols);
  size_t total_size = rows * cols;
  const size_t bytes = total_size * sizeof(int);

  // pinned host matrices (for async copy)
  int *h_out;
  cudaMallocHost(&h_out, bytes);
  cuda_error("cudaMallocHost h_out");

  // device matrices
  int *d_a;
  cudaMalloc(&d_a, bytes);
  cuda_error("cudaMalloc d_a");

  int *d_b;
  cudaMalloc(&d_b, bytes);
  cuda_error("cudaMalloc d_b");

  int *d_out;
  cudaMalloc(&d_out, bytes);
  cuda_error("cudaMalloc d_out");

  // cuRAND states (optimized - use fewer states to save memory)
  unsigned long seed = 42; // time(NULL);
  // Use 1024 states instead of 1M (64MB -> 64KB memory savings)
  const int num_states = 1024;
  curandState *d_states;
  cudaMalloc(&d_states, num_states * sizeof(curandState));
  cuda_error("cudaMalloc d_states");

  // Create two streams
  cudaStream_t streams[2];
  cudaStreamCreate(&streams[0]);
  cuda_error("cudaStreamCreate streams[0]");
  cudaStreamCreate(&streams[1]);
  cuda_error("cudaStreamCreate streams[1]");

  // kernel parameters for matrix multiplication
  dim3 threads_per_block(32, 32); // 32x32 = 1024 threads per block
  dim3 blocks_per_grid((cols + threads_per_block.x - 1) / threads_per_block.x,
                       (rows + threads_per_block.y - 1) / threads_per_block.y);

  {
    CudaTimer timer("Initializing cuRAND states");
    // Use 1D grid for state initialization
    int threads_init = 256;
    int blocks_init = (num_states + threads_init - 1) / threads_init;
    init_curand_states<<<blocks_init, threads_init, 0, streams[0]>>>(
        d_states, seed, num_states);
    cudaStreamSynchronize(streams[0]);
  }

  {
    CudaTimer timer("Generating random matrices on GPU");
    generate_random_matrix_int<<<blocks_per_grid, threads_per_block, 0,
                                 streams[0]>>>(d_states, d_a, rows, cols, 100,
                                               num_states);
    cuda_error("generate_random_matrix_int A");
    generate_random_matrix_int<<<blocks_per_grid, threads_per_block, 0,
                                 streams[1]>>>(d_states, d_b, rows, cols, 100,
                                               num_states);
    cuda_error("generate_random_matrix_int B");
    // Wait for both streams to complete
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  {
    CudaTimer timer("Matrix multiplication on GPU");
    matrix_multplication<<<blocks_per_grid, threads_per_block, 0, streams[0]>>>(
        d_a, d_b, d_out, cols, rows);
    cuda_error("matrix_multplication");
    cudaStreamSynchronize(streams[0]);
  }

  {
    CudaTimer timer("Copying results from device to host");
    cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, streams[0]);
    cuda_error("cudaMemcpyAsync d_out to h_out");
    cudaStreamSynchronize(streams[0]);
  }

#ifdef VERIFY_RESULTS
  // Verify GPU results with CPU computation
  int *h_a;
  cudaMallocHost(&h_a, bytes);
  cuda_error("cudaMallocHost h_a");

  int *h_b;
  cudaMallocHost(&h_b, bytes);
  cuda_error("cudaMallocHost h_b");

  {
    CudaTimer timer("VERIFICATION: Copying matrices from device to host");
    // Start verification copies in parallel
    cudaMemcpyAsync(h_a, d_a, bytes, cudaMemcpyDeviceToHost, streams[0]);
    cuda_error("cudaMemcpyAsync d_a to h_a");
    cudaMemcpyAsync(h_b, d_b, bytes, cudaMemcpyDeviceToHost, streams[1]);
    cuda_error("cudaMemcpyAsync d_b to h_b");
    // Wait for both streams to complete
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  int *result;
  cudaMallocHost(&result, bytes);
  cuda_error("VERIFICATION: cudaMallocHost result");

  {
    CudaTimer timer("VERIFICATION: Matrix multiplication on CPU");
    matrix_multiplication_cpu(h_a, h_b, result, cols);
  }

  // Compare results
  compare_results(h_out, result, total_size);

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
