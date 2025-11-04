#include "cuda_timer.cuh"
#include "random_matrix.cuh"
#include "reduction.cuh"
#include "utils.cuh"

void device_properties(cudaDeviceProp &device, int device_index = 0) {
  cudaGetDeviceProperties(&device, device_index);

  printf("  --- General information for device ---\n");
  printf("Name: %s;\n", device.name);
  printf("Compute capability: %d.%d\n", device.major, device.minor);
  printf("Total global memory: %ld\n", device.totalGlobalMem);
  printf("Total constant memory: %ld\n", device.totalConstMem);
  printf("Multiprocessor count: %d\n", device.multiProcessorCount);
  printf("Shared memory per block: %ld\n", device.sharedMemPerBlock);
  printf("Registers per block: %d\n", device.regsPerBlock);
  printf("Threads in warp: %d\n", device.warpSize);
  printf("Max threads Per Block: %d\n", device.maxThreadsPerBlock);
  printf("Max thread dimensions: (%d, %d, %d)\n", device.maxThreadsDim[0],
         device.maxThreadsDim[1], device.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", device.maxGridSize[0],
         device.maxGridSize[1], device.maxGridSize[2]);
  printf("  --- General information for device ---\n\n");
}

void wrapper() {
  cudaDeviceProp device;
  device_properties(device, 0);

  // sizes
  size_t rows = device.maxThreadsDim[0];
  size_t cols = device.maxThreadsDim[1];
  printf("Matrix of size %dx%d\n\n", rows, cols);
  size_t total_size = rows * cols;
  const size_t bytes = total_size * sizeof(int);

  // pinned host matrices (for async copy)
  int *h_a;
  int *h_b;
  cudaMallocHost(&h_a, bytes);
  cuda_error("cudaMallocHost h_a");
  cudaMallocHost(&h_b, bytes);
  cuda_error("cudaMallocHost h_b");

  // device matrices
  int *d_a;
  int *d_b;
  cudaMalloc(&d_a, bytes);
  cuda_error("cudaMalloc d_a");
  cudaMalloc(&d_b, bytes);
  cuda_error("cudaMalloc d_b");

  // states
  unsigned long seed = 42; // time(NULL);
  curandState *d_states;
  cudaMalloc(&d_states, total_size * sizeof(curandState));
  cuda_error("cudaMalloc d_states");

  // Create two streams
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cuda_error("cudaStreamCreate stream1");
  cudaStreamCreate(&stream2);
  cuda_error("cudaStreamCreate stream2");

  // kernel parameters
  int threads_per_blocks = 256;
  int blocks_per_grid =
      (total_size + threads_per_blocks - 1) / threads_per_blocks;

  {
    CudaTimer timer("Initializing cuRAND states");
    init_curand_states<<<blocks_per_grid, threads_per_blocks, 0, stream1>>>(
        d_states, seed, rows, cols);
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Generating random matrices on GPU");
    generate_random_matrix_int<<<blocks_per_grid, threads_per_blocks, 0,
                                 stream1>>>(d_states, d_a, rows, cols, 100);
    cuda_error("generate_random_matrix_int A");
    generate_random_matrix_int<<<blocks_per_grid, threads_per_blocks, 0,
                                 stream2>>>(d_states, d_b, rows, cols, 100);
    cuda_error("generate_random_matrix_int B");
    cudaDeviceSynchronize();
  }

  {
    CudaTimer timer("Async copying matrices from device to host");
    cudaMemcpyAsync(h_a, d_a, bytes, cudaMemcpyDeviceToHost, stream1);
    cuda_error("cudaMemcpyAsync d_a to h_a");
    cudaMemcpyAsync(h_b, d_b, bytes, cudaMemcpyDeviceToHost, stream2);
    cuda_error("cudaMemcpyAsync d_b to h_b");
    cudaDeviceSynchronize();
  }

  // Print matrices
  printf("\n");
  print_matrix(h_a, rows, cols);
  printf("\n");
  print_matrix(h_b, rows, cols);
  printf("\n");

  // Cleanup
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_states);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}
