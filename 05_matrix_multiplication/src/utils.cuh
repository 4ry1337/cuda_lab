#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <cstddef>
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

inline void cuda_error(const char *prefix) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", prefix, cudaGetErrorString(err));
    exit(1);
  }
}

inline void print_device_properties(cudaDeviceProp &device) {
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

template <typename T>
inline void print_matrix(T *h_matrix, int rows, int cols,
                         int max_display = 10) {
  std::size_t display_rows = (rows < max_display) ? rows : max_display;
  std::size_t display_cols = (cols < max_display) ? cols : max_display;

  printf("Matrix (%dx%d", rows, cols);
  if (rows > max_display || cols > max_display) {
    printf(", showing %zux%zu", display_rows, display_cols);
  }
  printf("):\n");

  for (int i = 0; i < display_rows; i++) {
    for (int j = 0; j < display_cols; j++) {
      printf("%g\t", (double)h_matrix[i * cols + j]);
    }
    if (cols > max_display) {
      printf("...");
    }
    printf("\n");
  }
  if (rows > max_display) {
    for (int j = 0; j < display_cols; j++) {
      printf(".\t");
    }
    if (cols > max_display) {
      printf("...");
    }
    printf("\n");
  }
}

#endif // UTILS_CUH_
