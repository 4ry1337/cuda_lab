#include <chrono>
#include <stdio.h>

void checkCudaError(const char *prefix) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", prefix, cudaGetErrorString(err));
    exit(1);
  }
}

int main() {
  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, 0);

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

  return 0;
}
