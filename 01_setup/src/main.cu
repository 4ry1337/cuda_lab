#include <chrono>
#include <stdio.h>

void cuda_error(const char *prefix) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", prefix, cudaGetErrorString(err));
    exit(1);
  }
}

__global__ void kernel() {
  unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
  printf("%d %d %d\n", x, y, z);
}

int main() {
  dim3 threads(2, 2, 2);
  dim3 block(2, 2, 2);

  auto start = std::chrono::high_resolution_clock::now();

  kernel<<<threads, block>>>();
  cuda_error("Launch failed");

  cudaDeviceSynchronize();
  cuda_error("Synchronization failed");

  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  printf("Execution time: %.3f ms\n", duration.count() / 1000.0);

  return 0;
}
