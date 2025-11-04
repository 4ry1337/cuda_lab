#ifndef RANDOM_MATRIX_CUH_
#define RANDOM_MATRIX_CUH_

#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA kernel to initialize cuRAND states
__global__ void init_curand_states(curandState *state, unsigned long seed,
                                   int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = rows * cols;

  if (idx < total_size) {
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, idx, 0, &state[idx]);
  }
}

// CUDA kernel to generate random float matrix [0.0, 1.0)
__global__ void generate_random_matrix(curandState *state, float *matrix,
                                       int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = rows * cols;

  if (idx < total_size) {
    curandState localState = state[idx];
    matrix[idx] = curand_uniform(&localState);
    state[idx] = localState;
  }
}

// CUDA kernel to generate random int matrix [0, max_val)
__global__ void generate_random_matrix_int(curandState *state, int *matrix,
                                           int rows, int cols, int max_val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = rows * cols;

  if (idx < total_size) {
    curandState localState = state[idx];
    matrix[idx] = curand(&localState) % max_val;
    state[idx] = localState;
  }
}

#endif // RANDOM_MATRIX_CUH_
