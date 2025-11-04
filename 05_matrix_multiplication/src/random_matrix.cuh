#ifndef RANDOM_MATRIX_CUH_
#define RANDOM_MATRIX_CUH_

#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA kernel to initialize cuRAND states (optimized - uses fewer states)
__global__ void init_curand_states(curandState *state, unsigned long seed,
                                   int num_states) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_states) {
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, idx, 0, &state[idx]);
  }
}

// CUDA kernel to generate random float matrix [0.0, 1.0) (optimized)
__global__ void generate_random_matrix(curandState *state, float *matrix,
                                       int rows, int cols, int num_states) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < rows && col < cols) {
    int matrix_idx = row * cols + col;
    int state_idx = matrix_idx % num_states;
    curandState localState = state[state_idx];
    matrix[matrix_idx] = curand_uniform(&localState);
    state[state_idx] = localState;
  }
}

// CUDA kernel to generate random int matrix [0, max_val) (optimized)
__global__ void generate_random_matrix_int(curandState *state, int *matrix,
                                           int rows, int cols, int max_val,
                                           int num_states) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < rows && col < cols) {
    int matrix_idx = row * cols + col;
    int state_idx = matrix_idx % num_states;
    curandState localState = state[state_idx];
    matrix[matrix_idx] = curand(&localState) % max_val;
    state[state_idx] = localState;
  }
}

#endif // RANDOM_MATRIX_CUH_
