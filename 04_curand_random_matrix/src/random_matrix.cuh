#ifndef RANDOM_MATRIX_CUH_
#define RANDOM_MATRIX_CUH_

#include <curand_kernel.h>

// CUDA kernel to initialize cuRAND states
__global__ void init_curand_states(curandState *state, unsigned long seed,
                                   uint num_states) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < num_states) {
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, idx, 0, &state[idx]);
  }
}

template <typename T>
__global__ void random_matrix(uint num_states, curandState *state, T *matrix,
                              uint rows, uint cols, T max_val = 10) {
  uint col = threadIdx.x + blockIdx.x * blockDim.x;
  uint row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row < rows && col < cols) {
    uint matrix_idx = row * cols + col;
    int state_idx = matrix_idx % num_states;
    curandState localState = state[state_idx];
    matrix[matrix_idx] = (T)(curand_uniform(&localState) * max_val);
    state[state_idx] = localState;
  }
}

#endif // RANDOM_MATRIX_CUH_
