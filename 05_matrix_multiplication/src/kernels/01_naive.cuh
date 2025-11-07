#ifndef NAIVE_CUH_
#define NAIVE_CUH_

// Naive matrix multiplication: C = A × B
// A is M×N, B is N×K, C is M×K
#include <cstddef>

__global__ void matrix_multplication_naive(int *d_a, int *d_b, int *d_out,
                                           std::size_t M, std::size_t N,
                                           std::size_t K) {
  std::size_t x = threadIdx.x + blockIdx.x * blockDim.x; // column (x-axis)
  std::size_t y = threadIdx.y + blockIdx.y * blockDim.y; // row (y-axis)

  // Check bounds: ensure we're within the output matrix C (M×K)
  if ((y < M) && (x < K)) {
    int val = 0;
    for (std::size_t i = 0; i < N; i++) {
      val += d_a[y * N + i] * d_b[i * K + x];
    }
    d_out[y * K + x] = val;
  }
}

#endif // !NAIVE_CUH_
