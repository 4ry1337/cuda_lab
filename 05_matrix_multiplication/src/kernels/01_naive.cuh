#ifndef NAIVE_CUH_
#define NAIVE_CUH_

// Naive matrix multiplication: C = A × B
// A is M×N, B is N×K, C is M×K
__global__ void matrix_multplication_naive(int *d_a, int *d_b, int *d_out,
                                           size_t M, size_t N, size_t K) {
  // Calculate global thread indices
  // x = column index in output matrix C (ranges 0 to K-1)
  // y = row index in output matrix C (ranges 0 to M-1)
  size_t x = threadIdx.x + blockIdx.x * blockDim.x; // column (x-axis)
  size_t y = threadIdx.y + blockIdx.y * blockDim.y; // row (y-axis)

  // Check bounds: ensure we're within the output matrix C (M×K)
  if ((y < M) && (x < K)) {
    int val = 0;

    // Compute dot product: C[y][x] = sum(A[y][i] * B[i][x]) for i=0 to N-1
    // - A[y][i]: element at row y, column i in matrix A (M×N)
    //   Linearized index: y * N + i (row-major storage)
    // - B[i][x]: element at row i, column x in matrix B (N×K)
    //   Linearized index: i * K + x (row-major storage)
    for (size_t i = 0; i < N; i++) {
      val += d_a[y * N + i] * d_b[i * K + x];
    }

    // Store result in C[y][x]
    // Linearized index: y * K + x (output matrix C is M×K)
    d_out[y * K + x] = val;
  }
}

#endif // !NAIVE_CUH_
