#ifndef NAIVE_CUH_
#define NAIVE_CUH_

// Naive matrix multiplication: C = A × B
// A[M][N] B[N][K] C[M][K]
__global__ void matrix_multplication_naive(int *d_a, int *d_b, int *d_c, uint m,
                                           uint n, uint k) {
  uint c_row = threadIdx.x + blockIdx.x * blockDim.x; // column (x-axis)
  uint c_col = threadIdx.y + blockIdx.y * blockDim.y; // row (y-axis)

  // Check bounds: ensure we're within the output matrix C (M×K)
  if ((c_col < m) && (c_row < k)) {
    int val = 0;
    for (uint i = 0; i < n; i++) {
      val += d_a[c_col * n + i] * d_b[i * k + c_row];
    }
    d_c[c_col * k + c_row] = val;
  }
}

#endif // !NAIVE_CUH_
