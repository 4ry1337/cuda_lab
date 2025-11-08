#ifndef NAIVE_CUH_
#define NAIVE_CUH_

// Naive matrix multiplication
template <typename T>
__global__ void matrix_multplication_naive(T *d_a, T *d_b, T *d_c, uint m,
                                           uint n, uint k) {
  const uint c_row = threadIdx.x + blockIdx.x * blockDim.x;
  const uint c_col = threadIdx.y + blockIdx.y * blockDim.y;

  // Check bounds: ensure we're within the output matrix C (M×K)
  if ((c_row < m) && (c_col < k)) {
    T val = 0;
    for (uint i = 0; i < n; i++) {
      val += d_a[c_row * n + i] * d_b[i * k + c_col];
    }
    d_c[c_row * k + c_col] = val;
  }
}

#endif // !NAIVE_CUH_
