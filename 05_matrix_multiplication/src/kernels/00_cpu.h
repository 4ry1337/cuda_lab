#ifndef CPU_H_
#define CPU_H_

// CPU matrix multiplication: C = A × B
// A[M][N] B[N][K] C[M][K]
// Used for verification of GPU results
template <typename T>
void matrix_multiplication_cpu(T *h_a, T *h_b, T *h_c, uint m, uint n, uint k) {
  for (uint c_row = 0; c_row < m; c_row++) {
    for (uint c_col = 0; c_col < k; c_col++) {
      T val = 0;
      for (uint i = 0; i < n; i++) {
        val += h_a[c_row * n + i] * h_b[i * k + c_col];
      }
      h_c[c_row * k + c_col] = val;
    }
  }
}

#endif // !CPU_H_
