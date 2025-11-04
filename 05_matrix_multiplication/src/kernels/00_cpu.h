#ifndef CPU_H_
#define CPU_H_

// CPU reference implementation: C = A × B
// A is M×N, B is N×K, C is M×K
// Used for verification of GPU results
void matrix_multiplication_cpu(int *h_a, int *h_b, int *h_out, size_t M,
                               size_t N, size_t K) {
  // Iterate over all rows of output matrix C
  for (size_t x = 0; x < M; x++) {
    // Iterate over all columns of output matrix C
    for (size_t y = 0; y < K; y++) {
      int val = 0;

      // Compute dot product: C[x][y] = sum(A[x][z] * B[z][y]) for z=0 to N-1
      for (size_t z = 0; z < N; z++) {
        val += h_a[x * N + z] * h_b[z * K + y];
      }

      // Store result in C[x][y]
      h_out[x * K + y] = val;
    }
  }
}

#endif // !CPU_H_
