#ifndef CPU_H_
#define CPU_H_

// CPU reference implementation: C = A × B
// A is M×N, B is N×K, C is M×K
// Used for verification of GPU results
#include <cstddef>

inline void matrix_multiplication_cpu(int *h_a, int *h_b, int *h_out,
                                      std::size_t M, std::size_t N,
                                      std::size_t K) {
  for (std::size_t x = 0; x < M; x++) {
    for (std::size_t y = 0; y < K; y++) {
      int val = 0;
      for (std::size_t z = 0; z < N; z++) {
        val += h_a[x * N + z] * h_b[z * K + y];
      }
      h_out[x * K + y] = val;
    }
  }
}

#endif // !CPU_H_
