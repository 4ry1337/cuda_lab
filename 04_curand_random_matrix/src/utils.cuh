#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <cstddef>
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

inline void cuda_error(const char *prefix) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", prefix, cudaGetErrorString(err));
    exit(1);
  }
}

template <typename T>
inline void print_matrix(T *h_matrix, int rows, int cols,
                         int max_display = 10) {
  std::size_t display_rows = (rows < max_display) ? rows : max_display;
  std::size_t display_cols = (cols < max_display) ? cols : max_display;

  printf("Matrix (%dx%d", rows, cols);
  if (rows > max_display || cols > max_display) {
    printf(", showing %zux%zu", display_rows, display_cols);
  }
  printf("):\n");

  for (int i = 0; i < display_rows; i++) {
    for (int j = 0; j < display_cols; j++) {
      printf("%g\t", (double)h_matrix[i * cols + j]);
    }
    if (cols > max_display) {
      printf("...");
    }
    printf("\n");
  }
  if (rows > max_display) {
    for (int j = 0; j < display_cols; j++) {
      printf(".\t");
    }
    if (cols > max_display) {
      printf("...");
    }
    printf("\n");
  }
}

#endif // UTILS_CUH_
