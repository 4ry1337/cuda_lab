#ifndef UTILS_CUH_
#define UTILS_CUH_

#include <iostream>

void cuda_error(const char *prefix) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", prefix, cudaGetErrorString(err));
    exit(1);
  }
}

// Utility function to print matrix
template <typename T>
void print_matrix(T *h_matrix, int rows, int cols, int max_display = 10) {
  int display_rows = (rows < max_display) ? rows : max_display;
  int display_cols = (cols < max_display) ? cols : max_display;

  std::cout << "Matrix (" << rows << "x" << cols;
  if (rows > max_display || cols > max_display) {
    std::cout << ", showing " << display_rows << "x" << display_cols;
  }
  std::cout << "):" << std::endl;

  for (int i = 0; i < display_rows; i++) {
    for (int j = 0; j < display_cols; j++) {
      std::cout << h_matrix[i * cols + j] << "\t";
    }
    if (cols > max_display) {
      std::cout << "...";
    }
    std::cout << std::endl;
  }
  if (rows > max_display) {
    for (int j = 0; j < display_cols; j++) {
      std::cout << ".\t";
    }
    if (cols > max_display) {
      std::cout << "...";
    }
    std::cout << std::endl;
  }
}

#endif // UTILS_CUH_
