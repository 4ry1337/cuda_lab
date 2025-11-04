#ifndef REDUCTION_H_
#define REDUCTION_H_

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <utility>

inline void cuda_error(const char *prefix) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s: %s\n", prefix, cudaGetErrorString(err));
    exit(1);
  }
}

extern void wrapper();

#endif // !REDUCTION_H_
