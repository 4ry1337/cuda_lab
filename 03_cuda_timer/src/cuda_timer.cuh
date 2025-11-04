#ifndef CUDA_TIMER_H_
#define CUDA_TIMER_H_

#include <cuda_runtime.h>
#include <stdio.h>

class CudaTimer {
private:
  cudaEvent_t start, stop;
  const char *label;

public:
  CudaTimer(const char *label = "Operation") : label(label) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
  }

  ~CudaTimer() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("ELAPSED TIME: %s: %.5f ms; \n", label, elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};

#endif // !CUDA_TIMER_H_
