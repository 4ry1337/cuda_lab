#ifndef MULTIPLICATION_CUH_
#define MULTIPLICATION_CUH_

#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

extern void wrapper();

#endif // !MULTIPLICATION_CUH_
