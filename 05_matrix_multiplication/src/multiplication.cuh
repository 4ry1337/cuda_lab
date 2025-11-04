#ifndef MULTIPLICATION_CUH_
#define MULTIPLICATION_CUH_

#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

enum KernelType { NAIVE, GMEM, SMEM, DBLOCK, DDBLOCK };

extern void wrapper(KernelType type, bool verify_result);

#endif // !MULTIPLICATION_CUH_
