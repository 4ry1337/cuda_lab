#ifndef MULTIPLICATION_CUH_
#define MULTIPLICATION_CUH_

enum KernelType { NAIVE, GMEM, SMEM, DBLOCK, DDBLOCK, VECTORIZE };

extern void wrapper(KernelType type, bool verify_result);

#endif // !MULTIPLICATION_CUH_
