#ifndef MULTIPLICATION_CUH_
#define MULTIPLICATION_CUH_

enum KernelType { NAIVE, GMEM, SMEM, DBLOCK, DDBLOCK, VECTORIZE };

void wrapper(KernelType type, bool verify_result);

inline const char *to_string(KernelType v) {
  switch (v) {
  case NAIVE:
    return "NAIVE";
  case GMEM:
    return "GMEM";
  case SMEM:
    return "SMEM";
  case DBLOCK:
    return "DBLOCK";
  case DDBLOCK:
    return "DDBLOCK";
  case VECTORIZE:
    return "VECTORIZE";
  default:
    return "[Unknown KernelType]";
  }
}

#endif // !MULTIPLICATION_CUH_
