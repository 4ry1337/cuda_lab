#include "multiplication.cuh"

int main(int argc, char *argv[]) {
  wrapper<float>(NAIVE, true);
  wrapper<float>(GMEM, true);
  wrapper<float>(SMEM, true);
  wrapper<float>(DBLOCK, true);
  wrapper<float>(DDBLOCK, true);
  wrapper<float>(VECTORIZE, true);
  return 0;
}
