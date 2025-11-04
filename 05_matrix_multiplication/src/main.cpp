#include "multiplication.cuh"

int main(int argc, char *argv[]) {
  wrapper(VECTORIZE, true);
  return 0;
}
