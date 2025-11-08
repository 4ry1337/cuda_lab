#ifndef DBLOCK_CUH_
#define DBLOCK_CUH_

#include <cassert>

// 1D Blocktiling Matrix Multiplication
template <typename T, const int BM, const int BN, const int BK, const int TM>
__global__ void matrix_multplication_1d_blocktailing(T *d_a, T *d_b, T *d_c,
                                                     uint m, uint n, uint k) {
  __shared__ T s_a[BM * BN]; // Shared A tile [BM×BN]
  __shared__ T s_b[BN * BK]; // Shared B tile [BN×BK]

  // Thread's position in tile
  const uint thread_col = threadIdx.x % BK; // Column in tile (0 to BK-1)
  const uint thread_row = threadIdx.x / BK; // Row group in tile (0 to BM/TM-1)

  const uint block_row = blockIdx.x; // Block row index (0 to M/BM-1)
  const uint block_col = blockIdx.y; // Block col index (0 to K/BK-1)

  // Calculate starting pointers for this block's region in global memory
  // Start at row=(out_row * BLOCKSIZE), col=0 in A
  d_a += block_row * BM * n;
  // Start at row=0, col=(out_col * BLOCKSIZE) in B
  d_b += block_col * BK;
  // Output position in C
  d_c += block_row * BM * k + block_col * BK;

  // Verify dimensions are compatible with thread block size
  assert(BM * BN == blockDim.x); // Loading A requires BM*BN elements
  assert(BN * BK == blockDim.x); // Loading B requires BN*BK elements

  const uint inner_a_col = threadIdx.x % BN;
  const uint inner_a_row = threadIdx.x / BN;
  const uint inner_b_col = threadIdx.x % BK;
  const uint inner_b_row = threadIdx.x / BK;

  // Thread-local storage: accumulate TM output values in registers
  T thread_results[TM] = {0};

  // Outer loop: iterate over tiles along the N dimension
  for (uint block_id = 0; block_id < n; block_id += BN) {
    // Cooperatively load tiles from global to shared memory
    // Each thread loads one element from A and one from B
    s_a[inner_a_row * BN + inner_a_col] = d_a[inner_a_row * n + inner_a_col];
    s_b[inner_b_row * BK + inner_b_col] = d_b[inner_b_row * k + inner_b_col];

    // Wait for all threads to finish loading the tiles
    // Critical: ENSURES SHARED MEMORY IS FULLY POPULATED BEFORE COMPUTATION
    __syncthreads();

    for (uint i = 0; i < BN; ++i) {
      T temp_b = s_b[i * BK + thread_col];

      // Each thread computes TM output elements (1D tiling)
      // This is the key optimization: one B load, TM A loads, TM multiplies
      for (uint result_index = 0; result_index < TM; ++result_index) {
        thread_results[result_index] +=
            s_a[(thread_row * TM + result_index) * BN + i] * temp_b;
      }
    }
    // Ensure computation done before loading next tiles
    __syncthreads();

    // Advance to next tile along N dimension
    d_a += BN;
    d_b += BN * k;
  }

  for (uint result_index = 0; result_index < TM; ++result_index) {
    d_c[(thread_row * TM + result_index) * k + thread_col] =
        thread_results[result_index];
  }
}

#endif // !DBLOCK_CUH_
