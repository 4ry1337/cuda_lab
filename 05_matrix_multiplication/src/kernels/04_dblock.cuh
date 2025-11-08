#ifndef DBLOCK_CUH_
#define DBLOCK_CUH_

#include <cassert>
template <const int BM, const int BN, const int BK, const int TM>
__global__ void matrix_multplication_1d_blocktailing(int *d_a, int *d_b,
                                                     int *d_out, size_t M,
                                                     size_t N, size_t K) {
  const uint out_row = blockIdx.x;
  const uint out_col = blockIdx.y;

  __shared__ int s_a[BM * BN];
  __shared__ int s_b[BN * BK];

  const uint thread_col = threadIdx.x % BK;
  const uint thread_row = threadIdx.x / BK;

  d_a += out_row * BM * N;
  d_b += out_col * BK;
  d_out += out_row * BM * K + out_col * BK;

  assert(BM * BN == blockDim.x);
  assert(BN * BK == blockDim.x);

  const uint inner_a_col = threadIdx.x % BN;
  const uint inner_a_row = threadIdx.x / BN;
  const uint inner_b_col = threadIdx.x % BK;
  const uint inner_b_row = threadIdx.x / BK;

  int thread_results[TM] = {0};

  for (int block_id = 0; block_id < N; block_id += BN) {
    s_a[inner_a_row * BN + inner_a_col] = d_a[inner_a_row * N + inner_a_col];
    s_b[inner_b_row * BK + inner_b_col] = d_b[inner_b_row * K + inner_b_col];

    __syncthreads();

    for (uint i = 0; i < BN; ++i) {
      int temp_b = s_b[i * BK + thread_col];
      for (uint result_index = 0; result_index < TM; ++result_index) {
        thread_results[result_index] +=
            s_a[(thread_row * TM + result_index) * BN + i] * temp_b;
      }
    }
    __syncthreads();

    d_a += BN;
    d_b += BN * K;
  }
  // write out the results
  for (uint result_index = 0; result_index < TM; ++result_index) {
    d_out[(thread_row * TM + result_index) * K + thread_col] =
        thread_results[result_index];
  }
}

#endif // !DBLOCK_CUH_
