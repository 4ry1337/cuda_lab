#ifndef DDBLOCK_CUH_
#define DDBLOCK_CUH_

// 2D Blocktiling implementation: C = A × B
// A is M×N, B is N×K, C is M×K
//
// Template parameters:
// - BM: Block tile size in M dimension (rows of output)
// - BN: Block tile size in N dimension (shared dimension, controls SMEM usage)
// - BK: Block tile size in K dimension (cols of output)
// - TM: Number of output rows each thread computes (vertical blocktiling)
// - TN: Number of output cols each thread computes (horizontal blocktiling)
//
// Each thread block processes a BM×BK tile of output matrix
// Each thread computes TM×TN output elements (a 2D tile)
// Uses shared memory + register caching for maximum data reuse
//
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matrix_multplication_2d_blocktailing(int *d_a, int *d_b,
                                                     int *d_out, size_t M,
                                                     size_t N, size_t K) {
  // Block-level tile indices in output matrix
  const uint out_row = blockIdx.x;
  const uint out_col = blockIdx.y;

  const uint total_results_blocktile = BM * BK;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint num_threads_per_blocktile = total_results_blocktile / (TM * TN);

  const uint thread_col = threadIdx.x % (BK / TN);
  const uint thread_row = threadIdx.x / (BK / TN);

  __shared__ int s_a[BM * BN];
  __shared__ int s_b[BN * BK];

  d_a += out_row * BM * N;                  // row=cRow, col=0
  d_b += out_col * BK;                      // row=0, col=cCol
  d_out += out_row * BM * K + out_col * BK; // row=cRow, col=cCol
                                            //
  // calculating the indices that this thread will load into SMEM
  const uint inner_a_row = threadIdx.x / BN;
  const uint inner_a_col = threadIdx.x % BN;
  // calculates the number of rows of s_a that are being loaded in a single step
  // by a single block
  const uint stride_a = num_threads_per_blocktile / BN;
  const uint inner_b_row = threadIdx.x / BK;
  const uint inner_b_col = threadIdx.x % BK;
  // for both s_a and s_b we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint stride_b = num_threads_per_blocktile / BK;

  int thread_results[TM * TN] = {0};

  // register caches for As and Bs
  int reg_m[TM] = {0};
  int reg_n[TN] = {0};

  for (uint block_id = 0; block_id < N; block_id += BN) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += stride_a) {
      s_a[(inner_a_row + loadOffset) * BN + inner_a_col] =
          d_a[(inner_a_row + loadOffset) * N + inner_a_col];
    }
    for (uint loadOffset = 0; loadOffset < BN; loadOffset += stride_b) {
      s_b[(inner_b_row + loadOffset) * BK + inner_b_col] =
          d_b[(inner_b_row + loadOffset) * K + inner_b_col];
    }
    __syncthreads();

    // advance blocktile
    d_a += BN;     // move BN columns to right
    d_b += BN * K; // move BN rows down

    // calculate per-thread results
    for (uint index = 0; index < BN; ++index) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        reg_m[i] = s_a[(thread_row * TM + i) * BN + index];
      }
      for (uint i = 0; i < TN; ++i) {
        reg_n[i] = s_b[index * BK + thread_col * TN + i];
      }
      for (uint result_index_m = 0; result_index_m < TM; ++result_index_m) {
        for (uint result_index_n = 0; result_index_n < TN; ++result_index_n) {
          thread_results[result_index_m * TN + result_index_n] +=
              reg_m[result_index_m] * reg_n[result_index_n];
        }
      }
    }
    __syncthreads();
  }
  // write out the results
  for (uint result_index_m = 0; result_index_m < TM; ++result_index_m) {
    for (uint result_index_n = 0; result_index_n < TN; ++result_index_n) {
      d_out[(thread_row * TM + result_index_m) * K + thread_col * TN +
            result_index_n] =
          thread_results[result_index_m * TN + result_index_n];
    }
  }
}

#endif // !DDBLOCK_CUH_
