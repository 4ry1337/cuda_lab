#ifndef VECTORIZE_CUH_
#define VECTORIZE_CUH_

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matrix_multplication_vectorize(int *d_a, int *d_b, int *d_out,
                                               size_t M, size_t N, size_t K) {
  const uint out_row = blockIdx.x;
  const uint out_col = blockIdx.y;

  // BK/TN are the number of threads to span a column
  const uint thread_col = threadIdx.x % (BK / TN);
  const uint thread_row = threadIdx.x / (BK / TN);

  // allocate space for the current blocktile in smem
  __shared__ int s_a[BM * BN];
  __shared__ int s_b[BN * BK];

  // Move blocktile to beginning of d_a's row and d_b's column
  d_a += out_row * BM * N;                  // row=cRow, col=0
  d_b += out_col * BK;                      // row=0, col=cCol
  d_out += out_row * BM * K + out_col * BK; // row=cRow, col=cCol

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint num_threads_per_blocktile = (BM * BK) / (TM * TN);
  const uint inner_a_row = threadIdx.x / (BN / 4);
  const uint inner_a_col = threadIdx.x % (BN / 4);
  const uint stride_a = num_threads_per_blocktile / (BN / 4);
  const uint inner_b_row = threadIdx.x / (BK / 4);
  const uint inner_b_col = threadIdx.x % (BK / 4);
  const uint stride_b = num_threads_per_blocktile / (BK / 4);

  // allocate thread-local cache for results in registerfile
  int thread_results[TM * TN] = {0};
  int reg_m[TM] = {0};
  int reg_n[TN] = {0};

  // outer-most loop over block tiles
  for (uint block_id = 0; block_id < N; block_id += BN) {
    // populate the SMEM caches
    // load A with vectorization (no transpose)
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += stride_a) {
      reinterpret_cast<int4 *>(
          &s_a[(inner_a_row + loadOffset) * BN + inner_a_col * 4])[0] =
          reinterpret_cast<int4 *>(
              &d_a[(inner_a_row + loadOffset) * N + inner_a_col * 4])[0];
    }
    for (uint loadOffset = 0; loadOffset < BN; loadOffset += stride_b) {
      reinterpret_cast<int4 *>(
          &s_b[(inner_b_row + loadOffset) * BK + inner_b_col * 4])[0] =
          reinterpret_cast<int4 *>(
              &d_b[(inner_b_row + loadOffset) * K + inner_b_col * 4])[0];
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
  for (uint result_index_m = 0; result_index_m < TM; result_index_m += 1) {
    for (uint result_index_n = 0; result_index_n < TN; result_index_n += 4) {
      // load C vector into registers
      int4 tmp = reinterpret_cast<int4 *>(
          &d_out[(thread_row * TM + result_index_m) * K + thread_col * TN +
                 result_index_n])[0];
      // perform GEMM update in reg
      tmp.x = thread_results[result_index_m * TN + result_index_n] + tmp.x;
      tmp.y = thread_results[result_index_m * TN + result_index_n + 1] + tmp.y;
      tmp.z = thread_results[result_index_m * TN + result_index_n + 2] + tmp.z;
      tmp.w = thread_results[result_index_m * TN + result_index_n + 3] + tmp.w;
      // write back
      reinterpret_cast<int4 *>(&d_out[(thread_row * TM + result_index_m) * K +
                                      thread_col * TN + result_index_n])[0] =
          tmp;
    }
  }
}

#endif // !VECTORIZE_CUH_
