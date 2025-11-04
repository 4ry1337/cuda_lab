#ifndef DBLOCK_CUH_
#define DBLOCK_CUH_

// 1D Blocktiling implementation: C = A × B
// A is M×N, B is N×K, C is M×K
//
// Template parameters:
// - BM: Block tile size in M dimension (rows of output)
// - BN: Block tile size in N dimension (shared dimension, controls SMEM usage)
// - BK: Block tile size in K dimension (cols of output)
// - TM: Number of output elements each thread computes (1D blocktiling factor)
//
// Each thread block processes a BM×BK tile of output matrix
// Each thread computes TM consecutive output elements in the row direction
// Uses shared memory to cache tiles of A and B for reuse
//
template <const int BM, const int BN, const int BK, const int TM>
__global__ void matrix_multplication_1d_blocktailing(int *d_a, int *d_b,
                                                     int *d_out, size_t M,
                                                     size_t N, size_t K) {
  // Block-level tile indices in output matrix
  // blockIdx.x = row block index (0 to M/BM - 1)
  // blockIdx.y = col block index (0 to K/BK - 1)
  // This configuration ensures better L2 cache hit rate (see comment below)
  const uint out_row = blockIdx.x;
  const uint out_col = blockIdx.y;

  // allocate space for the current blocktile in SMEM
  __shared__ int s_a[BM * BN];
  __shared__ int s_b[BN * BK];

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const uint thread_col = threadIdx.x % BK;
  const uint thread_row = threadIdx.x / BK;

  // Move blocktile to beginning of A's row and B's column
  d_a += out_row * BM * N;                  // row=cRow, col=0
  d_b += out_col * BK;                      // row=0, col=cCol
  d_out += out_row * BM * K + out_col * BK; // row=cRow, col=cCol

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BN == blockDim.x);
  assert(BN * BK == blockDim.x);

  const uint inner_a_col = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint inner_a_row = threadIdx.x / BN;
  const uint inner_b_col = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint inner_b_row = threadIdx.x / BK;

  // allocate thread-local cache for results in registerfile
  int thread_results[TM] = {0};

  for (int block_id = 0; block_id < N; block_id += BN) {
    // populate the SMEM caches
    s_a[inner_a_row * BN + inner_a_col] = d_a[inner_a_row * N + inner_a_col];
    s_b[inner_b_row * BK + inner_b_col] = d_b[inner_b_row * K + inner_b_col];

    // wait for all threads to finish loading
    __syncthreads();

    // calculate per-thread results
    for (uint i = 0; i < BN; ++i) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the s_b entry, which we can cache in a tmp var.
      int temp_b = s_b[i * BK + thread_col];
      for (uint result_index = 0; result_index < TM; ++result_index) {
        thread_results[result_index] +=
            s_a[(thread_row * TM + result_index) * BN + i] * temp_b;
      }
    }
    __syncthreads();

    // advance blocktile
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
