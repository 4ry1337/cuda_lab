#ifndef SMEM_CUH_
#define SMEM_CUH_

template <const int BLOCKSIZE>
__global__ void matrix_multplication_smem(int *d_a, int *d_b, int *d_out,
                                          size_t M, size_t N, size_t K) {
  // the output block that we want to compute in this threadblock
  const uint out_row = blockIdx.x;
  const uint out_col = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float s_a[BLOCKSIZE * BLOCKSIZE];
  __shared__ float s_b[BLOCKSIZE * BLOCKSIZE];

  const uint thread_col = threadIdx.x % BLOCKSIZE;
  const uint thread_row = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  d_a += out_row * BLOCKSIZE * N;                         // row=cRow, col=0
  d_b += out_col * BLOCKSIZE;                             // row=0, col=cCol
  d_out += out_row * BLOCKSIZE * K + out_col * BLOCKSIZE; // row=cRow, col=cCol

  int val;

  for (int block_id = 0; block_id < N; block_id += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    s_a[thread_row * BLOCKSIZE + thread_col] = d_a[thread_row * N + thread_col];
    s_b[thread_row * BLOCKSIZE + thread_col] = d_b[thread_row * K + thread_col];

    // block threads in this block until cache is fully populated
    __syncthreads();
    d_a += BLOCKSIZE;
    d_b += BLOCKSIZE * K;

    // execute the dotproduct on the currently cached block
    for (int i = 0; i < BLOCKSIZE; ++i) {
      val += s_a[thread_row * BLOCKSIZE + i] * s_b[i * BLOCKSIZE + thread_col];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  d_out[thread_row * N + thread_col] = val;
}

#endif // !SMEM_CUH_
