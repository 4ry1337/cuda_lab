#ifndef SMEM_CUH_
#define SMEM_CUH_

// Shared Memory (SMEM) optimized matrix multiplication: C = A × B
// A[M][N] B[N][K] C[M][K]
//
// Key optimization: Uses tiled computation with shared memory to reduce global
// memory accesses. Shared memory is ~100x faster than global memory.
template <const int BLOCKSIZE>
__global__ void matrix_multplication_smem(int *d_a, int *d_b, int *d_c,
                                          size_t m, size_t n, size_t k) {
  // Allocate shared memory buffers for tiles
  // These are shared by all threads in this block
  __shared__ float s_a[BLOCKSIZE * BLOCKSIZE];
  __shared__ float s_b[BLOCKSIZE * BLOCKSIZE];

  // Calculate thread's position within its tile (local coordinates)
  // Using 1D thread indexing (threadIdx.x) mapped to 2D tile positions
  const uint thread_col = threadIdx.x % BLOCKSIZE; // Column within tile
  const uint thread_row = threadIdx.x / BLOCKSIZE; // Row within tile

  // Calculate which output tile this block is responsible for
  const uint block_row = blockIdx.x; // Block row index in grid
  const uint block_col = blockIdx.y; // Block column index in grid

  //  Calculate starting pointers for this block's region in global
  // memory Each block processes one BLOCKSIZE×BLOCKSIZE tile of the output
  // matrix C
  //
  // Start at row=(out_row * BLOCKSIZE), col=0 in A
  d_a += block_row * BLOCKSIZE * n;
  // Start at row=0, col=(out_col * BLOCKSIZE) in B
  d_b += block_col * BLOCKSIZE;
  // Output position in C
  d_c += block_row * BLOCKSIZE * k + block_col * BLOCKSIZE;

  int val = 0;

  // Loop through tiles along the shared dimension (N)
  // For C[i][j] = sum(A[i][k] * B[k][j]), we process k in BLOCKSIZE chunks
  for (int block_id = 0; block_id < n; block_id += BLOCKSIZE) {
    // Cooperatively load one tile from A and B into shared memory
    // Each thread loads one element from global → shared memory
    s_a[thread_row * BLOCKSIZE + thread_col] = d_a[thread_row * n + thread_col];
    s_b[thread_row * BLOCKSIZE + thread_col] = d_b[thread_row * k + thread_col];

    // Wait for all threads to finish loading the tiles
    // Critical: ENSURES SHARED MEMORY IS FULLY POPULATED BEFORE COMPUTATION
    __syncthreads();

    // Move global pointers to next tiles for next iteration
    d_a += BLOCKSIZE;     // Move right in A (next BLOCKSIZE columns)
    d_b += BLOCKSIZE * k; // Move down in B (next BLOCKSIZE rows)

    // Compute partial dot product using data from shared memory
    // This is where we benefit: BLOCKSIZE reads from fast shared memory
    // instead of slow global memory
    for (int i = 0; i < BLOCKSIZE; i++) {
      val += s_a[thread_row * BLOCKSIZE + i] * s_b[i * BLOCKSIZE + thread_col];
    }
    // Wait for all threads to finish computing before loading next
    // tile Prevents overwriting shared memory while other threads still need it
    __syncthreads();
  }
  d_c[thread_row * n + thread_col] = val;
}

#endif // !SMEM_CUH_
