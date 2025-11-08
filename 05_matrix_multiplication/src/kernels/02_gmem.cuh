#ifndef GMEM_CUH_
#define GMEM_CUH_

// Global memory (GMEM) coalescing implementation
//
// BLOCKSIZE: compile-time constant (typically 32) for better memory access
// patterns
//
// IMPORTANT: This kernel requires 1D thread blocks of size BLOCKSIZE*BLOCKSIZE!
// Launch with: <<<grid, BLOCKSIZE*BLOCKSIZE, ...>>> e.g., <<<grid, 1024, ...>>>
//
// How 1D indexing improves memory coalescing:
// - Threads are numbered linearly: threadIdx.x = 0 to 1023 (for BLOCKSIZE=32)
// - Adjacent thread IDs map to adjacent columns (same row):
//     Thread 0  → [row 0, col 0]    writes to d_out[0*K + 0]
//     Thread 1  → [row 0, col 1]    writes to d_out[0*K + 1]  (adjacent!)
//     Thread 2  → [row 0, col 2]    writes to d_out[0*K + 2]  (adjacent!)
//     ...
//     Thread 31 → [row 0, col 31]   writes to d_out[0*K + 31]
//     Thread 32 → [row 1, col 0]    writes to d_out[1*K + 0]
//     ...
// - This ensures adjacent threads write to adjacent memory addresses
// - GPU can coalesce these into fewer memory transactions (better performance!)
template <const size_t BLOCKSIZE>
__global__ void matrix_multplication_gmem(int *d_a, int *d_b, int *d_c,
                                          size_t m, size_t n, size_t k) {
  const int c_row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int c_col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if ((c_row < m) && (c_col < k)) {
    int val = 0;
    for (uint i = 0; i < n; i++) {
      val += d_a[c_row * n + i] * d_b[i * k + c_col];
    }
    d_c[c_row * k + c_col] = val;
  }
}

#endif // !GMEM_CUH_
