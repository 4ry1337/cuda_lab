#ifndef GMEM_CUH_
#define GMEM_CUH_

// Global memory coalescing implementation: C = A × B
// A is M×N, B is N×K, C is M×K
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
//
template <const size_t BLOCKSIZE>
__global__ void matrix_multplication_gmem(int *d_a, int *d_b, int *d_out,
                                          size_t M, size_t N, size_t K) {
  // Calculate global thread indices using linear thread ID (threadIdx.x)
  // This improves memory coalescing by having adjacent threads access adjacent
  // memory addresses in row-major layout
  // x = column index in output matrix C (ranges 0 to K-1)
  // y = row index in output matrix C (ranges 0 to M-1)
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE); // column
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE); // row

  // Check bounds: ensure we're within the output matrix C (M×K)
  if ((y < M) && (x < K)) {
    int val = 0;

    // Compute dot product: C[y][x] = sum(A[y][i] * B[i][x]) for i=0 to N-1
    for (size_t i = 0; i < N; i++) {
      val += d_a[y * N + i] * d_b[i * K + x];
    }

    // Store result in C[y][x]
    d_out[y * K + x] = val;
  }
}

#endif // !GMEM_CUH_
