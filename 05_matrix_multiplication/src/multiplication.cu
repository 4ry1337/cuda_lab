/*
 * Matrix Multiplication: A × B = C
 *
 * DIMENSION CONVENTIONS:
 * =====================
 * Matrix A: M rows × N columns (M×N)
 * Matrix B: N rows × K columns (N×K)
 * Matrix C: M rows × K columns (M×K) [output]
 *
 * CUDA GRID/THREAD MAPPING:
 * =========================
 * For a matrix with R rows × C columns, the grid is configured as:
 *   dim3 grid{CEIL_DIV(C, 32), CEIL_DIV(R, 32)}
 *                     ^^^^                ^^^^
 *                   columns              rows
 *
 * This is because CUDA convention maps:
 *   - gridDim.x (1st parameter) → blockIdx.x → x-axis → COLUMNS
 *   - gridDim.y (2nd parameter) → blockIdx.y → y-axis → ROWS
 *
 * Inside kernel, each thread computes:
 *   int col = threadIdx.x + blockIdx.x * blockDim.x;  (column index)
 *   int row = threadIdx.y + blockIdx.y * blockDim.y;  (row index)
 *
 * MEMORY LAYOUT (Row-Major):
 * ==========================
 * Element at [row][col] is stored at linear index: row * num_cols + col
 *
 * Example: For 1024×512 matrix, element [5][10] is at: 5 * 512 + 10 = 2570
 */

#include <cassert>

#include "cuda_timer.cuh"
#include "multiplication.cuh"
#include "random_matrix.cuh"
#include "utils.cuh"

void device_properties(cudaDeviceProp &device) {
  printf("  --- General information for device ---\n");
  printf("Name: %s;\n", device.name);
  printf("Compute capability: %d.%d\n", device.major, device.minor);
  printf("Total global memory: %zu\n", device.totalGlobalMem);
  printf("Total constant memory: %zu\n", device.totalConstMem);
  printf("Multiprocessor count: %d\n", device.multiProcessorCount);
  printf("Shared memory per block: %zu\n", device.sharedMemPerBlock);
  printf("Registers per block: %d\n", device.regsPerBlock);
  printf("Threads in warp: %d\n", device.warpSize);
  printf("Max threads Per Block: %d\n", device.maxThreadsPerBlock);
  printf("Max thread dimensions: (%d, %d, %d)\n", device.maxThreadsDim[0],
         device.maxThreadsDim[1], device.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", device.maxGridSize[0],
         device.maxGridSize[1], device.maxGridSize[2]);
  printf("  --- General information for device ---\n\n");
}

// sources:
// [1] https://khushi-411.github.io/multidim_grids_and_data/#link2
// [2] https://siboehm.com/articles/22/CUDA-MMM

#ifdef NAIVE
// Naive matrix multiplication: C = A × B
// A is M×N, B is N×K, C is M×K
__global__ void matrix_multplication(int *d_a, int *d_b, int *d_out, size_t M,
                                     size_t N, size_t K) {
  // Calculate global thread indices
  // x = column index in output matrix C (ranges 0 to K-1)
  // y = row index in output matrix C (ranges 0 to M-1)
  size_t x = threadIdx.x + blockIdx.x * blockDim.x; // column (x-axis)
  size_t y = threadIdx.y + blockIdx.y * blockDim.y; // row (y-axis)

  // Check bounds: ensure we're within the output matrix C (M×K)
  if ((y < M) && (x < K)) {
    int val = 0;

    // Compute dot product: C[y][x] = sum(A[y][i] * B[i][x]) for i=0 to N-1
    // - A[y][i]: element at row y, column i in matrix A (M×N)
    //   Linearized index: y * N + i (row-major storage)
    // - B[i][x]: element at row i, column x in matrix B (N×K)
    //   Linearized index: i * K + x (row-major storage)
    for (size_t i = 0; i < N; i++) {
      val += d_a[y * N + i] * d_b[i * K + x];
    }

    // Store result in C[y][x]
    // Linearized index: y * K + x (output matrix C is M×K)
    d_out[y * K + x] = val;
  }
}
#endif

#ifdef GMEM
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
__global__ void matrix_multplication(int *d_a, int *d_b, int *d_out, size_t M,
                                     size_t N, size_t K) {
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
#endif

#ifdef SMEM
template <const int BLOCKSIZE>
__global__ void matrix_multplication(int *d_a, int *d_b, int *d_out, size_t M,
                                     size_t N, size_t K) {
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
#endif

#ifdef DBLOCK
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
__global__ void matrix_multplication(int *d_a, int *d_b, int *d_out, size_t M,
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
#endif

// CPU reference implementation: C = A × B
// A is M×N, B is N×K, C is M×K
// Used for verification of GPU results
void matrix_multiplication_cpu(int *h_a, int *h_b, int *h_out, size_t M,
                               size_t N, size_t K) {
  // Iterate over all rows of output matrix C
  for (size_t x = 0; x < M; x++) {
    // Iterate over all columns of output matrix C
    for (size_t y = 0; y < K; y++) {
      int val = 0;

      // Compute dot product: C[x][y] = sum(A[x][z] * B[z][y]) for z=0 to N-1
      for (size_t z = 0; z < N; z++) {
        val += h_a[x * N + z] * h_b[z * K + y];
      }

      // Store result in C[x][y]
      h_out[x * K + y] = val;
    }
  }
}

bool compare_results(int *gpu_result, int *cpu_result, size_t total_size,
                     int max_errors_to_show = 10) {
  bool results_match = true;
  int errors_shown = 0;

  for (size_t i = 0; i < total_size; i++) {
    if (gpu_result[i] != cpu_result[i]) {
      if (errors_shown < max_errors_to_show) {
        printf("VERIFICATION: Mismatch at index %zu: GPU=%d, CPU=%d\n", i,
               gpu_result[i], cpu_result[i]);
        errors_shown++;
      }
      results_match = false;
    }
  }

  if (results_match) {
    printf("\nVERIFICATION: Results match!\n");
  } else {
    printf("\nVERIFICATION: Results DO NOT match! (%d+ errors found)\n",
           errors_shown);
  }

  return results_match;
}

void wrapper() {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, device_id);
  // device_properties(device);

  // Matrix multiplication: A × B = C
  // A is M×N, B is N×K, C is M×K
  // Formula: (M rows × N cols) × (N rows × K cols) = (M rows × K cols)
  uint M = 1024, N = 512 + 256, K = 512;

  // Concrete example with current values (M=1024, N=768, K=512):
  //   A[1024×768] × B[768×512] = C[1024×512]
  //   (1024 rows,    (768 rows,    (1024 rows,
  //    768 cols)      512 cols)     512 cols)

  // pinned host matrices (for async copy)
  int *h_out;                                  // Output matrix C (M×K)
  cudaMallocHost(&h_out, M * K * sizeof(int)); // 1024×512 = 524,288 ints
  cuda_error("cudaMallocHost h_out");

  // device matrices
  int *d_a;                              // Matrix A (M×N)
  cudaMalloc(&d_a, M * N * sizeof(int)); // 1024×768 = 786,432 ints
  cuda_error("cudaMalloc d_a");

  int *d_b;                              // Matrix B (N×K)
  cudaMalloc(&d_b, N * K * sizeof(int)); // 768×512 = 393,216 ints
  cuda_error("cudaMalloc d_b");

  int *d_out;                              // Output matrix C (M×K)
  cudaMalloc(&d_out, M * K * sizeof(int)); // 1024×512 = 524,288 ints
  cuda_error("cudaMalloc d_out");

  // random matrix
  unsigned long seed = 42; // time(NULL);
  const int num_states = 1024;
  curandState *d_states;
  cudaMalloc(&d_states, num_states * sizeof(curandState));
  cuda_error("cudaMalloc d_states");

  // streams
  cudaStream_t streams[2];
  cudaStreamCreate(&streams[0]);
  cuda_error("cudaStreamCreate streams[0]");
  cudaStreamCreate(&streams[1]);
  cuda_error("cudaStreamCreate streams[1]");

  // CUDA thread block configuration
  // Each block has 32×32 threads = 1024 threads total
  // - blockDim.x = 32 (threads per block in x-direction, handles columns)
  // - blockDim.y = 32 (threads per block in y-direction, handles rows)
  dim3 threads_per_block(32, 32, 1);

  {
    CudaTimer timer("Initializing cuRAND states");
    init_curand_states<<<CEIL_DIV(num_states, 256), 256, 0, streams[0]>>>(
        d_states, seed, num_states);
    cudaStreamSynchronize(streams[0]);
  }

  {
    CudaTimer timer("Generating random matrices on GPU");

    // Generate matrix A (M×N = 1024 rows × 768 cols)
    // IMPORTANT: dim3{x, y} where x controls columns, y controls rows
    // Grid dimensions: {CEIL_DIV(cols, 32), CEIL_DIV(rows, 32)}
    //                = {CEIL_DIV(768, 32), CEIL_DIV(1024, 32)}
    //                = {24 blocks, 32 blocks}
    generate_random_matrix_int<<<dim3{CEIL_DIV(N, 32), CEIL_DIV(M, 32)},
                                 threads_per_block, 0, streams[0]>>>(
        d_states, d_a, M, N, 100, num_states);
    cuda_error("generate_random_matrix_int A");

    // Generate matrix B (N×K = 768 rows × 512 cols)
    // Grid dimensions: {CEIL_DIV(cols, 32), CEIL_DIV(rows, 32)}
    //                = {CEIL_DIV(512, 32), CEIL_DIV(768, 32)}
    //                = {16 blocks, 24 blocks}
    generate_random_matrix_int<<<dim3{CEIL_DIV(K, 32), CEIL_DIV(N, 32)},
                                 threads_per_block, 0, streams[1]>>>(
        d_states, d_b, N, K, 100, num_states);
    cuda_error("generate_random_matrix_int B");
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  {
    CudaTimer timer("Matrix multiplication on GPU");

    // Compute C = A × B where C is M×K (1024 rows × 512 cols)
    // Grid dimensions: {CEIL_DIV(cols, 32), CEIL_DIV(rows, 32)}
    //                = {CEIL_DIV(512, 32), CEIL_DIV(1024, 32)}
    //                = {16 blocks, 32 blocks}
    //
    // Each thread computes ONE element C[row][col]:
    //   - threadIdx.x + blockIdx.x * blockDim.x = column index (0 to K-1)
    //   - threadIdx.y + blockIdx.y * blockDim.y = row index (0 to M-1)
    //
    // Why {K, M} not {M, K}? Because CUDA convention:
    //   - gridDim.x (1st param) controls blockIdx.x → x-axis → columns
    //   - gridDim.y (2nd param) controls blockIdx.y → y-axis → rows
#ifdef NAIVE
    matrix_multplication<<<dim3{CEIL_DIV(K, 32), CEIL_DIV(M, 32)},
                           threads_per_block, 0, streams[0]>>>(d_a, d_b, d_out,
                                                               M, N, K);
#endif
#ifdef GMEM
    // GMEM kernel uses 1D thread blocks (32*32 = 1024 threads)
    // Instead of 2D blocks like NAIVE (32×32 threads)
    // This allows linear indexing: threadIdx.x maps to [row, col] via:
    //   col = threadIdx.x % 32  (adjacent threads → adjacent columns)
    //   row = threadIdx.x / 32  (for memory coalescing in row-major layout)
    matrix_multplication<32>
        <<<dim3{CEIL_DIV(K, 32), CEIL_DIV(M, 32)}, 32 * 32, 0, streams[0]>>>(
            d_a, d_b, d_out, M, N, K);
#endif
#ifdef SMEM
    matrix_multplication<32>
        <<<dim3{CEIL_DIV(K, 32), CEIL_DIV(M, 32)}, 32 * 32, 0, streams[0]>>>(
            d_a, d_b, d_out, M, N, K);
#endif
#ifdef DBLOCK
    const uint BM = 64;
    const uint BN = 8;
    const uint BK = 64;
    const uint TM = 8;
    // Template params: <BM, BN, BK, TM> (match kernel definition order!)
    // Grid: {rows, cols} because kernel uses blockIdx.x for out_row, blockIdx.y
    // for out_col
    matrix_multplication<BM, BN, BK, TM>
        <<<dim3{CEIL_DIV(M, BM), CEIL_DIV(K, BK)}, (BM * BK) / TM, 0,
           streams[0]>>>(d_a, d_b, d_out, M, N, K);
#endif
    cuda_error("matrix_multplication");
    cudaStreamSynchronize(streams[0]);
  }

  {
    CudaTimer timer("Copying results from device to host");
    cudaMemcpyAsync(h_out, d_out, M * K * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[0]);
    cuda_error("cudaMemcpyAsync d_out to h_out");
    cudaStreamSynchronize(streams[0]);
  }

#ifdef VERIFY_RESULTS
  int *h_a;
  cudaMallocHost(&h_a, M * N * sizeof(int));
  cuda_error("cudaMallocHost h_a");

  int *h_b;
  cudaMallocHost(&h_b, N * K * sizeof(int));
  cuda_error("cudaMallocHost h_b");

  {
    CudaTimer timer("VERIFICATION: Copying matrices from device to host");
    cudaMemcpyAsync(h_a, d_a, M * N * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[0]);
    cuda_error("cudaMemcpyAsync d_a to h_a");
    cudaMemcpyAsync(h_b, d_b, N * K * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[1]);
    cuda_error("cudaMemcpyAsync d_b to h_b");
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  int *result;
  cudaMallocHost(&result, M * K * sizeof(int));
  cuda_error("VERIFICATION: cudaMallocHost result");

  {
    CudaTimer timer("VERIFICATION: Matrix multiplication on CPU");
    matrix_multiplication_cpu(h_a, h_b, result, M, N, K);
  }

  // Compare results
  compare_results(h_out, result, M * K);

  // Cleanup
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(result);
#endif

  // Cleanup
  cudaFreeHost(h_out);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  cudaFree(d_states);
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
}
