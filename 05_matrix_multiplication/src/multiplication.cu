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
#include "kernels/00_cpu.h"
#include "kernels/01_naive.cuh"
#include "kernels/02_gmem.cuh"
#include "kernels/03_smem.cuh"
#include "kernels/04_dblock.cuh"
#include "kernels/05_ddblock.cuh"
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

bool compare_results(int *gpu_result, int *cpu_result, size_t total_size,
                     int max_errors_to_show = 10) {
  bool results_match = true;
  int errors_shown = 0;

  for (size_t i = 0; i < total_size; i++) {
    if (gpu_result[i] != cpu_result[i]) {
      if (errors_shown < max_errors_to_show) {
        printf("Mismatch at index %zu: GPU=%d, CPU=%d\n", i, gpu_result[i],
               cpu_result[i]);
        errors_shown++;
      }
      results_match = false;
    }
  }

  if (results_match) {
    printf("\nResults match!\n");
  } else {
    printf("\nResults DO NOT match! (%d+ errors found)\n", errors_shown);
  }

  return results_match;
}

void wrapper(KernelType kernel, bool verify_results) {
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

    switch (kernel) {
    case NAIVE: {
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
      matrix_multplication_naive<<<dim3{CEIL_DIV(K, 32), CEIL_DIV(M, 32)},
                                   threads_per_block, 0, streams[0]>>>(
          d_a, d_b, d_out, M, N, K);
      break;
    }
    case GMEM: {
      // GMEM kernel uses 1D thread blocks (32*32 = 1024 threads)
      // Instead of 2D blocks like NAIVE (32×32 threads)
      // This allows linear indexing: threadIdx.x maps to [row, col] via:
      //   col = threadIdx.x % 32  (adjacent threads → adjacent columns)
      //   row = threadIdx.x / 32  (for memory coalescing in row-major layout)
      matrix_multplication_gmem<32>
          <<<dim3{CEIL_DIV(K, 32), CEIL_DIV(M, 32)}, 32 * 32, 0, streams[0]>>>(
              d_a, d_b, d_out, M, N, K);
      break;
    }
    case SMEM: {
      matrix_multplication_smem<32>
          <<<dim3{CEIL_DIV(K, 32), CEIL_DIV(M, 32)}, 32 * 32, 0, streams[0]>>>(
              d_a, d_b, d_out, M, N, K);
      break;
    }
    case DBLOCK: {
      const uint DBLOCK_BM = 64;
      const uint DBLOCK_BN = 8;
      const uint DBLOCK_BK = 64;
      const uint DBLOCK_TM = 8;
      // Template params: <BM, BN, BK, TM> (match kernel definition order!)
      // Grid: {rows, cols} because kernel uses blockIdx.x for out_row,
      // blockIdx.y for out_col
      matrix_multplication_1d_blocktailing<DBLOCK_BM, DBLOCK_BN, DBLOCK_BK,
                                           DBLOCK_TM>
          <<<dim3{CEIL_DIV(M, DBLOCK_BM), CEIL_DIV(K, DBLOCK_BK)},
             (DBLOCK_BM * DBLOCK_BK) / DBLOCK_TM, 0, streams[0]>>>(
              d_a, d_b, d_out, M, N, K);
      break;
    }
    case DDBLOCK: {
      const uint DDBLOCK_BM = 64;
      const uint DDBLOCK_BN = 8;
      const uint DDBLOCK_BK = 64;
      const uint DDBLOCK_TM = 8;
      const uint DDBLOCK_TN = 8;

      matrix_multplication_2d_blocktailing<DDBLOCK_BM, DDBLOCK_BN, DDBLOCK_BK,
                                           DDBLOCK_TM, DDBLOCK_TN>
          <<<dim3{CEIL_DIV(M, DDBLOCK_BM), CEIL_DIV(K, DDBLOCK_BK)},
             (DDBLOCK_BM * DDBLOCK_BK) / (DDBLOCK_TM * DDBLOCK_TN), 0,
             streams[0]>>>(d_a, d_b, d_out, M, N, K);
      break;
    }
    default:
      throw std::invalid_argument("Unknown kernel number");
    }
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

  if (verify_results) {
    int *h_a;
    cudaMallocHost(&h_a, M * N * sizeof(int));
    cuda_error("cudaMallocHost h_a");

    int *h_b;
    cudaMallocHost(&h_b, N * K * sizeof(int));
    cuda_error("cudaMallocHost h_b");

    {
      CudaTimer timer("Verifing - copying matrices from device to host");
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
    cuda_error("cudaMallocHost result");

    {
      CudaTimer timer("Verifing - Matrix multiplication on CPU");
      matrix_multiplication_cpu(h_a, h_b, result, M, N, K);
    }

    // Compare results
    compare_results(h_out, result, M * K);

    // Cleanup
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(result);
  }

  // Cleanup
  cudaFreeHost(h_out);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  cudaFree(d_states);
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
}
