#include <stdexcept>

#include "cuda_timer.cuh"
#include "kernels/00_cpu.h"
#include "kernels/01_naive.cuh"
#include "kernels/02_gmem.cuh"
#include "kernels/03_smem.cuh"
#include "kernels/04_dblock.cuh"
#include "kernels/05_ddblock.cuh"
#include "kernels/06_vectorize.cuh"
#include "multiplication.cuh"
#include "random_matrix.cuh"
#include "utils.cuh"

// Sources:
// [1] https://khushi-411.github.io/multidim_grids_and_data/#link2
// [2] https://siboehm.com/articles/22/CUDA-MMM
void wrapper(KernelType kernel, bool verify_results) {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, device_id);
  print_device_properties(device);

  // streams
  cudaStream_t streams[2];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  // A[M][N] B[N][K] C[M][K]
  uint M = 1024, N = 512 + 256, K = 512;

  const size_t a_bytes = M * N * sizeof(int), b_bytes = N * K * sizeof(int),
               c_bytes = M * K * sizeof(int);

  // pinned host matrices (for async copy)
  int *h_c;
  cudaMallocHost(&h_c, c_bytes);

  // device matrices
  int *d_a;
  cudaMalloc(&d_a, a_bytes);

  int *d_b;
  cudaMalloc(&d_b, b_bytes);

  int *d_c;
  cudaMalloc(&d_c, c_bytes);

  // random matrix
  unsigned long seed = 42;
  const int max_val = 101;
  // unsigned long seed = time(NULL);
  const int num_states = 1024;
  curandState *d_states;
  cudaMalloc(&d_states, num_states * sizeof(curandState));

  {
    CudaTimer timer("Initializing cuRAND states");
    init_curand_states<<<CEIL_DIV(num_states, 256), 256, 0, streams[0]>>>(
        d_states, seed, num_states);
    cuda_error("FAILED: Initializing cuRAND states: kernel launch");
    cudaStreamSynchronize(streams[0]);
  }

  {
    // CUDA thread block configuration
    CudaTimer timer("Generating random matrices (d_a, d_b)");

    random_matrix<<<dim3{CEIL_DIV(N, 32), CEIL_DIV(M, 32)}, dim3{32, 32}, 0,
                    streams[0]>>>(num_states, d_states, d_a, M, N, max_val);
    cuda_error("FAIELD: Generating random matrix (d_a): kernel launch");

    random_matrix<<<dim3{CEIL_DIV(K, 32), CEIL_DIV(N, 32)}, dim3{32, 32}, 0,
                    streams[1]>>>(num_states, d_states, d_b, N, K, max_val);
    cuda_error("FAIELD: Generating random matrix (d_b): kernel launch");

    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
  }

  {
    char title[50] = "Matrix multiplication: ";
    CudaTimer timer(strcat(title, to_string(kernel)));
    switch (kernel) {
    case NAIVE: {
      matrix_multplication_naive<<<dim3{CEIL_DIV(M, 32), CEIL_DIV(K, 32)},
                                   dim3{32, 32}, 0, streams[0]>>>(d_a, d_b, d_c,
                                                                  M, N, K);
      cuda_error("FAIELD: MATRIX MULTPLICATION: NAIVE");
      break;
    }
    case GMEM: {
      matrix_multplication_gmem<32>
          <<<dim3{CEIL_DIV(M, 32), CEIL_DIV(K, 32)}, 32 * 32, 0, streams[0]>>>(
              d_a, d_b, d_c, M, N, K);
      cuda_error("FAIELD: MATRIX MULTPLICATION: GMEM");
      break;
    }
    case SMEM: {
      matrix_multplication_smem<32>
          <<<dim3{CEIL_DIV(M, 32), CEIL_DIV(K, 32)}, 32 * 32, 0, streams[0]>>>(
              d_a, d_b, d_c, M, N, K);
      cuda_error("FAIELD: MATRIX MULTPLICATION: SMEM");
      break;
    }
    case DBLOCK: {
      const uint BM = 64;
      const uint BN = 8;
      const uint BK = 64;
      const uint TM = 8;
      matrix_multplication_1d_blocktailing<BM, BN, BK, TM>
          <<<dim3{CEIL_DIV(M, BM), CEIL_DIV(K, BK)}, (BM * BK) / TM, 0,
             streams[0]>>>(d_a, d_b, d_c, M, N, K);
      cuda_error("FAIELD: MATRIX MULTPLICATION: DBLOCK");
      break;
    }
    case DDBLOCK: {
      const uint BM = 64;
      const uint BN = 8;
      const uint BK = 64;
      const uint TM = 8;
      const uint TN = 8;

      matrix_multplication_2d_blocktailing<BM, BN, BK, TM, TN>
          <<<dim3{CEIL_DIV(M, BM), CEIL_DIV(K, BK)}, (BM * BK) / (TM * TN), 0,
             streams[0]>>>(d_a, d_b, d_c, M, N, K);
      cuda_error("FAIELD: MATRIX MULTPLICATION: DDBLOCK");
      break;
    }
    case VECTORIZE: {
      const uint BM = 64;
      const uint BN = 8;
      const uint BK = 64;
      const uint TM = 8;
      const uint TN = 8;

      matrix_multplication_vectorize<BM, BN, BK, TM, TN>
          <<<dim3{CEIL_DIV(M, BM), CEIL_DIV(K, BK)}, (BM * BK) / (TM * TN), 0,
             streams[0]>>>(d_a, d_b, d_c, M, N, K);
      cuda_error("FAIELD: MATRIX MULTPLICATION: VECTORIZE");
      break;
    }
    default:
      throw std::invalid_argument("Unknown kernel number");
    }
    cuda_error("matrix_multplication");
    cudaStreamSynchronize(streams[0]);
  }

  {
    CudaTimer timer("Copying results (d_c to h_c)");
    cudaMemcpyAsync(h_c, d_c, M * K * sizeof(int), cudaMemcpyDeviceToHost,
                    streams[0]);
    cuda_error("cudaMemcpyAsync d_c to h_c");
    cudaStreamSynchronize(streams[0]);
  }

  if (verify_results) {
    int *h_a;
    cudaMallocHost(&h_a, a_bytes);
    cuda_error("cudaMallocHost h_a");

    int *h_b;
    cudaMallocHost(&h_b, b_bytes);
    cuda_error("cudaMallocHost h_b");

    {
      CudaTimer timer("Verifing: copying (d_a to h_a) and (d_b to h_b)");
      cudaMemcpyAsync(h_a, d_a, a_bytes, cudaMemcpyDeviceToHost, streams[0]);
      cuda_error("cudaMemcpyAsync d_a to h_a");
      cudaMemcpyAsync(h_b, d_b, b_bytes, cudaMemcpyDeviceToHost, streams[1]);
      cuda_error("cudaMemcpyAsync d_b to h_b");
      cudaStreamSynchronize(streams[0]);
      cudaStreamSynchronize(streams[1]);
    }

    int *cpu_c;
    cudaMallocHost(&cpu_c, c_bytes);
    cuda_error("cudaMallocHost result");

    {
      CudaTimer timer("Verifing: Matrix multiplication: CPU");
      matrix_multiplication_cpu(h_a, h_b, cpu_c, M, N, K);
    }

    bool results_match = true;
    int errors_shown = 0;

    for (size_t i = 0; i < M * K; i++) {
      if (h_c[i] != cpu_c[i]) {
        if (errors_shown < 10) {
          printf("Verifing: Mismatch: i=%zu, GPU=%d, CPU=%d\n", i, h_c[i],
                 cpu_c[i]);
          errors_shown++;
        }
        results_match = false;
      }
    }

    if (results_match) {
      printf("\nVerifing: Results match!\n");
    } else {
      printf("\nVerifing: Results DO NOT match! (%d+ errors found)\n",
             errors_shown);
    }

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(cpu_c);
  }

  cudaFreeHost(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_states);
  cudaStreamDestroy(streams[0]);
  cudaStreamDestroy(streams[1]);
}
