#include <cmath>
#include <stdexcept>
#include <type_traits>

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

template <typename T>
bool values_match(T a, T b, T epsilon = T(1e-5)) {
  if (std::is_floating_point<T>::value) {
    T diff = std::abs(a - b);
    T abs_a = std::abs(a);
    T abs_b = std::abs(b);
    T max_val = (abs_a > abs_b) ? abs_a : abs_b;
    return diff <= epsilon * max_val || diff < epsilon;
  } else {
    return a == b;
  }
}

// Sources:
// [1] https://khushi-411.github.io/multidim_grids_and_data/#link2
// [2] https://siboehm.com/articles/22/CUDA-MMM
template <typename T> void wrapper(KernelType kernel, bool verify_results) {
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, device_id);
  // print_device_properties(device);

  // streams
  cudaStream_t streams[2];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  // A[M][N] B[N][K] C[M][K]
  uint M = 1024, N = 512 + 256, K = 512;

  const size_t a_bytes = M * N * sizeof(T), b_bytes = N * K * sizeof(T),
               c_bytes = M * K * sizeof(T);

  // pinned host matrices (for async copy)
  T *h_c;
  cudaMallocHost(&h_c, c_bytes);

  // device matrices
  T *d_a;
  cudaMalloc(&d_a, a_bytes);

  T *d_b;
  cudaMalloc(&d_b, b_bytes);

  T *d_c;
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

    random_matrix<T><<<dim3{CEIL_DIV(N, 32), CEIL_DIV(M, 32)}, dim3{32, 32}, 0,
                       streams[0]>>>(num_states, d_states, d_a, M, N, max_val);
    cuda_error("FAIELD: Generating random matrix (d_a): kernel launch");

    random_matrix<T><<<dim3{CEIL_DIV(K, 32), CEIL_DIV(N, 32)}, dim3{32, 32}, 0,
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
      matrix_multplication_gmem<T, 32>
          <<<dim3{CEIL_DIV(M, 32), CEIL_DIV(K, 32)}, 32 * 32, 0, streams[0]>>>(
              d_a, d_b, d_c, M, N, K);
      cuda_error("FAIELD: MATRIX MULTPLICATION: GMEM");
      break;
    }
    case SMEM: {
      matrix_multplication_smem<T, 32>
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
      matrix_multplication_1d_blocktailing<T, BM, BN, BK, TM>
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

      matrix_multplication_2d_blocktailing<T, BM, BN, BK, TM, TN>
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

      matrix_multplication_vectorize<T, BM, BN, BK, TM, TN>
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
    cudaMemcpyAsync(h_c, d_c, M * K * sizeof(T), cudaMemcpyDeviceToHost,
                    streams[0]);
    cuda_error("cudaMemcpyAsync d_c to h_c");
    cudaStreamSynchronize(streams[0]);
  }

  if (verify_results) {
    T *h_a;
    cudaMallocHost(&h_a, a_bytes);
    cuda_error("cudaMallocHost h_a");

    T *h_b;
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

    T *cpu_c;
    cudaMallocHost(&cpu_c, c_bytes);
    cuda_error("cudaMallocHost result");

    {
      CudaTimer timer("Verifing: Matrix multiplication: CPU");
      matrix_multiplication_cpu(h_a, h_b, cpu_c, M, N, K);
    }

    bool results_match = true;
    int errors_shown = 0;
    T max_error = 0;

    for (size_t i = 0; i < M * K; i++) {
      if (!values_match(h_c[i], cpu_c[i])) {
        T error = std::abs(h_c[i] - cpu_c[i]);
        if (error > max_error) max_error = error;
        if (errors_shown < 10) {
          if (std::is_floating_point<T>::value) {
            printf("Verifing: Mismatch: i=%zu, GPU=%.6e, CPU=%.6e, diff=%.6e\n", 
                   i, (double)h_c[i], (double)cpu_c[i], (double)error);
          } else {
            printf("Verifing: Mismatch: i=%zu, GPU=%d, CPU=%d\n", 
                   i, (int)h_c[i], (int)cpu_c[i]);
          }
          errors_shown++;
        }
        results_match = false;
      }
    }

    if (results_match) {
      printf("\nVerifing: Results match!\n");
    } else {
      if (std::is_floating_point<T>::value) {
        printf("\nVerifing: Results DO NOT match! (%d+ errors found, max_error=%.6e)\n",
               errors_shown, (double)max_error);
      } else {
        printf("\nVerifing: Results DO NOT match! (%d+ errors found)\n",
               errors_shown);
      }
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

// Explicit template instantiations
template void wrapper<int>(KernelType kernel, bool verify_results);
template void wrapper<float>(KernelType kernel, bool verify_results);
template void wrapper<double>(KernelType kernel, bool verify_results);
