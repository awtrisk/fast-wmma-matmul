#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h> 
#include <mma.h>       

// WE HATE BOILERPLATE !!
#define CUDA_CHECK(call)                                                            \
do {                                                                                \
    cudaError_t err = call;                                                         \
    if (err != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,           \
                cudaGetErrorString(err));                                           \
        exit(EXIT_FAILURE);                                                         \
    }                                                                               \
} while (0)

// basic consts config
const int MATRIX_WIDTH = 256;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int BLOCK_TILE_M = 64; 
const int BLOCK_TILE_N = 128; 
const int BLOCK_TILE_K = 16;
const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16; 
const int THREADS_PER_BLOCK = BLOCK_DIM_X * BLOCK_DIM_Y;
const int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

static_assert(MATRIX_WIDTH % BLOCK_TILE_M == 0, "Matrix width must be multiple of Block Tile M");
static_assert(MATRIX_WIDTH % BLOCK_TILE_N == 0, "Matrix width must be multiple of Block Tile N");
static_assert(MATRIX_WIDTH % BLOCK_TILE_K == 0, "Matrix width must be multiple of Block Tile K");
static_assert(BLOCK_TILE_M % WMMA_M == 0, "Block Tile M must be multiple of WMMA M");
static_assert(BLOCK_TILE_N % WMMA_N == 0, "Block Tile N must be multiple of WMMA N");
static_assert(BLOCK_TILE_K % WMMA_K == 0, "Block Tile K must be multiple of WMMA K");
static_assert(THREADS_PER_BLOCK % 32 == 0, "Threads per block must be multiple of warp size");

using namespace nvcuda; 
// matmul using wmma for speed!
__global__ void __launch_bounds__(THREADS_PER_BLOCK) 
    wmmaKernelOptimized(const half* __restrict__ A,
                        const half* __restrict__ B,
                        float* __restrict__ C,
                        int width)
{
    __shared__ half As[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ half Bs[BLOCK_TILE_N][BLOCK_TILE_K]; 

    int blockCBaseRow = blockIdx.y * BLOCK_TILE_M;
    int blockCBaseCol = blockIdx.x * BLOCK_TILE_N;

    int thread_ix = threadIdx.x; 
    int thread_iy = threadIdx.y; 
    int thread_linear_id = thread_iy * blockDim.x + thread_ix;
    int warpId = thread_linear_id / 32;       
    int laneId = thread_linear_id % 32;       

    const int wmma_ops_m = BLOCK_TILE_M / WMMA_M; 
    const int wmma_ops_n = BLOCK_TILE_N / WMMA_N; 
    const int total_wmma_ops = wmma_ops_m * wmma_ops_n; 

    const int num_ops_per_warp = total_wmma_ops / WARPS_PER_BLOCK; 
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[num_ops_per_warp];

    #pragma unroll
    for (int i = 0; i < num_ops_per_warp; ++i) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    for (int k_tile_start = 0; k_tile_start < width; k_tile_start += BLOCK_TILE_K) {
        const int A_elements_to_load = BLOCK_TILE_M * BLOCK_TILE_K;
        const int A_loads_per_thread = (A_elements_to_load / 2) / THREADS_PER_BLOCK; 

        half2* As_h2 = reinterpret_cast<half2*>(As);
        const half2* A_global = reinterpret_cast<const half2*>(A);
        const int width_h2 = width / 2; 

        #pragma unroll
        for (int i = 0; i < A_loads_per_thread; ++i) {
            int load_idx_h2 = thread_linear_id * A_loads_per_thread + i;
            int row = load_idx_h2 / (BLOCK_TILE_K / 2); 
            int col = load_idx_h2 % (BLOCK_TILE_K / 2); 

            int gmem_row_start_A = blockCBaseRow + row;
            int gmem_col_start_A = (k_tile_start / 2) + col; 

            As_h2[row * (BLOCK_TILE_K / 2) + col] = A_global[gmem_row_start_A * width_h2 + gmem_col_start_A];
        }

        const int B_elements_to_load = BLOCK_TILE_N * BLOCK_TILE_K;
        const int B_loads_per_thread = (B_elements_to_load / 2) / THREADS_PER_BLOCK; 

        half2* Bs_h2 = reinterpret_cast<half2*>(Bs);
        const half* B_global = B; 

        #pragma unroll
        for (int i = 0; i < B_loads_per_thread; ++i) {
            int load_idx = thread_linear_id * B_loads_per_thread + i; 

            int n_coord = load_idx / (BLOCK_TILE_K / 2); 
            int k_coord_h2 = load_idx % (BLOCK_TILE_K / 2); 

            int gmem_row_start_B = k_tile_start + k_coord_h2 * 2; 
            int gmem_col_start_B = blockCBaseCol + n_coord;       

            half val1 = B_global[gmem_row_start_B * width + gmem_col_start_B];
            half val2 = B_global[(gmem_row_start_B + 1) * width + gmem_col_start_B];

            Bs_h2[n_coord * (BLOCK_TILE_K / 2) + k_coord_h2] = __halves2half2(val1, val2);
        }

        __syncthreads();

        #pragma unroll 
        for (int k_inner = 0; k_inner < BLOCK_TILE_K; k_inner += WMMA_K)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag; 

            #pragma unroll
            for (int i = 0; i < num_ops_per_warp; ++i) {

                int op_idx = warpId * num_ops_per_warp + i; 
                int tile_m_idx = op_idx / wmma_ops_n;      
                int tile_n_idx = op_idx % wmma_ops_n;      

                int shmem_a_row_start = tile_m_idx * WMMA_M;
                int shmem_b_row_start = tile_n_idx * WMMA_N; 
                int k_shmem_offset = k_inner;                   

                wmma::load_matrix_sync(a_frag, &As[shmem_a_row_start][k_shmem_offset], BLOCK_TILE_K); 

                wmma::load_matrix_sync(b_frag, &Bs[shmem_b_row_start][k_shmem_offset], BLOCK_TILE_K); 

                wmma::mma_sync(acc_frag[i], a_frag, b_frag, acc_frag[i]);
            }
        }

        __syncthreads();
    } 

    #pragma unroll
    for (int i = 0; i < num_ops_per_warp; ++i) {
        int op_idx = warpId * num_ops_per_warp + i;
        int tile_m_idx = op_idx / wmma_ops_n;
        int tile_n_idx = op_idx % wmma_ops_n;

        int c_tile_row_start = blockCBaseRow + tile_m_idx * WMMA_M;
        int c_tile_col_start = blockCBaseCol + tile_n_idx * WMMA_N;

        float* C_tile_ptr = C + c_tile_row_start * width + c_tile_col_start;

        wmma::store_matrix_sync(C_tile_ptr, acc_frag[i], width, wmma::mem_row_major);
    }
}

// quant fp32 -> fp16
std::vector<__half> floatToHalf(const std::vector<float>& floatVec) {
    std::vector<__half> halfVec(floatVec.size());
    #pragma omp parallel for 
    for (size_t i = 0; i < floatVec.size(); ++i) {
        halfVec[i] = __float2half(floatVec[i]);
    }
    return halfVec;
}

// result verification for smol cases
bool verifyResultAnalogousWMMA(const std::vector<float>& result, int width, float expectedValue) {
    const float relative_epsilon = 1e-3f; 
    const float absolute_epsilon = relative_epsilon * std::abs(expectedValue); 

    size_t size = (size_t)width * width;
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(result[i] - expectedValue);

        if (diff > absolute_epsilon) {
            fprintf(stderr, "Verification Error at index %zu: got %f expected %f and diff: %f (epsilon: %f)\n",
                    i, result[i], expectedValue, diff, absolute_epsilon);
            return false; 
        }
    }

    return true;
}

// weird things, just calculations, conversions, memory copies and timing.
int main() { 
    const int N = MATRIX_WIDTH;
    const size_t matrixSizeFloats = N * N;
    const size_t matrixSizeBytesFloat = matrixSizeFloats * sizeof(float);
    const size_t matrixSizeBytesHalf = matrixSizeFloats * sizeof(__half);

    std::vector<float> h_A_f32(matrixSizeFloats);
    std::vector<float> h_B_f32(matrixSizeFloats);
    std::vector<float> h_C_wmma(matrixSizeFloats, -1.0f);
    std::fill(h_A_f32.begin(), h_A_f32.end(), 1.0f);
    std::fill(h_B_f32.begin(), h_B_f32.end(), 2.0f);
    float expectedValue = static_cast<float>(N) * 1.0f * 2.0f;
    std::vector<__half> h_A_h = floatToHalf(h_A_f32);
    std::vector<__half> h_B_h = floatToHalf(h_B_f32);

    half *d_A_h = nullptr; 
    half *d_B_h = nullptr; 
    float *d_C_f32 = nullptr; 
    CUDA_CHECK(cudaMalloc(&d_A_h, matrixSizeBytesHalf));
    CUDA_CHECK(cudaMalloc(&d_B_h, matrixSizeBytesHalf));
    CUDA_CHECK(cudaMalloc(&d_C_f32, matrixSizeBytesFloat));

    cudaEvent_t start, stop; 
    cudaStream_t stream; 
    CUDA_CHECK(cudaEventCreate(&start)); 
    CUDA_CHECK(cudaEventCreate(&stop)); 
    CUDA_CHECK(cudaStreamCreate(&stream)); 

    std::cout << "running the funny wmma\n";
    std::cout << "cping data HtoD..." << std::endl;
    CUDA_CHECK(cudaMemcpyAsync(d_A_h, h_A_h.data(), matrixSizeBytesHalf, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_h, h_B_h.data(), matrixSizeBytesHalf, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_C_f32, 0, matrixSizeBytesFloat, stream));

    dim3 numBlocks( (N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (N + BLOCK_TILE_M - 1) / BLOCK_TILE_M );
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);

    std::cout << "warm-up run..." << std::endl;
    wmmaKernelOptimized<<<numBlocks, threadsPerBlock, 0, stream>>>(d_A_h, d_B_h, d_C_f32, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    std::cout << "timed run..." << std::endl;
    CUDA_CHECK(cudaMemsetAsync(d_C_f32, 0, matrixSizeBytesFloat, stream)); 
    CUDA_CHECK(cudaEventRecord(start, stream));

    wmmaKernelOptimized<<<numBlocks, threadsPerBlock, 0, stream>>>(d_A_h, d_B_h, d_C_f32, N);

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaGetLastError());

    std::cout << "cping result DtoH..." << std::endl;
    CUDA_CHECK(cudaMemcpyAsync(h_C_wmma.data(), d_C_f32, matrixSizeBytesFloat, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float wmmaKernelMillis = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&wmmaKernelMillis, start, stop));

    std::cout << "verifying..." << std::endl;
    if (verifyResultAnalogousWMMA(h_C_wmma, N, expectedValue)) { 
        std::cout << "WMMA Kernel Verification PASSED!" << std::endl;
    } else {
        std::cout << "WMMA Kernel Verification FAILED!" << std::endl;
    }

    double gflops = 2.0 * N * N * N / (wmmaKernelMillis / 1000.0) / 1e9;
    std::cout << "\nOptimized WMMA Kernel Time: " << wmmaKernelMillis << " ms" << std::endl;
    std::cout << "Achieved Performance: " << gflops << " GFLOPS (FP16 Tensor Core)" << std::endl;

    std::cout << "\n--- Cleaning up ---" << std::endl;
    CUDA_CHECK(cudaEventDestroy(start)); 
    CUDA_CHECK(cudaEventDestroy(stop)); 
    CUDA_CHECK(cudaStreamDestroy(stream)); 
    CUDA_CHECK(cudaFree(d_A_h)); 
    CUDA_CHECK(cudaFree(d_B_h)); 
    CUDA_CHECK(cudaFree(d_C_f32)); 

    int deviceId; 
    cudaDeviceProp prop; 
    CUDA_CHECK(cudaGetDevice(&deviceId)); 
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId)); 
    std::cout << "\nRunning on GPU: " << prop.name << " (Compute Capability: " << prop.major << "." << prop.minor << ")" << std::endl;

    return 0;
}
