# High-Performance CUDA Matrix Multiplication and Timing using FP16 WMMA (Optimized for T4 because Colab)

## Overview

Impl. of a matmul kernel aiming for maximal speed. Tested on a T4 mainly, and optimized for `sm_75`. Made for a challenge by a friend - the timing part can be removed.
I achieve maximum throughput by leveraging **FP16 Tensor Core operations via WMMA**, along with other optimization techniques.

## Key Features & Optimizations

*   **Tensor Core Acceleration:** Utilizes `nvcuda::wmma` intrinsics to perform the bulk of the multiplication using the GPU's specialized FP16 Tensor Cores, offering significantly higher theoretical FLOPS compared to standard FP32 CUDA cores.
*   **Mixed Precision:** Uses `__half` (FP16) data type for input matrices A and B and shared memory storage to reduce memory bandwidth and leverage Tensor Cores. Accumulation is performed in FP32 (`float`) for better numerical stability.
*   **Shared Memory Tiling:** Implements a classic tiled algorithm where thread blocks cooperatively load tiles of A and B into fast shared memory, maximizing data reuse and minimizing slow global memory access.
*   **Optimized Shared Memory Layout:** Matrix B is loaded into shared memory in a **transposed layout** (`Bs[N][K]`) to potentially improve memory bank efficiency and better align with WMMA's `load_matrix_sync` requirements for `matrix_b` fragments (when using `wmma::col_major`).
*   **Large Compute Tiles:** Each thread block computes a relatively large tile of the output matrix C (e.g., 128x64) to further enhance data reuse from shared memory.
*   **Asynchronous Operations:** Uses CUDA Streams (`cudaStream_t`) and asynchronous memory copies (`cudaMemcpyAsync`) and kernel launches to potentially overlap data transfers with computation.
*   **Warp-Level Collaboration:** Explicitly distributes WMMA operations across the warps within a thread block.
*   **`__launch_bounds__`:** Provides hints to the NVCC compiler regarding kernel launch parameters (threads per block) to potentially enable more aggressive register allocation without sacrificing occupancy.
*   **Vectorized Loads:** Attempts to use `half2` memory transactions when loading data into shared memory to reduce the number of load instructions.
*   **Pointer Aliasing Hint (`__restrict__`):** Informs the compiler that kernel pointer arguments do not alias, potentially enabling further optimizations.
*   **Kernel Warm-up:** Includes an initial untimed kernel execution to ensure caches are warm and GPU clocks are stable before performance measurement.
*   **Performance Measurement:** Uses `cudaEvent_t` for accurate GPU-side timing and calculates theoretical GFLOPS achieved.
*   **Verification:** Checks the result against a mathematically calculated expected value, accounting for the potential precision differences inherent in FP16 computation using an appropriate epsilon.

## Building

```bash
# Command optimized for T4 (sm_75)
nvcc wmma_optimized.cu -o wmma_optimized -arch=sm_75 -O3 --use_fast_math
```

*(Add `-Xcompiler -fopenmp -lgomp` if you want to enable OpenMP for the host-side FP16 conversion, though it's usually fast enough without it).*

## Running

The program will:
1.  Init host matrices A (1.0f) and B (2.0f).
2.  Convert matrices -> FP16.
3.  Allocate memory.
4.  Copy input matrices
5.  Perform a warm-up run of the WMMA kernel.
6.  Execute the timed run of the WMMA kernel.
7.  Copy the FP32 result matrix C back to the host.
8.  Verify the result against the expected value (N * 2.0), printing PASS or FAIL status and error metrics.
9.  Print the execution time of the WMMA kernel in milliseconds.
10. Calculate and print the achieved performance in GFLOPS (based on FP16 Tensor Core operations).
11. Clean up GPU resources.

## Code Structure & Configuration

Key constants defined at the top of `wmma_optimized.cu` control the kernel's behavior:

*   `MATRIX_WIDTH`: Dimension of the square matrices (must be divisible by tile sizes for this version).
*   `WMMA_M`, `WMMA_N`, `WMMA_K`: Dimensions of the fundamental hardware WMMA operation (usually 16x16x16 for FP16 on T4).
*   `BLOCK_TILE_M`, `BLOCK_TILE_N`, `BLOCK_TILE_K`: Dimensions of the larger tiles processed by one thread block in shared memory.
*   `BLOCK_DIM_X`, `BLOCK_DIM_Y`: Threads per CUDA block.

## Precision Considerations

This implementation relies heavily on **FP16 (`__half`) arithmetic**. So, we just accumulate within the kernel by using FP32. The final result is thus FP32. Intermediate mults are done in FP16 though, but an epsilon is used to try to prevent alot of error.
This kernel is a benchmark and assumes matrix dimensions are perfectly divisible by the chosen tile dimensions (`BLOCK_TILE_*`, `WMMA_*`) for simplicity and maximum performance. 


