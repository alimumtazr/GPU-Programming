#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s | File: %s | Line: %d\n",
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

int main() {
    int numDevices = 0;
    gpuErrchk( cudaGetDeviceCount(&numDevices) );
    printf("Found %d CUDA-capable GPU(s) on this system\n", numDevices);

    for (int gpuIdx = 0; gpuIdx < numDevices; ++gpuIdx) {
        cudaDeviceProp props;
        gpuErrchk( cudaGetDeviceProperties(&props, gpuIdx) );

        printf("\n=== GPU %d: \"%s\" ===\n", gpuIdx, props.name);

        /* ---- Identity ---- */
        printf("\n  [Architecture]\n");
        printf("  Compute capability:                  %d.%d — CUDA architecture version (major.minor)\n", props.major, props.minor);

        /* ---- Compute resources ---- */
        printf("\n  [Compute Resources]\n");
        printf("  Streaming multiprocessors (SMs):     %d   — Number of parallel processing units\n",     props.multiProcessorCount);
        printf("  CUDA cores per SM:                   128  — Fixed for Ampere (GA106) architecture\n");
        printf("  Warp size:                           %d   — Threads executed in lockstep per warp\n",   props.warpSize);
        printf("  Max threads per block:               %d   — Upper thread count limit per block\n",      props.maxThreadsPerBlock);
        printf("  Registers per block:                 %d   — Hardware registers available per block\n",  props.regsPerBlock);
        printf("  Core clock rate:                     %.2f GHz — GPU shader/core frequency\n",           props.clockRate * 1e-6f);

        /* ---- Thread hierarchy limits ---- */
        printf("\n  [Thread Hierarchy Limits]\n");
        printf("  Max block dimensions  (x, y, z):     %d x %d x %d — Per-dimension block size limits\n",
               props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("  Max grid dimensions   (x, y, z):     %d x %d x %d — Per-dimension grid size limits\n",
               props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);

        /* ---- Memory ---- */
        printf("\n  [Memory]\n");
        printf("  Global memory:                       %.2f GB — Total device DRAM available\n",          (float)props.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Constant memory:                     %.2f KB — Read-only cache memory per device\n",    (float)props.totalConstMem / 1024.0f);
        printf("  Shared memory per block:             %.2f KB — Fast on-chip memory per thread block\n", (float)props.sharedMemPerBlock / 1024.0f);
        printf("  L2 cache size:                       %.2f MB — Size of the device L2 cache\n",          (float)props.l2CacheSize / (1024.0f * 1024.0f));
        printf("  Memory bus width:                    %d-bit  — Width of the memory interface\n",        props.memoryBusWidth);
        printf("  Memory clock rate:                   %d MHz  — VRAM operating frequency\n",             props.memoryClockRate / 1000);

        /* ---- Derived metrics ---- */
        double memClockHz    = props.memoryClockRate * 1000.0;
        double busWidthBytes = props.memoryBusWidth  / 8.0;
        double peakBandwidth = (2.0 * memClockHz * busWidthBytes) / 1e9;  /* x2 for DDR */

        const int cudaCoresPerSM = 128;                                    /* Ampere GA106 */
        double    coreClockGHz   = props.clockRate * 1e-6;
        double    peakFP32       = (double)props.multiProcessorCount
                                   * cudaCoresPerSM * coreClockGHz * 2.0; /* x2 for FMA */

        printf("\n  [Derived Metrics]\n");
        printf("  Peak memory bandwidth:               %.2f GB/s  — (2 x mem_clock x bus_width)\n",  peakBandwidth);
        printf("  Peak FP32 throughput:                %.2f GFLOPS — (SMs x cores/SM x clock x 2)\n", peakFP32);
    }

    return 0;
}
