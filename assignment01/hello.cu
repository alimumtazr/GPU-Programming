#include <stdio.h>

// CUDA kernel
__global__ void hello_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    // Launch kernel: 1 block, 1 thread
    hello_gpu<<<1, 1>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}


