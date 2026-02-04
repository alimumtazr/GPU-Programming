# Task 01 — GPU Coding Environment Setup

## Environment
The GPU programming environment is set up using **Google Colab**, which provides a Linux-based virtual machine with **NVIDIA GPU access**.

---

## GPU Hardware Details
- **GPU Model:** NVIDIA Tesla T4  
- **GPU Memory:** 15 GB  
- **Driver Version:** 550.54.15  
- **CUDA Version (Driver):** 12.4  

Verified using:
```bash
!nvidia-smi
```

## CUDA Toolchain
- **CUDA Compilation Tools:** release 12.5
- **NVCC Version:** 12.5.82

Verified using:
```bash
!nvcc --version
```

## CUDA Program Verification
A simple CUDA kernel is executed to verify GPU functionality. The kernel prints a message from the GPU to confirm execution.

**hello.cu:**
```cuda
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
```

**Compilation:**
```bash
!nvcc -arch=sm_75 hello.cu -o hello
```

**Execution:**
```bash
!./hello
```

**Output:**
```
Hello from GPU!
```

