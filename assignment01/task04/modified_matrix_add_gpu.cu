#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>  // <-- include chrono for CPU timing

using namespace std;
using namespace std::chrono; // <-- fix for chrono in .cu file

// Allocating GPU memory
float* allocGPU(int r, int c) {
    float* m;
    cudaError_t err = cudaMalloc((void**)&m, r * c * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "GPU allocation failed: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
    return m;
}

// Copying matrix to/from GPU
void copyMatrix(float* dst, float* src, int r, int c, cudaMemcpyKind dir) {
    cudaError_t err = cudaMemcpy(dst, src, r * c * sizeof(float), dir);
    if (err != cudaSuccess) {
        cerr << "Matrix copy failed: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// GPU kernel for element-wise addition
__global__ void addKernel(float* A, float* B, float* C, int r, int c) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    if (i < r && j < c) {
        int idx = i * c + j;
        C[idx] = A[idx] + B[idx];
    }
}

// Allocate host memory (CPU helper)
float* allocHost(int r, int c) {
    float* m = new float[r * c];
    if (!m) {
        cerr << "Host allocation failed" << endl;
        exit(EXIT_FAILURE);
    }
    return m;
}

// Fill matrix with simple sequence
void fillMatrix(float* m, int r, int c, bool isA) {
    for (int i = 0; i < r * c; ++i) {
        m[i] = isA ? (i % 1000) : ((i % 1000) * 0.5f);
    }
}

// GPU matrix addition including data transfer
float matrixAddGPU(float* A, float* B, float* C, int r, int c) {
    float *A_d = allocGPU(r, c);
    float *B_d = allocGPU(r, c);
    float *C_d = allocGPU(r, c);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    copyMatrix(A_d, A, r, c, cudaMemcpyHostToDevice);
    copyMatrix(B_d, B, r, c, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(c / 16.0), ceil(r / 16.0));
    addKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, r, c);
    cudaDeviceSynchronize();

    copyMatrix(C, C_d, r, c, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop); // includes transfers + kernel

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms * 1000; // convert ms to μs for consistency with CPU CSV
}

int main() {
    int sizes[] = {2000, 4000, 6000, 8000, 10000};

    ofstream csv("task4_times.csv");
    csv << "size,cpu_time_us,gpu_time_us\n"; // header

    for (int sz : sizes) {
        int r = sz, c = sz;
        float *A = allocHost(r, c);
        float *B = allocHost(r, c);
        float *C = allocHost(r, c);

        fillMatrix(A, r, c, true);
        fillMatrix(B, r, c, false);

        // Measure GPU time
        float gpu_time = matrixAddGPU(A, B, C, r, c);

        // Measure CPU time
        auto t1 = high_resolution_clock::now();
        for (int i = 0; i < r * c; ++i) C[i] = A[i] + B[i];
        auto t2 = high_resolution_clock::now();
        auto cpu_time = duration_cast<microseconds>(t2 - t1).count();

        csv << sz << "," << cpu_time << "," << gpu_time << "\n";
        cout << "Size " << sz << " done, CPU: " << cpu_time << " μs, GPU: " << gpu_time << " μs\n";

        delete[] A; delete[] B; delete[] C;
    }

    csv.close();
    return 0;
}

