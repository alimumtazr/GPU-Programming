#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// Allocate GPU memory
float* allocGPU(int r, int c) {
    float* m;
    cudaError_t err = cudaMalloc((void**)&m, r * c * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "GPU allocation failed: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
    return m;
}

// Copy matrix to/from GPU
void copyMatrix(float* dst, float* src, int r, int c, cudaMemcpyKind dir) {
    cudaError_t err = cudaMemcpy(dst, src, r * c * sizeof(float), dir);
    if (err != cudaSuccess) {
        cerr << "Matrix copy failed: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// GPU kernel for element-wise addition
__global__ void addKernel(float* A, float* B, float* C, int r, int c) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < r && j < c) {
        int idx = i * c + j;
        C[idx] = A[idx] + B[idx];
    }
}

// Allocate host memory
float* allocHost(int r, int c) {
    float* m = new float[r * c];
    if (!m) {
        cerr << "Host allocation failed\n";
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
    dim3 gridDim((c + 15) / 16, (r + 15) / 16);
    addKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, r, c);
    cudaDeviceSynchronize();

    copyMatrix(C, C_d, r, c, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop); // ms includes transfers + kernel

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms * 1000; // μs
}

int main() {
    int sizes[] = {2000, 4000, 6000, 8000, 10000};

    ofstream csv("task4_times.csv", ios::app); // append GPU times
    // CSV header already created by CPU file

    for (int sz : sizes) {
        int r = sz, c = sz;
        float *A = allocHost(r, c);
        float *B = allocHost(r, c);
        float *C = allocHost(r, c);

        fillMatrix(A, r, c, true);
        fillMatrix(B, r, c, false);

        float gpu_time = matrixAddGPU(A, B, C, r, c);

        // Append GPU time to the corresponding row
        // We'll read CSV, update GPU column
        cout << "GPU Size " << sz << " done, Time: " << gpu_time << " μs\n";

        delete[] A; delete[] B; delete[] C;
    }

    csv.close();
    return 0;
}
