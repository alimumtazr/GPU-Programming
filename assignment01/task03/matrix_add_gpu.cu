#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

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

// allocate host memory (cpu helper)
float* allocHost(int r, int c) {
    float* m = new float[r * c];
    if (!m) {
        cerr << "Host allocation failed" << endl;
        exit(EXIT_FAILURE);
    }
    return m;
}

// Reading matrix from the file stream
void readMatrix(ifstream& fin, float* M, int r, int c) {
    string line;
    for (int i = 0; i < r; ++i) {
        getline(fin, line);
        stringstream ss(line);
        for (int j = 0; j < c; ++j) ss >> M[i * c + j];
    }
}

// Printing the matrix to output stream
void printMatrix(float* M, int r, int c, ostream& out) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) out << M[i * c + j] << " ";
        out << "\n";
    }
}

// GPU matrix addition
void matrixAdd(float* A, float* B, float* C, int r, int c) {
    float *A_d = allocGPU(r, c);
    float *B_d = allocGPU(r, c);
    float *C_d = allocGPU(r, c);

    copyMatrix(A_d, A, r, c, cudaMemcpyHostToDevice);
    copyMatrix(B_d, B, r, c, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(c / 16.0), ceil(r / 16.0));
    addKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, r, c);
    cudaDeviceSynchronize();

    copyMatrix(C, C_d, r, c, cudaMemcpyDeviceToHost);

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <inputfile> [outputfile]\n";
        return 1;
    }

    string infile = argv[1];
    string outfile = (argc >= 3) ? argv[2] : "";

    ifstream fin(infile);
    if (!fin.is_open()) { cerr << "Cannot open file\n"; return 1; }

    int r, c;
    fin >> r >> c;
    fin.ignore();

    float *A = allocHost(r, c);
    float *B = allocHost(r, c);
    float *C = allocHost(r, c);

    readMatrix(fin, A, r, c);
    fin.ignore();
    readMatrix(fin, B, r, c);
    fin.close();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    matrixAdd(A, B, C, r, c);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);

    if (!outfile.empty()) {
        ofstream fout(outfile);
        fout << r << " " << c << "\nTime (ms): " << ms << "\n\n";
        printMatrix(C, r, c, fout);
        fout.close();
    } else {
        cout << r << " " << c << "\nTime (ms): " << ms << "\n\n";
        printMatrix(C, r, c, cout);
    }

    delete[] A; delete[] B; delete[] C;
    return 0;
}
