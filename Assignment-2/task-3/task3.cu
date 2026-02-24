#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <new>

using namespace std;

#define BLOCK_SIZE 32

/* ── error checking ──────────────────────────────────────────────────────── */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s | File: %s | Line: %d\n",
                cudaGetErrorString(code), file, line);
        exit(EXIT_FAILURE);
    }
}

/* ── memory ──────────────────────────────────────────────────────────────── */
float* allocateMat(int rows, int cols) {
    float *mat = nullptr;
    try {
        mat = new float[(size_t)rows * cols];
    } catch (const bad_alloc& e) {
        cerr << "Memory allocation failed (" << rows << "x" << cols << ")\n";
        exit(EXIT_FAILURE);
    }
    return mat;
}

/* ── file parsing ────────────────────────────────────────────────────────── */
int parseKeyVal(const string& line) {
    size_t pos = line.find('=');
    if (pos == string::npos) {
        cerr << "Malformed key=value line: " << line << "\n";
        exit(EXIT_FAILURE);
    }
    return stoi(line.substr(pos + 1));
}

void skipToData(ifstream& f) {
    string line;
    while (f.peek() != EOF) {
        char c = (char)f.peek();
        if (c == '#' || c == '\n' || c == '\r')
            getline(f, line);
        else
            break;
    }
}

void readMat(ifstream& f, float *mat, int rows, int cols) {
    string line, token;
    int i = 0;
    while (i < rows && getline(f, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        int j = 0;
        while (j < cols && getline(ss, token, ' ')) {
            if (!token.empty())
                mat[i * cols + j++] = stof(token);
        }
        i++;
    }
}

bool loadMatrices(const string& path,
                  float *&A, float *&B,
                  int &rowsA, int &colsA,
                  int &rowsB, int &colsB) {
    ifstream f(path);
    if (!f.is_open()) {
        cerr << "Cannot open input file: " << path << "\n";
        return false;
    }

    string line;
    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("rows_A", 0) == 0) { rowsA = parseKeyVal(line); break; }
    }
    while (getline(f, line)) {
        if (line.rfind("cols_A", 0) == 0) { colsA = parseKeyVal(line); break; }
    }
    skipToData(f);
    A = allocateMat(rowsA, colsA);
    readMat(f, A, rowsA, colsA);

    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("rows_B", 0) == 0) { rowsB = parseKeyVal(line); break; }
    }
    while (getline(f, line)) {
        if (line.rfind("cols_B", 0) == 0) { colsB = parseKeyVal(line); break; }
    }
    skipToData(f);
    B = allocateMat(rowsB, colsB);
    readMat(f, B, rowsB, colsB);

    f.close();
    return true;
}

/* ── computational intensity ─────────────────────────────────────────────── */
/*
 * FLOPS      = 2 * N * M * K   (1 multiply + 1 add per k-iteration)
 * Mem traffic = (N*K + K*M + N*M) * 4 bytes  (read A, read B, write C)
 * CI          = FLOPS / Mem traffic
 */
double computeCI(int rowsA, int colsA, int colsB) {
    double flops    = 2.0 * rowsA * colsA * colsB;
    double memBytes = (double)((size_t)rowsA * colsA +
                               (size_t)colsA * colsB +
                               (size_t)rowsA * colsB) * sizeof(float);
    return flops / memBytes;
}

/* ── output ──────────────────────────────────────────────────────────────── */
void writeMat(ostream& out, const float *mat, int rows, int cols,
              const string& label, double ci,
              int rowsA, int colsA, int colsB) {
    out << "# Computational Intensity (Naive GPU)\n";
    out << "#   FLOPS      = 2 * N * M * K = 2 * "
        << rowsA << " * " << colsB << " * " << colsA << " = "
        << (long long)(2.0 * rowsA * colsA * colsB) << "\n";
    out << "#   Mem traffic = (N*K + K*M + N*M) * 4 bytes\n";
    out << "#   CI          = " << fixed << setprecision(4) << ci
        << " FLOPS/byte\n";
    out << "#\n";
    out << "# " << label << ": " << rows << " x " << cols << "\n";
    out << "rows_C=" << rows << "\n";
    out << "cols_C=" << cols << "\n\n";
    out << fixed << setprecision(4);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out << mat[i * cols + j];
            if (j < cols - 1) out << " ";
        }
        out << "\n";
    }
}

/* ── kernel ──────────────────────────────────────────────────────────────── */
__global__ void matmulKernel(const float *A, const float *B, float *C,
                              int rowsA, int colsB, int colsA) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float acc = 0.0f;
        for (int k = 0; k < colsA; k++)
            acc += A[row * colsA + k] * B[k * colsB + col];
        C[row * colsB + col] = acc;
    }
}

/* ── GPU driver ──────────────────────────────────────────────────────────── */
void multiplyMatsGPU(const float *A_h, const float *B_h, float *C_h,
                     int rowsA, int colsA, int colsB) {
    float *A_d, *B_d, *C_d;

    gpuErrchk( cudaMalloc((void**)&A_d, (size_t)rowsA * colsA * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&B_d, (size_t)colsA * colsB * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&C_d, (size_t)rowsA * colsB * sizeof(float)) );

    gpuErrchk( cudaMemcpy(A_d, A_h, (size_t)rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(B_d, B_h, (size_t)colsA * colsB * sizeof(float), cudaMemcpyHostToDevice) );

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 dimGrid((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    matmulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, rowsA, colsB, colsA);
    gpuErrchk( cudaGetLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(C_h, C_d, (size_t)rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost) );

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "USAGE: ./matmul_gpu <input_file> [output_file]\n";
        return 1;
    }

    float *A = nullptr, *B = nullptr;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

    if (!loadMatrices(argv[1], A, B, rowsA, colsA, rowsB, colsB))
        return 1;

    if (colsA != rowsB) {
        cerr << "Dimension mismatch: A is " << rowsA << "x" << colsA
             << " but B is " << rowsB << "x" << colsB << "\n";
        delete[] A;
        delete[] B;
        return 1;
    }

    float *C = allocateMat(rowsA, colsB);
    multiplyMatsGPU(A, B, C, rowsA, colsA, colsB);

    double ci = computeCI(rowsA, colsA, colsB);

    if (argc == 2) {
        writeMat(cout, C, rowsA, colsB, "Result Matrix C", ci, rowsA, colsA, colsB);
    } else {
        ofstream ofile(argv[2]);
        if (!ofile.is_open()) {
            cerr << "Cannot open output file: " << argv[2] << "\n";
            delete[] A; delete[] B; delete[] C;
            return 1;
        }
        writeMat(ofile, C, rowsA, colsB, "Result Matrix C", ci, rowsA, colsA, colsB);
        ofile.close();
        cout << "Output written to: " << argv[2] << "\n";
        cout << "Computational Intensity: " << fixed << setprecision(4) << ci << " FLOPS/byte\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
