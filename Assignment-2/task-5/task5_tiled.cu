#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <new>
#include <cmath>

using namespace std;

#define TILE_SIZE 32   /* max viable tile size for RTX A2000 */

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
 * Tiled CI vs Naive CI:
 *
 * Naive:  each thread loads its own elements from global memory
 *         => 2*N*M*K global memory reads
 *         CI_naive = 2*N*M*K / ((N*K + K*M + N*M) * 4)
 *
 * Tiled:  each tile of TILE^2 threads loads ONE tile from global memory
 *         and reuses it TILE times via shared memory
 *         => global memory reads reduced by factor of TILE
 *         CI_tiled = CI_naive * TILE_SIZE
 *
 * For TILE=32 and N=M=K=n:
 *   CI_naive = n/6
 *   CI_tiled = 32 * n/6 = ~5.33n
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
              const string& label, double ci_tiled, double ci_naive,
              int rowsA, int colsA, int colsB) {
    out << "# Computational Intensity — Tiled GPU (TILE_SIZE=" << TILE_SIZE << ")\n";
    out << "#   FLOPS        = 2 * N * M * K = "
        << (long long)(2.0 * rowsA * colsA * colsB) << "\n";
    out << "#   Mem traffic  = (N*K + K*M + N*M) * 4 bytes\n";
    out << "#                  (reduced by TILE_SIZE vs naive)\n";
    out << "#   CI (naive)   = " << fixed << setprecision(4) << ci_naive << " FLOPS/byte\n";
    out << "#   CI (tiled)   = " << fixed << setprecision(4) << ci_tiled << " FLOPS/byte\n";
    out << "#   Improvement  = " << fixed << setprecision(2)
        << (ci_tiled / ci_naive) << "x  (= TILE_SIZE)\n";
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

/* ── tiled kernel ────────────────────────────────────────────────────────── */
__global__ void tiledMatmulKernel(const float *A, const float *B, float *C,
                                   int rowsA, int colsB, int colsA) {
    __shared__ float Ads[TILE_SIZE][TILE_SIZE];
    __shared__ float Bds[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;
    int numPhases = (int)ceilf((float)colsA / TILE_SIZE);

    for (int phase = 0; phase < numPhases; phase++) {
        int aCol = phase * TILE_SIZE + threadIdx.x;
        int bRow = phase * TILE_SIZE + threadIdx.y;

        Ads[threadIdx.y][threadIdx.x] = (row < rowsA && aCol < colsA)
                                        ? A[row * colsA + aCol] : 0.0f;
        Bds[threadIdx.y][threadIdx.x] = (bRow < colsA && col < colsB)
                                        ? B[bRow * colsB + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            acc += Ads[threadIdx.y][k] * Bds[k][threadIdx.x];
        __syncthreads();
    }

    if (row < rowsA && col < colsB)
        C[row * colsB + col] = acc;
}

/* ── GPU tiled driver ────────────────────────────────────────────────────── */
void multiplyTiledGPU(const float *A_h, const float *B_h, float *C_h,
                      int rowsA, int colsA, int colsB) {
    float *A_d, *B_d, *C_d;

    gpuErrchk( cudaMalloc((void**)&A_d, (size_t)rowsA * colsA * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&B_d, (size_t)colsA * colsB * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&C_d, (size_t)rowsA * colsB * sizeof(float)) );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuErrchk( cudaMemcpy(A_d, A_h, (size_t)rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(B_d, B_h, (size_t)colsA * colsB * sizeof(float), cudaMemcpyHostToDevice) );

    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid((colsB + TILE_SIZE - 1) / TILE_SIZE,
                 (rowsA + TILE_SIZE - 1) / TILE_SIZE, 1);
    tiledMatmulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, rowsA, colsB, colsA);
    gpuErrchk( cudaGetLastError() );

    gpuErrchk( cudaMemcpy(C_h, C_d, (size_t)rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost) );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double elapsed = ms * 1000.0;

    cout << "Size: " << rowsA << "x" << colsA << "x" << colsB
         << "  GPU Tiled Time (H2D+kernel+D2H): " << elapsed << " us\n";

    /* append to tiled_timings.csv */
    ofstream csv("tiled_timings.csv", ios::app);
    if (csv.is_open()) {
        csv << rowsA << "x" << colsA << "x" << colsB << ","
            << elapsed << "\n";
        csv.close();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "USAGE: ./matmul_tiled <input_file> [output_file]\n";
        return 1;
    }

    float *A = nullptr, *B = nullptr;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

    if (!loadMatrices(argv[1], A, B, rowsA, colsA, rowsB, colsB))
        return 1;

    if (colsA != rowsB) {
        cerr << "Dimension mismatch: A is " << rowsA << "x" << colsA
             << " but B is " << rowsB << "x" << colsB << "\n";
        delete[] A; delete[] B;
        return 1;
    }

    float *C = allocateMat(rowsA, colsB);
    multiplyTiledGPU(A, B, C, rowsA, colsA, colsB);

    double ci_naive = computeCI(rowsA, colsA, colsB);
    double ci_tiled = ci_naive * TILE_SIZE;

    if (argc == 2) {
        writeMat(cout, C, rowsA, colsB, "Result Matrix C",
                 ci_tiled, ci_naive, rowsA, colsA, colsB);
    } else {
        ofstream ofile(argv[2]);
        if (!ofile.is_open()) {
            cerr << "Cannot open output file: " << argv[2] << "\n";
            delete[] A; delete[] B; delete[] C;
            return 1;
        }
        writeMat(ofile, C, rowsA, colsB, "Result Matrix C",
                 ci_tiled, ci_naive, rowsA, colsA, colsB);
        ofile.close();
        cout << "Output written to: " << argv[2] << "\n";
        cout << "CI (naive): " << fixed << setprecision(4) << ci_naive << " FLOPS/byte\n";
        cout << "CI (tiled): " << fixed << setprecision(4) << ci_tiled << " FLOPS/byte\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
