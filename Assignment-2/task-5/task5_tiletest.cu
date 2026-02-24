#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <new>
#include <cmath>

using namespace std;

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

/* ── tiled kernel — templated on tile size ───────────────────────────────── */
template <int TILE>
__global__ void tiledKernel(const float *A, const float *B, float *C,
                             int rowsA, int colsB, int colsA) {
    __shared__ float Ads[TILE][TILE];
    __shared__ float Bds[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int numPhases = (int)ceilf((float)colsA / TILE);

    for (int phase = 0; phase < numPhases; phase++) {
        int aCol = phase * TILE + threadIdx.x;
        int bRow = phase * TILE + threadIdx.y;

        Ads[threadIdx.y][threadIdx.x] = (row < rowsA && aCol < colsA)
                                        ? A[row * colsA + aCol] : 0.0f;
        Bds[threadIdx.y][threadIdx.x] = (bRow < colsA && col < colsB)
                                        ? B[bRow * colsB + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += Ads[threadIdx.y][k] * Bds[k][threadIdx.x];
        __syncthreads();
    }

    if (row < rowsA && col < colsB)
        C[row * colsB + col] = acc;
}

/* ── run one tile size, return time in microseconds ─────────────────────── */
template <int TILE>
double runTileTest(const float *A_h, const float *B_h,
                   int rowsA, int colsA, int colsB) {
    float *A_d, *B_d, *C_d;
    float *C_h = allocateMat(rowsA, colsB);

    gpuErrchk( cudaMalloc((void**)&A_d, (size_t)rowsA * colsA * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&B_d, (size_t)colsA * colsB * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&C_d, (size_t)rowsA * colsB * sizeof(float)) );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gpuErrchk( cudaMemcpy(A_d, A_h, (size_t)rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(B_d, B_h, (size_t)colsA * colsB * sizeof(float), cudaMemcpyHostToDevice) );

    dim3 dimBlock(TILE, TILE, 1);
    dim3 dimGrid((colsB + TILE - 1) / TILE, (rowsA + TILE - 1) / TILE, 1);
    tiledKernel<TILE><<<dimGrid, dimBlock>>>(A_d, B_d, C_d, rowsA, colsB, colsA);
    gpuErrchk( cudaGetLastError() );

    gpuErrchk( cudaMemcpy(C_h, C_d, (size_t)rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost) );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double elapsed = ms * 1000.0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    delete[] C_h;

    return elapsed;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "USAGE: ./tiletest <input_file>\n";
        return 1;
    }

    float *A = nullptr, *B = nullptr;
    int rowsA = 0, colsA = 0, rowsB = 0, colsB = 0;

    if (!loadMatrices(argv[1], A, B, rowsA, colsA, rowsB, colsB))
        return 1;

    if (colsA != rowsB) {
        cerr << "Dimension mismatch\n";
        delete[] A; delete[] B;
        return 1;
    }

    cout << "Testing tile sizes on "
         << rowsA << "x" << colsA << " x " << colsA << "x" << colsB << "\n\n";

    double t8  = runTileTest<8> (A, B, rowsA, colsA, colsB);
    double t16 = runTileTest<16>(A, B, rowsA, colsA, colsB);
    double t32 = runTileTest<32>(A, B, rowsA, colsA, colsB);

    cout << fixed << setprecision(2);
    cout << "Tile Size  8x8  : " << t8  << " us"
         << "  (shared mem: " << 2*8*8*4   << " bytes)\n";
    cout << "Tile Size 16x16 : " << t16 << " us"
         << "  (shared mem: " << 2*16*16*4 << " bytes)\n";
    cout << "Tile Size 32x32 : " << t32 << " us"
         << "  (shared mem: " << 2*32*32*4 << " bytes)\n";
    cout << "(Shared memory limit on RTX A2000: 49152 bytes = 48 KB)\n";
    cout << "(Max threads/block = 1024, so TILE > 32 is not possible)\n";

    /* pick fastest based on measured results */
    double times[3] = {t8, t16, t32};
    int    tiles[3] = {8, 16, 32};
    int best = 0;
    for (int i = 1; i < 3; i++)
        if (times[i] < times[best]) best = i;

    cout << "\n=> Fastest tile size: " << tiles[best] << "x" << tiles[best]
         << " (" << times[best] << " us)\n";

    /* write tile_timings.csv */
    ofstream csv("tile_timings.csv");
    csv << "tile_size,time_us\n";
    csv << "8,"  << t8  << "\n";
    csv << "16," << t16 << "\n";
    csv << "32," << t32 << "\n";
    csv.close();
    cout << "\nTile timings written to tile_timings.csv\n";

    delete[] A;
    delete[] B;
    return 0;
}
