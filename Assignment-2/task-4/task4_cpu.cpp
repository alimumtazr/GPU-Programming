#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <new>

using namespace std;

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

/* ── computation ─────────────────────────────────────────────────────────── */
void multiplyCPU(const float *A, const float *B, float *C,
                 int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float acc = 0.0f;
            for (int k = 0; k < colsA; k++)
                acc += A[i * colsA + k] * B[k * colsB + j];
            C[i * colsB + j] = acc;
        }
    }
}

/* ── output ──────────────────────────────────────────────────────────────── */
void writeMat(ostream& out, const float *mat, int rows, int cols,
              const string& label) {
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

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "USAGE: ./matmul_cpu <input_file> [output_file]\n";
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

    /* timing */
    auto tStart = chrono::high_resolution_clock::now();
    multiplyCPU(A, B, C, rowsA, colsA, colsB);
    auto tEnd   = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration_cast<chrono::microseconds>(tEnd - tStart).count();

    cout << "Size: " << rowsA << "x" << colsA << "x" << colsB
         << "  CPU Time: " << elapsed << " us\n";

    /* append to cpu_timings.csv */
    ofstream csv("cpu_timings.csv", ios::app);
    if (csv.is_open()) {
        csv << rowsA << "x" << colsA << "x" << colsB << ","
            << elapsed << "\n";
        csv.close();
    }

    /* write result */
    if (argc == 2) {
        writeMat(cout, C, rowsA, colsB, "Result Matrix C");
    } else {
        ofstream ofile(argv[2]);
        if (!ofile.is_open()) {
            cerr << "Cannot open output file: " << argv[2] << "\n";
            delete[] A; delete[] B; delete[] C;
            return 1;
        }
        writeMat(ofile, C, rowsA, colsB, "Result Matrix C");
        ofile.close();
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
