#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Allocating a float array
float* alloc(int r, int c) {
    float* m = new float[r * c];
    if (!m) {
        cerr << "Memory allocation failed\n";
        exit(EXIT_FAILURE);
    }
    return m;
}

// Reading a matrix from file Stream
void readMatrix(ifstream& fin, float* m, int r, int c) {
    string line;
    for (int i = 0; i < r; ++i) {
        getline(fin, line);
        stringstream ss(line);
        for (int j = 0; j < c; ++j) {
            ss >> m[i * c + j];
        }
    }
}

// Printing matrix to output stream
void printMatrix(float* m, int r, int c, ostream& out) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j)
            out << m[i * c + j] << " ";
        out << "\n";
    }
}

// Element-wise addition
float* addMatrix(float* a, float* b, int r, int c) {
    float* res = alloc(r, c);
    for (int i = 0; i < r * c; ++i)
        res[i] = a[i] + b[i];
    return res;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <inputfile> [outputfile]\n";
        return 1;
    }

    string infile = argv[1];
    string outfile = (argc >= 3) ? argv[2] : "";

    ifstream fin(infile);
    if (!fin.is_open()) {
        cerr << "Cannot open input file: " << infile << "\n";
        return 1;
    }

    int r, c;
    fin >> r >> c;
    fin.ignore(); // skipping remaining newline

    float *A = alloc(r, c);
    float *B = alloc(r, c);

    // Reading matrices
    readMatrix(fin, A, r, c);
    fin.ignore(); // skip blank line
    readMatrix(fin, B, r, c);
    fin.close();

    // Time element-wise addition
    auto t1 = high_resolution_clock::now();
    float* C = addMatrix(A, B, r, c);
    auto t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1);

    // Output
    if (!outfile.empty()) {
        ofstream fout(outfile);
        fout << r << " " << c << "\n";
        fout << "Time (μs): " << duration.count() << "\n\n";
        printMatrix(C, r, c, fout);
        fout.close();
    } else {
        cout << r << " " << c << "\n";
        cout << "Time (μs): " << duration.count() << "\n\n";
        printMatrix(C, r, c, cout);
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
