#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>
using namespace std;

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cerr << "USAGE: ./data_generator <rows_A> <cols_A> <cols_B> <output_file>\n";
        cerr << "  Matrix A: rows_A x cols_A\n";
        cerr << "  Matrix B: cols_A x cols_B\n";
        return 1;
    }

    int rowsA = stoi(argv[1]);
    int colsA = stoi(argv[2]);   /* also rows of B */
    int colsB = stoi(argv[3]);
    string outFile = argv[4];

    float *matA = nullptr, *matB = nullptr;
    try {
        matA = new float[(size_t)rowsA * colsA];
        matB = new float[(size_t)colsA * colsB];
    } catch (const bad_alloc& e) {
        cerr << "Memory allocation failed\n";
        return 1;
    }

    srand((unsigned int)time(0));
    for (int i = 0; i < rowsA * colsA; i++)
        matA[i] = (float)rand() / (float)RAND_MAX;
    for (int i = 0; i < colsA * colsB; i++)
        matB[i] = (float)rand() / (float)RAND_MAX;

    ofstream f(outFile);
    if (!f.is_open()) {
        cerr << "Could not open output file: " << outFile << "\n";
        return 1;
    }

    /* Matrix A */
    f << "# Matrix A: " << rowsA << " x " << colsA << "\n";
    f << "rows_A=" << rowsA << "\n";
    f << "cols_A=" << colsA << "\n\n";
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            f << matA[i * colsA + j];
            if (j < colsA - 1) f << " ";
        }
        f << "\n";
    }

    /* Matrix B */
    f << "\n# Matrix B: " << colsA << " x " << colsB << "\n";
    f << "rows_B=" << colsA << "\n";
    f << "cols_B=" << colsB << "\n\n";
    for (int i = 0; i < colsA; i++) {
        for (int j = 0; j < colsB; j++) {
            f << matB[i * colsB + j];
            if (j < colsB - 1) f << " ";
        }
        f << "\n";
    }

    f.close();
    cout << "Generated: " << outFile << "  (A: " << rowsA << "x" << colsA
         << ", B: " << colsA << "x" << colsB << ")\n";

    delete[] matA;
    delete[] matB;
    return 0;
}
