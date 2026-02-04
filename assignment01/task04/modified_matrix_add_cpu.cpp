#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Allocate a float array
float* alloc(int r, int c) {
    float* m = new float[r * c];
    if (!m) {
        cerr << "Memory allocation failed\n";
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

int main() {
    int sizes[] = {2000, 4000, 6000, 8000, 10000};

    ofstream csv("task4_times.csv");
    csv << "size,cpu_time_us,gpu_time_us\n"; // header for combined CSV

    for (int sz : sizes) {
        int r = sz, c = sz;
        float *A = alloc(r, c);
        float *B = alloc(r, c);
        float *C = alloc(r, c);

        fillMatrix(A, r, c, true);
        fillMatrix(B, r, c, false);

        // Measure CPU time
        auto t1 = high_resolution_clock::now();
        for (int i = 0; i < r * c; ++i) {
            C[i] = A[i] + B[i];
        }
        auto t2 = high_resolution_clock::now();
        auto cpu_time = duration_cast<microseconds>(t2 - t1).count();

        csv << sz << "," << cpu_time << ",0\n"; // GPU time 0 for now
        cout << "CPU Size " << sz << " done, Time: " << cpu_time << " μs\n";

        delete[] A; delete[] B; delete[] C;
    }

    csv.close();
    return 0;
}
