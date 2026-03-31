#ifndef MATRIX_ADD_GPU_H
#define MATRIX_ADD_GPU_H

// Function to perform matrix addition on GPU
// mat1_h, mat2_h, mat3_h: pointers to host matrices
// num_rows, num_cols: dimensions of the matrices
void matrixAdd(float *mat1_h, float *mat2_h, float *mat3_h, int num_rows, int num_cols);

#endif // MATRIX_ADD_GPU_H
