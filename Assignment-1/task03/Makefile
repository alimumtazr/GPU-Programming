# Makefile for GPU matrix addition
NVCC = nvcc
CFLAGS = -O2 -Xcompiler -Wall -std=c++11

all: matrix_add_gpu

matrix_add_gpu: matrix_add_gpu.cu matrix_add_gpu.h
	$(NVCC) $(CFLAGS) -arch=sm_75 -o matrix_add_gpu matrix_add_gpu.cu

clean:
	rm -f matrix_add_gpu


