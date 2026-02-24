NVCC      = nvcc
CXX       = g++
NVCCFLAGS = -O3 -Xcompiler -Wall,-Wextra -arch=sm_86
CXXFLAGS  = -O3 -Wall -Wextra -std=c++17
CPU_TARGET = matmul_cpu
GPU_TARGET = matmul_gpu

all: $(CPU_TARGET) $(GPU_TARGET)

$(CPU_TARGET): task4_cpu.cpp
	$(CXX) $(CXXFLAGS) -o $(CPU_TARGET) task4_cpu.cpp

$(GPU_TARGET): task4_gpu.cu
	$(NVCC) $(NVCCFLAGS) -o $(GPU_TARGET) task4_gpu.cu

clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET) cpu_timings.csv gpu_timings.csv comparison.png
