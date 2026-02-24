NVCC      = nvcc
NVCCFLAGS = -O3 -Xcompiler -Wall,-Wextra -arch=sm_86
TARGET    = matmul_gpu

all: $(TARGET)

$(TARGET): task3.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) task3.cu

clean:
	rm -f $(TARGET)
