NVCC     = nvcc
CFLAGS   = -Xcompiler -Wall,-Wextra
TARGET   = task01

all: $(TARGET)

$(TARGET): task1.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) task1.cu

clean:
	rm -f $(TARGET)
