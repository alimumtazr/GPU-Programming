NVCC      = nvcc
NVCCFLAGS = -O3 -Xcompiler -Wall,-Wextra -arch=sm_86
TILETEST  = tiletest
TILED     = matmul_tiled

all: $(TILETEST) $(TILED)

$(TILETEST): task5_tiletest.cu
	$(NVCC) $(NVCCFLAGS) -o $(TILETEST) task5_tiletest.cu

$(TILED): task5_tiled.cu
	$(NVCC) $(NVCCFLAGS) -o $(TILED) task5_tiled.cu

clean:
	rm -f $(TILETEST) $(TILED) tile_timings.csv tiled_timings.csv \
	      best_tile_size.png comparison.png
