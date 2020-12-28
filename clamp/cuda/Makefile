default:benchmark_cuda
asm: benchmark_cuda.ptx

CUDAC = nvcc
OPT = -O2

benchmark_cuda:main.cu benchmarks.cu
	$(CUDAC) $(OPT) $^ -o $@

benchmark_cuda.ptx:benchmarks.cu
	$(CUDAC) $(OPT) -ptx $^ -o $@

clean:
	rm benchmark_*

