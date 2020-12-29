default:benchmark_sse benchmark_avx2 benchmark_avx512

asm:benchmark_sse.s benchmark_avx2.s benchmark_avx512.s

CC = gcc
AVX512OPT = -DAVX512 -mavx512f -mavx512bw
AVX2OPT = -DAVX2 -mavx2
SSEOPT = -DSSE -msse4
OPT = -O2 -std=c11

benchmark_sse:main.c benchmarks.c
	$(CC) $(OPT) $(SSEOPT) $^ -o $@

benchmark_avx2:main.c benchmarks.c
	$(CC) $(OPT) $(AVX2OPT) $^ -o $@

benchmark_avx512:main.c benchmarks.c
	$(CC) $(OPT) $(AVX512OPT) $^ -o $@

benchmark_sse.s:benchmarks.c
	$(CC) $(OPT) $(SSEOPT) -S $^ -o $@

benchmark_avx2.s:benchmarks.c
	$(CC) $(OPT) $(AVX2OPT) -S $^ -o $@

benchmark_avx512.s: benchmarks.c
	$(CC) $(OPT) $(AVX512OPT) -S $^ -o $@

clean:
	rm benchmark_*

