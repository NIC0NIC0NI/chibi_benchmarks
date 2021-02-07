default:benchmark_sse benchmark_avx2 benchmark_avx512 benchmark_auto

asm:benchmark_sse.s benchmark_avx2.s benchmark_avx512.s benchmark_auto.s

CC = gcc
AVX512OPT = -DAVX512 -mavx512f -mavx512bw -O2
AVX2OPT = -DAVX2 -mavx2 -O2
SSEOPT = -DSSE -msse4 -O2
AUTOOPT = -mbranch-cost=5 -O3
OPT = -std=c11

benchmark_sse:main.c benchmarks.c
	$(CC) $(OPT) $(SSEOPT) $^ -o $@

benchmark_avx2:main.c benchmarks.c
	$(CC) $(OPT) $(AVX2OPT) $^ -o $@

benchmark_avx512:main.c benchmarks.c
	$(CC) $(OPT) $(AVX512OPT) $^ -o $@

benchmark_auto:main.c benchmarks.c
	$(CC) $(OPT) $(AUTOOPT) $^ -o $@

benchmark_sse.s:benchmarks.c
	$(CC) $(OPT) $(SSEOPT) -S $^ -o $@

benchmark_avx2.s:benchmarks.c
	$(CC) $(OPT) $(AVX2OPT) -S $^ -o $@

benchmark_avx512.s: benchmarks.c
	$(CC) $(OPT) $(AVX512OPT) -S $^ -o $@

benchmark_auto.s:benchmarks.c
	$(CC) $(OPT) $(AUTOOPT) -S $^ -o $@

clean:
	rm benchmark_*

