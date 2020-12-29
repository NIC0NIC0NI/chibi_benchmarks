default:benchmark_avx512 benchmark_avx2 benchmark_sse
asm: benchmark_avx512.s benchmark_avx2.s benchmark_sse.s

CC = gcc
AVX512OPT = -DAVX512 -mavx512f -mavx512bw
AVX2OPT = -DAVX2 -mavx2
SSEOPT = -DSSE -msse4
OPT = -O2 -std=c11

benchmark_avx512:benchmark.c main.c
	$(CC) $(OPT) $(AVX512OPT) $^ -o $@

benchmark_avx512.s:benchmark.c
	$(CC) $(OPT) $(AVX512OPT) -S $^ -o $@

benchmark_avx2:benchmark.c main.c
	$(CC) $(OPT) $(AVX2OPT) $^ -o $@

benchmark_avx2.s:benchmark.c
	$(CC) $(OPT) $(AVX2OPT) -S $^ -o $@

benchmark_sse:benchmark.c main.c
	$(CC) $(OPT) $(SSEOPT) $^ -o $@

benchmark_sse.s:benchmark.c
	$(CC) $(OPT) $(SSEOPT) -S $^ -o $@

