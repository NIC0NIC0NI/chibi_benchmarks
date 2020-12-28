default:benchmark_sse benchmark_avx2 benchmark_avx512

asm:benchmark_sse.s benchmark_avx2.s benchmark_avx512.s

CC = gcc
OPT = -O2 -std=c11

benchmark_sse:main.c benchmarks.c
	$(CC) $(OPT) -msse4 -DSSE $^ -o $@

benchmark_avx2:main.c benchmarks.c
	$(CC) $(OPT) -mavx2 -DAVX2 $^ -o $@

benchmark_avx512:main.c benchmarks.c
	$(CC) $(OPT) -mavx512f -DAVX512 $^ -o $@

benchmark_sse.s:benchmarks.c
	$(CC) $(OPT) -msse4 -DSSE -S $^ -o $@

benchmark_avx2.s:benchmarks.c
	$(CC) $(OPT) -mavx2 -DAVX2 -S $^ -o $@

benchmark_avx512.s: benchmarks.c
	$(CC) $(OPT) -mavx512f -DAVX512 -S $^ -o $@

clean:
	rm benchmark_*

