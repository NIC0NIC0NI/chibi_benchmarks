default:benchmark_avx512 benchmark_avx2 benchmark_sse benchmark_auto
asm: benchmark_avx512.s benchmark_avx2.s benchmark_sse.s benchmark_auto.s

CC = gcc
CXX = g++
AVX512OPT = -DAVX512 -mavx512f -mavx512bw -O2
AVX2OPT = -DAVX2 -mavx2 -O2
SSEOPT = -DSSE -msse4 -O2
AUTOOPT = -mbranch-cost=5 -O3
OPT =

benchmark_avx512:benchmark.c main.cpp
	$(CXX) $(OPT) $(AVX512OPT) $^ -o $@

benchmark_avx512.s:benchmark.c
	$(CC) $(OPT) $(AVX512OPT) -S $^ -o $@

benchmark_avx2:benchmark.c main.cpp
	$(CXX) $(OPT) $(AVX2OPT) $^ -o $@

benchmark_avx2.s:benchmark.c
	$(CC) $(OPT) $(AVX2OPT) -S $^ -o $@

benchmark_sse:benchmark.c main.cpp
	$(CXX) $(OPT) $(SSEOPT) $^ -o $@

benchmark_sse.s:benchmark.c
	$(CC) $(OPT) $(SSEOPT) -S $^ -o $@

benchmark_auto:benchmark.c main.cpp
	$(CXX) $(OPT) $(AUTOOPT) $^ -o $@

benchmark_auto.s:benchmark.c
	$(CC) $(OPT) $(AUTOOPT) -S $^ -o $@
