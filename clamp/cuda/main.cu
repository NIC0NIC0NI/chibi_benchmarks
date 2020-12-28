#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>

int32_t clamp_benchmark_1(int32_t* x, size_t n, size_t repetition);
int32_t clamp_benchmark_2(int32_t* x, size_t n, size_t repetition);

#define REPETITION 1000000ull
#define SIZE       1024ull * 8ull

int main() {
    double t1, t2;
    struct timeval start, stop;
    int32_t xx[SIZE], *yy;
    for(size_t i = 0; i < SIZE; ++i) {
        xx[i] = rand() - RAND_MAX / 2;
    }

    cudaMalloc(&yy, sizeof(int32_t) * SIZE);

    // 热 cache
    clamp_benchmark_1(yy, SIZE, 1);
    clamp_benchmark_2(yy, SIZE, 1);

    // benchmark arithmetics
    cudaMemcpy(yy, xx, sizeof(int32_t) * SIZE, cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);

    clamp_benchmark_1(yy, SIZE, REPETITION);

    gettimeofday(&stop, NULL);

    t1 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;

    // benchmark comparison
    cudaMemcpy(yy, xx, sizeof(int32_t) * SIZE, cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);

    clamp_benchmark_2(yy, SIZE, REPETITION);

    gettimeofday(&stop, NULL);

    t2 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;

    cudaFree(yy);

    printf("Clamp by arithmetics:\t%8.3f ms\nClamp by comparison:\t%8.3f ms\n", t1, t2);
    printf("Array size:\t%llu\nRepetitions:\t%llu\n", SIZE, REPETITION);

    return 0;
}
