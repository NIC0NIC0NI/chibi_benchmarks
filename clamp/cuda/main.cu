#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<sys/time.h>

void clamp_arithmetic_logic(int32_t* x, size_t n, size_t repetition);
void clamp_compare_select(int32_t* x, size_t n, size_t repetition);

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
    clamp_arithmetic_logic(yy, SIZE, 1);
    clamp_compare_select(yy, SIZE, 1);

    // benchmark arithmetics and logic
    cudaMemcpy(yy, xx, sizeof(int32_t) * SIZE, cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);

    clamp_arithmetic_logic(yy, SIZE, REPETITION);

    gettimeofday(&stop, NULL);

    t1 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;

    // benchmark comparison and selection
    cudaMemcpy(yy, xx, sizeof(int32_t) * SIZE, cudaMemcpyHostToDevice);

    gettimeofday(&start, NULL);

    clamp_compare_select(yy, SIZE, REPETITION);

    gettimeofday(&stop, NULL);

    t2 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;

    cudaFree(yy);

    printf("Clamp by arithmetics and logic:    %.3f ms\n", t1);
    printf("Clamp by comparison and selection: %.3f ms\n", t2);
    printf("Array size:                        %llu\n", SIZE);
    printf("Repetitions:                       %llu\n", REPETITION);

    return 0;
}
