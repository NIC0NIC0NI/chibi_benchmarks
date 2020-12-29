#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<sys/time.h>
#include<immintrin.h>

void clamp_arithmetic_logic(int32_t* x, size_t n);
void clamp_compare_select(int32_t* x, size_t n);

#define REPETITION 1000000ull
#define SIZE       (4321ull)

int main() {
    double t1, t2;
    struct timeval start, stop;
    int32_t *xx = (int32_t*)_mm_malloc(sizeof(int32_t) * SIZE, 64);
    int32_t *yy = (int32_t*)_mm_malloc(sizeof(int32_t) * SIZE, 64);
    int32_t *ref = (int32_t*)_mm_malloc(sizeof(int32_t) * SIZE, 64);
    for(size_t i = 0; i < SIZE; ++i) {
        xx[i] = rand() % 512 - 128;
    }

    // 热 cache
    memcpy(ref, xx, sizeof(int32_t) * SIZE);
    clamp_arithmetic_logic(ref, SIZE);
    memcpy(yy, xx, sizeof(int32_t) * SIZE);
    clamp_compare_select(yy, SIZE);
    assert(memcmp(yy, ref, sizeof(int32_t) * SIZE) == 0);
    _mm_free(ref);

    // benchmark arithmetics
    memcpy(yy, xx, sizeof(int32_t) * SIZE);

    gettimeofday(&start, NULL);
    for(size_t i = 0; i < REPETITION; ++i) {
        clamp_arithmetic_logic(yy, SIZE);
    }
    gettimeofday(&stop, NULL);

    t1 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;

    // benchmark comparison
    memcpy(yy, xx, sizeof(int32_t) * SIZE);

    gettimeofday(&start, NULL);
    for(size_t i = 0; i < REPETITION; ++i) {
        clamp_compare_select(yy, SIZE);
    }
    gettimeofday(&stop, NULL);

    t2 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;
    _mm_free(xx);
    _mm_free(yy);

    printf("Clamp by arithmetics and logic:    %.3f ms\n", t1);
    printf("Clamp by comparison and selection: %.3f ms\n", t2);
    printf("Array size:                        %llu\n", SIZE);
    printf("Repetitions:                       %llu\n", REPETITION);

    return 0;
}
