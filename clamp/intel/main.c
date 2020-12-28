#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<assert.h>
#include<sys/time.h>
#include<immintrin.h>

int32_t clamp_arithmetic(int32_t* x, size_t n);
int32_t clamp_comparison(int32_t* x, size_t n);

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
    clamp_arithmetic(ref, SIZE);
    memcpy(yy, xx, sizeof(int32_t) * SIZE);
    clamp_comparison(yy, SIZE);
    assert(memcmp(yy, ref, sizeof(int32_t) * SIZE) == 0);
    _mm_free(ref);

    // benchmark arithmetics
    memcpy(yy, xx, sizeof(int32_t) * SIZE);

    gettimeofday(&start, NULL);
    for(size_t i = 0; i < REPETITION; ++i) {
        clamp_arithmetic(yy, SIZE);
    }
    gettimeofday(&stop, NULL);

    t1 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;

    // benchmark comparison
    memcpy(yy, xx, sizeof(int32_t) * SIZE);

    gettimeofday(&start, NULL);
    for(size_t i = 0; i < REPETITION; ++i) {
        clamp_comparison(yy, SIZE);
    }
    gettimeofday(&stop, NULL);

    t2 = (stop.tv_sec - start.tv_sec) * 1e3 + (stop.tv_usec - start.tv_usec) * 1e-3;

    printf("Clamp by arithmetics:\t%8.3f ms\nClamp by comparison:\t%8.3f ms\n", t1, t2);
    printf("Array size:\t%lu\nRepetitions:\t%lu\n", SIZE, REPETITION);

    _mm_free(xx);
    _mm_free(yy);
    return 0;
}
