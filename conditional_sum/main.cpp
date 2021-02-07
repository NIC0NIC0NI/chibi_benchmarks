#include <cstdint>
#include <cassert>
#include <random>
#include <functional>
#include <algorithm>
#include <cstdio>
#include <sys/time.h>
#include <immintrin.h>

uint32_t conditional_sum(uint32_t* x, size_t n);

int main() {
    int size = 4321, rep = 1000000;
    uint32_t a = 0, b = 0;
    uint32_t *src = (uint32_t *) _mm_malloc(size * sizeof(uint32_t), 64);
    std::mt19937 gen;
    auto rd = std::bind(std::uniform_int_distribution<uint32_t>(0, 255), std::ref(gen));
    for (unsigned int i = 0; i < size - 1; i++) {
        src[i] = rd();
    }

    struct timeval tv_start;
    struct timeval tv_end;
    double t_ms;
    // warm up cache
    conditional_sum(src, size);

    gettimeofday(&tv_start, NULL);

    for (int i = 0; i < rep; i++) {
        a += conditional_sum(src, size);
    }

    gettimeofday(&tv_end, NULL);
    
    t_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1e3 + (tv_end.tv_usec - tv_start.tv_usec) * 1e-3;
    std::printf("Unordered:                            %.3lf ms\n", t_ms);

    std::sort(src, src + size);

    gettimeofday(&tv_start, NULL);

    for (int i = 0; i < rep; i++) {
        b += conditional_sum(src, size);
    }

    gettimeofday(&tv_end, NULL);

    assert(a == b);
    
    t_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1e3 + (tv_end.tv_usec - tv_start.tv_usec) * 1e-3;
    std::printf("Ordered:                              %.3lf ms\n", t_ms);
    std::printf("Size:                                 %ld\n", size);
    std::printf("Repetitions:                          %ld\n", rep);
    std::printf("Result:                               %ld\n", a);


    _mm_free(src);
}


