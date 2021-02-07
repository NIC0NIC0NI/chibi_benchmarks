#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <immintrin.h>

void init_table();
void to_upper_compare_select(const char *src, char *dest, size_t len);
void to_upper_arithmetic_logic(const char *src, char *dest, size_t len);
void to_upper_lookup_table(const char *src, char *dest, size_t len);

int main() {
    int size = 3415, rep = 1000000;
    char *src = (char *) _mm_malloc(size, 64);
    char *dest = (char *) _mm_malloc(size, 64);
    char *ref = (char *) _mm_malloc(size, 64);
    for (unsigned int i = 0; i < size - 1; i++) {
        src[i] = (i*29327)%93 + '!';
    }
    src[size - 1] = 0;
    init_table();

    struct timeval tv_start;
    struct timeval tv_end;
    double t_ms;

    to_upper_lookup_table(src, ref, size - 1);
    to_upper_arithmetic_logic(src, dest, size - 1);
    assert(strcmp(dest, ref) == 0);
    to_upper_compare_select(src, dest, size - 1);
    assert(strcmp(dest, ref) == 0);
    _mm_free(ref);

    gettimeofday(&tv_start, NULL);

    for (int i = 0; i < rep; i++) {
        to_upper_lookup_table(src, dest, size - 1);
    }

    gettimeofday(&tv_end, NULL);
    
    t_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1e3 + (tv_end.tv_usec - tv_start.tv_usec) * 1e-3;
    printf("To upper by look-up table:            %.3lf ms\n", t_ms);

    gettimeofday(&tv_start, NULL);

    for (int i = 0; i < rep; i++) {
        to_upper_arithmetic_logic(src, dest, size - 1);
    }

    gettimeofday(&tv_end, NULL);
    
    t_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1e3 + (tv_end.tv_usec - tv_start.tv_usec) * 1e-3;
    printf("To upper by arithmetics and logic:    %.3lf ms\n", t_ms);

    gettimeofday(&tv_start, NULL);

    for (int i = 0; i < rep; i++) {
        to_upper_compare_select(src, dest, size - 1);
    }

    gettimeofday(&tv_end, NULL);
    
    t_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1e3 + (tv_end.tv_usec - tv_start.tv_usec) * 1e-3;
    printf("To upper by comparison and selection: %.3lf ms\n", t_ms);
    printf("String length:                        %ld\n", size - 1);
    printf("Repetitions:                          %ld\n", rep);

    _mm_free(src);
    _mm_free(dest);
}


