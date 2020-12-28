#include <stdint.h>
#include <immintrin.h>

#ifdef AVX512
int32_t clamp_arithmetic(int32_t* x, size_t n) {
    const __m512i vec0 = _mm512_setzero_epi32();
    const __m512i vec255 = _mm512_set1_epi32(255);
    size_t j = 0;
    if(n >= 64) {
        for(; j < n - 63; j += 64) {
            __m512i x0 = _mm512_load_epi32(&x[j]);
            __m512i x1 = _mm512_load_epi32(&x[j+16]);
            __m512i x2 = _mm512_load_epi32(&x[j+32]);
            __m512i x3 = _mm512_load_epi32(&x[j+48]);
            __m512i y0 = _mm512_and_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec0, x0), 31), x0);
            __m512i y1 = _mm512_and_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec0, x1), 31), x1);
            __m512i y2 = _mm512_and_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec0, x2), 31), x2);
            __m512i y3 = _mm512_and_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec0, x3), 31), x3);
            __m512i z0 = _mm512_and_epi32(_mm512_or_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec255, y0), 31), y0), vec255);
            __m512i z1 = _mm512_and_epi32(_mm512_or_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec255, y1), 31), y1), vec255);
            __m512i z2 = _mm512_and_epi32(_mm512_or_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec255, y2), 31), y2), vec255);
            __m512i z3 = _mm512_and_epi32(_mm512_or_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec255, y3), 31), y3), vec255);
            _mm512_store_epi32(&x[j],    z0);
            _mm512_store_epi32(&x[j+16], z1);
            _mm512_store_epi32(&x[j+32], z2);
            _mm512_store_epi32(&x[j+48], z3);
        }
    }
    if(n >= 16) {
        for(; j < n - 15; j += 16) {
            __m512i x0 = _mm512_load_epi32(&x[j]);
            __m512i y0 = _mm512_and_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec0, x0), 31), x0);
            __m512i z0 = _mm512_and_epi32(_mm512_or_epi32(_mm512_srai_epi32(_mm512_sub_epi32(vec255, y0), 31), y0), vec255);
            _mm512_store_epi32(&x[j],    z0);
        }
    }
    for(; j < n; ++j) {
        int32_t x0 = x[j];
        int32_t y0 = ((-x0) >> 31) & x0;
        int32_t z0 = (((255 - y0) >> 31) | y0) & 255;
        x[j] = z0;
    }
    return 0;
}

int32_t clamp_comparison(int32_t* x, size_t n) {
    const __m512i vec0 = _mm512_setzero_epi32();
    const __m512i vec255 = _mm512_set1_epi32(255);
    size_t j = 0;
    if(n >= 64) {
        for(; j < n - 63; j += 64) {
            __m512i x0 = _mm512_load_epi32(&x[j]);
            __m512i x1 = _mm512_load_epi32(&x[j+16]);
            __m512i x2 = _mm512_load_epi32(&x[j+32]);
            __m512i x3 = _mm512_load_epi32(&x[j+48]);
            __m512i y0 = _mm512_max_epi32(vec0, x0);
            __m512i y1 = _mm512_max_epi32(vec0, x1);
            __m512i y2 = _mm512_max_epi32(vec0, x2);
            __m512i y3 = _mm512_max_epi32(vec0, x3);
            __m512i z0 = _mm512_min_epi32(vec255, y0);
            __m512i z1 = _mm512_min_epi32(vec255, y1);
            __m512i z2 = _mm512_min_epi32(vec255, y2);
            __m512i z3 = _mm512_min_epi32(vec255, y3);
            _mm512_store_epi32(&x[j],    z0);
            _mm512_store_epi32(&x[j+16], z1);
            _mm512_store_epi32(&x[j+32], z2);
            _mm512_store_epi32(&x[j+48], z3);
        }
    }
    if(n >= 16) {
        for(; j < n - 15; j += 16) {
            __m512i x0 = _mm512_load_epi32(&x[j]);
            __m512i y0 = _mm512_max_epi32(vec0, x0);
            __m512i z0 = _mm512_min_epi32(vec255, y0);
            _mm512_store_epi32(&x[j],    z0);
        }
    }
    for(; j < n; ++j) {
        int32_t x0 = x[j];
        int32_t y0 = (x0 >= 0) ? x0 : 0;
        int32_t z0 = (y0 <= 255) ? y0 : 255;
        x[j] = z0;
    }
    return 0;
}

#elif defined(AVX2)
int32_t clamp_arithmetic(int32_t* x, size_t n) {
    const __m256i vec0 = _mm256_setzero_si256();
    const __m256i vec255 = _mm256_set1_epi32(255);
    size_t j = 0;
    if(n >= 32) {
        for(; j < n - 31; j += 32) { // 循环展开
            __m256i x0 = _mm256_load_si256((__m256i*)&x[j]);
            __m256i x1 = _mm256_load_si256((__m256i*)&x[j+8]);
            __m256i x2 = _mm256_load_si256((__m256i*)&x[j+16]);
            __m256i x3 = _mm256_load_si256((__m256i*)&x[j+24]);
            __m256i y0 = _mm256_and_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec0, x0), 31), x0);
            __m256i y1 = _mm256_and_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec0, x1), 31), x1);
            __m256i y2 = _mm256_and_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec0, x2), 31), x2);
            __m256i y3 = _mm256_and_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec0, x3), 31), x3);
            __m256i z0 = _mm256_and_si256(_mm256_or_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec255, y0), 31), y0), vec255);
            __m256i z1 = _mm256_and_si256(_mm256_or_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec255, y1), 31), y1), vec255);
            __m256i z2 = _mm256_and_si256(_mm256_or_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec255, y2), 31), y2), vec255);
            __m256i z3 = _mm256_and_si256(_mm256_or_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec255, y3), 31), y3), vec255);
            _mm256_store_si256((__m256i*)&x[j],    z0);
            _mm256_store_si256((__m256i*)&x[j+8],  z1);
            _mm256_store_si256((__m256i*)&x[j+16], z2);
            _mm256_store_si256((__m256i*)&x[j+24], z3);
        }
    }
    if(n >= 8) {
        for(; j < n - 7; j += 8) {
            __m256i x0 = _mm256_load_si256((__m256i*)&x[j]);
            __m256i y0 = _mm256_and_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec0, x0), 31), x0);
            __m256i z0 = _mm256_and_si256(_mm256_or_si256(_mm256_srai_epi32(_mm256_sub_epi32(vec255, y0), 31), y0), vec255);
            _mm256_store_si256((__m256i*)&x[j],    z0);
        }
    }
    for(; j < n; ++j) {
        int32_t x0 = x[j];
        int32_t y0 = ((-x0) >> 31) & x0;
        int32_t z0 = (((255 - y0) >> 31) | y0) & 255;
        x[j] = z0;
    }
    return 0;
}

int32_t clamp_comparison(int32_t* x, size_t n) {
    const __m256i vec0 = _mm256_setzero_si256();
    const __m256i vec255 = _mm256_set1_epi32(255);
    size_t j = 0;
    if(n >= 32) {
        for(; j < n - 31; j += 32) { // 循环展开
            __m256i x0 = _mm256_load_si256((__m256i*)&x[j]);
            __m256i x1 = _mm256_load_si256((__m256i*)&x[j+8]);
            __m256i x2 = _mm256_load_si256((__m256i*)&x[j+16]);
            __m256i x3 = _mm256_load_si256((__m256i*)&x[j+24]);
            __m256i y0 = _mm256_max_epi32(vec0, x0);
            __m256i y1 = _mm256_max_epi32(vec0, x1);
            __m256i y2 = _mm256_max_epi32(vec0, x2);
            __m256i y3 = _mm256_max_epi32(vec0, x3);
            __m256i z0 = _mm256_min_epi32(vec255, y0);
            __m256i z1 = _mm256_min_epi32(vec255, y1);
            __m256i z2 = _mm256_min_epi32(vec255, y2);
            __m256i z3 = _mm256_min_epi32(vec255, y3);
            _mm256_store_si256((__m256i*)&x[j],    z0);
            _mm256_store_si256((__m256i*)&x[j+8],  z1);
            _mm256_store_si256((__m256i*)&x[j+16], z2);
            _mm256_store_si256((__m256i*)&x[j+24], z3);
        }
    }
    if(n >= 8) {
        for(; j < n - 7; j += 8) {
            __m256i x0 = _mm256_load_si256((__m256i*)&x[j]);
            __m256i y0 = _mm256_max_epi32(vec0, x0);
            __m256i z0 = _mm256_min_epi32(vec255, y0);
            _mm256_store_si256((__m256i*)&x[j],    z0);
        }
    }
    for(; j < n; ++j) {
        int32_t x0 = x[j];
        int32_t y0 = (x0 >= 0) ? x0 : 0;
        int32_t z0 = (y0 <= 255) ? y0 : 255;
        x[j] = z0;
    }
    return 0;
}

#else // SSE4

int32_t clamp_arithmetic(int32_t* x, size_t n) {
    const __m128i vec0 = _mm_setzero_si128();
    const __m128i vec255 = _mm_set1_epi32(255);
    size_t j = 0;
    if(n >= 16) {
        for(; j < n - 15; j += 16) { // 循环展开
            __m128i x0 = _mm_load_si128((__m128i*)&x[j]);
            __m128i x1 = _mm_load_si128((__m128i*)&x[j+4]);
            __m128i x2 = _mm_load_si128((__m128i*)&x[j+8]);
            __m128i x3 = _mm_load_si128((__m128i*)&x[j+12]);
            __m128i y0 = _mm_and_si128(_mm_srai_epi32(_mm_sub_epi32(vec0, x0), 31), x0);
            __m128i y1 = _mm_and_si128(_mm_srai_epi32(_mm_sub_epi32(vec0, x1), 31), x1);
            __m128i y2 = _mm_and_si128(_mm_srai_epi32(_mm_sub_epi32(vec0, x2), 31), x2);
            __m128i y3 = _mm_and_si128(_mm_srai_epi32(_mm_sub_epi32(vec0, x3), 31), x3);
            __m128i z0 = _mm_and_si128(_mm_or_si128(_mm_srai_epi32(_mm_sub_epi32(vec255, y0), 31), y0), vec255);
            __m128i z1 = _mm_and_si128(_mm_or_si128(_mm_srai_epi32(_mm_sub_epi32(vec255, y1), 31), y1), vec255);
            __m128i z2 = _mm_and_si128(_mm_or_si128(_mm_srai_epi32(_mm_sub_epi32(vec255, y2), 31), y2), vec255);
            __m128i z3 = _mm_and_si128(_mm_or_si128(_mm_srai_epi32(_mm_sub_epi32(vec255, y3), 31), y3), vec255);
            _mm_store_si128((__m128i*)&x[j],    z0);
            _mm_store_si128((__m128i*)&x[j+4],  z1);
            _mm_store_si128((__m128i*)&x[j+8],  z2);
            _mm_store_si128((__m128i*)&x[j+12], z3);
        }
    }
    if(n >= 4) {
        for(; j < n - 3; j += 4) { // 循环展开
            __m128i x0 = _mm_load_si128((__m128i*)&x[j]);
            __m128i y0 = _mm_and_si128(_mm_srai_epi32(_mm_sub_epi32(vec0, x0), 31), x0);
            __m128i z0 = _mm_and_si128(_mm_or_si128(_mm_srai_epi32(_mm_sub_epi32(vec255, y0), 31), y0), vec255);
            _mm_store_si128((__m128i*)&x[j],    z0);
        }
    }
    for(; j < n; ++j) {
        int32_t x0 = x[j];
        int32_t y0 = ((-x0) >> 31) & x0;
        int32_t z0 = (((255 - y0) >> 31) | y0) & 255;
        x[j] = z0;
    }
    return 0;
}

int32_t clamp_comparison(int32_t* x, size_t n) {
    const __m128i vec0 = _mm_setzero_si128();
    const __m128i vec255 = _mm_set1_epi32(255);
    size_t j = 0;
    if(n >= 16) {
        for(; j < n - 15; j += 16) { // 循环展开
            __m128i x0 = _mm_load_si128((__m128i*)&x[j]);
            __m128i x1 = _mm_load_si128((__m128i*)&x[j+4]);
            __m128i x2 = _mm_load_si128((__m128i*)&x[j+8]);
            __m128i x3 = _mm_load_si128((__m128i*)&x[j+12]);
            __m128i y0 = _mm_max_epi32(vec0, x0);
            __m128i y1 = _mm_max_epi32(vec0, x1);
            __m128i y2 = _mm_max_epi32(vec0, x2);
            __m128i y3 = _mm_max_epi32(vec0, x3);
            __m128i z0 = _mm_min_epi32(vec255, y0);
            __m128i z1 = _mm_min_epi32(vec255, y1);
            __m128i z2 = _mm_min_epi32(vec255, y2);
            __m128i z3 = _mm_min_epi32(vec255, y3);
            _mm_store_si128((__m128i*)&x[j],    z0);
            _mm_store_si128((__m128i*)&x[j+4],  z1);
            _mm_store_si128((__m128i*)&x[j+8],  z2);
            _mm_store_si128((__m128i*)&x[j+12], z3);
        }
    }
    if(n >= 4) {
        for(; j < n - 3; j += 4) { // 循环展开
            __m128i x0 = _mm_load_si128((__m128i*)&x[j]);
            __m128i y0 = _mm_max_epi32(vec0, x0);
            __m128i z0 = _mm_min_epi32(vec255, y0);
            _mm_store_si128((__m128i*)&x[j],    z0);
        }
    }
    for(; j < n; ++j) {
        int32_t x0 = x[j];
        int32_t y0 = (x0 >= 0) ? x0 : 0;
        int32_t z0 = (y0 <= 255) ? y0 : 255;
        x[j] = z0;
    }
    return 0;
}

#endif