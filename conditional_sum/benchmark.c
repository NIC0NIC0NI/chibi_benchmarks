#include <stdint.h>
#include <immintrin.h>

#ifdef AVX512
uint32_t conditional_sum(uint32_t* x, size_t n) {
    const __m512i vec0 = _mm512_setzero_epi32();
    const __m512i vec128 = _mm512_set1_epi32(128);
    __m512i s0 = vec0;
    uint32_t s = 0;
    size_t j = 0;
    if(n >= 64) {
        __m512i s1 = vec0, s2 = vec0, s3 = vec0;
        for(; j < n - 63; j += 64) {
            __m512i x0 = _mm512_load_epi32(&x[j]);
            __m512i x1 = _mm512_load_epi32(&x[j+16]);
            __m512i x2 = _mm512_load_epi32(&x[j+32]);
            __m512i x3 = _mm512_load_epi32(&x[j+48]);
            __mmask16 m0 = _mm512_cmpge_epu32_mask(x0, vec128);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(x1, vec128);
            __mmask16 m2 = _mm512_cmpge_epu32_mask(x2, vec128);
            __mmask16 m3 = _mm512_cmpge_epu32_mask(x3, vec128);
            __m512i y0 = _mm512_mask_blend_epi32(m0, vec0, x0);
            __m512i y1 = _mm512_mask_blend_epi32(m1, vec0, x1);
            __m512i y2 = _mm512_mask_blend_epi32(m2, vec0, x2);
            __m512i y3 = _mm512_mask_blend_epi32(m3, vec0, x3);
            s0 = _mm512_add_epi32(s0, y0);
            s1 = _mm512_add_epi32(s1, y1);
            s2 = _mm512_add_epi32(s2, y2);
            s3 = _mm512_add_epi32(s3, y3);
        }
        s0 = _mm512_add_epi32(_mm512_add_epi32(s0, s1), _mm512_add_epi32(s2, s3));
    }
    if(n >= 16) {
        for(; j < n - 15; j += 16) {
            __m512i x0 = _mm512_load_epi32(&x[j]);
            __mmask16 m0 = _mm512_cmpge_epu32_mask(x0, vec128);
            __m512i y0 = _mm512_mask_blend_epi32(m0, vec0, x0);
            s0 = _mm512_add_epi32(s0, y0);
        }
        s = _mm512_reduce_add_epi32(s0);
    }
    for(; j < n; ++j) {
        s = (x[j] >= 128) ? (s + x[j]) : s;
    }
    return s;
}

#elif defined(AVX2)
uint32_t conditional_sum(uint32_t* x, size_t n) {
    const __m256i vec0 = _mm256_setzero_si256();
    const __m256i vec128 = _mm256_set1_epi32(128);
    __m256i s0 = vec0;
    uint32_t s = 0;
    size_t j = 0;
    if(n >= 32) {
        __m256i s1 = vec0, s2 = vec0, s3 = vec0;
        for(; j < n - 31; j += 32) { // 循环展开
            __m256i x0 = _mm256_load_si256((__m256i*)&x[j]);
            __m256i x1 = _mm256_load_si256((__m256i*)&x[j+8]);
            __m256i x2 = _mm256_load_si256((__m256i*)&x[j+16]);
            __m256i x3 = _mm256_load_si256((__m256i*)&x[j+24]);
            __m256i m0 = _mm256_cmpgt_epi32(vec128, x0);
            __m256i m1 = _mm256_cmpgt_epi32(vec128, x1);
            __m256i m2 = _mm256_cmpgt_epi32(vec128, x2);
            __m256i m3 = _mm256_cmpgt_epi32(vec128, x3);
            __m256i y0 = _mm256_blendv_epi8(x0, vec0, m0);
            __m256i y1 = _mm256_blendv_epi8(x1, vec0, m1);
            __m256i y2 = _mm256_blendv_epi8(x2, vec0, m2);
            __m256i y3 = _mm256_blendv_epi8(x3, vec0, m3);
            s0 = _mm256_add_epi32(s0, y0);
            s1 = _mm256_add_epi32(s1, y1);
            s2 = _mm256_add_epi32(s2, y2);
            s3 = _mm256_add_epi32(s3, y3);
        }
        s0 = _mm256_add_epi32(_mm256_add_epi32(s0, s1), _mm256_add_epi32(s2, s3));
    }
    if(n >= 8) {
        for(; j < n - 7; j += 8) {
            __m256i x0 = _mm256_load_si256((__m256i*)&x[j]);
            __m256i m0 = _mm256_cmpgt_epi32(vec128, x0);
            __m256i y0 = _mm256_blendv_epi8(x0, vec0, m0);
            s0 = _mm256_add_epi32(s0, y0);
        }
        __m128i half_hi = _mm256_extracti128_si256(s0, 1);
        __m128i half_lo = _mm256_extracti128_si256(s0, 0);
        __m128i half = _mm_add_epi32(half_hi, half_lo);
        __m128i quarter = _mm_add_epi32(half, _mm_shuffle_epi32(half, 0x0E));
        uint32_t eighth_hi = _mm_extract_epi32(quarter, 1);
        uint32_t eighth_lo = _mm_extract_epi32(quarter, 0);
        s = eighth_hi + eighth_lo;
    }
    for(; j < n; ++j) {
        s = (x[j] >= 128) ? (s + x[j]) : s;
    }
    return s;
}

#elif defined(SSE)
uint32_t conditional_sum(uint32_t* x, size_t n) {
    const __m128i vec0 = _mm_setzero_si128();
    const __m128i vec128 = _mm_set1_epi32(128);
    __m128i s0 = vec0;
    uint32_t s = 0;
    size_t j = 0;
    if(n >= 16) {
        __m128i s1 = vec0, s2 = vec0, s3 = vec0;
        for(; j < n - 15; j += 16) { // 循环展开
            __m128i x0 = _mm_load_si128((__m128i*)&x[j]);
            __m128i x1 = _mm_load_si128((__m128i*)&x[j+4]);
            __m128i x2 = _mm_load_si128((__m128i*)&x[j+8]);
            __m128i x3 = _mm_load_si128((__m128i*)&x[j+12]);
            __m128i m0 = _mm_cmpgt_epi32(vec128, x0);
            __m128i m1 = _mm_cmpgt_epi32(vec128, x1);
            __m128i m2 = _mm_cmpgt_epi32(vec128, x2);
            __m128i m3 = _mm_cmpgt_epi32(vec128, x3);
            __m128i y0 = _mm_blendv_epi8(x0, vec0, m0);
            __m128i y1 = _mm_blendv_epi8(x1, vec0, m1);
            __m128i y2 = _mm_blendv_epi8(x2, vec0, m2);
            __m128i y3 = _mm_blendv_epi8(x3, vec0, m3);
            s0 = _mm_add_epi32(s0, y0);
            s1 = _mm_add_epi32(s1, y1);
            s2 = _mm_add_epi32(s2, y2);
            s3 = _mm_add_epi32(s3, y3);
        }
        s0 = _mm_add_epi32(_mm_add_epi32(s0, s1), _mm_add_epi32(s2, s3));
    }
    if(n >= 4) {
        for(; j < n - 3; j += 4) { // 循环展开
            __m128i x0 = _mm_load_si128((__m128i*)&x[j]);
            __m128i m0 = _mm_cmpgt_epi32(vec128, x0);
            __m128i y0 = _mm_blendv_epi8(x0, vec0, m0);
            s0 = _mm_add_epi32(s0, y0);
        }
        __m128i quarter = _mm_add_epi32(s0, _mm_shuffle_epi32(s0, 0x0E));
        uint32_t eighth_hi = _mm_extract_epi32(quarter, 1);
        uint32_t eighth_lo = _mm_extract_epi32(quarter, 0);
        s = eighth_hi + eighth_lo;
    }
    for(; j < n; ++j) {
        s = (x[j] >= 128) ? (s + x[j]) : s;
    }
    return s;
}

#else
uint32_t conditional_sum(uint32_t* x, size_t n) {
    uint32_t s = 0;
    for(size_t j = 0; j < n; ++j) {
        s = (x[j] >= 128) ? (s + x[j]) : s;
    }
    return s;
}


#endif