#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>

#ifdef AVX512

void to_upper_arithmetic_logic(const char *src, char *dest, size_t len) {
    const __m512i x7f = _mm512_set1_epi8(0x7f);
    const __m512i x05 = _mm512_set1_epi8(0x05);
    const __m512i x20 = _mm512_set1_epi8(0x20);
    size_t i;

    if(len >= 64 * 4) {
        for (i = 0; i < len - (64 * 4 - 1); i += 64 * 4) {
            __m512i src0 = _mm512_and_si512(_mm512_load_si512(&src[i + 64 * 0]), x7f);
            __m512i src1 = _mm512_and_si512(_mm512_load_si512(&src[i + 64 * 1]), x7f);
            __m512i src2 = _mm512_and_si512(_mm512_load_si512(&src[i + 64 * 2]), x7f);
            __m512i src3 = _mm512_and_si512(_mm512_load_si512(&src[i + 64 * 3]), x7f);
            __m512i a0 = _mm512_and_si512(_mm512_add_epi8(src0, x05), _mm512_add_epi8(src0, x7f));
            __m512i a1 = _mm512_and_si512(_mm512_add_epi8(src1, x05), _mm512_add_epi8(src1, x7f));
            __m512i a2 = _mm512_and_si512(_mm512_add_epi8(src2, x05), _mm512_add_epi8(src2, x7f));
            __m512i a3 = _mm512_and_si512(_mm512_add_epi8(src3, x05), _mm512_add_epi8(src3, x7f));
            __m512i b0 =_mm512_and_si512(_mm512_and_si512(a0, x20), _mm512_srai_epi16(a0, 1));
            __m512i b1 =_mm512_and_si512(_mm512_and_si512(a1, x20), _mm512_srai_epi16(a1, 1));
            __m512i b2 =_mm512_and_si512(_mm512_and_si512(a2, x20), _mm512_srai_epi16(a2, 1));
            __m512i b3 =_mm512_and_si512(_mm512_and_si512(a3, x20), _mm512_srai_epi16(a3, 1));
            _mm512_store_si512(&dest[i + 64 * 0], _mm512_sub_epi8(src0, b0));
            _mm512_store_si512(&dest[i + 64 * 1], _mm512_sub_epi8(src1, b1));
            _mm512_store_si512(&dest[i + 64 * 2], _mm512_sub_epi8(src2, b2));
            _mm512_store_si512(&dest[i + 64 * 3], _mm512_sub_epi8(src3, b3));
        }
    }
    if(len >= 64) {
        for (; i < len - (64 - 1); i += 64) {
            __m512i src0 = _mm512_and_si512(_mm512_load_si512(&src[i]), x7f);
            __m512i a0 = _mm512_and_si512(_mm512_add_epi8(src0, x05), _mm512_add_epi8(src0, x7f));
            __m512i b0 =_mm512_and_si512(_mm512_and_si512(a0, x20), _mm512_srai_epi16(a0, 1));
            _mm512_store_si512(&dest[i], _mm512_sub_epi8(src0, b0));
        }
    }
    for (; i < len; i++) {
        char src_char = src[i] & 0x7f;
        char a = (src_char + 0x05) & (src_char + 0x7f);
        char b = a & (a >> 1) & 0x20;
        dest[i] = src_char - b;
    }
    dest[len] = 0;
}

void to_upper_compare_select(const char *src, char *dest, size_t len) {
    const __m512i a = _mm512_set1_epi8('a');
    const __m512i z = _mm512_set1_epi8('z');
    const __m512i Aa = _mm512_set1_epi8('a' - 'A');
    size_t i;

    if(len >= 64 * 4) {
        for (i = 0; i < len - (64 * 4 - 1); i += 64 * 4) {
            __m512i src0 = _mm512_load_si512(&src[i + 64 * 0]);
            __m512i src1 = _mm512_load_si512(&src[i + 64 * 1]);
            __m512i src2 = _mm512_load_si512(&src[i + 64 * 2]);
            __m512i src3 = _mm512_load_si512(&src[i + 64 * 3]);
            __mmask64 mask0 = _kand_mask64(_mm512_cmple_epi8_mask(src0, z), _mm512_cmpge_epi8_mask(src0, a));
            __mmask64 mask1 = _kand_mask64(_mm512_cmple_epi8_mask(src1, z), _mm512_cmpge_epi8_mask(src1, a));
            __mmask64 mask2 = _kand_mask64(_mm512_cmple_epi8_mask(src2, z), _mm512_cmpge_epi8_mask(src2, a));
            __mmask64 mask3 = _kand_mask64(_mm512_cmple_epi8_mask(src3, z), _mm512_cmpge_epi8_mask(src3, a));
            _mm512_store_si512(&dest[i + 64 * 0], _mm512_mask_sub_epi8(src0, mask0, src0, Aa));
            _mm512_store_si512(&dest[i + 64 * 1], _mm512_mask_sub_epi8(src1, mask1, src1, Aa));
            _mm512_store_si512(&dest[i + 64 * 2], _mm512_mask_sub_epi8(src2, mask2, src2, Aa));
            _mm512_store_si512(&dest[i + 64 * 3], _mm512_mask_sub_epi8(src3, mask3, src3, Aa));
        }
    }
    if(len >= 64) {
        for (; i < len - (64 - 1); i += 64) {
            __m512i src0 = _mm512_load_si512(&src[i]);
            __mmask64 mask0 = _kand_mask64(_mm512_cmple_epi8_mask(src0, z), _mm512_cmpge_epi8_mask(src0, a));
            _mm512_store_si512(&dest[i], _mm512_mask_sub_epi8(src0, mask0, src0, Aa));
        }
    }
    for (; i < len; i++) {
        dest[i] = (src[i] >= 'a' && src[i] <= 'z') ? (src[i] - 'a' + 'A') : src[i];
    }
    dest[len] = 0;
}

#elif defined(AVX2)

void to_upper_arithmetic_logic(const char *src, char *dest, size_t len) {
    const __m256i x7f = _mm256_set1_epi8(0x7f);
    const __m256i x05 = _mm256_set1_epi8(0x05);
    const __m256i x20 = _mm256_set1_epi8(0x20);
    size_t i;

    if(len >= 32 * 4) {
        for (i = 0; i < len - (32 * 4 - 1); i += 32 * 4) {
            __m256i src0 = _mm256_and_si256(_mm256_load_si256((__m256i*)&src[i + 32 * 0]), x7f);
            __m256i src1 = _mm256_and_si256(_mm256_load_si256((__m256i*)&src[i + 32 * 1]), x7f);
            __m256i src2 = _mm256_and_si256(_mm256_load_si256((__m256i*)&src[i + 32 * 2]), x7f);
            __m256i src3 = _mm256_and_si256(_mm256_load_si256((__m256i*)&src[i + 32 * 3]), x7f);
            __m256i a0 = _mm256_and_si256(_mm256_add_epi8(src0, x05), _mm256_add_epi8(src0, x7f));
            __m256i a1 = _mm256_and_si256(_mm256_add_epi8(src1, x05), _mm256_add_epi8(src1, x7f));
            __m256i a2 = _mm256_and_si256(_mm256_add_epi8(src2, x05), _mm256_add_epi8(src2, x7f));
            __m256i a3 = _mm256_and_si256(_mm256_add_epi8(src3, x05), _mm256_add_epi8(src3, x7f));
            __m256i b0 =_mm256_and_si256(_mm256_and_si256(a0, x20), _mm256_srai_epi16(a0, 1));
            __m256i b1 =_mm256_and_si256(_mm256_and_si256(a1, x20), _mm256_srai_epi16(a1, 1));
            __m256i b2 =_mm256_and_si256(_mm256_and_si256(a2, x20), _mm256_srai_epi16(a2, 1));
            __m256i b3 =_mm256_and_si256(_mm256_and_si256(a3, x20), _mm256_srai_epi16(a3, 1));
            _mm256_store_si256((__m256i*)&dest[i + 32 * 0], _mm256_sub_epi8(src0, b0));
            _mm256_store_si256((__m256i*)&dest[i + 32 * 1], _mm256_sub_epi8(src1, b1));
            _mm256_store_si256((__m256i*)&dest[i + 32 * 2], _mm256_sub_epi8(src2, b2));
            _mm256_store_si256((__m256i*)&dest[i + 32 * 3], _mm256_sub_epi8(src3, b3));
        }
    }
    if(len >= 32) {
        for(; i < len - (32 - 1); i += 32) {
            __m256i src0 = _mm256_and_si256(_mm256_load_si256((__m256i*)&src[i]), x7f);
            __m256i a0 = _mm256_and_si256(_mm256_add_epi8(src0, x05), _mm256_add_epi8(src0, x7f));
            __m256i b0 =_mm256_and_si256(_mm256_and_si256(a0, x20), _mm256_srai_epi16(a0, 1));
            _mm256_store_si256((__m256i*)&dest[i], _mm256_sub_epi8(src0, b0));
        }
    }
    for (; i < len; i++) {
        char src_char = src[i] & 0x7f;
        char a = (src_char + 0x05) & (src_char + 0x7f);
        char b = a & (a >> 1) & 0x20;
        dest[i] = src_char - b;
    }
    dest[len] = 0;
}

void to_upper_compare_select(const char *src, char *dest, size_t len) {
    const __m256i a = _mm256_set1_epi8('a');
    const __m256i z = _mm256_set1_epi8('z');
    const __m256i Aa = _mm256_set1_epi8('a' - 'A');
    size_t i;

    if(len >= 32 * 4) {
        for (i = 0; i < len - (32 * 4 - 1); i += 32 * 4) {
            __m256i src0 = _mm256_load_si256((__m256i*)&src[i + 32 * 0]);
            __m256i src1 = _mm256_load_si256((__m256i*)&src[i + 32 * 1]);
            __m256i src2 = _mm256_load_si256((__m256i*)&src[i + 32 * 2]);
            __m256i src3 = _mm256_load_si256((__m256i*)&src[i + 32 * 3]);
            __m256i mask0 = _mm256_or_si256(_mm256_cmpgt_epi8(src0, z), _mm256_cmpgt_epi8(a, src0));
            __m256i mask1 = _mm256_or_si256(_mm256_cmpgt_epi8(src1, z), _mm256_cmpgt_epi8(a, src1));
            __m256i mask2 = _mm256_or_si256(_mm256_cmpgt_epi8(src2, z), _mm256_cmpgt_epi8(a, src2));
            __m256i mask3 = _mm256_or_si256(_mm256_cmpgt_epi8(src3, z), _mm256_cmpgt_epi8(a, src3));
            __m256i u0 = _mm256_sub_epi8(src0, Aa);
            __m256i u1 = _mm256_sub_epi8(src1, Aa);
            __m256i u2 = _mm256_sub_epi8(src2, Aa);
            __m256i u3 = _mm256_sub_epi8(src3, Aa);
            _mm256_store_si256((__m256i*)&dest[i + 32 * 0], _mm256_blendv_epi8(u0, src0, mask0));
            _mm256_store_si256((__m256i*)&dest[i + 32 * 1], _mm256_blendv_epi8(u1, src1, mask1));
            _mm256_store_si256((__m256i*)&dest[i + 32 * 2], _mm256_blendv_epi8(u2, src2, mask2));
            _mm256_store_si256((__m256i*)&dest[i + 32 * 3], _mm256_blendv_epi8(u3, src3, mask3));
        }
    }
    if(len >= 32) {
        for(; i < len - (32 - 1); i += 32) {
            __m256i src0 = _mm256_load_si256((__m256i*)&src[i]);
            __m256i mask0 = _mm256_or_si256(_mm256_cmpgt_epi8(src0, z), _mm256_cmpgt_epi8(a, src0));
            __m256i u0 = _mm256_sub_epi8(src0, Aa);
            _mm256_store_si256((__m256i*)&dest[i], _mm256_blendv_epi8(u0, src0, mask0));
        }
    }
    for (; i < len; i++) {
        dest[i] = (src[i] >= 'a' && src[i] <= 'z') ? (src[i] - 'a' + 'A') : src[i];
    }
    dest[len] = 0;
}

#elif defined(SSE)

void to_upper_arithmetic_logic(const char *src, char *dest, size_t len) {
    const __m128i x7f = _mm_set1_epi8(0x7f);
    const __m128i x05 = _mm_set1_epi8(0x05);
    const __m128i x20 = _mm_set1_epi8(0x20);
    size_t i;

    if(len >= 16 * 4) {
        for (i = 0; i < len - (16 * 4 - 1); i += 16 * 4) {
            __m128i src0 = _mm_and_si128(_mm_load_si128((__m128i*)&src[i + 16 * 0]), x7f);
            __m128i src1 = _mm_and_si128(_mm_load_si128((__m128i*)&src[i + 16 * 1]), x7f);
            __m128i src2 = _mm_and_si128(_mm_load_si128((__m128i*)&src[i + 16 * 2]), x7f);
            __m128i src3 = _mm_and_si128(_mm_load_si128((__m128i*)&src[i + 16 * 3]), x7f);
            __m128i a0 = _mm_and_si128(_mm_add_epi8(src0, x05), _mm_add_epi8(src0, x7f));
            __m128i a1 = _mm_and_si128(_mm_add_epi8(src1, x05), _mm_add_epi8(src1, x7f));
            __m128i a2 = _mm_and_si128(_mm_add_epi8(src2, x05), _mm_add_epi8(src2, x7f));
            __m128i a3 = _mm_and_si128(_mm_add_epi8(src3, x05), _mm_add_epi8(src3, x7f));
            __m128i b0 =_mm_and_si128(_mm_and_si128(a0, x20), _mm_srai_epi16(a0, 1));
            __m128i b1 =_mm_and_si128(_mm_and_si128(a1, x20), _mm_srai_epi16(a1, 1));
            __m128i b2 =_mm_and_si128(_mm_and_si128(a2, x20), _mm_srai_epi16(a2, 1));
            __m128i b3 =_mm_and_si128(_mm_and_si128(a3, x20), _mm_srai_epi16(a3, 1));
            _mm_store_si128((__m128i*)&dest[i + 16 * 0], _mm_sub_epi8(src0, b0));
            _mm_store_si128((__m128i*)&dest[i + 16 * 1], _mm_sub_epi8(src1, b1));
            _mm_store_si128((__m128i*)&dest[i + 16 * 2], _mm_sub_epi8(src2, b2));
            _mm_store_si128((__m128i*)&dest[i + 16 * 3], _mm_sub_epi8(src3, b3));
        }
    }
    if(len >= 16) {
        for(; i < len - (16 - 1); i += 16) {
            __m128i src0 = _mm_and_si128(_mm_load_si128((__m128i*)&src[i]), x7f);
            __m128i a0 = _mm_and_si128(_mm_add_epi8(src0, x05), _mm_add_epi8(src0, x7f));
            __m128i b0 =_mm_and_si128(_mm_and_si128(a0, x20), _mm_srai_epi16(a0, 1));
            _mm_store_si128((__m128i*)&dest[i], _mm_sub_epi8(src0, b0));
        }
    }
    for (; i < len; i++) {
        char src_char = src[i] & 0x7f;
        char a = (src_char + 0x05) & (src_char + 0x7f);
        char b = a & (a >> 1) & 0x20;
        dest[i] = src_char - b;
    }
    dest[len] = 0;
}

void to_upper_compare_select(const char *src, char *dest, size_t len) {
    const __m128i a = _mm_set1_epi8('a');
    const __m128i z = _mm_set1_epi8('z');
    const __m128i Aa = _mm_set1_epi8('a' - 'A');
    size_t i;

    if(len > 16 * 4) {
        for (i = 0; i < len - (16 * 4 - 1); i += 16 * 4) {
            __m128i src0 = _mm_load_si128((__m128i*)&src[i + 16 * 0]);
            __m128i src1 = _mm_load_si128((__m128i*)&src[i + 16 * 1]);
            __m128i src2 = _mm_load_si128((__m128i*)&src[i + 16 * 2]);
            __m128i src3 = _mm_load_si128((__m128i*)&src[i + 16 * 3]);
            __m128i mask0 = _mm_or_si128(_mm_cmpgt_epi8(src0, z), _mm_cmpgt_epi8(a, src0));
            __m128i mask1 = _mm_or_si128(_mm_cmpgt_epi8(src1, z), _mm_cmpgt_epi8(a, src1));
            __m128i mask2 = _mm_or_si128(_mm_cmpgt_epi8(src2, z), _mm_cmpgt_epi8(a, src2));
            __m128i mask3 = _mm_or_si128(_mm_cmpgt_epi8(src3, z), _mm_cmpgt_epi8(a, src3));
            __m128i u0 = _mm_sub_epi8(src0, Aa);
            __m128i u1 = _mm_sub_epi8(src1, Aa);
            __m128i u2 = _mm_sub_epi8(src2, Aa);
            __m128i u3 = _mm_sub_epi8(src3, Aa);
            _mm_store_si128((__m128i*)&dest[i + 16 * 0], _mm_blendv_epi8(u0, src0, mask0));
            _mm_store_si128((__m128i*)&dest[i + 16 * 1], _mm_blendv_epi8(u1, src1, mask1));
            _mm_store_si128((__m128i*)&dest[i + 16 * 2], _mm_blendv_epi8(u2, src2, mask2));
            _mm_store_si128((__m128i*)&dest[i + 16 * 3], _mm_blendv_epi8(u3, src3, mask3));
        }
    }
    if(len > 16) {
        for(; i < len - (16 - 1); i += 16) {
            __m128i src0 = _mm_load_si128((__m128i*)&src[i]);
            __m128i mask0 = _mm_or_si128(_mm_cmpgt_epi8(src0, z), _mm_cmpgt_epi8(a, src0));
            __m128i u0 = _mm_sub_epi8(src0, Aa);
            _mm_store_si128((__m128i*)&dest[i], _mm_blendv_epi8(u0, src0, mask0));
        }
    }
    for (; i < len; i++) {
        dest[i] = (src[i] >= 'a' && src[i] <= 'z') ? (src[i] - 'a' + 'A') : src[i];
    }
    dest[len] = 0;
}

#else

void to_upper_arithmetic_logic(const char *src, char *dest, size_t len) {
    size_t i = 0;
    if(len >= 8) {
        for (; i < len - 7; i += 8) {
            uint64_t src_uint = *((uint64_t*)&src[i]) & 0x7f7f7f7f7f7f7f7f;
            uint64_t a = (src_uint + 0x0505050505050505) & (src_uint + 0x7f7f7f7f7f7f7f7f);
            uint64_t b = a & (a >> 1) & 0x2020202020202020;
            *((uint64_t*)&src[i]) = src_uint - b;
        }
    }
    for (; i < len; i++) {
        char src_char = src[i] & 0x7f;
        char a = (src_char + 0x05) & (src_char + 0x7f);
        char b = a & (a >> 1) & 0x20;
        dest[i] = src_char - b;
    }
    dest[len] = 0;
}

void to_upper_compare_select(const char *src, char *dest, size_t len) {
    for (size_t i = 0; i < len; i++) {
        dest[i] = (src[i] >= 'a' && src[i] <= 'z') ? (src[i] - 'a' + 'A') : src[i];
    }
    dest[len] = 0;
}

#endif

static char lowercase_to_uppercase_map[256];
void init_table() {
    for (int i = 0; i < 256; i++) {
        if (i <= 'z' && i >= 'a') {
            lowercase_to_uppercase_map[i] = i - 'a' + 'A';
        } else {
            lowercase_to_uppercase_map[i] = i;
        }
    }
}

void to_upper_lookup_table(const char *src, char *dest, size_t len) {
    for (int i = 0; i < len; i++) {
        dest[i] = lowercase_to_uppercase_map[src[i]];
    }
    dest[len] = 0;
}

