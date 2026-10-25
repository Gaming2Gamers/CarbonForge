#ifndef SIMD_INTRINSICS_VEX_HPP
#define SIMD_INTRINSICS_VEX_HPP

/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * Intel Core Ultra 9 285 (Arrow Lake-S) VEX-Encoded SIMD Header
 * File: include/simd_intrinsics_vex.hpp
 * ============================================================================
 * Strict Silicon Architecture Invariant:
 * Arrow Lake-S does NOT support AVX-512. EVEX 0x62 prefix causes hardware fault.
 * Uses pure VEX encoding (_mm256_dpbusd_avx_epi32 or AVX2 fallback).
 */

#include <immintrin.h>
#include <intrin.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace g2g::carbon_forge::simd {

// Horizontal sum of 8 int32 lanes in a 256-bit register
static inline int32_t HorizontalSum256(__m256i v) noexcept {
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(vlow, vhigh);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0xB1));
    return _mm_cvtsi128_si32(sum128);
}

// Probes CPUID Leaf 1, Leaf 7.0, Leaf 7.1, and XCR0 for AVX2 and AVX-VNNI support
static inline int IsVnniSupported() noexcept {
    int cpuInfo[4] = {0};

    __cpuid(cpuInfo, 1);
    int osxsave = (cpuInfo[2] & (1 << 27)) != 0;
    int avx = (cpuInfo[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) {
        return 0;
    }

    unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 6) != 6) {
        return 0;
    }

    __cpuidex(cpuInfo, 7, 0);
    int avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    if (!avx2) {
        return 0;
    }

    __cpuidex(cpuInfo, 7, 1);
    int avx_vnni = (cpuInfo[0] & (1 << 4)) != 0;

    return (avx2 && avx_vnni) ? 1 : 0;
}

// Computes raw integer dot products between signed query and database rows
static inline void ComputeRawDotProductsInt8(
    const int8_t* db_matrix,
    const int8_t* query_vec,
    int num_vectors,
    int dim,
    int32_t* out_raw
) noexcept {
    static int s_has_vnni = -1;
    if (s_has_vnni == -1) {
        s_has_vnni = IsVnniSupported();
    }

    if (s_has_vnni) {
        uint8_t q_u8_stack[4096];
        uint8_t* query_u8 = (dim <= 4096) ? q_u8_stack : static_cast<uint8_t*>(std::malloc(dim));
        if (!query_u8) {
            query_u8 = q_u8_stack;
        }
        for (int j = 0; j < dim; ++j) {
            query_u8[j] = static_cast<uint8_t>(query_vec[j] + 128);
        }

        const __m256i ones_byte = _mm256_set1_epi8(1);

        for (int i = 0; i < num_vectors; ++i) {
            const int8_t* row = db_matrix + static_cast<size_t>(i) * dim;
            __m256i dot0 = _mm256_setzero_si256();
            __m256i dot1 = _mm256_setzero_si256();
            __m256i dot2 = _mm256_setzero_si256();
            __m256i dot3 = _mm256_setzero_si256();

            __m256i sum0 = _mm256_setzero_si256();
            __m256i sum1 = _mm256_setzero_si256();
            __m256i sum2 = _mm256_setzero_si256();
            __m256i sum3 = _mm256_setzero_si256();

            int j = 0;
            for (; j + 127 < dim; j += 128) {
                __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_u8 + j));
                __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j));
                dot0 = _mm256_dpbusd_avx_epi32(dot0, q0, r0);
                sum0 = _mm256_dpbusd_avx_epi32(sum0, ones_byte, r0);

                __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_u8 + j + 32));
                __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j + 32));
                dot1 = _mm256_dpbusd_avx_epi32(dot1, q1, r1);
                sum1 = _mm256_dpbusd_avx_epi32(sum1, ones_byte, r1);

                __m256i q2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_u8 + j + 64));
                __m256i r2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j + 64));
                dot2 = _mm256_dpbusd_avx_epi32(dot2, q2, r2);
                sum2 = _mm256_dpbusd_avx_epi32(sum2, ones_byte, r2);

                __m256i q3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_u8 + j + 96));
                __m256i r3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j + 96));
                dot3 = _mm256_dpbusd_avx_epi32(dot3, q3, r3);
                sum3 = _mm256_dpbusd_avx_epi32(sum3, ones_byte, r3);
            }

            for (; j + 31 < dim; j += 32) {
                __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_u8 + j));
                __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + j));
                dot0 = _mm256_dpbusd_avx_epi32(dot0, q0, r0);
                sum0 = _mm256_dpbusd_avx_epi32(sum0, ones_byte, r0);
            }

            __m256i dot_tot = _mm256_add_epi32(_mm256_add_epi32(dot0, dot1), _mm256_add_epi32(dot2, dot3));
            __m256i sum_tot = _mm256_add_epi32(_mm256_add_epi32(sum0, sum1), _mm256_add_epi32(sum2, sum3));

            int32_t raw_dot = HorizontalSum256(dot_tot);
            int32_t raw_sum = HorizontalSum256(sum_tot);

            for (; j < dim; ++j) {
                raw_dot += static_cast<int32_t>(query_u8[j]) * static_cast<int32_t>(row[j]);
                raw_sum += static_cast<int32_t>(row[j]);
            }

            out_raw[i] = raw_dot - 128 * raw_sum;
        }

        if (query_u8 != q_u8_stack) {
            std::free(query_u8);
        }
    } else {
        for (int i = 0; i < num_vectors; ++i) {
            const int8_t* row = db_matrix + static_cast<size_t>(i) * dim;
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();

            int j = 0;
            for (; j + 31 < dim; j += 32) {
                __m128i q_low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(query_vec + j));
                __m128i q_high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(query_vec + j + 16));
                __m128i r_low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j));
                __m128i r_high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j + 16));

                __m256i q_i16_0 = _mm256_cvtepi8_epi16(q_low);
                __m256i q_i16_1 = _mm256_cvtepi8_epi16(q_high);
                __m256i r_i16_0 = _mm256_cvtepi8_epi16(r_low);
                __m256i r_i16_1 = _mm256_cvtepi8_epi16(r_high);

                __m256i prod0 = _mm256_madd_epi16(q_i16_0, r_i16_0);
                __m256i prod1 = _mm256_madd_epi16(q_i16_1, r_i16_1);

                acc0 = _mm256_add_epi32(acc0, prod0);
                acc1 = _mm256_add_epi32(acc1, prod1);
            }

            __m256i total_acc = _mm256_add_epi32(acc0, acc1);
            int32_t sum = HorizontalSum256(total_acc);

            for (; j < dim; ++j) {
                sum += static_cast<int32_t>(query_vec[j]) * static_cast<int32_t>(row[j]);
            }

            out_raw[i] = sum;
        }
    }
}

} // namespace g2g::carbon_forge::simd

#endif // SIMD_INTRINSICS_VEX_HPP
