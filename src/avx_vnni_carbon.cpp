/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * C++20 Lowering Bridge for avx_vnni_cortex.carbon -> avx_vnni_carbon.dll
 * File: src/avx_vnni_carbon.cpp
 * ============================================================================
 */

#include "../include/carbon_interop.hpp"
#include "../include/simd_intrinsics_vex.hpp"

#include <cstdlib>
#include <cmath>
#include <algorithm>

using namespace g2g::carbon_forge;
using namespace g2g::carbon_forge::simd;

extern "C" {

CARBON_EXPORT int CARBON_CALL avx_vnni_is_supported(void) {
    return IsVnniSupported();
}

CARBON_EXPORT int CARBON_CALL avx_vnni_dot_product_int8(
    const int8_t* db_matrix,
    const int8_t* query_vec,
    const float* db_scales,
    float query_scale,
    int num_vectors,
    int dim,
    float* out_scores
) {
    if (!db_matrix || !query_vec || !db_scales || !out_scores || num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    int32_t raw_stack[1024];
    int32_t* raw_dots = (num_vectors <= 1024)
        ? raw_stack
        : static_cast<int32_t*>(std::malloc(static_cast<size_t>(num_vectors) * sizeof(int32_t)));
    if (!raw_dots) {
        return -2;
    }

    ComputeRawDotProductsInt8(db_matrix, query_vec, num_vectors, dim, raw_dots);

    for (int i = 0; i < num_vectors; ++i) {
        out_scores[i] = static_cast<float>(raw_dots[i]) * (query_scale * db_scales[i]);
    }

    if (raw_dots != raw_stack) {
        std::free(raw_dots);
    }

    return 0;
}

CARBON_EXPORT int CARBON_CALL avx_vnni_batch_cosine_similarity(
    const int8_t* db_matrix,
    const int8_t* query_vec,
    const float* db_norms,
    float query_norm,
    const float* db_scales,
    float query_scale,
    int num_vectors,
    int dim,
    float* out_sims
) {
    if (!db_matrix || !query_vec || !db_norms || !db_scales || !out_sims || num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    int32_t raw_stack[1024];
    int32_t* raw_dots = (num_vectors <= 1024)
        ? raw_stack
        : static_cast<int32_t*>(std::malloc(static_cast<size_t>(num_vectors) * sizeof(int32_t)));
    if (!raw_dots) {
        return -2;
    }

    ComputeRawDotProductsInt8(db_matrix, query_vec, num_vectors, dim, raw_dots);

    for (int i = 0; i < num_vectors; ++i) {
        float dot_float = static_cast<float>(raw_dots[i]) * (query_scale * db_scales[i]);
        float denom = query_norm * db_norms[i];
        float sim = (denom > 1e-9f) ? (dot_float / denom) : 0.0f;
        if (sim > 1.0f) sim = 1.0f;
        if (sim < -1.0f) sim = -1.0f;
        out_sims[i] = sim;
    }

    if (raw_dots != raw_stack) {
        std::free(raw_dots);
    }

    return 0;
}

} // extern "C"
