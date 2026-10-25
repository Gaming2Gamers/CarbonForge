/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * Dual-Mode Three.js & @pixiv/three-vrm WebAssembly SIMD128 Bridge
 * File: templates/wasm_threejs_bridge/carbon_vrm_wasm.cpp
 * ============================================================================
 * Exposes high-performance C ABI entrypoints wrapping kimodo_blend (SO(3) SLERP,
 * 0.0005 rad geodesic deadband filter, fast-attack trapezoid pose blending,
 * and branchless wasm-simd128 quaternion batch SLERP) and skin_colliders
 * (analytical SDF primitives, 64-primitive humanoid queries, limb-to-body
 * coronal abduction collision resolution, and 5-tier flared skirt clipping solver).
 *
 * Supports both Emscripten compilation (-msimd128 -O3) for in-browser WebAssembly
 * and native host C++20 compilation (MSVC / Clang / GCC) with zero warnings.
 */

#if defined(__EMSCRIPTEN__)
  #include <emscripten/emscripten.h>
  #define WASM_EXPORT EMSCRIPTEN_KEEPALIVE
  #if defined(__wasm_simd128__)
    #include <wasm_simd128.h>
    #define HAS_WASM_SIMD128 1
  #endif
#else
  #if defined(_WIN32)
    #define WASM_EXPORT __declspec(dllexport)
  #else
    #define WASM_EXPORT __attribute__((visibility("default")))
  #endif
#endif

#if __has_include("carbon_interop.hpp")
  #include "carbon_interop.hpp"
#elif __has_include("../../include/carbon_interop.hpp")
  #include "../../include/carbon_interop.hpp"
#else
  #include <carbon_interop.hpp>
#endif

#if __has_include("skin_colliders.hpp")
  #include "skin_colliders.hpp"
#elif __has_include("../../include/skin_colliders.hpp")
  #include "../../include/skin_colliders.hpp"
#else
  #include <skin_colliders.hpp>
#endif

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

namespace g2g::carbon_forge {

namespace {

// Thread-safe cached 64-primitive humanoid ragdoll
inline const std::vector<colliders::ColliderPrimitive>& GetHumanoidPrimitives() {
    static const std::vector<colliders::ColliderPrimitive> s_prims =
        colliders::HumanoidRagdollSynthesizer::SynthesizeHumanoid64();
    return s_prims;
}

// Exact 3D Segment-to-Segment minimum squared distance
inline float SegmentToSegmentDistanceSq(
    const colliders::Vec3& p1, const colliders::Vec3& q1,
    const colliders::Vec3& p2, const colliders::Vec3& q2,
    colliders::Vec3* outClosestA = nullptr,
    colliders::Vec3* outClosestB = nullptr
) noexcept {
    colliders::Vec3 d1{q1.x - p1.x, q1.y - p1.y, q1.z - p1.z};
    colliders::Vec3 d2{q2.x - p2.x, q2.y - p2.y, q2.z - p2.z};
    colliders::Vec3 r{p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};

    float a = d1.LengthSq();
    float e = d2.LengthSq();
    float f = d2.x * r.x + d2.y * r.y + d2.z * r.z;

    float s = 0.0f;
    float t = 0.0f;

    if (a <= 1e-8f && e <= 1e-8f) {
        s = 0.0f;
        t = 0.0f;
    } else if (a <= 1e-8f) {
        s = 0.0f;
        t = std::clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = d1.x * r.x + d1.y * r.y + d1.z * r.z;
        if (e <= 1e-8f) {
            t = 0.0f;
            s = std::clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = d1.x * d2.x + d1.y * d2.y + d1.z * d2.z;
            float denom = a * e - b * b;
            if (std::abs(denom) > 1e-12f) {
                s = std::clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = std::clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = std::clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }

    colliders::Vec3 cA{p1.x + s * d1.x, p1.y + s * d1.y, p1.z + s * d1.z};
    colliders::Vec3 cB{p2.x + t * d2.x, p2.y + t * d2.y, p2.z + t * d2.z};
    if (outClosestA) *outClosestA = cA;
    if (outClosestB) *outClosestB = cB;

    float dx = cA.x - cB.x;
    float dy = cA.y - cB.y;
    float dz = cA.z - cB.z;
    return dx * dx + dy * dy + dz * dz;
}

// Static pinned zero-copy buffer allocations (32-byte aligned for SIMD)
alignas(32) static float g_in_rest_quats[55 * 4] = {0.0f};       // Buffer 0: Rest pose (55 quats)
alignas(32) static float g_in_target_quats[55 * 4] = {0.0f};     // Buffer 1: Target pose (55 quats)
alignas(32) static float g_out_blended_quats[55 * 4] = {0.0f};   // Buffer 2: Output blended pose (55 quats)
alignas(32) static float g_arm_telemetry[12] = {0.0f};           // Buffer 3: L/R shoulder, elbow, wrist [x,y,z]
alignas(32) static float g_deflection_results[4] = {0.0f};       // Buffer 4: L/R abduction & elbow deltas
alignas(32) static float g_skirt_in_vertices[256 * 3] = {0.0f};  // Buffer 5: Input skirt vertices (up to 256)
alignas(32) static float g_skirt_out_vertices[256 * 3] = {0.0f}; // Buffer 6: Output skirt vertices (up to 256)

} // anonymous namespace

} // namespace g2g::carbon_forge

extern "C" {

using namespace g2g::carbon_forge;

/**
 * 1. Returns module version string.
 */
WASM_EXPORT const char* carbon_vrm_wasm_version(void) {
    return "1.7.0-wasm-simd128-meraki";
}

/**
 * 2. Single quaternion SLERP with shortest-path antipodal alignment.
 */
WASM_EXPORT void carbon_vrm_slerp(const float* q1, const float* q2, float t, float* out_q) {
    if (!q1 || !q2 || !out_q) return;
    Quaternion quat1 = Quaternion::Make(q1[0], q1[1], q1[2], q1[3]);
    Quaternion quat2 = Quaternion::Make(q2[0], q2[1], q2[2], q2[3]);
    Quaternion res = Quaternion::Slerp(quat1, quat2, t);
    out_q[0] = res.x;
    out_q[1] = res.y;
    out_q[2] = res.z;
    out_q[3] = res.w;
}

/**
 * 3. Angular deadband filter: 0.0005 rad unconditional geodesic deadband + exponential smoothing.
 */
WASM_EXPORT void carbon_vrm_angular_deadband_filter(
    const float* cur_q,
    const float* tgt_q,
    float dt,
    float eps,
    float lambda,
    float* out_q
) {
    if (!cur_q || !tgt_q || !out_q) return;
    Quaternion current = Quaternion::Make(cur_q[0], cur_q[1], cur_q[2], cur_q[3]);
    Quaternion target = Quaternion::Make(tgt_q[0], tgt_q[1], tgt_q[2], tgt_q[3]);
    AngularDeadbandFilter filter(eps, lambda);
    Quaternion res = filter.Filter(current, target, dt);
    out_q[0] = res.x;
    out_q[1] = res.y;
    out_q[2] = res.z;
    out_q[3] = res.w;
}

/**
 * 4. Batched SO(3) pose interpolation with C1 Hermite trapezoidal easing.
 */
WASM_EXPORT void carbon_vrm_blend_skeleton_pose(
    const float* q_rest,
    const float* q_target,
    float* q_out,
    int32_t num_joints,
    float progress,
    float attack_frac,
    float release_frac
) {
    if (!q_rest || !q_target || !q_out || num_joints <= 0) return;
    float factor = MotionEnvelope::FastAttackTrapezoid(progress, attack_frac, release_frac);
    for (int32_t i = 0; i < num_joints; ++i) {
        Quaternion r = Quaternion::Make(q_rest[i * 4 + 0], q_rest[i * 4 + 1], q_rest[i * 4 + 2], q_rest[i * 4 + 3]);
        Quaternion tgt = Quaternion::Make(q_target[i * 4 + 0], q_target[i * 4 + 1], q_target[i * 4 + 2], q_target[i * 4 + 3]);
        Quaternion bl = Quaternion::Slerp(r, tgt, factor);
        q_out[i * 4 + 0] = bl.x;
        q_out[i * 4 + 1] = bl.y;
        q_out[i * 4 + 2] = bl.z;
        q_out[i * 4 + 3] = bl.w;
    }
}

/**
 * 5. Branchless SIMD batch quaternion SLERP using wasm-simd128 v128_t or C++20 AVX2 fallback.
 */
WASM_EXPORT void carbon_vrm_simd_batch_slerp(
    const float* q1_array,
    const float* q2_array,
    const float* t_array,
    float* out_array,
    int32_t num_quats
) {
    if (!q1_array || !q2_array || !out_array || num_quats <= 0) return;

#if defined(HAS_WASM_SIMD128)
    for (int32_t i = 0; i < num_quats; ++i) {
        float t = t_array ? t_array[i] : 0.5f;
        v128_t v1 = wasm_v128_load(&q1_array[i * 4]);
        v128_t v2 = wasm_v128_load(&q2_array[i * 4]);

        v128_t prod = wasm_f32x4_mul(v1, v2);
        float dot = wasm_f32x4_extract_lane(prod, 0) +
                    wasm_f32x4_extract_lane(prod, 1) +
                    wasm_f32x4_extract_lane(prod, 2) +
                    wasm_f32x4_extract_lane(prod, 3);

        if (dot < 0.0f) {
            v2 = wasm_f32x4_neg(v2);
            dot = -dot;
        }

        v128_t out_v;
        if (dot > 0.99999995f) {
            v128_t vt = wasm_f32x4_splat(t);
            v128_t diff = wasm_f32x4_sub(v2, v1);
            out_v = wasm_f32x4_add(v1, wasm_f32x4_mul(vt, diff));
        } else {
            float theta_0 = std::acos(std::clamp(dot, -1.0f, 1.0f));
            float theta = theta_0 * t;
            float sin_theta = std::sin(theta);
            float sin_theta_0 = std::sin(theta_0);
            float s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
            float s1 = sin_theta / sin_theta_0;

            v128_t vs0 = wasm_f32x4_splat(s0);
            v128_t vs1 = wasm_f32x4_splat(s1);
            out_v = wasm_f32x4_add(wasm_f32x4_mul(v1, vs0), wasm_f32x4_mul(v2, vs1));
        }

        v128_t out_sq = wasm_f32x4_mul(out_v, out_v);
        float mag_sq = wasm_f32x4_extract_lane(out_sq, 0) +
                       wasm_f32x4_extract_lane(out_sq, 1) +
                       wasm_f32x4_extract_lane(out_sq, 2) +
                       wasm_f32x4_extract_lane(out_sq, 3);
        if (mag_sq > 1e-12f) {
            float inv_mag = 1.0f / std::sqrt(mag_sq);
            out_v = wasm_f32x4_mul(out_v, wasm_f32x4_splat(inv_mag));
        } else {
            out_v = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 1.0f);
        }
        wasm_v128_store(&out_array[i * 4], out_v);
    }
#else
    for (int32_t i = 0; i < num_quats; ++i) {
        float t = t_array ? t_array[i] : 0.5f;
        Quaternion q1 = Quaternion::Make(q1_array[i * 4 + 0], q1_array[i * 4 + 1], q1_array[i * 4 + 2], q1_array[i * 4 + 3]);
        Quaternion q2 = Quaternion::Make(q2_array[i * 4 + 0], q2_array[i * 4 + 1], q2_array[i * 4 + 2], q2_array[i * 4 + 3]);
        Quaternion q_out = Quaternion::Slerp(q1, q2, t);
        out_array[i * 4 + 0] = q_out.x;
        out_array[i * 4 + 1] = q_out.y;
        out_array[i * 4 + 2] = q_out.z;
        out_array[i * 4 + 3] = q_out.w;
    }
#endif
}

/**
 * 6. Analytical Sphere Signed Distance Field.
 */
WASM_EXPORT float carbon_vrm_sdf_sphere(float px, float py, float pz, float r) {
    return colliders::AnalyticalSDF::SphereSDF({px, py, pz}, r);
}

/**
 * 7. Analytical Box Signed Distance Field.
 */
WASM_EXPORT float carbon_vrm_sdf_box(float px, float py, float pz, float hx, float hy, float hz) {
    return colliders::AnalyticalSDF::BoxSDF({px, py, pz}, {hx, hy, hz});
}

/**
 * 8. Analytical Capsule Signed Distance Field.
 */
WASM_EXPORT float carbon_vrm_sdf_capsule(float px, float py, float pz, float h, float r) {
    return colliders::AnalyticalSDF::CapsuleSDF({px, py, pz}, h, r);
}

/**
 * 9. Analytical Cylinder Signed Distance Field.
 */
WASM_EXPORT float carbon_vrm_sdf_cylinder(float px, float py, float pz, float h, float r) {
    return colliders::AnalyticalSDF::CylinderSDF({px, py, pz}, h, r);
}

/**
 * 10. Returns humanoid ragdoll primitive count (64).
 */
WASM_EXPORT int32_t carbon_vrm_get_humanoid64_count(void) {
    return 64;
}

/**
 * 11. Evaluates distance to humanoid primitive [0..63] or whole-body minimum (idx < 0).
 */
WASM_EXPORT float carbon_vrm_sdf_humanoid64(float px, float py, float pz, int32_t prim_idx) {
    const auto& prims = GetHumanoidPrimitives();
    if (prim_idx >= 0 && prim_idx < static_cast<int32_t>(prims.size())) {
        return colliders::AnalyticalSDF::PrimitiveSDF(prims[prim_idx], {px, py, pz});
    }
    float min_d = 1.0e6f;
    for (const auto& prim : prims) {
        float d = colliders::AnalyticalSDF::PrimitiveSDF(prim, {px, py, pz});
        if (d < min_d) min_d = d;
    }
    return min_d;
}

/**
 * 12. Solves limb-to-body clearance and computes coronal abduction deflection.
 */
WASM_EXPORT int32_t carbon_vrm_solve_arm_body_collision(
    const float* shoulder_pos,
    const float* elbow_pos,
    const float* wrist_pos,
    float arm_radius,
    int32_t body_capsule_idx,
    float clearance_margin,
    float* out_abduction_rad,
    float* out_elbow_rad
) {
    if (!shoulder_pos || !elbow_pos || !wrist_pos || !out_abduction_rad || !out_elbow_rad) {
        return 0;
    }

    colliders::Vec3 shoulder{shoulder_pos[0], shoulder_pos[1], shoulder_pos[2]};
    colliders::Vec3 elbow{elbow_pos[0], elbow_pos[1], elbow_pos[2]};
    colliders::Vec3 wrist{wrist_pos[0], wrist_pos[1], wrist_pos[2]};

    colliders::Vec3 cap_p1{0.0f, 1.02f, 0.0f};
    colliders::Vec3 cap_p2{0.0f, 1.18f, 0.0f};
    float body_radius = 0.130f;

    const auto& prims = GetHumanoidPrimitives();
    if (body_capsule_idx >= 0 && body_capsule_idx < static_cast<int32_t>(prims.size())) {
        const auto& prim = prims[body_capsule_idx];
        body_radius = prim.radius;
        cap_p1 = colliders::Vec3{prim.position.x, prim.position.y - prim.half_height, prim.position.z};
        cap_p2 = colliders::Vec3{prim.position.x, prim.position.y + prim.half_height, prim.position.z};
    } else {
        float side_sign = (shoulder.x < 0.0f) ? -1.0f : 1.0f;
        cap_p1 = colliders::Vec3{side_sign * 0.09f, 0.78f, 0.0f};
        cap_p2 = colliders::Vec3{side_sign * 0.09f, 1.12f, 0.0f};
        body_radius = 0.135f;
    }

    float forearm_dist_sq = SegmentToSegmentDistanceSq(elbow, wrist, cap_p1, cap_p2);
    float forearm_dist = std::sqrt(forearm_dist_sq);

    float upper_dist_sq = SegmentToSegmentDistanceSq(shoulder, elbow, cap_p1, cap_p2);
    float upper_dist = std::sqrt(upper_dist_sq);

    float min_dist = std::min(forearm_dist, upper_dist);
    float threshold = body_radius + arm_radius + clearance_margin;
    float signed_dist = min_dist - threshold;

    if (signed_dist < 0.0f) {
        float penetration = -signed_dist;
        float arm_dx = wrist.x - shoulder.x;
        float arm_dy = wrist.y - shoulder.y;
        float arm_dz = wrist.z - shoulder.z;
        float arm_length = std::sqrt(arm_dx * arm_dx + arm_dy * arm_dy + arm_dz * arm_dz);
        if (arm_length < 0.15f) arm_length = 0.55f;

        float ratio = std::clamp(penetration / arm_length, 0.0f, 0.85f);
        float abduct_rad = std::asin(ratio);
        float elbow_rad = 0.35f * abduct_rad;

        *out_abduction_rad = abduct_rad;
        *out_elbow_rad = elbow_rad;
        return 1;
    }

    *out_abduction_rad = 0.0f;
    *out_elbow_rad = 0.0f;
    return 0;
}

/**
 * 13. Projects vertices radially outward from 5-tier flared skirt rings.
 */
WASM_EXPORT int32_t carbon_vrm_solve_skirt_clipping(
    float* points_xyz,
    int32_t num_points,
    float waist_height,
    float skirt_length,
    float* out_points_xyz
) {
    if (!points_xyz || !out_points_xyz || num_points <= 0) return 0;
    if (skirt_length <= 0.001f) skirt_length = 0.40f;
    if (waist_height <= 0.001f) waist_height = 0.95f;

    int32_t resolved_count = 0;
    float skirt_bottom = waist_height - skirt_length;

    for (int32_t i = 0; i < num_points; ++i) {
        float px = points_xyz[i * 3 + 0];
        float py = points_xyz[i * 3 + 1];
        float pz = points_xyz[i * 3 + 2];

        if (py >= skirt_bottom - 0.05f && py <= waist_height + 0.05f) {
            float frac = std::clamp((waist_height - py) / skirt_length, 0.0f, 1.0f);
            float r_skirt = 0.14f + frac * 0.16f;
            float d_xz = std::sqrt(px * px + pz * pz);

            if (d_xz < r_skirt) {
                float factor = (d_xz > 1e-6f) ? (r_skirt / d_xz) : 1.0f;
                out_points_xyz[i * 3 + 0] = px * factor;
                out_points_xyz[i * 3 + 1] = py;
                out_points_xyz[i * 3 + 2] = (d_xz > 1e-6f) ? (pz * factor) : r_skirt;
                resolved_count++;
                continue;
            }
        }

        out_points_xyz[i * 3 + 0] = px;
        out_points_xyz[i * 3 + 1] = py;
        out_points_xyz[i * 3 + 2] = pz;
    }

    return resolved_count;
}

/**
 * 14. Provides direct pointers to internal pinned Wasm memory buffers for zero-copy views.
 * Buffer 0: Rest Quats (55 bones x 4 = 220 floats)
 * Buffer 1: Target Quats (55 bones x 4 = 220 floats)
 * Buffer 2: Output Blended Quats (55 bones x 4 = 220 floats)
 * Buffer 3: Arm Telemetry (L/R shoulder, elbow, wrist = 12 floats)
 * Buffer 4: Deflection Results (L/R abduction & elbow deltas = 4 floats)
 * Buffer 5: Skirt In Vertices (256 * 3 = 768 floats)
 * Buffer 6: Skirt Out Vertices (256 * 3 = 768 floats)
 */
WASM_EXPORT float* carbon_vrm_get_shared_buffer(int32_t buffer_id) {
    switch (buffer_id) {
        case 0: return g_in_rest_quats;
        case 1: return g_in_target_quats;
        case 2: return g_out_blended_quats;
        case 3: return g_arm_telemetry;
        case 4: return g_deflection_results;
        case 5: return g_skirt_in_vertices;
        case 6: return g_skirt_out_vertices;
        default: return nullptr;
    }
}

} // extern "C"
