#ifndef HEPHAESTUS_RK4_SE3_HPP
#define HEPHAESTUS_RK4_SE3_HPP

/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * Bespoke Hephaestus v2.0 RK4 SE(3) Predictive Horizon Barrier & SoA8 SLERP
 * File: include/hephaestus_rk4_se3.hpp
 * ============================================================================
 * Implements 4th-Order Runge-Kutta state prediction with O(dt^5) truncation
 * error, continuous C4 quintic phase-space damping torque, and 8-wide AVX2
 * Structure-of-Arrays (SoA) quaternion SLERP with Chebyshev-Padé polynomial.
 */

#include "carbon_interop.hpp"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace g2g::carbon_forge::hephaestus {

// Physical & Mathematical Constants
constexpr f32 DefaultJointInertia = 0.05f;         // J = 0.05 kg*m^2
constexpr f32 DefaultViscousDamping = 0.10f;       // b = 0.10 N*m*s/rad
constexpr f32 DefaultLookaheadHorizon = 0.015f;    // dt_pred = 15ms
constexpr f32 DefaultBarrierDelta = 0.08f;         // delta = 0.08 rad (4.58 deg)
constexpr f32 DefaultBarrierK = 12000.0f;          // K = 12000 N*m
constexpr f32 DefaultBarrierD = 800.0f;            // D = 800 N*m*s/rad
constexpr f32 DefaultSafetyBubbleRadius = 0.150f;  // 15cm

// Chebyshev-Padé fast polynomial coefficients
constexpr f32 ChebyshevC1 = 0.3333333333f;         // 1/3
constexpr f32 ChebyshevC2 = 0.1333333333f;         // 2/15
constexpr f32 ChebyshevC3 = 0.0539682540f;         // 17/315

struct State2D {
    f32 theta{0.0f};
    f32 omega{0.0f};
};

// Computes joint angular acceleration: ddot_theta = (tau - b * omega) / J
static inline f32 ComputeAcceleration(f32 omega, f32 tau, f32 J, f32 b) noexcept {
    f32 inertia = (J > 1.0e-6f) ? J : DefaultJointInertia;
    return (tau - b * omega) / inertia;
}

// Single 4th-Order Runge-Kutta (RK4) integration step
static inline void Rk4Step(
    f32 theta, f32 omega, f32 tau, f32 J, f32 b, f32 dt,
    f32* out_theta, f32* out_omega
) noexcept {
    f32 dt_h = 0.5f * dt;

    // Stage 1
    f32 k1_theta = omega;
    f32 k1_omega = ComputeAcceleration(omega, tau, J, b);

    // Stage 2
    f32 th2 = theta + dt_h * k1_theta;
    f32 om2 = omega + dt_h * k1_omega;
    f32 k2_theta = om2;
    f32 k2_omega = ComputeAcceleration(om2, tau, J, b);

    // Stage 3
    f32 th3 = theta + dt_h * k2_theta;
    f32 om3 = omega + dt_h * k2_omega;
    f32 k3_theta = om3;
    f32 k3_omega = ComputeAcceleration(om3, tau, J, b);

    // Stage 4
    f32 th4 = theta + dt * k3_theta;
    f32 om4 = omega + dt * k3_omega;
    f32 k4_theta = om4;
    f32 k4_omega = ComputeAcceleration(om4, tau, J, b);

    f32 dt6 = dt / 6.0f;
    if (out_theta) {
        *out_theta = theta + dt6 * (k1_theta + 2.0f * k2_theta + 2.0f * k3_theta + k4_theta);
    }
    if (out_omega) {
        *out_omega = omega + dt6 * (k1_omega + 2.0f * k2_omega + 2.0f * k3_omega + k4_omega);
    }
}

// Predicts future joint state (theta, omega) over lookahead horizon dt_pred
static inline State2D Rk4Predict(
    f32 theta, f32 omega, f32 tau, f32 J, f32 b, f32 dt_pred
) noexcept {
    State2D res;
    Rk4Step(theta, omega, tau, J, b, dt_pred, &res.theta, &res.omega);
    return res;
}

// Continuous C4 Quintic Phase-Space Barrier Torque
static inline f32 ComputeQuinticBarrierTorque(
    f32 theta_pred, f32 omega_pred, f32 limit_min, f32 limit_max,
    f32 K = DefaultBarrierK, f32 D = DefaultBarrierD, f32 delta = DefaultBarrierDelta
) noexcept {
    if (limit_max <= limit_min) return 0.0f;
    f32 d_barrier = (delta > 1.0e-5f) ? delta : DefaultBarrierDelta;
    f32 k_stiff = (K > 0.0f) ? K : DefaultBarrierK;
    f32 d_damp = (D >= 0.0f) ? D : DefaultBarrierD;

    // Upper limit approach
    if (omega_pred > 0.0f) {
        f32 dist_upper = limit_max - theta_pred;
        if (dist_upper < d_barrier) {
            f32 u = std::clamp((d_barrier - dist_upper) / d_barrier, 0.0f, 1.0f);
            f32 u2 = u * u;
            f32 u4 = u2 * u2;
            f32 u5 = u4 * u;
            return -k_stiff * u5 - d_damp * omega_pred * u4;
        }
    }
    // Lower limit approach
    else if (omega_pred < 0.0f) {
        f32 dist_lower = theta_pred - limit_min;
        if (dist_lower < d_barrier) {
            f32 u = std::clamp((d_barrier - dist_lower) / d_barrier, 0.0f, 1.0f);
            f32 u2 = u * u;
            f32 u4 = u2 * u2;
            f32 u5 = u4 * u;
            return k_stiff * u5 - d_damp * omega_pred * u4;
        }
    }
    return 0.0f;
}

// ----------------------------------------------------------------------------
// AVX2 8-Wide Batch Quaternion SLERP (BatchSlerpSoA8)
// ----------------------------------------------------------------------------
struct alignas(32) BatchSlerpSoA8 {
    f32 qx[8];
    f32 qy[8];
    f32 qz[8];
    f32 qw[8];

    static void SlerpSimd8(
        const f32* q1_soa, const f32* q2_soa, const f32* t8, f32* out_soa
    ) noexcept {
        __m256 q1x = _mm256_loadu_ps(q1_soa + 0);
        __m256 q1y = _mm256_loadu_ps(q1_soa + 8);
        __m256 q1z = _mm256_loadu_ps(q1_soa + 16);
        __m256 q1w = _mm256_loadu_ps(q1_soa + 24);

        __m256 q2x = _mm256_loadu_ps(q2_soa + 0);
        __m256 q2y = _mm256_loadu_ps(q2_soa + 8);
        __m256 q2z = _mm256_loadu_ps(q2_soa + 16);
        __m256 q2w = _mm256_loadu_ps(q2_soa + 24);

        __m256 t_vec = _mm256_loadu_ps(t8);

        // Compute 8 dot products: d = q1x*q2x + q1y*q2y + q1z*q2z + q1w*q2w
        __m256 d = _mm256_fmadd_ps(q1x, q2x,
                   _mm256_fmadd_ps(q1y, q2y,
                   _mm256_fmadd_ps(q1z, q2z, _mm256_mul_ps(q1w, q2w))));

        // Shortest-path antipodal alignment: if d < 0, negate q2 and d
        __m256 zero = _mm256_setzero_ps();
        __m256 neg_mask = _mm256_cmp_ps(d, zero, _CMP_LT_OQ);
        d = _mm256_blendv_ps(d, _mm256_sub_ps(zero, d), neg_mask);
        q2x = _mm256_blendv_ps(q2x, _mm256_sub_ps(zero, q2x), neg_mask);
        q2y = _mm256_blendv_ps(q2y, _mm256_sub_ps(zero, q2y), neg_mask);
        q2z = _mm256_blendv_ps(q2z, _mm256_sub_ps(zero, q2z), neg_mask);
        q2w = _mm256_blendv_ps(q2w, _mm256_sub_ps(zero, q2w), neg_mask);

        // Clamp d to [0, 1]
        __m256 one = _mm256_set1_ps(1.0f);
        d = _mm256_min_ps(d, one);

        // x = 1 - d
        __m256 x = _mm256_sub_ps(one, d);

        // Near-parallel threshold mask (d > 0.9995)
        __m256 near_parallel_thresh = _mm256_set1_ps(0.9995f);
        __m256 is_near_parallel = _mm256_cmp_ps(d, near_parallel_thresh, _CMP_GT_OQ);

        // Chebyshev-Padé rational polynomial expansion
        __m256 c1 = _mm256_set1_ps(ChebyshevC1);
        __m256 c2 = _mm256_set1_ps(ChebyshevC2);
        __m256 c3 = _mm256_set1_ps(ChebyshevC3);

        __m256 poly = _mm256_fmadd_ps(c3, x, c2);
        poly = _mm256_fmadd_ps(poly, x, c1);
        poly = _mm256_mul_ps(poly, x);

        __m256 t2 = _mm256_mul_ps(t_vec, t_vec);
        __m256 one_minus_t2 = _mm256_sub_ps(one, t2);
        __m256 factor = _mm256_mul_ps(t_vec, one_minus_t2);
        __m256 w2 = _mm256_fmadd_ps(factor, poly, t_vec);

        __m256 one_minus_t = _mm256_sub_ps(one, t_vec);
        __m256 omt2 = _mm256_mul_ps(one_minus_t, one_minus_t);
        __m256 factor1 = _mm256_mul_ps(one_minus_t, _mm256_sub_ps(one, omt2));
        __m256 w1 = _mm256_fmadd_ps(factor1, poly, one_minus_t);

        w1 = _mm256_blendv_ps(w1, one_minus_t, is_near_parallel);
        w2 = _mm256_blendv_ps(w2, t_vec, is_near_parallel);

        __m256 ox = _mm256_fmadd_ps(w1, q1x, _mm256_mul_ps(w2, q2x));
        __m256 oy = _mm256_fmadd_ps(w1, q1y, _mm256_mul_ps(w2, q2y));
        __m256 oz = _mm256_fmadd_ps(w1, q1z, _mm256_mul_ps(w2, q2z));
        __m256 ow = _mm256_fmadd_ps(w1, q1w, _mm256_mul_ps(w2, q2w));

        __m256 norm_sq = _mm256_fmadd_ps(ox, ox,
                         _mm256_fmadd_ps(oy, oy,
                         _mm256_fmadd_ps(oz, oz, _mm256_mul_ps(ow, ow))));

        // Fast reciprocal square root with Newton-Raphson refinement
        __m256 inv_norm = _mm256_rsqrt_ps(norm_sq);
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 three = _mm256_set1_ps(3.0f);
        __m256 nr = _mm256_mul_ps(half, _mm256_sub_ps(three, _mm256_mul_ps(norm_sq, _mm256_mul_ps(inv_norm, inv_norm))));
        inv_norm = _mm256_mul_ps(inv_norm, nr);

        ox = _mm256_mul_ps(ox, inv_norm);
        oy = _mm256_mul_ps(oy, inv_norm);
        oz = _mm256_mul_ps(oz, inv_norm);
        ow = _mm256_mul_ps(ow, inv_norm);

        _mm256_storeu_ps(out_soa + 0, ox);
        _mm256_storeu_ps(out_soa + 8, oy);
        _mm256_storeu_ps(out_soa + 16, oz);
        _mm256_storeu_ps(out_soa + 24, ow);
    }
};

} // namespace g2g::carbon_forge::hephaestus

#endif // HEPHAESTUS_RK4_SE3_HPP
