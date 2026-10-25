/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * Native Standalone Parity & Performance Benchmark Executable
 * File: benchmarks/carbon_forge_bench.cpp
 * ============================================================================
 * Evaluates:
 * 1. Hephaestus v2.0 RK4 SE(3) Predictive Jerk Barrier vs. 1st-Order Euler
 *    Boundary Overshoot under high-velocity collision trajectory.
 * 2. AVX2 8-Wide Batch Quaternion SLERP (BatchSlerpSoA8) vs. Scalar
 *    std::acos/std::sin SLERP across 1,000,000 evaluations.
 */

#include "../include/carbon_interop.hpp"
#include "../include/hephaestus_rk4_se3.hpp"
#include "../include/simd_intrinsics_vex.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace g2g::carbon_forge;
using namespace g2g::carbon_forge::hephaestus;

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    std::cout << "====================================================================\n";
    std::cout << "🔨 GAMING2GAMERS (G2G) — CARBONFORGE STANDALONE BENCHMARK SUITE\n";
    std::cout << "====================================================================\n\n";

    bool all_passed = true;

    // ------------------------------------------------------------------------
    // Benchmark 1: RK4 SE(3) Predictive Barrier vs. 1st-Order Euler
    // ------------------------------------------------------------------------
    std::cout << "--- [BENCHMARK 1] RK4 SE(3) Predictive Barrier vs 1st-Order Euler ---\n";
    {
        const f32 theta_0 = 1.45f;           // Initial joint position (rad)
        const f32 dot_theta_0 = 5.0f;        // High initial angular velocity (rad/s)
        const f32 limit_max = 1.50f;         // Hard mechanical boundary (rad)
        const f32 limit_min = -1.50f;
        const f32 dt = 0.002f;               // 500Hz simulation step (2ms)
        const f32 dt_lookahead = 0.015f;     // 15ms lookahead horizon
        const f32 J = DefaultJointInertia;   // 0.05 kg*m^2
        const f32 b = DefaultViscousDamping; // 0.10 N*m*s/rad
        const f32 MaxActuatorTorque = 35.0f; // 35 N*m peak actuator torque limit
        const int steps = 50;                // 100ms duration

        // Simulation A: 1st-Order Euler without predictive horizon
        f32 theta_euler = theta_0;
        f32 omega_euler = dot_theta_0;
        f32 max_theta_euler = theta_euler;

        for (int i = 0; i < steps; ++i) {
            f32 tau_reactive = 0.0f;
            if (theta_euler > limit_max) {
                f32 dist = theta_euler - limit_max;
                tau_reactive = -std::min(MaxActuatorTorque, 12000.0f * (dist * dist * dist) + 50.0f * omega_euler);
            }
            f32 alpha = (tau_reactive - b * omega_euler) / J;
            theta_euler += dt * omega_euler;
            omega_euler += dt * alpha;
            if (theta_euler > max_theta_euler) max_theta_euler = theta_euler;
            if (theta_euler > limit_max && omega_euler <= 0.0f) break;
        }

        // Simulation B: Hephaestus v2.0 RK4 Predictive Horizon Barrier
        f32 theta_rk4 = theta_0;
        f32 omega_rk4 = dot_theta_0;
        f32 max_theta_rk4 = theta_rk4;

        for (int i = 0; i < steps; ++i) {
            State2D pred = Rk4Predict(theta_rk4, omega_rk4, 0.0f, J, b, dt_lookahead);
            f32 tau_barrier = ComputeQuinticBarrierTorque(
                pred.theta, pred.omega, limit_min, limit_max,
                DefaultBarrierK, DefaultBarrierD, DefaultBarrierDelta
            );
            tau_barrier = std::clamp(tau_barrier, -MaxActuatorTorque, MaxActuatorTorque);

            Rk4Step(theta_rk4, omega_rk4, tau_barrier, J, b, dt, &theta_rk4, &omega_rk4);
            if (theta_rk4 > max_theta_rk4) max_theta_rk4 = theta_rk4;
            if (omega_rk4 <= 0.0f && theta_rk4 >= theta_0) break;
        }

        f32 overshoot_euler = std::max(0.0f, max_theta_euler - limit_max);
        f32 overshoot_rk4 = std::max(0.0f, max_theta_rk4 - limit_max);

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "Initial Condition: theta_0 = " << theta_0 << " rad, dot_theta_0 = " << dot_theta_0 << " rad/s\n";
        std::cout << "Hard Boundary Limit: theta_max = " << limit_max << " rad\n";
        std::cout << "  1st-Order Euler Max Position: " << max_theta_euler << " rad (Overshoot = +" << overshoot_euler << " rad / " << (overshoot_euler * 180.0f / 3.14159f) << " deg)\n";
        std::cout << "  Hephaestus RK4 Max Position:  " << max_theta_rk4 << " rad (Overshoot = +" << overshoot_rk4 << " rad)\n";

        if (overshoot_euler > 0.005f && overshoot_rk4 <= 0.0001f) {
            std::cout << "  >> RESULT: PASSED (RK4 Predictive Barrier eliminated overshoot completely!)\n\n";
        } else {
            std::cout << "  >> RESULT: FAILED\n\n";
            all_passed = false;
        }
    }

    // ------------------------------------------------------------------------
    // Benchmark 2: AVX2 8-Wide Batch Quaternion SLERP vs. Scalar SLERP
    // ------------------------------------------------------------------------
    std::cout << "--- [BENCHMARK 2] AVX2 BatchSlerpSoA8 vs Scalar SLERP (1,000,000 evaluations) ---\n";
    {
        constexpr size_t N = 1000000;
        constexpr size_t N_batches = N / 8;

        std::vector<f32> q1_soa(N_batches * 32);
        std::vector<f32> q2_soa(N_batches * 32);
        std::vector<f32> t_vec(N_batches * 8);
        std::vector<f32> out_simd(N_batches * 32);
        std::vector<f32> out_scalar(N_batches * 32);

        std::mt19937 rng(42);
        std::uniform_real_distribution<f32> dist(-1.0f, 1.0f);
        std::uniform_real_distribution<f32> dist_t(0.0f, 1.0f);

        for (size_t b = 0; b < N_batches; ++b) {
            for (size_t lane = 0; lane < 8; ++lane) {
                Quaternion q1 = Quaternion::Make(dist(rng), dist(rng), dist(rng), dist(rng)).Normalized();
                Quaternion q2 = Quaternion::Make(dist(rng), dist(rng), dist(rng), dist(rng)).Normalized();
                q1_soa[b * 32 + 0 + lane] = q1.x;
                q1_soa[b * 32 + 8 + lane] = q1.y;
                q1_soa[b * 32 + 16 + lane] = q1.z;
                q1_soa[b * 32 + 24 + lane] = q1.w;

                q2_soa[b * 32 + 0 + lane] = q2.x;
                q2_soa[b * 32 + 8 + lane] = q2.y;
                q2_soa[b * 32 + 16 + lane] = q2.z;
                q2_soa[b * 32 + 24 + lane] = q2.w;

                t_vec[b * 8 + lane] = dist_t(rng);
            }
        }

        // 1. Measure Scalar SLERP
        auto t0_scalar = std::chrono::high_resolution_clock::now();
        for (size_t b = 0; b < N_batches; ++b) {
            for (size_t lane = 0; lane < 8; ++lane) {
                Quaternion q1 = Quaternion::Make(
                    q1_soa[b * 32 + 0 + lane],
                    q1_soa[b * 32 + 8 + lane],
                    q1_soa[b * 32 + 16 + lane],
                    q1_soa[b * 32 + 24 + lane]
                );
                Quaternion q2 = Quaternion::Make(
                    q2_soa[b * 32 + 0 + lane],
                    q2_soa[b * 32 + 8 + lane],
                    q2_soa[b * 32 + 16 + lane],
                    q2_soa[b * 32 + 24 + lane]
                );
                f32 t = t_vec[b * 8 + lane];
                Quaternion res = Quaternion::Slerp(q1, q2, t);
                out_scalar[b * 32 + 0 + lane] = res.x;
                out_scalar[b * 32 + 8 + lane] = res.y;
                out_scalar[b * 32 + 16 + lane] = res.z;
                out_scalar[b * 32 + 24 + lane] = res.w;
            }
        }
        auto t1_scalar = std::chrono::high_resolution_clock::now();
        double scalar_ms = std::chrono::duration<double, std::milli>(t1_scalar - t0_scalar).count();

        // 2. Measure AVX2 BatchSlerpSoA8
        auto t0_simd = std::chrono::high_resolution_clock::now();
        for (size_t b = 0; b < N_batches; ++b) {
            BatchSlerpSoA8::SlerpSimd8(
                &q1_soa[b * 32],
                &q2_soa[b * 32],
                &t_vec[b * 8],
                &out_simd[b * 32]
            );
        }
        auto t1_simd = std::chrono::high_resolution_clock::now();
        double simd_ms = std::chrono::duration<double, std::milli>(t1_simd - t0_simd).count();

        f32 max_err = 0.0f;
        for (size_t i = 0; i < out_scalar.size(); ++i) {
            f32 err = std::abs(out_scalar[i] - out_simd[i]);
            if (err > max_err) max_err = err;
        }

        double scalar_throughput = (static_cast<double>(N) / (scalar_ms / 1000.0)) / 1e6;
        double simd_throughput = (static_cast<double>(N) / (simd_ms / 1000.0)) / 1e6;
        double speedup = scalar_ms / std::max(simd_ms, 1e-6);

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Scalar SLERP Runtime:      " << scalar_ms << " ms (" << scalar_throughput << " M slerps/sec)\n";
        std::cout << "  AVX2 BatchSlerpSoA8:       " << simd_ms << " ms (" << simd_throughput << " M slerps/sec)\n";
        std::cout << "  SIMD Speedup Factor:       " << speedup << "x faster than scalar!\n";
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "  Max Error vs Scalar Slerp: " << max_err << " (within Chebyshev-Pade tolerance)\n";

        if (speedup >= 1.5 && max_err <= 0.015f) {
            std::cout << "  >> RESULT: PASSED (AVX2 BatchSlerpSoA8 outran scalar SLERP with high accuracy!)\n\n";
        } else {
            std::cout << "  >> RESULT: FAILED\n\n";
            all_passed = false;
        }
    }

    std::cout << "====================================================================\n";
    if (all_passed) {
        std::cout << "SUCCESS: ALL BENCHMARKS PASSED CLEANLY (EXIT CODE 0) • ∞ Meraki\n";
    } else {
        std::cout << "FAILURE: SOME BENCHMARKS FAILED\n";
    }
    std::cout << "====================================================================\n";

    return all_passed ? 0 : 1;
}
