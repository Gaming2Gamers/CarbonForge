/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * OpenArm Bimanual Hardware Safety Guard Implementation
 * File: examples/openarm_bimanual_guard/openarm_bimanual_guard.cpp
 * ============================================================================
 */

#include "openarm_bimanual_guard.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#if defined(_WIN32)
  // Windows platform includes handled in header
#else
  #include <sys/mman.h>
#endif

namespace g2g::carbon_forge {

// ----------------------------------------------------------------------------
// Mathematical & Geometric Helpers
// ----------------------------------------------------------------------------

static inline float Dot(const Vec3& u, const Vec3& v) noexcept {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

static inline Vec3 Cross(const Vec3& u, const Vec3& v) noexcept {
    return Vec3{
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    };
}

static inline Vec3 Add(const Vec3& u, const Vec3& v) noexcept {
    return Vec3{u.x + v.x, u.y + v.y, u.z + v.z};
}

static inline Vec3 Sub(const Vec3& u, const Vec3& v) noexcept {
    return Vec3{u.x - v.x, u.y - v.y, u.z - v.z};
}

static inline Vec3 Scale(const Vec3& u, float s) noexcept {
    return Vec3{u.x * s, u.y * s, u.z * s};
}

// Rotates vector v by axis (unit vector) and angle in radians (Rodrigues formula)
static inline Vec3 RotateAxisAngle(const Vec3& v, const Vec3& axis, float angle) noexcept {
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);
    Vec3 cross_prod = Cross(axis, v);
    float dot_prod = Dot(axis, v);
    return Vec3{
        v.x * cos_a + cross_prod.x * sin_a + axis.x * dot_prod * (1.0f - cos_a),
        v.y * cos_a + cross_prod.y * sin_a + axis.y * dot_prod * (1.0f - cos_a),
        v.z * cos_a + cross_prod.z * sin_a + axis.z * dot_prod * (1.0f - cos_a)
    };
}

// Analytical segment-to-segment distance algorithm (Lumelsky / Ericson)
// Returns shortest Euclidean distance between line segment (p1 -> q1) and (p2 -> q2)
static inline float SegmentSegmentDistance(
    const Vec3& p1, const Vec3& q1,
    const Vec3& p2, const Vec3& q2,
    Vec3* out_c1 = nullptr,
    Vec3* out_c2 = nullptr
) noexcept {
    Vec3 d1 = Sub(q1, p1); // Direction of segment S1
    Vec3 d2 = Sub(q2, p2); // Direction of segment S2
    Vec3 r  = Sub(p1, p2);

    float a = Dot(d1, d1); // Squared length of S1
    float e = Dot(d2, d2); // Squared length of S2
    float f = Dot(d2, r);

    float s = 0.0f;
    float t = 0.0f;

    // Check if either or both segments degenerate into points
    if (a <= 1.0e-7f && e <= 1.0e-7f) {
        if (out_c1) *out_c1 = p1;
        if (out_c2) *out_c2 = p2;
        return Sub(p1, p2).Length();
    }
    if (a <= 1.0e-7f) {
        s = 0.0f;
        t = std::clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = Dot(d1, r);
        if (e <= 1.0e-7f) {
            t = 0.0f;
            s = std::clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = Dot(d1, d2);
            float denom = a * e - b * b;

            if (denom != 0.0f) {
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

    Vec3 c1 = Add(p1, Scale(d1, s));
    Vec3 c2 = Add(p2, Scale(d2, t));

    if (out_c1) *out_c1 = c1;
    if (out_c2) *out_c2 = c2;

    return Sub(c1, c2).Length();
}

// ----------------------------------------------------------------------------
// BimanualGuardConfig Default Calibration
// ----------------------------------------------------------------------------

BimanualGuardConfig BimanualGuardConfig::Default() noexcept {
    BimanualGuardConfig cfg;

    // Right Arm Limits (IDs 1..8)
    cfg.limit_min[0] = -1.5708f; cfg.limit_max[0] = +1.5708f; // J1 Shoulder Yaw [-90°, +90°]
    cfg.limit_min[1] = -1.0000f; cfg.limit_max[1] = +1.5708f; // J2 Shoulder Pitch [-57.3°, +90°]
    cfg.limit_min[2] = -1.5708f; cfg.limit_max[2] = +1.5708f; // J3 Bicep Rotate [-90°, +90°]
    cfg.limit_min[3] =  0.0000f; cfg.limit_max[3] = +2.2000f; // J4 Elbow Pitch [0°, +126°]
    cfg.limit_min[4] = -1.5708f; cfg.limit_max[4] = +1.5708f; // J5 Forearm Roll [-90°, +90°]
    cfg.limit_min[5] = -0.7854f; cfg.limit_max[5] = +0.7854f; // J6 Wrist Pitch [-45°, +45°]
    cfg.limit_min[6] = -0.4189f; cfg.limit_max[6] = +0.4189f; // J7 Wrist Curl [-24°, +24°]
    cfg.limit_min[7] =  0.0000f; cfg.limit_max[7] = +0.4500f; // J8 Gripper Claw [0.0 tight, 0.45 open]

    // Left Arm Limits (IDs 11..18, decade mapping)
    cfg.limit_min[8]  = -1.5708f; cfg.limit_max[8]  = +1.5708f; // J11 Shoulder Yaw [-90°, +90°]
    cfg.limit_min[9]  = -1.5708f; cfg.limit_max[9]  = +1.0000f; // J12 Shoulder Pitch (mirrored) [-90°, +57.3°]
    cfg.limit_min[10] = -1.5708f; cfg.limit_max[10] = +1.5708f; // J13 Bicep Rotate [-90°, +90°]
    cfg.limit_min[11] =  0.0000f; cfg.limit_max[11] = +2.2000f; // J14 Elbow Pitch [0°, +126°]
    cfg.limit_min[12] = -1.5708f; cfg.limit_max[12] = +1.5708f; // J15 Forearm Roll [-90°, +90°]
    cfg.limit_min[13] = -0.7854f; cfg.limit_max[13] = +2.6180f; // J16 Wrist Pitch [-45°, +150°]
    cfg.limit_min[14] = -0.4189f; cfg.limit_max[14] = +0.4189f; // J17 Wrist Curl [-24°, +24°]
    cfg.limit_min[15] =  0.0000f; cfg.limit_max[15] = +0.4500f; // J18 Gripper Claw [0.0 tight, 0.45 open]

    // Actuator torque saturation ceilings
    for (std::size_t i = 0; i < 5; ++i) {
        cfg.tau_max[i]     = 35.0f; // Right Arm DaMiao MIT J1..J5
        cfg.tau_max[i + 8] = 35.0f; // Left Arm DaMiao MIT J11..J15
    }
    for (std::size_t i = 5; i < 7; ++i) {
        cfg.tau_max[i]     = 12.0f; // Right Arm SteadyWin J6..J7
        cfg.tau_max[i + 8] = 12.0f; // Left Arm SteadyWin J16..J17
    }
    // Lead screw gripper stall limit: 1500mA current clamp corresponds to ~8.0 N*m
    cfg.tau_max[7]  = 8.0f;
    cfg.tau_max[15] = 8.0f;

    // Inertia & Damping
    for (std::size_t i = 0; i < 3; ++i) {
        cfg.inertia[i]     = 0.050f;
        cfg.inertia[i + 8] = 0.050f;
    }
    for (std::size_t i = 3; i < 5; ++i) {
        cfg.inertia[i]     = 0.030f;
        cfg.inertia[i + 8] = 0.030f;
    }
    for (std::size_t i = 5; i < 8; ++i) {
        cfg.inertia[i]     = 0.015f;
        cfg.inertia[i + 8] = 0.015f;
    }
    for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
        cfg.damping[i] = 0.10f;
    }

    cfg.barrier_k = 12000.0f;
    cfg.barrier_d = 800.0f;
    cfg.barrier_delta = 0.080f;
    cfg.lookahead_sec = 0.015f;
    cfg.corridor_min = 0.050f;
    cfg.max_stall_current_ma = 1500.0f;

    return cfg;
}

// ----------------------------------------------------------------------------
// Bimanual Forward Kinematics Capsule Generator
// ----------------------------------------------------------------------------

struct ArmKinematicsCapsules {
    Capsule3D upper_arm;
    Capsule3D forearm;
    Capsule3D hand;
    Vec3 wrist_pos;
};

static ArmKinematicsCapsules ComputeArmCapsules(
    const float* q_arm,
    bool is_left_arm
) noexcept {
    // Shoulder base position: Right arm at Y = -0.175m, Left arm at Y = +0.175m
    Vec3 p0 = Vec3::Make(0.0f, is_left_arm ? 0.175f : -0.175f, 0.0f);

    float q_yaw   = q_arm[0];
    float q_pitch = q_arm[1];
    float q_roll  = q_arm[2];
    float q_elbow = q_arm[3];
    float q_froll = q_arm[4];
    float q_wpitch= q_arm[5];
    float q_wcurl = q_arm[6];

    // Upper arm orientation: Yaw around Z, then Pitch around Y, then Roll around X
    Vec3 v_fwd = Vec3::Make(1.0f, 0.0f, 0.0f);
    Vec3 z_axis = Vec3::Make(0.0f, 0.0f, 1.0f);
    Vec3 y_axis = Vec3::Make(0.0f, 1.0f, 0.0f);

    // Shoulder Yaw (around Z)
    v_fwd = RotateAxisAngle(v_fwd, z_axis, q_yaw);
    y_axis = RotateAxisAngle(y_axis, z_axis, q_yaw);

    // Shoulder Pitch (around current Y)
    v_fwd = RotateAxisAngle(v_fwd, y_axis, q_pitch);
    z_axis = RotateAxisAngle(z_axis, y_axis, q_pitch);

    // Upper arm link vector (length 0.25m)
    Vec3 upper_vec = Scale(v_fwd, 0.25f);
    Vec3 p1 = Add(p0, upper_vec); // Elbow joint position

    // Elbow pitch (around local Y)
    v_fwd = RotateAxisAngle(v_fwd, y_axis, q_elbow);
    z_axis = RotateAxisAngle(z_axis, y_axis, q_elbow);

    // Forearm roll (around v_fwd)
    y_axis = RotateAxisAngle(y_axis, v_fwd, q_froll);
    z_axis = RotateAxisAngle(z_axis, v_fwd, q_froll);

    // Forearm link vector (length 0.23m)
    Vec3 forearm_vec = Scale(v_fwd, 0.23f);
    Vec3 p2 = Add(p1, forearm_vec); // Wrist joint position

    // Wrist pitch & curl
    v_fwd = RotateAxisAngle(v_fwd, y_axis, q_wpitch);
    v_fwd = RotateAxisAngle(v_fwd, z_axis, q_wcurl);

    // Hand/gripper vector (length 0.18m)
    Vec3 hand_vec = Scale(v_fwd, 0.18f);
    Vec3 p3 = Add(p2, hand_vec); // Gripper tip position

    ArmKinematicsCapsules res;
    res.upper_arm = {p0, p1, 0.045f}; // radius 45mm
    res.forearm   = {p1, p2, 0.040f}; // radius 40mm
    res.hand      = {p2, p3, 0.050f}; // radius 50mm
    res.wrist_pos = p2;
    return res;
}

// ----------------------------------------------------------------------------
// BimanualGuardContext Implementation
// ----------------------------------------------------------------------------

BimanualGuardContext::BimanualGuardContext() noexcept
    : config(BimanualGuardConfig::Default()) {}

BimanualGuardContext::BimanualGuardContext(const BimanualGuardConfig& cfg) noexcept
    : config(cfg) {}

void BimanualGuardContext::EvaluateSDF(
    const float* q,
    float* out_min_dist,
    float* out_wrist_dist
) const noexcept {
    if (!q) return;

    ArmKinematicsCapsules right = ComputeArmCapsules(q, false);
    ArmKinematicsCapsules left  = ComputeArmCapsules(q + 8, true);

    const Capsule3D right_caps[3] = {right.upper_arm, right.forearm, right.hand};
    const Capsule3D left_caps[3]  = {left.upper_arm,  left.forearm,  left.hand};

    float min_clearance = 1.0e9f;

    // Test all 9 capsule pairs between Left and Right arms
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float seg_dist = SegmentSegmentDistance(
                left_caps[i].a, left_caps[i].b,
                right_caps[j].a, right_caps[j].b
            );
            float clearance = seg_dist - (left_caps[i].radius + right_caps[j].radius);
            if (clearance < min_clearance) {
                min_clearance = clearance;
            }
        }
    }

    // Wrist separation distance
    float wrist_seg_dist = Sub(left.wrist_pos, right.wrist_pos).Length();
    float wrist_clearance = wrist_seg_dist - (left.hand.radius + right.hand.radius);

    if (out_min_dist) *out_min_dist = min_clearance;
    if (out_wrist_dist) *out_wrist_dist = wrist_clearance;
}

// Single 4th-Order Runge-Kutta step: ddot_q = (tau - b * v) / J
static inline void Rk4JointStep(
    float q, float v, float tau, float J, float b, float dt,
    float* out_q, float* out_v
) noexcept {
    float dt_h = 0.5f * dt;
    float inv_J = 1.0f / (J > 1.0e-6f ? J : 0.05f);

    // Stage 1
    float k1_q = v;
    float k1_v = (tau - b * v) * inv_J;

    // Stage 2
    float q2 = q + dt_h * k1_q;
    float v2 = v + dt_h * k1_v;
    float k2_q = v2;
    float k2_v = (tau - b * v2) * inv_J;

    // Stage 3
    float q3 = q + dt_h * k2_q;
    float v3 = v + dt_h * k2_v;
    float k3_q = v3;
    float k3_v = (tau - b * v3) * inv_J;

    // Stage 4
    float q4 = q + dt * k3_q;
    float v4 = v + dt * k3_v;
    float k4_q = v4;
    float k4_v = (tau - b * v4) * inv_J;

    float dt6 = dt / 6.0f;
    if (out_q) *out_q = q + dt6 * (k1_q + 2.0f * k2_q + 2.0f * k3_q + k4_q);
    if (out_v) *out_v = v + dt6 * (k1_v + 2.0f * k2_v + 2.0f * k3_v + k4_v);
}

BimanualGuardOutput BimanualGuardContext::Step1kHz(
    const float* q_meas,
    const float* v_meas,
    const float* tau_cmd,
    float dt
) noexcept {
    BimanualGuardOutput out{};
    if (!q_meas || !v_meas || !tau_cmd || dt <= 0.0f) {
        out.status = -1;
        return out;
    }

    ++step_count;
    float current_min_dist = 0.0f;
    float current_wrist_dist = 0.0f;
    EvaluateSDF(q_meas, &current_min_dist, &current_wrist_dist);

    out.min_clearance = current_min_dist;
    out.wrist_distance = current_wrist_dist;

    // ------------------------------------------------------------------------
    // Phase 1: Predictive Horizon Joint Limit Barriers (C4 Quintic Damping)
    // ------------------------------------------------------------------------
    std::array<float, kOpenArmActuatorCount> tau_joint_barrier{};
    std::array<float, kOpenArmActuatorCount> q_pred{};
    std::array<float, kOpenArmActuatorCount> v_pred{};
    int32_t active_mask = 0;

    for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
        float q_i = q_meas[i];
        float v_i = v_meas[i];
        float t_i = tau_cmd[i];
        float J_i = config.inertia[i];
        float b_i = config.damping[i];

        // 15ms Predictive Horizon RK4 projection
        Rk4JointStep(q_i, v_i, t_i, J_i, b_i, config.lookahead_sec, &q_pred[i], &v_pred[i]);

        float q_min = config.limit_min[i];
        float q_max = config.limit_max[i];
        float delta = config.barrier_delta;
        float K = config.barrier_k;
        float D = config.barrier_d;

        float barrier_torque = 0.0f;

        // Approaching upper limit
        if (q_pred[i] > q_max - delta && v_pred[i] > 0.0f) {
            float u = std::clamp((q_pred[i] - (q_max - delta)) / delta, 0.0f, 1.0f);
            float u4 = u * u * u * u;
            float u5 = u4 * u;
            barrier_torque = -K * u5 - D * v_pred[i] * u4;
            active_mask |= (1 << i);
        }
        // Approaching lower limit
        else if (q_pred[i] < q_min + delta && v_pred[i] < 0.0f) {
            float u = std::clamp(((q_min + delta) - q_pred[i]) / delta, 0.0f, 1.0f);
            float u4 = u * u * u * u;
            float u5 = u4 * u;
            barrier_torque = +K * u5 - D * v_pred[i] * u4;
            active_mask |= (1 << i);
        }

        tau_joint_barrier[i] = barrier_torque;
    }

    // ------------------------------------------------------------------------
    // Phase 2: Predictive Bimanual Collision Barrier & Wrist Corridor Guard
    // ------------------------------------------------------------------------
    float pred_min_dist = 0.0f;
    float pred_wrist_dist = 0.0f;
    EvaluateSDF(q_pred.data(), &pred_min_dist, &pred_wrist_dist);

    float eff_dist = std::min(current_min_dist, pred_min_dist);
    std::array<float, kOpenArmActuatorCount> tau_collision_barrier{};

    // Numerical gradient evaluation for joint repulsion: d(Dist)/d(q_i)
    constexpr float eps = 0.002f;
    std::array<float, kOpenArmActuatorCount> grad{};
    std::array<float, kOpenArmActuatorCount> q_plus;
    std::array<float, kOpenArmActuatorCount> q_minus;
    std::memcpy(q_plus.data(), q_meas, sizeof(float) * kOpenArmActuatorCount);
    std::memcpy(q_minus.data(), q_meas, sizeof(float) * kOpenArmActuatorCount);

    float grad_sq = 0.0f;
    for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
        if ((i < 7) || (i >= 8 && i < 15)) {
            q_plus[i] = q_meas[i] + eps;
            q_minus[i] = q_meas[i] - eps;
            float d_p = 0.0f, d_m = 0.0f, dummy = 0.0f;
            EvaluateSDF(q_plus.data(), &d_p, &dummy);
            EvaluateSDF(q_minus.data(), &d_m, &dummy);
            grad[i] = (d_p - d_m) / (2.0f * eps);
            grad_sq += grad[i] * grad[i];
            q_plus[i] = q_meas[i];
            q_minus[i] = q_meas[i];
        }
    }
    float grad_norm = std::sqrt(grad_sq);

    if (eff_dist < config.corridor_min && grad_norm > 1.0e-5f) {
        float u_coll = std::clamp((config.corridor_min - eff_dist) / config.corridor_min, 0.0f, 1.0f);
        float u4 = u_coll * u_coll * u_coll * u_coll;
        float u5 = u4 * u_coll;

        // Closing speed along clearance gradient
        float d_dot = 0.0f;
        for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
            d_dot += grad[i] * v_meas[i];
        }
        float closing_speed = (d_dot < 0.0f) ? -d_dot : 0.0f;

        float rep_mag = config.barrier_k * u5 + config.barrier_d * closing_speed * u4;

        for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
            tau_collision_barrier[i] = (grad[i] / grad_norm) * rep_mag;
        }
        active_mask |= (1 << 30); // Bit 30 marks active bimanual collision barrier
    }

    // ------------------------------------------------------------------------
    // Phase 3: Actuator Command Clamping & Closed-Loop 1kHz Integration
    // ------------------------------------------------------------------------
    for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
        float raw_tau = tau_cmd[i] + tau_joint_barrier[i] + tau_collision_barrier[i];

        // Actuator torque saturation limit
        float t_limit = config.tau_max[i];
        float safe_tau = std::clamp(raw_tau, -t_limit, t_limit);

        // Gripper claw stall current clamp (Motor 8 and 18 <= 1500mA)
        if (i == 7 || i == 15) {
            safe_tau = std::clamp(safe_tau, -config.tau_max[i], config.tau_max[i]);
        }

        out.tau_safe[i] = safe_tau;

        // Integrate next state with safe torque
        float q_next = 0.0f;
        float v_next = 0.0f;
        Rk4JointStep(q_meas[i], v_meas[i], safe_tau, config.inertia[i], config.damping[i], dt, &q_next, &v_next);

        out.q_next[i] = q_next;
        out.v_next[i] = v_next;
    }

    // Safety projection for collision avoidance (Control Barrier Filter)
    float candidate_min_dist = 0.0f;
    float candidate_wrist_dist = 0.0f;
    EvaluateSDF(out.q_next.data(), &candidate_min_dist, &candidate_wrist_dist);

    constexpr float kCriticalCorridor = 0.025f; // 25mm critical boundary
    if (candidate_min_dist < kCriticalCorridor && grad_norm > 1.0e-5f) {
        float v_dot_grad = 0.0f;
        for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
            v_dot_grad += (grad[i] / grad_norm) * out.v_next[i];
        }
        if (v_dot_grad < 0.0f) {
            for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
                float n_i = grad[i] / grad_norm;
                out.v_next[i] -= v_dot_grad * n_i;
                out.q_next[i] = q_meas[i] + dt * out.v_next[i];
            }
        }
    }

    // Analytical boundary enforcement guaranteeing max_overshoot == 0.00000 rad
    for (std::size_t i = 0; i < kOpenArmActuatorCount; ++i) {
        float q_min = config.limit_min[i];
        float q_max = config.limit_max[i];

        if (out.q_next[i] > q_max) {
            out.q_next[i] = q_max;
            if (out.v_next[i] > 0.0f) out.v_next[i] = 0.0f;
        } else if (out.q_next[i] < q_min) {
            out.q_next[i] = q_min;
            if (out.v_next[i] < 0.0f) out.v_next[i] = 0.0f;
        }
    }

    // Compute updated clearance after 1ms integration
    float next_min_dist = 0.0f;
    float next_wrist_dist = 0.0f;
    EvaluateSDF(out.q_next.data(), &next_min_dist, &next_wrist_dist);

    out.min_clearance = next_min_dist;
    out.wrist_distance = next_wrist_dist;
    out.barrier_active_mask = active_mask;
    out.status = 0;

    return out;
}

} // namespace g2g::carbon_forge

// ----------------------------------------------------------------------------
// Pure C ABI Functions for Google Carbon & Python ctypes Interop
// ----------------------------------------------------------------------------

extern "C" {

OPENARM_EXPORT void* OPENARM_CALL openarm_bimanual_guard_init(void) {
    try {
        return new g2g::carbon_forge::BimanualGuardContext();
    } catch (...) {
        return nullptr;
    }
}

OPENARM_EXPORT void OPENARM_CALL openarm_bimanual_guard_destroy(void* ctx) {
    if (ctx) {
        delete static_cast<g2g::carbon_forge::BimanualGuardContext*>(ctx);
    }
}

OPENARM_EXPORT int32_t OPENARM_CALL openarm_bimanual_guard_step_1khz(
    void* ctx,
    const float* q_meas,
    const float* v_meas,
    const float* tau_cmd,
    float dt,
    float* q_next,
    float* v_next,
    float* tau_safe,
    float* out_clearance
) {
    if (!q_meas || !v_meas || !tau_cmd || dt <= 0.0f) {
        return -1;
    }

    g2g::carbon_forge::BimanualGuardContext* guard = static_cast<g2g::carbon_forge::BimanualGuardContext*>(ctx);
    g2g::carbon_forge::BimanualGuardContext local_guard;
    g2g::carbon_forge::BimanualGuardContext* active_guard = guard ? guard : &local_guard;

    g2g::carbon_forge::BimanualGuardOutput result = active_guard->Step1kHz(q_meas, v_meas, tau_cmd, dt);

    if (q_next) {
        std::memcpy(q_next, result.q_next.data(), sizeof(float) * g2g::carbon_forge::kOpenArmActuatorCount);
    }
    if (v_next) {
        std::memcpy(v_next, result.v_next.data(), sizeof(float) * g2g::carbon_forge::kOpenArmActuatorCount);
    }
    if (tau_safe) {
        std::memcpy(tau_safe, result.tau_safe.data(), sizeof(float) * g2g::carbon_forge::kOpenArmActuatorCount);
    }
    if (out_clearance) {
        *out_clearance = result.min_clearance;
    }

    return result.status;
}

OPENARM_EXPORT int32_t OPENARM_CALL openarm_bimanual_eval_sdf(
    const float* q,
    float* out_min_dist,
    float* out_wrist_dist
) {
    if (!q) return -1;
    g2g::carbon_forge::BimanualGuardContext ctx;
    ctx.EvaluateSDF(q, out_min_dist, out_wrist_dist);
    return 0;
}

OPENARM_EXPORT const char* OPENARM_CALL openarm_bimanual_guard_version(void) {
    return "OpenArm Bimanual Hardware Guard v1.7.0 • 1kHz Closed-Loop Closed-Form Math • ∞ Meraki";
}

} // extern "C"
