#ifndef CARBON_INTEROP_HPP
#define CARBON_INTEROP_HPP

/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * C++20 Lowering Bridge & Interop Header
 * File: include/carbon_interop.hpp
 * ============================================================================
 * Establishes C++20 lowering abstractions for Google Carbon types, C ABI
 * exports, 8-byte alignment, 72-byte structs, and common SO(3) math under
 * the canonical namespace g2g::carbon_forge.
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <numbers>
#include <memory>
#include <mutex>
#include <span>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #define CARBON_EXPORT __declspec(dllexport)
  #define CARBON_CALL __cdecl
#else
  #define CARBON_EXPORT __attribute__((visibility("default")))
  #define CARBON_CALL
#endif

namespace g2g::carbon_forge {

// ----------------------------------------------------------------------------
// Primitive Type Aliases (Mapping Carbon Types 1:1 to C++20)
// ----------------------------------------------------------------------------
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

// Mathematical Constants
constexpr f32 Pi = 3.14159265358979323846f;
constexpr f32 TwoPi = 6.28318530717958647692f;
constexpr f32 HalfPi = 1.57079632679489661923f;
constexpr f32 QuarterPi = 0.78539816339744830962f;

constexpr f32 DefaultDeadbandEpsilon = 0.0005f; // 0.0286479 deg
constexpr f32 DefaultDampingLambda = 18.0f;     // 18.0 s^-1

// ----------------------------------------------------------------------------
// Carbon SO(3) Unit Quaternion Class
// ----------------------------------------------------------------------------
struct alignas(4) Quaternion {
    f32 x{0.0f};
    f32 y{0.0f};
    f32 z{0.0f};
    f32 w{1.0f};

    static constexpr Quaternion Identity() noexcept {
        return Quaternion{0.0f, 0.0f, 0.0f, 1.0f};
    }

    static constexpr Quaternion Make(f32 x, f32 y, f32 z, f32 w) noexcept {
        return Quaternion{x, y, z, w};
    }

    [[nodiscard]] f32 NormSq() const noexcept {
        return x * x + y * y + z * z + w * w;
    }

    [[nodiscard]] f32 Norm() const noexcept {
        return std::sqrt(NormSq());
    }

    void Normalize() noexcept {
        f32 nsq = NormSq();
        if (nsq > 1.0e-12f) {
            f32 inv = 1.0f / std::sqrt(nsq);
            x *= inv;
            y *= inv;
            z *= inv;
            w *= inv;
        } else {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
            w = 1.0f;
        }
    }

    [[nodiscard]] Quaternion Normalized() const noexcept {
        Quaternion q = *this;
        q.Normalize();
        return q;
    }

    [[nodiscard]] Quaternion Conjugate() const noexcept {
        return Quaternion{-x, -y, -z, w};
    }

    [[nodiscard]] static f32 Dot(const Quaternion& q1, const Quaternion& q2) noexcept {
        return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    }

    [[nodiscard]] static f32 GeodesicAngle(const Quaternion& q1, const Quaternion& q2) noexcept {
        f32 xd = q1.w * q2.x - q1.x * q2.w - q1.y * q2.z + q1.z * q2.y;
        f32 yd = q1.w * q2.y + q1.x * q2.z - q1.y * q2.w - q1.z * q2.x;
        f32 zd = q1.w * q2.z - q1.x * q2.y + q1.y * q2.x - q1.z * q2.w;
        f32 wd = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        f32 v_norm = std::sqrt(xd * xd + yd * yd + zd * zd);
        f32 abs_w = std::abs(wd);
        return 2.0f * std::atan2(v_norm, abs_w);
    }

    [[nodiscard]] std::array<f32, 3> RotateVector(const std::array<f32, 3>& v) const noexcept {
        f32 tx = 2.0f * (y * v[2] - z * v[1]);
        f32 ty = 2.0f * (z * v[0] - x * v[2]);
        f32 tz = 2.0f * (x * v[1] - y * v[0]);
        return {
            v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx)
        };
    }

    [[nodiscard]] static Quaternion FromEulerXYZ(f32 rx, f32 ry, f32 rz) noexcept {
        f64 cx = std::cos(static_cast<f64>(rx) * 0.5);
        f64 cy = std::cos(static_cast<f64>(ry) * 0.5);
        f64 cz = std::cos(static_cast<f64>(rz) * 0.5);
        f64 sx = std::sin(static_cast<f64>(rx) * 0.5);
        f64 sy = std::sin(static_cast<f64>(ry) * 0.5);
        f64 sz = std::sin(static_cast<f64>(rz) * 0.5);

        f64 qw = cx * cy * cz - sx * sy * sz;
        f64 qx = sx * cy * cz + cx * sy * sz;
        f64 qy = cx * sy * cz - sx * cy * sz;
        f64 qz = cx * cy * sz + sx * sy * cz;

        f64 mag = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (mag > 1e-15) {
            f64 inv = 1.0 / mag;
            qx *= inv; qy *= inv; qz *= inv; qw *= inv;
        } else {
            qx = 0.0; qy = 0.0; qz = 0.0; qw = 1.0;
        }

        return Quaternion{
            static_cast<f32>(qx),
            static_cast<f32>(qy),
            static_cast<f32>(qz),
            static_cast<f32>(qw)
        };
    }

    [[nodiscard]] static Quaternion Slerp(const Quaternion& q1, const Quaternion& q2_in, f32 t_in) noexcept {
        Quaternion q2 = q2_in;
        f64 q1x = q1.x, q1y = q1.y, q1z = q1.z, q1w = q1.w;
        f64 q2x = q2.x, q2y = q2.y, q2z = q2.z, q2w = q2.w;

        f64 dot = q1x * q2x + q1y * q2y + q1z * q2z + q1w * q2w;
        if (dot < 0.0) {
            dot = -dot;
            q2x = -q2x; q2y = -q2y; q2z = -q2z; q2w = -q2w;
        }

        f64 t = std::clamp(static_cast<f64>(t_in), 0.0, 1.0);
        f64 ox, oy, oz, ow;

        if (dot > 0.99999995) {
            ox = q1x + t * (q2x - q1x);
            oy = q1y + t * (q2y - q1y);
            oz = q1z + t * (q2z - q1z);
            ow = q1w + t * (q2w - q1w);
        } else {
            f64 theta_0 = std::acos(std::clamp(dot, -1.0, 1.0));
            f64 theta = theta_0 * t;
            f64 sin_theta = std::sin(theta);
            f64 sin_theta_0 = std::sin(theta_0);
            f64 s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
            f64 s1 = sin_theta / sin_theta_0;
            ox = (s0 * q1x) + (s1 * q2x);
            oy = (s0 * q1y) + (s1 * q2y);
            oz = (s0 * q1z) + (s1 * q2z);
            ow = (s0 * q1w) + (s1 * q2w);
        }

        f64 mag = std::sqrt(ox * ox + oy * oy + oz * oz + ow * ow);
        if (mag > 1e-15) {
            f64 inv = 1.0 / mag;
            ox *= inv; oy *= inv; oz *= inv; ow *= inv;
        } else {
            ox = 0.0; oy = 0.0; oz = 0.0; ow = 1.0;
        }

        return Quaternion{
            static_cast<f32>(ox),
            static_cast<f32>(oy),
            static_cast<f32>(oz),
            static_cast<f32>(ow)
        };
    }
};

// ----------------------------------------------------------------------------
// MotionEnvelope: Fast-Attack Trapezoid with C1 Hermite Smoothstep Easing
// ----------------------------------------------------------------------------
class MotionEnvelope {
public:
    static constexpr f64 FastAttackTrapezoid(f64 progress, f64 attack_frac = 0.15, f64 release_frac = 0.20) noexcept {
        f64 p = std::clamp(progress, 0.0, 1.0);
        f64 att = std::clamp(attack_frac, 0.001, 0.999);
        f64 rel = std::clamp(release_frac, 0.001, 1.0 - att);

        if (p < att) {
            f64 s = p / att;
            return s * s * (3.0 - 2.0 * s);
        }
        if (p <= (1.0 - rel)) {
            return 1.0;
        }
        f64 s = (1.0 - p) / rel;
        return s * s * (3.0 - 2.0 * s);
    }

    static constexpr f32 FastAttackTrapezoid(f32 progress, f32 attack_frac = 0.15f, f32 release_frac = 0.20f) noexcept {
        return static_cast<f32>(FastAttackTrapezoid(static_cast<f64>(progress), static_cast<f64>(attack_frac), static_cast<f64>(release_frac)));
    }
};

// ----------------------------------------------------------------------------
// AngularDeadbandFilter: Unconditional 0.0005 rad Geodesic Deadband & Damping
// ----------------------------------------------------------------------------
class AngularDeadbandFilter {
private:
    f32 epsilon_{DefaultDeadbandEpsilon};
    f32 lambda_{DefaultDampingLambda};

public:
    constexpr AngularDeadbandFilter(f32 eps = DefaultDeadbandEpsilon, f32 lam = DefaultDampingLambda) noexcept
        : epsilon_(eps), lambda_(lam) {}

    [[nodiscard]] constexpr f32 GetEpsilon() const noexcept { return epsilon_; }
    constexpr void SetEpsilon(f32 eps) noexcept { epsilon_ = eps; }

    [[nodiscard]] constexpr f32 GetLambda() const noexcept { return lambda_; }
    constexpr void SetLambda(f32 lam) noexcept { lambda_ = lam; }

    [[nodiscard]] bool IsInDeadband(const Quaternion& current, const Quaternion& target) const noexcept {
        return Quaternion::GeodesicAngle(current, target) < epsilon_;
    }

    [[nodiscard]] Quaternion Filter(const Quaternion& current, const Quaternion& target, f32 delta_sec) const noexcept {
        f32 delta_rad = Quaternion::GeodesicAngle(current, target);
        if (delta_rad < epsilon_) {
            return current;
        }
        f32 dt_clamped = std::clamp(delta_sec, 0.0f, 0.03333333f);
        f32 alpha = 1.0f - std::exp(-lambda_ * dt_clamped);
        return Quaternion::Slerp(current, target, alpha);
    }
};

// ----------------------------------------------------------------------------
// BilateralClappingConstraint: Bimanual Validation
// ----------------------------------------------------------------------------
class BilateralClappingConstraint {
public:
    static constexpr f32 TargetWristSeparationM = 0.170f;
    static constexpr f32 MinPalmOppositionDot = -0.90f;
    static constexpr f32 MinThumbUpwardDot = 0.90f;

    struct ValidationResult {
        bool is_valid{false};
        f32 actual_separation_m{0.0f};
        f32 palm_dot{0.0f};
        f32 left_thumb_dot{0.0f};
        f32 right_thumb_dot{0.0f};
        bool no_x_crossover{true};
    };

    static ValidationResult Validate(
        const std::array<f32, 3>& left_wrist_pos,
        const std::array<f32, 3>& right_wrist_pos,
        const Quaternion& left_wrist_rot,
        const Quaternion& right_wrist_rot,
        f32 separation_tol_m = 0.03f
    ) noexcept {
        ValidationResult res{};
        f32 dx = right_wrist_pos[0] - left_wrist_pos[0];
        f32 dy = right_wrist_pos[1] - left_wrist_pos[1];
        f32 dz = right_wrist_pos[2] - left_wrist_pos[2];
        res.actual_separation_m = std::sqrt(dx * dx + dy * dy + dz * dz);

        bool sep_ok = std::abs(res.actual_separation_m - TargetWristSeparationM) <= separation_tol_m;

        std::array<f32, 3> left_palm_local{1.0f, 0.0f, 0.0f};
        std::array<f32, 3> right_palm_local{-1.0f, 0.0f, 0.0f};
        auto left_palm_world = left_wrist_rot.RotateVector(left_palm_local);
        auto right_palm_world = right_wrist_rot.RotateVector(right_palm_local);
        res.palm_dot = left_palm_world[0] * right_palm_world[0] +
                       left_palm_world[1] * right_palm_world[1] +
                       left_palm_world[2] * right_palm_world[2];
        bool palm_ok = res.palm_dot <= MinPalmOppositionDot;

        std::array<f32, 3> thumb_local{0.0f, 1.0f, 0.0f};
        auto left_thumb_world = left_wrist_rot.RotateVector(thumb_local);
        auto right_thumb_world = right_wrist_rot.RotateVector(thumb_local);
        res.left_thumb_dot = left_thumb_world[1];
        res.right_thumb_dot = right_thumb_world[1];
        bool thumb_ok = (res.left_thumb_dot >= MinThumbUpwardDot) &&
                        (res.right_thumb_dot >= MinThumbUpwardDot);

        res.no_x_crossover = (left_wrist_pos[0] < right_wrist_pos[0]);
        res.is_valid = sep_ok && palm_ok && thumb_ok && res.no_x_crossover;
        return res;
    }
};

// ----------------------------------------------------------------------------
// SoftBarrierEngine: C3 Soft-Spring Boundary Barrier
// ----------------------------------------------------------------------------
class SoftBarrierEngine {
public:
    static constexpr f64 DefaultBarrierStiffness = 10000.0;
    static constexpr f64 MaxBoundaryPenetrationRad = 0.01;

    static f64 ApplyCubicSoftLimit(f64 theta, f64 min_limit, f64 max_limit, f64 k_in = 10000.0) noexcept {
        f64 k = (k_in > 0.0) ? k_in : DefaultBarrierStiffness;
        if (theta > max_limit) {
            f64 e = theta - max_limit;
            f64 ke2 = k * e * e;
            f64 theta_soft = (!std::isfinite(ke2)) ? max_limit + 1.0 / std::sqrt(k) : max_limit + e / std::sqrt(1.0 + ke2);
            return std::min(theta_soft, max_limit + MaxBoundaryPenetrationRad);
        } else if (theta < min_limit) {
            f64 e = min_limit - theta;
            f64 ke2 = k * e * e;
            f64 theta_soft = (!std::isfinite(ke2)) ? min_limit - 1.0 / std::sqrt(k) : min_limit - e / std::sqrt(1.0 + ke2);
            return std::max(theta_soft, min_limit - MaxBoundaryPenetrationRad);
        }
        return theta;
    }

    static f64 ComputeRestoringTorque(f64 theta, f64 limit_min, f64 limit_max, f64 k_in = 10000.0) noexcept {
        f64 k = (k_in > 0.0) ? k_in : DefaultBarrierStiffness;
        if (theta > limit_max) {
            f64 e = theta - limit_max;
            return -k * e * e * e;
        } else if (theta < limit_min) {
            f64 e = limit_min - theta;
            return k * e * e * e;
        }
        return 0.0;
    }
};

} // namespace g2g::carbon_forge

// Seamless namespace alias for internal Carbon-to-C++20 lowering compatibility
namespace carbon = g2g::carbon_forge;

// ----------------------------------------------------------------------------
// Strict 72-Byte C ABI Struct Layouts (8-Byte Alignment Invariants)
// ----------------------------------------------------------------------------
#pragma pack(push, 8)

typedef struct alignas(8) {
    const char* model_path;          /* 8 bytes */
    const char* text_encoder_path;   /* 8 bytes */
    int32_t     compute_backend;     /* 4 bytes */
    int32_t     n_threads;           /* 4 bytes */
    int32_t     gpu_layers;          /* 4 bytes */
    int32_t     use_mmap;            /* 4 bytes */
    int32_t     use_mlock;           /* 4 bytes */
    int32_t     _pad0;               /* 4 bytes */
    int32_t     reserved[8];         /* 32 bytes */
} KimodoInitParams;

typedef struct alignas(8) {
    const char* prompt;              /* 8 bytes */
    float       duration_sec;        /* 4 bytes */
    int32_t     fps;                 /* 4 bytes */
    int32_t     ddim_steps;          /* 4 bytes */
    float       cfg_scale;           /* 4 bytes */
    uint64_t    seed;                /* 8 bytes */
    int32_t     rotation_format;     /* 4 bytes */
    int32_t     _pad0;               /* 4 bytes */
    int32_t     reserved[8];         /* 32 bytes */
} KimodoGenerationParams;

typedef struct alignas(8) {
    int32_t     num_frames;               /* 4 bytes */
    int32_t     num_joints;               /* 4 bytes */
    int32_t     fps;                      /* 4 bytes */
    int32_t     rotation_format;          /* 4 bytes */
    float*      root_positions;           /* 8 bytes */
    float*      joint_rotations;          /* 8 bytes */
    float*      root_linear_velocities;   /* 8 bytes */
    float*      root_angular_velocities;  /* 8 bytes */
    const char** joint_names;             /* 8 bytes */
    uint64_t    memory_size_bytes;        /* 8 bytes */
    void*       internal_handle;          /* 8 bytes */
} KimodoMotionBuffer;

typedef struct alignas(8) {
    char        model_name[64];
    char        architecture[32];
    int32_t     quant_type;
    int32_t     joint_count;
    int32_t     max_sequence_len;
    int32_t     embedding_dim;
    uint64_t    parameter_count;
    uint64_t    file_size_bytes;
} KimodoModelInfo;

typedef struct alignas(8) {
    char        name[128];
    uint32_t    n_dimensions;
    uint32_t    _pad0;
    uint64_t    dimensions[8];
    uint32_t    ggml_type;
    uint32_t    _pad1;
    uint64_t    byte_size;
    uint64_t    num_elements;
    uint64_t    file_offset;
} KimodoGgufTensorInfo;

#pragma pack(pop)

// Enforce Exact Size Invariants
static_assert(sizeof(KimodoInitParams) == 72, "KimodoInitParams must be exactly 72 bytes");
static_assert(sizeof(KimodoGenerationParams) == 72, "KimodoGenerationParams must be exactly 72 bytes");
static_assert(sizeof(KimodoMotionBuffer) == 72, "KimodoMotionBuffer must be exactly 72 bytes");

#endif // CARBON_INTEROP_HPP
