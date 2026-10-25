#ifndef OPENARM_BIMANUAL_GUARD_HPP
#define OPENARM_BIMANUAL_GUARD_HPP

/**
 * ============================================================================
 * 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
 * OpenArm Bimanual Hardware Safety Guard (1kHz Closed-Loop Closed-Form Math)
 * File: examples/openarm_bimanual_guard/openarm_bimanual_guard.hpp
 * ============================================================================
 * Fuses 4th-order Runge-Kutta predictive horizon integration (lookahead 15ms),
 * C4 quintic phase-space barrier damping, and analytical capsule-capsule signed
 * distance fields (SDF) with a 50mm wrist separation corridor for 16 CAN nodes:
 *   - Right Arm (CAN IDs 1..8): 7-DOF DaMiao MIT Cheetah + 1-DOF SteadyWin Claw
 *   - Left Arm  (CAN IDs 11..18): 7-DOF DaMiao MIT Cheetah + 1-DOF SteadyWin Claw
 *
 * Guarantees:
 *   - Joint limit overshoot = 0.00000 rad
 *   - Minimum bimanual clearance > 0.00 m
 *   - Motor 8 / 18 lead screw gripper stall current <= 1500mA
 */

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <array>
#include <algorithm>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <intrin.h>
  #if defined(OPENARM_BUILD_SHARED) || defined(KIMODO_BUILD_SHARED)
    #define OPENARM_EXPORT __declspec(dllexport)
  #else
    #define OPENARM_EXPORT
  #endif
  #define OPENARM_CALL __cdecl
#else
  #include <sys/mman.h>
  #include <cpuid.h>
  #define OPENARM_EXPORT __attribute__((visibility("default")))
  #define OPENARM_CALL
#endif

namespace g2g::carbon_forge {

// Total CAN actuators across both arms (Right IDs 1..8, Left IDs 11..18)
constexpr std::size_t kOpenArmActuatorCount = 16;

// Joint indexing:
//   Right Arm: [0]=J1 Yaw, [1]=J2 Pitch, [2]=J3 Bicep, [3]=J4 Elbow,
//              [4]=J5 Forearm Roll, [5]=J6 Wrist Pitch, [6]=J7 Wrist Curl, [7]=J8 Claw
//   Left Arm:  [8]=J11 Yaw, [9]=J12 Pitch, [10]=J13 Bicep, [11]=J14 Elbow,
//              [12]=J15 Forearm Roll, [13]=J16 Wrist Pitch, [14]=J17 Wrist Curl, [15]=J18 Claw

struct Vec3 {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    static constexpr Vec3 Zero() noexcept { return {0.0f, 0.0f, 0.0f}; }
    static constexpr Vec3 Make(float x, float y, float z) noexcept { return {x, y, z}; }

    [[nodiscard]] float LengthSq() const noexcept { return x * x + y * y + z * z; }
    [[nodiscard]] float Length() const noexcept { return std::sqrt(LengthSq()); }

    [[nodiscard]] Vec3 Normalized() const noexcept {
        float l = Length();
        return (l > 1.0e-7f) ? Vec3{x / l, y / l, z / l} : Zero();
    }
};

struct Capsule3D {
    Vec3 a{Vec3::Zero()};
    Vec3 b{Vec3::Zero()};
    float radius{0.045f};
};

struct BimanualGuardConfig {
    std::array<float, kOpenArmActuatorCount> limit_min;
    std::array<float, kOpenArmActuatorCount> limit_max;
    std::array<float, kOpenArmActuatorCount> tau_max;
    std::array<float, kOpenArmActuatorCount> inertia;
    std::array<float, kOpenArmActuatorCount> damping;

    float barrier_k{12000.0f};         // C4 quintic barrier stiffness (N*m)
    float barrier_d{800.0f};           // Quartic barrier velocity damping (N*m*s/rad)
    float barrier_delta{0.080f};       // Anticipatory braking corridor (rad) ~4.58 deg
    float lookahead_sec{0.015f};       // RK4 predictive horizon (15ms)
    float corridor_min{0.050f};        // Bimanual wrist/arm separation corridor (50mm)
    float max_stall_current_ma{1500.0f}; // Max claw stall current (mA)

    static BimanualGuardConfig Default() noexcept;
};

struct BimanualState {
    std::array<float, kOpenArmActuatorCount> q;
    std::array<float, kOpenArmActuatorCount> v;
    std::array<float, kOpenArmActuatorCount> tau;
};

struct BimanualGuardOutput {
    std::array<float, kOpenArmActuatorCount> q_next;
    std::array<float, kOpenArmActuatorCount> v_next;
    std::array<float, kOpenArmActuatorCount> tau_safe;
    float min_clearance{0.0f};
    float wrist_distance{0.0f};
    int32_t barrier_active_mask{0};
    int32_t status{0};
};

class BimanualGuardContext {
public:
    BimanualGuardConfig config;
    BimanualState state{};
    uint64_t step_count{0};

    BimanualGuardContext() noexcept;
    explicit BimanualGuardContext(const BimanualGuardConfig& cfg) noexcept;

    // Evaluates 1kHz closed-loop integration with predictive barriers and collision avoidance
    BimanualGuardOutput Step1kHz(
        const float* q_meas,
        const float* v_meas,
        const float* tau_cmd,
        float dt
    ) noexcept;

    // Computes analytical forward kinematics capsules and pairwise signed distances
    void EvaluateSDF(
        const float* q,
        float* out_min_dist,
        float* out_wrist_dist
    ) const noexcept;
};

} // namespace g2g::carbon_forge

// ----------------------------------------------------------------------------
// Pure C ABI Functions for Google Carbon & Python ctypes Interop
// ----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

OPENARM_EXPORT void* OPENARM_CALL openarm_bimanual_guard_init(void);

OPENARM_EXPORT void OPENARM_CALL openarm_bimanual_guard_destroy(void* ctx);

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
);

OPENARM_EXPORT int32_t OPENARM_CALL openarm_bimanual_eval_sdf(
    const float* q,
    float* out_min_dist,
    float* out_wrist_dist
);

OPENARM_EXPORT const char* OPENARM_CALL openarm_bimanual_guard_version(void);

#ifdef __cplusplus
}
#endif

#endif // OPENARM_BIMANUAL_GUARD_HPP
