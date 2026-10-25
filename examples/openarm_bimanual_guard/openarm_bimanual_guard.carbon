// ============================================================================
// 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
// Package: CarbonForge (Module: examples/openarm_bimanual_guard/openarm_bimanual_guard.carbon)
// Standard: Google Carbon Language Specification (api)
//
// OpenArm Bimanual Hardware Safety Guard Reference Architecture:
// 1kHz closed-loop integration combining 4th-order Runge-Kutta SE(3) predictive
// horizon integration (lookahead 15ms), C4 quintic phase-space barrier damping,
// and analytical capsule-capsule signed distance fields (SDF) with a 50mm wrist
// separation corridor for 16 CAN nodes:
//   - Right Arm (CAN IDs 1..8): 7-DOF DaMiao MIT Cheetah + 1-DOF SteadyWin Claw
//   - Left Arm  (CAN IDs 11..18): 7-DOF DaMiao MIT Cheetah + 1-DOF SteadyWin Claw
//
// Proposals Conformance:
//   - Proposal #0107: Packages & Libraries (package CarbonForge api;)
//   - Proposal #0722: Explicit Method Receivers ([self: Self], [addr self: Self*])
//   - Proposal #2106: Fixed-Size Arrays ([f32; 16]) and Pointer Types (T*)
//   - Proposal #0000: Zero C Preprocessor directives (#)
// ============================================================================

package CarbonForge api;

import Cpp library "openarm_bimanual_guard.hpp";

// ----------------------------------------------------------------------------
// Physics & Actuator Calibration Constants
// ----------------------------------------------------------------------------
let ActuatorCount: i32 = 16;

// C4 Quintic Phase-Space Barrier Calibration
let DefaultBarrierK: f32 = 12000.0;          // K = 12000.0 N*m
let DefaultBarrierD: f32 = 800.0;            // D = 800.0 N*m*s/rad
let DefaultBarrierDelta: f32 = 0.080;        // delta = 0.080 rad (~4.58 deg) anticipatory corridor
let DefaultLookaheadHorizon: f32 = 0.015;    // dt_pred = 15ms (0.015s)
let DefaultCorridorMin: f32 = 0.050;         // 50mm minimum separation corridor
let MaxStallCurrentMa: f32 = 1500.0;         // 1500mA claw stall current ceiling

// ----------------------------------------------------------------------------
// Vec3: 3D Single-Precision Float Vector
// ----------------------------------------------------------------------------
class Vec3 {
  var x: f32;
  var y: f32;
  var z: f32;

  fn Zero() -> Vec3 {
    return {.x = 0.0, .y = 0.0, .z = 0.0};
  }

  fn Make(x: f32, y: f32, z: f32) -> Vec3 {
    return {.x = x, .y = y, .z = z};
  }

  fn Add(a: Vec3, b: Vec3) -> Vec3 {
    return {.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z};
  }

  fn Sub(a: Vec3, b: Vec3) -> Vec3 {
    return {.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
  }

  fn Scale(a: Vec3, s: f32) -> Vec3 {
    return {.x = a.x * s, .y = a.y * s, .z = a.z * s};
  }

  fn Dot(a: Vec3, b: Vec3) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  fn LengthSq[self: Self]() -> f32 {
    return self.x * self.x + self.y * self.y + self.z * self.z;
  }

  fn Length[self: Self]() -> f32 {
    return Cpp.sqrt(self.LengthSq());
  }

  fn Normalized[self: Self]() -> Vec3 {
    let l: f32 = self.Length();
    if (l > 1.0e-7) {
      let inv: f32 = 1.0 / l;
      return {.x = self.x * inv, .y = self.y * inv, .z = self.z * inv};
    }
    return Vec3.Zero();
  }
}

// ----------------------------------------------------------------------------
// OpenArmBimanualState: 16-Actuator Positions, Velocities & Torques
// ----------------------------------------------------------------------------
class OpenArmBimanualState {
  var q: [f32; 16];
  var v: [f32; 16];
  var tau: [f32; 16];

  fn Zero() -> OpenArmBimanualState {
    var s: OpenArmBimanualState = {
      .q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      .v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      .tau = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    };
    return s;
  }

  fn GetRightArmPositions[self: Self]() -> [f32; 8] {
    var r: [f32; 8] = [
      self.q[0], self.q[1], self.q[2], self.q[3],
      self.q[4], self.q[5], self.q[6], self.q[7]
    ];
    return r;
  }

  fn GetLeftArmPositions[self: Self]() -> [f32; 8] {
    var l: [f32; 8] = [
      self.q[8], self.q[9], self.q[10], self.q[11],
      self.q[12], self.q[13], self.q[14], self.q[15]
    ];
    return l;
  }
}

// ----------------------------------------------------------------------------
// BimanualGuardConfig: Dynamic Calibration for 16 Actuators
// ----------------------------------------------------------------------------
class BimanualGuardConfig {
  var limit_min: [f32; 16];
  var limit_max: [f32; 16];
  var tau_max: [f32; 16];
  var inertia: [f32; 16];
  var damping: [f32; 16];
  var barrier_k: f32;
  var barrier_d: f32;
  var barrier_delta: f32;
  var lookahead_sec: f32;
  var corridor_min: f32;
  var max_stall_current_ma: f32;

  fn Default() -> BimanualGuardConfig {
    var cfg: BimanualGuardConfig = {
      // Right: J1..J8, Left: J11..J18
      .limit_min = [
        -1.5708, -1.0000, -1.5708, 0.0000, -1.5708, -0.7854, -0.4189, 0.0000,
        -1.5708, -1.5708, -1.5708, 0.0000, -1.5708, -0.7854, -0.4189, 0.0000
      ],
      .limit_max = [
        1.5708, 1.5708, 1.5708, 2.2000, 1.5708, 0.7854, 0.4189, 0.4500,
        1.5708, 1.0000, 1.5708, 2.2000, 1.5708, 2.6180, 0.4189, 0.4500
      ],
      .tau_max = [
        35.0, 35.0, 35.0, 35.0, 35.0, 12.0, 12.0, 8.0,
        35.0, 35.0, 35.0, 35.0, 35.0, 12.0, 12.0, 8.0
      ],
      .inertia = [
        0.050, 0.050, 0.050, 0.030, 0.030, 0.015, 0.015, 0.015,
        0.050, 0.050, 0.050, 0.030, 0.030, 0.015, 0.015, 0.015
      ],
      .damping = [
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,
        0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10
      ],
      .barrier_k = DefaultBarrierK,
      .barrier_d = DefaultBarrierD,
      .barrier_delta = DefaultBarrierDelta,
      .lookahead_sec = DefaultLookaheadHorizon,
      .corridor_min = DefaultCorridorMin,
      .max_stall_current_ma = MaxStallCurrentMa
    };
    return cfg;
  }
}

// ----------------------------------------------------------------------------
// BimanualSafetyResult: Guard Output Telemetry
// ----------------------------------------------------------------------------
class BimanualSafetyResult {
  var q_next: [f32; 16];
  var v_next: [f32; 16];
  var tau_safe: [f32; 16];
  var min_clearance: f32;
  var wrist_distance: f32;
  var barrier_active_mask: i32;
  var status: i32;

  fn IsSafe[self: Self]() -> bool {
    return self.min_clearance > 0.0 and self.status == 0;
  }

  fn HasActiveIntervention[self: Self]() -> bool {
    return self.barrier_active_mask != 0;
  }
}

// ----------------------------------------------------------------------------
// BimanualGuard: High-Level Safety Manager Class
// ----------------------------------------------------------------------------
class BimanualGuard {
  var config: BimanualGuardConfig;
  var state: OpenArmBimanualState;
  var step_count: i64;

  fn CreateDefault() -> BimanualGuard {
    var g: BimanualGuard = {
      .config = BimanualGuardConfig.Default(),
      .state = OpenArmBimanualState.Zero(),
      .step_count = 0
    };
    return g;
  }

  fn Step1kHz[addr self: Self*](
    q_meas: [f32; 16]*,
    v_meas: [f32; 16]*,
    tau_cmd: [f32; 16]*,
    dt: f32
  ) -> BimanualSafetyResult {
    var res: BimanualSafetyResult;
    var out_clearance: f32 = 0.0;

    let ret: i32 = Cpp.openarm_bimanual_guard_step_1khz(
      null,
      q_meas as f32*,
      v_meas as f32*,
      tau_cmd as f32*,
      dt,
      (res.q_next) as f32*,
      (res.v_next) as f32*,
      (res.tau_safe) as f32*,
      &out_clearance
    );

    res.min_clearance = out_clearance;
    res.status = ret;
    (*self).step_count = (*self).step_count + 1;
    return res;
  }

  fn EvalSdf[self: Self](q: [f32; 16]*) -> BimanualSafetyResult {
    var res: BimanualSafetyResult;
    var out_min_dist: f32 = 0.0;
    var out_wrist_dist: f32 = 0.0;

    let ret: i32 = Cpp.openarm_bimanual_eval_sdf(
      q as f32*,
      &out_min_dist,
      &out_wrist_dist
    );

    res.min_clearance = out_min_dist;
    res.wrist_distance = out_wrist_dist;
    res.status = ret;
    return res;
  }
}

// ----------------------------------------------------------------------------
// Pure C ABI Export Wrappers (Mapped 1:1 to C++20 Lowering Bridge)
// ----------------------------------------------------------------------------

fn openarm_bimanual_guard_init() -> i8* {
  return Cpp.openarm_bimanual_guard_init() as i8*;
}

fn openarm_bimanual_guard_destroy(ctx: i8*) -> void {
  Cpp.openarm_bimanual_guard_destroy(ctx as void*);
}

fn openarm_bimanual_guard_step_1khz(
  ctx: i8*,
  q_meas: f32*,
  v_meas: f32*,
  tau_cmd: f32*,
  dt: f32,
  q_next: f32*,
  v_next: f32*,
  tau_safe: f32*,
  out_clearance: f32*
) -> i32 {
  return Cpp.openarm_bimanual_guard_step_1khz(
    ctx as void*,
    q_meas,
    v_meas,
    tau_cmd,
    dt,
    q_next,
    v_next,
    tau_safe,
    out_clearance
  );
}

fn openarm_bimanual_eval_sdf(
  q: f32*,
  out_min_dist: f32*,
  out_wrist_dist: f32*
) -> i32 {
  return Cpp.openarm_bimanual_eval_sdf(
    q,
    out_min_dist,
    out_wrist_dist
  );
}

fn openarm_bimanual_guard_version() -> i8* {
  return Cpp.openarm_bimanual_guard_version() as i8*;
}
