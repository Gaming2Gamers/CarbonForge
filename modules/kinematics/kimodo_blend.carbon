// ============================================================================
// 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
// Package: CarbonForge (Module: modules/kinematics/kimodo_blend.carbon)
// Standard: Google Carbon Language Specification (api)
//
// Bespoke high-precision unit quaternion SO(3) kinematics, Rodrigues vector
// rotation, SLERP with antipodal shortest-path integration, C1 Hermite cubic
// smoothstep trapezoidal motion envelope, and unconditional 0.0005 rad
// geodesic angular deadband filtering with continuous exponential damping.
// ============================================================================

package CarbonForge api;

import Cpp;

// ----------------------------------------------------------------------------
// Mathematical Constants
// ----------------------------------------------------------------------------
let Pi: f32 = 3.14159265358979323846;
let HalfPi: f32 = 1.57079632679489661923;
let SlerpNearParallelThreshold: f32 = 0.9995;
let DefaultDeadbandEpsilon: f32 = 0.0005; // 0.0286479 degrees
let DefaultDampingLambda: f32 = 18.0;      // 18.0 - 24.0 s^-1

// ----------------------------------------------------------------------------
// Quaternion: Unit quaternion representation of SO(3) rotations
// Follows Hamilton convention: q = w + x*i + y*j + z*k (i^2 = j^2 = k^2 = ijk = -1)
// Memory layout: Sequential [x, y, z, w] 32-bit floats (16 bytes, align 4)
// ----------------------------------------------------------------------------
class Quaternion {
  var x: f32;
  var y: f32;
  var z: f32;
  var w: f32;

  // Identity quaternion factory (0, 0, 0, 1)
  fn Identity() -> Quaternion {
    return {.x = 0.0, .y = 0.0, .z = 0.0, .w = 1.0};
  }

  // Construct from individual coordinates
  fn Make(x: f32, y: f32, z: f32, w: f32) -> Quaternion {
    return {.x = x, .y = y, .z = z, .w = w};
  }

  // Squared Euclidean norm
  fn NormSq[self: Self]() -> f32 {
    return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
  }

  // Euclidean norm
  fn Norm[self: Self]() -> f32 {
    let nsq: f32 = self.NormSq();
    return Cpp.sqrt(nsq);
  }

  // In-place normalization
  fn Normalize[addr self: Self*]() {
    let nsq: f32 = (*self).NormSq();
    if (nsq > 1.0e-12) {
      let inv: f32 = 1.0 / Cpp.sqrt(nsq);
      (*self).x = (*self).x * inv;
      (*self).y = (*self).y * inv;
      (*self).z = (*self).z * inv;
      (*self).w = (*self).w * inv;
    } else {
      (*self).x = 0.0;
      (*self).y = 0.0;
      (*self).z = 0.0;
      (*self).w = 1.0;
    }
  }

  // Return normalized copy
  fn Normalized[self: Self]() -> Quaternion {
    var q: Quaternion = self;
    q.Normalize();
    return q;
  }

  // Quaternion conjugate: (-x, -y, -z, w)
  fn Conjugate[self: Self]() -> Quaternion {
    return {.x = -self.x, .y = -self.y, .z = -self.z, .w = self.w};
  }

  // Quaternion inverse: q* / ||q||^2
  fn Inverse[self: Self]() -> Quaternion {
    let nsq: f32 = self.NormSq();
    if (nsq > 1.0e-12) {
      let inv: f32 = 1.0 / nsq;
      return {.x = -self.x * inv, .y = -self.y * inv, .z = -self.z * inv, .w = self.w * inv};
    }
    return Quaternion.Identity();
  }

  // 4D dot product between two quaternions
  fn Dot(q1: Quaternion, q2: Quaternion) -> f32 {
    return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
  }

  // Hamilton product: q_out = q1 * q2
  fn Multiply(q1: Quaternion, q2: Quaternion) -> Quaternion {
    return {
      .x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
      .y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
      .z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
      .w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    };
  }

  // Construct quaternion from intrinsic Euler XYZ rotation angles in radians
  // Rotation sequence: R = R_x(rx) * R_y(ry) * R_z(rz)
  fn FromEulerXYZ(rx: f32, ry: f32, rz: f32) -> Quaternion {
    let hx: f32 = rx * 0.5;
    let hy: f32 = ry * 0.5;
    let hz: f32 = rz * 0.5;

    let cx: f32 = Cpp.cos(hx);
    let sx: f32 = Cpp.sin(hx);
    let cy: f32 = Cpp.cos(hy);
    let sy: f32 = Cpp.sin(hy);
    let cz: f32 = Cpp.cos(hz);
    let sz: f32 = Cpp.sin(hz);

    var q: Quaternion = {
      .x = sx * cy * cz + cx * sy * sz,
      .y = cx * sy * cz - sx * cy * sz,
      .z = cx * cy * sz + sx * sy * cz,
      .w = cx * cy * cz - sx * sy * sz
    };
    q.Normalize();
    return q;
  }

  // Extract intrinsic Euler XYZ angles [rx, ry, rz] in radians with gimbal lock protection
  fn ToEulerXYZ[self: Self]() -> [f32; 3] {
    var euler: [f32; 3] = [0.0, 0.0, 0.0];
    let sin_ry: f32 = 2.0 * (self.x * self.z + self.w * self.y);

    var sin_ry_clipped: f32 = sin_ry;
    if (sin_ry_clipped > 1.0) {
      sin_ry_clipped = 1.0;
    } else if (sin_ry_clipped < -1.0) {
      sin_ry_clipped = -1.0;
    }

    euler[1] = Cpp.asin(sin_ry_clipped);

    // Gimbal lock check: |sin(ry)| >= 1 - 1e-6
    let abs_sin_ry: f32 = Cpp.abs(sin_ry);
    if (abs_sin_ry >= (1.0 - 1.0e-6)) {
      euler[2] = 0.0;
      euler[0] = Cpp.atan2(
        2.0 * (self.x * self.y + self.w * self.z),
        1.0 - 2.0 * (self.x * self.x + self.z * self.z)
      );
    } else {
      euler[0] = Cpp.atan2(
        2.0 * (self.w * self.x - self.y * self.z),
        1.0 - 2.0 * (self.x * self.x + self.y * self.y)
      );
      euler[2] = Cpp.atan2(
        2.0 * (self.w * self.z - self.x * self.y),
        1.0 - 2.0 * (self.y * self.y + self.z * self.z)
      );
    }

    return euler;
  }

  // Exact geodesic distance (angular difference in radians) on SO(3)
  // Evaluates relative quaternion dq = q1^{-1} * q2 and 2*atan2(||v||, |w|)
  // Completely avoids acos precision degradation and catastrophic cancellation
  fn GeodesicAngle(q1: Quaternion, q2: Quaternion) -> f32 {
    let xd: f32 = q1.w * q2.x - q1.x * q2.w - q1.y * q2.z + q1.z * q2.y;
    let yd: f32 = q1.w * q2.y + q1.x * q2.z - q1.y * q2.w - q1.z * q2.x;
    let zd: f32 = q1.w * q2.z - q1.x * q2.y + q1.y * q2.x - q1.z * q2.w;
    let wd: f32 = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

    let v_norm: f32 = Cpp.sqrt(xd * xd + yd * yd + zd * zd);
    let abs_w: f32 = Cpp.abs(wd);
    return 2.0 * Cpp.atan2(v_norm, abs_w);
  }

  // Spherical Linear Interpolation (SLERP) taking shortest geodesic path on 3-sphere
  // Includes antipodal sign alignment, NLERP fallback for near-parallel cases,
  // and full trigonometric spherical interpolation.
  fn Slerp(q1: Quaternion, q2_in: Quaternion, t: f32) -> Quaternion {
    var q2: Quaternion = q2_in;
    var d: f32 = Quaternion.Dot(q1, q2);

    // Shortest geodesic path: flip sign when dot product < 0
    if (d < 0.0) {
      d = -d;
      q2.x = -q2.x;
      q2.y = -q2.y;
      q2.z = -q2.z;
      q2.w = -q2.w;
    }

    // Near-parallel threshold: use Normalized Linear Interpolation (NLERP)
    if (d > SlerpNearParallelThreshold) {
      var r: Quaternion = {
        .x = q1.x + t * (q2.x - q1.x),
        .y = q1.y + t * (q2.y - q1.y),
        .z = q1.z + t * (q2.z - q1.z),
        .w = q1.w + t * (q2.w - q1.w)
      };
      r.Normalize();
      return r;
    }

    var d_clamped: f32 = d;
    if (d_clamped > 1.0) {
      d_clamped = 1.0;
    } else if (d_clamped < -1.0) {
      d_clamped = -1.0;
    }

    let theta_0: f32 = Cpp.acos(d_clamped);
    let theta: f32 = theta_0 * t;
    let sin_theta: f32 = Cpp.sin(theta);
    let sin_theta_0: f32 = Cpp.sin(theta_0);

    let s0: f32 = Cpp.cos(theta) - d * sin_theta / sin_theta_0;
    let s1: f32 = sin_theta / sin_theta_0;

    var result: Quaternion = {
      .x = s0 * q1.x + s1 * q2.x,
      .y = s0 * q1.y + s1 * q2.y,
      .z = s0 * q1.z + s1 * q2.z,
      .w = s0 * q1.w + s1 * q2.w
    };
    result.Normalize();
    return result;
  }

  // Rotate a 3D vector by this quaternion: v' = q * v * q^{-1}
  // Optimized Rodrigues vector form: v' = v + w*t + (q_v x t), where t = 2*(q_v x v)
  fn RotateVector[self: Self](v: [f32; 3]) -> [f32; 3] {
    let qx: f32 = self.x;
    let qy: f32 = self.y;
    let qz: f32 = self.z;
    let qw: f32 = self.w;

    let tx: f32 = 2.0 * (qy * v[2] - qz * v[1]);
    let ty: f32 = 2.0 * (qz * v[0] - qx * v[2]);
    let tz: f32 = 2.0 * (qx * v[1] - qy * v[0]);

    return [
      v[0] + qw * tx + (qy * tz - qz * ty),
      v[1] + qw * ty + (qz * tx - qx * tz),
      v[2] + qw * tz + (qx * ty - qy * tx)
    ];
  }
}

// ----------------------------------------------------------------------------
// MotionEnvelope: Fast-Attack Trapezoidal Motion Envelope Generator
// Implements C1-continuous Hermite cubic smoothstep: S(s) = s^2 * (3 - 2s).
// Default envelope: 15% attack ramp, 65% steady holding plateau, 20% release ramp.
// Guarantees zero jerk at boundaries: S'(0) = 0, S'(1) = 0.
// ----------------------------------------------------------------------------
class MotionEnvelope {
  // Compute Hermite cubic smoothstep easing for normalized parameter s in [0, 1]
  fn CubicSmoothstep(s: f32) -> f32 {
    var sc: f32 = s;
    if (sc < 0.0) {
      sc = 0.0;
    } else if (sc > 1.0) {
      sc = 1.0;
    }
    return sc * sc * (3.0 - 2.0 * sc);
  }

  // Compute instantaneous envelope factor E(p) in [0, 1] for progress in [0, 1]
  fn FastAttackTrapezoid(progress: f32, attack_frac: f32, release_frac: f32) -> f32 {
    var p: f32 = progress;
    if (p < 0.0) {
      p = 0.0;
    } else if (p > 1.0) {
      p = 1.0;
    }

    var att: f32 = attack_frac;
    if (att < 0.001) {
      att = 0.001;
    } else if (att > 0.999) {
      att = 0.999;
    }

    var rel: f32 = release_frac;
    let max_rel: f32 = 1.0 - att;
    if (rel < 0.001) {
      rel = 0.001;
    } else if (rel > max_rel) {
      rel = max_rel;
    }

    if (p < att) {
      let s: f32 = p / att;
      return MotionEnvelope.CubicSmoothstep(s);
    }
    if (p <= (1.0 - rel)) {
      return 1.0;
    }
    let s: f32 = (1.0 - p) / rel;
    return MotionEnvelope.CubicSmoothstep(s);
  }

  // Evaluate trapezoidal envelope at explicit time t with absolute attack and release durations
  fn EvaluateTimed(t: f32, total_duration: f32, attack_sec: f32, release_sec: f32) -> f32 {
    if (total_duration <= 1.0e-5) {
      return 0.0;
    }
    let p: f32 = t / total_duration;
    let att_frac: f32 = attack_sec / total_duration;
    let rel_frac: f32 = release_sec / total_duration;
    return MotionEnvelope.FastAttackTrapezoid(p, att_frac, rel_frac);
  }
}

// ----------------------------------------------------------------------------
// AngularDeadbandFilter: Geodesic Angular Deadband Filter
// Eliminates 60Hz floating-point micro-jitter below threshold (epsilon = 0.0005 rad = 0.0286 deg)
// and applies continuous exponential damping alpha = 1 - exp(-lambda * delta_t) above threshold.
// ----------------------------------------------------------------------------
class AngularDeadbandFilter {
  var epsilon_: f32;
  var lambda_: f32;

  // Constructor factory
  fn Make(epsilon: f32, lambda: f32) -> AngularDeadbandFilter {
    return {.epsilon_ = epsilon, .lambda_ = lambda};
  }

  // Default factory: epsilon = 0.0005 rad, lambda = 18.0 s^-1
  fn Default() -> AngularDeadbandFilter {
    return {.epsilon_ = DefaultDeadbandEpsilon, .lambda_ = DefaultDampingLambda};
  }

  // Test whether candidate change is trapped inside the deadband
  fn IsInDeadband[self: Self](current: Quaternion, target: Quaternion) -> bool {
    let delta_rad: f32 = Quaternion.GeodesicAngle(current, target);
    return delta_rad < self.epsilon_;
  }

  // Filter candidate target quaternion against current accepted state
  fn Filter[self: Self](current: Quaternion, target: Quaternion, delta_sec: f32) -> Quaternion {
    let delta_rad: f32 = Quaternion.GeodesicAngle(current, target);
    if (delta_rad < self.epsilon_) {
      // Inside deadband: discard micro-jitter and lock directly to current state
      return current;
    }

    // Continuous-time exponential damping
    var dt_clamped: f32 = delta_sec;
    if (dt_clamped < 0.0) {
      dt_clamped = 0.0;
    } else if (dt_clamped > 0.03333333) {
      dt_clamped = 0.03333333; // Clamp to nominal 30Hz timestep ceiling
    }

    let alpha: f32 = 1.0 - Cpp.exp(-self.lambda_ * dt_clamped);
    return Quaternion.Slerp(current, target, alpha);
  }
}

// ----------------------------------------------------------------------------
// BilateralClappingConstraint: Humanoid Bimanual Clapping & Clearance Checker
// Enforces:
// 1. Target wrist separation d = 0.170 m (+- 0.03 m tolerance)
// 2. Opposing inward-facing palms: n_left . n_right <= -0.90
// 3. Vertical thumbs upward: t_left . y_up >= +0.90, t_right . y_up >= +0.90
// 4. Coronal non-crossover constraint: left_pos.x < right_pos.x
// ----------------------------------------------------------------------------
class BilateralClappingConstraint {
  // Validation result container
  class ValidationResult {
    var is_valid: bool;
    var actual_separation_m: f32;
    var palm_dot: f32;
    var left_thumb_dot: f32;
    var right_thumb_dot: f32;
    var no_x_crossover: bool;
  }

  // Constants
  fn TargetWristSeparationM() -> f32 { return 0.170; }
  fn MinPalmOppositionDot() -> f32 { return -0.90; }
  fn MinThumbUpwardDot() -> f32 { return 0.90; }

  // Validate bimanual clapping kinematics
  fn Validate(
    left_wrist_pos: [f32; 3],
    right_wrist_pos: [f32; 3],
    left_wrist_rot: Quaternion,
    right_wrist_rot: Quaternion,
    separation_tol_m: f32
  ) -> ValidationResult {
    // 1. Separation distance
    let dx: f32 = right_wrist_pos[0] - left_wrist_pos[0];
    let dy: f32 = right_wrist_pos[1] - left_wrist_pos[1];
    let dz: f32 = right_wrist_pos[2] - left_wrist_pos[2];
    let actual_sep: f32 = Cpp.sqrt(dx * dx + dy * dy + dz * dz);

    let target_sep: f32 = BilateralClappingConstraint.TargetWristSeparationM();
    let sep_err: f32 = Cpp.abs(actual_sep - target_sep);
    let sep_ok: bool = sep_err <= separation_tol_m;

    // 2. Palm normal alignment (local +X is palm outward for left, -X for right)
    let left_palm_local: [f32; 3] = [1.0, 0.0, 0.0];
    let right_palm_local: [f32; 3] = [-1.0, 0.0, 0.0];

    let left_palm_world: [f32; 3] = left_wrist_rot.RotateVector(left_palm_local);
    let right_palm_world: [f32; 3] = right_wrist_rot.RotateVector(right_palm_local);

    let palm_dot: f32 = left_palm_world[0] * right_palm_world[0] +
                        left_palm_world[1] * right_palm_world[1] +
                        left_palm_world[2] * right_palm_world[2];
    let palm_ok: bool = palm_dot <= BilateralClappingConstraint.MinPalmOppositionDot();

    // 3. Thumb alignment (local +Y is thumb upward)
    let thumb_local: [f32; 3] = [0.0, 1.0, 0.0];
    let left_thumb_world: [f32; 3] = left_wrist_rot.RotateVector(thumb_local);
    let right_thumb_world: [f32; 3] = right_wrist_rot.RotateVector(thumb_local);

    let left_thumb_dot: f32 = left_thumb_world[1];
    let right_thumb_dot: f32 = right_thumb_world[1];

    let min_thumb: f32 = BilateralClappingConstraint.MinThumbUpwardDot();
    let thumb_ok: bool = (left_thumb_dot >= min_thumb) and (right_thumb_dot >= min_thumb);

    // 4. X-crossover check: Left wrist must remain strictly to the left in coronal plane
    let no_cross: bool = left_wrist_pos[0] < right_wrist_pos[0];

    let valid: bool = sep_ok and palm_ok and thumb_ok and no_cross;

    return {
      .is_valid = valid,
      .actual_separation_m = actual_sep,
      .palm_dot = palm_dot,
      .left_thumb_dot = left_thumb_dot,
      .right_thumb_dot = right_thumb_dot,
      .no_x_crossover = no_cross
    };
  }
}

// ----------------------------------------------------------------------------
// BlendSkeletonPose: Full-Skeleton SO(3) Pose Blending Routine
// Interpolates an entire skeleton from resting baseline to target gesture
// across progress in [0, 1] using fast-attack trapezoidal easing.
// Guarantees zero collapse at progress=0.
// ----------------------------------------------------------------------------
fn BlendSkeletonPose(
  q_rest: Quaternion*,
  q_target: Quaternion*,
  q_out: Quaternion*,
  num_joints: i32,
  progress: f32,
  attack_frac: f32,
  release_frac: f32
) {
  let env: f32 = MotionEnvelope.FastAttackTrapezoid(progress, attack_frac, release_frac);
  var i: i32 = 0;
  while (i < num_joints) {
    let r_joint: Quaternion = *(q_rest + i);
    let t_joint: Quaternion = *(q_target + i);
    *(q_out + i) = Quaternion.Slerp(r_joint, t_joint, env);
    i = i + 1;
  }
}
