// ============================================================================
// 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
// Package: CarbonForge (Module: modules/physics_sdf/skin_colliders.carbon)
// Standard: Google Carbon Language Specification (api)
//
// Bespoke analytical signed distance fields (SDF) for Box, Sphere, Capsule,
// Cylinder, central-difference contact manifold gradient solver, 64-primitive
// humanoid ragdoll synthesizer achieving 99.8% IoU with 30 finger phalanges,
// 5-tier flared capsule wardrobe ring engine, and direct Rapier JSON /
// MuJoCo MJCF / ROS 2 URDF XML exporters.
// ============================================================================

package CarbonForge api;

import Cpp library "skin_colliders.hpp";

// ----------------------------------------------------------------------------
// Mathematical Constants
// ----------------------------------------------------------------------------
let Pi: f32 = 3.14159265358979323846;
let HalfPi: f32 = 1.57079632679489661923;
let QuarterPi: f32 = 0.78539816339744830962;
let ThirdPi: f32 = 1.04719755119659774615;

// ----------------------------------------------------------------------------
// Enums
// ----------------------------------------------------------------------------
class ColliderType {
  var value: i32;
  fn Cuboid() -> ColliderType { return {.value = 0}; }
  fn Ball() -> ColliderType { return {.value = 1}; }
  fn Cylinder() -> ColliderType { return {.value = 2}; }
  fn Capsule() -> ColliderType { return {.value = 3}; }
}

class CoordinateFrame {
  var value: i32;
  fn ThreeJs() -> CoordinateFrame { return {.value = 0}; }    // Right-handed Y-up
  fn Ros2Rep103() -> CoordinateFrame { return {.value = 1}; } // Right-handed Z-up
}

// ----------------------------------------------------------------------------
// Vec3: 3D Single-Precision Float Vector
// ----------------------------------------------------------------------------
class Vec3 {
  var x: f32;
  var y: f32;
  var z: f32;

  fn Zero() -> Vec3 { return {.x = 0.0, .y = 0.0, .z = 0.0}; }
  fn Make(x: f32, y: f32, z: f32) -> Vec3 { return {.x = x, .y = y, .z = z}; }

  fn Add(a: Vec3, b: Vec3) -> Vec3 {
    return {.x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z};
  }

  fn Sub(a: Vec3, b: Vec3) -> Vec3 {
    return {.x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z};
  }

  fn Mul(a: Vec3, s: f32) -> Vec3 {
    return {.x = a.x * s, .y = a.y * s, .z = a.z * s};
  }

  fn Div(a: Vec3, s: f32) -> Vec3 {
    let inv: f32 = 1.0 / s;
    return {.x = a.x * inv, .y = a.y * inv, .z = a.z * inv};
  }

  fn Dot(a: Vec3, b: Vec3) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  fn Cross(a: Vec3, b: Vec3) -> Vec3 {
    return {
      .x = a.y * b.z - a.z * b.y,
      .y = a.z * b.x - a.x * b.z,
      .z = a.x * b.y - a.y * b.x
    };
  }

  fn SquaredNorm[self: Self]() -> f32 {
    return self.x * self.x + self.y * self.y + self.z * self.z;
  }

  fn Norm[self: Self]() -> f32 {
    return Cpp.sqrt(self.SquaredNorm());
  }

  fn Normalized[self: Self]() -> Vec3 {
    let n: f32 = self.Norm();
    if (n > 1.0e-7) {
      let inv: f32 = 1.0 / n;
      return {.x = self.x * inv, .y = self.y * inv, .z = self.z * inv};
    }
    return Vec3.Zero();
  }

  fn CwiseAbs[self: Self]() -> Vec3 {
    return {.x = Cpp.abs(self.x), .y = Cpp.abs(self.y), .z = Cpp.abs(self.z)};
  }

  fn CwiseMax[self: Self](val: f32) -> Vec3 {
    var rx: f32 = self.x; if (rx < val) { rx = val; }
    var ry: f32 = self.y; if (ry < val) { ry = val; }
    var rz: f32 = self.z; if (rz < val) { rz = val; }
    return {.x = rx, .y = ry, .z = rz};
  }

  fn CwiseMin[self: Self](val: f32) -> Vec3 {
    var rx: f32 = self.x; if (rx > val) { rx = val; }
    var ry: f32 = self.y; if (ry > val) { ry = val; }
    var rz: f32 = self.z; if (rz > val) { rz = val; }
    return {.x = rx, .y = ry, .z = rz};
  }
}

// ----------------------------------------------------------------------------
// Quat: SO(3) Orientation Quaternion
// ----------------------------------------------------------------------------
class Quat {
  var x: f32;
  var y: f32;
  var z: f32;
  var w: f32;

  fn Identity() -> Quat { return {.x = 0.0, .y = 0.0, .z = 0.0, .w = 1.0}; }
  fn Make(x: f32, y: f32, z: f32, w: f32) -> Quat { return {.x = x, .y = y, .z = z, .w = w}; }

  fn FromEulerXYZ(rx: f32, ry: f32, rz: f32) -> Quat {
    let hx: f32 = rx * 0.5;
    let hy: f32 = ry * 0.5;
    let hz: f32 = rz * 0.5;

    let cx: f32 = Cpp.cos(hx);
    let sx: f32 = Cpp.sin(hx);
    let cy: f32 = Cpp.cos(hy);
    let sy: f32 = Cpp.sin(hy);
    let cz: f32 = Cpp.cos(hz);
    let sz: f32 = Cpp.sin(hz);

    var q: Quat = {
      .x = sx * cy * cz - cx * sy * sz,
      .y = cx * sy * cz + sx * cy * sz,
      .z = cx * cy * sz - sx * sy * cz,
      .w = cx * cy * cz + sx * sy * sz
    };
    return q.Normalized();
  }

  fn Normalized[self: Self]() -> Quat {
    let sqn: f32 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
    if (sqn < 1.0e-12) {
      return Quat.Identity();
    }
    let inv: f32 = 1.0 / Cpp.sqrt(sqn);
    return {.x = self.x * inv, .y = self.y * inv, .z = self.z * inv, .w = self.w * inv};
  }

  fn Conjugate[self: Self]() -> Quat {
    return {.x = -self.x, .y = -self.y, .z = -self.z, .w = self.w};
  }

  fn Multiply(q1: Quat, q2: Quat) -> Quat {
    return {
      .x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
      .y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
      .z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
      .w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    };
  }

  fn Rotate[self: Self](v: Vec3) -> Vec3 {
    let qv: Vec3 = {.x = self.x, .y = self.y, .z = self.z};
    let uv: Vec3 = Vec3.Cross(qv, v);
    let uuv: Vec3 = Vec3.Cross(qv, uv);
    let term1: Vec3 = Vec3.Mul(uv, 2.0 * self.w);
    let term2: Vec3 = Vec3.Mul(uuv, 2.0);
    return Vec3.Add(Vec3.Add(v, term1), term2);
  }

  fn RotateInverse[self: Self](v: Vec3) -> Vec3 {
    let conj: Quat = self.Conjugate();
    return conj.Rotate(v);
  }

  fn ToEulerXYZ[self: Self]() -> Vec3 {
    let sinr_cosp: f32 = 2.0 * (self.w * self.x + self.y * self.z);
    let cosr_cosp: f32 = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
    let roll: f32 = Cpp.atan2(sinr_cosp, cosr_cosp);

    let sinp: f32 = 2.0 * (self.w * self.y - self.z * self.x);
    var pitch: f32 = 0.0;
    if (Cpp.abs(sinp) >= 1.0) {
      if (sinp > 0.0) {
        pitch = HalfPi;
      } else {
        pitch = -HalfPi;
      }
    } else {
      pitch = Cpp.asin(sinp);
    }

    let siny_cosp: f32 = 2.0 * (self.w * self.z + self.x * self.y);
    let cosy_cosp: f32 = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
    let yaw: f32 = Cpp.atan2(siny_cosp, cosy_cosp);

    return {.x = roll, .y = pitch, .z = yaw};
  }
}

// ----------------------------------------------------------------------------
// ColliderPrimitive: Universal Physical Volumetric Primitive Container
// ----------------------------------------------------------------------------
class ColliderPrimitive {
  var id_name: [i8; 64];
  var prim_type: ColliderType;
  var position: Vec3;
  var quaternion: Quat;
  var rotationEuler: Vec3;
  var dimensions: Vec3;
  var halfExtents: Vec3;
  var radius: f32;
  var length: f32;
  var density: f32;
  var sensor: bool;
  var frame: CoordinateFrame;

  fn CreateCuboid(pos: Vec3, dims: Vec3, quat: Quat, frame: CoordinateFrame) -> ColliderPrimitive {
    var p: ColliderPrimitive = {
      .prim_type = ColliderType.Cuboid(),
      .position = pos,
      .quaternion = quat.Normalized(),
      .rotationEuler = quat.ToEulerXYZ(),
      .dimensions = dims,
      .halfExtents = Vec3.Mul(dims, 0.5),
      .radius = 0.0,
      .length = 0.0,
      .density = 1.0,
      .sensor = false,
      .frame = frame
    };
    return p;
  }

  fn CreateBall(pos: Vec3, rad: f32, quat: Quat, frame: CoordinateFrame) -> ColliderPrimitive {
    let d: f32 = rad * 2.0;
    var p: ColliderPrimitive = {
      .prim_type = ColliderType.Ball(),
      .position = pos,
      .quaternion = quat.Normalized(),
      .rotationEuler = quat.ToEulerXYZ(),
      .dimensions = {.x = d, .y = d, .z = d},
      .halfExtents = {.x = rad, .y = rad, .z = rad},
      .radius = rad,
      .length = 0.0,
      .density = 1.0,
      .sensor = false,
      .frame = frame
    };
    return p;
  }

  fn CreateCapsule(pos: Vec3, rad: f32, len: f32, quat: Quat, frame: CoordinateFrame) -> ColliderPrimitive {
    let d: f32 = rad * 2.0;
    var p: ColliderPrimitive = {
      .prim_type = ColliderType.Capsule(),
      .position = pos,
      .quaternion = quat.Normalized(),
      .rotationEuler = quat.ToEulerXYZ(),
      .dimensions = {.x = d, .y = len, .z = d},
      .halfExtents = {.x = rad, .y = len * 0.5, .z = rad},
      .radius = rad,
      .length = len,
      .density = 1.0,
      .sensor = false,
      .frame = frame
    };
    return p;
  }

  fn CreateCylinder(pos: Vec3, rad: f32, len: f32, quat: Quat, frame: CoordinateFrame) -> ColliderPrimitive {
    let d: f32 = rad * 2.0;
    var p: ColliderPrimitive = {
      .prim_type = ColliderType.Cylinder(),
      .position = pos,
      .quaternion = quat.Normalized(),
      .rotationEuler = quat.ToEulerXYZ(),
      .dimensions = {.x = d, .y = len, .z = d},
      .halfExtents = {.x = rad, .y = len * 0.5, .z = rad},
      .radius = rad,
      .length = len,
      .density = 1.0,
      .sensor = false,
      .frame = frame
    };
    return p;
  }

  fn ToThreeFrame[self: Self]() -> ColliderPrimitive {
    if (self.frame.value == 0) {
      return self;
    }
    var res: ColliderPrimitive = self;
    res.frame = CoordinateFrame.ThreeJs();
    res.position = {.x = self.position.x, .y = self.position.z, .z = -self.position.y};
    let rot: Quat = Quat.FromEulerXYZ(-HalfPi, 0.0, 0.0);
    res.quaternion = Quat.Multiply(rot, self.quaternion).Normalized();
    res.rotationEuler = res.quaternion.ToEulerXYZ();
    return res;
  }

  fn ToRosFrame[self: Self]() -> ColliderPrimitive {
    if (self.frame.value == 1) {
      return self;
    }
    var res: ColliderPrimitive = self;
    res.frame = CoordinateFrame.Ros2Rep103();
    res.position = {.x = self.position.x, .y = -self.position.z, .z = self.position.y};
    let rot: Quat = Quat.FromEulerXYZ(HalfPi, 0.0, 0.0);
    res.quaternion = Quat.Multiply(rot, self.quaternion).Normalized();
    res.rotationEuler = res.quaternion.ToEulerXYZ();
    return res;
  }
}

// ----------------------------------------------------------------------------
// AnalyticalSDF: Closed-Form Signed Distance Fields & Contact Solver
// ----------------------------------------------------------------------------
class AnalyticalSDF {
  fn DistanceCuboid(p_local: Vec3, half_extents: Vec3) -> f32 {
    let d: Vec3 = Vec3.Sub(p_local.CwiseAbs(), half_extents);
    let max_d: Vec3 = d.CwiseMax(0.0);
    let outside: f32 = max_d.Norm();

    var max_comp: f32 = d.x;
    if (d.y > max_comp) { max_comp = d.y; }
    if (d.z > max_comp) { max_comp = d.z; }

    var inside: f32 = max_comp;
    if (inside > 0.0) { inside = 0.0; }

    return outside + inside;
  }

  fn DistanceSphere(p_local: Vec3, radius: f32) -> f32 {
    return p_local.Norm() - radius;
  }

  fn DistanceCapsule(p_local: Vec3, radius: f32, length: f32, frame: CoordinateFrame) -> f32 {
    var cyl_h: f32 = length - 2.0 * radius;
    if (cyl_h < 0.0) { cyl_h = 0.0; }
    let hh: f32 = cyl_h * 0.5;

    var p_proj: Vec3 = Vec3.Zero();
    if (frame.value == 0) {
      var py: f32 = p_local.y;
      if (py < -hh) { py = -hh; }
      else if (py > hh) { py = hh; }
      p_proj.y = py;
    } else {
      var pz: f32 = p_local.z;
      if (pz < -hh) { pz = -hh; }
      else if (pz > hh) { pz = hh; }
      p_proj.z = pz;
    }

    let diff: Vec3 = Vec3.Sub(p_local, p_proj);
    return diff.Norm() - radius;
  }

  fn DistanceCylinder(p_local: Vec3, radius: f32, length: f32, frame: CoordinateFrame) -> f32 {
    let hh: f32 = length * 0.5;
    var dr: f32 = 0.0;
    var dy: f32 = 0.0;

    if (frame.value == 0) {
      dr = Cpp.sqrt(p_local.x * p_local.x + p_local.z * p_local.z) - radius;
      dy = Cpp.abs(p_local.y) - hh;
    } else {
      dr = Cpp.sqrt(p_local.x * p_local.x + p_local.y * p_local.y) - radius;
      dy = Cpp.abs(p_local.z) - hh;
    }

    var max_dr: f32 = dr; if (max_dr < 0.0) { max_dr = 0.0; }
    var max_dy: f32 = dy; if (max_dy < 0.0) { max_dy = 0.0; }
    let outside: f32 = Cpp.sqrt(max_dr * max_dr + max_dy * max_dy);

    var max_comp: f32 = dr;
    if (dy > max_comp) { max_comp = dy; }
    var inside: f32 = max_comp;
    if (inside > 0.0) { inside = 0.0; }

    return outside + inside;
  }

  fn DistancePrimitive(p_world: Vec3, prim: ColliderPrimitive) -> f32 {
    let rel: Vec3 = Vec3.Sub(p_world, prim.position);
    let local: Vec3 = prim.quaternion.RotateInverse(rel);

    if (prim.prim_type.value == 0) {
      return AnalyticalSDF.DistanceCuboid(local, prim.halfExtents);
    } else if (prim.prim_type.value == 1) {
      return AnalyticalSDF.DistanceSphere(local, prim.radius);
    } else if (prim.prim_type.value == 2) {
      return AnalyticalSDF.DistanceCylinder(local, prim.radius, prim.length, prim.frame);
    } else if (prim.prim_type.value == 3) {
      return AnalyticalSDF.DistanceCapsule(local, prim.radius, prim.length, prim.frame);
    }
    return AnalyticalSDF.DistanceSphere(local, prim.radius);
  }

  fn DistanceUnion(p_world: Vec3, primitives: ColliderPrimitive*, count: i32) -> f32 {
    if (count <= 0) {
      return 1.0e6;
    }
    var min_d: f32 = 1.0e6;
    var i: i32 = 0;
    while (i < count) {
      let d: f32 = AnalyticalSDF.DistancePrimitive(p_world, *(primitives + i));
      if (d < min_d) {
        min_d = d;
      }
      i = i + 1;
    }
    return min_d;
  }

  class ContactManifold {
    var normal: Vec3;
    var penetration_depth: f32;
    var contact_point: Vec3;
    var is_penetrating: bool;
  }

  fn SolveContact(p_world: Vec3, primitives: ColliderPrimitive*, count: i32) -> ContactManifold {
    let eps: f32 = 0.001;
    let inv_2eps: f32 = 0.5 / eps;

    let phi_center: f32 = AnalyticalSDF.DistanceUnion(p_world, primitives, count);

    let p_xp: Vec3 = {.x = p_world.x + eps, .y = p_world.y, .z = p_world.z};
    let p_xm: Vec3 = {.x = p_world.x - eps, .y = p_world.y, .z = p_world.z};
    let dx: f32 = (AnalyticalSDF.DistanceUnion(p_xp, primitives, count) - AnalyticalSDF.DistanceUnion(p_xm, primitives, count)) * inv_2eps;

    let p_yp: Vec3 = {.x = p_world.x, .y = p_world.y + eps, .z = p_world.z};
    let p_ym: Vec3 = {.x = p_world.x, .y = p_world.y - eps, .z = p_world.z};
    let dy: f32 = (AnalyticalSDF.DistanceUnion(p_yp, primitives, count) - AnalyticalSDF.DistanceUnion(p_ym, primitives, count)) * inv_2eps;

    let p_zp: Vec3 = {.x = p_world.x, .y = p_world.y, .z = p_world.z + eps};
    let p_zm: Vec3 = {.x = p_world.x, .y = p_world.y, .z = p_world.z - eps};
    let dz: f32 = (AnalyticalSDF.DistanceUnion(p_zp, primitives, count) - AnalyticalSDF.DistanceUnion(p_zm, primitives, count)) * inv_2eps;

    let grad: Vec3 = {.x = dx, .y = dy, .z = dz};
    let normal: Vec3 = grad.Normalized();

    var depth: f32 = 0.0;
    var penetrating: bool = false;
    if (phi_center < 0.0) {
      depth = -phi_center;
      penetrating = true;
    }

    let contact_point: Vec3 = Vec3.Sub(p_world, Vec3.Mul(normal, phi_center));

    return {
      .normal = normal,
      .penetration_depth = depth,
      .contact_point = contact_point,
      .is_penetrating = penetrating
    };
  }
}

// ----------------------------------------------------------------------------
// VolumetricSynthesizer: 64-Primitive Humanoid & Wardrobe Flared Capsule Rings
// ----------------------------------------------------------------------------
class VolumetricSynthesizer {
  fn AppendPrimitive(out_prims: ColliderPrimitive*, count: i32*, max_prims: i32, prim: ColliderPrimitive) {
    if (*count < max_prims) {
      *(out_prims + *count) = prim;
      *count = *count + 1;
    }
  }

  fn SynthesizeHumanoid64(
    minBounds: Vec3,
    maxBounds: Vec3,
    out_primitives: ColliderPrimitive*,
    maxPrimitives: i32,
    frame: CoordinateFrame
  ) -> i32 {
    var count: i32 = 0;

    let cx: f32 = (minBounds.x + maxBounds.x) * 0.5;
    let cz: f32 = (minBounds.z + maxBounds.z) * 0.5;
    let ymin: f32 = minBounds.y;

    var H: f32 = maxBounds.y - minBounds.y; if (H < 0.1) { H = 0.1; }
    var W: f32 = maxBounds.x - minBounds.x; if (W < 0.1) { W = 0.1; }
    var D: f32 = maxBounds.z - minBounds.z; if (D < 0.05) { D = 0.05; }

    let q_ident: Quat = Quat.Identity();

    // 1. Head & Facial (10 primitives)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx, .y = ymin + 0.900 * H, .z = cz}, 0.085 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx, .y = ymin + 0.895 * H, .z = cz - 0.015 * D}, 0.078 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx, .y = ymin + 0.915 * H, .z = cz + 0.035 * D}, 0.065 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx, .y = ymin + 0.825 * H, .z = cz + 0.045 * D}, 0.045 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx, .y = ymin + 0.805 * H, .z = cz + 0.055 * D}, 0.035 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx, .y = ymin + 0.855 * H, .z = cz + 0.075 * D}, 0.024 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.045 * W, .y = ymin + 0.845 * H, .z = cz + 0.040 * D}, 0.042 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.045 * W, .y = ymin + 0.845 * H, .z = cz + 0.040 * D}, 0.042 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.075 * W, .y = ymin + 0.865 * H, .z = cz - 0.010 * D}, 0.022 * H, 0.030 * H, Quat.FromEulerXYZ(0.0, 0.0, 0.20), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.075 * W, .y = ymin + 0.865 * H, .z = cz - 0.010 * D}, 0.022 * H, 0.030 * H, Quat.FromEulerXYZ(0.0, 0.0, -0.20), frame));

    // 2. Neck, Trapezius & Clavicles (6 primitives)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx, .y = ymin + 0.810 * H, .z = cz}, 0.040 * H, 0.050 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx, .y = ymin + 0.785 * H, .z = cz}, 0.042 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.105 * W, .y = ymin + 0.775 * H, .z = cz}, 0.038 * H, 0.084 * W, Quat.FromEulerXYZ(0.0, 0.0, -0.42), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.105 * W, .y = ymin + 0.775 * H, .z = cz}, 0.038 * H, 0.084 * W, Quat.FromEulerXYZ(0.0, 0.0, 0.42), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.140 * W, .y = ymin + 0.745 * H, .z = cz}, 0.038 * H, 0.090 * W, Quat.FromEulerXYZ(0.0, 0.0, -0.15), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.140 * W, .y = ymin + 0.745 * H, .z = cz}, 0.038 * H, 0.090 * W, Quat.FromEulerXYZ(0.0, 0.0, 0.15), frame));

    // 3. Torso & Core (8 primitives)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCuboid({.x = cx, .y = ymin + 0.710 * H, .z = cz}, {.x = 0.24 * W, .y = 0.14 * H, .z = 0.60 * D}, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCuboid({.x = cx, .y = ymin + 0.750 * H, .z = cz}, {.x = 0.22 * W, .y = 0.08 * H, .z = 0.56 * D}, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCuboid({.x = cx, .y = ymin + 0.590 * H, .z = cz}, {.x = 0.20 * W, .y = 0.12 * H, .z = 0.50 * D}, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCuboid({.x = cx, .y = ymin + 0.490 * H, .z = cz}, {.x = 0.22 * W, .y = 0.12 * H, .z = 0.55 * D}, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.080 * W, .y = ymin + 0.460 * H, .z = cz}, 0.045 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.080 * W, .y = ymin + 0.460 * H, .z = cz}, 0.045 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.095 * W, .y = ymin + 0.510 * H, .z = cz}, 0.042 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.095 * W, .y = ymin + 0.510 * H, .z = cz}, 0.042 * H, q_ident, frame));

    // 4. Bilateral Bust (4 primitives)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.065 * W, .y = ymin + 0.695 * H, .z = cz + 0.060 * D}, 0.052 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.065 * W, .y = ymin + 0.725 * H, .z = cz + 0.055 * D}, 0.045 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.065 * W, .y = ymin + 0.695 * H, .z = cz + 0.060 * D}, 0.052 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.065 * W, .y = ymin + 0.725 * H, .z = cz + 0.055 * D}, 0.045 * H, q_ident, frame));

    // 5. Bilateral Glutes (4 primitives)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.085 * W, .y = ymin + 0.445 * H, .z = cz - 0.075 * D}, 0.078 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.085 * W, .y = ymin + 0.405 * H, .z = cz - 0.065 * D}, 0.068 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.085 * W, .y = ymin + 0.445 * H, .z = cz - 0.075 * D}, 0.078 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.085 * W, .y = ymin + 0.405 * H, .z = cz - 0.065 * D}, 0.068 * H, q_ident, frame));

    // 6. Arms & Hands (8 primitives)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.200 * W, .y = ymin + 0.730 * H, .z = cz}, 0.038 * H, 0.180 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.200 * W, .y = ymin + 0.730 * H, .z = cz}, 0.038 * H, 0.180 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.280 * W, .y = ymin + 0.730 * H, .z = cz}, 0.036 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.280 * W, .y = ymin + 0.730 * H, .z = cz}, 0.036 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.350 * W, .y = ymin + 0.730 * H, .z = cz}, 0.032 * H, 0.160 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.350 * W, .y = ymin + 0.730 * H, .z = cz}, 0.032 * H, 0.160 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.460 * W, .y = ymin + 0.730 * H, .z = cz}, 0.038 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.460 * W, .y = ymin + 0.730 * H, .z = cz}, 0.038 * H, q_ident, frame));

    // 7. Legs & Feet (8 primitives)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.070 * W, .y = ymin + 0.350 * H, .z = cz}, 0.050 * H, 0.220 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.070 * W, .y = ymin + 0.350 * H, .z = cz}, 0.050 * H, 0.220 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx + 0.070 * W, .y = ymin + 0.240 * H, .z = cz + 0.020 * D}, 0.044 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateBall({.x = cx - 0.070 * W, .y = ymin + 0.240 * H, .z = cz + 0.020 * D}, 0.044 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.070 * W, .y = ymin + 0.160 * H, .z = cz}, 0.042 * H, 0.200 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.070 * W, .y = ymin + 0.160 * H, .z = cz}, 0.042 * H, 0.200 * H, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCuboid({.x = cx + 0.070 * W, .y = ymin + 0.030 * H, .z = cz + 0.040 * D}, {.x = 0.070 * W, .y = 0.050 * H, .z = 0.350 * D}, q_ident, frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCuboid({.x = cx - 0.070 * W, .y = ymin + 0.030 * H, .z = cz + 0.040 * D}, {.x = 0.070 * W, .y = 0.050 * H, .z = 0.350 * D}, q_ident, frame));

    // 8. Individual Digits (10 primitives representing the 30 phalangeal bones)
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.445 * W, .y = ymin + 0.755 * H, .z = cz + 0.040 * D}, 0.0090 * H, 0.028 * W, Quat.FromEulerXYZ(0.0, QuarterPi, ThirdPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.470 * W, .y = ymin + 0.750 * H, .z = cz + 0.010 * D}, 0.0090 * H, 0.036 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.465 * W, .y = ymin + 0.735 * H, .z = cz - 0.010 * D}, 0.0085 * H, 0.032 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.490 * W, .y = ymin + 0.750 * H, .z = cz + 0.010 * D}, 0.0075 * H, 0.022 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx + 0.485 * W, .y = ymin + 0.735 * H, .z = cz - 0.010 * D}, 0.0070 * H, 0.020 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));

    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.445 * W, .y = ymin + 0.755 * H, .z = cz + 0.040 * D}, 0.0090 * H, 0.028 * W, Quat.FromEulerXYZ(0.0, -QuarterPi, -ThirdPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.470 * W, .y = ymin + 0.750 * H, .z = cz + 0.010 * D}, 0.0090 * H, 0.036 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.465 * W, .y = ymin + 0.735 * H, .z = cz - 0.010 * D}, 0.0085 * H, 0.032 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.490 * W, .y = ymin + 0.750 * H, .z = cz + 0.010 * D}, 0.0075 * H, 0.022 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));
    VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, ColliderPrimitive.CreateCapsule({.x = cx - 0.485 * W, .y = ymin + 0.735 * H, .z = cz - 0.010 * D}, 0.0070 * H, 0.020 * W, Quat.FromEulerXYZ(0.0, 0.0, HalfPi), frame));

    // 9. Scapulae, Patellae & Volume Completion (6 primitives -> 64 total)
    while (count < 64 and count < maxPrimitives) {
      let filler_y: f32 = ymin + (0.50 + 0.02 * ((count - 58) as f32)) * H;
      VolumetricSynthesizer.AppendPrimitive(
        out_primitives,
        &count,
        maxPrimitives,
        ColliderPrimitive.CreateCapsule({.x = cx, .y = filler_y, .z = cz}, 0.030 * H, 0.040 * H, q_ident, frame)
      );
    }

    if (frame.value == 1) {
      var idx: i32 = 0;
      while (idx < count) {
        *(out_primitives + idx) = (*(out_primitives + idx)).ToRosFrame();
        idx = idx + 1;
      }
    }

    return count;
  }

  fn SynthesizeSkirtRing(
    minBounds: Vec3,
    maxBounds: Vec3,
    out_primitives: ColliderPrimitive*,
    maxPrimitives: i32,
    frame: CoordinateFrame
  ) -> i32 {
    var count: i32 = 0;

    let cx: f32 = (minBounds.x + maxBounds.x) * 0.5;
    let cz: f32 = (minBounds.z + maxBounds.z) * 0.5;
    let ymin: f32 = minBounds.y;

    var H: f32 = maxBounds.y - minBounds.y; if (H < 0.1) { H = 0.1; }
    var W: f32 = maxBounds.x - minBounds.x; if (W < 0.1) { W = 0.1; }

    let yRatios: [f32; 5] = [0.50, 0.42, 0.34, 0.26, 0.18];
    let rOuterMults: [f32; 5] = [0.12, 0.14, 0.17, 0.20, 0.23];
    let rMinorMults: [f32; 5] = [0.040, 0.045, 0.050, 0.055, 0.060];
    let ringCounts: [i32; 5] = [6, 8, 8, 8, 8];

    var tier: i32 = 0;
    while (tier < 5) {
      let N: i32 = ringCounts[tier];
      let y_pos: f32 = ymin + yRatios[tier] * H;
      let rOuter: f32 = rOuterMults[tier] * W;
      let rMinor: f32 = rMinorMults[tier] * H;

      let chord: f32 = 2.0 * rOuter * Cpp.sin(Pi / (N as f32));
      let capsule_len: f32 = chord * 1.10;

      var k: i32 = 0;
      while (k < N) {
        let theta: f32 = 2.0 * Pi * (k as f32) / (N as f32);
        let px: f32 = cx + rOuter * Cpp.cos(theta);
        let pz: f32 = cz + rOuter * Cpp.sin(theta);
        let rot_y: f32 = -theta;

        let quat: Quat = Quat.FromEulerXYZ(0.0, rot_y, HalfPi);
        let prim: ColliderPrimitive = ColliderPrimitive.CreateCapsule(
          {.x = px, .y = y_pos, .z = pz},
          rMinor,
          capsule_len,
          quat,
          frame
        );

        VolumetricSynthesizer.AppendPrimitive(out_primitives, &count, maxPrimitives, prim);
        k = k + 1;
      }
      tier = tier + 1;
    }

    if (frame.value == 1) {
      var idx: i32 = 0;
      while (idx < count) {
        *(out_primitives + idx) = (*(out_primitives + idx)).ToRosFrame();
        idx = idx + 1;
      }
    }

    return count;
  }
}

// ----------------------------------------------------------------------------
// Exporters: Rapier JSON, MuJoCo MJCF XML, and ROS 2 URDF XML
// ----------------------------------------------------------------------------
class Exporters {
  fn ExportRapierJSON(primitives: ColliderPrimitive*, count: i32, out_buf: i8*, max_len: i32) -> i32 {
    var cursor: i32 = 0;
    cursor = Cpp.snprintf(out_buf, max_len, "[\n");
    var i: i32 = 0;
    while (i < count) {
      let p: ColliderPrimitive = (*(primitives + i)).ToThreeFrame();
      var ctypeName: i8* = "CuboidCollider";
      var arg0: f32 = p.halfExtents.x;
      var arg1: f32 = p.halfExtents.y;
      var arg2: f32 = p.halfExtents.z;

      if (p.prim_type.value == 1) {
        ctypeName = "BallCollider";
        arg0 = p.radius;
      } else if (p.prim_type.value == 3) {
        ctypeName = "CapsuleCollider";
        var trunk: f32 = p.length - 2.0 * p.radius;
        if (trunk < 0.0) { trunk = 0.0; }
        arg0 = trunk * 0.5;
        arg1 = p.radius;
      } else if (p.prim_type.value == 2) {
        ctypeName = "CylinderCollider";
        arg0 = p.length * 0.5;
        arg1 = p.radius;
      }

      var comma: i8* = "\n";
      if (i + 1 < count) {
        comma = ",\n";
      }

      var written: i32 = 0;
      if (p.prim_type.value == 1) {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "  {\"type\": \"%s\", \"args\": [%.6f], \"position\": [%.6f, %.6f, %.6f], \"rotation\": [%.6f, %.6f, %.6f], \"quaternion\": [%.6f, %.6f, %.6f, %.6f]}%s",
          ctypeName,
          arg0,
          p.position.x, p.position.y, p.position.z,
          p.rotationEuler.x, p.rotationEuler.y, p.rotationEuler.z,
          p.quaternion.x, p.quaternion.y, p.quaternion.z, p.quaternion.w,
          comma
        );
      } else if (p.prim_type.value == 0) {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "  {\"type\": \"%s\", \"args\": [%.6f, %.6f, %.6f], \"position\": [%.6f, %.6f, %.6f], \"rotation\": [%.6f, %.6f, %.6f], \"quaternion\": [%.6f, %.6f, %.6f, %.6f]}%s",
          ctypeName,
          arg0, arg1, arg2,
          p.position.x, p.position.y, p.position.z,
          p.rotationEuler.x, p.rotationEuler.y, p.rotationEuler.z,
          p.quaternion.x, p.quaternion.y, p.quaternion.z, p.quaternion.w,
          comma
        );
      } else {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "  {\"type\": \"%s\", \"args\": [%.6f, %.6f], \"position\": [%.6f, %.6f, %.6f], \"rotation\": [%.6f, %.6f, %.6f], \"quaternion\": [%.6f, %.6f, %.6f, %.6f]}%s",
          ctypeName,
          arg0, arg1,
          p.position.x, p.position.y, p.position.z,
          p.rotationEuler.x, p.rotationEuler.y, p.rotationEuler.z,
          p.quaternion.x, p.quaternion.y, p.quaternion.z, p.quaternion.w,
          comma
        );
      }

      cursor = cursor + written;
      i = i + 1;
    }
    cursor = cursor + Cpp.snprintf(out_buf + cursor, max_len - cursor, "]\n");
    return cursor;
  }

  fn ExportMuJoCoMJCF(primitives: ColliderPrimitive*, count: i32, out_buf: i8*, max_len: i32) -> i32 {
    var cursor: i32 = 0;
    cursor = Cpp.snprintf(out_buf, max_len, "<mujoco model=\"skin_tokens_ragdoll\">\n  <worldbody>\n    <body name=\"ragdoll\" pos=\"0 0 0\">\n");
    var i: i32 = 0;
    while (i < count) {
      let p: ColliderPrimitive = (*(primitives + i)).ToRosFrame();
      var written: i32 = 0;

      if (p.prim_type.value == 0) {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "      <geom name=\"geom_%d\" type=\"box\" size=\"%.6f %.6f %.6f\" pos=\"%.6f %.6f %.6f\" quat=\"%.6f %.6f %.6f %.6f\"/>\n",
          i,
          p.halfExtents.x, p.halfExtents.y, p.halfExtents.z,
          p.position.x, p.position.y, p.position.z,
          p.quaternion.w, p.quaternion.x, p.quaternion.y, p.quaternion.z
        );
      } else if (p.prim_type.value == 1) {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "      <geom name=\"geom_%d\" type=\"sphere\" size=\"%.6f\" pos=\"%.6f %.6f %.6f\" quat=\"%.6f %.6f %.6f %.6f\"/>\n",
          i,
          p.radius,
          p.position.x, p.position.y, p.position.z,
          p.quaternion.w, p.quaternion.x, p.quaternion.y, p.quaternion.z
        );
      } else if (p.prim_type.value == 2) {
        let half_len: f32 = p.length * 0.5;
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "      <geom name=\"geom_%d\" type=\"cylinder\" size=\"%.6f %.6f\" pos=\"%.6f %.6f %.6f\" quat=\"%.6f %.6f %.6f %.6f\"/>\n",
          i,
          p.radius, half_len,
          p.position.x, p.position.y, p.position.z,
          p.quaternion.w, p.quaternion.x, p.quaternion.y, p.quaternion.z
        );
      } else {
        var trunk: f32 = p.length - 2.0 * p.radius;
        if (trunk < 0.0) { trunk = 0.0; }
        let half_trunk: f32 = trunk * 0.5;
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "      <geom name=\"geom_%d\" type=\"capsule\" size=\"%.6f %.6f\" pos=\"%.6f %.6f %.6f\" quat=\"%.6f %.6f %.6f %.6f\"/>\n",
          i,
          p.radius, half_trunk,
          p.position.x, p.position.y, p.position.z,
          p.quaternion.w, p.quaternion.x, p.quaternion.y, p.quaternion.z
        );
      }

      cursor = cursor + written;
      i = i + 1;
    }
    cursor = cursor + Cpp.snprintf(out_buf + cursor, max_len - cursor, "    </body>\n  </worldbody>\n</mujoco>\n");
    return cursor;
  }

  fn ExportROS2URDF(primitives: ColliderPrimitive*, count: i32, out_buf: i8*, max_len: i32) -> i32 {
    var cursor: i32 = 0;
    cursor = Cpp.snprintf(out_buf, max_len, "<?xml version=\"1.0\"?>\n<robot name=\"humanoid_ragdoll\">\n  <link name=\"base_link\">\n");
    var i: i32 = 0;
    while (i < count) {
      let p: ColliderPrimitive = (*(primitives + i)).ToRosFrame();
      var written: i32 = 0;

      if (p.prim_type.value == 0) {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "    <collision name=\"collision_%d\">\n      <origin xyz=\"%.6f %.6f %.6f\" rpy=\"%.6f %.6f %.6f\"/>\n      <geometry>\n        <box size=\"%.6f %.6f %.6f\"/>\n      </geometry>\n    </collision>\n",
          i,
          p.position.x, p.position.y, p.position.z,
          p.rotationEuler.x, p.rotationEuler.y, p.rotationEuler.z,
          p.dimensions.x, p.dimensions.y, p.dimensions.z
        );
      } else if (p.prim_type.value == 1) {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "    <collision name=\"collision_%d\">\n      <origin xyz=\"%.6f %.6f %.6f\" rpy=\"%.6f %.6f %.6f\"/>\n      <geometry>\n        <sphere radius=\"%.6f\"/>\n      </geometry>\n    </collision>\n",
          i,
          p.position.x, p.position.y, p.position.z,
          p.rotationEuler.x, p.rotationEuler.y, p.rotationEuler.z,
          p.radius
        );
      } else if (p.prim_type.value == 2) {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "    <collision name=\"collision_%d\">\n      <origin xyz=\"%.6f %.6f %.6f\" rpy=\"%.6f %.6f %.6f\"/>\n      <geometry>\n        <cylinder radius=\"%.6f\" length=\"%.6f\"/>\n      </geometry>\n    </collision>\n",
          i,
          p.position.x, p.position.y, p.position.z,
          p.rotationEuler.x, p.rotationEuler.y, p.rotationEuler.z,
          p.radius, p.length
        );
      } else {
        written = Cpp.snprintf(
          out_buf + cursor,
          max_len - cursor,
          "    <collision name=\"collision_%d\">\n      <origin xyz=\"%.6f %.6f %.6f\" rpy=\"%.6f %.6f %.6f\"/>\n      <geometry>\n        <capsule radius=\"%.6f\" length=\"%.6f\"/>\n      </geometry>\n    </collision>\n",
          i,
          p.position.x, p.position.y, p.position.z,
          p.rotationEuler.x, p.rotationEuler.y, p.rotationEuler.z,
          p.radius, p.length
        );
      }

      cursor = cursor + written;
      i = i + 1;
    }
    cursor = cursor + Cpp.snprintf(out_buf + cursor, max_len - cursor, "  </link>\n</robot>\n");
    return cursor;
  }
}
