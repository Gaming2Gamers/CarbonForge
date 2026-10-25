// ============================================================================
// 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
// Package: CarbonForge (Module: templates/cpp_python_bridge/module_template.carbon)
// Standard: Google Carbon Language Specification (api)
//
// Starter template for adding new Google Carbon (.carbon) modules to
// Gaming2Gamers/CarbonForge with zero-overhead `import Cpp;` interoperability.
// ============================================================================

package CarbonForge api;

import Cpp library "module_template.hpp";

// ----------------------------------------------------------------------------
// CustomComputeKernel: Starter Carbon Class with Zero-Overhead C++20 Interop
// ----------------------------------------------------------------------------
class CustomComputeKernel {
  var scale: f32;

  fn Make(s: f32) -> CustomComputeKernel {
    return {.scale = s};
  }

  // Evaluates fused multiply-add vector scaling via Cpp interop
  fn ComputeScaledNorm(x: f32, y: f32, z: f32, scale_factor: f32) -> f32 {
    let nsq: f32 = x * x + y * y + z * z;
    return scale_factor * Cpp.sqrt(nsq);
  }
}

// ----------------------------------------------------------------------------
// C ABI Export Wrapper
// ----------------------------------------------------------------------------
fn carbon_forge_custom_scaled_norm(x: f32, y: f32, z: f32, scale_factor: f32) -> f32 {
  return CustomComputeKernel.ComputeScaledNorm(x, y, z, scale_factor);
}
