#!/usr/bin/env bash
# ============================================================================
# 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
# Emscripten SIMD128 WebAssembly Build Pipeline
# File: templates/wasm_threejs_bridge/build_wasm.sh
# ============================================================================
# Compiles carbon_vrm_wasm.cpp into high-throughput WebAssembly with 128-bit
# SIMD (wasm-simd128) support, exporting modular ES6 bindings and zero-copy
# memory buffers for Three.js and @pixiv/three-vrm.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE_DIR="${SCRIPT_DIR}/../../include"
DIST_DIR="${SCRIPT_DIR}/dist"

mkdir -p "${DIST_DIR}"

echo "🔨 [CarbonForge] Building Dual-Mode Three.js Wasm Bridge with SIMD128..."

# Check for em++ compiler
if ! command -v em++ &> /dev/null; then
    echo "⚠️  [CarbonForge] em++ (Emscripten SDK) not found in PATH."
    echo "   To compile in-browser WebAssembly binaries, activate your emsdk environment:"
    echo "     source /path/to/emsdk/emsdk_env.sh"
    echo "   Skipping WebAssembly artifact emission on host without Emscripten."
    exit 0
fi

# Compile carbon_vrm_wasm.cpp into modular ES6 Wasm bundle
em++ -std=c++20 -O3 -flto -msimd128 \
  -I"${INCLUDE_DIR}" \
  "${SCRIPT_DIR}/carbon_vrm_wasm.cpp" \
  -o "${DIST_DIR}/carbon_vrm_wasm.js" \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s EXPORT_NAME="createCarbonVrmModule" \
  -s EXPORTED_FUNCTIONS='["_malloc","_free","_carbon_vrm_wasm_version","_carbon_vrm_slerp","_carbon_vrm_angular_deadband_filter","_carbon_vrm_blend_skeleton_pose","_carbon_vrm_simd_batch_slerp","_carbon_vrm_sdf_sphere","_carbon_vrm_sdf_box","_carbon_vrm_sdf_capsule","_carbon_vrm_sdf_cylinder","_carbon_vrm_get_humanoid64_count","_carbon_vrm_sdf_humanoid64","_carbon_vrm_solve_arm_body_collision","_carbon_vrm_solve_skirt_clipping","_carbon_vrm_get_shared_buffer"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF32","HEAPU8"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ENVIRONMENT="web,worker,node" \
  --no-entry

echo "✅ [CarbonForge] Emscripten SIMD128 build complete -> ${DIST_DIR}/carbon_vrm_wasm.js + .wasm"
