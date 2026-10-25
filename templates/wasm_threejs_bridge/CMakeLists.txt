# ============================================================================
# 🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
# Dual-Mode WebAssembly & Host Native C++20 Build Configuration
# File: templates/wasm_threejs_bridge/CMakeLists.txt
# ============================================================================

cmake_minimum_required(VERSION 3.20)
project(carbon_vrm_wasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

if(EMSCRIPTEN)
  message(STATUS "🔨 [CarbonForge] Configuring for Emscripten WebAssembly target (simd128)")

  add_executable(carbon_vrm_wasm carbon_vrm_wasm.cpp)
  set_target_properties(carbon_vrm_wasm PROPERTIES
    OUTPUT_NAME "carbon_vrm_wasm"
    SUFFIX ".js"
  )

  target_compile_options(carbon_vrm_wasm PRIVATE
    -O3
    -flto
    -msimd128
  )

  target_link_options(carbon_vrm_wasm PRIVATE
    -O3
    -flto
    -msimd128
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME="createCarbonVrmModule"
    -sEXPORTED_FUNCTIONS="['_malloc','_free','_carbon_vrm_wasm_version','_carbon_vrm_slerp','_carbon_vrm_angular_deadband_filter','_carbon_vrm_blend_skeleton_pose','_carbon_vrm_simd_batch_slerp','_carbon_vrm_sdf_sphere','_carbon_vrm_sdf_box','_carbon_vrm_sdf_capsule','_carbon_vrm_sdf_cylinder','_carbon_vrm_get_humanoid64_count','_carbon_vrm_sdf_humanoid64','_carbon_vrm_solve_arm_body_collision','_carbon_vrm_solve_skirt_clipping','_carbon_vrm_get_shared_buffer']"
    -sEXPORTED_RUNTIME_METHODS="['ccall','cwrap','getValue','setValue','HEAPF32','HEAPU8']"
    -sALLOW_MEMORY_GROWTH=1
    -sENVIRONMENT="web,worker,node"
    --no-entry
  )
else()
  message(STATUS "🔨 [CarbonForge] Configuring for Native Host C++20 target")

  add_library(carbon_vrm_wasm_native SHARED carbon_vrm_wasm.cpp)

  if(MSVC)
    target_compile_definitions(carbon_vrm_wasm_native PRIVATE
      WIN32
      _WINDOWS
      CARBON_BUILDING_DLL
      _CRT_SECURE_NO_WARNINGS
    )
    target_compile_options(carbon_vrm_wasm_native PRIVATE
      /O2
      /arch:AVX2
      /EHsc
      /W4
    )
  else()
    target_compile_options(carbon_vrm_wasm_native PRIVATE
      -O3
      -mavx2
      -mfma
      -Wall
      -Wextra
      -fPIC
    )
  endif()
endif()
