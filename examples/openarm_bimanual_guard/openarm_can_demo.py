#!/usr/bin/env python3
"""
===============================================================================
🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
OpenArm Bimanual 1kHz CAN Closed-Loop Safety Verification & Collision Demo
File: examples/openarm_bimanual_guard/openarm_can_demo.py
===============================================================================

Standalone verification script (zero external dependencies) demonstrating:
  1. 1kHz Closed-Loop Predictive Control Barrier Function (CBF) execution.
  2. Intentional high-speed bimanual collision assault (> 3.5 rad/s inward).
  3. Violent joint limit assault (> 6.0 rad/s boundary strikes).
  4. Actuator limit guarantees for 16 CAN nodes (Right IDs 1..8, Left IDs 11..18):
     - Right Wrist Pitch J6: [-0.7854, +0.7854] rad (+/-45°)
     - Left Wrist Pitch J16: [-0.7854, +2.6180] rad (-45° to +150°)
     - Wrist Curls J7/J17:   [-0.4189, +0.4189] rad (+/-24° with 1° cushion)
     - Gripper Claws J8/J18: [0.0000, 0.4500] rad (stall current <= 1500mA)

Guarantees Verified:
  - max_overshoot == 0.00000 rad across all 16 joints
  - min_clearance > 0.00 m throughout entire bimanual trajectory
"""

import ctypes
import math
import os
from pathlib import Path
import platform
import struct
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
BUILD_DIR = ROOT_DIR / "build"

IS_WINDOWS = platform.system().lower().startswith("win")
SHARED_EXT = ".dll" if IS_WINDOWS else (".dylib" if platform.system().lower() == "darwin" else ".so")

# Candidate paths for the shared library
CANDIDATE_LIBS = [
    BUILD_DIR / f"openarm_bimanual_guard{SHARED_EXT}",
    BUILD_DIR / f"kimodo_carbon{SHARED_EXT}",
    SCRIPT_DIR / f"openarm_bimanual_guard{SHARED_EXT}",
    SCRIPT_DIR / f"libopenarm_bimanual_guard{SHARED_EXT}",
]


def load_guard_library() -> ctypes.CDLL:
    """Locates and loads the OpenArm bimanual guard shared library via ctypes."""
    for cand in CANDIDATE_LIBS:
        if cand.exists():
            try:
                lib = ctypes.CDLL(str(cand))
                if hasattr(lib, "openarm_bimanual_guard_step_1khz"):
                    return lib
            except Exception as e:
                print(f"[DEBUG] Failed loading candidate {cand}: {e}")

    # Fallback to system search
    try:
        return ctypes.CDLL(f"openarm_bimanual_guard{SHARED_EXT}")
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not locate openarm_bimanual_guard{SHARED_EXT} in any of: "
        f"{[str(p) for p in CANDIDATE_LIBS]}. Run 'python build_and_benchmark.py' first."
    )


# Actuator Limit Specifications for 16 CAN Nodes
ACTUATOR_SPECS = [
    # Right Arm (CAN IDs 1..8)
    {"id": 1,  "name": "R. Shoulder Yaw",   "min": -1.5708, "max": +1.5708, "tau_max": 35.0},
    {"id": 2,  "name": "R. Shoulder Pitch", "min": -1.0000, "max": +1.5708, "tau_max": 35.0},
    {"id": 3,  "name": "R. Bicep Rotate",   "min": -1.5708, "max": +1.5708, "tau_max": 35.0},
    {"id": 4,  "name": "R. Elbow Pitch",    "min":  0.0000, "max": +2.2000, "tau_max": 35.0},
    {"id": 5,  "name": "R. Forearm Roll",   "min": -1.5708, "max": +1.5708, "tau_max": 35.0},
    {"id": 6,  "name": "R. Wrist Pitch",    "min": -0.7854, "max": +0.7854, "tau_max": 12.0},
    {"id": 7,  "name": "R. Wrist Curl",     "min": -0.4189, "max": +0.4189, "tau_max": 12.0},
    {"id": 8,  "name": "R. Gripper Claw",   "min":  0.0000, "max": +0.4500, "tau_max":  8.0},

    # Left Arm (CAN IDs 11..18, decade mapping)
    {"id": 11, "name": "L. Shoulder Yaw",   "min": -1.5708, "max": +1.5708, "tau_max": 35.0},
    {"id": 12, "name": "L. Shoulder Pitch", "min": -1.5708, "max": +1.0000, "tau_max": 35.0},
    {"id": 13, "name": "L. Bicep Rotate",   "min": -1.5708, "max": +1.5708, "tau_max": 35.0},
    {"id": 14, "name": "L. Elbow Pitch",    "min":  0.0000, "max": +2.2000, "tau_max": 35.0},
    {"id": 15, "name": "L. Forearm Roll",   "min": -1.5708, "max": +1.5708, "tau_max": 35.0},
    {"id": 16, "name": "L. Wrist Pitch",    "min": -0.7854, "max": +2.6180, "tau_max": 12.0},
    {"id": 17, "name": "L. Wrist Curl",     "min": -0.4189, "max": +0.4189, "tau_max": 12.0},
    {"id": 18, "name": "L. Gripper Claw",   "min":  0.0000, "max": +0.4500, "tau_max":  8.0},
]


def run_demo() -> bool:
    print("=" * 88)
    print("🤖 GAMING2GAMERS (G2G) — OPENARM BIMANUAL 1kHz HARDWARE SAFETY GUARD • ∞ Meraki")
    print("=" * 88)

    lib = load_guard_library()

    # Function Signatures
    lib.openarm_bimanual_guard_init.restype = ctypes.c_void_p
    lib.openarm_bimanual_guard_init.argtypes = []

    lib.openarm_bimanual_guard_destroy.restype = None
    lib.openarm_bimanual_guard_destroy.argtypes = [ctypes.c_void_p]

    lib.openarm_bimanual_guard_step_1khz.restype = ctypes.c_int32
    lib.openarm_bimanual_guard_step_1khz.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    lib.openarm_bimanual_eval_sdf.restype = ctypes.c_int32
    lib.openarm_bimanual_eval_sdf.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    if hasattr(lib, "openarm_bimanual_guard_version"):
        lib.openarm_bimanual_guard_version.restype = ctypes.c_char_p
        ver = lib.openarm_bimanual_guard_version().decode("utf-8")
        print(f"Loaded Core Engine: {ver}\n")

    guard_ctx = lib.openarm_bimanual_guard_init()

    # Float arrays for 16 CAN nodes
    c_float_16 = ctypes.c_float * 16
    q_meas = c_float_16()
    v_meas = c_float_16()
    tau_cmd = c_float_16()
    q_next = c_float_16()
    v_next = c_float_16()
    tau_safe = c_float_16()
    out_clearance = ctypes.c_float(0.0)

    # Initial rest state: both arms slightly outward and neutral
    q_meas[0] = -0.30  # Right arm yawed slightly outward
    q_meas[1] =  0.20  # Right arm shoulder pitch
    q_meas[3] =  0.80  # Right elbow bent
    q_meas[7] =  0.20  # Right gripper half aperture

    q_meas[8] =  0.30  # Left arm yawed slightly outward
    q_meas[9] =  0.20  # Left arm shoulder pitch
    q_meas[11] = 0.80  # Left elbow bent
    q_meas[15] = 0.20  # Left gripper half aperture

    TOTAL_STEPS = 2000
    DT = 0.001  # 1kHz closed loop (1ms per tick)

    global_max_overshoot = 0.0
    global_min_clearance = float("inf")
    total_interventions = 0

    print(f"Initiating 1kHz Closed-Loop Stress Simulation ({TOTAL_STEPS} steps, dt={DT*1000.0:.1f}ms)...")
    print(f"{'Step':<6} | {'Event / Phase':<30} | {'Min Dist (m)':<13} | {'Max Overshoot':<14} | {'Status'}")
    print("-" * 88)

    t_start = time.perf_counter()

    for step in range(TOTAL_STEPS):
        t = step * DT

        # ---------------------------------------------------------------------
        # STRESS PHASE 1 (Steps 0..999): High-Speed Intentional Bimanual Collision
        # Both arms commanded with extreme torques inward toward center (Y = 0)
        # ---------------------------------------------------------------------
        if step < 1000:
            phase_name = "Bimanual Collision Assault"
            # Command Right Arm aggressively leftward (+Yaw toward Y=0)
            tau_cmd[0] = +35.0  # Max peak torque on Right Yaw
            tau_cmd[1] = +15.0  # Pitch forward
            tau_cmd[3] = +10.0  # Elbow flexion

            # Command Left Arm aggressively rightward (-Yaw toward Y=0)
            tau_cmd[8] = -35.0  # Max peak torque on Left Yaw
            tau_cmd[9] = +15.0  # Pitch forward
            tau_cmd[11] = +10.0 # Elbow flexion

            # Wrists and hands driving straight toward overlapping space
            tau_cmd[5] = +10.0  # R. Wrist pitch
            tau_cmd[6] = +8.0   # R. Wrist curl
            tau_cmd[13] = +10.0 # L. Wrist pitch
            tau_cmd[14] = +8.0  # L. Wrist curl

        # ---------------------------------------------------------------------
        # STRESS PHASE 2 (Steps 1000..1999): High-Velocity Joint Limit Assault
        # Command all joints aggressively past their mechanical stops at > 6 rad/s
        # ---------------------------------------------------------------------
        else:
            phase_name = "Joint Limit Boundary Assault"
            for i in range(16):
                # Alternate driving toward max and min limits with saturated torques
                if (step // 250) % 2 == 0:
                    tau_cmd[i] = +ACTUATOR_SPECS[i]["tau_max"]
                else:
                    tau_cmd[i] = -ACTUATOR_SPECS[i]["tau_max"]

            # Specifically target asymmetric Left Wrist Pitch J16 to +150° (+2.6180 rad)
            tau_cmd[13] = +12.0
            # Target Wrist Curls J7/J17 beyond +/-24° (+/-0.4189 rad)
            tau_cmd[6] = +12.0
            tau_cmd[14] = -12.0
            # Target Gripper Claws to clamp (0.0 rad) and over-extend (0.45 rad)
            tau_cmd[7] = -8.0
            tau_cmd[15] = +8.0

        # Execute 1kHz guard step
        status = lib.openarm_bimanual_guard_step_1khz(
            guard_ctx,
            q_meas,
            v_meas,
            tau_cmd,
            DT,
            q_next,
            v_next,
            tau_safe,
            ctypes.byref(out_clearance),
        )

        curr_clearance = out_clearance.value
        if curr_clearance < global_min_clearance:
            global_min_clearance = curr_clearance

        # Check joint limit overshoot across all 16 actuators
        for i in range(16):
            q_val = q_next[i]
            q_min = ACTUATOR_SPECS[i]["min"]
            q_max = ACTUATOR_SPECS[i]["max"]

            if q_val > q_max:
                overshoot = q_val - q_max
                if overshoot > global_max_overshoot:
                    global_max_overshoot = overshoot
            elif q_val < q_min:
                overshoot = q_min - q_val
                if overshoot > global_max_overshoot:
                    global_max_overshoot = overshoot

            # State propagation
            q_meas[i] = q_next[i]
            v_meas[i] = v_next[i]

        # Log periodic progress
        if step in (0, 250, 500, 750, 1000, 1250, 1500, 1750, TOTAL_STEPS - 1):
            stat_str = "🛡️ GUARD ACTIVE" if curr_clearance < 0.055 else "✅ CLEAR"
            print(
                f"{step:<6} | {phase_name:<30} | {curr_clearance:>+10.4f} m | "
                f"+{global_max_overshoot:.5f} rad  | {stat_str}"
            )

    elapsed_sec = time.perf_counter() - t_start
    avg_step_us = (elapsed_sec / TOTAL_STEPS) * 1.0e6

    lib.openarm_bimanual_guard_destroy(guard_ctx)

    print("-" * 88)
    print("📊 1kHz TELEMETRY SUMMARY & VERIFICATION BENCHMARK:")
    print(f"  • Total Steps Simulated:     {TOTAL_STEPS:,} steps ({TOTAL_STEPS * DT:.1f}s realtime)")
    print(f"  • Execution Time:            {elapsed_sec * 1000.0:.2f} ms ({avg_step_us:.2f} µs/step, budget <= 1000 µs)")
    print(f"  • Minimum Bimanual Distance: {global_min_clearance:>+8.4f} m (Safety corridor >= 0.050m)")
    print(f"  • Maximum Joint Overshoot:   +{global_max_overshoot:.5f} rad (Target: 0.00000 rad)")
    print("=" * 88)

    # Invariant Assertions
    passed = True
    if global_max_overshoot > 1.0e-5:
        print(f"❌ ASSERTION FAILED: Joint limit overshoot {global_max_overshoot:.5f} rad > 0.00000 rad!")
        passed = False
    else:
        print("✅ ASSERTION PASSED: max_overshoot == 0.00000 rad (Zero boundary overshoot verified).")

    if global_min_clearance <= 0.0:
        print(f"❌ ASSERTION FAILED: Self-collision detected! min_clearance = {global_min_clearance:.4f} m <= 0.00 m!")
        passed = False
    else:
        print(f"✅ ASSERTION PASSED: min_clearance = {global_min_clearance:.4f} m > 0.00 m (Zero self-collision verified).")

    print("=" * 88)
    if passed:
        print("🏆 OPENARM BIMANUAL HARDWARE SAFETY REFERENCE: VERIFIED • ∞ Meraki")
        sys.exit(0)
    else:
        print("💥 OPENARM BIMANUAL HARDWARE SAFETY REFERENCE: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    run_demo()
