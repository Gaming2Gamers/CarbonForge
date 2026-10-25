#!/usr/bin/env python3
"""
===============================================================================
🔨 Gaming2Gamers (G2G) — CarbonForge • ∞ Meraki
Starter Python ctypes Zero-Copy Bridge Template
File: templates/cpp_python_bridge/ctypes_bridge_template.py
===============================================================================
Demonstrates how to bind a newly added CarbonForge C ABI export into Python
with zero-copy ctypes pointers.
"""

import ctypes
from pathlib import Path
import platform


def load_carbon_forge_lib(build_dir: Path) -> ctypes.CDLL:
    ext = ".dll" if platform.system().lower().startswith("win") else ".so"
    lib_path = build_dir / f"kimodo_carbon{ext}"
    lib = ctypes.CDLL(str(lib_path))

    # Bind kimodo_slerp as an example zero-copy C ABI call
    lib.kimodo_slerp.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.kimodo_slerp.restype = None
    return lib


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    dll = load_carbon_forge_lib(root / "build")
    q1 = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 1.0)
    q2 = (ctypes.c_float * 4)(0.0, 0.70710678, 0.0, 0.70710678)
    out = (ctypes.c_float * 4)()
    dll.kimodo_slerp(q1, q2, 0.5, out)
    print(f"CarbonForge SLERP(t=0.5) -> [{out[0]:.6f}, {out[1]:.6f}, {out[2]:.6f}, {out[3]:.6f}] • ∞ Meraki")
